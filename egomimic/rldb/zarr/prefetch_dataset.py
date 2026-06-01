"""Sliding-window NVMe pool for per-episode zipped zarr v3 storage.

Architecture
------------
ZipEpisodeResolver
    Reads ``catalog.json`` from the zip volume (``/mnt/zarr-zip``).
    Inherits key_map / transform_list / norm_stats / pause_removal_epsilon
    plumbing from EpisodeResolver. Frame counts come from the catalog.

EpisodePlan
    Deterministic permutation of the full train (or valid) catalog, fixed
    by seed for the whole run. Each epoch consumes a contiguous slice of
    ``episodes_per_epoch`` episodes from the plan. Same seed on all DDP
    ranks → identical episode order, no coordination required.

EpisodePool
    Flat NVMe cache keyed by episode_hash at ``<cache_dir>/pool/<hash>/``.
    Tracks used bytes, enforces a hard capacity ceiling, and evicts only
    episodes outside the current+lookahead window (so episodes the
    DataLoader is about to consume are never deleted).

PoolFillerThread
    One persistent background thread on rank 0 only. Submits extractions
    to a fixed-size ThreadPoolExecutor, staying ``lookahead_epochs``
    ahead of the training cursor. Before each submission it checks pool
    capacity; if over budget it triggers eviction first and waits if no
    eviction is possible (training is still behind).

PrefetchedMapDataset
    Map-style Dataset that reads from the pool. ``prepare_epoch(epoch)``
    builds a frame-level ``_index_map`` for the current window, blocking
    until every episode is materialized. With ``persistent_workers=False``
    workers fork after this returns and inherit the index_map + warm zarr
    handles.

Disk safety
    The pool capacity is a hard byte ceiling. The filler refuses to start
    an extraction that would push usage past the ceiling and instead waits
    for the training cursor to advance and free episodes. ENOSPC is treated
    as a transient error: the partial directory is cleaned up and the
    episode is retried after the next eviction.

DDP correctness
    Only rank 0 runs the filler (RANK env var). Non-leader ranks just poll
    the ``.done`` sentinel on the shared NVMe mount, so all ranks observe
    the same set of ready episodes without IPC.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import random
import shutil
import tarfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from egomimic.rldb.zarr.zarr_dataset_multi import (
    EpisodeResolver,
    ZarrDataset,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalog entry
# ---------------------------------------------------------------------------

@dataclass
class EpisodeCatalogEntry:
    """Lightweight descriptor for one zipped episode on the zip volume."""

    tar_path: Path
    episode_hash: str
    n_frames: int
    embodiment: str = "mecka_bimanual"


# ---------------------------------------------------------------------------
# Bounds-check mixin  (mirrors MultiDataset._check_bounds exactly)
# ---------------------------------------------------------------------------

class _BoundsCheckMixin:
    """Shared bounds-check logic for map-style and iterable datasets."""

    def _check_bounds(
        self, data: dict, dataset, idx: int, dataset_name: str
    ) -> str | None:
        if self.data_schematic is None:
            return None

        embodiment_id = data.get("embodiment")
        if embodiment_id is None:
            raise ValueError("data has no embodiment metadata")

        norm_stats = self.data_schematic.norm_stats.get(embodiment_id, {})
        if not norm_stats:
            return None

        self._n_samples_checked += 1
        prefix: str | None = None

        for key_name, stats in norm_stats.items():
            zarr_key = self.data_schematic.keyname_to_zarr_key(key_name, embodiment_id)
            if zarr_key is None or zarr_key not in data:
                continue

            v = data[zarr_key]
            if isinstance(v, torch.Tensor):
                arr = v.float()
            elif isinstance(v, np.ndarray):
                arr = torch.from_numpy(v).float()
            else:
                continue

            slack = getattr(self, "bounds_slack", 0.0)
            q_low = torch.as_tensor(
                stats.get("quantile_0_01", stats.get("quantile_0_1", stats["quantile_1"])),
                dtype=torch.float32,
            ) - slack
            q_high = torch.as_tensor(
                stats.get("quantile_99_99", stats.get("quantile_99_9", stats["quantile_99"])),
                dtype=torch.float32,
            ) + slack

            try:
                q_low = torch.broadcast_to(q_low, arr.shape)
                q_high = torch.broadcast_to(q_high, arr.shape)
            except RuntimeError:
                key_sig = (str(zarr_key), tuple(arr.shape), tuple(q_low.shape))
                if not hasattr(self, "_shape_mismatch_warned"):
                    self._shape_mismatch_warned = set()
                if key_sig not in self._shape_mismatch_warned:
                    self._shape_mismatch_warned.add(key_sig)
                    logger.warning(
                        "Skipping bounds check for key=%s: value=%s q_low=%s",
                        zarr_key, tuple(arr.shape), tuple(q_low.shape),
                    )
                continue

            if torch.any(torch.isnan(arr)) or torch.any(torch.isinf(arr)):
                ep_name = Path(getattr(dataset, "episode_path", dataset_name)).name
                prefix = f"NaN/Inf violation ep={ep_name} frame={idx} key={zarr_key}"
                break

            if torch.any(arr < q_low) or torch.any(arr > q_high):
                ep_name = Path(getattr(dataset, "episode_path", dataset_name)).name
                prefix = f"Bounds violation ep={ep_name} frame={idx} key={zarr_key}"
                break

        if prefix is not None:
            self._n_violation_samples += 1

        return prefix


# ---------------------------------------------------------------------------
# ZipEpisodeResolver
# ---------------------------------------------------------------------------

class ZipEpisodeResolver(EpisodeResolver):
    """Resolves episodes from catalog.json on the zip volume."""

    CATALOG_FILENAME = "catalog.json"

    def __init__(
        self,
        zip_dir: Path | str,
        key_map: dict | None = None,
        transform_list: list | None = None,
        norm_stats: dict | None = None,
        pause_removal_epsilon: float | None = None,
        valid_ratio: float = 0.1,
        debug: int | None = None,
        min_frames: int | None = None,
        seed: int = 42,
    ):
        super().__init__(
            Path(zip_dir),
            key_map,
            transform_list,
            norm_stats=norm_stats,
            pause_removal_epsilon=pause_removal_epsilon,
        )
        self.zip_dir = Path(zip_dir)
        self.valid_ratio = valid_ratio
        self.debug = debug
        self.min_frames = min_frames
        self.seed = seed
        self._catalog: list[EpisodeCatalogEntry] | None = None

    def load_catalog(self) -> list[EpisodeCatalogEntry]:
        if self._catalog is not None:
            return self._catalog

        catalog_path = self.zip_dir / self.CATALOG_FILENAME
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog not found: {catalog_path}. "
                "Run `zip_zarr_to_vol.py` first to populate the zip volume."
            )

        with open(catalog_path) as f:
            raw: list[dict] = json.load(f)

        entries: list[EpisodeCatalogEntry] = []
        n_missing = 0
        for e in raw:
            tar_path = self.zip_dir / e["tar_filename"]
            if not tar_path.exists():
                n_missing += 1
                continue
            entries.append(
                EpisodeCatalogEntry(
                    tar_path=tar_path,
                    episode_hash=e["episode_hash"],
                    n_frames=int(e["n_frames"]),
                    embodiment=e.get("embodiment", "mecka_bimanual"),
                )
            )

        if n_missing:
            logger.warning(
                "ZipEpisodeResolver: %d catalog entries missing from zip volume (skipped)",
                n_missing,
            )

        if self.debug:
            entries = entries[: int(self.debug)]
            logger.info("ZipEpisodeResolver: debug=%d — using first %d episodes", self.debug, len(entries))

        if self.min_frames:
            before = len(entries)
            entries = [e for e in entries if e.n_frames >= self.min_frames]
            logger.info(
                "ZipEpisodeResolver: min_frames=%d — kept %d/%d episodes",
                self.min_frames, len(entries), before,
            )

        logger.info(
            "ZipEpisodeResolver: %d episodes, %d total frames from %s",
            len(entries),
            sum(e.n_frames for e in entries),
            catalog_path,
        )
        self._catalog = entries
        return self._catalog

    def split_catalog(self, mode: str) -> list[EpisodeCatalogEntry]:
        catalog = self.load_catalog()
        rng = random.Random(self.seed)
        shuffled = list(catalog)
        rng.shuffle(shuffled)
        n_valid = max(1, int(len(shuffled) * self.valid_ratio))
        if mode == "valid":
            return shuffled[:n_valid]
        return shuffled[n_valid:]

    def total_frames(self, mode: str = "train") -> int:
        return sum(e.n_frames for e in self.split_catalog(mode))

    def resolve(self, filters=None, **kwargs):
        raise NotImplementedError(
            "ZipEpisodeResolver does not support resolve(). "
            "Use PrefetchedMapDataset(resolver=...) instead."
        )


# ---------------------------------------------------------------------------
# EpisodePlan — deterministic episode ordering across the run
# ---------------------------------------------------------------------------

class EpisodePlan:
    """Fixed-order schedule of episodes across all epochs.

    The plan is a single shuffle of the full split (same seed → same order
    on every DDP rank). Epoch ``e`` consumes plan[e*ept : (e+1)*ept] with
    wrap-around. No re-shuffle between epochs; randomness comes from the
    frame-level shuffle in ``prepare_epoch`` and from the plan's initial
    permutation.
    """

    def __init__(
        self,
        episodes: list[EpisodeCatalogEntry],
        episodes_per_epoch: int,
        seed: int = 42,
    ):
        if not episodes:
            raise ValueError("EpisodePlan: empty episode list")
        if episodes_per_epoch <= 0:
            raise ValueError(f"EpisodePlan: episodes_per_epoch must be > 0, got {episodes_per_epoch}")

        rng = random.Random(seed)
        order = list(episodes)
        rng.shuffle(order)
        self.episodes: list[EpisodeCatalogEntry] = order
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.n = len(order)
        self._by_hash = {e.episode_hash: e for e in order}

    def epoch_episodes(self, epoch: int) -> list[EpisodeCatalogEntry]:
        """Episodes scheduled for ``epoch`` (with wrap-around)."""
        start = (epoch * self.episodes_per_epoch) % self.n
        out: list[EpisodeCatalogEntry] = []
        for i in range(self.episodes_per_epoch):
            out.append(self.episodes[(start + i) % self.n])
        return out

    def episodes_in_window(self, start_epoch: int, end_epoch_exclusive: int) -> list[EpisodeCatalogEntry]:
        """All distinct episodes scheduled in [start_epoch, end_epoch_exclusive)."""
        seen: set[str] = set()
        out: list[EpisodeCatalogEntry] = []
        for e in range(start_epoch, end_epoch_exclusive):
            for ep in self.epoch_episodes(e):
                if ep.episode_hash not in seen:
                    seen.add(ep.episode_hash)
                    out.append(ep)
        return out

    def hashes_in_window(self, start_epoch: int, end_epoch_exclusive: int) -> set[str]:
        return {ep.episode_hash for ep in self.episodes_in_window(start_epoch, end_epoch_exclusive)}

    def episode_at(self, plan_idx: int) -> EpisodeCatalogEntry:
        return self.episodes[plan_idx % self.n]


# ---------------------------------------------------------------------------
# EpisodePool — flat byte-tracked NVMe cache
# ---------------------------------------------------------------------------

class EpisodePool:
    """Hash-keyed NVMe cache with a hard byte ceiling and future-aware eviction.

    Layout: ``<cache_dir>/pool/<episode_hash>/`` with ``.done`` sentinel.
    Tracks bytes-on-disk for every cached episode; ``used_bytes`` returns
    the running sum without a directory walk.

    ``evict_outside(keep_hashes)`` deletes every cached episode whose hash
    is not in ``keep_hashes`` and returns the number of bytes freed.
    Callers pass the set of episodes scheduled in the current + lookahead
    epochs, so the DataLoader never sees an episode disappear from under it.
    """

    def __init__(self, cache_dir: Path | str, capacity_gb: float):
        self.root = Path(cache_dir) / "pool"
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity_bytes = int(capacity_gb * 1e9)
        self._sizes: dict[str, int] = {}
        self._lock = threading.Lock()
        self._scan_existing()

    def _scan_existing(self) -> None:
        """Rebuild the byte map from disk (crash recovery)."""
        n = 0
        n_bad = 0
        for ep_dir in self.root.iterdir():
            if not ep_dir.is_dir():
                continue
            if (ep_dir / ".bad").exists():
                # Permanent known-bad marker; keep but don't count for capacity.
                n_bad += 1
                continue
            if not (ep_dir / ".done").exists():
                # Half-finished extraction from a previous run; drop it.
                shutil.rmtree(ep_dir, ignore_errors=True)
                continue
            try:
                size = sum(f.stat().st_size for f in ep_dir.rglob("*") if f.is_file())
            except FileNotFoundError:
                shutil.rmtree(ep_dir, ignore_errors=True)
                continue
            self._sizes[ep_dir.name] = size
            n += 1
        if n_bad:
            logger.info("EpisodePool: %d episodes carry .bad marker (will be skipped)", n_bad)
        if n:
            logger.info(
                "EpisodePool: restored %d episodes (%.1f GB) from existing cache at %s",
                n, sum(self._sizes.values()) / 1e9, self.root,
            )

    def episode_path(self, ep_hash: str) -> Path:
        return self.root / ep_hash

    def is_ready(self, ep_hash: str) -> bool:
        return (self.root / ep_hash / ".done").exists()

    def is_bad(self, ep_hash: str) -> bool:
        """True if this episode has been marked permanently unextractable."""
        return (self.root / ep_hash / ".bad").exists()

    def used_bytes(self) -> int:
        with self._lock:
            return sum(self._sizes.values())

    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes()

    def register(self, ep_hash: str, size_bytes: int) -> None:
        with self._lock:
            self._sizes[ep_hash] = size_bytes

    def drop(self, ep_hash: str) -> int:
        """Remove ``ep_hash`` from the pool. Returns bytes freed."""
        with self._lock:
            size = self._sizes.pop(ep_hash, 0)
        shutil.rmtree(self.root / ep_hash, ignore_errors=True)
        return size

    def evict_outside(self, keep_hashes: set[str]) -> int:
        """Evict every cached hash not in ``keep_hashes``. Returns bytes freed."""
        with self._lock:
            victims = [h for h in self._sizes if h not in keep_hashes]
        freed = 0
        for h in victims:
            freed += self.drop(h)
        if victims:
            logger.info(
                "EpisodePool: evicted %d episodes (%.1f GB freed, %.1f GB used / %.0f GB cap)",
                len(victims), freed / 1e9,
                self.used_bytes() / 1e9, self.capacity_bytes / 1e9,
            )
        return freed


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_tar_to_dir(tar_path: Path, dest: Path) -> int:
    """Extract ``tar_path`` into ``dest`` and return total bytes written.

    Caller is responsible for creating/cleaning ``dest``, touching ``.done``,
    and registering the size with the pool.  Raises ``OSError`` (including
    ENOSPC, errno 28) on failure.
    """
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(path=dest)
    return sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())


def _acquire_extract_lock(pool_dir: Path, ep_hash: str) -> int | None:
    """Cross-process per-episode lock used during extraction.

    Returns the file descriptor on success (caller must release it), or
    ``None`` if another process is already extracting this episode (the
    caller should wait for the ``.done`` sentinel).
    """
    lock_path = pool_dir / f".lock_{ep_hash}"
    try:
        return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None


def _release_extract_lock(pool_dir: Path, ep_hash: str, fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        (pool_dir / f".lock_{ep_hash}").unlink(missing_ok=True)
    except OSError:
        pass


class _ENOSPCError(Exception):
    """Raised inside PoolFillerThread to swallow ENOSPC without spamming logs."""
    pass


# Module-level registry of running PoolFillerThreads keyed by absolute cache_dir.
# trainHydra.py instantiates the train dataset twice (once for the actual
# DataLoader, once briefly for norm-stats inference). Without this registry,
# both instances would start their own filler against the same pool directory
# and race on rmtree+extract for the same episodes.
_FILLER_REGISTRY: dict[str, "PoolFillerThread"] = {}
_FILLER_REGISTRY_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# PoolFillerThread — persistent extractor on rank 0
# ---------------------------------------------------------------------------

class PoolFillerThread(threading.Thread):
    """Keeps the pool filled ``lookahead_episodes`` ahead of training.

    Single thread orchestrates extraction; actual tar-extracts run in a
    fixed-size ThreadPoolExecutor (``n_copy_threads`` workers, matching
    the previous per-epoch pool size). The filler is "soft-paced" — when
    the pool is near capacity it stops submitting and waits for the
    training cursor to advance (which triggers eviction).

    DDP: instantiate ONLY on rank 0. Other ranks read the ``.done``
    sentinel directly via ``EpisodePool.is_ready``.
    """

    def __init__(
        self,
        pool: EpisodePool,
        plan: EpisodePlan,
        n_copy_threads: int,
        lookahead_episodes: int,
        capacity_target_ratio: float = 0.92,
    ):
        super().__init__(daemon=True, name="pool-filler")
        self.pool = pool
        self.plan = plan
        self.lookahead_episodes = int(lookahead_episodes)
        self.capacity_target_ratio = capacity_target_ratio
        self.n_copy_threads = int(n_copy_threads)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.n_copy_threads, thread_name_prefix="pool-extract"
        )
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._train_cursor = 0          # plan index that training has just consumed
        self._extract_cursor = 0        # next plan index to consider for extraction
        self._inflight: dict[str, concurrent.futures.Future] = {}
        # Running estimate of the largest observed episode size (bytes); used
        # to gate submissions when free space might not fit one extraction.
        self._max_observed_size = 200 * 1024 * 1024  # 200 MB seed
        # ENOSPC backoff state
        self._enospc_count = 0
        self._last_enospc_at = 0.0

    def stop(self) -> None:
        self._stop_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def advance_training_cursor(self, plan_idx: int) -> None:
        """Called by the Dataset when training enters a new epoch.

        If training jumped forward (e.g. resumed from a mid-run checkpoint)
        bump ``_extract_cursor`` along with it — otherwise the filler keeps
        crawling from plan index 0 toward a window that's already well past
        and prepare_epoch times out waiting on episodes the filler will not
        reach for hours.
        """
        with self._state_lock:
            if plan_idx > self._train_cursor:
                self._train_cursor = plan_idx
            if plan_idx > self._extract_cursor:
                # Skip ahead so the filler immediately starts extracting the
                # episodes the resumed training cursor needs next.
                self._extract_cursor = plan_idx

    def stats(self) -> dict:
        with self._state_lock:
            return {
                "train_cursor": self._train_cursor,
                "extract_cursor": self._extract_cursor,
                "inflight": len(self._inflight),
                "used_gb": self.pool.used_bytes() / 1e9,
                "cap_gb": self.pool.capacity_bytes / 1e9,
                "enospc_count": self._enospc_count,
            }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        last_stats_log = time.monotonic()
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("PoolFillerThread: tick failed; sleeping 2s")
                time.sleep(2.0)
            # Periodic stats so we can see if the filler is keeping up.
            now = time.monotonic()
            if now - last_stats_log > 30.0:
                s = self.stats()
                logger.info(
                    "PoolFiller stats: train_cursor=%d extract_cursor=%d "
                    "inflight=%d used=%.1f GB / %.0f GB enospc=%d",
                    s["train_cursor"], s["extract_cursor"], s["inflight"],
                    s["used_gb"], s["cap_gb"], s["enospc_count"],
                )
                last_stats_log = now

    def _tick(self) -> None:
        # 1. Don't outrun lookahead.
        with self._state_lock:
            target = self._train_cursor + self.lookahead_episodes
            cursor = self._extract_cursor
            inflight = len(self._inflight)

        # Cap concurrent in-flight extractions to ~2x worker count so the
        # executor queue doesn't grow unbounded.
        if inflight >= self.n_copy_threads * 2:
            time.sleep(0.05)
            return

        if cursor >= target:
            # We're already ahead; nothing to do.
            time.sleep(0.2)
            return

        ep = self.plan.episode_at(cursor)

        # Skip if already ready or in-flight.
        if self.pool.is_ready(ep.episode_hash):
            with self._state_lock:
                self._extract_cursor = cursor + 1
            return
        with self._state_lock:
            if ep.episode_hash in self._inflight:
                self._extract_cursor = cursor + 1
                return

        # 2. Disk gate — refuse to start if we'd push past capacity.
        used = self.pool.used_bytes()
        budget = int(self.pool.capacity_bytes * self.capacity_target_ratio)
        # Reserve headroom for the largest observed extraction plus inflight.
        headroom = self._max_observed_size * (1 + inflight)
        if used + headroom > budget:
            # Can't fit safely; back off and let the training cursor advance.
            time.sleep(0.5)
            return

        # 3. Submit.
        fut = self._executor.submit(self._extract_one, ep)
        with self._state_lock:
            self._inflight[ep.episode_hash] = fut
            self._extract_cursor = cursor + 1
        fut.add_done_callback(lambda f, h=ep.episode_hash: self._on_done(h, f))

    def _on_done(self, ep_hash: str, fut: concurrent.futures.Future) -> None:
        with self._state_lock:
            self._inflight.pop(ep_hash, None)
        exc = fut.exception()
        if exc is not None and not isinstance(exc, _ENOSPCError):
            logger.warning("PoolFiller: extraction of %s failed: %s", ep_hash, exc)

    def _mark_bad(self, ep_hash: str, reason: str) -> None:
        """Atomically replace the episode dir with a single .bad sentinel."""
        dest = self.pool.episode_path(ep_hash)
        shutil.rmtree(dest, ignore_errors=True)
        try:
            dest.mkdir(parents=True, exist_ok=True)
            (dest / ".bad").touch()
        except OSError:
            logger.warning("PoolFiller: failed to write .bad for %s", ep_hash)
        logger.warning("PoolFiller: %s marked .bad — %s", ep_hash, reason)

    def _extract_one(self, entry: EpisodeCatalogEntry) -> None:
        dest = self.pool.episode_path(entry.episode_hash)
        if (dest / ".done").exists():
            return
        if (dest / ".bad").exists():
            return  # known-bad tar; don't retry

        lock_fd = _acquire_extract_lock(self.pool.root, entry.episode_hash)
        if lock_fd is None:
            # Another process is already extracting this episode; let it finish.
            return

        try:
            # Always start from a clean directory (a previous attempt may have
            # left a partial extraction behind after ENOSPC).
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            try:
                size = _extract_tar_to_dir(entry.tar_path, dest)
            except OSError as e:
                shutil.rmtree(dest, ignore_errors=True)
                if e.errno == 28:  # ENOSPC — transient, no .bad
                    with self._state_lock:
                        self._enospc_count += 1
                        self._last_enospc_at = time.monotonic()
                    if self._enospc_count <= 4 or self._enospc_count % 50 == 0:
                        logger.warning(
                            "PoolFiller: ENOSPC extracting %s (count=%d, used=%.1f GB)",
                            entry.episode_hash, self._enospc_count, self.pool.used_bytes() / 1e9,
                        )
                    raise _ENOSPCError(entry.episode_hash)
                # Any other OSError (missing tar, IO error, etc.) is
                # permanent for this episode — mark .bad so prepare_epoch
                # doesn't wait forever.
                self._mark_bad(entry.episode_hash, f"OSError on extract: {e}")
                return
            except Exception as e:
                # tarfile.ReadError, EOFError, etc. — corrupt archive.
                shutil.rmtree(dest, ignore_errors=True)
                self._mark_bad(entry.episode_hash, f"{type(e).__name__} on extract: {e}")
                return

            # Validate: zarr v3 needs `zarr.json` at the group root.
            # zarr v2 layout uses `.zgroup`. Accept either.
            if not ((dest / "zarr.json").exists() or (dest / ".zgroup").exists()):
                self._mark_bad(entry.episode_hash, "no zarr group metadata after extract")
                return

            self.pool.register(entry.episode_hash, size)
            (dest / ".done").touch()

            with self._state_lock:
                if size > self._max_observed_size:
                    self._max_observed_size = size

            if time.perf_counter() - t0 > 30:
                logger.info(
                    "PoolFiller: slow extract %s (%.1fs, %.0f MB)",
                    entry.episode_hash, time.perf_counter() - t0, size / 1e6,
                )
        finally:
            _release_extract_lock(self.pool.root, entry.episode_hash, lock_fd)


# ---------------------------------------------------------------------------
# PrefetchedMapDataset (sliding-window pool)
# ---------------------------------------------------------------------------

class PrefetchedMapDataset(_BoundsCheckMixin, torch.utils.data.Dataset):
    """Map-style dataset backed by an NVMe ``EpisodePool``.

    Parameters
    ----------
    resolver:
        ``ZipEpisodeResolver`` providing the catalog and data transforms.
    mode:
        ``"train"`` or ``"valid"``.
    episodes_per_epoch:
        Number of episodes scheduled per epoch in the ``EpisodePlan``. The
        actual frame count per epoch is ``sum(ep.n_frames for ep in window)``;
        Lightning's ``limit_train_batches`` truncates as needed.
    epoch_frames:
        Optional target frame count (used only for ``__len__`` so
        ``DistributedSampler`` partitions consistently across ranks). When
        ``None`` the real index_map length is used after ``prepare_epoch``.
    pool_size_gb:
        Hard NVMe ceiling for the episode pool. Sized so that
        current + lookahead episodes always fit with headroom.
    lookahead_epochs:
        Filler stays this many epochs ahead of the training cursor.
        1.5 means: by the time training reaches epoch N, episodes for
        epochs N+1 and half of N+2 should already be extracted.
    cache_dir:
        Local NVMe directory. ``<cache_dir>/pool/<hash>/`` holds episodes.
    n_copy_threads:
        Worker threads in the extractor pool.
    seed:
        Used for the global episode permutation and the per-epoch frame
        shuffle. Same seed → identical order on every DDP rank.
    """

    _STALL_WARN_INTERVAL_S = 30.0

    def __init__(
        self,
        resolver: ZipEpisodeResolver,
        mode: str = "train",
        episodes_per_epoch: int | None = None,
        epoch_frames: int | None = None,
        pool_size_gb: float = 150.0,
        lookahead_epochs: float = 1.5,
        cache_dir: str | Path = "/cache/zarr_cache",
        n_copy_threads: int = 16,
        seed: int = 42,
    ):
        super().__init__()
        self.resolver = resolver
        self.mode = mode
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_copy_threads = int(n_copy_threads)
        self.seed = seed

        self._episodes = resolver.split_catalog(mode)
        if not self._episodes:
            raise RuntimeError(f"Resolver returned 0 episodes for mode={mode}")

        total_frames = sum(e.n_frames for e in self._episodes)
        avg_frames_per_ep = max(1, total_frames // len(self._episodes))

        # Default episodes_per_epoch: enough episodes to cover epoch_frames
        # with a 20% safety margin (selecting variable-length episodes by
        # count can land short on the average). If neither is given, use
        # the whole split (matches the legacy behaviour).
        if episodes_per_epoch is None:
            if epoch_frames is not None:
                episodes_per_epoch = max(
                    1, int((epoch_frames * 1.2) // avg_frames_per_ep) + 1
                )
            else:
                episodes_per_epoch = len(self._episodes)
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.epoch_frames = (
            int(epoch_frames) if epoch_frames is not None
            else int(self.episodes_per_epoch * avg_frames_per_ep)
        )

        # Build the deterministic plan + pool + filler.
        self._plan = EpisodePlan(self._episodes, self.episodes_per_epoch, seed=seed)
        self._pool = EpisodePool(self.cache_dir, capacity_gb=pool_size_gb)
        self._lookahead_episodes = max(
            self.episodes_per_epoch,
            int(lookahead_epochs * self.episodes_per_epoch),
        )

        # Filler only on local rank 0 (shared NVMe across ranks).
        # Lightning sets LOCAL_RANK before spawning each DDP child; the global
        # RANK env var is only set after dist.init_process_group(), which runs
        # later in Trainer.fit(). Reading RANK here would default to "0" for
        # every child and start N fillers racing on the same pool directory.
        self._rank = int(
            os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))
        )
        self._is_filler_rank = self._rank == 0
        self._filler: PoolFillerThread | None = None
        if self.mode == "train" and self._is_filler_rank:
            registry_key = str(self.cache_dir.resolve())
            with _FILLER_REGISTRY_LOCK:
                existing = _FILLER_REGISTRY.get(registry_key)
                if existing is not None and existing.is_alive():
                    self._filler = existing
                    # Reuse the existing pool's tracking, otherwise our new
                    # _pool will register sizes the filler doesn't see.
                    self._pool = existing.pool
                    logger.info(
                        "PoolFillerThread already running for %s — reusing",
                        registry_key,
                    )
                else:
                    self._filler = PoolFillerThread(
                        pool=self._pool,
                        plan=self._plan,
                        n_copy_threads=self.n_copy_threads,
                        lookahead_episodes=self._lookahead_episodes,
                    )
                    self._filler.start()
                    _FILLER_REGISTRY[registry_key] = self._filler
                    logger.info(
                        "PoolFillerThread started: lookahead=%d eps, n_copy_threads=%d, "
                        "pool_capacity=%.0f GB",
                        self._lookahead_episodes, self.n_copy_threads, pool_size_gb,
                    )

        # Valid mode does not start a filler. prepare_epoch synchronously
        # extracts only the current window (rank 0); non-leader ranks poll
        # the shared .done sentinels written by rank 0.

        # Per-worker ZarrDataset cache (fork-inherited after prepare_epoch).
        self._zarr_cache: dict[str, ZarrDataset] = {}

        # Per-epoch index_map (set in prepare_epoch).
        self._index_map: list[tuple[str, int]] | None = None
        self._current_epoch: int = -1

        # Probe path for shape/norm inference before prepare_epoch.
        self._probe_zarr_path: str | None = None

        # Disk-synced epoch tracking for persistent_workers=True.
        # Main writes the current epoch atomically; workers stat the file
        # each __getitem__ and deterministically rebuild their local
        # index_map (the plan is identical across all ranks/workers, so
        # the same shuffle seed yields the same order).
        self._epoch_file = self.cache_dir / f"current_epoch_{mode}.txt"
        self._worker_epoch_loaded: int = -1
        # Track the lookahead-window keep_set per worker so we can drop
        # stale episodes from _zarr_cache when the window slides forward.
        self._worker_keep_paths: set[str] | None = None

        # Bounds-check state (_BoundsCheckMixin)
        self.data_schematic = None
        self.bounds_slack: float = 0.0
        self._n_samples_checked: int = 0
        self._n_violation_samples: int = 0
        self._violation_log_every: int = 100
        self._shape_mismatch_warned: set = set()

        logger.info(
            "PrefetchedMapDataset [%s]: %d episodes, %d total frames, "
            "episodes_per_epoch=%d, lookahead_epochs=%.1f, "
            "pool_size=%.0f GB, cache_dir=%s, rank=%d (filler=%s)",
            mode, len(self._episodes), total_frames,
            self.episodes_per_epoch, lookahead_epochs,
            pool_size_gb, cache_dir, self._rank,
            "yes" if self._is_filler_rank else "no",
        )

    # ------------------------------------------------------------------
    # DataSchematic wiring (matches MultiDataset API)
    # ------------------------------------------------------------------

    def set_data_schematic(self, data_schematic, bounds_slack: float = 0.0) -> None:
        self.data_schematic = data_schematic
        self.bounds_slack = bounds_slack
        if (
            hasattr(data_schematic, "norm_stats")
            and self.resolver.norm_stats is None
        ):
            self.resolver.norm_stats = data_schematic.norm_stats
        logger.info(
            "PrefetchedMapDataset [%s]: data_schematic set (bounds_slack=%.4f)",
            self.mode, bounds_slack,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return self.epoch_frames * world_size

    def __getitem__(self, idx: int) -> dict:
        # In worker context, check the epoch file and rebuild the local
        # index_map deterministically when the main process has moved on.
        # This is what makes persistent_workers=True viable: workers live
        # across epochs and refresh themselves rather than respawning.
        if torch.utils.data.get_worker_info() is not None:
            self._maybe_reload_worker_epoch()

        if self._index_map is None:
            # Pre-prepare_epoch path: shape + norm-stats inference via probe.
            if self._probe_zarr_path is None:
                self._probe_zarr_path = self._extract_probe()
            if not hasattr(self, "_probe_ds"):
                self._probe_ds = ZarrDataset(
                    self._probe_zarr_path,
                    key_map=self.resolver.key_map,
                    transform_list=self.resolver.transform_list,
                    norm_stats=self.resolver.norm_stats,
                    pause_removal_epsilon=self.resolver.pause_removal_epsilon,
                )
            return self._probe_ds[idx % len(self._probe_ds)]

        # Hot path: episode is in NVMe and zarr handle is cached. Try up to
        # a few times to find a usable frame, falling back to a different
        # idx if the chosen episode has been evicted under us (which can
        # happen briefly at epoch boundaries with persistent_workers).
        attempts = 0
        while attempts < 4:
            ep_path, frame_idx = self._index_map[idx % len(self._index_map)]
            if Path(ep_path, ".done").exists():
                if ep_path not in self._zarr_cache:
                    try:
                        self._zarr_cache[ep_path] = ZarrDataset(
                            ep_path,
                            key_map=self.resolver.key_map,
                            transform_list=self.resolver.transform_list,
                            norm_stats=self.resolver.norm_stats,
                            pause_removal_epsilon=self.resolver.pause_removal_epsilon,
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to open %s (%s); skipping to next frame",
                            Path(ep_path).name, e,
                        )
                        idx = (idx + 1) % len(self._index_map)
                        attempts += 1
                        continue
                return self._zarr_cache[ep_path][frame_idx]
            # Evicted episode (eviction race at epoch boundary). Skip to
            # next frame in the shuffled index_map.
            idx = (idx + 1) % len(self._index_map)
            attempts += 1

        # Last resort: bounded wait, then surface whatever's available.
        self._wait_for_episode_path(ep_path)
        if ep_path not in self._zarr_cache:
            self._zarr_cache[ep_path] = ZarrDataset(
                ep_path,
                key_map=self.resolver.key_map,
                transform_list=self.resolver.transform_list,
                norm_stats=self.resolver.norm_stats,
                pause_removal_epsilon=self.resolver.pause_removal_epsilon,
            )
        return self._zarr_cache[ep_path][frame_idx]

    def _wait_for_episode_path(self, ep_path: str) -> None:
        """Defensive stall — should never fire under correct sizing."""
        done = Path(ep_path) / ".done"
        t0 = time.perf_counter()
        last_warn = t0
        while not done.exists():
            time.sleep(0.05)
            now = time.perf_counter()
            if now - last_warn > self._STALL_WARN_INTERVAL_S:
                logger.warning(
                    "Dataloader stalled %.1fs waiting for %s — pool sizing may be too small",
                    now - t0, ep_path,
                )
                last_warn = now

    # ------------------------------------------------------------------
    # Valid-mode extraction (rank 0 only, synchronous per epoch)
    # ------------------------------------------------------------------

    def _extract_episode_into_pool(self, entry: EpisodeCatalogEntry) -> None:
        """Extract one episode into the pool, registering its size and
        touching the ``.done`` sentinel. Idempotent (no-op if already done).

        Cross-process safe via an O_EXCL lock — only one extractor (across
        any number of ranks/datasets) writes a given episode at a time.
        """
        if self._pool.is_ready(entry.episode_hash):
            return
        lock_fd = _acquire_extract_lock(self._pool.root, entry.episode_hash)
        if lock_fd is None:
            return  # another process is already extracting
        try:
            if self._pool.is_ready(entry.episode_hash):
                return  # someone else finished while we waited for the lock
            dest = self._pool.episode_path(entry.episode_hash)
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)
            try:
                size = _extract_tar_to_dir(entry.tar_path, dest)
            except Exception:
                shutil.rmtree(dest, ignore_errors=True)
                raise
            if not ((dest / "zarr.json").exists() or (dest / ".zgroup").exists()):
                shutil.rmtree(dest, ignore_errors=True)
                raise RuntimeError(
                    f"valid-extract {entry.episode_hash}: no zarr group metadata found"
                )
            self._pool.register(entry.episode_hash, size)
            (dest / ".done").touch()
        finally:
            _release_extract_lock(self._pool.root, entry.episode_hash, lock_fd)

    def _extract_window_sync(self, window_eps: list[EpisodeCatalogEntry]) -> None:
        """Synchronously extract any missing episodes in ``window_eps``.

        Used by valid mode (no background filler). Blocks until every
        episode is on NVMe with ``.done`` set. Logs failures and continues —
        a single bad tar shouldn't fail the whole validation pass.
        """
        missing = [ep for ep in window_eps if not self._pool.is_ready(ep.episode_hash)]
        if not missing:
            return
        logger.info(
            "Valid extractor: extracting %d/%d missing window episodes (%d threads)",
            len(missing), len(window_eps), self.n_copy_threads,
        )
        n_ok = n_err = 0
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.n_copy_threads, thread_name_prefix="valid-extract"
        ) as ex:
            futs = {ex.submit(self._extract_episode_into_pool, ep): ep for ep in missing}
            for fut in concurrent.futures.as_completed(futs):
                if fut.exception() is not None:
                    ep = futs[fut]
                    logger.warning(
                        "Valid extractor: %s failed: %s", ep.episode_hash, fut.exception()
                    )
                    n_err += 1
                else:
                    n_ok += 1
        logger.info("Valid extractor: window ready (%d ok, %d failed).", n_ok, n_err)

    # ------------------------------------------------------------------
    # Epoch lifecycle — called from Lightning callback on main process
    # ------------------------------------------------------------------

    def prepare_epoch(self, epoch: int) -> None:
        """Materialize this epoch's window and build the frame index_map.

        Side effects:
          * Advances the filler's training cursor to ``epoch * episodes_per_epoch``.
          * Evicts episodes outside the [current, current + lookahead) window.
          * Blocks until every episode in this epoch's window is ready on NVMe.
          * Rebuilds ``self._index_map`` for the new window.

        Workers fork after this returns and inherit the new state via fork
        (``persistent_workers=False`` is required).
        """
        self._current_epoch = epoch
        gen_label = epoch + 1
        full_window_eps = self._plan.epoch_episodes(epoch)
        # Episodes previously found to be unextractable (.bad sentinel) are
        # excluded here so neither the wait loop nor the index_map ever sees
        # them. The filler also skips them (.bad short-circuits _extract_one).
        window_eps = [e for e in full_window_eps if not self._pool.is_bad(e.episode_hash)]
        n_bad_in_window = len(full_window_eps) - len(window_eps)
        if n_bad_in_window:
            logger.warning(
                "prepare_epoch %d: skipping %d known-bad episodes from window",
                gen_label, n_bad_in_window,
            )
        ep_hashes = {e.episode_hash for e in window_eps}

        # Calculate how many epochs of lookahead in terms of epochs (rounded up).
        lookahead_epochs = max(1, (self._lookahead_episodes + self.episodes_per_epoch - 1)
                               // self.episodes_per_epoch)
        # Include the PREVIOUS epoch in keep_hashes: persistent_workers may
        # still hold in-flight batches from epoch N-1 in their prefetch
        # queue when prepare_epoch(N) runs eviction. Without the prev-epoch
        # buffer those workers would stall indefinitely on .done files that
        # just got removed. Costs at most 1 epoch (~900 eps) of extra disk.
        prev_epoch = max(0, epoch - 1)
        keep_hashes = self._plan.hashes_in_window(prev_epoch, epoch + 1 + lookahead_epochs)

        # Publish the new epoch FIRST so worker __getitem__ calls can rebuild
        # their local index_map before any eviction touches disk. With the
        # prev-epoch buffer above this is belt-and-suspenders.
        if self._is_filler_rank:
            tmp = self._epoch_file.with_suffix(".tmp")
            try:
                tmp.write_text(str(epoch))
                os.replace(tmp, self._epoch_file)
            except OSError as e:
                logger.warning("Failed to publish epoch file %s: %s", self._epoch_file, e)

        # Advance the filler's training cursor (rank 0 only).
        if self._filler is not None:
            self._filler.advance_training_cursor(epoch * self.episodes_per_epoch)

        # Evict episodes outside the current+lookahead window. Train mode
        # makes room for the filler to stage future episodes; valid mode
        # makes room for the synchronous extract below.
        if self._is_filler_rank:
            freed = self._pool.evict_outside(keep_hashes)
            if freed:
                logger.info(
                    "prepare_epoch %d: freed %.1f GB outside current window",
                    gen_label, freed / 1e9,
                )

        # Valid mode has no background filler; extract the window inline.
        # Cheap because validation runs rarely (check_val_every_n_epoch=200).
        if self.mode == "valid" and self._is_filler_rank:
            self._extract_window_sync(window_eps)

        # Drop stale ZarrDataset handles (episodes that left the window).
        new_paths = {str(self._pool.episode_path(h)) for h in ep_hashes}
        self._zarr_cache = {k: v for k, v in self._zarr_cache.items() if k in new_paths}

        # Block until every episode in this window is materialized. Episodes
        # that get marked .bad mid-wait (e.g. by the filler discovering a
        # corrupt tar after we entered this loop) are dropped from the window
        # so we don't time out.
        t0 = time.monotonic()
        ready_count = 0
        bad_during_wait: set[str] = set()
        last_log = t0
        deadline = t0 + 3600  # 1 hour
        pending = list(window_eps)
        while pending:
            still_pending = []
            for ep in pending:
                if self._pool.is_ready(ep.episode_hash):
                    ready_count += 1
                elif self._pool.is_bad(ep.episode_hash):
                    bad_during_wait.add(ep.episode_hash)
                else:
                    still_pending.append(ep)
            pending = still_pending

            if not pending:
                break
            now = time.monotonic()
            if now > deadline:
                raise RuntimeError(
                    f"prepare_epoch: timeout (1h) waiting for {len(pending)} episodes "
                    f"for epoch {gen_label}"
                )
            if now - last_log > 30.0:
                stats = self._filler.stats() if self._filler is not None else {}
                logger.info(
                    "prepare_epoch %d: %d/%d episodes ready (%.0fs) %s",
                    gen_label, ready_count, len(window_eps), now - t0,
                    f"filler={stats}" if stats else "",
                )
                last_log = now
            time.sleep(0.25)

        # If any episodes turned out to be bad during the wait, drop them
        # from window_eps/ep_hashes so the index_map and pre-warm don't
        # reference them.
        if bad_during_wait:
            window_eps = [e for e in window_eps if e.episode_hash not in bad_during_wait]
            ep_hashes -= bad_during_wait
            new_paths = {str(self._pool.episode_path(h)) for h in ep_hashes}
            logger.warning(
                "prepare_epoch %d: %d additional episodes marked .bad during wait — dropped",
                gen_label, len(bad_during_wait),
            )

        wait_s = time.monotonic() - t0
        if wait_s > 0.5:
            logger.info(
                "prepare_epoch %d: %d episodes ready in %.1fs",
                gen_label, len(window_eps), wait_s,
            )

        # Build frame-level index_map (deterministic per-epoch shuffle).
        t_build = time.perf_counter()
        index_map: list[tuple[str, int]] = []
        for ep in window_eps:
            ep_path = str(self._pool.episode_path(ep.episode_hash))
            for fi in range(ep.n_frames):
                index_map.append((ep_path, fi))
        if self.mode == "train":
            rng = random.Random(self.seed + epoch + 99999)
            rng.shuffle(index_map)
        self._index_map = index_map
        logger.info(
            "[Timing] prepare_epoch %d: index_map built in %.3fs (%d frames)",
            gen_label, time.perf_counter() - t_build, len(self._index_map),
        )

        # Pre-open zarr stores in the main process so workers inherit warm
        # handles via fork. Eliminates the cold-open stall on batch 0.
        # An occasional episode may have a corrupt extraction (missing
        # zarr.json, partial tar, etc.) — drop those rather than crash the
        # whole epoch. They'll be re-extracted on the next pass.
        t_open = time.perf_counter()
        broken: set[str] = set()
        for ep_path in new_paths:
            if ep_path in self._zarr_cache:
                continue
            try:
                self._zarr_cache[ep_path] = ZarrDataset(
                    ep_path,
                    key_map=self.resolver.key_map,
                    transform_list=self.resolver.transform_list,
                    norm_stats=self.resolver.norm_stats,
                    pause_removal_epsilon=self.resolver.pause_removal_epsilon,
                )
            except Exception as e:
                logger.warning(
                    "prepare_epoch %d: dropping broken episode %s (%s: %s)",
                    gen_label, Path(ep_path).name, type(e).__name__, e,
                )
                broken.add(ep_path)
                # Remove the bad pool entry so filler may re-extract.
                if self._is_filler_rank:
                    h = Path(ep_path).name
                    self._pool.drop(h)

        if broken:
            before = len(self._index_map)
            self._index_map = [
                (p, fi) for (p, fi) in self._index_map if p not in broken
            ]
            logger.warning(
                "prepare_epoch %d: skipped %d broken episodes, index_map shrank %d → %d",
                gen_label, len(broken), before, len(self._index_map),
            )

        logger.info(
            "[Timing] prepare_epoch %d: pre-opened %d zarr stores in %.3fs",
            gen_label, len(new_paths) - len(broken), time.perf_counter() - t_open,
        )

        # Publish the new epoch number atomically so persistent_workers can
        # refresh themselves on the next __getitem__. Atomic via os.replace.
        if self._is_filler_rank:
            tmp = self._epoch_file.with_suffix(".tmp")
            try:
                tmp.write_text(str(epoch))
                os.replace(tmp, self._epoch_file)
            except OSError as e:
                logger.warning("Failed to publish epoch file %s: %s", self._epoch_file, e)

    # ------------------------------------------------------------------
    # Persistent-worker epoch sync
    # ------------------------------------------------------------------

    def _maybe_reload_worker_epoch(self) -> None:
        """Called from worker __getitem__. Stat the epoch file and, if the
        epoch advanced since this worker last refreshed, rebuild the local
        ``_index_map`` deterministically and evict zarr stores that are no
        longer in the lookahead window."""
        try:
            with open(self._epoch_file) as f:
                published_epoch = int(f.read().strip())
        except (OSError, ValueError):
            return
        if published_epoch == self._worker_epoch_loaded:
            return

        # Build the same window the main process built. The plan is
        # deterministic (same seed across all processes), so the resulting
        # (ep_path, frame_idx) tuples match main's index_map exactly.
        full_window_eps = self._plan.epoch_episodes(published_epoch)
        window_eps = [e for e in full_window_eps if not self._pool.is_bad(e.episode_hash)]
        index_map: list[tuple[str, int]] = []
        for ep in window_eps:
            ep_path = str(self._pool.episode_path(ep.episode_hash))
            for fi in range(ep.n_frames):
                index_map.append((ep_path, fi))
        if self.mode == "train":
            rng = random.Random(self.seed + published_epoch + 99999)
            rng.shuffle(index_map)
        self._index_map = index_map

        # Evict zarr handles for episodes not in the current+lookahead
        # window so the worker's cache doesn't grow without bound.
        lookahead_epochs = max(
            1,
            (self._lookahead_episodes + self.episodes_per_epoch - 1)
            // self.episodes_per_epoch,
        )
        keep_hashes = self._plan.hashes_in_window(
            published_epoch, published_epoch + 1 + lookahead_epochs
        )
        keep_paths = {str(self._pool.episode_path(h)) for h in keep_hashes}
        if self._worker_keep_paths is None or self._worker_keep_paths != keep_paths:
            self._zarr_cache = {
                k: v for k, v in self._zarr_cache.items() if k in keep_paths
            }
            self._worker_keep_paths = keep_paths

        self._worker_epoch_loaded = published_epoch

    # ------------------------------------------------------------------
    # Probe extraction for shape / norm-stats inference
    # ------------------------------------------------------------------

    def _extract_probe(self) -> str:
        """Extract the plan's first episode to a stable probe directory.

        DDP-safe: uses O_CREAT|O_EXCL lock so only one rank extracts;
        others wait for the done sentinel. Probe lives outside the pool
        so it never gets evicted.
        """
        if not self._episodes:
            raise RuntimeError("ZipEpisodeResolver returned an empty catalog.")
        entry = self._plan.episodes[0]
        probe_dir = self.cache_dir / "probe" / entry.episode_hash
        done_file = probe_dir / ".done"

        if done_file.exists():
            return str(probe_dir)

        lock_dir = self.cache_dir / "probe"
        lock_dir.mkdir(parents=True, exist_ok=True)
        lock_path = lock_dir / f".probe_lock_{entry.episode_hash}"
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            probe_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Probe: extracting %s", entry.tar_path.name)
            _extract_tar_to_dir(entry.tar_path, probe_dir)
            done_file.touch()
        except FileExistsError:
            deadline = time.monotonic() + 120
            while not done_file.exists():
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Probe extraction timed out after 120s at {probe_dir}"
                    )
                time.sleep(0.1)

        return str(probe_dir)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        try:
            if getattr(self, "_filler", None) is not None:
                self._filler.stop()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------

PrefetchedIterableDataset = PrefetchedMapDataset
