"""Map-style dataset reading zarr episodes directly from R2 -- no tar-extract,
no per-epoch materialization wait.

Unlike ``PrefetchedMapDataset`` (bounded local NVMe pool populated by
extracting tars, evicted/refilled at each epoch boundary via
``PoolFillerThread``), this dataset samples across the *entire* corpus every
epoch and reads chunks on demand through
``egomimic.rldb.zarr.remote_store``'s cached fsspec filesystem. There is no
"stage this epoch's window, then train" barrier: ``prepare_epoch()`` is cheap
and non-blocking, and a background thread continuously reads ahead of the
DataLoader's current position so R2 chunks are already warm in the local disk
cache by the time they're actually needed, instead of the trainer blocking on
a synchronous extract phase between every epoch.

``episodes_per_epoch`` / ``pool_size_gb`` from the tar-pool config have no
equivalent here -- the whole split is always in scope; fsspec's own
``filecache`` layer does size-bounded LRU eviction on local disk
independently of any notion of "epoch".

NEW, not yet exercised against a live Modal run. Before trusting this for a
real training job:
  1. Run ``egomimic.rldb.zarr.remote_store.verify_remote_zarr()`` against one
     converted episode.
  2. Smoke-test with a few hundred steps and ``num_workers=0`` first, then
     with ``num_workers>0`` to confirm the fork-safety fix in
     ``remote_store._cached_async_fs`` (keyed by (cache_dir, pid)) actually
     holds up under real DataLoader worker forking.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import threading
import time

import numpy as np
import torch
import torch.utils.data

from egomimic.rldb.zarr.prefetch.bounds import _BoundsCheckMixin
from egomimic.rldb.zarr.prefetch.catalog import R2ZarrEpisodeResolver, RemoteEpisodeCatalogEntry
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

logger = logging.getLogger(__name__)


class _CacheWarmer(threading.Thread):
    """Continuously reads ahead of the dataset's current episode cursor,
    warming the local fsspec disk cache for upcoming episodes.

    Unlike ``PoolFillerThread``, this never goes idle between epochs -- it
    keeps running for the dataset's whole lifetime, since there is no
    per-epoch materialization barrier to wait on in the first place.
    """

    def __init__(self, dataset: "RemoteZarrMapDataset", ahead_episodes: int, n_threads: int):
        super().__init__(daemon=True, name="R2ZarrCacheWarmer")
        self._ds = dataset
        self._ahead_episodes = ahead_episodes
        self._n_threads = max(1, n_threads)
        self._stop_evt = threading.Event()
        self._warmed: set[str] = set()
        self._lock = threading.Lock()

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._n_threads) as pool:
            while not self._stop_evt.is_set():
                targets = self._ds._upcoming_episodes(self._ahead_episodes)
                pending = []
                with self._lock:
                    for entry in targets:
                        if entry.episode_hash in self._warmed:
                            continue
                        self._warmed.add(entry.episode_hash)
                        pending.append(entry)
                if not pending:
                    time.sleep(1.0)
                    continue
                futs = [pool.submit(self._warm_one, e) for e in pending]
                concurrent.futures.wait(futs)

    def _warm_one(self, entry: RemoteEpisodeCatalogEntry) -> None:
        if self._stop_evt.is_set():
            return
        try:
            ds = self._ds._open_episode(entry)
            n = len(ds)
            if n <= 0:
                return
            step = max(1, n // 32)  # sample across the episode, not just frame 0
            for i in range(0, n, step):
                if self._stop_evt.is_set():
                    return
                ds[i]
        except Exception as e:
            logger.warning("CacheWarmer: failed to warm %s (%s)", entry.episode_hash, e)

    def note_epoch_advance(self) -> None:
        """Forget warmed-but-stale entries periodically so episodes revisited
        much later (after fsspec's own cache may have evicted them) get
        re-warmed instead of being skipped forever."""
        with self._lock:
            if len(self._warmed) > 4 * max(1, self._ahead_episodes):
                self._warmed.clear()


class RemoteZarrMapDataset(_BoundsCheckMixin, torch.utils.data.Dataset):
    """Map-style dataset over the entire R2-backed corpus, no rotating pool.

    Parameters
    ----------
    resolver:
        ``R2ZarrEpisodeResolver`` providing the catalog and data transforms.
    mode:
        ``"train"`` or ``"valid"``.
    cache_warm_ahead_episodes:
        How many upcoming episodes the background warmer keeps primed in the
        local fsspec cache. 0 disables warming (every read is a cold R2 fetch
        on first touch of a chunk).
    cache_warm_threads:
        Concurrency for the warmer's read-ahead.
    """

    def __init__(
        self,
        resolver: R2ZarrEpisodeResolver,
        mode: str = "train",
        seed: int = 42,
        cache_warm_ahead_episodes: int = 200,
        cache_warm_threads: int = 16,
    ):
        self.resolver = resolver
        self.mode = mode
        self.seed = seed
        self.data_schematic = None
        self.bounds_slack: float = 0.0
        self._n_samples_checked: int = 0

        self._episodes: list[RemoteEpisodeCatalogEntry] = resolver.split_catalog(mode)
        if not self._episodes:
            raise ValueError(f"RemoteZarrMapDataset [{mode}]: empty split from resolver")

        # Fixed shuffle order for the whole run (deterministic per seed, same
        # on every DDP rank) -- reshuffled per-epoch only at the frame level
        # via idx % total_frames + torch's own sampler shuffle, exactly like
        # PrefetchedMapDataset relies on RandomSampler over __len__.
        rng = random.Random(seed)
        self._order: list[RemoteEpisodeCatalogEntry] = list(self._episodes)
        rng.shuffle(self._order)

        self._cum_frames = np.cumsum([0] + [e.n_frames for e in self._order])
        self._total_frames = int(self._cum_frames[-1])

        self._zarr_cache: dict[str, ZarrDataset] = {}
        self._zarr_cache_lock = threading.Lock()
        self._zarr_cache_cap = 256  # bounded handle cache; fsspec owns disk-level LRU

        self._epoch = 0
        self._cursor_lock = threading.Lock()
        self._episode_cursor = 0  # where the warmer currently reads ahead from

        self._warmer: _CacheWarmer | None = None
        if mode == "train" and cache_warm_ahead_episodes > 0:
            self._warmer = _CacheWarmer(self, cache_warm_ahead_episodes, cache_warm_threads)
            self._warmer.start()

        logger.info(
            "RemoteZarrMapDataset [%s]: %d episodes, %d total frames, "
            "cache_warm_ahead=%d threads=%d",
            mode, len(self._order), self._total_frames,
            cache_warm_ahead_episodes, cache_warm_threads,
        )

    # ------------------------------------------------------------------
    # Compatibility surface expected by trainHydra.py / modal/callbacks.py
    # ------------------------------------------------------------------

    def set_data_schematic(self, data_schematic, bounds_slack: float = 0.0) -> None:
        self.data_schematic = data_schematic
        self.bounds_slack = bounds_slack
        if hasattr(data_schematic, "norm_stats") and self.resolver.norm_stats is None:
            self.resolver.norm_stats = data_schematic.norm_stats
        logger.info(
            "RemoteZarrMapDataset [%s]: data_schematic set (bounds_slack=%.4f)",
            self.mode, bounds_slack,
        )

    def prepare_epoch(self, epoch: int) -> None:
        """Called by PrefetchEpochCallback exactly like PrefetchedMapDataset,
        but here it's cheap and non-blocking -- there is nothing to
        materialize. Just moves the warmer's read-ahead cursor forward so it
        keeps tracking roughly where training is in the corpus."""
        self._epoch = epoch
        with self._cursor_lock:
            step = max(1, len(self._order) // 20)
            self._episode_cursor = (epoch * step) % len(self._order)
        if self._warmer is not None:
            self._warmer.note_epoch_advance()

    # ------------------------------------------------------------------
    # Internals shared with _CacheWarmer
    # ------------------------------------------------------------------

    def _upcoming_episodes(self, ahead: int) -> list[RemoteEpisodeCatalogEntry]:
        with self._cursor_lock:
            start = self._episode_cursor
        n = len(self._order)
        ahead = min(ahead, n)
        return [self._order[(start + i) % n] for i in range(ahead)]

    def _open_episode(self, entry: RemoteEpisodeCatalogEntry) -> ZarrDataset:
        with self._zarr_cache_lock:
            ds = self._zarr_cache.get(entry.episode_hash)
            if ds is not None:
                return ds
        ds = ZarrDataset(
            entry.zarr_url,
            key_map=self.resolver.key_map,
            transform_list=self.resolver.transform_list,
            norm_stats=self.resolver.norm_stats,
            pause_removal_epsilon=self.resolver.pause_removal_epsilon,
        )
        with self._zarr_cache_lock:
            if len(self._zarr_cache) >= self._zarr_cache_cap:
                # Cheap to reopen later -- fsspec keeps the actual bytes
                # cached on local disk regardless of this handle cache.
                stale_hash = next(iter(self._zarr_cache), None)
                stale = self._zarr_cache.pop(stale_hash, None) if stale_hash else None
                if stale is not None:
                    try:
                        stale.close()
                    except Exception:
                        pass
            self._zarr_cache[entry.episode_hash] = ds
        return ds

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return self._total_frames * world_size

    def __getitem__(self, idx: int) -> dict:
        idx = idx % self._total_frames
        ep_idx = int(np.searchsorted(self._cum_frames, idx, side="right") - 1)
        entry = self._order[ep_idx]
        frame_idx = idx - int(self._cum_frames[ep_idx])

        attempts = 0
        while attempts < 8:
            try:
                ds = self._open_episode(entry)
                return ds[frame_idx]
            except Exception as e:
                logger.warning(
                    "RemoteZarrMapDataset: read failed for %s frame %s (%s); "
                    "skipping to a different episode",
                    entry.episode_hash, frame_idx, e,
                )
                with self._zarr_cache_lock:
                    self._zarr_cache.pop(entry.episode_hash, None)
                ep_idx = (ep_idx + 1) % len(self._order)
                entry = self._order[ep_idx]
                frame_idx = frame_idx % max(1, entry.n_frames)
                attempts += 1
        raise RuntimeError(f"RemoteZarrMapDataset: repeated read failures near idx={idx}")

    def __del__(self):
        warmer = getattr(self, "_warmer", None)
        if warmer is not None:
            warmer.stop()
