"""Rank-0 background thread that stages episodes ahead of the training cursor."""

from __future__ import annotations

import concurrent.futures
import logging
import shutil
import threading
import time

from egomimic.rldb.zarr.prefetch.catalog import EpisodeCatalogEntry
from egomimic.rldb.zarr.prefetch.extract import (
    _ENOSPCError,
    _acquire_extract_lock,
    _extract_tar_to_dir,
    _release_extract_lock,
)
from egomimic.rldb.zarr.prefetch.plan import EpisodePlan
from egomimic.rldb.zarr.prefetch.pool import EpisodePool

logger = logging.getLogger(__name__)

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
            except RuntimeError as e:
                # Executor/interpreter is tearing down (process exit or Modal
                # auto-restart): submit() raises "cannot schedule new futures
                # after ... shutdown". Stop the filler cleanly instead of
                # re-raising every 2s (which spams logs and stalls shutdown
                # past the grace period).
                if "shutdown" in str(e):
                    logger.info(
                        "PoolFillerThread: executor shutting down — stopping filler"
                    )
                    break
                logger.exception("PoolFillerThread: tick failed; sleeping 2s")
                time.sleep(2.0)
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


