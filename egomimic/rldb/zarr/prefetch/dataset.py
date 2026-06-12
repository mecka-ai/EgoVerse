"""Map-style dataset that reads episodes from the NVMe pool.

``prepare_epoch(epoch)`` builds the frame-level index for the current window,
blocking until every episode is materialized; workers fork after it returns and
inherit the index map plus warm zarr handles.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.utils.data

from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
from egomimic.rldb.zarr.prefetch.bounds import _BoundsCheckMixin
from egomimic.rldb.zarr.prefetch.catalog import EpisodeCatalogEntry, ZipEpisodeResolver
from egomimic.rldb.zarr.prefetch.extract import (
    _FILLER_REGISTRY,
    _FILLER_REGISTRY_LOCK,
    _acquire_extract_lock,
    _extract_tar_to_dir,
    _release_extract_lock,
)
from egomimic.rldb.zarr.prefetch.filler import PoolFillerThread
from egomimic.rldb.zarr.prefetch.plan import EpisodePlan
from egomimic.rldb.zarr.prefetch.pool import EpisodePool

logger = logging.getLogger(__name__)

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
        Optional target frame count hint used to compute a default
        ``episodes_per_epoch``. ``__len__`` now returns the real total frame
        count so Lightning's ``limit_train_batches`` truncates a fully-shuffled
        pool (same behaviour as zarr's ``MultiDataset``).
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
        prepare_timeout_s: float = 3600.0,
        background_filler: bool = True,
    ):
        super().__init__()
        self.resolver = resolver
        self.mode = mode
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_copy_threads = int(n_copy_threads)
        self.seed = seed
        # Max time prepare_epoch blocks waiting for the window to materialize.
        # Default 1h is plenty for a normal sliding window; raise it for
        # stage-all-first configs where episodes_per_epoch == a large subset
        # (e.g. 20k episodes from a ~150 MB/s volume takes ~3h to stage).
        self.prepare_timeout_s = float(prepare_timeout_s)
        self._background_filler = bool(background_filler)

        self._episodes = resolver.split_catalog(mode)
        if not self._episodes:
            raise RuntimeError(f"Resolver returned 0 episodes for mode={mode}")

        total_frames = sum(e.n_frames for e in self._episodes)
        self._total_frames = total_frames
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
        # Any non-valid dataset stages in the background via the filler; only the
        # valid split uses synchronous inline extraction (rare, small). This must
        # key off "not valid" rather than '== "train"': configs that select an
        # exact episode list via eps_to_use use mode="total" (verbatim, no split),
        # and those still need the background filler — otherwise nothing stages and
        # prepare_epoch blocks until its timeout (the deminf64 stage-all hang).
        if (
            self._background_filler
            and self.mode != "valid"
            and self._is_filler_rank
        ):
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
        # Return the real total frame count so PyTorch's RandomSampler shuffles
        # the full pool and Lightning's limit_train_batches truncates it —
        # identical to zarr MultiDataset behaviour.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return self._total_frames * world_size

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
        while attempts < 16:
            ep_path, frame_idx = self._index_map[idx % len(self._index_map)]
            if Path(ep_path, ".done").exists() and not Path(ep_path, ".bad").exists():
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
                            "Failed to open %s (%s); marking .bad, skipping",
                            Path(ep_path).name, e,
                        )
                        self._mark_episode_bad(ep_path, f"open failed: {e}")
                        idx = (idx + 1) % len(self._index_map)
                        attempts += 1
                        continue
                try:
                    return self._zarr_cache[ep_path][frame_idx]
                except Exception as e:
                    # Episode opened but a frame is unreadable/undecodable
                    # (e.g. corrupt JPEG that exhausts ZarrDataset's in-episode
                    # retries → "Entire episode bad"). Skip to a *different*
                    # episode and mark this one .bad so prepare_epoch and the
                    # filler drop it for the rest of the run instead of
                    # re-selecting it. Without this the exception propagates
                    # out and kills the DataLoader worker / training.
                    logger.warning(
                        "Read/decode failed for %s frame %s (%s); marking .bad, skipping",
                        Path(ep_path).name, frame_idx, e,
                    )
                    self._mark_episode_bad(ep_path, f"read/decode failed: {e}")
                    self._zarr_cache.pop(ep_path, None)
                    idx = (idx + 1) % len(self._index_map)
                    attempts += 1
                    continue
            # Evicted episode (eviction race at epoch boundary) or just-marked
            # .bad. Skip to next frame in the shuffled index_map.
            idx = (idx + 1) % len(self._index_map)
            attempts += 1

        # Last resort: bounded wait, then surface whatever's available — still
        # tolerating a bad episode by skipping to a different index.
        self._wait_for_episode_path(ep_path)
        try:
            if ep_path not in self._zarr_cache:
                self._zarr_cache[ep_path] = ZarrDataset(
                    ep_path,
                    key_map=self.resolver.key_map,
                    transform_list=self.resolver.transform_list,
                    norm_stats=self.resolver.norm_stats,
                    pause_removal_epsilon=self.resolver.pause_removal_epsilon,
                )
            return self._zarr_cache[ep_path][frame_idx]
        except Exception as e:
            logger.warning(
                "Read/decode failed for %s (%s) on last-resort path; marking .bad, skipping",
                Path(ep_path).name, e,
            )
            self._mark_episode_bad(ep_path, f"read/decode failed (last resort): {e}")
            self._zarr_cache.pop(ep_path, None)
            return self.__getitem__((idx + 1) % len(self._index_map))

    @staticmethod
    def _mark_episode_bad(ep_path: str, reason: str) -> None:
        """Best-effort ``.bad`` sentinel, safe to call from any DataLoader
        worker. Unlike ``PoolFiller._mark_bad`` we do NOT remove the episode
        dir — other workers/ranks may be reading it concurrently. The sentinel
        makes ``prepare_epoch`` drop the episode next epoch and short-circuits
        the filler, so a corrupt-but-openable episode is selected at most once.
        """
        try:
            Path(ep_path, ".bad").touch()
        except OSError:
            pass
        logger.warning(
            "PrefetchedMapDataset: %s marked .bad — %s", Path(ep_path).name, reason
        )

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

        # Evict episodes outside the current+lookahead window. Skipped when the
        # background filler is off — the whole split stays on NVMe once staged.
        if self._background_filler and self._is_filler_rank:
            freed = self._pool.evict_outside(keep_hashes)
            if freed:
                logger.info(
                    "prepare_epoch %d: freed %.1f GB outside current window",
                    gen_label, freed / 1e9,
                )

        # Without a background filler, extract the window synchronously on rank 0
        # (valid always; train/train_viz when stage-all-first is configured).
        if (self.mode == "valid" or not self._background_filler) and self._is_filler_rank:
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
        deadline = t0 + self.prepare_timeout_s
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
                    f"prepare_epoch: timeout ({self.prepare_timeout_s / 3600:.1f}h) waiting "
                    f"for {len(pending)} episodes for epoch {gen_label}"
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
