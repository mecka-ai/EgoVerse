"""TarShardMultiDataset — shard-per-epoch training as a MultiDataset + IterableDataset.

Epoch contract:
- one epoch consumes exactly one extracted tar shard.
- worker-0 handles shard swap + index build + prefetch orchestration.
- all DataLoader workers sample via MultiDataset.__getitem__ on a shared index map.

Synchronization contract:
- worker-0 writes epoch metadata (`epoch_meta.pkl`) atomically.
- every worker waits until metadata for its local epoch generation appears,
  then reads the same index map file.
"""

from __future__ import annotations

import logging
import math
import pickle
import random
import shutil
import tarfile
import threading
import time
import fcntl
from pathlib import Path
from typing import Iterator

import torch
import torch.utils.data

from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset, ZarrDataset

logger = logging.getLogger(__name__)

_EPISODES_PER_SHARD = 20  # nominal, used only for debug-mode shard count


class TarShardMultiDataset(MultiDataset, torch.utils.data.IterableDataset):
    """One tar shard per epoch; reuses MultiDataset index map and __getitem__."""

    def __init__(
        self,
        shard_dir: str,
        key_map=None,
        transform_list=None,
        norm_stats=None,
        pause_removal_epsilon: float | None = None,
        mode: str = "train",
        valid_ratio: float = 0.1,
        cache_dir: str = "/tmp/shard_cache",
        seed: int = 42,
        debug: int = 0,
        epoch_ready_timeout_s: int = 1800,
        prefetch_wait_timeout_s: float = 10.0,
        resolver=None,
        prefetch_shards: int | None = None,
        max_workers: int | None = None,
        **kwargs,
    ):
        # Kept for Hydra config compatibility (resolver / legacy prefetch knobs).
        self.shard_dir = Path(shard_dir)
        self.key_map = key_map
        self.transform_list = transform_list
        self.norm_stats = norm_stats
        self.pause_removal_epsilon = pause_removal_epsilon
        self.mode = mode
        self.valid_ratio = valid_ratio
        self.cache_dir = Path(cache_dir)
        self.seed = seed
        self.debug = debug
        self.epoch_ready_timeout_s = int(epoch_ready_timeout_s)
        self.prefetch_wait_timeout_s = float(prefetch_wait_timeout_s)
        self._probe_log_emitted = False

        all_shards = sorted(self.shard_dir.glob("shard-*.tar"))
        if not all_shards:
            raise RuntimeError(
                f"No shard-*.tar files found in {self.shard_dir}. "
                "Run shard_zarr_to_tar.py first."
            )

        if debug > 0:
            n_shards = max(1, math.ceil(debug / _EPISODES_PER_SHARD))
            all_shards = all_shards[:n_shards]
            logger.info(
                "debug=%d: using first %d shards (~%d episodes)",
                debug, len(all_shards), len(all_shards) * _EPISODES_PER_SHARD,
            )

        rng = random.Random(seed)
        shuffled = list(all_shards)
        rng.shuffle(shuffled)
        n_valid = max(1, int(len(shuffled) * valid_ratio))
        if mode == "valid":
            self._shards = shuffled[:n_valid]
        elif mode == "train":
            self._shards = shuffled[n_valid:]
        else:
            self._shards = shuffled

        self._epoch_gen = 0
        self.datasets: dict[str, ZarrDataset] = {}
        self._init_multidataset_state()

        # Prefetch state — only ever mutated inside worker-0's process.
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_ready = threading.Event()
        self._prefetch_ok = False

        torch.utils.data.Dataset.__init__(self)

        logger.info(
            "TarShardMultiDataset [%s]: %d shards, cache_dir=%s, shard_dir=%s",
            mode, len(self._shards), self.cache_dir, shard_dir,
        )

    def set_data_schematic(self, data_schematic, bounds_slack: float = 0.0) -> None:
        super().set_data_schematic(data_schematic, bounds_slack=bounds_slack)
        if hasattr(data_schematic, "norm_stats") and self.norm_stats is None:
            self.norm_stats = data_schematic.norm_stats
        logger.info(
            "TarShardMultiDataset: data_schematic set (bounds_slack=%.4f)",
            self.bounds_slack,
        )

    def _zarr_dataset_kwargs(self) -> dict:
        return dict(
            key_map=self.key_map,
            transform_list=self.transform_list,
            norm_stats=self.norm_stats,
            pause_removal_epsilon=self.pause_removal_epsilon,
        )

    def _datasets_from_extracted_dir(self, tar_dir: Path) -> dict[str, ZarrDataset]:
        """Build episode-name → ZarrDataset map for one extracted shard."""
        datasets: dict[str, ZarrDataset] = {}
        for ep_dir in sorted(tar_dir.iterdir()):
            if not ep_dir.is_dir() or not (ep_dir / "zarr.json").exists():
                continue
            try:
                datasets[ep_dir.name] = ZarrDataset(ep_dir, **self._zarr_dataset_kwargs())
            except Exception as exc:
                logger.warning("Skipping %s: %s", ep_dir.name, exc)
        return datasets

    def _rebuild_epoch_index(self, tar_dir: Path, *, shuffle: bool) -> None:
        """Populate MultiDataset index state from episodes in ``tar_dir``."""
        self.datasets = self._datasets_from_extracted_dir(tar_dir)
        self._build_index_map_from_datasets()
        if shuffle:
            random.Random(self.seed + self._epoch_gen).shuffle(self.index_map)
        logger.info(
            "ShardEpoch %d index: %d episodes, %d frames from %s",
            self._epoch_gen,
            len(self.datasets),
            len(self.index_map),
            tar_dir.name,
        )

    def __len__(self) -> int:
        if self.index_map:
            return len(self.index_map)
        return len(self._shards) * _EPISODES_PER_SHARD * 2000

    def __getitem__(self, idx: int):
        if not self._probe_log_emitted:
            logger.info(
                "TarShardMultiDataset __getitem__ called in main process "
                "(likely shape probe); running one-shot probe path."
            )
            self._probe_log_emitted = True
        if not self.index_map:
            self._ensure_probe_shard()
        return super().__getitem__(idx)

    def _ensure_probe_shard(self) -> None:
        if self._current_dir.exists() and self.index_map:
            return
        probe_shard = self._pick_shard(1)
        logger.info("Probe: extracting %s for shape inference", probe_shard.name)
        self._extract(probe_shard, self._current_dir)
        self._epoch_gen = 1
        self._rebuild_epoch_index(self._current_dir, shuffle=False)

    # ------------------------------------------------------------------
    # Stable paths (same across all worker processes via shared filesystem)
    # ------------------------------------------------------------------

    @property
    def _current_dir(self) -> Path:
        return self.cache_dir / "extractor" / "current"

    @property
    def _next_dir(self) -> Path:
        return self.cache_dir / "extractor" / "next"

    @property
    def _index_map_path(self) -> Path:
        return self.cache_dir / "index_map.pkl"

    @property
    def _epoch_meta_path(self) -> Path:
        return self.cache_dir / "epoch_meta.pkl"

    @property
    def _prefetch_meta_path(self) -> Path:
        return self.cache_dir / "prefetch_meta.pkl"

    # ------------------------------------------------------------------
    # Worker-0 helpers
    # ------------------------------------------------------------------

    def _pick_shard(self, epoch_gen: int) -> Path:
        """Random shard for train; sequential cycle for valid."""
        if self.mode == "valid":
            return self._shards[(epoch_gen - 1) % len(self._shards)]
        return self._shards[
            random.Random(self.seed + epoch_gen).randrange(len(self._shards))
        ]

    def _extract(self, shard_path: Path, dest: Path) -> None:
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        t0 = time.perf_counter()
        size_mb = shard_path.stat().st_size / 1e6
        with tarfile.open(shard_path, "r") as tf:
            tf.extractall(path=dest)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Extracted %s  %.0f MB in %.1fs (%.0f MB/s)",
            shard_path.name, size_mb, elapsed, size_mb / elapsed if elapsed else 0,
        )

    def _start_prefetch(self, shard_path: Path, *, for_epoch_gen: int) -> None:
        """Spawn a background thread to extract shard_path into _next_dir."""
        dest = self._next_dir
        self._prefetch_ready.clear()
        self._prefetch_ok = False
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._prefetch_meta_path.unlink(missing_ok=True)

        def _run() -> None:
            try:
                self._extract(shard_path, dest)
                self._prefetch_ok = True
                tmp_path = self._prefetch_meta_path.with_suffix(".tmp")
                with open(tmp_path, "wb") as f:
                    pickle.dump(
                        {
                            "ready_for_epoch_gen": int(for_epoch_gen),
                            "shard_name": shard_path.name,
                        },
                        f,
                    )
                tmp_path.replace(self._prefetch_meta_path)
            except Exception as exc:
                logger.warning("Prefetch failed for %s: %s", shard_path.name, exc)
            finally:
                self._prefetch_ready.set()

        self._prefetch_thread = threading.Thread(target=_run, daemon=True)
        self._prefetch_thread.start()
        logger.info("Prefetch started: %s → next/", shard_path.name)

    def _prefetch_ready_for_epoch(self, epoch_gen: int) -> bool:
        if not self._prefetch_meta_path.exists():
            return False
        try:
            with open(self._prefetch_meta_path, "rb") as f:
                meta = pickle.load(f)
            return int(meta.get("ready_for_epoch_gen", -1)) >= int(epoch_gen)
        except Exception:
            return False

    def _wait_for_prefetch(self, epoch_gen: int, timeout_s: float) -> float:
        """Wait for prefetch readiness for `epoch_gen`; returns waited seconds."""
        t0 = time.perf_counter()
        deadline = t0 + timeout_s
        while time.perf_counter() < deadline:
            if self._prefetch_ok or self._prefetch_ready_for_epoch(epoch_gen):
                return time.perf_counter() - t0
            if self._prefetch_ready.is_set() and not self._prefetch_ok:
                break
            time.sleep(0.1)
        return time.perf_counter() - t0

    def _write_epoch_meta(self, *, epoch_gen: int, index_map_path: Path) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self._epoch_meta_path.with_suffix(".tmp")
        payload = {"epoch_gen": int(epoch_gen), "index_map_path": str(index_map_path)}
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f)
        tmp_path.replace(self._epoch_meta_path)

    def _wait_for_epoch_meta(self, target_epoch_gen: int) -> dict:
        deadline = time.perf_counter() + self.epoch_ready_timeout_s
        last_log = 0.0
        while time.perf_counter() < deadline:
            if self._epoch_meta_path.exists():
                try:
                    with open(self._epoch_meta_path, "rb") as f:
                        meta = pickle.load(f)
                    if int(meta.get("epoch_gen", -1)) >= target_epoch_gen:
                        return meta
                except Exception:
                    pass
            now = time.perf_counter()
            if now - last_log > 30.0:
                last_log = now
                logger.info(
                    "Waiting for epoch metadata (epoch_gen=%d, timeout=%ds)...",
                    target_epoch_gen,
                    self.epoch_ready_timeout_s,
                )
            time.sleep(0.1)
        raise TimeoutError(
            f"Timed out waiting for epoch metadata for epoch_gen={target_epoch_gen}"
        )

    def _epoch_done_state_paths(self, epoch_gen: int) -> tuple[Path, Path]:
        done_path = self.cache_dir / f"epoch_done_{epoch_gen}.pkl"
        lock_path = self.cache_dir / f"epoch_done_{epoch_gen}.lock"
        return done_path, lock_path

    def _mark_worker_done_and_maybe_cleanup(
        self, *, epoch_gen: int, worker_id: int, n_workers: int
    ) -> None:
        """Mark this worker done; last worker deletes the fully-consumed shard."""
        done_path, lock_path = self._epoch_done_state_paths(epoch_gen)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        is_last_worker = False
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

            done_workers: set[int] = set()
            if done_path.exists():
                try:
                    with open(done_path, "rb") as f:
                        loaded = pickle.load(f)
                        done_workers = set(int(x) for x in loaded)
                except Exception:
                    done_workers = set()

            done_workers.add(int(worker_id))
            tmp_path = done_path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(sorted(done_workers), f)
            tmp_path.replace(done_path)
            is_last_worker = len(done_workers) >= int(n_workers)

        if not is_last_worker:
            return

        if self._current_dir.exists():
            shutil.rmtree(self._current_dir, ignore_errors=True)
            logger.info("Epoch %d fully consumed: deleted current shard cache", epoch_gen)

        try:
            done_path.unlink(missing_ok=True)
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    def _iter_probe_once(self) -> Iterator[dict]:
        """One-shot sample for main-process shape probing (no epoch coordination)."""
        self._ensure_probe_shard()
        if not self.index_map:
            raise RuntimeError("Probe: unable to build index from current shard")
        yield self[0]

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            yield from self._iter_probe_once()
            return

        worker_id = worker_info.id
        n_workers = worker_info.num_workers
        sequential = self.mode == "valid"
        self._epoch_gen += 1
        local_epoch_gen = self._epoch_gen

        if worker_id == 0:
            gen = local_epoch_gen
            if gen == 1:
                shard = self._pick_shard(gen)
                logger.info("ShardEpoch 1 cold start: extracting %s", shard.name)
                self._extract(shard, self._current_dir)
            else:
                waited = self._wait_for_prefetch(
                    gen, timeout_s=self.prefetch_wait_timeout_s
                )
                if waited > 1.0:
                    logger.warning(
                        "ShardEpoch %d: waited %.1fs for prefetch readiness "
                        "(limit=%.1fs)",
                        gen, waited,
                        self.prefetch_wait_timeout_s,
                    )
                if self._current_dir.exists():
                    shutil.rmtree(self._current_dir, ignore_errors=True)
                if self._prefetch_ok and self._next_dir.exists():
                    self._next_dir.rename(self._current_dir)
                    logger.info("ShardEpoch %d: swapped prefetch → current", gen)
                    self._prefetch_meta_path.unlink(missing_ok=True)
                else:
                    logger.warning(
                        "ShardEpoch %d: prefetch unavailable, extracting synchronously",
                        gen,
                    )
                    self._extract(self._pick_shard(gen), self._current_dir)

            logger.info("ShardEpoch %d: building frame index map from current shard", gen)
            self._rebuild_epoch_index(self._current_dir, shuffle=not sequential)
            self._index_map_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._index_map_path, "wb") as f:
                pickle.dump(self.index_map, f)
            self._write_epoch_meta(epoch_gen=gen, index_map_path=self._index_map_path)
            logger.info(
                "ShardEpoch %d: published index map with %d samples", gen, len(self.index_map)
            )
            self._start_prefetch(self._pick_shard(gen + 1), for_epoch_gen=gen + 1)

        meta = self._wait_for_epoch_meta(local_epoch_gen)
        index_path = Path(meta["index_map_path"])
        with open(index_path, "rb") as f:
            self.index_map = pickle.load(f)

        # Rebuild datasets dict and per-episode global-index lists for MultiDataset.__getitem__.
        self.datasets = self._datasets_from_extracted_dir(self._current_dir)
        self._global_indices_by_dataset = {
            dataset_name: [] for dataset_name in self.datasets
        }
        for global_idx, (dataset_name, _local_idx) in enumerate(self.index_map):
            if dataset_name in self._global_indices_by_dataset:
                self._global_indices_by_dataset[dataset_name].append(global_idx)

        sample_count = 0
        t_start = time.perf_counter()
        try:
            for global_idx in range(worker_id, len(self.index_map), n_workers):
                try:
                    yield self.__getitem__(global_idx)
                    sample_count += 1
                except Exception as exc:
                    dataset_name, local_idx = self.index_map[global_idx]
                    logger.debug(
                        "Sample error %s[%d]: %s", dataset_name, local_idx, exc
                    )
        finally:
            elapsed = time.perf_counter() - t_start
            logger.info(
                "Worker %d: %d samples in %.0fs (%.1f samples/s)",
                worker_id, sample_count, elapsed, sample_count / elapsed if elapsed else 0,
            )
            self._mark_worker_done_and_maybe_cleanup(
                epoch_gen=local_epoch_gen,
                worker_id=worker_id,
                n_workers=n_workers,
            )


# Backward-compatible alias for existing Hydra configs and scripts.
TarShardIterableDataset = TarShardMultiDataset
