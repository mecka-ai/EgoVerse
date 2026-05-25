"""TarShardIterableDataset: sequential shard reading for fast training I/O.

WHY MAP-STYLE DOESN'T WORK FOR TAR SHARDS
------------------------------------------
MultiDataset is a map-style Dataset (random index access). For tar shards, random
access means: each __getitem__ call potentially needs to extract a new 3.3 GB shard
to reach one episode. With 10k shards and random sampling, every access misses the
cache → the tar approach provides zero benefit over direct volume reads.

IterableDataset is the right abstraction: each worker is assigned a subset of shards,
extracts one shard at a time to local /tmp NVMe, reads all samples from it, deletes
it, then moves to the next shard. Every sample read after extraction is local (~0.1ms
vs ~312ms from the volume).

WORKER ASSIGNMENT
-----------------
Worker i (of N total) handles shards [i, i+N, i+2N, ...].
With 8 workers and 10k shards: each worker processes ~1250 shards.
Shards are shuffled at each epoch so episode ordering varies between epochs.

MEMORY / DISK USAGE
--------------------
Each worker holds one extracted shard at a time: 20 episodes × 167 MB = ~3.3 GB.
With 8 workers: ~26 GB of /tmp in use at any given moment.
Use +modal_ephemeral_disk_gb=100 for comfortable headroom.

INTEGRATION WITH EXISTING CODE
-------------------------------
- Reuses ZarrDataset for per-episode reading (same key_map, transforms, etc.)
- Implements set_data_schematic() so BoundsCheck still works
- DataLoader shuffle must be False (shuffling is internal); set in yaml

USAGE
-----
data config:
    _target_: egomimic.rldb.zarr.tar_shard_dataset.TarShardIterableDataset
    shard_dir: /mnt/zarr-wds
    key_map: ...
    transform_list: ...
    mode: train   # or valid
"""

from __future__ import annotations

import logging
import random
import shutil
import tarfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import torch
import torch.utils.data

if TYPE_CHECKING:
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

logger = logging.getLogger(__name__)


def load_episodes_from_tars(
    episode_hashes: list[str],
    shard_index: dict[str, str],
    wds_dir: str,
    tmp_dir: str,
    key_map: dict | None,
    transform_list: list | None,
    pause_removal_epsilon: float | None = None,
) -> "dict[str, ZarrDataset]":
    """Extract tar shards for the given episode hashes and return {hash: ZarrDataset}.

    Only extracts shards that contain at least one requested episode. The extracted
    directories persist in tmp_dir — caller is responsible for cleanup after reading.
    Episodes missing from shard_index are warned about and skipped.
    """
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

    wds_path = Path(wds_dir)
    tmp_path = Path(tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Group requested episodes by shard
    shard_to_episodes: dict[str, list[str]] = {}
    missing: list[str] = []
    for ep_hash in episode_hashes:
        shard_name = shard_index.get(ep_hash)
        if shard_name is None:
            missing.append(ep_hash)
        else:
            shard_to_episodes.setdefault(shard_name, []).append(ep_hash)

    if missing:
        logger.warning(
            "%d/%d episodes not in shard index — skipping: %s%s",
            len(missing), len(episode_hashes),
            ", ".join(missing[:3]), " ..." if len(missing) > 3 else "",
        )

    if not shard_to_episodes:
        raise RuntimeError(
            f"None of the {len(episode_hashes)} requested episodes are in the shard index. "
            "Run rebuild_shard_index first."
        )

    logger.info(
        "Loading %d episodes from %d unique shard(s) (%d not in index)",
        len(episode_hashes) - len(missing), len(shard_to_episodes), len(missing),
    )

    episodes: dict[str, ZarrDataset] = {}
    for shard_name, wanted_hashes in shard_to_episodes.items():
        shard_path = wds_path / shard_name
        if not shard_path.exists():
            logger.warning(
                "Shard %s not found at %s — skipping %d episode(s)",
                shard_name, shard_path, len(wanted_hashes),
            )
            continue

        shard_tmp = tmp_path / shard_name[:-4]  # strip .tar
        shard_tmp.mkdir(parents=True, exist_ok=True)
        wanted_set = set(wanted_hashes)

        try:
            t0 = time.perf_counter()
            size_mb = shard_path.stat().st_size / 1e6
            with tarfile.open(shard_path, "r") as tar:
                tar.extractall(path=shard_tmp)
                top_level = {m.name.split("/")[0] for m in tar.getmembers()}
            elapsed = time.perf_counter() - t0
            logger.info(
                "Extracted %s  %.0f MB in %.1fs (%.0f MB/s)",
                shard_name, size_mb, elapsed, size_mb / elapsed if elapsed > 0 else 0,
            )
        except Exception as exc:
            logger.warning("Failed to extract shard %s: %s", shard_name, exc)
            continue

        for ep_name in top_level:
            ep_hash = ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
            if ep_hash not in wanted_set:
                continue
            ep_path = shard_tmp / ep_name
            if not ep_path.is_dir():
                continue
            try:
                ds = ZarrDataset(
                    ep_path,
                    key_map=key_map,
                    transform_list=transform_list,
                    pause_removal_epsilon=pause_removal_epsilon,
                )
                episodes[ep_hash] = ds
            except Exception as exc:
                logger.warning("Failed to open ZarrDataset for %s: %s", ep_hash, exc)

    logger.info(
        "Loaded %d/%d requested episodes from tars",
        len(episodes), len(episode_hashes) - len(missing),
    )
    return episodes


class TarShardIterableDataset(torch.utils.data.IterableDataset):
    """Streams training samples from tar shards by extracting them to local NVMe."""

    def __init__(
        self,
        shard_dir: str,
        key_map: dict | None = None,
        transform_list: list | None = None,
        norm_stats: dict | None = None,
        pause_removal_epsilon: float | None = None,
        mode: str = "train",
        valid_ratio: float = 0.1,
        cache_dir: str = "/tmp/shard_cache",
        seed: int = 42,
        debug: int | None = None,
        resolver=None,  # accepted but ignored; lets trainHydra.py access resolver.key_map via raw config
    ):
        super().__init__()
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
        self.data_schematic = None

        all_shards = sorted(self.shard_dir.glob("shard-*.tar"))
        if not all_shards:
            raise RuntimeError(
                f"No shard-*.tar files found in {self.shard_dir}. "
                "Run shard_zarr_to_tar.py first."
            )

        # Deterministic train/valid split on shards
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

        if debug:
            self._shards = self._shards[: max(1, debug)]

        logger.info(
            "TarShardIterableDataset [%s]: %d shards in %s",
            mode, len(self._shards), shard_dir,
        )
    
    # ------------------------------------------------------------------
    # Required by MultiDataset / trainHydra for DataSchematic wiring
    # ------------------------------------------------------------------

    def set_data_schematic(self, data_schematic, bounds_slack: float = 0.0) -> None:
        self.data_schematic = data_schematic
        if hasattr(data_schematic, "norm_stats") and self.norm_stats is None:
            self.norm_stats = data_schematic.norm_stats

    def __len__(self) -> int:
        # Approximate: 20 episodes/shard × ~2000 frames/episode
        # Used by Lightning for progress bars only; not required to be exact.
        return len(self._shards) * 20 * 2000

    def __getitem__(self, idx: int):
        # trainHydra.py calls dataset[0] once for shape inference.
        # IterableDataset doesn't support random access, so we peek at the first
        # available sample by extracting the first shard. The generator is discarded
        # after one item; Python's GC triggers the finally-block cleanup.
        return next(iter(self))

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        n_workers = worker_info.num_workers if worker_info is not None else 1

        # Each worker gets a strided slice of shards
        my_shards = self._shards[worker_id::n_workers]

        sequential = self.mode == "valid"

        if not sequential:
            # Shuffle shard order — use a different seed per epoch by mixing in
            # a counter seeded from the global RNG state
            epoch_seed = self.seed + worker_id + int(time.time() * 1000) % 10000
            rng = random.Random(epoch_seed)
            rng.shuffle(my_shards)
        else:
            rng = random.Random(self.seed)

        worker_cache = self.cache_dir / f"worker_{worker_id}"
        worker_cache.mkdir(parents=True, exist_ok=True)

        for shard_path in my_shards:
            shard_local = worker_cache / shard_path.name[:-4]  # strip .tar
            try:
                # Extract shard to local NVMe
                ep_dirs = self._extract_shard(shard_path, shard_local)

                if not sequential:
                    # Shuffle episode order within shard
                    rng.shuffle(ep_dirs)

                for ep_dir in ep_dirs:
                    ep_hash = ep_dir.name
                    if ep_hash.endswith(".zarr"):
                        ep_hash = ep_hash[:-5]

                    try:
                        ds = ZarrDataset(
                            ep_dir,
                            key_map=self.key_map,
                            transform_list=self.transform_list,
                            norm_stats=self.norm_stats,
                            pause_removal_epsilon=self.pause_removal_epsilon,
                        )
                        if self.data_schematic is not None and hasattr(ds, "set_data_schematic"):
                            ds.set_data_schematic(
                                self.data_schematic, bounds_slack=0.0
                            )
                    except Exception as e:
                        logger.warning("Skipping %s: %s", ep_hash, e)
                        continue

                    n = len(ds)
                    if n == 0:
                        continue

                    # Sequential frames for val; shuffled for train
                    indices = list(range(n))
                    if not sequential:
                        rng.shuffle(indices)
                    for idx in indices:
                        try:
                            sample = ds[idx]
                            yield sample
                        except Exception as e:
                            logger.debug("Sample error %s[%d]: %s", ep_hash, idx, e)
                            continue

            except Exception as e:
                logger.warning("Failed to process shard %s: %s", shard_path.name, e)
            finally:
                # Always clean up extracted shard to keep /tmp bounded
                if shard_local.exists():
                    shutil.rmtree(shard_local, ignore_errors=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_shard(self, shard_path: Path, dest: Path) -> list[Path]:
        """Extract tar shard to dest, return list of episode dirs."""
        dest.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        size_mb = shard_path.stat().st_size / 1e6

        with tarfile.open(shard_path, "r") as tar:
            tar.extractall(path=dest)
            top_level = {m.name.split("/")[0] for m in tar.getmembers()}

        ep_dirs = [dest / name for name in top_level if (dest / name).is_dir()]
        elapsed = time.perf_counter() - t0
        bw = size_mb / elapsed if elapsed > 0 else 0
        logger.info(
            "Extracted %s  %.0f MB in %.1fs (%.0f MB/s) → %d episodes",
            shard_path.name, size_mb, elapsed, bw, len(ep_dirs),
        )
        return ep_dirs
