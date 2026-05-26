"""Load ZarrDataset episodes from tar shards for curation passes."""

from __future__ import annotations

import logging
import tarfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

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

    Only extracts shards that contain at least one requested episode. Extracted
    directories persist in tmp_dir — caller is responsible for cleanup.
    Episodes missing from shard_index are warned about and skipped.
    """
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

    wds_path = Path(wds_dir)
    tmp_path = Path(tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)

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
            "Run shard_zarr_to_tar.py first."
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
