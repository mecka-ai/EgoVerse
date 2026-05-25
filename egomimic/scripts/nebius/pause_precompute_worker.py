"""Per-shard SLURM worker for pause-filter precompute on Nebius.

One SLURM array task = one shard. The task reads its shard line from
``manifest.jsonl`` (selected by ``$SLURM_ARRAY_TASK_ID``), processes that
shard's episodes, and writes ``<out_dir>/shards/shard_<id>.json``.

Output JSON shape matches the cache contract consumed by
``egomimic.rldb.zarr.zarr_dataset_multi._apply_pause_precompute_cache``:

    {episode_hash: {"raw_total": int, "keep_indices": [int, ...]}}

Per-episode failures (missing keys, zarr open errors) emit
``{"raw_total": 0, "keep_indices": []}`` which the consumer treats as a
cache miss — it falls back to in-process precompute for those episodes.

The filter algorithm is shared with the in-process path:
``_build_pause_keep_mask`` checks BOTH left/right obs_ee_pose deltas AND,
when available, left/right obs_keypoints max-landmark deltas. Direct
import from zarr_dataset_multi is intentional — single source of truth.

Usage (typically invoked by sbatch, but runnable standalone for debugging)::

    python -m egomimic.scripts.nebius.pause_precompute_worker \\
        --manifest /shared/pause/run-XYZ/manifest.jsonl \\
        --out-dir /shared/pause/run-XYZ \\
        --shard-id $SLURM_ARRAY_TASK_ID
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from egomimic.rldb.zarr.zarr_dataset_multi import (
    PAUSE_DETECT_KEYPOINT_KEYS,
    PAUSE_DETECT_KEYS,
    _build_pause_keep_mask,
)


def _process_episode(
    episode_hash: str,
    path_str: str,
    epsilon: float,
) -> tuple[str, int, list[int]]:
    """Open one episode, build keep-mask, return cache entry tuple.

    Failures collapse to ``(hash, 0, [])`` — the consumer treats raw_total=0
    as a miss and falls through to its in-process fallback.
    """
    try:
        store = zarr.open_group(path_str, mode="r")
    except Exception:
        return (episode_hash, 0, [])

    left_key, right_key = PAUSE_DETECT_KEYS
    try:
        left_pose = np.asarray(store[left_key][:])
        right_pose = np.asarray(store[right_key][:])
    except KeyError:
        # Episode lacks ee_pose — keep all frames (matches precompute_pause_filter
        # behavior in zarr_dataset_multi.py).
        try:
            sample = next(iter(store.array_keys()), None)
            total = int(store[sample].shape[0]) if sample else 0
        except Exception:
            total = 0
        return (episode_hash, total, list(range(total)))
    except Exception:
        return (episode_hash, 0, [])

    left_kp_key, right_kp_key = PAUSE_DETECT_KEYPOINT_KEYS
    left_kp = right_kp = None
    try:
        left_kp = np.asarray(store[left_kp_key][:])
    except Exception:
        pass
    try:
        right_kp = np.asarray(store[right_kp_key][:])
    except Exception:
        pass

    try:
        keep = _build_pause_keep_mask(
            left_pose=left_pose,
            right_pose=right_pose,
            epsilon=epsilon,
            left_keypoints=left_kp,
            right_keypoints=right_kp,
        )
    except Exception:
        return (episode_hash, 0, [])

    indices = np.flatnonzero(keep).astype(np.int64).tolist()
    return (episode_hash, int(left_pose.shape[0]), indices)


def _process_shard(
    episodes: list[tuple[str, str]],
    epsilon: float,
    threads: int,
) -> dict[str, dict]:
    """Thread-pool fan-out within the shard. Each task is mostly I/O bound."""
    out: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for episode_hash, raw_total, indices in ex.map(
            lambda item: _process_episode(item[0], item[1], epsilon), episodes
        ):
            out[episode_hash] = {"raw_total": raw_total, "keep_indices": indices}
    return out


def _load_shard_line(manifest_path: Path, shard_id: int) -> dict:
    """Read the shard_id-th JSONL line from the manifest.

    Streaming so we don't materialize the whole manifest for huge runs.
    """
    with manifest_path.open() as f:
        for i, line in enumerate(f):
            if i == shard_id:
                return json.loads(line)
    raise IndexError(f"shard_id={shard_id} out of range for manifest {manifest_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--shard-id",
        type=int,
        default=None,
        help="Shard index. Defaults to $SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("EGOMIMIC_PAUSE_PRECOMPUTE_THREADS", "16")),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process and overwrite even if shard output already exists.",
    )
    args = parser.parse_args(argv)

    shard_id = args.shard_id
    if shard_id is None:
        env_val = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_val is None:
            print(
                "[pause-precompute-worker] --shard-id not given and "
                "$SLURM_ARRAY_TASK_ID not set",
                file=sys.stderr,
            )
            return 2
        shard_id = int(env_val)

    shards_dir = args.out_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    out_path = shards_dir / f"shard_{shard_id:05d}.json"

    if out_path.exists() and not args.force:
        print(
            f"[pause-precompute-worker] shard {shard_id}: output exists ({out_path}), "
            "skipping (use --force to override)"
        )
        return 0

    t0 = time.monotonic()
    shard = _load_shard_line(args.manifest, shard_id)
    epsilon = float(shard["epsilon"])
    episodes: list[tuple[str, str]] = [(e[0], e[1]) for e in shard["episodes"]]
    print(
        f"[pause-precompute-worker] shard {shard_id}: {len(episodes)} episodes, "
        f"epsilon={epsilon}, threads={args.threads}"
    )

    results = _process_shard(episodes, epsilon, args.threads)

    n_total = sum(v["raw_total"] for v in results.values())
    n_kept = sum(len(v["keep_indices"]) for v in results.values())
    n_err = sum(1 for v in results.values() if v["raw_total"] == 0)
    elapsed = time.monotonic() - t0
    pct = (100.0 * n_kept / n_total) if n_total else 100.0

    tmp_path = out_path.with_suffix(".json.tmp")
    with tmp_path.open("w") as f:
        json.dump(results, f)
    tmp_path.replace(out_path)

    print(
        f"[pause-precompute-worker] shard {shard_id} done: kept {n_kept}/{n_total} "
        f"({pct:.1f}%) | errors={n_err} | {elapsed:.1f}s | wrote {out_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
