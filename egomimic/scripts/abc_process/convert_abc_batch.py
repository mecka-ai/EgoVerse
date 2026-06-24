"""Resumable batch driver for converting XDOF/ABC-130k episodes to Zarr.

Reads an episode list (one in-repo episode dir per line, e.g. from
``enumerate_abc_episodes.py``) and converts each to ``<output-dir>/<uuid>.zarr``
using ``abc_to_zarr``. Idempotent: episodes whose ``.zarr`` already exists are
skipped, so an interrupted run resumes by re-invocation. Conversions run in a
process pool; with ``--source hf`` each worker streams its episode straight from
the Hub (no raw ``.mcap`` staged on disk).

Examples
--------
    # stream + convert all listed episodes, 8 workers, to the eva folder
    python -m egomimic.scripts.abc_process.convert_abc_batch \
        --source hf --episodes-file episodes.txt \
        --output-dir /workspace/eva/abc130k_zarr --arm both --workers 8

    # convert from already-downloaded local dirs instead
    python -m egomimic.scripts.abc_process.convert_abc_batch \
        --source local --episodes-file local_dirs.txt \
        --output-dir /workspace/eva/abc130k_zarr --workers 8
"""

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from egomimic.scripts.abc_process.abc_to_zarr import (
    _dataset_name_from,
    convert_episode,
    convert_hf_episode,
)

logger = logging.getLogger(__name__)


def _convert_one(
    ep_path: str,
    source: str,
    repo_id: str,
    output_dir: str,
    arm: str,
    fps: int,
    token: str | None,
    save_mp4: bool,
    chunk_timesteps: int,
) -> tuple[str, str]:
    """Worker: convert one episode. Returns (ep_path, status)."""
    out = Path(output_dir)
    name = _dataset_name_from(ep_path)
    if (out / f"{name}.zarr").exists():
        return ep_path, "skip"
    try:
        if source == "hf":
            convert_hf_episode(
                repo_id=repo_id,
                hf_episode_dir=ep_path,
                output_dir=out,
                dataset_name=name,
                arm=arm,
                fps=fps,
                token=token,
                save_mp4=save_mp4,
                chunk_timesteps=chunk_timesteps,
            )
        else:
            convert_episode(
                episode_dir=Path(ep_path),
                output_dir=out,
                dataset_name=name,
                arm=arm,
                fps=fps,
                save_mp4=save_mp4,
                chunk_timesteps=chunk_timesteps,
            )
        return ep_path, "ok"
    except Exception as e:  # noqa: BLE001 - keep the batch going
        logger.error("FAILED %s: %s", ep_path, e)
        return ep_path, f"fail:{type(e).__name__}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-file", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--source", choices=["local", "hf"], default="hf")
    ap.add_argument("--repo-id", default="XDOF/ABC-130k")
    ap.add_argument("--arm", choices=["left", "right", "both"], default="both")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunk-timesteps", type=int, default=100)
    ap.add_argument("--save-mp4", action="store_true")
    ap.add_argument("--hf-token", default="")
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of nodes splitting the work (for multi-node runs). "
        "Every node must pass the SAME --episodes-file and --num-shards.",
    )
    ap.add_argument(
        "--shard",
        type=int,
        default=0,
        help="This node's shard index in [0, num-shards). Each node processes a "
        "disjoint round-robin slice (episodes[shard::num_shards]), so no two "
        "nodes convert the same episode and the slices are exhaustive.",
    )
    args = ap.parse_args()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"--shard must be in [0, {args.num_shards}); got {args.shard}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    token = args.hf_token or os.environ.get("HF_TOKEN")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    episodes = [
        ln.strip()
        for ln in Path(args.episodes_file).read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    episodes = list(dict.fromkeys(episodes))  # dedup, keep order
    total = len(episodes)
    if args.num_shards > 1:
        # Round-robin slice: disjoint across shards, union == all episodes.
        # Every node must use the same episodes-file + num-shards.
        episodes = episodes[args.shard :: args.num_shards]
        logger.info(
            "Loaded %d episodes from %s -> shard %d/%d = %d episodes for this node",
            total,
            args.episodes_file,
            args.shard,
            args.num_shards,
            len(episodes),
        )
    logger.info("Converting %d episodes with %d workers", len(episodes), args.workers)

    counts = {"ok": 0, "skip": 0}
    fails = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(
                _convert_one,
                ep,
                args.source,
                args.repo_id,
                args.output_dir,
                args.arm,
                args.fps,
                token,
                args.save_mp4,
                args.chunk_timesteps,
            )
            for ep in episodes
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            ep, status = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if status.startswith("fail"):
                fails.append(ep)
            if i % 50 == 0 or i == len(futs):
                logger.info(
                    "[%d/%d] ok=%d skip=%d fail=%d",
                    i,
                    len(futs),
                    counts.get("ok", 0),
                    counts.get("skip", 0),
                    len(fails),
                )

    logger.info(
        "DONE ok=%d skip=%d fail=%d",
        counts.get("ok", 0),
        counts.get("skip", 0),
        len(fails),
    )
    if fails:
        fail_path = Path(args.output_dir) / "failed_episodes.txt"
        fail_path.write_text("\n".join(fails) + "\n")
        logger.info("Wrote %d failures to %s", len(fails), fail_path)


if __name__ == "__main__":
    main()
