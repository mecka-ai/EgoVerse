"""Benchmark parallel shard download throughput from the GS volume.

Sweeps concurrent download counts [1, 2, 4, 6, 8, 12, 16, 24] and measures:
  - Wall time to copy N shards
  - Per-shard copy time
  - Aggregate MB/s

This tells us the I/O saturation point of the volume mount so we can pick
the right parallelism for the shared-pool downloader architecture.

Usage:
    modal run --env robotics egomimic/modal/benchmark_shard_io.py
    modal run --env robotics egomimic/modal/benchmark_shard_io.py \
        --shard-subdir global_shuffle_debug300 --n-shards 20
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import modal

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

from modal_setup import app, image, gs_volume  # noqa: E402

GS_MOUNT = "/mnt/zarr-gs"
LOCAL_DST = "/tmp"
PARALLELISM_SWEEP = [1, 2, 4, 6, 8, 12, 16, 24]


@app.function(
    image=image,
    volumes={GS_MOUNT: gs_volume},
    cpu=32,
    memory=32768,
    timeout=1800,
)
def run_benchmark(shard_subdir: str = "global_shuffle_debug300", n_shards: int = 16):
    """Copy `n_shards` from the GS volume to /cache with varying parallelism."""
    import concurrent.futures
    import json
    import shutil
    import time

    from tabulate import tabulate  # already in image via pip

    shard_dir = Path(GS_MOUNT) / shard_subdir
    index = json.loads((shard_dir / "index.json").read_text())
    all_ids = [
        sid for sid in index.get("shard_ids", [])
        if (shard_dir / f"{sid}.mp4").exists() and (shard_dir / f"{sid}.npz").exists()
    ]

    if len(all_ids) < n_shards:
        n_shards = len(all_ids)
        print(f"Only {n_shards} shards available — adjusting n_shards to {n_shards}")

    shard_ids = all_ids[:n_shards]

    # Measure shard sizes
    sizes_mb = []
    for sid in shard_ids:
        mp4_bytes = (shard_dir / f"{sid}.mp4").stat().st_size
        npz_bytes = (shard_dir / f"{sid}.npz").stat().st_size
        sizes_mb.append((mp4_bytes + npz_bytes) / 1_048_576)

    total_mb = sum(sizes_mb)
    frames_per_shard = index.get("frames_per_shard", "N/A")
    n_covered_frames = index.get("n_covered_frames", "N/A")
    print(f"\nShard dir:         {shard_dir}")
    print(f"Total shards:      {len(all_ids)}")
    print(f"Frames per shard:  {frames_per_shard}")
    print(f"Total frames:      {n_covered_frames}")
    print(f"Shard size:        avg {total_mb/n_shards:.1f} MB  (mp4+npz)")
    print(f"Local dst:         {LOCAL_DST}\n")

    def copy_shard(sid: str, worker_id: int) -> float:
        """Copy one shard pair; return elapsed seconds."""
        t0 = time.perf_counter()
        dst_mp4 = Path(LOCAL_DST) / f"bench_{worker_id}_{sid}.mp4"
        dst_npz = Path(LOCAL_DST) / f"bench_{worker_id}_{sid}.npz"
        shutil.copy2(shard_dir / f"{sid}.mp4", dst_mp4)
        shutil.copy2(shard_dir / f"{sid}.npz", dst_npz)
        elapsed = time.perf_counter() - t0
        dst_mp4.unlink(missing_ok=True)
        dst_npz.unlink(missing_ok=True)
        return elapsed

    rows = []
    for n_parallel in PARALLELISM_SWEEP:
        if n_parallel > n_shards:
            break

        # Pick the first n_parallel shards for a fair comparison
        batch = shard_ids[:n_parallel]
        batch_mb = sum(sizes_mb[:n_parallel])

        # Warm up the volume mount (avoid cold-start noise on first sweep)
        if n_parallel == PARALLELISM_SWEEP[0]:
            print("Warming up volume mount...")
            copy_shard(batch[0], 99)

        t_wall_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as pool:
            futs = [pool.submit(copy_shard, sid, i) for i, sid in enumerate(batch)]
            per_shard_times = [f.result() for f in futs]
        wall = time.perf_counter() - t_wall_start

        throughput = batch_mb / wall
        shards_per_sec = n_parallel / wall
        avg_shard_s = sum(per_shard_times) / len(per_shard_times)
        max_shard_s = max(per_shard_times)

        rows.append([
            n_parallel,
            f"{wall:.2f}s",
            f"{avg_shard_s:.2f}s",
            f"{max_shard_s:.2f}s",
            f"{throughput:.0f} MB/s",
            f"{shards_per_sec:.2f}/s",
        ])
        print(f"  parallel={n_parallel:2d}  wall={wall:.2f}s  avg_shard={avg_shard_s:.2f}s  {throughput:.0f} MB/s  {shards_per_sec:.2f} shards/s")

    print("\n" + tabulate(
        rows,
        headers=["Parallel", "Wall time", "Avg shard", "Max shard", "Throughput", "Shards/s"],
        tablefmt="rounded_outline",
    ))


@app.local_entrypoint()
def main(
    shard_subdir: str = "global_shuffle_debug300",
    n_shards: int = 16,
):
    run_benchmark.remote(shard_subdir=shard_subdir, n_shards=n_shards)
