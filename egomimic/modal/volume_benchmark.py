"""Volume I/O benchmark for mecka_data_v2.

Usage:
    modal run --env robotics egomimic/modal/volume_benchmark.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import modal
from modal_setup import app, zarr_volume, CFG

MOUNT_PATH = CFG.volume_mount_path  # /mnt/zarr-data


@app.function(
    volumes={MOUNT_PATH: zarr_volume},
    cpu=8,
    memory=16384,
    timeout=600,
)
def run_benchmark() -> None:
    import concurrent.futures
    import os
    import random
    import time

    import numpy as np
    import zarr

    root = Path(MOUNT_PATH)

    # ------------------------------------------------------------------
    # 1. Volume overview
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VOLUME OVERVIEW")
    print("=" * 60)

    all_episodes = [p for p in root.iterdir() if p.is_dir()]
    n_episodes = len(all_episodes)
    print(f"Episodes (top-level dirs): {n_episodes:,}")

    # Sample 20 episodes to estimate average size
    sample = random.sample(all_episodes, min(20, n_episodes))
    sizes_mb = []
    for ep in sample:
        total = sum(f.stat().st_size for f in ep.rglob("*") if f.is_file())
        sizes_mb.append(total / 1e6)
    avg_mb = sum(sizes_mb) / len(sizes_mb)
    est_total_tb = avg_mb * n_episodes / 1e6
    print(f"Avg episode size (sample of {len(sample)}): {avg_mb:.1f} MB")
    print(f"Estimated total data: {est_total_tb:.1f} TB")

    # ------------------------------------------------------------------
    # 2. Zarr open latency (metadata reads)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ZARR OPEN LATENCY (metadata only)")
    print("=" * 60)

    N_OPEN = 50
    open_sample = random.sample(all_episodes, N_OPEN)
    open_times = []
    for ep_path in open_sample:
        t0 = time.perf_counter()
        store = zarr.open_group(str(ep_path), mode="r")
        _ = dict(store.attrs)
        open_times.append(time.perf_counter() - t0)

    open_times_ms = [t * 1000 for t in open_times]
    print(f"Samples: {N_OPEN}")
    print(f"  mean:   {sum(open_times_ms)/len(open_times_ms):.1f} ms")
    print(f"  median: {sorted(open_times_ms)[len(open_times_ms)//2]:.1f} ms")
    print(f"  p95:    {sorted(open_times_ms)[int(len(open_times_ms)*0.95)]:.1f} ms")
    print(f"  min:    {min(open_times_ms):.1f} ms")
    print(f"  max:    {max(open_times_ms):.1f} ms")

    # ------------------------------------------------------------------
    # 3. Single-frame read latency (what __getitem__ does)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SINGLE FRAME READ LATENCY")
    print("=" * 60)

    N_READS = 100
    read_sample = random.sample(all_episodes, N_READS)
    frame_times = []
    frame_sizes_kb = []

    for ep_path in read_sample:
        try:
            store = zarr.open_group(str(ep_path), mode="r")
            keys = list(store.keys())
            if not keys:
                continue
            arr = store[keys[0]]
            n_frames = arr.shape[0]
            if n_frames == 0:
                continue
            idx = random.randint(0, n_frames - 1)
            t0 = time.perf_counter()
            data = arr[idx]
            elapsed = time.perf_counter() - t0
            frame_times.append(elapsed)
            frame_sizes_kb.append(np.asarray(data).nbytes / 1024)
        except Exception:
            continue

    if frame_times:
        ft_ms = [t * 1000 for t in frame_times]
        print(f"Samples: {len(ft_ms)} (key=first array key per episode)")
        print(f"  mean:   {sum(ft_ms)/len(ft_ms):.1f} ms  |  {sum(frame_sizes_kb)/len(frame_sizes_kb):.1f} KB/frame avg")
        print(f"  median: {sorted(ft_ms)[len(ft_ms)//2]:.1f} ms")
        print(f"  p95:    {sorted(ft_ms)[int(len(ft_ms)*0.95)]:.1f} ms")
        print(f"  min:    {min(ft_ms):.1f} ms")
        print(f"  max:    {max(ft_ms):.1f} ms")

    # ------------------------------------------------------------------
    # 4. Sequential chunk throughput (read one full episode array)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SEQUENTIAL READ THROUGHPUT (full array, 5 episodes)")
    print("=" * 60)

    seq_sample = random.sample(all_episodes, 5)
    for ep_path in seq_sample:
        try:
            store = zarr.open_group(str(ep_path), mode="r")
            keys = list(store.keys())
            if not keys:
                continue
            arr = store[keys[0]]
            t0 = time.perf_counter()
            data = arr[:]
            elapsed = time.perf_counter() - t0
            size_mb = np.asarray(data).nbytes / 1e6
            bw = size_mb / elapsed if elapsed > 0 else 0
            print(f"  {ep_path.name[:24]:24s}  {size_mb:6.1f} MB  {elapsed*1000:6.0f} ms  {bw:6.1f} MB/s")
        except Exception as e:
            print(f"  {ep_path.name[:24]:24s}  ERROR: {e}")

    # ------------------------------------------------------------------
    # 5. Concurrent read throughput (simulate N workers)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CONCURRENT READ THROUGHPUT (simulated workers)")
    print("=" * 60)

    def _read_one(ep_path: Path) -> tuple[float, float]:
        """Returns (elapsed_s, bytes_read)."""
        try:
            store = zarr.open_group(str(ep_path), mode="r")
            keys = list(store.keys())
            if not keys:
                return 0.0, 0.0
            arr = store[keys[0]]
            n = arr.shape[0]
            if n == 0:
                return 0.0, 0.0
            idx = random.randint(0, n - 1)
            t0 = time.perf_counter()
            data = arr[idx]
            return time.perf_counter() - t0, float(np.asarray(data).nbytes)
        except Exception:
            return 0.0, 0.0

    for n_workers in [1, 2, 4, 8]:
        batch = random.sample(all_episodes, 32)
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_read_one, batch))
        wall = time.perf_counter() - t0
        total_bytes = sum(b for _, b in results)
        samples_per_sec = len(results) / wall
        mb_per_sec = total_bytes / 1e6 / wall
        print(
            f"  {n_workers:2d} threads:  {wall*1000:5.0f} ms wall  "
            f"{samples_per_sec:5.1f} samples/s  {mb_per_sec:5.1f} MB/s"
        )

    # ------------------------------------------------------------------
    # 6. Estimated training throughput
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ESTIMATED TRAINING THROUGHPUT")
    print("=" * 60)

    if frame_times:
        median_read_s = sorted(frame_times)[len(frame_times) // 2]
        batch_size = 32
        for n_workers in [1, 2, 4, 8, 16, 32]:
            # time to fill one batch = batch_size reads / n_workers (parallel)
            batch_time_s = (batch_size / n_workers) * median_read_s
            batches_per_sec = 1.0 / batch_time_s if batch_time_s > 0 else 0
            samples_per_sec = batches_per_sec * batch_size
            print(
                f"  {n_workers:2d} workers:  ~{batch_time_s*1000:5.0f} ms/batch  "
                f"~{batches_per_sec:.2f} batches/s  ~{samples_per_sec:.0f} samples/s"
            )
        print(f"\n  (based on median frame read time of {median_read_s*1000:.1f} ms)")
        print(f"  GPU step time unknown — check WandB train/step_time for comparison")

    print("\n" + "=" * 60 + "\n")


@app.local_entrypoint()
def main() -> None:
    run_benchmark.remote()
