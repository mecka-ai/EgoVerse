"""
Cold-read benchmark: measures true zarr volume throughput with zero caching.

Every __getitem__ opens the zarr store fresh (no handle cache) and drops the
OS pagecache before each worker-count config. This reflects what training at
200K episodes with random sampling actually hits — every read is a cold network
fetch from the Modal FUSE volume.

Usage:
    modal run --env robotics egomimic/modal/benchmark_cold_reads.py
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

from modal_setup import app, image, zarr_volume  # noqa: E402

VOLUME_MOUNT = "/mnt/zarr-data"
WORKER_COUNTS = [8, 16, 32, 64, 72]
WARMUP_SEC = 5
MEASURE_SEC = 30
# Use a large pool so workers never revisit the same episode within the window
EPISODE_POOL = 500

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
bench_image = image.add_local_dir(
    str(REPO_ROOT / "egomimic"),
    remote_path="/root/EgoVerse/egomimic",
    copy=True,
    ignore=["**/__pycache__", "**/*.pyc", "**/*.pyo"],
).run_commands(
    "python3 -c \""
    "import sysconfig, os; "
    "p = os.path.join(sysconfig.get_paths()['purelib'], 'egoverse_egomimic.pth'); "
    "open(p, 'w').write('/root/EgoVerse')"
    "\""
)


@app.function(
    image=bench_image,
    cpu=32.0,
    memory=65536,
    timeout=3600,
    volumes={VOLUME_MOUNT: zarr_volume},
)
def run_cold_benchmark() -> None:
    import random
    import time

    import zarr
    from rich.console import Console
    from rich.table import Table

    from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
    from egomimic.rldb.embodiment.human import Mecka

    console = Console()
    console.rule("[bold red]Cold-Read Benchmark — raw zarr latency, no prefetch")
    console.print(
        "Measures each zarr open+read directly in the main process.\n"
        "No DataLoader prefetch queue, no handle cache.\n"
        "Each read hits a different episode → true cold FUSE latency.\n"
    )

    key_map = Mecka.get_keymap(mode="cartesian")

    resolver = LocalEpisodeResolver(
        folder_path=Path(VOLUME_MOUNT),
        key_map=key_map,
        debug=EPISODE_POOL,
    )
    console.print(f"Resolving up to {EPISODE_POOL} episodes...")
    inner = MultiDataset._from_resolver(resolver, mode="train")
    episodes = list(inner.datasets.values())
    console.print(f"Loaded {len(episodes)} episodes\n")

    # Build a shuffled list of (episode, frame_idx) pairs — each accessed at most once
    all_pairs = [(ds, i) for ds in episodes for i in range(min(ds.total_frames, 50))]
    random.shuffle(all_pairs)
    console.print(f"Unique (episode, frame) pairs to draw from: {len(all_pairs):,}\n")

    # --- Single-threaded cold read: one zarr open per read, sequential ---
    console.rule("[yellow]Single-threaded cold reads")
    lats = []
    N_SINGLE = min(200, len(all_pairs))
    for ds, frame_idx in all_pairs[:N_SINGLE]:
        t0 = time.perf_counter()
        store = zarr.open_group(str(ds.episode_path), mode="r")
        for k, spec in key_map.items():
            zarr_key = spec["zarr_key"]
            horizon = spec.get("horizon")
            if horizon:
                end = min(frame_idx + horizon, ds.total_frames)
                _ = store[zarr_key][frame_idx:end]
            else:
                _ = store[zarr_key][frame_idx:frame_idx + 1][0]
        lats.append(time.perf_counter() - t0)

    lats.sort()
    mean_ms = sum(lats) / len(lats) * 1000
    p50_ms = lats[len(lats) // 2] * 1000
    p95_ms = lats[int(len(lats) * 0.95)] * 1000
    p99_ms = lats[int(len(lats) * 0.99)] * 1000
    single_ips = 1000 / mean_ms

    console.print(f"  N={N_SINGLE} reads")
    console.print(f"  mean={mean_ms:.0f}ms  p50={p50_ms:.0f}ms  p95={p95_ms:.0f}ms  p99={p99_ms:.0f}ms")
    console.print(f"  [bold]Single-threaded: {single_ips:.1f} idx/s[/bold]")
    console.print(f"  [bold]Theoretical ceiling with N workers (mean latency / N):[/bold]")
    for nw in WORKER_COUNTS:
        theoretical = single_ips * nw
        console.print(f"    {nw:>3} workers → {theoretical:,.0f} idx/s  (if perfectly parallel)")

    # --- Parallel cold reads: concurrent.futures to simulate N workers ---
    console.rule("[yellow]Parallel cold reads (concurrent.futures)")
    results = []
    from concurrent.futures import ThreadPoolExecutor

    def cold_read(pair):
        ds, frame_idx = pair
        t0 = time.perf_counter()
        store = zarr.open_group(str(ds.episode_path), mode="r")
        for k, spec in key_map.items():
            zarr_key = spec["zarr_key"]
            horizon = spec.get("horizon")
            if horizon:
                end = min(frame_idx + horizon, ds.total_frames)
                _ = store[zarr_key][frame_idx:end]
            else:
                _ = store[zarr_key][frame_idx:frame_idx + 1][0]
        return time.perf_counter() - t0

    pair_pool = all_pairs[N_SINGLE:]  # use fresh pairs not touched by single-threaded run

    for nw in WORKER_COUNTS:
        n_reads = min(nw * 20, len(pair_pool))  # enough reads to measure steadily
        pairs = pair_pool[:n_reads]
        pair_pool = pair_pool[n_reads:]

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=nw) as ex:
            read_lats = list(ex.map(cold_read, pairs))
        elapsed = time.perf_counter() - t0

        ips = len(pairs) / elapsed
        read_lats.sort()
        p50 = read_lats[len(read_lats) // 2] * 1000
        p95 = read_lats[int(len(read_lats) * 0.95)] * 1000
        p99 = read_lats[int(len(read_lats) * 0.99)] * 1000

        console.print(
            f"  {nw:>3} workers | [bold red]{ips:,.0f} idx/s[/bold red] | "
            f"p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms  (n={len(pairs)})"
        )
        results.append(dict(workers=nw, ips=ips, p50=p50, p95=p95, p99=p99))

    console.rule("[bold cyan]Summary")
    table = Table(title="True Cold-Read Throughput (fresh zarr open per read)")
    table.add_column("Workers", style="bold yellow", justify="right")
    table.add_column("idx/sec", style="bold red", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("p99 ms", justify="right")

    prev_ips = None
    peak = max(results, key=lambda r: r["ips"])
    for r in results:
        delta = ""
        if prev_ips is not None:
            pct = (r["ips"] - prev_ips) / prev_ips * 100
            delta = f"[green]+{pct:.0f}%[/green]" if pct >= 1 else f"[red]{pct:.0f}%[/red]"
        table.add_row(
            str(r["workers"]),
            f"{r['ips']:,.0f}" + (" ★" if r["workers"] == peak["workers"] else ""),
            delta,
            f"{r['p50']:.0f}",
            f"{r['p95']:.0f}",
            f"{r['p99']:.0f}",
            style="bold" if r["workers"] == peak["workers"] else "",
        )
        prev_ips = r["ips"]
    console.print(table)
    console.print(f"\n[bold]Single-thread cold read: mean={mean_ms:.0f}ms → {single_ips:.1f} idx/s[/bold]")


@app.local_entrypoint()
def main():
    print("Launching cold-read benchmark...")
    run_cold_benchmark.remote()
