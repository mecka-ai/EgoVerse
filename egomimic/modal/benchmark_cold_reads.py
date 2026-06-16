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
WORKER_COUNTS = [8, 16, 32, 64]
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


class ColdReadDataset:
    """
    Wraps a MultiDataset but opens a fresh zarr store on every __getitem__,
    bypassing the per-worker handle cache entirely. Simulates 200K-episode
    random access where every read is a cold FUSE fetch.
    """

    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx: int):
        import zarr

        dataset_name, local_idx = self.inner.index_map[idx]
        ds = self.inner.datasets[dataset_name]

        # Force a fresh zarr open — no cached handle
        store = zarr.open_group(str(ds.episode_path), mode="r")

        data = {}
        for k, spec in ds.key_map.items():
            zarr_key = spec["zarr_key"]
            horizon = spec.get("horizon")
            if horizon:
                end = min(local_idx + horizon, ds.total_frames)
                raw = store[zarr_key][local_idx:end]
            else:
                raw = store[zarr_key][local_idx:local_idx + 1][0]
            data[k] = raw

        return idx  # we only care about timing, not the actual values


@app.function(
    image=bench_image,
    cpu=32.0,
    memory=65536,
    timeout=3600,
    volumes={VOLUME_MOUNT: zarr_volume},
)
def run_cold_benchmark() -> None:
    import subprocess
    import time

    from torch.utils.data import DataLoader
    from rich.console import Console
    from rich.table import Table

    from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
    from egomimic.rldb.embodiment.human import Mecka

    console = Console()
    console.rule("[bold red]Cold-Read Benchmark (no caching)")
    console.print(
        "Every __getitem__ opens a fresh zarr store.\n"
        "Pagecache is dropped between configs.\n"
        "This is the floor for 200K-episode random-sampling training.\n"
    )

    key_map = Mecka.get_keymap(mode="cartesian")
    transform_list = Mecka.get_transform_list(mode="cartesian")

    resolver = LocalEpisodeResolver(
        folder_path=Path(VOLUME_MOUNT),
        key_map=key_map,
        transform_list=transform_list,
        debug=EPISODE_POOL,
    )
    console.print(f"Resolving up to {EPISODE_POOL} episodes...")
    inner = MultiDataset._from_resolver(resolver, mode="train")
    dataset = ColdReadDataset(inner)
    console.print(f"Dataset: {len(dataset):,} frames from {len(inner.datasets)} episodes\n")

    def drop_caches():
        try:
            subprocess.run(["sync"], check=True)
            Path("/proc/sys/vm/drop_caches").write_text("3")
            console.print("  [dim]pagecache dropped[/dim]")
        except Exception as e:
            console.print(f"  [dim]drop_caches skipped ({e})[/dim]")

    results: list[dict] = []

    for nw in WORKER_COUNTS:
        console.rule(f"[yellow]num_workers = {nw}")
        drop_caches()

        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=nw,
            shuffle=True,
            collate_fn=lambda x: x,
            prefetch_factor=2,
            persistent_workers=False,  # don't persist — each iter is a fresh cold start
            pin_memory=False,
        )
        it = iter(loader)

        # warmup
        t_end = time.perf_counter() + WARMUP_SEC
        warmup = 0
        while time.perf_counter() < t_end:
            try:
                next(it)
                warmup += 1
            except StopIteration:
                it = iter(loader)
        console.print(f"  warmup: {warmup} frames discarded")

        # measure
        t0 = time.perf_counter()
        t_end = t0 + MEASURE_SEC
        count = 0
        lats: list[float] = []
        while time.perf_counter() < t_end:
            ts = time.perf_counter()
            try:
                next(it)
            except StopIteration:
                it = iter(loader)
                continue
            lats.append(time.perf_counter() - ts)
            count += 1

        elapsed = time.perf_counter() - t0
        ips = count / elapsed
        lats.sort()
        p50 = lats[len(lats) // 2] * 1000 if lats else 0
        p95 = lats[int(len(lats) * 0.95)] * 1000 if lats else 0
        p99 = lats[int(len(lats) * 0.99)] * 1000 if lats else 0

        console.print(
            f"  [bold red]{ips:,.1f} idx/s[/bold red]  "
            f"count={count:,}  p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms"
        )
        results.append(dict(workers=nw, ips=ips, count=count, p50=p50, p95=p95, p99=p99))
        del loader, it

    console.rule("[bold cyan]Summary")
    table = Table(title="Cold-Read Throughput (zero cache, 200K-episode simulation)")
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
            f"{r['ips']:.1f}" + (" ★" if r["workers"] == peak["workers"] else ""),
            delta,
            f"{r['p50']:.0f}",
            f"{r['p95']:.0f}",
            f"{r['p99']:.0f}",
            style="bold" if r["workers"] == peak["workers"] else "",
        )
        prev_ips = r["ips"]
    console.print(table)


@app.local_entrypoint()
def main():
    print("Launching cold-read benchmark...")
    run_cold_benchmark.remote()
