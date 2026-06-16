"""
Benchmark DataLoader __getitem__ throughput on the Modal zarr volume.

Sweeps num_workers = [8, 12, 16, 24, 32, 64] using the exact same
LocalEpisodeResolver → MultiDataset pipeline as training (Mecka cartesian
key_map + transforms), so the numbers reflect real training data throughput.

Usage:
    modal run --env robotics egomimic/modal/benchmark_dataloader.py
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
WORKER_COUNTS = [8, 12, 16, 24, 32, 64]
WARMUP_SEC = 10
MEASURE_SEC = 30
MAX_EPISODES = 300

# Bake the egomimic package into the image so LocalEpisodeResolver is importable
# without a repo clone. modal_setup.image already has all runtime deps.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
bench_image = image.add_local_dir(
    str(REPO_ROOT / "egomimic"),
    remote_path="/root/EgoVerse/egomimic",
    copy=True,
).run_commands(
    # Register the package on sys.path via a .pth file (same trick as _prepare_repo)
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
def run_benchmark() -> None:
    import time

    import torch
    from torch.utils.data import DataLoader

    from rich.console import Console
    from rich.table import Table

    from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
    from egomimic.rldb.embodiment.human import Mecka

    console = Console()
    console.rule("[bold cyan]EgoVerse Volume DataLoader Benchmark")

    key_map = Mecka.get_keymap(mode="cartesian")
    transform_list = Mecka.get_transform_list(mode="cartesian")
    console.print(f"key_map keys: {list(key_map.keys())}")
    console.print(f"transforms:   {[type(t).__name__ for t in transform_list]}\n")

    resolver = LocalEpisodeResolver(
        folder_path=Path(VOLUME_MOUNT),
        key_map=key_map,
        transform_list=transform_list,
        debug=1000,
    )
    console.print("Resolving episodes from volume...")
    dataset = MultiDataset._from_resolver(resolver, mode="train")

    # Cap to MAX_EPISODES so each run is consistent
    all_keys = list(dataset.datasets.keys())[:MAX_EPISODES]
    dataset._build_index_map_from_datasets(
        {k: dataset.datasets[k] for k in all_keys}
    )

    console.print(
        f"Dataset: [bold]{len(dataset):,}[/bold] frames from {len(dataset.datasets)} episodes\n"
        f"Warmup: {WARMUP_SEC}s  |  Measure: {MEASURE_SEC}s  per config\n"
    )

    results: list[dict] = []

    for nw in WORKER_COUNTS:
        console.rule(f"[yellow]num_workers = {nw}")

        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=nw,
            shuffle=True,
            collate_fn=lambda x: x,  # skip collation — just raw __getitem__ output
            prefetch_factor=4,
            persistent_workers=True,
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
            f"  [bold green]{ips:,.0f} idx/s[/bold green]  "
            f"count={count:,}  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms"
        )
        results.append(dict(workers=nw, ips=ips, count=count, p50=p50, p95=p95, p99=p99))
        del loader, it

    # Summary table
    console.rule("[bold cyan]Summary")
    table = Table(title="DataLoader __getitem__ Throughput vs Workers")
    table.add_column("Workers", style="bold yellow", justify="right")
    table.add_column("idx/sec", style="bold green", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("p99 ms", justify="right")

    peak = max(results, key=lambda r: r["ips"])
    prev_ips = None
    for r in results:
        delta = ""
        if prev_ips is not None:
            pct = (r["ips"] - prev_ips) / prev_ips * 100
            delta = f"[green]+{pct:.0f}%[/green]" if pct >= 1 else f"[red]{pct:.0f}%[/red]"
        table.add_row(
            str(r["workers"]),
            f"{r['ips']:,.0f}" + (" ★" if r["workers"] == peak["workers"] else ""),
            delta,
            f"{r['p50']:.1f}",
            f"{r['p95']:.1f}",
            f"{r['p99']:.1f}",
            style="bold" if r["workers"] == peak["workers"] else "",
        )
        prev_ips = r["ips"]

    console.print(table)
    console.print(
        f"\n[bold]Peak:[/bold] [bold green]{peak['ips']:,.0f} idx/s[/bold green] "
        f"at [bold yellow]{peak['workers']} workers[/bold yellow]"
    )

    # Contention = first step where marginal gain < 5%
    for i in range(1, len(results)):
        gain = (results[i]["ips"] - results[i - 1]["ips"]) / results[i - 1]["ips"]
        if gain < 0.05:
            console.print(
                f"[bold]Contention detected at[/bold] [bold red]{results[i]['workers']} workers[/bold red]"
                f" (only +{gain*100:.0f}% gain over {results[i-1]['workers']})"
            )
            break
    else:
        console.print("[bold]No contention — throughput scaled across all worker counts.[/bold]")


@app.local_entrypoint()
def main():
    print("Launching dataloader benchmark on Modal (robotics env)...")
    run_benchmark.remote()
