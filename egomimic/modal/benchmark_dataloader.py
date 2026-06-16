"""
Benchmark raw zarr volume dataloader throughput on Modal.

Sweeps num_workers = [8, 12, 16, 24, 32, 64] and measures how many
indices/second can be pulled from the mecka_data_v2 volume using a
plain torch DataLoader — no collation, no batching, just raw frame reads.

Usage:
    modal run egomimic/modal/benchmark_dataloader.py --env robotics
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

# ---------------------------------------------------------------------------
# Minimal image: zarr + torch only, fast rebuild
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "zarr==3.1.5",
        "numpy",
        "rich",
    )
)

zarr_volume = modal.Volume.from_name("mecka_data_v2")
VOLUME_MOUNT = "/mnt/zarr-data"

app = modal.App("egoverse-dataloader-benchmark", image=image)

WORKER_COUNTS = [8, 12, 16, 24, 32, 64]
WARMUP_SEC = 10      # discard first N seconds to let workers spin up
MEASURE_SEC = 30     # measure window per worker count
MAX_EPISODES = 200   # cap episodes scanned (enough for a stable benchmark)


# ---------------------------------------------------------------------------
# Dataset: flat frame index over all zarr episodes on the volume
# ---------------------------------------------------------------------------

class ZarrVolumeDataset:
    """
    Scans ZARR_ROOT for episode directories, builds a flat (episode, frame)
    index, and reads one frame per __getitem__ call.

    Designed to stress-test volume I/O: opens zarr on first worker access
    (fork-safe), reads the smallest available array key to minimise decode cost.
    """

    def __init__(self, zarr_root: str, max_episodes: int = MAX_EPISODES):
        self.zarr_root = zarr_root
        # Collect (episode_path, frame_count) pairs
        self.index: list[tuple[str, int]] = []
        self._flat: list[tuple[str, int]] = []   # (path, local_frame_idx)
        self._ep_readers: dict[str, object] = {}  # per-worker cache, keyed by path
        self._key: str | None = None              # zarr key to read (set on first open)
        self._build_index(max_episodes)

    def _build_index(self, max_episodes: int) -> None:
        import zarr as _zarr

        root = Path(self.zarr_root)
        if not root.is_dir():
            raise RuntimeError(f"Volume path not found: {root}")

        candidates = sorted(
            p for p in root.iterdir()
            if p.is_dir() and (p.name.endswith(".zarr") or not "." in p.name)
        )
        print(f"[benchmark] Found {len(candidates)} episode dirs, using first {max_episodes}")
        candidates = candidates[:max_episodes]

        key_chosen = None
        for ep_path in candidates:
            try:
                store = _zarr.open_group(str(ep_path), mode="r")
                attrs = dict(store.attrs)
                total_frames = attrs.get("total_frames", 0)
                if total_frames <= 0:
                    continue
                # Pick the smallest array key once (prefer non-image for speed)
                if key_chosen is None:
                    features = attrs.get("features", {})
                    # prefer action / state keys over images (smaller payload)
                    preferred = [k for k in features if "action" in k or "state" in k or "ee_pose" in k]
                    fallback = [k for k in features if "image" not in k and "jpeg" not in k and features[k].get("dtype") != "jpeg"]
                    all_keys = list(features.keys())
                    key_chosen = (preferred or fallback or all_keys or [None])[0]
                self.index.append((str(ep_path), total_frames))
                for fi in range(total_frames):
                    self._flat.append((str(ep_path), fi))
            except Exception as e:
                print(f"[benchmark] skip {ep_path.name}: {e}")
                continue

        self._key = key_chosen
        print(f"[benchmark] Index: {len(self.index)} episodes, {len(self._flat)} frames total")
        print(f"[benchmark] Reading key: {self._key!r}")

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, idx: int):
        import zarr as _zarr

        ep_path, frame_idx = self._flat[idx % len(self._flat)]

        # Open zarr store once per worker process and cache it
        if ep_path not in self._ep_readers:
            self._ep_readers[ep_path] = _zarr.open_group(ep_path, mode="r")

        store = self._ep_readers[ep_path]
        if self._key:
            # zarr 3.x: navigate nested keys via attribute access
            node = store
            for part in self._key.split("."):
                node = node[part]
            _ = node[frame_idx]       # trigger actual I/O read
        # return a small sentinel — we only care about throughput, not values
        return idx


# ---------------------------------------------------------------------------
# Modal function — runs the full sweep inside the container
# ---------------------------------------------------------------------------

@app.function(
    cpu=32.0,
    memory=32768,
    timeout=3600,
    volumes={VOLUME_MOUNT: zarr_volume},
)
def run_benchmark() -> None:
    import torch
    from torch.utils.data import DataLoader

    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.rule("[bold cyan]EgoVerse Volume Dataloader Benchmark")

    ds = ZarrVolumeDataset(VOLUME_MOUNT, max_episodes=MAX_EPISODES)
    n = len(ds)
    console.print(f"Dataset: [bold]{n:,}[/bold] frames from {len(ds.index)} episodes")
    console.print(f"Warmup: {WARMUP_SEC}s  |  Measure: {MEASURE_SEC}s  per worker count\n")

    results: list[dict] = []

    for nw in WORKER_COUNTS:
        console.rule(f"[yellow]workers = {nw}")

        # Build fresh DataLoader for each worker count
        loader = DataLoader(
            ds,
            batch_size=1,
            num_workers=nw,
            shuffle=True,
            collate_fn=lambda x: x,       # no collation overhead
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=False,
        )

        iter_loader = iter(loader)

        # ---- warmup ----
        t_warmup_end = time.perf_counter() + WARMUP_SEC
        warmup_count = 0
        while time.perf_counter() < t_warmup_end:
            try:
                next(iter_loader)
                warmup_count += 1
            except StopIteration:
                iter_loader = iter(loader)

        console.print(f"  warmup done ({warmup_count} frames discarded)")

        # ---- measurement ----
        t0 = time.perf_counter()
        t_end = t0 + MEASURE_SEC
        count = 0
        latencies: list[float] = []

        while True:
            t_now = time.perf_counter()
            if t_now >= t_end:
                break
            t_item = time.perf_counter()
            try:
                next(iter_loader)
            except StopIteration:
                iter_loader = iter(loader)
                continue
            latencies.append(time.perf_counter() - t_item)
            count += 1

        elapsed = time.perf_counter() - t0
        ips = count / elapsed

        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[len(latencies_sorted) // 2] * 1000 if latencies_sorted else 0
        p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)] * 1000 if latencies_sorted else 0
        p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)] * 1000 if latencies_sorted else 0

        console.print(
            f"  [bold green]{ips:,.0f} idx/s[/bold green]  "
            f"(count={count:,}  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms)"
        )
        results.append(dict(workers=nw, ips=ips, count=count, p50=p50, p95=p95, p99=p99))

        # Clean up workers before next run
        del loader, iter_loader

    # ---------------------------------------------------------------------------
    # Final summary table
    # ---------------------------------------------------------------------------
    console.rule("[bold cyan]Summary")
    table = Table(title="Volume Dataloader Throughput Sweep")
    table.add_column("Workers", style="bold yellow", justify="right")
    table.add_column("idx/sec", style="bold green", justify="right")
    table.add_column("vs prev", justify="right")
    table.add_column("p50 (ms)", justify="right")
    table.add_column("p95 (ms)", justify="right")
    table.add_column("p99 (ms)", justify="right")

    prev_ips = None
    peak_row = max(results, key=lambda r: r["ips"]) if results else None
    for r in results:
        delta = ""
        if prev_ips is not None:
            pct = (r["ips"] - prev_ips) / prev_ips * 100
            delta = f"[green]+{pct:.0f}%[/green]" if pct >= 1 else f"[red]{pct:.0f}%[/red]"
        is_peak = peak_row and r["workers"] == peak_row["workers"]
        row_style = "bold" if is_peak else ""
        table.add_row(
            str(r["workers"]),
            f"{r['ips']:,.0f}" + (" ★" if is_peak else ""),
            delta,
            f"{r['p50']:.1f}",
            f"{r['p95']:.1f}",
            f"{r['p99']:.1f}",
            style=row_style,
        )
        prev_ips = r["ips"]

    console.print(table)

    if peak_row:
        console.print(
            f"\n[bold]Peak throughput:[/bold] [bold green]{peak_row['ips']:,.0f} idx/s[/bold green] "
            f"at [bold yellow]{peak_row['workers']} workers[/bold yellow]"
        )

    # Detect contention: first worker count where gains < 10%
    contention_at = None
    for i in range(1, len(results)):
        gain = (results[i]["ips"] - results[i - 1]["ips"]) / results[i - 1]["ips"]
        if gain < 0.05:
            contention_at = results[i]["workers"]
            break
    if contention_at:
        console.print(
            f"[bold]Worker contention detected at:[/bold] [bold red]{contention_at} workers[/bold red] "
            f"(gains dropped below 5%)"
        )
    else:
        console.print("[bold]No contention detected — throughput scaled across all worker counts.[/bold]")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Submitting dataloader benchmark to Modal (robotics env)...")
    run_benchmark.remote()
