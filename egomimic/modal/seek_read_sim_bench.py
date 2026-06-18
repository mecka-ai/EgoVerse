"""Synthetic seek-latency benchmark for HDD-style dataloader access.

This benchmark intentionally does not import torch or zarr. It isolates the
storage pattern researchers are pointing at:

* main/zarr-like: every sample/camera pays two non-contiguous disk operations
  (shard index + JPEG payload).
* pack-block-like: workers read contiguous frame blocks from packed camera
  files; one seek per camera/block, then nearby samples are served from memory.

The disk is modeled as a single seek/transfer queue by serializing every disk
operation behind one multiprocessing lock. That approximates the thing HDDs are
bad at: many workers contending for one mechanical head.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any


_DISK_LOCK: Any = None
_SEEK_MS = 8.0
_SEQUENTIAL_MBPS = 200.0
_SERIALIZE = True


@dataclass
class CaseResult:
    label: str
    samples: int
    workers: int
    cameras: int
    seconds: float
    samples_per_s: float
    payload_mb_per_s: float
    disk_mb_per_s: float
    disk_ops: int
    seeks: int
    seeks_per_sample: float
    disk_bytes: int
    payload_bytes: int


def _init_worker(
    disk_lock: Any,
    seek_ms: float,
    sequential_mbps: float,
    serialize: bool,
) -> None:
    global _DISK_LOCK, _SEEK_MS, _SEQUENTIAL_MBPS, _SERIALIZE
    _DISK_LOCK = disk_lock
    _SEEK_MS = float(seek_ms)
    _SEQUENTIAL_MBPS = float(sequential_mbps)
    _SERIALIZE = bool(serialize)


def _disk_op(nbytes: int, *, seeks: int = 1) -> None:
    transfer_s = nbytes / max(_SEQUENTIAL_MBPS, 1e-9) / 1_000_000.0
    delay_s = (max(0, seeks) * _SEEK_MS / 1000.0) + transfer_s
    if _SERIALIZE and _DISK_LOCK is not None:
        with _DISK_LOCK:
            time.sleep(delay_s)
    else:
        time.sleep(delay_s)


def _main_worker(args: tuple[int, int, int, int, int]) -> tuple[int, int, int, int]:
    worker_id, n_samples, cameras, payload_bytes, index_bytes = args
    rng = random.Random(10_000 + worker_id)
    disk_ops = seeks = disk_bytes = payload_read = 0
    for _ in range(n_samples):
        # Touch the RNG so this remains a random-read workload shape even though
        # the synthetic delay is independent of the chosen episode/frame.
        rng.randrange(1_000_000)
        for _cam in range(cameras):
            _disk_op(index_bytes, seeks=1)
            _disk_op(payload_bytes, seeks=1)
            disk_ops += 2
            seeks += 2
            disk_bytes += index_bytes + payload_bytes
            payload_read += payload_bytes
    return disk_ops, seeks, disk_bytes, payload_read


def _pack_block_worker(
    args: tuple[int, int, int, int, int]
) -> tuple[int, int, int, int]:
    worker_id, n_samples, cameras, payload_bytes, block_size = args
    rng = random.Random(20_000 + worker_id)
    disk_ops = seeks = disk_bytes = payload_read = 0
    remaining = n_samples
    while remaining > 0:
        block_n = min(block_size, remaining)
        # Blocks are shuffled at episode/block granularity, not frame granularity.
        rng.randrange(1_000_000)
        for _cam in range(cameras):
            nbytes = block_n * payload_bytes
            _disk_op(nbytes, seeks=1)
            disk_ops += 1
            seeks += 1
            disk_bytes += nbytes
            payload_read += nbytes
        remaining -= block_n
    return disk_ops, seeks, disk_bytes, payload_read


def _split_work(samples: int, workers: int) -> list[int]:
    base = samples // workers
    extra = samples % workers
    return [base + (1 if i < extra else 0) for i in range(workers)]


def _run_case(
    *,
    label: str,
    worker_fn: Any,
    worker_args: list[tuple],
    workers: int,
    samples: int,
    cameras: int,
    seek_ms: float,
    sequential_mbps: float,
    serialize: bool,
) -> CaseResult:
    lock = mp.Lock()
    t0 = time.perf_counter()
    with mp.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(lock, seek_ms, sequential_mbps, serialize),
    ) as pool:
        parts = pool.map(worker_fn, worker_args)
    dt = time.perf_counter() - t0
    disk_ops = sum(p[0] for p in parts)
    seeks = sum(p[1] for p in parts)
    disk_bytes = sum(p[2] for p in parts)
    payload_bytes = sum(p[3] for p in parts)
    return CaseResult(
        label=label,
        samples=samples,
        workers=workers,
        cameras=cameras,
        seconds=dt,
        samples_per_s=samples / dt,
        payload_mb_per_s=payload_bytes / dt / 1_000_000.0,
        disk_mb_per_s=disk_bytes / dt / 1_000_000.0,
        disk_ops=disk_ops,
        seeks=seeks,
        seeks_per_sample=seeks / max(samples, 1),
        disk_bytes=disk_bytes,
        payload_bytes=payload_bytes,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    work = _split_work(args.samples, args.workers)
    main_args = [
        (i, n, args.cameras, args.payload_kb * 1024, args.index_kb * 1024)
        for i, n in enumerate(work)
    ]
    pack_args = [
        (i, n, args.cameras, args.payload_kb * 1024, args.block_size)
        for i, n in enumerate(work)
    ]

    main = _run_case(
        label="main_zarr_random",
        worker_fn=_main_worker,
        worker_args=main_args,
        workers=args.workers,
        samples=args.samples,
        cameras=args.cameras,
        seek_ms=args.seek_ms,
        sequential_mbps=args.sequential_mbps,
        serialize=args.serialize_seeks,
    )
    pack = _run_case(
        label="pr_pack_block_readahead",
        worker_fn=_pack_block_worker,
        worker_args=pack_args,
        workers=args.workers,
        samples=args.samples,
        cameras=args.cameras,
        seek_ms=args.seek_ms,
        sequential_mbps=args.sequential_mbps,
        serialize=args.serialize_seeks,
    )
    summary = {
        "config": {
            "samples": args.samples,
            "workers": args.workers,
            "cameras": args.cameras,
            "payload_kb": args.payload_kb,
            "index_kb": args.index_kb,
            "seek_ms": args.seek_ms,
            "sequential_mbps": args.sequential_mbps,
            "block_size": args.block_size,
            "serialize_seeks": args.serialize_seeks,
            "expected_main_seeks_per_sample": args.cameras * 2,
            "expected_pack_seeks_per_sample": (
                args.cameras * math.ceil(args.samples / args.workers / args.block_size)
                * args.workers
                / args.samples
            ),
        },
        "main": asdict(main),
        "pack": asdict(pack),
        "speedup": {
            "wall_clock_x": main.seconds / pack.seconds,
            "samples_per_s_x": pack.samples_per_s / main.samples_per_s,
            "seek_reduction_x": main.seeks / max(pack.seeks, 1),
            "disk_op_reduction_x": main.disk_ops / max(pack.disk_ops, 1),
            "seeks_per_sample_reduction_x": main.seeks_per_sample
            / max(pack.seeks_per_sample, 1e-12),
        },
    }
    return summary


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=1024)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--cameras", type=int, default=1)
    p.add_argument("--payload-kb", type=int, default=100)
    p.add_argument("--index-kb", type=int, default=46)
    p.add_argument("--seek-ms", type=float, default=8.0)
    p.add_argument("--sequential-mbps", type=float, default=200.0)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--serialize-seeks", action=argparse.BooleanOptionalAction, default=True)
    return p


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    summary = run_benchmark(args)
    print("SEEK_SIM_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    return summary


try:
    import modal

    _app = modal.App(
        "egoverse-seek-sim",
        image=modal.Image.debian_slim().add_local_file(
            __file__, remote_path="/root/seek_read_sim_bench.py"
        ),
    )

    @_app.function(cpu=8.0, memory=2048, timeout=1800)
    def run_remote(argv: list[str]) -> dict[str, Any]:
        return main(argv)

    @_app.local_entrypoint()
    def modal_main(*argv: str) -> None:
        summary = run_remote.remote(list(argv))
        print("MODAL_SEEK_SIM_RESULT", json.dumps(summary, sort_keys=True), flush=True)

except Exception:
    pass


if __name__ == "__main__":
    main()
