"""Separate volume-read from local-write, on cold episodes only.

The first benchmark reused one episode across thread counts, so everything
after the first pass was served from page cache and the numbers were garbage.
Here every configuration gets its own disjoint slice of episodes, and read and
write are timed independently.
"""
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("bench-stage2", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)


def _files(ep: Path):
    return [p for p in ep.rglob("*") if p.is_file()]


@app.function(volumes={MOUNT: vol}, cpu=16, memory=32768, timeout=3600,
              ephemeral_disk=512 * 1024, cloud="aws")
def bench():
    root = Path(MOUNT)
    all_eps = sorted((root / "cleaning-sanitation" / "top").glob("*.zarr"))
    print(f"pool: {len(all_eps):,} episodes\n")

    print("=== where does what live ===")
    for p in ("/tmp", "/cache", "/", "/root"):
        if Path(p).exists():
            st = os.statvfs(p)
            print(f"  {p:8s} {st.f_blocks*st.f_frsize/1e9:8.0f} GB total")
    print(subprocess.run(["df", "-h"], capture_output=True, text=True).stdout)

    cursor = 0

    def take(n):
        nonlocal cursor
        s = all_eps[cursor:cursor + n]
        cursor += n
        return s

    # --- pure read: volume -> memory, nothing written ---
    print("=== pure READ from volume (no write) ===")
    for n_eps, threads in ((1, 32), (4, 128), (16, 512), (32, 1024)):
        eps = take(n_eps)
        jobs = [f for e in eps for f in _files(e)]
        def rd(p):
            with open(p, "rb") as fh:
                n = 0
                while chunk := fh.read(4 << 20):
                    n += len(chunk)
            return n
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            tot = sum(ex.map(rd, jobs))
        el = time.perf_counter() - t0
        print(f"  {n_eps:3d} eps / {len(jobs):5d} files / {threads:5d} thr : "
              f"{el:6.1f}s  {tot/1e6/el:7.1f} MB/s")

    # --- pure write: memory -> local disk ---
    print("\n=== pure WRITE to local disk (data already in RAM) ===")
    eps = take(8)
    blobs = []
    for e in eps:
        for f in _files(e):
            blobs.append((f.relative_to(e.parent), f.read_bytes()))
    tot = sum(len(b) for _, b in blobs)
    for dest_root, label in ((Path("/tmp/w"), "/tmp"), (Path("/cache/w"), "/cache")):
        if not dest_root.parent.exists():
            continue
        shutil.rmtree(dest_root, ignore_errors=True)
        t0 = time.perf_counter()
        for rel, b in blobs:
            d = dest_root / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_bytes(b)
        el = time.perf_counter() - t0
        print(f"  {label:8s}: {el:6.1f}s  {tot/1e6/el:7.1f} MB/s")
        shutil.rmtree(dest_root, ignore_errors=True)

    # --- full stage, cold, scaling episode concurrency ---
    print("\n=== full stage (read+write), cold episodes, 18 threads/ep ===")
    for n_eps in (4, 16, 48):
        eps = take(n_eps)
        dest = Path(f"/tmp/s{n_eps}")
        shutil.rmtree(dest, ignore_errors=True)

        def stage(e):
            fs = _files(e)
            def one(s):
                d = dest / e.name / s.relative_to(e)
                d.parent.mkdir(parents=True, exist_ok=True)
                with open(s, "rb") as fi, open(d, "wb") as fo:
                    shutil.copyfileobj(fi, fo, length=4 << 20)
                return d.stat().st_size
            with ThreadPoolExecutor(max_workers=32) as ex:
                return sum(ex.map(one, fs))

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_eps) as ex:
            tot = sum(ex.map(stage, eps))
        el = time.perf_counter() - t0
        print(f"  {n_eps:3d} eps concurrent : {el:6.1f}s  {tot/1e6/el:7.1f} MB/s "
              f"({n_eps/el:5.2f} eps/s)")
        shutil.rmtree(dest, ignore_errors=True)
    return "ok"


@app.local_entrypoint()
def main():
    print(bench.remote())
