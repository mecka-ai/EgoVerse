"""Can concurrency alone rescue the mid-sized arrays?

Episodes are 1 x 37.6 MB image file (streams at ~280 MB/s) plus 17 smaller
files. Reading everything runs ~6x slower than reading images alone, so the
small files dominate staging. If throwing threads at JUST those files fixes
them, staging is a config change. If not, the data layout has to change.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("bench-small", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)


@app.function(volumes={MOUNT: vol}, cpu=8, memory=16384, timeout=3600)
def bench():
    root = Path(MOUNT)
    eps = sorted((root / "cleaning-sanitation" / "top").glob("*.zarr"))
    cursor = 0

    def take(n):
        nonlocal cursor
        s = eps[cursor:cursor + n]
        cursor += n
        return s

    def rd(p):
        with open(p, "rb") as fh:
            n = 0
            while c := fh.read(8 << 20):
                n += len(c)
        return n

    def run(paths, threads, label):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            tot = sum(ex.map(rd, paths))
        el = time.perf_counter() - t0
        print(f"  {label:34s} {len(paths):5d} files {tot/1e6:7.1f} MB "
              f"{el:6.1f}s  {len(paths)/el:7.1f} files/s  {tot/1e6/el:6.1f} MB/s")

    # non-image, non-json data files: the keypoint/pose arrays, 0.15-1.5 MB each
    print("=== mid-sized arrays only (no images, no json) ===")
    for thr in (32, 128, 512, 1024):
        sel = take(32)
        paths = [
            p for e in sel for p in e.rglob("*")
            if p.is_file() and p.name != "zarr.json" and "images.front_1" not in str(p)
        ]
        run(paths, thr, f"32 eps, {thr} threads")

    print("\n=== images only, for reference ===")
    for thr in (32, 128):
        sel = take(32)
        paths = [e / "images.front_1" / "c" / "0" for e in sel]
        paths = [p for p in paths if p.exists()]
        run(paths, thr, f"32 eps, {thr} threads")
    return "ok"


@app.local_entrypoint()
def main():
    print(bench.remote())
