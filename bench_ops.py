"""Is the volume capped on file-opens/sec rather than bytes/sec?

If throughput is set by file count, then reading N files should take time
proportional to N regardless of how big they are -- and the cure for staging is
to cut files per episode, not bytes per episode.

Every configuration reads a disjoint, never-touched slice so nothing is warm.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("bench-ops", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)


@app.function(volumes={MOUNT: vol}, cpu=8, memory=16384, timeout=3600)
def bench():
    root = Path(MOUNT)
    eps = sorted((root / "cleaning-sanitation" / "top").glob("*.zarr"))
    print(f"{len(eps):,} episodes available\n")

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
        print(f"  {label:38s} {len(paths):5d} files {tot/1e6:8.1f} MB "
              f"{el:6.1f}s  {len(paths)/el:6.1f} files/s  {tot/1e6/el:7.1f} MB/s")
        return len(paths) / el

    cursor = 0

    def take(n):
        nonlocal cursor
        s = eps[cursor:cursor + n]
        cursor += n
        return s

    print("=== ONLY the big image file (1 file/episode) ===")
    for n_eps, thr in ((32, 32), (64, 64)):
        sel = take(n_eps)
        paths = [e / "images.front_1" / "c" / "0" for e in sel]
        paths = [p for p in paths if p.exists()]
        run(paths, thr, f"{n_eps} eps, images only, {thr} thr")

    print("\n=== ONLY the tiny json files (many tiny files) ===")
    for n_eps, thr in ((32, 32), (64, 64)):
        sel = take(n_eps)
        paths = [p for e in sel for p in e.rglob("zarr.json")]
        run(paths, thr, f"{n_eps} eps, zarr.json only, {thr} thr")

    print("\n=== WHOLE episodes (18 files each) ===")
    for n_eps, thr in ((32, 32), (64, 64)):
        sel = take(n_eps)
        paths = [p for e in sel for p in e.rglob("*") if p.is_file()]
        run(paths, thr, f"{n_eps} eps, everything, {thr} thr")
    return "ok"


@app.local_entrypoint()
def main():
    print(bench.remote())
