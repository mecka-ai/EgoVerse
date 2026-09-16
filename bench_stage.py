"""Where do the 156 seconds of a zarr episode stage actually go?

Splits the directory copy into its two phases -- the recursive metadata walk
and the byte copy -- because they have completely different fixes.
"""
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("bench-stage", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)


def _walk_seq(src: Path):
    return [p for p in src.rglob("*") if p.is_file()]


def _walk_threaded(src: Path, threads: int = 64):
    """os.scandir level by level, fanned out -- rglob is one sequential stream."""
    files, dirs = [], [src]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        while dirs:
            results = list(ex.map(lambda d: list(os.scandir(d)), dirs))
            dirs = []
            for entries in results:
                for e in entries:
                    (dirs if e.is_dir() else files).append(Path(e.path))
    return files


def _copy(jobs, threads):
    def one(a):
        s, d = a
        with open(s, "rb") as fi, open(d, "wb") as fo:
            shutil.copyfileobj(fi, fo, length=1024 * 1024)
    with ThreadPoolExecutor(max_workers=threads) as ex:
        list(ex.map(one, jobs))


@app.function(volumes={MOUNT: vol}, cpu=16, memory=32768, timeout=3600,
              ephemeral_disk=512 * 1024, cloud="aws")
def bench():
    root = Path(MOUNT)
    eps = sorted((root / "cleaning-sanitation" / "top").glob("*.zarr"))[:6]
    print(f"{len(eps)} episodes under test\n")

    ep = eps[0]
    t0 = time.perf_counter(); files_seq = _walk_seq(ep); t_seq = time.perf_counter() - t0
    t0 = time.perf_counter(); files_thr = _walk_threaded(ep); t_thr = time.perf_counter() - t0
    nbytes = sum(f.stat().st_size for f in files_seq)
    print(f"episode {ep.name}: {len(files_seq):,} files, {nbytes/1e6:.0f} MB")
    print(f"  walk rglob sequential : {t_seq:6.1f}s")
    print(f"  walk scandir threaded : {t_thr:6.1f}s   ({t_seq/max(t_thr,1e-6):.1f}x)")

    for threads in (64, 256, 512):
        dest = Path(f"/tmp/stage_{threads}"); shutil.rmtree(dest, ignore_errors=True)
        jobs = []
        for s in files_seq:
            d = dest / s.relative_to(ep)
            d.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((s, d))
        t0 = time.perf_counter(); _copy(jobs, threads); el = time.perf_counter() - t0
        print(f"  copy {threads:4d} threads     : {el:6.1f}s  = {nbytes/1e6/el:5.1f} MB/s")
        shutil.rmtree(dest, ignore_errors=True)

    # what the filler actually does today: N episodes at once, each walking
    # sequentially then copying with 64 threads
    def stage_current(e):
        dest = Path("/tmp/cur") / e.name
        fs = _walk_seq(e)
        jobs = []
        for s in fs:
            d = dest / s.relative_to(e); d.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((s, d))
        _copy(jobs, 64)
        return sum(f.stat().st_size for f in fs)

    shutil.rmtree("/tmp/cur", ignore_errors=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=6) as ex:
        tot = sum(ex.map(stage_current, eps))
    el = time.perf_counter() - t0
    print(f"\n6 episodes concurrent, current code: {el:.1f}s "
          f"= {tot/1e6/el:.1f} MB/s aggregate")
    shutil.rmtree("/tmp/cur", ignore_errors=True)
    return {"files_per_ep": len(files_seq), "walk_seq_s": t_seq, "walk_thr_s": t_thr}


@app.local_entrypoint()
def main():
    print(bench.remote())
