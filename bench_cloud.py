"""Is the 7 MB/s volume read a cross-cloud penalty?

The volumes live in rkd. Everything in this session pins cloud="aws" to escape
the rkd scheduling queue, and cross-cloud reachability was verified -- but never
cross-cloud *throughput*. Same read, same episodes, three placements.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("bench-cloud", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)


def _read(skip: int, n_eps: int, threads: int) -> dict:
    root = Path(MOUNT)
    eps = sorted((root / "cleaning-sanitation" / "top").glob("*.zarr"))[skip:skip + n_eps]
    jobs = [f for e in eps for f in e.rglob("*") if f.is_file()]

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
    return {"eps": len(eps), "files": len(jobs), "threads": threads,
            "sec": round(el, 1), "mb_s": round(tot / 1e6 / el, 1)}


@app.function(volumes={MOUNT: vol}, cpu=16, memory=32768, timeout=1800, cloud="aws")
def read_aws(skip: int = 100, n_eps: int = 16, threads: int = 256):
    return {"cloud": "aws", **_read(skip, n_eps, threads)}


@app.function(volumes={MOUNT: vol}, cpu=16, memory=32768, timeout=1800, cloud="oci")
def read_rkd(skip: int = 200, n_eps: int = 16, threads: int = 256):
    return {"cloud": "rkd", **_read(skip, n_eps, threads)}


@app.function(volumes={MOUNT: vol}, cpu=16, memory=32768, timeout=1800)
def read_default(skip: int = 300, n_eps: int = 16, threads: int = 256):
    return {"cloud": "default", **_read(skip, n_eps, threads)}


@app.local_entrypoint()
def main():
    for f in (read_default, read_aws, read_rkd):
        try:
            print(f.remote())
        except Exception as e:
            print(f"{f}: FAILED {type(e).__name__}: {e}")
