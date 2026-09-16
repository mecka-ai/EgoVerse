"""Is the volume slow per-file, or slow per-byte?

If one big file reads far faster than the same bytes split across many files,
the fix is how episodes are stored (fewer, larger shards). If both are equally
slow, it's a hard bandwidth cap and the fix has to be streaming instead of
bulk staging. Also checks cold vs warm, since staging only ever does first
touch.
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
SCRATCH = "/scratch"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("bench-filesize", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)
scratch = modal.Volume.from_name("bench-filesize-scratch", create_if_missing=True)


def _read_all(paths, threads):
    def rd(p):
        with open(p, "rb") as fh:
            n = 0
            while c := fh.read(8 << 20):
                n += len(c)
        return n
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as ex:
        tot = sum(ex.map(rd, paths))
    return tot, time.perf_counter() - t0


@app.function(volumes={MOUNT: vol, SCRATCH: scratch}, cpu=8, memory=16384,
              timeout=3600)
def prep():
    """Write the same bytes as one big file and as many small files."""
    root = Path(MOUNT)
    eps = sorted((root / "cleaning-sanitation" / "top").glob("*.zarr"))[:8]
    blob = bytearray()
    for e in eps:
        for f in sorted(e.rglob("*")):
            if f.is_file():
                blob += f.read_bytes()
    print(f"collected {len(blob)/1e6:.0f} MB from {len(eps)} episodes")

    big = Path(SCRATCH) / "one_big.bin"
    big.write_bytes(blob)

    small_dir = Path(SCRATCH) / "many_small"
    small_dir.mkdir(exist_ok=True)
    chunk = 2_300_000  # ~ the real per-file size
    for i in range(0, len(blob), chunk):
        (small_dir / f"part_{i//chunk:05d}.bin").write_bytes(blob[i:i + chunk])
    n = len(list(small_dir.glob("*.bin")))
    scratch.commit()
    print(f"wrote 1 file of {len(blob)/1e6:.0f} MB and {n} files of ~2.3 MB")
    return {"mb": len(blob) / 1e6, "n_small": n}


@app.function(volumes={SCRATCH: scratch}, cpu=8, memory=16384, timeout=3600)
def read_back():
    scratch.reload()
    big = Path(SCRATCH) / "one_big.bin"
    smalls = sorted((Path(SCRATCH) / "many_small").glob("*.bin"))

    out = {}
    for threads in (1, 16, 64):
        tot, el = _read_all([big], threads if threads == 1 else 1)
        out[f"big_1file"] = round(tot / 1e6 / el, 1)
        print(f"  ONE {tot/1e6:5.0f} MB file, 1 stream      : "
              f"{el:6.1f}s  {tot/1e6/el:7.1f} MB/s")
        break

    for threads in (16, 64, 256):
        tot, el = _read_all(smalls, threads)
        out[f"small_{threads}thr"] = round(tot / 1e6 / el, 1)
        print(f"  {len(smalls):3d} files x ~2.3 MB, {threads:3d} threads : "
              f"{el:6.1f}s  {tot/1e6/el:7.1f} MB/s")

    # warm re-read of the big file: page cache / local caching effects
    tot, el = _read_all([big], 1)
    out["big_warm"] = round(tot / 1e6 / el, 1)
    print(f"  ONE file again (warm)             : {el:6.1f}s  {tot/1e6/el:7.1f} MB/s")
    return out


@app.local_entrypoint()
def main():
    print(prep.remote())
    print(read_back.remote())
