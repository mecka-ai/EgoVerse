"""Actual bytes for the two 20k-episode training pools.

The staged path copies an epoch window to local NVMe before step 1, so pool
size decides both whether it fits pool_size_gb and how long the GPU sits idle.
Sized from real stat() calls, not a per-episode extrapolation.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("measure-pools", image=image)
vol = modal.Volume.from_name("mecka-zarr-train", create_if_missing=False)


@app.function(volumes={MOUNT: vol}, cpu=8, memory=16384, timeout=3600)
def measure():
    root = Path(MOUNT)
    cat = json.loads((root / "_catalog_cache.json").read_text())
    by_hash = {e["episode_hash"]: root / e["rel_path"] for e in cat}
    print(f"catalog: {len(cat):,} episodes")

    def ep_bytes(p: Path) -> int:
        try:
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        except Exception:
            return 0

    out = {}
    for name in ("train_pool_2600h", "train_pool_6000h"):
        f = root / "_manifest" / f"{name}.json"
        if not f.exists():
            print(f"{name}: MISSING at {f}")
            continue
        hashes = json.load(open(f))
        paths = [by_hash[h] for h in hashes if h in by_hash]
        with ThreadPoolExecutor(max_workers=256) as ex:
            sizes = list(ex.map(ep_bytes, paths))
        tot = sum(sizes)
        out[name] = {"eps_in_manifest": len(hashes), "eps_found": len(paths),
                     "gb": round(tot / 1e9, 1),
                     "mb_per_ep": round(tot / 1e6 / max(len(paths), 1), 1)}
        print(f"\n{name}: {len(paths):,}/{len(hashes):,} found, "
              f"{tot/1e9:.0f} GB, {tot/1e6/max(len(paths),1):.1f} MB/ep")
        for rate in (8.0, 15.0, 19.0):
            print(f"   stage entire pool at {rate:4.1f} MB/s: {tot/1e6/rate/3600:6.1f} h")
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(measure.remote(), indent=2))
