"""Actual on-volume bytes per arm, so the staging estimate isn't an extrapolation.

Each of the 9 planned runs stages its whole arm onto the training container's
NVMe before step 1, and prepare_epoch blocks for the duration. Sizing that wait
correctly decides prepare_timeout_s.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("measure-arms", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)


@app.function(volumes={MOUNT: vol}, cpu=4, memory=8192, timeout=3600)
def measure():
    root = Path(MOUNT)
    arms_dir = root / "_arms"
    arms = sorted(arms_dir.glob("*.json"))
    print(f"{len(arms)} arm manifests\n")

    # episode hash -> on-disk path, from the catalog we already built
    cat = json.loads((root / "_catalog_cache.json").read_text())
    by_hash = {e["episode_hash"]: root / e["rel_path"] for e in cat}

    def ep_bytes(p: Path) -> int:
        try:
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        except Exception:
            return 0

    out = {}
    for a in arms:
        hashes = json.load(open(a))
        paths = [by_hash[h] for h in hashes if h in by_hash]
        with ThreadPoolExecutor(max_workers=256) as ex:
            sizes = list(ex.map(ep_bytes, paths))
        tot = sum(sizes)
        out[a.stem] = {"eps": len(paths), "gb": round(tot / 1e9, 1),
                       "mb_per_ep": round(tot / 1e6 / max(len(paths), 1), 1)}
        print(f"{a.stem:45s} {len(paths):5d} eps  {tot/1e9:7.1f} GB  "
              f"{tot/1e6/max(len(paths),1):6.1f} MB/ep")

    tot_gb = sum(v["gb"] for v in out.values())
    print(f"\nall arms: {tot_gb:.0f} GB")
    for rate in (7.6, 15.0, 19.0):
        print(f"  stage one avg arm at {rate:5.1f} MB/s: "
              f"{tot_gb/len(out)*1000/rate/3600:5.2f} h")
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(measure.remote(), indent=2))
