"""Do the catalog's recorded paths actually exist on the volume?"""
import json
import random
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("check-catalog-paths", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)


@app.function(volumes={MOUNT: vol}, cpu=8, memory=16384, timeout=1800, cloud="aws")
def check():
    from concurrent.futures import ThreadPoolExecutor

    root = Path(MOUNT)
    raw = json.loads((root / "_catalog_cache.json").read_text())
    print(f"catalog: {len(raw):,} entries")

    rng = random.Random(0)
    sample = rng.sample(raw, min(3000, len(raw)))
    with ThreadPoolExecutor(max_workers=64) as ex:
        exists = list(ex.map(lambda e: Path(e["path"]).exists(), sample))
    miss = [e for e, ok in zip(sample, exists) if not ok]
    print(f"sampled {len(sample):,}: {len(miss):,} missing "
          f"({100*len(miss)/len(sample):.2f}%)")
    for e in miss[:5]:
        p = Path(e["path"])
        print(f"  MISSING {p}")
        # is it somewhere else under the same domain?
        dom = p.parent.parent
        hits = list(dom.glob(f"*/{p.name}")) if dom.exists() else []
        print(f"     elsewhere under {dom.name}: {[str(h.parent.name) for h in hits]}")

    # the specific ones the run tripped on
    for eid in ("6a8418504431b6c060572b3c", "6a8d7df217e559288f3a444a"):
        hits = list((root / "cleaning-sanitation").glob(f"*/{eid}.zarr"))
        entry = next((e for e in raw if e["episode_hash"] == eid), None)
        print(f"\n{eid}:")
        print(f"  catalog says: {entry['path'] if entry else '<not in catalog>'}")
        print(f"  actually at:  {[str(h) for h in hits] or '<nowhere under cleaning-sanitation>'}")
        anywhere = list(root.glob(f"*/*/{eid}.zarr"))
        print(f"  anywhere:     {[str(a) for a in anywhere] or '<not on volume>'}")
    return {"missing_pct": 100 * len(miss) / len(sample)}


@app.local_entrypoint()
def main():
    print(check.remote())
