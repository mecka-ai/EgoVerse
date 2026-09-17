"""How many arm episodes are NOT at the expected 126x224?

Two runs died with
    stack expects each tensor to be equal size,
    but got [3, 126, 224] at entry 0 and [3, 360, 640] at entry 44
so the reconversion did not land on every episode. Any batch that happens to mix
resolutions kills the run, which makes this probabilistic and every run exposed.

Reads the stored image shape from each episode's zarr metadata, per arm, so the
offenders can be excluded or reconverted.
"""
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("scan-resolutions", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)

DOMAINS = ["cleaning-sanitation", "organization-stocking", "dish-handling"]
ARMS = ["top", "bottom", "random"]


@app.function(volumes={MOUNT: vol}, cpu=8, memory=16384, timeout=7200)
def scan():
    root = Path(MOUNT)
    cat = json.loads((root / "_catalog_cache.json").read_text())
    by_hash = {e["episode_hash"]: root / e["rel_path"] for e in cat}

    def shape_of(h):
        p = by_hash.get(h)
        if p is None:
            return h, None
        try:
            m = json.loads((p / "images.front_1" / "zarr.json").read_text())
            attrs = json.loads((p / "zarr.json").read_text()).get("attributes", {})
            feat = (attrs.get("features") or {}).get("images.front_1") or {}
            hw = feat.get("shape") or feat.get("image_shape")
            if hw:
                return h, tuple(hw[:2]) if len(hw) >= 2 else None
            return h, tuple(m.get("shape", [])[:1]) or None
        except Exception:
            return h, None

    out = {}
    offenders = defaultdict(list)
    for dom in DOMAINS:
        for arm in ARMS:
            f = root / "_arms" / f"{dom}__{arm}.json"
            if not f.exists():
                continue
            hashes = json.load(open(f))
            with ThreadPoolExecutor(max_workers=256) as ex:
                res = list(ex.map(shape_of, hashes))
            c = Counter(s for _, s in res)
            out[f"{dom}__{arm}"] = {str(k): v for k, v in c.most_common()}
            for h, s in res:
                if s is not None and s != (126, 224):
                    offenders[f"{dom}__{arm}"].append(h)
            print(f"{dom}__{arm}: {dict(c.most_common(5))}")

    allbad = sorted({h for v in offenders.values() for h in v})
    (root / "_arms" / "_wrong_resolution.json").write_text(json.dumps(allbad))
    vol.commit()
    print(f"\ntotal distinct wrong-resolution episodes: {len(allbad)}")
    for k, v in offenders.items():
        print(f"  {k}: {len(v)}")
    return {"counts": out, "n_wrong": len(allbad)}


@app.local_entrypoint()
def main():
    print(json.dumps(scan.remote(), indent=2)[:3000])
