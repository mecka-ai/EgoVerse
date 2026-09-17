"""Drop the one un-reconverted episode from the arms that contain it.

6a91bcfd87ec349d4b502e59 is still stored at 360x640 while every other episode is
126x224 -- it is the single episode the QA_exp reconversion missed (58,481 of
58,482). Any batch that happens to draw it alongside a normal episode dies with

    stack expects each tensor to be equal size,
    but got [3, 126, 224] at entry 0 and [3, 360, 640] at entry 44

which is exactly how organization-stocking bottom and random died. It appears in
no other arm, so no other run is exposed.
"""
import json
from pathlib import Path

import modal

BAD = "6a91bcfd87ec349d4b502e59"
MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("fix-arm-manifests", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)


@app.function(volumes={MOUNT: vol}, cpu=2, memory=4096, timeout=1800)
def fix():
    arms = Path(MOUNT) / "_arms"
    changed = {}
    for f in sorted(arms.glob("*.json")):
        if f.name.startswith("_"):
            continue
        hashes = json.load(open(f))
        if BAD not in hashes:
            continue
        kept = [h for h in hashes if h != BAD]
        f.write_text(json.dumps(kept))
        changed[f.stem] = {"before": len(hashes), "after": len(kept)}
        print(f"{f.stem}: {len(hashes)} -> {len(kept)}")
    vol.commit()
    if not changed:
        print("no arm manifest contained the bad episode")
    return changed


@app.local_entrypoint()
def main():
    print(json.dumps(fix.remote(), indent=2))
