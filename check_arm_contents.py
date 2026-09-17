"""Is the bad episode actually in the org-stocking arm manifests?

fix_arm_manifests reported "no arm manifest contained the bad episode", which
contradicts the resolution scan that found it BY iterating those same files.
One of the two is wrong; look at the raw contents rather than trust either.
"""
import json
from pathlib import Path

import modal

BAD = "6a91bcfd87ec349d4b502e59"
MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("check-arm-contents", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)


@app.function(volumes={MOUNT: vol}, cpu=2, memory=4096, timeout=1800)
def check():
    vol.reload()
    arms = Path(MOUNT) / "_arms"
    print("files in _arms:")
    for f in sorted(arms.iterdir()):
        print("  ", f.name)
    for name in ("organization-stocking__bottom", "organization-stocking__random",
                 "organization-stocking__top"):
        f = arms / f"{name}.json"
        if not f.exists():
            print(f"{name}: MISSING")
            continue
        h = json.load(open(f))
        print(f"{name}: n={len(h)} contains_bad={BAD in h} "
              f"sample={h[:1]} type={type(h[0]).__name__}")
    return "ok"


@app.local_entrypoint()
def main():
    print(check.remote())
