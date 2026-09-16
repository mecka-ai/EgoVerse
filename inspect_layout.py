"""What are the 18 files in an episode, and how big is each?

Staging throughput is set by file count, not bytes, so the win from re-sharding
depends on how the bytes are currently distributed across those files.
"""
import json
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("inspect-layout", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)


@app.function(volumes={MOUNT: vol}, cpu=4, memory=8192, timeout=1800)
def inspect():
    root = Path(MOUNT)
    ep = sorted((root / "cleaning-sanitation" / "top").glob("*.zarr"))[0]
    print(f"episode: {ep.name}\n")
    files = sorted((f for f in ep.rglob("*") if f.is_file()),
                   key=lambda f: -f.stat().st_size)
    tot = sum(f.stat().st_size for f in files)
    for f in files:
        print(f"  {f.stat().st_size/1e6:9.2f} MB  {f.relative_to(ep)}")
    print(f"\n  {len(files)} files, {tot/1e6:.1f} MB total")

    # chunk/shard geometry of the dominant arrays
    print("\n--- array metadata ---")
    for sub in sorted(ep.iterdir()):
        zj = sub / "zarr.json"
        if sub.is_dir() and zj.exists():
            m = json.loads(zj.read_text())
            codecs = [c.get("name") for c in m.get("codecs", [])]
            shard = None
            for c in m.get("codecs", []):
                if c.get("name") == "sharding_indexed":
                    shard = c.get("configuration", {}).get("chunk_shape")
            n = len([f for f in sub.rglob("*") if f.is_file()])
            print(f"  {sub.name:28s} shape={m.get('shape')} "
                  f"chunks={m.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape')} "
                  f"shard_inner={shard} codecs={codecs} files={n}")
    return {"files": len(files), "mb": round(tot / 1e6, 1)}


@app.local_entrypoint()
def main():
    print(inspect.remote())
