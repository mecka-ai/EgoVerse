"""Exhaustively search the outputs volume for ANY checkpoint from the arm runs.

An earlier check globbed only <run>/checkpoints/*.ckpt and reported zero. Round 2
trained to epoch ~198, so "no checkpoints at all" deserves verification against
the whole tree rather than one assumed path.
"""
import json
import os
import time
from pathlib import Path

import modal

OUT = "/root/EgoVerse/logs"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("find-ckpts", image=image)
out_vol = modal.Volume.from_name("egoverse-training-outputs", create_if_missing=False)


@app.function(volumes={OUT: out_vol}, cpu=4, memory=8192, timeout=3600)
def find():
    out_vol.reload()
    root = Path(OUT)

    qaexp_dirs = sorted(p for p in root.iterdir()
                        if p.is_dir() and p.name.startswith("qaexp-"))
    print(f"{len(qaexp_dirs)} qaexp run roots\n")

    total = 0
    for d in qaexp_dirs:
        found = []
        for dirpath, _dirnames, filenames in os.walk(d):
            for f in filenames:
                if f.endswith((".ckpt", ".pt", ".safetensors")):
                    fp = Path(dirpath) / f
                    try:
                        st = fp.stat()
                        found.append((str(fp.relative_to(d)), st.st_size,
                                      time.strftime("%m-%d %H:%M",
                                                    time.gmtime(st.st_mtime))))
                    except Exception:
                        found.append((str(fp.relative_to(d)), -1, "?"))
        total += len(found)
        # what DID each run write, so "nothing" is distinguishable from
        # "written somewhere else"
        subdirs = sorted({Path(dp).relative_to(d).parts[0]
                          for dp, _, fs in os.walk(d) if fs and Path(dp) != d})
        print(f"{d.name}: {len(found)} model files | subdirs={subdirs[:6]}")
        for rel, size, mt in found[:5]:
            print(f"    {size/1e9:6.2f} GB  {mt}  {rel}")

    print(f"\nTOTAL model files under qaexp-* roots: {total}")

    # also: does a checkpoints/ dir exist anywhere at all?
    any_ckpt_dirs = [str(Path(dp).relative_to(root))
                     for dp, dn, _ in os.walk(root) if Path(dp).name == "checkpoints"]
    print(f"checkpoints/ dirs anywhere on volume: {len(any_ckpt_dirs)}")
    for c in any_ckpt_dirs[:10]:
        print(f"    {c}")
    return {"total_model_files": total, "checkpoint_dirs": len(any_ckpt_dirs)}


@app.local_entrypoint()
def main():
    print(json.dumps(find.remote(), indent=2))
