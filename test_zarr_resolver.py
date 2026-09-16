"""Smoke-test ZarrDirEpisodeResolver against the real 6k volume.

Checks the two things that would silently break staging: that n_frames is
actually recoverable from each store's metadata, and that the tar_path entries
point at directories (so _extract_tar_to_dir takes the copytree branch).
"""

import modal

MOUNT = "/vol/zarr_output"
EGOVERSE = "/home/mecka/EgoVerse/.claude/worktrees/fix-frame-count-offbyone"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("zarr>=3.0", "numpy", "numcodecs")
    .add_local_dir(f"{EGOVERSE}/egomimic", remote_path="/root/EgoVerse/egomimic")
)
app = modal.App("test-zarr-resolver", image=image)
vol = modal.Volume.from_name("mecka-zarr-6k-v2", create_if_missing=False)


@app.function(volumes={MOUNT: vol}, cpu=8, memory=16384, timeout=1800, cloud="aws",
              ephemeral_disk=524288)
def check():
    import shutil
    import sys
    import time
    from pathlib import Path

    sys.path.insert(0, "/root/EgoVerse")
    # import the two units directly -- importing the package pulls in torch
    import importlib.util

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m

    ex = load("pf_extract", "/root/EgoVerse/egomimic/rldb/zarr/prefetch/extract.py")

    root = Path(MOUNT)
    groups = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))[:3]
    print(f"scanning groups: {[g.name for g in groups]}")

    # replicate the resolver's catalog scan without importing torch
    import json

    def n_frames(ep: Path):
        try:
            attrs = json.loads((ep / "zarr.json").read_text()).get("attributes", {})
        except Exception:
            return None
        n = attrs.get("total_frames") or attrs.get("n_frames")
        if n:
            return int(n)
        for k, v in (attrs.get("features") or {}).items():
            if v.get("dtype") == "jpeg":
                try:
                    return int(json.loads((ep / k / "zarr.json").read_text())["shape"][0])
                except Exception:
                    return None
        return None

    t0 = time.perf_counter()
    cat = []
    for g in groups:
        for ep in sorted(g.glob("*.zarr")):
            n = n_frames(ep)
            if n:
                cat.append((ep, ep.stem, n))
    el = time.perf_counter() - t0
    print(f"catalog: {len(cat)} episodes in {el:.1f}s")
    if not cat:
        print("!! EMPTY CATALOG -- n_frames not recoverable from zarr metadata")
        return {"ok": False}
    for p, h, n in cat[:4]:
        print(f"  {h[:24]}  n_frames={n:5d}  is_dir={p.is_dir()}")

    nfiles = sum(1 for f in cat[0][0].rglob("*") if f.is_file())
    print(f"files per episode: {nfiles}")
    import subprocess
    print(subprocess.run(["df","-h","/tmp"],capture_output=True,text=True).stdout.strip().splitlines()[-1])
    dest = Path("/tmp/staged_ep")
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    import os
    for threads in (1, 16, 64, 192):
        os.environ["ZARR_STAGE_COPY_THREADS"] = str(threads)
        ex._COPY_THREADS = threads
        d2 = Path(f"/tmp/staged_{threads}")
        if d2.exists():
            shutil.rmtree(d2)
        d2.mkdir(parents=True)
        src = cat[min(threads % len(cat), len(cat) - 1)][0]
        t0 = time.perf_counter()
        nbytes = ex._extract_tar_to_dir(src, d2)
        el = time.perf_counter() - t0
        print(f"  copy threads={threads:4d}: {nbytes/1e6:6.1f} MB in {el:6.1f}s = {nbytes/el/1e6:6.1f} MB/s")
    shutil.copytree(Path("/tmp/staged_64"), dest, dirs_exist_ok=True)

    import zarr

    z = zarr.open(str(dest), mode="r")
    a = z["images.front_1"]
    print(f"staged store opens: images.front_1 shape={a.shape}")
    print(f"round-trip frame 0: {len(a[0:1][0])} bytes")

    # local read speed after staging -- the whole point of staging
    import random
    times = []
    for _ in range(20):
        i = random.randrange(a.shape[0])
        t = time.perf_counter()
        _ = a[i : i + 1]
        times.append(time.perf_counter() - t)
    times.sort()
    print(f"local read after staging: median {times[len(times)//2]*1000:.2f} ms/frame")
    return {"ok": True, "episodes": len(cat)}


@app.local_entrypoint()
def main():
    print(check.remote())
