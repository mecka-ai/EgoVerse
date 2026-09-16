"""Pre-build the zarr catalog cache on a CPU container.

The resolver builds this on first use, but on a training box that is an idle
H200 sitting through tens of thousands of small FUSE reads. Doing it once here,
threaded, means every training run starts on a cache hit.

    modal run -e robotics build_catalog.py --volume mecka-zarr-qaexp
"""

import json
import time
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("build-catalog", image=image)
qaexp = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)
train = modal.Volume.from_name("mecka-zarr-train", create_if_missing=False)


def _scan(root: Path, threads: int = 128):
    from concurrent.futures import ThreadPoolExecutor

    seen: set[str] = set()
    eps: list[Path] = []
    for pattern in ("*/*.zarr", "*/*/*.zarr"):
        for ep in sorted(root.glob(pattern)):
            if ep.parts[len(root.parts)].startswith("_"):
                continue
            if ep.stem in seen:
                continue
            seen.add(ep.stem)
            eps.append(ep)
    print(f"  {len(eps):,} distinct episodes discovered")

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
    with ThreadPoolExecutor(max_workers=threads) as ex:
        counts = list(ex.map(n_frames, eps))
    el = time.perf_counter() - t0
    print(f"  metadata read in {el:.1f}s ({len(eps)/el:.0f} eps/s, {threads} threads)")

    # relative to the volume root: this container mounts at /vol/zarr_output but
    # training mounts the same volume at /mnt/zarr-data, so absolute paths here
    # resolve to nothing there.
    return [
        {"rel_path": str(ep.relative_to(root)), "episode_hash": ep.stem,
         "n_frames": n, "group": "/".join(ep.parts[len(root.parts):-1])}
        for ep, n in zip(eps, counts) if n
    ]


@app.function(volumes={MOUNT: qaexp}, cpu=16, memory=16384, timeout=3600, cloud="aws")
def build_qaexp():
    root = Path(MOUNT)
    raw = _scan(root)
    out = root / "_catalog_cache.json"
    out.write_text(json.dumps(raw))
    qaexp.commit()
    print(f"wrote {out}: {len(raw):,} entries, "
          f"{sum(r['n_frames'] for r in raw):,} frames")
    return {"entries": len(raw)}


@app.function(volumes={MOUNT: train}, cpu=16, memory=16384, timeout=3600, cloud="aws")
def build_train():
    root = Path(MOUNT)
    raw = _scan(root)
    out = root / "_catalog_cache.json"
    out.write_text(json.dumps(raw))
    train.commit()
    print(f"wrote {out}: {len(raw):,} entries, "
          f"{sum(r['n_frames'] for r in raw):,} frames")
    return {"entries": len(raw)}


@app.local_entrypoint()
def main(volume: str = "both"):
    if volume in ("both", "mecka-zarr-qaexp"):
        print("qaexp:", build_qaexp.remote())
    if volume in ("both", "mecka-zarr-train"):
        print("train:", build_train.remote())
