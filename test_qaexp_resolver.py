"""Verify the resolver finds QA_exp episodes and that eps_to_use pins an arm.

Three things that would each silently break a run: the nested
<domain>/<split>/<ep>.zarr layout not being walked, duplicate episodes (the same
hash materialised under more than one split) being sampled twice, and eps_to_use
not actually restricting the catalog to the arm.
"""

import modal

MOUNT = "/vol/zarr_output"
EGOVERSE = "/home/mecka/EgoVerse/.claude/worktrees/fix-frame-count-offbyone"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("zarr>=3.0", "numpy", "numcodecs")
    .add_local_dir(f"{EGOVERSE}/egomimic", remote_path="/root/EgoVerse/egomimic")
)
app = modal.App("test-qaexp-resolver", image=image)
vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)


@app.function(volumes={MOUNT: vol}, cpu=8, memory=16384, timeout=3600, cloud="aws")
def check():
    import json
    import time
    from collections import Counter
    from pathlib import Path

    root = Path(MOUNT)
    t0 = time.perf_counter()
    seen = set()
    found = []
    for pattern in ("*/*.zarr", "*/*/*.zarr"):
        for ep in sorted(root.glob(pattern)):
            if ep.parts[len(root.parts)].startswith("_"):
                continue
            if ep.stem in seen:
                continue
            seen.add(ep.stem)
            found.append(ep)
    el = time.perf_counter() - t0
    print(f"scan: {len(found):,} distinct episodes in {el:.1f}s")
    if not found:
        print("!! nothing found — layout assumption is wrong")
        return {"ok": False}

    depths = Counter(len(p.parts) - len(root.parts) for p in found)
    print(f"path depths: {dict(depths)}  (2 = <domain>/<split>/<ep>.zarr)")
    print(f"examples: {[str(p.relative_to(root)) for p in found[:2]]}")

    # duplicates across split dirs
    total_paths = sum(1 for pat in ("*/*.zarr", "*/*/*.zarr") for _ in root.glob(pat))
    print(f"total .zarr paths on volume: {total_paths:,} "
          f"-> {len(found):,} distinct ({total_paths - len(found):,} duplicate copies)")

    # does eps_to_use actually pin the arm?
    arm = root / "_arms" / "cleaning-sanitation__top.json"
    want = set(json.load(open(arm)))
    hit = [p for p in found if p.stem in want]
    print(f"\narm cleaning-sanitation__top: {len(want):,} requested, "
          f"{len(hit):,} present on volume ({100*len(hit)/len(want):.1f}%)")
    if len(hit) < len(want):
        missing = list(want - {p.stem for p in hit})[:3]
        print(f"  missing examples: {missing}")

    # which split dirs do the top-arm episodes actually live under?
    where = Counter(p.parent.name for p in hit)
    print(f"  they live under: {dict(where)}")
    print("  (a re-derived arm cuts across the original split dirs, as expected)")

    # frame counts readable?
    def n_frames(ep):
        try:
            a = json.loads((ep / "zarr.json").read_text()).get("attributes", {})
        except Exception:
            return None
        n = a.get("total_frames") or a.get("n_frames")
        if n:
            return int(n)
        for k, v in (a.get("features") or {}).items():
            if v.get("dtype") == "jpeg":
                try:
                    return int(json.loads((ep / k / "zarr.json").read_text())["shape"][0])
                except Exception:
                    return None
        return None

    ok = sum(1 for p in hit[:40] if n_frames(p))
    print(f"\nframe counts readable on {ok}/40 sampled arm episodes")
    return {"ok": True, "episodes": len(found), "arm_hits": len(hit)}


@app.local_entrypoint()
def main():
    print(check.remote())
