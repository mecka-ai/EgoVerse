"""Count converted episodes on mecka-zarr-train against the union manifest.

Fanned out across containers: a single container walking ~1,777 group dirs on a
FUSE mount is far too slow.
"""

import json
from pathlib import Path

import modal

MOUNT = "/vol/zarr_output"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("count-train", image=image)
vol = modal.Volume.from_name("mecka-zarr-train", create_if_missing=False)


@app.function(volumes={MOUNT: vol}, cpu=1, memory=1024, timeout=1800,
              max_containers=200, cloud="aws")
def count_group(group_id: str) -> tuple:
    d = Path(MOUNT) / group_id
    return (group_id, [p.stem for p in d.glob("*.zarr")] if d.exists() else [])


@app.function(volumes={MOUNT: vol}, cpu=2, memory=4096, timeout=7200, cloud="aws")
def run() -> dict:
    import csv

    root = Path(MOUNT)
    want = {}
    with open(root / "_manifest" / "train_union.csv") as f:
        for row in csv.DictReader(f):
            want.setdefault(row["group_id"], set()).add(row["episode_id"])
    groups = sorted(want)
    print(f"manifest: {sum(len(v) for v in want.values()):,} episodes in {len(groups):,} groups")

    have = {}
    for gid, eids in count_group.map(groups):
        have[gid] = set(eids)

    got = sum(len(have.get(g, set()) & want[g]) for g in groups)
    exp = sum(len(v) for v in want.values())
    missing = [(g, e) for g in groups for e in (want[g] - have.get(g, set()))]
    print(f"converted: {got:,} / {exp:,}  = {100*got/exp:.2f}%")
    print(f"missing:   {len(missing):,}")

    # confirm both arms survived -- an arm short of episodes would quietly
    # unbalance the comparison it exists to make
    for pool in ("2600h", "6000h"):
        p = root / "_manifest" / f"train_pool_{pool}.json"
        if p.exists():
            ids = set(json.load(open(p)))
            present = sum(1 for g in groups for e in (have.get(g, set()) & ids))
            print(f"  pool {pool}: {present:,}/{len(ids):,} present "
                  f"({100*present/len(ids):.2f}%)")

    with open(root / "_manifest" / "train_missing.json", "w") as f:
        json.dump([e for _, e in missing], f)
    vol.commit()
    return {"converted": got, "expected": exp, "missing": len(missing),
            "coverage_pct": 100 * got / exp}


@app.local_entrypoint()
def main():
    print(json.dumps(run.remote(), indent=2))
