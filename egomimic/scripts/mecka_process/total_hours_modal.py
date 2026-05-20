"""
Total-hours summarizer for the mecka_data_v2 Modal Volume.

Reads only each episode's zarr.json (a small metadata JSON at the root of every
zarr v3 group) — never opens an array. Runs in a single CPU-only container with
a 64-thread reader pool. Cold start dominates; full scan of ~200k episodes
costs well under a cent of compute.

Usage:
    source emimic/bin/activate
    export MODAL_ENVIRONMENT=robotics
    modal run egomimic/scripts/mecka_process/total_hours_modal.py

Optional flags (everything has a sensible default):
    --dataset-root .        path within the volume to walk (default: root)
    --include-deleted       count episodes whose attrs say is_deleted=true
    --max-workers 64        threadpool size inside the container

The script prints a per-embodiment breakdown and the grand total in hours.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

DATA_VOLUME_NAME = os.environ.get("EGOVERSE_DATA_VOLUME", "mecka_data_v2")
DATA_MOUNT = "/data"

image = modal.Image.debian_slim(python_version="3.11")

app = modal.App("egomimic-total-hours", image=image)
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)


@app.function(
    cpu=2.0,
    memory=2048,
    timeout=1800,
    volumes={DATA_MOUNT: data_vol},
)
def aggregate(
    dataset_root: str = ".",
    include_deleted: bool = False,
    max_workers: int = 64,
) -> dict:
    """Walk the volume, read every zarr.json, sum frames per embodiment."""
    import json
    import time
    from concurrent.futures import ThreadPoolExecutor

    data_vol.reload()

    root = Path(DATA_MOUNT) / dataset_root.lstrip("/").rstrip("/") if dataset_root not in ("", ".") else Path(DATA_MOUNT)
    if not root.is_dir():
        return {"error": f"root not found: {root}"}

    t0 = time.time()
    entries = [n for n in os.listdir(root) if not n.startswith(".")]
    t_list = time.time() - t0
    print(f"[total-hours] listed {len(entries):,} entries under {root} in {t_list:.1f}s")

    def _read(name: str):
        p = root / name / "zarr.json"
        try:
            with p.open("rb") as f:
                meta = json.load(f)
        except FileNotFoundError:
            return ("missing_metadata", name, 0, 0.0, False)
        except Exception as e:
            return (f"error:{type(e).__name__}", name, 0, 0.0, False)

        attrs = meta.get("attributes", {}) if isinstance(meta, dict) else {}
        embodiment = str(attrs.get("embodiment") or "UNKNOWN").upper()
        try:
            frames = int(attrs.get("total_frames") or 0)
        except (TypeError, ValueError):
            frames = 0
        try:
            fps = float(attrs.get("fps") or 30.0)
        except (TypeError, ValueError):
            fps = 30.0
        if fps <= 0:
            fps = 30.0
        is_deleted = bool(attrs.get("is_deleted") or False)
        return (embodiment, name, frames, fps, is_deleted)

    by_emb_frames: dict[str, int] = {}
    by_emb_seconds: dict[str, float] = {}
    by_emb_count: dict[str, int] = {}
    missing = 0
    errors: dict[str, int] = {}
    skipped_deleted = 0

    t1 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, row in enumerate(ex.map(_read, entries), start=1):
            tag, _name, frames, fps, is_deleted = row
            if tag == "missing_metadata":
                missing += 1
                continue
            if tag.startswith("error:"):
                errors[tag] = errors.get(tag, 0) + 1
                continue
            if is_deleted and not include_deleted:
                skipped_deleted += 1
                continue
            by_emb_frames[tag] = by_emb_frames.get(tag, 0) + frames
            by_emb_seconds[tag] = by_emb_seconds.get(tag, 0.0) + (frames / fps if fps > 0 else 0.0)
            by_emb_count[tag] = by_emb_count.get(tag, 0) + 1
            if i % 25000 == 0:
                print(f"[total-hours] processed {i:,} / {len(entries):,}")
    t_read = time.time() - t1

    total_frames = sum(by_emb_frames.values())
    total_seconds = sum(by_emb_seconds.values())

    return {
        "dataset_root": str(root),
        "entries_listed": len(entries),
        "episodes_counted": sum(by_emb_count.values()),
        "missing_metadata": missing,
        "errors": errors,
        "skipped_deleted": skipped_deleted,
        "include_deleted": include_deleted,
        "by_embodiment": {
            emb: {
                "episodes": by_emb_count[emb],
                "frames": by_emb_frames[emb],
                "hours": by_emb_seconds[emb] / 3600.0,
            }
            for emb in sorted(by_emb_count.keys())
        },
        "total_frames": total_frames,
        "total_hours": total_seconds / 3600.0,
        "list_seconds": t_list,
        "read_seconds": t_read,
    }


@app.local_entrypoint()
def main(
    dataset_root: str = ".",
    include_deleted: bool = False,
    max_workers: int = 64,
):
    result = aggregate.remote(dataset_root, include_deleted, max_workers)

    if "error" in result:
        print(f"[total-hours] ERROR: {result['error']}")
        return

    print("")
    print(f"Dataset root        : {result['dataset_root']}")
    print(f"Entries listed      : {result['entries_listed']:,}")
    print(f"Episodes counted    : {result['episodes_counted']:,}")
    print(f"Missing zarr.json   : {result['missing_metadata']:,}")
    if result["errors"]:
        print(f"Read errors         : {result['errors']}")
    print(f"Skipped (deleted)   : {result['skipped_deleted']:,}  (include_deleted={result['include_deleted']})")
    print(f"List time           : {result['list_seconds']:.1f}s")
    print(f"Read time           : {result['read_seconds']:.1f}s")
    print("")
    print("By embodiment:")
    print(f"  {'embodiment':<24}{'episodes':>12}{'frames':>16}{'hours':>14}")
    for emb, row in result["by_embodiment"].items():
        print(f"  {emb:<24}{row['episodes']:>12,}{row['frames']:>16,}{row['hours']:>14,.2f}")
    print("")
    print(f"TOTAL frames        : {result['total_frames']:,}")
    print(f"TOTAL hours         : {result['total_hours']:,.2f}")
