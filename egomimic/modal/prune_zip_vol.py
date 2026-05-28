"""Prune mecka_data_zip volume to at most 70 000 complete episodes.

Pruning rules
-------------
1. Remove orphans: any .tar without a matching .done, or .done without a .tar.
2. Keep only the first 70 000 episodes when sorted by episode_hash (alphabetical).
   This naturally preserves the first 1 000 entries (used by debug=1000 training runs).
3. Delete both .tar and .done for every episode beyond the 70 000 cap.
4. Rewrite catalog.json with only the surviving episodes.

Usage
-----
# Dry run — show what would be deleted without touching anything:
MODAL_ENVIRONMENT=robotics modal run egomimic/modal/prune_zip_vol.py -- --dry-run

# Live prune:
MODAL_ENVIRONMENT=robotics modal run egomimic/modal/prune_zip_vol.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import modal
from modal_setup import app, zip_volume, image

ZIP_MOUNT = "/mnt/zarr-zip"
KEEP_CAP = 70_000


@app.function(
    image=image,
    volumes={ZIP_MOUNT: zip_volume},
    cpu=4,
    memory=16384,
    timeout=1800,
)
def prune(dry_run: bool = False) -> dict:
    """Prune the zip volume in-container and return a summary dict."""
    zip_path = Path(ZIP_MOUNT)

    # -----------------------------------------------------------------------
    # 1. Inventory
    # -----------------------------------------------------------------------
    tars  = {p.stem for p in zip_path.glob("*.tar")}
    dones = {p.stem for p in zip_path.glob("*.done")}

    valid    = sorted(tars & dones)          # both present — sorted by episode_hash
    orphan_tars  = tars - dones              # tar without done (partial write)
    orphan_dones = dones - tars              # done without tar (stale sentinel)

    print(f"Inventory: {len(valid):,} valid pairs, "
          f"{len(orphan_tars):,} orphan tars, "
          f"{len(orphan_dones):,} orphan dones")

    # -----------------------------------------------------------------------
    # 2. Decide what to keep / delete
    # -----------------------------------------------------------------------
    keep_set   = set(valid[:KEEP_CAP])
    prune_set  = set(valid[KEEP_CAP:])      # valid pairs beyond the cap

    print(f"Keeping {len(keep_set):,} episodes (first {KEEP_CAP:,} by episode_hash sort)")
    print(f"Pruning {len(prune_set):,} episodes beyond cap")
    print(f"Removing {len(orphan_tars):,} orphan tars + {len(orphan_dones):,} orphan dones")

    if dry_run:
        if prune_set:
            sample = sorted(prune_set)[:5]
            print(f"Dry run — first 5 episodes that would be deleted: {sample}")
        print("Dry run — no files deleted.")
        return {
            "keep": len(keep_set),
            "prune": len(prune_set),
            "orphan_tars": len(orphan_tars),
            "orphan_dones": len(orphan_dones),
            "dry_run": True,
        }

    # -----------------------------------------------------------------------
    # 3. Delete pruned episodes (both .tar and .done)
    # -----------------------------------------------------------------------
    deleted_pairs = 0
    for ep_hash in sorted(prune_set):
        (zip_path / f"{ep_hash}.tar").unlink(missing_ok=True)
        (zip_path / f"{ep_hash}.done").unlink(missing_ok=True)
        deleted_pairs += 1

    # -----------------------------------------------------------------------
    # 4. Delete orphan files
    # -----------------------------------------------------------------------
    deleted_orphan_tars = deleted_orphan_dones = 0
    for ep_hash in sorted(orphan_tars):
        (zip_path / f"{ep_hash}.tar").unlink(missing_ok=True)
        deleted_orphan_tars += 1
    for ep_hash in sorted(orphan_dones):
        (zip_path / f"{ep_hash}.done").unlink(missing_ok=True)
        deleted_orphan_dones += 1

    print(f"Deleted {deleted_pairs:,} episode pairs, "
          f"{deleted_orphan_tars:,} orphan tars, "
          f"{deleted_orphan_dones:,} orphan dones")

    # -----------------------------------------------------------------------
    # 5. Rewrite catalog.json
    # -----------------------------------------------------------------------
    catalog_path = zip_path / "catalog.json"
    existing_meta: dict[str, dict] = {}
    if catalog_path.exists():
        try:
            for e in json.loads(catalog_path.read_text()):
                existing_meta[e["episode_hash"]] = e
        except Exception as exc:
            print(f"Warning: could not read existing catalog.json: {exc}")

    new_catalog = []
    for ep_hash in valid[:KEEP_CAP]:
        if ep_hash in existing_meta:
            new_catalog.append(existing_meta[ep_hash])
        else:
            # episode has no catalog entry (was never cataloged) — store minimal record
            new_catalog.append({
                "tar_filename": f"{ep_hash}.tar",
                "episode_hash": ep_hash,
                "n_frames": 0,
                "embodiment": "mecka_bimanual",
            })

    catalog_path.write_text(json.dumps(new_catalog, indent=2))
    total_frames = sum(e["n_frames"] for e in new_catalog)
    print(f"catalog.json rewritten: {len(new_catalog):,} episodes, {total_frames:,} frames")

    zip_volume.commit()
    print("Volume committed.")

    return {
        "keep": len(keep_set),
        "prune": deleted_pairs,
        "orphan_tars": deleted_orphan_tars,
        "orphan_dones": deleted_orphan_dones,
        "catalog_entries": len(new_catalog),
        "dry_run": False,
    }


@app.local_entrypoint()
def main(dry_run: bool = False) -> None:
    """Prune mecka_data_zip to at most 70 000 valid episodes.

    Args:
        dry_run: Print what would be deleted without removing anything.
    """
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"=== prune_zip_vol [{mode}] — cap={KEEP_CAP:,} episodes ===")
    result = prune.remote(dry_run=dry_run)
    print(f"\nResult: {result}")
