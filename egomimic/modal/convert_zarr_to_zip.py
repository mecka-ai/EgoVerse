"""Convert mecka_data_v2 zarr episode directories to single-file ZipStore format.

WHY
---
The benchmark (volume_benchmark.py) showed 161ms median zarr-open latency on the
mecka_data_v2 volume. This is caused by the directory-per-episode structure: opening
one episode requires ~10-50 separate network metadata reads (.zgroup, .zattrs,
one .zarray per array key, plus the directory listing itself).

ZipStore packages the entire zarr into a single file. On open, one sequential read
loads the zip central directory (all entry offsets) into memory. Subsequent metadata
reads (.zgroup, .zattrs, .zarray) become in-memory lookups instead of network
round-trips, cutting open latency by ~5-10x.

The zarr chunk data (images, poses, actions) is stored uncompressed inside the zip
so zarr's own Blosc/LZ4 chunk compression is preserved without double-compression.

USAGE
-----
# Dry run
modal run --env robotics egomimic/modal/convert_zarr_to_zip.py -- --dry-run

# Convert everything (~198k episodes, runs in parallel)
modal run --env robotics egomimic/modal/convert_zarr_to_zip.py

# Test with 5 batches first
modal run --env robotics egomimic/modal/convert_zarr_to_zip.py -- --max-batches 5

OUTPUT
------
New volume: mecka_data_zip  (mounted at /mnt/zarr-zip inside containers)
Layout mirrors mecka_data_v2 but every episode_hash.zarr/ directory becomes
episode_hash.zarr.zip (single file, ZipStore format).

TRAINING
--------
Pass  +modal_volume=mecka_data_zip  when launching trainModal.py.
The ZarrEpisode class auto-detects .zarr.zip paths and opens them via ZipStore.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import modal
from modal_setup import app, image, zarr_volume, CFG

# ---------------------------------------------------------------------------
# Output volume
# ---------------------------------------------------------------------------

zip_volume = modal.Volume.from_name("mecka_data_zip", create_if_missing=True)
ZIP_MOUNT = "/mnt/zarr-zip"
SRC_MOUNT = CFG.volume_mount_path  # /mnt/zarr-data

EPISODES_PER_BATCH = 500        # episodes per Modal function call
_EPHEMERAL_DISK_MB = 200_000    # 200 GB local NVMe — buffer for one batch

# ---------------------------------------------------------------------------
# Remote: convert one batch of episodes
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={
        SRC_MOUNT: zarr_volume,
        ZIP_MOUNT: zip_volume,
    },
    cpu=4,
    memory=8192,
    ephemeral_disk=_EPHEMERAL_DISK_MB,
    timeout=3600,
)
def convert_batch(episode_dirs: list[str]) -> dict:
    """Convert a list of zarr directory episodes to .zarr.zip (ZipStore) format.

    Each source episode is read from the input volume, converted to a single-file
    ZipStore written to /tmp first, then copied to the output volume.
    Existing outputs are skipped so the job is safe to re-run after failures.
    """
    import shutil
    import time
    import zipfile

    import zarr
    import zarr.storage

    n_ok = n_skip = n_err = 0
    t_batch = time.perf_counter()

    for ep_dir_str in episode_dirs:
        ep_path = Path(ep_dir_str)
        ep_name = ep_path.name  # e.g. "abc123.zarr" or "abc123"
        base = ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
        zip_name = f"{base}.zarr.zip"

        out_path = Path(ZIP_MOUNT) / zip_name
        if out_path.exists():
            n_skip += 1
            continue

        tmp_path = Path("/tmp") / zip_name
        try:
            t0 = time.perf_counter()

            # Open source zarr (directory store on network volume)
            src_store = zarr.storage.DirectoryStore(str(ep_path))

            # Write to ZipStore on local /tmp — ZIP_STORED keeps zarr's own
            # chunk compression intact without adding a second compression layer
            with zarr.ZipStore(str(tmp_path), mode="w", compression=zipfile.ZIP_STORED) as dst_store:
                zarr.copy_store(src_store, dst_store, if_exists="replace")

            elapsed = time.perf_counter() - t0
            size_mb = tmp_path.stat().st_size / 1e6

            # Copy from /tmp to output volume (single sequential write)
            shutil.copy2(tmp_path, out_path)
            tmp_path.unlink()

            n_ok += 1

        except Exception as e:
            print(f"ERROR {base}: {e}")
            if tmp_path.exists():
                tmp_path.unlink()
            n_err += 1

    zip_volume.commit()
    elapsed_batch = time.perf_counter() - t_batch
    print(f"Batch done: {n_ok} converted, {n_skip} skipped, {n_err} errors — {elapsed_batch:.0f}s")
    return {"n_ok": n_ok, "n_skip": n_skip, "n_err": n_err}


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(dry_run: bool = False, max_batches: int = 0) -> None:
    src_root = Path(SRC_MOUNT)
    all_episodes = sorted(p for p in src_root.iterdir() if p.is_dir())
    n_episodes = len(all_episodes)

    batches = [
        [str(p) for p in all_episodes[i : i + EPISODES_PER_BATCH]]
        for i in range(0, n_episodes, EPISODES_PER_BATCH)
    ]
    if max_batches > 0:
        batches = batches[:max_batches]

    print(f"Episodes:        {n_episodes:,}")
    print(f"Episodes/batch:  {EPISODES_PER_BATCH}")
    print(f"Batches:         {len(batches)}")
    print(f"Output volume:   mecka_data_zip")

    if dry_run:
        print("\nDry run — not launching.")
        return

    print(f"\nLaunching {len(batches)} parallel Modal functions...")
    results = list(convert_batch.map(batches, return_exceptions=True))

    ok_results = [r for r in results if isinstance(r, dict)]
    errors = [r for r in results if isinstance(r, Exception)]
    total_ok = sum(r["n_ok"] for r in ok_results)
    total_skip = sum(r["n_skip"] for r in ok_results)
    total_err = sum(r["n_err"] for r in ok_results)

    print(f"\nConversion complete:")
    print(f"  Converted: {total_ok:,}")
    print(f"  Skipped:   {total_skip:,}  (already existed)")
    print(f"  Errors:    {total_err + len(errors):,}")
