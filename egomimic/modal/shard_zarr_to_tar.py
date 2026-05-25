"""Convert mecka_data_v2 zarr episodes into tar shards for fast sequential training I/O.

WHY
---
The mecka_data_v2 volume has ~312ms random-read latency (measured by volume_benchmark.py).
Each zarr episode is a directory of many small files; random access across 198k directories
is extremely slow on network-attached storage.

This script bundles N episodes per tar shard (default: 20). At training time a worker reads
one shard sequentially — one network read — extracts it to /tmp (local NVMe), then accesses
the zarr stores locally at ~GB/s instead of ~1 MB/s.

USAGE
-----
# Dry run — show what would be converted
modal run --env robotics egomimic/modal/shard_zarr_to_tar.py -- --dry-run

# Convert all episodes (dispatches ~10k parallel Modal functions)
modal run --env robotics egomimic/modal/shard_zarr_to_tar.py

# Convert a limited number of shards (for testing)
modal run --env robotics egomimic/modal/shard_zarr_to_tar.py -- --max-shards 10

OUTPUT VOLUME
-------------
mecka_data_wds  mounted at /mnt/zarr-wds inside the container.
Layout:
    /mnt/zarr-wds/
        shard-000000.tar   (20 zarr episode dirs bundled as-is)
        shard-000001.tar
        ...
        shard_index.json   (maps episode_hash -> shard id)

TRAINING INTEGRATION
---------------------
At training time, workers:
  1. Pick a shard sequentially (shuffle shard list each epoch)
  2. Read the tar from the volume into /tmp  (~3 GB one sequential read, ~30s vs 6600s random)
  3. Extract to /tmp, open zarr stores locally (fast)
  4. Randomly sample (episode, frame_idx) pairs to build batches
  5. Delete /tmp shard when done
"""

from __future__ import annotations

import json
import os
import sys
import tarfile
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import modal
from modal_setup import app, zarr_volume, CFG, image

# ---------------------------------------------------------------------------
# Output volume
# ---------------------------------------------------------------------------

wds_volume = modal.Volume.from_name("mecka_data_wds_v2", create_if_missing=True)
WDS_MOUNT = "/mnt/zarr-wds"
INPUT_MOUNT = CFG.volume_mount_path  # /mnt/zarr-data

EPISODES_PER_SHARD = 20

# Each shard = 20 episodes × ~167 MB = ~3.3 GB written to /tmp then copied to volume.
# No special ephemeral disk needed; default Modal filesystem handles ~4 GB comfortably.


# ---------------------------------------------------------------------------
# Remote: convert one batch of episode dirs → one tar shard
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={
        INPUT_MOUNT: zarr_volume,
        WDS_MOUNT: wds_volume,
    },
    cpu=2,
    memory=8192,
    timeout=3600,
    max_containers=1000,
)
def convert_shard(episode_dirs: list[str], shard_id: int) -> dict:
    """Bundle a list of zarr episode directories into a single tar shard.

    The zarr directory tree is added verbatim — existing chunk compression is
    preserved and no data is re-encoded. The shard is written to /tmp first
    (local NVMe), then copied to the output volume.
    """
    import shutil
    import time

    shard_name = f"shard-{shard_id:06d}.tar"
    tmp_path = Path("/tmp") / shard_name
    out_path = Path(WDS_MOUNT) / shard_name

    if out_path.exists():
        print(f"[shard {shard_id:06d}] already exists, skipping")
        return {"shard_id": shard_id, "n_episodes": 0, "skipped": True}

    n_ok = 0
    n_err = 0
    episodes_in_shard = []
    t0 = time.perf_counter()

    with tarfile.open(tmp_path, "w") as tar:
        for ep_dir in episode_dirs:
            ep_path = Path(ep_dir)
            if not ep_path.is_dir():
                n_err += 1
                continue
            try:
                # Add entire zarr directory tree with arcname = just the dir name
                # so the tar contains  episode_hash.zarr/...  entries
                tar.add(str(ep_path), arcname=ep_path.name)
                episodes_in_shard.append(ep_path.name)
                n_ok += 1
            except Exception as e:
                print(f"[shard {shard_id:06d}] ERROR on {ep_path.name}: {e}")
                n_err += 1

    elapsed = time.perf_counter() - t0
    size_mb = tmp_path.stat().st_size / 1e6

    # Copy shard from /tmp to volume (single sequential write)
    shutil.copy2(tmp_path, out_path)
    tmp_path.unlink()
    wds_volume.commit()

    print(
        f"[shard {shard_id:06d}] {n_ok} episodes, "
        f"{size_mb:.0f} MB, {elapsed:.0f}s — copied to volume"
    )
    return {
        "shard_id": shard_id,
        "n_episodes": n_ok,
        "n_errors": n_err,
        "size_mb": size_mb,
        "elapsed_s": elapsed,
        "episodes": episodes_in_shard,
    }


# ---------------------------------------------------------------------------
# Remote: write the shard index
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={WDS_MOUNT: wds_volume},
    cpu=1,
    memory=4096,
    timeout=300,
)
def write_shard_index(results: list[dict]) -> None:
    """Write shard_index.json mapping episode_hash -> shard_id to the volume."""
    index = {}
    for r in results:
        if r.get("skipped"):
            continue
        for ep_name in r.get("episodes", []):
            ep_hash = ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
            index[ep_hash] = r["shard_id"]

    out = Path(WDS_MOUNT) / "shard_index.json"
    out.write_text(json.dumps(index, indent=2))
    wds_volume.commit()
    print(f"Wrote shard_index.json with {len(index)} episodes")


# ---------------------------------------------------------------------------
# Remote: list available episode directories
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={INPUT_MOUNT: zarr_volume},
    cpu=1,
    memory=4096,
    timeout=120,
)
def list_episodes() -> list[str]:
    """Return sorted list of episode directory paths from the input volume."""
    input_root = Path(INPUT_MOUNT)
    return sorted(str(p) for p in input_root.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(dry_run: bool = False, max_shards: int = 0) -> None:
    """Dispatch parallel shard conversion jobs.

    Args:
        dry_run:    Print plan without launching any jobs.
        max_shards: If >0, only convert this many shards (for testing).
    """
    print("Listing episodes from volume...")
    all_episodes = list_episodes.remote()
    n_episodes = len(all_episodes)

    # Split into batches of EPISODES_PER_SHARD
    batches: list[list[str]] = []
    for i in range(0, n_episodes, EPISODES_PER_SHARD):
        batch = all_episodes[i : i + EPISODES_PER_SHARD]
        batches.append(batch)

    if max_shards > 0:
        batches = batches[:max_shards]

    n_shards = len(batches)
    est_size_gb = n_shards * EPISODES_PER_SHARD * 0.167
    print(f"Episodes:      {n_episodes:,}  (using first {n_shards * EPISODES_PER_SHARD})")
    print(f"Shards:        {n_shards}  ({EPISODES_PER_SHARD} episodes each)")
    print(f"Est. output:   {est_size_gb:.0f} GB")
    print(f"Output volume: mecka_data_wds")

    if dry_run:
        print("\nDry run — not launching any jobs.")
        return

    print(f"\nLaunching {n_shards} parallel Modal functions...")
    shard_ids = list(range(n_shards))
    results = list(convert_shard.map(batches, shard_ids, return_exceptions=True))

    # Filter out exceptions for reporting
    ok = [r for r in results if isinstance(r, dict) and not r.get("skipped")]
    errs = [r for r in results if isinstance(r, Exception)]

    total_episodes = sum(r["n_episodes"] for r in ok)
    total_mb = sum(r.get("size_mb", 0) for r in ok)

    print(f"\nConversion complete:")
    print(f"  Shards written: {len(ok)}")
    print(f"  Episodes:       {total_episodes:,}")
    print(f"  Total size:     {total_mb/1000:.1f} GB")
    if errs:
        print(f"  Errors:         {len(errs)} shards failed")

    # Write shard index
    print("\nWriting shard index...")
    write_shard_index.remote(ok)
    print("Done.")
