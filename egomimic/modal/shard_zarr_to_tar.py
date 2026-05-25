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

import hashlib
import json
import math
import os
import sys
import tarfile
import time
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import modal
from modal_setup import app, zarr_volume, CFG, image, _prepare_repo_light

# ---------------------------------------------------------------------------
# Output volume
# ---------------------------------------------------------------------------

wds_volume = modal.Volume.from_name("mecka_data_wds_v2", create_if_missing=True)
WDS_MOUNT = "/mnt/zarr-wds"
INPUT_MOUNT = CFG.volume_mount_path  # /mnt/zarr-data

EPISODES_PER_SHARD = 20

# Each shard = 20 episodes × ~167 MB = ~3.3 GB written to /tmp then copied to volume.
# No special ephemeral disk needed; default Modal filesystem handles ~4 GB comfortably.


def _shard_name(episode_dirs: list[str]) -> str:
    """Compute a content-addressed shard filename from the episode hashes it contains.

    Hashing the sorted episode names means the same set of episodes always produces
    the same filename, making re-runs fully idempotent.
    """
    ep_names = sorted(Path(d).name for d in episode_dirs)
    digest = hashlib.sha256("\n".join(ep_names).encode()).hexdigest()[:16]
    return f"shard-{digest}.tar"


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
def convert_shard(episode_dirs: list[str], output_subdir: str = "") -> dict:
    """Bundle a list of zarr episode directories into a single content-addressed tar shard.

    The shard filename is a SHA-256 hash of the sorted episode names, so the same
    set of episodes always produces the same filename. Re-runs skip existing shards.
    The zarr directory tree is added verbatim — existing chunk compression is preserved.

    output_subdir: relative path under WDS_MOUNT to write the shard (default: root).
    """
    import shutil
    import time

    shard_name = _shard_name(episode_dirs)
    if output_subdir:
        out_dir = Path(WDS_MOUNT) / output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(WDS_MOUNT)
    tmp_path = Path("/tmp") / shard_name
    out_path = out_dir / shard_name

    if out_path.exists():
        print(f"[{shard_name}] already exists, skipping")
        return {"shard_name": shard_name, "n_episodes": 0, "skipped": True}

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
                print(f"[{shard_name}] ERROR on {ep_path.name}: {e}")
                n_err += 1

    elapsed = time.perf_counter() - t0
    size_mb = tmp_path.stat().st_size / 1e6

    # Copy shard from /tmp to volume (single sequential write)
    shutil.copy2(tmp_path, out_path)
    tmp_path.unlink()
    wds_volume.commit()

    print(f"[{shard_name}] {n_ok} episodes, {size_mb:.0f} MB, {elapsed:.0f}s — copied to volume")
    return {
        "shard_name": shard_name,
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
    """Write shard_index.json mapping episode_hash -> shard_name to the volume."""
    index = {}
    for r in results:
        if r.get("skipped"):
            continue
        for ep_name in r.get("episodes", []):
            ep_hash = ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
            index[ep_hash] = r["shard_name"]

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
# Remote: list episodes for specific tasks (SQL + zarr volume check)
# ---------------------------------------------------------------------------


def _task_shard_dir(task_name: str) -> str:
    """Deterministic per-task subfolder: tasks/{task_name}_{sha6}."""
    task_hash = hashlib.sha256(task_name.encode()).hexdigest()[:6]
    return f"tasks/{task_name}_{task_hash}"


@app.function(
    image=image,
    volumes={INPUT_MOUNT: zarr_volume},
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    cpu=2,
    memory=8192,
    timeout=300,
)
def _list_task_episodes_remote(
    task_names: list[str],
    git_remote: str,
    git_commit: str,
) -> dict[str, list[str]]:
    """SQL lookup + zarr volume existence check; returns {task_name: [ep_dir_path]}."""
    import sys as _sys
    _prepare_repo_light(git_remote=git_remote, git_commit=git_commit)
    _sys.path.insert(0, CFG.remote_repo_dir)

    from egomimic.utils.aws.aws_data_utils import load_env
    from egomimic.utils.aws.aws_sql import episode_table_to_df, create_default_engine
    load_env()

    engine = create_default_engine()
    full_df = episode_table_to_df(engine)
    if "is_deleted" in full_df.columns:
        full_df = full_df[full_df["is_deleted"] != True]  # noqa: E712

    task_set = set(task_names)
    if "task" not in full_df.columns:
        raise RuntimeError("SQL table has no 'task' column — check DB schema")

    task_df = full_df[full_df["task"].isin(task_set)]
    input_root = Path(INPUT_MOUNT)
    by_task: dict[str, list[str]] = {}

    for _, row in task_df.iterrows():
        ep_hash = str(row["episode_hash"])
        task = str(row["task"])
        for candidate in [input_root / ep_hash, input_root / f"{ep_hash}.zarr"]:
            if candidate.is_dir():
                by_task.setdefault(task, []).append(str(candidate))
                break

    for task, dirs in sorted(by_task.items()):
        print(f"  [{task}] {len(dirs)} episodes on zarr volume")
    return by_task


# ---------------------------------------------------------------------------
# Remote: write per-task shard indexes and metadata
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={WDS_MOUNT: wds_volume},
    cpu=1,
    memory=2048,
    timeout=120,
)
def _write_task_indexes_remote(task_results: dict[str, list[dict]]) -> None:
    """Write shard_index.json + metadata.json into each per-task shard dir."""
    import time as _t

    for task_name, results in task_results.items():
        task_dir = Path(WDS_MOUNT) / _task_shard_dir(task_name)
        task_dir.mkdir(parents=True, exist_ok=True)

        index: dict[str, str] = {}
        for r in results:
            if r.get("skipped"):
                continue
            for ep_name in r.get("episodes", []):
                ep_hash = ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
                index[ep_hash] = r["shard_name"]

        (task_dir / "shard_index.json").write_text(json.dumps(index, indent=2))
        meta = {
            "task": task_name,
            "n_episodes": len(index),
            "n_shards": len([r for r in results if not r.get("skipped")]),
            "created_at": _t.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (task_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(f"  [{_task_shard_dir(task_name)}] {len(index)} episodes")

    wds_volume.commit()


# ---------------------------------------------------------------------------
# Local entrypoint: task-partitioned shard conversion
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def shard_by_task(tasks: str = "") -> None:
    """Convert zarr episodes for specific tasks into per-task tar shards.

    Each task gets a deterministic subfolder in the WDS volume:
        tasks/{task_name}_{sha6}/shard-{content_hash}.tar
        tasks/{task_name}_{sha6}/shard_index.json
        tasks/{task_name}_{sha6}/metadata.json

    Usage:
        modal run --env robotics egomimic/modal/shard_zarr_to_tar.py::shard_by_task \\
            -- --tasks cutting_plastic,organizing_containers
    """
    from modal_setup import _resolve_git_state

    task_names = [t.strip() for t in tasks.split(",") if t.strip()]
    if not task_names:
        raise SystemExit("Specify tasks: --tasks task1,task2,...")

    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: uncommitted local changes — Modal uses the last pushed commit.")

    print(f"Querying zarr volume for tasks: {task_names}")
    by_task: dict[str, list[str]] = _list_task_episodes_remote.remote(
        task_names, git_remote, git_commit
    )
    if not by_task:
        raise SystemExit("No episodes found on zarr volume for the specified tasks.")

    # Build (episode_dirs, output_subdir) pairs for parallel convert_shard
    episode_batches: list[list[str]] = []
    output_subdirs: list[str] = []
    batch_task_labels: list[str] = []

    for task_name, ep_dirs in sorted(by_task.items()):
        subdir = _task_shard_dir(task_name)
        n_shards = math.ceil(len(ep_dirs) / EPISODES_PER_SHARD)
        print(f"  {task_name}: {len(ep_dirs)} episodes → {n_shards} shards → {subdir}/")
        for i in range(0, len(ep_dirs), EPISODES_PER_SHARD):
            batch = ep_dirs[i : i + EPISODES_PER_SHARD]
            episode_batches.append(batch)
            output_subdirs.append(subdir)
            batch_task_labels.append(task_name)

    print(f"\nLaunching {len(episode_batches)} parallel shard conversions...")
    results = list(convert_shard.map(episode_batches, output_subdirs, return_exceptions=True))

    # Group results by task and write per-task indexes
    task_results: dict[str, list[dict]] = {t: [] for t in by_task}
    n_errors = 0
    for i, r in enumerate(results):
        if isinstance(r, dict):
            task_results[batch_task_labels[i]].append(r)
        else:
            n_errors += 1
            print(f"  Shard error ({batch_task_labels[i]}): {r}")

    print(f"\nWriting per-task shard indexes ({n_errors} shard error(s))...")
    _write_task_indexes_remote.remote(task_results)

    print("\nTask shard conversion complete:")
    for task_name in sorted(by_task.keys()):
        ok = [r for r in task_results[task_name] if isinstance(r, dict) and not r.get("skipped")]
        print(f"  {_task_shard_dir(task_name)}: {sum(r.get('n_episodes', 0) for r in ok)} episodes")


# ---------------------------------------------------------------------------
# Remote: rebuild full shard index from all tars in the volume
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={WDS_MOUNT: wds_volume},
    cpu=2,
    memory=4096,
    timeout=600,
)
def _rebuild_shard_index_remote() -> int:
    """Read tar headers from every shard-*.tar in the volume; write shard_index.json."""
    wds = Path(WDS_MOUNT)
    shards = sorted(wds.glob("shard-*.tar"))
    print(f"Scanning {len(shards)} shards for tar headers ...")
    index = {}
    for shard_path in shards:
        shard_name = shard_path.name
        try:
            with tarfile.open(shard_path, "r") as tar:
                top_level = {m.name.split("/")[0] for m in tar.getmembers()}
        except Exception as exc:
            print(f"[{shard_name}] ERROR reading headers: {exc}")
            continue
        for ep_name in top_level:
            ep_hash = ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
            index[ep_hash] = shard_name

    out = wds / "shard_index.json"
    out.write_text(json.dumps(index, indent=2))
    wds_volume.commit()
    print(f"Wrote shard_index.json: {len(index)} episodes across {len(shards)} shards")
    return len(index)


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def rebuild_shard_index() -> None:
    """Rebuild shard_index.json from ALL tar shards currently in the WDS volume.

    The default write_shard_index only indexes newly-created shards. Run this
    once after adding shards out-of-band (e.g. from a previous conversion run)
    to ensure the index is complete before starting curation.
    """
    n = _rebuild_shard_index_remote.remote()
    print(f"Index rebuild complete — {n} episodes indexed.")


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
    results = list(convert_shard.map(batches, return_exceptions=True))

    ok = [r for r in results if isinstance(r, dict) and not r.get("skipped")]
    skipped = [r for r in results if isinstance(r, dict) and r.get("skipped")]
    errs = [r for r in results if isinstance(r, Exception)]

    total_episodes = sum(r["n_episodes"] for r in ok)
    total_mb = sum(r.get("size_mb", 0) for r in ok)

    print(f"\nConversion complete:")
    print(f"  Shards written:  {len(ok)}")
    print(f"  Shards skipped:  {len(skipped)} (already existed)")
    print(f"  Episodes:        {total_episodes:,}")
    print(f"  Total size:      {total_mb/1000:.1f} GB")
    if errs:
        print(f"  Errors:          {len(errs)} shards failed")

    # Write shard index
    print("\nWriting shard index...")
    write_shard_index.remote(ok)
    print("Done.")
