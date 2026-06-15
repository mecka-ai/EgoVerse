"""Convert mecka_data_v2 zarr episodes into tar shards.

Usage:
    modal run --env robotics egomimic/modal/shard_zarr_to_tar.py -- --dry-run
    modal run --detach --env robotics egomimic/modal/shard_zarr_to_tar.py
    modal run --env robotics egomimic/modal/shard_zarr_to_tar.py -- --max-episodes 100 --episodes-per-shard 10 --max-containers 500
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

WDS_VOLUME_NAME = "mecka_data_wds_v2"
WDS_MOUNT = "/mnt/zarr-wds"
INPUT_MOUNT = CFG.volume_mount_path

_IO_BUFFER = 64 * 1024 * 1024
_DEFAULT_MAX_CONTAINERS = 3500
EPISODES_PER_SHARD = 200

wds_volume = modal.Volume.from_name(
    WDS_VOLUME_NAME,
    create_if_missing=True,
)


def _shard_name(episode_dirs: list[str]) -> str:
    ep_names = sorted(Path(d).name for d in episode_dirs)
    digest = hashlib.sha256("\n".join(ep_names).encode()).hexdigest()[:16]
    return f"shard-{digest}.tar"


def _ep_hash(ep_path: str) -> str:
    name = Path(ep_path).name
    return name[:-5] if name.endswith(".zarr") else name


@app.function(
    image=image,
    volumes={WDS_MOUNT: wds_volume},
    cpu=1,
    memory=2048,
    timeout=120,
)
def get_sharded_episode_hashes() -> list[str]:
    index_path = Path(WDS_MOUNT) / "shard_index.json"
    if not index_path.exists():
        return []
    index = json.loads(index_path.read_text())
    print(f"shard_index.json: {len(index):,} episodes already sharded")
    return list(index.keys())


@app.function(
    image=image,
    volumes={INPUT_MOUNT: zarr_volume},
    cpu=1,
    memory=4096,
    timeout=120,
)
def list_episodes() -> list[str]:
    input_root = Path(INPUT_MOUNT)
    return sorted(str(p) for p in input_root.iterdir() if p.is_dir())


@app.function(
    image=image,
    volumes={
        INPUT_MOUNT: zarr_volume,
        WDS_MOUNT: wds_volume,
    },
    cpu=2,
    memory=8192,
    timeout=3600,
    max_containers=_DEFAULT_MAX_CONTAINERS,
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
    done_path = out_dir / (shard_name + ".done")

    # Already fully completed.
    if done_path.exists():
        return {"shard_name": shard_name, "n_episodes": 0, "skipped": True}

    # out_path exists but no .done → previous upload was interrupted.
    # Delete the partial tar and retry cleanly.
    if out_path.exists():
        out_path.unlink()
        wds_volume.commit()

    # Phase 1: build entire tar in /tmp — volume is untouched.
    n_ok = n_err = 0
    episodes_in_shard: list[str] = []
    t0 = time.perf_counter()

    try:
        with tarfile.open(tmp_path, "w", bufsize=_IO_BUFFER) as tar:
            for ep_dir in episode_dirs:
                ep_path = Path(ep_dir)
                if not ep_path.is_dir():
                    print(f"[{shard_name}] MISSING dir: {ep_path.name}")
                    n_err += 1
                    continue
                try:
                    tar.add(str(ep_path), arcname=ep_path.name, recursive=True)
                    episodes_in_shard.append(ep_path.name)
                    n_ok += 1
                except Exception as e:
                    print(f"[{shard_name}] ERROR adding {ep_path.name}: {e}")
                    n_err += 1
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    if n_ok == 0:
        tmp_path.unlink(missing_ok=True)
        return {"shard_name": shard_name, "n_episodes": 0, "n_errors": n_err, "skipped": True}

    build_elapsed = time.perf_counter() - t0
    size_mb = tmp_path.stat().st_size / 1e6

    # Phase 2: upload completed tar, write .done marker, commit.
    t1 = time.perf_counter()
    shutil.copy2(tmp_path, out_path)
    tmp_path.unlink(missing_ok=True)
    done_path.write_text("")
    wds_volume.commit()
    upload_elapsed = time.perf_counter() - t1

    bw = size_mb / upload_elapsed if upload_elapsed > 0 else 0
    print(
        f"[{shard_name}] {n_ok} eps  {size_mb:.0f} MB  "
        f"build={build_elapsed:.0f}s  upload={upload_elapsed:.0f}s ({bw:.0f} MB/s)"
    )
    return {
        "shard_name": shard_name,
        "n_episodes": n_ok,
        "n_errors": n_err,
        "size_mb": size_mb,
        "elapsed_s": time.perf_counter() - t0,
        "episodes": episodes_in_shard,
    }



@app.function(
    image=image,
    volumes={WDS_MOUNT: wds_volume},
    cpu=1,
    memory=2048,
    timeout=300,
)
def cleanup_partials() -> dict:
    """Delete orphaned tars that have no .done marker (interrupted uploads).

    Also cleans up any legacy .building.* files from the old code path.
    """
    root = Path(WDS_MOUNT)
    deleted: list[str] = []

    # New scheme: tar without a matching .done marker = interrupted upload.
    for tar in sorted(root.glob("shard-*.tar")):
        if not (root / f"{tar.name}.done").exists():
            print(f"Incomplete upload (no .done): {tar.name}  ({tar.stat().st_size / 1e6:.0f} MB) — deleting")
            tar.unlink()
            deleted.append(tar.name)

    # Legacy: .building.* from the old streaming-to-volume approach.
    for p in sorted(root.glob(".building.*.tar")):
        print(f"Legacy orphan: {p.name} — deleting")
        p.unlink()
        deleted.append(p.name)

    if deleted:
        wds_volume.commit()

    print(f"Deleted {len(deleted)} incomplete/orphaned files. Volume is clean.")
    return {"deleted": deleted}


@app.function(
    image=image,
    volumes={WDS_MOUNT: wds_volume},
    cpu=1,
    memory=2048,
    timeout=300,
)
def clear_volume() -> None:
    wds_root = Path(WDS_MOUNT)
    removed = 0
    for f in wds_root.iterdir():
        if (f.name.endswith(".tar") or f.name.endswith(".tar.done")
                or f.name == "shard_index.json" or f.name.startswith(".building.")):
            f.unlink()
            removed += 1
    wds_volume.commit()
    print(f"Cleared {removed} files from {WDS_VOLUME_NAME}")


@app.function(
    image=image,
    volumes={WDS_MOUNT: wds_volume},
    cpu=1,
    memory=4096,
    timeout=300,
)
def write_shard_index(results: list[dict], override: bool = False) -> None:
    out = Path(WDS_MOUNT) / "shard_index.json"
    index: dict = {}
    if not override and out.exists():
        try:
            index = json.loads(out.read_text())
        except Exception:
            pass

    prev_count = len(index)
    for r in results:
        if r.get("skipped"):
            continue
        for ep_name in r.get("episodes", []):
            ep_hash = ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
            index[ep_hash] = r["shard_name"]

    out.write_text(json.dumps(index, indent=2))
    wds_volume.commit()
    print(f"shard_index.json: {len(index) - prev_count} new + {prev_count} existing = {len(index)} total")


@app.function(
    image=image,
    volumes={
        INPUT_MOUNT: zarr_volume,
        WDS_MOUNT: wds_volume,
    },
    cpu=2,
    memory=8192,
    timeout=600,
)
def rebuild_shard_index(max_episodes: int = 70_000, episodes_per_shard: int = 20) -> None:
    """Reconstruct shard_index.json from the completed shards on the volume.

    Uses the same deterministic episode ordering and batching as convert_shard,
    so the shard names are computed without reading tar content.
    Only shards with a .done marker are included.
    """
    all_episodes = sorted(str(p) for p in Path(INPUT_MOUNT).iterdir() if p.is_dir())
    target = all_episodes[:max_episodes]
    batches = [target[i: i + episodes_per_shard] for i in range(0, len(target), episodes_per_shard)]

    index: dict[str, str] = {}
    missing: list[str] = []
    for batch in batches:
        shard_name = _shard_name(batch)
        done_path = Path(WDS_MOUNT) / f"{shard_name}.done"
        if done_path.exists():
            for ep_dir in batch:
                ep_name = Path(ep_dir).name
                ep_hash = ep_name[:-5] if ep_name.endswith(".zarr") else ep_name
                index[ep_hash] = shard_name
        else:
            missing.append(shard_name)

    out = Path(WDS_MOUNT) / "shard_index.json"
    out.write_text(json.dumps(index, indent=2))
    wds_volume.commit()
    print(f"shard_index.json written: {len(index):,} episodes across {len(index) // episodes_per_shard} shards")
    if missing:
        print(f"WARNING: {len(missing)} shards missing .done marker (not included in index):")
        for s in missing[:10]:
            print(f"  {s}")

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

        if not index:
            print(f"  [{_task_shard_dir(task_name)}] no successful shards — skipping index write")
            continue
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
    results = list(convert_shard.map(episode_batches, output_subdirs, return_exceptions=True, wrap_returned_exceptions=False))

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


@app.function(
    image=image,
    volumes={WDS_MOUNT: wds_volume},
    cpu=1,
    memory=512,
    timeout=120,
)
def _delete_task_shard_dir(task_name: str) -> str:
    """Remove the per-task shard directory from the WDS volume."""
    import shutil
    task_dir = Path(WDS_MOUNT) / _task_shard_dir(task_name)
    if task_dir.exists():
        shutil.rmtree(task_dir)
        wds_volume.commit()
        return f"deleted {task_dir}"
    return f"not found: {task_dir}"


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def delete_task_shards(task: str = "") -> None:
    """Delete a task's per-task shard directory from the WDS volume.

    Usage:
        modal run --env robotics egomimic/modal/shard_zarr_to_tar.py::delete_task_shards -- --task dishwashing
    """
    if not task:
        raise ValueError("Pass --task <task_name>")
    result = _delete_task_shard_dir.remote(task)
    print(result)


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
def main(
    dry_run: bool = False,
    max_episodes: int = 0,
    episodes_per_shard: int = 20,
    max_containers: int = _DEFAULT_MAX_CONTAINERS,
    override: bool = False,
    cleanup: bool = False,
    rebuild_index: bool = False,
) -> None:
    if cleanup:
        cleanup_partials.remote()
        return

    if rebuild_index:
        n = max_episodes if max_episodes > 0 else 70_000
        print(f"Rebuilding shard index for first {n:,} episodes...")
        rebuild_shard_index.remote(max_episodes=n, episodes_per_shard=episodes_per_shard)
        return

    if override:
        print(f"override=True: clearing all existing data from {WDS_VOLUME_NAME}...")
        if not dry_run:
            clear_volume.remote()
        already_sharded: set[str] = set()
    else:
        already_sharded = set(get_sharded_episode_hashes.remote())
    print(f"Already sharded: {len(already_sharded):,} episodes")

    all_episodes = list_episodes.remote()  # sorted, deterministic — never shuffled
    print(f"Total available: {len(all_episodes):,} episodes")

    to_shard = [ep for ep in all_episodes if _ep_hash(ep) not in already_sharded]
    if max_episodes > 0:
        to_shard = to_shard[:max_episodes]
    print(f"To shard:        {len(to_shard):,} episodes")

    batches = [
        to_shard[i : i + episodes_per_shard]
        for i in range(0, len(to_shard), episodes_per_shard)
    ]

    print(f"\nPlan: {len(batches):,} shards x {episodes_per_shard} episodes -> {WDS_VOLUME_NAME}")
    print(f"      max_containers={max_containers}")

    if dry_run:
        print("Dry run — not launching any jobs.")
        return

    print(f"Launching {len(batches):,} parallel Modal functions...")
    results = list(convert_shard.map(batches, return_exceptions=True))

    ok      = [r for r in results if isinstance(r, dict) and not r.get("skipped")]
    skipped = [r for r in results if isinstance(r, dict) and r.get("skipped")]
    errs    = [r for r in results if isinstance(r, Exception)]

    print(f"\nDone: {len(ok)} written, {len(skipped)} skipped, {len(errs)} errors")
    print(f"      {sum(r['n_episodes'] for r in ok):,} episodes, {sum(r.get('size_mb', 0) for r in ok) / 1000:.1f} GB")
    if errs:
        for e in errs[:5]:
            print(f"  ERROR: {e}")

    # Use spawn() so these survive after the local process disconnects in detach mode.
    write_shard_index.remote(ok, override=override)
    print("Index updated.")
    cleanup_partials.remote()