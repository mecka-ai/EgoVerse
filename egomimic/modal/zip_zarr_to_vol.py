"""Convert mecka_data_v2 zarr v3 episodes → per-episode uncompressed tar files on mecka_data_zip.

Each episode becomes one ``episode_hash.tar`` whose top-level entries are the raw zarr v3
files/directories (no wrapping directory), so ``tarfile.extractall(path=dest)`` yields a
ready-to-open zarr v3 store at ``dest``.

A ``episode_hash.done`` sentinel is written to the zip volume alongside every successfully
committed tar.  At the end of a run, ``scan_and_fix`` finds any tars whose ``.done`` is
missing (container was preempted mid-write), deletes them, and re-queues conversion.

A ``catalog.json`` is written to the zip volume root after all conversions complete.

Usage
-----
# 1 000-episode debug run (non-destructive, idempotent):
MODAL_ENVIRONMENT=robotics modal run egomimic/modal/zip_zarr_to_vol.py -- --debug 1000

# Full 70K conversion:
MODAL_ENVIRONMENT=robotics modal run egomimic/modal/zip_zarr_to_vol.py

# Dry run — list plan without converting:
MODAL_ENVIRONMENT=robotics modal run egomimic/modal/zip_zarr_to_vol.py -- --debug 1000 --dry-run

# Force-reconvert everything (overwrites existing tars):
MODAL_ENVIRONMENT=robotics modal run egomimic/modal/zip_zarr_to_vol.py -- --force
"""

from __future__ import annotations

import json
import os
import sys
import tarfile
import time
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import modal
from modal_setup import app, zarr_volume, zip_volume, image

INPUT_MOUNT = "/mnt/zarr-data"
ZIP_MOUNT = "/mnt/zarr-zip"


# ---------------------------------------------------------------------------
# Remote: list episode directories
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={INPUT_MOUNT: zarr_volume},
    cpu=2,
    memory=8192,
    timeout=300,
)
def list_episodes() -> list[str]:
    """Return sorted list of zarr episode directory paths from mecka_data_v2."""
    input_root = Path(INPUT_MOUNT)
    return sorted(str(p) for p in input_root.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Remote: convert one episode → one tar + .done sentinel
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={
        INPUT_MOUNT: zarr_volume,
        ZIP_MOUNT: zip_volume,
    },
    cpu=2,
    memory=8192,
    timeout=600,
    max_containers=3000,
)
def convert_episode(ep_dir: str, force: bool = False) -> dict | None:
    """Create a per-episode uncompressed tar on the zip volume.

    The tar's top-level entries are the raw zarr v3 store files/directories (no
    wrapping directory), so ``tarfile.extractall(path=dest)`` yields a
    ready-to-open zarr v3 store at ``dest``.

    A ``episode_hash.done`` sentinel is written after the tar is committed.
    The skip check requires BOTH the tar AND the sentinel to exist — if only
    the tar exists the episode was partially written (container preempted) and
    will be reconverted.

    Args:
        ep_dir: path to the zarr episode directory on the input volume.
        force:  if True, overwrite any existing tar (fixes corrupted tars).

    Returns a catalog dict on success, or None on failure.
    """
    import shutil
    import zarr

    ep_path = Path(ep_dir)
    ep_name = ep_path.name
    episode_hash = ep_name[:-5] if ep_name.endswith(".zarr") else ep_name

    tar_filename = f"{episode_hash}.tar"
    done_filename = f"{episode_hash}.done"
    out_tar = Path(ZIP_MOUNT) / tar_filename
    out_done = Path(ZIP_MOUNT) / done_filename
    tmp_tar = Path("/tmp") / tar_filename

    # Read zarr metadata (needed even for skip path to return accurate catalog entry)
    try:
        store = zarr.open_group(str(ep_path), mode="r")
        attrs = dict(store.attrs)
        n_frames = int(attrs["total_frames"])
        embodiment = str(attrs.get("embodiment", "mecka_bimanual"))
    except Exception as exc:
        print(f"[{episode_hash}] ERROR opening zarr: {exc}")
        return None

    # Skip only when both tar AND .done exist (both missing or only tar = redo)
    if out_tar.exists() and out_done.exists() and not force:
        print(f"[{episode_hash}] already exists — skipping")
        return {
            "tar_filename": tar_filename,
            "episode_hash": episode_hash,
            "n_frames": n_frames,
            "embodiment": embodiment,
            "skipped": True,
        }

    # Remove any stale/partial tar and done before writing
    out_tar.unlink(missing_ok=True)
    out_done.unlink(missing_ok=True)
    tmp_tar.unlink(missing_ok=True)

    # Create uncompressed tar with recursive=False so each item in rglob("*") is
    # added exactly once.  Without this, tf.add() on a directory recursively adds
    # its entire subtree, and rglob then re-adds every file N times (once per
    # ancestor directory), bloating /tmp until the write truncates.
    t0 = time.perf_counter()
    try:
        with tarfile.open(tmp_tar, "w") as tf:
            for item in sorted(ep_path.rglob("*")):
                tf.add(
                    str(item),
                    arcname=str(item.relative_to(ep_path)),
                    recursive=False,
                )
    except Exception as exc:
        print(f"[{episode_hash}] ERROR creating tar: {exc}")
        tmp_tar.unlink(missing_ok=True)
        return None

    size_mb = tmp_tar.stat().st_size / 1e6
    elapsed_tar = time.perf_counter() - t0

    try:
        shutil.move(str(tmp_tar), str(out_tar))
        # Write .done sentinel — this is the commit gate.  If the container is
        # preempted between shutil.move and here, the tar exists without a .done
        # and scan_and_fix will delete and redo it.
        out_done.touch()
        zip_volume.commit()
    except Exception as exc:
        print(f"[{episode_hash}] ERROR writing to volume: {exc}")
        tmp_tar.unlink(missing_ok=True)
        return None

    elapsed = time.perf_counter() - t0
    print(
        f"[{episode_hash}] {n_frames} frames, {size_mb:.0f} MB, "
        f"tar={elapsed_tar:.1f}s total={elapsed:.1f}s"
    )
    return {
        "tar_filename": tar_filename,
        "episode_hash": episode_hash,
        "n_frames": n_frames,
        "embodiment": embodiment,
        "skipped": False,
    }


# ---------------------------------------------------------------------------
# Remote: scan zip volume and delete tars without .done sentinels
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={ZIP_MOUNT: zip_volume},
    cpu=2,
    memory=8192,
    timeout=300,
)
def scan_and_fix() -> list[str]:
    """Find tars without a matching .done sentinel, delete them, return their hashes.

    These correspond to episodes where the container was preempted after the
    tar was moved to the volume but before (or during) commit.  Deleting them
    ensures the next conversion pass starts from a clean state.
    """
    zip_path = Path(ZIP_MOUNT)
    tars  = {p.stem for p in zip_path.glob("*.tar")}
    dones = {p.stem for p in zip_path.glob("*.done")}

    incomplete = tars - dones
    if not incomplete:
        print("scan_and_fix: all tars have .done sentinels — nothing to fix")
        zip_volume.commit()
        return []

    print(f"scan_and_fix: {len(incomplete)} incomplete tar(s) — deleting")
    for ep_hash in sorted(incomplete):
        tar = zip_path / f"{ep_hash}.tar"
        tar.unlink(missing_ok=True)
        print(f"  deleted {ep_hash}.tar")

    zip_volume.commit()
    return sorted(incomplete)


# ---------------------------------------------------------------------------
# Remote: write catalog.json (merge-idempotent)
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={ZIP_MOUNT: zip_volume},
    cpu=1,
    memory=4096,
    timeout=120,
)
def write_catalog(entries: list[dict]) -> None:
    """Write catalog.json to the zip volume root, merging with any existing entries."""
    catalog_path = Path(ZIP_MOUNT) / "catalog.json"

    existing: dict[str, dict] = {}
    if catalog_path.exists():
        try:
            for e in json.loads(catalog_path.read_text()):
                existing[e["episode_hash"]] = e
        except Exception:
            pass

    for e in entries:
        existing[e["episode_hash"]] = {
            "tar_filename": e["tar_filename"],
            "episode_hash": e["episode_hash"],
            "n_frames": e["n_frames"],
            "embodiment": e.get("embodiment", "mecka_bimanual"),
        }

    catalog = sorted(existing.values(), key=lambda x: x["episode_hash"])
    catalog_path.write_text(json.dumps(catalog, indent=2))
    zip_volume.commit()
    print(f"catalog.json: {len(catalog)} episodes, {sum(e['n_frames'] for e in catalog):,} frames")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(debug: int = 0, dry_run: bool = False, force: bool = False) -> None:
    """Dispatch parallel per-episode conversion jobs to mecka_data_zip.

    After the main conversion pass, scans for any tars missing their .done
    sentinel (container preempted mid-commit) and reconverts those episodes.
    Repeats until the volume is fully clean, then writes catalog.json.

    Args:
        debug:   If > 0, limit conversion to this many episodes (for smoke-testing).
        dry_run: Print plan without launching any jobs.
        force:   If True, overwrite existing tars (use to fix corrupted tars).
    """
    print("Listing episodes from mecka_data_v2 (/mnt/zarr-data)...")
    all_episodes = list_episodes.remote()
    n_total = len(all_episodes)
    print(f"Found {n_total:,} episodes")

    # Build hash → zarr_dir map for fast lookup during fix passes
    def _ep_hash(ep_dir: str) -> str:
        name = Path(ep_dir).name
        return name[:-5] if name.endswith(".zarr") else name

    hash_to_dir: dict[str, str] = {_ep_hash(ep): ep for ep in all_episodes}

    if debug > 0:
        all_episodes = all_episodes[:debug]
        print(f"Debug mode: converting first {len(all_episodes)} episodes")

    est_gb = len(all_episodes) * 0.167
    print(f"Est. output: {est_gb:.0f} GB → mecka_data_zip (/mnt/zarr-zip)")
    if force:
        print("Force mode: will overwrite existing tars")

    if dry_run:
        print("Dry run — exiting without converting.")
        return

    def _run_conversion(episodes: list[str], label: str) -> tuple[list[dict], list]:
        """Starmap convert_episode over episodes; return (good_results, errors)."""
        pairs = [(ep, force) for ep in episodes]
        results = []
        try:
            for item in convert_episode.starmap(pairs, return_exceptions=True):
                results.append(item)
        except Exception as exc:
            print(
                f"Warning: starmap() raised after {len(results)}/{len(episodes)} episodes "
                f"({label}): {exc}"
            )
        ok      = [r for r in results if isinstance(r, dict) and not r.get("skipped")]
        skipped = [r for r in results if isinstance(r, dict) and r.get("skipped")]
        errs    = [r for r in results if isinstance(r, Exception) or r is None]
        total_frames = sum(r["n_frames"] for r in ok)
        print(
            f"\n[{label}] Converted: {len(ok):,}  Skipped: {len(skipped):,}  "
            f"Errors: {len(errs):,}  ({total_frames:,} frames)"
        )
        return ok + skipped, errs

    # --- Pass 1: main conversion ---
    print(f"\nLaunching {len(all_episodes):,} parallel convert_episode calls "
          f"(max_containers=3000)...")
    t0 = time.time()
    good, _ = _run_conversion(all_episodes, "pass-1")
    print(f"Pass 1 complete in {time.time() - t0:.0f}s")

    # --- Fix passes: scan for tars without .done, delete, reconvert ---
    _MAX_FIX_PASSES = 3
    all_good = list(good)
    for fix_pass in range(1, _MAX_FIX_PASSES + 1):
        print(f"\nScanning zip volume for incomplete tars (fix pass {fix_pass})...")
        incomplete_hashes = scan_and_fix.remote()
        if not incomplete_hashes:
            print("No incomplete tars — volume is clean.")
            break

        fix_dirs = [hash_to_dir[h] for h in incomplete_hashes if h in hash_to_dir]
        unknown  = [h for h in incomplete_hashes if h not in hash_to_dir]
        if unknown:
            print(f"  Warning: {len(unknown)} hashes not found in input volume (already deleted?): {unknown[:5]}")

        if not fix_dirs:
            break

        print(f"Reconverting {len(fix_dirs)} incomplete episodes...")
        t_fix = time.time()
        fix_good, _ = _run_conversion(fix_dirs, f"fix-pass-{fix_pass}")
        print(f"Fix pass {fix_pass} complete in {time.time() - t_fix:.0f}s")
        all_good.extend(fix_good)
    else:
        print(f"Warning: still have incomplete tars after {_MAX_FIX_PASSES} fix passes.")

    # --- Write catalog ---
    # Deduplicate by episode_hash (later entries win)
    seen: dict[str, dict] = {}
    for r in all_good:
        if isinstance(r, dict):
            seen[r["episode_hash"]] = r
    good_unique = list(seen.values())

    if good_unique:
        print(f"\nWriting catalog.json ({len(good_unique):,} entries)...")
        write_catalog.remote(good_unique)
        print("Done.")
    else:
        print("No successful conversions — catalog.json not written.")

    print(f"\nTotal wall time: {time.time() - t0:.0f}s")
