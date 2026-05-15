"""Modal-parallelized cache stage for DemInf curation.

What this does
--------------
Fans one container out per zarr episode on the ``mecka_data_v2`` volume,
loads (observations, actions, embodiment, metadata) via the shared
``egomimic.curation.utils.load_episode_from_path``, and writes a compact
``<hash>.npz`` to ``egoverse-training-outputs/curation/<run-id>/embeddings/``.

Why "cache raw arrays" and not "embed per-episode"
--------------------------------------------------
DemInf's StateEmbedder / ActionEmbedder are global estimators — their
Gaussian normalisation stats and random projections are fit across the
entire dataset (or per-embodiment group), so a meaningful embedding cannot
be produced one episode at a time without first observing the whole dataset.
We therefore parallelise the *I/O-heavy* zarr-to-numpy step here. The
companion ``score_and_filter_modal.py`` then loads the cached arrays into
a single container, fits the embedders globally, runs KSG, and writes the
keep/drop manifest. This split keeps the per-episode container CPU-only
(zarr reads are I/O bound) and matches the algorithm's actual structure.

Outputs (on egoverse-training-outputs volume)
---------------------------------------------
- ``curation/<run-id>/embeddings/<hash>.npz`` per episode (obs, actions,
  embodiment, episode_hash, length, metadata-json)
- ``curation/<run-id>/cache_summary.json`` aggregate stats

Usage
-----
    source emimic/bin/activate
    export MODAL_ENVIRONMENT=robotics
    modal run egomimic/curation/modal/embed_episodes_modal.py \\
        -- --dataset-root . --smoke
"""

from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import modal

# ─── Tunables ─────────────────────────────────────────────────────────────────
MAX_CONTAINERS = int(os.environ.get("EGOVERSE_MAX_CONTAINERS", "50"))
SMOKE_EPISODE_COUNT = 8

# ─── Volume / image config ────────────────────────────────────────────────────
DATA_VOLUME_NAME = os.environ.get("EGOVERSE_DATA_VOLUME", "mecka_data_v2")
OUTPUT_VOLUME_NAME = "egoverse-training-outputs"
DATA_MOUNT = "/data"
OUTPUT_MOUNT = "/egoverse-training-outputs"

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "numpy",
            "zarr==3.1.5",
            "tqdm",
            "scipy",
            "omegaconf",
            "simplejpeg",
            "cloudpathlib",
            "sqlalchemy",
            "opencv-python-headless",
            "pandas",
        ]
    )
    .add_local_python_source("egomimic")
)

app = modal.App("egomimic-curation-cache", image=image)
data_vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
output_vol = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


# ─── Episode discovery ────────────────────────────────────────────────────────


@app.function(volumes={DATA_MOUNT: data_vol}, timeout=600)
def discover_episodes(dataset_root: str) -> list[tuple[str, str]]:
    """Return list of (path_str, episode_hash) for every zarr store under
    ``DATA_MOUNT/<dataset_root>``. Cheap single os.listdir, no .zattrs read."""
    root = Path(DATA_MOUNT) / dataset_root.lstrip("/")
    if not root.is_dir():
        print(f"[cache] root does not exist: {root}")
        return []

    out: list[tuple[str, str]] = []
    for name in sorted(os.listdir(root)):
        if name.startswith("."):
            continue
        episode_hash = name[:-5] if name.endswith(".zarr") else name
        out.append((str(root / name), episode_hash))
    print(f"[cache] listed {len(out)} entries under {root}")
    return out


# ─── Per-episode cache worker ─────────────────────────────────────────────────


@app.function(
    volumes={DATA_MOUNT: data_vol, OUTPUT_MOUNT: output_vol},
    timeout=600,
    max_containers=MAX_CONTAINERS,
    cpu=2.0,
    memory=4096,
)
def cache_episode(ep_path_str: str, episode_hash: str, run_id: str) -> dict:
    """Load one zarr episode, serialise (obs, actions, embodiment, metadata)
    to ``curation/<run-id>/embeddings/<hash>.npz`` on the output volume.

    Returns a small result dict per episode; never raises so .starmap()
    doesn't abort on a single bad episode.
    """
    import numpy as np

    from egomimic.curation.utils import load_episode_from_path

    out_dir = Path(OUTPUT_MOUNT) / "curation" / run_id / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{episode_hash}.npz"

    try:
        ep = load_episode_from_path(Path(ep_path_str), episode_hash=episode_hash)
        if ep is None:
            return {
                "episode_hash": episode_hash,
                "ep_path": ep_path_str,
                "status": "skip",
                "reason": "load_returned_none",
            }

        # Metadata can contain non-JSON-friendly bits — coerce to JSON via str.
        meta_json = json.dumps(ep.metadata, default=str)

        np.savez_compressed(
            out_path,
            observations=ep.observations.astype(np.float32),
            actions=ep.actions.astype(np.float32),
            embodiment=np.array(ep.embodiment),
            episode_hash=np.array(ep.episode_hash),
            length=np.array(len(ep.actions), dtype=np.int64),
            metadata=np.array(meta_json),
        )

        return {
            "episode_hash": episode_hash,
            "ep_path": ep_path_str,
            "status": "ok",
            "embodiment": ep.embodiment,
            "length": int(len(ep.actions)),
            "obs_dim": int(ep.observations.shape[-1])
            if ep.observations.ndim >= 2
            else 1,
            "action_dim": int(ep.actions.shape[-1]) if ep.actions.ndim >= 2 else 1,
            "out_path": str(out_path),
        }

    except Exception as exc:
        print(f"[ERROR] ep={episode_hash}: {exc}")
        return {
            "episode_hash": episode_hash,
            "ep_path": ep_path_str,
            "status": "error",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


# ─── Summary writer ───────────────────────────────────────────────────────────


@app.function(
    volumes={OUTPUT_MOUNT: output_vol},
    timeout=600,
)
def save_cache_summary(results: list[dict], run_id: str, runtime_seconds: float) -> str:
    run_dir = Path(OUTPUT_MOUNT) / "curation" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skip"]
    errors = [r for r in results if r["status"] == "error"]

    per_embodiment: dict[str, int] = {}
    total_timesteps = 0
    for r in ok:
        per_embodiment[r["embodiment"]] = per_embodiment.get(r["embodiment"], 0) + 1
        total_timesteps += int(r.get("length", 0))

    summary = {
        "run_id": run_id,
        "stage": "cache",
        "total_input": len(results),
        "cached_ok": len(ok),
        "skipped": len(skipped),
        "errors": len(errors),
        "per_embodiment": per_embodiment,
        "total_timesteps": total_timesteps,
        "runtime_seconds": runtime_seconds,
    }
    (run_dir / "cache_summary.json").write_text(json.dumps(summary, indent=2))

    with open(run_dir / "cache_errors.jsonl", "w") as fh:
        for r in errors:
            fh.write(json.dumps(r) + "\n")

    output_vol.commit()
    return str(run_dir)


# ─── Local entrypoint ─────────────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    dataset_root: str = ".",
    run_id: str = "",
    smoke: bool = False,
    pct: float = 100.0,
):
    """Cache raw episode arrays in parallel across Modal.

    Args mapped from CLI flags (underscore → dash):
      --dataset-root  path within mecka_data_v2 (default: root)
      --run-id        optional run label (default: UTC timestamp)
      --smoke         process only ``SMOKE_EPISODE_COUNT`` episodes
      --pct           percentage of discovered episodes (ignored when --smoke)
    """
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    t0 = time.time()

    print(f"[cache] run_id      = {run_id}")
    print(f"[cache] dataset     = {dataset_root}")
    print(f"[cache] smoke       = {smoke}")

    raw: list[tuple[str, str]] = discover_episodes.remote(dataset_root)
    if not raw:
        print("[cache] no episodes found — exiting")
        return

    if smoke:
        raw = raw[:SMOKE_EPISODE_COUNT]
        print(f"[cache] smoke mode: caching first {len(raw)} episodes")
    elif pct < 100.0:
        import random

        k = max(1, int(round(len(raw) * pct / 100.0)))
        raw = sorted(random.Random(42).sample(raw, k))
        print(f"[cache] sampling {k} episodes ({pct:.1f}%)")

    args = [(p, h, run_id) for p, h in raw]

    results: list[dict] = []
    n_done = n_err = 0
    for result in cache_episode.starmap(
        args, order_outputs=False, return_exceptions=True
    ):
        n_done += 1
        if isinstance(result, BaseException):
            n_err += 1
            results.append(
                {
                    "episode_hash": f"__unhandled_{n_done}",
                    "ep_path": "",
                    "status": "error",
                    "error": str(result),
                    "traceback": "",
                }
            )
        else:
            results.append(result)
            if result["status"] == "error":
                n_err += 1

    elapsed = time.time() - t0
    run_path = save_cache_summary.remote(results, run_id, elapsed)
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(
        f"[cache] done — ok={n_ok} err={n_err} runtime={elapsed:.1f}s  out={run_path}"
    )
    print(f"[cache] run_id={run_id}")
