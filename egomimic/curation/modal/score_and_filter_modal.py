"""Modal scoring + filtering stage for DemInf curation.

Loads the cached per-episode NPZ files written by
``embed_episodes_modal.py`` (under ``curation/<run-id>/embeddings/``), runs
the full ``DemInfCurator`` pipeline globally inside a single container,
and writes a keep/drop manifest + summary to the same run directory.

This stage is intentionally single-container: KSG and the embedder Gaussian
stats need the full dataset in one place. Fan-out happens at the cache stage.

Outputs (on egoverse-training-outputs volume)
---------------------------------------------
- ``curation/<run-id>/manifest.jsonl`` — one JSON line per episode with
  ``{episode_hash, embodiment, status: kept|low_mi|removed, mi_score}``
- ``curation/<run-id>/summary.json``  — aggregate stats from CurationResult
- ``curation/<run-id>/scores.json``   — full {hash: score} map
- ``curation/<run-id>/filter.yaml``   — DemInfCurator.export_sql_filter(yaml)

Usage
-----
    modal run egomimic/curation/modal/score_and_filter_modal.py \\
        -- --run-id <from cache stage> --smoke
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

OUTPUT_VOLUME_NAME = "egoverse-training-outputs"
OUTPUT_MOUNT = "/egoverse-training-outputs"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        [
            "numpy",
            "scipy",
            "torch",
            "torchvision",
            "tqdm",
            "omegaconf",
            "pandas",
            "cloudpathlib",
            "sqlalchemy",
            "opencv-python-headless",
            "zarr==3.1.5",
            "simplejpeg",
        ]
    )
    .add_local_python_source("egomimic")
)

app = modal.App("egomimic-curation-score", image=image)
output_vol = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)


@app.function(
    volumes={OUTPUT_MOUNT: output_vol},
    timeout=3600,
    cpu=8.0,
    memory=32768,
)
def score_and_filter(
    run_id: str,
    filter_ratio: float = 0.3,
    state_mode: str = "proprioceptive",
    latent_dim: int = 32,
    k_min: int = 3,
    k_max: int = 7,
    pause_epsilon: float = 1e-2,
    min_trajectory_length: int = 20,
    cross_embodiment_mode: str = "independent",
    max_episodes: int | None = None,
) -> dict:
    """Run DemInfCurator over all NPZs in ``curation/<run-id>/embeddings/``."""
    import numpy as np

    from egomimic.curation import DemInfCurator, Episode

    # Reload the volume so we see whatever the cache stage just committed.
    output_vol.reload()

    run_dir = Path(OUTPUT_MOUNT) / "curation" / run_id
    emb_dir = run_dir / "embeddings"
    if not emb_dir.is_dir():
        raise FileNotFoundError(
            f"No embeddings directory at {emb_dir} — run embed_episodes_modal first"
        )

    npz_paths = sorted(emb_dir.glob("*.npz"))
    if max_episodes is not None:
        npz_paths = npz_paths[:max_episodes]
    if not npz_paths:
        raise RuntimeError(f"No .npz files found under {emb_dir}")

    print(f"[score] loading {len(npz_paths)} cached episodes from {emb_dir}")

    episodes: list[Episode] = []
    for p in npz_paths:
        try:
            z = np.load(p, allow_pickle=False)
            episodes.append(
                Episode(
                    episode_hash=str(z["episode_hash"]),
                    observations=z["observations"],
                    actions=z["actions"],
                    embodiment=str(z["embodiment"]),
                    metadata=json.loads(str(z["metadata"])) if "metadata" in z else {},
                )
            )
        except Exception as exc:
            print(f"[score] failed to load {p.name}: {exc}")

    print(f"[score] loaded {len(episodes)} episodes; running curation …")
    curator = DemInfCurator(
        filter_ratio=filter_ratio,
        state_mode=state_mode,
        latent_dim=latent_dim,
        k_range=(k_min, k_max),
        pause_epsilon=pause_epsilon,
        min_trajectory_length=min_trajectory_length,
        cross_embodiment_mode=cross_embodiment_mode,
        device="cpu",
    )
    result = curator.curate(episodes)

    # Manifest: one JSON line per episode with keep/drop status.
    embodiment_by_hash = {ep.episode_hash: ep.embodiment for ep in episodes}
    kept = set(result.kept_hashes)
    low_mi = set(result.low_mi_hashes)
    pre_removed = set(result.removed_hashes)

    manifest_path = run_dir / "manifest.jsonl"
    with open(manifest_path, "w") as fh:
        for ep in episodes:
            h = ep.episode_hash
            if h in kept:
                status = "kept"
            elif h in low_mi:
                status = "low_mi"
            elif h in pre_removed:
                status = "removed_preprocess"
            else:
                status = "unknown"
            fh.write(
                json.dumps(
                    {
                        "episode_hash": h,
                        "embodiment": embodiment_by_hash.get(h),
                        "status": status,
                        "mi_score": result.scores.get(h),
                    }
                )
                + "\n"
            )

    curator.save_result(run_dir)
    try:
        curator.export_sql_filter(run_dir / "filter.yaml", format="yaml")
    except Exception as exc:
        print(f"[score] export_sql_filter failed (non-fatal): {exc}")

    output_vol.commit()
    return {
        "run_dir": str(run_dir),
        "kept": len(result.kept_hashes),
        "low_mi": len(result.low_mi_hashes),
        "pre_removed": len(result.removed_hashes),
        "total_scored": len(result.scores),
        "stats": result.stats,
    }


@app.local_entrypoint()
def main(
    run_id: str,
    filter_ratio: float = 0.3,
    state_mode: str = "proprioceptive",
    latent_dim: int = 32,
    k_min: int = 3,
    k_max: int = 7,
    pause_epsilon: float = 1e-2,
    min_trajectory_length: int = 20,
    cross_embodiment_mode: str = "independent",
    smoke: bool = False,
):
    """Score cached episodes under ``curation/<run-id>/`` and write a manifest.

    --run-id     required, matches the cache stage's run id
    --smoke      cap to first 8 episodes (matches embed_episodes_modal --smoke)
    """
    if not run_id:
        raise SystemExit("--run-id is required (output of embed_episodes_modal)")

    t0 = time.time()
    print(f"[score] run_id={run_id} smoke={smoke}")
    out = score_and_filter.remote(
        run_id=run_id,
        filter_ratio=filter_ratio,
        state_mode=state_mode,
        latent_dim=latent_dim,
        k_min=k_min,
        k_max=k_max,
        pause_epsilon=pause_epsilon,
        min_trajectory_length=min_trajectory_length,
        cross_embodiment_mode=cross_embodiment_mode,
        max_episodes=8 if smoke else None,
    )
    elapsed = time.time() - t0
    print(
        f"[score] done — kept={out['kept']} low_mi={out['low_mi']} "
        f"pre_removed={out['pre_removed']} runtime={elapsed:.1f}s  out={out['run_dir']}"
    )
