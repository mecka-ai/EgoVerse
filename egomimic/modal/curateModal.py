"""Modal curation entrypoints for DemInf scoring at scale.

Architecture: one container per task — each loads only its task's episodes,
applies preprocessing filters, and runs KSG scoring independently.  The
orchestrator aggregates scores from all task containers into a single JSON.

Usage
-----
Fire-and-forget:
    modal run --env robotics egomimic/modal/curateModal.py::submit_curate -- \\
        name=my_run description=test

Blocking (streams logs):
    modal run --env robotics egomimic/modal/curateModal.py::run_curate_cmd -- \\
        name=my_run description=test

Override data or model:
    ... -- data=deminf_mecka model=deminf_default model.scorer.k_range=[5,9]

KSG is CPU-bound; no GPU needed by default. For StateEmbedder mode=image:
    ... -- +modal_gpu=A100
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import modal

# modal_setup.py lives next to this file locally (egomimic/modal/) and is baked
# into the image at /root/ so it is importable before the repo is cloned.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modal_setup import (  # noqa: E402
    CFG,
    _local_wandb_key,
    _prepare_repo,
    _prepare_repo_light,
    _resolve_git_state,
    app,
    training_outputs_volume,
    zarr_volume,
)

_CURATE_GPU = os.environ.get("MODAL_GPU") or None
_CURATE_CPU = float(os.environ.get("MODAL_CPU", "32"))
_CURATE_MEMORY_MB = (
    int(float(os.environ.get("MODAL_MEMORY_GB")) * 1024)
    if os.environ.get("MODAL_MEMORY_GB")
    else int(os.environ.get("MODAL_MEMORY_MB", "65536"))
)


# ---------------------------------------------------------------------------
# Per-task worker — one container per task group
# ---------------------------------------------------------------------------


@app.function(
    gpu=None,
    cpu=8,
    memory=32768,
    timeout=3600,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={CFG.volume_mount_path: zarr_volume},
)
def _score_task(
    task_name: str,
    path_hash_pairs: list,
    git_remote: str,
    git_commit: str,
    hydra_args: tuple[str, ...],
) -> tuple[str, dict]:
    """Load and KSG-score all episodes for one task. Returns (task_name, {hash: score})."""
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path as _Path
    import sys as _sys

    _prepare_repo_light(git_remote=git_remote, git_commit=git_commit)
    _sys.path.insert(0, CFG.remote_repo_dir)

    import hydra as _hydra
    from egomimic.curation.utils import load_episode_from_path
    from egomimic.curation.filters import apply_filters

    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        cfg = _hydra.compose("curate", overrides=list(hydra_args))

    algo = _hydra.utils.instantiate(cfg.model)

    def _load_one(pair):
        path_str, episode_hash = pair
        ep = load_episode_from_path(_Path(path_str), episode_hash=episode_hash)
        if ep is not None:
            ep.metadata["task_name"] = task_name
        return ep

    n_threads = min(16, max(1, len(path_hash_pairs)))
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        results = list(pool.map(_load_one, path_hash_pairs))

    episodes = [ep for ep in results if ep is not None]
    print(f"[{task_name}] loaded {len(episodes)}/{len(path_hash_pairs)} episodes")

    if not episodes:
        return task_name, {}

    filtered_episodes, _ = apply_filters(episodes, algo.preprocessing)
    print(f"[{task_name}] {len(filtered_episodes)}/{len(episodes)} after preprocessing")

    if len(filtered_episodes) < 2:
        print(f"[{task_name}] too few episodes for KSG — returning nan scores")
        return task_name, {ep.episode_hash: float("nan") for ep in filtered_episodes}

    algo._scorer.fit(filtered_episodes)
    scores = algo._scorer.get_scores()
    print(f"[{task_name}] scored {len(scores)} episodes")
    return task_name, scores


# ---------------------------------------------------------------------------
# Curation orchestrator — SQL grouping + per-task fan-out
# ---------------------------------------------------------------------------


@app.function(
    gpu=_CURATE_GPU,
    cpu=_CURATE_CPU,
    memory=_CURATE_MEMORY_MB,
    timeout=CFG.timeout_seconds,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_curate(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    wandb_api_key: str = "",
) -> str:
    """SQL task grouping + per-task container fan-out for DemInf curation."""
    import sys as _sys
    import time as _time
    import numpy as _np
    from pathlib import Path as _Path

    _prepare_repo(git_remote=git_remote, git_commit=git_commit)
    _sys.path.insert(0, CFG.remote_repo_dir)

    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    # ── 1. Load Hydra config ──────────────────────────────────────────────────
    import hydra as _hydra

    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        cfg = _hydra.compose("curate", overrides=list(hydra_args))

    import lightning as _L
    from egomimic.rldb.zarr.utils import set_global_seed

    if cfg.get("seed"):
        _L.seed_everything(cfg.seed, workers=True)
        set_global_seed(cfg.seed)

    from egomimic.utils.aws.aws_data_utils import load_env
    load_env()

    # ── 2. Scan episode paths from data config ────────────────────────────────
    from omegaconf import OmegaConf
    from egomimic.rldb.filters import DatasetFilter
    from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver

    train_datasets = OmegaConf.to_container(
        cfg.data.train_datasets, resolve=True, throw_on_missing=False
    )
    all_paths: list = []
    for ds_name, ds_cfg in train_datasets.items():
        resolver_cfg = (ds_cfg or {}).get("resolver") or {}
        folder_path = resolver_cfg.get("folder_path")
        if not folder_path:
            continue
        filter_lambdas = list(
            (((ds_cfg or {}).get("filters") or {}).get("filter_lambdas")) or []
        )
        exclude_hashes: set[str] = set()
        eps_to_ignore_path = resolver_cfg.get("eps_to_ignore")
        if eps_to_ignore_path:
            try:
                with open(eps_to_ignore_path) as _f:
                    exclude_hashes.update(json.load(_f))
                print(f"[scan] {ds_name}: eps_to_ignore — {len(exclude_hashes)} hashes")
            except OSError as exc:
                print(f"[scan] {ds_name}: eps_to_ignore not accessible ({exc}) — skipping")
        include_hashes: set[str] | None = None
        eps_to_use_path = resolver_cfg.get("eps_to_use")
        if eps_to_use_path:
            try:
                with open(eps_to_use_path) as _f:
                    include_hashes = set(json.load(_f))
                print(f"[scan] {ds_name}: eps_to_use — {len(include_hashes)} hashes")
            except OSError as exc:
                print(f"[scan] {ds_name}: eps_to_use not accessible ({exc}) — skipping")
        debug = resolver_cfg.get("debug")
        try:
            pairs = LocalEpisodeResolver._get_local_filtered_paths(
                search_path=_Path(folder_path),
                filters=DatasetFilter(filter_lambdas),
                debug=debug,
            )
            before = len(pairs)
            if exclude_hashes:
                pairs = [(p, h) for p, h in pairs if h not in exclude_hashes]
            if include_hashes is not None:
                pairs = [(p, h) for p, h in pairs if h in include_hashes]
            if len(pairs) != before:
                print(f"[scan] {ds_name}: {before} → {len(pairs)} after eps filters")
            all_paths.extend(pairs)
            print(f"[scan] {ds_name}: {len(pairs)} paths from {folder_path}")
        except Exception as exc:
            print(f"[scan] {ds_name}: failed — {exc}")

    total = len(all_paths)
    print(f"Total episode paths: {total}")
    if total == 0:
        print("No episodes found — check resolver.folder_path in data config")
        return ""

    # ── 3. SQL task grouping ──────────────────────────────────────────────────
    all_hashes = [h for _, h in all_paths]
    path_by_hash = {h: str(p) for p, h in all_paths}

    try:
        from egomimic.utils.aws.aws_sql import batch_get_task_names, create_default_engine
        engine = create_default_engine()
        task_map = batch_get_task_names(engine, all_hashes)
        print(
            f"[sql] got task names for {len(task_map)}/{total} episodes "
            f"({len(set(task_map.values()))} distinct tasks)"
        )
    except Exception as exc:
        print(f"[sql] task name lookup failed — grouping all as 'unknown': {exc}")
        task_map = {}

    by_task: dict[str, list] = {}
    for h in all_hashes:
        task = task_map.get(h, "unknown")
        by_task.setdefault(task, []).append((path_by_hash[h], h))

    n_tasks = len(by_task)
    summary = ", ".join(f"{t}:{len(p)}" for t, p in sorted(by_task.items())[:5])
    print(f"Task groups: {n_tasks} ({summary}{'...' if n_tasks > 5 else ''})")

    # ── 4. WandB setup ────────────────────────────────────────────────────────
    # hydra.compose() does not set HydraConfig, so logger configs that reference
    # ${hydra:runtime.output_dir} or ${now:...} would fail. We init WandB directly.
    wandb_run = None
    try:
        import wandb as _wandb
        _wandb.init(
            project="mecka-robotics",
            entity="kevin_yam1_-mecka-ai",
            name=f"{cfg.name}_{cfg.description}",
            config={
                "name": cfg.name,
                "description": cfg.description,
                "model": str(cfg.model._target_),
            },
        )
        wandb_run = _wandb.run
        print(f"[wandb] run initialised: {wandb_run.name}")
    except Exception as exc:
        print(f"[wandb] init skipped: {exc}")

    # ── 5. Output dir ─────────────────────────────────────────────────────────
    timestamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = _Path(CFG.output_mount_path) / cfg.name / f"{cfg.description}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 6. Fan-out per-task scoring ───────────────────────────────────────────
    task_args = [
        (task_name, pairs, git_remote, git_commit, hydra_args)
        for task_name, pairs in sorted(by_task.items())
    ]
    print(f"Fanning out {len(task_args)} task containers …")
    t0 = _time.time()

    scores_by_task: dict[str, dict[str, float]] = {}
    n_task_failures = 0

    for task_result in _score_task.starmap(task_args, return_exceptions=True):
        if isinstance(task_result, Exception):
            n_task_failures += 1
            print(f"[task] FAILED (skipping): {task_result}")
        else:
            t_name, t_scores = task_result
            scores_by_task[t_name] = t_scores

    elapsed = _time.time() - t0
    print(
        f"Scoring done in {elapsed:.1f}s — "
        f"{len(scores_by_task)}/{len(task_args)} tasks succeeded "
        f"({n_task_failures} failed)"
    )

    # ── 7. Aggregate stats ────────────────────────────────────────────────────
    flat_scores: dict[str, float] = {}
    for t_scores in scores_by_task.values():
        flat_scores.update(t_scores)

    all_score_vals = _np.array([s for s in flat_scores.values() if _np.isfinite(s)])

    per_task_stats: dict[str, dict] = {}
    for t_name, t_scores in scores_by_task.items():
        vals = _np.array([s for s in t_scores.values() if _np.isfinite(s)])
        per_task_stats[t_name] = {
            "count": len(t_scores),
            "mi_mean": float(_np.nanmean(vals)) if len(vals) else float("nan"),
            "mi_std": float(_np.nanstd(vals)) if len(vals) else float("nan"),
            "mi_median": float(_np.nanmedian(vals)) if len(vals) else float("nan"),
            "mi_min": float(_np.nanmin(vals)) if len(vals) else float("nan"),
            "mi_max": float(_np.nanmax(vals)) if len(vals) else float("nan"),
        }

    stats = {
        "total_input": total,
        "n_tasks": len(scores_by_task),
        "n_task_failures": n_task_failures,
        "scored": len(flat_scores),
        "elapsed_seconds": round(elapsed, 1),
        "mi_mean": float(all_score_vals.mean()) if len(all_score_vals) else float("nan"),
        "mi_std": float(all_score_vals.std()) if len(all_score_vals) else float("nan"),
        "mi_median": float(_np.median(all_score_vals)) if len(all_score_vals) else float("nan"),
        "mi_min": float(all_score_vals.min()) if len(all_score_vals) else float("nan"),
        "mi_max": float(all_score_vals.max()) if len(all_score_vals) else float("nan"),
        "per_task": per_task_stats,
    }

    # ── 8. Save outputs ───────────────────────────────────────────────────────
    with open(output_dir / "scores.json", "w") as f:
        json.dump(flat_scores, f, indent=2)
    with open(output_dir / "scores_by_task.json", "w") as f:
        json.dump(scores_by_task, f, indent=2)
    with open(output_dir / "kept_hashes.json", "w") as f:
        json.dump(list(flat_scores.keys()), f, indent=2)
    with open(output_dir / "curation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"Curation done — scored={len(flat_scores)}  "
        f"n_tasks={len(scores_by_task)}  output={output_dir}"
    )

    # ── 9. WandB logging ─────────────────────────────────────────────────────
    if wandb_run is not None:
        try:
            metrics: dict = {
                "curation/total_input": total,
                "curation/scored": len(flat_scores),
                "curation/n_tasks": len(scores_by_task),
                "curation/n_task_failures": n_task_failures,
                "curation/mi_mean": stats["mi_mean"],
                "curation/mi_std": stats["mi_std"],
                "curation/mi_median": stats["mi_median"],
            }
            for t_name, ts in per_task_stats.items():
                safe = t_name.replace(".", "_").replace("/", "_")[:64]
                metrics[f"curation/task/{safe}/count"] = ts["count"]
                metrics[f"curation/task/{safe}/mi_mean"] = ts["mi_mean"]
                metrics[f"curation/task/{safe}/mi_median"] = ts["mi_median"]
            wandb_run.log(metrics)
        except Exception as exc:
            print(f"WandB logging failed: {exc}")
        try:
            wandb_run.finish()
        except Exception:
            pass

    zarr_volume.commit()
    training_outputs_volume.commit()

    return str(output_dir)


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def submit_curate(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a curation job and return immediately."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo has uncommitted changes. Modal will run the last committed state only.")
    print(f"Submitting curation at commit {git_commit[:12]} from {git_remote}")
    handle = run_curate.spawn(
        tuple(hydra_args), git_remote, git_commit, _local_wandb_key()
    )
    print(f"Submitted Modal curation job: {handle.object_id}")
    print("Monitor at: https://modal.com/apps/egomimic-training")


@app.local_entrypoint()
def run_curate_cmd(*hydra_args: str) -> None:
    """Blocking run: streams curation logs to stdout and waits for completion."""
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo has uncommitted changes. Modal will run the last committed state only.")
    print(f"Running curation at commit {git_commit[:12]} from {git_remote}")
    result = run_curate.remote(
        tuple(hydra_args), git_remote, git_commit, _local_wandb_key()
    )
    print(f"Curation complete: {result}")
