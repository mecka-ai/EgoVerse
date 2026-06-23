"""DemInf v2 curation pipeline — three-phase orchestrator.

Phase 1: build MP4+NPZ shards from zarr episodes (50 parallel workers)
Phase 2: embed state (L40S GPU) + action (CPU) in parallel per task
Phase 3: KSG mutual-information scoring from pre-computed latents

Usage:
  python egomimic/modal/curate_v2.py \\
    name=<run_name> description=<desc> \\
    data=mecka_all_zarr model=deminf_default trainer=ddp_modal

The pipeline replaces the WDS tar sharding + combined GPU embed of curateModal.py
with a decoupled format that separates state/action embedding onto different hardware.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import modal
from modal_setup import (
    CFG,
    CURATE_ORCHESTRATOR,
    MODAL_COMPUTE_ARG_MAP,
    ModalCompute,
    _boot_container,
    _local_hf_token,
    _prepare_repo,
    _resolve_git_state,
    app,
    app_name_from_hydra_args,
    deminf_v2_volume,
    DEMINF_V2_MOUNT,
    image,
    launch_detached,
    pop_init_submodules,
    training_outputs_volume,
    zarr_volume,
)
from build_deminf_shards import build_shards_for_task, _build_shards_worker  # noqa: F401
from embed_deminf_shards import _embed_state_shards, _embed_action_shards  # noqa: F401

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]

# Per-task KSG scoring: 32 CPUs so n_threads can be set high.
_SCORE_COMPUTE = ModalCompute(gpu=None, cpu=32.0, memory_mb=49152)


def _load_cfg(hydra_args: tuple[str, ...]):
    import hydra as _hydra
    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        return _hydra.compose("curate", overrides=list(hydra_args))


# ---------------------------------------------------------------------------
# Phase 3: KSG scoring from pre-computed latents
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=_SCORE_COMPUTE.gpu,
    cpu=_SCORE_COMPUTE.cpu,
    memory=_SCORE_COMPUTE.memory_mb,
    timeout=86400,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.output_mount_path: training_outputs_volume,
    },
)
def _score_task_v2(
    task_name: str,
    run_name: str,
    hydra_args: tuple[str, ...],
    action_mean: list,
    action_std: list,
    output_dir: str,
    git_remote: str,
    git_commit: str,
    hf_token: str = "",
) -> tuple[str, dict[str, float]]:
    """Load pre-computed latents and run KSG scoring. Returns (task_name, scores)."""
    import time as _time
    import numpy as _np

    _boot_container(git_remote, git_commit, hf_token)

    from egomimic.curation.config import apply_curation_seed, select_seed
    from egomimic.curation.scoring import trajectory_scorer_from_cfg
    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()
    cfg = _load_cfg(hydra_args)
    apply_curation_seed(select_seed(cfg))

    tag = f"[{task_name}][score]"
    t_start = _time.perf_counter()

    lat_dir = Path(output_dir) / "latents_v2" / task_name
    state_path = lat_dir / "state.npz"
    action_path = lat_dir / "action.npz"

    if not state_path.exists() or not action_path.exists():
        print(f"{tag} latent files not found — skipping KSG")
        return task_name, {}

    state_npz = _np.load(str(state_path), allow_pickle=True)
    action_npz = _np.load(str(action_path), allow_pickle=True)

    # Match episodes present in both state and action latents
    state_keys = set(state_npz.files)
    action_keys = set(action_npz.files)
    common = sorted(state_keys & action_keys)
    missing = (state_keys ^ action_keys)
    if missing:
        print(f"{tag} {len(missing)} episodes in only one latent file — will be skipped")

    if not common:
        print(f"{tag} no episodes with both state and action latents — skipping KSG")
        return task_name, {}

    # Build stacked arrays for scorer
    state_latents_list = []
    action_latents_list = []
    episode_hashes = []
    ep_lengths = []

    for ep_hash in common:
        s = _np.asarray(state_npz[ep_hash], dtype=_np.float32)
        a = _np.asarray(action_npz[ep_hash], dtype=_np.float32)
        if len(s) == 0 or len(a) == 0:
            continue
        if len(s) != len(a):
            print(f"{tag} {ep_hash[:8]}: state/action length mismatch ({len(s)} vs {len(a)}) — trimming")
            T = min(len(s), len(a))
            s, a = s[:T], a[:T]
        state_latents_list.append(s)
        action_latents_list.append(a)
        episode_hashes.append(ep_hash)
        ep_lengths.append(len(s))

    if not episode_hashes:
        print(f"{tag} no valid episodes after length checks — returning empty")
        return task_name, {}

    s_all = _np.concatenate(state_latents_list, axis=0)
    a_all = _np.concatenate(action_latents_list, axis=0)
    n_total = s_all.shape[0]

    print(
        f"{tag} KSG: {len(episode_hashes)} episodes, {n_total} timesteps, "
        f"state_dim={s_all.shape[1]}, action_dim={a_all.shape[1]}"
    )

    t_ksg = _time.perf_counter()
    scorer = trajectory_scorer_from_cfg(cfg)
    scores = scorer.score_latents(s_all, a_all, episode_hashes, ep_lengths)
    print(f"{tag} KSG done in {_time.perf_counter() - t_ksg:.1f}s — {len(scores)} episodes scored")

    # t-SNE visualization (non-fatal)
    try:
        from egomimic.curation.tsne_viz import (
            TsneVizSettings,
            export_task_tsne3d,
            make_task_tsne_plots,
        )
        from egomimic.curation.config import select_tsne_viz_config

        tsne_cfg = select_tsne_viz_config(cfg)
        viz_settings = TsneVizSettings(
            every_n=tsne_cfg.every_n,
            seed=select_seed(cfg),
            include_state_lang=tsne_cfg.include_state_lang,
            include_language=tsne_cfg.include_language,
            include_state_by_lang=tsne_cfg.include_state_by_lang,
            state_color_by=tsne_cfg.state_color_by,
        )
        tsne_dir = Path(output_dir) / "tsne_v2"
        tsne3d_dir = Path(output_dir) / "tsne3d_v2"
        make_task_tsne_plots(
            task_name,
            state_latents_list,
            action_latents_list,
            tsne_dir,
            settings=viz_settings,
        )
        export_task_tsne3d(
            task_name,
            state_latents_list,
            action_latents_list,
            episode_hashes,
            tsne3d_dir,
            settings=viz_settings,
        )
    except Exception as exc:
        import traceback
        print(f"{tag} t-SNE viz FAILED (non-fatal): {exc}")
        traceback.print_exc()

    # Write per-task scores
    scores_dir = Path(output_dir) / "scores_v2"
    scores_dir.mkdir(parents=True, exist_ok=True)
    with open(scores_dir / f"{task_name}_scores.json", "w") as f:
        json.dump(dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)), f, indent=2)
    training_outputs_volume.commit()

    print(f"{tag} total: {_time.perf_counter() - t_start:.1f}s")
    return task_name, scores


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=CURATE_ORCHESTRATOR.gpu,
    cpu=CURATE_ORCHESTRATOR.cpu,
    memory=CURATE_ORCHESTRATOR.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        DEMINF_V2_MOUNT: deminf_v2_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_curate_v2(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    run_name: str,
    hf_token: str = "",
) -> str:
    """Three-phase DemInf v2 orchestrator: build shards → embed → KSG."""
    import sys as _sys
    import time as _time
    import numpy as _np

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo(git_remote=git_remote, git_commit=git_commit, init_submodules=False)
    _sys.path.insert(0, CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    import hydra as _hydra
    from omegaconf import OmegaConf
    from egomimic.curation.config import (
        apply_curation_seed,
        load_action_norm_stats,
        select_seed,
        select_tensor_keys,
    )
    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()

    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        cfg = _hydra.compose("curate", overrides=list(hydra_args))

    apply_curation_seed(select_seed(cfg))

    # ── 1. SQL task lookup ────────────────────────────────────────────────────
    print("Running SQL task lookup …")
    from egomimic.utils.aws.aws_sql import episode_table_to_df, create_default_engine

    engine = create_default_engine()
    full_df = episode_table_to_df(engine)
    if "is_deleted" in full_df.columns:
        full_df = full_df[full_df["is_deleted"] != True]  # noqa: E712

    hash_to_task: dict[str, str] = {}
    if "task" in full_df.columns:
        hash_to_task = dict(
            zip(full_df["episode_hash"], full_df["task"].fillna("unknown"))
        )

    # ── 2. Resolve episodes via data-config resolver ──────────────────────────
    by_task: dict[str, list[str]] = {}  # task → list of zarr episode dirs

    zarr_root = Path(CFG.volume_mount_path)
    for ds_name, ds_cfg in cfg.data.train_datasets.items():
        resolver = _hydra.utils.instantiate(ds_cfg.resolver)
        dataset_filter = (
            _hydra.utils.instantiate(ds_cfg.filters) if "filters" in ds_cfg else None
        )
        resolved = resolver.resolve(filters=dataset_filter)
        print(f"[{ds_name}] {len(resolved)} episodes after resolver")

        for episode_hash in resolved:
            task = hash_to_task.get(episode_hash) or "unknown"
            if str(task) in ("nan", "None", ""):
                task = "unknown"
            # Resolve zarr dir path on volume
            ep_dir: str | None = None
            for cand in (zarr_root / episode_hash, zarr_root / f"{episode_hash}.zarr"):
                if cand.is_dir():
                    ep_dir = str(cand)
                    break
            if ep_dir is None:
                continue
            by_task.setdefault(task, []).append(ep_dir)

    total_episodes = sum(len(v) for v in by_task.values())
    print(
        f"Episode partition: {total_episodes} episodes across {len(by_task)} tasks — "
        + ", ".join(f"{t}:{len(h)}" for t, h in sorted(by_task.items())[:5])
        + ("…" if len(by_task) > 5 else "")
    )
    if total_episodes == 0:
        print("No episodes found — check data config resolver")
        return ""

    # ── 3. Load norm stats ────────────────────────────────────────────────────
    action_key, _ = select_tensor_keys(cfg)
    try:
        action_mean_arr, action_std_arr = load_action_norm_stats(
            cfg,
            action_key,
            search_roots=[CFG.output_mount_path, CFG.remote_repo_dir, Path.cwd()],
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load norm stats: {exc}") from exc

    action_mean = action_mean_arr.tolist()
    action_std = action_std_arr.tolist()

    # ── 4. Output directory ───────────────────────────────────────────────────
    timestamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(CFG.output_mount_path) / cfg.name / f"{cfg.description}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir)

    print(f"Output dir: {output_dir_str}")

    # ── 5. Per-task fan-out ───────────────────────────────────────────────────
    print(f"Processing {len(by_task)} task(s) …")
    t0 = _time.time()

    scores_by_task: dict[str, dict[str, float]] = {}
    n_failures = 0

    for task_name, ep_dirs in sorted(by_task.items()):
        print(f"\n── [{task_name}] {len(ep_dirs)} episodes ──")
        t_task = _time.perf_counter()

        # Phase 1: build MP4+NPZ shards
        print(f"[{task_name}] Phase 1: building shards …")
        shard_pairs = build_shards_for_task(
            ep_dirs,
            task_name,
            run_name,
            hydra_args,
            git_remote,
            git_commit,
            hf_token,
        )
        if not shard_pairs:
            print(f"[{task_name}] No shards built — skipping")
            n_failures += 1
            continue
        print(f"[{task_name}] Phase 1 done: {len(shard_pairs)} shards")

        # Phase 2: parallel state + action embedding
        print(f"[{task_name}] Phase 2: spawning state + action embedders …")
        state_handle = _embed_state_shards.spawn(
            shard_pairs, task_name, run_name, hydra_args,
            action_mean, action_std, git_remote, git_commit, output_dir_str, hf_token,
        )
        action_handle = _embed_action_shards.spawn(
            shard_pairs, task_name, run_name, hydra_args,
            action_mean, action_std, git_remote, git_commit, output_dir_str, hf_token,
        )
        try:
            state_path = state_handle.get(timeout=14400)
            action_path = action_handle.get(timeout=14400)
        except Exception as exc:
            print(f"[{task_name}] embedding FAILED: {exc}")
            n_failures += 1
            continue
        print(f"[{task_name}] Phase 2 done: state={state_path}, action={action_path}")

        # Phase 3: KSG scoring
        print(f"[{task_name}] Phase 3: KSG scoring …")
        try:
            _, task_scores = _score_task_v2.remote(
                task_name, run_name, hydra_args,
                action_mean, action_std, output_dir_str,
                git_remote, git_commit, hf_token,
            )
            scores_by_task[task_name] = task_scores
        except Exception as exc:
            print(f"[{task_name}] KSG FAILED: {exc}")
            n_failures += 1
            continue

        print(
            f"[{task_name}] complete — {len(task_scores)} scores "
            f"in {_time.perf_counter() - t_task:.1f}s"
        )

    elapsed = _time.time() - t0

    # ── 6. Aggregate + save ───────────────────────────────────────────────────
    flat_scores: dict[str, float] = {}
    for t_scores in scores_by_task.values():
        flat_scores.update(t_scores)

    all_vals = _np.array([s for s in flat_scores.values() if _np.isfinite(s)])

    per_task_stats: dict[str, dict] = {}
    for t_name, t_scores in scores_by_task.items():
        vals = _np.array([s for s in t_scores.values() if _np.isfinite(s)])
        per_task_stats[t_name] = {
            "count":     len(t_scores),
            "mi_mean":   float(_np.nanmean(vals))   if len(vals) else float("nan"),
            "mi_std":    float(_np.nanstd(vals))    if len(vals) else float("nan"),
            "mi_median": float(_np.nanmedian(vals)) if len(vals) else float("nan"),
            "mi_min":    float(_np.nanmin(vals))    if len(vals) else float("nan"),
            "mi_max":    float(_np.nanmax(vals))    if len(vals) else float("nan"),
        }

    def _sort_scores(d: dict) -> dict:
        return dict(sorted(d.items(), key=lambda kv: kv[1] if _np.isfinite(kv[1]) else float("-inf"), reverse=True))

    sorted_flat = _sort_scores(flat_scores)
    sorted_by_task = {t: _sort_scores(s) for t, s in scores_by_task.items()}

    with open(output_dir / "scores.json", "w") as f:
        json.dump(sorted_flat, f, indent=2)
    with open(output_dir / "scores_by_task.json", "w") as f:
        json.dump(sorted_by_task, f, indent=2)
    with open(output_dir / "kept_hashes.json", "w") as f:
        json.dump(list(sorted_flat.keys()), f, indent=2)
    with open(output_dir / "curation_stats_v2.json", "w") as f:
        json.dump({
            "total_input":     total_episodes,
            "n_tasks":         len(scores_by_task),
            "n_task_failures": n_failures,
            "scored":          len(flat_scores),
            "elapsed_seconds": round(elapsed, 1),
            "mi_mean":   float(all_vals.mean())       if len(all_vals) else float("nan"),
            "mi_std":    float(all_vals.std())        if len(all_vals) else float("nan"),
            "mi_median": float(_np.median(all_vals))  if len(all_vals) else float("nan"),
            "per_task":  per_task_stats,
        }, f, indent=2)

    zarr_volume.commit()
    training_outputs_volume.commit()

    print(
        f"\nDemInf v2 curation done — scored={len(flat_scores)} episodes "
        f"across {len(scores_by_task)}/{len(by_task)} tasks "
        f"in {elapsed:.1f}s\nOutput: {output_dir_str}"
    )
    return output_dir_str


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def submit_curate_v2(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a DemInf v2 curation job."""
    hydra_args, _ = pop_init_submodules(hydra_args)
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo has uncommitted changes.")

    # Derive run_name from hydra args (name= key)
    run_name = "deminf_v2"
    for arg in hydra_args:
        key, sep, val = arg.lstrip("+").partition("=")
        if sep and key == "name":
            run_name = val.strip()
            break

    print(f"Submitting DemInf v2 curation: run_name={run_name} at commit {git_commit[:12]}")
    handle = run_curate_v2.spawn(
        tuple(hydra_args), git_remote, git_commit, run_name,
        hf_token=_local_hf_token(),
    )
    _env = os.environ.get("MODAL_ENVIRONMENT", "robotics")
    _app = os.environ.get("MODAL_APP_NAME", "egomimic-training")
    print(f"Submitted: {handle.object_id}")
    print(f"Monitor: https://modal.com/apps/mecka/{_env}/apps/{_app}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    modal_env = os.environ.copy()
    hydra_args: list[str] = []
    _MODAL_FLAGS = {"--detach", "--env"}
    for arg in sys.argv[1:]:
        if arg in _MODAL_FLAGS:
            continue
        key, sep, val = arg.lstrip("+").partition("=")
        if sep and key in MODAL_COMPUTE_ARG_MAP:
            modal_env[MODAL_COMPUTE_ARG_MAP[key]] = val
        else:
            hydra_args.append(arg)

    modal_env["MODAL_APP_NAME"] = app_name_from_hydra_args(hydra_args)
    print(f"DemInf v2 — app: {modal_env['MODAL_APP_NAME']}")
    launch_detached(Path(__file__).resolve(), "submit_curate_v2", hydra_args, modal_env)
