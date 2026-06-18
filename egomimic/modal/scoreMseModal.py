"""Modal entrypoint: per-episode action-MSE scoring with a pretrained pi checkpoint.

Architecture (mirrors curateModal.py, one fan-out level)
--------------------------------------------------------
run_score_mse (orchestrator, CPU)
  • SQL task lookup partitions resolved episode hashes by task.
  • Shards each task into chunks of ≤ max_episodes_per_shard.
  • Fans out one _score_shard GPU container per shard (all in parallel).
  • Collects per-shard {hash: mse}, merges by task, writes the score artifacts.

_score_shard (GPU worker, one per shard)
  • Builds the PI algo once (pretrained base + optional fine-tuned ckpt).
  • Resolves its shard's episodes via ModalEpisodeResolver(allowed_episode_ids=…).
  • Scores each episode (forward_eval diffusion → unnormalized paired MSE).

Outputs (on egoverse-training-outputs at mse_scores/<name>/<desc>_<ts>/):
  • mse_scores.json   {task: [[episode_hash, mse], ...]}  (ascending; lower = better)
  • scores_meta.json  {source, metric, higher_is_worse: false, ...}  (viewer reads this)
  • mse_stats.json    run summary (per-task + global stats, settings, elapsed)

Usage
-----
    python egomimic/modal/scoreMseModal.py \\
        name=my_run description=mse \\
        model.robomimic_model.config.pytorch_weight_path=pi_checkpoints/pi05_base_pytorch \\
        norm_stats.precomputed_norm_path=precomputed_norm_stats/mecka_all_zarr \\
        +modal_gpu=L40S:1 +modal_cpu=24 +modal_memory_gb=128

    # score a fine-tuned run instead of the base, on a subset, cheaply:
    python egomimic/modal/scoreMseModal.py \\
        name=my_run description=mse ckpt_path=<run>/checkpoints/last.ckpt \\
        model.robomimic_model.config.pytorch_weight_path=pi_checkpoints/pi05_base_pytorch \\
        norm_stats.precomputed_norm_path=precomputed_norm_stats/mecka_all_zarr \\
        data.train_datasets.mecka_bimanual.resolver.debug=20 score.every_n=10
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import modal

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modal_setup import (  # noqa: E402
    CFG,
    MODAL_COMPUTE_ARG_MAP,
    ModalCompute,
    _local_hf_token,
    _prepare_repo,
    _resolve_git_state,
    app,
    app_name_from_hydra_args,
    launch_detached,
    pop_init_submodules,
    training_outputs_volume,
    zarr_volume,
)

# GPU shard workers — override at launch via +modal_gpu / +modal_cpu / +modal_memory_gb.
SHARD_COMPUTE = ModalCompute.from_environ(
    default_gpu="L40S",
    default_cpu=24.0,
    default_memory_mb=131072,
)

# Orchestrator: SQL + resolve + spawn only (fixed lightweight CPU).
ORCHESTRATOR = ModalCompute(gpu=None, cpu=4.0, memory_mb=16384)

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]

# Path-override keys rewritten from repo-relative to the outputs-volume mount —
# same set as trainModal._resolve_volume_paths (the YAML defaults for these are
# dead SLURM absolute paths; see CLAUDE.md / the pytorch_weight_path gotcha).
_PATH_KEYS = {
    "ckpt_path",
    "norm_stats.precomputed_norm_path",
    "model.robomimic_model.config.paligemma_weight_path",
    "model.robomimic_model.config.pytorch_weight_path",
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_volume_paths(hydra_args: tuple[str, ...]) -> tuple[str, ...]:
    """Rewrite relative path overrides to absolute container (outputs-volume) paths."""
    fixed = []
    for arg in hydra_args:
        key, sep, val = arg.partition("=")
        if (
            sep
            and key in _PATH_KEYS
            and val
            and val != "null"
            and not val.startswith("/")
        ):
            arg = f"{key}={CFG.output_mount_path}/{val}"
        fixed.append(arg)
    return tuple(fixed)


def _boot_container(git_remote: str, git_commit: str, hf_token: str) -> None:
    """Clone the repo (with openpi + the pi transformers overlay) and set env.

    Uses the FULL _prepare_repo (init_submodules=True) — PI imports openpi, whose
    pi0.5 pytorch model requires the transformers==4.53.2 overlay. openpi becomes
    importable in-process via the .pth _prepare_repo writes.
    """
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo(git_remote=git_remote, git_commit=git_commit, init_submodules=True)
    sys.path.insert(0, CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")
    # PI.forward_eval's sampler is @torch.compile; the image ships no C compiler,
    # so force eager (matches trainModal's pi handling).
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _openpi_src = f"{CFG.remote_repo_dir}/external/openpi/src"
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{_openpi_src}:{existing}" if existing else _openpi_src


def _load_cfg(hydra_args: tuple[str, ...]):
    """Compose the score_mse Hydra config inside an already-booted container."""
    import hydra as _hydra

    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        return _hydra.compose("score_mse", overrides=list(hydra_args))


def _resolve_episode_map(cfg, hydra, allowed_hashes: list[str] | None = None) -> dict:
    """Instantiate the data config's resolvers; return {episode_hash: ZarrDataset}.

    When ``allowed_hashes`` is given, restrict every resolver to exactly those
    hashes (ModalEpisodeResolver.allowed_episode_ids) — used to scope a shard.
    """
    from omegaconf import open_dict

    episodes: dict = {}
    for ds_name, ds_cfg in cfg.data.train_datasets.items():
        resolver_cfg = copy.deepcopy(ds_cfg.resolver)
        if allowed_hashes is not None:
            with open_dict(resolver_cfg):
                resolver_cfg.allowed_episode_ids = list(allowed_hashes)
        resolver = hydra.utils.instantiate(resolver_cfg)
        filters = (
            hydra.utils.instantiate(ds_cfg.filters) if "filters" in ds_cfg else None
        )
        resolved = resolver.resolve(filters=filters)
        episodes.update(resolved)
        print(f"  [{ds_name}] {len(resolved)} episodes resolved")
    return episodes


def _embodiment_name(cfg) -> str:
    from omegaconf import OmegaConf

    name = OmegaConf.select(cfg, "score.embodiment_name", default=None)
    return name or next(iter(cfg.data.train_datasets))


# ---------------------------------------------------------------------------
# GPU shard worker
# ---------------------------------------------------------------------------


@app.function(
    gpu=SHARD_COMPUTE.gpu,
    cpu=SHARD_COMPUTE.cpu,
    memory=SHARD_COMPUTE.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def _score_shard(
    task_name: str,
    shard_idx: int,
    shard_hashes: list[str],
    git_remote: str,
    git_commit: str,
    hydra_args: tuple[str, ...],
    hf_token: str = "",
) -> tuple[str, dict]:
    """Score one shard of episodes on a GPU. Returns (task_name, {hash: {mse, n_frames}})."""
    import time as _time

    import torch
    from omegaconf import OmegaConf

    t0 = _time.perf_counter()
    _boot_container(git_remote, git_commit, hf_token)
    import hydra as _hydra

    from egomimic.eval.mse_episode_scorer import (
        build_collate_fn,
        build_data_schematic,
        build_pi_algo,
        register_omegaconf_resolvers,
        score_episode_map,
    )
    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()
    register_omegaconf_resolvers()

    cfg = _load_cfg(_resolve_volume_paths(hydra_args))
    tag = f"[{task_name}][shard {shard_idx}]"
    if not shard_hashes:
        print(f"{tag} empty shard — skipping")
        return task_name, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embodiment_name = _embodiment_name(cfg)
    precomputed = OmegaConf.select(
        cfg, "norm_stats.precomputed_norm_path", default=None
    )
    ckpt_path = OmegaConf.select(cfg, "ckpt_path", default=None)

    print(f"{tag} {len(shard_hashes)} episodes — building model on {device}")
    data_schematic = build_data_schematic(
        cfg, embodiment_name, precomputed_norm_path=precomputed
    )
    algo = build_pi_algo(cfg, data_schematic, device=device, ckpt_path=ckpt_path)
    collate_fn = build_collate_fn(cfg)

    episodes = _resolve_episode_map(cfg, _hydra, allowed_hashes=shard_hashes)
    if not episodes:
        print(f"{tag} no episodes resolved for shard hashes — skipping")
        return task_name, {}

    scores = score_episode_map(
        algo,
        episodes,
        embodiment_name,
        collate_fn,
        batch_size=int(OmegaConf.select(cfg, "score.batch_size", default=16)),
        every_n=int(OmegaConf.select(cfg, "score.every_n", default=1)),
        max_frames=OmegaConf.select(cfg, "score.max_frames", default=None),
        device=device,
        autocast_dtype=OmegaConf.select(
            cfg, "score.autocast_dtype", default="bfloat16"
        ),
        progress=tag,
    )
    n_frames = sum(v["n_frames"] for v in scores.values())
    print(
        f"{tag} scored {len(scores)}/{len(episodes)} episodes, {n_frames} frames "
        f"in {_time.perf_counter() - t0:.1f}s"
    )
    return task_name, scores


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@app.function(
    gpu=ORCHESTRATOR.gpu,
    cpu=ORCHESTRATOR.cpu,
    memory=ORCHESTRATOR.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_score_mse(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    init_submodules: bool = True,
    hf_token: str = "",
) -> str:
    """Orchestrator: SQL task grouping + per-shard GPU fan-out + artifact write."""
    import time as _time

    import numpy as _np
    from omegaconf import OmegaConf

    _boot_container(git_remote, git_commit, hf_token)
    import hydra as _hydra

    from egomimic.eval.mse_episode_scorer import register_omegaconf_resolvers
    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()
    register_omegaconf_resolvers()
    cfg = _load_cfg(_resolve_volume_paths(hydra_args))

    # ── 1. SQL task lookup: episode_hash → task_name ──────────────────────────
    print("Running SQL task lookup …")
    from egomimic.utils.aws.aws_sql import create_default_engine, episode_table_to_df

    full_df = episode_table_to_df(create_default_engine())
    if "is_deleted" in full_df.columns:
        full_df = full_df[full_df["is_deleted"] != True]  # noqa: E712
    hash_to_task: dict[str, str] = {}
    if "task" in full_df.columns:
        hash_to_task = dict(
            zip(full_df["episode_hash"], full_df["task"].fillna("unknown"))
        )

    # ── 2. Resolve episodes + partition by task ───────────────────────────────
    episodes = _resolve_episode_map(cfg, _hydra, allowed_hashes=None)
    by_task: dict[str, list[str]] = {}
    for episode_hash in episodes:
        task = hash_to_task.get(episode_hash) or "unknown"
        if str(task) in ("nan", "None", ""):
            task = "unknown"
        by_task.setdefault(task, []).append(episode_hash)

    total_episodes = sum(len(v) for v in by_task.values())
    max_per_shard = int(OmegaConf.select(cfg, "max_episodes_per_shard", default=200))
    if total_episodes == 0:
        print("No episodes resolved — check the data config resolver/filters.")
        return ""
    print(
        f"Resolved {total_episodes} episodes across {len(by_task)} task(s); "
        f"sharding at max_per_shard={max_per_shard}"
    )

    # ── 3. Output dir ─────────────────────────────────────────────────────────
    timestamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(CFG.output_mount_path)
        / "mse_scores"
        / cfg.name
        / f"{cfg.description}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. Fan out one GPU shard worker per (task, shard) ─────────────────────
    spawn_args = []
    for task_name, hashes in sorted(by_task.items()):
        ordered = sorted(hashes)
        for shard_idx, i in enumerate(range(0, len(ordered), max_per_shard)):
            spawn_args.append((task_name, shard_idx, ordered[i : i + max_per_shard]))

    print(f"Spawning {len(spawn_args)} GPU shard worker(s) …")
    t0 = _time.time()
    handles = [
        (
            task_name,
            _score_shard.spawn(
                task_name,
                shard_idx,
                shard_hashes,
                git_remote,
                git_commit,
                hydra_args,
                hf_token,
            ),
        )
        for task_name, shard_idx, shard_hashes in spawn_args
    ]

    scores_by_task: dict[str, dict] = {}
    n_failures = 0
    for task_name, handle in handles:
        try:
            _label, shard_scores = handle.get(timeout=CFG.timeout_seconds)
            scores_by_task.setdefault(task_name, {}).update(shard_scores)
        except Exception as exc:
            n_failures += 1
            print(f"[shard] FAILED ({task_name}): {exc}")
    elapsed = _time.time() - t0
    print(
        f"Scoring done in {elapsed:.1f}s — {len(spawn_args) - n_failures}/"
        f"{len(spawn_args)} shards succeeded ({n_failures} failed)"
    )

    # ── 5. Aggregate + write artifacts ────────────────────────────────────────
    # mse_scores.json: {task: [[hash, mse], ...]} sorted ascending (lower = better).
    mse_scores: dict[str, list] = {}
    per_task_stats: dict[str, dict] = {}
    all_vals: list[float] = []
    total_scored = 0
    total_frames = 0
    for task_name, ep_scores in scores_by_task.items():
        pairs = sorted(
            ((h, d["mse"]) for h, d in ep_scores.items()), key=lambda kv: kv[1]
        )
        mse_scores[task_name] = [[h, m] for h, m in pairs]
        vals = _np.array([m for _, m in pairs], dtype=float)
        all_vals.extend(vals.tolist())
        total_scored += len(ep_scores)
        total_frames += sum(d["n_frames"] for d in ep_scores.values())
        per_task_stats[task_name] = {
            "count": len(ep_scores),
            "mse_mean": float(vals.mean()) if len(vals) else float("nan"),
            "mse_median": float(_np.median(vals)) if len(vals) else float("nan"),
            "mse_min": float(vals.min()) if len(vals) else float("nan"),
            "mse_max": float(vals.max()) if len(vals) else float("nan"),
        }

    source = (
        OmegaConf.select(cfg, "ckpt_path", default=None)
        or OmegaConf.select(
            cfg, "model.robomimic_model.config.pytorch_weight_path", default=None
        )
        or "pretrained_base"
    )
    scores_meta = {
        "source": str(source),
        "metric": "paired_mse_unnorm",
        "higher_is_worse": False,
        "every_n": int(OmegaConf.select(cfg, "score.every_n", default=1)),
        "max_frames": OmegaConf.select(cfg, "score.max_frames", default=None),
        "num_sampling_steps": OmegaConf.select(
            cfg, "model.robomimic_model.config.num_sampling_steps", default=10
        ),
    }
    arr = _np.array(all_vals, dtype=float)
    stats = {
        "total_resolved": total_episodes,
        "scored": total_scored,
        "n_tasks": len(scores_by_task),
        "n_shards": len(spawn_args),
        "n_shard_failures": n_failures,
        "total_frames_scored": total_frames,
        "max_episodes_per_shard": max_per_shard,
        "elapsed_seconds": round(elapsed, 1),
        "mse_mean": float(arr.mean()) if len(arr) else float("nan"),
        "mse_median": float(_np.median(arr)) if len(arr) else float("nan"),
        "mse_min": float(arr.min()) if len(arr) else float("nan"),
        "mse_max": float(arr.max()) if len(arr) else float("nan"),
        "per_task": per_task_stats,
    }

    (output_dir / "mse_scores.json").write_text(json.dumps(mse_scores, indent=2))
    (output_dir / "scores_meta.json").write_text(json.dumps(scores_meta, indent=2))
    (output_dir / "mse_stats.json").write_text(json.dumps(stats, indent=2))
    # episode_hashes.json: full resolved universe (for the viewer's keep-all-uncovered).
    (output_dir / "episode_hashes.json").write_text(json.dumps(sorted(episodes.keys())))
    training_outputs_volume.commit()

    rel = os.path.relpath(output_dir, CFG.output_mount_path)
    print(
        f"MSE scoring done — scored={total_scored}/{total_episodes}  "
        f"tasks={len(scores_by_task)}  output={rel}"
    )
    print(
        "Build the viewer:\n"
        f"  python egomimic/scripts/build_mse_viewer.py {rel}/mse_scores.json "
        f"--volume egoverse-training-outputs --out mse_viewer.html"
    )
    return rel


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def submit_score_mse(*hydra_args: str) -> None:
    """Fire-and-forget: spawn an MSE-scoring job from an already-pushed commit."""
    hydra_args, init_submodules = pop_init_submodules(hydra_args)
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Submitting MSE scoring at commit {git_commit[:12]} from {git_remote}")
    handle = run_score_mse.spawn(
        tuple(hydra_args),
        git_remote,
        git_commit,
        init_submodules=init_submodules,
        hf_token=_local_hf_token(),
    )
    _env = os.environ.get("MODAL_ENVIRONMENT", "robotics")
    _app = os.environ.get("MODAL_APP_NAME", "egomimic-training")
    print(f"Submitted Modal MSE-scoring job: {handle.object_id}")
    print(f"Monitor: https://modal.com/apps/mecka/{_env}/apps/{_app}")


if __name__ == "__main__":
    modal_env = os.environ.copy()
    hydra_args: list[str] = []
    for arg in sys.argv[1:]:
        key, sep, val = arg.lstrip("+").partition("=")
        if sep and key in MODAL_COMPUTE_ARG_MAP:
            modal_env[MODAL_COMPUTE_ARG_MAP[key]] = val
        else:
            hydra_args.append(arg)

    modal_env["MODAL_APP_NAME"] = app_name_from_hydra_args(hydra_args)

    shard_compute = ModalCompute.from_mapping(
        modal_env, default_gpu="L40S", default_cpu=24.0, default_memory_mb=131072
    )
    print(f"Modal app:                 {modal_env['MODAL_APP_NAME']}")
    print(f"Modal orchestrator (fixed): {ORCHESTRATOR.summary()}")
    print(f"Modal shard GPU worker:     {shard_compute.summary()}")

    launch_detached(Path(__file__).resolve(), "submit_score_mse", hydra_args, modal_env)
