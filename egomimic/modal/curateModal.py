"""Modal curation entrypoints for DemInf scoring at scale.

Architecture
------------
run_curate (orchestrator, CPU)
  • SQL task lookup partitions episode hashes by task.
  • Fans out one _score_task_split container per task.

_score_task_split (CPU worker, one per task)
  • Reads precomputed action norm stats (required — raises if missing).
  • Shards episode hashes into chunks of ≤ max_episodes_per_shard (default 100).
  • Spawns one _embed_task_shard GPU container per shard (all in parallel).
  • Collects shard latents, concatenates in shard-index order.
  • Pass 2 (KSG): mutual-information scoring on combined latents.

_embed_task_shard (GPU worker, one per shard of ≤ max_episodes_per_shard episodes)
  • Receives pre-computed action_mean/std from _score_task_split.
  • Injects allowed_episode_ids = shard_hashes into every resolver.
  • Pass 1 (embed): parallel bulk load + GPU embed → returns (state, action) latents.

Norm stats are always precomputed offline (model.precomputed_norm_stats in
deminf_default.yaml). Sharding is controlled by max_episodes_per_shard in
curate.yaml. Per-shard GPU compute is overridden at launch via
+modal_gpu / +modal_cpu / +modal_memory_gb.
Architecture
------------
run_curate (orchestrator, CPU)
  • SQL task lookup partitions episode hashes by task.
  • Fans out one _score_task_split container per task.

_score_task_split (CPU worker, one per task)
  • Reads precomputed action norm stats (required — raises if missing).
  • Shards episode hashes into chunks of ≤ max_episodes_per_shard (default 100).
  • Spawns one _embed_task_shard GPU container per shard (all in parallel).
  • Collects shard latents, concatenates in shard-index order.
  • Pass 2 (KSG): mutual-information scoring on combined latents.

_embed_task_shard (GPU worker, one per shard of ≤ max_episodes_per_shard episodes)
  • Receives pre-computed action_mean/std from _score_task_split.
  • Injects allowed_episode_ids = shard_hashes into every resolver.
  • Pass 1 (embed): parallel bulk load + GPU embed → returns (state, action) latents.

Norm stats are always precomputed offline (model.precomputed_norm_stats in
deminf_default.yaml). Sharding is controlled by max_episodes_per_shard in
curate.yaml. Per-shard GPU compute is overridden at launch via
+modal_gpu / +modal_cpu / +modal_memory_gb.

Usage
-----
    python egomimic/modal/curateModal.py \\
        name=my_run description=test

    python egomimic/modal/curateModal.py \\
        name=my_run description=test init_submodules=false \\
        +modal_gpu=L40S:1 +modal_cpu=32 +modal_memory_gb=128

    python egomimic/modal/curateModal.py \\
        name=my_run description=test

    python egomimic/modal/curateModal.py \\
        name=my_run description=test init_submodules=false \\
        +modal_gpu=L40S:1 +modal_cpu=32 +modal_memory_gb=128

    modal run --env robotics egomimic/modal/curateModal.py::submit_curate -- \\
        name=my_run description=test
"""

from __future__ import annotations

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
    CURATE_ORCHESTRATOR,
    MODAL_COMPUTE_ARG_MAP,
    ModalCompute,
    _local_hf_token,
    CURATE_ORCHESTRATOR,
    MODAL_COMPUTE_ARG_MAP,
    ModalCompute,
    _local_hf_token,
    _prepare_repo,
    _prepare_repo_light,
    _resolve_git_state,
    launch_detached,
    pop_init_submodules,
    app,
    training_outputs_volume,
    zarr_volume,
)

# GPU embed-shard workers — override at launch via +modal_gpu / +modal_cpu / +modal_memory_gb.
TASK_COMPUTE = ModalCompute.from_environ(
    default_gpu="L40S",
    default_cpu=16.0,
    default_memory_mb=131072,
)

# Per-task CPU orchestrator (shard fan-out + KSG) — fixed, no GPU.
TASK_SCORE_COMPUTE = ModalCompute(gpu=None, cpu=16.0, memory_mb=32768)

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]
# GPU embed-shard workers — override at launch via +modal_gpu / +modal_cpu / +modal_memory_gb.
TASK_COMPUTE = ModalCompute.from_environ(
    default_gpu="L40S",
    default_cpu=16.0,
    default_memory_mb=131072,
)

# Per-task CPU orchestrator (shard fan-out + KSG) — fixed, no GPU.
TASK_SCORE_COMPUTE = ModalCompute(gpu=None, cpu=16.0, memory_mb=32768)

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]


# ---------------------------------------------------------------------------
# Shared container boot helpers
# Shared container boot helpers
# ---------------------------------------------------------------------------


def _boot_container(git_remote: str, git_commit: str, hf_token: str) -> None:
    """Set up the remote container: env vars, repo clone, sys.path."""
def _boot_container(git_remote: str, git_commit: str, hf_token: str) -> None:
    """Set up the remote container: env vars, repo clone, sys.path."""
    import sys as _sys

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo_light(git_remote=git_remote, git_commit=git_commit)
    _sys.path.insert(0, CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")


def _load_cfg(hydra_args: tuple[str, ...]):
    """Compose the curate Hydra config inside an already-booted container."""
def _load_cfg(hydra_args: tuple[str, ...]):
    """Compose the curate Hydra config inside an already-booted container."""
    import hydra as _hydra

    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        return _hydra.compose("curate", overrides=list(hydra_args))
        return _hydra.compose("curate", overrides=list(hydra_args))


def _inject_episode_ids(cfg, hashes: list[str]):
    """Deep-copy cfg and inject allowed_episode_ids into every resolver node."""
    import copy
    from omegaconf import open_dict

    task_cfg = copy.deepcopy(cfg)
    with open_dict(task_cfg):
        for ds_name in task_cfg.data.train_datasets:
            ds_node = task_cfg.data.train_datasets[ds_name]
            if "resolver" in ds_node:
                ds_node.resolver.allowed_episode_ids = list(hashes)
    return task_cfg


def _build_episode_map(task_cfg, hydra) -> dict:
    """Instantiate MultiDatasets from task_cfg; return {hash: ZarrDataset}."""
    episodes: dict = {}
    for ds_name in task_cfg.data.train_datasets:
        ds_node = task_cfg.data.train_datasets[ds_name]
        try:
            md = hydra.utils.instantiate(ds_node)
            episodes.update(md.datasets)
            print(f"  [{ds_name}] {len(md.datasets)} episodes loaded")
        except Exception as exc:
            print(f"  [{ds_name}] dataset build FAILED: {exc}")
    return episodes


# ---------------------------------------------------------------------------
# Pass-1 GPU embed shard (one per shard of ≤ max_episodes_per_shard episodes)
# ---------------------------------------------------------------------------


@app.function(
    gpu=TASK_COMPUTE.gpu,
    cpu=TASK_COMPUTE.cpu,
    memory=TASK_COMPUTE.memory_mb,
    timeout=86400,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def _embed_task_shard(
    task_name: str,
    shard_idx: int,
    shard_hashes: list[str],
    action_mean: "np.ndarray",
    action_std: "np.ndarray",
    git_remote: str,
    git_commit: str,
    hydra_args: tuple[str, ...],
    hf_token: str = "",
) -> tuple[int, list, list, list[str], list[int]]:
    """
    GPU embed worker for one shard of ≤ max_episodes_per_shard episodes.

    Loads shard_hashes from the zarr volume, builds embedders from the
    provided precomputed norm stats, and returns per-episode latents.

    Returns
    -------
    (shard_idx, state_latents, action_latents, hashes, ep_lengths)
    where state_latents / action_latents are lists of per-episode (T, D) arrays.
    """
    import numpy as _np
    import torch
    from tqdm import tqdm

    import time as _time
    t_shard_start = _time.perf_counter()

    _boot_container(git_remote, git_commit, hf_token)
    import hydra as _hydra
    from egomimic.utils.aws.aws_data_utils import load_env
    load_env()

    cfg = _load_cfg(hydra_args)
    tag = f"[{task_name}][shard {shard_idx}]"
    print(f"{tag} {len(shard_hashes)} episodes — embedding")

    t_map = _time.perf_counter()
    task_cfg = _inject_episode_ids(cfg, shard_hashes)
    all_episodes = _build_episode_map(task_cfg, _hydra)
    print(f"{tag} Episode map built: {len(all_episodes)} episodes in {_time.perf_counter() - t_map:.2f}s")

    if not all_episodes:
        print(f"{tag} No episodes loaded — returning empty shard")
        return shard_idx, [], [], [], []

    from egomimic.curation.config import (
        apply_curation_seed,
        select_curation_loader,
        select_embedder_settings,
        select_seed,
        select_tensor_keys,
    )
    from egomimic.curation.episode_pipeline import build_embedders, run_pass2_embed_episodes

    apply_curation_seed(select_seed(cfg))
    embed_cfg = select_embedder_settings(cfg)
    device = torch.device(embed_cfg.device if torch.cuda.is_available() else "cpu")
    action_key, image_key = select_tensor_keys(cfg)
    ds_name = next(iter(task_cfg.data.train_datasets))
    loader_cfg = select_curation_loader(cfg, ds_name)

    t_embed_build = _time.perf_counter()
    action_embedder, state_embedder = build_embedders(
        embed_cfg,
        _np.asarray(action_mean, dtype=_np.float32),
        _np.asarray(action_std, dtype=_np.float32),
        device,
        select_seed(cfg),
        global_frame_batch_size=loader_cfg.global_frame_batch_size,
    )
    print(
        f"{tag} Embedders ready in {_time.perf_counter() - t_embed_build:.2f}s — "
        f"backbone={embed_cfg.state_image.backbone}, device={device}, "
        f"global_frame_batch={loader_cfg.global_frame_batch_size}"
    )

    t_pass2 = _time.perf_counter()
    state_latents, action_latents, hashes, ep_lengths = run_pass2_embed_episodes(
        all_episodes,
        set(all_episodes.keys()),
        action_key,
        image_key,
        loader_cfg,
        action_embedder,
        state_embedder,
        progress=f"{tag} embed",
    )
    n_frames = sum(ep_lengths)
    print(
        f"{tag} Pass2 done in {_time.perf_counter() - t_pass2:.2f}s — "
        f"{len(hashes)} episodes, {n_frames} frames"
    )
    print(f"{tag} Shard total: {_time.perf_counter() - t_shard_start:.2f}s")
    return shard_idx, state_latents, action_latents, hashes, ep_lengths


# ---------------------------------------------------------------------------
# Per-task CPU orchestrator: load norm stats → shard → fan-out → KSG
# ---------------------------------------------------------------------------


@app.function(
    gpu=TASK_SCORE_COMPUTE.gpu,
    cpu=TASK_SCORE_COMPUTE.cpu,
    memory=TASK_SCORE_COMPUTE.memory_mb,
    timeout=86400,
    secrets=_SHARED_SECRETS,
    volumes={CFG.output_mount_path: training_outputs_volume},
)
def _score_task_split(
    task_name: str,
    task_episode_hashes: list[str],
    git_remote: str,
    git_commit: str,
    hydra_args: tuple[str, ...],
    hf_token: str = "",
) -> tuple[str, dict]:
    """
    CPU orchestrator for one task: norm stats → shard GPU embed → KSG.

    Steps
    -----
    1. Read precomputed action norm stats (raises if not configured).
    2. Shard task_episode_hashes into chunks of ≤ max_episodes_per_shard.
    3. Spawn one _embed_task_shard GPU container per shard (all in parallel).
    4. Collect and concatenate latents from all shards (in shard-index order).
    5. KSG mutual-information scoring on combined latents (CPU).
    """
    import time as _time
    import numpy as _np
    from omegaconf import OmegaConf

    t_task_start = _time.perf_counter()
    _boot_container(git_remote, git_commit, hf_token)
    import hydra as _hydra
    from egomimic.utils.aws.aws_data_utils import load_env
    load_env()

    cfg = _load_cfg(hydra_args)
    tag = f"[{task_name}]"

    if not task_episode_hashes:
        print(f"{tag} No episodes — skipping")
        return task_name, {}

    max_per_shard = int(OmegaConf.select(cfg, "max_episodes_per_shard", default=100))
    print(f"{tag} {len(task_episode_hashes)} episodes, max_per_shard={max_per_shard}")

    from egomimic.curation.config import (
        apply_curation_seed,
        load_action_norm_stats,
        select_embedder_settings,
        select_seed,
        select_tensor_keys,
    )

    apply_curation_seed(select_seed(cfg))
    embed_cfg = select_embedder_settings(cfg)
    action_key, _ = select_tensor_keys(cfg)

    # ── Precomputed norm stats (required) ─────────────────────────────────────
    t_norm = _time.perf_counter()
    action_mean, action_std = load_action_norm_stats(
        cfg,
        action_key=action_key,
        norm_min_std=embed_cfg.norm_min_std,
        search_roots=[CFG.output_mount_path, CFG.remote_repo_dir, Path.cwd()],
    )
    print(
        f"{tag} Norm stats loaded in {_time.perf_counter() - t_norm:.2f}s — "
        f"action_key={action_key}, dim={action_mean.shape}"
    )

    # ── Shard episodes ────────────────────────────────────────────────────────
    episode_list = sorted(task_episode_hashes)
    shards = [
        episode_list[i : i + max_per_shard]
        for i in range(0, len(episode_list), max_per_shard)
    ]
    print(f"{tag} {len(episode_list)} episodes → {len(shards)} shard(s)")

    # ── Spawn GPU embed shards in parallel ────────────────────────────────────
    t_spawn = _time.perf_counter()
    handles = [
        (
            shard_idx,
            _embed_task_shard.spawn(
                task_name,
                shard_idx,
                shard_hashes,
                action_mean,
                action_std,
                git_remote,
                git_commit,
                hydra_args,
                hf_token,
            ),
        )
        for shard_idx, shard_hashes in enumerate(shards)
    ]
    print(f"{tag} {len(handles)} embed shard(s) spawned in {_time.perf_counter() - t_spawn:.2f}s — collecting …")

    # ── Collect shard results ordered by shard_idx ────────────────────────────
    t_collect = _time.perf_counter()
    results_by_idx: dict[int, tuple] = {}
    n_shard_failures = 0
    for shard_idx, handle in handles:
        t_shard = _time.perf_counter()
        try:
            idx, s_lats, a_lats, hashes, lengths = handle.get(
                timeout=CFG.timeout_seconds
            )
            results_by_idx[idx] = (s_lats, a_lats, hashes, lengths)
            print(
                f"{tag}[shard {shard_idx}] collected: {len(hashes)} episodes, "
                f"{sum(lengths)} frames in {_time.perf_counter() - t_shard:.2f}s"
            )
        except Exception as exc:
            n_shard_failures += 1
            print(f"{tag}[shard {shard_idx}] FAILED after {_time.perf_counter() - t_shard:.2f}s: {exc}")

    print(f"{tag} All shards collected in {_time.perf_counter() - t_collect:.2f}s ({n_shard_failures} failure(s))")

    state_latents: list = []
    action_latents: list = []
    scored_hashes: list[str] = []
    ep_lengths: list[int] = []
    for idx in sorted(results_by_idx):
        s_lats, a_lats, hashes, lengths = results_by_idx[idx]
        state_latents.extend(s_lats)
        action_latents.extend(a_lats)
        scored_hashes.extend(hashes)
        ep_lengths.extend(lengths)

    if not state_latents:
        print(
            f"{tag} No latents collected ({n_shard_failures} shard failure(s)) "
            "— returning empty"
        )

def _inject_episode_ids(cfg, hashes: list[str]):
    """Deep-copy cfg and inject allowed_episode_ids into every resolver node."""
    import copy
    from omegaconf import open_dict

    task_cfg = copy.deepcopy(cfg)
    with open_dict(task_cfg):
        for ds_name in task_cfg.data.train_datasets:
            ds_node = task_cfg.data.train_datasets[ds_name]
            if "resolver" in ds_node:
                ds_node.resolver.allowed_episode_ids = list(hashes)
    return task_cfg


def _build_episode_map(task_cfg, hydra) -> dict:
    """Instantiate MultiDatasets from task_cfg; return {hash: ZarrDataset}."""
    episodes: dict = {}
    for ds_name in task_cfg.data.train_datasets:
        ds_node = task_cfg.data.train_datasets[ds_name]
        try:
            md = hydra.utils.instantiate(ds_node)
            episodes.update(md.datasets)
            print(f"  [{ds_name}] {len(md.datasets)} episodes loaded")
        except Exception as exc:
            print(f"  [{ds_name}] dataset build FAILED: {exc}")
    return episodes


# ---------------------------------------------------------------------------
# Pass-1 GPU embed shard (one per shard of ≤ max_episodes_per_shard episodes)
# ---------------------------------------------------------------------------


@app.function(
    gpu=TASK_COMPUTE.gpu,
    cpu=TASK_COMPUTE.cpu,
    memory=TASK_COMPUTE.memory_mb,
    timeout=86400,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def _embed_task_shard(
    task_name: str,
    shard_idx: int,
    shard_hashes: list[str],
    action_mean: "np.ndarray",
    action_std: "np.ndarray",
    git_remote: str,
    git_commit: str,
    hydra_args: tuple[str, ...],
    hf_token: str = "",
) -> tuple[int, list, list, list[str], list[int]]:
    """
    GPU embed worker for one shard of ≤ max_episodes_per_shard episodes.

    Loads shard_hashes from the zarr volume, builds embedders from the
    provided precomputed norm stats, and returns per-episode latents.

    Returns
    -------
    (shard_idx, state_latents, action_latents, hashes, ep_lengths)
    where state_latents / action_latents are lists of per-episode (T, D) arrays.
    """
    import numpy as _np
    import torch
    from tqdm import tqdm

    import time as _time
    t_shard_start = _time.perf_counter()

    _boot_container(git_remote, git_commit, hf_token)
    import hydra as _hydra
    from egomimic.utils.aws.aws_data_utils import load_env
    load_env()

    cfg = _load_cfg(hydra_args)
    tag = f"[{task_name}][shard {shard_idx}]"
    print(f"{tag} {len(shard_hashes)} episodes — embedding")

    t_map = _time.perf_counter()
    task_cfg = _inject_episode_ids(cfg, shard_hashes)
    all_episodes = _build_episode_map(task_cfg, _hydra)
    print(f"{tag} Episode map built: {len(all_episodes)} episodes in {_time.perf_counter() - t_map:.2f}s")

    if not all_episodes:
        print(f"{tag} No episodes loaded — returning empty shard")
        return shard_idx, [], [], [], []

    from egomimic.curation.config import (
        apply_curation_seed,
        select_curation_loader,
        select_embedder_settings,
        select_seed,
        select_tensor_keys,
    )
    from egomimic.curation.episode_pipeline import build_embedders, run_pass2_embed_episodes

    apply_curation_seed(select_seed(cfg))
    embed_cfg = select_embedder_settings(cfg)
    device = torch.device(embed_cfg.device if torch.cuda.is_available() else "cpu")
    action_key, image_key = select_tensor_keys(cfg)
    ds_name = next(iter(task_cfg.data.train_datasets))
    loader_cfg = select_curation_loader(cfg, ds_name)

    t_embed_build = _time.perf_counter()
    action_embedder, state_embedder = build_embedders(
        embed_cfg,
        _np.asarray(action_mean, dtype=_np.float32),
        _np.asarray(action_std, dtype=_np.float32),
        device,
        select_seed(cfg),
        global_frame_batch_size=loader_cfg.global_frame_batch_size,
    )
    print(
        f"{tag} Embedders ready in {_time.perf_counter() - t_embed_build:.2f}s — "
        f"backbone={embed_cfg.state_image.backbone}, device={device}, "
        f"global_frame_batch={loader_cfg.global_frame_batch_size}"
    )

    t_pass2 = _time.perf_counter()
    state_latents, action_latents, hashes, ep_lengths = run_pass2_embed_episodes(
        all_episodes,
        set(all_episodes.keys()),
        action_key,
        image_key,
        loader_cfg,
        action_embedder,
        state_embedder,
        progress=f"{tag} embed",
    )
    n_frames = sum(ep_lengths)
    print(
        f"{tag} Pass2 done in {_time.perf_counter() - t_pass2:.2f}s — "
        f"{len(hashes)} episodes, {n_frames} frames"
    )
    print(f"{tag} Shard total: {_time.perf_counter() - t_shard_start:.2f}s")
    return shard_idx, state_latents, action_latents, hashes, ep_lengths


# ---------------------------------------------------------------------------
# Per-task CPU orchestrator: load norm stats → shard → fan-out → KSG
# ---------------------------------------------------------------------------


@app.function(
    gpu=TASK_SCORE_COMPUTE.gpu,
    cpu=TASK_SCORE_COMPUTE.cpu,
    memory=TASK_SCORE_COMPUTE.memory_mb,
    timeout=86400,
    secrets=_SHARED_SECRETS,
    volumes={CFG.output_mount_path: training_outputs_volume},
)
def _score_task_split(
    task_name: str,
    task_episode_hashes: list[str],
    git_remote: str,
    git_commit: str,
    hydra_args: tuple[str, ...],
    hf_token: str = "",
) -> tuple[str, dict]:
    """
    CPU orchestrator for one task: norm stats → shard GPU embed → KSG.

    Steps
    -----
    1. Read precomputed action norm stats (raises if not configured).
    2. Shard task_episode_hashes into chunks of ≤ max_episodes_per_shard.
    3. Spawn one _embed_task_shard GPU container per shard (all in parallel).
    4. Collect and concatenate latents from all shards (in shard-index order).
    5. KSG mutual-information scoring on combined latents (CPU).
    """
    import time as _time
    import numpy as _np
    from omegaconf import OmegaConf

    t_task_start = _time.perf_counter()
    _boot_container(git_remote, git_commit, hf_token)
    import hydra as _hydra
    from egomimic.utils.aws.aws_data_utils import load_env
    load_env()

    cfg = _load_cfg(hydra_args)
    tag = f"[{task_name}]"

    if not task_episode_hashes:
        print(f"{tag} No episodes — skipping")
        return task_name, {}

    max_per_shard = int(OmegaConf.select(cfg, "max_episodes_per_shard", default=100))
    print(f"{tag} {len(task_episode_hashes)} episodes, max_per_shard={max_per_shard}")

    from egomimic.curation.config import (
        apply_curation_seed,
        load_action_norm_stats,
        select_embedder_settings,
        select_seed,
        select_tensor_keys,
    )

    apply_curation_seed(select_seed(cfg))
    embed_cfg = select_embedder_settings(cfg)
    action_key, _ = select_tensor_keys(cfg)

    # ── Precomputed norm stats (required) ─────────────────────────────────────
    t_norm = _time.perf_counter()
    action_mean, action_std = load_action_norm_stats(
        cfg,
        action_key=action_key,
        norm_min_std=embed_cfg.norm_min_std,
        search_roots=[CFG.output_mount_path, CFG.remote_repo_dir, Path.cwd()],
    )
    print(
        f"{tag} Norm stats loaded in {_time.perf_counter() - t_norm:.2f}s — "
        f"action_key={action_key}, dim={action_mean.shape}"
    )

    # ── Shard episodes ────────────────────────────────────────────────────────
    episode_list = sorted(task_episode_hashes)
    shards = [
        episode_list[i : i + max_per_shard]
        for i in range(0, len(episode_list), max_per_shard)
    ]
    print(f"{tag} {len(episode_list)} episodes → {len(shards)} shard(s)")

    # ── Spawn GPU embed shards in parallel ────────────────────────────────────
    t_spawn = _time.perf_counter()
    handles = [
        (
            shard_idx,
            _embed_task_shard.spawn(
                task_name,
                shard_idx,
                shard_hashes,
                action_mean,
                action_std,
                git_remote,
                git_commit,
                hydra_args,
                hf_token,
            ),
        )
        for shard_idx, shard_hashes in enumerate(shards)
    ]
    print(f"{tag} {len(handles)} embed shard(s) spawned in {_time.perf_counter() - t_spawn:.2f}s — collecting …")

    # ── Collect shard results ordered by shard_idx ────────────────────────────
    t_collect = _time.perf_counter()
    results_by_idx: dict[int, tuple] = {}
    n_shard_failures = 0
    for shard_idx, handle in handles:
        t_shard = _time.perf_counter()
        try:
            idx, s_lats, a_lats, hashes, lengths = handle.get(
                timeout=CFG.timeout_seconds
            )
            results_by_idx[idx] = (s_lats, a_lats, hashes, lengths)
            print(
                f"{tag}[shard {shard_idx}] collected: {len(hashes)} episodes, "
                f"{sum(lengths)} frames in {_time.perf_counter() - t_shard:.2f}s"
            )
        except Exception as exc:
            n_shard_failures += 1
            print(f"{tag}[shard {shard_idx}] FAILED after {_time.perf_counter() - t_shard:.2f}s: {exc}")

    print(f"{tag} All shards collected in {_time.perf_counter() - t_collect:.2f}s ({n_shard_failures} failure(s))")

    state_latents: list = []
    action_latents: list = []
    scored_hashes: list[str] = []
    ep_lengths: list[int] = []
    for idx in sorted(results_by_idx):
        s_lats, a_lats, hashes, lengths = results_by_idx[idx]
        state_latents.extend(s_lats)
        action_latents.extend(a_lats)
        scored_hashes.extend(hashes)
        ep_lengths.extend(lengths)

    if not state_latents:
        print(
            f"{tag} No latents collected ({n_shard_failures} shard failure(s)) "
            "— returning empty"
        )
        return task_name, {}

    n_total = sum(ep_lengths)
    print(
        f"{tag} KSG input: {len(scored_hashes)} episodes, {n_total} timesteps "
        f"({n_shard_failures} shard failure(s))"
    )

    # ── KSG scoring on combined latents ───────────────────────────────────────
    from egomimic.curation.scoring import trajectory_scorer_from_cfg

    t_concat = _time.perf_counter()
    s_all = _np.concatenate(state_latents, axis=0)
    a_all = _np.concatenate(action_latents, axis=0)
    del state_latents, action_latents
    print(
        f"{tag} Latent concat: state={s_all.shape}, action={a_all.shape}, "
        f"{_time.perf_counter() - t_concat:.2f}s"
    )

    t_ksg = _time.perf_counter()
    scorer = trajectory_scorer_from_cfg(cfg)
    scores = scorer.score_latents(s_all, a_all, scored_hashes, ep_lengths)
    del s_all, a_all
    print(
        f"{tag} KSG done in {_time.perf_counter() - t_ksg:.2f}s — "
        f"scored {len(scores)} episodes ({n_total} timesteps)"
    )
    print(f"{tag} Task total: {_time.perf_counter() - t_task_start:.2f}s")
    n_total = sum(ep_lengths)
    print(
        f"{tag} KSG input: {len(scored_hashes)} episodes, {n_total} timesteps "
        f"({n_shard_failures} shard failure(s))"
    )

    # ── KSG scoring on combined latents ───────────────────────────────────────
    from egomimic.curation.scoring import trajectory_scorer_from_cfg

    t_concat = _time.perf_counter()
    s_all = _np.concatenate(state_latents, axis=0)
    a_all = _np.concatenate(action_latents, axis=0)
    del state_latents, action_latents
    print(
        f"{tag} Latent concat: state={s_all.shape}, action={a_all.shape}, "
        f"{_time.perf_counter() - t_concat:.2f}s"
    )

    t_ksg = _time.perf_counter()
    scorer = trajectory_scorer_from_cfg(cfg)
    scores = scorer.score_latents(s_all, a_all, scored_hashes, ep_lengths)
    del s_all, a_all
    print(
        f"{tag} KSG done in {_time.perf_counter() - t_ksg:.2f}s — "
        f"scored {len(scores)} episodes ({n_total} timesteps)"
    )
    print(f"{tag} Task total: {_time.perf_counter() - t_task_start:.2f}s")
    return task_name, scores


# ---------------------------------------------------------------------------
# Orchestrator: SQL task grouping + per-task fan-out
# Orchestrator: SQL task grouping + per-task fan-out
# ---------------------------------------------------------------------------


@app.function(
    gpu=CURATE_ORCHESTRATOR.gpu,
    cpu=CURATE_ORCHESTRATOR.cpu,
    memory=CURATE_ORCHESTRATOR.memory_mb,
    gpu=CURATE_ORCHESTRATOR.gpu,
    cpu=CURATE_ORCHESTRATOR.cpu,
    memory=CURATE_ORCHESTRATOR.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=_SHARED_SECRETS,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_curate(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    init_submodules: bool = True,
    hf_token: str = "",
    init_submodules: bool = True,
    hf_token: str = "",
) -> str:
    """Orchestrator: SQL task grouping + per-task container fan-out."""
    """Orchestrator: SQL task grouping + per-task container fan-out."""
    import sys as _sys
    import time as _time
    import numpy as _np

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo(
        git_remote=git_remote,
        git_commit=git_commit,
        init_submodules=init_submodules,
    )

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo(
        git_remote=git_remote,
        git_commit=git_commit,
        init_submodules=init_submodules,
    )
    _sys.path.insert(0, CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    import hydra as _hydra
    from omegaconf import OmegaConf
    from egomimic.curation.config import apply_curation_seed, select_seed
    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()
    from omegaconf import OmegaConf
    from egomimic.curation.config import apply_curation_seed, select_seed
    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()

    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        cfg = _hydra.compose("curate", overrides=list(hydra_args))

    apply_curation_seed(select_seed(cfg))
    apply_curation_seed(select_seed(cfg))

    # ── 1. SQL task lookup: episode_hash → task_name ──────────────────────────
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

    # ── 2. Resolve episodes via data-config resolvers ─────────────────────────
    by_task: dict[str, list[str]] = {}

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
            by_task.setdefault(task, []).append(episode_hash)

    total_episodes = sum(len(v) for v in by_task.values())
    max_per_shard = int(OmegaConf.select(cfg, "max_episodes_per_shard", default=100))
    total_shards = sum(-(-len(v) // max_per_shard) for v in by_task.values())
    print(
        f"Episode partition: {total_episodes} episodes across {len(by_task)} tasks "
        f"({total_shards} total embed shards at max_per_shard={max_per_shard}) — "
        + ", ".join(f"{t}:{len(h)}" for t, h in sorted(by_task.items())[:5])
        + ("…" if len(by_task) > 5 else "")
    )

    if total_episodes == 0:
        print("No episodes found — check data config resolver settings")
    # ── 1. SQL task lookup: episode_hash → task_name ──────────────────────────
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

    # ── 2. Resolve episodes via data-config resolvers ─────────────────────────
    by_task: dict[str, list[str]] = {}

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
            by_task.setdefault(task, []).append(episode_hash)

    total_episodes = sum(len(v) for v in by_task.values())
    max_per_shard = int(OmegaConf.select(cfg, "max_episodes_per_shard", default=100))
    total_shards = sum(-(-len(v) // max_per_shard) for v in by_task.values())
    print(
        f"Episode partition: {total_episodes} episodes across {len(by_task)} tasks "
        f"({total_shards} total embed shards at max_per_shard={max_per_shard}) — "
        + ", ".join(f"{t}:{len(h)}" for t, h in sorted(by_task.items())[:5])
        + ("…" if len(by_task) > 5 else "")
    )

    if total_episodes == 0:
        print("No episodes found — check data config resolver settings")
        return ""

    # ── 3. Output dir ─────────────────────────────────────────────────────────
    # ── 3. Output dir ─────────────────────────────────────────────────────────
    timestamp = _time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(CFG.output_mount_path) / cfg.name / f"{cfg.description}_{timestamp}"
    )
    output_dir = (
        Path(CFG.output_mount_path) / cfg.name / f"{cfg.description}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. Fan-out per-task scoring containers ────────────────────────────────
    print(f"Spawning {len(by_task)} per-task CPU orchestrators …")
    # ── 4. Fan-out per-task scoring containers ────────────────────────────────
    print(f"Spawning {len(by_task)} per-task CPU orchestrators …")
    t0 = _time.time()

    handles = [
        (
            task_name,
            _score_task_split.spawn(
                task_name,
                episode_hashes,
                git_remote,
                git_commit,
                hydra_args,
                hf_token,
            ),
        )
        for task_name, episode_hashes in sorted(by_task.items())
    ]
    print(f"All {len(handles)} task containers spawned — collecting results …")

    handles = [
        (
            task_name,
            _score_task_split.spawn(
                task_name,
                episode_hashes,
                git_remote,
                git_commit,
                hydra_args,
                hf_token,
            ),
        )
        for task_name, episode_hashes in sorted(by_task.items())
    ]
    print(f"All {len(handles)} task containers spawned — collecting results …")

    scores_by_task: dict[str, dict[str, float]] = {}
    n_failures = 0
    n_failures = 0

    for task_name, handle in handles:
        try:
            _label, task_scores = handle.get(timeout=CFG.timeout_seconds)
            scores_by_task[task_name] = task_scores
        except Exception as exc:
            n_failures += 1
            print(f"[task] FAILED ({task_name}): {exc}")
    for task_name, handle in handles:
        try:
            _label, task_scores = handle.get(timeout=CFG.timeout_seconds)
            scores_by_task[task_name] = task_scores
        except Exception as exc:
            n_failures += 1
            print(f"[task] FAILED ({task_name}): {exc}")

    elapsed = _time.time() - t0
    print(
        f"Scoring done in {elapsed:.1f}s — "
        f"{len(scores_by_task)}/{len(by_task)} tasks succeeded "
        f"({n_failures} failed)"
        f"{len(scores_by_task)}/{len(by_task)} tasks succeeded "
        f"({n_failures} failed)"
    )

    # ── 5. Aggregate + save outputs ───────────────────────────────────────────
    # ── 5. Aggregate + save outputs ───────────────────────────────────────────
    flat_scores: dict[str, float] = {}
    for t_scores in scores_by_task.values():
        flat_scores.update(t_scores)

    all_score_vals = _np.array([s for s in flat_scores.values() if _np.isfinite(s)])

    per_task_stats: dict[str, dict] = {}
    for t_name, t_scores in scores_by_task.items():
        vals = _np.array([s for s in t_scores.values() if _np.isfinite(s)])
        per_task_stats[t_name] = {
            "count":     len(t_scores),
            "mi_mean":   float(_np.nanmean(vals))   if len(vals) else float("nan"),
            "mi_std":    float(_np.nanstd(vals))    if len(vals) else float("nan"),
            "count":     len(t_scores),
            "mi_mean":   float(_np.nanmean(vals))   if len(vals) else float("nan"),
            "mi_std":    float(_np.nanstd(vals))    if len(vals) else float("nan"),
            "mi_median": float(_np.nanmedian(vals)) if len(vals) else float("nan"),
            "mi_min":    float(_np.nanmin(vals))    if len(vals) else float("nan"),
            "mi_max":    float(_np.nanmax(vals))    if len(vals) else float("nan"),
            "mi_min":    float(_np.nanmin(vals))    if len(vals) else float("nan"),
            "mi_max":    float(_np.nanmax(vals))    if len(vals) else float("nan"),
        }

    stats = {
        "total_input":           total_episodes,
        "n_tasks":               len(scores_by_task),
        "n_task_failures":       n_failures,
        "scored":                len(flat_scores),
        "elapsed_seconds":       round(elapsed, 1),
        "max_episodes_per_shard": max_per_shard,
        "mi_mean":   float(all_score_vals.mean())       if len(all_score_vals) else float("nan"),
        "mi_std":    float(all_score_vals.std())        if len(all_score_vals) else float("nan"),
        "mi_median": float(_np.median(all_score_vals))  if len(all_score_vals) else float("nan"),
        "mi_min":    float(all_score_vals.min())        if len(all_score_vals) else float("nan"),
        "mi_max":    float(all_score_vals.max())        if len(all_score_vals) else float("nan"),
        "per_task":  per_task_stats,
        "total_input":           total_episodes,
        "n_tasks":               len(scores_by_task),
        "n_task_failures":       n_failures,
        "scored":                len(flat_scores),
        "elapsed_seconds":       round(elapsed, 1),
        "max_episodes_per_shard": max_per_shard,
        "mi_mean":   float(all_score_vals.mean())       if len(all_score_vals) else float("nan"),
        "mi_std":    float(all_score_vals.std())        if len(all_score_vals) else float("nan"),
        "mi_median": float(_np.median(all_score_vals))  if len(all_score_vals) else float("nan"),
        "mi_min":    float(all_score_vals.min())        if len(all_score_vals) else float("nan"),
        "mi_max":    float(all_score_vals.max())        if len(all_score_vals) else float("nan"),
        "per_task":  per_task_stats,
    }

    def _sort_scores(d: dict) -> dict:
        return dict(
            sorted(
                d.items(),
                key=lambda kv: kv[1] if _np.isfinite(kv[1]) else float("-inf"),
                reverse=True,
            )
        )
        return dict(
            sorted(
                d.items(),
                key=lambda kv: kv[1] if _np.isfinite(kv[1]) else float("-inf"),
                reverse=True,
            )
        )

    sorted_flat = _sort_scores(flat_scores)
    sorted_by_task = {t: _sort_scores(s) for t, s in scores_by_task.items()}

    with open(output_dir / "scores.json", "w") as f:
        json.dump(sorted_flat, f, indent=2)
    with open(output_dir / "scores_by_task.json", "w") as f:
        json.dump(sorted_by_task, f, indent=2)
    with open(output_dir / "kept_hashes.json", "w") as f:
        json.dump(list(sorted_flat.keys()), f, indent=2)
    with open(output_dir / "curation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"Curation done — scored={len(flat_scores)}  "
        f"n_tasks={len(scores_by_task)}  output={output_dir}"
    )

    zarr_volume.commit()
    training_outputs_volume.commit()

    return str(output_dir)


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def submit_curate(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a curation job from an already-pushed commit."""
    hydra_args, init_submodules = pop_init_submodules(hydra_args)
    hydra_args, init_submodules = pop_init_submodules(hydra_args)
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Submitting curation at commit {git_commit[:12]} from {git_remote}")
    if not init_submodules:
        print("Skipping git submodule init (init_submodules=false)")
    handle = run_curate.spawn(
        tuple(hydra_args), git_remote, git_commit,
        init_submodules=init_submodules,
        hf_token=_local_hf_token(),
    )
    print(f"Submitted Modal curation job: {handle.object_id}")
    print("Monitor: https://modal.com/apps/mecka/robotics")


# ---------------------------------------------------------------------------
# python egomimic/modal/curateModal.py name=my_run description=test [overrides…]
# python egomimic/modal/curateModal.py name=my_run description=test [overrides…]
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    modal_env = os.environ.copy()
    hydra_args: list[str] = []
    hydra_args: list[str] = []
    for arg in sys.argv[1:]:
        key, sep, val = arg.lstrip("+").partition("=")
        if sep and key in MODAL_COMPUTE_ARG_MAP:
            modal_env[MODAL_COMPUTE_ARG_MAP[key]] = val
        if sep and key in MODAL_COMPUTE_ARG_MAP:
            modal_env[MODAL_COMPUTE_ARG_MAP[key]] = val
        else:
            hydra_args.append(arg)

    task_compute = ModalCompute.from_mapping(
        modal_env,
        default_gpu="L40S",
        default_cpu=16.0,
        default_memory_mb=131072,
    )
    print(f"Modal curation orchestrator (fixed):          {CURATE_ORCHESTRATOR.summary()}")
    print(f"Modal curation per-task CPU orchestrator:     {TASK_SCORE_COMPUTE.summary()}")
    print(f"Modal curation embed-shard GPU worker:        {task_compute.summary()}")

    launch_detached(Path(__file__).resolve(), "submit_curate", hydra_args, modal_env)
