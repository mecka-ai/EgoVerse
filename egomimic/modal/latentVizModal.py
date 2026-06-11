"""Latent t-SNE export for ANY data config on Modal — no curation required.

Where curateModal.py exists to *score* episodes (KSG mutual information over
tar-sharded tasks), this entry point exists only to *visualize*: it resolves
whatever episodes a data config selects, embeds every frame (image state +
action chunk), and writes the ``tsne3d_<group>.json`` files the interactive
viewer consumes. No scores, no tar shards, no required norm-stats file.

Pipeline (single GPU container)
-------------------------------
1. Resolve episodes from ``data=<config>`` ``train_datasets`` (+
   ``valid_datasets`` when present) — resolver + filters + eps_to_use lists.
   With ``force_modal_resolver`` (default) any resolver type is re-rooted at
   the mounted ``mecka_data_v2`` zarr volume, so zip/S3 training configs work
   unmodified.
2. Group episodes (``group_by``: task | dataset | none).
3. Action norm stats: precomputed file when configured, else fit from a
   sample of the resolved episodes (actions-only zarr reads — cheap).
4. Per group: stream-load + GPU-embed all frames
   (egomimic.curation.episode_pipeline) and export 3-D t-SNE JSON + PNGs +
   raw latents (egomimic.curation.tsne_viz).
5. Write episode_hashes.json / val_episodes.json / viz_manifest.json so the
   viewer (latent_viz_app.py) and the MP4 renderer (episode_preview.py) can
   pick the run up directly.

Outputs land on the ``egoverse-training-outputs`` volume under
``latent_viz/<name>/<description>_<timestamp>/``.

Usage
-----
    python egomimic/modal/latentVizModal.py data=deminf_mecka name=my_viz description=test

    python egomimic/modal/latentVizModal.py data=mecka_all_zip name=my_viz \\
        description=zip group_by=dataset model.latent_dim=64 \\
        +modal_gpu=L40S:1 +modal_cpu=32 +modal_memory_gb=128

    modal run --env robotics egomimic/modal/latentVizModal.py::submit_latent_viz -- \\
        data=deminf_mecka name=my_viz description=test

Then serve it:
    LATENT_VIZ_RUN=latent_viz/<name>/<description>_<ts> \\
        MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/latent_viz_app.py
"""

from __future__ import annotations

import json
import os
import re
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

# Single embed/export container — override at launch via +modal_gpu / +modal_cpu
# / +modal_memory_gb.
VIZ_COMPUTE = ModalCompute.from_environ(
    default_gpu="L40S",
    default_cpu=16.0,
    default_memory_mb=131072,
)

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]

# Resolver fields carried over when force_modal_resolver rebuilds a data
# config's resolver as a ModalEpisodeResolver on the zarr volume.
_RESOLVER_CARRY_KEYS = (
    "key_map",
    "transform_list",
    "pause_removal_epsilon",
    "eps_to_use",
    "eps_to_ignore",
    "exclude_hashes",
    "debug",
)


def _safe_group_name(name: str) -> str:
    """Group names become tsne3d_<group>.json filenames — keep them path-safe."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)) or "unknown"


@app.function(
    gpu=VIZ_COMPUTE.gpu,
    cpu=VIZ_COMPUTE.cpu,
    memory=VIZ_COMPUTE.memory_mb,
    timeout=CFG.timeout_seconds,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def run_latent_viz(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    init_submodules: bool = False,
    hf_token: str = "",
) -> str:
    """Resolve → group → embed → export tsne3d JSONs for the viewer."""
    import sys as _sys
    import time as _time
    from datetime import datetime

    import numpy as _np

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _prepare_repo(
        git_remote=git_remote,
        git_commit=git_commit,
        init_submodules=init_submodules,
    )
    _sys.path.insert(0, CFG.remote_repo_dir)
    os.chdir(CFG.remote_repo_dir)
    os.environ["MODAL_IS_REMOTE"] = "1"
    os.environ.setdefault("HYDRA_FULL_ERROR", "1")

    import hydra as _hydra
    import torch
    from omegaconf import OmegaConf

    from egomimic.curation.config import (
        CurationLoaderSettings,
        apply_curation_seed,
        load_action_norm_stats,
        select_curation_loader,
        select_embedder_settings,
        select_seed,
        select_tensor_keys,
    )
    from egomimic.curation.embedders import _fit_gaussian_stats
    from egomimic.curation.episode_pipeline import (
        _release_episode_cache,
        build_embedders,
        run_pass2_embed_episodes,
    )
    from egomimic.curation.tsne_viz import export_task_tsne3d, make_task_tsne_plots
    from egomimic.utils.aws.aws_data_utils import load_env

    load_env()

    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        cfg = _hydra.compose("latent_viz", overrides=list(hydra_args))

    seed = select_seed(cfg)
    apply_curation_seed(seed)
    action_key, image_key = select_tensor_keys(cfg)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (
        Path(CFG.output_mount_path)
        / "latent_viz"
        / str(cfg.name)
        / f"{cfg.description}_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_rel = run_dir.relative_to(CFG.output_mount_path)

    # ── 1. Resolve episodes from the data config ──────────────────────────────
    def _resolver_cfg(ds_cfg):
        resolver_cfg = ds_cfg.resolver
        if not bool(OmegaConf.select(cfg, "force_modal_resolver", default=True)):
            return resolver_cfg
        out: dict = {
            "_target_": "egomimic.rldb.zarr.zarr_dataset_multi.ModalEpisodeResolver",
            "folder_path": CFG.volume_mount_path,
        }
        for key in _RESOLVER_CARRY_KEYS:
            node = OmegaConf.select(resolver_cfg, key, default=None)
            if node is not None:
                out[key] = (
                    OmegaConf.to_container(node, resolve=True)
                    if OmegaConf.is_config(node)
                    else node
                )
        return OmegaConf.create(out)

    datasets: dict = {}          # episode_hash -> ZarrDataset
    ds_of_hash: dict[str, str] = {}
    val_hashes: set[str] = set()
    ds_names: list[str] = []
    for split, block_key in (("train", "data.train_datasets"), ("valid", "data.valid_datasets")):
        block = OmegaConf.select(cfg, block_key, default=None)
        if not block:
            continue
        for ds_name, ds_cfg in block.items():
            if "resolver" not in ds_cfg:
                print(f"[{split}/{ds_name}] no resolver block — skipped")
                continue
            ds_names.append(str(ds_name))
            resolver = _hydra.utils.instantiate(_resolver_cfg(ds_cfg))
            dataset_filter = (
                _hydra.utils.instantiate(ds_cfg.filters) if "filters" in ds_cfg else None
            )
            resolved = resolver.resolve(filters=dataset_filter)
            print(f"[{split}/{ds_name}] resolved {len(resolved)} episodes")
            for ep_hash, zarr_ds in resolved.items():
                datasets.setdefault(ep_hash, zarr_ds)
                ds_of_hash.setdefault(ep_hash, str(ds_name))
                if split == "valid":
                    val_hashes.add(ep_hash)

    if not datasets:
        raise ValueError("Data config resolved no episodes — check data= and filters.")

    # VAL badge list for the viewer. A valid_datasets block that resolves to the
    # exact same episode set as train (index-split val) carries no information.
    if val_hashes and val_hashes != set(datasets):
        with open(run_dir / "val_episodes.json", "w") as f:
            json.dump(sorted(val_hashes), f)
        print(f"val_episodes.json: {len(val_hashes)} episodes")
    elif val_hashes:
        print("valid_datasets resolved to the full episode set — skipping VAL list")

    # ── 2. Group episodes ─────────────────────────────────────────────────────
    group_by = str(OmegaConf.select(cfg, "group_by", default="task"))
    hash_to_task: dict[str, str] = {}
    if group_by == "task":
        try:
            from egomimic.utils.aws.aws_sql import (
                create_default_engine,
                episode_table_to_df,
            )

            df = episode_table_to_df(create_default_engine())
            if "is_deleted" in df.columns:
                df = df[df["is_deleted"] != True]  # noqa: E712
            if "task" in df.columns:
                hash_to_task = dict(
                    zip(df["episode_hash"], df["task"].fillna("unknown"))
                )
        except Exception as exc:
            print(f"Task lookup failed ({exc}) — falling back to group_by=dataset")
            group_by = "dataset"

    groups: dict[str, list[str]] = {}
    for ep_hash in datasets:
        if group_by == "task":
            group = str(hash_to_task.get(ep_hash) or "unknown")
            if group in ("nan", "None", ""):
                group = "unknown"
        elif group_by == "dataset":
            group = ds_of_hash[ep_hash]
        else:
            group = "all"
        groups.setdefault(_safe_group_name(group), []).append(ep_hash)

    print(
        f"Episode partition ({group_by}): {len(datasets)} episodes across "
        f"{len(groups)} group(s) — "
        + ", ".join(f"{g}:{len(h)}" for g, h in sorted(groups.items())[:8])
        + ("…" if len(groups) > 8 else "")
    )

    # ── 3. Action norm stats ──────────────────────────────────────────────────
    embed_cfg = select_embedder_settings(cfg)
    try:
        loader = select_curation_loader(cfg, ds_names[0])
        print(f"loader settings from data.curation_loader.{ds_names[0]}")
    except KeyError:
        lb = OmegaConf.select(cfg, "loader", default=None) or {}
        loader = CurationLoaderSettings(
            episode_workers=int(lb.get("episode_workers", 4)),
            pass2_image_decode_workers=int(lb.get("pass2_image_decode_workers", 2)),
            frame_queue_maxsize=int(lb.get("frame_queue_maxsize", 32)),
            global_frame_batch_size=int(lb.get("global_frame_batch_size", 512)),
        )

    if OmegaConf.select(cfg, "model.precomputed_norm_stats.path", default=None):
        action_mean, action_std = load_action_norm_stats(
            cfg,
            action_key,
            norm_min_std=embed_cfg.norm_min_std,
            search_roots=[CFG.output_mount_path, CFG.remote_repo_dir],
        )
        print(f"Action norm stats: precomputed (feat_dim={action_mean.shape[0]})")
    else:
        sample_n = int(OmegaConf.select(cfg, "norm_sample_episodes", default=64))
        rng = _np.random.default_rng(seed)
        sample = sorted(datasets)
        if len(sample) > sample_n:
            sample = sorted(rng.choice(sample, size=sample_n, replace=False))
        t0 = _time.perf_counter()
        act_parts: list = []
        for ep_hash in sample:
            zarr_ds = datasets[ep_hash]
            try:
                actions, _ = zarr_ds._collect_curation_batched(
                    action_key=action_key,
                    image_key=image_key,
                    image_decode_workers=0,
                    load_images=False,
                )
                if actions is not None and len(actions):
                    act_parts.append(actions.reshape(len(actions), -1).astype(_np.float32))
            finally:
                _release_episode_cache(zarr_ds)
        if not act_parts:
            raise RuntimeError(
                f"Could not load actions ({action_key}) from any of "
                f"{len(sample)} sampled episodes — cannot fit norm stats."
            )
        action_mean, action_std = _fit_gaussian_stats(
            act_parts, min_std=embed_cfg.norm_min_std
        )
        print(
            f"Action norm stats: fit from {len(act_parts)}/{len(sample)} episodes "
            f"(feat_dim={action_mean.shape[0]}) in {_time.perf_counter() - t0:.1f}s"
        )

    # ── 4. Embedders (shared across groups for consistent latent spaces) ──────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    action_embedder, state_embedder = build_embedders(
        embed_cfg,
        action_mean,
        action_std,
        device,
        seed,
        global_frame_batch_size=loader.global_frame_batch_size,
    )

    # ── 5. Per-group embed + t-SNE export ─────────────────────────────────────
    every_n_base = int(OmegaConf.select(cfg, "every_n", default=10))
    max_points = int(OmegaConf.select(cfg, "max_points_per_group", default=30000))
    write_pngs = bool(OmegaConf.select(cfg, "write_pngs", default=True))
    manifest_groups: dict = {}
    for group in sorted(groups):
        hashes = groups[group]
        t0 = _time.perf_counter()
        state_latents, action_latents, done_hashes, ep_lengths = run_pass2_embed_episodes(
            {h: datasets[h] for h in hashes},
            set(hashes),
            action_key,
            image_key,
            loader,
            action_embedder,
            state_embedder,
            progress=group,
        )
        if not done_hashes:
            print(f"[{group}] no episodes embedded — skipped")
            continue

        total_frames = sum(ep_lengths)
        every_n = max(every_n_base, -(-total_frames // max_points))
        if every_n > every_n_base:
            print(
                f"[{group}] every_n {every_n_base} → {every_n} "
                f"({total_frames} frames, cap {max_points} points)"
            )

        tsne3d_json = export_task_tsne3d(
            group,
            state_latents,
            action_latents,
            done_hashes,
            run_dir / "tsne3d",
            every_n=every_n,
            seed=seed,
        )
        if write_pngs:
            make_task_tsne_plots(
                group,
                state_latents,
                action_latents,
                run_dir / "tsne",
                every_n=every_n,
                seed=seed,
            )
        # Raw latents: re-project later (different t-SNE params / UMAP / 2-D
        # vs 3-D) without re-running the GPU embed pass.
        lat_dir = run_dir / "latents"
        lat_dir.mkdir(parents=True, exist_ok=True)
        _np.savez_compressed(
            lat_dir / f"latents_{group}.npz",
            state=_np.concatenate(state_latents, axis=0),
            action=_np.concatenate(action_latents, axis=0),
            lengths=_np.asarray(ep_lengths, dtype=_np.int64),
            hashes=_np.asarray(done_hashes),
        )
        training_outputs_volume.commit()
        manifest_groups[group] = {
            "episodes": len(done_hashes),
            "frames": total_frames,
            "every_n": every_n,
        }
        print(
            f"[{group}] {len(done_hashes)} episodes, {total_frames} frames "
            f"embedded + exported in {_time.perf_counter() - t0:.1f}s "
            f"→ {tsne3d_json}"
        )

    # ── 6. Manifest + episode list ────────────────────────────────────────────
    with open(run_dir / "episode_hashes.json", "w") as f:
        json.dump(sorted(datasets), f)
    with open(run_dir / "viz_manifest.json", "w") as f:
        json.dump(
            {
                "name": str(cfg.name),
                "description": str(cfg.description),
                "overrides": list(hydra_args),
                "git_commit": git_commit,
                "group_by": group_by,
                "action_key": action_key,
                "image_key": image_key,
                "latent_dim": embed_cfg.latent_dim,
                "state_backbone": embed_cfg.state_image.backbone,
                "seed": seed,
                "groups": manifest_groups,
            },
            f,
            indent=2,
        )
    training_outputs_volume.commit()

    if bool(OmegaConf.select(cfg, "render_videos", default=False)):
        try:
            render = modal.Function.from_name(
                "egoverse-episode-preview",
                "render_episode",
                environment_name=os.environ.get("MODAL_ENVIRONMENT", "robotics"),
            )
            for ep_hash in sorted(datasets):
                render.spawn(ep_hash)
            print(f"Spawned {len(datasets)} MP4 render(s) on egoverse-episode-preview")
        except Exception as exc:
            print(f"render_videos failed ({exc}) — render manually (see below)")

    print(
        f"\nDone: {sum(g['episodes'] for g in manifest_groups.values())} episodes "
        f"in {len(manifest_groups)} group(s) → {run_rel} (egoverse-training-outputs)\n"
        f"Serve the viewer:\n"
        f"  LATENT_VIZ_RUN={run_rel} MODAL_ENVIRONMENT=robotics "
        f"modal deploy egomimic/modal/latent_viz_app.py\n"
        f"Render episode MP4s (video previews), if not already rendered:\n"
        f"  modal volume get --env robotics egoverse-training-outputs "
        f"{run_rel}/episode_hashes.json /tmp/eph.json\n"
        f"  MODAL_ENVIRONMENT=robotics modal run "
        f"egomimic/modal/episode_preview.py::render_all --hashes-file /tmp/eph.json"
    )
    return str(run_rel)


@app.local_entrypoint()
def submit_latent_viz(*hydra_args: str) -> None:
    """Fire-and-forget: spawn a latent-viz export from an already-pushed commit."""
    hydra_args, init_submodules = pop_init_submodules(hydra_args)
    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print(
            "Warning: local repo has uncommitted changes. "
            "Modal will run the last committed state only."
        )
    print(f"Submitting latent-viz export at commit {git_commit[:12]} from {git_remote}")
    handle = run_latent_viz.spawn(
        tuple(hydra_args), git_remote, git_commit,
        init_submodules=init_submodules,
        hf_token=_local_hf_token(),
    )
    _env = os.environ.get("MODAL_ENVIRONMENT", "robotics")
    _app = os.environ.get("MODAL_APP_NAME", "egomimic-training")
    print(f"Submitted Modal latent-viz job: {handle.object_id}")
    print(f"Monitor: https://modal.com/apps/mecka/{_env}/apps/{_app}")


# ---------------------------------------------------------------------------
# python egomimic/modal/latentVizModal.py data=<config> name=my_viz [overrides…]
# ---------------------------------------------------------------------------

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

    viz_compute = ModalCompute.from_mapping(
        modal_env,
        default_gpu="L40S",
        default_cpu=16.0,
        default_memory_mb=131072,
    )
    print(f"Modal app:                    {modal_env['MODAL_APP_NAME']}")
    print(f"Modal latent-viz container:   {viz_compute.summary()}")

    launch_detached(Path(__file__).resolve(), "submit_latent_viz", hydra_args, modal_env)
