"""Visualise the KSG joint embedding space across episodes.

Runs the same embed pipeline as curation (pass 2: build action + state
embedders, embed every episode), then reduces to 2D with UMAP (PCA fallback)
and saves scatter PNGs to the zarr volume.

Usage
-----
    python egomimic/modal/plot_embedding_space.py \
        name=<run_name> \
        data=deminf_mecka \
        model=deminf_default \
        norm_stats.precomputed_norm_path=precomputed_norm_stats/mecka_all_zarr \
        [init_submodules=false] \
        [plot.max_episodes=500] \
        [plot.output_dir=/mnt/zarr-data/embedding_plots/<run_name>]

Any Hydra overrides that work for curation runs work here.  Extra keys:

    plot.max_episodes   int   cap on episode count (0 = all, default 0)
    plot.output_dir     str   where to write PNGs + embeddings.npz
    plot.umap_neighbors int   UMAP n_neighbors (default 15)
    plot.umap_min_dist  float UMAP min_dist (default 0.1)
    plot.timestep_sample int  max timestep points in the per-timestep plot (default 5000)

Outputs saved to plot.output_dir:
    action_umap.png     per-episode mean action embeddings (UMAP 2-D)
    state_umap.png      per-episode mean state embeddings
    joint_umap.png      per-episode mean joint (action ‖ state) embeddings
    timestep_umap.png   sampled per-timestep action embeddings, coloured by episode
    embeddings.npz      raw arrays (s_ep, a_ep, lengths, hashes) for offline use

Download after the run:
    modal volume get egoverse-zarr-data embedding_plots/ ./embedding_plots/
"""

from __future__ import annotations

import os
import sys

from egomimic.modal.modal_setup import pop_init_submodules
from egomimic.modal.modal_config import CFG, app, vol

_submodules = pop_init_submodules(sys.argv)


@app.function(
    gpu=CFG.gpu,
    cpu=CFG.cpu,
    memory=CFG.memory,
    timeout=CFG.timeout,
    volumes={"/mnt/zarr-data": vol},
)
def plot_embeddings(
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
) -> str:
    """Embed all episodes and produce UMAP scatter plots. Returns output dir path."""
    import os as _os
    import sys as _sys
    import time as _time

    import numpy as _np

    # ── boot: clone repo, set up sys.path (same as curation shards) ──────────
    from egomimic.modal.curateModal import _boot_container, _build_episode_map, _load_cfg
    _boot_container(git_remote, git_commit, "")
    _sys.path.insert(0, CFG.remote_repo_dir)
    _os.chdir(CFG.remote_repo_dir)

    import torch
    from omegaconf import OmegaConf
    from pathlib import Path
    import hydra as _hydra

    from egomimic.curation.config import (
        apply_curation_seed,
        load_action_norm_stats,
        select_curation_loader,
        select_embedder_settings,
        select_seed,
        select_tensor_keys,
    )
    from egomimic.curation.episode_pipeline import build_embedders, run_pass2_embed_episodes

    t0 = _time.perf_counter()
    cfg = _load_cfg(hydra_args)

    # ── plot knobs ─────────────────────────────────────────────────────────────
    max_episodes = int(OmegaConf.select(cfg, "plot.max_episodes", default=0)) or None
    output_dir = str(
        OmegaConf.select(cfg, "plot.output_dir", default="/mnt/zarr-data/embedding_plots")
    )
    umap_neighbors = int(OmegaConf.select(cfg, "plot.umap_neighbors", default=15))
    umap_min_dist = float(OmegaConf.select(cfg, "plot.umap_min_dist", default=0.1))
    timestep_sample = int(OmegaConf.select(cfg, "plot.timestep_sample", default=5000))
    _os.makedirs(output_dir, exist_ok=True)

    # ── norm stats ─────────────────────────────────────────────────────────────
    apply_curation_seed(select_seed(cfg))
    embed_cfg = select_embedder_settings(cfg)
    action_key, image_key = select_tensor_keys(cfg)

    action_mean, action_std = load_action_norm_stats(
        cfg,
        action_key=action_key,
        norm_min_std=embed_cfg.norm_min_std,
        search_roots=[CFG.output_mount_path, CFG.remote_repo_dir, Path.cwd()],
    )
    print(f"Norm stats: action_key={action_key}, dim={action_mean.shape}")

    # ── build episode map ──────────────────────────────────────────────────────
    all_episodes = _build_episode_map(cfg, _hydra)
    all_hashes = sorted(all_episodes.keys())
    if max_episodes and len(all_hashes) > max_episodes:
        print(f"Capping at {max_episodes} of {len(all_hashes)} episodes")
        all_hashes = all_hashes[:max_episodes]
        ep_map = {h: all_episodes[h] for h in all_hashes}
    else:
        ep_map = all_episodes

    device = torch.device(embed_cfg.device if torch.cuda.is_available() else "cpu")
    ds_name = next(iter(cfg.data.train_datasets))
    loader_cfg = select_curation_loader(cfg, ds_name)
    print(f"Embedding {len(ep_map)} episodes on {device}")

    # ── embed ──────────────────────────────────────────────────────────────────
    action_embedder, state_embedder = build_embedders(
        embed_cfg, action_mean, action_std, device, select_seed(cfg),
        global_frame_batch_size=loader_cfg.global_frame_batch_size,
    )
    state_latents, action_latents, hashes, ep_lengths = run_pass2_embed_episodes(
        ep_map,
        scored_hashes=set(),
        action_key=action_key,
        image_key=image_key,
        loader=loader_cfg,
        action_embedder=action_embedder,
        state_embedder=state_embedder,
    )
    print(f"Embedded {len(hashes)} episodes in {_time.perf_counter() - t0:.1f}s")

    if not hashes:
        print("No episodes embedded — nothing to plot")
        return output_dir

    # ── per-episode mean embeddings ────────────────────────────────────────────
    a_ep = _np.stack([a.mean(0) for a in action_latents])   # (N, latent_dim)
    s_ep = _np.stack([s.mean(0) for s in state_latents])    # (N, latent_dim)
    j_ep = _np.concatenate([a_ep, s_ep], axis=1)            # (N, 2*latent_dim)
    lengths = _np.array(ep_lengths, dtype=_np.int32)

    # ── per-timestep arrays (sampled) ─────────────────────────────────────────
    a_all = _np.concatenate(action_latents, axis=0)          # (N_ts, latent_dim)
    ep_ids = _np.concatenate(
        [_np.full(t, i, dtype=_np.int32) for i, t in enumerate(ep_lengths)]
    )
    if len(a_all) > timestep_sample:
        rng = _np.random.default_rng(42)
        idx = rng.choice(len(a_all), size=timestep_sample, replace=False)
        a_ts, ep_ts = a_all[idx], ep_ids[idx]
    else:
        a_ts, ep_ts = a_all, ep_ids

    # ── save raw arrays ────────────────────────────────────────────────────────
    _np.savez_compressed(
        _os.path.join(output_dir, "embeddings.npz"),
        s_ep=s_ep, a_ep=a_ep, lengths=lengths,
        hashes=_np.array(hashes, dtype=object),
    )
    print(f"Saved embeddings.npz ({len(hashes)} episodes)")

    # ── dimensionality reduction ───────────────────────────────────────────────
    def _reduce_2d(X: "_np.ndarray", label: str) -> "_np.ndarray":
        try:
            import umap as _umap
            print(f"  UMAP {label} {X.shape} …")
            reducer = _umap.UMAP(
                n_neighbors=umap_neighbors, min_dist=umap_min_dist,
                random_state=42, verbose=False,
            )
            return reducer.fit_transform(X.astype(_np.float32))
        except ImportError:
            from sklearn.decomposition import PCA
            print(f"  umap-learn not found — PCA fallback for {label}")
            return PCA(n_components=2, random_state=42).fit_transform(X.astype(_np.float32))

    print("Running dimensionality reduction …")
    umap_a  = _reduce_2d(a_ep, "action/episode")
    umap_s  = _reduce_2d(s_ep, "state/episode")
    umap_j  = _reduce_2d(j_ep, "joint/episode")
    umap_ts = _reduce_2d(a_ts, "action/timestep")

    # ── plots ──────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _ep_scatter(xy, color_vals, title, cbar_label, path) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            xy[:, 0], xy[:, 1], c=color_vals,
            cmap="viridis", s=14, alpha=0.75, linewidths=0,
        )
        plt.colorbar(sc, ax=ax, label=cbar_label)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"  saved {path}")

    n_ep = len(hashes)
    _ep_scatter(
        umap_a, lengths,
        f"Action embeddings ({n_ep} episodes, per-episode mean)\ncolour = episode length",
        "episode length (steps)",
        _os.path.join(output_dir, "action_umap.png"),
    )
    _ep_scatter(
        umap_s, lengths,
        f"State embeddings ({n_ep} episodes, per-episode mean)\ncolour = episode length",
        "episode length (steps)",
        _os.path.join(output_dir, "state_umap.png"),
    )
    _ep_scatter(
        umap_j, lengths,
        f"Joint (action ‖ state) embeddings ({n_ep} episodes, per-episode mean)\ncolour = episode length",
        "episode length (steps)",
        _os.path.join(output_dir, "joint_umap.png"),
    )

    # timestep scatter — colour by episode index mod 20
    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        umap_ts[:, 0], umap_ts[:, 1],
        c=ep_ts % 20, cmap="tab20", vmin=0, vmax=19,
        s=4, alpha=0.45, linewidths=0,
    )
    ax.set_title(
        f"Action embeddings — {len(a_ts):,} sampled timesteps (/{len(a_all):,})\n"
        "colour = episode index mod 20",
        fontsize=11,
    )
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    fig.tight_layout()
    ts_path = _os.path.join(output_dir, "timestep_umap.png")
    fig.savefig(ts_path, dpi=130)
    plt.close(fig)
    print(f"  saved {ts_path}")

    vol.commit()
    print(f"\nAll outputs written to {output_dir}")
    return output_dir


# ── local submit entrypoint ───────────────────────────────────────────────────

def _submit() -> None:
    import subprocess

    hydra_args = tuple(a for a in sys.argv[1:] if not a.startswith("--"))
    git_remote = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], text=True
    ).strip()
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    print("Submitting plot_embeddings:")
    print(f"  commit    : {git_commit[:12]}")
    print(f"  submodules: {sorted(_submodules) if _submodules else 'none'}")
    print(f"  args      : {hydra_args}")

    handle = plot_embeddings.spawn(
        hydra_args=hydra_args,
        git_remote=git_remote,
        git_commit=git_commit,
    )
    print(f"\nJob spawned. Stream logs:\n  modal logs {app.name}\n")
    output_dir = handle.get(timeout=7200)
    print(f"\nDone. Outputs at: {output_dir}")
    print(
        "Download:\n"
        "  modal volume get egoverse-zarr-data embedding_plots/ ./embedding_plots/"
    )


if __name__ == "__main__":
    with app.run():
        _submit()
