"""Phase 1 of the DemInf v2 pipeline: build MP4+NPZ shards from zarr episodes.

Each episode becomes exactly one shard pair:
  {ep_hash}.mp4  — H.264 video (30 fps, gop=30, bf=0, crf=18)
  {ep_hash}.npz  — action (T, flat_dim) float32 + episode_hash scalar

Shards are written to the egoverse-deminf-v2 volume at:
  /mnt/deminf-v2/{run_name}/shards/{task_name}/

Usage (from curate_v2.py):
  shard_pairs = build_shards_for_task(episode_dirs, task_name, run_name, ...)
"""

from __future__ import annotations

import math
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
    DEMINF_V2_MOUNT,
    _boot_container as _boot_container_fn,
    _local_hf_token,
    _resolve_git_state,
    app,
    deminf_v2_volume,
    image,
    training_outputs_volume,
    zarr_volume,
    pop_init_submodules,
    launch_detached,
)

# Re-export _boot_container/_load_cfg to avoid depending on curateModal at import time.
def _boot_container(git_remote: str, git_commit: str, hf_token: str) -> None:
    import sys as _sys
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    _boot_container_fn(git_remote, git_commit, hf_token)


def _load_cfg(hydra_args: tuple[str, ...]):
    import hydra as _hydra
    with _hydra.initialize_config_dir(
        config_dir=f"{CFG.remote_repo_dir}/egomimic/hydra_configs",
        version_base="1.3",
    ):
        return _hydra.compose("curate", overrides=list(hydra_args))


_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]

N_WORKERS = 50   # max parallel shard builder containers
FPS = 30
GOP = 30


# ---------------------------------------------------------------------------
# MP4 encoding helper
# ---------------------------------------------------------------------------


def _encode_mp4(frames_chw: "np.ndarray", out_path: str) -> None:
    """Encode (T, C, H, W) float32 [0,1] → H.264 MP4 with deterministic keyframes."""
    import av
    import numpy as np

    T, C, H, W = frames_chw.shape
    container = av.open(str(out_path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = W
    stream.height = H
    stream.pix_fmt = "yuv420p"
    stream.gop_size = GOP
    # bf=0: no B-frames → deterministic byte offsets per keyframe
    # preset=faster: balance encode speed vs file size for storage
    stream.options = {"bf": "0", "preset": "faster", "crf": "18"}

    for t in range(T):
        hwc = (frames_chw[t].transpose(1, 2, 0) * 255.0).clip(0, 255).astype("uint8")
        av_frame = av.VideoFrame.from_ndarray(hwc, format="rgb24")
        av_frame.pts = t
        for pkt in stream.encode(av_frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


# ---------------------------------------------------------------------------
# Per-worker Modal function
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    cpu=4,
    memory=32768,
    timeout=7200,
    max_containers=N_WORKERS,
    secrets=_SHARED_SECRETS,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        DEMINF_V2_MOUNT: deminf_v2_volume,
    },
)
def _build_shards_worker(
    episode_dirs: list[str],
    task_name: str,
    worker_id: int,
    run_name: str,
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    hf_token: str = "",
) -> list[tuple[str, str]]:
    """Build one MP4+NPZ shard per episode for a batch of zarr episodes.

    Returns a list of (mp4_path, npz_path) absolute paths on the deminf_v2 volume.
    """
    import time as _time
    import numpy as _np

    _boot_container(git_remote, git_commit, hf_token)

    import hydra as _hydra
    from omegaconf import OmegaConf
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
    from egomimic.curation.config import select_tensor_keys

    cfg = _load_cfg(hydra_args)
    ds_name = next(iter(cfg.data.train_datasets))
    resolver_cfg = cfg.data.train_datasets[ds_name].resolver
    key_map = _hydra.utils.instantiate(resolver_cfg.key_map)
    transform_list = _hydra.utils.instantiate(resolver_cfg.transform_list)
    pause_eps = OmegaConf.select(resolver_cfg, "pause_removal_epsilon")
    action_key, image_key = select_tensor_keys(cfg)

    out_dir = Path(DEMINF_V2_MOUNT) / run_name / "shards" / task_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"[{task_name}][worker {worker_id}]"
    shard_pairs: list[tuple[str, str]] = []
    t_worker = _time.perf_counter()

    for ep_dir in episode_dirs:
        ep_hash = Path(ep_dir).name.removesuffix(".zarr")
        mp4_path = str(out_dir / f"{ep_hash}.mp4")
        npz_path = str(out_dir / f"{ep_hash}.npz")

        done_flag = out_dir / f"{ep_hash}.done"
        if done_flag.exists():
            shard_pairs.append((mp4_path, npz_path))
            print(f"{tag} {ep_hash[:8]}: already built — skipping")
            continue

        t_ep = _time.perf_counter()
        try:
            zarr_ds = ZarrDataset(
                ep_dir,
                key_map=key_map,
                transform_list=transform_list,
                pause_removal_epsilon=pause_eps,
            )
            actions, images, _ = zarr_ds.collect_curation_episode(
                action_key=action_key,
                image_key=image_key,
                image_decode_workers=2,
            )
        except Exception as exc:
            print(f"{tag} {ep_hash[:8]}: ZarrDataset load FAILED — {exc}")
            continue
        finally:
            try:
                zarr_ds._zarr_bulk_cache = None
            except Exception:
                pass

        if actions is None or images is None:
            print(f"{tag} {ep_hash[:8]}: quality filter removed all frames — skipping")
            continue

        T = images.shape[0]
        if T == 0:
            print(f"{tag} {ep_hash[:8]}: 0 frames after filtering — skipping")
            continue

        try:
            _encode_mp4(images, mp4_path)
        except Exception as exc:
            print(f"{tag} {ep_hash[:8]}: MP4 encode FAILED — {exc}")
            continue

        flat_actions = actions.reshape(T, -1).astype(_np.float32)
        _np.savez_compressed(
            npz_path,
            action=flat_actions,
            episode_hash=_np.str_(ep_hash),
        )
        done_flag.touch()

        elapsed = _time.perf_counter() - t_ep
        print(f"{tag} {ep_hash[:8]}: {T} frames → built in {elapsed:.1f}s")
        shard_pairs.append((mp4_path, npz_path))

    deminf_v2_volume.commit()
    print(
        f"{tag} done — {len(shard_pairs)}/{len(episode_dirs)} shards built "
        f"in {_time.perf_counter() - t_worker:.1f}s"
    )
    return shard_pairs


# ---------------------------------------------------------------------------
# Task-level orchestration helper (not a Modal function)
# ---------------------------------------------------------------------------


def build_shards_for_task(
    episode_dirs: list[str],
    task_name: str,
    run_name: str,
    hydra_args: tuple[str, ...],
    git_remote: str,
    git_commit: str,
    hf_token: str = "",
    n_workers: int = N_WORKERS,
) -> list[tuple[str, str]]:
    """Distribute episode dirs across n_workers Modal containers and collect shard pairs."""
    n = len(episode_dirs)
    if n == 0:
        return []

    batch_size = max(1, math.ceil(n / n_workers))
    batches = [episode_dirs[i : i + batch_size] for i in range(0, n, batch_size)]

    print(
        f"[{task_name}] launching {len(batches)} shard-builder workers "
        f"for {n} episodes (batch_size≈{batch_size})"
    )

    results = list(
        _build_shards_worker.starmap(
            [
                (batch, task_name, wid, run_name, hydra_args, git_remote, git_commit, hf_token)
                for wid, batch in enumerate(batches)
            ],
            return_exceptions=True,
        )
    )

    shard_pairs: list[tuple[str, str]] = []
    for wid, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"[{task_name}] worker {wid} FAILED: {r}")
        else:
            shard_pairs.extend(r)

    print(f"[{task_name}] {len(shard_pairs)} shards built across {len(batches)} workers")
    return shard_pairs
