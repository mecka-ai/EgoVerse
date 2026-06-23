"""Phase 2 of the DemInf v2 pipeline: embed MP4+NPZ shards into state/action latents.

Three Modal functions:
  _embed_state_shards     — L40S GPU, MP4s → StateEmbedder; producer-consumer (decode thread)
  _embed_action_shards    — CPU only, NPZs → ActionEmbedder (gaussian, fast)
  _embed_action_shards_gpu— L40S GPU, NPZs → CheckpointActionEmbedder; producer-consumer

Both write NPZ files to the training_outputs_volume at:
  {output_dir}/latents_v2/{task_name}/state.npz
  {output_dir}/latents_v2/{task_name}/action.npz

Each NPZ has one key per episode: {ep_hash: array(T, D)}.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import modal
from modal_setup import (
    CFG,
    DEMINF_V2_MOUNT,
    _boot_container as _boot_container_fn,
    app,
    deminf_v2_volume,
    image,
    training_outputs_volume,
    zarr_volume,
)

_SHARED_SECRETS = [modal.Secret.from_name(name) for name in CFG.secret_names]

_DONE = object()  # end-sentinel for producer-consumer queues


def _boot_container(git_remote: str, git_commit: str, hf_token: str) -> None:
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


# ---------------------------------------------------------------------------
# MP4 frame decoder
# ---------------------------------------------------------------------------


def _decode_mp4_frames(mp4_path: str) -> "np.ndarray":
    """Decode MP4 → (T, 3, H, W) float32 [0,1] using PyAV (one frame at a time)."""
    import av
    import numpy as np

    frames = []
    container = av.open(str(mp4_path))
    try:
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
            frames.append(arr.transpose(2, 0, 1).astype(np.float32) / 255.0)
    finally:
        container.close()
    if not frames:
        return np.empty((0, 3, 1, 1), dtype=np.float32)
    return np.stack(frames)  # (T, 3, H, W)


# ---------------------------------------------------------------------------
# State embedding — GPU (L40S)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="H200",
    cpu=4,
    memory=65536,
    timeout=14400,
    secrets=_SHARED_SECRETS,
    volumes={
        DEMINF_V2_MOUNT: deminf_v2_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def _embed_state_shards(
    shard_pairs: list[tuple[str, str]],
    task_name: str,
    run_name: str,
    hydra_args: tuple[str, ...],
    action_mean: list,
    action_std: list,
    git_remote: str,
    git_commit: str,
    output_dir: str,
    hf_token: str = "",
    batch_size: int = 512,
    prefetch: int = 4,
) -> str:
    """Embed state (image) latents from MP4 shards. Returns path to state.npz.

    Producer-consumer: a background thread decodes MP4s into a bounded queue
    (capacity=prefetch) while the GPU thread consumes and embeds batches.
    This overlaps ~17s of MP4 decode with ~5s of GPU inference per episode.
    """
    import queue as _queue
    import threading as _threading
    import time as _time
    import numpy as _np
    import torch

    _boot_container(git_remote, git_commit, hf_token)

    from egomimic.curation.config import (
        apply_curation_seed,
        select_embedder_settings,
        select_seed,
        select_tensor_keys,
    )
    from egomimic.curation.episode_pipeline import build_embedders

    cfg = _load_cfg(hydra_args)
    apply_curation_seed(select_seed(cfg))
    embed_cfg = select_embedder_settings(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_key, _ = select_tensor_keys(cfg)

    _, state_embedder = build_embedders(
        embed_cfg,
        _np.asarray(action_mean, dtype=_np.float32),
        _np.asarray(action_std, dtype=_np.float32),
        device,
        select_seed(cfg),
    )

    tag = f"[{task_name}][state]"
    t_total = _time.perf_counter()

    # ── producer: background thread loads NPZ + decodes MP4 ──────────────────
    loaded_q: "_queue.Queue[object]" = _queue.Queue(maxsize=prefetch)

    def _loader() -> None:
        for mp4_path, npz_path in shard_pairs:
            try:
                npz = _np.load(npz_path, allow_pickle=True)
                ep_hash = str(npz["episode_hash"])
            except Exception as exc:
                print(f"{tag} failed to read NPZ {npz_path}: {exc}")
                continue
            try:
                frames_chw = _decode_mp4_frames(mp4_path)  # (T, 3, H, W) float32
            except Exception as exc:
                print(f"{tag} {ep_hash[:8]}: MP4 decode FAILED — {exc}")
                continue
            if len(frames_chw) == 0:
                print(f"{tag} {ep_hash[:8]}: 0 frames — skipping")
                continue
            loaded_q.put((ep_hash, frames_chw))
        loaded_q.put(_DONE)

    loader = _threading.Thread(target=_loader, daemon=True)
    loader.start()

    # ── consumer: GPU thread embeds frames from queue ─────────────────────────
    ep_to_state: dict[str, "_np.ndarray"] = {}
    while True:
        item = loaded_q.get()
        if item is _DONE:
            break
        ep_hash, frames_chw = item
        T = len(frames_chw)
        t_ep = _time.perf_counter()
        ep_latents = []
        for start in range(0, T, batch_size):
            batch = _np.ascontiguousarray(frames_chw[start : start + batch_size])
            lats = state_embedder.embed(batch)
            ep_latents.append(_np.asarray(lats))
        ep_arr = _np.concatenate(ep_latents, axis=0)  # (T, D)
        ep_to_state[ep_hash] = ep_arr
        print(
            f"{tag} {ep_hash[:8]}: {T} frames → {ep_arr.shape[1]}d "
            f"in {_time.perf_counter() - t_ep:.1f}s"
        )

    loader.join()

    if not ep_to_state:
        print(f"{tag} WARNING: no episodes embedded")

    lat_dir = Path(output_dir) / "latents_v2" / task_name
    lat_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(lat_dir / "state.npz")
    _np.savez_compressed(out_path, **ep_to_state)
    training_outputs_volume.commit()

    print(
        f"{tag} done — {len(ep_to_state)} episodes, "
        f"saved to {out_path} in {_time.perf_counter() - t_total:.1f}s"
    )
    return out_path


# ---------------------------------------------------------------------------
# Action embedding — CPU
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=None,
    cpu=4,
    memory=16384,
    timeout=7200,
    secrets=_SHARED_SECRETS,
    volumes={
        DEMINF_V2_MOUNT: deminf_v2_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def _embed_action_shards(
    shard_pairs: list[tuple[str, str]],
    task_name: str,
    run_name: str,
    hydra_args: tuple[str, ...],
    action_mean: list,
    action_std: list,
    git_remote: str,
    git_commit: str,
    output_dir: str,
    hf_token: str = "",
    batch_size: int = 256,
) -> str:
    """Embed action latents from NPZ shards (gaussian/CPU). Returns path to action.npz.

    For VAE checkpoint mode use _embed_action_shards_gpu (L40S, producer-consumer).
    """
    import time as _time
    import numpy as _np
    import torch

    _boot_container(git_remote, git_commit, hf_token)

    from egomimic.curation.config import (
        apply_curation_seed,
        select_embedder_settings,
        select_seed,
        select_tensor_keys,
    )
    from egomimic.curation.episode_pipeline import build_embedders

    cfg = _load_cfg(hydra_args)
    apply_curation_seed(select_seed(cfg))
    embed_cfg = select_embedder_settings(cfg)
    device = torch.device("cpu")
    action_key, _ = select_tensor_keys(cfg)

    action_embedder, _ = build_embedders(
        embed_cfg,
        _np.asarray(action_mean, dtype=_np.float32),
        _np.asarray(action_std, dtype=_np.float32),
        device,
        select_seed(cfg),
    )

    tag = f"[{task_name}][action]"
    t_total = _time.perf_counter()
    ep_to_action: dict[str, "_np.ndarray"] = {}

    for _, npz_path in shard_pairs:
        try:
            npz = _np.load(npz_path, allow_pickle=True)
            ep_hash = str(npz["episode_hash"])
            actions = npz[action_key].astype(_np.float32)  # (T, flat_dim)
        except Exception as exc:
            print(f"{tag} failed to read NPZ {npz_path}: {exc}")
            continue

        T = len(actions)
        if T == 0:
            print(f"{tag} {ep_hash[:8]}: 0 actions — skipping")
            continue

        t_ep = _time.perf_counter()
        ep_latents = []
        for start in range(0, T, batch_size):
            batch = actions[start : start + batch_size]
            lats = action_embedder.embed(batch)
            ep_latents.append(_np.asarray(lats))

        ep_arr = _np.concatenate(ep_latents, axis=0)  # (T, D)
        ep_to_action[ep_hash] = ep_arr
        print(
            f"{tag} {ep_hash[:8]}: {T} actions → {ep_arr.shape[1]}d "
            f"in {_time.perf_counter() - t_ep:.1f}s"
        )

    if not ep_to_action:
        print(f"{tag} WARNING: no episodes embedded")

    lat_dir = Path(output_dir) / "latents_v2" / task_name
    lat_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(lat_dir / "action.npz")
    _np.savez_compressed(out_path, **ep_to_action)
    training_outputs_volume.commit()

    print(
        f"{tag} done — {len(ep_to_action)} episodes, "
        f"saved to {out_path} in {_time.perf_counter() - t_total:.1f}s"
    )
    return out_path


# ---------------------------------------------------------------------------
# Action embedding — GPU (L40S, CheckpointActionEmbedder only)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    timeout=7200,
    secrets=_SHARED_SECRETS,
    volumes={
        DEMINF_V2_MOUNT: deminf_v2_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
)
def _embed_action_shards_gpu(
    shard_pairs: list[tuple[str, str]],
    task_name: str,
    run_name: str,
    hydra_args: tuple[str, ...],
    action_mean: list,
    action_std: list,
    git_remote: str,
    git_commit: str,
    output_dir: str,
    hf_token: str = "",
    batch_size: int = 256,
    prefetch: int = 8,
) -> str:
    """Embed action latents using a VAE checkpoint on GPU. Returns path to action.npz.

    Use this variant when the config specifies a CheckpointActionEmbedder (VAE ckpt).
    Producer-consumer: background thread loads NPZ actions into a bounded queue
    while the GPU thread batches and embeds. For the CPU gaussian case use
    _embed_action_shards instead (no GPU needed, already fast).
    """
    import queue as _queue
    import threading as _threading
    import time as _time
    import numpy as _np
    import torch

    _boot_container(git_remote, git_commit, hf_token)

    from egomimic.curation.config import (
        apply_curation_seed,
        select_embedder_settings,
        select_seed,
        select_tensor_keys,
    )
    from egomimic.curation.episode_pipeline import build_embedders

    cfg = _load_cfg(hydra_args)
    apply_curation_seed(select_seed(cfg))
    embed_cfg = select_embedder_settings(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_key, _ = select_tensor_keys(cfg)

    action_embedder, _ = build_embedders(
        embed_cfg,
        _np.asarray(action_mean, dtype=_np.float32),
        _np.asarray(action_std, dtype=_np.float32),
        device,
        select_seed(cfg),
    )

    tag = f"[{task_name}][action-gpu]"
    t_total = _time.perf_counter()

    # ── producer: background thread loads NPZ action arrays ──────────────────
    loaded_q: "_queue.Queue[object]" = _queue.Queue(maxsize=prefetch)

    def _loader() -> None:
        for _, npz_path in shard_pairs:
            try:
                npz = _np.load(npz_path, allow_pickle=True)
                ep_hash = str(npz["episode_hash"])
                actions = npz[action_key].astype(_np.float32)
            except Exception as exc:
                print(f"{tag} failed to read NPZ {npz_path}: {exc}")
                continue
            if len(actions) == 0:
                print(f"{tag} {ep_hash[:8]}: 0 actions — skipping")
                continue
            loaded_q.put((ep_hash, actions))
        loaded_q.put(_DONE)

    loader = _threading.Thread(target=_loader, daemon=True)
    loader.start()

    # ── consumer: GPU thread embeds action batches ────────────────────────────
    ep_to_action: dict[str, "_np.ndarray"] = {}
    while True:
        item = loaded_q.get()
        if item is _DONE:
            break
        ep_hash, actions = item
        T = len(actions)
        t_ep = _time.perf_counter()
        ep_latents = []
        for start in range(0, T, batch_size):
            batch = actions[start : start + batch_size]
            lats = action_embedder.embed(batch)
            ep_latents.append(_np.asarray(lats))
        ep_arr = _np.concatenate(ep_latents, axis=0)  # (T, D)
        ep_to_action[ep_hash] = ep_arr
        print(
            f"{tag} {ep_hash[:8]}: {T} actions → {ep_arr.shape[1]}d "
            f"in {_time.perf_counter() - t_ep:.1f}s"
        )

    loader.join()

    if not ep_to_action:
        print(f"{tag} WARNING: no episodes embedded")

    lat_dir = Path(output_dir) / "latents_v2" / task_name
    lat_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(lat_dir / "action.npz")
    _np.savez_compressed(out_path, **ep_to_action)
    training_outputs_volume.commit()

    print(
        f"{tag} done — {len(ep_to_action)} episodes, "
        f"saved to {out_path} in {_time.perf_counter() - t_total:.1f}s"
    )
    return out_path
