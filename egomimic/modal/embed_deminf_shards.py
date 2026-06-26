"""Phase 2 of the DemInf v2 pipeline: embed MP4+NPZ shards into state/action latents.

Architecture (inspired by GlobalShuffleShardDataset in egomimic/rldb/gs_shard_dataset.py)
-----------
  Volume (network)
      │
      │  background downloader thread
      │  sliding window of n_dl_threads concurrent shutil.copy2 calls
      │  pool_size bounded queue provides backpressure
      ▼
  Local /cache SSD  ← per-shard copies live here until embedding is done
      │
      │  GPU consumer thread
      │  decodes MP4 frame-by-frame (never holds full episode in RAM)
      │  accumulates batch_size frames → state_embedder.embed()
      │  after embedding: delete local copies in background thread
      ▼
  per-episode latent cache (local tmpdir .npy files)
      │
      ▼
  state.npz / action.npz  →  training_outputs_volume.commit()

Peak RAM = pool_size × one_shard_size_compressed (~30 MB each) + batch_size × one_frame (~600 KB)
This is O(100 MB) regardless of dataset size — RAM no longer grows with episode count.
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


# ---------------------------------------------------------------------------
# Container boot helpers
# ---------------------------------------------------------------------------


def _ep_hash_str(npz) -> str:
    """Decode the episode_hash scalar from a shard npz to a clean str.

    Shards store it as a numpy bytes scalar; ``str()`` on that yields the mangled
    ``np.bytes_(b'...')`` repr. Unwrap the 0-d array and decode bytes → str so the
    latent key is the bare episode hash (needed to map back to the zarr dir).
    """
    v = npz["episode_hash"]
    v = v.item() if hasattr(v, "item") else v
    return v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)


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


def _local_cache_dir() -> Path:
    """Fast local SSD — prefer /cache (Modal's ephemeral NVMe), fall back to /tmp."""
    for candidate in ["/cache", os.environ.get("TMPDIR"), "/tmp"]:
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return Path("/tmp")


# ---------------------------------------------------------------------------
# Frame-by-frame MP4 decoder (generator — never holds full episode in RAM)
# ---------------------------------------------------------------------------


def _iter_mp4_frames(mp4_path: str):
    """Yield (3, H, W) float32 [0,1] frames one at a time via PyAV.

    Keeps only ONE frame in RAM at a time (~600 KB). Never use tio.read_video or
    a full np.stack — those load the entire episode as a (T,C,H,W) tensor (~1.5 GB).
    """
    import av
    import numpy as np

    container = av.open(str(mp4_path))
    try:
        for frame in container.decode(video=0):
            arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
            yield arr.transpose(2, 0, 1).astype(np.float32) / 255.0
    finally:
        container.close()


# ---------------------------------------------------------------------------
# Shard downloader (volume → local SSD, sliding window of concurrent copies)
# ---------------------------------------------------------------------------


def _run_shard_downloader(
    shard_pairs: list[tuple[str, str]],
    local_dir: Path,
    path_q: "queue.Queue",
    n_dl_threads: int,
    tag: str,
    _np,
) -> None:
    """Background thread: copies shards from network volume to local SSD.

    Uses a sliding window of n_dl_threads concurrent shutil.copy2 calls so
    the network volume is kept saturated. path_q.put() blocks when pool_size
    shards are already ready — this is the backpressure mechanism.

    After copying, reads episode_hash from the local NPZ so the GPU consumer
    doesn't touch the network volume at all.
    """
    import concurrent.futures
    import shutil
    import uuid as _uuid

    def _copy_one(mp4_src: str, npz_src: str):
        uid = _uuid.uuid4().hex[:8]
        local_mp4 = local_dir / f"embed_{uid}.mp4"
        local_npz = local_dir / f"embed_{uid}.npz"
        shutil.copy2(mp4_src, local_mp4)
        shutil.copy2(npz_src, local_npz)
        npz = _np.load(str(local_npz), allow_pickle=True)
        ep_hash = _ep_hash_str(npz)
        return ep_hash, local_mp4, local_npz

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=n_dl_threads)
    pending = list(shard_pairs)
    futures: list = []
    idx = 0

    def _submit():
        nonlocal idx
        if idx < len(pending):
            mp4, npz = pending[idx]
            futures.append(pool.submit(_copy_one, mp4, npz))
            idx += 1

    for _ in range(min(n_dl_threads, len(pending))):
        _submit()

    while futures:
        fut = futures.pop(0)
        try:
            ep_hash, local_mp4, local_npz = fut.result()
            _submit()
            path_q.put((ep_hash, local_mp4, local_npz))  # blocks when pool full
        except Exception as exc:
            print(f"{tag} shard copy failed: {exc}", flush=True)
            _submit()

    pool.shutdown(wait=False, cancel_futures=True)
    path_q.put(_DONE)


# ---------------------------------------------------------------------------
# Shared embedder builder + latent cache helpers
# ---------------------------------------------------------------------------


def _make_embedders(hydra_args, action_mean, action_std, device, _np, torch):
    from egomimic.curation.config import (
        apply_curation_seed,
        select_embedder_settings,
        select_seed,
    )
    from egomimic.curation.episode_pipeline import build_embedders

    cfg = _load_cfg(hydra_args)
    apply_curation_seed(select_seed(cfg))
    embed_cfg = select_embedder_settings(cfg)

    action_embedder, state_embedder = build_embedders(
        embed_cfg,
        _np.asarray(action_mean, dtype=_np.float32),
        _np.asarray(action_std, dtype=_np.float32),
        device,
        select_seed(cfg),
    )
    # NPZ shards always store actions under "action" (see build_deminf_shards.py).
    # The config's action_key ("actions_cartesian") is the training-data key, not the shard key.
    return action_embedder, state_embedder


def _cache_write(cache_dir: Path, ep_hash: str, arr: "np.ndarray", _np) -> None:
    _np.save(str(cache_dir / f"{ep_hash}.npy"), arr)


def _cache_publish_flat(cache_dir: Path, out_dir: Path, mod: str, _np) -> int:
    """Stream per-episode ``{hash}.npy`` into a flat ``_<mod>.npy`` + ``_<mod>_manifest.json``.

    Rows are laid out in sorted-hash order; the flat array is filled via an
    open_memmap so peak RAM stays flat regardless of run size. The orchestrator's
    assemble step later merges the state/action flats into the canonical store.
    Returns the episode count.
    """
    import json as _json
    import shutil

    files = sorted((f for f in cache_dir.iterdir() if f.suffix == ".npy"), key=lambda p: p.stem)
    if not files:
        return 0

    eps: list[dict] = []
    n_rows = 0
    dim: int | None = None
    for f in files:
        a = _np.load(str(f), mmap_mode="r")
        T, D = int(a.shape[0]), int(a.shape[1])
        if dim is None:
            dim = D
        eps.append({"hash": f.stem, "row_start": n_rows, "n_frames": T})
        n_rows += T

    out_dir.mkdir(parents=True, exist_ok=True)
    flat = _np.lib.format.open_memmap(
        str(out_dir / f"_{mod}.npy"), mode="w+", dtype=_np.float32, shape=(n_rows, dim)
    )
    for ep, f in zip(eps, files):
        flat[ep["row_start"] : ep["row_start"] + ep["n_frames"]] = _np.load(str(f))
    flat.flush()
    del flat
    with open(out_dir / f"_{mod}_manifest.json", "w") as mf:
        _json.dump({"dim": dim, "n_rows": n_rows, "episodes": eps}, mf)
    shutil.rmtree(str(cache_dir))
    return len(eps)


# ---------------------------------------------------------------------------
# State embedding — GPU (H200)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="H200",
    cpu=16,
    memory=131072,
    ephemeral_disk=2 * 1024 * 1024,
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
    batch_size: int = 2048,
    pool_size: int = 4,
    n_dl_threads: int = 4,
) -> str:
    """Embed state (image) latents from MP4 shards. Returns path to state.npz.

    GlobalShuffle-style pipeline:
    1. Background thread downloads shards from network volume → local /cache SSD
       using a sliding window of n_dl_threads concurrent copies.
    2. GPU consumer decodes each shard frame-by-frame (peak RAM = one frame, ~600 KB),
       accumulates batch_size frames, runs state_embedder.embed(), writes latents to
       a local cache file.
    3. After embedding each shard, deletes local copies in a background thread so
       the GPU consumer never waits on disk I/O.
    4. At the end: collect cache files → savez_compressed → volume commit.

    Peak RAM = pool_size × compressed_shard_size (~30 MB) + batch_size × frame (~600 KB).
    Constant regardless of dataset size.
    """
    import queue as _queue
    import tempfile as _tempfile
    import threading as _threading
    import time as _time

    import numpy as _np
    import torch

    # Disable CUDA caching allocator — pure inference, no allocation reuse needed.
    # Without this, PyTorch holds freed VRAM in an internal free-list that grows
    # across hundreds of episodes and eventually causes OOM via unified memory spill.
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

    _boot_container(git_remote, git_commit, hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, state_embedder = _make_embedders(hydra_args, action_mean, action_std, device, _np, torch)

    tag = f"[{task_name}][state]"
    t_total = _time.perf_counter()
    local_dir = _local_cache_dir()
    cache_dir = Path(_tempfile.mkdtemp(prefix="state_lat_", dir=local_dir))

    # ── downloader: fills pool_size-bounded queue from network volume ─────────
    path_q: "_queue.Queue[object]" = _queue.Queue(maxsize=pool_size)

    dl_thread = _threading.Thread(
        target=_run_shard_downloader,
        args=(shard_pairs, local_dir, path_q, n_dl_threads, tag, _np),
        daemon=True,
    )
    dl_thread.start()

    # ── GPU consumer: decode frame-by-frame → batch → embed → cache ──────────
    # Timing decomposition (cumulative, to expose GPU starvation):
    #   cum_wait   — blocked on path_q.get() (downloader/network can't keep up)
    #   cum_decode — CPU PyAV decode + batch stacking (GPU idle between embeds)
    #   cum_embed  — state_embedder.embed() (GPU busy)
    # "GPU waiting for frames" = cum_wait + cum_decode.
    n_done = 0
    cum_wait = 0.0
    cum_decode = 0.0
    cum_embed = 0.0
    while True:
        t_get = _time.perf_counter()
        item = path_q.get()
        cum_wait += _time.perf_counter() - t_get
        if item is _DONE:
            break

        ep_hash, local_mp4, local_npz = item
        t_ep = _time.perf_counter()

        # Stream frames one at a time — accumulate into GPU batches
        frame_buf: list = []
        ep_latents: list = []
        T = 0
        ep_embed = 0.0

        for frame_chw in _iter_mp4_frames(str(local_mp4)):
            frame_buf.append(frame_chw)
            T += 1
            if len(frame_buf) == batch_size:
                batch = _np.ascontiguousarray(_np.stack(frame_buf))
                t_e = _time.perf_counter()
                ep_latents.append(_np.asarray(state_embedder.embed(batch)))
                ep_embed += _time.perf_counter() - t_e
                del batch, frame_buf
                frame_buf = []

        # Flush remaining frames
        if frame_buf:
            batch = _np.ascontiguousarray(_np.stack(frame_buf))
            t_e = _time.perf_counter()
            ep_latents.append(_np.asarray(state_embedder.embed(batch)))
            ep_embed += _time.perf_counter() - t_e
            del batch, frame_buf

        ep_total = _time.perf_counter() - t_ep
        ep_decode = max(ep_total - ep_embed, 0.0)
        cum_decode += ep_decode
        cum_embed += ep_embed

        # Delete local shard files in background — GPU moves to next immediately
        _p1, _p2 = local_mp4, local_npz
        _threading.Thread(
            target=lambda a=_p1, b=_p2: (a.unlink(missing_ok=True), b.unlink(missing_ok=True)),
            daemon=True,
        ).start()

        if not ep_latents:
            print(f"{tag} {ep_hash[:8]}: 0 frames decoded — skipping")
            continue

        ep_arr = _np.concatenate(ep_latents, axis=0)
        del ep_latents
        _cache_write(cache_dir, ep_hash, ep_arr, _np)
        del ep_arr

        n_done += 1
        print(
            f"{tag} {ep_hash[:8]}: {T} frames → {batch_size}-frame batches "
            f"in {ep_total:.1f}s (embed={ep_embed:.1f}s decode={ep_decode:.1f}s) "
            f"({n_done}/{len(shard_pairs)})"
        )
        if n_done <= 5 or n_done % 25 == 0:
            rss_gb = 0.0
            try:
                with open("/proc/self/status") as _sf:
                    for _ln in _sf:
                        if _ln.startswith("VmRSS:"):
                            rss_gb = int(_ln.split()[1]) / 1024 ** 2
                            break
            except Exception:
                rss_gb = -1.0
            vram_alloc_gb = torch.cuda.memory_allocated() / 1024 ** 3
            cum_busy = cum_wait + cum_decode + cum_embed
            gpu_pct = 100.0 * cum_embed / cum_busy if cum_busy > 0 else 0.0
            idle_pct = 100.0 - gpu_pct
            print(
                f"{tag} [mem/{n_done}] RSS={rss_gb:.1f}GB VRAM_alloc={vram_alloc_gb:.1f}GB "
                f"| GPU busy={gpu_pct:.0f}% idle={idle_pct:.0f}% "
                f"(embed={cum_embed:.0f}s decode={cum_decode:.0f}s shard_wait={cum_wait:.0f}s)",
                flush=True,
            )

    dl_thread.join()

    if n_done == 0:
        print(f"{tag} WARNING: no episodes embedded")

    lat_dir = Path(output_dir) / "latents" / task_name
    lat_dir.mkdir(parents=True, exist_ok=True)
    n_saved = _cache_publish_flat(cache_dir, lat_dir, "state", _np)
    training_outputs_volume.commit()
    out_path = str(lat_dir / "_state.npy")

    _cum_busy = cum_wait + cum_decode + cum_embed
    _gpu_pct = 100.0 * cum_embed / _cum_busy if _cum_busy > 0 else 0.0
    print(
        f"{tag} done — {n_saved} episodes, {out_path}, "
        f"total {_time.perf_counter() - t_total:.1f}s "
        f"| GPU busy={_gpu_pct:.0f}% "
        f"(embed={cum_embed:.0f}s decode={cum_decode:.0f}s shard_wait={cum_wait:.0f}s)"
    )
    return out_path


# ---------------------------------------------------------------------------
# Action embedding — CPU (gaussian, fast — no pool needed)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=None,
    cpu=4,
    memory=16384,
    ephemeral_disk=2 * 1024 * 1024,
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
    """Embed action latents (gaussian/CPU path). Returns path to action.npz.

    Gaussian embedding is a whitening transform — near-zero cost. No pool/staging
    needed. For VAE checkpoint mode use _embed_action_shards_gpu instead.
    """
    import time as _time

    import numpy as _np
    import torch

    _boot_container(git_remote, git_commit, hf_token)

    device = torch.device("cpu")
    action_embedder, _ = _make_embedders(
        hydra_args, action_mean, action_std, device, _np, torch
    )

    import tempfile as _tempfile

    tag = f"[{task_name}][action]"
    t_total = _time.perf_counter()
    cache_dir = Path(_tempfile.mkdtemp(prefix="action_lat_", dir=_local_cache_dir()))
    n_emb = 0

    for _, npz_path in shard_pairs:
        try:
            npz = _np.load(npz_path, allow_pickle=True)
            ep_hash = _ep_hash_str(npz)
            actions = npz["action"].astype(_np.float32)  # shard NPZ always uses "action"
        except Exception as exc:
            print(f"{tag} failed to read NPZ {npz_path}: {exc}")
            continue

        T = len(actions)
        if T == 0:
            print(f"{tag} {ep_hash[:8]}: 0 actions — skipping")
            continue

        ep_latents = []
        for start in range(0, T, batch_size):
            lats = action_embedder.embed(actions[start : start + batch_size])
            ep_latents.append(_np.asarray(lats))

        _cache_write(cache_dir, ep_hash, _np.concatenate(ep_latents, axis=0), _np)
        n_emb += 1

    if n_emb == 0:
        print(f"{tag} WARNING: no episodes embedded")

    lat_dir = Path(output_dir) / "latents" / task_name
    lat_dir.mkdir(parents=True, exist_ok=True)
    n_saved = _cache_publish_flat(cache_dir, lat_dir, "action", _np)
    training_outputs_volume.commit()
    out_path = str(lat_dir / "_action.npy")

    print(
        f"{tag} done — {n_saved} episodes, {out_path}, "
        f"total {_time.perf_counter() - t_total:.1f}s"
    )
    return out_path


# ---------------------------------------------------------------------------
# Action embedding — GPU (L40S, CheckpointActionEmbedder / VAE path)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    ephemeral_disk=2 * 1024 * 1024,
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
    pool_size: int = 8,
    n_dl_threads: int = 4,
) -> str:
    """Embed action latents via VAE checkpoint on GPU. Returns path to action.npz.

    Same GlobalShuffle-style pipeline as _embed_state_shards. Action NPZs are tiny
    (~1 MB each) so the downloader keeps the pool full with negligible overhead.
    Use _embed_action_shards for the CPU gaussian case.
    """
    import queue as _queue
    import tempfile as _tempfile
    import threading as _threading
    import time as _time

    import numpy as _np
    import torch

    _boot_container(git_remote, git_commit, hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_embedder, _ = _make_embedders(
        hydra_args, action_mean, action_std, device, _np, torch
    )

    tag = f"[{task_name}][action-gpu]"
    t_total = _time.perf_counter()
    local_dir = _local_cache_dir()
    cache_dir = Path(_tempfile.mkdtemp(prefix="action_lat_", dir=local_dir))

    # ── downloader: pre-fetch NPZ shards to local SSD ────────────────────────
    path_q: "_queue.Queue[object]" = _queue.Queue(maxsize=pool_size)

    dl_thread = _threading.Thread(
        target=_run_shard_downloader,
        args=(shard_pairs, local_dir, path_q, n_dl_threads, tag, _np),
        daemon=True,
    )
    dl_thread.start()

    # ── GPU consumer ──────────────────────────────────────────────────────────
    n_done = 0
    while True:
        item = path_q.get()
        if item is _DONE:
            break

        ep_hash, _local_mp4, local_npz = item
        t_ep = _time.perf_counter()

        try:
            npz = _np.load(str(local_npz), allow_pickle=True)
            actions = npz["action"].astype(_np.float32)  # shard NPZ always uses "action"
        except Exception as exc:
            print(f"{tag} failed to read NPZ {local_npz}: {exc}")
            _local_mp4.unlink(missing_ok=True)
            local_npz.unlink(missing_ok=True)
            continue

        T = len(actions)
        ep_latents = []
        for start in range(0, T, batch_size):
            lats = action_embedder.embed(actions[start : start + batch_size])
            ep_latents.append(_np.asarray(lats))

        del actions

        _p1, _p2 = _local_mp4, local_npz
        _threading.Thread(
            target=lambda a=_p1, b=_p2: (a.unlink(missing_ok=True), b.unlink(missing_ok=True)),
            daemon=True,
        ).start()

        ep_arr = _np.concatenate(ep_latents, axis=0)
        del ep_latents
        _cache_write(cache_dir, ep_hash, ep_arr, _np)
        del ep_arr

        n_done += 1
        print(
            f"{tag} {ep_hash[:8]}: {T} actions in {_time.perf_counter() - t_ep:.1f}s "
            f"({n_done}/{len(shard_pairs)})"
        )

    dl_thread.join()

    if n_done == 0:
        print(f"{tag} WARNING: no episodes embedded")

    lat_dir = Path(output_dir) / "latents" / task_name
    lat_dir.mkdir(parents=True, exist_ok=True)
    n_saved = _cache_publish_flat(cache_dir, lat_dir, "action", _np)
    training_outputs_volume.commit()
    out_path = str(lat_dir / "_action.npy")

    print(
        f"{tag} done — {n_saved} episodes, {out_path}, "
        f"total {_time.perf_counter() - t_total:.1f}s"
    )
    return out_path
