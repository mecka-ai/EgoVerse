"""Per-episode bulk zarr load, transform, and embed for DemInf curation."""

from __future__ import annotations

import logging
import math
import queue as _queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from egomimic.curation.config import CurationLoaderSettings, EmbedderSettings
from egomimic.curation.embedders import (
    ActionEmbedder,
    CheckpointActionEmbedder,
    OATActionEmbedder,
    StateEmbedder,
)

if TYPE_CHECKING:
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

logger = logging.getLogger(__name__)


def _release_episode_cache(zarr_ds: ZarrDataset) -> None:
    zarr_ds._zarr_bulk_cache = None


def build_embedders(
    embed_cfg: EmbedderSettings,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: Any,
    seed: int,
    *,
    global_frame_batch_size: int = 512,
) -> tuple[ActionEmbedder | OATActionEmbedder | CheckpointActionEmbedder, StateEmbedder]:
    """Construct fitted action and image embedders from config and precomputed norm stats.

    Three action embedder types are supported (``embed_cfg.action_embedder.type``):

    ``gaussian`` (default):
        Gaussian normalisation with precomputed mean/std + random orthogonal projection.

    ``oat``:
        Loads a trained OAT tokenizer checkpoint and uses the encoder to produce
        continuous pre-quantization latents for KSG mutual information estimation.
        Requires ``action_embedder.checkpoint_path`` and the OAT architecture
        config blocks (``oat_encoder_cfg``, ``oat_decoder_cfg``, ``oat_quantizer_cfg``).

    ``checkpoint`` (legacy):
        General model checkpoint with a configurable encode method.
    """
    ae_cfg = getattr(embed_cfg, "action_embedder", None)
    ae_type = ae_cfg.type.lower() if ae_cfg is not None else "gaussian"

    if ae_type == "oat":
        if ae_cfg is None or not ae_cfg.checkpoint_path:
            raise ValueError(
                "action_embedder.type=oat requires action_embedder.checkpoint_path to be set"
            )
        if not ae_cfg.oat_encoder_cfg or not ae_cfg.oat_decoder_cfg or not ae_cfg.oat_quantizer_cfg:
            raise ValueError(
                "action_embedder.type=oat requires oat_encoder_cfg, oat_decoder_cfg, "
                "and oat_quantizer_cfg to be set in the curation model config."
            )
        action_embedder: ActionEmbedder | OATActionEmbedder = OATActionEmbedder(
            checkpoint_path=ae_cfg.checkpoint_path,
            encoder_cfg=ae_cfg.oat_encoder_cfg,
            decoder_cfg=ae_cfg.oat_decoder_cfg,
            quantizer_cfg=ae_cfg.oat_quantizer_cfg,
            device=device,
            latent_dim=embed_cfg.latent_dim,
            action_chunk_size=ae_cfg.oat_action_chunk_size,
            action_dim=ae_cfg.oat_action_dim,
            seed=seed,
        )
        action_embedder.fit([])
    elif ae_type == "checkpoint":
        if ae_cfg is None or not ae_cfg.checkpoint_path:
            raise ValueError(
                "action_embedder.type=checkpoint requires action_embedder.checkpoint_path to be set"
            )
        action_embedder = CheckpointActionEmbedder(
            checkpoint_path=ae_cfg.checkpoint_path,
            device=device,
            latent_dim=embed_cfg.latent_dim,
            encode_method=ae_cfg.encode_method,
        )
        action_embedder.fit([])
    else:
        # Default: gaussian Gaussian normalisation + random projection
        action_embedder = ActionEmbedder(
            latent_dim=embed_cfg.latent_dim,
            seed=seed,
            norm_min_std=embed_cfg.norm_min_std,
        )
        action_embedder.set_precomputed_stats(action_mean, action_std)
        action_embedder.fit([])

    img = embed_cfg.state_image
    state_embedder = StateEmbedder(
        mode="image",
        latent_dim=embed_cfg.latent_dim,
        device=device,
        image_batch_size=global_frame_batch_size,
        image_backbone=img.backbone,
        dinov3_model_name=img.dinov3_model_name,
        dinov3_dtype=img.dinov3_dtype,
        seed=seed,
    )
    state_embedder.fit([])
    return action_embedder, state_embedder


def _load_episode(
    item: tuple[str, "ZarrDataset", str, str, int],
) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load and decode one episode (CPU-only, runs in thread pool)."""
    episode_hash, zarr_ds, action_key, image_key, image_decode_workers = item
    t0 = time.perf_counter()
    try:
        actions, images, frame_idx = zarr_ds.collect_curation_episode(
            action_key=action_key,
            image_key=image_key,
            image_decode_workers=image_decode_workers,
            return_frame_indices=True,
        )
        if actions is None or images is None:
            logger.debug("episode %s: skip (None data returned)", episode_hash[:8])
            return episode_hash, None, None, None
        t = actions.shape[0]
        flat_actions = actions.reshape(t, -1).astype(np.float32)
        img_f32 = images.astype(np.float32)
        elapsed = time.perf_counter() - t0
        logger.debug(
            "episode %s loaded in %.2fs: images %s dtype=%s | actions %s dtype=%s",
            episode_hash[:8], elapsed,
            img_f32.shape, img_f32.dtype,
            flat_actions.shape, flat_actions.dtype,
        )
        del actions, images
        return episode_hash, flat_actions, img_f32, frame_idx
    finally:
        _release_episode_cache(zarr_ds)


def run_pass2_embed_episodes(
    episodes: dict[str, "ZarrDataset"],
    scored_hashes: set[str],
    action_key: str,
    image_key: str,
    loader: CurationLoaderSettings,
    action_embedder: ActionEmbedder | OATActionEmbedder,
    state_embedder: StateEmbedder,
    *,
    progress: str | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[int], list[np.ndarray]]:
    """
    Pass 2: producer-consumer pipeline.

    Producer (thread pool, ``episode_workers`` threads):
        Loads episodes in parallel via ``_load_episode`` (zarr read + JPEG
        decode + transform). Pushes completed episodes onto a bounded queue.
        Blocks when the queue is full, providing backpressure so RAM stays
        bounded to ``frame_queue_maxsize`` loaded episodes at once.

    Consumer (main thread, GPU):
        Pulls episodes off the queue and accumulates their frames. Once the
        frame buffer reaches ``global_frame_batch_size``, it embeds that batch
        (image preprocess → GPU forward, action normalize → project) and clears
        the buffer. Loading and embedding run concurrently — the GPU never waits
        for a full episode chunk to finish loading.

    Returns ``(state_latents, action_latents, episode_hashes, ep_lengths,
    raw_frame_indices)`` — the last is one ``(T,)`` int64 array per episode of
    RAW video-frame indices (pre pause/pose filtering), row-aligned with that
    episode's latents.
    """
    items = [
        (h, episodes[h], action_key, image_key, loader.pass2_image_decode_workers)
        for h in episodes
        if h in scored_hashes
    ]
    n_items = len(items)
    global_bs = loader.global_frame_batch_size

    logger.info(
        "Pass2 embed: %d episodes | global_frame_batch=%d, episode_workers=%d, "
        "queue_maxsize=%d (episodes), image_decode_workers=%d",
        n_items, global_bs, loader.episode_workers,
        loader.frame_queue_maxsize, loader.pass2_image_decode_workers,
    )

    # ------------------------------------------------------------------
    # Producer thread: submit all load jobs, push results onto the queue.
    # Queue is bounded so producers block when the consumer falls behind,
    # keeping at most frame_queue_maxsize loaded episodes in RAM.
    # ------------------------------------------------------------------
    ep_queue: _queue.Queue = _queue.Queue(maxsize=loader.frame_queue_maxsize)

    def _producer() -> None:
        try:
            with ThreadPoolExecutor(max_workers=loader.episode_workers) as pool:
                futures = {pool.submit(_load_episode, item): item[0] for item in items}
                for future in as_completed(futures):
                    ep_hash = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.error("episode %s load failed: %s", ep_hash[:8], exc)
                        result = (ep_hash, None, None, None)
                    ep_queue.put(result)  # blocks when queue full → backpressure
        finally:
            ep_queue.put(None)  # sentinel: all episodes submitted

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    # ------------------------------------------------------------------
    # Consumer: accumulate frames, flush in global_bs batches, scatter back.
    # ------------------------------------------------------------------
    state_latents: list[np.ndarray] = []
    action_latents: list[np.ndarray] = []
    hashes: list[str] = []
    ep_lengths: list[int] = []
    raw_indices: list[np.ndarray] = []
    n_skipped = 0
    t0 = time.perf_counter()

    # Frame buffer: whole episodes queued here before embedding
    buf_imgs: list[np.ndarray] = []
    buf_acts: list[np.ndarray] = []
    buf_ep_meta: list[tuple[str, int, np.ndarray]] = []  # (hash, n_frames, raw_idx) in order
    buf_n_frames = 0

    def _flush() -> None:
        nonlocal buf_n_frames
        if not buf_imgs:
            return

        all_images = np.concatenate(buf_imgs, axis=0)
        all_actions = np.concatenate(buf_acts, axis=0)
        n = all_images.shape[0]
        n_batches = math.ceil(n / global_bs)
        t_flush = time.perf_counter()

        logger.info(
            "flush: %d frames from %d episodes → %d frame-batch(es) of %d "
            "| [images] %s | [actions] %s",
            n, len(buf_ep_meta), n_batches, global_bs,
            all_images.shape, all_actions.shape,
        )

        s_parts: list[np.ndarray] = []
        a_parts: list[np.ndarray] = []
        for batch_idx, start in enumerate(range(0, n, global_bs)):
            end = min(start + global_bs, n)
            img_batch = all_images[start:end]
            act_batch = all_actions[start:end]
            tb = time.perf_counter()
            logger.info(
                "  frame-batch %d/%d [%d:%d/%d] | [images] %s | [actions] %s",
                batch_idx + 1, n_batches, start, end, n,
                img_batch.shape, act_batch.shape,
            )
            s = state_embedder.embed(img_batch)
            t_img = time.perf_counter()
            logger.info("    [images]  → %s in %.2fs", s.shape, t_img - tb)
            a = action_embedder.embed(act_batch)
            t_act = time.perf_counter()
            logger.info("    [actions] → %s in %.2fs", a.shape, t_act - t_img)
            s_parts.append(s)
            a_parts.append(a)

        del all_images, all_actions
        s_all = np.concatenate(s_parts, axis=0)
        a_all = np.concatenate(a_parts, axis=0)

        elapsed_flush = time.perf_counter() - t_flush
        logger.info(
            "flush done: %.2fs (%.0f frames/s)",
            elapsed_flush, n / elapsed_flush if elapsed_flush > 0 else 0,
        )

        # Scatter latents back to per-episode slices
        offset = 0
        for h, ep_n, fidx in buf_ep_meta:
            state_latents.append(s_all[offset : offset + ep_n])
            action_latents.append(a_all[offset : offset + ep_n])
            hashes.append(h)
            ep_lengths.append(ep_n)
            raw_indices.append(fidx)
            offset += ep_n

        buf_imgs.clear()
        buf_acts.clear()
        buf_ep_meta.clear()
        buf_n_frames = 0

    pbar = (
        tqdm(total=n_items, desc=progress, unit="ep", dynamic_ncols=True)
        if progress is not None
        else None
    )

    while True:
        item = ep_queue.get()
        if item is None:  # sentinel: producer finished
            break

        ep_hash, flat_actions, img_f32, frame_idx = item
        if pbar is not None:
            pbar.update(1)

        if flat_actions is None or img_f32 is None:
            n_skipped += 1
            logger.debug("episode %s: skipped", ep_hash[:8])
            continue

        n_ep = img_f32.shape[0]
        if frame_idx is None:
            frame_idx = np.arange(n_ep, dtype=np.int64)
        buf_imgs.append(img_f32)
        buf_acts.append(flat_actions)
        buf_ep_meta.append((ep_hash, n_ep, np.asarray(frame_idx, dtype=np.int64)))
        buf_n_frames += n_ep
        logger.debug(
            "received episode %s: %d frames (buffer: %d frames, %d eps)",
            ep_hash[:8], n_ep, buf_n_frames, len(buf_ep_meta),
        )

        if buf_n_frames >= global_bs:
            _flush()

    if pbar is not None:
        pbar.close()

    _flush()  # remaining frames that didn't fill a full batch
    producer_thread.join()

    elapsed = time.perf_counter() - t0
    n_frames_total = sum(ep_lengths)
    logger.info(
        "Pass2 done: %d/%d episodes embedded, %d skipped, %d frames, "
        "%.1fs (%.1f eps/s, %.0f frames/s)",
        len(hashes), n_items, n_skipped, n_frames_total, elapsed,
        len(hashes) / elapsed if elapsed > 0 else 0,
        n_frames_total / elapsed if elapsed > 0 else 0,
    )

    return state_latents, action_latents, hashes, ep_lengths, raw_indices
