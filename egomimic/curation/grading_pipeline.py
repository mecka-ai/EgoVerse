"""Per-episode featurise pass + npz cache for k-NN action-consistency grading.

GPU shards call ``run_featurize_episodes`` to write one ``<hash>.npz`` per
episode under ``<features_root>/<feature_version>/<task>/``; the CPU scorer
loads them with ``load_episode_features`` (which derives chunk geometry and
drops the raw chunks). Caches are keyed by ``feature_version``, so scoring
re-runs and metric iteration never touch a GPU.
"""

from __future__ import annotations

import json
import logging
import queue as _queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from egomimic.curation.config import CurationLoaderSettings
from egomimic.curation.embedders import StateEmbedder
from egomimic.curation.knn_grading import (
    EpisodeFeatures,
    KnnGradeSettings,
    compute_chunk_features,
)

if TYPE_CHECKING:
    from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GradingFeaturizeSettings:
    """``featurize`` block of knn_grade.yaml (+ top-level feature_version)."""

    stride: int = 6
    image_key: str = "observations.images.front_img_1"
    action_key: str = "actions_cartesian"
    proprio_key: str = "observations.state.ee_pose"
    feature_version: str = "v1"
    backbone: str = "dinov3"
    dinov3_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    dinov3_dtype: str = "float16"

    @classmethod
    def from_cfg(cls, cfg: Any) -> "GradingFeaturizeSettings":
        """Build from a composed knn_grade Hydra config."""
        from omegaconf import OmegaConf

        f = OmegaConf.select(cfg, "featurize", default=None)
        d = OmegaConf.to_container(f, resolve=True) if f is not None else {}
        state_image = d.pop("state_image", {}) or {}
        dinov3 = state_image.get("dinov3", {}) or {}
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if state_image.get("backbone"):
            kwargs["backbone"] = str(state_image["backbone"]).lower().strip()
        if dinov3.get("model_name"):
            kwargs["dinov3_model_name"] = str(dinov3["model_name"])
        if dinov3.get("dtype"):
            kwargs["dinov3_dtype"] = str(dinov3["dtype"])
        version = OmegaConf.select(cfg, "feature_version", default=None)
        if version is not None:
            kwargs["feature_version"] = str(version)
        return cls(**kwargs)


def build_grading_featurizer(
    settings: GradingFeaturizeSettings,
    device: Any,
    image_batch_size: int = 512,
    seed: int = 42,
) -> StateEmbedder:
    """Frozen image backbone returning *unprojected* pooled features."""
    featurizer = StateEmbedder(
        mode="image",
        device=device,
        image_batch_size=image_batch_size,
        image_backbone=settings.backbone,
        dinov3_model_name=settings.dinov3_model_name,
        dinov3_dtype=settings.dinov3_dtype,
        seed=seed,
        project=False,
    )
    featurizer.fit([])
    return featurizer


# --------------------------------------------------------------------------- #
# npz cache IO
# --------------------------------------------------------------------------- #
def task_feature_dir(
    features_root: Path | str, feature_version: str, task: str
) -> Path:
    return Path(features_root) / feature_version / task


def episode_feature_path(
    features_root: Path | str, feature_version: str, task: str, ep_hash: str
) -> Path:
    return task_feature_dir(features_root, feature_version, task) / f"{ep_hash}.npz"


def save_episode_features(
    path: Path,
    image_feats: np.ndarray,
    proprio: np.ndarray,
    chunks: np.ndarray,
    frame_idx: np.ndarray,
    n_logical: int,
    meta: dict[str, Any],
) -> None:
    """Atomic-ish npz write (tmp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy appends ".npz" to names that lack it — keep the tmp name compliant.
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        image_feats=image_feats.astype(np.float16),
        proprio=proprio.astype(np.float32),
        chunks=chunks.astype(np.float16),
        frame_idx=frame_idx.astype(np.int64),
        n_logical=np.int64(n_logical),
        meta=json.dumps(meta),
    )
    tmp.replace(path)


def load_episode_features(
    path: Path | str,
    ep_hash: str,
    knn_settings: KnnGradeSettings,
) -> EpisodeFeatures:
    """Load one cached episode and derive chunk geometry (drops raw chunks)."""
    with np.load(path, allow_pickle=False) as data:
        chunks = data["chunks"].astype(np.float32)
        return EpisodeFeatures(
            ep_hash=ep_hash,
            image_feats=data["image_feats"].astype(np.float32),
            proprio=data["proprio"].astype(np.float32),
            chunk_feats=compute_chunk_features(chunks, knn_settings),
            frame_idx=data["frame_idx"].astype(np.int64),
            ep_len=int(data["n_logical"]),
        )


# --------------------------------------------------------------------------- #
# Featurise pass (GPU shard worker body)
# --------------------------------------------------------------------------- #
def _load_grading_episode(
    item: tuple[str, "ZarrDataset", GradingFeaturizeSettings, int],
) -> tuple[str, dict | None]:
    """Load + transform one episode (CPU-only, runs in thread pool)."""
    ep_hash, zarr_ds, settings, decode_workers = item
    try:
        collected = zarr_ds.collect_grading_episode(
            action_key=settings.action_key,
            image_key=settings.image_key,
            proprio_key=settings.proprio_key,
            image_decode_workers=decode_workers,
            frame_stride=settings.stride,
        )
        return ep_hash, collected
    finally:
        zarr_ds._zarr_bulk_cache = None


def run_featurize_episodes(
    episodes: dict[str, "ZarrDataset"],
    featurizer: StateEmbedder,
    loader: CurationLoaderSettings,
    settings: GradingFeaturizeSettings,
    out_dir: Path,
    *,
    skip_existing: bool = True,
    progress: str | None = None,
) -> list[dict[str, Any]]:
    """
    Featurise every episode and write one npz per episode to ``out_dir``.

    Producer (thread pool): loads/transforms episodes (zarr read, JPEG decode
    at ``settings.stride``). Consumer (main thread): embeds frames on the GPU
    and writes the cache file. Returns a manifest of
    ``{"hash", "path", "n_states", "n_logical", "skipped"}`` entries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    items = []
    for ep_hash, zarr_ds in episodes.items():
        path = out_dir / f"{ep_hash}.npz"
        if skip_existing and path.is_file():
            manifest.append({"hash": ep_hash, "path": str(path), "cached": True})
            continue
        items.append((ep_hash, zarr_ds, settings, loader.pass2_image_decode_workers))

    logger.info(
        "Featurize: %d episodes to embed (%d cached) | stride=%d, workers=%d",
        len(items),
        len(manifest),
        settings.stride,
        loader.episode_workers,
    )
    if not items:
        return manifest

    ep_queue: _queue.Queue = _queue.Queue(maxsize=loader.frame_queue_maxsize)

    def _producer() -> None:
        try:
            with ThreadPoolExecutor(max_workers=loader.episode_workers) as pool:
                futures = {
                    pool.submit(_load_grading_episode, item): item[0] for item in items
                }
                for future in as_completed(futures):
                    ep_hash = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.error("episode %s load failed: %s", ep_hash[:8], exc)
                        result = (ep_hash, None)
                    ep_queue.put(result)  # blocks when full → backpressure
        finally:
            ep_queue.put(None)

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    meta = asdict(settings)
    n_done = n_skipped = 0
    t0 = time.perf_counter()
    while True:
        item = ep_queue.get()
        if item is None:
            break
        ep_hash, collected = item
        if collected is None:
            n_skipped += 1
            manifest.append({"hash": ep_hash, "skipped": True})
            continue

        feats = featurizer.embed(collected["images"])
        path = out_dir / f"{ep_hash}.npz"
        save_episode_features(
            path,
            image_feats=feats,
            proprio=collected["proprio"],
            chunks=collected["actions"],
            frame_idx=collected["frame_idx"],
            n_logical=collected["n_logical"],
            meta=meta,
        )
        n_done += 1
        manifest.append(
            {
                "hash": ep_hash,
                "path": str(path),
                "n_states": int(len(collected["frame_idx"])),
                "n_logical": int(collected["n_logical"]),
            }
        )
        if progress and n_done % 10 == 0:
            elapsed = time.perf_counter() - t0
            logger.info(
                "%s %d/%d episodes (%.1f eps/min)",
                progress,
                n_done,
                len(items),
                60.0 * n_done / max(elapsed, 1e-9),
            )

    producer_thread.join()
    logger.info(
        "Featurize done: %d embedded, %d skipped, %d cached in %.1fs",
        n_done,
        n_skipped,
        sum(1 for m in manifest if m.get("cached")),
        time.perf_counter() - t0,
    )
    return manifest
