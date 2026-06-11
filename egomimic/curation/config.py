"""OmegaConf helpers for DemInf curation (Modal + local)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf


@dataclass(frozen=True)
class CurationLoaderSettings:
    """
    Per-dataset curation I/O (``data.curation_loader.<dataset>``).

    ``episode_workers``: threads loading episodes in parallel.
    ``pass2_image_decode_workers``: threads decoding JPEGs within each episode.
    ``frame_queue_maxsize``: max fully-loaded episodes held in the producer queue.
        Bounds peak RAM: frame_queue_maxsize × avg_episode_RAM.
        Producers block (backpressure) when the queue is full, so the GPU
        consumer controls how fast loading proceeds.
    ``global_frame_batch_size``: frames per GPU embed call. Frames from multiple
        episodes are concatenated and sliced at this stride regardless of episode
        boundaries. Set to fill GPU VRAM.
    """

    episode_workers: int = 4
    pass2_image_decode_workers: int = 2
    frame_queue_maxsize: int = 32
    global_frame_batch_size: int = 512


@dataclass(frozen=True)
class StateImageSettings:
    """State image backbone (``model.state_image``)."""

    backbone: str = "resnet18"
    dinov3_model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    dinov3_dtype: str = "float16"


@dataclass(frozen=True)
class ActionEmbedderSettings:
    """
    Action embedder type selection (``model.action_embedder``).

    type: ``gaussian``   — Gaussian normalisation + random orthogonal projection (default).
          ``checkpoint`` — Load a trained model from ``checkpoint_path`` and call
                          ``encode_method`` to produce action latents.
          ``oat``        — Load a trained OAT tokenizer checkpoint and use the encoder
                          to produce continuous pre-quantization latents suitable for
                          KSG mutual information estimation.

    For ``type=oat``:
      checkpoint_path:       Path to the OATTokenizerTrainer Lightning ``.ckpt``.
      oat_action_chunk_size: Timesteps per action chunk (must match training config).
      oat_action_dim:        Action dimensionality (must match training config).
      oat_encoder_cfg:       Encoder block from the OAT model YAML (as a dict).
      oat_decoder_cfg:       Decoder block (needed to instantiate OATTok).
      oat_quantizer_cfg:     Quantizer block.
    """

    type: str = "gaussian"
    checkpoint_path: str | None = None
    encode_method: str = "encode"
    oat_action_chunk_size: int = 100
    oat_action_dim: int = 12
    oat_encoder_cfg: dict | None = None
    oat_decoder_cfg: dict | None = None
    oat_quantizer_cfg: dict | None = None


@dataclass(frozen=True)
class LanguageConditioningSettings:
    """
    Language-conditioned DemInf scoring (``model.language_conditioning``).

    When ``enabled=false``, curation uses I(S; A) as before.

    When enabled, ``mode`` selects the estimator:
      ``concat``     — I(S, L; A): concatenate state + language latents, one KSG pass.
      ``stratified`` — I(S; A | L): partition timesteps by instruction text, KSG
                       within each language cluster, then assign per-sample MI.
    """

    enabled: bool = False
    mode: str = "concat"
    source: str = "qwen3"
    language_key: str = "annotations"
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    max_length: int = 512
    batch_size: int = 64
    dtype: str = "float16"
    stratified_min_cluster_size: int = 10


@dataclass(frozen=True)
class EmbedderSettings:
    """Model embedder settings (``model.*``)."""

    latent_dim: int = 32
    device: str = "cuda"
    norm_min_std: float = 1e-6
    state_image: StateImageSettings = field(default_factory=StateImageSettings)
    action_embedder: ActionEmbedderSettings = field(
        default_factory=ActionEmbedderSettings
    )
    language_conditioning: LanguageConditioningSettings = field(
        default_factory=LanguageConditioningSettings
    )


def select_seed(cfg: Any) -> int:
    """Global RNG seed from ``curate.yaml`` top-level ``seed``."""
    return int(OmegaConf.select(cfg, "seed", default=42))


def apply_curation_seed(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and Lightning from one value."""
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import lightning as L

        L.seed_everything(seed, workers=True)
    except ImportError:
        pass


def select_curation_loader(cfg: Any, dataset_name: str) -> CurationLoaderSettings:
    """Read ``data.curation_loader.<dataset_name>`` (required for Modal curation)."""
    base = f"data.curation_loader.{dataset_name}"
    if OmegaConf.select(cfg, base, default=None) is None:
        raise KeyError(
            f"Missing {base} in data config. "
            "Add a curation_loader block (see deminf_mecka.yaml)."
        )
    return CurationLoaderSettings(
        episode_workers=int(
            OmegaConf.select(cfg, f"{base}.episode_workers", default=4)
        ),
        pass2_image_decode_workers=int(
            OmegaConf.select(cfg, f"{base}.pass2_image_decode_workers", default=2)
        ),
        frame_queue_maxsize=int(
            OmegaConf.select(cfg, f"{base}.frame_queue_maxsize", default=32)
        ),
        global_frame_batch_size=int(
            OmegaConf.select(cfg, f"{base}.global_frame_batch_size", default=512)
        ),
    )


def select_state_image_settings(cfg: Any) -> StateImageSettings:
    """Read ``model.state_image`` backbone settings."""
    backbone = str(
        OmegaConf.select(cfg, "model.state_image.backbone", default="resnet18")
    ).lower().strip()
    if backbone == "dino":
        backbone = "dinov3"

    defaults = StateImageSettings()
    dinov3_model_name = defaults.dinov3_model_name
    dinov3_dtype = defaults.dinov3_dtype
    if backbone == "dinov3":
        dinov3 = OmegaConf.select(cfg, "model.state_image.dinov3", default={}) or {}
        dinov3_model_name = str(dinov3.get("model_name", dinov3_model_name))
        dinov3_dtype = str(dinov3.get("dtype", dinov3_dtype))

    return StateImageSettings(
        backbone=backbone,
        dinov3_model_name=dinov3_model_name,
        dinov3_dtype=dinov3_dtype,
    )


def select_action_embedder_settings(cfg: Any) -> ActionEmbedderSettings:
    """Read ``model.action_embedder`` settings; defaults to gaussian if absent."""
    block = OmegaConf.select(cfg, "model.action_embedder", default=None)
    if block is None:
        return ActionEmbedderSettings()
    ae_type = str(block.get("type", "gaussian")).lower().strip()
    ckpt = block.get("checkpoint_path", None)
    encode_method = str(block.get("encode_method", "encode"))

    # OAT-specific fields
    oat_chunk = int(block.get("oat_action_chunk_size", 100))
    oat_dim = int(block.get("oat_action_dim", 12))
    oat_enc = block.get("oat_encoder_cfg", None)
    oat_dec = block.get("oat_decoder_cfg", None)
    oat_qtz = block.get("oat_quantizer_cfg", None)

    # Convert OmegaConf sub-nodes to plain dicts for hydra.utils.instantiate
    def _to_dict(x):
        if x is None:
            return None
        try:
            from omegaconf import OmegaConf as _OC
            return _OC.to_container(x, resolve=True)
        except Exception:
            return x

    return ActionEmbedderSettings(
        type=ae_type,
        checkpoint_path=str(ckpt) if ckpt is not None else None,
        encode_method=encode_method,
        oat_action_chunk_size=oat_chunk,
        oat_action_dim=oat_dim,
        oat_encoder_cfg=_to_dict(oat_enc),
        oat_decoder_cfg=_to_dict(oat_dec),
        oat_quantizer_cfg=_to_dict(oat_qtz),
    )


def select_language_conditioning_settings(cfg: Any) -> LanguageConditioningSettings:
    """Read ``model.language_conditioning``; disabled when block is absent."""
    block = OmegaConf.select(cfg, "model.language_conditioning", default=None)
    if block is None:
        return LanguageConditioningSettings()
    enabled = bool(block.get("enabled", False))
    mode = str(block.get("mode", "concat")).lower().strip()
    if mode not in ("concat", "stratified"):
        raise ValueError(
            f"model.language_conditioning.mode must be 'concat' or 'stratified', got {mode!r}"
        )
    source = str(block.get("source", "qwen3")).lower().strip()
    if source not in ("qwen3", "precomputed"):
        raise ValueError(
            f"model.language_conditioning.source must be 'qwen3' or 'precomputed', "
            f"got {source!r}"
        )
    qwen3 = block.get("qwen3", {}) or {}
    defaults = LanguageConditioningSettings()
    return LanguageConditioningSettings(
        enabled=enabled,
        mode=mode,
        source=source,
        language_key=str(block.get("language_key", defaults.language_key)),
        model_name=str(qwen3.get("model_name", block.get("model_name", defaults.model_name))),
        max_length=int(qwen3.get("max_length", block.get("max_length", defaults.max_length))),
        batch_size=int(qwen3.get("batch_size", block.get("batch_size", defaults.batch_size))),
        dtype=str(qwen3.get("dtype", block.get("dtype", defaults.dtype))),
        stratified_min_cluster_size=int(
            block.get("stratified_min_cluster_size", defaults.stratified_min_cluster_size)
        ),
    )


def select_embedder_settings(cfg: Any) -> EmbedderSettings:
    """Read embedder settings from ``model`` config."""
    return EmbedderSettings(
        latent_dim=int(OmegaConf.select(cfg, "model.latent_dim", default=32)),
        device=str(OmegaConf.select(cfg, "model.device", default="cuda")),
        norm_min_std=float(
            OmegaConf.select(cfg, "model.norm_min_std", default=1e-6)
        ),
        state_image=select_state_image_settings(cfg),
        action_embedder=select_action_embedder_settings(cfg),
        language_conditioning=select_language_conditioning_settings(cfg),
    )


def select_tensor_keys(cfg: Any) -> tuple[str, str]:
    """Return ``(action_key, image_key)`` from ``model.keys``."""
    action = OmegaConf.select(
        cfg, "model.keys.action", default="actions_cartesian"
    )
    image = OmegaConf.select(
        cfg,
        "model.keys.image",
        default="observations.images.front_img_1",
    )
    return str(action), str(image)


def resolve_norm_stats_json(path: str | None, search_roots: list[str | Path]) -> Path | None:
    """Resolve a directory or file path to ``norm_stats.json``."""
    if not path:
        return None
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir() and (p / "norm_stats.json").is_file():
        return p / "norm_stats.json"
    for root in search_roots:
        root = Path(root)
        candidate = root / path
        if candidate.is_file():
            return candidate
        if (candidate / "norm_stats.json").is_file():
            return candidate / "norm_stats.json"
    return None


def load_action_norm_stats(
    cfg: Any,
    action_key: str,
    norm_min_std: float = 1e-6,
    search_roots: list[str | Path] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load flattened action mean/std from ``model.precomputed_norm_stats`` (required).

    Stats must match Pass 2 ``actions.reshape(T, -1)`` (chunk_size × action_dim).
    Raises ``ValueError`` if ``model.precomputed_norm_stats`` is not configured.
    """
    block = OmegaConf.select(cfg, "model.precomputed_norm_stats", default=None)
    if block is None:
        raise ValueError(
            "model.precomputed_norm_stats is required but not configured. "
            "Run the offline norm-stats script first and set "
            "model.precomputed_norm_stats.path in the model config."
        )
    rel_path = block.get("path") if hasattr(block, "get") else None
    if not rel_path:
        raise ValueError(
            "model.precomputed_norm_stats.path is empty. "
            "Set it to the directory or file containing norm_stats.json."
        )

    json_path = resolve_norm_stats_json(
        str(rel_path),
        search_roots=search_roots or [Path.cwd()],
    )
    if json_path is None:
        raise FileNotFoundError(
            f"precomputed_norm_stats.path={rel_path!r} — no norm_stats.json found "
            f"(searched {search_roots or [Path.cwd()]})"
        )

    with open(json_path) as f:
        data = json.load(f)

    embodiment_id = str(block.get("embodiment_id", "9"))
    emb_stats = data.get("stats", {}).get(embodiment_id, {})
    if action_key not in emb_stats:
        raise KeyError(
            f"action key {action_key!r} not in norm_stats.json for embodiment {embodiment_id!r}"
        )

    s = emb_stats[action_key]
    mean = np.asarray(s["mean"], dtype=np.float32).reshape(-1)
    std = np.maximum(np.asarray(s["std"], dtype=np.float32).reshape(-1), norm_min_std)
    return mean, std
