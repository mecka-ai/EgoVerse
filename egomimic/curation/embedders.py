"""State and action embedders for DemInf curation (fit / embed API)."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_STATE_IMAGE_BACKBONES = frozenset({"resnet18", "dinov3"})
_DINOV3_DEFAULT_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _parse_torch_dtype(name: str) -> "torch.dtype":
    import torch

    key = str(name).lower().replace(" ", "")
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if key not in mapping:
        raise ValueError(
            f"Unsupported dtype {name!r}; expected one of {sorted(mapping)}"
        )
    return mapping[key]


def _chw_batch_to_rgb_images(chunk: np.ndarray) -> list[np.ndarray]:
    """Convert (N, C, H, W) uint8 or float [0, 1] to N HWC RGB uint8 arrays."""
    if chunk.dtype == np.uint8:
        nhwc = np.transpose(chunk, (0, 2, 3, 1))
    else:
        nhwc = np.transpose(
            (np.clip(chunk.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8),
            (0, 2, 3, 1),
        )
    return [nhwc[i] for i in range(nhwc.shape[0])]


def _fit_gaussian_stats(
    arrays: list[np.ndarray],
    min_std: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute global mean and std from a list of (T, D) arrays."""
    stacked = np.concatenate(arrays, axis=0)
    mean = stacked.mean(axis=0)
    std = np.maximum(stacked.std(axis=0), min_std)
    return mean, std


def _build_random_projection(in_dim: int, out_dim: int, seed: int = 42) -> np.ndarray | None:
    """
    Build a random orthogonal projection matrix (in_dim → out_dim).

    Returns None when in_dim ≤ out_dim (no projection needed).
    Columns are orthonormal via QR decomposition, preserving MI up to
    the Johnson-Lindenstrauss guarantee.
    """
    if in_dim <= out_dim:
        return None
    rng = np.random.default_rng(seed=seed)
    Q, _ = np.linalg.qr(rng.standard_normal((in_dim, out_dim)))
    return Q[:, :out_dim].astype(np.float32)


class StateEmbedder:
    """
    Embed proprioceptive states or egocentric images to a fixed latent dim.

    Args:
        mode: "proprioceptive" or "image".
        latent_dim: Output dimensionality (default 32).
        device: Torch device for image mode.
        image_batch_size: Batch size for GPU inference in image mode.
        image_backbone: For image mode, ``resnet18`` or ``dinov3``.
        dinov3_model_name: HuggingFace model id when ``image_backbone=dinov3``.
        dinov3_dtype: Inference dtype for DINOv3 (e.g. ``float16``).
        norm_min_std: Minimum std used in Gaussian normalisation to avoid divide-by-zero.
        project: When False (image mode), skip the random projection and return
            the raw pooled backbone features (512-d resnet18 / hidden-dim DINOv3).
            Used by k-NN grading, where retrieval quality needs the full features.
    """

    def __init__(
        self,
        mode: str = "proprioceptive",
        latent_dim: int = 32,
        device: str | torch.device = "cpu",
        image_batch_size: int = 256,
        image_backbone: str = "resnet18",
        dinov3_model_name: str = _DINOV3_DEFAULT_MODEL,
        dinov3_dtype: str = "float16",
        seed: int = 42,
        norm_min_std: float = 1e-6,
        project: bool = True,
    ) -> None:
        self.mode = mode
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.image_batch_size = image_batch_size
        self.image_backbone = image_backbone.lower().strip()
        if self.image_backbone == "dino":
            self.image_backbone = "dinov3"
        self.dinov3_model_name = dinov3_model_name
        self.dinov3_dtype = dinov3_dtype
        self._seed = seed
        self.norm_min_std = norm_min_std
        self.project = project
        self._fitted = False

        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._proj: np.ndarray | None = None
        self._backbone: nn.Module | None = None
        self._proj_layer: nn.Linear | None = None
        self._processor: Any | None = None
        self._num_patches: int = 0

    def set_precomputed_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """
        Provide precomputed normalisation stats from an external source (e.g. norm_stats.json).

        When called before fit(), the embedder skips inline stats computation and
        uses these values directly. The random projection is still built during fit().
        """
        self._mean = np.asarray(mean, dtype=np.float32)
        self._std = np.asarray(std, dtype=np.float32)
        self._precomputed = True
        logger.info(
            "StateEmbedder: using precomputed stats (obs_dim=%d)", self._mean.shape[0]
        )

    def fit(self, episodes: list | None = None) -> None:
        """Compute normalisation stats (and build projection) from all episodes."""
        if self.mode == "proprioceptive":
            if episodes and not getattr(self, "_precomputed", False):
                obs_list = [ep.observations for ep in episodes]
                self._mean, self._std = _fit_gaussian_stats(obs_list, min_std=self.norm_min_std)
            self._proj = _build_random_projection(self._mean.shape[0], self.latent_dim, seed=self._seed)
            logger.info(
                "StateEmbedder (proprio): obs_dim=%d → latent_dim=%d",
                self._mean.shape[0], min(self._mean.shape[0], self.latent_dim),
            )
        else:
            if self.image_backbone not in _STATE_IMAGE_BACKBONES:
                raise ValueError(
                    f"Unknown state image backbone {self.image_backbone!r}; "
                    f"expected one of {sorted(_STATE_IMAGE_BACKBONES)}"
                )
            if self.image_backbone == "resnet18":
                self._fit_resnet()
            else:
                self._fit_dinov3()
        self._fitted = True

    def _fit_resnet(self) -> None:
        """Build frozen ResNet-18 + fixed Xavier-initialised projection (512 → latent_dim)."""
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise ImportError("torchvision is required for resnet18 image backbone") from exc

        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self._backbone = nn.Sequential(*list(backbone.children())[:-1])
        for p in self._backbone.parameters():
            p.requires_grad_(False)
        self._backbone = self._backbone.to(self.device).eval()

        self._proj_layer = nn.Linear(512, self.latent_dim, bias=True).to(self.device)
        nn.init.xavier_uniform_(self._proj_layer.weight)
        nn.init.zeros_(self._proj_layer.bias)
        for p in self._proj_layer.parameters():
            p.requires_grad_(False)

        logger.info(
            "StateEmbedder (image/resnet18): 512 → %d (fixed random proj)",
            self.latent_dim,
        )

    def _fit_dinov3(self) -> None:
        """Build frozen DINOv3 + mean-pooled patch tokens → latent_dim projection."""
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError(
                "transformers is required for dinov3 image backbone"
            ) from exc

        dtype = _parse_torch_dtype(self.dinov3_dtype)
        logger.info("Loading DINOv3 for curation: %s", self.dinov3_model_name)
        proc = AutoImageProcessor.from_pretrained(self.dinov3_model_name)
        self._backbone = AutoModel.from_pretrained(
            self.dinov3_model_name, torch_dtype=dtype
        )
        self._backbone.to(self.device).eval()

        cfg = self._backbone.config
        side = int(cfg.image_size) // int(cfg.patch_size)
        self._num_patches = side * side
        hidden_dim = int(cfg.hidden_size)

        self._proj_layer = nn.Linear(hidden_dim, self.latent_dim, bias=True).to(
            self.device
        )
        nn.init.xavier_uniform_(self._proj_layer.weight)
        nn.init.zeros_(self._proj_layer.bias)
        for p in self._proj_layer.parameters():
            p.requires_grad_(False)

        # Extract preprocessing params from the HF processor so the hot path
        # can skip PIL entirely and run resize/normalize on the GPU.
        _mean = list(getattr(proc, "image_mean", None) or [0.485, 0.456, 0.406])
        _std  = list(getattr(proc, "image_std",  None) or [0.229, 0.224, 0.225])
        _crop = getattr(proc, "crop_size", None) or {}
        if isinstance(_crop, dict):
            crop_h = int(_crop.get("height", cfg.image_size))
            crop_w = int(_crop.get("width",  cfg.image_size))
        else:
            crop_h = crop_w = int(_crop)
        _sz = getattr(proc, "size", None) or {}
        if isinstance(_sz, dict):
            resize_to = int(_sz.get("shortest_edge", _sz.get("height", crop_h)))
        else:
            resize_to = int(_sz) if _sz else crop_h

        self._dino_mean_t  = torch.tensor(_mean, dtype=dtype, device=self.device).view(1, 3, 1, 1)
        self._dino_std_t   = torch.tensor(_std,  dtype=dtype, device=self.device).view(1, 3, 1, 1)
        self._dino_crop_hw = (crop_h, crop_w)
        self._dino_resize_to = resize_to
        self._processor = None  # no longer needed at inference time

        logger.info(
            "StateEmbedder (image/dinov3): %d patches, hidden=%d → %d, "
            "resize=%d crop=%dx%d (GPU fast path)",
            self._num_patches, hidden_dim, self.latent_dim,
            resize_to, crop_h, crop_w,
        )

    def embed(self, data: np.ndarray) -> np.ndarray:
        """
        Embed a batch of observations.

        Args:
            data: (N, obs_dim) for proprioceptive or (N, C, H, W) for image.

        Returns:
            (N, latent_dim) float32 array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        if self.mode == "proprioceptive":
            normalised = (data.astype(np.float32) - self._mean) / self._std
            return normalised @ self._proj if self._proj is not None else normalised
        if self.image_backbone == "resnet18":
            return self._embed_image_resnet(data)
        return self._embed_image_dinov3(data)

    def _embed_image_resnet(self, data: np.ndarray) -> np.ndarray:
        """
        Batched ResNet-18 inference with ImageNet normalisation.

        Expects uint8 (N, C, H, W) in [0, 255] or float32 in [0, 1].
        Center-crops to 224×224 after resizing.
        """
        try:
            import torchvision.transforms.functional as TF
        except ImportError as exc:
            raise ImportError("torchvision is required for resnet18 image backbone") from exc

        n_total = data.shape[0]
        outputs: list[np.ndarray] = []
        t0 = time.perf_counter()
        for start in range(0, n_total, self.image_batch_size):
            chunk = data[start : start + self.image_batch_size]
            tb = time.perf_counter()
            tensor = torch.from_numpy(chunk).float() / 255.0 if chunk.dtype == np.uint8 else torch.from_numpy(chunk.astype(np.float32))
            tensor = TF.resize(tensor, [224], antialias=True)
            tensor = TF.center_crop(tensor, [224, 224])
            tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            with torch.no_grad():
                feats = self._backbone(tensor.to(self.device)).flatten(1)
                if not self.project:
                    outputs.append(feats.float().cpu().numpy())
                else:
                    outputs.append(self._proj_layer(feats).cpu().numpy())
            logger.debug(
                "ResNet18 batch [%d:%d] %.3fs (%.0f imgs/s)",
                start, start + len(chunk), time.perf_counter() - tb,
                len(chunk) / (time.perf_counter() - tb) if (time.perf_counter() - tb) > 0 else 0,
            )
        elapsed = time.perf_counter() - t0
        logger.info(
            "[images] ResNet18 embed: %d images in %.2fs (%.0f imgs/s)",
            n_total, elapsed, n_total / elapsed if elapsed > 0 else 0,
        )
        return np.concatenate(outputs, axis=0)

    def _embed_image_dinov3(self, data: np.ndarray) -> np.ndarray:
        """
        Batched DINOv3 inference (patch-token mean pool → latent_dim).

        Expects uint8 (N, C, H, W) in [0, 255] or float32 in [0, 1].
        Uses direct GPU tensor ops (resize + normalize) instead of the HF
        PIL-based processor, which avoids per-frame Python object allocation.
        """
        import torch.nn.functional as F

        dtype = _parse_torch_dtype(self.dinov3_dtype)
        n_total = data.shape[0]
        outputs: list[np.ndarray] = []
        crop_h, crop_w = self._dino_crop_hw
        resize_to = self._dino_resize_to
        t0 = time.perf_counter()

        for start in range(0, n_total, self.image_batch_size):
            chunk = data[start : start + self.image_batch_size]
            tb = time.perf_counter()

            # Single numpy→GPU copy for the whole batch (NCHW uint8 → float).
            if chunk.dtype == np.uint8:
                tensor = torch.from_numpy(chunk).to(self.device, dtype=dtype, non_blocking=True)
                tensor = tensor.div_(255.0)
            else:
                tensor = torch.from_numpy(chunk.astype(np.float32)).to(self.device, dtype=dtype, non_blocking=True)
                tensor = tensor.clamp_(0.0, 1.0)

            # GPU resize → center-crop (matches HF processor shortest-edge logic).
            h, w = tensor.shape[-2], tensor.shape[-1]
            if h != crop_h or w != crop_w:
                if resize_to != crop_h:
                    # Resize shortest edge to resize_to, then center-crop.
                    if h <= w:
                        new_h, new_w = resize_to, max(crop_w, int(w * resize_to / h))
                    else:
                        new_h, new_w = max(crop_h, int(h * resize_to / w)), resize_to
                    tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
                    top  = (tensor.shape[-2] - crop_h) // 2
                    left = (tensor.shape[-1] - crop_w) // 2
                    tensor = tensor[:, :, top : top + crop_h, left : left + crop_w]
                else:
                    tensor = F.interpolate(tensor, size=(crop_h, crop_w), mode="bilinear", align_corners=False)

            # Normalize with pre-cached GPU tensors.
            tensor = (tensor - self._dino_mean_t) / self._dino_std_t

            t_preproc = time.perf_counter()
            with torch.no_grad():
                hidden = self._backbone(pixel_values=tensor).last_hidden_state
                patches = hidden[:, -self._num_patches :]
                pooled = patches.float().mean(dim=1)
                if not self.project:
                    outputs.append(pooled.cpu().numpy())
                else:
                    proj_in = self._proj_layer.weight.dtype
                    outputs.append(self._proj_layer(pooled.to(proj_in)).cpu().numpy())
            t_fwd = time.perf_counter()
            logger.debug(
                "DINOv3 batch [%d:%d] preproc=%.3fs forward=%.3fs total=%.3fs (%.0f imgs/s)",
                start, start + len(chunk),
                t_preproc - tb, t_fwd - t_preproc,
                t_fwd - tb, len(chunk) / (t_fwd - tb) if (t_fwd - tb) > 0 else 0,
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "[images] DINOv3 embed: %d images in %.2fs (%.0f imgs/s, batch_size=%d)",
            n_total, elapsed, n_total / elapsed if elapsed > 0 else 0, self.image_batch_size,
        )
        return np.concatenate(outputs, axis=0)


class OATActionEmbedder:
    """Action embedder backed by a trained OAT tokenizer checkpoint.

    Loads a Lightning ``.ckpt`` written by ``model=oat_tokenizer`` and uses the
    OATTok encoder to produce **continuous** pre-quantization latents for each
    action chunk.  These latents are suitable for KSG mutual information
    estimation in the curation pipeline.

    The encoder maps ``(B, S, action_dim)`` → ``(B, num_registers, latent_dim)``
    (register tokens).  We mean-pool the register dimension to obtain a single
    ``(B, latent_dim)`` embedding per chunk, then project down to ``latent_dim``
    (= ``embed_latent_dim`` from the config) if needed via random projection.

    Args:
        checkpoint_path: Path to the OATTokenizerTrainer Lightning ``.ckpt``.
        encoder_cfg: Hydra-structured dict describing the OATTok encoder
            (same block used in the model YAML under ``action_tokenizer.encoder``).
        decoder_cfg: Hydra-structured dict for the OATTok decoder (needed to
            instantiate OATTok; weights are loaded but only the encoder is called).
        quantizer_cfg: Hydra-structured dict for the OATTok FSQ quantizer.
        device: Torch device for inference (default ``"cpu"``).
        latent_dim: Output dimensionality after optional random projection.
            Set to ``None`` to return the raw encoder embedding dimension.
        action_chunk_size: Expected number of timesteps ``S`` per chunk.
            Used to reshape flat ``(T * action_dim,)`` inputs.
        action_dim: Action dimension ``D``.  Together with ``action_chunk_size``
            this determines how flat arrays are reshaped into ``(B, S, D)`` chunks
            before encoding.
        batch_size: Max chunks per forward call.
        seed: RNG seed for the optional random projection.
    """

    def __init__(
        self,
        checkpoint_path: str,
        encoder_cfg: dict,
        decoder_cfg: dict,
        quantizer_cfg: dict,
        device: str | torch.device = "cpu",
        latent_dim: int = 128,
        action_chunk_size: int = 100,
        action_dim: int = 12,
        batch_size: int = 256,
        seed: int = 42,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.encoder_cfg = encoder_cfg
        self.decoder_cfg = decoder_cfg
        self.quantizer_cfg = quantizer_cfg
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.batch_size = batch_size
        self._seed = seed
        self._encoder: nn.Module | None = None
        self._proj: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # fit / set_precomputed_stats
    # ------------------------------------------------------------------

    def set_precomputed_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """No-op: OAT encoder handles normalisation internally via identity transform."""
        logger.debug(
            "OATActionEmbedder: ignoring external norm stats "
            "(encoder uses identity normalizer seeded at OAT training time)"
        )

    def fit(self, episodes: list | None = None) -> None:
        """Load the OAT tokenizer from checkpoint and extract the encoder.

        Instantiates the full ``OATTok`` (encoder + quantizer + decoder) using the
        provided configs, loads weights from ``checkpoint_path``, then holds only
        the encoder for inference (avoids loading the decoder weights into GPU RAM
        unnecessarily).
        """
        import hydra.utils
        from omegaconf import OmegaConf

        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required for OATActionEmbedder")

        logger.info("OATActionEmbedder: loading tokenizer from %s", self.checkpoint_path)

        # Instantiate the full tokenizer to correctly load the state dict.
        tok = hydra.utils.instantiate(
            OmegaConf.create(
                {
                    "_target_": "oat.tokenizer.oat.tokenizer.OATTok",
                    "encoder": self.encoder_cfg,
                    "decoder": self.decoder_cfg,
                    "quantizer": self.quantizer_cfg,
                }
            )
        )

        # Load the Lightning checkpoint — keys are prefixed with "nets.tokenizer."
        state_dict_prefix = "nets.tokenizer."
        sd = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )["state_dict"]
        p = len(state_dict_prefix)
        sub = {k[p:]: v for k, v in sd.items() if k.startswith(state_dict_prefix)}
        tok.load_state_dict(sub, strict=True)

        # Hold encoder only; freeze parameters.
        self._encoder = tok.encoder.to(self.device).eval()
        for param in self._encoder.parameters():
            param.requires_grad_(False)

        # Build optional random projection: encoder_latent_embed → latent_dim.
        # encoder_latent_embed = num_registers * latent_dim_per_register
        # We mean-pool over registers first to get (latent_dim_per_register,).
        encoder_latent_per_register = self.encoder_cfg.get(
            "latent_dim", getattr(self._encoder, "latent_dim", 5)
        )
        self._proj = _build_random_projection(
            encoder_latent_per_register, self.latent_dim, seed=self._seed
        )

        self._fitted = True
        logger.info(
            "OATActionEmbedder: encoder ready on %s, "
            "latent_per_register=%d → output_dim=%d",
            self.device,
            encoder_latent_per_register,
            min(encoder_latent_per_register, self.latent_dim),
        )

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Encode a batch of action chunks into latent representations.

        Args:
            data: ``(N, action_chunk_size * action_dim)`` or
                  ``(N, action_chunk_size, action_dim)`` float32 array.
                  Values should be pre-normalized to ``[-1, 1]`` (the same
                  normalization applied during OAT training).

        Returns:
            ``(N, latent_dim)`` float32 array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        if len(data) == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)

        # Reshape to (N, S, D) chunks
        n = len(data)
        chunks = data.reshape(n, self.action_chunk_size, self.action_dim).astype(
            np.float32
        )

        outputs: list[np.ndarray] = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, n, self.batch_size):
                batch = torch.from_numpy(
                    chunks[start : start + self.batch_size]
                ).to(self.device)  # (B, S, D)

                # OAT encoder: (B, S, D) → (B, num_registers, latent_dim_per_reg)
                z = self._encoder(batch)  # continuous pre-quantization latents

                # Mean-pool over register dimension → (B, latent_dim_per_reg)
                z_pooled = z.mean(dim=1).float().cpu().numpy()

                # Optional random projection to output latent_dim
                if self._proj is not None:
                    z_pooled = z_pooled @ self._proj
                outputs.append(z_pooled)

        elapsed = time.perf_counter() - t0
        result = np.concatenate(outputs, axis=0)
        logger.debug(
            "[actions] OATActionEmbedder: %d chunks in %.3fs → latent_dim=%d",
            n, elapsed, result.shape[1],
        )
        return result


class CheckpointActionEmbedder:
    """
    Action embedder backed by a trained model checkpoint.

    Loads a PyTorch checkpoint (Lightning ``.ckpt`` or raw ``.pt``) and calls
    ``model.<encode_method>(tensor)`` to produce per-timestep action latents.

    The loaded model must accept a ``(T, action_dim)`` float32 ``torch.Tensor``
    and return a ``(T, latent_dim)`` ``torch.Tensor``.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Torch device for inference.
        latent_dim: Expected output dimensionality (used for empty-batch shortcut).
        encode_method: Name of the encode method on the loaded model (default ``"encode"``).
        batch_size: Maximum timesteps per forward call. Reduce if OOM.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | torch.device = "cpu",
        latent_dim: int = 32,
        encode_method: str = "encode",
        batch_size: int = 4096,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.latent_dim = latent_dim
        self.encode_method = encode_method
        self.batch_size = batch_size
        self._model: nn.Module | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # fit / set_precomputed_stats — kept for API compatibility; norm stats
    # are handled inside the loaded model, so these are no-ops here.
    # ------------------------------------------------------------------

    def set_precomputed_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """No-op: checkpoint model handles normalisation internally."""
        logger.debug("CheckpointActionEmbedder: ignoring external norm stats (model handles normalisation)")

    def fit(self, episodes: list | None = None) -> None:
        """Load and prepare the model from checkpoint."""
        import torch as _torch

        path = self.checkpoint_path
        if not path:
            raise ValueError("checkpoint_path is required for CheckpointActionEmbedder")

        logger.info("CheckpointActionEmbedder: loading from %s", path)
        checkpoint = _torch.load(path, map_location=self.device, weights_only=False)

        # Support Lightning .ckpt (has "state_dict" + optional hyper_parameters)
        # and plain .pt dicts with a "model" or "state_dict" key, or bare state dicts.
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
        elif isinstance(checkpoint, dict):
            # Try Hydra-instantiated model stored under "model" key
            if "model" in checkpoint and isinstance(checkpoint["model"], nn.Module):
                model = checkpoint["model"]
            else:
                # Lightning checkpoint — need to instantiate the model class first.
                # We look for a "hyper_parameters" block with a "_target_" key so we
                # can use hydra.utils.instantiate, falling back to a raw state-dict load.
                if "hyper_parameters" in checkpoint:
                    hp = checkpoint["hyper_parameters"]
                    target = hp.get("_target_", None) if isinstance(hp, dict) else None
                    if target:
                        try:
                            from hydra.utils import instantiate
                            from omegaconf import OmegaConf
                            model = instantiate(OmegaConf.create(hp))
                            sd = checkpoint.get("state_dict", {})
                            # Lightning prefixes keys with "model." — strip it
                            sd = {k.removeprefix("model."): v for k, v in sd.items()}
                            missing, unexpected = model.load_state_dict(sd, strict=False)
                            if missing:
                                logger.warning("CheckpointActionEmbedder: missing keys: %s", missing[:5])
                        except Exception as exc:
                            raise RuntimeError(
                                f"Could not instantiate model from checkpoint hyper_parameters: {exc}"
                            ) from exc
                    else:
                        raise RuntimeError(
                            "Checkpoint has hyper_parameters but no _target_; "
                            "cannot auto-instantiate. Pass the model instance directly "
                            "or use a checkpoint that stores the module under 'model'."
                        )
                else:
                    raise RuntimeError(
                        f"Unsupported checkpoint format at {path!r}. "
                        "Expected: nn.Module, dict with 'model' key, or Lightning .ckpt."
                    )
        else:
            raise RuntimeError(
                f"Unrecognised checkpoint type {type(checkpoint).__name__} at {path!r}."
            )

        if not hasattr(model, self.encode_method):
            raise AttributeError(
                f"Loaded model {type(model).__name__!r} has no method "
                f"{self.encode_method!r}. Set action_embedder.encode_method in the config."
            )

        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self._model = model
        self._fitted = True
        logger.info(
            "CheckpointActionEmbedder: %s.%s ready on %s",
            type(model).__name__, self.encode_method, self.device,
        )

    def embed(self, data: np.ndarray) -> np.ndarray:
        """
        Encode a batch of actions through the checkpoint model.

        Args:
            data: ``(T, action_dim)`` float32 array.

        Returns:
            ``(T, latent_dim)`` float32 array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        if len(data) == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)

        data = data.reshape(len(data), -1).astype(np.float32)
        encode_fn = getattr(self._model, self.encode_method)
        outputs: list[np.ndarray] = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(data), self.batch_size):
                chunk = torch.from_numpy(data[start : start + self.batch_size]).to(self.device)
                out = encode_fn(chunk)
                if isinstance(out, tuple):
                    out = out[0]
                outputs.append(out.float().cpu().numpy())
        elapsed = time.perf_counter() - t0
        result = np.concatenate(outputs, axis=0)
        logger.debug(
            "[actions] CheckpointActionEmbedder: %d timesteps → latent_dim=%d in %.3fs",
            len(data), result.shape[1], elapsed,
        )
        return result




class ActionEmbedder:
    """
    Embed actions: Gaussian normalisation → random orthogonal linear projection.

    A random orthogonal projection preserves MI (bijective when feat_dim ≤
    latent_dim; MI-preserving via Johnson-Lindenstrauss otherwise).

    Accepts per-timestep actions (T, feat_dim), where each frame's action is the
    full post-transform actions_cartesian chunk (chunk_size, action_dim) flattened
    to feat_dim = chunk_size * action_dim, in end-effector Cartesian pose (head
    frame). The whole chunk is embedded — not a single step.

    Args:
        latent_dim: Output dimensionality (default 32).
        seed: RNG seed for the random orthogonal projection matrix.
        norm_min_std: Minimum std used in Gaussian normalisation to avoid divide-by-zero.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        seed: int = 42,
        norm_min_std: float = 1e-6,
    ) -> None:
        self.latent_dim = latent_dim
        self.seed = seed
        self.norm_min_std = norm_min_std
        self._fitted = False
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._proj: np.ndarray | None = None

    def set_precomputed_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """
        Provide precomputed normalisation stats from an external source (e.g. norm_stats.json).

        Stats should be flattened to 1-D to match the embedder's internal flat representation
        (e.g. shape (chunk_size * action_dim,) for action chunks).
        When called before fit(), inline stats computation is skipped.
        """
        self._mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self._std = np.asarray(std, dtype=np.float32).reshape(-1)
        self._precomputed = True
        logger.info(
            "ActionEmbedder: using precomputed stats (feat_dim=%d)", self._mean.shape[0]
        )

    def fit(self, episodes: list | None = None) -> None:
        """Compute normalisation stats and random orthogonal projection."""
        if episodes and not getattr(self, "_precomputed", False):
            act_list = [ep.actions.reshape(len(ep.actions), -1) for ep in episodes]
            self._mean, self._std = _fit_gaussian_stats(act_list, min_std=self.norm_min_std)
        feat_dim = self._mean.shape[0]
        self._proj = _build_random_projection(feat_dim, self.latent_dim, seed=self.seed)
        self._fitted = True
        logger.info(
            "ActionEmbedder: feat_dim=%d → latent_dim=%d (proj=%s)",
            feat_dim, min(feat_dim, self.latent_dim), self._proj is not None,
        )

    def embed(self, data: np.ndarray) -> np.ndarray:
        """
        Embed a batch of actions.

        Args:
            data: (T, ...) float32 — per-timestep action chunks. Anything past
                the first axis is flattened to feat_dim = chunk_size * action_dim
                (so (T, chunk_size, action_dim) is accepted directly).

        Returns:
            (T, latent_dim) float32 array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        data = data.reshape(len(data), -1)
        if len(data) == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)
        t0 = time.perf_counter()
        normalised = (data.astype(np.float32) - self._mean) / self._std
        result = normalised @ self._proj if self._proj is not None else normalised
        elapsed = time.perf_counter() - t0
        logger.debug(
            "[actions] ActionEmbedder: %d timesteps, feat_dim=%d → latent_dim=%d in %.3fs",
            len(data), data.shape[1], result.shape[1], elapsed,
        )
        return result
