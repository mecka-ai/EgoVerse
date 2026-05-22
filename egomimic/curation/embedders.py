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
        self._processor = AutoImageProcessor.from_pretrained(self.dinov3_model_name)
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

        logger.info(
            "StateEmbedder (image/dinov3): %d patches, hidden=%d → %d",
            self._num_patches,
            hidden_dim,
            self.latent_dim,
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
        Uses the model's HuggingFace image processor for resize/normalise.
        """
        dtype = _parse_torch_dtype(self.dinov3_dtype)
        n_total = data.shape[0]
        outputs: list[np.ndarray] = []
        t0 = time.perf_counter()
        for start in range(0, n_total, self.image_batch_size):
            chunk = data[start : start + self.image_batch_size]
            tb = time.perf_counter()
            images = _chw_batch_to_rgb_images(chunk)
            t_decode = time.perf_counter()
            inputs = self._processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, dtype=dtype)
            t_preproc = time.perf_counter()
            with torch.no_grad():
                hidden = self._backbone(pixel_values=pixel_values).last_hidden_state
                patches = hidden[:, -self._num_patches :]
                pooled = patches.float().mean(dim=1)
                proj_in = self._proj_layer.weight.dtype
                outputs.append(
                    self._proj_layer(pooled.to(proj_in)).cpu().numpy()
                )
            t_fwd = time.perf_counter()
            logger.debug(
                "DINOv3 batch [%d:%d] decode=%.3fs preproc=%.3fs forward=%.3fs total=%.3fs (%.0f imgs/s)",
                start, start + len(chunk),
                t_decode - tb, t_preproc - t_decode, t_fwd - t_preproc,
                t_fwd - tb, len(chunk) / (t_fwd - tb) if (t_fwd - tb) > 0 else 0,
            )
        elapsed = time.perf_counter() - t0
        logger.info(
            "[images] DINOv3 embed: %d images in %.2fs (%.0f imgs/s, batch_size=%d)",
            n_total, elapsed, n_total / elapsed if elapsed > 0 else 0, self.image_batch_size,
        )
        return np.concatenate(outputs, axis=0)


class ActionEmbedder:
    """
    Embed actions: Gaussian normalisation → random orthogonal linear projection.

    A random orthogonal projection preserves MI (bijective when action_dim ≤
    latent_dim; MI-preserving via Johnson-Lindenstrauss otherwise).

    Accepts per-timestep actions (T, action_dim) — the step-0 action vector
    from the post-transform actions_cartesian chunk, in end-effector Cartesian
    pose (head frame).

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
            data: (T, action_dim) float32 — per-timestep action vectors.

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
