"""State and action embedders for DemInf curation (fit / embed API)."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_STATE_IMAGE_BACKBONES = frozenset({"resnet18", "dinov3", "wes"})
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
        wes_checkpoint_path: str | None = None,
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
        self.wes_checkpoint_path = wes_checkpoint_path
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
            elif self.image_backbone == "wes":
                self._fit_wes()
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

    def _fit_wes(self) -> None:
        """Load frozen WES (YOLO11-L-pose) backbone + Xavier projection (512 → latent_dim).

        Only layers 0-10 (Conv/C3k2/SPPF/C2PSA backbone) are used for feature
        extraction. All backbone layers in YOLO11 are sequential (f=-1), so they
        can be run with a simple loop without tracking intermediate outputs.
        """
        if not self.wes_checkpoint_path:
            raise ValueError(
                "wes_checkpoint_path must be set when using backbone=wes"
            )
        ckpt = torch.load(self.wes_checkpoint_path, map_location="cpu", weights_only=False)
        raw_model = ckpt["model"].float().eval()
        for p in raw_model.parameters():
            p.requires_grad_(False)
        self._wes_model = raw_model.to(self.device)

        # Probe backbone output shape with a dummy input.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640, device=self.device)
            feat = dummy
            for i in range(11):
                feat = self._wes_model.model[i](feat)
        feat_dim = feat.shape[1]  # 512 for YOLO11-L

        self._proj_layer = nn.Linear(feat_dim, self.latent_dim, bias=True).to(self.device)
        nn.init.xavier_uniform_(self._proj_layer.weight)
        nn.init.zeros_(self._proj_layer.bias)
        for p in self._proj_layer.parameters():
            p.requires_grad_(False)

        logger.info(
            "StateEmbedder (image/wes): backbone_dim=%d → %d (fixed random proj)",
            feat_dim, self.latent_dim,
        )

    def _embed_image_wes(self, data: np.ndarray) -> np.ndarray:
        """Batched WES backbone inference (YOLO11-L layers 0-10 → GAP → latent_dim).

        Expects uint8 (N, C, H, W) in [0, 255] or float32 in [0, 1]. Resizes to
        640×640 (YOLO's native resolution). No additional mean/std normalization —
        YOLO models normalize only by dividing by 255.
        """
        import torch.nn.functional as F

        n_total = data.shape[0]
        outputs: list[np.ndarray] = []
        t0 = time.perf_counter()

        for start in range(0, n_total, self.image_batch_size):
            chunk = data[start : start + self.image_batch_size]
            tb = time.perf_counter()

            if chunk.dtype == np.uint8:
                tensor = torch.from_numpy(chunk).to(self.device, dtype=torch.float32, non_blocking=True)
                tensor = tensor.div_(255.0)
            else:
                tensor = torch.from_numpy(chunk.astype(np.float32)).to(self.device, non_blocking=True)
                tensor = tensor.clamp_(0.0, 1.0)

            h, w = tensor.shape[-2], tensor.shape[-1]
            if h != 640 or w != 640:
                tensor = F.interpolate(tensor, size=(640, 640), mode="bilinear", align_corners=False)

            with torch.no_grad():
                feat = tensor
                for i in range(11):
                    feat = self._wes_model.model[i](feat)
                # feat: (B, 512, 20, 20) — global average pool → (B, 512)
                pooled = feat.mean(dim=[2, 3])
                outputs.append(self._proj_layer(pooled).cpu().numpy())

            logger.debug(
                "WES batch [%d:%d] %.3fs (%.0f imgs/s)",
                start, start + len(chunk), time.perf_counter() - tb,
                len(chunk) / (time.perf_counter() - tb) if (time.perf_counter() - tb) > 0 else 0,
            )

        elapsed = time.perf_counter() - t0
        logger.info(
            "[images] WES embed: %d images in %.2fs (%.0f imgs/s)",
            n_total, elapsed, n_total / elapsed if elapsed > 0 else 0,
        )
        return np.concatenate(outputs, axis=0)

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
        if self.image_backbone == "wes":
            return self._embed_image_wes(data)
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
                proj_in = self._proj_layer.weight.dtype
                outputs.append(self._proj_layer(pooled.to(proj_in)).cpu().numpy())
            del tensor, hidden, patches, pooled
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


def _last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Last-token pooling (Qwen3 recipe), robust to left/right padding."""
    left_padded = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padded:
        return last_hidden_states[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(
        last_hidden_states.size(0), device=last_hidden_states.device
    )
    return last_hidden_states[batch_idx, seq_lens]


class LanguageEmbedder:
    """
    Frozen text embedder for language-conditioned DemInf curation.

    Supports live Qwen3 embedding from instruction strings or pass-through /
    projection of precomputed per-frame embeddings stored in zarr.

    Args:
        source: ``qwen3`` (embed strings) or ``precomputed`` (project float arrays).
        latent_dim: Output dimensionality (random projection when raw dim is larger).
        model_name: HuggingFace id for ``source=qwen3``.
        max_length: Tokenizer truncation for Qwen3.
        batch_size: Max strings per forward call.
        dtype: Inference dtype for Qwen3.
        seed: RNG seed for optional random projection.
    """

    _QWEN3_DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        source: str = "qwen3",
        latent_dim: int = 32,
        device: str | torch.device = "cpu",
        model_name: str = _QWEN3_DEFAULT_MODEL,
        max_length: int = 512,
        batch_size: int = 64,
        dtype: str = "float16",
        seed: int = 42,
        instruction: str = "",
    ) -> None:
        self.source = source.lower().strip()
        if self.source not in ("qwen3", "precomputed"):
            raise ValueError(
                f"Unknown language source {source!r}; expected 'qwen3' or 'precomputed'"
            )
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.dtype_name = dtype
        self._seed = seed
        # Qwen3-Embedding is instruction-tuned: a task instruction prepended as
        # "Instruct: {instruction}\nQuery: {text}" steers the representation toward
        # the requested aspect (e.g. verbs + handedness rather than objects).
        self.instruction = instruction.strip()
        self._fitted = False
        self._proj: np.ndarray | None = None
        self._raw_dim: int | None = None
        self._tokenizer: Any | None = None
        self._model: nn.Module | None = None

    def fit(self, episodes: list | None = None) -> None:
        """Load Qwen3 (if needed) and build optional random projection."""
        if self.source == "qwen3":
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for language source=qwen3"
                ) from exc

            dtype = _parse_torch_dtype(self.dtype_name)
            logger.info("LanguageEmbedder: loading Qwen3 %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, padding_side="left"
            )
            self._model = AutoModel.from_pretrained(self.model_name, dtype=dtype)
            self._model.to(self.device).eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            self._raw_dim = int(self._model.config.hidden_size)
        else:
            self._raw_dim = None
            self._proj = None

        if self.source == "qwen3":
            assert self._raw_dim is not None
            out_dim = min(self._raw_dim, self.latent_dim)
            self._proj = _build_random_projection(
                self._raw_dim, self.latent_dim, seed=self._seed
            )
            logger.info(
                "LanguageEmbedder (qwen3): raw_dim=%d → latent_dim=%d (proj=%s)",
                self._raw_dim,
                out_dim,
                self._proj is not None,
            )
        else:
            logger.info(
                "LanguageEmbedder (precomputed): projection built on first embed()"
            )
        self._fitted = True

    def embed(self, data: np.ndarray | list[str]) -> np.ndarray:
        """
        Embed language inputs.

        Args:
            data: For ``qwen3``: length-N list of instruction strings (or (N,) object
                array). For ``precomputed``: (N, D) float32 array.

        Returns:
            (N, latent_dim) float32 array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        if self.source == "qwen3":
            if isinstance(data, np.ndarray):
                texts = [str(x) for x in data.reshape(-1)]
            else:
                texts = [str(x) for x in data]
            if not texts:
                return np.empty((0, self.latent_dim), dtype=np.float32)
            return self._embed_texts(texts)
        return self._embed_precomputed(np.asarray(data, dtype=np.float32))

    @torch.no_grad()
    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        assert self._model is not None and self._tokenizer is not None
        outputs: list[np.ndarray] = []
        t0 = time.perf_counter()
        if self.instruction:
            texts = [f"Instruct: {self.instruction}\nQuery: {t}" for t in texts]
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokens = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            hidden = self._model(**tokens).last_hidden_state
            pooled = _last_token_pool(hidden, tokens["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            emb = pooled.cpu().numpy()
            if self._proj is not None:
                emb = emb @ self._proj
            outputs.append(emb.astype(np.float32))
        elapsed = time.perf_counter() - t0
        result = np.concatenate(outputs, axis=0)
        logger.info(
            "[language] Qwen3 embed: %d strings in %.2fs (%.0f/s)",
            len(texts),
            elapsed,
            len(texts) / elapsed if elapsed > 0 else 0,
        )
        return result

    def _embed_precomputed(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if len(data) == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)
        in_dim = data.shape[1]
        if self._raw_dim is None:
            self._raw_dim = in_dim
            self._proj = _build_random_projection(
                in_dim, self.latent_dim, seed=self._seed
            )
            logger.info(
                "LanguageEmbedder (precomputed): in_dim=%d → latent_dim=%d (proj=%s)",
                in_dim,
                min(in_dim, self.latent_dim),
                self._proj is not None,
            )
        elif in_dim != self._raw_dim:
            raise ValueError(
                f"precomputed language dim {in_dim} != expected {self._raw_dim}"
            )
        if self._proj is not None:
            return (data @ self._proj).astype(np.float32)
        return data[:, : self.latent_dim].astype(np.float32)


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


class CheckpointStateEmbedder:
    """State embedder backed by a trained StateVAETrainer checkpoint.

    Loads a Lightning ``.ckpt`` written by ``model=state_vae`` and uses the
    StateVAETrainer.encode() method to embed front-camera images.

    Args:
        checkpoint_path: Path to the StateVAETrainer Lightning ``.ckpt``.
        device: Torch device for inference.
        batch_size: Max images per forward call.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | torch.device = "cpu",
        batch_size: int = 256,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._model = None
        self._fitted = False

    def set_precomputed_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        logger.debug("CheckpointStateEmbedder: ignoring external norm stats (VAE handles normalisation internally)")

    def fit(self, episodes: list | None = None) -> None:
        from egomimic.pl_utils.pl_model import ModelWrapper
        logger.info("CheckpointStateEmbedder: loading StateVAE from %s", self.checkpoint_path)
        wrapper = ModelWrapper.load_from_checkpoint(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self._model = wrapper.model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._fitted = True
        logger.info("CheckpointStateEmbedder: ready, latent_dim=%d", self._model.latent_dim)

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed front-camera images.

        Args:
            data: (N, C, H, W) uint8 numpy array.

        Returns:
            (N, latent_dim) float32 array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        if len(data) == 0:
            return np.empty((0, self._model.latent_dim), dtype=np.float32)
        outputs: list[np.ndarray] = []
        t0 = time.perf_counter()
        for start in range(0, len(data), self.batch_size):
            chunk = data[start : start + self.batch_size]
            outputs.append(self._model.encode(chunk))
        elapsed = time.perf_counter() - t0
        result = np.concatenate(outputs, axis=0)
        logger.info(
            "[images] CheckpointStateEmbedder: %d images in %.2fs (%.0f imgs/s)",
            len(data), elapsed, len(data) / elapsed if elapsed > 0 else 0,
        )
        return result


class CheckpointActionEmbedder:
    """Action embedder backed by a trained ActionVAETrainer checkpoint.

    Loads a Lightning ``.ckpt`` written by ``model=action_vae`` and uses the
    ActionVAETrainer.encode() method to embed action chunks.

    Args:
        checkpoint_path: Path to the ActionVAETrainer Lightning ``.ckpt``.
        device: Torch device for inference.
        batch_size: Max chunks per forward call.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | torch.device = "cpu",
        batch_size: int = 512,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self._model = None
        self._fitted = False

    def set_precomputed_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        logger.debug("CheckpointActionEmbedder: ignoring external norm stats (VAE normalises from data_schematic)")

    def fit(self, episodes: list | None = None) -> None:
        from egomimic.pl_utils.pl_model import ModelWrapper
        logger.info("CheckpointActionEmbedder: loading ActionVAE from %s", self.checkpoint_path)
        wrapper = ModelWrapper.load_from_checkpoint(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self._model = wrapper.model.to(self.device).eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        self._fitted = True
        logger.info("CheckpointActionEmbedder: ready, latent_dim=%d", self._model.latent_dim)

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed action chunks.

        Args:
            data: (N, flat_dim) or (N, horizon, action_dim) float32 array.

        Returns:
            (N, latent_dim) float32 array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        if len(data) == 0:
            return np.empty((0, self._model.latent_dim), dtype=np.float32)
        flat = data.reshape(len(data), -1).astype(np.float32)
        outputs: list[np.ndarray] = []
        t0 = time.perf_counter()
        for start in range(0, len(flat), self.batch_size):
            chunk = flat[start : start + self.batch_size]
            outputs.append(self._model.encode(chunk))
        elapsed = time.perf_counter() - t0
        result = np.concatenate(outputs, axis=0)
        logger.debug(
            "[actions] CheckpointActionEmbedder: %d chunks in %.3fs → latent_dim=%d",
            len(data), elapsed, result.shape[1],
        )
        return result


class TCNActionEmbedder:
    """Span-level action embedder backed by a trained TemporalCNNAutoencoder checkpoint.

    Unlike the per-frame action embedders, this encodes one variable-length annotation
    span at a time into a single fixed-dim latent, mirroring the span training path
    (``SpanActionDataset`` + ``TemporalCNNAutoencoderTrainer``):

        raw span trajectory ``(T, D)`` → ActionNorms (shape-normalize → ``(L, D)``)
        → ``TemporalCNNAutoencoderTrainer.encode`` (DataSchematic per-channel
        normalize + CNN bottleneck) → ``(latent_dim,)``.

    ActionNorms must match the settings the checkpoint trained on (see the data
    config's ``norms`` block). The DataSchematic stats are restored from the
    checkpoint, so no external norm stats are needed.

    Args:
        checkpoint_path: Path to the TemporalCNNAutoencoderTrainer Lightning ``.ckpt``.
        norms: ``ActionNormsSettings`` or a dict of its fields (``model.action_embedder.norms``).
        device: Torch device for inference.
        batch_size: Max spans per forward call.
    """

    def __init__(
        self,
        checkpoint_path: str,
        norms: Any = None,
        device: str | torch.device = "cpu",
        batch_size: int = 256,
    ) -> None:
        from egomimic.algo.action_norms import ActionNorms, ActionNormsSettings

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.batch_size = batch_size
        if isinstance(norms, ActionNormsSettings):
            ns = norms
        elif isinstance(norms, dict):
            ns = ActionNormsSettings(**norms)
        else:
            # No norms given → span shape-normalization ON with library defaults
            # (arc-length resample to L=100, deltas, centroid-translate, path-scale).
            ns = ActionNormsSettings(enabled=True)
        self.norms = ActionNorms(ns)
        self._norms_settings = ns
        self._model = None
        self._fitted = False
        self.latent_dim: int | None = None

    def set_precomputed_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        logger.debug(
            "TCNActionEmbedder: ignoring external norm stats "
            "(DataSchematic stats restored from checkpoint; ActionNorms applied upstream)"
        )

    def fit(self, episodes: list | None = None) -> None:
        from egomimic.pl_utils.pl_model import ModelWrapper

        logger.info("TCNActionEmbedder: loading TemporalCNNAutoencoder from %s", self.checkpoint_path)
        wrapper = ModelWrapper.load_from_checkpoint(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        # TemporalCNNAutoencoderTrainer is an Algo (not an nn.Module): its trainable
        # submodules live in .nets (already on the trainer's device from __init__), and
        # .encode() handles device placement + DataSchematic normalization internally.
        self._model = wrapper.model
        self._model.nets.eval()
        for p in self._model.nets.parameters():
            p.requires_grad_(False)
        self.latent_dim = int(self._model.nets["autoencoder"].latent_dim)
        self._fitted = True
        logger.info(
            "TCNActionEmbedder: ready, latent_dim=%d, norms=%s",
            self.latent_dim, self._norms_settings,
        )

    def embed_spans(self, trajectories: list[np.ndarray]) -> np.ndarray:
        """Encode raw per-span action trajectories into one latent each.

        Args:
            trajectories: list of ``(T_i, D)`` float arrays — one raw per-frame action
                trajectory per annotation span (variable ``T_i``).

        Returns:
            ``(n_spans, latent_dim)`` float32 array, aligned to ``trajectories``.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed_spans()")
        if not trajectories:
            return np.empty((0, self.latent_dim or 0), dtype=np.float32)

        # ActionNorms → fixed-length (L, D); all spans become the same shape so they batch.
        normed = np.stack(
            [self.norms.apply(np.asarray(t, dtype=np.float32)) for t in trajectories]
        ).astype(np.float32)  # (n, L, D)

        outputs: list[np.ndarray] = []
        t0 = time.perf_counter()
        for start in range(0, len(normed), self.batch_size):
            batch = torch.from_numpy(normed[start : start + self.batch_size]).to(self.device)
            z = self._model.encode(batch)  # (B, latent_dim); applies DataSchematic normalize
            outputs.append(np.asarray(z.detach().cpu().numpy(), dtype=np.float32))
        result = np.concatenate(outputs, axis=0)
        logger.info(
            "[actions] TCNActionEmbedder: %d spans → %s in %.2fs",
            len(trajectories), result.shape, time.perf_counter() - t0,
        )
        return result


class QuestTokenEmbedder:
    """Per-chunk QueST tokenizer for token-level span visualization.

    Loads a ``QuestTokenizerTrainer`` (SkillVAE) checkpoint and encodes each action chunk
    ``(T_chunk, action_dim)`` into ``num_tokens`` continuous pre-quantization token
    embeddings (``encoder_dim`` each) via ``SkillVAE.encode`` — the same path training used,
    so chunks must be the training horizon length (Mecka keymap: 30). DataSchematic
    normalization (restored from the checkpoint) is applied before encoding, matching
    ``QuestTokenizerTrainer.process_batch_for_training``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | torch.device = "cpu",
        batch_size: int = 256,
        quest_horizon: int = 100,
        action_dim: int = 18,
        latent_dim: int = 32,
        seed: int = 42,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.quest_horizon = quest_horizon
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self._seed = seed
        self._model = None
        self._eid = None
        self._ac_key = None
        self.num_tokens: int | None = None
        self.encoder_dim: int | None = None
        self._proj: np.ndarray | None = None
        self._fitted = False

    def set_precomputed_stats(self, mean: np.ndarray, std: np.ndarray) -> None:
        """No-op: QueST encoder normalises via data_schematic from checkpoint."""
        logger.debug("QuestTokenEmbedder: ignoring external norm stats")

    def fit(self, episodes: list | None = None) -> None:
        import re as _re
        import subprocess as _sp
        import sys as _sys
        from pathlib import Path as _Path
        from egomimic.pl_utils.pl_model import ModelWrapper

        # Auto-select the latest .ckpt if a directory was passed.
        ckpt_path = str(self.checkpoint_path)
        _cp = _Path(ckpt_path)
        if _cp.is_dir():
            _ckpts = sorted(
                _cp.glob("*.ckpt"),
                key=lambda f: int(m.group(1)) if (m := _re.search(r"epoch=(\d+)", f.name)) else 0,
            )
            if not _ckpts:
                raise FileNotFoundError(f"No *.ckpt files in {_cp}")
            ckpt_path = str(_ckpts[-1])
            logger.info("QuestTokenEmbedder: auto-selected checkpoint %s", ckpt_path)

        # Ensure external/quest is on sys.path (submodule may not be initialized yet).
        try:
            import quest  # noqa: F401
        except ImportError:
            try:
                from egomimic.modal.modal_setup import CFG as _CFG
                _qd = f"{_CFG.remote_repo_dir}/external/quest"
                _sp.run(
                    ["git", "-C", _CFG.remote_repo_dir, "submodule", "update", "--init", "external/quest"],
                    check=True,
                )
                if _Path(_qd).is_dir() and _qd not in _sys.path:
                    _sys.path.insert(0, _qd)
            except Exception:
                pass  # best-effort; load below will fail if quest is still not importable

        logger.info("QuestTokenEmbedder: loading QueST tokenizer from %s", ckpt_path)
        wrapper = ModelWrapper.load_from_checkpoint(
            ckpt_path, map_location=self.device, weights_only=False
        )
        self._model = wrapper.model  # QuestTokenizerTrainer (Algo): submodules in .nets
        self._model.nets.eval()
        for _param in self._model.nets.parameters():
            _param.requires_grad_(False)
        self._eid = next(iter(self._model.ac_keys_by_id))
        self._ac_key = self._model.ac_keys_by_id[self._eid]
        self._fitted = True

    def embed(self, data: np.ndarray) -> np.ndarray:
        """Embed flat per-frame actions into per-frame latent vectors.

        Tiles consecutive frames into non-overlapping ``quest_horizon``-step windows,
        encodes each window through the QueST encoder (mean-pooled over token dim),
        then up-samples back to per-frame by repeating each window embedding.

        Args:
            data: ``(N, flat_dim)`` float32 array. The first ``action_dim`` columns are
                  the per-frame executed action (index-0 step of the stored chunk).

        Returns:
            ``(N, latent_dim)`` float32 array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed()")
        data = np.asarray(data, dtype=np.float32)
        N = len(data)
        if N == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)

        # Extract per-step action (first action_dim cols handle both per-step and
        # flattened-chunk formats: for flat_dim==action_dim this is a no-op copy).
        per_step = data[:, : self.action_dim]  # (N, action_dim)

        H = self.quest_horizon
        pad = (H - N % H) % H
        if pad > 0:
            per_step = np.concatenate([per_step, np.repeat(per_step[-1:], pad, axis=0)], axis=0)
        n_chunks = len(per_step) // H
        windows = per_step.reshape(n_chunks, H, self.action_dim)

        tok_emb = self.embed_chunks(windows)  # (n_chunks, num_tokens, encoder_dim)
        pooled = tok_emb.mean(axis=1)  # (n_chunks, encoder_dim)

        if self._proj is None:
            self._proj = _build_random_projection(pooled.shape[1], self.latent_dim, seed=self._seed)
        projected = (pooled @ self._proj).astype(np.float32)

        # Up-sample: each chunk embedding repeated H times, trim to original N.
        return np.repeat(projected, H, axis=0)[:N]

    def embed_chunks(self, chunks: np.ndarray) -> np.ndarray:
        """Encode action chunks into per-token embeddings.

        Args:
            chunks: ``(N, T_chunk, action_dim)`` float array (T_chunk = training horizon).

        Returns:
            ``(N, num_tokens, encoder_dim)`` float32 array.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before embed_chunks()")
        chunks = np.asarray(chunks, dtype=np.float32)
        if len(chunks) == 0:
            return np.empty((0, self.num_tokens or 0, self.encoder_dim or 0), dtype=np.float32)
        vae = self._model.nets["tokenizer"]
        ds = self._model.data_schematic
        outputs: list[np.ndarray] = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(chunks), self.batch_size):
                b = torch.from_numpy(chunks[start : start + self.batch_size]).to(self.device).float()
                normed = ds.normalize_data({self._ac_key: b}, self._eid)[self._ac_key]
                z = vae.encode(normed.to(self.device))  # (B, num_tokens, encoder_dim)
                outputs.append(np.asarray(z.detach().cpu().numpy(), dtype=np.float32))
        out = np.concatenate(outputs, axis=0)
        self.num_tokens, self.encoder_dim = int(out.shape[1]), int(out.shape[2])
        logger.info(
            "[actions] QuestTokenEmbedder: %d chunks → %s (%d tok/chunk) in %.2fs",
            len(chunks), out.shape, self.num_tokens, time.perf_counter() - t0,
        )
        return out


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
