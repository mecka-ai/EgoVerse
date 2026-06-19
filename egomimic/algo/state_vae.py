"""StateVAETrainer — CNN Variational Autoencoder over egocentric images.

Training:
  Input:  (B, C, H, W) uint8 or float images from the zarr pipeline
  Preprocess: scale to [0, 1], resize to image_size × image_size
  Encode: CNN → (mu, logvar) shape (B, latent_dim)
  Sample: z = mu + eps * exp(0.5 * logvar)
  Decode: CNN → (B, C, image_size, image_size) → Sigmoid
  Loss:   MSE reconstruction + β * KL(N(mu, σ²) ‖ N(0, I))

DemInf embedding:
  After training, call encode(images) where images is (T, C, H, W) uint8 numpy.
  Returns (T, latent_dim) using mu only (no sampling).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from egomimic.algo.algo import Algo
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id

log = logging.getLogger(__name__)

# VIZ_IMAGE_KEY keyname from Mecka embodiment — DataSchematic keyname after remap
_FRONT_IMG_KEYNAME = "observations.images.front_img_1"


class _StateVAENet(nn.Module):
    """CNN encoder + decoder for image reconstruction VAE."""

    def __init__(self, latent_dim: int, image_size: int, channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.channels = channels

        # Encoder: 4 stride-2 conv blocks → AdaptiveAvgPool(4,4) → linear
        self.encoder_convs = nn.Sequential(
            nn.Conv2d(channels, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        flat_dim = 256 * 4 * 4  # 4096
        self.encoder_fc = nn.Sequential(nn.Linear(flat_dim, 512), nn.SiLU())
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        # Decoder: linear → unflatten → 4 stride-2 transposed conv blocks
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.SiLU(),
            nn.Linear(512, flat_dim),
            nn.SiLU(),
        )
        self.decoder_convs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, image_size, image_size) float [0, 1]."""
        h = self.encoder_convs(x)
        h = self.pool(h).flatten(1)
        h = self.encoder_fc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, latent_dim) → (B, C, 64, 64) float [0, 1]."""
        h = self.decoder_fc(z).reshape(z.shape[0], 256, 4, 4)
        return self.decoder_convs(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def _prep_images(img: torch.Tensor, image_size: int) -> torch.Tensor:
    """Normalise + resize images to (B, C, image_size, image_size) float [0,1]."""
    # Flatten any batch prefix dims except the last 3 (C, H, W).
    orig_shape = img.shape
    if img.ndim > 4:
        img = img.reshape(-1, *orig_shape[-3:])
    # Normalise to [0, 1].
    if img.dtype == torch.uint8:
        img = img.float() / 255.0
    else:
        img = img.float().clamp(0.0, 1.0)
    # Resize if needed.
    h, w = img.shape[-2], img.shape[-1]
    if h != image_size or w != image_size:
        img = F.interpolate(img, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return img


class StateVAETrainer(Algo):
    """VAE over egocentric images for DemInf state embedding.

    Trains a CNN β-VAE that compresses front-camera frames to a 32-dim latent.
    After training the checkpoint is consumed by CheckpointStateEmbedder in
    egomimic/curation/embedders.py.

    Args:
        data_schematic: Injected by ModelWrapper from trainHydra.
        domains: List of embodiment names (e.g. ["mecka_bimanual"]).
        image_keys: Zarr key per domain (e.g. {"mecka_bimanual": "images.front_1"}).
        latent_dim: Encoder output dimension.
        image_size: Square size images are resized to before encoding.
        beta: KL weight in β-VAE loss.
    """

    def __init__(
        self,
        data_schematic: Any,
        domains: list[str],
        image_keys: dict[str, str] | None = None,
        latent_dim: int = 32,
        image_size: int = 64,
        beta: float = 1.0,
        viz_func: Any = None,
    ) -> None:
        super().__init__()
        self.data_schematic = data_schematic
        self.domains = domains
        self.latent_dim = latent_dim
        self.image_size = image_size
        self.beta = beta

        _image_keys = image_keys or {}
        self.img_keys: dict[int, str] = {}
        self.nets = nn.ModuleDict()

        for domain_name in domains:
            embodiment_id = get_embodiment_id(domain_name)
            zarr_img_key = _image_keys.get(domain_name, "images.front_1")
            keyname = data_schematic.zarr_key_to_keyname(zarr_img_key, embodiment_id)
            # Fall back to the known VIZ_IMAGE_KEY if the remapping returns None.
            self.img_keys[embodiment_id] = keyname if keyname else _FRONT_IMG_KEYNAME
            self.nets[str(embodiment_id)] = _StateVAENet(
                latent_dim=latent_dim,
                image_size=image_size,
            )

    @property
    def device(self) -> torch.device:
        return next(iter(self.nets.parameters())).device

    @device.setter
    def device(self, value) -> None:
        pass  # Lightning sets this; actual device tracked from parameters via .to(device)

    # ------------------------------------------------------------------
    # Algo interface
    # ------------------------------------------------------------------

    def process_batch_for_training(self, batch: dict) -> dict:
        processed: dict = {}
        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            img_key_full = self.img_keys[embodiment_id]           # observations.images.front_img_1
            img_key_short = img_key_full.split(".")[-1]           # front_img_1

            # The batch uses short keynames; try full path then short form then any 4D tensor.
            raw = _batch.get(img_key_full)
            if raw is None:
                raw = _batch.get(img_key_short)
            if raw is None:
                for k, v in _batch.items():
                    if isinstance(v, torch.Tensor) and v.ndim >= 3:
                        raw = v
                        break
            if raw is None:
                raise KeyError(
                    f"StateVAETrainer: no image tensor found in batch. "
                    f"Tried {img_key_full!r}, {img_key_short!r}. "
                    f"Available: {list(_batch.keys())}"
                )

            img = _prep_images(raw.to(self.device), self.image_size)
            processed[embodiment_id] = {"image": img}
        return processed

    def forward_training(self, batch: dict) -> dict:
        predictions: dict = OrderedDict()
        for embodiment_id, _batch in batch.items():
            img = _batch["image"]  # (B, C, image_size, image_size) float [0, 1]
            net = self.nets[str(embodiment_id)]
            recon, mu, logvar = net(img)

            recon_loss = F.mse_loss(recon, img)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            emb_name = get_embodiment(embodiment_id).lower()
            predictions[f"{emb_name}_recon_loss"] = recon_loss
            predictions[f"{emb_name}_kl_loss"] = kl_loss
            predictions[f"{emb_name}_mu"] = mu
        return predictions

    def forward_eval(self, batch: dict) -> dict:
        with torch.no_grad():
            return self.forward_training(batch)

    def compute_losses(self, predictions: dict, batch: dict) -> dict:
        total_recon = torch.tensor(0.0, device=self.device)
        total_kl = torch.tensor(0.0, device=self.device)
        n = 0
        for embodiment_id in batch:
            emb_name = get_embodiment(embodiment_id).lower()
            total_recon = total_recon + predictions[f"{emb_name}_recon_loss"]
            total_kl = total_kl + predictions[f"{emb_name}_kl_loss"]
            n += 1
        n = max(n, 1)
        recon = total_recon / n
        kl = total_kl / n
        return {
            "action_loss": recon + self.beta * kl,
            "recon_loss": recon,
            "kl_loss": kl,
        }

    def log_info(self, info: dict) -> dict:
        losses = info.get("losses", {})
        label_map = {"action_loss": "Loss", "recon_loss": "ReconLoss", "kl_loss": "KLLoss"}
        return {
            label_map[k]: (v.item() if isinstance(v, torch.Tensor) else float(v))
            for k in label_map
            if k in losses
        }

    # ------------------------------------------------------------------
    # DemInf embedding hook
    # ------------------------------------------------------------------

    def encode(self, images: np.ndarray) -> np.ndarray:
        """Encode egocentric images to latent means.

        Args:
            images: (T, C, H, W) uint8 numpy array.

        Returns:
            (T, latent_dim) float32 numpy array — posterior means, no sampling.
        """
        if len(images) == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)
        t = torch.from_numpy(images).to(self.device)
        t = _prep_images(t, self.image_size)
        net = next(iter(self.nets.values()))
        was_training = net.training
        net.eval()
        with torch.no_grad():
            mu, _ = net.encode(t)
        if was_training:
            net.train()
        return mu.cpu().numpy().astype(np.float32)
