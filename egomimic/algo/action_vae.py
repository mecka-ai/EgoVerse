"""ActionVAETrainer — Variational Autoencoder over action chunks.

Trains a VAE that compresses per-timestep action chunks into a compact
latent representation for use in DemInf mutual-information scoring.

Training:
  Input:  (B, sample_horizon, action_dim) normalised action chunks
  Encode: MLP → (mu, logvar) shape (B, latent_dim)
  Sample: z = mu + eps * exp(0.5 * logvar)
  Decode: MLP → (B, sample_horizon * action_dim) → reshape to (B, sample_horizon, action_dim)
  Loss:   MSE reconstruction + β * KL(N(mu,σ²) ‖ N(0,I))

DemInf embedding:
  After training, call encode(actions_flat) where actions_flat is (T, flat_dim)
  float32 numpy.  Returns (T, latent_dim) using mu only (no sampling).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from egomimic.algo.algo import Algo
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLP building block
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden_dims: list[int], out_dim: int, act=nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# VAE network
# ---------------------------------------------------------------------------

class _ActionVAENet(nn.Module):
    """Encoder + decoder MLP pair."""

    def __init__(
        self,
        flat_dim: int,
        latent_dim: int,
        encoder_hidden: list[int],
        decoder_hidden: list[int],
    ) -> None:
        super().__init__()
        self.flat_dim = flat_dim
        self.latent_dim = latent_dim

        if encoder_hidden:
            trunk_layers: list[nn.Module] = []
            prev = flat_dim
            for h in encoder_hidden:
                trunk_layers += [nn.Linear(prev, h), nn.SiLU()]
                prev = h
            self.encoder_trunk: nn.Module = nn.Sequential(*trunk_layers)
            enc_out = encoder_hidden[-1]
        else:
            self.encoder_trunk = nn.Identity()
            enc_out = flat_dim

        self.fc_mu = nn.Linear(enc_out, latent_dim)
        self.fc_logvar = nn.Linear(enc_out, latent_dim)

        self.decoder = _mlp(latent_dim, decoder_hidden, flat_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# ---------------------------------------------------------------------------
# Algo wrapper
# ---------------------------------------------------------------------------

class ActionVAETrainer(Algo):
    """Variational Autoencoder trainer for action chunks.

    Designed to plug into the egomimic training pipeline exactly like HPT/ACT:
    uses the same data pipeline (zarr or zip), same ModelWrapper, same trainHydra
    entrypoint.

    After training the checkpoint can be used by DemInf as an action embedder:

        action_embedder:
          type: checkpoint
          checkpoint_path: <path/to/lightning.ckpt>
          encode_method: encode

    The ``encode`` method accepts (T, flat_dim) float32 numpy and returns
    (T, latent_dim) float32 numpy using the posterior mean (no sampling).
    """

    def __init__(
        self,
        data_schematic: Any,
        domains: list[str],
        ac_keys: dict[str, str],
        sample_horizon: int,
        sample_dim: int,
        latent_dim: int = 32,
        encoder_hidden: list[int] | None = None,
        decoder_hidden: list[int] | None = None,
        beta: float = 1.0,
        viz_func: Any = None,
    ) -> None:
        super().__init__()

        self.data_schematic = data_schematic
        self.domains = domains
        self.viz_func = viz_func
        self.latent_dim = latent_dim
        self.sample_horizon = sample_horizon
        self.sample_dim = sample_dim
        self.beta = beta

        flat_dim = sample_horizon * sample_dim
        enc_hid = encoder_hidden or [512, 256]
        dec_hid = decoder_hidden or [256, 512]

        # Build one VAE per domain (allows different action dims in future).
        self.ac_keys: dict[int, str] = {}
        self.nets = nn.ModuleDict()
        for domain_name in domains:
            embodiment_id = get_embodiment_id(domain_name)
            ac_key_zarr = ac_keys.get(domain_name, "actions_cartesian")
            self.ac_keys[embodiment_id] = data_schematic.zarr_key_to_keyname(
                ac_key_zarr, embodiment_id
            )
            self.nets[str(embodiment_id)] = _ActionVAENet(
                flat_dim=flat_dim,
                latent_dim=latent_dim,
                encoder_hidden=enc_hid,
                decoder_hidden=dec_hid,
            )

        # Norm stats for the encode() DemInf hook — populated on first call.
        self._norm_mean: np.ndarray | None = None
        self._norm_std: np.ndarray | None = None

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
            proc: dict = {}
            for key, value in _batch.items():
                key_name = self.data_schematic.zarr_key_to_keyname(key, embodiment_id)
                if key_name is not None:
                    proc[key_name] = value
            proc = self.data_schematic.normalize_data(proc, embodiment_id)
            for k, v in proc.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(self.device)
                    if v.is_floating_point():
                        v = v.float()
                    proc[k] = v
            processed[embodiment_id] = proc
        return processed

    def forward_training(self, batch: dict) -> dict:
        predictions: dict = OrderedDict()
        for embodiment_id, _batch in batch.items():
            ac_key = self.ac_keys[embodiment_id]
            actions = _batch[ac_key]  # (B, S, D)
            B = actions.shape[0]
            x = actions.reshape(B, -1)  # (B, S*D)

            net = self.nets[str(embodiment_id)]
            recon, mu, logvar = net(x)

            recon_loss = nn.functional.mse_loss(recon, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            emb_name = get_embodiment(embodiment_id).lower()
            predictions[f"{emb_name}_recon_loss"] = recon_loss
            predictions[f"{emb_name}_kl_loss"] = kl_loss
            predictions[f"{emb_name}_mu"] = mu
            predictions[f"{emb_name}_recon"] = recon.reshape(B, self.sample_horizon, self.sample_dim)

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
        out: dict = {}
        for k in ("action_loss", "recon_loss", "kl_loss"):
            if k in losses:
                label = {"action_loss": "Loss", "recon_loss": "ReconLoss", "kl_loss": "KLLoss"}[k]
                v = losses[k]
                out[label] = v.item() if isinstance(v, torch.Tensor) else float(v)
        return out

    # ------------------------------------------------------------------
    # DemInf embedding hook
    # ------------------------------------------------------------------

    def _get_norm_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract action norm stats from data_schematic for the first domain."""
        if self._norm_mean is not None:
            return self._norm_mean, self._norm_std
        embodiment_id = get_embodiment_id(self.domains[0])
        ac_key = self.ac_keys[embodiment_id]
        ns = self.data_schematic.norm_stats.get(embodiment_id, {}).get(ac_key, {})
        mean = np.asarray(ns.get("mean", [0.0]), dtype=np.float32).reshape(-1)
        std = np.asarray(ns.get("std", [1.0]), dtype=np.float32).reshape(-1)
        std = np.where(std < 1e-6, 1.0, std)
        self._norm_mean = mean
        self._norm_std = std
        return mean, std

    def encode(self, actions: np.ndarray) -> np.ndarray:
        """Encode flat action chunks to latent means.

        Args:
            actions: (T, flat_dim) float32 numpy array.
                     flat_dim = sample_horizon * sample_dim (e.g. 1200 for 100×12).

        Returns:
            (T, latent_dim) float32 numpy array — posterior means, no sampling.
        """
        if len(actions) == 0:
            return np.empty((0, self.latent_dim), dtype=np.float32)
        mean, std = self._get_norm_stats()
        x_norm = (actions.reshape(len(actions), -1).astype(np.float32) - mean) / std
        x_t = torch.from_numpy(x_norm).to(self.device)
        net = next(iter(self.nets.values()))
        was_training = net.training
        net.eval()
        with torch.no_grad():
            mu, _ = net.encode(x_t)
        if was_training:
            net.train()
        return mu.cpu().numpy()
