"""Trainer for a small temporal-CNN sequence autoencoder over action spans.

Mirrors the OAT/QueST tokenizer trainers (egomimic/algo/oat_tokenizer.py) but is a
plain continuous autoencoder: a 1D temporal CNN encodes a fixed-length action
trajectory ``(B, S, D)`` to a bottleneck ``(B, latent_dim)`` and reconstructs it.
Trained on per-annotation-span trajectories (resampled to S=sample_horizon and
shape-normalized by ActionNorms inside SpanActionDataset). The curation pipeline
loads the checkpoint and calls ``encode`` to get one embedding per span.

ActionNorms is applied UPSTREAM (the dataset for training, the curation span path
for inference), so this model stays a clean autoencoder; DataSchematic provides the
per-channel statistical normalization, like OAT/QueST.
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


class TemporalCNNAutoencoder(nn.Module):
    """1D temporal-CNN autoencoder for fixed-length action trajectories ``(B, S, D)``."""

    def __init__(
        self,
        action_dim: int,
        sample_horizon: int = 100,
        latent_dim: int = 128,
        channels: tuple[int, ...] = (32, 64, 128),
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.sample_horizon = int(sample_horizon)
        self.latent_dim = int(latent_dim)
        pad = kernel_size // 2

        enc, c_in = [], self.action_dim
        for c_out in channels:
            enc += [nn.Conv1d(c_in, c_out, kernel_size, stride=2, padding=pad), nn.GELU()]
            c_in = c_out
        self.encoder_conv = nn.Sequential(*enc)
        # length after len(channels) stride-2 convs (ceil division)
        self._conv_len = self.sample_horizon
        for _ in channels:
            self._conv_len = (self._conv_len + 1) // 2
        self._flat = channels[-1] * self._conv_len
        self.to_latent = nn.Linear(self._flat, self.latent_dim)

        self.from_latent = nn.Linear(self.latent_dim, self._flat)
        self._dec_c0 = channels[-1]
        dec, c_in = [], channels[-1]
        for c_out in list(reversed(channels[:-1])) + [self.action_dim]:
            dec += [
                nn.ConvTranspose1d(c_in, c_out, kernel_size, stride=2, padding=pad, output_padding=1),
                nn.GELU() if c_out != self.action_dim else nn.Identity(),
            ]
            c_in = c_out
        self.decoder_conv = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, S, D)`` → ``(B, latent_dim)``."""
        h = self.encoder_conv(x.transpose(1, 2))          # (B, C, L')
        return self.to_latent(h.flatten(start_dim=1))     # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, latent_dim)`` → ``(B, S, D)``."""
        h = self.from_latent(z).view(-1, self._dec_c0, self._conv_len)
        h = self.decoder_conv(h)                           # (B, D, ~S)
        h = F.interpolate(h, size=self.sample_horizon, mode="linear", align_corners=False)
        return h.transpose(1, 2)                           # (B, S, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class TemporalCNNAutoencoderTrainer(Algo):
    """Train the temporal-CNN action autoencoder on span trajectories."""

    def __init__(
        self,
        data_schematic,
        domains: List[str],
        ac_keys: Dict[str, str],
        autoencoder: TemporalCNNAutoencoder,
        viz_func: Optional[dict] = None,
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic
        self.viz_func = viz_func
        self.domains = list(domains)
        self.ac_keys = dict(ac_keys)

        device_arg = kwargs.get("device")
        if device_arg is not None:
            self.device = torch.device(device_arg)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ac_keys_by_id: Dict[int, str] = {}
        for embodiment in self.domains:
            embodiment_id = get_embodiment_id(embodiment)
            for key in data_schematic.keys_of_type("action_keys", embodiment_id):
                if (
                    data_schematic.is_key_with_embodiment(key, embodiment_id)
                    and key == self.ac_keys[embodiment]
                ):
                    self.ac_keys_by_id[embodiment_id] = key
            if embodiment_id not in self.ac_keys_by_id:
                raise ValueError(
                    f"Could not resolve action key {self.ac_keys[embodiment]!r} "
                    f"for embodiment {embodiment!r} in data_schematic."
                )

        self.nets["autoencoder"] = autoencoder
        self.nets = self.nets.float().to(self.device)
        self.training_step = 0

    @override
    def process_batch_for_training(self, batch):
        processed = {}
        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            processed[embodiment_id] = {}
            for key, value in _batch.items():
                key_name = self.data_schematic.zarr_key_to_keyname(key, embodiment_id)
                if key_name is not None:
                    processed[embodiment_id][key_name] = value

            ac_key = self.ac_keys_by_id[embodiment_id]
            if ac_key not in processed[embodiment_id]:
                raise KeyError(
                    f"Action key {ac_key!r} missing for embodiment {embodiment_name!r}; "
                    f"have {list(processed[embodiment_id].keys())}"
                )
            ac = processed[embodiment_id][ac_key]
            if ac.ndim != 3:
                raise ValueError(
                    f"Expected actions of shape (B, S, D), got {tuple(ac.shape)}"
                )

            processed[embodiment_id] = self.data_schematic.normalize_data(
                processed[embodiment_id], embodiment_id
            )
            for key, value in processed[embodiment_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed[embodiment_id][key] = value
        return processed

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        self.training_step += 1
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            actions = _batch[self.ac_keys_by_id[embodiment_id]]
            recon = self.nets["autoencoder"](actions)
            predictions[f"{embodiment_name}_loss"] = F.mse_loss(recon, actions)
        return predictions

    @override
    def forward_eval(self, batch):
        recons = {}
        with torch.inference_mode():
            for embodiment_id, _batch in batch.items():
                embodiment_name = get_embodiment(embodiment_id).lower()
                ac_key = self.ac_keys_by_id[embodiment_id]
                recons[f"{embodiment_name}_{ac_key}"] = self.nets["autoencoder"](_batch[ac_key])
        return recons

    def forward_eval_logging(self, batch):
        metrics: Dict[str, torch.Tensor] = {}
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            actions = _batch[self.ac_keys_by_id[embodiment_id]]
            with torch.inference_mode():
                metrics[f"{embodiment_name}_reconst_mse"] = F.mse_loss(
                    self.nets["autoencoder"](actions), actions
                )
        return metrics, {}

    def visualize_preds(self, predictions, batch):
        return None

    @override
    def compute_losses(self, predictions, batch):
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for embodiment_id, _ in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            loss = predictions[f"{embodiment_name}_loss"]
            loss_dict[f"{embodiment_name}_loss"] = loss
            total = total + loss
        loss_dict["action_loss"] = total / max(len(self.domains), 1)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        return log

    # ------------------------------------------------------------------
    # Curation interface (used by CheckpointActionEmbedder / span scorer)
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode span trajectories ``(S, D)`` or ``(B, S, D)`` → ``(latent_dim,)`` / ``(B, latent_dim)``.

        Input is expected to be ActionNorms-normalized already (applied upstream);
        this applies the trained DataSchematic per-channel normalization, then the encoder.
        """
        squeeze = actions.ndim == 2
        if squeeze:
            actions = actions.unsqueeze(0)
        actions = actions.float().to(self.device)
        eid = next(iter(self.ac_keys_by_id))
        ac_key = self.ac_keys_by_id[eid]
        normed = self.data_schematic.normalize_data({ac_key: actions}, eid)[ac_key]
        z = self.nets["autoencoder"].encode(normed.to(self.device))
        return z.squeeze(0) if squeeze else z
