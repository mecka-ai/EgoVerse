"""Standalone trainer for the QueST SkillVAE action tokenizer.

Wraps a SkillVAE instance to enable independent training on action reconstruction
via PyTorch Lightning.  Only the encoder + FSQ + decoder are trained; the loss
combines MSE reconstruction with the VQ commitment penalty.

Action chunks come out of the egomimic data pipeline already normalized to [-1, 1]
by ``DataSchematic.normalize_data``.  SkillVAE has no internal normalizer so data
passes through as-is.
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id


class QuestTokenizerTrainer(Algo):
    """Train only the QueST SkillVAE action tokenizer on egomimic action chunks."""

    def __init__(
        self,
        data_schematic,
        domains: List[str],
        ac_keys: Dict[str, str],
        skill_vae,
        viz_func: Optional[dict] = None,
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic
        self.viz_func = viz_func
        self.domains = list(domains)
        self.ac_keys = dict(ac_keys)
        self.skill_vae = skill_vae

        if not torch.cuda.is_available():
            raise RuntimeError(
                "QuestTokenizerTrainer requires CUDA; no CUDA device is available."
            )
        device_arg = kwargs.get("device")
        if device_arg is not None:
            self.device = torch.device(device_arg)
            if self.device.type != "cuda":
                raise ValueError(
                    f"device must be CUDA (e.g. cuda or cuda:0), got {self.device!r}"
                )
        else:
            self.device = torch.device("cuda")

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

        self.nets["tokenizer"] = skill_vae
        self.nets = self.nets.float().to(self.device)

        self.training_step_count = 0

    @override
    def process_batch_for_training(self, batch):
        processed_batch = {}
        for embodiment_name, _batch in batch.items():
            embodiment_id = get_embodiment_id(embodiment_name)
            processed_batch[embodiment_id] = {}
            for key, value in _batch.items():
                key_name = self.data_schematic.zarr_key_to_keyname(key, embodiment_id)
                if key_name is not None:
                    processed_batch[embodiment_id][key_name] = value

            ac_key = self.ac_keys_by_id[embodiment_id]
            if ac_key not in processed_batch[embodiment_id]:
                raise KeyError(
                    f"Action key {ac_key!r} missing from batch for embodiment "
                    f"{embodiment_name!r}; available keys: "
                    f"{list(processed_batch[embodiment_id].keys())}"
                )
            ac = processed_batch[embodiment_id][ac_key]
            if ac.ndim != 3:
                raise ValueError(
                    f"Expected actions of shape (B, S, D) for tokenizer training, "
                    f"got {tuple(ac.shape)}"
                )

            processed_batch[embodiment_id] = self.data_schematic.normalize_data(
                processed_batch[embodiment_id], embodiment_id
            )
            processed_batch[embodiment_id]["embodiment"] = torch.tensor(
                [embodiment_id], device=self.device, dtype=torch.int64
            )
            for key, value in processed_batch[embodiment_id].items():
                if isinstance(value, torch.Tensor):
                    value = value.to(self.device)
                    if value.is_floating_point():
                        value = value.float()
                    processed_batch[embodiment_id][key] = value

        return processed_batch

    @override
    def forward_training(self, batch):
        predictions = OrderedDict()
        self.training_step_count += 1
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys_by_id[embodiment_id]
            actions = _batch[ac_key]
            recon, pp, pp_sample, commit_loss, codes = self.nets["tokenizer"](actions)
            recon_loss = F.mse_loss(recon, actions)
            predictions[f"{embodiment_name}_recon_loss"] = recon_loss
            predictions[f"{embodiment_name}_commit_loss"] = commit_loss.mean()
            predictions[f"{embodiment_name}_perplexity"] = pp
        return predictions

    @override
    def forward_eval(self, batch):
        recons_dict = {}
        with torch.inference_mode():
            for embodiment_id, _batch in batch.items():
                embodiment_name = get_embodiment(embodiment_id).lower()
                ac_key = self.ac_keys_by_id[embodiment_id]
                actions = _batch[ac_key]
                recon, pp, pp_sample, commit_loss, codes = self.nets["tokenizer"](actions)
                recons_dict[f"{embodiment_name}_{ac_key}"] = recon
        return recons_dict

    def forward_eval_logging(self, batch):
        metrics: Dict[str, torch.Tensor] = {}
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys_by_id[embodiment_id]
            actions = _batch[ac_key]
            with torch.inference_mode():
                recon, pp, pp_sample, commit_loss, codes = self.nets["tokenizer"](actions)
                mse = F.mse_loss(recon, actions)
            metrics[f"{embodiment_name}_reconst_mse"] = mse
            metrics[f"{embodiment_name}_perplexity"] = pp.mean()
        return metrics, {}

    def visualize_preds(self, predictions, batch):
        return None

    @override
    def compute_losses(self, predictions, batch):
        total = torch.tensor(0.0, device=self.device)
        loss_dict = OrderedDict()
        for embodiment_id, _ in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            recon_loss = predictions[f"{embodiment_name}_recon_loss"]
            commit_loss = predictions[f"{embodiment_name}_commit_loss"]
            combined = recon_loss + commit_loss
            loss_dict[f"{embodiment_name}_recon_loss"] = recon_loss
            loss_dict[f"{embodiment_name}_commit_loss"] = commit_loss
            loss_dict[f"{embodiment_name}_loss"] = combined
            total = total + combined
        loss_dict["action_loss"] = total / max(len(self.domains), 1)
        return loss_dict

    @override
    def log_info(self, info):
        log = OrderedDict()
        log["Loss"] = info["losses"]["action_loss"].item()
        for k, v in info["losses"].items():
            log[k] = v.item()
        for k, v in info.get("predictions", {}).items():
            if "perplexity" in k and torch.is_tensor(v):
                log[k] = v.item()
        return log

    @torch.inference_mode()
    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode action chunks into continuous latent embeddings.

        Maps ``(T, action_dim)`` → ``(T, num_latent_tokens * codebook_dim)``
        (pre-quantization, for use as action embeddings in the curation pipeline).

        Args:
            actions: Float tensor ``(T, action_dim)`` or ``(B, T, action_dim)``,
                values in the normalized action space (already in [-1, 1]).

        Returns:
            Float tensor ``(B_or_T, latent_tokens * encoder_dim)``.
        """
        vae = self.nets["tokenizer"]
        squeeze = actions.ndim == 2
        if squeeze:
            actions = actions.unsqueeze(0)
        actions = actions.float().to(self.device)
        # encode() produces the pre-quantization latent from the encoder
        z = vae.encode(actions)  # (B, num_tokens, encoder_dim)
        out = z.flatten(start_dim=1)  # (B, num_tokens * encoder_dim)
        if squeeze:
            out = out.squeeze(0)
        return out
