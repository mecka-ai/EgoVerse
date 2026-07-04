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
        loss_block_weights: Optional[List[dict]] = None,
        vision_encoder: Optional[nn.Module] = None,
        image_key: str = "front_img_1",
        **kwargs,
    ):
        self.nets = nn.ModuleDict()
        self.data_schematic = data_schematic
        self.viz_func = viz_func
        self.domains = list(domains)
        self.ac_keys = dict(ac_keys)
        self.skill_vae = skill_vae
        # Per-block reconstruction weighting: list of {name, start, end, weight}.
        # Loss = sum_b weight_b * mean(MSE over dims [start:end]) / sum_b weight_b,
        # so each block contributes by weight, not by dim count / raw scale.
        self.loss_blocks = [dict(b) for b in loss_block_weights] if loss_block_weights else None
        self.image_key = image_key

        device_arg = kwargs.get("device")
        if device_arg is not None:
            self.device = torch.device(device_arg)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            # MPS is skipped: SkillVAE.quantize uses numpy on device tensors which
            # fails on MPS (no float64). CPU works for local debug runs.
            self.device = torch.device("cpu")

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
        if vision_encoder is not None:
            # obs_emb conditioning: trainable image encoder whose token both the
            # SkillVAE encoder and decoder attend to.
            self.nets["vision"] = vision_encoder
        self.nets = self.nets.float().to(self.device)

        self.training_step_count = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs_emb(self, _batch):
        """Vision conditioning token (B, 1, emb_dim), or None when not configured."""
        if "vision" not in self.nets:
            return None
        if self.image_key not in _batch:
            raise KeyError(
                f"vision_encoder is configured but image key {self.image_key!r} is "
                f"missing from the batch; available keys: {list(_batch.keys())}"
            )
        return self.nets["vision"](_batch[self.image_key])

    def _recon_loss(self, recon, actions):
        """(loss, per_block) — per-block weighted mean MSE, or plain MSE if unset."""
        if not self.loss_blocks:
            return F.mse_loss(recon, actions), {}
        per_block = {}
        total = 0.0
        weight_sum = 0.0
        for blk in self.loss_blocks:
            s, e, w = int(blk["start"]), int(blk["end"]), float(blk["weight"])
            blk_loss = F.mse_loss(recon[..., s:e], actions[..., s:e])
            per_block[blk["name"]] = blk_loss
            total = total + w * blk_loss
            weight_sum += w
        return total / max(weight_sum, 1e-8), per_block

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
            obs_emb = self._obs_emb(_batch)
            recon, pp, pp_sample, commit_loss, codes = self.nets["tokenizer"](
                actions, obs_emb=obs_emb
            )
            recon_loss, per_block = self._recon_loss(recon, actions)
            predictions[f"{embodiment_name}_recon_loss"] = recon_loss
            predictions[f"{embodiment_name}_commit_loss"] = commit_loss.mean()
            predictions[f"{embodiment_name}_perplexity"] = pp
            for blk_name, blk_loss in per_block.items():
                predictions[f"{embodiment_name}_block_{blk_name}_loss"] = blk_loss
        return predictions

    @override
    def forward_eval(self, batch):
        recons_dict = {}
        with torch.inference_mode():
            for embodiment_id, _batch in batch.items():
                embodiment_name = get_embodiment(embodiment_id).lower()
                ac_key = self.ac_keys_by_id[embodiment_id]
                actions = _batch[ac_key]
                obs_emb = self._obs_emb(_batch)
                recon, pp, pp_sample, commit_loss, codes = self.nets["tokenizer"](
                    actions, obs_emb=obs_emb
                )
                recons_dict[f"{embodiment_name}_{ac_key}"] = recon
        return recons_dict

    def forward_eval_logging(self, batch):
        metrics: Dict[str, torch.Tensor] = {}
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = self.ac_keys_by_id[embodiment_id]
            actions = _batch[ac_key]
            with torch.inference_mode():
                obs_emb = self._obs_emb(_batch)
                recon, pp, pp_sample, commit_loss, codes = self.nets["tokenizer"](
                    actions, obs_emb=obs_emb
                )
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
            # Surface per-block losses (palm vs fingertips) for logging.
            prefix = f"{embodiment_name}_block_"
            for k, v in predictions.items():
                if k.startswith(prefix):
                    loss_dict[k] = v
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
