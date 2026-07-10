"""Arc Tokenizer — a first-class algo for progress-based action tokens.

Standalone trainer for the Arc Tokenizer representation (sequence actions by
PROGRESS instead of time): each chunk covers a fixed travelled distance (30 cm of
combined wrist path), resampled to M waypoints equally spaced in arc length, with
a STILL branch for pauses and a packed per-waypoint velocity feature. The action
layout is 49-dim: [L palm 6D (0:9) | L tips (9:24) | R palm (24:33) |
R tips (33:48) | path speed (48:49)] — produced by the
``cartesian_wristframe_6d_fingertips_arctok`` data transform.

Separate from ``QuestTokenizerTrainer`` so the Arc Tokenizer can evolve
independently; the underlying VQ network (``ArcTok``) subclasses our SkillVAE
fork (pre-quant LayerNorm, optional positional embedding) — the quest submodule
is still required for the network internals.

Actions arrive normalized to [-1, 1] by ``DataSchematic.normalize_data``.
"""

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.algo.skill_vae_mecka import SkillVAEMecka
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id

# Default 49-dim block layout of the arctok representation. Shape blocks dominate
# (the representation exists to retain trajectory shape); speed is auxiliary.
ARCTOK_DEFAULT_LOSS_BLOCKS = [
    {"name": "palm_left", "start": 0, "end": 9, "weight": 1.0},
    {"name": "tips_left", "start": 9, "end": 24, "weight": 2.0},
    {"name": "palm_right", "start": 24, "end": 33, "weight": 1.0},
    {"name": "tips_right", "start": 33, "end": 48, "weight": 2.0},
    {"name": "speed", "start": 48, "end": 49, "weight": 0.5},
]


class ArcTok(SkillVAEMecka):
    """The learned VQ network over arc-length action tokens.

    A named subclass of our SkillVAE fork with arc-appropriate defaults baked in
    (no encoder positional embedding — waypoints are indexed by progress, not
    time; pre-quant LayerNorm — FSQ saturation collapse is structurally
    impossible). All constructor args of SkillVAE/SkillVAEMecka pass through.
    """

    def __init__(
        self,
        *args,
        use_positional_emb: bool = False,
        normalize_pre_quant: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            use_positional_emb=use_positional_emb,
            normalize_pre_quant=normalize_pre_quant,
            **kwargs,
        )
        # SkillVAE stores skill_block_size but not downsample_factor; keep the
        # token count directly (conv stack halves ceil-wise per stride-2 level).
        import numpy as np

        n = int(kwargs["skill_block_size"])
        for _ in range(int(np.log2(int(kwargs["downsample_factor"])))):
            n = (n + 1) // 2
        self.num_tokens = n


class ArcTokenizerTrainer(Algo):
    """Train the Arc Tokenizer on arc-length action chunks.

    Interface-compatible with the tokenizer eval stack (``forward_eval``,
    ``ac_keys_by_id``, ``viz_func``, ``data_schematic``) so
    ``EvalTokenizerVideo`` / ``TrainVizTokenizerEvalVideo`` work unchanged.
    """

    def __init__(
        self,
        data_schematic,
        domains: List[str],
        ac_keys: Dict[str, str],
        tokenizer: ArcTok,
        arc_tokenizer: Optional[nn.Module] = None,
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
        # The deterministic Arc Tokenizer module (egomimic.models.arc_tokenizer)
        # — the canonical tokenization spec. Parameter-free today, so the actual
        # tokenization executes in dataloader workers via ApplyArcTokenizer; when
        # it gains learned params, move its call into process_batch_for_training
        # and register it in self.nets so the optimizer sees it.
        self.arc_tokenizer = arc_tokenizer
        # Per-block weighted mean MSE (each block contributes by weight, not by
        # dim count). None -> the arctok 49-dim default layout.
        self.loss_blocks = [
            dict(b)
            for b in (
                loss_block_weights
                if loss_block_weights is not None
                else ARCTOK_DEFAULT_LOSS_BLOCKS
            )
        ]
        self.image_key = image_key

        device_arg = kwargs.get("device")
        if device_arg is not None:
            self.device = torch.device(device_arg)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            # MPS skipped: SkillVAE.quantize uses numpy on device tensors, which
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

        self.nets["tokenizer"] = tokenizer
        if vision_encoder is not None:
            # obs_emb conditioning: trainable image encoder whose token both the
            # tokenizer encoder and decoder attend to.
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
        """(loss, per_block) — per-block weighted mean MSE."""
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

    # ------------------------------------------------------------------
    # Algo interface
    # ------------------------------------------------------------------

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
            # Per-block losses (palm / tips / speed) for W&B.
            prefix = f"{embodiment_name}_block_"
            for k, v in predictions.items():
                if k.startswith(prefix):
                    loss_dict[k] = v
            # Codebook perplexity so FSQ saturation collapse is visible in
            # telemetry, not just val videos.
            pp_key = f"{embodiment_name}_perplexity"
            if pp_key in predictions:
                loss_dict[pp_key] = predictions[pp_key].detach()
            total = total + combined
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
    # Curation / analysis interface
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode arc chunks into continuous pre-quantization latents.

        ``(T, action_dim)`` or ``(B, T, action_dim)`` (normalized space) →
        ``(B_or_T, num_tokens * encoder_dim)`` for the curation pipeline.
        """
        tok = self.nets["tokenizer"]
        squeeze = actions.ndim == 2
        if squeeze:
            actions = actions.unsqueeze(0)
        actions = actions.float().to(self.device)
        z = tok.encode(actions)  # (B, num_tokens, encoder_dim)
        out = z.flatten(start_dim=1)
        if squeeze:
            out = out.squeeze(0)
        return out

    @torch.inference_mode()
    def get_indices(self, actions: torch.Tensor) -> torch.Tensor:
        """Discrete arc-token indices ``(B, num_tokens)`` for token analysis."""
        squeeze = actions.ndim == 2
        if squeeze:
            actions = actions.unsqueeze(0)
        idx = self.nets["tokenizer"].get_indices(actions.float().to(self.device))
        return idx.squeeze(0) if squeeze else idx
