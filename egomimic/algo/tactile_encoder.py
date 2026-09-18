"""
Cross-embodiment tactile encoder: masked tactile autoencoding (MTAE) +
margin-gated Sinkhorn optimal-transport (OT) alignment between a human
tactile-glove stream and a robot tactile-sensor stream.

This pretrains a shared latent space that is later frozen and fed into a
downstream policy's cross-attention layer (e.g. pi_0.5), so a policy trained
mostly on robot data can transfer contact-rich behavior learned from cheap,
abundant human tele-op/glove demonstrations.

Maps directly onto the four-phase spec:
    Phase 1 (architecture): `egomimic.models.tactile_nets` -- input adapters
        A_h/A_r, shared backbone E_shared, decoders D_h/D_r, projection head g.
    Phase 2 (losses): computed in `forward_training` -- masked reconstruction
        (MTAE) per platform, and the proxy-guided, margin-gated Sinkhorn OT
        distance between the two platforms' projected latents.
    Phase 3 (schedule): a step-counted, two-stage schedule -- MTAE-only warmup
        for `warmup_steps` (K1), then a `sigmoid_anneal`-weighted OT term
        layered on top until `alignment_end_steps` (K2). See `compute_losses`.
    Phase 4 (deployment): `export_policy_backbone` freezes the adapters +
        shared encoder and discards the decoders/projector.
"""

import copy
import math
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss
from overrides import override

from egomimic.algo.algo import Algo
from egomimic.models.tactile_nets import (
    PlatformAdapter,
    PlatformDecoder,
    ProjectionHead,
    SharedBackboneEncoder,
    TactileEncoderBackbone,
    compute_proxy_features,
    make_proxy_guided_cost_fn,
    random_masking,
)


def sigmoid_anneal(
    step: int,
    start_step: int,
    end_step: int,
    max_value: float,
    steepness: float = 10.0,
) -> float:
    """
    Sigmoid schedule for lambda_OT: exactly 0 for `step <= start_step`, exactly
    `max_value` for `step >= end_step`, and a monotonically increasing sigmoid
    curve in between.

    A raw sigmoid only asymptotes to its endpoints, so the curve is
    min-max renormalized against its own value at `progress=0` and
    `progress=1` to hit the boundaries exactly -- otherwise Stage 1 would
    leak a small nonzero OT weight from the very first alignment step.
    """
    if end_step <= start_step:
        raise ValueError(
            f"alignment_end_steps ({end_step}) must be > warmup_steps ({start_step})"
        )
    if step <= start_step:
        return 0.0
    if step >= end_step:
        return max_value

    progress = (step - start_step) / (end_step - start_step)
    x = (progress - 0.5) * steepness
    s = 1.0 / (1.0 + math.exp(-x))
    s0 = 1.0 / (1.0 + math.exp(steepness / 2.0))
    s1 = 1.0 / (1.0 + math.exp(-steepness / 2.0))
    return max_value * (s - s0) / (s1 - s0)


class TactileEncoder(Algo):
    """
    Self-supervised, cross-embodiment tactile representation learner.

    Consumes paired-or-unpaired batches of the form
        {platform_name: {"tactile": Tensor[B, T, C, H, W]}, ...}
    for exactly two platforms (e.g. "human", "robot"), configured via
    `platforms`. Each platform gets its own light input adapter and
    reconstruction decoder; a single shared transformer backbone and a single
    alignment projection head are used for both.
    """

    def __init__(
        self,
        data_schematic=None,
        viz_func=None,
        # ---------------------------
        # Platform IO specs, e.g.:
        #   platforms:
        #     human: {input_shape: [T, C, H, W], patch_size: [pt, ph, pw]}
        #     robot: {input_shape: [T, C, H, W], patch_size: [pt, ph, pw]}
        # ---------------------------
        platforms: Optional[Dict] = None,
        # ---------------------------
        # Shared backbone / projector dims
        # ---------------------------
        d_model: int = 256,
        d_latent: int = 128,
        d_proj: int = 64,
        proj_hidden_dim: int = 128,
        num_encoder_layers: int = 6,
        num_encoder_heads: int = 8,
        encoder_dim_feedforward: Optional[int] = None,
        decoder_num_layers: int = 2,
        decoder_num_heads: int = 4,
        dropout: float = 0.0,
        # ---------------------------
        # Phase 2: Masked Tactile Autoencoding (MTAE)
        # ---------------------------
        mask_ratio: float = 0.75,
        # ---------------------------
        # Phase 2: proxy-guided, margin-gated Sinkhorn OT
        # ---------------------------
        alpha: float = 1.0,  # proxy-feature weight in the cost matrix C_ij
        margin: float = 0.0,  # margin m
        contact_threshold: float = 0.0,
        sinkhorn_blur: float = 0.05,
        sinkhorn_scaling: float = 0.9,
        # ---------------------------
        # Phase 3: two-stage schedule
        # ---------------------------
        warmup_steps: int = 10_000,  # K1: MTAE-only warmup
        alignment_end_steps: int = 40_000,  # K2: lambda_OT reaches lambda_max
        lambda_max: float = 1.0,
        sigmoid_steepness: float = 10.0,
        **kwargs,
    ):
        super().__init__()

        if platforms is None or len(platforms) != 2:
            raise ValueError(
                "TactileEncoder requires exactly 2 platforms (e.g. 'human' and "
                f"'robot'), got: {list(platforms.keys()) if platforms else platforms}"
            )

        self.data_schematic = data_schematic
        self.viz_func = viz_func
        self.platform_names = list(platforms.keys())
        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        adapters = {}
        decoders = {}
        self.num_patches = {}
        self.patch_dims = {}
        for name, spec in platforms.items():
            T, C, H, W = spec["input_shape"]
            pt, ph, pw = spec["patch_size"]
            if T % pt or H % ph or W % pw:
                raise ValueError(
                    f"Platform '{name}': input_shape {spec['input_shape']} is not "
                    f"divisible by patch_size {spec['patch_size']}"
                )
            adapters[name] = PlatformAdapter(
                in_channels=C, patch_t=pt, patch_h=ph, patch_w=pw, d_model=d_model
            )
            self.num_patches[name] = (T // pt) * (H // ph) * (W // pw)
            self.patch_dims[name] = pt * ph * pw * C

        encoder = SharedBackboneEncoder(
            d_model=d_model,
            d_latent=d_latent,
            num_layers=num_encoder_layers,
            num_heads=num_encoder_heads,
            dim_feedforward=encoder_dim_feedforward or d_model * 4,
            dropout=dropout,
        )

        for name in platforms:
            decoders[name] = PlatformDecoder(
                d_latent=d_latent,
                d_model=d_model,
                patch_dim=self.patch_dims[name],
                num_patches=self.num_patches[name],
                num_layers=decoder_num_layers,
                num_heads=decoder_num_heads,
                dropout=dropout,
            )

        projector = ProjectionHead(d_latent, proj_hidden_dim, d_proj)

        model = nn.ModuleDict(
            {
                "adapters": nn.ModuleDict(adapters),
                "encoder": encoder,
                "decoders": nn.ModuleDict(decoders),
                "projector": projector,
            }
        )
        self.nets = nn.ModuleDict()
        self.nets["policy"] = model

        self.d_proj = d_proj
        self.mask_ratio = mask_ratio
        self.alpha = alpha
        self.margin = margin
        self.contact_threshold = contact_threshold
        self.warmup_steps = warmup_steps
        self.alignment_end_steps = alignment_end_steps
        self.lambda_max = lambda_max
        self.sigmoid_steepness = sigmoid_steepness

        self._sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=sinkhorn_blur,
            scaling=sinkhorn_scaling,
            cost=make_proxy_guided_cost_fn(d_proj, alpha),
        )

        # Step counter driving the two-stage schedule (Phase 3). Incremented
        # once per `forward_training` call, mirroring `egomimic.algo.hpt.HPT`.
        self.training_step = 0

        self.nets = self.nets.float().to(self.device)

    # ------------------------------------------------------------------
    # Algo interface
    # ------------------------------------------------------------------

    @override
    def process_batch_for_training(self, batch):
        """
        Args:
            batch (dict): {platform_name: {"tactile": Tensor[B,T,C,H,W]}} for
                each configured platform.
        Returns:
            batch (dict): same structure, tensors moved to the model device
                and cast to float.
        """
        processed = {}
        for name in self.platform_names:
            if name not in batch:
                raise KeyError(
                    f"Missing platform '{name}' in batch; got keys {list(batch.keys())}"
                )
            tactile = batch[name]["tactile"].to(self.device)
            if tactile.is_floating_point():
                tactile = tactile.float()
            processed[name] = {"tactile": tactile}
        return processed

    @override
    def forward_training(self, batch):
        """
        One iteration of training: adapters -> random masking -> shared
        encoder -> per-platform decoder (MTAE) and projector (OT). Also
        computes the raw and margin-gated OT distance once Stage 2 begins.

        Args:
            batch (dict): output of `process_batch_for_training`.
        Returns:
            predictions (dict): per-platform MTAE losses/latents, the
                current lambda_OT, and the raw/gated OT distance.
        """
        self.training_step += 1
        model = self.nets["policy"]
        adapters, encoder = model["adapters"], model["encoder"]
        decoders, projector = model["decoders"], model["projector"]

        predictions = OrderedDict()
        z_proj = {}
        phi = {}

        for name in self.platform_names:
            x = batch[name]["tactile"]
            tokens, target_patches, _grid = adapters[name](x)
            if target_patches.shape[1] != self.num_patches[name]:
                raise ValueError(
                    f"Platform '{name}': expected {self.num_patches[name]} patches "
                    f"from configured input_shape/patch_size, got "
                    f"{target_patches.shape[1]} from the actual batch tensor "
                    f"(shape {tuple(x.shape)}). Check the platform config."
                )

            visible_tokens, mask, _ids_restore, _ids_keep = random_masking(
                tokens, self.mask_ratio
            )
            z, _tokens_out = encoder(visible_tokens)
            recon = decoders[name](z)

            mask_f = mask.to(recon.dtype)
            mse_per_patch = F.mse_loss(recon, target_patches, reduction="none").mean(-1)
            num_masked = mask_f.sum(dim=1).clamp(min=1.0)
            mtae_loss = ((mse_per_patch * mask_f).sum(dim=1) / num_masked).mean()

            predictions[f"{name}_mtae_loss"] = mtae_loss
            predictions[f"{name}_z"] = z
            phi[name] = compute_proxy_features(
                x, contact_threshold=self.contact_threshold
            )
            z_proj[name] = projector(z)

        lambda_ot = sigmoid_anneal(
            self.training_step,
            self.warmup_steps,
            self.alignment_end_steps,
            self.lambda_max,
            self.sigmoid_steepness,
        )
        predictions["lambda_ot"] = torch.as_tensor(lambda_ot, device=self.device)

        # Stage 1 (warmup): skip the OT branch entirely -- only
        # {A_h, A_r, E_shared, D_h, D_r} are exercised/optimized.
        if self.training_step > self.warmup_steps:
            name_h, name_r = self.platform_names
            combined_h = torch.cat([z_proj[name_h], phi[name_h]], dim=-1)
            combined_r = torch.cat([z_proj[name_r], phi[name_r]], dim=-1)
            w_eps = self._sinkhorn(combined_h, combined_r)
            predictions["ot_raw"] = w_eps
            predictions["ot_loss"] = torch.clamp(w_eps - self.margin, min=0.0)
        else:
            zero = torch.zeros((), device=self.device)
            predictions["ot_raw"] = zero
            predictions["ot_loss"] = zero

        return predictions

    @override
    def forward_eval(self, batch):
        """
        Deployment-time forward pass: adapters + shared encoder only, run on
        the *full* (unmasked) patch sequence for each platform present in the
        batch. This is the representation Phase 4 hands to a downstream
        policy -- see also `export_policy_backbone`.

        Args:
            batch (dict): output of `process_batch_for_training`. May contain
                either or both platforms.
        Returns:
            predictions (dict): {"{platform}_z": Tensor[B, d_latent],
                "{platform}_tokens": Tensor[B, 1+N, d_model]}.
        """
        model = self.nets["policy"]
        adapters, encoder = model["adapters"], model["encoder"]

        predictions = OrderedDict()
        for name in self.platform_names:
            if name not in batch:
                continue
            tokens, _target_patches, _grid = adapters[name](batch[name]["tactile"])
            z, tokens_out = encoder(tokens)
            predictions[f"{name}_z"] = z
            predictions[f"{name}_tokens"] = tokens_out
        return predictions

    @override
    def compute_losses(self, predictions, batch):
        """
        Combine the per-platform MTAE losses with the annealed, margin-gated
        OT loss:
            Stage 1 (step <= K1): L_total = L_MTAE                 (lambda_OT == 0)
            Stage 2 (K1 < step < K2): L_total = L_MTAE + lambda_OT * L_OT
            (step >= K2): lambda_OT == lambda_max

        Args:
            predictions (dict): output of `forward_training`.
            batch (dict): unused (losses are read out of `predictions`,
                matching the ACT/HPT convention), kept for interface parity.
        Returns:
            losses (dict): includes `action_loss`, the total scalar objective
                that `ModelWrapper.training_step` backpropagates through.
        """
        del batch  # unused; losses were already computed in forward_training

        lambda_ot = predictions["lambda_ot"]
        mtae_total = torch.stack(
            [predictions[f"{name}_mtae_loss"] for name in self.platform_names]
        ).sum()
        ot_loss = predictions["ot_loss"]

        losses = OrderedDict()
        for name in self.platform_names:
            losses[f"{name}_mtae_loss"] = predictions[f"{name}_mtae_loss"]
        losses["mtae_loss"] = mtae_total
        losses["ot_raw"] = predictions["ot_raw"]
        losses["ot_loss"] = ot_loss
        losses["lambda_ot"] = lambda_ot
        # Required key: `pl_utils.pl_model.ModelWrapper.training_step` reads
        # `losses["action_loss"]` as the scalar to backpropagate, regardless
        # of algo -- ACT and HPT repurpose the same key for their own totals.
        losses["action_loss"] = mtae_total + lambda_ot * ot_loss
        return losses

    @override
    def log_info(self, info):
        """
        Args:
            info (dict): {"losses": output of `compute_losses`}.
        Returns:
            log (dict): scalar values for tensorboard/wandb logging.
        """
        losses = info["losses"]
        log = OrderedDict()
        log["Loss"] = losses["action_loss"].item()
        for key, value in losses.items():
            if key == "action_loss":
                continue
            log[key] = value.item() if torch.is_tensor(value) else value
        log["stage"] = 2.0 if self.training_step > self.warmup_steps else 1.0
        return log

    # ------------------------------------------------------------------
    # Phase 4: deployment hand-off
    # ------------------------------------------------------------------

    def export_policy_backbone(self) -> TactileEncoderBackbone:
        """
        Freeze the input adapters (A_h, A_r) and the shared backbone encoder
        (E_shared); discard the reconstruction decoders (D_h, D_r) and the
        alignment projection head (g).

        Returns:
            A frozen `TactileEncoderBackbone` producing the latent `z` (and,
            if wanted, the full token sequence) to pass into a downstream
            policy's cross-attention layer alongside visual tokens (e.g. for
            pi_0.5 fine-tuning).
        """
        model = self.nets["policy"]
        return TactileEncoderBackbone(
            adapters=copy.deepcopy(model["adapters"]),
            encoder=copy.deepcopy(model["encoder"]),
        )
