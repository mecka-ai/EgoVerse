"""Mecka fork of the QueST SkillVAE + a vision obs encoder for conditioning.

``SkillVAEMecka`` subclasses ``quest.algos.quest_modules.skill_vae.SkillVAE`` with one
change: the additive positional embedding in ``encode`` is optional
(``use_positional_emb=False`` removes it). The causal conv stack already provides
implicit position, and the decoder keeps its own ``fixed_positional_emb`` for queries,
so reconstruction still works — but the code a token gets assigned no longer carries
"I am token k" (FSQ codes were position-dominated: NMI 0.28 position vs 0.07 language).

``ImageObsEncoder`` produces ``obs_emb`` for SkillVAE's (previously unused) vision
conditioning: frozen-architecture torchvision backbone -> linear projection ->
``(B, 1, emb_dim)`` token that both ``encode`` and ``decode`` attend to.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from quest.algos.quest_modules.skill_vae import SkillVAE


class SkillVAEMecka(SkillVAE):
    """SkillVAE with an optional encoder positional embedding."""

    def __init__(self, *args, use_positional_emb: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_positional_emb = bool(use_positional_emb)

    def encode(self, act, obs_emb=None):
        # Mirrors SkillVAE.encode, with add_positional_emb gated by the flag.
        x = self.action_proj(act)
        x = self.conv_block(x)
        B, H, D = x.shape

        if obs_emb is not None:
            x = torch.cat([obs_emb, x], dim=1)
        if self.use_positional_emb:
            x = self.add_positional_emb(x)

        if self.use_causal_encoder:
            mask = nn.Transformer.generate_square_subsequent_mask(
                x.size(1), device=x.device
            )
            x = self.encoder(x, mask=mask, is_causal=True)
        else:
            x = self.encoder(x)

        x = x[:, -H:]
        return x


# ImageNet normalization constants (torchvision pretrained backbones).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageObsEncoder(nn.Module):
    """Encode the ego image into a single obs token for SkillVAE conditioning.

    Input: ``(B, C, H, W)`` uint8 [0,255] or float [0,1] (camera keys pass through
    DataSchematic.normalize_data untouched). Output: ``(B, 1, emb_dim)``.
    """

    def __init__(
        self,
        emb_dim: int,
        pretrained: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        import torchvision

        weights = (
            torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        backbone = torchvision.models.resnet18(weights=weights)
        # Strip the fc head; keep global-avg-pooled (B, 512, 1, 1) features.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(512, emb_dim)
        self.image_size = int(image_size)
        self.register_buffer(
            "img_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "img_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.float()
        if x.max() > 2.0:  # uint8-range input
            x = x / 255.0
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
        x = (x - self.img_mean) / self.img_std
        feat = self.backbone(x).flatten(start_dim=1)  # (B, 512)
        return self.proj(feat).unsqueeze(1)           # (B, 1, emb_dim)
