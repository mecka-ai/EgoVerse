"""Minimal EgoVideo visual-encoder loader.

Loads PretrainVisionTransformer + image_projection from a checkpoint without
instantiating the text encoder or requiring easydict.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from .vision_encoder import PretrainVisionTransformer

logger = logging.getLogger(__name__)


def build_visual_model(
    ckpt_path: str,
    num_frames: int = 4,
) -> tuple[PretrainVisionTransformer, nn.Parameter]:
    """
    Instantiate the EgoVideo ViT-H/14 visual encoder and load weights.

    Returns:
        visual: PretrainVisionTransformer — 40-block ViT, output shape (B, 768)
                via the internal AttentionPoolingBlock (clip_projector).
        image_projection: nn.Parameter of shape (768, 512) — projects visual
                features to the shared 512-dim CLIP embedding space.

    Usage:
        visual, image_proj = build_visual_model(ckpt_path, num_frames=4)
        out = visual(video_tensor)   # video_tensor: (B, 3, T, 224, 224)
        feats = out[1] @ image_proj  # (B, 512)
        feats = F.normalize(feats, dim=-1)
    """
    visual = PretrainVisionTransformer(
        img_size=224,
        num_frames=num_frames,
        tubelet_size=1,
        patch_size=14,
        embed_dim=1408,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type="l2",
        clip_return_layer=6,
        clip_student_return_interval=1,
        use_checkpoint=False,
        checkpoint_num=0,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        sep_image_video_pos_embed=False,
    )

    image_projection = nn.Parameter(torch.empty(768, 512))
    nn.init.trunc_normal_(image_projection, std=768 ** -0.5)

    if ckpt_path:
        logger.info("Loading EgoVideo checkpoint from %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Strip DataParallel 'module.' prefix
        new_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

        # Visual encoder: keys are 'visual.*' → strip prefix
        visual_weights = {
            k[len("visual."):]: v
            for k, v in new_ckpt.items()
            if k.startswith("visual.")
        }
        missing, unexpected = visual.load_state_dict(visual_weights, strict=False)
        if missing:
            logger.warning("EgoVideo visual encoder — missing keys: %s", missing[:5])
        if unexpected:
            logger.warning("EgoVideo visual encoder — unexpected keys: %s", unexpected[:5])

        if "image_projection" in new_ckpt:
            image_projection = nn.Parameter(new_ckpt["image_projection"].float())
            logger.info("Loaded image_projection from checkpoint (shape: %s)", image_projection.shape)
        else:
            logger.warning("image_projection not found in checkpoint — using random init")

    return visual, image_projection
