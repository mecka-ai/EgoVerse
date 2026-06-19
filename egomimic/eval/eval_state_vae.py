"""Validation evaluator for StateVAETrainer.

Each validation step:
  - Runs the CNN VAE forward pass to reconstruct egocentric images
  - Logs per-embodiment reconstruction MSE
  - Renders side-by-side GT | reconstruction frames for validation videos
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment

_LABEL_H = 24
_DIVIDER_W = 4


def _state_recon_frames(gt: torch.Tensor, recon: torch.Tensor) -> np.ndarray:
    """Build side-by-side GT | reconstruction frames.

    Args:
        gt, recon: (B, C, H, W) float tensors in [0, 1].

    Returns:
        (B, H + label_h, 2*W + divider, 3) uint8 numpy array.
    """
    gt_u8 = (
        gt.detach().clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0
    ).astype(np.uint8)
    recon_u8 = (
        recon.detach().clamp(0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy() * 255.0
    ).astype(np.uint8)

    frames = []
    for b in range(gt_u8.shape[0]):
        h, w = gt_u8[b].shape[:2]
        canvas = np.full((h + _LABEL_H, 2 * w + _DIVIDER_W, 3), 255, dtype=np.uint8)
        canvas[_LABEL_H:, :w] = gt_u8[b]
        canvas[_LABEL_H:, w + _DIVIDER_W :] = recon_u8[b]
        canvas[_LABEL_H : _LABEL_H + h, w : w + _DIVIDER_W] = 180
        cv2.putText(
            canvas,
            "GT",
            (8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 128, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Recon",
            (w + _DIVIDER_W + 8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 200),
            1,
            cv2.LINE_AA,
        )
        frames.append(canvas)
    return np.stack(frames, axis=0)


class StateVAEEvalVideo(EvalVideo):
    """Validation evaluator for StateVAETrainer."""

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics: dict = {}
        images_dict: dict = {}

        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            recon_key = f"{embodiment_name}_recon"

            if recon_key not in preds:
                continue

            gt = _batch["image"]
            recon = preds[recon_key]
            metrics[f"Valid/{embodiment_name}_recon_mse"] = F.mse_loss(recon, gt)

            if not self.trainer.is_global_zero:
                continue

            images_dict[embodiment_id] = _state_recon_frames(gt, recon)

        return metrics, images_dict
