"""Validation evaluator for ActionVAETrainer.

Each validation step:
  - Runs the VAE forward pass to reconstruct action chunks
  - Logs per-embodiment reconstruction MSE
  - Renders GT (green) vs reconstruction (red) trajectory overlays on the ego
    image when ``viz_func`` is configured (same as OAT tokenizer / HPT eval)
  - Falls back to per-sample action-trajectory plots when no viz_func is available
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment

log = logging.getLogger(__name__)


def _action_recon_plot_frames(gt: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """Render GT vs reconstruction action trajectories as RGB frames.

    Args:
        gt, recon: (B, S, D) float numpy arrays.

    Returns:
        (B, H, W, 3) uint8 numpy array.
    """
    import io

    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = []
    target_size = (640, 480)  # W, H — fixed so batched video frames stack cleanly.
    horizon = gt.shape[1]
    t = np.arange(horizon)
    # Plot left/right hand xyz (dims 0-2 and 6-8 for 12-dim cartesian chunks).
    plot_dims = [(0, "Lx"), (1, "Ly"), (2, "Lz"), (6, "Rx"), (7, "Ry"), (8, "Rz")]
    plot_dims = [(d, label) for d, label in plot_dims if d < gt.shape[-1]]

    for b in range(gt.shape[0]):
        fig, axes = plt.subplots(
            len(plot_dims), 1, figsize=(6, 2 * len(plot_dims)), sharex=True
        )
        if len(plot_dims) == 1:
            axes = [axes]
        for ax, (dim, label) in zip(axes, plot_dims):
            ax.plot(t, gt[b, :, dim], color="green", linewidth=1.5, label="GT")
            ax.plot(
                t,
                recon[b, :, dim],
                color="red",
                linewidth=1.5,
                alpha=0.85,
                label="Recon",
            )
            ax.set_ylabel(label, fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=7)
        axes[-1].set_xlabel("timestep", fontsize=8)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        frames.append(img)
    return np.stack(frames, axis=0)


def _find_viz_image_key(algo, embodiment_id: int, batch: dict) -> str | None:
    """Pick a camera key present in the batch for trajectory overlays."""
    for key in algo.data_schematic.keys_of_type("camera_keys", embodiment_id):
        if key in batch:
            return key
    return None


class ActionVAEEvalVideo(EvalVideo):
    """Validation evaluator for ActionVAETrainer."""

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)

        metrics: dict = {}
        images_dict: dict = {}

        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]
            recon_key = f"{embodiment_name}_recon"
            pred_viz_key = f"{embodiment_name}_{ac_key}"

            if recon_key not in preds:
                continue

            actions = _batch[ac_key]
            recons = preds[recon_key]
            metrics[f"Valid/{embodiment_name}_recon_mse"] = F.mse_loss(recons, actions)

            if not self.trainer.is_global_zero:
                continue

            unnorm_batch = algo.data_schematic.unnormalize_data(_batch, embodiment_id)
            unnorm_recon = algo.data_schematic.unnormalize_data(
                {ac_key: recons}, embodiment_id
            )[ac_key]

            ims = None
            viz_fn = None if algo.viz_func is None else algo.viz_func.get(embodiment_name)
            img_key = _find_viz_image_key(algo, embodiment_id, unnorm_batch)
            if viz_fn is not None and img_key is not None:
                try:
                    ims = viz_fn({pred_viz_key: unnorm_recon}, unnorm_batch)
                except Exception as exc:
                    log.warning(
                        "ActionVAEEvalVideo: viz_func failed for %s (%r); using plot fallback",
                        embodiment_name,
                        exc,
                    )

            if ims is None:
                gt_np = unnorm_batch[ac_key].detach().cpu().numpy()
                recon_np = unnorm_recon.detach().cpu().numpy()
                ims = _action_recon_plot_frames(gt_np, recon_np)

            images_dict[embodiment_id] = ims

        return metrics, images_dict
