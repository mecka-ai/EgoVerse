import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as tvio
from torchmetrics import MeanSquaredError

from egomimic.eval.eval_video import EvalVideo
from egomimic.rldb.embodiment.embodiment import get_embodiment


class WAMEvalVideo(EvalVideo):
    """Evaluator for WAM (World-Action Model). Per embodiment it writes TWO
    separate validation videos:

      1. ``validation_video_*.mp4`` — predicted vs GT **action** overlay drawn on
         the GT observation frame (via ``viz_func``); BC loss + action MSE metrics.
      2. ``predicted_video_*.mp4``  — the world model's **predicted future frames**
         (sampled jointly with the action chunk, VAE-decoded). Written from the
         frames forward_eval stashes on the algo as ``_eval_frames[eid]``.
    """

    def __init__(self, viz_func=None, limit_val_batches: int = 400):
        # This fork's EvalVideo takes only limit_val_batches; the per-embodiment
        # action-overlay viz callables come in via viz_func (evaluator/viz config).
        super().__init__(limit_val_batches=limit_val_batches)
        self.viz_func = viz_func

    def compute_metrics_and_viz(self, batch):
        algo = self.model
        preds = algo.forward_eval(batch)  # samples actions + future frames

        metrics = {}
        images_dict = {}
        mse = MeanSquaredError()
        total_loss = None
        n_loss = 0

        for embodiment_id, _batch in batch.items():
            _batch = algo.data_schematic.unnormalize_data(_batch, embodiment_id)
            name = get_embodiment(embodiment_id).lower()
            ac_key = algo.ac_keys[embodiment_id]

            # The action overlay needs a single GT frame, not the clip — use the
            # current (first) frame of each camera clip.
            for ck in algo.camera_keys.get(embodiment_id, []):
                if (
                    ck in _batch
                    and torch.is_tensor(_batch[ck])
                    and _batch[ck].dim() == 5
                ):
                    _batch[ck] = _batch[ck][:, 0]

            loss_key = f"{name}_loss"
            if loss_key in preds:
                loss_val = preds[loss_key]
                metrics[f"Valid/{loss_key}"] = loss_val
                total_loss = loss_val if total_loss is None else total_loss + loss_val
                n_loss += 1

            pred_key = f"{name}_{ac_key}"
            if pred_key in preds:
                metrics[f"Valid/{pred_key}_paired_mse_avg"] = mse(
                    preds[pred_key].cpu(), _batch[ac_key].cpu()
                )

            # Video 1: action overlay on the GT frame.
            images_dict[embodiment_id] = self._visualize_preds(preds, _batch)

        if total_loss is not None and n_loss > 0:
            metrics["Valid/action_loss"] = total_loss / n_loss

        return metrics, images_dict

    def _visualize_preds(self, predictions, batch):
        if self.viz_func is None:
            raise ValueError("viz_func is not set")
        name = get_embodiment(batch["embodiment"][0].item()).lower()
        return self.viz_func[name](predictions, batch)

    # --- Video 2: predicted future-frame video (separate file) --------------
    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Base writes the action-overlay video + logs metrics (it calls
        # compute_metrics_and_viz, which runs forward_eval and stashes frames).
        super().on_validation_step(batch, batch_idx, dataloader_idx)
        if not hasattr(self, "_pred_buffer"):
            self._pred_buffer = {}
        for eid, video in getattr(self.model, "_eval_frames", {}).items():
            if video is None:
                continue
            clip = self._predicted_clip(video)  # (T, H, W, 3) uint8, one sample
            self._pred_buffer.setdefault(eid, []).extend(torch.from_numpy(clip))

    def on_validation_end(self):
        for eid, buf in getattr(self, "_pred_buffer", {}).items():
            if not buf:
                continue
            out_dir = os.path.join(
                self.video_dir(),
                f"epoch_{self.trainer.current_epoch}",
                str(get_embodiment(eid)),
            )
            os.makedirs(out_dir, exist_ok=True)
            tvio.write_video(
                os.path.join(out_dir, "predicted_video_0.mp4"),
                torch.stack(buf),
                fps=10,
                video_codec="h264",
            )
        self._pred_buffer = {}
        super().on_validation_end()  # writes remaining action-overlay video

    @staticmethod
    def _predicted_clip(video: torch.Tensor, hw=(360, 640)) -> np.ndarray:
        """(B, C, T, H, W) in [-1, 1] -> (T, Hd, Wd, 3) uint8 video of the FIRST
        sample only (no batch tiling), resized to the GT display resolution so
        the aspect ratio matches a real frame (the world model runs at 256x256)."""
        v = video[0].clamp(-1, 1)  # (C, T, H, W)
        v = F.interpolate(
            v.permute(1, 0, 2, 3), size=tuple(hw), mode="bilinear", align_corners=False
        )  # (T, C, Hd, Wd)
        v = ((v + 1.0) * 127.5).to(torch.uint8).cpu()
        return v.permute(0, 2, 3, 1).numpy()  # (T, Hd, Wd, C)
