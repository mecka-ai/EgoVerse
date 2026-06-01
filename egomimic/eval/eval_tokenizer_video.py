"""Validation evaluator for the OAT action tokenizer.

Each validation step:
  - Encodes GT action chunks through the tokenizer encoder + quantizer
  - Decodes them back to action space
  - Draws GT (green) and reconstruction (red) overlays on the ego image
  - Logs per-embodiment reconstruction MSE

Videos are written as H.264 MP4 at 30 fps into
``<run_dir>/videos/epoch_<N>/<EmbodimentName>/validation_video_<K>.mp4``
and also pushed to W&B (when a WandbLogger is attached) under
``videos/<embodiment_name>/epoch_<N>_chunk_<K>``.
"""

import logging
import os

import torch
import torch.nn.functional as F
import torchvision.io as tvio

from egomimic.eval.eval import Eval
from egomimic.rldb.embodiment.embodiment import get_embodiment

log = logging.getLogger(__name__)

# Flush buffered frames to disk + W&B every N frames so per-epoch video files
# stay roughly 30-60s long at 30 fps.
_FLUSH_FRAMES = 1000


class EvalTokenizerVideo(Eval):
    """Validation evaluator for the OAT tokenizer.

    Each step: encodes GT actions -> decodes -> draws GT (green) and
    reconstruction (red) on the ego image. Also logs reconstruction MSE.
    Renders videos via the per-embodiment ``model.viz_func`` (typically
    ``Mecka.viz_gt_preds`` from ``hydra_configs/visualization/cartesian.yaml``).
    """

    def __init__(self):
        super().__init__()
        self.trainer = None
        self.model = None
        self.val_image_buffer = {}
        self.val_counter = {}
        self.override_dict = {
            "strategy": "ddp_find_unused_parameters_true",
            "limit_train_batches": 0,
            "limit_val_batches": 400,
            "check_val_every_n_epoch": 1,
            "profiler": "simple",
            "max_epochs": 1,
            "min_epochs": 1,
        }

    def video_dir(self):
        return os.path.join(self.root_dir(), "videos")

    def _wandb_logger(self):
        """Return the WandbLogger experiment handle, or None if not attached."""
        if self.trainer is None or not self.trainer.is_global_zero:
            return None
        loggers = self.trainer.loggers or []
        for lgr in loggers:
            # Lightning's WandbLogger exposes .experiment (the wandb.Run).
            exp = getattr(lgr, "experiment", None)
            if exp is None:
                continue
            # Duck-type: a wandb Run exposes .log and .id.
            if hasattr(exp, "log") and hasattr(exp, "id"):
                return exp
        return None

    def _flush_buffer(self, embodiment_id):
        """Write buffered frames to disk + log to W&B, then clear the buffer."""
        if not self.trainer.is_global_zero:
            return
        buffer = self.val_image_buffer.get(embodiment_id, [])
        if not buffer:
            return

        embodiment = get_embodiment(embodiment_id)
        embodiment_name = embodiment.lower()
        epoch = self.trainer.current_epoch
        chunk_idx = self.val_counter[embodiment_id]

        out_dir = os.path.join(
            self.video_dir(),
            f"epoch_{epoch}",
            str(embodiment),
        )
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"validation_video_{chunk_idx}.mp4")

        frames = torch.stack(buffer)
        tvio.write_video(path, frames, fps=30, video_codec="h264")
        log.info(
            "[EvalTokenizerVideo] wrote %d frames -> %s",
            frames.shape[0],
            path,
        )

        wandb_run = self._wandb_logger()
        if wandb_run is not None:
            try:
                import wandb

                wandb_run.log(
                    {
                        f"videos/{embodiment_name}/recon_overlay": wandb.Video(
                            path, fps=30, format="mp4",
                            caption=f"epoch={epoch} chunk={chunk_idx}",
                        ),
                        "epoch": epoch,
                    }
                )
            except Exception as e:  # pragma: no cover — never block training on viz
                log.warning(
                    "[EvalTokenizerVideo] wandb video upload failed for %s: %s",
                    path, e,
                )

        self.val_image_buffer[embodiment_id] = []
        self.val_counter[embodiment_id] = chunk_idx + 1

    def on_validation_start(self):
        if self.trainer.is_global_zero:
            os.makedirs(
                os.path.join(self.video_dir(), f"epoch_{self.trainer.current_epoch}"),
                exist_ok=True,
            )

    def on_validation_end(self):
        # Flush whatever frames are left at end-of-epoch and reset state.
        for embodiment_id in list(self.val_image_buffer.keys()):
            if self.val_image_buffer[embodiment_id]:
                self._flush_buffer(embodiment_id)
            self.val_counter[embodiment_id] = 0
            self.val_image_buffer[embodiment_id] = []

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        model = self.model  # OATTokenizerTrainer

        # Run encode -> decode once for all embodiments in the batch.
        recons_dict = model.forward_eval(batch)

        metrics = {}
        for embodiment_id, _batch in batch.items():
            embodiment_name = get_embodiment(embodiment_id).lower()
            ac_key = model.ac_keys_by_id[embodiment_id]
            recons_key = f"{embodiment_name}_{ac_key}"

            actions = _batch[ac_key]               # normalized GT actions
            recons = recons_dict[recons_key]       # normalized reconstructions

            metrics[f"Valid/{embodiment_name}_reconst_mse"] = F.mse_loss(
                recons, actions
            )

            # Only rank-0 buffers frames + writes videos. viz_func may also be
            # None when an embodiment has no configured visualizer.
            if not self.trainer.is_global_zero:
                continue
            if model.viz_func is None or embodiment_name not in model.viz_func:
                continue

            # Unnormalize GT batch (actions + any image keys), then unnormalize
            # reconstructions through the same schematic so both live in the
            # original action space when fed to viz_func.
            unnorm_batch = model.data_schematic.unnormalize_data(_batch, embodiment_id)
            unnorm_recons = model.data_schematic.unnormalize_data(
                {ac_key: recons}, embodiment_id
            )[ac_key]

            # viz_gt_preds expects {f"{embodiment_name}_{ac_key}": pred_actions}
            predictions = {recons_key: unnorm_recons}
            ims = model.viz_func[embodiment_name](predictions, unnorm_batch)
            # ims: (B, H, W, 3) numpy uint8

            if embodiment_id not in self.val_image_buffer:
                self.val_image_buffer[embodiment_id] = []
                self.val_counter[embodiment_id] = 0

            self.val_image_buffer[embodiment_id].extend(torch.from_numpy(ims))

            if len(self.val_image_buffer[embodiment_id]) >= _FLUSH_FRAMES:
                self._flush_buffer(embodiment_id)

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }
        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)
