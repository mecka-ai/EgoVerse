import os
from abc import abstractmethod

import torch
import torchvision.io as tvio

from egomimic.eval.eval import Eval
from egomimic.rldb.embodiment.embodiment import get_embodiment


class EvalVideo(Eval):
    """
    Base evaluator that buffers per-embodiment frames and writes them out as
    validation videos. Subclasses implement `compute_metrics_and_viz` to compute
    model-specific metrics and produce the frames to buffer.

    Eval is split into a cheap part (metrics, computed every validation pass) and
    an expensive part (per-embodiment visualization + mp4 writes to the shared
    volume). The expensive viz is gated by `viz_every_n_steps` so it runs far less
    often than the metrics. Gating is **step based** (trainer.global_step) to match
    a step-based training regime (val_check_interval in steps). `viz_every_n_steps`
    should be an integer multiple of trainer.val_check_interval so the modulo lines
    up with validation passes (e.g. val_check_interval=2000, viz_every_n_steps=10000
    -> metrics every 2000 steps, viz every 5th validation = every 10000 steps).
    """

    def __init__(self, limit_val_batches: int = 400, viz_every_n_steps: int = 1):
        super().__init__()
        self.trainer = None
        self.model = None
        self.val_image_buffer = {}
        self.val_counter = {}
        self.viz_every_n_steps = viz_every_n_steps
        self.override_dict = {
            "strategy": "ddp_find_unused_parameters_true",
            "limit_train_batches": 0,
            "limit_val_batches": limit_val_batches,
            "check_val_every_n_epoch": 1,
            "profiler": "simple",
            "max_epochs": 1,
            "min_epochs": 1,
        }

    def video_dir(self):
        return os.path.join(self.root_dir(), "videos")

    def _viz_subdir(self) -> str:
        """Per-eval output subdir, keyed by global step (step-based regime)."""
        return f"step_{self.trainer.global_step}"

    def _should_viz(self) -> bool:
        """Whether to render/write visualization videos this validation pass.

        Cheap metrics (MSE) are logged every validation pass regardless; only the
        expensive viz is gated. Fires when trainer.global_step is a multiple of
        viz_every_n_steps. viz_every_n_steps <= 0 disables viz entirely. In
        standalone eval mode global_step == 0, so viz always fires there.
        """
        if not self.viz_every_n_steps or self.viz_every_n_steps <= 0:
            return False
        return (int(self.trainer.global_step) % int(self.viz_every_n_steps)) == 0

    @abstractmethod
    def compute_metrics_and_viz(self, batch, do_viz=True):
        """
        Run the model's eval forward and compute metrics and visualization frames.

        Args:
            batch (dict): processed batch produced by the algo's
                `process_batch_for_training`.
            do_viz (bool): when False, skip the expensive per-embodiment
                visualization and return an empty images dict (metrics still
                computed). The caller passes the result of `_should_viz()`.
        Returns:
            metrics (dict[str, torch.Tensor | float])
            images_dict (dict[embodiment_id, np.ndarray (B, H, W, 3)])
        """
        raise NotImplementedError

    def on_validation_start(self):
        if self.trainer.is_global_zero and self._should_viz():
            os.makedirs(
                os.path.join(self.video_dir(), self._viz_subdir()),
                exist_ok=True,
            )

    def on_validation_end(self):
        # Rank-0 only, matching on_validation_step / on_validation_start: non-zero
        # ranks never buffered frames, but guard explicitly so the residual mp4
        # flush below stays single-writer and never collides across DDP ranks.
        if not (self._should_viz() and self.trainer.is_global_zero):
            return
        for key, buffer in self.val_image_buffer.items():
            os.makedirs(
                os.path.join(
                    self.video_dir(),
                    self._viz_subdir(),
                    str(get_embodiment(key)),
                ),
                exist_ok=True,
            )
            if len(buffer) != 0:
                frames = torch.stack(buffer)
                path = os.path.join(
                    self.video_dir(),
                    self._viz_subdir(),
                    str(get_embodiment(key)),
                    f"validation_video_{self.val_counter[key]}.mp4",
                )
                tvio.write_video(path, frames, fps=30, video_codec="h264")

            self.val_counter[key] = 0
            self.val_image_buffer[key] = []

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Viz (frame buffering + mp4 writes) is rank-0 only. Under DDP this hook
        # runs on every rank, each holding a different val shard, but the output
        # path is identical across ranks
        # (step_{global_step}/<embodiment>/validation_video_{counter}.mp4) because
        # global_step and val_counter are DDP-synchronized. Writing from all ranks
        # streams to the same file concurrently -> corrupt mp4 / silently dropped
        # frames. Gating do_viz on is_global_zero also skips wasted frame rendering
        # on non-zero ranks. Metrics are unaffected: they are computed regardless
        # of do_viz and logged on every rank for the sync_dist all-reduce below.
        do_viz = self._should_viz() and self.trainer.is_global_zero
        metrics, images_dict = self.compute_metrics_and_viz(batch, do_viz=do_viz)

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }

        ## images is now a dict
        if do_viz:
            for key, images in images_dict.items():
                os.makedirs(
                    os.path.join(
                        self.video_dir(),
                        self._viz_subdir(),
                        str(get_embodiment(key)),
                    ),
                    exist_ok=True,
                )
                if (
                    key not in self.val_image_buffer
                    or self.val_image_buffer[key] is None
                ):
                    self.val_image_buffer[key] = []
                    self.val_counter[key] = 0
                self.val_image_buffer[key].extend(torch.from_numpy(images))
                if len(self.val_image_buffer[key]) >= 1000:
                    frames = torch.stack(self.val_image_buffer[key])
                    path = os.path.join(
                        self.video_dir(),
                        self._viz_subdir(),
                        str(get_embodiment(key)),
                        f"validation_video_{self.val_counter[key]}.mp4",
                    )
                    tvio.write_video(path, frames, fps=30, video_codec="h264")
                    self.val_image_buffer[key].clear()
                    self.val_counter[key] += 1

        # add_dataloader_idx=False keeps metric key names stable. When train_viz
        # is enabled, val_dataloader returns [eval, train_viz] and Lightning would
        # otherwise append "/dataloader_idx_0" / "/dataloader_idx_1" to every key,
        # diverging from the single-loader (train_viz-disabled) names and from the
        # held-out loss keys (pl_model already logs those with add_dataloader_idx=
        # False). The eval ("Valid/...") and train_viz ("train_viz/Valid/...") keys
        # are already disambiguated by the train_viz/ prefix, so no collision.
        self.trainer.lightning_module.log_dict(
            metrics, sync_dist=True, add_dataloader_idx=False
        )
