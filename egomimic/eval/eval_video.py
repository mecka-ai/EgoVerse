import os
from abc import abstractmethod

import torch
import torchvision.io as tvio

from egomimic.eval.eval import Eval
from egomimic.rldb.embodiment.embodiment import get_embodiment

# Validation-time dataloader_idx → on-disk subdir / metric-suffix.
# Idx 0 is the canonical eval pass (existing behavior, kept under a clean
# "eval/" subfolder when multi-loader is active so train comparisons sit
# next to it). Idx 1 is the secondary train-split pass enabled by setting
# MultiDataModuleWrapper(include_train_split_in_val=True).
_SPLIT_NAMES: dict[int, str] = {0: "eval", 1: "train_split"}


def _split_name(dataloader_idx: int) -> str:
    return _SPLIT_NAMES.get(dataloader_idx, f"loader_{dataloader_idx}")


class EvalVideo(Eval):
    """
    Base evaluator that buffers per-embodiment frames and writes them out as
    validation videos. Subclasses implement `compute_metrics_and_viz` to compute
    model-specific metrics and produce the frames to buffer.

    When the datamodule enables the train-split val pass
    (``include_train_split_in_val=True``), Lightning provides multiple val
    dataloaders. Videos are then written under split-specific subfolders:

        videos/eval/epoch_<N>/<embodiment>/...        # dataloader_idx 0
        videos/train_split/epoch_<N>/<embodiment>/... # dataloader_idx 1

    and metrics from idx>0 get a ``/train_split`` suffix so they don't collide
    with the canonical ``Valid/...`` keys downstream dashboards already expect.

    With a single val dataloader (the default), behavior is unchanged —
    videos still land in ``videos/epoch_<N>/<embodiment>/`` and metric names
    are not modified.
    """

    def __init__(self, limit_val_batches: int = 400):
        super().__init__()
        self.trainer = None
        self.model = None
        # Buffer keys: (dataloader_idx, embodiment_id) so train-split frames
        # don't accidentally tail-flush into the eval video.
        self.val_image_buffer: dict[tuple[int, int], list[torch.Tensor]] = {}
        self.val_counter: dict[tuple[int, int], int] = {}
        self.override_dict = {
            "strategy": "ddp_find_unused_parameters_true",
            "limit_train_batches": 0,
            "limit_val_batches": limit_val_batches,
            "check_val_every_n_epoch": 1,
            "profiler": "simple",
            "max_epochs": 1,
            "min_epochs": 1,
        }

    def video_dir(self) -> str:
        return os.path.join(self.root_dir(), "videos")

    @abstractmethod
    def compute_metrics_and_viz(self, batch):
        """
        Run the model's eval forward and compute metrics and visualization frames.

        Args:
            batch (dict): processed batch produced by the algo's
                `process_batch_for_training`.
        Returns:
            metrics (dict[str, torch.Tensor | float])
            images_dict (dict[embodiment_id, np.ndarray (B, H, W, 3)])
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Split-aware path / key helpers
    # ------------------------------------------------------------------

    def _is_multi_loader(self) -> bool:
        loaders = getattr(self.trainer, "val_dataloaders", None)
        if loaders is None:
            return False
        # Lightning may expose the val loaders as a list, tuple, or a single
        # object (CombinedLoader) depending on configuration. Multi-loader
        # only when we have a sequence with >1 entries.
        return isinstance(loaders, (list, tuple)) and len(loaders) > 1

    def _split_subdir(self, dataloader_idx: int) -> str:
        """Returns "" for single-loader (legacy layout) or "eval" / "train_split" for multi."""
        return _split_name(dataloader_idx) if self._is_multi_loader() else ""

    def _video_subdir(self, dataloader_idx: int, embodiment_id) -> str:
        return os.path.join(
            self.video_dir(),
            self._split_subdir(dataloader_idx),
            f"epoch_{self.trainer.current_epoch}",
            str(get_embodiment(embodiment_id)),
        )

    def _maybe_relabel_metrics(self, metrics: dict, dataloader_idx: int) -> dict:
        """Append /<split> to metric names when reading from a non-zero loader.

        Idx 0 keeps the existing names so dashboards/queries don't break;
        idx>0 (currently only the train split) is clearly tagged.
        """
        if not self._is_multi_loader() or dataloader_idx == 0:
            return metrics
        suffix = "/" + _split_name(dataloader_idx)
        return {k + suffix: v for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Lightning lifecycle hooks
    # ------------------------------------------------------------------

    def on_validation_start(self):
        if not self.trainer.is_global_zero:
            return
        # Eagerly create the per-split epoch dirs (one when single-loader,
        # two when train-split is enabled) so writers don't have to check.
        loader_count = (
            len(self.trainer.val_dataloaders)
            if isinstance(self.trainer.val_dataloaders, (list, tuple))
            else 1
        )
        for idx in range(loader_count):
            os.makedirs(
                os.path.join(
                    self.video_dir(),
                    self._split_subdir(idx),
                    f"epoch_{self.trainer.current_epoch}",
                ),
                exist_ok=True,
            )

    def on_validation_end(self):
        # Flush any partial buffers, keyed by (dataloader_idx, embodiment_id).
        for key, buffer in self.val_image_buffer.items():
            dataloader_idx, embodiment_id = key
            os.makedirs(
                self._video_subdir(dataloader_idx, embodiment_id), exist_ok=True
            )
            if len(buffer) != 0:
                frames = torch.stack(buffer)
                path = os.path.join(
                    self._video_subdir(dataloader_idx, embodiment_id),
                    f"validation_video_{self.val_counter[key]}.mp4",
                )
                tvio.write_video(path, frames, fps=30, video_codec="h264")
            self.val_counter[key] = 0
            self.val_image_buffer[key] = []

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics, images_dict = self.compute_metrics_and_viz(batch)
        metrics = self._maybe_relabel_metrics(metrics, dataloader_idx)

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }

        ## images is now a dict (embodiment_id -> frames)
        for embodiment_id, images in images_dict.items():
            buf_key = (dataloader_idx, embodiment_id)
            os.makedirs(
                self._video_subdir(dataloader_idx, embodiment_id), exist_ok=True
            )
            if (
                buf_key not in self.val_image_buffer
                or self.val_image_buffer[buf_key] is None
            ):
                self.val_image_buffer[buf_key] = []
                self.val_counter[buf_key] = 0
            self.val_image_buffer[buf_key].extend(torch.from_numpy(images))
            if len(self.val_image_buffer[buf_key]) >= 1000:
                frames = torch.stack(self.val_image_buffer[buf_key])
                path = os.path.join(
                    self._video_subdir(dataloader_idx, embodiment_id),
                    f"validation_video_{self.val_counter[buf_key]}.mp4",
                )
                tvio.write_video(path, frames, fps=30, video_codec="h264")
                self.val_image_buffer[buf_key].clear()
                self.val_counter[buf_key] += 1

        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)
