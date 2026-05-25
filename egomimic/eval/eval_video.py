import os
from abc import abstractmethod

import torch
import torchvision.io as tvio

from egomimic.eval.eval import Eval
from egomimic.rldb.embodiment.embodiment import get_embodiment

# Maps validation dataloader_idx → on-disk subfolder name.
# idx 0 is the canonical eval split; idx 1 is the optional train-split pass.
_SPLIT_DIRS: dict[int, str] = {0: "eval", 1: "train_eval"}


def _split_name(dataloader_idx: int) -> str:
    return _SPLIT_DIRS.get(dataloader_idx, f"loader_{dataloader_idx}")


class EvalVideo(Eval):
    """
    Base evaluator that buffers per-embodiment frames and writes them as
    validation videos. Subclasses implement ``compute_metrics_and_viz``.

    Single val dataloader (default)
    --------------------------------
    Videos land in ``videos/epoch_<N>/<embodiment>/`` and metric names are
    unchanged — identical to the pre-split behavior.

    Multiple val dataloaders (``include_train_split_in_val=True``)
    ---------------------------------------------------------------
    Lightning assigns each loader a ``dataloader_idx``. Videos are written
    under split-specific subdirectories:

        videos/eval/epoch_<N>/<embodiment>/         # idx 0
        videos/train_split/epoch_<N>/<embodiment>/  # idx 1

    Metrics from idx > 0 get a ``/<split>`` suffix so they don't collide
    with the ``Valid/...`` keys that downstream dashboards expect.
    """

    def __init__(self, limit_val_batches: int = 400):
        super().__init__()
        self.trainer = None
        self.model = None
        # Keyed by (dataloader_idx, embodiment_id) so train and eval frames
        # never share a buffer.
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
        Run the model's eval forward and produce metrics and visualization frames.

        Args:
            batch (dict): processed batch from ``process_batch_for_training``.
        Returns:
            metrics (dict[str, torch.Tensor | float])
            images_dict (dict[embodiment_id, np.ndarray (B, H, W, 3)])
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Path / metric helpers
    # ------------------------------------------------------------------

    def _is_multi_loader(self) -> bool:
        loaders = getattr(self.trainer, "val_dataloaders", None)
        return isinstance(loaders, (list, tuple)) and len(loaders) > 1

    def _split_subdir(self, dataloader_idx: int) -> str:
        """Returns "" for single-loader (legacy layout) or the split name for multi."""
        return _split_name(dataloader_idx) if self._is_multi_loader() else ""

    def _video_subdir(self, dataloader_idx: int, embodiment_id) -> str:
        return os.path.join(
            self.video_dir(),
            self._split_subdir(dataloader_idx),
            f"epoch_{self.trainer.current_epoch}",
            str(get_embodiment(embodiment_id)),
        )

    def _relabel_metrics(self, metrics: dict, dataloader_idx: int) -> dict:
        """Append /<split> to metric keys for non-zero loaders.

        idx 0 keeps names unchanged for backward compatibility with dashboards.
        """
        split = self._split_subdir(dataloader_idx)
        if not split or dataloader_idx == 0:
            return metrics
        return {f"{k}/{split}": v for k, v in metrics.items()}

    def _write_video(self, buf_key: tuple[int, int]) -> None:
        """Write the current buffer contents for buf_key to disk."""
        dataloader_idx, embodiment_id = buf_key
        subdir = self._video_subdir(dataloader_idx, embodiment_id)
        os.makedirs(subdir, exist_ok=True)
        path = os.path.join(subdir, f"validation_video_{self.val_counter[buf_key]}.mp4")
        tvio.write_video(
            path, torch.stack(self.val_image_buffer[buf_key]), fps=30, video_codec="h264"
        )

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_validation_start(self):
        if not self.trainer.is_global_zero:
            return
        loaders = getattr(self.trainer, "val_dataloaders", None)
        n_loaders = len(loaders) if isinstance(loaders, (list, tuple)) else 1
        for idx in range(n_loaders):
            os.makedirs(
                os.path.join(
                    self.video_dir(),
                    self._split_subdir(idx),
                    f"epoch_{self.trainer.current_epoch}",
                ),
                exist_ok=True,
            )

    def on_validation_end(self):
        for buf_key, buffer in self.val_image_buffer.items():
            if buffer:
                self._write_video(buf_key)
            self.val_image_buffer[buf_key] = []
            self.val_counter[buf_key] = 0

    def on_validation_step(self, batch, batch_idx, dataloader_idx=0):
        metrics, images_dict = self.compute_metrics_and_viz(batch)
        metrics = self._relabel_metrics(metrics, dataloader_idx)

        device = self.trainer.lightning_module.device
        metrics = {
            k: (v.to(device) if torch.is_tensor(v) else torch.tensor(v, device=device))
            for k, v in metrics.items()
        }

        for embodiment_id, images in images_dict.items():
            buf_key = (dataloader_idx, embodiment_id)
            if buf_key not in self.val_image_buffer:
                self.val_image_buffer[buf_key] = []
                self.val_counter[buf_key] = 0
            self.val_image_buffer[buf_key].extend(torch.from_numpy(images))
            if len(self.val_image_buffer[buf_key]) >= 1000:
                self._write_video(buf_key)
                self.val_image_buffer[buf_key].clear()
                self.val_counter[buf_key] += 1

        self.trainer.lightning_module.log_dict(metrics, sync_dist=True)
