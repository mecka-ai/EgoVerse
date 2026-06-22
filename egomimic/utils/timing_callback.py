from __future__ import annotations

import os
import time

import torch
from pytorch_lightning import Callback

from egomimic.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DataLoaderStallLogger(Callback):
    """Log when the training loop waits on the DataLoader (GPU has no batch yet).

    Measures time between ``on_train_batch_end`` and the next ``on_train_batch_start``,
    which is when Lightning blocks on ``next(train_dataloader)`` after the previous step.

    Also logs throughput/frames_per_sec and throughput/total_frames to WandB at
    epoch end with wall_time_s as the x-axis (Charts section).
    """

    def __init__(
        self,
        stall_threshold_s: float = 0.5,
        log_ready_after_stall: bool = True,
    ):
        self.stall_threshold_s = stall_threshold_s
        self.log_ready_after_stall = log_ready_after_stall
        self._epoch_start: float | None = None
        self._last_batch_end: float | None = None
        self._was_stalled = False
        self._train_start: float | None = None
        self._epoch_frames: int = 0
        self._total_frames: int = 0
        self._gs_datasets: list = []  # GlobalShuffleShardDataset instances on this rank

    @classmethod
    def _collated_batch_size(cls, batch) -> int | None:
        if not isinstance(batch, dict):
            return None
        # CombinedLoader: {"mecka_bimanual": {tensor keys...}}
        for value in batch.values():
            if isinstance(value, dict):
                nested = cls._collated_batch_size(value)
                if nested is not None:
                    return nested
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
        return None

    def on_train_start(self, trainer, pl_module) -> None:
        self._train_start = time.perf_counter()
        self._total_frames = 0
        # Discover any GlobalShuffleShardDataset instances so we can drain their
        # download metrics at epoch end. Only runs once; zero cost during training.
        self._gs_datasets = []
        dm = getattr(trainer, "datamodule", None)
        if dm is not None:
            for ds in getattr(dm, "train_datasets", {}).values():
                if hasattr(ds, "drain_download_metrics"):
                    self._gs_datasets.append(ds)
        if trainer.is_global_zero and trainer.logger and hasattr(trainer.logger, "experiment"):
            try:
                trainer.logger.experiment.define_metric("throughput/*", step_metric="wall_time_s")
            except Exception:
                pass

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start = time.perf_counter()
        self._last_batch_end = None
        self._was_stalled = False
        self._epoch_frames = 0

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        if not trainer.is_global_zero:
            return

        now = time.perf_counter()
        if self._last_batch_end is not None:
            wait_s = now - self._last_batch_end
        elif self._epoch_start is not None:
            wait_s = now - self._epoch_start
        else:
            return

        bs = self._collated_batch_size(batch)
        bs_note = f" batch_size={bs}" if bs is not None else ""

        if wait_s >= self.stall_threshold_s:
            log.info(f"GPU idle: waited {wait_s:.2f}s for train batch {batch_idx}{bs_note} (no batch ready)")
            self._was_stalled = True
        elif self._was_stalled and self.log_ready_after_stall:
            log.info(f"Train batch {batch_idx} ready for GPU (dataloader wait {wait_s:.2f}s){bs_note}")
            self._was_stalled = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._last_batch_end = time.perf_counter()
        bs = self._collated_batch_size(batch)
        if bs is not None:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            self._epoch_frames += bs * world_size

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if not trainer.is_global_zero or self._train_start is None or self._epoch_start is None:
            return
        now = time.perf_counter()
        wall_time_s = now - self._train_start
        epoch_s = now - self._epoch_start
        self._total_frames += self._epoch_frames
        fps = self._epoch_frames / max(epoch_s, 1e-6)

        metrics: dict = {
            "wall_time_s": wall_time_s,
            "throughput/frames_per_sec": fps,
            "throughput/total_frames": self._total_frames,
        }

        # Drain download stats from GlobalShuffleShardDataset (rank 0 only).
        # put_nowait in the downloader ensures this never blocked training.
        if self._gs_datasets:
            total_shards, total_bytes = 0, 0.0
            for ds in self._gs_datasets:
                n, b, _ = ds.drain_download_metrics()
                total_shards += n
                total_bytes += b
            if epoch_s > 0:
                world_size = getattr(trainer, "world_size", 1) or 1
                # per-rank stats (rank 0)
                metrics["throughput/dl_shards_per_sec"] = total_shards / epoch_s
                metrics["throughput/dl_mb_per_sec"] = total_bytes / epoch_s / 1e6
                # total across all ranks (estimated — each rank downloads its own slice)
                metrics["throughput/dl_total_shards_per_sec"] = total_shards / epoch_s * world_size
                metrics["throughput/dl_total_mb_per_sec"] = total_bytes / epoch_s / 1e6 * world_size

        if trainer.logger:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)


class WandbProfilerLogger(Callback):
    """Logs Lightning profiler durations to W&B every N steps."""

    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps != 0:
            return

        if trainer.profiler is not None and hasattr(
            trainer.profiler, "recorded_durations"
        ):
            metrics_to_log = {}
            for action_name, durations in trainer.profiler.recorded_durations.items():
                if len(durations) > 0:
                    recent_time = durations[-1]
                    metrics_to_log[f"profiler/{action_name}_time_sec"] = recent_time

            if metrics_to_log and trainer.logger:
                trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)
