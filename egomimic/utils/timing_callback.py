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
        fps_log_every_n_batches: int = 50,
    ):
        self.stall_threshold_s = stall_threshold_s
        self.log_ready_after_stall = log_ready_after_stall
        # Log a windowed throughput/frames_per_sec every N batches so fps still
        # updates frequently when an epoch spans many batches (e.g.
        # limit_train_batches=null → a full pass is thousands of batches and the
        # epoch-end number alone would update only every several minutes).
        # Set to 0 to disable intra-epoch logging.
        self.fps_log_every_n_batches = fps_log_every_n_batches
        self._epoch_start: float | None = None
        self._last_batch_end: float | None = None
        self._was_stalled = False
        self._train_start: float | None = None
        self._epoch_frames: int = 0
        self._total_frames: int = 0
        # Windowed (intra-epoch) throughput accumulators, reset each window.
        self._window_start: float | None = None
        self._window_frames: int = 0
        # Sum of GPU-idle waits (gap between batch_end and next batch_start)
        # observed within the current window — i.e. time the train loop blocked
        # on next(dataloader). Lets us report the idle fraction of wall time.
        self._window_wait_s: float = 0.0

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
        if (
            trainer.is_global_zero
            and trainer.logger
            and hasattr(trainer.logger, "experiment")
        ):
            try:
                trainer.logger.experiment.define_metric(
                    "throughput/*", step_metric="wall_time_s"
                )
            except Exception:
                pass

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._epoch_start = time.perf_counter()
        self._last_batch_end = None
        self._was_stalled = False
        self._epoch_frames = 0
        self._window_start = self._epoch_start
        self._window_frames = 0
        self._window_wait_s = 0.0

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

        # Accumulate the GPU-idle wait into the current throughput window so the
        # periodic log can report what fraction of wall time was spent blocked on
        # the dataloader (the headline "where the stall is" number).
        self._window_wait_s += wait_s

        bs = self._collated_batch_size(batch)
        bs_note = f" batch_size={bs}" if bs is not None else ""

        if wait_s >= self.stall_threshold_s:
            log.info(
                f"GPU idle: waited {wait_s:.2f}s for train batch {batch_idx}{bs_note} (no batch ready)"
            )
            self._was_stalled = True
        elif self._was_stalled and self.log_ready_after_stall:
            log.info(
                f"Train batch {batch_idx} ready for GPU (dataloader wait {wait_s:.2f}s){bs_note}"
            )
            self._was_stalled = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        self._last_batch_end = time.perf_counter()
        bs = self._collated_batch_size(batch)
        if bs is None:
            return
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        frames = bs * world_size
        self._epoch_frames += frames
        self._window_frames += frames

        # Periodic intra-epoch throughput so fps updates frequently regardless of
        # epoch length. The first window of an epoch includes the boundary
        # pipeline-refill stall; later windows reflect the steady-state rate.
        n = self.fps_log_every_n_batches
        if (
            n
            and trainer.is_global_zero
            and (batch_idx + 1) % n == 0
            and self._window_start is not None
        ):
            now = self._last_batch_end
            window_s = now - self._window_start
            fps = self._window_frames / max(window_s, 1e-6)
            idle_s = self._window_wait_s
            idle_pct = 100.0 * idle_s / max(window_s, 1e-6)
            wall_time_s = now - self._train_start if self._train_start else 0.0
            # Console first so fps/idle are visible even without a W&B logger.
            log.info(
                f"[fps] step {trainer.global_step}: {fps:.1f} frames/s "
                f"over {window_s:.1f}s ({n} batches) | GPU idle on dataloader "
                f"{idle_s:.1f}s ({idle_pct:.0f}% of window)"
            )
            if trainer.logger:
                trainer.logger.log_metrics(
                    {
                        "wall_time_s": wall_time_s,
                        "throughput/frames_per_sec": fps,
                        "throughput/total_frames": self._total_frames
                        + self._epoch_frames,
                        # Fraction of this window the train loop spent blocked on
                        # the dataloader. High => dataloader-bound; ~0 => GPU-bound.
                        "throughput/gpu_idle_sec": idle_s,
                        "throughput/gpu_idle_pct": idle_pct,
                    },
                    step=trainer.global_step,
                )
            self._window_start = now
            self._window_frames = 0
            self._window_wait_s = 0.0

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if (
            not trainer.is_global_zero
            or self._train_start is None
            or self._epoch_start is None
        ):
            return
        wall_time_s = time.perf_counter() - self._train_start
        epoch_s = time.perf_counter() - self._epoch_start
        self._total_frames += self._epoch_frames
        fps = self._epoch_frames / max(epoch_s, 1e-6)
        if trainer.logger:
            trainer.logger.log_metrics(
                {
                    "wall_time_s": wall_time_s,
                    "throughput/frames_per_sec": fps,
                    "throughput/total_frames": self._total_frames,
                },
                step=trainer.global_step,
            )


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
