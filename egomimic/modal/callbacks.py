"""Modal-specific Lightning callbacks.

Importable by any Modal function after the repo has been cloned.
No egomimic imports at module level — only stdlib and lightning.
"""

from __future__ import annotations

import json
import logging
import os
import time

from lightning import Callback, Trainer

log = logging.getLogger(__name__)


class ModalAutoRestartCallback(Callback):
    """Save a checkpoint and spawn a detached continuation job ~30 min before
    the Modal container timeout, then stop the current run gracefully.

    Enabled automatically when MODAL_IS_REMOTE=1 and MODAL_TIMEOUT_SECONDS are
    set. trainModal.py injects these env vars into the container via run_hydra_train.

    The callback writes a restart-request file; the parent run_hydra_train reads
    it after the subprocess exits and self-spawns the continuation.

    Required env vars (set by trainModal.py::run_hydra_train):
        MODAL_TIMEOUT_SECONDS     container timeout in seconds (e.g. 86400)
        MODAL_RESTART_MARGIN_SEC  save + restart this many sec before timeout
        MODAL_START_TIME          unix timestamp when the container started
        MODAL_HYDRA_ARGS          JSON-encoded list of the original hydra overrides
        MODAL_RESTART_REQUEST_FILE path the parent polls for the continuation args
    """

    _RESTART_MARGIN_SEC = 1800  # default: save + restart 30 min before timeout

    def __init__(self) -> None:
        super().__init__()
        self._triggered = False
        self._start = float(os.environ.get("MODAL_START_TIME", time.time()))
        self._timeout = int(os.environ.get("MODAL_TIMEOUT_SECONDS", 86400))
        self._margin = int(
            os.environ.get("MODAL_RESTART_MARGIN_SEC", self._RESTART_MARGIN_SEC)
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._triggered:
            return
        remaining = self._timeout - (time.time() - self._start)
        if remaining < self._margin:
            self._triggered = True
            self._auto_restart(trainer)

    def _auto_restart(self, trainer: Trainer) -> None:
        ckpt_path = os.path.join(
            trainer.default_root_dir, "checkpoints", "modal_auto_restart.ckpt"
        )
        trainer.save_checkpoint(ckpt_path)
        log.info(f"[ModalAutoRestart] Checkpoint saved → {ckpt_path}")

        # Parent run_hydra_train commits the outputs volume before spawning.
        wandb_run_id = None
        for lgr in trainer.loggers:
            if hasattr(lgr, "experiment") and hasattr(lgr.experiment, "id"):
                wandb_run_id = lgr.experiment.id
                break

        raw_args: list = json.loads(os.environ.get("MODAL_HYDRA_ARGS", "[]"))
        new_args = [
            a for a in raw_args
            if not a.startswith("ckpt_path=") and not a.startswith("wandb_run_id=")
        ]
        new_args.append(f"ckpt_path={ckpt_path}")
        if wandb_run_id:
            new_args.append(f"wandb_run_id={wandb_run_id}")

        # Hand continuation args to the parent (it self-spawns the next job).
        restart_file = os.environ.get("MODAL_RESTART_REQUEST_FILE")
        if trainer.is_global_zero and restart_file:
            try:
                with open(restart_file, "w") as f:
                    json.dump({"hydra_args": new_args}, f)
                log.info(f"[ModalAutoRestart] Wrote restart request → {restart_file}")
            except Exception as exc:
                log.error(f"[ModalAutoRestart] Failed to write restart request: {exc}")
        elif not restart_file:
            log.error(
                "[ModalAutoRestart] MODAL_RESTART_REQUEST_FILE unset — "
                "continuation will NOT be spawned"
            )

        # Clear min_epochs/min_steps (default config sets min_epochs=2000) so
        # should_stop is honored now instead of at the hard timeout.
        fit_loop = getattr(trainer, "fit_loop", None)
        epoch_loop = getattr(fit_loop, "epoch_loop", None)
        for obj, attr in ((fit_loop, "min_epochs"), (epoch_loop, "min_steps")):
            if obj is not None and hasattr(obj, attr):
                try:
                    setattr(obj, attr, 0)
                except AttributeError:
                    log.warning("[ModalAutoRestart] could not clear %s", attr)

        trainer.should_stop = True
        log.info("[ModalAutoRestart] Stopping current run — parent will spawn continuation")


class PrefetchEpochCallback(Callback):
    """Call ``prepare_epoch`` on every PrefetchedMapDataset before each epoch.

    This must run before the DataLoader begins iterating so that workers
    (spawned after the callback) inherit the fresh index_map via fork.

    Both train AND valid datasets are prepared. Preparing valid is required for
    DDP correctness: without it the valid dataset stays in its probe-path
    fallback (``_index_map is None``), and the rank-0-only pool staging means
    ranks enter the validation loop doing unequal work — they then desync at the
    first ``sync_dist`` all-reduce and hang until the 30-min NCCL watchdog kills
    the run. ``prepare_epoch`` is deterministic by seed, so every rank builds an
    identical valid index_map and blocks together until the shared-NVMe episodes
    are ready — an in-lockstep sync point before any val collective fires.
    """

    def __init__(self, train_datasets: dict, valid_datasets: dict | None = None) -> None:
        super().__init__()
        self._train_datasets = train_datasets
        self._valid_datasets = valid_datasets or {}

    def _prepare(self, datasets: dict, epoch: int, split: str) -> None:
        for name, ds in datasets.items():
            if hasattr(ds, "prepare_epoch"):
                log.info(
                    "PrefetchEpochCallback: prepare_epoch(%d) for %s/%s", epoch, split, name
                )
                ds.prepare_epoch(epoch)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._prepare(self._train_datasets, trainer.current_epoch, "train")

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        # All ranks call this before the val loop → identical valid index_map +
        # a synchronized barrier point, preventing the DDP collective desync.
        self._prepare(self._valid_datasets, trainer.current_epoch, "valid")
