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

    Required env vars (set by trainModal.py::run_hydra_train):
        MODAL_TIMEOUT_SECONDS   container timeout in seconds (e.g. 86400)
        MODAL_START_TIME        unix timestamp when the container started
        MODAL_HYDRA_ARGS        JSON-encoded list of the original hydra overrides
        MODAL_GIT_REMOTE        remote URL used to clone the repo
        MODAL_GIT_COMMIT        git SHA checked out in the container
    """

    _RESTART_MARGIN_SEC = 1800  # save + spawn 30 min before timeout

    def __init__(self) -> None:
        super().__init__()
        self._triggered = False
        self._start = float(os.environ.get("MODAL_START_TIME", time.time()))
        self._timeout = int(os.environ.get("MODAL_TIMEOUT_SECONDS", 86400))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._triggered:
            return
        remaining = self._timeout - (time.time() - self._start)
        if remaining < self._RESTART_MARGIN_SEC:
            self._triggered = True
            self._auto_restart(trainer)

    def _auto_restart(self, trainer: Trainer) -> None:
        ckpt_path = os.path.join(
            trainer.default_root_dir, "checkpoints", "modal_auto_restart.ckpt"
        )
        trainer.save_checkpoint(ckpt_path)
        log.info(f"[ModalAutoRestart] Checkpoint saved → {ckpt_path}")

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

        git_remote = os.environ.get("MODAL_GIT_REMOTE", "")
        git_commit = os.environ.get("MODAL_GIT_COMMIT", "")
        wandb_api_key = os.environ.get("WANDB_API_KEY", "")

        if trainer.is_global_zero:
            try:
                import modal as _modal

                fn = _modal.Function.from_name(
                    "egomimic-training",
                    "run_hydra_train",
                    environment_name="robotics",
                )
                handle = fn.spawn(
                    tuple(new_args), git_remote, git_commit, wandb_api_key
                )
                log.info(f"[ModalAutoRestart] Spawned continuation: {handle.object_id}")
            except Exception as exc:
                log.error(f"[ModalAutoRestart] Failed to spawn continuation: {exc}")

        trainer.should_stop = True
        log.info("[ModalAutoRestart] Stopping current run — continuation job is running")
