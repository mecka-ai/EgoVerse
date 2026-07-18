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

        # Persist the checkpoint to the outputs volume BEFORE spawning the
        # continuation. run_hydra_train only commits after trainer.fit() unwinds,
        # which is after this callback returns — so without an explicit commit
        # here the continuation can start reading ckpt_path before it exists on
        # the volume. Commit once, on rank zero (the mount is shared).
        if trainer.is_global_zero:
            try:
                import modal as _modal

                _modal.Volume.from_name("egoverse-training-outputs").commit()
                log.info("[ModalAutoRestart] Committed checkpoint to outputs volume")
            except Exception as exc:
                log.error(f"[ModalAutoRestart] Volume commit failed: {exc}")

        wandb_run_id = None
        for lgr in trainer.loggers:
            if hasattr(lgr, "experiment") and hasattr(lgr.experiment, "id"):
                wandb_run_id = lgr.experiment.id
                break

        raw_args: list = json.loads(os.environ.get("MODAL_HYDRA_ARGS", "[]"))
        new_args = [
            a
            for a in raw_args
            if not a.startswith("ckpt_path=") and not a.startswith("wandb_run_id=")
        ]
        new_args.append(f"ckpt_path={ckpt_path}")
        if wandb_run_id:
            new_args.append(f"wandb_run_id={wandb_run_id}")

        git_remote = os.environ.get("MODAL_GIT_REMOTE", "")
        git_commit = os.environ.get("MODAL_GIT_COMMIT", "")
        wandb_api_key = os.environ.get("WANDB_API_KEY", "")
        from modal_setup import decode_submodules

        submodules = decode_submodules(os.environ.get("MODAL_INIT_SUBMODULES", ""))

        if trainer.is_global_zero:
            try:
                fn = self._continuation_fn()
                handle = fn.spawn(
                    tuple(new_args),
                    git_remote,
                    git_commit,
                    wandb_api_key,
                    submodules=submodules,
                )
                log.info(f"[ModalAutoRestart] Spawned continuation: {handle.object_id}")
            except Exception as exc:
                log.error(f"[ModalAutoRestart] Failed to spawn continuation: {exc}")

        trainer.should_stop = True
        log.info(
            "[ModalAutoRestart] Stopping current run — continuation job is running"
        )

    @staticmethod
    def _continuation_fn():
        """Resolve the run_hydra_train function to spawn the continuation on.

        Prefer the hydrated function object from the in-container trainModal
        module (the entrypoint Modal loaded for this very job): spawning it
        targets the SAME ephemeral app, so the continuation inherits this run's
        app name, GPU type, and image. Same-app sibling spawns are the
        established pattern (see curateModal.py fan-out).

        Falls back to a deployed-app lookup only if the module isn't loaded —
        note Function.from_name resolves DEPLOYED apps only, so the fallback
        works only if an `egomimic-training` app has been `modal deploy`ed.
        """
        import sys

        tm = sys.modules.get("trainModal")
        if tm is not None and hasattr(tm, "run_hydra_train"):
            return tm.run_hydra_train

        import modal as _modal

        log.warning(
            "[ModalAutoRestart] trainModal module not loaded — falling back to "
            "deployed-app lookup (requires a deployed 'egomimic-training' app)"
        )
        return _modal.Function.from_name(
            "egomimic-training",
            "run_hydra_train",
            environment_name="robotics",
        )


class VolumeCommitCallback(Callback):
    """Commit the training-outputs volume after every checkpoint save.

    Modal volume writes are not durable across a container kill until committed.
    Preemptible GPUs can reclaim the container at any time, so without an explicit
    commit the most recent ModelCheckpoint save is lost and the restarted
    container falls back to the launch checkpoint. Committing on each save makes
    the run's own latest checkpoint durable, so trainHydra's resume-from-own-ckpt
    logic can pick up where the preempted container left off (bounded to the
    checkpoint interval). Rank-zero only; the mount is shared.
    """

    def on_save_checkpoint(self, trainer: Trainer, pl_module, checkpoint) -> None:
        if not trainer.is_global_zero:
            return
        try:
            import modal as _modal

            _modal.Volume.from_name("egoverse-training-outputs").commit()
            log.info("[VolumeCommit] Committed checkpoint to outputs volume")
        except Exception as exc:  # noqa: BLE001 — commit is best-effort
            log.error(f"[VolumeCommit] Volume commit failed: {exc}")


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

    def __init__(
        self,
        train_datasets: dict,
        valid_datasets: dict | None = None,
        train_viz_datasets: dict | None = None,
    ) -> None:
        super().__init__()
        self._train_datasets = train_datasets
        self._valid_datasets = valid_datasets or {}
        # train_viz is the dataloader_idx=1 leg of the val loop. As a prefetch
        # dataset it must be staged on the same schedule as valid, else it stays
        # in the probe-path fallback and desyncs ranks at the val all-reduce.
        self._train_viz_datasets = train_viz_datasets or {}

    def _prepare(self, datasets: dict, epoch: int, split: str) -> None:
        for name, ds in datasets.items():
            if hasattr(ds, "prepare_epoch"):
                log.info(
                    "PrefetchEpochCallback: prepare_epoch(%d) for %s/%s",
                    epoch,
                    split,
                    name,
                )
                ds.prepare_epoch(epoch)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._prepare(self._train_datasets, trainer.current_epoch, "train")

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        # All ranks call this before the val loop → identical valid index_map +
        # a synchronized barrier point, preventing the DDP collective desync.
        self._prepare(self._valid_datasets, trainer.current_epoch, "valid")
        self._prepare(self._train_viz_datasets, trainer.current_epoch, "train_viz")
