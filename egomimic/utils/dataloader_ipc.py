"""Configure PyTorch DataLoader multiprocessing IPC for large image batches.

Lightning DDP launches training in a child process. ``set_sharing_strategy`` only
affects the process that calls it — setting it in Hydra ``main()`` does NOT apply
to the DDP rank process, which still defaults to ``file_descriptor`` (/dev/shm).
Call :func:`configure_dataloader_ipc` in the process that constructs the
DataLoader (e.g. ``LightningDataModule.train_dataloader``).
"""

from __future__ import annotations

import logging
import os

import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

_configured = False


def configure_dataloader_ipc(*, force: bool = False) -> str | None:
    """Use file-backed tensor sharing when TMPDIR is set (Modal NVMe).

    Returns the active sharing strategy name, or None if unchanged.
    """
    global _configured
    if _configured and not force:
        return mp.get_sharing_strategy()

    if os.environ.get("MODAL_IS_REMOTE") != "1":
        return mp.get_sharing_strategy()

    tmpdir = os.environ.get("TMPDIR", "")
    if tmpdir:
        mp.set_sharing_strategy("file_system")
        _configured = True
        return "file_system"

    logger.warning(
        "MODAL_IS_REMOTE but TMPDIR is unset — DataLoader IPC uses /dev/shm "
        "(SIGBUS likely with num_workers>8 and 480×640 images). Pass "
        "+modal_ephemeral_disk_gb=600 so trainModal sets TMPDIR=/cache/torch_tmp."
    )
    _configured = True
    return mp.get_sharing_strategy()


def chain_worker_init_fn(user_init=None):
    """Return a worker_init_fn that configures IPC then calls *user_init*."""
    configure_dataloader_ipc()

    if user_init is None:

        def _init(worker_id: int) -> None:
            # Force file_system in EVERY worker — fork inheritance of the sharing
            # strategy is not reliable, so workers that default to file_descriptor
            # route batch tensors through the small /dev/shm and SIGBUS once it
            # fills (the bus errors we saw at high worker/prefetch counts).
            configure_dataloader_ipc(force=True)

        return _init

    def _init(worker_id: int) -> None:
        configure_dataloader_ipc(force=True)
        user_init(worker_id)

    return _init


def apply_ipc_dataloader_params(params: dict) -> dict:
    """Configure IPC and inject worker_init_fn into a DataLoader kwargs dict."""
    configure_dataloader_ipc()
    params = dict(params)
    params["worker_init_fn"] = chain_worker_init_fn(params.get("worker_init_fn"))
    return params
