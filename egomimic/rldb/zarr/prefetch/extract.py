"""Tar extraction helpers, the running-filler registry, and the ENOSPC sentinel.

Pure orchestration: no torch/numpy. Shared by the filler (background staging)
and the dataset (synchronous valid-mode extraction).
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # forward-ref only — importing filler here would cycle
    from egomimic.rldb.zarr.prefetch.filler import PoolFillerThread

logger = logging.getLogger(__name__)

def _extract_tar_to_dir(
    tar_path: Path,
    dest: Path,
    *,
    expected_size_bytes: int | None = None,
) -> int:
    """Extract ``tar_path`` into ``dest`` and return total bytes written.

    Caller is responsible for creating/cleaning ``dest``, touching ``.done``,
    and registering the size with the pool.  Raises ``OSError`` (including
    ENOSPC, errno 28) on failure.
    """
    tar_bin = shutil.which("tar")
    if tar_bin:
        try:
            subprocess.run(
                [tar_bin, "-xf", str(tar_path), "-C", str(dest)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"tar extraction failed for {tar_path}: {msg}") from e
    else:
        with tarfile.open(tar_path, "r") as tf:
            tf.extractall(path=dest)

    if expected_size_bytes is not None:
        return int(expected_size_bytes)
    return sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())


def _acquire_extract_lock(pool_dir: Path, ep_hash: str) -> int | None:
    """Cross-process per-episode lock used during extraction.

    Returns the file descriptor on success (caller must release it), or
    ``None`` if another process is already extracting this episode (the
    caller should wait for the ``.done`` sentinel).
    """
    lock_path = pool_dir / f".lock_{ep_hash}"
    try:
        return os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None


def _release_extract_lock(pool_dir: Path, ep_hash: str, fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        (pool_dir / f".lock_{ep_hash}").unlink(missing_ok=True)
    except OSError:
        pass


class _ENOSPCError(Exception):
    """Raised inside PoolFillerThread to swallow ENOSPC without spamming logs."""
    pass


# Module-level registry of running PoolFillerThreads keyed by absolute cache_dir.
# trainHydra.py instantiates the train dataset twice (once for the actual
# DataLoader, once briefly for norm-stats inference). Without this registry,
# both instances would start their own filler against the same pool directory
# and race on rmtree+extract for the same episodes.
_FILLER_REGISTRY: dict[str, "PoolFillerThread"] = {}
_FILLER_REGISTRY_LOCK = threading.Lock()


def shutdown_registered_fillers() -> None:
    """Stop all registered background pool fillers.

    The filler thread is daemonized, but its ThreadPoolExecutor workers are
    normal threads. If the global registry keeps a filler reachable after
    training, relying on ``__del__`` is not enough to let the interpreter exit.
    """
    with _FILLER_REGISTRY_LOCK:
        fillers = list(_FILLER_REGISTRY.items())
        _FILLER_REGISTRY.clear()

    for cache_dir, filler in fillers:
        try:
            filler.stop()
            logger.info("Stopped PoolFillerThread for %s", cache_dir)
        except Exception:
            logger.exception("Failed to stop PoolFillerThread for %s", cache_dir)


atexit.register(shutdown_registered_fillers)
