"""Tar extraction helpers, the running-filler registry, and the ENOSPC sentinel.

Pure orchestration: no torch/numpy. Shared by the filler (background staging)
and the dataset (synchronous valid-mode extraction).
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # forward-ref only — importing filler here would cycle
    from egomimic.rldb.zarr.prefetch.filler import PoolFillerThread

logger = logging.getLogger(__name__)

def _extract_tar_to_dir(source: Path, dest: Path) -> int:
    """Materialise ``source`` into ``dest`` and return total bytes written.

    ``source`` is either a tar archive (zip volume) or an already-unpacked
    ``.zarr`` directory (zarr volume). The directory case is a straight copy,
    which skips tar decompression entirely -- staging off a volume is bandwidth
    bound at ~150 MB/s per container, so avoiding the extra CPU pass keeps the
    filler from competing with the dataloader workers for cores.

    Caller is responsible for creating/cleaning ``dest``, touching ``.done``,
    and registering the size with the pool.  Raises ``OSError`` (including
    ENOSPC, errno 28) on failure.
    """
    if source.is_dir():
        _copy_tree_parallel(source, dest)
    else:
        with tarfile.open(source, "r") as tf:
            tf.extractall(path=dest)
    return sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())


# A zarr episode is thousands of tiny chunk files, and on a FUSE volume the
# per-file round trip -- not bandwidth -- dominates. shutil.copytree is
# sequential and adds a stat+copystat per file, which measured 6 MB/s against
# ~150 MB/s of available bandwidth. Copying the files concurrently and skipping
# metadata preservation recovers most of that gap.
_COPY_THREADS = int(os.environ.get("ZARR_STAGE_COPY_THREADS", "64"))


def _copy_one(args: tuple[Path, Path]) -> None:
    src, dst = args
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        shutil.copyfileobj(fi, fo, length=1024 * 1024)


def _copy_tree_parallel(source: Path, dest: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    jobs: list[tuple[Path, Path]] = []
    for src in source.rglob("*"):
        rel = src.relative_to(source)
        dst = dest / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((src, dst))
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=_COPY_THREADS) as ex:
        list(ex.map(_copy_one, jobs))


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


