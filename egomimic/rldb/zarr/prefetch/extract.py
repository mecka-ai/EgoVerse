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


# A zarr episode here is ~18 files: one ~37 MB image shard plus a handful of
# 0.15-1.5 MB pose/keypoint arrays and their metadata. Measured on the volume:
# the image shard alone streams at 100-280 MB/s, the mid-sized arrays manage
# only 21-29 MB/s and do NOT improve with more threads (flat from 32 to 1024),
# and whole episodes land at 40-55 MB/s. shutil.copytree is sequential and adds
# a stat+copystat per file; copying concurrently and skipping metadata
# preservation recovers most of the gap that is recoverable here.
_COPY_THREADS = int(os.environ.get("ZARR_STAGE_COPY_THREADS", "64"))


def _copy_one(args: tuple[Path, Path]) -> None:
    src, dst = args
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        # 8 MiB: the image shard is ~37 MB, so larger reads mean far fewer
        # round trips on the file that carries ~90% of the episode's bytes.
        shutil.copyfileobj(fi, fo, length=8 * 1024 * 1024)


def _copy_tree_parallel(source: Path, dest: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=_COPY_THREADS) as ex:
        # Discover the tree with the directory reads fanned out. Path.rglob is
        # a single sequential stream of FUSE round trips, which measured ~1.0s
        # per episode against ~0.01s for the same walk issued concurrently --
        # most of a stage was spent listing the directory, not copying it.
        files: list[Path] = []
        level = [source]
        while level:
            results = list(ex.map(lambda d: list(os.scandir(d)), level))
            level = []
            for entries in results:
                for e in entries:
                    (level if e.is_dir() else files).append(Path(e.path))

        jobs: list[tuple[Path, Path]] = []
        for src in files:
            dst = dest / src.relative_to(source)
            dst.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((src, dst))
        if not jobs:
            return
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


