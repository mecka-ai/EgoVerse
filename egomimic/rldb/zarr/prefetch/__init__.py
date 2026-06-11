"""Sliding-window NVMe pool for per-episode zipped zarr v3 storage.

Architecture
------------
ZipEpisodeResolver
    Discovers ``{hash}.tar`` + ``{hash}.done`` pairs on the zip volume
    (``/mnt/zarr-zip``) and reads frame counts from each tar's ``zarr.json``.
    Inherits key_map / transform_list / norm_stats / pause_removal_epsilon
    plumbing from EpisodeResolver.

EpisodePlan
    Deterministic permutation of the full train (or valid) catalog, fixed
    by seed for the whole run. Each epoch consumes a contiguous slice of
    ``episodes_per_epoch`` episodes from the plan. Same seed on all DDP
    ranks → identical episode order, no coordination required.

EpisodePool
    Flat NVMe cache keyed by episode_hash at ``<cache_dir>/pool/<hash>/``.
    Tracks used bytes, enforces a hard capacity ceiling, and evicts only
    episodes outside the current+lookahead window (so episodes the
    DataLoader is about to consume are never deleted).

PoolFillerThread
    One persistent background thread on rank 0 only. Submits extractions
    to a fixed-size ThreadPoolExecutor, staying ``lookahead_epochs``
    ahead of the training cursor. Before each submission it checks pool
    capacity; if over budget it triggers eviction first and waits if no
    eviction is possible (training is still behind).

PrefetchedMapDataset
    Map-style Dataset that reads from the pool. ``prepare_epoch(epoch)``
    builds a frame-level ``_index_map`` for the current window, blocking
    until every episode is materialized. With ``persistent_workers=False``
    workers fork after this returns and inherit the index_map + warm zarr
    handles.

Disk safety
    The pool capacity is a hard byte ceiling. The filler refuses to start
    an extraction that would push usage past the ceiling and instead waits
    for the training cursor to advance and free episodes. ENOSPC is treated
    as a transient error: the partial directory is cleaned up and the
    episode is retried after the next eviction.

DDP correctness
    Only rank 0 runs the filler (RANK env var). Non-leader ranks just poll
    the ``.done`` sentinel on the shared NVMe mount, so all ranks observe
    the same set of ready episodes without IPC.
"""

from egomimic.rldb.zarr.prefetch.bounds import _BoundsCheckMixin
from egomimic.rldb.zarr.prefetch.catalog import EpisodeCatalogEntry, ZipEpisodeResolver
from egomimic.rldb.zarr.prefetch.dataset import (
    PrefetchedIterableDataset,
    PrefetchedMapDataset,
)
from egomimic.rldb.zarr.prefetch.filler import PoolFillerThread
from egomimic.rldb.zarr.prefetch.plan import EpisodePlan
from egomimic.rldb.zarr.prefetch.pool import EpisodePool

__all__ = [
    "EpisodeCatalogEntry",
    "ZipEpisodeResolver",
    "EpisodePlan",
    "EpisodePool",
    "PoolFillerThread",
    "PrefetchedMapDataset",
    "PrefetchedIterableDataset",
    "_BoundsCheckMixin",
]
