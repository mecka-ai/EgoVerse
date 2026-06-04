"""Backward-compat shim — the implementation now lives in the ``prefetch`` package.

This module was split into ``egomimic/rldb/zarr/prefetch/`` (catalog, plan, pool,
extract, filler, dataset). Hydra configs still reference
``egomimic.rldb.zarr.prefetch_dataset.PrefetchedMapDataset`` and ``.ZipEpisodeResolver``,
so this re-exports the public API to keep those ``_target_`` paths resolving.
Prefer importing from ``egomimic.rldb.zarr.prefetch`` in new code.
"""

from __future__ import annotations

from egomimic.rldb.zarr.prefetch import (  # noqa: F401
    EpisodeCatalogEntry,
    EpisodePlan,
    EpisodePool,
    PoolFillerThread,
    PrefetchedIterableDataset,
    PrefetchedMapDataset,
    ZipEpisodeResolver,
    _BoundsCheckMixin,
)

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
