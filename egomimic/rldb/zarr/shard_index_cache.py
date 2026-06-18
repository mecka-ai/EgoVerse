"""Process-local cache for zarr v3 sharding indexes.

Zarr's sharding codec loads and decodes the shard index on each partial chunk
read. EgoVerse training repeatedly reads many frames from the same image shard,
so this saves one random range read per sample on seek-bound filesystems.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

_INSTALLED = False
_LOCK = Lock()
_LOGGED = False


def _cache_size() -> int:
    raw = os.environ.get("EGOMIMIC_ZARR_SHARD_INDEX_CACHE", "512")
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid EGOMIMIC_ZARR_SHARD_INDEX_CACHE=%r; using 512 entries", raw
        )
        return 512


def _store_key(byte_getter: Any) -> tuple[str, str] | None:
    store = getattr(byte_getter, "store", None)
    path = getattr(byte_getter, "path", None)
    if store is None or path is None:
        return None
    return (str(store), str(path))


def install_zarr_shard_index_cache() -> None:
    """Install an LRU cache around zarr.codecs.sharding.ShardingCodec.

    Safe to call repeatedly. If zarr internals change or zarr is unavailable,
    this becomes a no-op.
    """
    global _INSTALLED, _LOGGED
    max_entries = _cache_size()
    if max_entries <= 0:
        return

    with _LOCK:
        if _INSTALLED:
            return
        try:
            from zarr.codecs.sharding import ShardingCodec
        except Exception:
            return

        original = getattr(ShardingCodec, "_load_shard_index_maybe", None)
        if original is None or getattr(original, "_egomimic_cached", False):
            _INSTALLED = True
            return

        cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        cache_lock = Lock()

        async def cached_load_shard_index_maybe(self, byte_getter, chunks_per_shard):
            nonlocal cache
            global _LOGGED

            base_key = _store_key(byte_getter)
            if base_key is None:
                return await original(self, byte_getter, chunks_per_shard)

            try:
                shard_index_size = self._shard_index_size(chunks_per_shard)
            except Exception:
                shard_index_size = None
            key = (
                base_key[0],
                base_key[1],
                tuple(chunks_per_shard),
                getattr(self.index_location, "value", self.index_location),
                shard_index_size,
            )

            with cache_lock:
                hit = cache.get(key)
                if hit is not None:
                    cache.move_to_end(key)
                    return hit

            index = await original(self, byte_getter, chunks_per_shard)
            if index is None:
                return None

            with cache_lock:
                cache[key] = index
                cache.move_to_end(key)
                while len(cache) > max_entries:
                    cache.popitem(last=False)

            if not _LOGGED:
                logger.info(
                    "Installed zarr shard-index cache (%d entries per process)",
                    max_entries,
                )
                _LOGGED = True
            return index

        cached_load_shard_index_maybe._egomimic_cached = True
        setattr(ShardingCodec, "_load_shard_index_maybe", cached_load_shard_index_maybe)
        _INSTALLED = True
