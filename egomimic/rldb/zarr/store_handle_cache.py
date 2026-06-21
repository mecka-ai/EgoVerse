"""Make zarr v3 ``LocalStore`` cheap to read over NFS.

Stock ``LocalStore._get`` does ``with path.open("rb")`` on *every* chunk read.
Over NFS each ``open()`` is a close-to-open revalidation GETATTR (~3.3 ms vs
~10 µs on local overlay), and the sharding codec issues two reads per frame: a
``SuffixByteRequest`` for the shard-index footer (identical for every frame, the
shard is immutable in ``mode="r"``) and a ``RangeByteRequest`` for the chunk.

This patch removes both taxes without bulk-staging to RAM:
  1. fd reuse — open each shard once per process, serve reads via ``os.pread``.
  2. readahead advice — see ``EGOMIMIC_ZARR_FDCACHE_FADV`` (default ``random``:
     disables kernel readahead so a cold 64 KB chunk read fetches ~64 KB instead
     of up to 1 MB; ``normal``/``sequential`` keep/boost readahead, which warms
     whole shards into the page cache faster across epochs).
  3. shard-index suffix cache — cache the footer bytes per (path, suffix-len) for
     read-only stores only; zarr still decodes/CRC-checks them ⇒ byte-identical.

**Bounded at scale.** The fd cache + index cache are capped at
``EGOMIMIC_ZARR_FDCACHE_MAX`` shards per process (default 8192). On overflow the
oldest-inserted shard's fd is closed and its index dropped, so memory and fd
count stay bounded by the cap, not the dataset size (~33 KB index + 1 KB fd per
cached shard). A reaccess to an evicted shard simply re-opens it (one open),
still cheaper than stock (which reopens per read). ``close_store`` clears
everything for a store at episode teardown.

Toggle the whole patch off with ``EGOMIMIC_ZARR_FDCACHE=0``.

Usage:
    from egomimic.rldb.zarr import store_handle_cache as hc
    hc.ensure_enabled()        # idempotent; respects an explicit disable() + env flag
    ...
    hc.close_store(group)      # on episode teardown: close fds, drop caches
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections import OrderedDict

from zarr.abc.store import (
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import LocalStore

_orig_get = LocalStore.get
_orig_partial = LocalStore.get_partial_values

# Set by an explicit disable(); makes ensure_enabled() a no-op until the next
# explicit enable() (lets benchmarks run a true "stock" arm).
_force_disabled = False

# PER-PROCESS GLOBAL fd cache (NOT per-store). The fd cache used to live on each
# LocalStore instance, so total open fds = (open episode stores) x cap, which is
# UNBOUNDED as the dataset's working set slides -> ~33k fds/proc, exhausting the
# system file table (fs.file-max) and crashing with "Too many open files". Making
# the cache global bounds TOTAL open fds per process to EGOMIMIC_ZARR_FDCACHE_MAX
# (LRU across ALL stores), independent of how many episode stores are open.
_GLOBAL_FDCACHE = None  # OrderedDict[path_str -> fd]
_GLOBAL_FDLOCK = None
_GLOBAL_FDPID = None


def _fd_max() -> int:
    """Max cached shard fds per process (and index-cache entries). Read each
    eviction check so it can be tuned via env without restart-in-test."""
    try:
        return max(1, int(os.environ.get("EGOMIMIC_ZARR_FDCACHE_MAX", "8192")))
    except ValueError:
        return 8192


def _fadvise(fd: int) -> None:
    """Apply readahead advice to a freshly opened shard fd.

    EGOMIMIC_ZARR_FDCACHE_FADV: 'random' (default) disables kernel readahead
    (best for shuffled per-frame access); 'normal' leaves the kernel default
    (whole-shard readahead → better cross-epoch page-cache warming);
    'sequential' requests aggressive readahead.
    """
    mode = os.environ.get("EGOMIMIC_ZARR_FDCACHE_FADV", "random").strip().lower()
    advice = {
        "random": getattr(os, "POSIX_FADV_RANDOM", None),
        "sequential": getattr(os, "POSIX_FADV_SEQUENTIAL", None),
        "normal": getattr(os, "POSIX_FADV_NORMAL", None),
        "off": getattr(os, "POSIX_FADV_NORMAL", None),
        "none": getattr(os, "POSIX_FADV_NORMAL", None),
        "": getattr(os, "POSIX_FADV_RANDOM", None),
    }.get(mode, getattr(os, "POSIX_FADV_RANDOM", None))
    if advice is None:
        return
    try:
        os.posix_fadvise(fd, 0, 0, advice)
    except (AttributeError, OSError):
        pass


def _fd_for(self, path_str: str) -> int:
    """Return a cached read fd for ``path_str``, opened once per process and
    capped at ``EGOMIMIC_ZARR_FDCACHE_MAX``.

    fork-safe: the fd cache *and* the suffix-index cache are reset together when
    the pid changes, so each DataLoader worker opens its own fds.
    """
    global _GLOBAL_FDCACHE, _GLOBAL_FDLOCK, _GLOBAL_FDPID
    pid = os.getpid()
    # (Re)init the per-process global fd cache on first use or after fork.
    if _GLOBAL_FDCACHE is None or _GLOBAL_FDPID != pid:
        _GLOBAL_FDCACHE = OrderedDict()
        _GLOBAL_FDLOCK = threading.Lock()
        _GLOBAL_FDPID = pid
    # Per-store suffix (shard-index footer) byte cache is still per-store; reset
    # on fork. It stores small immutable footers, not fds, so it isn't the leak.
    if (
        getattr(self, "_egomimic_fdc_suffix", None) is None
        or getattr(self, "_fdpid", None) != pid
    ):
        self._egomimic_fdc_suffix = {}
        self._fdpid = pid
    cache = _GLOBAL_FDCACHE
    fd = cache.get(path_str)
    if fd is not None:
        try:
            cache.move_to_end(path_str)  # LRU bump
        except KeyError:
            pass
        return fd
    with _GLOBAL_FDLOCK:
        fd = cache.get(path_str)
        if fd is not None:
            cache.move_to_end(path_str)
            return fd
        fd = os.open(path_str, os.O_RDONLY)
        _fadvise(fd)
        cache[path_str] = fd
        # Bound TOTAL fds per process (LRU across all stores), so fds stay capped
        # at EGOMIMIC_ZARR_FDCACHE_MAX no matter how many episode stores are open.
        maxn = _fd_max()
        while len(cache) > maxn:
            _old_path, old_fd = cache.popitem(last=False)
            try:
                os.close(old_fd)
            except OSError:
                pass
    return fd


def _drop_fd(self, path_str: str) -> None:
    """Remove + close a possibly-stale cached fd so the next read re-opens.
    Used to recover from the rare race where a concurrent insert evicted+closed
    an fd between a lookup and its read."""
    cache = _GLOBAL_FDCACHE
    if not cache:
        return
    if _GLOBAL_FDLOCK is not None:
        with _GLOBAL_FDLOCK:
            fd = cache.pop(path_str, None)
    else:
        fd = cache.pop(path_str, None)
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


def _pread_all(fd: int, n: int, offset: int) -> bytes:
    """Positioned read of ``n`` bytes; loops over NFS short reads."""
    out = bytearray()
    while len(out) < n:
        b = os.pread(fd, n - len(out), offset + len(out))
        if not b:
            break
        out += b
    return bytes(out)


def _read_range(fd: int, byte_range) -> bytes:
    if byte_range is None:
        return _pread_all(fd, os.fstat(fd).st_size, 0)
    if isinstance(byte_range, RangeByteRequest):
        return _pread_all(fd, byte_range.end - byte_range.start, byte_range.start)
    if isinstance(byte_range, OffsetByteRequest):
        size = os.fstat(fd).st_size
        return _pread_all(fd, size - byte_range.offset, byte_range.offset)
    if isinstance(byte_range, SuffixByteRequest):
        size = os.fstat(fd).st_size
        return _pread_all(fd, byte_range.suffix, max(0, size - byte_range.suffix))
    raise TypeError(f"Unexpected byte_range: {byte_range!r}")


def _read_cached(self, fd: int, path_str: str, byte_range) -> bytes:
    """Read bytes for ``byte_range``; serve the immutable shard-index footer
    (``SuffixByteRequest``) from a per-store byte cache on read-only stores.
    Only suffix reads are cached — never chunk (Range/Offset) data."""
    if isinstance(byte_range, SuffixByteRequest) and getattr(self, "read_only", False):
        scache = getattr(self, "_egomimic_fdc_suffix", None)
        if scache is None:
            self._egomimic_fdc_suffix = scache = {}
        ckey = (path_str, byte_range.suffix)
        cached = scache.get(ckey)
        if cached is None:
            cached = _read_range(fd, byte_range)
            scache[ckey] = cached
        return cached
    return _read_range(fd, byte_range)


async def _cached_get(self, key, prototype=None, byte_range=None):
    if prototype is None:
        prototype = default_buffer_prototype()
    path_str = str(self.root / key)

    def work():
        for attempt in (0, 1):
            try:
                fd = _fd_for(self, path_str)
                return prototype.buffer.from_bytes(
                    _read_cached(self, fd, path_str, byte_range)
                )
            except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
                return None
            except OSError:
                # Rare: a concurrent insert evicted+closed this fd between the
                # lookup and the read. Drop the stale fd and re-open once.
                if attempt == 1:
                    raise
                _drop_fd(self, path_str)
        return None

    return await asyncio.to_thread(work)


async def _cached_get_partial_values(self, prototype, key_ranges):
    async def one(key, br):
        return await _cached_get(self, key, prototype, br)

    return await asyncio.gather(*[one(k, br) for (k, br) in key_ranges])


def close_store(obj) -> None:
    """Close cached fds and clear the fd + suffix caches for a store.

    Accepts a ``LocalStore`` or any object exposing ``.store`` (e.g. a zarr
    ``Group``). Call on reader teardown so fds don't accumulate as the working
    set slides. No-op on objects without a cache.
    """
    store = getattr(obj, "store", None) or obj
    # fds now live in the per-process global cache keyed by full path; drop the
    # ones under this store's root (the global LRU would evict them anyway, but
    # closing promptly on teardown is good hygiene).
    cache = _GLOBAL_FDCACHE
    if cache:
        try:
            root = str(getattr(store, "root", "") or "")
        except Exception:
            root = ""
        if root:
            lock = _GLOBAL_FDLOCK
            if lock is not None:
                lock.acquire()
            try:
                for p in [k for k in list(cache.keys()) if k.startswith(root)]:
                    fd = cache.pop(p, None)
                    if fd is not None:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
            finally:
                if lock is not None:
                    lock.release()
    scache = getattr(store, "_egomimic_fdc_suffix", None)
    if scache:
        scache.clear()


def _env_enabled() -> bool:
    """Kill-switch: ``EGOMIMIC_ZARR_FDCACHE=0`` disables the whole patch."""
    return os.environ.get("EGOMIMIC_ZARR_FDCACHE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def enable() -> None:
    """Install the patch (unless the env kill-switch is off). Clears any prior
    explicit ``disable()`` so ``ensure_enabled()`` will re-apply it."""
    global _force_disabled
    _force_disabled = False
    if not _env_enabled():
        return
    LocalStore.get = _cached_get
    LocalStore.get_partial_values = _cached_get_partial_values


def disable() -> None:
    """Restore stock zarr and pin it off: ``ensure_enabled()`` is a no-op until
    the next explicit ``enable()``."""
    global _force_disabled
    _force_disabled = True
    LocalStore.get = _orig_get
    LocalStore.get_partial_values = _orig_partial


def ensure_enabled() -> None:
    """Idempotent enable for the hot read path. Respects an explicit ``disable()``
    and the ``EGOMIMIC_ZARR_FDCACHE`` kill-switch. Safe to call on every episode
    open (runs in every fork/spawn worker, and for ``num_workers=0``)."""
    if _force_disabled:
        return
    enable()
