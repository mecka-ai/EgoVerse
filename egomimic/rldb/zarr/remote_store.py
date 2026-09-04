"""Direct zarr reads from R2 -- no local tar-extract step.

Builds a locally-cached, async fsspec filesystem so ``zarr.open_group()`` can
read an episode's chunks straight from R2 on demand. Each chunk is cached to
local disk on first touch via fsspec's own ``filecache`` wrapper (a
transparent, size-bounded LRU disk cache) -- this replaces the
tar-download-then-extract-then-read-locally step entirely; there is no
separate "materialize the whole episode before reading" phase.

Credentials come from the same env vars the ``mecka-r2`` Modal secret
already provides: ``R2_ENDPOINT``, ``R2_ACCESS_KEY``, ``R2_SECRET_KEY`` (see
``~/mecka/robotics_pipeline/setup_secret.sh``). Requires ``s3fs`` in any
image that imports this module (zarr itself only depends on ``fsspec``).

Verified against the exact pinned version used by the training image
(``zarr==3.1.5``, ``fsspec==2026.4.0`` locally) -- ``zarr.storage.FsspecStore``
requires an *async* fsspec filesystem, but ``fsspec.filesystem("filecache",
...)`` returns a sync-wrapping one, so ``_ensure_async`` below applies the
same sync->async wrap zarr's own (private) ``FsspecStore.from_url`` helper
uses internally, without reaching into zarr's private API.

NEW, not yet exercised against a live Modal run -- smoke-test before relying
on it for a real training job (see ``verify_remote_zarr`` at the bottom).
"""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_FS_LOCK = threading.Lock()
# Keyed by (cache_dir, pid): an async fsspec filesystem typically runs its own
# event loop on a background thread, and threads do not survive fork(). A
# DataLoader with num_workers>0 forks worker processes *after* the parent may
# have already built one of these, so every entry is invalidated the moment
# os.getpid() no longer matches the pid that built it -- each process (parent
# and every worker) gets its own fs instance instead of inheriting a dead one.
_FS_CACHE: dict[tuple[str, int], object] = {}


def _r2_storage_options() -> dict:
    endpoint = os.environ.get("R2_ENDPOINT")
    key = os.environ.get("R2_ACCESS_KEY")
    secret = os.environ.get("R2_SECRET_KEY")
    if not (endpoint and key and secret):
        raise RuntimeError(
            "R2_ENDPOINT/R2_ACCESS_KEY/R2_SECRET_KEY must be set to read zarr "
            "episodes directly from R2 -- attach the mecka-r2 secret to this "
            "Modal function (see modal_setup.py::CFG.secret_names)."
        )
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    return {"key": key, "secret": secret, "client_kwargs": {"endpoint_url": endpoint}}


def _ensure_async(fs):
    """Wrap a sync fsspec filesystem for zarr's async FsspecStore.

    Mirrors ``zarr.storage._fsspec._make_async`` (verified against the
    installed zarr==3.1.5 source) without importing that private helper.
    """
    if getattr(fs, "async_impl", False) and getattr(fs, "asynchronous", False):
        return fs
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    return AsyncFileSystemWrapper(fs, asynchronous=True)


def _cached_async_fs(cache_dir: str):
    """One shared, async, locally-caching S3 filesystem per cache_dir per process."""
    import fsspec

    cache_key = (cache_dir, os.getpid())
    with _FS_LOCK:
        fs = _FS_CACHE.get(cache_key)
        if fs is not None:
            return fs

        os.makedirs(cache_dir, exist_ok=True)
        cached_sync_fs = fsspec.filesystem(
            "filecache",
            target_protocol="s3",
            target_options=_r2_storage_options(),
            cache_storage=cache_dir,
        )
        fs = _ensure_async(cached_sync_fs)
        _FS_CACHE[cache_key] = fs
        return fs


def parse_r2_url(url: str) -> tuple[str, str]:
    """``r2://bucket/key/prefix`` -> ``(bucket, "key/prefix")``."""
    if not url.startswith("r2://"):
        raise ValueError(f"expected an r2:// URL, got {url!r}")
    _, _, rest = url.partition("r2://")
    bucket, _, key_prefix = rest.partition("/")
    return bucket, key_prefix


def open_remote_zarr_group(url: str, cache_dir: str, mode: str = "r"):
    """Open a zarr group whose chunks live on R2, through a local disk cache.

    ``url`` is ``r2://<bucket>/<key-prefix>`` pointing at the *root* of one
    episode's zarr store (no trailing filename). Repeated reads of the same
    chunk after the first hit the local cache in ``cache_dir`` instead of R2.
    """
    import zarr
    from zarr.storage import FsspecStore

    bucket, key_prefix = parse_r2_url(url)
    fs = _cached_async_fs(cache_dir)
    store = FsspecStore(fs=fs, path=f"{bucket}/{key_prefix}", read_only=(mode == "r"))
    return zarr.open_group(store=store, mode=mode)


def verify_remote_zarr(url: str, cache_dir: str = "/tmp/remote_zarr_verify") -> dict:
    """Smoke test: open one episode and read its attrs. Run this manually
    before pointing a real training job at R2ZarrEpisodeResolver, e.g.:

        python -c "from egomimic.rldb.zarr.remote_store import verify_remote_zarr as v; \
                    print(v('r2://robotics/fold_clothes_zarr/<hash>.zarr'))"
    """
    group = open_remote_zarr_group(url, cache_dir)
    attrs = dict(group.attrs)
    return {
        "total_frames": attrs.get("total_frames"),
        "embodiment": attrs.get("embodiment"),
        "keys": list(attrs.get("features", {})),
    }
