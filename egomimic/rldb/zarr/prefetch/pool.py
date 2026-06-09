"""Flat NVMe episode cache with a hard byte ceiling and window-safe eviction."""

from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class EpisodePool:
    """Hash-keyed NVMe cache with a hard byte ceiling and future-aware eviction.

    Layout: ``<cache_dir>/pool/<episode_hash>/`` with ``.done`` sentinel.
    Tracks bytes-on-disk for every cached episode; ``used_bytes`` returns
    the running sum without a directory walk.

    ``evict_outside(keep_hashes)`` deletes every cached episode whose hash
    is not in ``keep_hashes`` and returns the number of bytes freed.
    Callers pass the set of episodes scheduled in the current + lookahead
    epochs, so the DataLoader never sees an episode disappear from under it.
    """

    def __init__(self, cache_dir: Path | str, capacity_gb: float):
        self.root = Path(cache_dir) / "pool"
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity_bytes = int(capacity_gb * 1e9)
        self._sizes: dict[str, int] = {}
        self._lock = threading.Lock()
        self._scan_existing()

    def _scan_existing(self) -> None:
        """Rebuild the byte map from disk (crash recovery)."""
        n = 0
        n_bad = 0
        for ep_dir in self.root.iterdir():
            if not ep_dir.is_dir():
                continue
            if (ep_dir / ".bad").exists():
                # Permanent known-bad marker; keep but don't count for capacity.
                n_bad += 1
                continue
            if not (ep_dir / ".done").exists():
                # Half-finished extraction from a previous run; drop it.
                shutil.rmtree(ep_dir, ignore_errors=True)
                continue
            try:
                size = sum(f.stat().st_size for f in ep_dir.rglob("*") if f.is_file())
            except FileNotFoundError:
                shutil.rmtree(ep_dir, ignore_errors=True)
                continue
            self._sizes[ep_dir.name] = size
            n += 1
        if n_bad:
            logger.info("EpisodePool: %d episodes carry .bad marker (will be skipped)", n_bad)
        if n:
            logger.info(
                "EpisodePool: restored %d episodes (%.1f GB) from existing cache at %s",
                n, sum(self._sizes.values()) / 1e9, self.root,
            )

    def episode_path(self, ep_hash: str) -> Path:
        return self.root / ep_hash

    def is_ready(self, ep_hash: str) -> bool:
        return (self.root / ep_hash / ".done").exists()

    def is_bad(self, ep_hash: str) -> bool:
        """True if this episode has been marked permanently unextractable."""
        return (self.root / ep_hash / ".bad").exists()

    def used_bytes(self) -> int:
        with self._lock:
            return sum(self._sizes.values())

    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes()

    def register(self, ep_hash: str, size_bytes: int) -> None:
        with self._lock:
            self._sizes[ep_hash] = size_bytes

    def drop(self, ep_hash: str) -> int:
        """Remove ``ep_hash`` from the pool. Returns bytes freed."""
        with self._lock:
            size = self._sizes.pop(ep_hash, 0)
        shutil.rmtree(self.root / ep_hash, ignore_errors=True)
        return size

    def evict_outside(self, keep_hashes: set[str]) -> int:
        """Evict every cached hash not in ``keep_hashes``. Returns bytes freed."""
        with self._lock:
            victims = [h for h in self._sizes if h not in keep_hashes]
        freed = 0
        for h in victims:
            freed += self.drop(h)
        if victims:
            logger.info(
                "EpisodePool: evicted %d episodes (%.1f GB freed, %.1f GB used / %.0f GB cap)",
                len(victims), freed / 1e9,
                self.used_bytes() / 1e9, self.capacity_bytes / 1e9,
            )
        return freed


