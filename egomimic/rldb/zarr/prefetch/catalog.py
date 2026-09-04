"""Episode catalog: the per-episode entry and the zip-volume resolver.

``ZipEpisodeResolver`` reads ``catalog.json`` from the zip volume and inherits
key_map / transform_list / norm_stats plumbing from ``EpisodeResolver``.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

from egomimic.rldb.zarr.zarr_dataset_multi import EpisodeResolver

logger = logging.getLogger(__name__)

@dataclass
class EpisodeCatalogEntry:
    """Lightweight descriptor for one zipped episode on the zip volume."""

    tar_path: Path
    episode_hash: str
    n_frames: int
    embodiment: str = "mecka_bimanual"



class ZipEpisodeResolver(EpisodeResolver):
    """Resolves episodes from catalog.json on the zip volume."""

    CATALOG_FILENAME = "catalog.json"

    def __init__(
        self,
        zip_dir: Path | str,
        key_map: dict | None = None,
        transform_list: list | None = None,
        norm_stats: dict | None = None,
        pause_removal_epsilon: float | None = None,
        valid_ratio: float = 0.1,
        debug: int | None = None,
        min_frames: int | None = None,
        seed: int = 42,
    ):
        super().__init__(
            Path(zip_dir),
            key_map,
            transform_list,
            norm_stats=norm_stats,
            pause_removal_epsilon=pause_removal_epsilon,
        )
        self.zip_dir = Path(zip_dir)
        self.valid_ratio = valid_ratio
        self.debug = debug
        self.min_frames = min_frames
        self.seed = seed
        self._catalog: list[EpisodeCatalogEntry] | None = None

    def load_catalog(self) -> list[EpisodeCatalogEntry]:
        if self._catalog is not None:
            return self._catalog

        catalog_path = self.zip_dir / self.CATALOG_FILENAME
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Catalog not found: {catalog_path}. "
                "Run `zip_zarr_to_vol.py` first to populate the zip volume."
            )

        with open(catalog_path) as f:
            raw: list[dict] = json.load(f)

        entries: list[EpisodeCatalogEntry] = []
        n_missing = 0
        for e in raw:
            tar_path = self.zip_dir / e["tar_filename"]
            if not tar_path.exists():
                n_missing += 1
                continue
            entries.append(
                EpisodeCatalogEntry(
                    tar_path=tar_path,
                    episode_hash=e["episode_hash"],
                    n_frames=int(e["n_frames"]),
                    embodiment=e.get("embodiment", "mecka_bimanual"),
                )
            )

        if n_missing:
            logger.warning(
                "ZipEpisodeResolver: %d catalog entries missing from zip volume (skipped)",
                n_missing,
            )

        if self.debug:
            entries = entries[: int(self.debug)]
            logger.info("ZipEpisodeResolver: debug=%d — using first %d episodes", self.debug, len(entries))

        if self.min_frames:
            before = len(entries)
            entries = [e for e in entries if e.n_frames >= self.min_frames]
            logger.info(
                "ZipEpisodeResolver: min_frames=%d — kept %d/%d episodes",
                self.min_frames, len(entries), before,
            )

        logger.info(
            "ZipEpisodeResolver: %d episodes, %d total frames from %s",
            len(entries),
            sum(e.n_frames for e in entries),
            catalog_path,
        )
        self._catalog = entries
        return self._catalog

    def split_catalog(self, mode: str) -> list[EpisodeCatalogEntry]:
        catalog = self.load_catalog()
        rng = random.Random(self.seed)
        shuffled = list(catalog)
        rng.shuffle(shuffled)
        n_valid = max(1, int(len(shuffled) * self.valid_ratio))
        if mode == "valid":
            return shuffled[:n_valid]
        return shuffled[n_valid:]

    def total_frames(self, mode: str = "train") -> int:
        return sum(e.n_frames for e in self.split_catalog(mode))

    def resolve(self, filters=None, **kwargs):
        raise NotImplementedError(
            "ZipEpisodeResolver does not support resolve(). "
            "Use PrefetchedMapDataset(resolver=...) instead."
        )


@dataclass
class RemoteEpisodeCatalogEntry:
    """Descriptor for one episode whose zarr store lives on R2 (no local tar)."""

    zarr_url: str  # r2://<bucket>/<key-prefix>, root of the episode's zarr store
    episode_hash: str
    n_frames: int
    embodiment: str = "mecka_bimanual"


class R2ZarrEpisodeResolver(EpisodeResolver):
    """Resolves episodes straight from R2 zarr stores -- no tar, no local
    volume, no PoolFillerThread staging step. Pairs with
    ``egomimic.rldb.zarr.prefetch.remote_dataset.RemoteZarrMapDataset``.

    Catalog format (produced by
    ``~/mecka/robotics_pipeline/convert_to_r2_zarr.py``): a JSON list of
    ``{"episode_hash", "zarr_key_prefix", "n_frames", "embodiment"}``, either
    at a local path (downloaded once ahead of time) or on R2 itself (in which
    case it is fetched via the same credentials used for episode reads).
    """

    CATALOG_FILENAME = "catalog.json"

    def __init__(
        self,
        bucket: str,
        key_prefix: str,
        catalog_path: str | Path | None = None,
        key_map: dict | None = None,
        transform_list: list | None = None,
        norm_stats: dict | None = None,
        pause_removal_epsilon: float | None = None,
        valid_ratio: float = 0.1,
        debug: int | None = None,
        min_frames: int | None = None,
        seed: int = 42,
    ):
        super().__init__(
            Path(f"r2://{bucket}/{key_prefix}"),
            key_map,
            transform_list,
            norm_stats=norm_stats,
            pause_removal_epsilon=pause_removal_epsilon,
        )
        self.bucket = bucket
        self.key_prefix = key_prefix.rstrip("/")
        # Local path to a pre-fetched catalog.json; if unset, read it from R2.
        self.catalog_path = Path(catalog_path) if catalog_path else None
        self.valid_ratio = valid_ratio
        self.debug = debug
        self.min_frames = min_frames
        self.seed = seed
        self._catalog: list[RemoteEpisodeCatalogEntry] | None = None

    def _read_catalog_json(self) -> list[dict]:
        if self.catalog_path is not None:
            if not self.catalog_path.exists():
                raise FileNotFoundError(f"Catalog not found: {self.catalog_path}")
            return json.loads(self.catalog_path.read_text())

        # Fetch catalog.json directly from R2 -- small file, no caching needed.
        from egomimic.rldb.zarr.remote_store import _cached_async_fs

        import asyncio

        fs = _cached_async_fs(os.environ.get("R2_ZARR_CACHE_DIR", "/cache/r2_zarr_chunks"))
        key = f"{self.bucket}/{self.key_prefix}/{self.CATALOG_FILENAME}"
        raw = asyncio.run(fs._cat_file(key)) if hasattr(fs, "_cat_file") else fs.cat_file(key)
        return json.loads(raw)

    def load_catalog(self) -> list[RemoteEpisodeCatalogEntry]:
        if self._catalog is not None:
            return self._catalog

        raw = self._read_catalog_json()
        entries = [
            RemoteEpisodeCatalogEntry(
                zarr_url=f"r2://{self.bucket}/{e['zarr_key_prefix']}",
                episode_hash=e["episode_hash"],
                n_frames=int(e["n_frames"]),
                embodiment=e.get("embodiment", "mecka_bimanual"),
            )
            for e in raw
        ]

        if self.debug:
            entries = entries[: int(self.debug)]
            logger.info("R2ZarrEpisodeResolver: debug=%d — using first %d episodes", self.debug, len(entries))

        if self.min_frames:
            before = len(entries)
            entries = [e for e in entries if e.n_frames >= self.min_frames]
            logger.info(
                "R2ZarrEpisodeResolver: min_frames=%d — kept %d/%d episodes",
                self.min_frames, len(entries), before,
            )

        logger.info(
            "R2ZarrEpisodeResolver: %d episodes, %d total frames from r2://%s/%s",
            len(entries), sum(e.n_frames for e in entries), self.bucket, self.key_prefix,
        )
        self._catalog = entries
        return self._catalog

    def split_catalog(self, mode: str) -> list[RemoteEpisodeCatalogEntry]:
        catalog = self.load_catalog()
        rng = random.Random(self.seed)
        shuffled = list(catalog)
        rng.shuffle(shuffled)
        n_valid = max(1, int(len(shuffled) * self.valid_ratio))
        if mode == "valid":
            return shuffled[:n_valid]
        return shuffled[n_valid:]

    def total_frames(self, mode: str = "train") -> int:
        return sum(e.n_frames for e in self.split_catalog(mode))

    def resolve(self, filters=None, **kwargs):
        raise NotImplementedError(
            "R2ZarrEpisodeResolver does not support resolve(). "
            "Use RemoteZarrMapDataset(resolver=...) instead."
        )


