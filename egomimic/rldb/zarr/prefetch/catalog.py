"""Episode catalog: the per-episode entry and the zip-volume resolver.

``ZipEpisodeResolver`` reads ``catalog.json`` from the zip volume and inherits
key_map / transform_list / norm_stats plumbing from ``EpisodeResolver``.
"""

from __future__ import annotations

import json
import logging
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
        read_block_size: int = 1,
        read_block_cache_blocks: int = 2,
        decode_images: bool = True,
    ):
        super().__init__(
            Path(zip_dir),
            key_map,
            transform_list,
            norm_stats=norm_stats,
            pause_removal_epsilon=pause_removal_epsilon,
            read_block_size=read_block_size,
            read_block_cache_blocks=read_block_cache_blocks,
            decode_images=decode_images,
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
