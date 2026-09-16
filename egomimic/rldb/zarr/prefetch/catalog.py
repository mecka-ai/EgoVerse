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


class ZarrDirEpisodeResolver(ZipEpisodeResolver):
    """Stage plain ``.zarr`` directories instead of tar archives.

    Same prefetch machinery as ZipEpisodeResolver -- the pool, filler and
    dataset are unchanged, because ``_extract_tar_to_dir`` copies a directory
    source rather than untarring it. What differs is where the catalog comes
    from: a zarr volume has no ``catalog.json``, so entries are discovered by
    walking ``<root>/<group>/<episode>.zarr`` and reading ``n_frames`` from
    each store's ``.zattrs``.

    Reading zattrs for every episode is one small read each and is done once at
    startup, but at ~150 MB/s per container that is still minutes for tens of
    thousands of episodes -- so the result is cached to ``catalog_cache`` (on
    the volume, next to the data) and reused on subsequent runs and restarts.

    ``eps_to_use`` takes a JSON list of episode hashes, which is how a curated
    subset (e.g. a quality-ranked selection) is pinned for a run.
    """

    CATALOG_FILENAME = "zarr_catalog.json"

    def __init__(self, zarr_dir, eps_to_use: str | None = None,
                 catalog_cache: str | None = None, **kwargs):
        super().__init__(zarr_dir, **kwargs)
        self.eps_to_use = eps_to_use
        self.catalog_cache = catalog_cache

    def _n_frames(self, ep: Path) -> int | None:
        try:
            attrs = json.loads((ep / "zarr.json").read_text()).get("attributes", {})
        except Exception:
            return None
        n = attrs.get("total_frames") or attrs.get("n_frames")
        if n:
            return int(n)
        feats = attrs.get("features") or {}
        for k, v in feats.items():
            if v.get("dtype") == "jpeg":
                try:
                    meta = json.loads((ep / k / "zarr.json").read_text())
                    return int(meta["shape"][0])
                except Exception:
                    return None
        return None

    def load_catalog(self) -> list[EpisodeCatalogEntry]:
        if self._catalog is not None:
            return self._catalog

        cache = Path(self.catalog_cache) if self.catalog_cache else (
            self.zip_dir / self.CATALOG_FILENAME
        )
        raw: list[dict] | None = None
        if cache.exists():
            try:
                raw = json.loads(cache.read_text())
                logger.info("ZarrDirEpisodeResolver: catalog cache hit (%d entries) %s",
                            len(raw), cache)
            except Exception:
                raw = None

        if raw is None:
            logger.info("ZarrDirEpisodeResolver: scanning %s (no cache)", self.zip_dir)
            # Layouts vary: the 6k corpus is <group>/<ep>.zarr, the QA_exp
            # volume is <domain>/<split>/<ep>.zarr. Cover both rather than
            # assume a depth. Deeper than this would mean rglob, which is far
            # too slow over a FUSE mount with tens of thousands of entries.
            seen: set[str] = set()
            raw = []
            for pattern in ("*/*.zarr", "*/*/*.zarr"):
                for ep in sorted(self.zip_dir.glob(pattern)):
                    if ep.parts[len(self.zip_dir.parts)].startswith("_"):
                        continue
                    # An episode can be materialised under more than one split
                    # directory; keep the first and skip the copies, or it would
                    # be sampled twice within an epoch.
                    if ep.stem in seen:
                        continue
                    n = self._n_frames(ep)
                    if n:
                        seen.add(ep.stem)
                        raw.append({"path": str(ep), "episode_hash": ep.stem,
                                    "n_frames": n,
                                    "group": "/".join(ep.parts[len(self.zip_dir.parts):-1])})
            try:
                cache.write_text(json.dumps(raw))
                logger.info("ZarrDirEpisodeResolver: wrote catalog cache %s", cache)
            except Exception as e:  # read-only mount is not fatal
                logger.warning("could not write catalog cache: %s", e)

        keep: set[str] | None = None
        if self.eps_to_use:
            with open(self.eps_to_use) as f:
                keep = set(json.load(f))
            logger.info("ZarrDirEpisodeResolver: eps_to_use — %d hashes from %s",
                        len(keep), self.eps_to_use)

        entries = [
            EpisodeCatalogEntry(
                tar_path=Path(e["path"]),      # a directory here, copied not untarred
                episode_hash=e["episode_hash"],
                n_frames=int(e["n_frames"]),
            )
            for e in raw
            if keep is None or e["episode_hash"] in keep
        ]

        if self.debug:
            entries = entries[: int(self.debug)]
        if self.min_frames:
            entries = [e for e in entries if e.n_frames >= self.min_frames]

        logger.info("ZarrDirEpisodeResolver: %d episodes, %d total frames",
                    len(entries), sum(e.n_frames for e in entries))
        self._catalog = entries
        return self._catalog


