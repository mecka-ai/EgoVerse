"""Episode catalog: the per-episode entry and the zip-volume resolver.

``ZipEpisodeResolver`` discovers complete episodes on the zip volume from
``{episode_hash}.tar`` + ``{episode_hash}.done`` pairs and reads frame counts
from each tar's root ``zarr.json``.  An optional ``catalog.json`` is ignored.
"""

from __future__ import annotations

import json
import logging
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path

from egomimic.rldb.zarr.zarr_dataset_multi import EpisodeResolver

logger = logging.getLogger(__name__)


def _read_episode_meta_from_tar(tar_path: Path) -> tuple[int, str]:
    """Read ``total_frames`` and ``embodiment`` from a zipped zarr v3 episode."""
    with tarfile.open(tar_path, "r") as tf:
        try:
            member = tf.getmember("zarr.json")
        except KeyError as exc:
            raise ValueError(f"{tar_path.name}: missing zarr.json in tar") from exc
        with tf.extractfile(member) as f:
            zarr_json = json.load(f)

    attrs = zarr_json.get("attributes")
    if not isinstance(attrs, dict):
        attrs = {}
    n_frames = int(attrs.get("total_frames", 0) or 0)
    embodiment = str(attrs.get("embodiment", "mecka_bimanual"))
    return n_frames, embodiment


def _discover_complete_episodes(
    zip_dir: Path,
    include_hashes: set[str] | None,
) -> list[tuple[str, Path]]:
    """Return ``(episode_hash, tar_path)`` for every complete tar+.done pair."""
    if include_hashes is not None:
        pairs: list[tuple[str, Path]] = []
        for episode_hash in sorted(include_hashes):
            tar_path = zip_dir / f"{episode_hash}.tar"
            if tar_path.exists() and (zip_dir / f"{episode_hash}.done").exists():
                pairs.append((episode_hash, tar_path))
        return pairs

    done_hashes = {p.stem for p in zip_dir.glob("*.done")}
    pairs = []
    for episode_hash in sorted(done_hashes):
        tar_path = zip_dir / f"{episode_hash}.tar"
        if tar_path.exists():
            pairs.append((episode_hash, tar_path))
    return pairs

@dataclass
class EpisodeCatalogEntry:
    """Lightweight descriptor for one zipped episode on the zip volume."""

    tar_path: Path
    episode_hash: str
    n_frames: int
    embodiment: str = "mecka_bimanual"



class ZipEpisodeResolver(EpisodeResolver):
    """Resolves episodes from tar+.done pairs on the zip volume."""

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
        eps_to_use: str | None = None,
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
        # Optional episode-hash allowlist (path to a JSON list of hashes). The
        # zip catalog carries no task labels, so restricting to a specific task
        # set (e.g. the 14-task curation episodes) is done by episode_hash.
        self.include_hashes: set[str] | None = None
        if eps_to_use:
            with open(eps_to_use) as f:
                self.include_hashes = set(json.load(f))
            logger.info(
                "ZipEpisodeResolver: eps_to_use=%s (%d hashes)",
                eps_to_use, len(self.include_hashes),
            )
        self._catalog: list[EpisodeCatalogEntry] | None = None

    def load_catalog(self) -> list[EpisodeCatalogEntry]:
        if self._catalog is not None:
            return self._catalog

        pairs = _discover_complete_episodes(self.zip_dir, self.include_hashes)
        if not pairs:
            raise FileNotFoundError(
                f"No complete episodes (.tar + .done) found in {self.zip_dir}. "
                "Run `zip_zarr_to_vol.py` first to populate the zip volume."
            )

        if self.include_hashes is not None:
            logger.info(
                "ZipEpisodeResolver: eps_to_use — found %d/%d episodes on zip volume",
                len(pairs),
                len(self.include_hashes),
            )

        if self.debug:
            pairs = pairs[: int(self.debug)]
            logger.info(
                "ZipEpisodeResolver: debug=%d — using first %d episodes",
                self.debug,
                len(pairs),
            )

        entries: list[EpisodeCatalogEntry] = []
        n_meta_errors = 0
        for episode_hash, tar_path in pairs:
            try:
                n_frames, embodiment = _read_episode_meta_from_tar(tar_path)
            except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
                n_meta_errors += 1
                logger.warning(
                    "ZipEpisodeResolver: skipping %s — could not read tar metadata: %s",
                    episode_hash,
                    exc,
                )
                continue
            entries.append(
                EpisodeCatalogEntry(
                    tar_path=tar_path,
                    episode_hash=episode_hash,
                    n_frames=n_frames,
                    embodiment=embodiment,
                )
            )

        if n_meta_errors:
            logger.warning(
                "ZipEpisodeResolver: skipped %d episodes with unreadable tar metadata",
                n_meta_errors,
            )

        if self.min_frames:
            before = len(entries)
            entries = [e for e in entries if e.n_frames >= self.min_frames]
            logger.info(
                "ZipEpisodeResolver: min_frames=%d — kept %d/%d episodes",
                self.min_frames, len(entries), before,
            )

        if not entries:
            raise FileNotFoundError(
                f"No usable episodes found in {self.zip_dir} "
                "(complete tar+.done pairs exist but metadata could not be read)."
            )

        logger.info(
            "ZipEpisodeResolver: %d episodes, %d total frames from %s",
            len(entries),
            sum(e.n_frames for e in entries),
            self.zip_dir,
        )
        self._catalog = entries
        return self._catalog

    def split_catalog(self, mode: str) -> list[EpisodeCatalogEntry]:
        catalog = self.load_catalog()
        if mode == "total":
            # Use the whole (eps_to_use-filtered) catalog, no train/valid split.
            # Used by train_viz datasets whose eps_to_use is already a curated set.
            return list(catalog)
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


