"""Deterministic per-run episode permutation consumed one window per epoch."""

from __future__ import annotations

import logging
import random

from egomimic.rldb.zarr.prefetch.catalog import EpisodeCatalogEntry

logger = logging.getLogger(__name__)

class EpisodePlan:
    """Fixed-order schedule of episodes across all epochs.

    The plan is a single shuffle of the full split (same seed → same order
    on every DDP rank). Epoch ``e`` consumes plan[e*ept : (e+1)*ept] with
    wrap-around. No re-shuffle between epochs; randomness comes from the
    frame-level shuffle in ``prepare_epoch`` and from the plan's initial
    permutation.
    """

    def __init__(
        self,
        episodes: list[EpisodeCatalogEntry],
        episodes_per_epoch: int,
        seed: int = 42,
    ):
        if not episodes:
            raise ValueError("EpisodePlan: empty episode list")
        if episodes_per_epoch <= 0:
            raise ValueError(f"EpisodePlan: episodes_per_epoch must be > 0, got {episodes_per_epoch}")

        rng = random.Random(seed)
        order = list(episodes)
        rng.shuffle(order)
        self.episodes: list[EpisodeCatalogEntry] = order
        self.episodes_per_epoch = int(episodes_per_epoch)
        self.n = len(order)
        self._by_hash = {e.episode_hash: e for e in order}

    def epoch_episodes(self, epoch: int) -> list[EpisodeCatalogEntry]:
        """Episodes scheduled for ``epoch`` (with wrap-around)."""
        start = (epoch * self.episodes_per_epoch) % self.n
        out: list[EpisodeCatalogEntry] = []
        for i in range(self.episodes_per_epoch):
            out.append(self.episodes[(start + i) % self.n])
        return out

    def episodes_in_window(self, start_epoch: int, end_epoch_exclusive: int) -> list[EpisodeCatalogEntry]:
        """All distinct episodes scheduled in [start_epoch, end_epoch_exclusive)."""
        seen: set[str] = set()
        out: list[EpisodeCatalogEntry] = []
        for e in range(start_epoch, end_epoch_exclusive):
            for ep in self.epoch_episodes(e):
                if ep.episode_hash not in seen:
                    seen.add(ep.episode_hash)
                    out.append(ep)
        return out

    def hashes_in_window(self, start_epoch: int, end_epoch_exclusive: int) -> set[str]:
        return {ep.episode_hash for ep in self.episodes_in_window(start_epoch, end_epoch_exclusive)}

    def episode_at(self, plan_idx: int) -> EpisodeCatalogEntry:
        return self.episodes[plan_idx % self.n]


