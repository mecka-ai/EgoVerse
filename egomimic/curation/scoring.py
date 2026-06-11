"""DemInf mutual-information scoring (Hejna et al. 2025).

Used by ``egomimic/modal/curateModal.py`` after per-episode zarr embedding.
Configure via ``hydra_configs/model/deminf_default.yaml``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from omegaconf import OmegaConf

from egomimic.curation.config import (
    select_language_conditioning_settings,
    select_seed,
)
from egomimic.curation.ksg import ksg_mi_averaged

logger = logging.getLogger(__name__)


def _optional_ksg_int(value: Any) -> int | None:
    """Parse ``model.ksg`` int fields; ``null`` / missing → disabled."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("", "null", "none", "~"):
        return None
    return int(value)


def aggregate_scores(
    per_timestep_mi: np.ndarray,
    episode_hashes: list[str],
    lengths: list[int],
) -> dict[str, float]:
    """Group per-timestep MI contributions back to episodes by nanmean."""
    scores: dict[str, float] = {}
    start = 0
    for ep_hash, length in zip(episode_hashes, lengths):
        ep_mi = per_timestep_mi[start : start + length]
        scores[ep_hash] = float(np.nanmean(ep_mi)) if length > 0 else float("nan")
        start += length
    return scores


def trajectory_scorer_from_cfg(cfg: Any) -> "TrajectoryScorer":
    """Build a TrajectoryScorer from a composed Hydra ``curate`` config."""
    ksg = OmegaConf.select(cfg, "model.ksg", default={}) or {}
    lang = select_language_conditioning_settings(cfg)
    return TrajectoryScorer(
        k_range=tuple(int(k) for k in ksg.get("k_range", [3, 7])),
        n_threads=int(ksg.get("n_threads", 12)),
        chunked_threshold=_optional_ksg_int(ksg.get("chunked_threshold")),
        chunked_max_points=_optional_ksg_int(ksg.get("chunked_max_points")),
        batch_size=int(ksg.get("batch_size", 10_000)),
        seed=select_seed(cfg),
        language_enabled=lang.enabled,
        language_mode=lang.mode,
        stratified_min_cluster_size=lang.stratified_min_cluster_size,
    )


def _flatten_language_labels(
    language_texts_by_episode: list[list[str]] | None,
    language_latents_by_episode: list[np.ndarray] | None,
) -> list[str]:
    """Build one label per timestep for stratified KSG."""
    if language_texts_by_episode:
        labels: list[str] = []
        for texts in language_texts_by_episode:
            labels.extend(texts)
        return labels
    if language_latents_by_episode:
        labels = []
        for lat in language_latents_by_episode:
            for row in lat:
                labels.append(row.tobytes())
        return labels
    return []


class TrajectoryScorer:
    """
    KSG mutual-information scoring over pre-embedded (state, action) latents.

    With ``language_enabled``, estimates either I(S, L; A) (``language_mode=concat``)
    or I(S; A | L) (``language_mode=stratified``).

    When ``chunked_threshold`` and ``chunked_max_points`` are both set, uses
    approximate subsampled KSG above the threshold. When either is ``null``,
    always uses exact KSG on all points. Chunked mode is disabled when language
    conditioning is enabled.
    """

    def __init__(
        self,
        k_range: tuple[int, int] = (3, 7),
        n_threads: int = 12,
        chunked_threshold: int | None = None,
        chunked_max_points: int | None = None,
        batch_size: int = 10_000,
        seed: int = 42,
        noise_scale: float = 1e-10,
        language_enabled: bool = False,
        language_mode: str = "concat",
        stratified_min_cluster_size: int = 10,
    ) -> None:
        self.k_range = k_range
        self.n_threads = n_threads
        self.chunked_threshold = chunked_threshold
        self.chunked_max_points = chunked_max_points
        self.batch_size = batch_size
        self.seed = seed
        self.noise_scale = noise_scale
        self.language_enabled = language_enabled
        self.language_mode = language_mode
        self.stratified_min_cluster_size = stratified_min_cluster_size

    def score_latents(
        self,
        state_latents: np.ndarray,
        action_latents: np.ndarray,
        episode_hashes: list[str],
        lengths: list[int],
        language_latents: np.ndarray | None = None,
        language_texts_by_episode: list[list[str]] | None = None,
        language_latents_by_episode: list[np.ndarray] | None = None,
    ) -> dict[str, float]:
        """
        Score episodes from stacked (T, latent_dim) state/action embeddings.

        When ``language_enabled``, pass ``language_latents`` with the same leading
        dimension as ``state_latents``. For ``language_mode=stratified``, also pass
        per-episode instruction strings (or per-episode language latent arrays).

        Returns:
            ``{episode_hash: mean_mi}`` per episode.
        """
        s_all = np.asarray(state_latents, dtype=np.float64)
        a_all = np.asarray(action_latents, dtype=np.float64)
        n_total = s_all.shape[0]
        n_episodes = len(episode_hashes)

        logger.info(
            "score_latents: %d episodes, %d total timesteps, state_dim=%d, action_dim=%d, "
            "language=%s",
            n_episodes,
            n_total,
            s_all.shape[1],
            a_all.shape[1],
            self.language_mode if self.language_enabled else "off",
        )

        if n_total < 2:
            logger.warning("score_latents: not enough timesteps (%d) — returning nan", n_total)
            return {h: float("nan") for h in episode_hashes}

        if self.language_enabled:
            if language_latents is None:
                raise ValueError(
                    "language_latents required when model.language_conditioning.enabled=true"
                )
            l_all = np.asarray(language_latents, dtype=np.float64)
            if l_all.shape[0] != n_total:
                raise ValueError(
                    f"language_latents length {l_all.shape[0]} != state length {n_total}"
                )

        use_chunked = (
            not self.language_enabled
            and self.chunked_threshold is not None
            and self.chunked_max_points is not None
            and n_total > self.chunked_threshold
        )
        logger.info(
            "score_latents: using %s KSG (chunked_threshold=%s, chunked_max_points=%s)",
            "chunked" if use_chunked else "exact",
            self.chunked_threshold, self.chunked_max_points,
        )

        t0 = time.perf_counter()
        if self.language_enabled:
            assert language_latents is not None
            l_all = np.asarray(language_latents, dtype=np.float64)
            if self.language_mode == "concat":
                x_all = np.hstack([s_all, l_all])
                mi = self._ksg_exact(x_all, a_all)
            else:
                labels = _flatten_language_labels(
                    language_texts_by_episode, language_latents_by_episode
                )
                if len(labels) != n_total:
                    raise ValueError(
                        f"stratified language labels length {len(labels)} != {n_total}"
                    )
                mi = self._mi_stratified(s_all, a_all, labels)
        elif use_chunked:
            mi = self._mi_chunked(s_all, a_all)
        else:
            mi = self._ksg_exact(s_all, a_all)
        logger.info("KSG scoring took %.2fs for %d timesteps", time.perf_counter() - t0, n_total)

        scores = aggregate_scores(mi, episode_hashes, lengths)
        _log_score_stats(scores)
        return scores

    def _ksg_exact(self, x_all: np.ndarray, a_all: np.ndarray) -> np.ndarray:
        return ksg_mi_averaged(
            x_all,
            a_all,
            k_range=self.k_range,
            noise_scale=self.noise_scale,
            n_workers=self.n_threads,
            batch_threshold=self.batch_size,
            batch_size=self.batch_size,
            seed=self.seed,
        )

    def _mi_stratified(
        self,
        s_all: np.ndarray,
        a_all: np.ndarray,
        labels: list[str] | list[bytes],
    ) -> np.ndarray:
        """I(S; A | L): KSG within each language cluster, NaN for small clusters."""
        n_total = len(s_all)
        mi = np.full(n_total, np.nan, dtype=np.float64)
        k_max = self.k_range[1]
        min_size = max(self.stratified_min_cluster_size, k_max + 1)

        clusters: dict[str | bytes, list[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)

        logger.info(
            "Stratified KSG: %d language clusters (min_cluster_size=%d)",
            len(clusters),
            min_size,
        )

        n_scored = 0
        for label, indices in clusters.items():
            if len(indices) < min_size:
                logger.debug(
                    "Skipping language cluster size=%d (< %d): %r",
                    len(indices),
                    min_size,
                    label[:80] if isinstance(label, str) else label,
                )
                continue
            idx = np.asarray(indices, dtype=np.int64)
            cluster_mi = self._ksg_exact(s_all[idx], a_all[idx])
            mi[idx] = cluster_mi
            n_scored += len(indices)

        logger.info(
            "Stratified KSG: scored %d / %d timesteps across %d clusters",
            n_scored,
            n_total,
            sum(1 for idx in clusters.values() if len(idx) >= min_size),
        )
        return mi

    def _mi_chunked(self, s_all: np.ndarray, a_all: np.ndarray) -> np.ndarray:
        """Approximate KSG via a subsampled reference set (large N)."""
        n_total = len(s_all)
        max_points = self.chunked_max_points
        assert max_points is not None

        if n_total > max_points:
            rng = np.random.default_rng(seed=self.seed)
            ref_idx = np.sort(
                rng.choice(n_total, size=max_points, replace=False)
            )
            s_ref, a_ref = s_all[ref_idx], a_all[ref_idx]
            logger.info(
                "Chunked KSG: %d / %d timesteps in reference set",
                max_points,
                n_total,
            )
        else:
            s_ref, a_ref = s_all, a_all

        n_ref = len(s_ref)
        k_approx = (self.k_range[0] + self.k_range[1]) // 2

        tree_z = cKDTree(np.hstack([s_ref, a_ref]))
        tree_s = cKDTree(s_ref)
        tree_a = cKDTree(a_ref)

        dists, _ = tree_z.query(
            np.hstack([s_all, a_all]), k=k_approx + 1, p=np.inf, workers=self.n_threads
        )
        eps_strict = dists[:, k_approx] * (1.0 - 1e-10)

        n_s = np.empty(n_total, dtype=np.float64)
        n_a = np.empty(n_total, dtype=np.float64)
        bs = self.batch_size
        for start in range(0, n_total, bs):
            end = min(start + bs, n_total)
            n_s[start:end] = tree_s.query_ball_point(
                s_all[start:end],
                eps_strict[start:end],
                p=np.inf,
                return_length=True,
                workers=self.n_threads,
            )
            n_a[start:end] = tree_a.query_ball_point(
                a_all[start:end],
                eps_strict[start:end],
                p=np.inf,
                return_length=True,
                workers=self.n_threads,
            )
        n_s = np.maximum(n_s - 1.0, 0.0)
        n_a = np.maximum(n_a - 1.0, 0.0)
        return (
            digamma(k_approx) - digamma(n_s + 1) - digamma(n_a + 1) + digamma(n_ref)
        )


def _log_score_stats(scores: dict[str, float]) -> None:
    vals = np.array([v for v in scores.values() if np.isfinite(v)])
    if len(vals) == 0:
        logger.warning("No finite MI scores computed")
        return
    logger.info(
        "Scoring complete. MI mean=%.4f std=%.4f min=%.4f max=%.4f",
        vals.mean(),
        vals.std(),
        vals.min(),
        vals.max(),
    )
