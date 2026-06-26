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
    cscore = OmegaConf.select(cfg, "model.cluster_scoring", default={}) or {}
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
        n_clusters=lang.n_clusters,
        n_clusters_max=lang.n_clusters_max,
        clustered_min_spans=lang.clustered_min_spans,
        cluster_score_algo=str(cscore.get("algo", "ksg")).lower().strip(),
        centroid_space=str(cscore.get("centroid_space", "action")).lower().strip(),
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
        n_clusters: int | str = "auto",
        n_clusters_max: int = 20,
        clustered_min_spans: int = 8,
        cluster_score_algo: str = "ksg",
        centroid_space: str = "action",
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
        self.n_clusters = n_clusters
        self.n_clusters_max = n_clusters_max
        self.clustered_min_spans = clustered_min_spans
        self.cluster_score_algo = cluster_score_algo
        self.centroid_space = centroid_space

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

    def _cluster_spans(self, emb: np.ndarray) -> np.ndarray:
        """K-means cluster span embeddings; ``n_clusters="auto"`` picks via silhouette."""
        from sklearn.cluster import KMeans

        n = emb.shape[0]
        if n <= 1:
            return np.zeros(n, dtype=np.int64)

        if self.n_clusters == "auto":
            from sklearn.metrics import silhouette_score

            k_min = 2
            k_max_c = min(int(self.n_clusters_max), n - 1)
            if k_max_c < k_min:
                return np.zeros(n, dtype=np.int64)

            # Silhouette is O(n^2); subsample for the score to keep auto-k fast.
            sample_size = min(2000, n)
            best_k, best_score, best_labels = k_min, -1.0, None
            for k in range(k_min, k_max_c + 1):
                km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
                lab = km.fit_predict(emb)
                try:
                    sc = silhouette_score(
                        emb, lab, sample_size=sample_size, random_state=self.seed
                    )
                except Exception:
                    continue
                logger.info("Clustered KSG: k=%d silhouette=%.4f", k, sc)
                if sc > best_score:
                    best_k, best_score, best_labels = k, sc, lab
            logger.info(
                "Clustered KSG: auto-selected k=%d (silhouette=%.4f)", best_k, best_score
            )
            return (
                best_labels if best_labels is not None else np.zeros(n, dtype=np.int64)
            )

        k = min(int(self.n_clusters), n)
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10)
        return km.fit_predict(emb)

    # ------------------------------------------------------------------
    # Modular per-cluster span scoring (selected by self.cluster_score_algo)
    # ------------------------------------------------------------------

    def _span_pooled(self, s: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Pool a span's per-frame latents to one vector in the configured space."""
        space = self.centroid_space
        if space == "state":
            return np.asarray(s, dtype=np.float64).mean(axis=0)
        if space == "joint":
            return np.concatenate(
                [np.asarray(s, dtype=np.float64).mean(axis=0),
                 np.asarray(a, dtype=np.float64).mean(axis=0)]
            )
        return np.asarray(a, dtype=np.float64).mean(axis=0)  # default: action

    def _cluster_scores_centroid(
        self, cs_state: list[np.ndarray], cs_action: list[np.ndarray]
    ) -> list[float]:
        """Score each span by its distance to the cluster centroid (per-span pooled).

        Centroid = mean of the cluster's pooled span vectors in ``centroid_space``;
        each span's score is the Euclidean distance from its pooled vector to that
        centroid (smaller = more representative of the cluster).
        """
        vecs = np.stack([self._span_pooled(s, a) for s, a in zip(cs_state, cs_action)])
        centroid = vecs.mean(axis=0)
        return [float(d) for d in np.linalg.norm(vecs - centroid, axis=1)]

    def _cluster_scores_ksg(
        self, cs_state: list[np.ndarray], cs_action: list[np.ndarray]
    ) -> list[float]:
        """Score each span by the mean per-frame KSG MI over its frames."""
        s_cat = np.concatenate([np.asarray(s, dtype=np.float64) for s in cs_state], axis=0)
        a_cat = np.concatenate([np.asarray(a, dtype=np.float64) for a in cs_action], axis=0)
        mi = self._ksg_exact(s_cat, a_cat)
        out: list[float] = []
        start = 0
        for s in cs_state:
            length = len(s)
            seg = mi[start : start + length]
            out.append(float(np.nanmean(seg)) if length > 0 else float("nan"))
            start += length
        return out

    def _score_cluster_spans(
        self, cs_state: list[np.ndarray], cs_action: list[np.ndarray]
    ) -> list[float]:
        """Dispatch to the configured per-cluster scoring algorithm."""
        if self.cluster_score_algo == "centroid":
            return self._cluster_scores_centroid(cs_state, cs_action)
        return self._cluster_scores_ksg(cs_state, cs_action)

    def score_clusters(
        self,
        span_state: list[np.ndarray],
        span_action: list[np.ndarray],
        span_ids: list[str],
        span_texts: list[str],
        span_meta: list[dict],
        text_embeddings: np.ndarray,
    ) -> dict[str, dict]:
        """
        Cluster annotation spans by language, then KSG-score each cluster.

        The atomic unit is the annotation span (a contiguous ``[start, end)`` frame
        range with one instruction). Spans are clustered by their Qwen3 language
        embedding; for each cluster, KSG MI is estimated over all frames belonging
        to its spans, and each span is scored by the mean per-frame MI over its own
        frames. Clusters with fewer than ``max(clustered_min_spans, k_max+1)`` spans
        are still formed but skipped for scoring (logged, spans recorded with
        ``score=None``).

        Returns:
            ``{cluster_id: {label, n_spans, scored, reason?, spans: {span_id: {...}}}}``,
            spans within each scored cluster sorted by score descending.
        """
        from collections import Counter

        n_spans = len(span_ids)
        emb = np.asarray(text_embeddings, dtype=np.float64)
        if emb.shape[0] != n_spans:
            raise ValueError(
                f"text_embeddings rows {emb.shape[0]} != n_spans {n_spans}"
            )

        labels = self._cluster_spans(emb)
        n_clusters = (int(labels.max()) + 1) if len(labels) else 0
        logger.info("Clustered KSG: %d spans → %d clusters", n_spans, n_clusters)

        k_max = self.k_range[1]
        min_spans = max(self.clustered_min_spans, k_max + 1)

        by_cluster: dict[int, list[int]] = {}
        for i, c in enumerate(labels):
            by_cluster.setdefault(int(c), []).append(i)

        def _dropped(idxs: list[int], label: str, reason: str) -> dict:
            return {
                "label": label,
                "n_spans": len(idxs),
                "scored": False,
                "reason": reason,
                "spans": {span_ids[i]: {**span_meta[i], "score": None} for i in idxs},
            }

        result: dict[str, dict] = {}
        for c in sorted(by_cluster):
            idxs = by_cluster[c]
            texts = [span_texts[i] for i in idxs]
            label = Counter(texts).most_common(1)[0][0] if texts else ""
            cluster_key = f"cluster_{c}"

            if len(idxs) < min_spans:
                logger.info(
                    "Clustered KSG: %s dropped (%d spans < %d)",
                    cluster_key, len(idxs), min_spans,
                )
                result[cluster_key] = _dropped(
                    idxs, label, f"only {len(idxs)} spans (< {min_spans})"
                )
                continue

            cs_state = [span_state[i] for i in idxs]
            cs_action = [span_action[i] for i in idxs]
            total_frames = sum(len(s) for s in cs_state)

            # KSG needs more frames than k; centroid only needs the per-span vectors.
            if self.cluster_score_algo != "centroid" and total_frames <= k_max:
                result[cluster_key] = _dropped(
                    idxs, label, f"only {total_frames} frames (<= k_max={k_max})"
                )
                continue

            scores = self._score_cluster_spans(cs_state, cs_action)

            spans_out: dict[str, dict] = {}
            for i, sc in zip(idxs, scores):
                score = sc if (sc is not None and np.isfinite(sc)) else None
                spans_out[span_ids[i]] = {**span_meta[i], "score": score}

            spans_out = dict(
                sorted(
                    spans_out.items(),
                    key=lambda kv: kv[1]["score"] if kv[1]["score"] is not None else float("-inf"),
                    reverse=True,
                )
            )
            result[cluster_key] = {
                "label": label,
                "n_spans": len(idxs),
                "scored": True,
                "spans": spans_out,
            }
            logger.info(
                "Clustered scoring (%s): %s scored %d spans (label=%r)",
                self.cluster_score_algo, cluster_key, len(idxs), label[:60],
            )

        n_scored = sum(1 for v in result.values() if v["scored"])
        logger.info(
            "Clustered KSG: scored %d / %d clusters (%d spans total)",
            n_scored, len(result), n_spans,
        )
        return result

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
