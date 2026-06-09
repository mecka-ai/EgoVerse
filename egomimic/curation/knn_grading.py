"""Cross-episode k-NN action-consistency grading.

For every (subsampled) state in an episode, retrieve the k nearest states from
*other* episodes of the same task (cosine over image-feature + proprio + phase
keys) and measure how far this episode's action chunk deviates from what the
neighbors did. Produces two per-state scores:

  * disagreement z — distance from the query chunk to the neighbor-chunk
    centroid, normalised by the neighbors' own spread ("is this episode the
    odd one out here?")
  * ambiguity — the neighbors' spread itself ("is this state a branch point
    for everyone?")

Chunk distance is decomposed into a *spatial* component (arc-length-resampled
EE paths — strategy divergence) and a *velocity* component (per-time-bin path
length profiles — retiming noise), so off-mode strategies and slow/fast
operators are flagged separately.

Pure numpy — no torch/Modal imports — so per-task scoring is CPU-cheap and the
metric can be iterated on without re-running the GPU featurise pass. Used by
``egomimic/modal/knnGradeModal.py``; configure via ``hydra_configs/knn_grade.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KnnGradeSettings:
    """Tunables for retrieval keys, neighbor search, and chunk metrics."""

    # Retrieval
    k: int = 15
    min_neighbors: int = 5
    pca_dim: int = 128
    image_weight: float = 1.0
    proprio_weight: float = 1.0
    phase_weight: float = 0.5
    # Coverage gate: a state is "covered" when its 1-NN distance is within
    # median + multiplier × MAD of the task's 1-NN distance distribution.
    # (A percentile gate would cap the gated fraction and mis-score tasks
    # where rare states are common.)
    coverage_mad_multiplier: float = 5.0
    query_block: int = 256

    # Chunk metrics
    n_resample: int = 20
    n_speed_bins: int = 10
    gripper_weight: float = 0.25
    spread_floor: float = 0.005  # metres — z denominator floor (spatial)
    speed_floor: float = 0.005  # metres/bin — z denominator floor (velocity)

    # Flagging / aggregation
    z_threshold: float = 2.0
    min_episodes: int = 8
    max_debug_states_per_episode: int = 3
    debug_neighbors: int = 5

    # Action vector layout (Mecka cartesian: [left xyzypr(6), right xyzypr(6)])
    left_pos: tuple[int, ...] = (0, 1, 2)
    right_pos: tuple[int, ...] = (6, 7, 8)
    gripper_dims: tuple[int, ...] = ()

    @classmethod
    def from_cfg(cls, knn_cfg: Any) -> "KnnGradeSettings":
        """Build from the ``knn`` block of a composed knn_grade Hydra config."""
        from omegaconf import OmegaConf

        d = OmegaConf.to_container(knn_cfg, resolve=True) if knn_cfg is not None else {}
        layout = d.pop("action_layout", {}) or {}
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for src, dst in (
            ("left_pos", "left_pos"),
            ("right_pos", "right_pos"),
            ("gripper_dims", "gripper_dims"),
        ):
            if layout.get(src) is not None:
                kwargs[dst] = tuple(int(i) for i in layout[src])
        return cls(**kwargs)


@dataclass
class ChunkFeatures:
    """Per-state geometric features derived from action chunks (T, H, A)."""

    left_path: np.ndarray  # (T, n_resample, 3) arc-length-resampled left EE path
    right_path: np.ndarray  # (T, n_resample, 3)
    left_speed: np.ndarray  # (T, n_speed_bins) path length per time bin
    right_speed: np.ndarray  # (T, n_speed_bins)
    gripper: np.ndarray | None  # (T, n_resample, n_grip) time-resampled traces


@dataclass
class EpisodeFeatures:
    """One episode's cached features, ready for grading."""

    ep_hash: str
    image_feats: np.ndarray  # (T, D_img) float — pooled backbone features
    proprio: np.ndarray  # (T, D_prop)
    chunk_feats: ChunkFeatures
    frame_idx: np.ndarray  # (T,) logical frame indices (post pause-filter)
    ep_len: int  # logical episode length the indices refer to


# --------------------------------------------------------------------------- #
# Retrieval keys
# --------------------------------------------------------------------------- #
@dataclass
class PcaWhitener:
    mean: np.ndarray
    components: np.ndarray  # (D, n_components)
    scales: np.ndarray  # (n_components,)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X.astype(np.float32) - self.mean) @ self.components) * self.scales


def fit_pca_whitener(
    X: np.ndarray, n_components: int, eps: float = 1e-6
) -> PcaWhitener:
    """Fit PCA whitening via eigendecomposition of the (D, D) covariance."""
    X = X.astype(np.float32)
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = (Xc.T @ Xc) / max(len(X) - 1, 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][: min(n_components, X.shape[1])]
    w, V = w[order], V[:, order]
    scales = 1.0 / np.sqrt(np.maximum(w, eps))
    return PcaWhitener(
        mean=mean, components=V.astype(np.float32), scales=scales.astype(np.float32)
    )


def _l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), eps)


def _standardize(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    X = X.astype(np.float32)
    return (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), eps)


def build_retrieval_keys(
    image_feats: np.ndarray,
    proprio: np.ndarray,
    phase: np.ndarray,
    settings: KnnGradeSettings,
    whitener: PcaWhitener | None = None,
) -> np.ndarray:
    """
    Weighted concat of [whitened image feats | standardized proprio | phase],
    L2-normalised so dot product = cosine similarity.

    Each block is L2-normalised before weighting so the weights control the
    blocks' *relative* influence regardless of raw dimensionality.
    """
    if whitener is None:
        whitener = fit_pca_whitener(image_feats, settings.pca_dim)
    blocks = [
        _l2_normalize(whitener.transform(image_feats)) * settings.image_weight,
        _l2_normalize(_standardize(proprio)) * settings.proprio_weight,
        (phase.astype(np.float32).reshape(-1, 1) - 0.5) * 2.0 * settings.phase_weight,
    ]
    return _l2_normalize(np.concatenate(blocks, axis=1).astype(np.float32))


# --------------------------------------------------------------------------- #
# Chunk geometry
# --------------------------------------------------------------------------- #
def resample_paths_by_arclength(
    paths: np.ndarray, n_points: int, block: int = 8192, eps: float = 1e-9
) -> np.ndarray:
    """
    Resample each (H, 3) path to ``n_points`` uniformly spaced *by arc length*.

    Removes execution-speed information: the same geometric path traversed at
    different speeds resamples to (near-)identical point sets. Degenerate
    paths (total length < eps) collapse to their first point repeated.
    """
    paths = np.asarray(paths, dtype=np.float32)
    N, H, _ = paths.shape
    seg = np.linalg.norm(np.diff(paths, axis=1), axis=-1)  # (N, H-1)
    cum = np.concatenate([np.zeros((N, 1), np.float32), np.cumsum(seg, axis=1)], axis=1)
    total = cum[:, -1]  # (N,)
    u = np.linspace(0.0, 1.0, n_points, dtype=np.float32)  # (n,)

    out = np.empty((N, n_points, 3), dtype=np.float32)
    for start in range(0, N, block):
        end = min(start + block, N)
        c = cum[start:end]  # (B, H)
        t = total[start:end, None] * u[None, :]  # (B, n)
        # batched searchsorted: idx of the segment containing each target
        idx = (c[:, None, :] <= t[:, :, None]).sum(axis=-1) - 1  # (B, n)
        idx = np.clip(idx, 0, H - 2)
        rows = np.arange(end - start)[:, None]
        seg_len = np.maximum(seg[start:end][rows, idx], eps)
        frac = np.clip((t - c[rows, idx]) / seg_len, 0.0, 1.0)[..., None]
        p0 = paths[start:end][rows, idx]
        p1 = paths[start:end][rows, idx + 1]
        out[start:end] = p0 + frac * (p1 - p0)

    degenerate = total < eps
    if degenerate.any():
        out[degenerate] = (
            paths[degenerate, :1][:, None, :]
            .repeat(n_points, axis=1)
            .reshape(-1, n_points, 3)
        )
    return out


def speed_profiles(paths: np.ndarray, n_bins: int) -> np.ndarray:
    """Path length per contiguous time bin: (N, H, 3) → (N, n_bins), metres."""
    paths = np.asarray(paths, dtype=np.float32)
    seg = np.linalg.norm(np.diff(paths, axis=1), axis=-1)  # (N, H-1)
    edges = np.linspace(0, seg.shape[1], n_bins + 1).astype(np.int64)[:-1]
    return np.add.reduceat(seg, edges, axis=1)


def time_resample_traces(traces: np.ndarray, n_points: int) -> np.ndarray:
    """Linear time-resample (N, H, D) → (N, n_points, D)."""
    traces = np.asarray(traces, dtype=np.float32)
    H = traces.shape[1]
    x = np.linspace(0.0, H - 1, n_points, dtype=np.float32)
    i0 = np.clip(np.floor(x).astype(np.int64), 0, H - 2)
    frac = (x - i0).astype(np.float32)[None, :, None]
    return traces[:, i0] + frac * (traces[:, i0 + 1] - traces[:, i0])


def compute_chunk_features(
    chunks: np.ndarray, settings: KnnGradeSettings
) -> ChunkFeatures:
    """Derive geometric features from raw action chunks (T, H, A)."""
    chunks = np.asarray(chunks, dtype=np.float32)
    left = chunks[:, :, list(settings.left_pos)]
    right = chunks[:, :, list(settings.right_pos)]
    grip = None
    if settings.gripper_dims:
        grip = time_resample_traces(
            chunks[:, :, list(settings.gripper_dims)], settings.n_resample
        )
    return ChunkFeatures(
        left_path=resample_paths_by_arclength(left, settings.n_resample),
        right_path=resample_paths_by_arclength(right, settings.n_resample),
        left_speed=speed_profiles(left, settings.n_speed_bins),
        right_speed=speed_profiles(right, settings.n_speed_bins),
        gripper=grip,
    )


def _mean_point_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean per-point Euclidean distance over the resample axis (… , n, 3)."""
    return np.linalg.norm(a - b, axis=-1).mean(axis=-1)


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def longest_true_run(mask: np.ndarray) -> int:
    """Length of the longest run of consecutive True values."""
    best = run = 0
    for v in np.asarray(mask, dtype=bool):
        run = run + 1 if v else 0
        best = max(best, run)
    return int(best)


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Each value's percentile rank within the array, in [0, 100]."""
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    denom = max(len(values) - 1, 1)
    return ranks / denom * 100.0


# --------------------------------------------------------------------------- #
# Main per-task grading
# --------------------------------------------------------------------------- #
def grade_task(
    episodes: list[EpisodeFeatures],
    settings: KnnGradeSettings,
) -> dict[str, Any]:
    """
    Grade every episode of one task by cross-episode action consistency.

    Returns a JSON-serialisable dict:
      ``task_summary``  — counts, coverage threshold, skip reason if any
      ``per_episode``   — {hash: metrics} incl. ``primary_score``
                          (= frac of covered states with spatial z > threshold;
                          higher = worse)
      ``debug_states``  — {hash: worst states + their neighbors} for grid
                          rendering and human review
    """
    result: dict[str, Any] = {
        "task_summary": {"n_episodes": len(episodes)},
        "per_episode": {},
        "debug_states": {},
        "settings": asdict(settings),
    }
    if len(episodes) < settings.min_episodes:
        result["task_summary"]["skipped"] = (
            f"only {len(episodes)} episodes (< min_episodes={settings.min_episodes})"
        )
        return result

    # ── Concatenate states; episodes stay contiguous so same-episode masking
    # is a single column-slice per query block. ──────────────────────────────
    hashes = [ep.ep_hash for ep in episodes]
    lengths = [len(ep.frame_idx) for ep in episodes]
    bounds = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    N = int(bounds[-1])
    if N == 0:
        result["task_summary"]["skipped"] = "no states"
        return result

    image_feats = np.concatenate([ep.image_feats for ep in episodes]).astype(np.float32)
    proprio = np.concatenate([ep.proprio for ep in episodes]).astype(np.float32)
    phase = np.concatenate(
        [ep.frame_idx / max(ep.ep_len - 1, 1) for ep in episodes]
    ).astype(np.float32)
    frame_idx_all = np.concatenate([ep.frame_idx for ep in episodes]).astype(np.int64)

    keys = build_retrieval_keys(image_feats, proprio, phase, settings)
    del image_feats, proprio

    left_path = np.concatenate([ep.chunk_feats.left_path for ep in episodes])
    right_path = np.concatenate([ep.chunk_feats.right_path for ep in episodes])
    left_speed = np.concatenate([ep.chunk_feats.left_speed for ep in episodes])
    right_speed = np.concatenate([ep.chunk_feats.right_speed for ep in episodes])
    grip = None
    if settings.gripper_dims and episodes[0].chunk_feats.gripper is not None:
        grip = np.concatenate([ep.chunk_feats.gripper for ep in episodes])

    # ── Pass 1: leave-one-episode-out top-k neighbor search ─────────────────
    k = min(settings.k, N - max(lengths))
    if k < settings.min_neighbors:
        result["task_summary"]["skipped"] = (
            f"only {k} cross-episode neighbors available (< min_neighbors)"
        )
        return result

    nbr_idx = np.empty((N, k), dtype=np.int64)
    nbr_sim = np.empty((N, k), dtype=np.float32)
    for e in range(len(episodes)):
        e_start, e_end = int(bounds[e]), int(bounds[e + 1])
        for q0 in range(e_start, e_end, settings.query_block):
            q1 = min(q0 + settings.query_block, e_end)
            sims = keys[q0:q1] @ keys.T  # (B, N)
            sims[:, e_start:e_end] = -np.inf  # leave own episode out
            top = np.argpartition(sims, -k, axis=1)[:, -k:]
            rows = np.arange(q1 - q0)[:, None]
            top_sims = sims[rows, top]
            order = np.argsort(-top_sims, axis=1)
            nbr_idx[q0:q1] = top[rows, order]
            nbr_sim[q0:q1] = top_sims[rows, order]

    # ── Coverage gate: 1-NN distance an outlier vs the task's bulk → the state
    # is rare, so "disagreement" there is unreliable (don't punish coverage).──
    nn_dist = 1.0 - nbr_sim[:, 0]
    med = float(np.median(nn_dist))
    mad = float(np.median(np.abs(nn_dist - med)))
    coverage_threshold = med + settings.coverage_mad_multiplier * max(mad, 1e-6)
    covered = nn_dist <= coverage_threshold

    # ── Pass 2: disagreement z + ambiguity per covered state ────────────────
    z_spatial = np.full(N, np.nan, dtype=np.float32)
    z_velocity = np.full(N, np.nan, dtype=np.float32)
    ambiguity = np.full(N, np.nan, dtype=np.float32)

    cov_idx = np.flatnonzero(covered)
    for b0 in range(0, len(cov_idx), settings.query_block):
        q = cov_idx[b0 : b0 + settings.query_block]
        nb = nbr_idx[q]  # (B, k)

        # spatial: distance to neighbor centroid in arc-length path space
        dev, spread = _centroid_dev_spread(q, nb, left_path, right_path, grip, settings)
        z_spatial[q] = dev / np.maximum(spread, settings.spread_floor)
        ambiguity[q] = spread

        # velocity: same construction on per-bin speed profiles
        dev_v, spread_v = _speed_dev_spread(q, nb, left_speed, right_speed)
        z_velocity[q] = dev_v / np.maximum(spread_v, settings.speed_floor)

    ambiguity_pct = np.full(N, np.nan, dtype=np.float32)
    if covered.any():
        ambiguity_pct[covered] = _percentile_ranks(ambiguity[covered])

    # ── Per-episode aggregation ──────────────────────────────────────────────
    state_to_hash = np.repeat(np.arange(len(episodes)), lengths)
    for e, ep_hash in enumerate(hashes):
        s = slice(int(bounds[e]), int(bounds[e + 1]))
        result["per_episode"][ep_hash] = _episode_metrics(
            z_spatial[s],
            z_velocity[s],
            ambiguity[s],
            ambiguity_pct[s],
            covered[s],
            settings,
        )
        result["debug_states"][ep_hash] = _episode_debug_states(
            int(bounds[e]),
            z_spatial[s],
            z_velocity[s],
            covered[s],
            frame_idx_all,
            nbr_idx,
            nbr_sim,
            state_to_hash,
            hashes,
            settings,
        )

    flagged_any = ambiguity_pct[covered] >= 90.0 if covered.any() else np.array([])
    result["task_summary"].update(
        n_states=N,
        n_covered=int(covered.sum()),
        coverage_threshold=coverage_threshold,
        k=k,
        frac_states_high_ambiguity=float(np.mean(flagged_any))
        if len(flagged_any)
        else float("nan"),
    )
    return result


def _centroid_dev_spread(
    q: np.ndarray,
    nb: np.ndarray,
    left_path: np.ndarray,
    right_path: np.ndarray,
    grip: np.ndarray | None,
    settings: KnnGradeSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Combined spatial (and gripper) deviation-from-centroid and neighbor spread."""
    dev = np.zeros(len(q), dtype=np.float32)
    spread = np.zeros(len(q), dtype=np.float32)
    for paths, w in ((left_path, 0.5), (right_path, 0.5)):
        nbr = paths[nb]  # (B, k, n, 3)
        centroid = nbr.mean(axis=1)  # (B, n, 3)
        dev += w * _mean_point_dist(paths[q], centroid)
        spread += w * _mean_point_dist(nbr, centroid[:, None]).mean(axis=1)
    if grip is not None:
        nbr_g = grip[nb]  # (B, k, n, D)
        centroid_g = nbr_g.mean(axis=1)
        dev += settings.gripper_weight * np.abs(grip[q] - centroid_g).mean(
            axis=(-1, -2)
        )
        spread += settings.gripper_weight * np.abs(nbr_g - centroid_g[:, None]).mean(
            axis=(-1, -2)
        ).mean(axis=1)
    return dev, spread


def _speed_dev_spread(
    q: np.ndarray,
    nb: np.ndarray,
    left_speed: np.ndarray,
    right_speed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dev = np.zeros(len(q), dtype=np.float32)
    spread = np.zeros(len(q), dtype=np.float32)
    for profiles in (left_speed, right_speed):
        nbr = profiles[nb]  # (B, k, nb_bins)
        centroid = nbr.mean(axis=1)
        dev += 0.5 * np.abs(profiles[q] - centroid).mean(axis=-1)
        spread += 0.5 * np.abs(nbr - centroid[:, None]).mean(axis=-1).mean(axis=1)
    return dev, spread


def _episode_metrics(
    z_sp: np.ndarray,
    z_vel: np.ndarray,
    amb: np.ndarray,
    amb_pct: np.ndarray,
    covered: np.ndarray,
    settings: KnnGradeSettings,
) -> dict[str, Any]:
    n = len(covered)
    cov = covered & np.isfinite(z_sp)
    flagged_sp = cov & (z_sp > settings.z_threshold)
    flagged_vel = cov & np.isfinite(z_vel) & (z_vel > settings.z_threshold)

    def _safe(fn, arr, mask):
        vals = arr[mask]
        return float(fn(vals)) if len(vals) else float("nan")

    frac_flagged = float(flagged_sp.sum() / cov.sum()) if cov.any() else float("nan")
    return {
        "n_states": int(n),
        "coverage_frac": float(cov.sum() / n) if n else 0.0,
        "frac_flagged_spatial": frac_flagged,
        "mean_z_spatial": _safe(np.mean, z_sp, cov),
        "p90_z_spatial": _safe(lambda v: np.percentile(v, 90), z_sp, cov),
        "longest_flagged_run": longest_true_run(flagged_sp),
        "frac_flagged_velocity": float(flagged_vel.sum() / cov.sum())
        if cov.any()
        else float("nan"),
        "mean_z_velocity": _safe(np.mean, z_vel, cov & np.isfinite(z_vel)),
        "mean_ambiguity": _safe(np.mean, amb, cov),
        "mean_ambiguity_pctile": _safe(np.mean, amb_pct, cov),
        "primary_score": frac_flagged,
    }


def _episode_debug_states(
    offset: int,
    z_sp: np.ndarray,
    z_vel: np.ndarray,
    covered: np.ndarray,
    frame_idx_all: np.ndarray,
    nbr_idx: np.ndarray,
    nbr_sim: np.ndarray,
    state_to_ep: np.ndarray,
    hashes: list[str],
    settings: KnnGradeSettings,
) -> list[dict[str, Any]]:
    """Worst-z states of one episode with their retrieved neighbors."""
    cov = covered & np.isfinite(z_sp)
    if not cov.any():
        return []
    order = np.argsort(np.where(cov, z_sp, -np.inf))[::-1]
    out = []
    for local in order[: settings.max_debug_states_per_episode]:
        if not cov[local]:
            break
        g = offset + int(local)
        nbrs = [
            {
                "hash": hashes[int(state_to_ep[j])],
                "frame_idx": int(frame_idx_all[j]),
                "sim": float(nbr_sim[g, rank]),
            }
            for rank, j in enumerate(nbr_idx[g, : settings.debug_neighbors])
        ]
        out.append(
            {
                "frame_idx": int(frame_idx_all[g]),
                "z_spatial": float(z_sp[local]),
                "z_velocity": float(z_vel[local])
                if np.isfinite(z_vel[local])
                else None,
                "neighbors": nbrs,
            }
        )
    return out
