"""Kraskov-Stögbauer-Grassberger (KSG) mutual information estimator.

Implements Algorithm 1 from:
    Kraskov, Stögbauer, Grassberger (2004).
    "Estimating mutual information."
    Physical Review E, 69(6).
"""

from __future__ import annotations

import logging
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
from tqdm import tqdm

logger = logging.getLogger(__name__)

_BATCH_THRESHOLD = 10_000


def ksg_mi(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    noise_scale: float = 1e-10,
    n_workers: int = 12,
    batch_threshold: int = _BATCH_THRESHOLD,
    batch_size: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Estimate per-sample MI contributions using KSG Algorithm 1.

    For each point i:
        1. Find the k-th NN distance ε_i in joint (x, y) Chebyshev space.
        2. Count n_x(i), n_y(i) in marginal spaces strictly within ε_i.
        3. mi_i = ψ(k) − ψ(n_x+1) − ψ(n_y+1) + ψ(N)

    Args:
        x: (N, dx) state embeddings.
        y: (N, dy) action embeddings.
        k: Number of nearest neighbours (≥ 1).
        noise_scale: Jitter to break exact ties (avoids ε=0 queries).
        n_workers: Thread-count for cKDTree queries.
        batch_threshold: Switch to batched NN counting above this N.
        batch_size: Points per batch in batched mode.

    Returns:
        (N,) float64 per-sample MI contributions; mean ≈ I(X; Y) in nats.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]

    N = len(x)
    if N <= k:
        raise ValueError(f"k={k} must be < N={N}")

    logger.info("KSG MI: N=%d, k=%d, dx=%d, dy=%d, n_workers=%d", N, k, x.shape[1], y.shape[1], n_workers)

    rng = np.random.default_rng(seed=seed)
    x = x + rng.standard_normal(x.shape) * noise_scale
    y = y + rng.standard_normal(y.shape) * noise_scale

    t_tree = time.perf_counter()
    z = np.hstack([x, y])
    tree_z = cKDTree(z)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)
    logger.info("KSG tree build: %.2fs", time.perf_counter() - t_tree)

    t_nn = time.perf_counter()
    dists, _ = tree_z.query(z, k=k + 1, p=np.inf, workers=n_workers)
    eps = dists[:, k]
    logger.info("KSG joint NN query (k=%d): %.2fs", k, time.perf_counter() - t_nn)

    eps_strict = eps * (1.0 - 1e-10)

    t_marginals = time.perf_counter()
    if N <= batch_threshold:
        n_x = _count_neighbours(tree_x, x, eps_strict, n_workers=n_workers)
        n_y = _count_neighbours(tree_y, y, eps_strict, n_workers=n_workers)
    else:
        n_x = _count_neighbours_batched(tree_x, x, eps_strict, batch_size=batch_size, n_workers=n_workers)
        n_y = _count_neighbours_batched(tree_y, y, eps_strict, batch_size=batch_size, n_workers=n_workers)
    logger.info("KSG marginal counts: %.2fs (batched=%s)", time.perf_counter() - t_marginals, N > batch_threshold)

    mi = digamma(k) - digamma(n_x + 1) - digamma(n_y + 1) + digamma(N)
    logger.info("KSG k=%d done: mi_mean=%.4f mi_std=%.4f", k, float(mi.mean()), float(mi.std()))
    return mi.astype(np.float64)


def _count_neighbours(
    tree: cKDTree,
    points: np.ndarray,
    radii: np.ndarray,
    n_workers: int = 12,
) -> np.ndarray:
    """Count neighbours within radii[i] for each point i, excluding self."""
    counts = tree.query_ball_point(points, radii, p=np.inf, return_length=True, workers=n_workers)
    return np.maximum(np.asarray(counts, dtype=np.float64) - 1.0, 0.0)


def _count_neighbours_batched(
    tree: cKDTree,
    points: np.ndarray,
    radii: np.ndarray,
    batch_size: int = 10_000,
    n_workers: int = 12,
) -> np.ndarray:
    """Batched version of _count_neighbours for large N."""
    N = len(points)
    counts = np.empty(N, dtype=np.float64)
    for start in tqdm(range(0, N, batch_size), desc="KSG marginals", unit="batch", dynamic_ncols=True):
        end = min(start + batch_size, N)
        batch_counts = tree.query_ball_point(
            points[start:end], radii[start:end], p=np.inf, return_length=True, workers=n_workers
        )
        counts[start:end] = np.maximum(np.asarray(batch_counts, dtype=np.float64) - 1.0, 0.0)
    return counts


def ksg_mi_averaged(
    x: np.ndarray,
    y: np.ndarray,
    k_range: tuple[int, int] = (3, 7),
    noise_scale: float = 1e-10,
    n_workers: int = 12,
    batch_threshold: int = _BATCH_THRESHOLD,
    batch_size: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Average per-sample MI estimates over a range of k values.

    Averaging over a small k range reduces variance without significant
    compute overhead.

    Args:
        x: (N, dx) state embeddings.
        y: (N, dy) action embeddings.
        k_range: (k_min, k_max) inclusive range to average over.
        noise_scale: Jitter for tie-breaking.
        n_workers: Thread-count for cKDTree queries.
        batch_threshold: Switch to batched counting above this N.
        batch_size: Points per batch in batched mode.

    Returns:
        (N,) float64 per-sample MI contributions, averaged over k values.
    """
    k_min, k_max = k_range
    k_values = list(range(k_min, k_max + 1))
    if not k_values:
        raise ValueError(f"Invalid k_range: {k_range}")

    logger.info("KSG averaged MI: k_range=%s (%d values)", k_range, len(k_values))
    t0 = time.perf_counter()
    estimates = np.stack(
        [
            ksg_mi(
                x,
                y,
                k=k,
                noise_scale=noise_scale,
                n_workers=n_workers,
                batch_threshold=batch_threshold,
                batch_size=batch_size,
                seed=seed,
            )
            for k in tqdm(k_values, desc="KSG k-values", unit="k", dynamic_ncols=True)
        ],
        axis=0,
    )
    result = estimates.mean(axis=0)
    logger.info(
        "KSG averaged MI done: %.2fs total, mi_mean=%.4f mi_std=%.4f",
        time.perf_counter() - t0, float(result.mean()), float(result.std()),
    )
    return result
