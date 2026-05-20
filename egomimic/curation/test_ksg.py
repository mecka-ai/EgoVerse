"""Unit tests for the KSG MI estimator.

KSG is numerically sensitive, so we test against the closed-form MI of a
correlated bivariate Gaussian:

    MI(X, Y) = -0.5 * log(1 - rho^2)  (nats)

We allow a generous tolerance because KSG variance grows with dimensionality
and shrinks with sample size — these tests use N=4000 to keep them fast while
still hitting the analytical answer within ~0.05 nats.
"""

from __future__ import annotations

import numpy as np
import pytest

from egomimic.curation.ksg import ksg_mi, ksg_mi_averaged


def _gaussian_pair(rho: float, n: int = 4000, seed: int = 0):
    """Sample (X, Y) jointly Gaussian with corr(X, Y) = rho."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    x = z[:, 0:1]
    y = rho * z[:, 0:1] + np.sqrt(max(1.0 - rho**2, 0.0)) * z[:, 1:2]
    return x, y


def _true_mi(rho: float) -> float:
    return -0.5 * np.log(max(1.0 - rho**2, 1e-12))


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_ksg_mi_matches_gaussian_truth(rho: float) -> None:
    x, y = _gaussian_pair(rho, n=4000, seed=1)
    per_sample = ksg_mi(x, y, k=5)
    estimate = float(per_sample.mean())
    truth = _true_mi(rho)
    assert per_sample.shape == (len(x),)
    assert (
        abs(estimate - truth) < 0.08
    ), f"rho={rho}: estimate={estimate:.4f} truth={truth:.4f}"


def test_ksg_mi_independence_near_zero() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal((2000, 1))
    y = rng.standard_normal((2000, 1))
    estimate = float(ksg_mi(x, y, k=5).mean())
    assert abs(estimate) < 0.08, f"Independent estimate too large: {estimate:.4f}"


def test_ksg_mi_averaged_matches_truth() -> None:
    rho = 0.7
    x, y = _gaussian_pair(rho, n=4000, seed=7)
    per_sample = ksg_mi_averaged(x, y, k_range=(3, 7))
    estimate = float(per_sample.mean())
    truth = _true_mi(rho)
    assert per_sample.shape == (len(x),)
    assert (
        abs(estimate - truth) < 0.08
    ), f"averaged estimate={estimate:.4f} truth={truth:.4f}"


def test_ksg_mi_handles_1d_input() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(500)
    y = rng.standard_normal(500)
    per_sample = ksg_mi(x, y, k=5)
    assert per_sample.shape == (500,)
    assert np.isfinite(per_sample).all()


def test_ksg_mi_rejects_too_few_points() -> None:
    x = np.zeros((3, 1))
    y = np.zeros((3, 1))
    with pytest.raises(ValueError):
        ksg_mi(x, y, k=5)


def test_ksg_mi_averaged_rejects_empty_k_range() -> None:
    x, y = _gaussian_pair(0.5, n=200, seed=0)
    with pytest.raises(ValueError):
        ksg_mi_averaged(x, y, k_range=(5, 3))
