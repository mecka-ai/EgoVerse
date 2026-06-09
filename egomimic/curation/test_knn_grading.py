"""Tests for cross-episode k-NN action-consistency grading (pure math)."""

from __future__ import annotations

import numpy as np
import pytest

from egomimic.curation.knn_grading import (
    EpisodeFeatures,
    KnnGradeSettings,
    build_retrieval_keys,
    compute_chunk_features,
    fit_pca_whitener,
    grade_task,
    longest_true_run,
    resample_paths_by_arclength,
    speed_profiles,
)

SETTINGS = KnnGradeSettings(
    k=8,
    min_neighbors=3,
    pca_dim=4,
    min_episodes=4,
    n_resample=12,
    n_speed_bins=6,
    query_block=64,
)


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def _line_path(speed_profile: np.ndarray) -> np.ndarray:
    """Straight-line (H, 3) path with per-step speeds given by speed_profile."""
    x = np.concatenate([[0.0], np.cumsum(speed_profile)])
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=-1)


def test_arclength_resample_is_speed_invariant():
    H = 40
    slow = _line_path(np.full(H - 1, 0.01))
    # Same geometric path: fast first half, dwell at the end point after.
    fast_steps = np.concatenate(
        [np.full((H - 1) // 2, 0.02), np.zeros(H - 1 - (H - 1) // 2)]
    )
    fast = _line_path(fast_steps)
    fast = fast * (slow[-1, 0] / fast[-1, 0])  # match total extent exactly

    rs = resample_paths_by_arclength(np.stack([slow, fast]), n_points=16)
    np.testing.assert_allclose(rs[0], rs[1], atol=1e-5)


def test_arclength_resample_degenerate_path():
    paths = np.zeros((2, 10, 3), dtype=np.float32)
    paths[1] += 3.0  # stationary at a non-origin point
    rs = resample_paths_by_arclength(paths, n_points=5)
    assert rs.shape == (2, 5, 3)
    np.testing.assert_allclose(rs[1], 3.0)


def test_speed_profiles_distinguish_retiming():
    H = 41
    steady = _line_path(np.full(H - 1, 0.01))
    rushed = _line_path(np.concatenate([np.full(20, 0.02), np.zeros(H - 1 - 20)]))
    profiles = speed_profiles(np.stack([steady, rushed]), n_bins=8)
    assert profiles.shape == (2, 8)
    # Same total length, very different distribution over time bins.
    np.testing.assert_allclose(
        profiles.sum(axis=1)[0], profiles.sum(axis=1)[1], rtol=1e-4
    )
    assert np.abs(profiles[0] - profiles[1]).max() > 0.01


def test_longest_true_run():
    assert longest_true_run(np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=bool)) == 3
    assert longest_true_run(np.zeros(5, dtype=bool)) == 0
    assert longest_true_run(np.ones(4, dtype=bool)) == 4


# --------------------------------------------------------------------------- #
# Retrieval keys
# --------------------------------------------------------------------------- #
def test_pca_whitener_whitens():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 8)) @ np.diag([5, 3, 2, 1, 0.5, 0.3, 0.2, 0.1])
    W = fit_pca_whitener(X, n_components=4)
    Xw = W.transform(X)
    cov = np.cov(Xw.T)
    np.testing.assert_allclose(cov, np.eye(4), atol=0.15)


def test_build_retrieval_keys_unit_norm():
    rng = np.random.default_rng(1)
    keys = build_retrieval_keys(
        rng.standard_normal((100, 16)).astype(np.float32),
        rng.standard_normal((100, 6)).astype(np.float32),
        rng.uniform(size=100).astype(np.float32),
        SETTINGS,
    )
    np.testing.assert_allclose(np.linalg.norm(keys, axis=1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# End-to-end synthetic task grading
# --------------------------------------------------------------------------- #
def _make_episode(
    ep_hash: str,
    direction: np.ndarray,
    rng: np.random.Generator,
    T: int = 30,
    H: int = 25,
    speed: float = 1.0,
    key_offset: float = 0.0,
) -> EpisodeFeatures:
    """
    Synthetic episode: states progress along a fixed 1-D task manifold; the
    action chunk from each state moves in ``direction`` at ``speed``. Retrieval
    keys are a deterministic function of task progress (+ small noise), so
    episodes of the same task land near each other in key space.
    """
    progress = np.linspace(0.0, 1.0, T, dtype=np.float32)
    image_feats = np.stack(
        [
            np.sin(2 * np.pi * progress),
            np.cos(2 * np.pi * progress),
            progress,
            np.full_like(progress, key_offset),
        ],
        axis=1,
    ) + rng.normal(scale=0.01, size=(T, 4)).astype(np.float32)
    proprio = np.stack([progress, progress**2], axis=1).astype(np.float32)
    proprio = np.tile(proprio, (1, 6))  # (T, 12) like ee_pose

    # Action chunks: straight paths from the current state in `direction`.
    # `speed` retimes the traversal but keeps total path length constant:
    # the first (H-1)/speed steps move faster, then the chunk dwells.
    if speed == 1.0:
        base_steps = np.full(H - 1, 0.01, dtype=np.float32)
    else:
        n_active = max(1, int(round((H - 1) / speed)))
        base_steps = np.zeros(H - 1, dtype=np.float32)
        base_steps[:n_active] = 0.01 * (H - 1) / n_active
    steps = np.cumsum(np.concatenate([[0.0], base_steps]))
    chunks = np.zeros((T, H, 12), dtype=np.float32)
    for arm_off in (0, 6):
        for d in range(3):
            chunks[:, :, arm_off + d] = (
                progress[:, None] * 0.1 + steps[None, :] * direction[d]
            )

    return EpisodeFeatures(
        ep_hash=ep_hash,
        image_feats=image_feats,
        proprio=proprio
        + rng.normal(scale=0.005, size=proprio.shape).astype(np.float32),
        chunk_feats=compute_chunk_features(chunks, SETTINGS),
        frame_idx=np.arange(T, dtype=np.int64) * 6,
        ep_len=T * 6,
    )


def test_grade_task_flags_offmode_episode():
    rng = np.random.default_rng(7)
    dir_a = np.array([1.0, 0.0, 0.0])
    dir_b = np.array([0.0, 1.0, 0.0])  # different strategy from same states
    episodes = [_make_episode(f"good_{i}", dir_a, rng) for i in range(11)]
    episodes.append(_make_episode("offmode", dir_b, rng))

    result = grade_task(episodes, SETTINGS)
    per_ep = result["per_episode"]

    assert per_ep["offmode"]["frac_flagged_spatial"] > 0.8
    for i in range(11):
        assert per_ep[f"good_{i}"]["frac_flagged_spatial"] < 0.2, (
            f"good_{i} over-flagged"
        )
    # Off-mode episode should also dominate the primary-score ranking.
    worst = max(per_ep, key=lambda h: per_ep[h]["primary_score"])
    assert worst == "offmode"
    # Debug states exist and reference real neighbors.
    dbg = result["debug_states"]["offmode"]
    assert dbg and all(n["hash"].startswith("good_") for n in dbg[0]["neighbors"])


def test_grade_task_velocity_only_outlier():
    rng = np.random.default_rng(11)
    dir_a = np.array([1.0, 0.0, 0.0])
    episodes = [_make_episode(f"good_{i}", dir_a, rng) for i in range(11)]
    # Same geometric path, executed at 2.5x speed then dwelling: chunk covers
    # the same shape but with a very different time profile.
    fast = _make_episode("fast", dir_a, rng, speed=2.5)
    episodes.append(fast)

    result = grade_task(episodes, SETTINGS)
    per_ep = result["per_episode"]
    # Velocity flagging should exceed spatial flagging for the retimed episode.
    assert (
        per_ep["fast"]["frac_flagged_velocity"] > per_ep["fast"]["frac_flagged_spatial"]
    )
    assert per_ep["fast"]["frac_flagged_velocity"] > 0.5


def test_grade_task_no_coverage_not_flagged():
    rng = np.random.default_rng(3)
    dir_a = np.array([1.0, 0.0, 0.0])
    episodes = [_make_episode(f"good_{i}", dir_a, rng) for i in range(11)]
    # Rare-state episode: far away in key space AND different actions. It must
    # be gated as no-coverage, not condemned as disagreeing.
    rare = _make_episode("rare", np.array([0.0, 0.0, 1.0]), rng, key_offset=50.0)
    episodes.append(rare)

    result = grade_task(episodes, SETTINGS)
    per_ep = result["per_episode"]
    assert per_ep["rare"]["coverage_frac"] < 0.2
    flagged = per_ep["rare"]["frac_flagged_spatial"]
    assert np.isnan(flagged) or flagged < 0.5


def test_grade_task_skips_small_tasks():
    rng = np.random.default_rng(5)
    episodes = [_make_episode(f"e{i}", np.array([1.0, 0, 0]), rng) for i in range(2)]
    result = grade_task(episodes, SETTINGS)
    assert "skipped" in result["task_summary"]
    assert result["per_episode"] == {}


def test_settings_from_cfg_roundtrip():
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.create(
        {
            "k": 9,
            "z_threshold": 1.5,
            "action_layout": {
                "left_pos": [0, 1, 2],
                "right_pos": [7, 8, 9],
                "gripper_dims": [6, 13],
            },
        }
    )
    s = KnnGradeSettings.from_cfg(cfg)
    assert s.k == 9 and s.z_threshold == 1.5
    assert s.right_pos == (7, 8, 9) and s.gripper_dims == (6, 13)


def test_compute_chunk_features_gripper_layout():
    settings = KnnGradeSettings(
        left_pos=(0, 1, 2), right_pos=(7, 8, 9), gripper_dims=(6, 13), n_resample=8
    )
    chunks = np.random.default_rng(0).standard_normal((5, 20, 14)).astype(np.float32)
    feats = compute_chunk_features(chunks, settings)
    assert feats.left_path.shape == (5, 8, 3)
    assert feats.gripper is not None and feats.gripper.shape == (5, 8, 2)
