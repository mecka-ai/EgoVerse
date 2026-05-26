"""Unit tests for the Modal pause-precompute fan-out script.

Worker round-trip via ``_pause_precompute_shard.local(...)`` — writes a
synthetic zarr group with ``{left,right}.obs_ee_pose`` (+ optionally
``{left,right}.obs_keypoints``) and confirms the worker's keep-mask
matches ``_build_pause_keep_mask`` from zarr_dataset_multi (the single
source of truth used by the in-process precompute path and by the Nebius
worker).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from egomimic.modal.pause_precompute_modal import _pause_precompute_shard
from egomimic.rldb.zarr.zarr_dataset_multi import (
    PAUSE_DETECT_KEYPOINT_KEYS,
    PAUSE_DETECT_KEYS,
    _build_pause_keep_mask,
)


@pytest.fixture(autouse=True)
def _stub_volume_reload(monkeypatch):
    """``zarr_volume.reload()`` only works inside a real Modal container.

    For ``_pause_precompute_shard.local(...)`` to run in pytest we need to
    no-op the reload (worker uses local filesystem paths via tmp_path —
    nothing to reload anyway).
    """
    import egomimic.modal.pause_precompute_modal as mod

    monkeypatch.setattr(mod.zarr_volume, "reload", lambda *a, **kw: None)


def _synthetic_poses(
    *, n_frames: int, pause_spans: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Build left/right ee_pose arrays where pause-span frames are stationary
    and non-pause frames step by 0.1 m in x (well above eps=0.005)."""
    left = np.zeros((n_frames, 7), dtype=np.float64)
    right = np.zeros((n_frames, 7), dtype=np.float64)
    left[:, 6] = 1.0  # quaternion w
    right[:, 6] = 1.0
    motion = 0.0
    for i in range(n_frames):
        in_pause = any(s <= i < e for s, e in pause_spans)
        if not in_pause:
            motion += 0.1
        left[i, 0] = motion
        right[i, 0] = motion + 0.5
    return left, right


def _write_episode(
    path: Path,
    *,
    n_frames: int,
    pause_spans: list[tuple[int, int]],
    include_keypoints: bool = False,
    omit_ee_pose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Write a minimal zarr v3 group for the worker to consume.

    Returns ``(left, right, left_kp, right_kp)`` — the same arrays the
    worker will see, so reference checks can be computed from them.
    """
    left, right = _synthetic_poses(n_frames=n_frames, pause_spans=pause_spans)
    left_kp = right_kp = None
    store = zarr.open(str(path), mode="w", zarr_format=3)

    if not omit_ee_pose:
        left_key, right_key = PAUSE_DETECT_KEYS
        store.create_array(left_key, data=left, chunks=(min(100, n_frames), 7))
        store.create_array(right_key, data=right, chunks=(min(100, n_frames), 7))
    else:
        # Worker's fallback path needs some array to size from.
        store.create_array(
            "obs_head_pose",
            data=np.zeros((n_frames, 7), dtype=np.float64),
            chunks=(min(100, n_frames), 7),
        )

    if include_keypoints:
        # 5 landmarks * 3 coords = 15 features. Move all landmarks in lockstep
        # with the ee_pose so the keypoint AND ee_pose pause votes agree.
        n_landmarks = 5
        left_kp = np.repeat(left[:, :3], n_landmarks, axis=1).reshape(
            n_frames, n_landmarks * 3
        )
        right_kp = np.repeat(right[:, :3], n_landmarks, axis=1).reshape(
            n_frames, n_landmarks * 3
        )
        left_kp_key, right_kp_key = PAUSE_DETECT_KEYPOINT_KEYS
        store.create_array(
            left_kp_key, data=left_kp, chunks=(min(100, n_frames), n_landmarks * 3)
        )
        store.create_array(
            right_kp_key, data=right_kp, chunks=(min(100, n_frames), n_landmarks * 3)
        )

    return left, right, left_kp, right_kp


def _expected_entry(
    left: np.ndarray,
    right: np.ndarray,
    epsilon: float,
    left_kp: np.ndarray | None = None,
    right_kp: np.ndarray | None = None,
) -> dict:
    """Reference cache entry computed via the canonical helper."""
    keep = _build_pause_keep_mask(
        left_pose=left,
        right_pose=right,
        epsilon=epsilon,
        left_keypoints=left_kp,
        right_keypoints=right_kp,
    )
    return {
        "raw_total": int(left.shape[0]),
        "keep_indices": np.flatnonzero(keep).astype(np.int64).tolist(),
    }


def test_worker_keep_mask_matches_canonical_helper(tmp_path):
    """End-to-end: worker on a single shard matches ``_build_pause_keep_mask``."""
    ep_path = tmp_path / "ep_simple"
    left, right, _, _ = _write_episode(ep_path, n_frames=30, pause_spans=[(10, 20)])
    _, result = _pause_precompute_shard.local(0, 0.005, [("hash_simple", str(ep_path))])
    expected = _expected_entry(left, right, epsilon=0.005)
    assert result == {"hash_simple": expected}
    # Sanity: pause span dropped frames 11..19 (first pause frame kept as
    # transition; subsequent in-pause frames dropped until motion resumes).
    kept = set(result["hash_simple"]["keep_indices"])
    for t in range(11, 20):
        assert t not in kept, f"frame {t} should be dropped (inside pause span)"
    assert 10 in kept, "first pause frame is the transition — must be kept"
    assert 20 in kept, "first post-pause frame must be kept (motion resumes)"


def test_worker_with_keypoints_matches_canonical_helper(tmp_path):
    """Keypoint-aware path: keypoints moving in lockstep with ee_pose →
    keep-mask identical to ee_pose-only result."""
    ep_path = tmp_path / "ep_kp"
    left, right, left_kp, right_kp = _write_episode(
        ep_path, n_frames=40, pause_spans=[(5, 15), (25, 35)], include_keypoints=True
    )
    _, result = _pause_precompute_shard.local(0, 0.005, [("hash_kp", str(ep_path))])
    expected = _expected_entry(
        left, right, epsilon=0.005, left_kp=left_kp, right_kp=right_kp
    )
    assert result == {"hash_kp": expected}


def test_worker_keypoint_veto_blocks_pause(tmp_path):
    """If hand keypoints flex while wrist is stationary, frame is NOT a pause.

    Build an episode where ee_pose is locked (would be classified as paused
    by ee_pose-only filter), but keypoints have non-trivial deltas — keep-
    mask should keep all frames.
    """
    ep_path = tmp_path / "ep_veto"
    n = 20
    # Locked wrist
    left = np.zeros((n, 7), dtype=np.float64)
    right = np.zeros((n, 7), dtype=np.float64)
    left[:, 6] = right[:, 6] = 1.0
    # Wiggling fingers — 5 landmarks * 3 coords. delta > 0.005 per frame.
    n_landmarks = 5
    base = np.arange(n).reshape(-1, 1) * 0.1
    left_kp = np.tile(base, (1, n_landmarks * 3))
    right_kp = np.tile(base, (1, n_landmarks * 3))
    store = zarr.open(str(ep_path), mode="w", zarr_format=3)
    left_key, right_key = PAUSE_DETECT_KEYS
    left_kp_key, right_kp_key = PAUSE_DETECT_KEYPOINT_KEYS
    store.create_array(left_key, data=left, chunks=(n, 7))
    store.create_array(right_key, data=right, chunks=(n, 7))
    store.create_array(left_kp_key, data=left_kp, chunks=(n, n_landmarks * 3))
    store.create_array(right_kp_key, data=right_kp, chunks=(n, n_landmarks * 3))

    _, result = _pause_precompute_shard.local(0, 0.005, [("hash_veto", str(ep_path))])
    entry = result["hash_veto"]
    assert entry["raw_total"] == n
    # All frames kept because keypoint motion vetoes the pause vote.
    assert entry["keep_indices"] == list(range(n))


def test_worker_missing_ee_pose_returns_all_frames(tmp_path):
    """Fallback: episode without ee_pose keys → keep ALL frames (matches the
    in-process ``precompute_pause_filter`` behavior, treated as 100% kept)."""
    ep_path = tmp_path / "ep_no_ee"
    _ = _write_episode(ep_path, n_frames=12, pause_spans=[], omit_ee_pose=True)
    _, result = _pause_precompute_shard.local(0, 0.005, [("hash_no_ee", str(ep_path))])
    entry = result["hash_no_ee"]
    assert entry["raw_total"] == 12
    assert entry["keep_indices"] == list(range(12))


def test_worker_unreadable_episode_collapses_to_miss(tmp_path):
    """Worker must not crash on a bad path — returns ``raw_total=0`` so the
    cache consumer surfaces a clear cache-miss error at training time."""
    bogus = tmp_path / "does_not_exist"
    _, result = _pause_precompute_shard.local(0, 0.005, [("hash_missing", str(bogus))])
    assert result == {"hash_missing": {"raw_total": 0, "keep_indices": []}}


def test_worker_handles_mixed_shard(tmp_path):
    """Single shard with a mix of good + bad episodes returns all entries
    with appropriate raw_total values (0 means miss)."""
    good_path = tmp_path / "good"
    left, right, _, _ = _write_episode(good_path, n_frames=20, pause_spans=[(5, 12)])
    bad_path = tmp_path / "bad"

    episodes = [
        ("good_hash", str(good_path)),
        ("bad_hash", str(bad_path)),
    ]
    shard_id, result = _pause_precompute_shard.local(7, 0.005, episodes)

    assert shard_id == 7
    assert set(result) == {"good_hash", "bad_hash"}
    assert result["bad_hash"] == {"raw_total": 0, "keep_indices": []}
    assert result["good_hash"] == _expected_entry(left, right, epsilon=0.005)


def test_worker_short_episode_keeps_all(tmp_path):
    """T<2 corner case: only one frame → no deltas to evaluate → keep it."""
    ep_path = tmp_path / "ep_short"
    n = 1
    left = np.zeros((n, 7), dtype=np.float64)
    right = np.zeros((n, 7), dtype=np.float64)
    left[:, 6] = right[:, 6] = 1.0
    store = zarr.open(str(ep_path), mode="w", zarr_format=3)
    left_key, right_key = PAUSE_DETECT_KEYS
    store.create_array(left_key, data=left, chunks=(n, 7))
    store.create_array(right_key, data=right, chunks=(n, 7))
    _, result = _pause_precompute_shard.local(0, 0.005, [("hash_short", str(ep_path))])
    assert result == {"hash_short": {"raw_total": 1, "keep_indices": [0]}}


# ---------------------------------------------------------------------------
# Smoke: epsilon=0 means every step is paused → all-but-first frame dropped
# ---------------------------------------------------------------------------


def test_worker_epsilon_zero_drops_all_repeated_frames(tmp_path):
    """With epsilon=0 and identical frames, every step satisfies
    delta < epsilon=False (since 0 < 0 is False), so nothing is paused.
    But with the SLIGHTEST motion (1e-9) and epsilon=1e-3, all frames
    register as paused — keep only frame 0 and the transition (frame 1)."""
    ep_path = tmp_path / "ep_tiny_motion"
    n = 6
    left = np.zeros((n, 7), dtype=np.float64)
    right = np.zeros((n, 7), dtype=np.float64)
    left[:, 6] = right[:, 6] = 1.0
    left[:, 0] = np.arange(n) * 1e-9
    right[:, 0] = np.arange(n) * 1e-9
    store = zarr.open(str(ep_path), mode="w", zarr_format=3)
    left_key, right_key = PAUSE_DETECT_KEYS
    store.create_array(left_key, data=left, chunks=(n, 7))
    store.create_array(right_key, data=right, chunks=(n, 7))

    _, result = _pause_precompute_shard.local(0, 1e-3, [("hash_tiny", str(ep_path))])
    expected = _expected_entry(left, right, epsilon=1e-3)
    assert result["hash_tiny"] == expected
    # Algorithm keeps frame 0 (pre-pause) and frame 1 (pause transition).
    assert result["hash_tiny"]["keep_indices"] == [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
