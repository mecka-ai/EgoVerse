"""Tests for the episode-level pause/idle precompute."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from egomimic.rldb.zarr.zarr_dataset_multi import (
    PAUSE_DETECT_KEYS,
    PAUSE_PRECOMPUTE_CACHE_ENV,
    LocalEpisodeResolver,
    ZarrDataset,
    _build_pause_keep_mask,
    _keypoint_max_delta,
)


def _reference_compress_keep_mask(chunk: np.ndarray, epsilon: float) -> np.ndarray:
    H = len(chunk)
    if H < 2:
        return np.ones(H, dtype=bool)
    deltas = np.linalg.norm(np.diff(chunk, axis=0), axis=-1)
    keep = np.ones(H, dtype=bool)
    in_pause = False
    for t in range(1, H):
        if deltas[t - 1] < epsilon:
            if in_pause:
                keep[t] = False
            else:
                in_pause = True
        else:
            in_pause = False
    return keep


def _synthetic_poses(
    *,
    n_frames: int,
    pause_spans: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build left/right obs_ee_pose arrays where frames inside any pause span
    are exactly identical (zero delta) and frames outside step by 0.1 in x.
    """
    left = np.zeros((n_frames, 7), dtype=np.float64)
    right = np.zeros((n_frames, 7), dtype=np.float64)
    left[:, 6] = 1.0  # quat w = 1
    right[:, 6] = 1.0
    motion = 0.0
    for i in range(n_frames):
        in_pause = any(s <= i < e for s, e in pause_spans)
        if not in_pause:
            motion += 0.1
        left[i, 0] = motion
        right[i, 0] = motion + 0.5  # offset right hand
    return left, right


def _synthetic_keypoints(
    *,
    n_frames: int,
    wiggle_spans: list[tuple[int, int]] | None = None,
    n_landmarks: int = 21,
) -> np.ndarray:
    """Build a flat (T, K*3) keypoint trajectory.

    Inside each ``wiggle_spans`` interval, ONE landmark wiggles 1 mm per
    frame (way above any reasonable epsilon). Outside the wiggles every
    landmark is exactly identical — so a "no wiggle" frame is paused-y on
    the keypoint signal too.
    """
    wiggle_spans = wiggle_spans or []
    kp = np.zeros((n_frames, n_landmarks, 3), dtype=np.float64)
    for s, e in wiggle_spans:
        for i in range(max(s, 0), min(e, n_frames)):
            kp[i, 0, 0] = (i - s + 1) * 0.001  # ramp one landmark by 1 mm/frame
    return kp.reshape(n_frames, n_landmarks * 3)


def _write_synthetic_mecka_episode(
    path: Path,
    *,
    n_frames: int,
    pause_spans: list[tuple[int, int]],
    left_kp_wiggle_spans: list[tuple[int, int]] | None = None,
    right_kp_wiggle_spans: list[tuple[int, int]] | None = None,
    write_keypoints: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Write a minimal zarr v3 group at path with the keys ZarrDataset reads.

    When ``write_keypoints=True``, also emits left.obs_keypoints /
    right.obs_keypoints arrays controlled by the ``*_wiggle_spans`` args.
    """
    left, right = _synthetic_poses(n_frames=n_frames, pause_spans=pause_spans)

    head = np.zeros((n_frames, 7), dtype=np.float64)
    head[:, 6] = 1.0

    store = zarr.open(str(path), mode="w", zarr_format=3)
    store.create_array("left.obs_ee_pose", data=left, chunks=(min(100, n_frames), 7))
    store.create_array("right.obs_ee_pose", data=right, chunks=(min(100, n_frames), 7))
    store.create_array("obs_head_pose", data=head, chunks=(min(100, n_frames), 7))

    features = {
        "left.obs_ee_pose": {"dtype": "float64", "shape": [7], "names": ["dim_0"]},
        "right.obs_ee_pose": {"dtype": "float64", "shape": [7], "names": ["dim_0"]},
        "obs_head_pose": {"dtype": "float64", "shape": [7], "names": ["dim_0"]},
    }

    if write_keypoints:
        left_kp = _synthetic_keypoints(
            n_frames=n_frames, wiggle_spans=left_kp_wiggle_spans
        )
        right_kp = _synthetic_keypoints(
            n_frames=n_frames, wiggle_spans=right_kp_wiggle_spans
        )
        store.create_array(
            "left.obs_keypoints", data=left_kp, chunks=(min(100, n_frames), 63)
        )
        store.create_array(
            "right.obs_keypoints", data=right_kp, chunks=(min(100, n_frames), 63)
        )
        features["left.obs_keypoints"] = {
            "dtype": "float64",
            "shape": [63],
            "names": ["dim_0"],
        }
        features["right.obs_keypoints"] = {
            "dtype": "float64",
            "shape": [63],
            "names": ["dim_0"],
        }

    store.attrs.update(
        {
            "embodiment": "MECKA_BIMANUAL",
            "total_frames": n_frames,
            "fps": 30,
            "features": features,
        }
    )
    return left, right


@pytest.mark.parametrize(
    "pause_spans,n,expected_dropped",
    [
        ([], 10, 0),
        ([(3, 8)], 10, 4),
        ([(0, 5)], 10, 3),
        ([(2, 4), (6, 9)], 10, 3),
    ],
)
def test_build_pause_keep_mask_matches_reference(pause_spans, n, expected_dropped):
    left, right = _synthetic_poses(n_frames=n, pause_spans=pause_spans)
    keep = _build_pause_keep_mask(left_pose=left, right_pose=right, epsilon=0.005)

    expected_left = _reference_compress_keep_mask(left, 0.005)
    expected_right = _reference_compress_keep_mask(right, 0.005)
    assert np.array_equal(keep, expected_left)
    assert np.array_equal(keep, expected_right)

    assert int((~keep).sum()) == expected_dropped


def test_build_pause_keep_mask_short_episode():
    for n in (0, 1):
        left = np.zeros((n, 7))
        right = np.zeros((n, 7))
        keep = _build_pause_keep_mask(left_pose=left, right_pose=right, epsilon=1.0)
        assert len(keep) == n
        assert keep.all()


def test_zarr_dataset_precompute_alters_length(tmp_path):
    ep = tmp_path / "ep_test.zarr"
    n_frames = 30
    pause_spans = [(5, 12), (20, 25)]  # 7-frame pause and 5-frame pause
    _write_synthetic_mecka_episode(ep, n_frames=n_frames, pause_spans=pause_spans)

    key_map = {
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
        },
        "right.action_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": "right.obs_ee_pose",
            "horizon": 5,
        },
    }

    ds_off = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=None)
    assert len(ds_off) == n_frames
    assert ds_off.keep_indices is None

    ds_on = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    assert len(ds_on) == n_frames

    n_raw, n_kept = ds_on.precompute_pause_filter()
    assert n_raw == n_frames
    expected_keep = _build_pause_keep_mask(
        left_pose=ds_on.episode_reader._store[PAUSE_DETECT_KEYS[0]][:],
        right_pose=ds_on.episode_reader._store[PAUSE_DETECT_KEYS[1]][:],
        epsilon=0.005,
    )
    assert n_kept == int(expected_keep.sum())
    assert len(ds_on) == n_kept
    assert n_kept < n_frames


def test_zarr_dataset_getitem_uses_keep_indices(tmp_path):
    ep = tmp_path / "ep_idx.zarr"
    n_frames = 20
    pause_spans = [(4, 10)]
    _write_synthetic_mecka_episode(ep, n_frames=n_frames, pause_spans=pause_spans)

    key_map = {
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    ds.precompute_pause_filter()

    keep = ds.keep_indices
    assert keep is not None
    assert int(keep[0]) == 0
    assert int(keep[-1]) < n_frames

    sample_0 = ds[0]
    raw_left = ds.episode_reader._store["left.obs_ee_pose"][:]
    np.testing.assert_allclose(
        sample_0["left.obs_ee_pose"].numpy(), raw_left[int(keep[0])]
    )

    mid = len(keep) // 2
    sample_mid = ds[mid]
    np.testing.assert_allclose(
        sample_mid["left.obs_ee_pose"].numpy(), raw_left[int(keep[mid])]
    )


def test_zarr_dataset_precompute_is_idempotent(tmp_path):
    ep = tmp_path / "ep_idem.zarr"
    _write_synthetic_mecka_episode(ep, n_frames=15, pause_spans=[(5, 10)])
    key_map = {
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    a = ds.precompute_pause_filter()
    b = ds.precompute_pause_filter()
    assert a == b
    assert ds.keep_indices is not None


def test_run_pause_precompute_runs_in_process(tmp_path):
    ep = tmp_path / "ep_inprocess.zarr"
    _write_synthetic_mecka_episode(ep, n_frames=20, pause_spans=[(4, 10)])
    key_map = {
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        key_map=key_map,
        pause_removal_epsilon=0.005,
    )
    resolver._run_pause_precompute({"ep_inprocess": ds})
    assert ds.keep_indices is not None
    assert len(ds.keep_indices) < 20


def _make_dataset(tmp_path, *, name="ep", n_frames=20, pause_spans=None):
    ep = tmp_path / f"{name}.zarr"
    _write_synthetic_mecka_episode(
        ep, n_frames=n_frames, pause_spans=pause_spans or [(4, 10)]
    )
    key_map = {
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    resolver = LocalEpisodeResolver(
        folder_path=tmp_path,
        key_map=key_map,
        pause_removal_epsilon=0.005,
    )
    return resolver, ds


def test_run_pause_precompute_consumes_cache_when_env_set(tmp_path, monkeypatch):
    """Resolver populates keep_indices from the JSON cache when the env var points at it."""
    resolver, ds = _make_dataset(tmp_path, name="ep_cache")

    # Synthesize a cache: drop the first 5 frames.
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "ep_cache": {
                    "raw_total": 20,
                    "keep_indices": list(range(5, 20)),
                }
            }
        )
    )
    monkeypatch.setenv(PAUSE_PRECOMPUTE_CACHE_ENV, str(cache_path))

    resolver._run_pause_precompute({"ep_cache": ds})

    assert ds._raw_total_frames == 20
    assert ds.keep_indices is not None
    assert ds.keep_indices.tolist() == list(range(5, 20))


def test_run_pause_precompute_cache_miss_falls_back_in_process(tmp_path, monkeypatch):
    """An episode missing from the cache still gets processed locally."""
    resolver, ds = _make_dataset(tmp_path, name="ep_miss")

    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps({"some_other_hash": {"raw_total": 0, "keep_indices": []}})
    )
    monkeypatch.setenv(PAUSE_PRECOMPUTE_CACHE_ENV, str(cache_path))

    resolver._run_pause_precompute({"ep_miss": ds})

    assert ds.keep_indices is not None
    assert len(ds.keep_indices) < 20  # in-process precompute trimmed pauses


def test_run_pause_precompute_cache_with_raw_total_zero_falls_back(
    tmp_path, monkeypatch
):
    """raw_total==0 in the cache means the worker hit an error; fall back per-episode."""
    resolver, ds = _make_dataset(tmp_path, name="ep_err")

    cache_path = tmp_path / "cache.json"
    cache_path.write_text(json.dumps({"ep_err": {"raw_total": 0, "keep_indices": []}}))
    monkeypatch.setenv(PAUSE_PRECOMPUTE_CACHE_ENV, str(cache_path))

    resolver._run_pause_precompute({"ep_err": ds})

    assert ds.keep_indices is not None
    assert len(ds.keep_indices) < 20


def test_run_pause_precompute_no_env_var_runs_in_process(tmp_path, monkeypatch):
    """With no cache env var set, everything goes through the in-process path."""
    monkeypatch.delenv(PAUSE_PRECOMPUTE_CACHE_ENV, raising=False)
    resolver, ds = _make_dataset(tmp_path, name="ep_noenv")

    resolver._run_pause_precompute({"ep_noenv": ds})

    assert ds.keep_indices is not None
    assert len(ds.keep_indices) < 20


def test_pause_precompute_shard_is_registered_on_training_app():
    """Worker still lives on the training app for trainModal to call .map() against."""
    pytest.importorskip("modal")
    from egomimic.modal import modal_setup

    assert hasattr(modal_setup, "pause_precompute_shard")
    assert modal_setup.pause_precompute_shard.app is modal_setup.app


# ---------------------------------------------------------------------------
# Keypoint signal
# ---------------------------------------------------------------------------


def test_keypoint_max_delta_picks_largest_landmark_motion():
    """A flat (T, K*3) array where one landmark wiggles → that one drives the result."""
    T, K = 5, 21
    kp = np.zeros((T, K, 3), dtype=np.float64)
    # Wiggle landmark 7 by 0.01m per frame; everything else stationary.
    kp[:, 7, 0] = np.arange(T) * 0.01
    d = _keypoint_max_delta(kp.reshape(T, K * 3))
    assert d.shape == (T - 1,)
    np.testing.assert_allclose(d, 0.01)


def test_keypoint_max_delta_handles_short_episodes():
    assert _keypoint_max_delta(np.zeros((0, 63))).shape == (0,)
    assert _keypoint_max_delta(np.zeros((1, 63))).shape == (0,)


def test_build_pause_keep_mask_drops_when_wrist_and_keypoints_both_still():
    """Wrist stationary AND keypoints stationary → pause."""
    n = 10
    pose_zero = np.tile(np.array([0, 0, 0, 0, 0, 0, 1.0]), (n, 1))
    kp_zero = np.zeros((n, 63))
    keep = _build_pause_keep_mask(
        left_pose=pose_zero,
        right_pose=pose_zero,
        epsilon=0.005,
        left_keypoints=kp_zero,
        right_keypoints=kp_zero,
    )
    # Frame 0 always kept; frame 1 is the transition (first paused frame);
    # frames 2.. are dropped.
    assert keep[0] is np.True_ or bool(keep[0])
    assert bool(keep[1])
    assert not bool(keep[5])
    assert (~keep).sum() == n - 2


def test_build_pause_keep_mask_keeps_frame_when_only_keypoints_move():
    """Wrist stationary but a hand keypoint moves → motion, keep the frame."""
    n = 10
    pose_zero = np.tile(np.array([0, 0, 0, 0, 0, 0, 1.0]), (n, 1))

    # Left hand has a single landmark moving 1 cm/frame; right hand idle.
    moving_left = _synthetic_keypoints(
        n_frames=n, wiggle_spans=[(0, n)], n_landmarks=21
    )
    # Reset to 1cm/frame so it's clearly > 0.005 m
    moving_left = (moving_left / 0.001) * 0.01  # rescale from 1mm to 1cm
    still_kp = np.zeros((n, 63))

    keep = _build_pause_keep_mask(
        left_pose=pose_zero,
        right_pose=pose_zero,
        epsilon=0.005,
        left_keypoints=moving_left,
        right_keypoints=still_kp,
    )
    # No pause anywhere — keypoints disagree with "wrist quiet"
    assert keep.all()


def test_build_pause_keep_mask_falls_back_when_keypoints_missing():
    """No keypoint arrays supplied → reduces to ee_pose-only detection."""
    n = 10
    pose_zero = np.tile(np.array([0, 0, 0, 0, 0, 0, 1.0]), (n, 1))
    keep_with_kp = _build_pause_keep_mask(
        left_pose=pose_zero,
        right_pose=pose_zero,
        epsilon=0.005,
        left_keypoints=None,
        right_keypoints=None,
    )
    # With no keypoint signal, behavior matches the original pose-only path.
    expected = _build_pause_keep_mask(
        left_pose=pose_zero, right_pose=pose_zero, epsilon=0.005
    )
    np.testing.assert_array_equal(keep_with_kp, expected)


def test_precompute_uses_keypoints_when_present(tmp_path):
    """Episode with stationary wrists but moving keypoints should keep all frames."""
    ep = tmp_path / "ep_kp_motion.zarr"
    n = 15
    # Wrist stationary throughout (empty pause_spans means ee_pose ramps,
    # which means ee_pose is *moving* — set pause_spans to whole episode so
    # ee_pose is fully stationary).
    _write_synthetic_mecka_episode(
        ep,
        n_frames=n,
        pause_spans=[(0, n)],
        write_keypoints=True,
        left_kp_wiggle_spans=[(0, n)],  # left fingers active for all frames
        right_kp_wiggle_spans=None,  # right keypoints idle
    )
    # ee_pose is stationary AND right keypoints are stationary, BUT left
    # keypoints are wiggling at 1mm/frame. epsilon=0.0005 catches them.
    key_map = {
        "left.obs_ee_pose": {
            "key_type": "proprio_keys",
            "zarr_key": "left.obs_ee_pose",
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.0005)
    n_raw, n_kept = ds.precompute_pause_filter()
    assert n_raw == n
    # Frame 0 always kept; frames 1..n-1 all have left-kp motion > eps so
    # NONE qualify as pause-steps → no drops.
    assert n_kept == n


# ---------------------------------------------------------------------------
# Fully-filtered action chunks
# ---------------------------------------------------------------------------


def test_horizon_read_walks_through_keep_indices(tmp_path):
    """When keep_indices is set, the action chunk is H consecutive *filtered* frames."""
    ep = tmp_path / "ep_chunk.zarr"
    n = 20
    pause_spans = [(3, 8)]  # raw frames 3..7 paused (idx 3 kept as edge, 4..7 dropped)
    _write_synthetic_mecka_episode(ep, n_frames=n, pause_spans=pause_spans)

    key_map = {
        "left.action_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": "left.obs_ee_pose",
            "horizon": 5,
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    ds.precompute_pause_filter()

    keep = ds.keep_indices
    assert keep is not None
    # Sample filtered-idx 0; chunk = keep[0:5] = first 5 surviving raw indices.
    sample = ds[0]
    chunk = sample["left.action_ee_pose"].numpy()
    raw = ds.episode_reader._store["left.obs_ee_pose"][:]
    expected = raw[keep[0:5]]
    np.testing.assert_allclose(chunk, expected)
    # And critically: the chunk's raw indices skip the dropped pause region.
    assert set(int(i) for i in keep[0:5]).isdisjoint({4, 5, 6, 7})


def test_horizon_read_unchanged_when_filter_disabled(tmp_path):
    """With pause_removal_epsilon=None the chunk is still the contiguous raw slice."""
    ep = tmp_path / "ep_nofilter.zarr"
    n = 20
    _write_synthetic_mecka_episode(ep, n_frames=n, pause_spans=[(3, 8)])
    key_map = {
        "left.action_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": "left.obs_ee_pose",
            "horizon": 5,
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=None)
    assert ds.keep_indices is None
    sample = ds[2]
    raw = ds.episode_reader._store["left.obs_ee_pose"][:]
    np.testing.assert_allclose(sample["left.action_ee_pose"].numpy(), raw[2:7])


def test_horizon_read_pads_when_chunk_clipped_near_episode_end(tmp_path):
    """Asking for a horizon that runs past the filtered episode end pads via _pad_sequences."""
    ep = tmp_path / "ep_clip.zarr"
    n = 10
    _write_synthetic_mecka_episode(ep, n_frames=n, pause_spans=[])
    key_map = {
        "left.action_ee_pose": {
            "key_type": "action_keys",
            "zarr_key": "left.obs_ee_pose",
            "horizon": 6,
        },
    }
    ds = ZarrDataset(ep, key_map=key_map, pause_removal_epsilon=0.005)
    ds.precompute_pause_filter()
    # Filtered idx near the end where horizon overruns the episode.
    last_filtered = len(ds) - 1
    sample = ds[last_filtered]
    chunk = sample["left.action_ee_pose"].numpy()
    assert chunk.shape[0] == 6  # padded to horizon length
