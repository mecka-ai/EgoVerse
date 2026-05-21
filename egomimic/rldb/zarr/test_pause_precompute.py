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


def _write_synthetic_mecka_episode(
    path: Path,
    *,
    n_frames: int,
    pause_spans: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Write a minimal zarr v3 group at path with the keys ZarrDataset reads."""
    left, right = _synthetic_poses(n_frames=n_frames, pause_spans=pause_spans)

    head = np.zeros((n_frames, 7), dtype=np.float64)
    head[:, 6] = 1.0

    store = zarr.open(str(path), mode="w", zarr_format=3)
    store.create_array("left.obs_ee_pose", data=left, chunks=(min(100, n_frames), 7))
    store.create_array("right.obs_ee_pose", data=right, chunks=(min(100, n_frames), 7))
    store.create_array("obs_head_pose", data=head, chunks=(min(100, n_frames), 7))
    store.attrs.update(
        {
            "embodiment": "MECKA_BIMANUAL",
            "total_frames": n_frames,
            "fps": 30,
            "features": {
                "left.obs_ee_pose": {
                    "dtype": "float64",
                    "shape": [7],
                    "names": ["dim_0"],
                },
                "right.obs_ee_pose": {
                    "dtype": "float64",
                    "shape": [7],
                    "names": ["dim_0"],
                },
                "obs_head_pose": {
                    "dtype": "float64",
                    "shape": [7],
                    "names": ["dim_0"],
                },
            },
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
