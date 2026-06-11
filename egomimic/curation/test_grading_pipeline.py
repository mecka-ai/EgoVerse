"""Tests for the k-NN grading featurize pass and npz cache (no GPU/zarr)."""

from __future__ import annotations

import numpy as np
import pytest

from egomimic.curation.config import CurationLoaderSettings
from egomimic.curation.grading_pipeline import (
    GradingFeaturizeSettings,
    episode_feature_path,
    load_episode_features,
    run_featurize_episodes,
    save_episode_features,
)
from egomimic.curation.knn_grading import KnnGradeSettings, grade_task

KNN_SETTINGS = KnnGradeSettings(n_resample=8, n_speed_bins=4)


class _FakeZarrDataset:
    """Stands in for ZarrDataset.collect_grading_episode."""

    def __init__(self, T: int = 12, fail: bool = False):
        self.T = T
        self.fail = fail
        self._zarr_bulk_cache = None

    def collect_grading_episode(self, **kwargs):
        if self.fail:
            return None
        rng = np.random.default_rng(self.T)
        return {
            "actions": rng.standard_normal((self.T, 10, 12)).astype(np.float32),
            "proprio": rng.standard_normal((self.T, 12)).astype(np.float32),
            "images": rng.uniform(size=(self.T, 3, 32, 32)).astype(np.float32),
            "frame_idx": np.arange(self.T, dtype=np.int64)
            * kwargs.get("frame_stride", 1),
            "n_logical": self.T * kwargs.get("frame_stride", 1),
        }


class _FakeFeaturizer:
    """Stands in for StateEmbedder(project=False): mean-pools image pixels."""

    def embed(self, images: np.ndarray) -> np.ndarray:
        return images.reshape(len(images), 3, -1).mean(axis=-1).astype(np.float32)


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    path = episode_feature_path(tmp_path, "v1", "taskA", "abc123")
    save_episode_features(
        path,
        image_feats=rng.standard_normal((7, 16)).astype(np.float32),
        proprio=rng.standard_normal((7, 12)).astype(np.float32),
        chunks=rng.standard_normal((7, 10, 12)).astype(np.float32),
        frame_idx=np.arange(7, dtype=np.int64) * 6,
        n_logical=42,
        meta={"stride": 6},
    )
    assert path.is_file() and not path.with_name(path.stem + ".tmp.npz").exists()

    ep = load_episode_features(path, "abc123", KNN_SETTINGS)
    assert ep.ep_hash == "abc123"
    assert ep.image_feats.shape == (7, 16) and ep.image_feats.dtype == np.float32
    assert ep.proprio.shape == (7, 12)
    assert ep.chunk_feats.left_path.shape == (7, KNN_SETTINGS.n_resample, 3)
    assert ep.chunk_feats.gripper is None
    assert ep.ep_len == 42
    np.testing.assert_array_equal(ep.frame_idx, np.arange(7) * 6)


def test_run_featurize_episodes_writes_cache_and_skips(tmp_path):
    loader = CurationLoaderSettings(episode_workers=2, frame_queue_maxsize=4)
    settings = GradingFeaturizeSettings(stride=3, feature_version="v9")
    episodes = {
        "ep_a": _FakeZarrDataset(T=10),
        "ep_b": _FakeZarrDataset(T=14),
        "ep_bad": _FakeZarrDataset(fail=True),
    }
    out_dir = tmp_path / "v9" / "taskA"

    manifest = run_featurize_episodes(
        episodes, _FakeFeaturizer(), loader, settings, out_dir
    )
    by_hash = {m["hash"]: m for m in manifest}
    assert (out_dir / "ep_a.npz").is_file() and (out_dir / "ep_b.npz").is_file()
    assert by_hash["ep_bad"].get("skipped") is True
    assert by_hash["ep_a"]["n_states"] == 10

    # Second run: everything already cached, nothing re-embedded.
    manifest2 = run_featurize_episodes(
        {"ep_a": _FakeZarrDataset(T=10), "ep_b": _FakeZarrDataset(T=14)},
        _FakeFeaturizer(),
        loader,
        settings,
        out_dir,
    )
    assert all(m.get("cached") for m in manifest2)

    # Cached episodes feed straight into grading (small task → skipped cleanly).
    eps = [
        load_episode_features(out_dir / f"{h}.npz", h, KNN_SETTINGS)
        for h in ("ep_a", "ep_b")
    ]
    result = grade_task(eps, KNN_SETTINGS)
    assert "skipped" in result["task_summary"]


def test_featurize_settings_from_cfg():
    omegaconf = pytest.importorskip("omegaconf")
    cfg = omegaconf.OmegaConf.create(
        {
            "feature_version": "v3",
            "featurize": {
                "stride": 4,
                "image_key": "observations.images.front_img_1",
                "state_image": {
                    "backbone": "dinov3",
                    "dinov3": {"model_name": "some/model", "dtype": "bfloat16"},
                },
            },
        }
    )
    s = GradingFeaturizeSettings.from_cfg(cfg)
    assert s.stride == 4
    assert s.feature_version == "v3"
    assert s.backbone == "dinov3"
    assert s.dinov3_model_name == "some/model"
    assert s.dinov3_dtype == "bfloat16"
