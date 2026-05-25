"""Tests for split-aware output routing in EvalVideo.

Covers:
  - Directory layout: single-loader keeps legacy paths; multi-loader gets
    split-specific subdirs.
  - Buffer keying: (dataloader_idx, embodiment_id) ensures train and eval
    frames never share a buffer.
  - Metric relabeling: idx 0 unchanged; idx 1 gets a /train_eval suffix.
  - on_validation_start: creates the right epoch dirs for each configuration.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from egomimic.eval.eval_video import EvalVideo, _split_name


class _StubEval(EvalVideo):
    """Minimal concrete subclass for instantiation."""

    def compute_metrics_and_viz(self, batch):
        return {}, {}


def _make_evaluator(*, val_dataloaders, current_epoch: int = 3, root_dir: str = "/tmp/r"):
    ev = _StubEval()
    ev.trainer = SimpleNamespace(
        default_root_dir=root_dir,
        current_epoch=current_epoch,
        val_dataloaders=val_dataloaders,
        is_global_zero=True,
        lightning_module=SimpleNamespace(device="cpu", log_dict=lambda *a, **kw: None),
    )
    return ev


# ------------------------------------------------------------------
# _split_name
# ------------------------------------------------------------------


def test_split_name_known_indices():
    assert _split_name(0) == "eval"
    assert _split_name(1) == "train_eval"
    assert _split_name(2) == "loader_2"


# ------------------------------------------------------------------
# Single-loader: legacy layout unchanged
# ------------------------------------------------------------------


def test_single_loader_legacy_layout(tmp_path):
    ev = _make_evaluator(val_dataloaders=object(), root_dir=str(tmp_path))
    with patch("egomimic.eval.eval_video.get_embodiment", lambda eid: f"emb{eid}"):
        assert ev._is_multi_loader() is False
        assert ev._split_subdir(0) == ""
        assert ev._video_subdir(0, 9) == str(tmp_path / "videos/epoch_3/emb9")
        # Metric names untouched for either idx in single-loader mode.
        assert ev._relabel_metrics({"Valid/foo": 1.0}, 0) == {"Valid/foo": 1.0}
        assert ev._relabel_metrics({"Valid/foo": 1.0}, 1) == {"Valid/foo": 1.0}


# ------------------------------------------------------------------
# Multi-loader: split-aware paths and metric suffixes
# ------------------------------------------------------------------


def test_multi_loader_routes_by_dataloader_idx(tmp_path):
    ev = _make_evaluator(val_dataloaders=[object(), object()], root_dir=str(tmp_path))
    with patch("egomimic.eval.eval_video.get_embodiment", lambda eid: f"emb{eid}"):
        assert ev._is_multi_loader() is True
        assert ev._split_subdir(0) == "eval"
        assert ev._split_subdir(1) == "train_eval"
        assert ev._video_subdir(0, 9) == str(tmp_path / "videos/eval/epoch_3/emb9")
        assert ev._video_subdir(1, 9) == str(tmp_path / "videos/train_eval/epoch_3/emb9")
        # idx 0 keeps original keys (backward compat).
        assert ev._relabel_metrics({"Valid/foo": 1.0}, 0) == {"Valid/foo": 1.0}
        # idx 1 gets a clear suffix.
        assert ev._relabel_metrics({"Valid/foo": 1.0}, 1) == {"Valid/foo/train_eval": 1.0}


# ------------------------------------------------------------------
# Buffer keys are (dataloader_idx, embodiment_id)
# ------------------------------------------------------------------


def test_buffer_keys_are_split_aware():
    ev = _make_evaluator(val_dataloaders=[object(), object()])
    assert ev.val_image_buffer == {}
    ev.val_image_buffer[(0, 9)] = ["eval_frame"]
    ev.val_image_buffer[(1, 9)] = ["train_frame"]
    # Same embodiment_id under different splits is a distinct buffer.
    assert ev.val_image_buffer[(0, 9)] != ev.val_image_buffer[(1, 9)]


# ------------------------------------------------------------------
# on_validation_start creates the right epoch dirs
# ------------------------------------------------------------------


def test_on_validation_start_single_loader(tmp_path):
    ev = _make_evaluator(val_dataloaders=object(), root_dir=str(tmp_path))
    ev.on_validation_start()
    assert (tmp_path / "videos/epoch_3").is_dir()


def test_on_validation_start_multi_loader(tmp_path):
    ev = _make_evaluator(val_dataloaders=[object(), object()], root_dir=str(tmp_path))
    ev.on_validation_start()
    assert (tmp_path / "videos/eval/epoch_3").is_dir()
    assert (tmp_path / "videos/train_eval/epoch_3").is_dir()
