"""Tests for the split-aware output routing in EvalVideo.

These are pure helper-level tests (no Lightning trainer) — the goal is to
lock in the directory + buffer-key + metric-naming behavior so train-split
videos can never accidentally tail-flush into the eval video and vice
versa, and so single-loader runs keep their legacy layout.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from egomimic.eval.eval_video import EvalVideo, _split_name


class _Stub(EvalVideo):
    """Subclass providing the abstract method so we can instantiate."""

    def compute_metrics_and_viz(self, batch):
        return {}, {}


def _make_evaluator(
    *, val_dataloaders, current_epoch: int = 3, root_dir: str = "/tmp/r"
):
    ev = _Stub()
    ev.trainer = SimpleNamespace(
        default_root_dir=root_dir,
        current_epoch=current_epoch,
        val_dataloaders=val_dataloaders,
        is_global_zero=True,
        lightning_module=SimpleNamespace(device="cpu", log_dict=lambda *a, **kw: None),
    )
    return ev


def test_split_name_known_indices():
    assert _split_name(0) == "eval"
    assert _split_name(1) == "train_split"
    assert _split_name(2) == "loader_2"


def test_single_loader_keeps_legacy_layout(tmp_path):
    """One CombinedLoader (not a list) → behavior identical to pre-split code."""
    ev = _make_evaluator(val_dataloaders=object(), root_dir=str(tmp_path))
    # Pretend embodiment id resolves to a stable string.
    with patch("egomimic.eval.eval_video.get_embodiment", lambda eid: f"emb{eid}"):
        assert ev._is_multi_loader() is False
        assert ev._split_subdir(0) == ""
        # Legacy layout: videos/epoch_3/emb9/...
        assert ev._video_subdir(0, 9) == str(tmp_path / "videos/epoch_3/emb9")
        # Metric names unchanged regardless of idx.
        assert ev._maybe_relabel_metrics({"Valid/foo": 1.0}, 0) == {"Valid/foo": 1.0}
        assert ev._maybe_relabel_metrics({"Valid/foo": 1.0}, 1) == {"Valid/foo": 1.0}


def test_multi_loader_routes_by_dataloader_idx(tmp_path):
    """List of 2 dataloaders → split-aware paths and metric suffixes."""
    ev = _make_evaluator(val_dataloaders=[object(), object()], root_dir=str(tmp_path))
    with patch("egomimic.eval.eval_video.get_embodiment", lambda eid: f"emb{eid}"):
        assert ev._is_multi_loader() is True
        assert ev._split_subdir(0) == "eval"
        assert ev._split_subdir(1) == "train_split"
        assert ev._video_subdir(0, 9) == str(tmp_path / "videos/eval/epoch_3/emb9")
        assert ev._video_subdir(1, 9) == str(
            tmp_path / "videos/train_split/epoch_3/emb9"
        )
        # Idx 0 metrics keep original keys (backward compat for dashboards).
        assert ev._maybe_relabel_metrics({"Valid/foo": 1.0}, 0) == {"Valid/foo": 1.0}
        # Idx 1 (train split) gets a clear suffix.
        assert ev._maybe_relabel_metrics({"Valid/foo": 1.0}, 1) == {
            "Valid/foo/train_split": 1.0
        }


def test_buffer_keys_are_dataloader_aware():
    """Buffers must be keyed (dataloader_idx, embodiment_id) so train+eval don't mix."""
    ev = _make_evaluator(val_dataloaders=[object(), object()])
    # Simulate the typing/initialization done in __init__.
    assert ev.val_image_buffer == {}
    assert ev.val_counter == {}
    # Manual insert mirroring on_validation_step's path:
    ev.val_image_buffer[(0, 9)] = ["frame_eval"]
    ev.val_image_buffer[(1, 9)] = ["frame_train"]
    assert ev.val_image_buffer[(0, 9)] == ["frame_eval"]
    assert ev.val_image_buffer[(1, 9)] == ["frame_train"]
    # Same embodiment_id under different splits is a distinct buffer.
    assert (0, 9) != (1, 9)


def test_on_validation_start_creates_per_split_dirs(tmp_path):
    """Single loader → one epoch dir. Two loaders → two parallel dirs."""
    # Single loader
    ev_single = _make_evaluator(val_dataloaders=object(), root_dir=str(tmp_path))
    ev_single.on_validation_start()
    assert (tmp_path / "videos/epoch_3").is_dir()

    # Multi loader
    tmp_path_multi = tmp_path / "multi"
    tmp_path_multi.mkdir()
    ev_multi = _make_evaluator(
        val_dataloaders=[object(), object()], root_dir=str(tmp_path_multi)
    )
    ev_multi.on_validation_start()
    assert (tmp_path_multi / "videos/eval/epoch_3").is_dir()
    assert (tmp_path_multi / "videos/train_split/epoch_3").is_dir()
