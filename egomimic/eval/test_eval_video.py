"""Regression tests for EvalVideo's viz scheduling and DDP rank gating.

These exercise the metrics/viz split without a GPU or a real Trainer: a
``SimpleNamespace`` stands in for ``self.trainer`` so we can drive
``current_epoch``/``is_global_zero`` directly.
"""

import os
from types import SimpleNamespace

import numpy as np
import torch

from egomimic.eval.eval_video import EvalVideo


class _StubEvalVideo(EvalVideo):
    """Minimal concrete EvalVideo: records do_viz and emits one tiny frame set."""

    def __init__(self, root, **kw):
        super().__init__(**kw)
        self._root = root
        self.compute_calls = []

    def root_dir(self):
        return self._root

    def compute_metrics_and_viz(self, batch, do_viz=True):
        self.compute_calls.append(do_viz)
        metrics = {"loss": torch.tensor(1.0)}
        images_dict = {}
        if do_viz:
            images_dict = {"mecka_bimanual": np.zeros((4, 8, 8, 3), dtype=np.uint8)}
        return metrics, images_dict


def _fake_trainer(current_epoch=0, is_global_zero=True):
    lm = SimpleNamespace(device="cpu", log_dict=lambda *a, **k: None)
    return SimpleNamespace(
        current_epoch=current_epoch,
        is_global_zero=is_global_zero,
        lightning_module=lm,
    )


def _rendered_epochs(viz_every_n, check_val, max_epochs):
    """Epochs at which a video is actually written: Lightning runs validation
    (``(e + 1) % check_val == 0``) AND ``_should_viz()`` fires."""
    ev = _StubEvalVideo("/tmp", viz_every_n_epochs=viz_every_n)
    out = []
    for e in range(max_epochs):
        if (e + 1) % check_val != 0:
            continue  # Lightning does not run validation this epoch
        ev.trainer = _fake_trainer(current_epoch=e)
        if ev._should_viz():
            out.append(e)
    return out


# --- viz / validation alignment ------------------------------------------------


def test_viz_aligns_with_validation_trigger():
    # The canonical production case (viz_every_n=200, check_val=20) that
    # previously rendered ZERO videos: viz must land on validation-trigger
    # epochs (e where (e+1)%20==0), i.e. 199 and 399, not 200/400.
    assert _rendered_epochs(200, 20, 400) == [199, 399]


def test_viz_every_epoch_when_check_val_1():
    # Matches the L40S smoke test: videos at epoch_0 and epoch_1.
    assert _rendered_epochs(1, 1, 4) == [0, 1, 2, 3]


def test_viz_multiple_of_check_val():
    assert _rendered_epochs(2, 1, 4) == [1, 3]
    assert _rendered_epochs(40, 20, 200) == [39, 79, 119, 159, 199]


def test_viz_disabled_for_nonpositive():
    assert _rendered_epochs(0, 1, 4) == []


def test_old_buggy_gate_rendered_nothing():
    # Documents the regression this fix closes: the pre-fix gate
    # ``current_epoch % N == 0`` never coincides with the validation-trigger
    # epochs when gcd(N, check_val) > 1, so no video was ever written.
    val_epochs = [e for e in range(400) if (e + 1) % 20 == 0]
    buggy = [e for e in val_epochs if e % 200 == 0]
    assert buggy == []


# --- DDP rank gating -----------------------------------------------------------


def test_only_rank0_buffers_frames(tmp_path):
    # Non-zero DDP rank: metrics are still computed (so sync_dist log_dict does
    # not deadlock), but no frames are buffered and no videos dir is created.
    ev = _StubEvalVideo(str(tmp_path / "nonzero"), viz_every_n_epochs=1)
    ev.trainer = _fake_trainer(current_epoch=0, is_global_zero=False)
    ev.on_validation_start()
    ev.on_validation_step({}, 0)
    assert ev.compute_calls == [False]
    assert ev.val_image_buffer == {}
    assert not os.path.exists(ev.video_dir())


def test_rank0_buffers_frames_and_creates_dir(tmp_path):
    ev = _StubEvalVideo(str(tmp_path / "rank0"), viz_every_n_epochs=1)
    ev.trainer = _fake_trainer(current_epoch=0, is_global_zero=True)
    ev.on_validation_start()
    ev.on_validation_step({}, 0)
    assert ev.compute_calls == [True]
    assert len(ev.val_image_buffer["mecka_bimanual"]) == 4
    assert os.path.isdir(os.path.join(ev.video_dir(), "epoch_0"))
