"""Regression tests for EvalVideo's viz scheduling and DDP rank gating.

These exercise the metrics/viz split without a GPU or a real Trainer: a
``SimpleNamespace`` stands in for ``self.trainer`` so we can drive
``global_step``/``is_global_zero`` directly.

Gating is **step based** (``trainer.global_step``) to match the step-based
training regime (``val_check_interval`` measured in steps). Cheap metrics log
on every validation pass; the expensive viz/video rendering fires only when
``global_step`` is a multiple of ``viz_every_n_steps``. Unlike the epoch-based
gate, there is no off-by-one alignment subtlety: at validation time
``global_step`` is exactly the step count Lightning triggered on, so a video
renders whenever ``viz_every_n_steps`` is an integer multiple of
``val_check_interval``.
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


def _fake_trainer(global_step=0, is_global_zero=True):
    lm = SimpleNamespace(device="cpu", log_dict=lambda *a, **k: None)
    return SimpleNamespace(
        global_step=global_step,
        is_global_zero=is_global_zero,
        lightning_module=lm,
    )


def _rendered_steps(viz_every_n, val_check_interval, max_steps):
    """Steps at which a video is actually written: Lightning runs validation
    (``step % val_check_interval == 0``) AND ``_should_viz()`` fires."""
    ev = _StubEvalVideo("/tmp", viz_every_n_steps=viz_every_n)
    out = []
    for s in range(1, max_steps + 1):
        if s % val_check_interval != 0:
            continue  # Lightning does not run validation this step
        ev.trainer = _fake_trainer(global_step=s)
        if ev._should_viz():
            out.append(s)
    return out


# --- viz / validation alignment ------------------------------------------------


def test_should_viz_step_modulo():
    # _should_viz fires exactly on multiples of viz_every_n_steps.
    ev = _StubEvalVideo("/tmp", viz_every_n_steps=10000)
    ev.trainer = _fake_trainer(global_step=10000)
    assert ev._should_viz()
    ev.trainer = _fake_trainer(global_step=20000)
    assert ev._should_viz()
    ev.trainer = _fake_trainer(global_step=12000)
    assert not ev._should_viz()


def test_viz_aligns_with_validation_trigger():
    # The canonical production case (eval_pi.yaml): val_check_interval=2000,
    # viz_every_n_steps=10000 -> metrics every 2000 steps, a video every 5th
    # validation, i.e. at 10000 and 20000.
    assert _rendered_steps(10000, 2000, 20000) == [10000, 20000]


def test_viz_every_validation_when_equal():
    # viz_every_n_steps == val_check_interval -> a video on every validation.
    assert _rendered_steps(2000, 2000, 8000) == [2000, 4000, 6000, 8000]


def test_viz_multiple_of_check_val():
    assert _rendered_steps(4000, 2000, 12000) == [4000, 8000, 12000]


def test_viz_fires_at_step_zero_standalone_eval():
    # Standalone eval runs with global_step == 0; 0 % N == 0 so viz fires.
    ev = _StubEvalVideo("/tmp", viz_every_n_steps=10000)
    ev.trainer = _fake_trainer(global_step=0)
    assert ev._should_viz()


def test_viz_disabled_for_nonpositive():
    # A non-positive cadence disables viz entirely, on every step.
    for viz_every_n in (0, -1):
        ev = _StubEvalVideo("/tmp", viz_every_n_steps=viz_every_n)
        for s in (0, 2000, 10000):
            ev.trainer = _fake_trainer(global_step=s)
            assert not ev._should_viz()


# --- DDP rank gating -----------------------------------------------------------


def test_only_rank0_buffers_frames(tmp_path):
    # Non-zero DDP rank: metrics are still computed (so sync_dist log_dict does
    # not deadlock), but no frames are buffered and no videos dir is created.
    ev = _StubEvalVideo(str(tmp_path / "nonzero"), viz_every_n_steps=1)
    ev.trainer = _fake_trainer(global_step=0, is_global_zero=False)
    ev.on_validation_start()
    ev.on_validation_step({}, 0)
    assert ev.compute_calls == [False]
    assert ev.val_image_buffer == {}
    assert not os.path.exists(ev.video_dir())


def test_rank0_buffers_frames_and_creates_dir(tmp_path):
    ev = _StubEvalVideo(str(tmp_path / "rank0"), viz_every_n_steps=1)
    ev.trainer = _fake_trainer(global_step=0, is_global_zero=True)
    ev.on_validation_start()
    ev.on_validation_step({}, 0)
    assert ev.compute_calls == [True]
    assert len(ev.val_image_buffer["mecka_bimanual"]) == 4
    assert os.path.isdir(os.path.join(ev.video_dir(), "step_0"))
