"""Unit tests for the curation filters."""

from __future__ import annotations

import numpy as np

from egomimic.curation.filters import (
    ActionClipFilter,
    MinLengthFilter,
    PauseFilter,
    apply_filters,
)
from egomimic.curation.utils import Episode


def _make_episode(
    actions: np.ndarray,
    observations: np.ndarray | None = None,
    embodiment: str = "test",
    episode_hash: str = "ep0",
) -> Episode:
    if observations is None:
        observations = actions.copy()
    return Episode(
        episode_hash=episode_hash,
        observations=observations.astype(np.float32),
        actions=actions.astype(np.float32),
        embodiment=embodiment,
        metadata={},
    )


# ---------------------------------------------------------------------------
# PauseFilter
# ---------------------------------------------------------------------------


def test_pause_filter_keeps_motion_frames() -> None:
    actions = np.arange(20, dtype=np.float32).reshape(-1, 1)
    ep = _make_episode(actions)
    out = PauseFilter(epsilon=0.5)(ep)
    assert out is not None
    assert len(out.actions) == 20


def test_pause_filter_collapses_pause_run() -> None:
    # Frames 5..14 are an identical pause; PauseFilter keeps one representative.
    actions = np.zeros((20, 1), dtype=np.float32)
    actions[:5, 0] = np.arange(5)
    actions[5:15, 0] = 4.0  # 10 consecutive identical frames → pause
    actions[15:, 0] = 4.0 + np.arange(1, 6)
    ep = _make_episode(actions)
    out = PauseFilter(epsilon=1e-3)(ep)
    assert out is not None
    # 5 motion + 1 pause-representative + 5 motion = 11
    assert len(out.actions) == 11
    # Returned arrays remain consistent in length
    assert len(out.observations) == len(out.actions)


def test_pause_filter_short_episode_passes_through() -> None:
    actions = np.array([[0.0], [1.0]], dtype=np.float32)
    ep = _make_episode(actions)
    out = PauseFilter(epsilon=0.5)(ep)
    assert out is not None
    assert len(out.actions) == 2


def test_pause_filter_embodiment_override() -> None:
    actions = np.zeros((10, 1), dtype=np.float32)
    actions[:, 0] = np.arange(10) * 0.01  # delta = 0.01 per step
    ep_default = _make_episode(actions, embodiment="x")
    ep_override = _make_episode(actions, embodiment="x")

    # epsilon=0.005 → 0.01 deltas exceed → no pauses
    assert len(PauseFilter(epsilon=0.005)(ep_default).actions) == 10
    # Override to 0.05 → every delta is a pause. PauseFilter keeps the first
    # frame plus the first pause-representative (frame 1), then drops the rest.
    out = PauseFilter(epsilon=0.005, embodiment_overrides={"x": 0.05})(ep_override)
    assert out is not None
    assert len(out.actions) == 2


# ---------------------------------------------------------------------------
# ActionClipFilter
# ---------------------------------------------------------------------------


def test_action_clip_filter_normalises_and_clips() -> None:
    actions = np.array(
        [[100.0], [101.0], [102.0], [103.0], [10000.0]], dtype=np.float32
    )
    ep = _make_episode(actions)
    out = ActionClipFilter(clip=1.0)(ep)
    assert out is not None
    assert out.actions.max() <= 1.0 + 1e-6
    assert out.actions.min() >= -1.0 - 1e-6
    # Length is preserved by ActionClipFilter
    assert len(out.actions) == len(actions)


def test_action_clip_filter_handles_zero_std() -> None:
    actions = np.ones((10, 2), dtype=np.float32)
    ep = _make_episode(actions)
    out = ActionClipFilter(clip=1.0)(ep)
    assert out is not None
    assert np.isfinite(out.actions).all()


# ---------------------------------------------------------------------------
# MinLengthFilter
# ---------------------------------------------------------------------------


def test_min_length_filter_drops_short() -> None:
    ep = _make_episode(np.zeros((5, 1), dtype=np.float32))
    assert MinLengthFilter(min_length=10)(ep) is None


def test_min_length_filter_keeps_long() -> None:
    ep = _make_episode(np.zeros((20, 1), dtype=np.float32))
    out = MinLengthFilter(min_length=10)(ep)
    assert out is ep


# ---------------------------------------------------------------------------
# apply_filters
# ---------------------------------------------------------------------------


def test_apply_filters_chains_and_records_removals() -> None:
    short_ep = _make_episode(np.zeros((3, 1), dtype=np.float32), episode_hash="short")
    long_ep = _make_episode(
        np.arange(50, dtype=np.float32).reshape(-1, 1), episode_hash="long"
    )

    kept, removed = apply_filters(
        [short_ep, long_ep],
        filters=[MinLengthFilter(min_length=10)],
    )
    assert [ep.episode_hash for ep in kept] == ["long"]
    assert removed == ["short"]


def test_apply_filters_handles_filter_returning_none() -> None:
    ep = _make_episode(np.zeros((50, 1), dtype=np.float32), episode_hash="zero_run")
    # PauseFilter collapses 50 identical frames to a single one → MinLength drops it
    kept, removed = apply_filters(
        [ep],
        filters=[PauseFilter(epsilon=1e-3), MinLengthFilter(min_length=10)],
    )
    assert kept == []
    assert removed == ["zero_run"]
