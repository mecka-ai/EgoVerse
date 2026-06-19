"""Tests for global-shuffle worker episode partitioning."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).with_name("build_global_shuffle_shards.py")
_spec = importlib.util.spec_from_file_location("build_global_shuffle_shards", _MODULE_PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)

_batch = _mod._batch_episodes_for_workers
_assert = _mod._assert_valid_batches
_summarize = _mod._summarize_batches
EPISODES_PER_WORKER = _mod.EPISODES_PER_WORKER


def _paths(n: int) -> list[str]:
    return [f"/mnt/zarr-data/ep_{i:05d}" for i in range(n)]


def _flatten(batches: list[list[str]]) -> list[str]:
    return [ep for batch in batches for ep in batch]


@pytest.mark.parametrize(
    ("n_episodes", "num_workers", "expected_workers"),
    [
        (0, 50, 0),
        (1, 50, 1),
        (30, 50, 30),
        (50, 50, 50),
        (75, 50, 50),
        (100, 50, 50),
        (5000, 50, 50),
        (5003, 50, 50),
    ],
)
def test_batch_count(n_episodes: int, num_workers: int, expected_workers: int) -> None:
    episodes = _paths(n_episodes)
    batches = _batch(episodes, num_workers)
    _assert(episodes, batches, num_workers)
    assert len(batches) == expected_workers


def test_exact_partition_no_duplicates() -> None:
    episodes = _paths(5003)
    batches = _batch(episodes, 50)
    flat = _flatten(batches)
    assert flat == episodes
    assert len(set(flat)) == len(episodes)


def test_balanced_sizes_differ_by_at_most_one() -> None:
    episodes = _paths(5003)
    batches = _batch(episodes, 50)
    sizes = [len(b) for b in batches]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 5003


def test_legacy_fallback_when_num_workers_non_positive() -> None:
    episodes = _paths(120)
    batches = _batch(episodes, 0)
    assert len(batches) == 3
    assert [len(b) for b in batches] == [50, 50, 20]


def test_summarize_batches() -> None:
    batches = _batch(_paths(75), 50)
    n_workers, min_batch, max_batch = _summarize(batches)
    assert n_workers == 50
    assert min_batch == 1
    assert max_batch == 2


def test_invalid_split_detected() -> None:
  episodes = _paths(10)
  bad_batches = [episodes[:5], episodes[4:]]
  with pytest.raises(RuntimeError, match="multiple workers"):
      _assert(episodes, bad_batches, 5)
