"""
Plain torch Datasets for the two tactile zarr corpora produced by
`egomimic/scripts/tactile_process/{human_glove,trex_robot}_to_zarr.py`,
feeding `egomimic.algo.tactile_encoder.TactileEncoder` via
`egomimic.pl_utils.pl_data_utils.MultiDataModuleWrapper`.

These deliberately do NOT go through `MultiDataset`/`LocalEpisodeResolver`
(the camera/proprio/action-keyed pipeline other `hydra_configs/data/*.yaml`
use) -- there is no camera/proprio/action schema here, just a per-episode
tactile stream. `MultiDataModuleWrapper.train_dataloader()` only requires a
plain `torch.utils.data.Dataset` (`__len__` + `__getitem__` returning
`{"tactile": Tensor[T, C, H, W]}`); `default_collate` + `CombinedLoader`
handle the rest, batching to `Tensor[B, T, C, H, W]` and nesting per platform
name automatically.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Out of 13,197 converted episodes, at least one (tactile_human/
# 6a4b8bdb4fa21115522b9374.zarr) turned out to be an empty/corrupted zarr
# store -- zarr.open_group raised GroupNotFoundError deep in a DataLoader
# worker and killed an otherwise-healthy multi-hour training run. Rather
# than pre-validate all episodes up front (13,197 extra network-mount
# opens just at dataset construction), __getitem__ retries a different
# random episode on failure -- a bad sample should be rare enough that
# this never meaningfully biases what gets seen.
_MAX_LOAD_RETRIES = 5


def _list_episode_dirs(root_dir: str) -> list[Path]:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"root_dir does not exist: {root_dir}")
    dirs = sorted(p for p in root.iterdir() if p.name.endswith(".zarr"))
    if not dirs:
        raise FileNotFoundError(f"No .zarr episodes found under {root_dir}")
    return dirs


def _split_paths(paths: list[Path], split: str, valid_frac: float, seed: int) -> list[Path]:
    order = list(range(len(paths)))
    random.Random(seed).shuffle(order)
    n_valid = max(1, int(round(len(paths) * valid_frac))) if len(paths) > 1 else 0
    valid_idx = set(order[:n_valid])
    if split == "valid":
        return [paths[i] for i in sorted(valid_idx)]
    if split == "train":
        return [paths[i] for i in sorted(set(order) - valid_idx)]
    raise ValueError(f"split must be 'train' or 'valid', got {split!r}")


def _window_start(t: int, window: int, rng: random.Random) -> int:
    return rng.randint(0, t - window) if t > window else 0


def _slice_window(arr, t: int, window: int, start: int) -> np.ndarray:
    """Slice a `window`-length span from axis 0 starting at `start`; edge-pad if too short."""
    if t < window:
        out = np.asarray(arr[:t])
        pad_width = [(0, window - t)] + [(0, 0)] * (out.ndim - 1)
        return np.pad(out, pad_width, mode="edge")
    return np.asarray(arr[start : start + window])


class HumanGloveTactileDataset(Dataset):
    """
    Each episode zarr has `tactile_left`/`tactile_right` (T_hand, 460) float32
    arrays, sampled independently per hand (not frame-aligned to each other
    or to any real clock beyond their own `*_time_ns`). This dataset takes
    the SAME window-start index into both -- an approximation, not a claim
    of true synchrony -- and stacks them as 2 channels of a 1x460 "image"
    (H=1: the glove's real taxel->pixel layout is unknown, so no 2D grid is
    fabricated). Output: {"tactile": Tensor[window, 2, 1, 460]}.
    """

    def __init__(
        self,
        root_dir: str,
        window: int = 16,
        split: str = "train",
        valid_frac: float = 0.05,
        seed: int = 0,
        scale: float = 1.0,
    ):
        self.window = window
        self.scale = scale
        self.episode_paths = _split_paths(_list_episode_dirs(root_dir), split, valid_frac, seed)
        self._rng = random.Random()

    def __len__(self) -> int:
        return len(self.episode_paths)

    def _load(self, idx: int) -> dict:
        path = self.episode_paths[idx]
        z = zarr.open_group(str(path), mode="r")
        left, right = z["tactile_left"], z["tactile_right"]
        t = min(left.shape[0], right.shape[0])
        w = self.window
        # Shared start so both hands use the same window offset (an
        # approximation -- they are not truly frame-synchronized).
        start = _window_start(t, w, self._rng)
        left_win = _slice_window(left, t, w, start)
        right_win = _slice_window(right, t, w, start)
        tactile = np.stack([left_win, right_win], axis=1)  # [w, 2, 460]
        tactile = tactile[:, :, None, :]  # [w, 2, 1, 460]
        tensor = torch.from_numpy(np.ascontiguousarray(tactile, dtype=np.float32)) * self.scale
        return {"tactile": tensor}

    def __getitem__(self, idx: int) -> dict:
        for attempt in range(_MAX_LOAD_RETRIES):
            try:
                return self._load(idx)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"HumanGloveTactileDataset: failed to load "
                    f"{self.episode_paths[idx]} ({type(e).__name__}: {e}); "
                    f"retrying with a different episode ({attempt + 1}/{_MAX_LOAD_RETRIES})"
                )
                idx = self._rng.randrange(len(self.episode_paths))
        raise RuntimeError(
            f"HumanGloveTactileDataset: {_MAX_LOAD_RETRIES} consecutive episode loads failed"
        )


class RobotTrexTactileDataset(Dataset):
    """
    Each episode zarr has `tactile_raw` (T, 10, 240, 320) uint8 -- 10
    per-fingertip (5 fingers x 2 hands) raw grayscale tactile-sensor images
    at native resolution, in the channel order recorded in the episode's
    `tactile_raw_channel_order` attr. Output: {"tactile": Tensor[window, 10,
    240, 320]}, scaled to [0, 1].
    """

    def __init__(
        self,
        root_dir: str,
        window: int = 16,
        split: str = "train",
        valid_frac: float = 0.05,
        seed: int = 0,
    ):
        self.window = window
        self.episode_paths = _split_paths(_list_episode_dirs(root_dir), split, valid_frac, seed)
        self._rng = random.Random()

    def __len__(self) -> int:
        return len(self.episode_paths)

    def _load(self, idx: int) -> dict:
        path = self.episode_paths[idx]
        z = zarr.open_group(str(path), mode="r")
        raw = z["tactile_raw"]
        t = raw.shape[0]
        start = _window_start(t, self.window, self._rng)
        arr = _slice_window(raw, t, self.window, start)
        tensor = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)) / 255.0
        return {"tactile": tensor}

    def __getitem__(self, idx: int) -> dict:
        for attempt in range(_MAX_LOAD_RETRIES):
            try:
                return self._load(idx)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    f"RobotTrexTactileDataset: failed to load "
                    f"{self.episode_paths[idx]} ({type(e).__name__}: {e}); "
                    f"retrying with a different episode ({attempt + 1}/{_MAX_LOAD_RETRIES})"
                )
                idx = self._rng.randrange(len(self.episode_paths))
        raise RuntimeError(
            f"RobotTrexTactileDataset: {_MAX_LOAD_RETRIES} consecutive episode loads failed"
        )
