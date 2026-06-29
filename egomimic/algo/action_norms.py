"""Modular, toggleable normalization for action trajectories (shape-preserving).

`ActionNorms` turns a variable-length action trajectory into a fixed-length, shape-
normalized one so that *shape* survives and duration / position / scale (and optionally
orientation) are removed. It is used in BOTH places:

  * training — inside SpanActionDataset, so the temporal-CNN autoencoder learns on
    normalized spans;
  * curation — as an optional layer on top of the action embedding.

Each operation is independent and individually toggleable. Order is fixed:
    resample (to L) → translate → deltas → scale → rotate

Operating on positions before `deltas` makes `translate` a true "start/centroid → origin";
with `deltas` on, the channels become frame-to-frame velocities (position-invariant).
Numpy-native (`apply`); `apply_torch` wraps it for the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionNormsSettings:
    """Config for ActionNorms (``model.action_embedder.norms``).

    Defaults: arc-length resample + deltas + centroid-translate + path-scale ON,
    rotation OFF. ``enabled=false`` makes ``apply`` a pass-through.
    """

    enabled: bool = False
    resample_len: int = 100
    resample: str = "arc_length"   # arc_length | uniform | off
    deltas: bool = True
    translate: str = "centroid"    # start | centroid | off
    scale: str = "path"            # path | bbox | off
    rotate: str = "off"            # pca | off


class ActionNorms:
    """Apply shape-preserving normalization to action trajectories."""

    def __init__(self, settings: ActionNormsSettings | None = None) -> None:
        self.s = settings or ActionNormsSettings()

    # ------------------------------------------------------------------
    def apply(self, x: np.ndarray) -> np.ndarray:
        """Normalize a trajectory ``(T, D)`` or batch ``(N, T, D)`` → ``(L, D)`` / ``(N, L, D)``.

        Returns the input unchanged when ``enabled`` is false (pass-through).
        """
        if not self.s.enabled:
            return np.asarray(x, dtype=np.float32)
        arr = np.asarray(x, dtype=np.float64)
        single = arr.ndim == 2
        if single:
            arr = arr[None]
        out = np.stack([self._apply_one(t) for t in arr])
        return out[0] if single else out

    def apply_torch(self, x):
        """Torch wrapper — applies the numpy pipeline and returns a tensor on the same device."""
        import torch

        dev, dt = x.device, x.dtype
        out = self.apply(x.detach().cpu().numpy())
        return torch.as_tensor(out, device=dev, dtype=dt)

    # ------------------------------------------------------------------
    def _apply_one(self, t: np.ndarray) -> np.ndarray:
        if self.s.resample != "off":
            t = self._resample(t, self.s.resample_len, self.s.resample)

        # scale factor is measured on the positional trajectory (pre-deltas)
        scale = 1.0
        if self.s.scale == "path":
            scale = self._path_length(t)
        elif self.s.scale == "bbox":
            scale = self._bbox_diag(t)
        scale = scale if scale > 1e-8 else 1.0

        if self.s.translate == "start":
            t = t - t[0:1]
        elif self.s.translate == "centroid":
            t = t - t.mean(axis=0, keepdims=True)

        if self.s.deltas:
            d = np.zeros_like(t)
            d[1:] = np.diff(t, axis=0)
            t = d

        if self.s.scale != "off":
            t = t / scale

        if self.s.rotate == "pca":
            t = self._pca_align(t)

        return t.astype(np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def _interp(t: np.ndarray, frac: np.ndarray) -> np.ndarray:
        base = np.arange(t.shape[0])
        return np.stack([np.interp(frac, base, t[:, d]) for d in range(t.shape[1])], axis=1)

    def _resample(self, t: np.ndarray, length: int, mode: str) -> np.ndarray:
        T = t.shape[0]
        if T < 2:
            return np.repeat(t, length, axis=0)[:length] if T == 1 else np.zeros((length, t.shape[1]))
        if mode == "uniform":
            return self._interp(t, np.linspace(0, T - 1, length))
        # arc_length: sample points equidistant along the path (speed/duration invariant)
        seg = np.linalg.norm(np.diff(t, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        total = cum[-1]
        if total <= 1e-8:
            return self._interp(t, np.linspace(0, T - 1, length))
        frac = np.interp(np.linspace(0.0, total, length), cum, np.arange(T))
        return self._interp(t, frac)

    @staticmethod
    def _path_length(t: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())

    @staticmethod
    def _bbox_diag(t: np.ndarray) -> float:
        return float(np.linalg.norm(t.max(axis=0) - t.min(axis=0)))

    @staticmethod
    def _pca_align(t: np.ndarray) -> np.ndarray:
        mean = t.mean(axis=0, keepdims=True)
        centered = t - mean
        # right singular vectors = principal axes; rotate coords onto them
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ vt.T + mean
