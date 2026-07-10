"""Two-view contrastive dataset over annotation-span action trajectories.

One item = one annotation span, returned as TWO independently-augmented views of
its raw action trajectory, each ActionNorms-normalized to a fixed length:
``{action_key: (num_views, L, action_dim)}``. Used by the InfoNCE self-supervised
action encoder (egomimic/algo/action_contrastive.py) — the augmentations DEFINE
the invariances of the learned embedding:

  crop      -> phase / sub-segment invariance ("which part of the repetition")
  rotation  -> facing-direction invariance (one shared rotation for BOTH hands,
               applied to positions and orientations, so the bimanual relative
               structure — e.g. "left holds still, right oscillates" — survives)
  noise     -> tracking-jitter invariance
  ActionNorms (arc-length resample + centroid + path-scale) -> speed, absolute
               position and scale invariance (shared by both views)

Deliberately NOT augmented: left/right mirroring (handedness is part of the
bimanual structure) and time reversal (direction of motion matters).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.zarr_dataset_span_action import SpanActionDataset

# 12-dim bimanual cartesian layout: [L xyz ypr | R xyz ypr] (head frame).
_XYZ_SLICES = (slice(0, 3), slice(6, 9))
_YPR_SLICES = (slice(3, 6), slice(9, 12))


class ContrastiveSpanActionDataset(SpanActionDataset):
    """SpanActionDataset that yields ``num_views`` augmented views per span."""

    def __init__(
        self,
        resolver,
        num_views: int = 2,
        crop_min_frac: float = 0.6,
        rotation_max_deg: float = 45.0,
        rotation_axis: str = "y",
        noise_xyz_m: float = 0.003,
        noise_rot_rad: float = 0.02,
        **kwargs,
    ) -> None:
        super().__init__(resolver, **kwargs)
        self.num_views = int(num_views)
        self.crop_min_frac = float(crop_min_frac)
        self.rotation_max_deg = float(rotation_max_deg)
        self.rotation_axis = rotation_axis
        self.noise_xyz_m = float(noise_xyz_m)
        self.noise_rot_rad = float(noise_rot_rad)

    # ------------------------------------------------------------------
    def _augment(self, traj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One augmented view of a raw (T, 12) span trajectory (pre-ActionNorms)."""
        t = np.asarray(traj, dtype=np.float64).copy()
        T = len(t)

        # 1) temporal crop: random contiguous sub-span.
        if T > 4 and self.crop_min_frac < 1.0:
            n = max(4, int(T * rng.uniform(self.crop_min_frac, 1.0)))
            i0 = int(rng.integers(0, T - n + 1))
            t = t[i0 : i0 + n]

        # 2) shared rotation about the (approx. gravity) axis — SAME rotation for
        # both hands, applied to positions AND orientations, preserving the
        # bimanual relative structure.
        if self.rotation_max_deg > 0:
            ang = np.deg2rad(rng.uniform(-self.rotation_max_deg, self.rotation_max_deg))
            r_aug = R.from_euler(self.rotation_axis, ang)
            m_aug = r_aug.as_matrix()
            for sl_xyz, sl_ypr in zip(_XYZ_SLICES, _YPR_SLICES):
                t[:, sl_xyz] = t[:, sl_xyz] @ m_aug.T
                rot = R.from_euler("ZYX", t[:, sl_ypr])
                t[:, sl_ypr] = (r_aug * rot).as_euler("ZYX")

        # 3) temporally SMOOTH noise (slow tracking drift): i.i.d. control points
        # every ~1 s, linearly interpolated. Never i.i.d. per-frame noise — arc
        # length integrates jitter, so per-frame noise would give a static
        # holding hand ~a metre of fake path and destroy the hold-vs-move
        # bimanual structure the embedding is supposed to capture.
        n = len(t)
        if n >= 2 and (self.noise_xyz_m > 0 or self.noise_rot_rad > 0):
            n_ctrl = max(2, n // 30)
            xs = np.linspace(0, n - 1, n_ctrl)
            grid = np.arange(n)

            def smooth(sigma, dim):
                ctrl = rng.normal(0.0, sigma, (n_ctrl, dim))
                return np.stack(
                    [np.interp(grid, xs, ctrl[:, d]) for d in range(dim)], axis=1
                )

            if self.noise_xyz_m > 0:
                for sl_xyz in _XYZ_SLICES:
                    t[:, sl_xyz] += smooth(self.noise_xyz_m, 3)
            if self.noise_rot_rad > 0:
                for sl_ypr in _YPR_SLICES:
                    t[:, sl_ypr] += smooth(self.noise_rot_rad, 3)

        return t

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        ep, s, e = self.index[idx]
        full = self._episode_actions(ep)
        T = full.shape[0]
        s2, e2 = max(0, min(s, T)), max(0, min(e, T))
        traj = full[s2:e2] if e2 > s2 else full[:1]

        rng = np.random.default_rng()  # fresh entropy per call (worker-safe)
        views = []
        for _ in range(self.num_views):
            aug = self._augment(traj, rng)
            norm = self.norms.apply(aug)  # (L, D)
            views.append(np.asarray(norm, dtype=np.float32))
        return {self.action_key: torch.from_numpy(np.stack(views))}  # (V, L, D)
