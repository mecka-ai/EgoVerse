"""Arc Tokenizer — the progress-based action tokenizer as a model-format module.

THE canonical implementation of the Arc Tokenizer algorithm (sequence actions by
progress instead of time). It is an ``nn.Module`` with zero learned parameters
today: tokenization is deterministic (arc-length resampling + still branch +
velocity features). It is deliberately in model format so that learned variants
(learned velocity features, learned resampler, learned still detection) can drop
in by overriding the hook methods — without restructuring the pipeline.

Execution today happens per-sample inside dataloader workers via the thin
``ApplyArcTokenizer`` data-transform adapter (parallel CPU, numpy). If the module
gains parameters, move the call into the trainer's ``process_batch_for_training``
(GPU, batched ``forward``) — the interface is already shaped for that.

Algorithm (per anchor window):
  1. Cumulative arc length s over the window: the combined translation metric of
     ``distance poses`` (both wrists: sqrt(|dL|^2 + |dR|^2) per step).
  2. s_end = min(delta_s, s_T); N_t = frames to FIRST reach s_end.
  3. STILL branch — s_end < eps (no EEF motion) or N_t > n_max (pause/too slow):
     translation = anchor repeated ``num_waypoints`` times, velocity 0 — but
     rotation and extra dims (fingertips/joints) are NOT repeated or resampled:
     they pass through as raw time samples (first ``num_waypoints`` frames), so
     in-place wrist rotation and finger articulation survive the still token.
  4. Else RESAMPLE at linspace(0, s_end, num_waypoints): the interpolation warp
     is computed from the EEF translations ONLY, and the SAME interpolating
     factor is applied to rotation (slerp keyed on s) and all extra dims
     (linear keyed on s) — hybrid actions never get their own metric. Partial
     paths (eps <= s_end < delta_s) resample as-is.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


class ArcTokenizer(nn.Module):
    """Parameter-free Arc Tokenizer in model format.

    Args:
        delta_s: distance unit per token chunk (metres of combined wrist travel).
        num_waypoints: waypoints per chunk (M), equally spaced in arc length.
        eps: minimum available distance below which the anchor is a STILL token.
        n_max: max frames allowed to cover delta_s before the anchor is treated
            as a pause (STILL token).
        velocity_mode: "per_waypoint" (local ds/dframe at each waypoint via the
            inverse map t(s)) or "mean" (scalar mean speed broadcast).
    """

    def __init__(
        self,
        delta_s: float = 0.30,
        num_waypoints: int = 100,
        eps: float = 0.03,
        n_max: int = 450,
        velocity_mode: Literal["per_waypoint", "mean"] = "per_waypoint",
    ):
        super().__init__()
        self.delta_s = float(delta_s)
        self.num_waypoints = int(num_waypoints)
        self.eps = float(eps)
        self.n_max = int(n_max)
        self.velocity_mode = velocity_mode

    # ------------------------------------------------------------------
    # Core (numpy, per window) — called from dataloader workers today
    # ------------------------------------------------------------------

    def arc_length(self, pose_chunks: list[np.ndarray]) -> np.ndarray:
        """Cumulative combined arc length s (T,) from pose chunk translations."""
        xyz = np.concatenate(
            [np.asarray(c, dtype=np.float64)[:, :3] for c in pose_chunks], axis=-1
        )
        ds = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)
        return np.concatenate([[0.0], np.cumsum(ds)])

    def compute_velocity(self, u: np.ndarray, t_m: np.ndarray, n_t: int) -> np.ndarray:
        """Velocity feature (M,) — override in learned variants."""
        M = self.num_waypoints
        if self.velocity_mode == "mean":
            return np.full(M, (u[-1] - u[0]) / max(n_t - 1, 1))
        speed = np.empty(M)
        dt = np.maximum(t_m[1:] - t_m[:-1], 1e-9)
        du = u[1:] - u[:-1]
        speed[1:-1] = (u[2:] - u[:-2]) / np.maximum(t_m[2:] - t_m[:-2], 1e-9)
        speed[0] = du[0] / dt[0]
        speed[-1] = du[-1] / dt[-1]
        return speed

    def tokenize_window(
        self,
        poses: dict[str, np.ndarray],
        points: dict[str, np.ndarray],
        distance_keys: list[str],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
        """Tokenize one raw window.

        Args:
            poses: name -> (T, 7) xyzwxyz pose chunk (resampled with slerp).
            points: name -> (T, K, 3) point chunks (resampled linearly).
            distance_keys: subset of ``poses`` defining the progress metric.

        Returns:
            (poses_out, points_out, speed): each output chunk has
            ``num_waypoints`` rows; speed is (num_waypoints, 1).
        """
        M = self.num_waypoints
        s = self.arc_length([poses[k] for k in distance_keys])
        s_end = min(self.delta_s, float(s[-1]))
        # Frames to FIRST reach s_end (plateaus at s_end must not inflate N_t).
        n_t = int(np.searchsorted(s, s_end, side="left")) + 1

        if s_end < self.eps or n_t > self.n_max:
            # STILL / pause token. Translation is repeated at the anchor and
            # velocity is 0 (no EEF progress) — but rotation and extra dims are
            # NOT repeated or resampled: they pass through as raw time samples
            # (first M frames), so in-place wrist rotation and finger
            # articulation survive the still token.
            def first_m(v):
                v = np.asarray(v, dtype=np.float64)
                if len(v) >= M:
                    return v[:M].copy()
                return np.concatenate(
                    [v, np.repeat(v[-1:], M - len(v), axis=0)], axis=0
                )

            poses_out = {}
            for k, v in poses.items():
                raw = first_m(v)
                raw[:, :3] = np.asarray(v, dtype=np.float64)[0, :3]  # anchor xyz
                poses_out[k] = raw  # rotation columns stay raw time samples
            points_out = {k: first_m(v) for k, v in points.items()}
            return poses_out, points_out, np.zeros((M, 1), dtype=np.float64)

        keep = np.concatenate([[True], np.diff(s) > 1e-12])
        s_k = s[keep]
        f_k = np.arange(len(s))[keep].astype(np.float64)
        u = np.linspace(0.0, s_end, M)

        poses_out = {}
        for k, v in poses.items():
            chunk = np.asarray(v, dtype=np.float64)[keep]
            xyz_i = np.stack([np.interp(u, s_k, chunk[:, d]) for d in range(3)], axis=1)
            rots = R.from_quat(chunk[:, [4, 5, 6, 3]])  # wxyz -> xyzw
            quat_i = Slerp(s_k, rots)(u).as_quat()
            poses_out[k] = np.concatenate([xyz_i, quat_i[:, [3, 0, 1, 2]]], axis=1)

        points_out = {}
        for k, v in points.items():
            pts = np.asarray(v, dtype=np.float64)[keep]
            flat = pts.reshape(len(s_k), -1)
            out = np.stack(
                [np.interp(u, s_k, flat[:, d]) for d in range(flat.shape[1])],
                axis=1,
            )
            points_out[k] = out.reshape(M, *pts.shape[1:])

        t_m = np.interp(u, s_k, f_k)
        speed = self.compute_velocity(u, t_m, n_t)
        return poses_out, points_out, speed[:, None]

    # ------------------------------------------------------------------
    # Model-format entry point (future learned variants run here, batched)
    # ------------------------------------------------------------------

    def forward(
        self,
        poses: dict[str, torch.Tensor],
        points: dict[str, torch.Tensor],
        distance_keys: list[str],
    ):
        """Batched tokenization: (B, T, ...) tensors in, (B, M, ...) tensors out.

        Deterministic today (loops the numpy core per batch element on CPU).
        A learned variant overrides this with a batched differentiable version.
        """
        B = next(iter(poses.values())).shape[0]
        p_out, x_out, v_out = [], [], []
        for b in range(B):
            p = {k: v[b].detach().cpu().numpy() for k, v in poses.items()}
            x = {k: v[b].detach().cpu().numpy() for k, v in points.items()}
            po, xo, sp = self.tokenize_window(p, x, distance_keys)
            p_out.append(po)
            x_out.append(xo)
            v_out.append(sp)
        device = next(iter(poses.values())).device
        poses_t = {
            k: torch.as_tensor(
                np.stack([po[k] for po in p_out]), dtype=torch.float32, device=device
            )
            for k in poses
        }
        points_t = {
            k: torch.as_tensor(
                np.stack([xo[k] for xo in x_out]), dtype=torch.float32, device=device
            )
            for k in points
        }
        speed_t = torch.as_tensor(np.stack(v_out), dtype=torch.float32, device=device)
        return poses_t, points_t, speed_t
