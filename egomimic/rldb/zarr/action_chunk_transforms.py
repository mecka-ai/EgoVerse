"""
Embodiment-dependent action chunk transforms for ZarrDataset.

Replicates the prestacking transformations from aria_to_lerobot.py / eva_to_lerobot.py,
applied at load time instead of at data creation time. Raw action frames are loaded
as (action_horizon, action_dim) and interpolated to (chunk_length, action_dim).

Translation (xyz) and gripper dimensions use linear interpolation.
Rotation (euler ypr) dimensions use np.unwrap before interpolation and rewrap after,
matching the behaviour of egomimicUtils.interpolate_arr_euler.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import numpy as np
import torch
from projectaria_tools.core.sophus import SE3
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from egomimic.utils.pose_utils import (
    _interpolate_euler,
    _interpolate_linear,
    _interpolate_linear_batch,
    _interpolate_quat_wxyz,
    _interpolate_quat_wxyz_batch,
    _interpolate_xyz,
    _matrix_to_xyz,
    _matrix_to_xyz6d,
    _matrix_to_xyzwxyz,
    _matrix_to_xyzypr,
    _xyz6d_to_matrix,
    _xyz_to_matrix,
    _xyzwxyz_to_matrix,
    _xyzypr_to_matrix,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)

# ---------------------------------------------------------------------------
# Base Transform
# ---------------------------------------------------------------------------


class Transform:
    """Base Class for all transforms."""

    @abstractmethod
    def transform(self, batch: dict) -> dict:
        """Transform the data."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Interpolation Transforms
# ---------------------------------------------------------------------------


class InterpolatePose(Transform):
    """Interpolate a pose chunk of shape (T, 6) or (T, 7)."""

    def __init__(
        self,
        new_chunk_length: int,
        action_key: str,
        output_action_key: str,
        stride: int = 1,
        mode: Literal["xyzwxyz", "xyzypr"] = "xyzwxyz",
    ):
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.new_chunk_length = new_chunk_length
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.stride = int(stride)
        self.mode = mode

    def transform(self, batch: dict) -> dict:
        actions = np.asarray(batch[self.action_key])
        actions = actions[:: self.stride]
        if self.mode == "xyzwxyz":
            if actions.ndim != 2 or actions.shape[-1] != 7:
                raise ValueError(
                    f"InterpolatePose expects (T, 7) when is_quat=True, got "
                    f"{actions.shape} for key '{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_quat_wxyz(
                actions, self.new_chunk_length
            )
        elif self.mode == "xyzypr":
            if actions.ndim != 2 or actions.shape[-1] != 6:
                raise ValueError(
                    f"InterpolatePose expects (T, 6), got {actions.shape} for key "
                    f"'{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_euler(
                actions, self.new_chunk_length
            )
        else:
            if actions.shape[-1] != 3:
                raise ValueError(
                    f"InterpolatePose expects (T, 3) or (T, K, 3), got {actions.shape} for key "
                    f"'{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_xyz(
                actions, self.new_chunk_length
            )
        return batch

    def transform_batch(self, batch: dict) -> dict:
        """Vectorized: (B, H, D) → (B, new_chunk_length, D) after stride."""
        actions = np.asarray(batch[self.action_key])  # (B, H, D)
        actions = actions[:, :: self.stride, :]  # (B, H//stride, D)
        if self.mode == "xyzwxyz":
            if actions.shape[-1] != 7:
                raise ValueError(
                    f"InterpolatePose.transform_batch (xyzwxyz): expected last dim 7, "
                    f"got {actions.shape} for key '{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_quat_wxyz_batch(
                actions, self.new_chunk_length
            )
        elif self.mode == "xyzypr":
            if actions.shape[-1] != 6:
                raise ValueError(
                    f"InterpolatePose.transform_batch (xyzypr): expected last dim 6, "
                    f"got {actions.shape} for key '{self.action_key}'"
                )
            batch[self.output_action_key] = _interpolate_linear_batch(
                actions, self.new_chunk_length
            )
        else:
            batch[self.output_action_key] = _interpolate_linear_batch(
                actions, self.new_chunk_length
            )
        return batch


class InterpolateLinear(Transform):
    """Interpolate any chunk of shape (T, D) with linear interpolation."""

    def __init__(
        self,
        new_chunk_length: int,
        action_key: str,
        output_action_key: str,
        stride: int = 1,
    ):
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        self.new_chunk_length = new_chunk_length
        self.action_key = action_key
        self.output_action_key = output_action_key
        self.stride = int(stride)

    def transform(self, batch: dict) -> dict:
        actions = np.asarray(batch[self.action_key])
        if actions.ndim != 2:
            raise ValueError(
                f"InterpolateLinear expects (T, D), got {actions.shape} for key "
                f"'{self.action_key}'"
            )
        actions = actions[:: self.stride]
        batch[self.output_action_key] = _interpolate_linear(
            actions, self.new_chunk_length
        )
        return batch


# ---------------------------------------------------------------------------
# Coordinate Transforms
# ---------------------------------------------------------------------------


class ActionChunkCoordinateFrameTransform(Transform):
    def __init__(
        self,
        target_world: str,
        chunk_world: str,
        transformed_key_name: str,
        extra_batch_key: dict = None,
        mode: Literal["xyz", "xyzwxyz", "xyzypr", "xyz6d"] = "xyzwxyz",
        inverse: bool = True,
    ):
        """
        args:
            target_world:
            chunk_world:
            transformed_key_name:
            is_quat: if True, inputs are xyz + quat(wxyz); otherwise xyz + ypr.
        """
        self.target_world = target_world
        self.chunk_world = chunk_world
        self.transformed_key_name = transformed_key_name
        self.extra_batch_key = extra_batch_key
        self.mode = mode
        self.inverse = inverse

    def transform(self, batch):
        """
        args:
            batch:
                if is_quat=False, inputs are xyz + ypr.
                if is_quat=True, inputs are xyz + quat(wxyz).
                Input shape validation is delegated to the selected to-matrix helper.
                transformed_key_name: str, name of the new key to store the transformed chunk world in

        returns
            batch with new key containing transformed chunk world in target frame:
                if is_quat=False: (T, 6) xyz + ypr
                if is_quat=True: (T, 7) xyz + quat(wxyz)
        """
        # flatten to (T, D)
        # target world is head pose, chunk world is keypoints
        batch.update(self.extra_batch_key or {})
        target_world = np.asarray(batch[self.target_world])
        chunk_world = np.asarray(batch[self.chunk_world])
        chunk_world_shape = None

        if chunk_world.ndim > 2:
            chunk_world_shape = chunk_world.shape
            chunk_world = chunk_world.reshape(-1, chunk_world_shape[-1])

        to_matrix_fn = None
        if self.mode == "xyzwxyz":
            to_matrix_fn = _xyzwxyz_to_matrix
        elif self.mode == "xyzypr":
            to_matrix_fn = _xyzypr_to_matrix
        elif self.mode == "xyz6d":
            # Gram-Schmidt re-orthonormalization happens here when reverting a
            # (possibly non-orthonormal) model 6D prediction back to a frame.
            to_matrix_fn = _xyz6d_to_matrix
        elif self.mode == "xyz":
            to_matrix_fn = _xyz_to_matrix
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Dispatch the target-world parser by its width: 7 -> xyz+quat(wxyz),
        # 9 -> xyz+6D columns, else xyz+ypr.
        target_width = target_world.shape[-1]
        if target_width == 7:
            target_world_to_matrix_fn = _xyzwxyz_to_matrix
        elif target_width == 9:
            target_world_to_matrix_fn = _xyz6d_to_matrix
        else:
            target_world_to_matrix_fn = _xyzypr_to_matrix
        # Convert to SE3 for transformation
        target_se3 = SE3.from_matrix(
            target_world_to_matrix_fn(target_world[None, :])[0]
        )  # (4, 4)
        chunk_se3 = SE3.from_matrix(to_matrix_fn(chunk_world))  # (T, 4, 4)

        # Compute relative transform and apply to chunk
        if self.inverse:
            chunk_in_target_frame = target_se3.inverse() @ chunk_se3
        else:
            chunk_in_target_frame = target_se3 @ chunk_se3
        chunk_mats = chunk_in_target_frame.to_matrix()
        if chunk_mats.ndim == 2:
            chunk_mats = chunk_mats[None, ...]

        if self.mode == "xyzwxyz":
            chunk_in_target_frame = _matrix_to_xyzwxyz(chunk_mats)
        elif self.mode == "xyzypr":
            chunk_in_target_frame = _matrix_to_xyzypr(chunk_mats)
        elif self.mode == "xyz6d":
            chunk_in_target_frame = _matrix_to_xyz6d(chunk_mats)
        elif self.mode == "xyz":
            chunk_in_target_frame = _matrix_to_xyz(chunk_mats)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if chunk_world_shape is not None:
            chunk_in_target_frame = chunk_in_target_frame.reshape(*chunk_world_shape)

        # Store transformed chunk back in batch
        batch[self.transformed_key_name] = chunk_in_target_frame

        return batch

    def transform_batch(self, batch: dict) -> dict:
        """
        Vectorized SE3 transform over a full episode.

        Inputs (B = timesteps):
            target_world : (B, 7)          — head pose at each timestep
            chunk_world  : (B, H, D) or (B, D)  — action chunk or single pose

        Computes ``target_inv @ chunk`` analytically without projectaria_tools,
        using R^{-1} = R^T for rotation matrices.
        """
        batch.update(self.extra_batch_key or {})
        target_world = np.asarray(batch[self.target_world], dtype=np.float64)  # (B, D)
        chunk_world = np.asarray(
            batch[self.chunk_world], dtype=np.float64
        )  # (B, H, D) or (B, D)
        B = target_world.shape[0]

        chunk_3d = chunk_world.ndim == 3
        if chunk_3d:
            H = chunk_world.shape[1]
            chunk_2d = chunk_world.reshape(B * H, chunk_world.shape[-1])
        else:
            chunk_2d = chunk_world  # (B, D)

        _to_mat = {
            "xyzwxyz": _xyzwxyz_to_matrix,
            "xyzypr": _xyzypr_to_matrix,
            "xyz": _xyz_to_matrix,
        }[self.mode]
        _from_mat = {
            "xyzwxyz": _matrix_to_xyzwxyz,
            "xyzypr": _matrix_to_xyzypr,
            "xyz": _matrix_to_xyz,
        }[self.mode]
        _tgt_to_mat = (
            _xyzwxyz_to_matrix if target_world.shape[-1] == 7 else _xyzypr_to_matrix
        )

        # SE3 inverse: [[R, t], [0,1]]^{-1} = [[R^T, -R^T t], [0, 1]]
        tgt_mats = _tgt_to_mat(target_world)  # (B, 4, 4)
        R = tgt_mats[:, :3, :3]  # (B, 3, 3)
        t = tgt_mats[:, :3, 3:]  # (B, 3, 1)
        R_T = R.swapaxes(-1, -2)  # (B, 3, 3)
        tgt_inv = np.zeros_like(tgt_mats)
        tgt_inv[:, :3, :3] = R_T
        tgt_inv[:, :3, 3:] = -(R_T @ t)
        tgt_inv[:, 3, 3] = 1.0

        chunk_mats = _to_mat(chunk_2d)  # (B*H, 4, 4) or (B, 4, 4)

        if chunk_3d:
            chunk_mats = chunk_mats.reshape(B, H, 4, 4)
            if self.inverse:
                result_mats = (tgt_inv[:, None] @ chunk_mats).reshape(B * H, 4, 4)
            else:
                result_mats = (tgt_mats[:, None] @ chunk_mats).reshape(B * H, 4, 4)
        else:
            result_mats = (
                (tgt_inv @ chunk_mats) if self.inverse else (tgt_mats @ chunk_mats)
            )

        result_flat = _from_mat(result_mats)  # (B*H, D) or (B, D)

        if chunk_3d:
            result = result_flat.reshape(B, H, result_flat.shape[-1])
        else:
            result = result_flat

        src_dtype = np.asarray(batch[self.chunk_world]).dtype
        out_dtype = src_dtype if np.issubdtype(src_dtype, np.floating) else np.float64
        batch[self.transformed_key_name] = result.astype(out_dtype, copy=False)
        return batch


class QuaternionPoseToYPR(Transform):
    """Convert a single pose from xyz + quat(x,y,z,w) to xyz + ypr."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.shape != (7,):
            raise ValueError(
                f"QuaternionPoseToYPR expects shape (7,), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:3]
        xyzw = wxyz_to_xyzw(pose[3:7])
        ypr = R.from_quat(xyzw).as_euler("ZYX", degrees=False)
        batch[self.output_key] = np.concatenate([xyz, ypr], axis=0)
        return batch


class YPRToQuaternionPose(Transform):
    """Convert a single pose from xyz + ypr to xyz + quat(x,y,z,w)."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.shape != (6,):
            raise ValueError(
                f"YPRToQuaternionPose expects shape (6,), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:3]
        quat = R.from_euler("ZYX", pose[3:6], degrees=False).as_quat()  # (x,y,z,w)
        quat = xyzw_to_wxyz(quat)
        batch[self.output_key] = np.concatenate([xyz, quat], axis=0)
        return batch


class BatchQuaternionPoseToYPR(Transform):
    """Convert a batch of poses from xyz + quat(x,y,z,w) to xyz + ypr."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.ndim != 2 or pose.shape[-1] != 7:
            raise ValueError(
                f"BatchQuaternionPoseToYPR expects shape (N, 7), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:, :3]
        xyzw = wxyz_to_xyzw(pose[:, 3:7])
        ypr = R.from_quat(xyzw).as_euler("ZYX", degrees=False)  # (N, 3)
        batch[self.output_key] = np.concatenate([xyz, ypr], axis=1)
        return batch


class BatchYPRToQuaternionPose(Transform):
    """Convert a batch of poses from xyz + ypr to xyz + quat(x,y,z,w)."""

    def __init__(self, pose_key: str, output_key: str):
        self.pose_key = pose_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        pose = np.asarray(batch[self.pose_key])
        if pose.ndim != 2 or pose.shape[-1] != 6:
            raise ValueError(
                f"BatchYPRToQuaternionPose expects shape (N, 6), got {pose.shape} for key "
                f"'{self.pose_key}'"
            )
        xyz = pose[:, :3]
        quat = R.from_euler("ZYX", pose[:, 3:6], degrees=False).as_quat()  # (N, 4)
        quat = xyzw_to_wxyz(quat)
        batch[self.output_key] = np.concatenate([xyz, quat], axis=1)
        return batch


class PoseCoordinateFrameTransform(Transform):
    """Transform a single pose into a target frame pose."""

    def __init__(
        self,
        target_world: str,
        pose_world: str,
        transformed_key_name: str,
        mode: Literal["xyzwxyz", "xyzypr", "xyz"] = "xyzwxyz",
    ):
        self.target_world = target_world
        self.pose_world = pose_world
        self.transformed_key_name = transformed_key_name
        self.mode = mode
        self._chunk_transform = ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=pose_world,
            transformed_key_name=transformed_key_name,
            mode=mode,
        )

    def transform(self, batch: dict) -> dict:
        pose_world = np.asarray(batch[self.pose_world])
        transformed = self._chunk_transform.transform(
            {
                self.target_world: batch[self.target_world],
                self.pose_world: pose_world[None, :],
            }
        )
        batch[self.transformed_key_name] = np.asarray(
            transformed[self.transformed_key_name]
        )[0]
        return batch

    def transform_batch(self, batch: dict) -> dict:
        """Vectorized: target (B, 7), pose (B, D) → transformed pose (B, D)."""
        sub = {
            self.target_world: batch[self.target_world],
            self.pose_world: batch[self.pose_world],
        }
        result = self._chunk_transform.transform_batch(sub)
        batch[self.transformed_key_name] = result[self.transformed_key_name]
        return batch


class DeleteKeys(Transform):
    def __init__(self, keys_to_delete):
        self.keys_to_delete = keys_to_delete

    def transform(self, batch):
        for key in self.keys_to_delete:
            batch.pop(key, None)
        return batch

    transform_batch = transform


class XYZWXYZ_to_XYZYPR(Transform):
    """Convert listed keys from xyz+quat(wxyz) to xyz+ypr in-place."""

    def __init__(self, keys: list[str]):
        self.keys = list(keys)

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            value = np.asarray(batch[key])
            if value.ndim == 1 and value.shape[0] == 7:
                batch[key] = _matrix_to_xyzypr(_xyzwxyz_to_matrix(value[None, :]))[0]
            elif value.ndim == 2 and value.shape[1] == 7:
                batch[key] = _matrix_to_xyzypr(_xyzwxyz_to_matrix(value))
            else:
                raise ValueError(
                    f"XYZWXYZ_to_XYZYPR expects key '{key}' to have shape (7,) "
                    f"or (T, 7), got {value.shape}"
                )
        return batch

    def transform_batch(self, batch: dict) -> dict:
        """
        Vectorized: (B, 7) obs → (B, 6), (B, H, 7) chunks → (B, H, 6).

        Reshape to (B*H, 7), convert, reshape back — no Python loop over B.
        """
        for key in self.keys:
            value = np.asarray(batch[key])
            if value.ndim == 2 and value.shape[-1] == 7:
                # (B, 7) single pose per timestep
                batch[key] = _matrix_to_xyzypr(_xyzwxyz_to_matrix(value))  # (B, 6)
            elif value.ndim == 3 and value.shape[-1] == 7:
                # (B, H, 7) action chunk
                B, H = value.shape[:2]
                flat = _matrix_to_xyzypr(_xyzwxyz_to_matrix(value.reshape(B * H, 7)))
                batch[key] = flat.reshape(B, H, 6)
            else:
                raise ValueError(
                    f"XYZWXYZ_to_XYZYPR.transform_batch: key '{key}' shape {value.shape} "
                    f"— expected (B, 7) or (B, H, 7)"
                )
        return batch


class XYZWXYZ_to_XYZ6D(Transform):
    """Convert listed keys from xyz+quat(wxyz) to xyz+6D-columns in-place.

    The 6D representation (Zhou et al. / 6DRepNet) is the first two columns of
    the rotation matrix and is continuous everywhere (no ±pi wraparound), which
    is what makes per-dimension normalization meaningful. Ported from pi-6d.
    """

    def __init__(self, keys: list[str]):
        self.keys = list(keys)

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            value = np.asarray(batch[key])
            if value.ndim == 1 and value.shape[0] == 7:
                batch[key] = _matrix_to_xyz6d(_xyzwxyz_to_matrix(value[None, :]))[0]
            elif value.ndim == 2 and value.shape[1] == 7:
                batch[key] = _matrix_to_xyz6d(_xyzwxyz_to_matrix(value))
            else:
                raise ValueError(
                    f"XYZWXYZ_to_XYZ6D expects key '{key}' to have shape (7,) "
                    f"or (T, 7), got {value.shape}"
                )
        return batch

    def transform_batch(self, batch: dict) -> dict:
        """Vectorized: (B, 7) obs → (B, 9), (B, H, 7) chunks → (B, H, 9)."""
        for key in self.keys:
            value = np.asarray(batch[key])
            if value.ndim == 2 and value.shape[-1] == 7:
                batch[key] = _matrix_to_xyz6d(_xyzwxyz_to_matrix(value))  # (B, 9)
            elif value.ndim == 3 and value.shape[-1] == 7:
                B, H = value.shape[:2]
                flat = _matrix_to_xyz6d(_xyzwxyz_to_matrix(value.reshape(B * H, 7)))
                batch[key] = flat.reshape(B, H, 9)
            else:
                raise ValueError(
                    f"XYZWXYZ_to_XYZ6D.transform_batch: key '{key}' shape {value.shape} "
                    f"— expected (B, 7) or (B, H, 7)"
                )
        return batch


class XYZ6D_to_XYZYPR(Transform):
    """Convert listed keys from xyz+6D-columns to xyz+ypr in-place.

    Runs Gram-Schmidt (via ``_xyz6d_to_matrix``) to re-orthonormalize the two
    columns, then reads Euler angles off the matrix. Used at the tail of the
    revert pipeline so viz keeps seeing ypr while the model predicts 6D. Ported
    from pi-6d.
    """

    def __init__(self, keys: list[str]):
        self.keys = list(keys)

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            value = np.asarray(batch[key])
            if value.ndim == 1 and value.shape[0] == 9:
                batch[key] = _matrix_to_xyzypr(_xyz6d_to_matrix(value[None, :]))[0]
            elif value.ndim == 2 and value.shape[1] == 9:
                batch[key] = _matrix_to_xyzypr(_xyz6d_to_matrix(value))
            else:
                raise ValueError(
                    f"XYZ6D_to_XYZYPR expects key '{key}' to have shape (9,) "
                    f"or (T, 9), got {value.shape}"
                )
        return batch

    def transform_batch(self, batch: dict) -> dict:
        """Vectorized: (B, 9) obs → (B, 6), (B, H, 9) chunks → (B, H, 6)."""
        for key in self.keys:
            value = np.asarray(batch[key])
            if value.ndim == 2 and value.shape[-1] == 9:
                batch[key] = _matrix_to_xyzypr(_xyz6d_to_matrix(value))  # (B, 6)
            elif value.ndim == 3 and value.shape[-1] == 9:
                B, H = value.shape[:2]
                flat = _matrix_to_xyzypr(_xyz6d_to_matrix(value.reshape(B * H, 9)))
                batch[key] = flat.reshape(B, H, 6)
            else:
                raise ValueError(
                    f"XYZ6D_to_XYZYPR.transform_batch: key '{key}' shape {value.shape} "
                    f"— expected (B, 9) or (B, H, 9)"
                )
        return batch


class CartesianWithGripperCoordinateTransform(Transform):
    def __init__(
        self,
        left_target_world: str,
        right_target_world: str,
        chunk_world: str,
        transformed_key_name: str,
        extra_batch_key: dict = None,
    ):
        """
        args:
            left_target_world: string key for left target world pose in batch (6D: xyz + ypr)
            right_target_world: string key for right target world pose in batch (6D: xyz + ypr)
            chunk_world: string key for chunk world pose in batch (14D: xyz + ypr + gripper * 2 arms)
            transformed_key_name: string key to store transformed chunk world in batch (14D)
        """
        self.left_target_world = left_target_world
        self.right_target_world = right_target_world
        self.chunk_world = chunk_world
        self.transformed_key_name = transformed_key_name
        self.extra_batch_key = extra_batch_key

    def transform(self, batch):
        """
        args:
            batch:
                left_target_world: numpy(6): xyz + ypr
                right_target_world: numpy(6): xyz + ypr
                chunk_world: numpy(T, 14): [left xyz+ypr+gripper, right xyz+ypr+gripper]
                transformed_key_name: str, name of the new key to store the transformed chunk world in

        returns
            batch with new key containing transformed chunk world in target frame: (T, 14)
        """
        batch.update(self.extra_batch_key or {})
        left_target_world = batch[self.left_target_world]
        right_target_world = batch[self.right_target_world]
        chunk_world = batch[self.chunk_world]

        if left_target_world.shape != (6,):
            raise ValueError(
                f"Expected left_target_world shape (6,), got {left_target_world.shape}"
            )
        if right_target_world.shape != (6,):
            raise ValueError(
                f"Expected right_target_world shape (6,), got {right_target_world.shape}"
            )
        if chunk_world.ndim != 2 or chunk_world.shape[1] != 14:
            raise ValueError(
                f"Expected chunk_world shape (T, 14), got {chunk_world.shape}"
            )

        # Chunk layout: [left xyz+ypr+gripper, right xyz+ypr+gripper]
        left_pose_world = chunk_world[:, :6]
        right_pose_world = chunk_world[:, 7:13]

        left_target_se3 = SE3.from_matrix(
            _xyzypr_to_matrix(left_target_world[None, :])[0]
        )
        right_target_se3 = SE3.from_matrix(
            _xyzypr_to_matrix(right_target_world[None, :])[0]
        )
        left_target_inv = left_target_se3.inverse()
        right_target_inv = right_target_se3.inverse()

        left_pose_in_target = _matrix_to_xyzypr(
            (
                left_target_inv @ SE3.from_matrix(_xyzypr_to_matrix(left_pose_world))
            ).to_matrix()
        )
        right_pose_in_target = _matrix_to_xyzypr(
            (
                right_target_inv @ SE3.from_matrix(_xyzypr_to_matrix(right_pose_world))
            ).to_matrix()
        )

        chunk_in_target_frame = np.empty_like(chunk_world)
        chunk_in_target_frame[:, :6] = left_pose_in_target
        chunk_in_target_frame[:, 6] = chunk_world[:, 6]  # left gripper unchanged
        chunk_in_target_frame[:, 7:13] = right_pose_in_target
        chunk_in_target_frame[:, 13] = chunk_world[:, 13]  # right gripper unchanged

        batch[self.transformed_key_name] = chunk_in_target_frame
        return batch


# ---------------------------------------------------------------------------
# Shape Transforms
# ---------------------------------------------------------------------------
class SplitKeys(Transform):
    def __init__(self, input_key: str, output_key_list: list[(str, int)]):
        self.input_key = input_key
        self.output_key_list = list(output_key_list)

    def transform(self, batch: dict) -> dict:
        prev_end = 0
        for key, size in self.output_key_list:
            batch[key] = batch[self.input_key][..., prev_end : prev_end + size]
            prev_end += size
        return batch


class ConcatKeys(Transform):
    def __init__(self, key_list, new_key_name, delete_old_keys=False):
        self.key_list = list(key_list)
        self.new_key_name = new_key_name
        self.delete_old_keys = delete_old_keys

    def transform(self, batch):
        arrays = [np.asarray(batch[k]) for k in self.key_list]
        try:
            batch[self.new_key_name] = np.concatenate(arrays, axis=-1)
        except ValueError as e:
            shapes = {k: np.asarray(batch[k]).shape for k in self.key_list}
            raise ValueError(
                f"ConcatKeys failed for keys {self.key_list} with shapes {shapes}"
            ) from e

        if self.delete_old_keys:
            for k in self.key_list:
                batch.pop(k, None)

        return batch

    transform_batch = transform


class Reshape(Transform):
    def __init__(self, input_key: str, output_key: str, shape: tuple):
        self.input_key = input_key
        self.output_key = output_key
        self.shape = shape

    def transform(self, batch: dict) -> dict:
        batch[self.output_key] = batch[self.input_key].reshape(*self.shape)
        return batch


def _pose_to_matrix_by_width(poses: np.ndarray) -> np.ndarray:
    """Dispatch pose→matrix by last-dim width: 7→xyzwxyz, 9→xyz6d, 6→xyzypr."""
    width = poses.shape[-1]
    if width == 7:
        return _xyzwxyz_to_matrix(poses)
    if width == 9:
        return _xyz6d_to_matrix(poses)
    if width == 6:
        return _xyzypr_to_matrix(poses)
    raise ValueError(f"Unsupported pose width {width} (expected 6, 7, or 9)")


def _matrix_to_pose_by_width(mats: np.ndarray, width: int) -> np.ndarray:
    if width == 7:
        return _matrix_to_xyzwxyz(mats)
    if width == 9:
        return _matrix_to_xyz6d(mats)
    if width == 6:
        return _matrix_to_xyzypr(mats)
    raise ValueError(f"Unsupported pose width {width} (expected 6, 7, or 9)")


def _se3_inverse_batch(mats: np.ndarray) -> np.ndarray:
    """Vectorized SE3 inverse: [[R,t],[0,1]]^-1 = [[R^T, -R^T t],[0,1]] for (T,4,4)."""
    R = mats[:, :3, :3]
    t = mats[:, :3, 3]
    Rt = np.transpose(R, (0, 2, 1))
    out = np.broadcast_to(np.eye(4, dtype=mats.dtype), mats.shape).copy()
    out[:, :3, :3] = Rt
    out[:, :3, 3] = -np.einsum("tij,tj->ti", Rt, t)
    return out


class SanitizeQuatPoseChunk(Transform):
    """Forward/backward-fill invalid rows (zero-norm or non-finite quats) in an
    xyzwxyz pose chunk.

    Mecka episodes contain tracking-dropout rows where obs_wrist_pose is all
    zeros; ``scipy.Rotation.from_quat`` raises on them ("Found zero norm
    quaternions"). Filling an invalid row with the previous valid pose makes the
    consecutive delta across it the identity (= no motion), which is the sane
    reading of "tracking lost, hold last pose". Leading invalid rows are
    backward-filled from the first valid row. Raises if the whole chunk is
    invalid (the dataset's retry-with-random-index fallback handles that).

    ``anchor_key``: optional single-pose key to overwrite with sanitized row 0,
    preserving the obs == chunk[0] anchor invariant.
    """

    def __init__(
        self, chunk_key: str, anchor_key: str | None = None, eps: float = 1e-6
    ):
        self.chunk_key = chunk_key
        self.anchor_key = anchor_key
        self.eps = float(eps)

    def transform(self, batch: dict) -> dict:
        chunk = np.asarray(batch[self.chunk_key]).copy()  # (T, 7)
        if chunk.ndim != 2 or chunk.shape[-1] != 7:
            raise ValueError(
                f"SanitizeQuatPoseChunk expects (T, 7) xyzwxyz, got {chunk.shape} "
                f"for key '{self.chunk_key}'"
            )
        quat = chunk[:, 3:7]
        valid = np.isfinite(chunk).all(axis=-1) & (
            np.linalg.norm(quat, axis=-1) > self.eps
        )
        if not valid.all():
            if not valid.any():
                raise ValueError(
                    f"SanitizeQuatPoseChunk: no valid pose rows in chunk "
                    f"'{self.chunk_key}' (all zero/non-finite)"
                )
            # forward-fill: index of the most recent valid row at each t
            idx = np.where(valid, np.arange(len(valid)), -1)
            idx = np.maximum.accumulate(idx)
            first_valid = int(np.argmax(valid))
            idx[idx < 0] = first_valid  # leading invalids -> backward-fill
            chunk = chunk[idx]
        batch[self.chunk_key] = chunk
        if self.anchor_key is not None:
            batch[self.anchor_key] = chunk[0].copy()
        return batch


class ConsecutiveDeltaChunk(Transform):
    """Pose chunk → step-wise (frame-to-frame) deltas: A_t = P_{t-1}^{-1} ∘ P_t.

    A_0 = identity. Output has the same representation/width as the input chunk.
    Unlike the single-anchor wrist-frame transform (all steps relative to P_0),
    this yields per-step velocities — the cumulative "distance since anchor"
    signal is removed at the representation level.
    """

    def __init__(self, chunk_key: str, output_key: str):
        self.chunk_key = chunk_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        chunk = np.asarray(batch[self.chunk_key])
        if chunk.ndim != 2:
            raise ValueError(
                f"ConsecutiveDeltaChunk expects (T, D), got {chunk.shape} for key "
                f"'{self.chunk_key}'"
            )
        width = chunk.shape[-1]
        mats = _pose_to_matrix_by_width(chunk)  # (T, 4, 4)
        deltas = _se3_inverse_batch(mats[:-1]) @ mats[1:]  # (T-1, 4, 4)
        eye = np.broadcast_to(np.eye(4, dtype=mats.dtype), (1, 4, 4))
        out = np.concatenate([eye, deltas], axis=0)  # (T, 4, 4), A_0 = I
        batch[self.output_key] = _matrix_to_pose_by_width(out, width)
        return batch


class PerTimestepCoordinateFrameTransform(Transform):
    """Express chunk_t in target_t's frame, per timestep.

    ``target_chunk``: (T, 6|7|9) pose per timestep. ``chunk``: (T, K, 3) points
    (mode="xyz"). inverse=True → target_t^{-1} ∘ chunk_t (world → target frame);
    inverse=False → target_t ∘ chunk_t (target frame → world/parent).
    Unlike ActionChunkCoordinateFrameTransform (one target for the whole chunk),
    the target here varies per timestep — e.g. fingertips in the wrist frame of
    the SAME timestep (pure articulation, no cumulative palm motion).
    """

    def __init__(
        self,
        target_chunk: str,
        chunk: str,
        transformed_key_name: str,
        mode: Literal["xyz"] = "xyz",
        inverse: bool = True,
    ):
        if mode != "xyz":
            raise ValueError(
                "PerTimestepCoordinateFrameTransform supports mode='xyz' only"
            )
        self.target_chunk = target_chunk
        self.chunk = chunk
        self.transformed_key_name = transformed_key_name
        self.inverse = inverse

    def transform(self, batch: dict) -> dict:
        target = np.asarray(batch[self.target_chunk])  # (T, 6|7|9)
        points = np.asarray(batch[self.chunk])  # (T, K, 3)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                f"PerTimestepCoordinateFrameTransform expects (T, K, 3) points, got "
                f"{points.shape} for key '{self.chunk}'"
            )
        if target.shape[0] != points.shape[0]:
            raise ValueError(
                f"target/chunk timestep mismatch: {target.shape[0]} vs {points.shape[0]}"
            )
        mats = _pose_to_matrix_by_width(target)  # (T, 4, 4)
        if self.inverse:
            mats = _se3_inverse_batch(mats)
        R = mats[:, :3, :3]
        t = mats[:, :3, 3]
        out = np.einsum("tij,tkj->tki", R, points) + t[:, None, :]
        batch[self.transformed_key_name] = out
        return batch


class CumulativeComposeChunk(Transform):
    """Chain-compose step-wise deltas back to absolute poses (revert for viz).

    ``P_t = anchor ∘ A_0 ∘ A_1 ∘ … ∘ A_t`` (A_0 is identity in the encoding, so
    P_0 = anchor). Inverse of ConsecutiveDeltaChunk given the anchor pose.
    Output uses the anchor's representation width.
    """

    def __init__(self, anchor_key: str, delta_key: str, output_key: str):
        self.anchor_key = anchor_key
        self.delta_key = delta_key
        self.output_key = output_key

    def transform(self, batch: dict) -> dict:
        anchor = np.asarray(batch[self.anchor_key])  # (6|7|9,)
        deltas = np.asarray(batch[self.delta_key])  # (T, 6|7|9)
        width = anchor.shape[-1]
        anchor_mat = _pose_to_matrix_by_width(anchor[None])[0]  # (4, 4)
        delta_mats = _pose_to_matrix_by_width(deltas)  # (T, 4, 4)
        out = np.empty_like(delta_mats)
        running = anchor_mat
        for i in range(delta_mats.shape[0]):
            running = running @ delta_mats[i]
            out[i] = running
        batch[self.output_key] = _matrix_to_pose_by_width(out, width)
        return batch


class ArcLengthResampleChunks(Transform):
    """Resample chunks at equal ARC-LENGTH spacing over a fixed path distance.

    Distance-based action sampling: instead of a chunk covering a fixed number of
    frames, it covers a fixed travelled distance (``total_distance`` metres),
    resampled to ``num_samples`` points equally spaced along the path — so speed
    and duration are erased and the spatial SHAPE of the motion is retained
    (the same idea as ActionNorms' arc_length resample, applied to raw chunks).

    The metric is the COMBINED path of ``distance_keys``: the R^{3n} norm of the
    concatenated translations' per-step deltas (for both wrists: sqrt(|dL|²+|dR|²)).
    One shared time-warp keeps all keys synchronized. ``pose_keys`` ((T,7) xyzwxyz)
    are resampled with xyz linear-in-s and rotation slerp-in-s; ``point_keys``
    ((T,K,3)) linearly-in-s. Zero-motion plateaus (pauses, repeat-padded rows) add
    no distance and are deduped, so end-of-episode padding is harmless.

    Raises ValueError when the window covers less than ``total_distance`` (a
    still/pause anchor with no 30 cm of shape) — the dataset's retry-with-random-
    index fallback then picks a new anchor. Run zero-quat sanitization BEFORE this
    transform: an all-zero dropout row would otherwise fake a huge jump to the
    origin and corrupt the metric.
    """

    def __init__(
        self,
        distance_keys: list[str],
        pose_keys: list[str],
        point_keys: list[str],
        total_distance: float,
        num_samples: int,
    ):
        self.distance_keys = list(distance_keys)
        self.pose_keys = list(pose_keys)
        self.point_keys = list(point_keys)
        self.total_distance = float(total_distance)
        self.num_samples = int(num_samples)

    def transform(self, batch: dict) -> dict:
        # Combined per-step travelled distance over the distance keys' translations.
        xyz = np.concatenate(
            [np.asarray(batch[k], dtype=np.float64)[:, :3] for k in self.distance_keys],
            axis=-1,
        )  # (T, 3n)
        ds = np.linalg.norm(np.diff(xyz, axis=0), axis=-1)  # (T-1,)
        s = np.concatenate([[0.0], np.cumsum(ds)])  # (T,)
        if s[-1] < self.total_distance:
            raise ValueError(
                f"ArcLengthResampleChunks: window covers {s[-1]:.3f} m "
                f"< required {self.total_distance:.3f} m (still/pause anchor)"
            )

        # Equal-arc-length query positions and a strictly-increasing key grid
        # (dedupe zero-motion steps for interp/slerp validity).
        u = np.linspace(0.0, self.total_distance, self.num_samples)
        keep = np.concatenate([[True], ds > 1e-12])
        s_k = s[keep]

        for k in self.pose_keys:
            chunk = np.asarray(batch[k], dtype=np.float64)[keep]  # (Tk, 7)
            xyz_i = np.stack([np.interp(u, s_k, chunk[:, d]) for d in range(3)], axis=1)
            rots = R.from_quat(chunk[:, [4, 5, 6, 3]])  # wxyz -> xyzw
            quat_i = Slerp(s_k, rots)(u).as_quat()  # (num_samples, 4) xyzw
            batch[k] = np.concatenate(
                [xyz_i, quat_i[:, [3, 0, 1, 2]]], axis=1
            )  # (num_samples, 7) xyzwxyz

        for k in self.point_keys:
            pts = np.asarray(batch[k], dtype=np.float64)[keep]  # (Tk, K, 3)
            flat = pts.reshape(len(s_k), -1)
            out = np.stack(
                [np.interp(u, s_k, flat[:, d]) for d in range(flat.shape[1])], axis=1
            )
            batch[k] = out.reshape(self.num_samples, *pts.shape[1:])

        return batch


class SelectKeypoints(Transform):
    """Select specific keypoint indices from a flattened ``(..., n_keypoints*item_dim)`` array.

    Reshapes ``(..., n_keypoints, item_dim)`` and gathers ``indices`` along the
    keypoint axis, returning ``(..., len(indices), item_dim)``. Used to pull the 5
    MANO fingertips (indices 4/8/12/16/20) out of the 21-keypoint (63-dim) vector.
    """

    def __init__(self, input_key, output_key, indices, n_keypoints=21, item_dim=3):
        self.input_key = input_key
        self.output_key = output_key
        self.indices = list(indices)
        self.n_keypoints = int(n_keypoints)
        self.item_dim = int(item_dim)

    def transform(self, batch: dict) -> dict:
        arr = np.asarray(batch[self.input_key])
        lead = arr.shape[:-1]
        arr = arr.reshape(*lead, self.n_keypoints, self.item_dim)
        batch[self.output_key] = arr[..., self.indices, :]
        return batch

    transform_batch = transform


# ---------------------------------------------------------------------------
# Type Transforms
# ---------------------------------------------------------------------------


class NumpyToTensor(Transform):
    def __init__(self, keys: list[str]):
        self.keys = keys

    def transform(self, batch: dict) -> dict:
        for key in self.keys:
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.from_numpy(batch[key])
            elif isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].clone()
            else:
                raise ValueError(
                    f"NumpyToTensor expects key '{key}' to be a numpy array or torch tensor, got {type(batch[key])}"
                )
        return batch
