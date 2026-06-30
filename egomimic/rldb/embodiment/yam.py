"""I2RT YAM bimanual robot embodiment.

YAM is a bimanual arm robot with the *same* data conventions as Eva:
per-arm end-effector pose + gripper, a front camera plus two wrist cameras, and
a cartesian "wrist-frame" action representation. It therefore reuses Eva's
transform pipeline and keymap verbatim — the only robot-specific difference is
the camera->base extrinsics, so this subclass simply swaps the ``extrinsics_key``
to ``"yam"`` (see ``egomimic.utils.egomimicUtils.EXTRINSICS["yam"]``).

This is deliberately distinct from ``Mecka`` (which is a Human-derived,
egocentric/keypoint embodiment), because YAM is a real bimanual robot like Eva.
"""

from __future__ import annotations

from typing import Literal

from egomimic.rldb.embodiment.eva import (
    Eva,
    _build_eva_bimanual_eef_frame_transform_list,
    _build_eva_bimanual_transform_list,
)
from egomimic.rldb.zarr.action_chunk_transforms import Transform


class Yam(Eva):
    # Camera->base extrinsics key for the YAM rig (egomimicUtils.EXTRINSICS).
    EXTRINSICS_KEY = "yam"
    # Intrinsics for action-overlay visualization: the Atlas front camera's
    # rectified-pinhole intrinsics (egomimicUtils.INTRINSICS["yam"]).
    VIZ_INTRINSICS_KEY = "yam"

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal[
            "cartesian",
            "cartesian_6d",
            "cartesian_wristframe_ypr",
            "cartesian_wristframe_6d",
            "cartesian_wristframe_quat",
        ],
    ) -> list[Transform]:
        # Same pipeline as Eva, but with the YAM extrinsics. The ``_6d`` modes
        # emit the continuous 6D rotation representation (so per-dimension
        # normalization runs on the 6D columns, not wrapped Euler angles); see
        # the rot_repr="6d" branches in the Eva builders.
        if mode == "cartesian":
            return _build_eva_bimanual_transform_list(
                is_quat=True, extrinsics_key=cls.EXTRINSICS_KEY
            )
        elif mode == "cartesian_6d":
            return _build_eva_bimanual_transform_list(
                is_quat=True, rot_repr="6d", extrinsics_key=cls.EXTRINSICS_KEY
            )
        elif mode == "cartesian_wristframe_ypr":
            return _build_eva_bimanual_eef_frame_transform_list(
                is_quat=False, extrinsics_key=cls.EXTRINSICS_KEY
            )
        elif mode == "cartesian_wristframe_6d":
            return _build_eva_bimanual_eef_frame_transform_list(
                rot_repr="6d", extrinsics_key=cls.EXTRINSICS_KEY
            )
        elif mode == "cartesian_wristframe_quat":
            return _build_eva_bimanual_eef_frame_transform_list(
                is_quat=True, extrinsics_key=cls.EXTRINSICS_KEY
            )
        raise ValueError(f"Unsupported mode '{mode}'")

    # _get_keymap and dinov3_keymap are inherited from Eva — YAM zarr episodes
    # use the same keys (images.front_1 / left_wrist / right_wrist, left/right
    # .obs_ee_pose / .obs_gripper / .cmd_ee_pose / .cmd_gripper).
