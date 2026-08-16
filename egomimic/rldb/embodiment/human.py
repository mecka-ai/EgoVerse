from __future__ import annotations

from abc import abstractmethod
from typing import Literal

from egomimic.rldb.embodiment.embodiment import Embodiment
from egomimic.rldb.zarr.action_chunk_transforms import (
    ActionChunkCoordinateFrameTransform,
    BatchQuaternionPoseToYPR,
    ConcatKeys,
    ConsecutiveDeltaChunk,
    CumulativeComposeChunk,
    DeleteKeys,
    InterpolatePose,
    PadActionGripper,
    PerTimestepCoordinateFrameTransform,
    PoseCoordinateFrameTransform,
    QuaternionPoseToYPR,
    Reshape,
    SanitizeQuatPoseChunk,
    SelectKeypoints,
    SplitKeys,
    Transform,
    XYZ6D_to_XYZYPR,
    XYZWXYZ_to_XYZ6D,
    XYZWXYZ_to_XYZYPR,
)
from egomimic.utils.viz_utils import (
    ColorPalette,
    _viz_fingertips,
    _viz_keypoints,
)


class Human(Embodiment):
    ACTION_STRIDE = 3

    @classmethod
    def viz(
        cls,
        image,
        viz_data,
        mode=Literal[
            "traj", "traj+rotation", "axes", "annotations", "keypoints", "fingertips"
        ],
        intrinsics_key=None,
        **kwargs,
    ):
        if mode == "fingertips":
            intrinsics_key = intrinsics_key or cls.VIZ_INTRINSICS_KEY
            return _viz_fingertips(
                image=image, actions=viz_data, intrinsics_key=intrinsics_key, **kwargs
            )
        if mode == "keypoints":
            intrinsics_key = intrinsics_key or cls.VIZ_INTRINSICS_KEY
            color = kwargs.get("color", None)
            if color is not None and ColorPalette.is_valid(color):
                n = len(cls.FINGER_COLORS)
                colors = {
                    finger: ColorPalette.to_rgb(color, value=(i + 1) / (n + 1))
                    for i, finger in enumerate(cls.FINGER_COLORS)
                }
                dot_color = ColorPalette.to_rgb(color, value=0.7)
            else:
                colors = cls.FINGER_COLORS
                dot_color = cls.DOT_COLOR
            return _viz_keypoints(
                image=image,
                actions=viz_data,
                intrinsics_key=intrinsics_key,
                edges=cls.FINGER_EDGES,
                edge_ranges=cls.FINGER_EDGE_RANGES,
                colors=colors,
                dot_color=dot_color,
                **kwargs,
            )
        return super().viz(
            image, viz_data, mode=mode, intrinsics_key=intrinsics_key, **kwargs
        )

    @abstractmethod
    def _get_keymap(
        cls, mode: Literal["cartesian", "keypoints"], annotation_key: str = None
    ):
        pass

    @abstractmethod
    def get_transform_list(
        cls,
        mode: str,
    ) -> list[Transform]:
        pass


class Aria(Human):
    VIZ_INTRINSICS_KEY = "base"
    ACTION_STRIDE = 3
    FINGER_EDGES = [
        (
            5,
            6,
        ),
        (6, 7),
        (7, 0),  # thumb
        (5, 8),
        (8, 9),
        (9, 10),
        (9, 1),  # index
        (5, 11),
        (11, 12),
        (12, 13),
        (13, 2),  # middle
        (5, 14),
        (14, 15),
        (15, 16),
        (16, 3),  # ring
        (5, 17),
        (17, 18),
        (18, 19),
        (19, 4),  # pinky
    ]
    FINGER_COLORS = {
        "thumb": (255, 100, 100),  # red
        "index": (100, 255, 100),  # green
        "middle": (100, 100, 255),  # blue
        "ring": (255, 255, 100),  # yellow
        "pinky": (255, 100, 255),  # magenta
    }
    FINGER_EDGE_RANGES = [
        ("thumb", 0, 3),
        ("index", 3, 7),
        ("middle", 7, 11),
        ("ring", 11, 15),
        ("pinky", 15, 19),
    ]
    DOT_COLOR = (255, 165, 0)

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal[
            "cartesian",
            "keypoints_headframe_ypr",
            "keypoints_headframe_quat",
            "keypoints_wristframe_ypr",
            "keypoints_wristframe_quat",
        ],
    ) -> list[Transform]:
        if mode == "cartesian":
            return _build_aria_cartesian_bimanual_transform_list(
                stride=cls.ACTION_STRIDE
            )
        elif mode == "keypoints_headframe_ypr":
            return _build_aria_keypoints_bimanual_transform_list(
                stride=cls.ACTION_STRIDE, is_quat=False
            )
        elif mode == "keypoints_headframe_quat":
            return _build_aria_keypoints_bimanual_transform_list(
                stride=cls.ACTION_STRIDE, is_quat=True
            )
        elif mode == "keypoints_wristframe_ypr":
            return _build_aria_keypoints_eef_frame_transform_list(
                stride=cls.ACTION_STRIDE, is_quat=False
            )
        elif mode == "keypoints_wristframe_quat":
            return _build_aria_keypoints_eef_frame_transform_list(
                stride=cls.ACTION_STRIDE, is_quat=True
            )

    @classmethod
    def _get_keymap(
        cls,
        keymap_mode: Literal["cartesian", "keypoints"],
    ):
        if keymap_mode == "cartesian":
            return {
                cls.VIZ_IMAGE_KEY: {
                    "key_type": "camera_keys",
                    "zarr_key": "images.front_1",
                },
                "right.action_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "right.obs_ee_pose",
                    "horizon": 30,
                },
                "left.action_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "left.obs_ee_pose",
                    "horizon": 30,
                },
                "right.obs_ee_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_ee_pose",
                },
                "left.obs_ee_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_ee_pose",
                },
                "obs_head_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "obs_head_pose",
                },
            }
        elif keymap_mode == "keypoints":
            return {
                cls.VIZ_IMAGE_KEY: {
                    "key_type": "camera_keys",
                    "zarr_key": "images.front_1",
                },
                "left.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": "left.obs_keypoints",
                    "horizon": 30,
                },
                "right.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": "right.obs_keypoints",
                    "horizon": 30,
                },
                "left.action_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_wrist_pose",
                    "horizon": 30,
                },
                "right.action_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_wrist_pose",
                    "horizon": 30,
                },
                "left.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_keypoints",
                },
                "right.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_keypoints",
                },
                "left.obs_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_wrist_pose",
                },
                "right.obs_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_wrist_pose",
                },
                "obs_head_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "obs_head_pose",
                },
            }


class Scale(Human):
    VIZ_INTRINSICS_KEY = "scale"
    ACTION_STRIDE = 1

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal["cartesian",],
    ) -> list[Transform]:
        if mode == "cartesian":
            return _build_aria_cartesian_bimanual_transform_list(
                stride=cls.ACTION_STRIDE,
            )

    @classmethod
    def _get_keymap(
        cls,
        keymap_mode: Literal["cartesian", "keypoints"],
    ):
        if keymap_mode == "cartesian":
            return {
                cls.VIZ_IMAGE_KEY: {
                    "key_type": "camera_keys",
                    "zarr_key": "images.front_1",
                },
                "right.action_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "right.obs_ee_pose",
                    "horizon": 30,
                },
                "left.action_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "left.obs_ee_pose",
                    "horizon": 30,
                },
                "right.obs_ee_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_ee_pose",
                },
                "left.obs_ee_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_ee_pose",
                },
            }
        elif keymap_mode == "keypoints":
            return {
                cls.VIZ_IMAGE_KEY: {
                    "key_type": "camera_keys",
                    "zarr_key": "images.front_1",
                },
                "left.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": "left.obs_keypoints",
                    "horizon": 30,
                },
                "right.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": "right.obs_keypoints",
                    "horizon": 30,
                },
                "left.action_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_wrist_pose",
                    "horizon": 30,
                },
                "right.action_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_wrist_pose",
                    "horizon": 30,
                },
                "left.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_keypoints",
                },
                "right.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_keypoints",
                },
                "left.obs_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_wrist_pose",
                },
                "right.obs_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_wrist_pose",
                },
            }


class Mecka(Human):
    VIZ_INTRINSICS_KEY = "mecka"
    ACTION_STRIDE = 1

    @classmethod
    def get_transform_list(
        cls,
        mode: Literal[
            "cartesian",
            "cartesian_wristframe_6d",
            "cartesian_wristframe_6d_fingertips_nointerp",
        ],
        pad_gripper: bool = False,
    ) -> list[Transform]:
        if mode == "cartesian":
            tl = _build_aria_cartesian_bimanual_transform_list(
                stride=cls.ACTION_STRIDE,
            )
            if pad_gripper:
                # Cotrain with a 14-D shared head: pad human 12-D actions to the
                # robot's 14-D [L6, gripper, R6, gripper] layout at the data level so
                # GT stays 14-D (aligned with 14-D predictions in loss + val metrics).
                tl.append(PadActionGripper(action_key="actions_cartesian"))
            return tl
        elif mode == "cartesian_wristframe_6d":
            # The pi-6d 6D wrist-frame representation: each arm's ee-pose action chunk
            # expressed relative to the current ee pose (single-anchor delta), rotation
            # as continuous 6D columns → 9 per arm, 18 bimanual.
            return _build_mecka_cartesian_wristframe_bimanual_transform_list(
                stride=cls.ACTION_STRIDE,
                rot_repr="6d",
            )
        elif mode == "cartesian_wristframe_6d_fingertips_nointerp":
            # 6D wrist-frame palm pose (9) + 5 MANO fingertips (indices 4/8/12/16/20) in
            # the wrist frame (15) per hand → 24 per hand, 48 bimanual. No resampling:
            # reads a full 100-frame horizon of real frames.
            return _build_mecka_wristframe_6d_fingertips_transform_list(
                interpolate=False,
            )
        elif mode == "cartesian_wristframe_6d_fingertips_stepwise":
            # v2: STEP-WISE palm deltas (frame-to-frame velocities, 6D) + fingertips in
            # the wrist frame of the SAME timestep (pure articulation). 48 bimanual,
            # no resampling. Kills the cumulative-displacement signal.
            return _build_mecka_wf6d_fingertips_stepwise_transform_list()

    # ----------------------------------------------------------------------
    # WAM (World-Action Model) keymap/transform: a frame CLIP + frame-aligned
    # action/state ee-pose chunks (world -> head frame), left-first 12-dim. The
    # Wan VAE compresses 4x temporally, so cam_horizon = 4*k+1 pixel frames ->
    # k+1 latent frames -> k predicted blocks; action_horizon = npb*k; state=k.
    # ----------------------------------------------------------------------
    @classmethod
    def get_wam_keymap(
        cls,
        cam_horizon: int = 17,
        action_horizon: int = 16,
        state_horizon: int = 4,
        norm_mode: bool = False,
        annotation_key=None,
    ):
        key_map = {
            cls.VIZ_IMAGE_KEY: {
                "key_type": "camera_keys",
                "zarr_key": "images.front_1",
                "horizon": cam_horizon,
            },
            "right.action_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "right.obs_ee_pose",
                "horizon": action_horizon,
            },
            "left.action_ee_pose": {
                "key_type": "action_keys",
                "zarr_key": "left.obs_ee_pose",
                "horizon": action_horizon,
            },
            "right.state_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "right.obs_ee_pose",
                "horizon": state_horizon,
            },
            "left.state_ee_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "left.obs_ee_pose",
                "horizon": state_horizon,
            },
            # Current head pose (single, xyz+quat) — target frame for the
            # head/camera-frame transform; deleted after the transform runs.
            "obs_head_pose": {
                "key_type": "proprio_keys",
                "zarr_key": "obs_head_pose",
            },
            # Chunked read of the same zarr key so the batch also carries the
            # per-raw-frame head pose across the whole camera-clip window.
            # ``key_type=metadata_keys`` keeps it out of the schematic's
            # proprio registry (so WAM._to_wam_data doesn't append it to
            # ``data["state"]`` and change model input) and out of norm
            # inference. Used at viz time (WAMEvalVideo) to re-project each
            # action into the head pose CURRENT at the displayed pixel —
            # otherwise the single-pose head reference at frame 0 drifts as
            # the head moves and the overlay walks off the hand.
            "obs_head_pose_chunk": {
                "key_type": "metadata_keys",
                "zarr_key": "obs_head_pose",
                "horizon": cam_horizon,
            },
        }
        if norm_mode:  # norm stats: drop the image clip (camera) key
            for k in [
                k for k, v in key_map.items() if v.get("key_type") == "camera_keys"
            ]:
                del key_map[k]
        return key_map

    @classmethod
    def get_wam_transform_list(cls):
        # Raw ee-poses are stored in the WORLD frame, so we reference both the
        # action and state chunks to the current head pose (obs_head_pose) ->
        # head/camera frame, exactly like the VLA cartesian pipeline
        # (_build_human_cartesian_bimanual_transform_list). This is REQUIRED:
        # the viz projects with intrinsics only (extrinsics=None), so points
        # must be in the camera frame to land on the hands. We use
        # ActionChunkCoordinateFrameTransform for BOTH (state is a chunk here,
        # not a single pose) and SKIP InterpolatePose to keep the action/state
        # horizons frame-aligned with the video clip. Then quat -> ypr
        # (xyzwxyz 7 -> xyzypr 6) and concat L+R left-first -> 12-dim (6/arm),
        # the layout _split_action_pose expects.
        return [
            ActionChunkCoordinateFrameTransform(
                target_world="obs_head_pose",
                chunk_world="left.action_ee_pose",
                transformed_key_name="left.action_ee_pose_hf",
                mode="xyzwxyz",
            ),
            ActionChunkCoordinateFrameTransform(
                target_world="obs_head_pose",
                chunk_world="right.action_ee_pose",
                transformed_key_name="right.action_ee_pose_hf",
                mode="xyzwxyz",
            ),
            ActionChunkCoordinateFrameTransform(
                target_world="obs_head_pose",
                chunk_world="left.state_ee_pose",
                transformed_key_name="left.state_ee_pose_hf",
                mode="xyzwxyz",
            ),
            ActionChunkCoordinateFrameTransform(
                target_world="obs_head_pose",
                chunk_world="right.state_ee_pose",
                transformed_key_name="right.state_ee_pose_hf",
                mode="xyzwxyz",
            ),
            XYZWXYZ_to_XYZYPR(
                keys=[
                    "left.action_ee_pose_hf",
                    "right.action_ee_pose_hf",
                    "left.state_ee_pose_hf",
                    "right.state_ee_pose_hf",
                ]
            ),
            ConcatKeys(
                ["left.action_ee_pose_hf", "right.action_ee_pose_hf"],
                "actions_cartesian",
                delete_old_keys=True,
            ),
            ConcatKeys(
                # Batch keys are ZARR keys; the data_schematic maps mecka proprio
                # key_name "ee_pose" -> zarr_key "observations.state.ee_pose".
                # Norm-stats (keyname_to_zarr_key) and process_batch
                # (zarr_key_to_keyname) both key off that zarr_key, so the
                # concatenated head-frame state must be named accordingly, or no
                # proprio matches (KeyError 'state' in _prep_state_action).
                ["left.state_ee_pose_hf", "right.state_ee_pose_hf"],
                "observations.state.ee_pose",
                delete_old_keys=True,
            ),
            # Drop the raw world-frame keys: ActionChunkCoordinateFrameTransform
            # COPIES into the _hf keys (consumed by ConcatKeys above) but leaves
            # the originals behind. _to_wam_data concatenates ALL proprio keys
            # into the state, so stray raw left/right.state_ee_pose (7-dim each)
            # would inflate state 12 -> 26. Delete them (+ obs_head_pose target).
            DeleteKeys(
                keys_to_delete=[
                    "obs_head_pose",
                    "left.action_ee_pose",
                    "right.action_ee_pose",
                    "left.state_ee_pose",
                    "right.state_ee_pose",
                ]
            ),
        ]

    @classmethod
    def get_keymap(
        cls,
        mode: Literal[
            "cartesian",
            "cartesian_wristframe_6d",
            "cartesian_wristframe_6d_fingertips_nointerp",
            "cartesian_wristframe_6d_fingertips_stepwise",
            "keypoints",
        ],
        annotations: bool = False,
        norm_mode: bool = False,
    ):
        # The 6D+fingertips modes need the keypoints raw keys (wrist pose + keypoints
        # chunks) at a full 100-frame horizon of real frames, routed to the
        # keypoints keymap.
        kpts_horizon = 30
        if mode in (
            "cartesian_wristframe_6d_fingertips_nointerp",
            "cartesian_wristframe_6d_fingertips_stepwise",
        ):
            kpts_horizon = 100
            mode = "keypoints"
        # cartesian + cartesian_wristframe_6d consume the same raw keys (action/obs
        # ee poses + head pose); only the transform frame differs.
        if mode in ("cartesian", "cartesian_wristframe_6d"):
            action_horizon = 30
            key_map = {
                cls.VIZ_IMAGE_KEY: {
                    "key_type": "camera_keys",
                    "zarr_key": "images.front_1",
                },
                "right.action_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "right.obs_ee_pose",
                    "horizon": action_horizon,
                },
                "left.action_ee_pose": {
                    "key_type": "action_keys",
                    "zarr_key": "left.obs_ee_pose",
                    "horizon": action_horizon,
                },
                "right.obs_ee_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_ee_pose",
                },
                "left.obs_ee_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_ee_pose",
                },
                "obs_head_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "obs_head_pose",
                },
            }
        elif mode == "keypoints":
            key_map = {
                cls.VIZ_IMAGE_KEY: {
                    "key_type": "camera_keys",
                    "zarr_key": "images.front_1",
                },
                "left.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": "left.obs_keypoints",
                    "horizon": 30,
                },
                "right.action_keypoints": {
                    "key_type": "action_keys",
                    "zarr_key": "right.obs_keypoints",
                    "horizon": 30,
                },
                "left.action_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_wrist_pose",
                    "horizon": 30,
                },
                "right.action_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_wrist_pose",
                    "horizon": 30,
                },
                "left.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_keypoints",
                },
                "right.obs_keypoints": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_keypoints",
                },
                "left.obs_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "left.obs_wrist_pose",
                },
                "right.obs_wrist_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "right.obs_wrist_pose",
                },
                "obs_head_pose": {
                    "key_type": "proprio_keys",
                    "zarr_key": "obs_head_pose",
                },
            }
        else:
            raise ValueError(
                f"Unsupported mode '{mode}'. Expected one of: 'cartesian', 'keypoints'."
            )
        # No-interp fingertips reads a full 100-frame horizon (no resampling), so the
        # keypoint/wrist action chunks are read at kpts_horizon instead of 30.
        if kpts_horizon != 30:
            for _k in (
                "left.action_keypoints",
                "right.action_keypoints",
                "left.action_wrist_pose",
                "right.action_wrist_pose",
            ):
                if _k in key_map:
                    key_map[_k] = {**key_map[_k], "horizon": kpts_horizon}
        if annotations:
            key_map["annotations"] = {
                "key_type": "annotation_keys",
                "zarr_key": "annotations",
            }
        if norm_mode:
            key_map = {
                k: v
                for k, v in key_map.items()
                if v.get("key_type") not in ("camera_keys", "annotation_keys")
            }
        return key_map


# this works for quat and ypr since actionChunkCoordinateFrameTransform works for both
def _build_aria_keypoints_revert_eef_frame_transform_list(
    *,
    action_key: str = "actions_keypoints",
    obs_key: str = "observations.state.keypoints",
    left_keypoints_action_wristframe: str = "left.action_keypoints_wristframe",
    right_keypoints_action_wristframe: str = "right.action_keypoints_wristframe",
    left_wrist_obs_headframe: str = "left.obs_wrist_pose_headframe",
    right_wrist_obs_headframe: str = "right.obs_wrist_pose_headframe",
    left_wrist_action_headframe: str = "left.action_wrist_pose_headframe",
    right_wrist_action_headframe: str = "right.action_wrist_pose_headframe",
    left_wrist_action_wristframe: str = "left.action_wrist_pose_wristframe",
    right_wrist_action_wristframe: str = "right.action_wrist_pose_wristframe",
    left_keypoints_action_headframe: str = "left.action_keypoints_headframe",
    right_keypoints_action_headframe: str = "right.action_keypoints_headframe",
    left_keypoints_obs_wristframe: str = "left.obs_keypoints_wristframe",
    right_keypoints_obs_wristframe: str = "right.obs_keypoints_wristframe",
    is_quat: bool = True,
) -> list[Transform]:
    if is_quat:
        pose_shape = 7
    else:
        pose_shape = 6
    transform_list = [
        SplitKeys(
            input_key=obs_key,
            output_key_list=[
                (left_wrist_obs_headframe, pose_shape),
                (left_keypoints_obs_wristframe, 63),
                (right_wrist_obs_headframe, pose_shape),
                (right_keypoints_obs_wristframe, 63),
            ],
        ),
        SplitKeys(
            input_key=action_key,
            output_key_list=[
                (left_wrist_action_wristframe, pose_shape),
                (left_keypoints_action_wristframe, 63),
                (right_wrist_action_wristframe, pose_shape),
                (right_keypoints_action_wristframe, 63),
            ],
        ),
        Reshape(
            input_key=left_keypoints_action_wristframe,
            output_key=left_keypoints_action_wristframe,
            shape=(100, 21, 3),
        ),
        Reshape(
            input_key=right_keypoints_action_wristframe,
            output_key=right_keypoints_action_wristframe,
            shape=(100, 21, 3),
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=left_wrist_obs_headframe,
            chunk_world=left_keypoints_action_wristframe,
            transformed_key_name=left_keypoints_action_headframe,
            mode="xyz",
            inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_wrist_obs_headframe,
            chunk_world=right_keypoints_action_wristframe,
            transformed_key_name=right_keypoints_action_headframe,
            mode="xyz",
            inverse=False,
        ),
        Reshape(
            input_key=left_keypoints_action_headframe,
            output_key=left_keypoints_action_headframe,
            shape=(100, 63),
        ),
        Reshape(
            input_key=right_keypoints_action_headframe,
            output_key=right_keypoints_action_headframe,
            shape=(100, 63),
        ),
        ConcatKeys(
            key_list=[
                left_keypoints_action_headframe,
                right_keypoints_action_headframe,
            ],
            new_key_name=action_key,
            delete_old_keys=True,
        ),
    ]
    return transform_list


def _build_aria_keypoints_eef_frame_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    left_keypoints_action_world: str = "left.action_keypoints",
    right_keypoints_action_world: str = "right.action_keypoints",
    left_keypoints_obs_pose: str = "left.obs_keypoints",
    right_keypoints_obs_pose: str = "right.obs_keypoints",
    left_keypoints_action_headframe: str = "left.action_keypoints_headframe",
    right_keypoints_action_headframe: str = "right.action_keypoints_headframe",
    left_keypoints_obs_headframe: str = "left.obs_keypoints_headframe",
    right_keypoints_obs_headframe: str = "right.obs_keypoints_headframe",
    left_wrist_action_world: str = "left.action_wrist_pose",
    right_wrist_action_world: str = "right.action_wrist_pose",
    left_keypoints_action_wristframe: str = "left.action_keypoints_wristframe",
    right_keypoints_action_wristframe: str = "right.action_keypoints_wristframe",
    left_wrist_action_wristframe: str = "left.action_wrist_pose_wristframe",
    right_wrist_action_wristframe: str = "right.action_wrist_pose_wristframe",
    left_wrist_obs_pose: str = "left.obs_wrist_pose",
    right_wrist_obs_pose: str = "right.obs_wrist_pose",
    left_wrist_action_headframe: str = "left.action_wrist_pose_headframe",
    right_wrist_action_headframe: str = "right.action_wrist_pose_headframe",
    left_wrist_obs_headframe: str = "left.obs_wrist_pose_headframe",
    right_wrist_obs_headframe: str = "right.obs_wrist_pose_headframe",
    left_keypoints_obs_wristframe: str = "left.obs_keypoints_wristframe",
    right_keypoints_obs_wristframe: str = "right.obs_keypoints_wristframe",
    delete_target_world: bool = True,
    chunk_length: int = 100,
    stride: int = 3,
    is_quat: bool = True,
) -> list[Transform]:
    transform_list = _build_aria_keypoints_bimanual_transform_list(
        target_world=target_world,
        target_world_ypr=target_world_ypr,
        target_world_is_quat=target_world_is_quat,
        delete_target_world=delete_target_world,
        chunk_length=chunk_length,
        stride=stride,
        concat_keys=False,
        is_quat=True,
    )
    delete_keys = [
        left_keypoints_action_world,
        right_keypoints_action_world,
        left_keypoints_obs_pose,
        right_keypoints_obs_pose,
        left_wrist_action_world,
        right_wrist_action_world,
        left_wrist_obs_pose,
        right_wrist_obs_pose,
        left_keypoints_action_headframe,
        right_keypoints_action_headframe,
        left_keypoints_obs_headframe,
        right_keypoints_obs_headframe,
        left_wrist_action_headframe,
        right_wrist_action_headframe,
    ]
    if delete_target_world:
        delete_keys.append(target_world)
        if target_world_is_quat:
            delete_keys.append(target_world_ypr)
    transform_list.extend(
        [
            Reshape(
                input_key=left_keypoints_action_headframe,
                output_key=left_keypoints_action_headframe,
                shape=(chunk_length, 21, 3),
            ),
            Reshape(
                input_key=right_keypoints_action_headframe,
                output_key=right_keypoints_action_headframe,
                shape=(chunk_length, 21, 3),
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=left_wrist_obs_headframe,
                chunk_world=left_keypoints_action_headframe,
                transformed_key_name=left_keypoints_action_wristframe,
                mode="xyz",
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=right_wrist_obs_headframe,
                chunk_world=right_keypoints_action_headframe,
                transformed_key_name=right_keypoints_action_wristframe,
                mode="xyz",
            ),
            Reshape(
                input_key=left_keypoints_action_wristframe,
                output_key=left_keypoints_action_wristframe,
                shape=(chunk_length, 63),
            ),
            Reshape(
                input_key=right_keypoints_action_wristframe,
                output_key=right_keypoints_action_wristframe,
                shape=(chunk_length, 63),
            ),
            Reshape(
                input_key=left_keypoints_obs_headframe,
                output_key=left_keypoints_obs_headframe,
                shape=(21, 3),
            ),
            Reshape(
                input_key=right_keypoints_obs_headframe,
                output_key=right_keypoints_obs_headframe,
                shape=(21, 3),
            ),
            PoseCoordinateFrameTransform(
                target_world=left_wrist_obs_headframe,
                pose_world=left_keypoints_obs_headframe,
                transformed_key_name=left_keypoints_obs_wristframe,
                mode="xyz",
            ),
            PoseCoordinateFrameTransform(
                target_world=right_wrist_obs_headframe,
                pose_world=right_keypoints_obs_headframe,
                transformed_key_name=right_keypoints_obs_wristframe,
                mode="xyz",
            ),
            Reshape(
                input_key=left_keypoints_obs_wristframe,
                output_key=left_keypoints_obs_wristframe,
                shape=(63,),
            ),
            Reshape(
                input_key=right_keypoints_obs_wristframe,
                output_key=right_keypoints_obs_wristframe,
                shape=(63,),
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=left_wrist_obs_headframe,
                chunk_world=left_wrist_action_headframe,
                transformed_key_name=left_wrist_action_wristframe,
                mode="xyzwxyz",
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=right_wrist_obs_headframe,
                chunk_world=right_wrist_action_headframe,
                transformed_key_name=right_wrist_action_wristframe,
                mode="xyzwxyz",
            ),
        ]
    )
    if not is_quat:
        transform_list.extend(
            [
                BatchQuaternionPoseToYPR(
                    pose_key=left_wrist_action_wristframe,
                    output_key=left_wrist_action_wristframe,
                ),
                BatchQuaternionPoseToYPR(
                    pose_key=right_wrist_action_wristframe,
                    output_key=right_wrist_action_wristframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=left_wrist_obs_headframe,
                    output_key=left_wrist_obs_headframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=right_wrist_obs_headframe,
                    output_key=right_wrist_obs_headframe,
                ),
            ]
        )
    transform_list.extend(
        [
            ConcatKeys(
                key_list=[
                    left_wrist_action_wristframe,
                    left_keypoints_action_wristframe,
                    right_wrist_action_wristframe,
                    right_keypoints_action_wristframe,
                ],
                new_key_name="actions_keypoints",
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[
                    left_wrist_obs_headframe,
                    left_keypoints_obs_wristframe,
                    right_wrist_obs_headframe,
                    right_keypoints_obs_wristframe,
                ],
                new_key_name="observations.state.keypoints",
                delete_old_keys=True,
            ),
            DeleteKeys(keys_to_delete=delete_keys),
        ]
    )
    return transform_list


def _build_aria_keypoints_bimanual_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    left_keypoints_action_world: str = "left.action_keypoints",
    right_keypoints_action_world: str = "right.action_keypoints",
    left_keypoints_obs_pose: str = "left.obs_keypoints",
    right_keypoints_obs_pose: str = "right.obs_keypoints",
    left_keypoints_action_headframe: str = "left.action_keypoints_headframe",
    right_keypoints_action_headframe: str = "right.action_keypoints_headframe",
    left_keypoints_obs_headframe: str = "left.obs_keypoints_headframe",
    right_keypoints_obs_headframe: str = "right.obs_keypoints_headframe",
    left_wrist_action_world: str = "left.action_wrist_pose",
    right_wrist_action_world: str = "right.action_wrist_pose",
    left_wrist_obs_pose: str = "left.obs_wrist_pose",
    right_wrist_obs_pose: str = "right.obs_wrist_pose",
    left_wrist_action_headframe: str = "left.action_wrist_pose_headframe",
    right_wrist_action_headframe: str = "right.action_wrist_pose_headframe",
    left_wrist_obs_headframe: str = "left.obs_wrist_pose_headframe",
    right_wrist_obs_headframe: str = "right.obs_wrist_pose_headframe",
    delete_target_world: bool = True,
    chunk_length: int = 100,
    stride: int = 3,
    concat_keys: bool = True,
    is_quat: bool = True,
) -> list[Transform]:
    keys_to_delete = list(
        {
            left_keypoints_action_world,
            right_keypoints_action_world,
            left_keypoints_obs_pose,
            right_keypoints_obs_pose,
            left_wrist_action_world,
            right_wrist_action_world,
            left_wrist_obs_pose,
            right_wrist_obs_pose,
            left_keypoints_action_headframe,
            right_keypoints_action_headframe,
            left_keypoints_obs_headframe,
            right_keypoints_obs_headframe,
            left_wrist_action_headframe,
            right_wrist_action_headframe,
            left_wrist_obs_headframe,
            right_wrist_obs_headframe,
        }
    )
    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)
    transform_list: list[Transform] = [
        Reshape(
            input_key=left_keypoints_action_world,
            output_key=left_keypoints_action_world,
            shape=(30, 21, 3),
        ),
        Reshape(
            input_key=right_keypoints_action_world,
            output_key=right_keypoints_action_world,
            shape=(30, 21, 3),
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=left_keypoints_action_world,
            transformed_key_name=left_keypoints_action_headframe,
            mode="xyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=right_keypoints_action_world,
            transformed_key_name=right_keypoints_action_headframe,
            mode="xyz",
        ),
        Reshape(
            input_key=left_keypoints_obs_pose,
            output_key=left_keypoints_obs_pose,
            shape=(21, 3),
        ),
        Reshape(
            input_key=right_keypoints_obs_pose,
            output_key=right_keypoints_obs_pose,
            shape=(21, 3),
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=left_keypoints_obs_pose,
            transformed_key_name=left_keypoints_obs_headframe,
            mode="xyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=right_keypoints_obs_pose,
            transformed_key_name=right_keypoints_obs_headframe,
            mode="xyz",
        ),
        Reshape(
            input_key=left_keypoints_obs_headframe,
            output_key=left_keypoints_obs_headframe,
            shape=(63,),
        ),
        Reshape(
            input_key=right_keypoints_obs_headframe,
            output_key=right_keypoints_obs_headframe,
            shape=(63,),
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_keypoints_action_headframe,
            output_action_key=left_keypoints_action_headframe,
            stride=stride,
            mode="xyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_keypoints_action_headframe,
            output_action_key=right_keypoints_action_headframe,
            stride=stride,
            mode="xyz",
        ),
        Reshape(
            input_key=left_keypoints_action_headframe,
            output_key=left_keypoints_action_headframe,
            shape=(chunk_length, 63),
        ),
        Reshape(
            input_key=right_keypoints_action_headframe,
            output_key=right_keypoints_action_headframe,
            shape=(chunk_length, 63),
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=left_wrist_action_world,
            transformed_key_name=left_wrist_action_headframe,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_world,
            chunk_world=right_wrist_action_world,
            transformed_key_name=right_wrist_action_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=left_wrist_obs_pose,
            transformed_key_name=left_wrist_obs_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_world,
            pose_world=right_wrist_obs_pose,
            transformed_key_name=right_wrist_obs_headframe,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_wrist_action_headframe,
            output_action_key=left_wrist_action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_wrist_action_headframe,
            output_action_key=right_wrist_action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
    ]
    if not is_quat:
        transform_list.extend(
            [
                BatchQuaternionPoseToYPR(
                    pose_key=left_wrist_action_headframe,
                    output_key=left_wrist_action_headframe,
                ),
                BatchQuaternionPoseToYPR(
                    pose_key=right_wrist_action_headframe,
                    output_key=right_wrist_action_headframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=left_wrist_obs_headframe,
                    output_key=left_wrist_obs_headframe,
                ),
                QuaternionPoseToYPR(
                    pose_key=right_wrist_obs_headframe,
                    output_key=right_wrist_obs_headframe,
                ),
            ]
        )
    if concat_keys:
        transform_list.extend(
            [
                ConcatKeys(
                    key_list=[
                        left_wrist_action_headframe,
                        left_keypoints_action_headframe,
                        right_wrist_action_headframe,
                        right_keypoints_action_headframe,
                    ],
                    new_key_name="actions_keypoints",
                    delete_old_keys=True,
                ),
                ConcatKeys(
                    key_list=[
                        left_wrist_obs_headframe,
                        left_keypoints_obs_headframe,
                        right_wrist_obs_headframe,
                        right_keypoints_obs_headframe,
                    ],
                    new_key_name="observations.state.keypoints",
                    delete_old_keys=True,
                ),
                DeleteKeys(keys_to_delete=keys_to_delete),
            ]
        )
    return transform_list


def _build_aria_cartesian_bimanual_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    left_action_world: str = "left.action_ee_pose",
    right_action_world: str = "right.action_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_action_headframe: str = "left.action_ee_pose_headframe",
    right_action_headframe: str = "right.action_ee_pose_headframe",
    left_obs_headframe: str = "left.obs_ee_pose_headframe",
    right_obs_headframe: str = "right.obs_ee_pose_headframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 3,
    delete_target_world: bool = True,
) -> list[Transform]:
    """Canonical ARIA bimanual transform pipeline used by tests and notebooks.

    Aria human data does not have commanded ee poses; action chunks are built
    from stacked observed ee poses (typically with a horizon on
    ``left/right.action_ee_pose`` mapped from ``left/right.obs_ee_pose``).
    """
    keys_to_delete = list(
        {
            left_action_world,
            right_action_world,
            left_obs_pose,
            right_obs_pose,
        }
    )
    target_pose_key = target_world
    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)

    transform_list: list[Transform] = [
        # Tracking-dropout rows (all-zero / non-finite quats) in some episodes make
        # scipy from_quat raise "Found zero norm quaternions" inside the SE3
        # conversion of the coordinate-frame transforms below. Forward/backward-fill
        # those rows first (holds the last valid pose = no motion). No-op on clean
        # data; the obs anchor (== action_chunk[0]) is kept consistent via anchor_key.
        SanitizeQuatPoseChunk(chunk_key=left_action_world, anchor_key=left_obs_pose),
        SanitizeQuatPoseChunk(chunk_key=right_action_world, anchor_key=right_obs_pose),
        ActionChunkCoordinateFrameTransform(
            target_world=target_pose_key,
            chunk_world=left_action_world,
            transformed_key_name=left_action_headframe,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=target_pose_key,
            chunk_world=right_action_world,
            transformed_key_name=right_action_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_headframe,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=left_action_headframe,
            output_action_key=left_action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
        InterpolatePose(
            new_chunk_length=chunk_length,
            action_key=right_action_headframe,
            output_action_key=right_action_headframe,
            stride=stride,
            mode="xyzwxyz",
        ),
    ]

    if target_world_is_quat:
        transform_list.append(
            XYZWXYZ_to_XYZYPR(
                keys=[
                    left_action_headframe,
                    right_action_headframe,
                    left_obs_headframe,
                    right_obs_headframe,
                ]
            )
        )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[left_action_headframe, right_action_headframe],
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[left_obs_headframe, right_obs_headframe],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(keys_to_delete=keys_to_delete),
        ]
    )
    return transform_list


def _build_mecka_cartesian_wristframe_bimanual_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    left_action_world: str = "left.action_ee_pose",
    right_action_world: str = "right.action_ee_pose",
    left_obs_pose: str = "left.obs_ee_pose",
    right_obs_pose: str = "right.obs_ee_pose",
    left_action_wristframe: str = "left.action_ee_pose_wristframe",
    right_action_wristframe: str = "right.action_ee_pose_wristframe",
    left_obs_headframe: str = "left.obs_ee_pose_headframe",
    right_obs_headframe: str = "right.obs_ee_pose_headframe",
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    stride: int = 3,
    interpolate: bool = True,
    rot_repr: str = "ypr",
    delete_target_world: bool = True,
) -> list[Transform]:
    """Cartesian bimanual pipeline with actions in the WRIST frame (delta pose).

    Identical to ``_build_aria_cartesian_bimanual_transform_list`` except each arm's
    action ee-pose chunk is expressed relative to THAT arm's current ee pose
    (``left/right.obs_ee_pose``) instead of the head pose. The result is a per-step
    delta pose in the wrist frame: step 0 is ~identity (zero delta, since the chunk
    starts at the current frame) and later steps are the relative transform from the
    current ee pose. ``actions_cartesian`` stays (chunk_length, 12) — xyz+ypr per arm —
    so the tokenizer model config is unchanged.

    Proprio (``observations.state.ee_pose``) is kept in the HEAD frame, matching the
    cartesian pipeline, so it stays informative and non-degenerate for norm stats
    (a self-referential wrist-frame obs would be all-zeros).

    ``interpolate``: when True (default), the raw action horizon (post-``stride``) is
    resampled to ``chunk_length`` via ``InterpolatePose``. When False, no resampling is
    done — the action chunk is used as read, so the keymap MUST supply a horizon of
    exactly ``chunk_length`` real frames (i.e. ``action_ee_pose`` horizon == chunk_length).

    ``rot_repr``: ``"ypr"`` → xyz + Euler (6 per arm, 12 bimanual); ``"6d"`` → xyz +
    continuous 6D rotation columns (9 per arm, 18 bimanual; Zhou et al.). The frame
    math is identical to pi-6d's ``cartesian_wristframe_6d`` (single-anchor: each action
    relative to the current ee pose); only the rotation encoding differs.
    """
    keys_to_delete = list(
        {
            left_action_world,
            right_action_world,
            left_obs_pose,
            right_obs_pose,
        }
    )
    target_pose_key = target_world
    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)

    transform_list: list[Transform] = [
        # ACTION chunk → each arm's OWN current ee frame (wrist frame / delta pose).
        # inverse=True (default) computes target^{-1} @ chunk, i.e. each future ee
        # pose expressed relative to the current ee pose.
        ActionChunkCoordinateFrameTransform(
            target_world=left_obs_pose,
            chunk_world=left_action_world,
            transformed_key_name=left_action_wristframe,
            mode="xyzwxyz",
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_obs_pose,
            chunk_world=right_action_world,
            transformed_key_name=right_action_wristframe,
            mode="xyzwxyz",
        ),
        # PROPRIO stays in head frame (unchanged from the cartesian pipeline).
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=left_obs_pose,
            transformed_key_name=left_obs_headframe,
            mode="xyzwxyz",
        ),
        PoseCoordinateFrameTransform(
            target_world=target_pose_key,
            pose_world=right_obs_pose,
            transformed_key_name=right_obs_headframe,
            mode="xyzwxyz",
        ),
    ]

    if interpolate:
        # Resample the (post-stride) raw horizon to chunk_length. When False, the
        # action chunk is used as-is (keymap must read exactly chunk_length frames).
        transform_list.extend(
            [
                InterpolatePose(
                    new_chunk_length=chunk_length,
                    action_key=left_action_wristframe,
                    output_action_key=left_action_wristframe,
                    stride=stride,
                    mode="xyzwxyz",
                ),
                InterpolatePose(
                    new_chunk_length=chunk_length,
                    action_key=right_action_wristframe,
                    output_action_key=right_action_wristframe,
                    stride=stride,
                    mode="xyzwxyz",
                ),
            ]
        )

    if target_world_is_quat:
        rot_encoder = XYZWXYZ_to_XYZ6D if rot_repr == "6d" else XYZWXYZ_to_XYZYPR
        transform_list.append(
            rot_encoder(
                keys=[
                    left_action_wristframe,
                    right_action_wristframe,
                    left_obs_headframe,
                    right_obs_headframe,
                ]
            )
        )

    transform_list.extend(
        [
            ConcatKeys(
                key_list=[left_action_wristframe, right_action_wristframe],
                new_key_name=actions_key,
                delete_old_keys=True,
            ),
            ConcatKeys(
                key_list=[left_obs_headframe, right_obs_headframe],
                new_key_name=obs_key,
                delete_old_keys=True,
            ),
            DeleteKeys(keys_to_delete=keys_to_delete),
        ]
    )
    return transform_list


def _build_mecka_cartesian_revert_wristframe_transform_list(
    *,
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    left_action_wristframe: str = "left.action_ee_pose_wristframe",
    right_action_wristframe: str = "right.action_ee_pose_wristframe",
    left_obs_headframe: str = "left.obs_ee_pose_headframe",
    right_obs_headframe: str = "right.obs_ee_pose_headframe",
    left_action_headframe: str = "left.action_ee_pose_headframe",
    right_action_headframe: str = "right.action_ee_pose_headframe",
    is_quat: bool = False,
    rot_repr: str = "ypr",
) -> list[Transform]:
    """Revert wrist-frame cartesian actions back to head (camera) frame for viz.

    Inverse of ``_build_mecka_cartesian_wristframe_bimanual_transform_list``: the
    action chunk lives in each arm's wrist frame, the proprio ee-poses live in head
    frame. Re-composes ``obs_headframe @ action_wristframe`` (inverse=False) so the
    action chunk is back in head/camera frame, which ``viz_gt_preds`` can project.
    ``rot_repr="6d"`` reverts a model 6D prediction (per-arm width 9, ``xyz6d`` with
    Gram-Schmidt) then collapses to ypr. Wired via the visualization config's
    ``transform_list`` (viz_gt_preds applies it to both GT and prediction). Ported
    from pi-6d's ``_build_human_cartesian_revert_eef_frame_transform_list``.
    """
    if rot_repr == "6d":
        pose_shape = 9
        mode = "xyz6d"
    else:
        pose_shape = 7 if is_quat else 6
        mode = "xyzwxyz" if is_quat else "xyzypr"
    transform_list = [
        SplitKeys(
            input_key=obs_key,
            output_key_list=[
                (left_obs_headframe, pose_shape),
                (right_obs_headframe, pose_shape),
            ],
        ),
        SplitKeys(
            input_key=action_key,
            output_key_list=[
                (left_action_wristframe, pose_shape),
                (right_action_wristframe, pose_shape),
            ],
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=left_obs_headframe,
            chunk_world=left_action_wristframe,
            transformed_key_name=left_action_headframe,
            mode=mode,
            inverse=False,
        ),
        ActionChunkCoordinateFrameTransform(
            target_world=right_obs_headframe,
            chunk_world=right_action_wristframe,
            transformed_key_name=right_action_headframe,
            mode=mode,
            inverse=False,
        ),
    ]
    if rot_repr == "6d":
        transform_list.append(
            XYZ6D_to_XYZYPR(keys=[left_action_headframe, right_action_headframe])
        )
    transform_list.append(
        ConcatKeys(
            key_list=[left_action_headframe, right_action_headframe],
            new_key_name=action_key,
            delete_old_keys=True,
        ),
    )
    return transform_list


# MANO fingertip keypoint indices in the 21-keypoint layout: thumb, index, middle,
# ring, pinky tips = 4, 8, 12, 16, 20 (each finger is 4 joints, tip = last).
_MANO_FINGERTIP_INDICES = (4, 8, 12, 16, 20)
_N_FINGERTIPS = len(_MANO_FINGERTIP_INDICES)  # 5
_FINGERTIP_DIMS = _N_FINGERTIPS * 3  # 15


def _build_mecka_wristframe_6d_fingertips_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    horizon: int = 30,
    stride: int = 1,
    interpolate: bool = True,
    delete_target_world: bool = True,
) -> list[Transform]:
    """Wrist-frame 6D palm pose + 5 MANO fingertips per hand (single-anchor).

    Per hand: palm (``obs_wrist_pose``) as a 6D wrist-frame delta relative to the
    current wrist pose (9 dims) + the 5 MANO fingertips (keypoints 4/8/12/16/20)
    expressed in that same current-wrist frame (5*3 = 15 dims) = 24. Bimanual
    ``actions_cartesian`` is ``(chunk_length, 48)``. Proprio ``observations.state.ee_pose``
    holds each current wrist pose in head frame as 6D (9/hand, 18 total) so the revert
    can re-anchor palm + fingertips back to the camera frame for viz. Uses the keypoints
    keymap keys.

    ``interpolate``: when True the raw ``horizon`` chunk is resampled to ``chunk_length``
    (palm via quaternion slerp, fingertips linearly). When False no resampling is done —
    the keymap must supply ``horizon == chunk_length`` real frames.
    """
    transform_list: list[Transform] = []
    keys_to_delete: list[str] = []
    action_parts: list[str] = []
    obs_parts: list[str] = []

    for side in ("left", "right"):
        aw = f"{side}.action_wrist_pose"  # (H, 7) world wrist poses
        akp = f"{side}.action_keypoints"  # (H, 63) world keypoints
        ow = f"{side}.obs_wrist_pose"  # (7,) current world wrist pose
        palm_wf = f"{side}.palm_wristframe"  # (H,7) -> (chunk,9)
        tips = f"{side}.fingertips"  # (H,5,3) -> (chunk,15)
        obs_wf_head = f"{side}.obs_wrist_headframe"  # (7,) -> (9,)

        transform_list += [
            # Palm: action wrist chunk relative to the CURRENT wrist pose (single-anchor
            # delta). inverse=True -> obs_wrist^-1 @ action_wrist.
            ActionChunkCoordinateFrameTransform(
                target_world=ow,
                chunk_world=aw,
                transformed_key_name=palm_wf,
                mode="xyzwxyz",
            ),
            # Fingertips: select the 5 MANO tips (indices 4/8/12/16/20) -> (H,5,3),
            # expressed in the current wrist frame (xyz points, inverse=True).
            SelectKeypoints(
                input_key=akp,
                output_key=tips,
                indices=_MANO_FINGERTIP_INDICES,
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=ow,
                chunk_world=tips,
                transformed_key_name=tips,
                mode="xyz",
            ),
            # Proprio: current wrist pose -> head frame (revert anchor).
            PoseCoordinateFrameTransform(
                target_world=target_world,
                pose_world=ow,
                transformed_key_name=obs_wf_head,
                mode="xyzwxyz",
            ),
        ]
        if interpolate:
            # Resample H -> chunk_length (palm as quats/slerp, fingertips as xyz).
            transform_list += [
                InterpolatePose(
                    new_chunk_length=chunk_length,
                    action_key=palm_wf,
                    output_action_key=palm_wf,
                    stride=stride,
                    mode="xyzwxyz",
                ),
                InterpolatePose(
                    new_chunk_length=chunk_length,
                    action_key=tips,
                    output_action_key=tips,
                    stride=stride,
                    mode="xyz",
                ),
            ]
        transform_list += [
            # Encode rotations as continuous 6D; flatten fingertips back to (chunk,15).
            XYZWXYZ_to_XYZ6D(keys=[palm_wf, obs_wf_head]),
            Reshape(
                input_key=tips, output_key=tips, shape=(chunk_length, _FINGERTIP_DIMS)
            ),
            # Per-hand action = palm(9) + fingertips(15) = 24.
            ConcatKeys(
                key_list=[palm_wf, tips],
                new_key_name=f"{side}.hand_action",
                delete_old_keys=True,
            ),
        ]
        action_parts.append(f"{side}.hand_action")
        obs_parts.append(obs_wf_head)
        keys_to_delete += [aw, akp, ow, f"{side}.obs_keypoints"]

    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)

    transform_list += [
        ConcatKeys(
            key_list=action_parts, new_key_name=actions_key, delete_old_keys=True
        ),
        ConcatKeys(key_list=obs_parts, new_key_name=obs_key, delete_old_keys=True),
        DeleteKeys(keys_to_delete=keys_to_delete),
    ]
    return transform_list


def _build_mecka_revert_wristframe_6d_fingertips_transform_list(
    *,
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
) -> list[Transform]:
    """Revert 6D wrist-frame palm + fingertips to head-frame positions for viz.

    Inverse of ``_build_mecka_wristframe_6d_fingertips_transform_list`` for viz.
      action(48): per hand [palm 6D wrist(9) | fingertips wrist(15)].
      obs(18):    per hand [wrist pose head 6D(9)] (the revert anchor).
    Output action(36): per hand [palm xyz head(3) | 5 fingertips xyz head(15)] — the
    camera-frame positions Mecka.viz(mode="fingertips") projects. Wired via the
    visualization config's transform_list (applied to both GT and prediction).
    """
    transform_list: list[Transform] = [
        SplitKeys(
            input_key=obs_key, output_key_list=[("left.owh", 9), ("right.owh", 9)]
        ),
        SplitKeys(
            input_key=action_key,
            output_key_list=[("left.hand", 24), ("right.hand", 24)],
        ),
    ]
    viz_parts: list[str] = []
    for side in ("left", "right"):
        owh, hand = f"{side}.owh", f"{side}.hand"
        palm, tips = f"{side}.palm6d", f"{side}.tips"
        palm_head, palm_xyz, palm_rot = (
            f"{side}.palm_head",
            f"{side}.palm_xyz",
            f"{side}.palm_rot",
        )
        transform_list += [
            SplitKeys(input_key=hand, output_key_list=[(palm, 9), (tips, 15)]),
            # palm 6D wrist -> head (re-anchor), keep only xyz for the point viz.
            ActionChunkCoordinateFrameTransform(
                target_world=owh,
                chunk_world=palm,
                transformed_key_name=palm_head,
                mode="xyz6d",
                inverse=False,
            ),
            SplitKeys(
                input_key=palm_head, output_key_list=[(palm_xyz, 3), (palm_rot, 6)]
            ),
            # fingertips wrist -> head.
            Reshape(
                input_key=tips, output_key=tips, shape=(chunk_length, _N_FINGERTIPS, 3)
            ),
            ActionChunkCoordinateFrameTransform(
                target_world=owh,
                chunk_world=tips,
                transformed_key_name=tips,
                mode="xyz",
                inverse=False,
            ),
            Reshape(
                input_key=tips, output_key=tips, shape=(chunk_length, _FINGERTIP_DIMS)
            ),
            ConcatKeys(
                key_list=[palm_xyz, tips],
                new_key_name=f"{side}.viz",
                delete_old_keys=True,
            ),
        ]
        viz_parts.append(f"{side}.viz")
    transform_list.append(
        ConcatKeys(key_list=viz_parts, new_key_name=action_key, delete_old_keys=True)
    )
    return transform_list


def _build_mecka_wf6d_fingertips_stepwise_transform_list(
    *,
    target_world: str = "obs_head_pose",
    target_world_ypr: str = "obs_head_pose_ypr",
    target_world_is_quat: bool = True,
    actions_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
    delete_target_world: bool = True,
) -> list[Transform]:
    """STEP-WISE 6D palm deltas + per-timestep wrist-frame MANO fingertips (48-dim).

    v2 of the fingertips representation, killing the cumulative-displacement signal:
      palm (9/hand):  A_t = P_{t-1}^{-1} ∘ P_t — frame-to-frame delta (velocity), 6D.
                      A_0 = identity. No monotone "distance since anchor" axis.
      tips (15/hand): the 5 MANO fingertips (4/8/12/16/20) expressed in the wrist
                      frame of the SAME timestep — pure articulation / grasp
                      aperture, carrying no palm displacement at all.
    No resampling: reads a full ``chunk_length`` horizon of real frames. Proprio
    stays the current wrist pose in head frame as 6D (18 total) — the anchor the
    revert chain-composes from for viz.
    """
    transform_list: list[Transform] = []
    keys_to_delete: list[str] = []
    action_parts: list[str] = []
    obs_parts: list[str] = []

    for side in ("left", "right"):
        aw = f"{side}.action_wrist_pose"  # (T, 7) world wrist poses
        akp = f"{side}.action_keypoints"  # (T, 63) world keypoints
        ow = f"{side}.obs_wrist_pose"  # (7,) current world wrist pose
        palm_sw = f"{side}.palm_stepwise"  # (T,7) -> (T,9)
        tips = f"{side}.fingertips"  # (T,5,3) -> (T,15)
        obs_wf_head = f"{side}.obs_wrist_headframe"  # (7,) -> (9,)

        transform_list += [
            # Tracking-dropout rows (zero-quat) crash from_quat; forward-fill them
            # (delta across a dropout = identity) and re-anchor ow to row 0.
            SanitizeQuatPoseChunk(chunk_key=aw, anchor_key=ow),
            # Palm: consecutive frame-to-frame deltas (A_0 = identity).
            ConsecutiveDeltaChunk(chunk_key=aw, output_key=palm_sw),
            # Fingertips: MANO tips in the wrist frame of the SAME timestep.
            SelectKeypoints(
                input_key=akp,
                output_key=tips,
                indices=_MANO_FINGERTIP_INDICES,
            ),
            PerTimestepCoordinateFrameTransform(
                target_chunk=aw,
                chunk=tips,
                transformed_key_name=tips,
                inverse=True,
            ),
            # Proprio: current wrist pose -> head frame (revert anchor).
            PoseCoordinateFrameTransform(
                target_world=target_world,
                pose_world=ow,
                transformed_key_name=obs_wf_head,
                mode="xyzwxyz",
            ),
            # 6D-encode rotations; flatten fingertips to (T, 15).
            XYZWXYZ_to_XYZ6D(keys=[palm_sw, obs_wf_head]),
            Reshape(
                input_key=tips, output_key=tips, shape=(chunk_length, _FINGERTIP_DIMS)
            ),
            ConcatKeys(
                key_list=[palm_sw, tips],
                new_key_name=f"{side}.hand_action",
                delete_old_keys=True,
            ),
        ]
        action_parts.append(f"{side}.hand_action")
        obs_parts.append(obs_wf_head)
        keys_to_delete += [aw, akp, ow, f"{side}.obs_keypoints"]

    if delete_target_world:
        keys_to_delete.append(target_world)
        if target_world_is_quat:
            keys_to_delete.append(target_world_ypr)

    transform_list += [
        ConcatKeys(
            key_list=action_parts, new_key_name=actions_key, delete_old_keys=True
        ),
        ConcatKeys(key_list=obs_parts, new_key_name=obs_key, delete_old_keys=True),
        DeleteKeys(keys_to_delete=keys_to_delete),
    ]
    return transform_list


def _build_mecka_revert_wf6d_ft_stepwise_transform_list(
    *,
    action_key: str = "actions_cartesian",
    obs_key: str = "observations.state.ee_pose",
    chunk_length: int = 100,
) -> list[Transform]:
    """Revert step-wise 6D palm + per-timestep fingertips to head frame for viz.

    Chain-composes the palm deltas from the head-frame wrist anchor
    (``P_t = anchor ∘ A_0 ∘ … ∘ A_t``), then re-anchors each timestep's fingertips
    by that timestep's recovered wrist pose. Output (36): per hand
    [palm_xyz(3) | 5 fingertips xyz(15)] in head/camera frame — same layout the
    single-anchor revert emits, so Mecka.viz(mode="fingertips") is unchanged.
    """
    transform_list: list[Transform] = [
        SplitKeys(
            input_key=obs_key, output_key_list=[("left.owh", 9), ("right.owh", 9)]
        ),
        SplitKeys(
            input_key=action_key,
            output_key_list=[("left.hand", 24), ("right.hand", 24)],
        ),
    ]
    viz_parts: list[str] = []
    for side in ("left", "right"):
        owh, hand = f"{side}.owh", f"{side}.hand"
        palm_sw, tips = f"{side}.palm_sw", f"{side}.tips"
        wrist_head = f"{side}.wrist_head"
        palm_xyz, palm_rot = f"{side}.palm_xyz", f"{side}.palm_rot"
        transform_list += [
            SplitKeys(input_key=hand, output_key_list=[(palm_sw, 9), (tips, 15)]),
            # Chain-compose step-wise deltas from the head-frame anchor.
            CumulativeComposeChunk(
                anchor_key=owh,
                delta_key=palm_sw,
                output_key=wrist_head,
            ),
            SplitKeys(
                input_key=wrist_head, output_key_list=[(palm_xyz, 3), (palm_rot, 6)]
            ),
            # Fingertips: per-timestep wrist frame -> head frame via wrist_t.
            Reshape(
                input_key=tips, output_key=tips, shape=(chunk_length, _N_FINGERTIPS, 3)
            ),
            PerTimestepCoordinateFrameTransform(
                target_chunk=wrist_head,
                chunk=tips,
                transformed_key_name=tips,
                inverse=False,
            ),
            Reshape(
                input_key=tips, output_key=tips, shape=(chunk_length, _FINGERTIP_DIMS)
            ),
            ConcatKeys(
                key_list=[palm_xyz, tips],
                new_key_name=f"{side}.viz",
                delete_old_keys=True,
            ),
        ]
        viz_parts.append(f"{side}.viz")
    transform_list.append(
        ConcatKeys(key_list=viz_parts, new_key_name=action_key, delete_old_keys=True)
    )
    return transform_list
