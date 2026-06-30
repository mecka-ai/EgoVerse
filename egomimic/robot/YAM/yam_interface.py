"""Robot interface for the I2RT YAM bimanual arms.

Wraps the ``i2rt`` SDK (vendored as the ``external/i2rt`` git submodule) behind
the same contract used by ``ARXInterface`` so the rest of EgoVerse (rollout,
data collection, eval) can drive YAM arms without code changes.

Conventions (identical to ``ARXInterface``):
  * Per-arm joint vector is shape ``(7,)`` = 6 arm joints (rad) + gripper.
  * Gripper is normalized to ``[0, 1]`` where ``1.0`` is fully open. The YAM
    SDK already normalizes the gripper into this range via its ``JointMapper``,
    so no manual (de)normalization is needed here.
  * End-effector pose is ``xyz`` + ``ZYX`` intrinsic Euler angles (rad).
  * ``get_obs`` packs both arms into length-14 vectors: ``left`` in ``[0:7]``,
    ``right`` in ``[7:14]``.
"""

import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

# --- Make the vendored i2rt submodule importable without `pip install` -------
# Repo layout: <repo_root>/external/i2rt/i2rt/...  and this file lives at
# <repo_root>/egomimic/robot/YAM/yam_interface.py
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_I2RT_PATH = os.path.join(_REPO_ROOT, "external", "i2rt")
if os.path.isdir(_I2RT_PATH) and _I2RT_PATH not in sys.path:
    sys.path.insert(0, _I2RT_PATH)

# The base class lives at egomimic/robot/eva/eva_ws/src/eva/robot_interface.py
# and is imported elsewhere in this repo via a sys.path hack to that dir.
_EVA_SRC = os.path.join(_REPO_ROOT, "egomimic", "robot", "eva", "eva_ws", "src", "eva")
if os.path.isdir(_EVA_SRC) and _EVA_SRC not in sys.path:
    sys.path.insert(0, _EVA_SRC)

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

try:
    # Inherit the abstract contract (set_joints/get_pose/...) when available.
    from robot_interface import Robot_Interface
except Exception:  # pragma: no cover - base import is optional for YAM
    Robot_Interface = object


# Site (frame) used for FK/IK in the YAM MuJoCo model.
_GRASP_SITE = "grasp_site"

# Default CAN channel per arm. Override via the `channels` ctor arg.
_DEFAULT_CHANNELS = {"left": "can0", "right": "can1"}

_ARM_OFFSET = {"left": 0, "right": 7}


class YAMInterface(Robot_Interface):
    """Bimanual (or single-arm) interface to YAM arms over CAN."""

    N_ARM_JOINTS = 6

    def __init__(
        self,
        arms,
        channels=None,
        gripper_type=GripperType.LINEAR_4310,
        arm_type=ArmType.YAM,
        zero_gravity_mode=False,
        gripper_limits_override=None,
        camera_names=None,
        cameras_cfg=None,
        front_raw=False,
    ):
        """
        Args:
            arms (list[str]): subset of ["left", "right"].
            channels (dict[str, str] | None): CAN channel per arm, e.g.
                {"left": "can0", "right": "can1"}. Defaults to _DEFAULT_CHANNELS.
            gripper_type (GripperType): YAM gripper variant.
            arm_type (ArmType): YAM arm variant.
            zero_gravity_mode (bool): start in gravity-comp idle (no PD hold).
                Use False for policy execution so the arm holds commanded poses.
            gripper_limits_override (np.ndarray | None): [closed, open] limits to
                skip auto-calibration on startup.
            camera_names (dict[str, str] | None): serial->name map for the wrist
                cameras. Defaults to yam_cameras.DEFAULT_CAMERA_NAMES.
            cameras_cfg (dict | None): optional ARX-style explicit camera config
                ``{name: {type, enabled, ...}}``. None -> auto-discovery (Atlas front
                + connected D405 wrists).
            front_raw (bool): if True (auto-discovery only), capture the Atlas front
                camera RAW — the un-rectified side-by-side fisheye pair, no
                rectify/re-aim/fuse/crop. For (intrinsic) camera calibration; leave
                False for normal rollout/data collection.

        The interface ALWAYS opens its own cameras and raises if none are
        available (a rollout — or black-frame demos — must not run blind).
        """
        # NOTE: do not call super().__init__(); the base ctor is ARX5-specific.
        self.arms = list(arms)
        self.channels = dict(channels) if channels else dict(_DEFAULT_CHANNELS)
        self.gripper_type = gripper_type
        self.arm_type = arm_type
        self._zero_gravity_mode = zero_gravity_mode
        self._gripper_limits_override = gripper_limits_override

        # Controllers: one YAM robot + kinematics solver per arm.
        self.controller = {}
        self.kinematics = {}
        # Per-arm convergence flag from the last solve_ik; set_pose refuses to
        # command an arm whose IK did not converge (see set_pose / solve_ik).
        self._last_ik_success = {}
        self._create_controllers(cfg={})
        self._create_cam_recorders(
            cameras_cfg=cameras_cfg,
            wrist_serial_to_name=camera_names,
            front_raw=front_raw,
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _create_controllers(self, cfg):
        """Instantiate one YAM robot + kinematics solver per arm."""
        # Kinematics runs on the clean 6-DOF arm (NO_GRIPPER): the real gripper
        # models add linkage qpos (8 total) that the 6 arm joints can't drive,
        # which breaks mink's fk/ik. This matches the SDK's own kinematics tests.
        # Note: with NO_GRIPPER the grasp_site sits at the wrist flange, so
        # get_pose() reports the flange frame (add a fixed gripper offset
        # downstream if you need the fingertip grasp point).
        kin_model_path = combine_arm_and_gripper_xml(self.arm_type, GripperType.NO_GRIPPER)

        for arm in self.arms:
            if arm not in self.channels:
                raise ValueError(f"No CAN channel configured for arm '{arm}'")
            self.controller[arm] = get_yam_robot(
                channel=self.channels[arm],
                arm_type=self.arm_type,
                gripper_type=self.gripper_type,
                zero_gravity_mode=self._zero_gravity_mode,
                gripper_limits_override=self._gripper_limits_override,
            )

            # === TEMP: soften the follower PD for safe rollout bring-up ========
            # Lower the ARM-joint kp/kd (NOT the gripper, kept full so grasps
            # still hold) so an off prediction nudges instead of slamming. This
            # is a temporary bring-up aid — set YAM_PD_SCALE=1.0 to restore the
            # SDK defaults, or delete this block. Applied live via update_kp_kd
            # (control loop reads self._kp/_kd each tick).
            _pd_scale = float(os.environ.get("YAM_PD_SCALE", "0.5"))
            if _pd_scale != 1.0:
                ctrl = self.controller[arm]
                new_kp = np.asarray(ctrl._kp, dtype=float).copy()
                new_kd = np.asarray(ctrl._kd, dtype=float).copy()
                new_kp[:6] *= _pd_scale  # arm joints only; leave gripper [6] as-is
                new_kd[:6] *= _pd_scale
                ctrl.update_kp_kd(new_kp, new_kd)
                print(
                    f"[YAM][TEMP] {arm}: scaled arm PD by {_pd_scale} -> "
                    f"kp[:6]={np.round(new_kp[:6], 2)} kd[:6]={np.round(new_kd[:6], 2)} "
                    f"(YAM_PD_SCALE=1.0 to disable)"
                )
            # === END TEMP =====================================================

            # Each arm gets its own solver: mink.Configuration holds internal
            # state, so sharing one across arms would race.
            self.kinematics[arm] = Kinematics(kin_model_path, _GRASP_SITE)

    def _create_cam_recorders(self, cameras_cfg=None, wrist_serial_to_name=None,
                              front_raw=False):
        """Build per-camera recorders into ``self.recorders`` / ``self.camera_res``.

        Mirrors ``ARXInterface.__create_cam_recorders``: ALL camera setup lives
        here, so ``__init__`` only calls this once. The interface ALWAYS builds &
        owns its own recorders (auto-discovery, or the ARX-style explicit
        ``cameras_cfg``); they are closed in ``close()``. Each recorder exposes
        ``.get_image()`` (latest BGR uint8 frame) and ``.res`` ((H, W)).

        Individual camera failures are warned-about and skipped inside
        ``create_camera_recorders``; if the net result is empty this RAISES — a
        rollout (or black-frame demos) must not run blind.

        Raises:
            RuntimeError: no cameras are available (USB links down / held by
                another app or PipeWire).
        """
        from yam_cameras import create_camera_recorders

        self.recorders = create_camera_recorders(
            cameras_cfg=cameras_cfg, wrist_serial_to_name=wrist_serial_to_name,
            front_raw=front_raw,
        )
        if not self.recorders:
            raise RuntimeError(
                "YAMInterface: NO cameras are available. Check the Atlas front cam "
                "+ RealSense D405 wrists (USB links / another app or PipeWire "
                "holding the devices)."
            )
        self.camera_res = {name: rec.res for name, rec in self.recorders.items()}
        # Wait for first frames (per-recorder; never raises).
        for rec in self.recorders.values():
            waiter = getattr(rec, "wait_until_ready", None)
            if callable(waiter):
                try:
                    waiter()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    def set_joints(self, desired_position, arm):
        """Command joint positions for one arm.

        Args:
            desired_position (np.ndarray): shape (7,) = 6 joints (rad) + gripper [0, 1].
            arm (str): "left" or "right".
        """
        desired_position = np.asarray(desired_position, dtype=np.float64)
        if desired_position.shape != (7,):
            raise ValueError(
                f"YAM expected joint command of shape (7,), got {desired_position.shape}"
            )
        # The SDK clips arm joints to limits and remaps the [0,1] gripper to
        # motor space internally.
        self.controller[arm].command_joint_pos(desired_position)

    def set_pose(self, pose, arm, skip_on_ik_failure=True):
        """Command an end-effector pose for one arm via IK.

        Args:
            pose (np.ndarray): shape (7,) = xyz + ZYX euler (rad) + gripper [0, 1].
            arm (str): "left" or "right".
            skip_on_ik_failure (bool): SAFETY DEFAULT. If True and IK does not
                converge, DO NOT command the arm (return None) — the SDK's
                "best-effort" joints for an unreachable target can fling the arm.
                Set False only if you explicitly want the best-effort command.
        Returns:
            np.ndarray | None: the (7,) joint command that was sent, or None if
                IK failed and skip_on_ik_failure is True (no command sent).
        """
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,):
            raise ValueError(f"YAM expected pose of shape (7,), got {pose.shape}")
        arm_joints = self.solve_ik(pose[:6], arm)
        if skip_on_ik_failure and not self._last_ik_success.get(arm, True):
            print(
                f"[YAMInterface] IK did not converge for arm '{arm}'; "
                f"NOT commanding (target likely unreachable). Holding."
            )
            return None
        joints = np.concatenate([arm_joints, [pose[6]]])
        self.set_joints(joints, arm)
        return joints

    def set_home(self):
        """Move every arm to the zero configuration with the gripper open."""
        for arm in self.arms:
            home = np.zeros(7, dtype=np.float64)
            home[6] = 1.0  # gripper open
            # Smooth interpolated move when available; else a single command.
            mover = getattr(self.controller[arm], "move_joints", None)
            if callable(mover):
                mover(home)
            else:
                self.set_joints(home, arm)
                time.sleep(1.0)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def get_obs(self):
        """Return joint positions and EE poses for all arms.

        Returns:
            dict with:
                "joint_positions": np.ndarray (14,) — left[0:7], right[7:14]
                "ee_poses":        np.ndarray (14,) — xyz+ZYX_euler+gripper per arm
        """
        obs = {}
        joint_positions = np.zeros(14)
        ee_poses = np.zeros(14)

        for arm in self.arms:
            offset = _ARM_OFFSET[arm]
            joints = self.get_joints(arm)
            joint_positions[offset : offset + 7] = joints
            xyz, rot = self.get_pose(arm, se3=False)
            ee_poses[offset : offset + 7] = np.concatenate(
                [xyz, rot.as_euler("ZYX", degrees=False), [joints[6]]]
            )

        obs["joint_positions"] = joint_positions
        obs["ee_poses"] = ee_poses

        # Camera frames (BGR HWC uint8), ARX-style: one recorder per camera, each
        # polled via .get_image(). Keys are the friendly names the rollout obs
        # pipeline expects (front_img_1 / left_wrist_img / right_wrist_img).
        for name, recorder in self.recorders.items():
            obs[name] = recorder.get_image()
        return obs

    def get_joints(self, arm):
        """Current joint state for one arm.

        Returns:
            np.ndarray: shape (7,) = 6 joints (rad) + gripper normalized to [0, 1].
        """
        # get_joint_pos() returns the full chain (6 arm + 1 gripper) with the
        # gripper already normalized into [0, 1] by the SDK's JointMapper.
        joints = np.asarray(self.controller[arm].get_joint_pos(), dtype=np.float64)
        if joints.shape[0] < 7:
            # No gripper configured — pad so callers always get (7,).
            joints = np.concatenate([joints, np.zeros(7 - joints.shape[0])])
        return joints[:7]

    def get_pose(self, arm, se3=False):
        """Forward kinematics for one arm.

        Args:
            arm (str): "left" or "right".
            se3 (bool): if True return a 4x4 SE(3) matrix; else (xyz, Rotation).
        Returns:
            np.ndarray (4,4) if se3, else (np.ndarray (3,), scipy Rotation).
        """
        joints = self.get_joints(arm)
        T = self.kinematics[arm].fk(joints[: self.N_ARM_JOINTS])  # (4, 4)
        if se3:
            return T
        pos = T[:3, 3]
        rot = R.from_matrix(T[:3, :3])
        return pos, rot

    def get_pose_6d(self, arm):
        """End-effector pose as xyz + ZYX euler (rad), shape (6,)."""
        pos, rot = self.get_pose(arm, se3=False)
        return np.concatenate([pos, rot.as_euler("ZYX", degrees=False)])

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------
    def solve_ik(self, ee_pose, arm):
        """Inverse kinematics for a 6D end-effector pose.

        Args:
            ee_pose (np.ndarray): shape (6,) = xyz + ZYX euler (rad).
            arm (str): "left" or "right".
        Returns:
            np.ndarray: arm joint angles (6,).
        """
        ee_pose = np.asarray(ee_pose, dtype=np.float64)
        if ee_pose.shape != (6,):
            raise ValueError(f"YAM IK expected shape (6,), got {ee_pose.shape}")

        target = np.eye(4)
        target[:3, :3] = R.from_euler("ZYX", ee_pose[3:6], degrees=False).as_matrix()
        target[:3, 3] = ee_pose[:3]

        init_q = self.get_joints(arm)[: self.N_ARM_JOINTS]
        # IK robustness near wrist singularities (gripper ~vertical, pitch~88).
        # The i2rt defaults (pos 1e-4 m, ori 1e-4 rad ~= 0.006 deg, damping 1e-4)
        # are far tighter than execution needs: at a singularity differential IK
        # cannot null the orientation error to 0.006 deg in the uncontrollable
        # direction, so it burns all max_iters and returns success=False even
        # though it is sitting well under 1 mm / 1 deg. The tiny damping also lets
        # joint velocities blow up in the singular direction -> the wrist flip.
        # Loosen to execution tolerances (mm / deg) and add real DLS damping so the
        # solver stops early near the seed and stays in-branch. Env-tunable.
        pos_tol = float(os.environ.get("YAM_IK_POS_TOL_MM", "2.0")) / 1000.0
        ori_tol = np.deg2rad(float(os.environ.get("YAM_IK_ORI_TOL_DEG", "2.0")))
        ik_damping = float(os.environ.get("YAM_IK_DAMPING", "1e-1"))
        success, q = self.kinematics[arm].ik(
            target,
            _GRASP_SITE,
            init_q=init_q,
            pos_threshold=pos_tol,
            ori_threshold=ori_tol,
            damping=ik_damping,
        )
        self._last_ik_success[arm] = bool(success)
        if not success:
            # Detail per failure: rejected target vs current (reachable) pose.
            # Large dpos => target positionally out of reach (frame/extrinsic/
            # cross-embodiment workspace); small dpos => orientation/singularity.
            T_cur = self.kinematics[arm].fk(init_q)
            cur_xyz = T_cur[:3, 3]
            cur_ypr = R.from_matrix(T_cur[:3, :3]).as_euler("ZYX", degrees=True)
            dpos = float(np.linalg.norm(np.asarray(ee_pose[:3]) - cur_xyz) * 1000)
            # |target| / |current| are distances from the ARM BASE (FK origin).
            # YAM reach ~= 400-500 mm. If |target| >> reach => commanded pose is
            # OUT OF WORKSPACE (model/data reaches where the arm can't). If
            # |target| ~ |current| (both ~reach) and dpos small => the position is
            # fine and IK is choking on the near-vertical (pitch~88) orientation.
            tgt_reach = float(np.linalg.norm(np.asarray(ee_pose[:3])) * 1000)
            cur_reach = float(np.linalg.norm(cur_xyz) * 1000)
            print(
                f"[YAMInterface][ik-fail] {arm}: |target|={tgt_reach:.0f}mm "
                f"|current|={cur_reach:.0f}mm dpos={dpos:.0f}mm  "
                f"(reach~400-500mm; |target|>>reach=>OUT OF WORKSPACE; "
                f"|target|~|current| & small dpos=>orientation/singularity) | "
                f"TARGET xyz={np.round(ee_pose[:3], 3)} "
                f"ypr={np.round(np.rad2deg(ee_pose[3:6]), 1)} | "
                f"CURRENT xyz={np.round(cur_xyz, 3)} ypr={np.round(cur_ypr, 1)}"
            )
        q = np.asarray(q, dtype=np.float64)[: self.N_ARM_JOINTS]
        # Flag IK branch flips: a large joint jump from the warm-start seed for a
        # (gradual) EE target means the solver landed in a different/flipped branch
        # — the usual cause of the wrist physically flipping even when the commanded
        # EE pose is sane (common near wrist singularities, e.g. gripper vertical).
        dq = np.abs(((q - init_q + np.pi) % (2 * np.pi)) - np.pi)
        reject_jump = np.deg2rad(float(os.environ.get("YAM_IK_MAX_JUMP_DEG", "90")))
        if dq.max() > np.deg2rad(45):
            j = int(np.argmax(dq))
            print(
                f"[YAMInterface][ik-jump] {arm}: joint {j} "
                f"{np.rad2deg(init_q[j]):+.0f}->{np.rad2deg(q[j]):+.0f} deg "
                f"(|dq|={np.rad2deg(dq).round(0)}) for a target that should be near "
                f"the seed — IK picked a flipped branch."
            )
            # A jump this large for a target ~at the seed is a branch flip, not a
            # real motion. Commanding it IS the visible wrist flip. Treat as an IK
            # failure so set_pose holds instead of flinging the arm. Tunable via
            # YAM_IK_MAX_JUMP_DEG (raise to allow large legitimate reconfigs).
            if dq.max() > reject_jump:
                print(
                    f"[YAMInterface][ik-jump] {arm}: jump exceeds "
                    f"{np.rad2deg(reject_jump):.0f} deg -> rejecting (holding) to "
                    f"avoid commanding a flip."
                )
                self._last_ik_success[arm] = False
        return q

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self):
        """Set all torques to zero and release the CAN connections."""
        for arm in self.arms:
            closer = getattr(self.controller[arm], "close", None)
            if callable(closer):
                closer()
        # Recorders are always interface-owned now (no borrowing) — release them.
        for rec in self.recorders.values():
            try:
                rec.close()
            except BaseException:
                pass


if __name__ == "__main__":
    ri = YAMInterface(arms=["right"])
    print("joints:", ri.get_joints("right"))
    print("ee pose 6d:", ri.get_pose_6d("right"))
    ri.close()
