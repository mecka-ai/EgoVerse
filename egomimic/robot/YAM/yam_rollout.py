"""Rollout for the I2RT YAM bimanual arms — the YAM counterpart of rollout.py.

Standalone, paralleling collect_yam_demo.py: drives YAM arms with a trained pi
policy via YAMInterface and the ``yam_bimanual`` embodiment. Forked from the
generic rollout engine (PolicyRollout transform pipeline + keyboard-intervention
control loop) so the sibling rollout.py stays EVA-only.

Launch (from the repo root):
    python egomimic/robot/YAM/yam_rollout.py \\
        --policy-path <ckpt> --arms both --cartesian \\
        --annotation-path <prompt.txt> \\
        --yam-left-can can_follower_l --yam-right-can can_follower_r
"""

import os
import sys

# --- Make sibling (YAM/), robot/, eva src, and the vendored i2rt importable ---
# This file lives at <repo>/egomimic/robot/YAM/yam_rollout.py.
_THIS = os.path.dirname(os.path.abspath(__file__))  # .../egomimic/robot/YAM
_ROBOT_DIR = os.path.dirname(_THIS)  # .../egomimic/robot
_EGOMIMIC_DIR = os.path.dirname(_ROBOT_DIR)  # .../egomimic
_REPO_ROOT = os.path.dirname(_EGOMIMIC_DIR)  # .../<repo root>
for _p in (
    _ROBOT_DIR,  # robot_utils
    _THIS,  # yam_interface, yam_cameras
    os.path.join(
        _ROBOT_DIR, "eva", "eva_ws", "src", "eva"
    ),  # robot_interface (ARX base)
    os.path.join(_REPO_ROOT, "external", "i2rt"),  # i2rt SDK
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import threading
import time
import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings("ignore", message="Can't initialize NVML")

# Set during shutdown so the thread excepthook can swallow the harmless CAN
# socket-teardown errors the i2rt SDK raises from background threads when the
# socket closes mid-send (ValueError: file descriptor cannot be a negative
# integer (-1)). Outside shutdown, defer to the default handler.
_SHUTTING_DOWN = threading.Event()


def _quiet_thread_excepthook(args):
    if _SHUTTING_DOWN.is_set() and issubclass(args.exc_type, (ValueError, OSError)):
        return
    threading.__excepthook__(args)


# (sys.path for robot_utils / robot_interface / yam_interface / i2rt is set up at
# the top of this module so it applies before the imports above.)
import select
import termios
import tty

import cv2
import h5py
import numpy as np
import torch
from robot_utils import RateLoop
from scipy.spatial.transform import Rotation as R
from torch.utils.data import default_collate

from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_data_utils import build_tokenized_collate
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.embodiment.eva import (
    Eva,
    _build_eva_bimanual_revert_eef_frame_transform_list,
)
from egomimic.rldb.embodiment.yam import Yam
from egomimic.utils.egomimicUtils import (
    CameraTransforms,
    cam_frame_to_base_frame,
    interpolate_arr,
)
from egomimic.utils.pose_utils import xyzw_to_wxyz

R_t_e = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ],
    dtype=float,
)

inv_R_t_e = np.linalg.inv(R_t_e)


def ee_pose_to_rot_ee_frame_batch(pose):
    pose = np.asarray(pose)
    xyz = pose[..., :3]
    ypr = pose[..., 3:6]
    R_ee = R.from_euler("ZYX", ypr).as_matrix()
    R_rot = R_t_e @ R_ee
    ypr_rot = R.from_matrix(R_rot).as_euler("ZYX")
    return np.concatenate([xyz, ypr_rot], axis=-1)


def rot_ee_frame_to_ee_pose_batch(pose_rot):
    pose_rot = np.asarray(pose_rot)
    xyz = pose_rot[..., :3]
    ypr = pose_rot[..., 3:6]
    R_rot = R.from_euler("ZYX", ypr).as_matrix()
    R_ee = inv_R_t_e @ R_rot
    ypr_ee = R.from_matrix(R_ee).as_euler("ZYX")
    return np.concatenate([xyz, ypr_ee], axis=-1)


def ee_pose_to_rot_ee_frame(pose):
    return ee_pose_to_rot_ee_frame_batch(pose[None, ...])[0]


# YAM-native flange -> EVA flange axis relabel (columns = EVA axes in YAM flange
# coords): EVA X = -Z_yam, EVA Y = -X_yam, EVA Z = +Y_yam. Measured on the
# training zarrs (grasp-event decomposition, 2026-07-05): EVA/ABC pre-grasp
# descent is +Z in EE coords while YAM's is +Y, and the 180-deg roll ambiguity is
# pinned by -Z_yam pointing world-down at level-approach frames. A constant LOCAL
# relabel (right-multiply) commutes with every world/cam-side transform in the
# pipeline (all left-multiplies), so remapping obs at ingestion and inverting on
# the outgoing command is exact. Valid ONLY from the stock "default" grasp_site
# convention the YAM demos were collected with (see --ee-convention).
YAM_TO_EVA_FLANGE = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=float,
)


def relabel_flange_ypr_batch(pose, E):
    """Post-multiply the rotation of [..., xyz ypr(ZYX)] pose rows by the
    constant axis relabel E. Pure frame relabel: xyz (and anything past index 6)
    is untouched."""
    pose = np.asarray(pose, dtype=np.float64).copy()
    ypr = pose[..., 3:6]
    R_m = R.from_euler("ZYX", ypr).as_matrix()
    pose[..., 3:6] = R.from_matrix(R_m @ E).as_euler("ZYX")
    return pose


def _slerp_resample_cartesian(chunk, target_len):
    """Resample a cartesian action chunk to target_len, interpolating rotation
    with SLERP (quaternion geodesic) instead of linear Euler.

    chunk: (T, D) of per-arm [xyz(3) ypr(3) grip(1)] blocks (D = 7 or 14).
    Linear interp for xyz/gripper; SLERP for the ypr block so interpolation is
    correct THROUGH gimbal lock (pitch~=90, where yaw/roll are degenerate and
    linear-Euler interp sweeps the wrist the wrong way -> flip). Each row's
    rotation is preserved exactly; only the intermediate samples change.
    """
    import scipy.interpolate as _si
    from scipy.spatial.transform import Slerp

    chunk = np.asarray(chunk, dtype=np.float64)
    T, D = chunk.shape
    if T < 2:
        return chunk.astype(np.float32, copy=False)
    old_t = np.linspace(0.0, 1.0, T)
    new_t = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, D), dtype=np.float64)
    for off in range(0, D, 7):  # one 7-dim [xyz ypr grip] block per arm
        blk = chunk[:, off : off + 7]
        out[:, off : off + 3] = _si.interp1d(old_t, blk[:, :3], axis=0)(new_t)
        rots = R.from_euler("ZYX", blk[:, 3:6])
        out[:, off + 3 : off + 6] = Slerp(old_t, rots)(new_t).as_euler("ZYX")
        out[:, off + 6 : off + 7] = _si.interp1d(old_t, blk[:, 6:7], axis=0)(new_t)
    return out.astype(np.float32, copy=False)


def rot_ee_frame_to_ee_pose(pose_rot):
    return rot_ee_frame_to_ee_pose_batch(pose_rot[None, ...])[0]


# Control parameters
DEFAULT_FREQUENCY = 30  # Hz
QUERY_FREQUENCY = 30
DEFAULT_RESAMPLE_LENGTH = 45


def _build_robot_interface(
    arms_list,
    robot="eva",
    yam_channels=None,
    yam_ee_convention="default",
):
    if robot == "yam":
        # YAMInterface is a sibling module (this file lives in egomimic/robot/YAM/);
        # YAM/ is already on sys.path (see the path setup at the top of this file).
        from yam_interface import YAMInterface

        # YAMInterface always opens its cameras, so get_obs() includes
        # front_img_1 / left_wrist_img / right_wrist_img — the keys the obs
        # pipeline consumes (and it raises if no cameras are available).
        return YAMInterface(
            arms=arms_list, channels=yam_channels, ee_frame_convention=yam_ee_convention
        )

    from robot_interface import ARXInterface

    return ARXInterface(arms=arms_list)


class _KeyPoll:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)  # no Enter needed
        return self

    def __exit__(self, exc_type, exc, tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def getch(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


class Rollout(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def rollout_step(self, i):
        pass


class ReplayRollout(Rollout):
    def __init__(self, dataset_path, cartesian):
        super().__init__()
        self.dataset_path = dataset_path
        if not os.path.isfile(self.dataset_path):
            raise FileNotFoundError(f"HDF5 not found: {self.dataset_path}")
        with h5py.File(self.dataset_path, "r") as f:
            if cartesian:
                self.actions = np.asarray(f["actions"]["eepose"][...], dtype=np.float32)
            else:
                self.actions = np.asarray(
                    f["observations"]["joint_positions"][...], dtype=np.float32
                )

    def rollout_step(self, i):
        if i < self.actions.shape[0]:
            return self.actions[i]
        else:
            return None


class PolicyRollout(Rollout):
    # Embodiment class per robot family; both share the cartesian_wristframe_ypr
    # transform pipeline, differing only in the camera extrinsics baked in.
    _ROBOT_EMBODIMENT_CLASS = {"eva": Eva, "yam": Yam}
    _ARM_TO_SUFFIX = {"both": "bimanual", "right": "right_arm", "left": "left_arm"}

    def __init__(
        self,
        arm,
        policy_path,
        query_frequency,
        cartesian,
        extrinsics_key,
        resampled_action_len=None,
        annotation_path=None,
        annotation_text=None,
        robot="eva",
        no_embodiment_prompt=False,
        flange_frame="native",
    ):
        super().__init__()
        self.arm = arm
        self.robot = robot
        # Prompt parity knob: when True the tokenized collate drops the
        # "Embodiment: <name>" block (Task/State blocks unchanged). Must match
        # the checkpoint's data config (embodiment_label: false).
        self.no_embodiment_prompt = no_embodiment_prompt
        if flange_frame not in ("native", "eva"):
            raise ValueError(f"unknown flange_frame '{flange_frame}'")
        # "eva": the checkpoint speaks EVA's flange convention -> relabel the
        # robot's native flange rotations into it on the obs side (R @ E) and
        # back on the command side (R @ E^T). Identity for --robot eva.
        self.flange_remap = (
            YAM_TO_EVA_FLANGE if (flange_frame == "eva" and robot == "yam") else None
        )
        if flange_frame == "eva":
            if robot != "yam":
                print(
                    "[rollout] --flange-frame eva is a no-op for --robot eva "
                    "(the checkpoint convention IS the native one)."
                )
            else:
                print(
                    "[rollout] flange-frame remap ON: YAM-native flange -> EVA "
                    "convention on obs, inverted on outgoing commands "
                    "(EVA X=-Z_yam, Y=-X_yam, Z=+Y_yam)."
                )
        self.policy_path = policy_path
        self.query_frequency = query_frequency
        self.cartesian = cartesian
        # Embodiment name/id derived from robot family + arm, e.g.
        # ("yam","both") -> "yam_bimanual" -> id 17.
        self.embodiment_name = f"{robot}_{self._ARM_TO_SUFFIX[self.arm]}"
        self.embodiment_id = get_embodiment_id(self.embodiment_name)
        self.embodiment_cls = self._ROBOT_EMBODIMENT_CLASS[robot]

        # Camera frames MUST match the embodiment:
        #   * extrinsics (camera->arm-base) drive the predicted-action cam->base
        #     mapping in rollout_step  -> EXTRINSICS["yam"]["left"|"right"] for YAM.
        #   * intrinsics drive the action-overlay viz                -> INTRINSICS["yam"]
        #     (== YAM_INTRINSICS) for YAM.
        # The intrinsics key is taken straight from the embodiment so it can never
        # drift from the viz path, and we PREFER the embodiment's own extrinsics key
        # (Yam.EXTRINSICS_KEY = "yam") so self.extrinsics stays consistent with the
        # transform pipeline, which also uses embodiment_cls.EXTRINSICS_KEY.
        intrinsics_key = self.embodiment_cls.VIZ_INTRINSICS_KEY
        emb_extrinsics_key = getattr(
            self.embodiment_cls, "EXTRINSICS_KEY", extrinsics_key
        )
        if extrinsics_key != emb_extrinsics_key:
            print(
                f"[rollout] WARNING: --extrinsics-key='{extrinsics_key}' != "
                f"{self.embodiment_name} default '{emb_extrinsics_key}'; using "
                f"'{emb_extrinsics_key}' to stay consistent with the transform pipeline."
            )
            extrinsics_key = emb_extrinsics_key
        self.camera_transforms = CameraTransforms(
            intrinsics_key=intrinsics_key, extrinsics_key=extrinsics_key
        )
        self.extrinsics = self.camera_transforms.extrinsics
        self.intrinsics = self.camera_transforms.intrinsics
        print(
            f"[rollout] camera frames -> extrinsics='{extrinsics_key}' "
            f"(EXTRINSICS, arms={sorted(self.extrinsics.keys())}), "
            f"intrinsics='{intrinsics_key}' (INTRINSICS)"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_device = self.device
        print(f"[rollout] Loading policy from {self.policy_path}")
        self.policy = self._load_policy()
        # Camera-frame copy of the latest predicted chunk (post-revert, pre
        # cam->base) — the frame the preview overlay projects in.
        self.debug_actions = None
        self.resampled_action_len = resampled_action_len

        # Decouple the MODEL embodiment from the HARDWARE robot. The checkpoint's
        # norm stats, action converter, and forward_eval output key are all keyed to
        # its OWN embodiment (its config `domains`), which need not equal the robot
        # we drive. e.g. an eva-keyed checkpoint (trained on yam data ported into the
        # eva embodiment) runs on the YAM robot: hardware/cameras/extrinsics stay
        # --robot yam (embodiment_cls=Yam), but batch keying / normalize / from32 /
        # the preds key must use the checkpoint's 'eva_bimanual'. Without this,
        # process_batch would key the batch as 'yam_bimanual' (id 17) and
        # normalize_data raises "Missing normalization stats for embodiment 17".
        model_domains = getattr(self.policy.model, "domains", None)
        model_domain = model_domains[0] if model_domains else self.embodiment_name
        if model_domain != self.embodiment_name:
            print(
                f"[rollout] checkpoint embodiment '{model_domain}' != robot embodiment "
                f"'{self.embodiment_name}': using '{model_domain}' for MODEL I/O "
                f"(norm stats / converter / preds key) and robot='{self.robot}' "
                f"(embodiment_cls={self.embodiment_cls.__name__}) for HARDWARE + cameras "
                f"+ extrinsics. The wrist-frame action repr is extrinsic-invariant, so "
                f"the cross-embodiment run hinges on matching EE-pose/frame conventions."
            )
        self.embodiment_name = model_domain
        self.embodiment_id = get_embodiment_id(model_domain)

        # Match the action representation to the LOADED checkpoint's converter, so a
        # 6D checkpoint (RobotBimanualCartesian6D, trained on cartesian_wristframe_6d)
        # and a legacy ypr/euler checkpoint (RobotBimanualCartesianEuler) both work
        # without a flag. The converter is reconstructed from the ckpt's own config
        # at load (action_registry), so it is authoritative — hardcoding a mode here
        # would silently mis-split the obs/action vectors for the other kind.
        ac_key = self.policy.model.ac_keys[self.embodiment_id]
        _conv = self.policy.model.action_registry.get(self.embodiment_id, ac_key)
        self.use_6d = "6D" in type(_conv).__name__
        print(
            f"[rollout] action representation: "
            f"{'6D wristframe' if self.use_6d else 'ypr wristframe'} "
            f"(checkpoint converter = {type(_conv).__name__})"
        )
        if self.use_6d:
            # obs/proprio and the model's 32-block actions are continuous-6D
            # (per-arm 9-dim pose). The revert collapses 6D->ypr at its tail, so
            # everything downstream of revert stays the 14-dim camframe-ypr contract.
            self.transform_list = self.embodiment_cls.get_transform_list(
                mode="cartesian_wristframe_6d"
            )
            self.revert_transform_list = (
                _build_eva_bimanual_revert_eef_frame_transform_list(rot_repr="6d")
            )
        else:
            # Legacy ypr: obs/proprio are 14-dim ypr and from32 already yields ypr.
            self.transform_list = self.embodiment_cls.get_transform_list(
                mode="cartesian_wristframe_ypr"
            )
            self.revert_transform_list = (
                _build_eva_bimanual_revert_eef_frame_transform_list(is_quat=False)
            )
        self.annotation = None
        self._tokenizer = None
        self.collate_fn = default_collate
        if annotation_path is not None:
            if not os.path.isfile(annotation_path):
                print(
                    f"[rollout] WARNING: annotation file not found: {annotation_path}  (continuing without annotation)"
                )
            else:
                with open(annotation_path, "r") as f:
                    self.annotation = f.read().strip()
                self.collate_fn = self._make_collate(self.annotation)
        # Inline prompt (e.g. --annotation "Fold the shirt") takes effect only if
        # no annotation file was successfully loaded above.
        if self.annotation is None and annotation_text:
            self.annotation = annotation_text.strip()
            self.collate_fn = self._make_collate(self.annotation)
            print(f"[rollout] Using inline annotation prompt: '{self.annotation}'")

    def _make_collate(self, default_prompt):
        """Tokenizing collate with the SAME prompt format the checkpoint trained on.

        The pi0.5 training configs (e.g. data=yam_pick_hat_wrist_pi) set
        ``proprio: true`` + ``embodiment_label: true``, so every training prompt
        is ``"Task: <text>, Embodiment: <name>, State: <256-bin proprio>;\\nAction: "``.
        Rollout previously built the collate WITHOUT those flags, so the model
        was conditioned on a bare prompt it never saw in training — no Task
        anchor, no Embodiment block, and no discretized State splice (a proprio
        pathway the model learned to read). NOTE: proprio_keys must be passed
        explicitly here (this branch's collate has no default), and the batch's
        "embodiment" key must be the integer id for the Embodiment splice.

        --no-embodiment-prompt drops ONLY the Embodiment block (for checkpoints
        trained with embodiment_label: false); Task/State stay.
        """
        return build_tokenized_collate(
            max_length=128,
            model_name="google/paligemma-3b-mix-224",
            sampling_mode="first",
            annotation_key="annotations",
            default_prompt=default_prompt,
            proprio_keys=["observations.state.ee_pose"],
            state_num_bins=256,
            proprio=True,
            embodiment_label=not self.no_embodiment_prompt,
        )

    LOCAL_WEIGHT_PATH = os.path.join(
        _EGOMIMIC_DIR, "algo", "pi_checkpoints", "pi05_base_pytorch"
    )

    @classmethod
    def _patch_checkpoint_paths(cls, ckpt_path):
        """Rewrite pytorch_weight_path in the checkpoint's saved config
        to point to the local base model weights."""
        import torch as _torch
        from omegaconf import DictConfig, OmegaConf

        ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ht = ckpt.get("hyper_parameters", {}).get("config_tree")
        if ht is None:
            return ckpt_path
        if isinstance(ht, DictConfig):
            cfg = OmegaConf.to_container(ht, resolve=True)
        else:
            cfg = ht
        # Navigate to pytorch_weight_path in the config
        robomimic = cfg.get("model", {}).get("robomimic_model", {})
        config = robomimic.get("config", {})
        old_path = config.get("pytorch_weight_path")
        if old_path is None or old_path == cls.LOCAL_WEIGHT_PATH:
            return ckpt_path
        print(
            f"[rollout] Patching pytorch_weight_path: {old_path} -> {cls.LOCAL_WEIGHT_PATH}"
        )
        config["pytorch_weight_path"] = cls.LOCAL_WEIGHT_PATH
        ckpt["hyper_parameters"]["config_tree"] = OmegaConf.create(cfg)
        patched_path = ckpt_path + ".patched"
        _torch.save(ckpt, patched_path)
        print(f"[rollout] Patched checkpoint saved to {patched_path}")
        return patched_path

    def _load_policy(self):
        patched_path = self._patch_checkpoint_paths(self.policy_path)
        policy = ModelWrapper.load_from_checkpoint(
            patched_path, weights_only=False, map_location="cpu"
        )
        policy = policy.to(self.policy_device)
        policy.eval()
        policy.model.device = self.policy_device

        # Unwrap torch.compile on sample_actions to avoid massive first-call
        # compilation overhead (~50s). The compiled version (instance attribute)
        # shadows the original class method; deleting it restores the fast
        # uncompiled path which is sufficient for real-time rollout.
        pi0 = policy.model.nets["policy"]
        if "sample_actions" in vars(pi0):
            del pi0.sample_actions
            print(
                "[rollout] Disabled torch.compile on sample_actions for rollout inference"
            )

        # Verify model is on GPU
        try:
            p = next(pi0.parameters())
            print(f"[rollout] Model device: {p.device}, dtype: {p.dtype}")
            if not p.is_cuda:
                print(
                    "[rollout] WARNING: model is NOT on GPU — inference will be very slow!"
                )
        except StopIteration:
            pass

        if getattr(policy.model, "diffusion", False):
            for head in policy.model.nets.policy.heads:
                if isinstance(policy.model.nets.policy.heads[head], DenoisingPolicy):
                    policy.model.nets.policy.heads[head].num_inference_steps = 10
        return policy

    def _downsample_chunk(self, chunk: np.ndarray, target_len: int) -> np.ndarray:
        if target_len is None or target_len <= 0 or chunk.shape[0] == target_len:
            return chunk.astype(np.float32, copy=False)

        # chunk: (T, D) -> (1, T, D) and back
        if self.cartesian:
            # SLERP rotation interp is correct through gimbal lock (pitch~=90),
            # where linear-Euler interp sweeps the wrist the wrong way -> flip.
            out = _slerp_resample_cartesian(chunk, target_len)
        else:
            out = interpolate_arr(chunk[None, ...], target_len)[0]

        return out.astype(np.float32, copy=False)

    def rollout_step(self, i, obs):
        if i % self.query_frequency == 0:
            start_infer_t = time.time()
            transform_list_batch = self.process_obs_for_transform_list(obs)
            for transform in self.transform_list:
                transform_list_batch = transform.transform(transform_list_batch)
            # Current EE pose in CAMERA frame (un-normalized) — needed to revert
            # the wrist-frame predictions back to camera frame below. Width matches
            # self.use_6d: bimanual (20,) in 6D (per-arm xyz+c1+c2 (9) + grip (1))
            # or (14,) in ypr (per-arm xyz+ypr (6) + grip (1)). Grab it BEFORE
            # collate (collate adds a batch dim) and BEFORE the model normalizes it
            # inside process_batch_for_training.
            obs_state_camframe = np.asarray(
                transform_list_batch["observations.state.ee_pose"]
            )
            transform_list_batch = self.collate_fn([transform_list_batch])
            embodiment_name = self.embodiment_name
            batch = {
                embodiment_name: transform_list_batch,
            }
            processed_batch = self.policy.model.process_batch_for_training(batch)
            preds = self.policy.model.forward_eval(processed_batch)[
                f"{embodiment_name}_actions_cartesian"
            ]
            self.actions = preds.detach().cpu().numpy().squeeze()
            # Predictions are eef/wrist-relative (6D (T,20) if self.use_6d, else
            # ypr (T,14)). Revert them to CAMERA frame by composing with the
            # current EE pose so the overlay (camera-frame pinhole projection) and
            # the cam->base mapping below are both correct. Skipping this projects
            # the origin-centered wrist-frame chunk to garbage and commands the
            # wrong pose. The 6D revert collapses 6D->ypr at its tail, so this
            # always emits a (T, 14) camframe-ypr chunk; single-arm is unsupported.
            if self.cartesian and self.arm == "both":
                revert_batch = {
                    "observations.state.ee_pose": obs_state_camframe,
                    "actions_cartesian": self.actions,
                }
                for _t in self.revert_transform_list:
                    revert_batch = _t.transform(revert_batch)
                self.actions = np.asarray(
                    revert_batch["actions_cartesian"], dtype=np.float32
                )
            self.debug_actions = self.actions.copy()

            if self.cartesian:
                if self.arm == "both":
                    left_actions = self.actions[:, :7]
                    right_actions = self.actions[:, 7:]

                    transformed_left = cam_frame_to_base_frame(
                        left_actions[:, :6].copy(), self.extrinsics["left"]
                    )
                    transformed_right = cam_frame_to_base_frame(
                        right_actions[:, :6].copy(), self.extrinsics["right"]
                    )
                    transformed_left = rot_ee_frame_to_ee_pose_batch(transformed_left)
                    transformed_right = rot_ee_frame_to_ee_pose_batch(transformed_right)
                    if self.flange_remap is not None:
                        # Commands leave the policy in the checkpoint's flange
                        # convention; relabel back to the robot's native frame
                        # (inverse of the obs-side remap) before IK.
                        transformed_left = relabel_flange_ypr_batch(
                            transformed_left, self.flange_remap.T
                        )
                        transformed_right = relabel_flange_ypr_batch(
                            transformed_right, self.flange_remap.T
                        )
                    gripper_left = left_actions[:, 6:7]
                    gripper_right = right_actions[:, 6:7]
                    if left_actions.shape[1] == 7:
                        left_actions = np.hstack([transformed_left, gripper_left])
                    else:
                        left_actions = transformed_left
                    if right_actions.shape[1] == 7:
                        right_actions = np.hstack([transformed_right, gripper_right])
                    else:
                        right_actions = transformed_right
                    self.actions = np.hstack([left_actions, right_actions])
                else:
                    eepose = rot_ee_frame_to_ee_pose_batch(self.actions[:, :6].copy())
                    self.actions[:, :6] = eepose
                    transformed_6dof = cam_frame_to_base_frame(
                        self.actions[:, :6].copy(), self.extrinsics[self.arm]
                    )
                    if self.flange_remap is not None:
                        transformed_6dof = relabel_flange_ypr_batch(
                            transformed_6dof, self.flange_remap.T
                        )
                    # Preserve gripper if present (7th value)
                    gripper = self.actions[:, 6:7]
                    if self.actions.shape[1] == 7:
                        self.actions = np.hstack([transformed_6dof, gripper])
                    else:
                        self.actions = transformed_6dof

            if self.resampled_action_len is not None:
                self.actions = self._downsample_chunk(
                    self.actions, self.resampled_action_len
                )

            print(f"Inference time: {time.time() - start_infer_t:.2f}s")

        act_i = i % self.query_frequency
        return self.actions[act_i]

    def process_obs_for_transform_list(self, obs):
        # front camera: obs["front_img_1"] is BGR, shape [H, W, 3]
        front = torch.from_numpy(obs["front_img_1"][None, ...])  # [1, H, W, 3]
        front = front[..., [2, 1, 0]]  # BGR -> RGB
        front = front.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
        front = front.squeeze()
        data = {
            # Keep rollout-local keys, PI schematic aliases, and canonical
            # dataset zarr keys so checkpoints with different data schematics
            # can all resolve the same image tensor.
            "front_img_1": front,
            "base_0_rgb": front,
            "observations.images.front_img_1": front,
            "pad_mask": torch.ones((1, 100, 1), dtype=torch.bool),
        }

        eepose = obs["ee_poses"]
        if self.flange_remap is not None:
            # Relabel each arm's native flange rotation into the checkpoint's
            # convention (base-frame [xyz ypr grip] blocks; xyz/grip untouched).
            eepose = np.asarray(eepose, dtype=np.float64).copy()
            eepose[0:6] = relabel_flange_ypr_batch(eepose[0:6], self.flange_remap)
            eepose[7:13] = relabel_flange_ypr_batch(eepose[7:13], self.flange_remap)

        if self.arm in ["right", "both"]:
            right = torch.from_numpy(
                obs["right_wrist_img"][None, ...]
            )  # [1, H, W, 3] BGR
            right = right[..., [2, 1, 0]]  # BGR -> RGB
            right = right.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
            data["right_wrist_img"] = right.squeeze()
            data["right_wrist_0_rgb"] = data["right_wrist_img"]
            data["observations.images.right_wrist_img"] = data["right_wrist_img"]
            right_ee_pose = eepose[7:13]
            right_ee_pose = ee_pose_to_rot_ee_frame(right_ee_pose)
            right_ypr = right_ee_pose[..., 3:6]
            right_xyzw = R.from_euler("ZYX", right_ypr).as_quat()
            right_wxyz = xyzw_to_wxyz(right_xyzw)
            right_xyzwxyz = np.concatenate([eepose[7:10], right_wxyz], axis=-1)
            data["right.obs_ee_pose"] = torch.from_numpy(right_xyzwxyz).reshape(-1)
            data["right.obs_gripper"] = torch.from_numpy(eepose[13:14]).reshape(-1)
            right_gripper = torch.from_numpy(eepose[13:14]).view(1, 1).repeat(45, 1)
            data["right.cmd_gripper"] = right_gripper
            right_cmd_ee_pose = torch.from_numpy(right_xyzwxyz).view(1, 7).repeat(45, 1)
            data["right.cmd_ee_pose"] = right_cmd_ee_pose

        if self.arm in ["left", "both"]:
            left = torch.from_numpy(
                obs["left_wrist_img"][None, ...]
            )  # [1, H, W, 3] BGR
            left = left[..., [2, 1, 0]]  # BGR -> RGB
            left = left.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
            data["left_wrist_img"] = left.squeeze()
            data["left_wrist_0_rgb"] = data["left_wrist_img"]
            data["observations.images.left_wrist_img"] = data["left_wrist_img"]
            left_ee_pose = eepose[0:6]
            left_ee_pose = ee_pose_to_rot_ee_frame(left_ee_pose)
            left_ypr = left_ee_pose[..., 3:6]
            left_xyzw = R.from_euler("ZYX", left_ypr).as_quat()
            left_wxyz = xyzw_to_wxyz(left_xyzw)
            left_xyzwxyz = np.concatenate([eepose[:3], left_wxyz], axis=-1)
            data["left.obs_ee_pose"] = torch.from_numpy(left_xyzwxyz).reshape(-1)
            data["left.obs_gripper"] = torch.from_numpy(eepose[6:7]).reshape(-1)
            left_gripper = torch.from_numpy(eepose[6:7]).view(1, 1).repeat(45, 1)
            data["left.cmd_gripper"] = left_gripper
            left_cmd_ee_pose = torch.from_numpy(left_xyzwxyz).view(1, 7).repeat(45, 1)
            data["left.cmd_ee_pose"] = left_cmd_ee_pose

        # Integer ids, matching training's ZarrDataset (which stores
        # get_embodiment_id(...) for both keys). The collate's Embodiment splice
        # does int(sample["embodiment"]) — a string here would crash it.
        data["embodiment"] = self.embodiment_id
        data["metadata.robot_name"] = self.embodiment_id

        if self.annotation is not None:
            data["annotations"] = [self.annotation]

        return data

    def load_annotation(self, annotation_path):
        """Load a new annotation file, building the tokenized collate only if needed.

        The annotation text flows through data["annotations"] at each inference
        step, so updating self.annotation is sufficient when the tokenized
        collate already exists.  We only build it when the collate is still the
        plain default_collate (i.e. no annotation was provided at init time).

        Returns True on success, False if the file could not be loaded.
        """
        if not os.path.isfile(annotation_path):
            print(f"[rollout] WARNING: annotation file not found: {annotation_path}")
            return False
        with open(annotation_path, "r") as f:
            self.annotation = f.read().strip()
        if self.collate_fn is default_collate:
            self.collate_fn = self._make_collate(self.annotation)
        print(
            f"[rollout] Loaded new annotation from {annotation_path}: '{self.annotation}'"
        )
        return True

    def reset(self):
        self.actions = None
        self.debug_actions = None
        self.policy.eval()


# Training front-image geometry: the zarr conversion (eva_to_zarr
# resize_video_thwc) stores front_img_1 STRETCH-resized from the stereo
# pipeline's native re-aimed crop (~1420x880) to 640x480, so the policy,
# eval viz, and INTRINSICS["yam"] all live at 640x480. The live
# AtlasStereoCamera emits the NATIVE crop instead.
_FRONT_TRAINING_WH = (640, 480)


def _match_training_front_resolution(obs):
    """Stretch-resize ``obs["front_img_1"]`` in place to the training resolution.

    Without this the policy sees a differently-scaled, un-stretched scene at
    rollout than it trained on (a real train/deploy distribution shift), and
    the preview overlays project with a K that does not match the image.
    Applied once at obs ingestion so the model input, the preview overlays, and
    ``INTRINSICS["yam"]`` stay mutually consistent. No-op for frames already at
    640x480 (e.g. the wrist cams or a future collection stack that resizes
    on-camera).
    """
    f = obs.get("front_img_1")
    if f is not None and f.shape[:2] != (_FRONT_TRAINING_WH[1], _FRONT_TRAINING_WH[0]):
        obs["front_img_1"] = cv2.resize(
            f, _FRONT_TRAINING_WH, interpolation=cv2.INTER_AREA
        )
    return obs


PREVIEW_WINDOW = "pi predicted actions"


def _front_img_to_uint8_hwc(front_img):
    """Coerce a front-cam frame (tensor or ndarray, CHW or HWC) to HWC uint8."""
    if isinstance(front_img, torch.Tensor):
        if front_img.dim() == 4:
            front_img = front_img[0].permute(1, 2, 0).cpu().numpy()
        elif front_img.dim() == 3 and front_img.shape[0] == 3:
            front_img = front_img.permute(1, 2, 0).cpu().numpy()
        else:
            front_img = front_img.cpu().numpy()
    elif front_img.ndim == 3 and front_img.shape[0] == 3:
        front_img = front_img.transpose(1, 2, 0)
    return np.ascontiguousarray(front_img).astype(np.uint8)


def render_pred_overlay(actions, front_img):
    """Draw the predicted action chunk over the front image.

    ``actions`` must be the policy output in the CAMERA frame (i.e.
    ``policy.debug_actions``, captured before the cam->base transform). Returns a
    BGR uint8 image via the same overlay pipeline as the offline eval.
    """
    front_hwc = _front_img_to_uint8_hwc(front_img)
    actions = np.asarray(actions).squeeze()
    viz_batch = {
        "observations.images.front_img_1": torch.from_numpy(front_hwc[None, ...]),
        "actions_cartesian": torch.from_numpy(
            actions.astype(np.float32, copy=False)[None, ...]
        ),
    }
    return Yam.viz_transformed_batch(viz_batch, mode="traj+rotation")


def _compose_with_wrist_cams(overlay, obs, arms):
    """Stack the front-cam action overlay with the raw wrist-cam frames so the
    preview window shows ``[left wrist | front (+overlay) | right wrist]``.

    Wrist frames (obs["left_wrist_img"] / ["right_wrist_img"], BGR HWC) are
    resized to the overlay height and hstacked; a missing cam (single-arm run)
    is skipped. Returns the composite BGR image, or the bare overlay if there
    are no wrist cams to add.
    """
    h = overlay.shape[0]

    def _panel(key, label):
        img = obs.get(key)
        if img is None:
            return None
        p = _front_img_to_uint8_hwc(img)
        scale = h / p.shape[0]
        p = cv2.resize(p, (max(1, int(round(p.shape[1] * scale))), h))
        cv2.putText(p, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(
            p, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        return p

    panels = []
    if arms in ("left", "both"):
        lp = _panel("left_wrist_img", "left wrist")
        if lp is not None:
            panels.append(lp)
    panels.append(overlay)
    if arms in ("right", "both"):
        rp = _panel("right_wrist_img", "right wrist")
        if rp is not None:
            panels.append(rp)
    return np.hstack(panels) if len(panels) > 1 else overlay


def preview_and_confirm(actions, obs, step_i, annotation=None, arms="both"):
    """Show the predicted-action overlay and BLOCK until the operator decides.

    The arm is NOT commanded until this returns. Keys (focus the preview window):
        ENTER / SPACE / e -> execute this chunk
        s                 -> skip this chunk (arm holds until the next prediction)
        q / ESC           -> quit the rollout
    Returns 'execute', 'skip', or 'quit'.
    """
    overlay = render_pred_overlay(actions, obs["front_img_1"])
    overlay = _compose_with_wrist_cams(overlay, obs, arms)
    banner = f"step {step_i}" + (f"  |  prompt: {annotation}" if annotation else "")
    hint = "ENTER/space = EXECUTE    s = skip    q = quit"
    h = overlay.shape[0]
    for txt, y, fg in ((banner, 24, (0, 255, 255)), (hint, h - 14, (255, 255, 255))):
        cv2.putText(overlay, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(overlay, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fg, 1)

    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.imshow(PREVIEW_WINDOW, overlay)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (13, 32, ord("e")):  # Enter / Space / e
            return "execute"
        if key == ord("s"):
            return "skip"
        if key in (ord("q"), 27):  # q / ESC
            return "quit"


def preview_live_update(actions, obs, step_i, annotation=None, arms="both"):
    """Non-blocking live overlay: refresh the window with the latest predicted
    chunk and return immediately (the rollout keeps executing). Returns True if
    the operator pressed q/ESC in the window (quit), else False."""
    overlay = render_pred_overlay(actions, obs["front_img_1"])
    overlay = _compose_with_wrist_cams(overlay, obs, arms)
    banner = f"step {step_i} (LIVE)" + (f"  |  {annotation}" if annotation else "")
    cv2.putText(overlay, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(
        overlay, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
    )
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.imshow(PREVIEW_WINDOW, overlay)
    return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)


def preview_pump():
    """Pump the OpenCV GUI event loop so the live window stays responsive
    between predictions. No-op if no window exists."""
    cv2.waitKey(1)


def reset_rollout(ri, policy):
    print("Resetting rollout: going home + clearing policy state")
    if isinstance(policy, ReplayRollout):
        return
    ri.set_home()
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "actions"):
        policy.actions = None
    if hasattr(policy, "debug_actions"):
        policy.debug_actions = None


def main(
    arms,
    frequency,
    cartesian,
    query_frequency=None,
    policy_path=None,
    dataset_path=None,
    resampled_action_len=None,
    annotation_path=None,
    annotation_text=None,
    robot="yam",
    extrinsics_key=None,
    yam_channels=None,
    yam_ee_convention="default",
    preview=False,
    preview_live=False,
    no_execute=False,
    no_embodiment_prompt=False,
    flange_frame="native",
):
    threading.excepthook = _quiet_thread_excepthook

    # YAM_TO_EVA_FLANGE was measured against demos collected with the stock
    # "default" grasp_site convention; from any other hardware frame the
    # constant is wrong, so refuse rather than silently command bad rotations.
    if flange_frame == "eva" and robot == "yam" and yam_ee_convention != "default":
        raise ValueError(
            "--flange-frame eva requires --ee-convention default (the relabel "
            f"constant is measured from that frame), got '{yam_ee_convention}'."
        )

    # Default camera extrinsics per robot family (override with --extrinsics-key).
    if extrinsics_key is None:
        extrinsics_key = "yam" if robot == "yam" else "x5Dec13_2"

    if arms == "both":
        arms_list = ["right", "left"]
    elif arms == "right":
        arms_list = ["right"]
    else:
        arms_list = ["left"]

    ri = _build_robot_interface(
        arms_list=arms_list,
        robot=robot,
        yam_channels=yam_channels,
        yam_ee_convention=yam_ee_convention,
    )

    if policy_path is not None:
        rollout_type = "policy"
        policy = PolicyRollout(
            arm=arms,
            policy_path=policy_path,
            query_frequency=query_frequency,
            cartesian=cartesian,
            extrinsics_key=extrinsics_key,
            resampled_action_len=resampled_action_len,
            annotation_path=annotation_path,
            annotation_text=annotation_text,
            robot=robot,
            no_embodiment_prompt=no_embodiment_prompt,
            flange_frame=flange_frame,
        )
    elif dataset_path is not None:
        rollout_type = "replay"
        policy = ReplayRollout(dataset_path=dataset_path, cartesian=cartesian)
    else:
        raise ValueError("Must provide either --policy-path or --dataset-path.")

    if no_execute:
        print(
            "[rollout] --no-execute: DRY RUN. Predicting + visualizing only; the arm "
            "will NOT be commanded by the policy (the initial set_home still runs)."
        )

    def _enter_intervention(kp, policy, rollout_type):
        """Pause rollout and wait for user command.

        Restores the terminal to cooked mode so the user can type full
        commands, then re-enters cbreak mode before returning.

        Returns one of:
            "continue"  – resume rollout
            "restart"   – restart rollout
            "quit"      – exit program
        """
        # Restore normal terminal so the user can type freely
        termios.tcsetattr(kp.fd, termios.TCSADRAIN, kp.old)
        print("\n--- INTERVENTION (rollout paused) ---")
        print("  c            : continue rollout")
        print("  h            : send arms to home (does not clear policy state)")
        print("  a <path>     : load new annotation file")
        print("  r            : restart rollout")
        print("  q            : quit")

        while True:
            try:
                cmd = input("> ").strip()
            except EOFError:
                tty.setcbreak(kp.fd)
                return "quit"

            if cmd == "c":
                print("Resuming rollout.")
                tty.setcbreak(kp.fd)
                return "continue"
            elif cmd == "q":
                tty.setcbreak(kp.fd)
                return "quit"
            elif cmd == "h":
                print("Sending arms to home...")
                ri.set_home()
                print(
                    "Arms at home. Still paused — c to continue, r to restart, q to quit."
                )
            elif cmd == "r":
                tty.setcbreak(kp.fd)
                return "restart"
            elif cmd.startswith("a "):
                ann_path = cmd[2:].strip()
                if not ann_path:
                    print("Usage: a <annotation_path>")
                    continue
                if rollout_type != "policy" or not isinstance(policy, PolicyRollout):
                    print("Annotation loading is only supported for policy rollouts.")
                    continue
                policy.load_annotation(ann_path)
            else:
                print(f"Unknown command: '{cmd}'. Use c / h / a <path> / r / q.")

    try:
        with _KeyPoll() as kp:
            reset_rollout(ri, policy)
            # Enter intervention at startup so the user decides when to begin
            result = _enter_intervention(kp, policy, rollout_type)
            if result == "quit":
                print("Quit requested.")
                return
            if result == "restart":
                reset_rollout(ri, policy)

            chunk_approved = True  # gates execution when --preview is on
            while True:  # restartable
                with RateLoop(frequency=frequency, verbose=True) as loop:
                    for step_i in loop:
                        ch = kp.getch()
                        if ch is not None:
                            # Any key press triggers intervention
                            result = _enter_intervention(kp, policy, rollout_type)
                            if result == "quit":
                                print("Quit requested.")
                                return
                            elif result == "restart":
                                print("Restart requested.")
                                reset_rollout(ri, policy)
                                result = _enter_intervention(kp, policy, rollout_type)
                                if result == "quit":
                                    return
                                if result == "restart":
                                    reset_rollout(ri, policy)
                                break
                            if hasattr(policy, "actions"):
                                policy.actions = None
                            break

                        actions = None
                        if rollout_type == "policy":
                            obs = ri.get_obs()
                            _match_training_front_resolution(obs)
                            actions = policy.rollout_step(step_i, obs)
                        elif rollout_type == "replay":
                            actions = policy.rollout_step(step_i)
                        else:
                            raise ValueError(f"Invalid rollout type: {rollout_type}")

                        if actions is None:
                            print("Finish rollout.")
                            reset_rollout(ri, policy)
                            result = _enter_intervention(kp, policy, rollout_type)
                            if result == "quit":
                                return
                            if result == "restart":
                                reset_rollout(ri, policy)
                            break

                        # Preview-and-confirm gate: once per predicted chunk, show
                        # the overlay and block until the operator approves. The
                        # decision holds for the whole chunk (the next prediction,
                        # query_frequency steps later, re-prompts).
                        if (
                            preview
                            and rollout_type == "policy"
                            and step_i % query_frequency == 0
                        ):
                            decision = preview_and_confirm(
                                policy.debug_actions,
                                obs,
                                step_i,
                                annotation=getattr(policy, "annotation", None),
                                arms=arms,
                            )
                            if decision == "quit":
                                print("Quit requested from preview.")
                                return
                            chunk_approved = decision == "execute"
                            if not chunk_approved:
                                print(
                                    f"[preview] Skipping chunk at step {step_i}; arm holds."
                                )

                        # Live (non-blocking) preview: refresh the window with each
                        # new predicted chunk and KEEP executing. q/ESC in the window
                        # quits. This is the "watch it run" path (vs --preview's gate).
                        if preview_live and rollout_type == "policy":
                            if step_i % query_frequency == 0:
                                if preview_live_update(
                                    policy.debug_actions,
                                    obs,
                                    step_i,
                                    annotation=getattr(policy, "annotation", None),
                                    arms=arms,
                                ):
                                    print("Quit requested from live preview.")
                                    return
                            else:
                                preview_pump()  # keep the window responsive between chunks

                        if chunk_approved and not no_execute:
                            for arm in arms_list:
                                arm_offset = (
                                    7 if (arm == "right" and arms == "both") else 0
                                )
                                arm_action = actions[arm_offset : arm_offset + 7]
                                if cartesian:
                                    ri.set_pose(arm_action, arm)
                                else:
                                    ri.set_joints(arm_action, arm)

    except KeyboardInterrupt:
        print("KeyboardInterrupt detected, exiting rollout.")
        return
    finally:
        # Always release the robot interface (cameras + CAN). Without this, a
        # crash leaves the non-daemon CAN control threads alive (process hangs)
        # and the RealSense pipelines open, so the cameras stay "Device busy"
        # on the next launch.
        _SHUTTING_DOWN.set()
        if preview or preview_live:
            try:
                cv2.destroyAllWindows()
            except BaseException:
                pass
        closer = getattr(ri, "close", None)
        if callable(closer):
            try:
                closer()
            except BaseException as e:
                print(f"[rollout] interface close error: {e}")


def build_arg_parser(description="Rollout robot model."):
    import argparse

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--arms",
        type=str,
        default="right",
        choices=["left", "right", "both"],
        help="Which arm(s) to control",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=DEFAULT_FREQUENCY,
        help="Control loop frequency in Hz",
    )
    parser.add_argument(
        "--query_frequency",
        type=int,
        default=QUERY_FREQUENCY,
        help="Frames which model does inference",
    )
    parser.add_argument("--policy-path", type=str, help="policy checkpoint path")
    parser.add_argument("--dataset-path", type=str, help="dataset path for replay")
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="control in cartesian space instead of joint space",
    )
    parser.add_argument(
        "--resampled-action-len",
        type=int,
        default=DEFAULT_RESAMPLE_LENGTH,
        help="Resample each predicted action chunk to this length (e.g., 100 -> 45).",
    )
    parser.add_argument(
        "--annotation-path",
        type=str,
        help="path to the annotation file",
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default=None,
        help='inline language prompt, e.g. --annotation "Fold the shirt". '
        "Used only if --annotation-path is not provided.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show the predicted action chunk overlaid on the front camera in a "
        "GUI window and wait for confirmation before executing it "
        "(ENTER/space=execute, s=skip, q=quit).",
    )
    parser.add_argument(
        "--preview-live",
        action="store_true",
        help="continuously show each predicted action chunk overlaid on the front "
        "camera WITHOUT pausing (the rollout keeps running). q/ESC in the window "
        "quits. Use this to watch the chunks evolve; --preview is the confirm gate.",
    )
    parser.add_argument(
        "--no-execute",
        action="store_true",
        help="DRY RUN: predict + visualize each chunk but NEVER command the arm "
        "from the policy (initial set_home still runs). Use with --preview-live to "
        "validate predictions/overlay/calibration safely before letting it move.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="yam",
        choices=["eva", "yam"],
        help="Robot backend (defaults to 'yam' in this script; 'eva' kept for parity).",
    )
    parser.add_argument(
        "--extrinsics-key",
        type=str,
        default=None,
        help="Camera extrinsics key (EXTRINSICS). Defaults to 'yam' for --robot yam, "
        "else 'x5Dec13_2'.",
    )
    parser.add_argument(
        "--yam-left-can",
        type=str,
        default="can_follower_l",
        help="CAN channel for the left YAM follower arm (--robot yam).",
    )
    parser.add_argument(
        "--yam-right-can",
        type=str,
        default="can_follower_r",
        help="CAN channel for the right YAM follower arm (--robot yam).",
    )
    parser.add_argument(
        "--ee-convention",
        type=str,
        default="default",
        choices=["default", "libero"],
        help="YAM grasp_site frame convention (--robot yam). MUST match the "
        "convention the training demos were collected with. 'libero' = both arms "
        "x fwd / y left / z up (LIBERO/EVA-congruent); see yam_interface.",
    )
    parser.add_argument(
        "--no-embodiment-prompt",
        action="store_true",
        help="drop the 'Embodiment: <name>' block from the conditioning prompt "
        "(Task/State blocks stay). Use with checkpoints trained with "
        "embodiment_label: false; must match the checkpoint's data config.",
    )
    parser.add_argument(
        "--flange-frame",
        type=str,
        default="native",
        choices=["native", "eva"],
        help="flange/wrist axis convention the CHECKPOINT speaks. 'native' "
        "(default) = current behavior, poses pass through unchanged. 'eva' = "
        "checkpoint trained on EVA-convention data (Z=approach): YAM-native obs "
        "flange rotations are relabeled into the EVA convention before inference "
        "and outgoing commands relabeled back (EVA X=-Z_yam, Y=-X_yam, Z=+Y_yam). "
        "Distinct from --ee-convention, which changes the HARDWARE frame and "
        "must stay 'default' when this is 'eva'.",
    )
    return parser


def run_from_args(args):
    return main(
        arms=args.arms,
        frequency=args.frequency,
        query_frequency=args.query_frequency,
        policy_path=args.policy_path,
        dataset_path=args.dataset_path,
        cartesian=args.cartesian,
        resampled_action_len=args.resampled_action_len,
        annotation_path=args.annotation_path,
        annotation_text=args.annotation,
        robot=args.robot,
        extrinsics_key=args.extrinsics_key,
        yam_channels={"left": args.yam_left_can, "right": args.yam_right_can},
        yam_ee_convention=args.ee_convention,
        preview=args.preview,
        preview_live=args.preview_live,
        no_execute=args.no_execute,
        no_embodiment_prompt=args.no_embodiment_prompt,
        flange_frame=args.flange_frame,
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_args(args)
