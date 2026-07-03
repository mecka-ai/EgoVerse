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
_THIS = os.path.dirname(os.path.abspath(__file__))            # .../egomimic/robot/YAM
_ROBOT_DIR = os.path.dirname(_THIS)                           # .../egomimic/robot
_EGOMIMIC_DIR = os.path.dirname(_ROBOT_DIR)                   # .../egomimic
_REPO_ROOT = os.path.dirname(_EGOMIMIC_DIR)                   # .../<repo root>
for _p in (
    _ROBOT_DIR,                                               # robot_utils
    _THIS,                                                    # yam_interface, yam_cameras
    os.path.join(_ROBOT_DIR, "eva", "eva_ws", "src", "eva"),  # robot_interface (ARX base)
    os.path.join(_REPO_ROOT, "external", "i2rt"),             # i2rt SDK
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

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import default_collate
from robot_utils import RateLoop
from scipy.spatial.transform import Rotation as R

from egomimic.models.denoising_policy import DenoisingPolicy
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.pl_utils.pl_data_utils import build_tokenized_collate
from egomimic.rldb.embodiment.embodiment import get_embodiment, get_embodiment_id
from egomimic.rldb.embodiment.eva import (
    Eva,
    _build_eva_bimanual_revert_eef_frame_transform_list,
)
from egomimic.rldb.embodiment.yam import Yam
from egomimic.robot.eva.eva_kinematics import EvaMinkKinematicsSolver
from egomimic.utils.egomimicUtils import (
    CameraTransforms,
    cam_frame_to_base_frame,
    draw_actions,
    interpolate_arr,
    interpolate_arr_euler,
)
from egomimic.utils.pose_utils import xyzw_to_wxyz

# (sys.path for robot_utils / robot_interface / yam_interface / i2rt is set up at
# the top of this module so it applies before the imports above.)

import select
import termios
import tty


def visualize_actions(ims, actions, extrinsics, intrinsics, arm="both"):
    if actions.shape[-1] == 7 or actions.shape[-1] == 14:
        ac_type = "joints"
    elif actions.shape[-1] == 3 or actions.shape[-1] == 6:
        ac_type = "xyz"
    else:
        raise ValueError(f"Unknown action type with shape {actions.shape}")

    ims = draw_actions(
        ims, ac_type, "Purples", actions, extrinsics, intrinsics, arm=arm
    )

    return ims


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


# --- one-shot orientation diagnostic (localize a wrist flip) -----------------
def _geodesic_deg(Ra, Rb):
    """Angle (deg) between two rotation matrices."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _bimanual_rot_mats(vec, use_6d):
    """(left_R, right_R) 3x3 from a bimanual pose row.

    use_6d  -> per-arm [xyz(3) c1(3) c2(3) (grip)], rot cols at [0:9]/[10:19].
    not 6d  -> per-arm ypr [xyz(3) ypr(3) (grip)], ypr at [3:6]/[10:13] (ZYX).
    """
    vec = np.asarray(vec, dtype=float)
    if vec.ndim > 1:
        vec = vec[0]  # first chunk row
    if use_6d:
        from egomimic.utils.pose_utils import _xyz6d_to_matrix
        L = _xyz6d_to_matrix(vec[0:9][None])[0, :3, :3]
        Rr = _xyz6d_to_matrix(vec[10:19][None])[0, :3, :3]
    else:
        L = R.from_euler("ZYX", vec[3:6]).as_matrix()
        Rr = R.from_euler("ZYX", vec[10:13]).as_matrix()
    return L, Rr


def _ypr_deg(Rm):
    return np.round(R.from_matrix(Rm).as_euler("ZYX", degrees=True), 1)


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


def viz_rot_ee_pose(image, eepose, action_image_path, rot_image_path):
    """
    Save both cartesian-action and orientation-axis visualizations for an EVA
    action chunk using the same conventions as the debug path.
    """
    arr = np.asarray(eepose, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, ...]
    if arr.ndim != 2 or arr.shape[1] not in (12, 14):
        raise ValueError(f"Expected eepose shape (T, 12|14), got {arr.shape}")

    os.makedirs(os.path.dirname(action_image_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(rot_image_path) or ".", exist_ok=True)

    img = np.asarray(image)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(
            f"Expected image shape (H, W, 3) or (3, H, W), got {img.shape}"
        )
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)

    if arr.shape[1] == 14:
        left_xyz = arr[:, :3]
        right_xyz = arr[:, 7:10]
    else:
        left_xyz = arr[:, :3]
        right_xyz = arr[:, 6:9]
    action_xyz = np.hstack([left_xyz, right_xyz]).astype(np.float32, copy=False)

    camera_transforms = CameraTransforms(
        intrinsics_key="base", extrinsics_key="x5Dec13_2"
    )
    im_action = visualize_actions(
        img.copy(),
        action_xyz,
        camera_transforms.extrinsics,
        camera_transforms.intrinsics,
        arm="both",
    )
    cv2.imwrite(action_image_path, im_action)

    eva_viz_batch = {
        "observations.images.front_img_1": torch.from_numpy(img[None, ...]),
        "actions_cartesian": torch.from_numpy(arr[None, ...]),
    }
    im_rot = Eva.viz_transformed_batch(eva_viz_batch, mode="palm_axes")
    cv2.imwrite(rot_image_path, im_rot)
    return im_action, im_rot


GRIPPER_WIDTH = 0.09
# Control parameters
DEFAULT_FREQUENCY = 30  # Hz
QUERY_FREQUENCY = 30
DEFAULT_RESAMPLE_LENGTH = 45

RIGHT_CAM_SERIAL = ""
LEFT_CAM_SERIAL = ""

EMBODIMENT_MAP = {
    "both": 8,
    "left": 7,
    "right": 6,
}

TEMP_DIR = "/home/robot/temp_dir"


def _build_robot_interface(
    arms_list, robot="eva", offline_debug=False, offline_episode_path=None, yam_channels=None
):
    if robot == "yam":
        if offline_debug:
            raise ValueError("--offline-debug is not supported for --robot yam")
        # YAMInterface is a sibling module (this file lives in egomimic/robot/YAM/);
        # YAM/ is already on sys.path (see the path setup at the top of this file).
        from yam_interface import YAMInterface

        # YAMInterface always opens its cameras, so get_obs() includes
        # front_img_1 / left_wrist_img / right_wrist_img — the keys the obs
        # pipeline consumes (and it raises if no cameras are available).
        return YAMInterface(arms=arms_list, channels=yam_channels)

    if offline_debug:
        from robot_interface import OfflineARXInterface

        return OfflineARXInterface(arms=arms_list, dataset_path=offline_episode_path)

    from robot_interface import ARXInterface

    return ARXInterface(arms=arms_list)


def _get_model_xml_path():
    candidates = [
        "/home/robot/robot_ws/egomimic/resources/model_x5.xml",
        os.path.join(_EGOMIMIC_DIR, "resources", "model_x5.xml"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[-1]


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
        debug=False,
        annotation_path=None,
        annotation_text=None,
        robot="eva",
    ):
        super().__init__()
        self.arm = arm
        self.robot = robot
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
        emb_extrinsics_key = getattr(self.embodiment_cls, "EXTRINSICS_KEY", extrinsics_key)
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
        self.debug_actions = None
        self._printed_diag = False  # one-shot input-alignment dump (see _diagnose_inputs)
        self.resampled_action_len = resampled_action_len
        self.debug = debug

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
                print(f"[rollout] WARNING: annotation file not found: {annotation_path}  (continuing without annotation)")
            else:
                with open(annotation_path, "r") as f:
                    self.annotation = f.read().strip()
                self.collate_fn = build_tokenized_collate(
                    max_length=128,
                    model_name="google/paligemma-3b-mix-224",
                    sampling_mode="first",
                    annotation_key="annotations",
                    default_prompt=self.annotation,
                )
        # Inline prompt (e.g. --annotation "Fold the shirt") takes effect only if
        # no annotation file was successfully loaded above.
        if self.annotation is None and annotation_text:
            self.annotation = annotation_text.strip()
            self.collate_fn = build_tokenized_collate(
                max_length=128,
                model_name="google/paligemma-3b-mix-224",
                sampling_mode="first",
                annotation_key="annotations",
                default_prompt=self.annotation,
            )
            print(f"[rollout] Using inline annotation prompt: '{self.annotation}'")

    LOCAL_WEIGHT_PATH = os.path.join(
        _EGOMIMIC_DIR, "algo", "pi_checkpoints", "pi05_base_pytorch"
    )

    @classmethod
    def _patch_checkpoint_paths(cls, ckpt_path):
        """Rewrite pytorch_weight_path in the checkpoint's saved config
        to point to the local base model weights."""
        import torch as _torch
        from omegaconf import OmegaConf, DictConfig
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
        print(f"[rollout] Patching pytorch_weight_path: {old_path} -> {cls.LOCAL_WEIGHT_PATH}")
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
            print("[rollout] Disabled torch.compile on sample_actions for rollout inference")

        # Verify model is on GPU
        try:
            p = next(pi0.parameters())
            print(f"[rollout] Model device: {p.device}, dtype: {p.dtype}")
            if not p.is_cuda:
                print("[rollout] WARNING: model is NOT on GPU — inference will be very slow!")
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
            # SLERP rotation interp (default) is correct through gimbal lock
            # (pitch~=90); linear-Euler interp sweeps the wrist there -> flip.
            # Set YAM_EULER_INTERP=1 to fall back to the old linear-Euler path
            # (for A/B-ing the wrist-flip fix).
            if os.environ.get("YAM_EULER_INTERP", "0") == "1":
                if self.arm == "both":
                    left = interpolate_arr_euler(chunk[:, :7][None, ...], target_len)[0]
                    right = interpolate_arr_euler(chunk[:, 7:14][None, ...], target_len)[0]
                    out = np.hstack([left, right])
                else:
                    out = interpolate_arr_euler(chunk[None, ...], target_len)[0]
            else:
                out = _slerp_resample_cartesian(chunk, target_len)
        else:
            out = interpolate_arr(chunk[None, ...], target_len)[0]

        return out.astype(np.float32, copy=False)

    def _diagnose_inputs(self, obs, fed_batch, processed_batch, preds):
        """One-shot dump of what actually reaches pi0.5, to localize 'noise'
        predictions: key-mapping / image / state / prompt misalignment, and the
        scale of the (unnormalized) action output."""
        m = self.policy.model
        eid = self.embodiment_id
        pb = processed_batch.get(eid, {})
        print("\n========== [diag] pi0.5 input alignment ==========")
        print(f"[diag] embodiment   : {self.embodiment_name} (id={eid})")
        print(f"[diag] camera_keys  : {m.camera_keys.get(eid)}")
        print(f"[diag] proprio_keys : {m.proprio_keys.get(eid)}")
        print(f"[diag] lang_keys    : {m.lang_keys.get(eid)}")
        print(f"[diag] ac_key       : {m.ac_keys.get(eid)}")
        print(f"[diag] pi_cam_keys  : {getattr(m, 'pi_cam_keys', None)}")
        print(f"[diag] fed keys     : {sorted(fed_batch.keys())}")
        print(f"[diag] survived keys: {sorted(pb.keys())}")

        # Cameras: did they survive the keymap, and is the content real (not black)?
        for k in getattr(m, "pi_cam_keys", []) or []:
            if k in pb and hasattr(pb[k], "shape"):
                t = pb[k].float()
                print(f"[diag] cam '{k}': shape={tuple(t.shape)} "
                      f"min={t.min():.3f} max={t.max():.3f} mean={t.mean():.3f}")
            else:
                print(f"[diag] cam '{k}': MISSING from processed_batch (will be duplicated+masked)")
        f = obs.get("front_img_1")
        if f is not None:
            f = np.asarray(f)
            print(f"[diag] raw front_img_1: shape={f.shape} dtype={f.dtype} "
                  f"min={f.min()} max={f.max()} mean={f.mean():.2f}  (~0 everywhere => black cam)")

        # Proprio/state: present and right width?
        for k in (m.proprio_keys.get(eid) or []):
            present = k in pb and hasattr(pb[k], "shape")
            print(f"[diag] proprio '{k}': present={present} "
                  f"shape={tuple(pb[k].shape) if present else None}")

        # Language conditioning
        tok = pb.get("tokenized_prompt")
        has_tok = tok is not None and hasattr(tok, "numel") and tok.numel() > 0
        print(f"[diag] tokenized_prompt nonempty: {has_tok}  annotation='{self.annotation}'")

        # Output scale: pure noise tends to be ~constant std with no structure.
        p = preds.detach().float().cpu()
        print(f"[diag] preds(unnorm) shape={tuple(p.shape)} min={p.min():.4f} "
              f"max={p.max():.4f} mean={p.mean():.4f} std={p.std():.4f}")
        print("==================================================\n")

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
            _wf_preds = self.actions.copy()  # wrist-frame prediction (pre-revert)
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

            # --- per-inference debug: policy actions in WRIST frame (raw pred,
            # pre-revert) and CAMERA frame (post-revert). The full action's wrist
            # rotation is decoded to ZYX ypr degrees for readability. Prints on
            # every policy step (once per inference / chunk).
            if self.cartesian:
                from egomimic.utils.pose_utils import _xyz6d_to_matrix

                def _fmt(a):
                    return np.array2string(
                        np.asarray(a, dtype=float), precision=4, suppress_small=True
                    )

                def _wf_ypr(block):
                    # per-arm wrist-frame slice -> ZYX ypr (deg). 6D block is
                    # [xyz c1 c2 (g)] (rotation from the two 6D columns); ypr block
                    # is [xyz ypr (g)] (ypr already, radians).
                    block = np.asarray(block, dtype=float)
                    if self.use_6d:
                        Rm = _xyz6d_to_matrix(block[:9][None])[0, :3, :3]
                        return _ypr_deg(Rm)
                    return np.round(np.rad2deg(block[3:6]), 1)

                _wf = np.atleast_2d(_wf_preds)           # wrist-frame chunk (pre-revert)
                _cf = np.atleast_2d(self.debug_actions)  # camera-frame chunk (post-revert)
                _both = self.arm == "both"
                _wfw = 10 if self.use_6d else 7          # per-arm wrist-frame width

                # WRISTFRAME (always available). Per-arm blocks: L first, R second.
                _wfL = _wf[0, 0:_wfw]
                print(f"[dbg step {i}] WRISTFRAME pred (pre-revert) shape={_wf.shape}")
                line = f"    L: pos={_fmt(_wfL[:3])} ypr(deg)={_fmt(_wf_ypr(_wfL))}"
                if _both:
                    _wfR = _wf[0, _wfw:2 * _wfw]
                    line += f"    R: pos={_fmt(_wfR[:3])} ypr(deg)={_fmt(_wf_ypr(_wfR))}"
                print(line)
                print(f"    row0 action={_fmt(_wf[0])}")

                # CAMFRAME only exists after the both-arm revert (14-dim ypr:
                # per-arm [xyz ypr grip]); single-arm skips the revert.
                if _both:
                    _cfL, _cfR = _cf[0, 0:7], _cf[0, 7:14]
                    print(f"[dbg step {i}] CAMFRAME (post-revert) shape={_cf.shape}")
                    print(
                        f"    L: pos={_fmt(_cfL[:3])} "
                        f"ypr(deg)={_fmt(np.round(np.rad2deg(_cfL[3:6]), 1))}"
                        f"    R: pos={_fmt(_cfR[:3])} "
                        f"ypr(deg)={_fmt(np.round(np.rad2deg(_cfR[3:6]), 1))}"
                    )
                    print(f"    row0 action={_fmt(_cf[0])}")

            if not self._printed_diag:
                self._printed_diag = True
                self._diagnose_inputs(obs, transform_list_batch, processed_batch, preds)
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
                    # Preserve gripper if present (7th value)
                    gripper = self.actions[:, 6:7]
                    if self.actions.shape[1] == 7:
                        self.actions = np.hstack([transformed_6dof, gripper])
                    else:
                        self.actions = transformed_6dof

            # --- one-shot orientation diagnostic: localize a wrist flip -------
            # Prints orientation at each stage for the FIRST chunk row, with the
            # two key angles: |wristframe pred from identity| (~180 => the model
            # itself predicts a flip => obs/proprio wrong or OOD) and |reverted vs
            # obs| (~180 => the revert/frame composition flips it; ~0 => the flip
            # is downstream in cam->base / rot_ee_frame / IK).
            if (
                self.cartesian
                and self.arm == "both"
                and not getattr(self, "_printed_orient_diag", False)
            ):
                self._printed_orient_diag = True
                try:
                    oL, oR = _bimanual_rot_mats(obs_state_camframe, self.use_6d)
                    wL, wR = _bimanual_rot_mats(_wf_preds, self.use_6d)
                    rL, rR = _bimanual_rot_mats(self.debug_actions, False)
                    bL, bR = _bimanual_rot_mats(self.actions, False)
                    # Robot's ACTUAL current EE orientation (base frame, ZYX).
                    # eepose = [L xyz ypr g | R xyz ypr g] -> ypr at [3:6]/[10:13].
                    raw = np.asarray(obs["ee_poses"], dtype=float)
                    cL = R.from_euler("ZYX", raw[3:6]).as_matrix()
                    cR = R.from_euler("ZYX", raw[10:13]).as_matrix()
                    I3 = np.eye(3)
                    print(f"\n===== [orient-diag] step {i} (first chunk row) =====")
                    for nm, o, w, r, b, c in (
                        ("LEFT", oL, wL, rL, bL, cL),
                        ("RIGHT", oR, wR, rR, bR, cR),
                    ):
                        print(f"  {nm}:")
                        print(f"    obs camframe ypr(deg)      = {_ypr_deg(o)}")
                        print(f"    wristframe PRED ypr(deg)   = {_ypr_deg(w)}"
                              f"   |angle from identity| = {_geodesic_deg(I3, w):6.1f} deg"
                              f"   (small=stay-put, ~180=model flips)")
                        print(f"    reverted camframe ypr(deg) = {_ypr_deg(r)}"
                              f"   |angle vs obs|        = {_geodesic_deg(o, r):6.1f} deg"
                              f"   (~0=revert ok, ~180=revert/frame flip)")
                        print(f"    base ee-pose ypr(deg)      = {_ypr_deg(b)}")
                        print(f"    RAW current ee ypr(deg)    = {_ypr_deg(c)}"
                              f"   |base vs raw current| = {_geodesic_deg(b, c):6.1f} deg"
                              f"   (~0=stay-put closes, large=decode broken)")
                    # Full commanded-chunk trajectory: does the base ee-pose
                    # SWING/flip across the chunk rows (model traj / 6D->ypr
                    # collapse), or stay put (=> flip is in IK / execution)?
                    acts = np.asarray(self.actions, dtype=float)
                    if acts.ndim == 2 and acts.shape[1] >= 13:
                        for nm, off in (("LEFT", 0), ("RIGHT", 7)):
                            rr = R.from_euler("ZYX", acts[:, off + 3 : off + 6]).as_matrix()
                            devs = np.array(
                                [_geodesic_deg(rr[0], rr[k]) for k in range(len(rr))]
                            )
                            k = int(np.argmax(devs))
                            print(f"  {nm} chunk traj: max |row{k} vs row0| = {devs[k]:6.1f} deg"
                                  f"  over {len(rr)} rows"
                                  f"  (large => commanded trajectory itself swings/flips;"
                                  f" small => flip is in IK/execution)")
                    print("======================================================\n")
                except Exception as _e:
                    print(f"[orient-diag] failed: {_e!r}")

            if self.resampled_action_len is not None:
                self.actions = self._downsample_chunk(
                    self.actions, self.resampled_action_len
                )
            # print(f"actions: {self.actions[6:7]}, debug_actions: {self.debug_actions[6:7]}")

            print(f"Inference time: {(time.time() - start_infer_t)}s")

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

        if self.arm in ["right", "both"]:
            right = torch.from_numpy(
                obs["right_wrist_img"][None, ...]
            )  # [1, H, W, 3] BGR
            right = right[..., [2, 1, 0]]  # BGR -> RGB
            right = (
                right.permute(0, 3, 1, 2).to(dtype=torch.float32) / 255.0
            )
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

        data["embodiment"] = self.embodiment_name
        data["metadata.robot_name"] = self.embodiment_name
        
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
            self.collate_fn = build_tokenized_collate(
                max_length=128,
                model_name="google/paligemma-3b-mix-224",
                sampling_mode="first",
                annotation_key="annotations",
                default_prompt=self.annotation,
            )
        print(f"[rollout] Loaded new annotation from {annotation_path}: '{self.annotation}'")
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
    the debug/preview overlays project with a K that does not match the image.
    Applied once at obs ingestion so the model input, ``debug_policy`` /
    ``preview_and_confirm`` overlays, and ``INTRINSICS["yam"]`` stay mutually
    consistent. No-op for frames already at 640x480 (e.g. the wrist cams or a
    future collection stack that resizes on-camera).
    """
    f = obs.get("front_img_1")
    if f is not None and f.shape[:2] != (_FRONT_TRAINING_WH[1], _FRONT_TRAINING_WH[0]):
        obs["front_img_1"] = cv2.resize(
            f, _FRONT_TRAINING_WH, interpolation=cv2.INTER_AREA
        )
    return obs


def debug_policy(
    actions, front_img, step_i
):
    os.makedirs("debug", exist_ok=True)

    if isinstance(front_img, torch.Tensor):
        if front_img.dim() == 4:
            front_img = front_img[0].permute(1, 2, 0).cpu().numpy()
        elif front_img.dim() == 3:
            if front_img.shape[0] == 3:
                front_img = front_img.permute(1, 2, 0).cpu().numpy()
            else:
                front_img = front_img.cpu().numpy()
    elif front_img.ndim == 3 and front_img.shape[0] == 3:
        front_img = front_img.transpose(1, 2, 0)
    front_img = front_img.astype(np.uint8)

    actions = actions.squeeze()
    eva_viz_batch = {
        "observations.images.front_img_1": torch.from_numpy(front_img[None, ...]),
        "actions_cartesian": torch.from_numpy(
            actions.astype(np.float32, copy=False)[None, ...]
        ),
    }
    im_viz = Yam.viz_transformed_batch(eva_viz_batch, mode="traj+rotation")

    cv2.imwrite(f"debug/debug_{step_i}.png", im_viz)
    breakpoint()


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
    BGR uint8 image via the same overlay pipeline as the offline eval / --debug.
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


def preview_and_confirm(actions, front_img, step_i, annotation=None):
    """Show the predicted-action overlay and BLOCK until the operator decides.

    The arm is NOT commanded until this returns. Keys (focus the preview window):
        ENTER / SPACE / e -> execute this chunk
        s                 -> skip this chunk (arm holds until the next prediction)
        q / ESC           -> quit the rollout
    Returns 'execute', 'skip', or 'quit'.
    """
    overlay = render_pred_overlay(actions, front_img)
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
        if key in (13, 32, ord("e")):   # Enter / Space / e
            return "execute"
        if key == ord("s"):
            return "skip"
        if key in (ord("q"), 27):       # q / ESC
            return "quit"


def preview_live_update(actions, front_img, step_i, annotation=None):
    """Non-blocking live overlay: refresh the window with the latest predicted
    chunk and return immediately (the rollout keeps executing). Returns True if
    the operator pressed q/ESC in the window (quit), else False."""
    overlay = render_pred_overlay(actions, front_img)
    banner = f"step {step_i} (LIVE)" + (f"  |  {annotation}" if annotation else "")
    cv2.putText(overlay, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
    cv2.putText(overlay, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
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
    debug=False,
    resampled_action_len=None,
    offline_debug=False,
    offline_episode_path=None,
    annotation_path=None,
    annotation_text=None,
    robot="yam",
    extrinsics_key=None,
    yam_channels=None,
    preview=False,
    preview_live=False,
    no_execute=False,
):
    threading.excepthook = _quiet_thread_excepthook

    # Default camera extrinsics per robot family (override with --extrinsics-key).
    if extrinsics_key is None:
        extrinsics_key = "yam" if robot == "yam" else "x5Dec13_2"

    if arms == "both":
        arms_list = ["right", "left"]
    elif arms == "right":
        arms_list = ["right"]
    else:
        arms_list = ["left"]

    if offline_episode_path is not None and not offline_debug:
        raise ValueError("--offline-episode-path requires --offline-debug.")
    if policy_path is not None and offline_debug and offline_episode_path is None:
        raise ValueError(
            "--policy-path requires --offline-episode-path in --offline-debug mode."
        )

    ri = _build_robot_interface(
        arms_list=arms_list,
        robot=robot,
        offline_debug=offline_debug,
        offline_episode_path=offline_episode_path,
        yam_channels=yam_channels,
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
            debug=debug,
            annotation_path=annotation_path,
            annotation_text=annotation_text,
            robot=robot,
        )
    elif dataset_path is not None:
        rollout_type = "replay"
        policy = ReplayRollout(dataset_path=dataset_path, cartesian=cartesian)
    else:
        raise ValueError(
            "Must provide either --policy-path or --dataset-path (and optionally --repo-id)."
        )

    print(f"Cartesian value {cartesian}")
    if no_execute:
        print(
            "[rollout] --no-execute: DRY RUN. Predicting + visualizing only; the arm "
            "will NOT be commanded by the policy (the initial set_home still runs)."
        )

    # EVA-only helpers (unused on the YAM path, and EvaMinkKinematicsSolver
    # would try to load the EVA x5 MJCF which isn't relevant for YAM).
    if robot == "eva":
        camera_transforms = CameraTransforms(
            intrinsics_key="base", extrinsics_key=extrinsics_key
        )
        kinematics_solver = EvaMinkKinematicsSolver(model_path=_get_model_xml_path())

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
                print(f"Unknown command: '{cmd}'. Use c / a <path> / r / q.")

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
                        elif rollout_type == "replay_lerobot":
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

                        if debug and rollout_type == "policy" and step_i % query_frequency == 0:
                            debug_actions = policy.debug_actions
                            front_img = obs["front_img_1"]
                            debug_policy(
                                debug_actions,
                                front_img,
                                step_i,
                            )

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
                                obs["front_img_1"],
                                step_i,
                                annotation=getattr(policy, "annotation", None),
                            )
                            if decision == "quit":
                                print("Quit requested from preview.")
                                return
                            chunk_approved = decision == "execute"
                            if not chunk_approved:
                                print(f"[preview] Skipping chunk at step {step_i}; arm holds.")

                        # Live (non-blocking) preview: refresh the window with each
                        # new predicted chunk and KEEP executing. q/ESC in the window
                        # quits. This is the "watch it run" path (vs --preview's gate).
                        if preview_live and rollout_type == "policy":
                            if step_i % query_frequency == 0:
                                if preview_live_update(
                                    policy.debug_actions,
                                    obs["front_img_1"],
                                    step_i,
                                    annotation=getattr(policy, "annotation", None),
                                ):
                                    print("Quit requested from live preview.")
                                    return
                            else:
                                preview_pump()  # keep the window responsive between chunks

                        if chunk_approved and not no_execute:
                            for arm in arms_list:
                                arm_offset = 7 if (arm == "right" and arms == "both") else 0
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
        "--offline-debug",
        action="store_true",
        help="use the offline dummy robot interface for rollout debugging",
    )
    parser.add_argument(
        "--offline-episode-path",
        type=str,
        help="local EVA Zarr episode path used as observation source in offline debug mode",
    )
    parser.add_argument(
        "--cartesian",
        action="store_true",
        help="control in cartesian space instead of joint space",
    )
    parser.add_argument(
        "--resampled-action-len",
        type=int,
        default=DEFAULT_RESAMPLE_LENGTH,
        help="Resample each predicted action chunk to this length (e.g., 100 -> 45). Euler if --cartesian.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="enable debug visualization of actions on images",
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
        help="inline language prompt, e.g. --annotation \"Fold the shirt\". "
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
    return parser


def run_from_args(args):
    print(f"Resampling actions to {args.resampled_action_len}")
    return main(
        arms=args.arms,
        frequency=args.frequency,
        query_frequency=args.query_frequency,
        policy_path=args.policy_path,
        dataset_path=args.dataset_path,
        cartesian=args.cartesian,
        debug=args.debug,
        resampled_action_len=args.resampled_action_len,
        offline_debug=args.offline_debug,
        offline_episode_path=args.offline_episode_path,
        annotation_path=args.annotation_path,
        annotation_text=args.annotation,
        robot=args.robot,
        extrinsics_key=args.extrinsics_key,
        yam_channels={"left": args.yam_left_can, "right": args.yam_right_can},
        preview=args.preview,
        preview_live=args.preview_live,
        no_execute=args.no_execute,
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    run_from_args(args)
