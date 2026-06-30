"""DEFINITIVE: run the real checkpoint on this episode's REAL obs (images+proprio)
and compare its wristframe prediction to the ground-truth eva action.

- preds ~= GT  -> model is good; live YAM failure is OOD live-obs / domain gap.
- preds != GT  -> checkpoint/normalization defect (predicts wrong even on training obs).

Reuses PolicyRollout's own pipeline (transform_list -> collate -> process_batch ->
forward_eval) so it's faithful to the live rollout inference path.
"""
import sys, io
import numpy as np
import zarr
from PIL import Image
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "/home/mecka/EgoVerse")
from egomimic.robot.YAM.yam_rollout import (
    PolicyRollout, ee_pose_to_rot_ee_frame, inv_R_t_e,
)
from egomimic.rldb.embodiment.yam import Yam

EP = "/home/mecka/EgoVerse/debug/eva_replay/0088c8d3-78ec-4738-b149-db27cf4acaee.zarr"
CKPT = "/home/mecka/EgoVerse/logs/abc_foldclothes_eva_wrist/pi05_eva_abc_foldclothes_wrist_2026-06-29_23-29-16/checkpoints/last.ckpt.patched"

z = zarr.open(EP, mode="r")
L_obs = np.asarray(z["left.obs_ee_pose"]); R_obs = np.asarray(z["right.obs_ee_pose"])
L_cmd = np.asarray(z["left.cmd_ee_pose"]); R_cmd = np.asarray(z["right.cmd_ee_pose"])
Lg = np.asarray(z["left.obs_gripper"]); Rg = np.asarray(z["right.obs_gripper"])
T = L_obs.shape[0]


def jpeg(arr_elem):
    s = arr_elem
    while isinstance(s, np.ndarray):
        s = s.item()
    return np.asarray(Image.open(io.BytesIO(s)).convert("RGB"))  # HWC RGB


def wxyz_to_raw_ypr(pose_wxyz):
    """zarr obs is post-R_t_e xyz+wxyz; recover RAW robot xyz+ypr (pre-R_t_e) so
    get_obs's ee_pose_to_rot_ee_frame re-applies R_t_e to reproduce the zarr value."""
    q = pose_wxyz[3:]
    Rmat = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()  # wxyz->xyzw
    raw = inv_R_t_e @ Rmat
    return np.concatenate([pose_wxyz[:3], R.from_matrix(raw).as_euler("ZYX")])


def build_obs(t):
    lp = wxyz_to_raw_ypr(L_obs[t]); rp = wxyz_to_raw_ypr(R_obs[t])
    ee = np.concatenate([lp, Lg[t], rp, Rg[t]])  # 14: Lxyz Lypr Lg Rxyz Rypr Rg
    rgb2bgr = lambda im: im[..., ::-1].copy()
    return {
        "ee_poses": ee,
        "front_img_1": rgb2bgr(jpeg(z["images.front_1"][t])),
        "left_wrist_img": rgb2bgr(jpeg(z["images.left_wrist"][t])),
        "right_wrist_img": rgb2bgr(jpeg(z["images.right_wrist"][t])),
    }


# GT wristframe action (forward transform of real eva cmd) for comparison
fwd = Yam.get_transform_list(mode="cartesian_wristframe_6d")
H = 16
def gt_wristframe(t):
    sl = slice(t, min(t + H, T))
    b = {"left.obs_ee_pose": L_obs[t].copy(), "right.obs_ee_pose": R_obs[t].copy(),
         "left.obs_gripper": Lg[t].copy(), "right.obs_gripper": Rg[t].copy(),
         "left.cmd_ee_pose": L_cmd[sl].copy(), "right.cmd_ee_pose": R_cmd[sl].copy(),
         "left.cmd_gripper": Lg[sl].copy(), "right.cmd_gripper": Rg[sl].copy()}
    for tr in fwd:
        b = tr.transform(b)
    return np.asarray(b["actions_cartesian"])  # (H,20)


print("Loading policy (21GB)...")
ro = PolicyRollout(arm="both", policy_path=CKPT, query_frequency=1,
                   cartesian=True, extrinsics_key="yam", robot="yam",
                   annotation_text="Fold the tank top")
import torch
ro.policy.eval()

# sample frames within the FOLD phase (annotation start_idx 475)
STEPS = [120, 300, 500, 600, 700]
print(f"\n{'step':>5} {'arm':>5} {'pos_err_mm':>10} {'ori_err_deg':>11}  (model pred vs GT, first action in chunk)")
for t in STEPS:
    obs = build_obs(t)
    tlb = ro.process_obs_for_transform_list(obs)
    for tr in ro.transform_list:
        tlb = tr.transform(tlb)
    tlb = ro.collate_fn([tlb])
    batch = {ro.embodiment_name: tlb}
    pb = ro.policy.model.process_batch_for_training(batch)
    with torch.no_grad():
        preds = ro.policy.model.forward_eval(pb)[f"{ro.embodiment_name}_actions_cartesian"]
    pred = preds.detach().cpu().numpy().squeeze()  # (H,20) wristframe 6D
    gt = gt_wristframe(t)
    n = min(pred.shape[0], gt.shape[0])
    for arm, off in [("left", 0), ("right", 10)]:
        # per-arm: xyz(3) c1(3) c2(3) grip(1)
        pp, gg = pred[:n, off:off+3], gt[:n, off:off+3]
        pos_err = np.linalg.norm(pp[0] - gg[0]) * 1000
        # orientation: reconstruct R from 6D cols, geodesic angle
        def rot6d(v):
            a, b = v[:3], v[3:6]
            a = a/np.linalg.norm(a); b = b - a*(a@b); b = b/np.linalg.norm(b)
            return np.stack([a, b, np.cross(a, b)], axis=1)
        Rp = rot6d(pred[0, off:off+6]); Rg_ = rot6d(gt[0, off:off+6])
        ang = np.degrees(np.linalg.norm(R.from_matrix(Rp.T@Rg_).as_rotvec()))
        print(f"{t:>5} {arm:>5} {pos_err:>10.1f} {ang:>11.1f}")
