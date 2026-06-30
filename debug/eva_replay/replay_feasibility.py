"""Decisive test: do REAL eva fold-tank-top actions decode to YAM-feasible IK targets?

Feeds the episode's real obs/cmd ee_poses through the rollout's EXACT pipeline
(forward transform_list -> wristframe 6D -> revert -> cam_frame_to_base ->
inv_R_t_e) and runs the YAM IK on each resulting target, exactly like
yam_interface.solve_ik. If real (demonstrated) actions are YAM-feasible, the
rollout's live infeasibility is the MODEL's predictions (OOD/checkpoint), not the
decode pipeline / cross-embodiment workspace. If real actions are ALSO infeasible,
the eva workspace doesn't fit the YAM and it's a fundamental mismatch.
"""
import sys
import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, "/home/mecka/EgoVerse")

from egomimic.rldb.embodiment.yam import Yam
from egomimic.robot.YAM.yam_rollout import (
    cam_frame_to_base_frame,
    rot_ee_frame_to_ee_pose_batch,
)
from egomimic.rldb.embodiment.eva import (
    _build_eva_bimanual_revert_eef_frame_transform_list,
)
from egomimic.utils.egomimicUtils import EXTRINSICS
from i2rt.robots.kinematics import Kinematics
from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

EP = "/home/mecka/EgoVerse/debug/eva_replay/0088c8d3-78ec-4738-b149-db27cf4acaee.zarr"
GRASP = "grasp_site"

# YAM kinematics (NO_GRIPPER, grasp_site at flange) — same as yam_interface.
kpath = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER)
kin = {"left": Kinematics(kpath, GRASP), "right": Kinematics(kpath, GRASP)}

z = zarr.open(EP, mode="r")
L_obs = np.asarray(z["left.obs_ee_pose"])   # (T,7) xyz+wxyz (post R_t_e)
R_obs = np.asarray(z["right.obs_ee_pose"])
L_cmd = np.asarray(z["left.cmd_ee_pose"])
R_cmd = np.asarray(z["right.cmd_ee_pose"])
Lg = np.asarray(z["left.cmd_gripper"]); Rg = np.asarray(z["right.cmd_gripper"])
T = L_obs.shape[0]

fwd = Yam.get_transform_list(mode="cartesian_wristframe_6d")
revert = _build_eva_bimanual_revert_eef_frame_transform_list(rot_repr="6d")

H = 16  # action-chunk horizon to encode per step


def build_batch(t):
    """Mirror process_obs_for_transform_list's key contract using REAL data."""
    sl = slice(t, min(t + H, T))
    return {
        "left.obs_ee_pose": L_obs[t].copy(), "right.obs_ee_pose": R_obs[t].copy(),
        "left.obs_gripper": Lg[t].copy(), "right.obs_gripper": Rg[t].copy(),
        "left.cmd_ee_pose": L_cmd[sl].copy(), "right.cmd_ee_pose": R_cmd[sl].copy(),
        "left.cmd_gripper": Lg[sl].copy(), "right.cmd_gripper": Rg[sl].copy(),
    }


def ik_feasibility(target_xyz_ypr, arm, seed):
    Tm = np.eye(4)
    Tm[:3, :3] = R.from_euler("ZYX", target_xyz_ypr[3:6]).as_matrix()
    Tm[:3, 3] = target_xyz_ypr[:3]
    ok, q = kin[arm].ik(Tm, GRASP, init_q=seed, pos_threshold=2e-3,
                        ori_threshold=np.deg2rad(2), damping=1e-2, max_iters=400)
    Tf = kin[arm].fk(q)
    pos = np.linalg.norm(Tf[:3, 3] - Tm[:3, 3]) * 1000
    dR = Tf[:3, :3].T @ Tm[:3, :3]
    ang = np.degrees(np.linalg.norm(R.from_matrix(dR).as_rotvec()))
    # position-only achievable check (orientation-free)
    ok2, q2 = kin[arm].ik(Tm, GRASP, init_q=seed, pos_threshold=2e-3,
                          ori_threshold=np.deg2rad(180), damping=1e-2, max_iters=400)
    Tf2 = kin[arm].fk(q2)
    pos_only = np.linalg.norm(Tf2[:3, 3] - Tm[:3, 3]) * 1000
    reach = np.linalg.norm(Tm[:3, 3]) * 1000
    return ok, pos, ang, pos_only, reach, q


from egomimic.robot.YAM.yam_rollout import ee_pose_to_rot_ee_frame
from egomimic.utils.pose_utils import xyzw_to_wxyz


def yam_obs_from_q(q, arm):
    """Build the rollout obs_ee_pose (xyz raw + wxyz post-R_t_e) from a YAM joint
    config, exactly as get_obs does: FK -> base ee_pose -> ee_pose_to_rot_ee_frame."""
    Tf = kin[arm].fk(q)
    xyz = Tf[:3, 3]
    ypr = R.from_matrix(Tf[:3, :3]).as_euler("ZYX")
    rot = ee_pose_to_rot_ee_frame(np.concatenate([xyz, ypr]))  # R_t_e on rotation
    wxyz = xyzw_to_wxyz(R.from_euler("ZYX", rot[3:6]).as_quat())
    return np.concatenate([xyz, wxyz])


def decode_actions_at_anchor(yam_q_l, yam_q_r, eva_actions20):
    """Ground eva wristframe deltas at the YAM pose (revert anchor = YAM obs),
    exactly like the live rollout, then cam->base->inv_R_t_e -> base targets."""
    ob = build_batch(0)  # any obs; we only need the anchor shape, then overwrite
    ob = {k: v for k, v in build_batch(0).items()}
    ob["left.obs_ee_pose"] = yam_obs_from_q(yam_q_l, "left")
    ob["right.obs_ee_pose"] = yam_obs_from_q(yam_q_r, "right")
    ob["left.cmd_ee_pose"] = np.tile(ob["left.obs_ee_pose"], (H, 1))
    ob["right.cmd_ee_pose"] = np.tile(ob["right.obs_ee_pose"], (H, 1))
    for tr in fwd:
        ob = tr.transform(ob)
    yam_obs_state = np.asarray(ob["observations.state.ee_pose"])
    rb = {"observations.state.ee_pose": yam_obs_state, "actions_cartesian": eva_actions20.copy()}
    for tr in revert:
        rb = tr.transform(rb)
    cam = np.asarray(rb["actions_cartesian"])
    tl = rot_ee_frame_to_ee_pose_batch(cam_frame_to_base_frame(cam[:, :6].copy(), EXTRINSICS["yam"]["left"]))
    tr_ = rot_ee_frame_to_ee_pose_batch(cam_frame_to_base_frame(cam[:, 7:13].copy(), EXTRINSICS["yam"]["right"]))
    return tl, tr_


def eva_action_delta(t):
    b = build_batch(t)
    for tr in fwd:
        b = tr.transform(b)
    return np.asarray(b["actions_cartesian"])  # (H,20) eva wristframe delta


READY = {"left": np.deg2rad([0, -30, 60, 0, 40, 0]), "right": np.deg2rad([0, -30, 60, 0, 40, 0])}
HOME = {"left": np.zeros(6), "right": np.zeros(6)}
STEPS = list(range(0, T - H, 50))

for anchor_name, anchor in [("HOME(joints=0)", HOME), ("READY(elbow bent)", READY)]:
    stats = {"left": [], "right": []}
    for t in STEPS:
        eva_act = eva_action_delta(t)
        tl, tr_ = decode_actions_at_anchor(anchor["left"], anchor["right"], eva_act)
        for arm, tgt in [("left", tl[0]), ("right", tr_[0])]:
            ok, pos, ang, pos_only, reach, q = ik_feasibility(tgt, arm, anchor[arm])
            stats[arm].append((ok, pos, ang, pos_only, reach))
    print(f"\n=== ANCHOR = {anchor_name}: eva fold deltas grounded at YAM pose ===")
    for arm in ["left", "right"]:
        a = np.array([[s[0], s[1], s[2], s[3], s[4]] for s in stats[arm]], dtype=float)
        print(f"  {arm}: IK ok {a[:,0].mean()*100:3.0f}% | median pos_err={np.median(a[:,1]):6.1f}mm "
              f"ori_err={np.median(a[:,2]):5.1f}deg posOnly={np.median(a[:,3]):6.1f}mm | "
              f"|tgt| {a[:,4].min():.0f}-{a[:,4].max():.0f}mm")
