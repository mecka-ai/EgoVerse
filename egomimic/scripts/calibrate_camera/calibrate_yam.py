"""Hand-eye calibration for the YAM front camera -> per-arm-base extrinsic.

Computes ``EXTRINSICS["yam"][arm]`` (the Atlas forward-stereo ``front_img_1`` ->
arm-base transform) from a recorded YAM demo, using an AprilTag fixed to the
moving end-effector and the STATIONARY front camera (eye-to-hand). This is the
YAM analogue of ``calibrate_eva.py``, wired for:

  * the YAM demo HDF5 layout  (observations/images/front_img_1, observations/eepose),
  * ``INTRINSICS["yam"]``      (the rectified+re-aimed+cropped pinhole K — because
                                front_img_1 is now a true pinhole, AprilTag pose
                                estimation with ZERO distortion is correct),
  * the vendored AprilTag detector in external/rpl_vision_utils (pupil_apriltags).

PREREQS
  uv pip install pupil-apriltags          # the only missing dep (PyPI)
  # external/rpl_vision_utils is a vendored submodule (added to sys.path below).

DATA COLLECTION
  Rigidly mount the AprilTag (default 90 mm, tag36h11) to ONE end-effector, then
  record a demo with collect_yam_demo.py while moving THAT arm through many varied
  poses (vary ROTATION, not just translation — hand-eye is ill-conditioned without
  rotation diversity), keeping the tag clearly in the front camera's view.
  Calibrate each arm separately (move + record the tag on that arm).

RUN
  python egomimic/scripts/calibrate_camera/calibrate_yam.py \
      --h5py-path /path/to/demo_0.hdf5 --arm left --tag-size 0.090 --debug

  Prints the 4x4 camera->base T, ready to paste into EXTRINSICS["yam"]["left"]
  in egomimic/utils/egomimicUtils.py. Use --store-npy to also save it next to the
  HDF5. Repeat with --arm right for the right arm.
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from tqdm import tqdm

# --- Make the vendored rpl_vision_utils submodule + egomimic importable ---------
# This file: <repo>/egomimic/scripts/calibrate_camera/calibrate_yam.py
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "external", "rpl_vision_utils"),
           os.path.join(_REPO_ROOT, "external", "i2rt"),
           os.path.join(_REPO_ROOT, "egomimic", "robot", "YAM")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from yam_cameras import (ATLAS_SERIAL, load_ds_intrinsics, load_rig_aim,
                         stereo_front_output_intrinsics)
# NOTE: egomimicUtils and the i2rt mink kinematics are imported LAZILY (after the
# detection loop). Both pull heavy libs (torch/torchvision via egomimicUtils;
# mujoco/mink via i2rt) whose presence perturbs the pupil_apriltags C-extension
# enough to SEGFAULT mid detection loop. Keep them off the stack until detection
# is done.


def _resolve_intrinsics(key):
    """3x4 K for INTRINSICS[key]. For 'yam' use the light yam_cameras derivation
    (numpy+sqlite) to avoid importing egomimicUtils (torch/…) on the detector path."""
    if key == "yam":
        return np.asarray(stereo_front_output_intrinsics(), dtype=np.float64)
    from egomimic.utils.egomimicUtils import INTRINSICS
    return np.asarray(INTRINSICS[key], dtype=np.float64)


def _expected_front_size(serial=ATLAS_SERIAL):
    """(w, h) of the front_img_1 the CURRENT pipeline produces for ``serial``
    (cam0 full width/height minus the rig_aim ROI crop). The recorded demo frames
    must match this, or the intrinsics no longer describe them and the solve is
    silently wrong."""
    cam0 = load_ds_intrinsics(serial, 0)
    cfg = load_rig_aim(serial)
    w = cam0["width"] - int(cfg["crop_left"]) - int(cfg["crop_right"])
    h = cam0["height"] - int(cfg["crop_top"]) - int(cfg["crop_bottom"])
    return w, h

# Raw Atlas stereo (cam0 | cam1) as stored by collect_yam_demo: 1920 + 1920 wide.
# calibrate needs the RECTIFIED pinhole (to match INTRINSICS['yam'] + zero
# distortion), so raw frames are undistorted on the fly via StereoFrontProcessor.
_RAW_STEREO_W = 3840
_RAW_STEREO_HALF = 1920


def _make_detector(family):
    """Import the vendored AprilTag detector lazily (so --help works pre-install)."""
    try:
        from rpl_vision_utils.utils.apriltag_detector import AprilTagDetector
    except ImportError as e:
        raise SystemExit(
            f"Could not import the AprilTag detector ({e}).\n"
            f"  - ensure the submodule exists: external/rpl_vision_utils\n"
            f"  - install its dep:  uv pip install pupil-apriltags"
        )
    return AprilTagDetector(families=family, quad_decimate=1.0)

# Per-arm slice into the 14-dim eepose (xyz + ZYX euler + gripper), matching
# YAMInterface.get_obs / collect_yam_demo (left[0:7], right[7:14]).
_ARM_OFFSET = {"left": 0, "right": 7}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5py-path", required=True, help="recorded YAM demo HDF5")
    p.add_argument("--arm", required=True, choices=["left", "right"],
                   help="which arm carries the tag (calibrate one arm per run)")
    p.add_argument("--tag-size", type=float, default=0.090,
                   help="AprilTag side length in METERS (default 0.090 = 90mm)")
    p.add_argument("--serial", type=int, default=328,
                   help="Atlas serial for rectification (StereoFrontProcessor)")
    p.add_argument("--rectify", default="auto", choices=["auto", "on", "off"],
                   help="raw fisheye stereo -> rectified pinhole so INTRINSICS['yam'] "
                        f"applies. 'auto' rectifies only when the stored image is the "
                        f"raw {_RAW_STEREO_W}px-wide stereo pair (default)")
    p.add_argument("--tag-family", default="tag36h11")
    p.add_argument("--tag-id", type=int, default=None,
                   help="restrict to this tag id (else require exactly one detection)")
    p.add_argument("--intrinsics-key", default="yam",
                   help="INTRINSICS[...] key for the front camera (default 'yam')")
    p.add_argument("--cam-name", default="front_img_1",
                   help="image key under observations/images (default front_img_1)")
    p.add_argument("--every-k", type=int, default=5, help="sample every k frames")
    p.add_argument("--max-reproj-px", type=float, default=4.0,
                   help="drop frames whose tag reprojection error exceeds this (px)")
    p.add_argument("--method", default="PARK",
                   choices=["PARK", "TSAI", "HORAUD", "ANDREFF", "DANIILIDIS"],
                   help="cv2.calibrateHandEye solver")
    p.add_argument("--store-npy", action="store_true",
                   help="also save the 4x4 T as <h5_dir>/extrinsics_yam_<arm>.npy")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def reproject_residual(det, K, tag_size, distCoeffs=None):
    """Mean px error between detected corners and the tag pose reprojected via K."""
    if distCoeffs is None:
        distCoeffs = np.zeros(5)
    s = float(tag_size) / 2.0
    # pupil_apriltags detected-corner order (CCW from top-left in the tag frame);
    # the y-flipped order makes a correct pose reproject ~100px off.
    objp = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], np.float64)
    rvec, _ = cv2.Rodrigues(det.pose_R.astype(np.float64))
    tvec = det.pose_t.reshape(3, 1).astype(np.float64)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, distCoeffs)
    return float(np.linalg.norm(proj.reshape(-1, 2) - np.array(det.corners, np.float64),
                                axis=1).mean())


def _make_rectifier(rectify_mode, serial, stored_w):
    """Return a fn(raw_stereo)->rectified pinhole, or None if no rectify needed.

    calibrate assumes front_img_1 is a pinhole matching INTRINSICS['yam'] with zero
    distortion. Raw demos store the 3840px fisheye stereo pair, so undistort it via
    the SAME StereoFrontProcessor rollout uses (double-sphere -> pinhole + re-aim +
    crop), keeping detection and intrinsics consistent."""
    want = rectify_mode == "on" or (rectify_mode == "auto" and stored_w == _RAW_STEREO_W)
    if not want:
        return None
    from yam_cameras import StereoFrontProcessor
    proc = StereoFrontProcessor(serial=serial)
    print(f"[calib] rectifying raw {stored_w}px stereo -> {proc.out_w}x{proc.out_h} "
          f"pinhole (serial {serial}); must match INTRINSICS.")

    def rectify(frame):
        left = frame[:, :_RAW_STEREO_HALF]
        right = frame[:, _RAW_STEREO_HALF:2 * _RAW_STEREO_HALF]
        return proc.process_split(left, right)
    return rectify


def main():
    args = parse_args()

    K3x4 = _resolve_intrinsics(args.intrinsics_key)
    fx, fy, cx, cy = K3x4[0, 0], K3x4[1, 1], K3x4[0, 2], K3x4[1, 2]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    intr = {"fx": fx, "fy": fy, "cx": cx, "cy": cy}
    print(f"[calib] intrinsics '{args.intrinsics_key}': fx={fx:.2f} fy={fy:.2f} "
          f"cx={cx:.2f} cy={cy:.2f}")

    detector = _make_detector(args.tag_family)

    R_base2gripper, t_base2gripper = [], []
    R_target2cam, t_target2cam = [], []
    used = missed = rejected = 0

    with h5py.File(args.h5py_path, "r") as f:
        imgs = f[f"observations/images/{args.cam_name}"]
        # Gripper pose is recomputed from joint_positions with the (now-fixed) mink
        # grasp_site FK -- the SAME model YAMInterface.get_pose/eepose uses, so the
        # extrinsic lands in the grasp_site frame that rollout's obs is in. We use
        # joint_positions (not the stored `eepose`) so this works on demos recorded
        # BEFORE the no_gripper.yml joint-6 fix, whose baked eepose is stale.
        joints = f["observations/joint_positions"]   # (T, 14): [L(7), R(7)]
        q_arm = []    # per-used-frame 6-joint configs; FK'd in one batch below
        T = imgs.shape[0]
        off = _ARM_OFFSET[args.arm]
        rectify = _make_rectifier(args.rectify, args.serial, imgs.shape[2])

        # Guard: the frames we detect on MUST match the resolution the current
        # intrinsics describe, else K's principal point / focal no longer align
        # with the pixels and the solve is silently wrong. Raw demos are
        # re-processed to the current size above (always matches); an
        # already-processed demo is checked as-stored -- this catches reusing a
        # demo from an OLDER crop/serial/re-aim pipeline. Recollect instead.
        probe = imgs[0]
        if rectify is not None:
            probe = rectify(probe)
        exp_w, exp_h = _expected_front_size(args.serial)
        if (int(probe.shape[1]), int(probe.shape[0])) != (exp_w, exp_h):
            raise SystemExit(
                f"[calib] front_img_1 is {probe.shape[1]}x{probe.shape[0]} but the "
                f"current pipeline (serial {args.serial} + rig_aim crop) produces "
                f"{exp_w}x{exp_h}. This demo was recorded with a different "
                f"crop/serial/re-aim, so its frames no longer match "
                f"INTRINSICS['{args.intrinsics_key}']. Recollect with the current "
                f"pipeline (see stream_stereo.py / rig_aim.json).")

        for t in tqdm(range(0, T, args.every_k)):
            img = imgs[t]
            if rectify is not None:
                img = rectify(img)
            dets = detector.detect(img, intrinsics=intr, tag_size=args.tag_size)
            if args.tag_id is not None:
                dets = [d for d in dets if d.tag_id == args.tag_id]
            if len(dets) != 1:
                missed += 1
                continue
            det = dets[0]
            err = reproject_residual(det, K, args.tag_size)
            if err > args.max_reproj_px:
                rejected += 1
                if args.debug:
                    print(f"[t={t}] reject: reproj err {err:.2f}px > {args.max_reproj_px}")
                continue
            if args.debug:
                print(f"[t={t}] ok: reproj err {err:.2f}px")

            q = np.asarray(joints[t])
            assert q.shape == (14,), f"joint_positions shape {q.shape} != (14,)"
            # Defer FK: torch calls interleaved with the AprilTag C-extension can
            # segfault, so only collect joints here and batch-FK after the loop.
            q_arm.append(q[off:off + 6])
            R_target2cam.append(det.pose_R)
            t_target2cam.append(det.pose_t.reshape(3, 1))
            used += 1

    # Batch FK once, outside the detector loop (see note above). Uses the fixed
    # mink grasp_site model -- same as YAMInterface.get_pose/eepose -- so the
    # extrinsic is in the grasp_site frame the rollout obs uses.
    if q_arm:
        from i2rt.robots.kinematics import Kinematics   # lazy: mujoco/mink off the detector loop
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml
        from yam_interface import _patch_kinematics_xml  # lazy: pulls the i2rt SDK
        # Apply the SAME per-arm model fixes as YAMInterface (joint-6 axis flip
        # for both arms; left additionally gets the grasp_site z-roll): an
        # unpatched/wrong-arm model here would solve an extrinsic in a different
        # frame than the one collection/rollout FK reports.
        kin = Kinematics(
            _patch_kinematics_xml(
                combine_arm_and_gripper_xml(ArmType.YAM, GripperType.NO_GRIPPER),
                args.arm,
            ),
            "grasp_site",
        )
        for q in q_arm:
            T_bg = kin.fk(np.asarray(q))          # 4x4 base->grasp_site
            R_bg, pos = T_bg[:3, :3], T_bg[:3, 3]
            # Eye-to-hand (stationary cam, tag on EE): feed base->gripper (inverse
            # of the EE pose) so calibrateHandEye returns camera->base directly.
            R_base2gripper.append(R_bg.T)
            t_base2gripper.append(-R_bg.T @ pos[:, None])

    print(f"[calib] used {used} frames | missed {missed} (no/!=1 tag) | "
          f"rejected {rejected} (high reproj err)")
    if used < 8:
        raise SystemExit(
            f"Only {used} usable frames — need ~15+ with VARIED EE rotation for a "
            f"stable hand-eye solve. Collect more, keeping the tag in view.")

    method = getattr(cv2, f"CALIB_HAND_EYE_{args.method}")
    R, t = cv2.calibrateHandEye(R_base2gripper, t_base2gripper,
                                R_target2cam, t_target2cam, method=method)
    Tmat = np.eye(4)
    Tmat[:3, :3], Tmat[:3, 3] = R, t.reshape(3)

    # Self-consistency residual: with the solved T (camera->base), the tag pose in
    # the GRIPPER frame, T_gt_i = [base->gripper]_i @ T @ [tag->cam]_i, must be the
    # same rigid transform in every frame (the tag is bolted to the EE). Its spread
    # is the end-to-end solve quality — a wrong FK model or intrinsics shows up
    # here as tens of degrees, which calibrateHandEye itself never reports.
    T_gt = []
    for R_bg_inv, t_bg_inv, R_ct, t_ct in zip(
        R_base2gripper, t_base2gripper, R_target2cam, t_target2cam
    ):
        A = np.eye(4)
        A[:3, :3], A[:3, 3] = R_bg_inv, t_bg_inv.reshape(3)
        B = np.eye(4)
        B[:3, :3], B[:3, 3] = R_ct, t_ct.reshape(3)
        T_gt.append(A @ Tmat @ B)
    T_gt = np.stack(T_gt)
    rots = Rot.from_matrix(T_gt[:, :3, :3])
    ang = np.degrees((rots.mean().inv() * rots).magnitude())
    dist_mm = np.linalg.norm(
        T_gt[:, :3, 3] - T_gt[:, :3, 3].mean(axis=0), axis=1
    ) * 1000.0
    print(f"[calib] hand-eye residual (gripper<-tag spread over {len(T_gt)} frames): "
          f"rot median {np.median(ang):.2f} / max {ang.max():.2f} deg | "
          f"trans median {np.median(dist_mm):.1f} / max {dist_mm.max():.1f} mm")
    if np.median(ang) > 5.0 or np.median(dist_mm) > 20.0:
        print("[calib] WARNING: residual is LARGE — the solve is likely wrong "
              "(FK model / intrinsics / tag detections). Do NOT deploy this T.")

    print("\n========== camera -> base extrinsic (paste into EXTRINSICS) ==========")
    print(f'    "yam": {{ ..., "{args.arm}": np.array([')
    for row in Tmat:
        print("        [" + ", ".join(f"{v: .8f}" for v in row) + "],")
    print("    ]) }")
    print("======================================================================")

    if args.store_npy:
        out = os.path.join(os.path.dirname(os.path.abspath(args.h5py_path)),
                           f"extrinsics_yam_{args.arm}.npy")
        np.save(out, Tmat)
        print(f"[calib] saved {out}")


if __name__ == "__main__":
    main()
