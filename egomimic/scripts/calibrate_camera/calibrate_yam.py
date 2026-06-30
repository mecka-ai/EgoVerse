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
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "external", "rpl_vision_utils")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from egomimic.utils.egomimicUtils import INTRINSICS


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
    objp = np.array([[-s, -s, 0.0], [s, -s, 0.0], [s, s, 0.0], [-s, s, 0.0]], np.float64)
    rvec, _ = cv2.Rodrigues(det.pose_R.astype(np.float64))
    tvec = det.pose_t.reshape(3, 1).astype(np.float64)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, distCoeffs)
    return float(np.linalg.norm(proj.reshape(-1, 2) - np.array(det.corners, np.float64),
                                axis=1).mean())


def main():
    args = parse_args()

    K3x4 = np.asarray(INTRINSICS[args.intrinsics_key], dtype=np.float64)
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
        eepose = f["observations/eepose"]          # (T, 14)
        T = imgs.shape[0]
        off = _ARM_OFFSET[args.arm]
        for t in tqdm(range(0, T, args.every_k)):
            img = imgs[t]
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

            pose = np.asarray(eepose[t])
            assert pose.shape == (14,), f"eepose shape {pose.shape} != (14,)"
            pos = pose[off:off + 3]
            rot = Rot.from_euler("ZYX", pose[off + 3:off + 6], degrees=False)
            # Eye-to-hand (stationary cam, tag on EE): feed base->gripper (inverse
            # of the EE pose) so calibrateHandEye returns camera->base directly.
            R_base2gripper.append(rot.as_matrix().T)
            t_base2gripper.append(-rot.as_matrix().T @ np.asarray(pos)[:, None])
            R_target2cam.append(det.pose_R)
            t_target2cam.append(det.pose_t.reshape(3, 1))
            used += 1

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
