#!/usr/bin/env python3
"""Verify the pose (ypr AND xyz) mapping through the YAM intrinsic/extrinsic chain.

Tests the ACTUAL runtime chain used by rollout's obs:
    get_pose (mink grasp_site FK, now joint-6-corrected) -> EXTRINSICS['yam']
    -> INTRINSICS['yam'] -> pixels.
At each commanded pose it overlays two readouts on the front image:

  * PREDICTED (solid arrows + white dot): the EE frame from ri.get_pose(), pushed
    through the extrinsic + intrinsic with the real pipeline fns
    (ee_pose_to_cam_pixels / base_frame_to_cam_frame). This is exactly what the
    policy's camframe observation sees.
  * MEASURED (thin arrows + yellow ring): the AprilTag pose_R/pose_t -- the true
    cam-frame pose (independent ground truth).

PASS = the white dot sits on the tag ring across the xyz sweep, and the solid axes
track the tag's axes across the ypr sweep. A rigid offset => extrinsic error; a
drift that grows toward frame edges => intrinsic error.

MOTION: the arm moves OUT to a start pose, then sweeps -- autonomously. Ensure
clearance + e-stop. Commanding + FK now share the corrected mink model, so the
commanded labels are physically meaningful. Returns to start at the end.

RUN (on the robot):
    python debug/yam/rotation_verify.py --arm left --present 0.10,0,0.05
    # --present dx,dy,dz (m, base frame, added to the start pose)
    # --ypr-deltas -30,-15,0,15,30 (deg)  --xyz-deltas -0.06,-0.03,0,0.03,0.06 (m)
    # --hold-sec 1.5  --tag-size 0.140  --out-dir /tmp/rotverify
"""
import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# The i2rt DMChainCanInterface.close() shuts the CAN socket without joining its
# control thread, so an in-flight bus.send() hits fd=-1 and raises in that daemon
# thread during shutdown. Harmless -- swallow it (same as collect_yam_demo).
_SHUTTING_DOWN = threading.Event()


def _quiet_thread_excepthook(args):
    if _SHUTTING_DOWN.is_set() and issubclass(args.exc_type, (ValueError, OSError)):
        return
    threading.__excepthook__(args)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "external", "rpl_vision_utils"),
           os.path.join(_REPO, "egomimic", "robot", "YAM")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from egomimic.utils.egomimicUtils import (           # noqa: E402
    INTRINSICS, EXTRINSICS, ee_pose_to_cam_pixels, cam_frame_to_cam_pixels,
    base_frame_to_cam_frame,
)
from yam_interface import YAMInterface                # noqa: E402
# NOTE: the AprilTag detector (pupil_apriltags C-extension) is imported LAZILY
# AFTER the arm is connected. Loading/creating it before YAMInterface connects
# can disrupt the tight-timing DM motor handshake and make motor init fail.

_X, _Y, _Z = (0, 0, 255), (0, 255, 0), (255, 0, 0)   # BGR: X red, Y green, Z blue
GOOD, WEAK = 40.0, 25.0    # decision_margin thresholds (same as tag_quality_live)
_GREEN, _YELLOW, _RED = (0, 220, 0), (0, 220, 220), (0, 0, 255)


def _margin_color(m):
    return _GREEN if m >= GOOD else _YELLOW if m >= WEAK else _RED


def _pxi(r):
    return (int(round(r[0])), int(round(r[1])))


def annotate_dets(img, dets):
    """Draw EVERY detection (outline + id/margin/edge/dist) color-coded by
    decision_margin -- shows WHY a tag is dropping (weak margin, multiple
    detections, too small/far). Returns the best detection or None."""
    best = None
    for d in dets:
        c = np.array(d.corners, np.int32)
        col = _margin_color(d.decision_margin)
        cv2.polylines(img, [c], True, col, 2)
        edge = float(np.linalg.norm(np.array(d.corners[0]) - np.array(d.corners[1])))
        dist = float(np.linalg.norm(d.pose_t)) if d.pose_t is not None else float("nan")
        cv2.putText(img, f"id{d.tag_id} m={d.decision_margin:.0f} {edge:.0f}px {dist:.2f}m",
                    (int(c[:, 0].min()), int(c[:, 1].min()) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
        if best is None or d.decision_margin > best.decision_margin:
            best = d
    return best


def draw_axes(img, px4, thick, tip):
    """px4: (4,2) [origin, x_tip, y_tip, z_tip]."""
    o = _pxi(px4[0])
    for k, col in ((1, _X), (2, _Y), (3, _Z)):
        cv2.arrowedLine(img, o, _pxi(px4[k]), col, thick, tipLength=tip)


def axes_from_base(T_cam_base, K, p_base, R_base, L=0.06):
    pts = np.stack([p_base, p_base + R_base[:, 0] * L,
                    p_base + R_base[:, 1] * L, p_base + R_base[:, 2] * L])
    return ee_pose_to_cam_pixels(pts, T_cam_base, K)[:, :2]


def axes_from_cam(K, p_cam, R_cam, L=0.06):
    pts = np.stack([p_cam, p_cam + R_cam[:, 0] * L,
                    p_cam + R_cam[:, 1] * L, p_cam + R_cam[:, 2] * L])
    return cam_frame_to_cam_pixels(pts, K)[:, :2]


def build_sweep(ypr_deltas, xyz_deltas):
    sweep = []
    for name, idx in (("yaw", 3), ("pitch", 4), ("roll", 5)):
        for d in ypr_deltas:
            sweep.append(("rot", name, idx, np.deg2rad(d), f"{d:+.0f}deg"))
    for name, idx in (("x", 0), ("y", 1), ("z", 2)):
        for d in xyz_deltas:
            sweep.append(("trn", name, idx, d, f"{d*100:+.0f}cm"))
    return sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="left", choices=["left", "right", "both"])
    ap.add_argument("--present", default="0,0,0",
                    help="dx,dy,dz (m) added to the start pose before sweeping")
    ap.add_argument("--ypr-deltas", default="-30,-15,0,15,30")
    ap.add_argument("--xyz-deltas", default="-0.06,-0.03,0,0.03,0.06")
    ap.add_argument("--channels", default=None,
                    help="comma list arm=canbus (default left=can_follower_l,right=can_follower_r); "
                         "YAMInterface's default can0/can1 doesn't match this rig")
    ap.add_argument("--ee-convention", default="default", choices=["default", "libero"],
                    help="grasp_site frame convention for FK/IK; 'libero' = both arms "
                         "x fwd / y left / z up (see yam_interface._patch_kinematics_xml)")
    ap.add_argument("--hold-sec", type=float, default=1.5)
    ap.add_argument("--tag-size", type=float, default=0.140)
    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--out-dir", default="/tmp/rotverify")
    ap.add_argument("--no-display", action="store_true",
                    help="headless: skip the live cv2 window, keep PNGs + table")
    args = ap.parse_args()

    threading.excepthook = _quiet_thread_excepthook   # silence harmless CAN teardown noise
    arms = ["left", "right"] if args.arm == "both" else [args.arm]
    if args.channels:
        channels = dict(kv.split("=") for kv in args.channels.split(","))
    else:
        channels = {a: f"can_follower_{a[0]}" for a in arms}   # this rig's follower buses
    present = np.array([float(x) for x in args.present.split(",")], dtype=np.float64)
    ypr_deltas = [float(d) for d in args.ypr_deltas.split(",")]
    xyz_deltas = [float(d) for d in args.xyz_deltas.split(",")]
    sweep = build_sweep(ypr_deltas, xyz_deltas)
    os.makedirs(args.out_dir, exist_ok=True)

    K = np.asarray(INTRINSICS["yam"], dtype=np.float64)
    intr = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}

    print(f"[rotverify] arms={arms} present={present.tolist()} | {len(sweep)} poses/arm "
          f"| arm MOVES autonomously in 3s -- clearance + e-stop!")
    for s in (3, 2, 1):
        print(f"  {s}...", flush=True); time.sleep(1.0)

    ri = YAMInterface(arms=arms, channels=channels, zero_gravity_mode=False,
                      ee_frame_convention=args.ee_convention)
    # Import + build the pupil detector AFTER the arm is up (see note at import).
    from rpl_vision_utils.utils.apriltag_detector import AprilTagDetector
    detector = AprilTagDetector(families=args.tag_family, quad_decimate=1.0)
    try:
        for arm in arms:
            if arm not in EXTRINSICS["yam"]:
                print(f"[rotverify] no EXTRINSICS['yam']['{arm}'] -- skip"); continue
            T_cam_base = np.asarray(EXTRINSICS["yam"][arm], dtype=np.float64)

            # 1) move OUT to the start/test pose (get_pose is mink grasp_site, base frame)
            start = np.asarray(ri.get_pose_6d(arm), dtype=np.float64)   # xyz + ZYX ypr
            start[:3] += present
            start = np.append(start, [1.0])
            ri.set_pose(start, arm); time.sleep(args.hold_sec)

            print(f"\n=== arm {arm} ===")
            print(f"{'kind':>4} {'axis':>5} {'d':>7} | {'pred origin px':>16} "
                  f"{'tag ctr px':>14} {'pos err px':>10} | {'pred cam ypr':>20} "
                  f"{'tag cam ypr':>20} | tag")
            front_rec = ri.recorders.get("front_img_1")
            win = "rotverify (q/ESC abort)"
            for kind, axis, idx, dval, lbl in sweep:
                tgt = start.copy(); tgt[idx] += dval
                ri.set_pose(tgt, arm)

                # Live viewer during the hold (like tag_quality_live): every frame
                # shows ALL detections color-coded by decision_margin plus the
                # predicted axes -- so you can SEE why the tag drops (weak margin,
                # glare, too small/far, several tags) while the arm settles.
                img = None; best = None
                t_end = time.perf_counter() + args.hold_sec
                while time.perf_counter() < t_end:
                    raw = front_rec.get_image() if front_rec is not None \
                        else ri.get_obs()["front_img_1"]
                    img = np.ascontiguousarray(raw).copy()
                    T_bg = ri.get_pose(arm, se3=True)   # base->EE (mink, corrected)
                    p_base, R_base = T_bg[:3, 3], T_bg[:3, :3]
                    pred_ax = axes_from_base(T_cam_base, K, p_base, R_base)
                    draw_axes(img, pred_ax, thick=3, tip=0.25)
                    pred_o = pred_ax[0]
                    cv2.circle(img, _pxi(pred_o), 6, (255, 255, 255), 2)

                    dets = detector.detect(img, intrinsics=intr, tag_size=args.tag_size)
                    best = annotate_dets(img, dets)
                    m = best.decision_margin if best is not None else 0.0
                    perr_live = float("nan")
                    if best is not None:
                        # GROUND TRUTH overlay: tag axes (thin) + tag center
                        # (yellow ring) + white error line to the predicted dot.
                        draw_axes(img, axes_from_cam(K, best.pose_t.reshape(3),
                                                     best.pose_R), thick=1, tip=0.2)
                        tag_ctr = np.array(best.center)
                        cv2.circle(img, _pxi(tag_ctr), 8, (0, 255, 255), 2)
                        cv2.line(img, _pxi(pred_o), _pxi(tag_ctr), (255, 255, 255), 1)
                        perr_live = float(np.linalg.norm(pred_o - tag_ctr))
                    verdict = "GOOD" if m >= GOOD else "WEAK" if m >= WEAK else "NONE/BAD"
                    cv2.rectangle(img, (0, 0), (img.shape[1], 52), (0, 0, 0), -1)
                    cv2.putText(img, f"{kind} {axis} {lbl}  |  {verdict} margin={m:.0f} "
                                     f"dets={len(dets)}  err={perr_live:.0f}px", (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, _margin_color(m), 2)
                    cv2.putText(img, "THICK+white dot = PREDICTED (FK->extr->K)   "
                                     "thin+yellow ring = TAG ground truth",
                                (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1)
                    if not args.no_display:
                        cv2.imshow(win, img)
                        if (cv2.waitKey(30) & 0xFF) in (27, ord("q")):
                            raise KeyboardInterrupt
                    else:
                        time.sleep(0.03)

                # Measurement from the LAST hold frame / best detection.
                pose6 = np.concatenate([p_base, R.from_matrix(R_base).as_euler("ZYX")])[None]
                pred_cam_ypr = np.rad2deg(base_frame_to_cam_frame(pose6, T_cam_base)[0, 3:6])
                if best is not None:
                    draw_axes(img, axes_from_cam(K, best.pose_t.reshape(3), best.pose_R),
                              thick=1, tip=0.2)
                    tag_ctr = np.array(best.center)
                    cv2.circle(img, _pxi(tag_ctr), 8, (0, 255, 255), 2)
                    perr = float(np.linalg.norm(pred_o - tag_ctr))
                    typr = R.from_matrix(best.pose_R).as_euler("ZYX", degrees=True)
                    tstr = f"{typr[0]:6.1f},{typr[1]:6.1f},{typr[2]:6.1f}"
                    cstr = f"{tag_ctr[0]:6.0f},{tag_ctr[1]:6.0f}"
                    seen = f"m={best.decision_margin:.0f}"
                else:
                    perr = float("nan"); tstr = " " * 19; cstr = " " * 13; seen = "none"
                print(f"{kind:>4} {axis:>5} {lbl:>7} | {pred_o[0]:7.0f},{pred_o[1]:6.0f} "
                      f"{cstr:>14} {perr:10.1f} | {pred_cam_ypr[0]:6.1f},"
                      f"{pred_cam_ypr[1]:6.1f},{pred_cam_ypr[2]:6.1f} | {tstr} | {seen}")
                cv2.imwrite(os.path.join(args.out_dir, f"{arm}_{kind}_{axis}_{lbl}.png"), img)
            ri.set_pose(start, arm); time.sleep(args.hold_sec)
        print(f"\n[rotverify] frames -> {args.out_dir}")
        print("[rotverify] PASS: white dot tracks the tag ring across the xyz sweep, and "
              "solid axes track the tag axes across the ypr sweep.")
    except KeyboardInterrupt:
        print("\n[rotverify] aborted; arm holds last pose.")
    except Exception:
        import traceback
        print("[rotverify] ERROR in sweep:")
        traceback.print_exc()
    finally:
        _SHUTTING_DOWN.set()   # from here, the CAN-teardown thread error is expected
        ri.close()
        cv2.destroyAllWindows()
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
