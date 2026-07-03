#!/usr/bin/env python3
"""Live AprilTag detection-quality checker for the YAM front camera.

Opens the SAME processed front_img_1 the calibration/rollout uses (Atlas cam0,
double-sphere->pinhole rectify + re-aim + crop, via AtlasStereoCamera) and
overlays, per frame:
  * the detected tag outline + id,
  * decision_margin (the single best predictor of whether calibrate_yam will
    keep the frame): GREEN >=40 good, YELLOW 25-40 weak, RED <25 / none,
  * apparent tag edge length in px and estimated distance (m),
  * a rolling "good-frame" fraction so you can sweep the arm through poses and
    watch coverage.

Use it BEFORE recording a calibration demo: position the tag, fix glare/lighting,
and confirm the margin stays green across the poses you plan to record. Weak/rare
detections here => a demo that yields 0 usable frames (see calibrate_yam.py).

Shows decision_margin AND reproj (mean px between the detected corners and the
tag pose reprojected via K -- the exact quantity calibrate_yam gates on at 4px):
GREEN <=1.5px, YELLOW <=4px, RED >4px.

RUN live (on the robot, Atlas camera connected; no arms/CAN needed):
    python debug/yam/tag_quality_live.py
    # options: --tag-family tag36h11  --tag-size 0.090  --serial 328  --snapshot-dir /tmp

RUN playback (replay a recorded demo, no camera needed):
    python debug/yam/tag_quality_live.py --h5py-path calibration_left/demo_0.hdf5
    # options: --every-k 1  --fps 30  --cam-name front_img_1
Keys: q/ESC quit | s snapshot | (playback) SPACE pause | , / . step when paused
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (_REPO, os.path.join(_REPO, "external", "rpl_vision_utils"),
           os.path.join(_REPO, "egomimic", "robot", "YAM")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import pupil_apriltags as apriltag          # noqa: E402
from yam_cameras import AtlasStereoCamera, ATLAS_SERIAL   # noqa: E402

GOOD, WEAK = 40.0, 25.0     # decision_margin thresholds
REPROJ_GOOD, REPROJ_OK = 1.5, 4.0   # px; 4.0 == calibrate_yam's default gate
_GREEN, _YELLOW, _RED = (0, 220, 0), (0, 220, 220), (0, 0, 255)


def color_for(margin):
    return _GREEN if margin >= GOOD else _YELLOW if margin >= WEAK else _RED


def reproj_color(err):
    return _GREEN if err <= REPROJ_GOOD else _YELLOW if err <= REPROJ_OK else _RED


def reproj_err(det, cam_params, tag_size):
    """Mean px between detected corners and the tag pose reprojected via K --
    identical to calibrate_yam.reproject_residual (rectified pinhole, zero
    distortion), so this mirrors the exact quantity the calibration gates on."""
    if det.pose_R is None or det.pose_t is None:
        return float("nan")
    fx, fy, cx, cy = cam_params
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    s = float(tag_size) / 2.0
    # pupil_apriltags detected-corner order (CCW from top-left); must match calibrate_yam.
    objp = np.array([[-s, s, 0.0], [s, s, 0.0], [s, -s, 0.0], [-s, -s, 0.0]], np.float64)
    rvec, _ = cv2.Rodrigues(det.pose_R.astype(np.float64))
    tvec = det.pose_t.reshape(3, 1).astype(np.float64)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, np.zeros(5))
    return float(np.linalg.norm(proj.reshape(-1, 2) - np.array(det.corners, np.float64),
                                axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-family", default="tag36h11")
    ap.add_argument("--tag-size", type=float, default=0.090, help="tag side (m)")
    ap.add_argument("--serial", type=int, default=ATLAS_SERIAL)
    ap.add_argument("--snapshot-dir", default="/tmp")
    ap.add_argument("--h5py-path", default=None,
                    help="replay front_img_1 from a recorded demo instead of the "
                         "live camera (e.g. calibration_left/demo_0.hdf5)")
    ap.add_argument("--cam-name", default="front_img_1",
                    help="image key under observations/images for playback")
    ap.add_argument("--every-k", type=int, default=1, help="playback: step every k frames")
    ap.add_argument("--fps", type=float, default=30.0, help="playback speed cap (Hz)")
    args = ap.parse_args()

    playback = args.h5py_path is not None
    cam = h5 = None
    if playback:
        import h5py
        from egomimic.utils.egomimicUtils import INTRINSICS
        h5 = h5py.File(args.h5py_path, "r")
        imgs = h5[f"observations/images/{args.cam_name}"]
        n_frames = imgs.shape[0]
        K = np.asarray(INTRINSICS["yam"], dtype=np.float64)   # derived front K
        print(f"[tagcheck] PLAYBACK {args.h5py_path} [{args.cam_name}] "
              f"{n_frames} frames @ {imgs.shape[1]}x{imgs.shape[2]}")
    else:
        cam = AtlasStereoCamera(serial=args.serial)
        cam.start()
        if not cam.wait_until_ready(10.0):
            print("[tagcheck] camera never produced a frame; is the Atlas connected?")
            cam.close(); os._exit(1)
        K = np.asarray(cam.intrinsics, dtype=np.float64)   # 3x4 (K|0) from the processor

    cam_params = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]  # fx,fy,cx,cy
    detector = apriltag.Detector(families=args.tag_family, quad_decimate=1.0)

    recent = deque(maxlen=90)   # ~3 s of good/bad flags at 30 fps
    t_prev = None               # for FPS (avoid Date.now-style calls; use perf loop)
    fps = 0.0
    idx = 0                     # playback frame cursor
    paused = False
    win = "YAM front tag quality"
    print(f"[tagcheck] {args.tag_family} tag-size={args.tag_size}m "
          f"| K fx={cam_params[0]:.1f} cx={cam_params[2]:.1f} cy={cam_params[3]:.1f}")
    print("[tagcheck] q/ESC quit | s snapshot" +
          (" | SPACE pause | ,/. step when paused" if playback else ""))

    while True:
        if playback:
            img = np.ascontiguousarray(imgs[idx % n_frames])
        else:
            img = cam.get_image()                 # BGR uint8, current pipeline
        vis = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = detector.detect(gray, estimate_tag_pose=True,
                               camera_params=cam_params, tag_size=args.tag_size)

        best = max((d.decision_margin for d in dets), default=0.0)
        best_reproj = float("nan")
        recent.append(best >= GOOD)
        for d in dets:
            c = np.array(d.corners, np.int32)
            col = color_for(d.decision_margin)
            cv2.polylines(vis, [c], True, col, 2)
            edge = float(np.linalg.norm(np.array(d.corners[0]) - np.array(d.corners[1])))
            dist = float(np.linalg.norm(d.pose_t)) if d.pose_t is not None else float("nan")
            err = reproj_err(d, cam_params, args.tag_size)
            if d.decision_margin >= best:
                best_reproj = err
            cv2.putText(vis, f"id{d.tag_id} m={d.decision_margin:.0f} {edge:.0f}px "
                             f"{dist:.2f}m reproj={err:.2f}px",
                        (int(c[:, 0].min()), int(c[:, 1].min()) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, reproj_color(err), 2)

        # header: verdict + rolling coverage + best reproj
        verdict = ("GOOD" if best >= GOOD else "WEAK" if best >= WEAK else "NONE/BAD")
        frac = 100.0 * sum(recent) / max(1, len(recent))
        pos = (f"f{idx % n_frames}/{n_frames}" if playback else f"{fps:.0f}fps")
        rp = "--" if best_reproj != best_reproj else f"{best_reproj:.2f}px"  # nan check
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(vis, f"{verdict}  margin={best:.0f}  reproj={rp}  "
                         f"good(3s)={frac:.0f}%  dets={len(dets)}  {pos}"
                         f"{'  PAUSED' if paused else ''}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color_for(best), 2)

        cv2.imshow(win, vis)
        delay = max(1, int(1000.0 / args.fps)) if playback else 1
        k = cv2.waitKey(delay) & 0xFF
        if k in (27, ord("q")):
            break
        if k == ord("s"):
            tag = f"f{idx % n_frames}" if playback else f"{int(frac):03d}pct"
            p = os.path.join(args.snapshot_dir, f"tagcheck_{tag}.png")
            cv2.imwrite(p, vis); print("[tagcheck] saved", p)
        if playback:
            if k == ord(" "):
                paused = not paused
            if paused and k == ord("."):
                idx += args.every_k
            elif paused and k == ord(","):
                idx = max(0, idx - args.every_k)
            elif not paused:
                idx += args.every_k

        now = time.perf_counter()
        if t_prev is not None:
            dt = now - t_prev
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
        t_prev = now

    if cam is not None:
        cam.close()
    if h5 is not None:
        h5.close()
    cv2.destroyAllWindows()
    os._exit(0)     # avoid pupil_apriltags/h5py-style teardown segfaults


if __name__ == "__main__":
    main()
