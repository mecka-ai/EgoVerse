"""Offline viz of a collected YAM episode: overlay the recorded action chunk
(trajectory dots + EE orientation arrows) on front_img_1, using the SAME
rectified 'yam' intrinsics + 'yam' extrinsics that rollout uses.

For each sampled frame t it takes the next --horizon EE poses (base frame,
per-arm [xyz, ZYX-euler, gripper]), converts them base->camera with
EXTRINSICS['yam'][arm], and renders via Yam.viz(mode='traj+axes'):
  * dots        = the future EE position trajectory (the "action chunk"),
  * red/green/blue arrows = the +x/+y/+z of the EE orientation (the ypr),
    drawn at the chunk start and every --axes-stride poses after it.

Axis ARROWS are the physical EE axes in the PLAIN camera frame (base->camera,
no relabel) — the convention validated against AprilTag ground truth by
debug/yam/rotation_verify.py. The ypr TEXT readout uses the rollout's 'rot-ee'
convention (R_t_e relabel applied base-side, then base->camera) so the numbers
match the rollout CAMFRAME debug prints; the two differ by a rigid rotation.

The wrist cameras are shown alongside the front view as a horizontal strip
[ left_wrist | front_img_1 (+overlay) | right_wrist ], each wrist resized to the
front height (disable with --no-wrists). Wrist keys absent from the episode are
skipped with a warning.

Use this to visually confirm the intrinsics/extrinsics/shift line up with the
arm WITHOUT running the robot.

RUN
  python debug/eva_replay/viz_episode.py --h5 demos/demo_0.hdf5 --out viz_demo0.mp4
  # options: --arm both|left|right  --source actions|observations  --horizon 16
  #          --every-k 30  --axes-stride 4  --no-wrists  --png-dir frames/
  # NOTE: --out must end in a video extension (.mp4/.avi/.mkv); otherwise OpenCV
  #       falls back to its image-sequence writer and writes nothing.
"""
import argparse
import os
import sys

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "egomimic", "robot", "YAM"))

from egomimic.utils.egomimicUtils import EXTRINSICS, base_frame_to_cam_frame
from egomimic.rldb.embodiment.yam import Yam

# Raw Atlas stereo (cam0 | cam1) as stored by collect_yam_demo(raw): 1920+1920 wide.
_RAW_STEREO_W = 3840
_RAW_STEREO_HALF = 1920

# INTRINSICS['yam'] lives in the 640x480 training space: the rollout stretch-
# resizes front_img_1 to this before projecting (commits a2af51b/ea896fa). The
# stored/rectified front is the processor-native 1220x880, so we must resize to
# match K here or every projected point lands ~0.53x too small (compressed into
# the top-left corner, off the arms).
_K_WH = (640, 480)

# eepose = 14-dim per-arm [xyz(3), ZYX euler(3), gripper(1)]; left[0:7] right[7:14].
_ARM = {"left": 0, "right": 7}

# Wrist camera image keys under observations/images.
_WRIST_KEYS = {"left": "left_wrist_img", "right": "right_wrist_img"}

# Fixed tool-frame relabel (yam_rollout.R_t_e). Applied AFTER base->camera so the
# displayed ypr + axis arrows match the rollout's CAMFRAME debug convention
# (ee_pose_to_rot_ee_frame). This is orientation-only; positions are unchanged.
_R_T_E = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True, help="collected YAM demo HDF5")
    p.add_argument("--arm", default="both", choices=["both", "left", "right"])
    p.add_argument("--source", default="actions", choices=["actions", "observations"],
                   help="'actions' = commanded chunk (the action); "
                        "'observations' = executed proprio trajectory")
    p.add_argument("--horizon", type=int, default=64,
                   help="action-chunk length shown, in frames (longer = further)")
    p.add_argument("--every-k", type=int, default=1, help="render every k frames")
    p.add_argument("--axes-stride", type=int, default=0,
                   help="draw EE orientation arrows every N poses along the chunk "
                        "(0 = only the first pose [default], 1 = every pose)")
    p.add_argument("--extrinsics-key", default="yam")
    p.add_argument("--serial", type=int, default=328, help="Atlas serial for rectify")
    p.add_argument("--rectify", default="auto", choices=["auto", "on", "off"],
                   help="raw fisheye stereo -> rectified pinhole (so 'yam' K applies). "
                        "'auto' rectifies only when the stored image is the raw "
                        f"{_RAW_STEREO_W}px-wide stereo pair")
    p.add_argument("--cam-name", default="front_img_1")
    p.add_argument("--wrists", default=True, action=argparse.BooleanOptionalAction,
                   help="show left/right wrist cameras flanking the front view "
                        "(default on; --no-wrists for front only)")
    p.add_argument("--out", default=None, help="output mp4 (default: <h5>_viz.mp4)")
    p.add_argument("--fps", type=int, default=15, help="output video fps")
    p.add_argument("--png-dir", default=None, help="also dump per-frame PNGs here")
    return p.parse_args()


def chunk_cam_frame(ee, t, H, arm, extr_key, rot_ee=False):
    """Base-frame eepose[t:t+H] for one arm -> (n,6) [xyz,ypr] in camera frame.

    rot_ee=False (drawing): PLAIN camera frame (base->camera only). This is the
    physical EE orientation on the image — the convention validated against the
    AprilTag ground truth in debug/yam/rotation_verify.py (axes_from_base). The
    axis arrows MUST use this, or they render rigidly rotated by |R_t_e| = 120°.

    rot_ee=True (text readout): the rollout's 'rot-ee' convention. The rollout
    applies the R_t_e relabel in the BASE frame FIRST (ee_pose_to_rot_ee_frame
    on the raw FK pose) and THEN maps base->camera, i.e. R_cam @ R_t_e @ R_ee —
    NOT R_t_e @ (R_cam @ R_ee); the two differ because composition doesn't
    commute. Use this only for ypr text meant to match the rollout CAMFRAME
    debug prints, never for drawing physical axes."""
    off = _ARM[arm]
    seg = ee[t:t + H, off:off + 6].astype(np.float64)    # xyz + ZYX euler (base)
    if rot_ee:
        seg = seg.copy()
        R_base = Rotation.from_euler("ZYX", seg[:, 3:6]).as_matrix()   # (n,3,3)
        seg[:, 3:6] = Rotation.from_matrix(_R_T_E @ R_base).as_euler("ZYX")
    T_cam_base = np.asarray(EXTRINSICS[extr_key][arm], dtype=np.float64)
    return base_frame_to_cam_frame(seg, T_cam_base)      # (n,6)


def build_viz_data(ee, t, H, arm, extr_key, rot_ee=False):
    """12-dim [L xyz ypr, R xyz ypr] in camera frame for _split_action_pose.

    rot_ee: see chunk_cam_frame — False for drawing (physical axes), True for
    the rollout-convention ypr text.

    Arms not requested are pushed behind the camera (z<0) so they project to
    nothing rather than drawing spurious dots."""
    n = min(H, ee.shape[0] - t)
    hidden = np.tile([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], (n, 1))
    left = (chunk_cam_frame(ee, t, H, "left", extr_key, rot_ee)
            if arm in ("both", "left") else hidden)
    right = (chunk_cam_frame(ee, t, H, "right", extr_key, rot_ee)
             if arm in ("both", "right") else hidden)
    return np.concatenate([left, right], axis=1).astype(np.float32)  # (n,12)


def label_ypr(img, viz12):
    """Stamp the chunk-start ypr (deg) for each arm in a corner."""
    deg = lambda v: np.round(np.rad2deg(v), 1)
    l_ypr, r_ypr = deg(viz12[0, 3:6]), deg(viz12[0, 9:12])
    for i, (txt, col) in enumerate([
        (f"L ypr(cam_rot_ee) {l_ypr}", (255, 180, 80)),
        (f"R ypr(cam_rot_ee) {r_ypr}", (80, 180, 255)),
    ]):
        y = img.shape[0] - 12 - i * 18
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    return img


def _load_wrist_bgr(f, side, t, target_h):
    """Wrist frame t as BGR, resized to target_h (aspect-preserving).

    Returns None if the episode has no image dataset for that wrist (the
    collector saves only cameras that opened, so this can legitimately happen)."""
    key = f"observations/images/{_WRIST_KEYS[side]}"
    if key not in f:
        return None
    frame_bgr = np.ascontiguousarray(f[key][t][..., ::-1])   # RGB->BGR for cv2
    h, w = frame_bgr.shape[:2]
    new_w = max(1, int(round(w * target_h / h)))
    return cv2.resize(frame_bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _label_panel(img, text):
    """Top-left caption on a camera panel."""
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
    cv2.putText(img, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 1)
    return img


def compose_strip(front_bgr, left_wrist, right_wrist):
    """Horizontal strip [ left_wrist | front (+overlay) | right_wrist ].

    Wrist panels are already sized to the front height; absent ones are omitted.
    A camera is present (or not) for the whole episode, so the strip width is
    stable across frames."""
    panels = []
    if left_wrist is not None:
        panels.append(_label_panel(left_wrist, "left_wrist"))
    panels.append(front_bgr)
    if right_wrist is not None:
        panels.append(_label_panel(right_wrist, "right_wrist"))
    return np.hstack(panels)


def _make_rectifier(a, stored_w):
    """Return a fn(raw_bgr_stereo)->rectified_bgr, or None if no rectify needed."""
    want = a.rectify == "on" or (a.rectify == "auto" and stored_w == _RAW_STEREO_W)
    if not want:
        return None
    from yam_cameras import StereoFrontProcessor
    proc = StereoFrontProcessor(serial=a.serial)
    print(f"[viz] rectifying raw {stored_w}px stereo -> {proc.out_w}x{proc.out_h} "
          f"pinhole (matches INTRINSICS['yam']).")

    def rectify(frame_bgr):
        left = frame_bgr[:, :_RAW_STEREO_HALF]
        right = frame_bgr[:, _RAW_STEREO_HALF:2 * _RAW_STEREO_HALF]
        return proc.process_split(left, right)
    return rectify


def main():
    a = parse_args()
    out_path = a.out or (os.path.splitext(a.h5)[0] + "_viz.mp4")
    # cv2.VideoWriter selects its backend from the extension. Without a video
    # container extension it falls back to the image-sequence writer (which
    # needs a %0Nd pattern) and writes nothing — coerce to .mp4 so a bare name
    # like "viz_demo0" still produces a video.
    _VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".m4v")
    if os.path.splitext(out_path)[1].lower() not in _VIDEO_EXTS:
        print(f"[viz] --out '{out_path}' has no video extension; using '{out_path}.mp4'")
        out_path += ".mp4"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if a.png_dir:
        os.makedirs(a.png_dir, exist_ok=True)
    with h5py.File(a.h5, "r") as f:
        imgs = f[f"observations/images/{a.cam_name}"]        # (T,H,W,3) RGB
        ee = np.asarray(f[f"{a.source}/eepose"])             # (T,14) base frame
        T = imgs.shape[0]
        rectify = _make_rectifier(a, imgs.shape[2])
        if a.wrists:
            missing = [s for s in ("left", "right")
                       if f"observations/images/{_WRIST_KEYS[s]}" not in f]
            if missing:
                print(f"[viz] WARNING: episode has no {', '.join(missing)} wrist "
                      f"camera(s); those panels will be omitted.")
        # Arrows only at the first pose of the chunk unless a stride is requested.
        stride = None if a.axes_stride == 0 else a.axes_stride
        writer = None
        n = 0
        for t in range(0, T, a.every_k):
            frame_bgr = np.ascontiguousarray(imgs[t][..., ::-1])   # RGB->BGR for cv2
            if rectify is not None:
                frame_bgr = rectify(frame_bgr)
            # Match the front image to the space INTRINSICS['yam'] is defined in
            # (640x480), exactly as the rollout does, so the overlay lines up.
            if frame_bgr.shape[1::-1] != _K_WH:
                frame_bgr = cv2.resize(frame_bgr, _K_WH, interpolation=cv2.INTER_AREA)
            # Drawn arrows: PLAIN camera frame (tag-validated convention).
            viz12 = build_viz_data(ee, t, a.horizon, a.arm, a.extrinsics_key)
            overlay = Yam.viz(image=frame_bgr, viz_data=viz12,
                              mode="traj+axes", axes_stride=stride)
            # Text ypr: rollout 'rot-ee' convention (base-side relabel), so the
            # numbers match the rollout CAMFRAME debug prints.
            ypr12 = build_viz_data(ee, t, 1, a.arm, a.extrinsics_key, rot_ee=True)
            overlay = label_ypr(overlay, ypr12)
            if a.wrists:
                target_h = overlay.shape[0]
                left_wrist = _load_wrist_bgr(f, "left", t, target_h)
                right_wrist = _load_wrist_bgr(f, "right", t, target_h)
                overlay = compose_strip(overlay, left_wrist, right_wrist)
            if writer is None:
                h, w = overlay.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         a.fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(
                        f"cv2.VideoWriter failed to open '{out_path}' ({w}x{h} @ "
                        f"{a.fps}fps, mp4v). Check the FFMPEG backend / output path."
                    )
            writer.write(overlay)
            if a.png_dir:
                cv2.imwrite(os.path.join(a.png_dir, f"frame_{t:05d}.png"), overlay)
            n += 1
        if writer is not None:
            writer.release()
    print(f"[viz] wrote {n} frames -> {out_path} @ {a.fps}fps "
          f"(horizon={a.horizon}, intrinsics=yam, extrinsics={a.extrinsics_key})")


if __name__ == "__main__":
    main()
