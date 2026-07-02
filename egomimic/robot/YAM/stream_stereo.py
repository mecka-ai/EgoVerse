"""Live-stream the Atlas forward STEREO feed (cam0 + cam1), rectified, in one window.

The Atlas "main" UVC stream ("Altas Nexus2") is a 4000x1200 MJPG frame that packs
the two forward fisheye cameras side by side:

    cam0 (left)  = columns [0:1920]      cam1 (right) = columns [1920:3840]
    (the last 160px are padding)         each 1920x1200, double-sphere model

This tool:
  * auto-discovers the Nexus2 capture node by V4L2 name (robust to renumbering),
  * splits the frame into cam0 / cam1,
  * rectifies each fisheye to a pinhole image using its TRUE double-sphere
    intrinsics (looked up by serial from the embedded calibration sqlite),
  * optionally RE-AIMS the virtual camera with a real rotation (pitch/yaw/roll)
    applied to the output rays before projection — this is the correct way to
    cancel the headset's physical mount tilt (do NOT nudge cx/cy, that warps a
    fisheye and only moves horizontally),
  * optionally row-aligns the pair via the calibrated cam0<->cam1 extrinsic
    (``--stereo-rectify``),
  * center-crops to drop the black fisheye-rectification border,
  * FUSES the two eyes into ONE combined image (both eyes are rectified to the
    same virtual camera, so they overlay directly); pass --side-by-side to see
    them as separate panes (cam0 | cam1) instead.

WHY pitch/yaw and not cx: the rectify maps the output-CENTER ray (0,0,1) to source
pixel (cx,cy) = the true optical axis. Adding a constant to cx just translates the
source-sampling lattice (pans horizontally only, and warps off-axis on a fisheye) —
it can never fix a vertical "up" tilt. A genuine re-aim rotates the ray bundle.

Sign convention (derived from the double-sphere projection):
    +pitch  -> virtual axis aims UP    -> pulls a too-HIGH subject down to center
    -yaw    -> virtual axis aims LEFT  -> pulls a too-LEFT subject right to center
    +roll   -> rotates the image in-plane (horizon leveling)

CALIBRATING the re-aim (the rig tilt is NOT in the calibration DB — cam0 is the
calibration origin, so it must be measured): run ``--calibrate``, put a known
straight-ahead / eye-level reference under the crosshair, nudge pitch/yaw/roll
until it sits at center and level, then press 'k' to save the angles to the rig
config (keyed by serial). Subsequent normal runs auto-load those angles.

Layout from Downloads/split.py; rectify math from cam_stream_test_rectified.py
(DS->pinhole map with FoV validity masking).

Examples:
    # live window, both cameras rectified, auto-loading saved re-aim angles:
    python stream_stereo.py

    # one-time: measure the mount-tilt re-aim interactively and save it:
    python stream_stereo.py --calibrate

    # explicit re-aim (e.g. tilt view up 6deg to cancel a downward mount):
    python stream_stereo.py --pitch-deg 6 --yaw-deg -1.5

    # row-aligned stereo + crop the black border to 85%, half-size window:
    python stream_stereo.py --stereo-rectify --crop 0.85 --scale 0.5 --fps 60

    # no display (headless) — save one rectified stereo frame and exit:
    python stream_stereo.py --snapshot /tmp/stereo_rect.jpg
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import cv2

# Reuse the verified DS calibration loader from yam_cameras (true cx/cy, no hacks).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yam_cameras import (  # noqa: E402
    ATLAS_SERIAL,
    DEFAULT_RIG_CONFIG,
    load_ds_intrinsics,
    load_rig_aim,
    save_rig_aim,
    reaim_rotation,
    build_ds_undistort_map,
    ds_valid_mask,
    stereo_rectify_rotations,
    fuse_eyes,
    edge_crop,
)

MAIN_CAM_NAME = "Nexus2"   # match the invariant token: firmware ships both "Atlas"/"Altas" (sic) spellings
MAIN_W, MAIN_H = 4000, 1200
# Side-by-side split (Downloads/split.py): each forward cam is 1920x1200.
CAM0_COLS = slice(0, 1920)       # left forward fisheye
CAM1_COLS = slice(1920, 3840)    # right forward fisheye
CAM_ROWS = slice(0, 1200)

# NOTE: the rectify/re-aim/fuse/crop pipeline (reaim_rotation, build_ds_undistort_map,
# ds_valid_mask, stereo_rectify_rotations, fuse_eyes, edge_crop, load/save_rig_aim)
# lives in yam_cameras.py as the SINGLE source of truth, so this live viewer and the
# recorded observation (AtlasStereoCamera) produce identical images from rig_aim.json.


# ---------------------------------------------------------------------------
# Device discovery / capture
# ---------------------------------------------------------------------------
def find_node(expected_name=MAIN_CAM_NAME):
    """/dev/video* nodes whose V4L2 name matches `expected_name`, lowest first."""
    out = []
    for path in sorted(glob.glob("/dev/video*"),
                       key=lambda p: int(p.replace("/dev/video", "") or -1)):
        n = os.path.basename(path)
        try:
            name = open(f"/sys/class/video4linux/{n}/name").read().strip()
        except OSError:
            continue
        if expected_name.lower() in name.lower():
            out.append(path)
    return out


def open_main_stream(device=None, fps=30):
    """Open the Nexus2 capture node at 4000x1200 MJPG. Tries candidates until one
    actually delivers a >=3840-wide frame (the metadata node won't)."""
    candidates = [device] if device else (
        [os.getenv("ATLAS_MAIN_NODE")] if os.getenv("ATLAS_MAIN_NODE") else find_node()
    )
    candidates = [c for c in candidates if c]
    if not candidates:
        raise RuntimeError(f"Atlas main node ('{MAIN_CAM_NAME}') not found in /dev/video*")
    last = None
    for dev in candidates:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            last = f"{dev}: could not open (held by another app/PipeWire?)"
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, MAIN_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, MAIN_H)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok, probe = False, None
        for _ in range(15):
            ok, probe = cap.read()
            if ok and probe is not None:
                break
        if ok and probe is not None and probe.shape[1] >= 3840:
            print(f"[stereo] streaming {dev} @ {probe.shape[1]}x{probe.shape[0]} MJPG")
            return cap, dev
        cap.release()
        last = f"{dev}: delivered {None if probe is None else probe.shape} (need >=3840 wide)"
    raise RuntimeError(f"no usable Atlas main node. last: {last}")


def center_crop(img, frac):
    if frac >= 0.999:
        return img
    h, w = img.shape[:2]
    cw, ch = int(round(w * frac)), int(round(h * frac))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def draw_overlay(img, pitch, yaw, roll, calibrating, side_by_side=False):
    """Crosshair + horizon + angle readout on the output image (for aiming)."""
    h, w = img.shape[:2]
    g = (0, 255, 0)
    if side_by_side:
        # per-eye vertical crosshair; red line marks the eye seam at w/2
        for cx in (w // 4, 3 * w // 4):
            cv2.line(img, (cx, 0), (cx, h), g, 1)
        cv2.line(img, (w // 2, 0), (w // 2, h), (0, 0, 255), 1)
    else:
        cv2.line(img, (w // 2, 0), (w // 2, h), g, 1)    # single center crosshair
    cv2.line(img, (0, h // 2), (w, h // 2), g, 1)        # horizon
    txt = f"pitch {pitch:+.2f}  yaw {yaw:+.2f}  roll {roll:+.2f} (deg)"
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
    if calibrating:
        hint = "w/s pitch  a/d yaw  z/x roll  (CAPS=1deg, else 0.25)  r reset  k save  ESC quit"
        cv2.putText(img, hint, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(img, hint, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="explicit /dev/videoN (default: auto-find 'Altas Nexus2')")
    ap.add_argument("--serial", type=int, default=ATLAS_SERIAL,
                    help="Atlas serial for calibration lookup (default %(default)s)")
    ap.add_argument("--crop", type=float, default=1.0,
                    help="center-crop fraction per camera to drop black border "
                         "(1.0=full, e.g. 0.85). Default %(default)s")
    ap.add_argument("--scale", type=float, default=0.0,
                    help="display scale (0=auto-fit to ~1600px wide). 1.0=full res")
    ap.add_argument("--fps", type=int, default=30, help="capture fps (30 or 60)")
    # Re-aim (mount-tilt correction). Default None => load from rig config.
    ap.add_argument("--pitch-deg", type=float, default=None,
                    help="virtual-camera pitch; +up cancels a downward mount tilt "
                         "(default: value saved in --rig-config, else 0)")
    ap.add_argument("--yaw-deg", type=float, default=None,
                    help="virtual-camera yaw; -left (default: rig-config, else 0)")
    ap.add_argument("--roll-deg", type=float, default=None,
                    help="virtual-camera roll for horizon leveling (default: rig-config, else 0)")
    ap.add_argument("--rig-config", default=DEFAULT_RIG_CONFIG,
                    help="JSON of saved re-aim angles per serial (default %(default)s)")
    ap.add_argument("--stereo-rectify", action="store_true",
                    help="row-align the two eyes using the calibrated cam0<->cam1 extrinsic")
    ap.add_argument("--side-by-side", action="store_true",
                    help="show the two eyes as separate panes (cam0 | cam1) instead of one fused image")
    ap.add_argument("--fuse-mode", choices=["cam0", "fill", "blend"], default="cam0",
                    help="how to combine the eyes into one image: 'cam0' (cam0 only — single optical "
                         "center, no ghosting; default), 'fill' (cam0 base, cam1 fills its out-of-FoV "
                         "border) or 'blend' (average the overlap)")
    # ROI crop (px trimmed per edge of the fused image) to isolate the table+arms.
    # Default None => load from rig-config, else 0 (no crop).
    ap.add_argument("--crop-left", type=int, default=None, help="px to trim from the LEFT edge")
    ap.add_argument("--crop-right", type=int, default=None, help="px to trim from the RIGHT edge")
    ap.add_argument("--crop-top", type=int, default=None, help="px to trim from the TOP edge")
    ap.add_argument("--crop-bottom", type=int, default=None, help="px to trim from the BOTTOM edge")
    ap.add_argument("--calibrate", action="store_true",
                    help="interactive: live crosshair; nudge pitch/yaw/roll and press 'k' to save")
    ap.add_argument("--raw", action="store_true",
                    help="show the raw (un-rectified) split instead of rectified")
    ap.add_argument("--snapshot", default=None,
                    help="headless: save ONE concatenated rectified frame to this path and exit")
    args = ap.parse_args()

    # Per-camera DS intrinsics (TRUE values; no cx/cy hacks).
    cam0 = load_ds_intrinsics(args.serial, 0)
    cam1 = load_ds_intrinsics(args.serial, 1)
    for i, c in ((0, cam0), (1, cam1)):
        if (c["width"], c["height"]) != (1920, 1200):
            raise ValueError(f"cam{i} is {c['width']}x{c['height']}, expected 1920x1200 "
                             f"(forward stereo)")

    # Re-aim angles + ROI crop: CLI flag overrides saved rig-config; rig-config overrides default.
    saved = load_rig_aim(args.serial, args.rig_config)
    pitch = saved["pitch_deg"] if args.pitch_deg is None else args.pitch_deg
    yaw = saved["yaw_deg"] if args.yaw_deg is None else args.yaw_deg
    roll = saved["roll_deg"] if args.roll_deg is None else args.roll_deg
    crop_left = saved["crop_left"] if args.crop_left is None else args.crop_left
    crop_right = saved["crop_right"] if args.crop_right is None else args.crop_right
    crop_top = saved["crop_top"] if args.crop_top is None else args.crop_top
    crop_bottom = saved["crop_bottom"] if args.crop_bottom is None else args.crop_bottom

    # Optional per-eye row-alignment rotations (composed with the re-aim below).
    R0_rect, R1_rect = (None, None)
    if args.stereo_rectify:
        R0_rect, R1_rect = stereo_rectify_rotations(args.serial)

    # Build (and rebuild, in --calibrate) the undistort maps + in-FoV masks,
    # using the shared yam_cameras pipeline (single source of truth).
    def build_maps(pitch, yaw, roll):
        R = reaim_rotation(pitch, yaw, roll)
        R0 = R if R0_rect is None else R @ R0_rect
        R1 = R if R1_rect is None else R @ R1_rect
        m0 = build_ds_undistort_map(cam0, R0)
        m1 = build_ds_undistort_map(cam1, R1)
        return m0, m1, ds_valid_mask(m0, cam0), ds_valid_mask(m1, cam1)

    map0, map1, mask0, mask1 = build_maps(pitch, yaw, roll)
    print(f"[stereo] cam0 fx={cam0['fx']:.1f} cx={cam0['cx']:.1f} | "
          f"cam1 fx={cam1['fx']:.1f} cx={cam1['cx']:.1f} | "
          f"re-aim pitch={pitch:+.2f} yaw={yaw:+.2f} roll={roll:+.2f} "
          f"{'+stereo-rectify' if args.stereo_rectify else ''}")

    cap, dev = open_main_stream(args.device, args.fps)

    def process(frame, overlay=False):
        left = frame[CAM_ROWS, CAM0_COLS]
        right = frame[CAM_ROWS, CAM1_COLS]
        if args.raw:                         # raw split can't be fused (not a common frame)
            out = np.hstack([center_crop(left, args.crop), center_crop(right, args.crop)])
        else:
            left = cv2.remap(left, map0[0], map0[1], cv2.INTER_LINEAR)
            right = cv2.remap(right, map1[0], map1[1], cv2.INTER_LINEAR)
            if args.side_by_side:
                out = np.hstack([center_crop(left, args.crop), center_crop(right, args.crop)])
            else:
                # fuse the two eyes -> ROI crop (shared yam_cameras pipeline)
                out = fuse_eyes(left, right, mask0, mask1, args.fuse_mode)
                out = edge_crop(out, crop_left, crop_right, crop_top, crop_bottom)
                out = center_crop(out, args.crop)
        if overlay:
            out = draw_overlay(out, pitch, yaw, roll, args.calibrate,
                               side_by_side=args.side_by_side or args.raw)
        return out

    # Headless snapshot mode (no display) — handy on machines without a GUI.
    if args.snapshot:
        ok, frame = False, None
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None and frame.shape[1] >= 3840:
                break
        cap.release()
        if not ok or frame is None:
            sys.exit("no frame for snapshot")
        out = process(frame)
        cv2.imwrite(args.snapshot, out)
        print(f"[stereo] saved {out.shape[1]}x{out.shape[0]} -> {args.snapshot}")
        return

    win = "Atlas stereo (cam0 | cam1) — rectified | q/ESC to quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    show_overlay = args.calibrate
    t0, frames = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.shape[1] < 3840:
                time.sleep(0.005)
                continue
            out = process(frame, overlay=show_overlay)

            scale = args.scale
            if scale <= 0:                       # auto-fit to ~1600px wide
                scale = min(1.0, 1600.0 / out.shape[1])
            disp = out if scale == 1.0 else cv2.resize(
                out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                print(f"[stereo] {frames / dt:4.1f} fps  out={out.shape[1]}x{out.shape[0]}")
                t0, frames = time.time(), 0

            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):            # q or ESC
                break
            if args.calibrate and key != 255 and 32 <= key < 128:
                ch = chr(key)
                step = 1.0 if ch.isupper() else 0.25   # hold Shift (CAPS) for 1deg nudges
                k = ch.lower()
                changed = True
                if k == "w":
                    pitch += step
                elif k == "s":
                    pitch -= step
                elif k == "a":
                    yaw -= step
                elif k == "d":
                    yaw += step
                elif k == "x":
                    roll += step
                elif k == "z":
                    roll -= step
                elif k == "r":
                    pitch = yaw = roll = 0.0
                elif k == "k":
                    save_rig_aim(args.serial,
                                 {"pitch_deg": pitch, "yaw_deg": yaw, "roll_deg": roll,
                                  "crop_left": crop_left, "crop_right": crop_right,
                                  "crop_top": crop_top, "crop_bottom": crop_bottom},
                                 args.rig_config)
                    changed = False
                else:
                    changed = False
                if changed:
                    map0, map1, mask0, mask1 = build_maps(pitch, yaw, roll)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
