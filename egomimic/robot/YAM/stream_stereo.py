"""Live-stream the Atlas forward STEREO feed (cam0 + cam1), rectified, in one window.

The Atlas "main" UVC stream ("Altas Nexus2") is a 4000x1200 MJPG frame that packs
the two forward fisheye cameras side by side:

    cam0 (left)  = columns [0:1920]      cam1 (right) = columns [1920:3840]
    (the last 160px are padding)         each 1920x1200, double-sphere model

This tool:
  * auto-discovers the Nexus2 capture node by V4L2 name (robust to renumbering),
  * splits the frame into cam0 / cam1,
  * rectifies each to a pinhole image using its double-sphere intrinsics
    (looked up by serial from the embedded calibration_db sqlite),
  * optionally center-crops each to drop the black fisheye-rectification border,
  * concatenates them into ONE image (cam0 | cam1) and shows it live.

Layout from Downloads/split.py; rectify math from cam_stream_test_rectified.py
(DS->pinhole map with FoV validity masking and zoom_out_factor hook).

Examples:
    # live window, both cameras rectified, full size:
    python stream_stereo.py

    # crop the black border to 85%, half-size window, 60 fps capture:
    python stream_stereo.py --crop 0.85 --scale 0.5 --fps 60

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

# Reuse the verified DS calibration loader from yam_cameras.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yam_cameras import (  # noqa: E402
    ATLAS_SERIAL,
    load_ds_intrinsics,
)


def build_undistort_map(cam, zoom_out_factor=1.0):
    """Return (map1, map2) for cv2.remap that turns a raw fisheye frame into pinhole.

    Supports 'kb4' (OpenCV fisheye) and 'ds' (Double Sphere) models.
    Pixels outside the lens FoV are set to -1 so cv2.remap renders them black.
    zoom_out_factor < 1.0 zooms out the output (show more of the scene, smaller).
    """
    w, h = cam["width"], cam["height"]
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], np.float64)

    if cam["model"] == "kb4":
        D = np.array([cam["k1"], cam["k2"], cam["k3"], cam["k4"]], np.float64)
        return cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_32FC1)

    # Double Sphere -> pinhole
    cx_out, cy_out = w / 2.0, h / 2.0
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    fx_virtual = cam["fx"] * zoom_out_factor
    fy_virtual = cam["fy"] * zoom_out_factor

    mx = (u - cx_out) / fx_virtual
    my = (v - cy_out) / fy_virtual
    x, y, z = mx, my, np.ones_like(mx)
    xi, alpha = cam["xi"], cam["alpha"]
    d1 = np.sqrt(x * x + y * y + z * z)
    zxi = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + zxi * zxi)
    div = alpha * d2 + (1 - alpha) * zxi
    map1 = (cam["fx"] * x / div + cam["cx"]).astype(np.float32)
    map2 = (cam["fy"] * y / div + cam["cy"]).astype(np.float32)
    # Mark pixels outside the lens FoV as invalid (-> black after remap).
    w1 = alpha / (1 - alpha) if alpha <= 0.5 else (1 - alpha) / alpha
    w2 = w1 + xi / np.sqrt(2 * w1 * xi + xi * xi + 1)
    invalid = z <= -w2 * d1
    map1[invalid] = -1.0
    map2[invalid] = -1.0
    return map1, map2

MAIN_CAM_NAME = "Altas Nexus2"   # firmware spells it "Altas" (sic); matched case-insensitively
MAIN_W, MAIN_H = 4000, 1200
# Side-by-side split (Downloads/split.py): each forward cam is 1920x1200.
CAM0_COLS = slice(0, 1920)       # left forward fisheye
CAM1_COLS = slice(1920, 3840)    # right forward fisheye
CAM_ROWS = slice(0, 1200)


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
    ap.add_argument("--raw", action="store_true",
                    help="show the raw (un-rectified) split instead of rectified")
    ap.add_argument("--snapshot", default=None,
                    help="headless: save ONE concatenated rectified frame to this path and exit")
    args = ap.parse_args()

    # Per-camera DS intrinsics + undistort maps (cam0 left, cam1 right).
    cam0 = load_ds_intrinsics(args.serial, 0)
    cam1 = load_ds_intrinsics(args.serial, 1)
    cam0["cx"] += 150
    cam1["cx"] += 150
    for i, c in ((0, cam0), (1, cam1)):
        if (c["width"], c["height"]) != (1920, 1200):
            raise ValueError(f"cam{i} is {c['width']}x{c['height']}, expected 1920x1200 "
                             f"(forward stereo)")
    map0 = build_undistort_map(cam0)
    map1 = build_undistort_map(cam1)
    print(f"[stereo] cam0 fx={cam0['fx']:.1f} cx={cam0['cx']:.1f} | "
          f"cam1 fx={cam1['fx']:.1f} cx={cam1['cx']:.1f}")

    cap, dev = open_main_stream(args.device, args.fps)

    def process(frame):
        left = frame[CAM_ROWS, CAM0_COLS]
        right = frame[CAM_ROWS, CAM1_COLS]
        if not args.raw:
            left = cv2.remap(left, map0[0], map0[1], cv2.INTER_LINEAR)
            right = cv2.remap(right, map1[0], map1[1], cv2.INTER_LINEAR)
        left = center_crop(left, args.crop)
        right = center_crop(right, args.crop)
        return np.hstack([left, right])  # one image: cam0 | cam1

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
    t0, frames = time.time(), 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None or frame.shape[1] < 3840:
                time.sleep(0.005)
                continue
            out = process(frame)

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
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
