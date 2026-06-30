"""Cameras for the YAM stack — Atlas front cam + RealSense wrists, ARX-style.

A single module for all YAM camera capture (formerly split across atlas_camera.py
and yam_cameras.py). Every camera is an independent **recorder** exposing the same
small contract — modeled on ``egomimic/robot/eva/.../stream_d405.py`` +
``ARXInterface.__create_cam_recorders``:

    get_image()          -> latest BGR uint8 frame (black until ready, never None)
    res                  -> (H, W)
    frame_count / error_count / last_error   -> live stall monitoring
    wait_until_ready(timeout) / close()

Two recorder kinds:
  * ``AtlasFrontCamera`` — the robot's front view (``front_img_1``), the Mecka
    **Atlas** 6-camera fisheye rig's down-looking cam3, rectified (double-sphere ->
    pinhole) and center-cropped. NOT a RealSense.
  * ``RealSenseRecorder`` — one D405/D435i wrist camera.

``create_camera_recorders()`` is the shared factory (the YAM analogue of ARX's
``__create_cam_recorders``): it returns a ``{friendly_name: recorder}`` dict consumed
by both ``YAMInterface`` (rollout) and ``collect_yam_demo`` (via the thin
``YamCameraRig`` aggregator). Individual camera failures are warned-about and skipped
so a session never crashes on a bad camera.

Atlas details (calibration_db README): cam0/1 forward stereo (1920x1200, "Altas
Nexus2" main stream), cam2 Left SLAM, cam3 Bottom SLAM (DOWN), cam4 Right SLAM,
cam5 Top SLAM (all 640x480, "Altas Nexus4" aux stream). Ported from
Downloads/rectify_6cam_videos.py (DS->pinhole map), Downloads/split.py (aux layout),
Downloads/calibration_db (serial-indexed sqlite, schema v1).
"""

import glob
import json
import os
import sqlite3
import threading
import time

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


# =====================================================================
# Atlas front camera (front_img_1): down-looking cam3, rectified + cropped.
# =====================================================================
ATLAS_SERIAL = 2952            # ATLASHX2952 (calibration_db primary key)
ATLAS_DEVICE_ID = "ATLASHX2952"
FRONT_CAM_INDEX = 3            # cam3 = Bottom SLAM = down-looking (the "third camera")
CROP_FRACTION = 0.75           # center-crop the rectified image to 75% of each dim

# Embedded calibration DB (copied from Downloads/calibration_db/out).
_CALIB_DB = os.path.join(os.path.dirname(__file__), "calibration", "calibrations.sqlite")

# Aux stream ("Altas Nexus4"): leftmost data tile + cam2,cam3,cam4,cam5 each 640
# wide. Width 3104 = 544 (data) + 4*640. cam3 is the 2nd of the four camera tiles
# (3rd tile overall counting the data tile) -> the down-looking view.
ATLAS_AUX_NAME = "Nexus4"   # match invariant token: firmware ships both "Atlas"/"Altas" (sic) spellings
AUX_W, AUX_H = 3104, 480
AUX_CAM_W = 640                   # per-camera tile width in the aux stream (cam2..5)
AUX_DATA_W = AUX_W - 4 * AUX_CAM_W  # 544: leftmost non-camera data tile
_CAM3_X0 = AUX_DATA_W + (FRONT_CAM_INDEX - 2) * AUX_CAM_W   # 544 + 640 = 1184
CAM3_COLS = slice(_CAM3_X0, _CAM3_X0 + AUX_CAM_W)           # [1184:1824]
CAM3_ROWS = slice(0, 480)
CAM3_W, CAM3_H = 640, 480

# Main stream ("Altas Nexus2"): 4000x1200 side-by-side forward stereo.
# cam0 (left) = cols [0:1920], cam1 (right) = cols [1920:3840], last 160px padding.
ATLAS_MAIN_NAME = "Nexus2"   # match invariant token (firmware "Atlas"/"Altas" sic)
ATLAS_MAIN_W, ATLAS_MAIN_H = 4000, 1200
STEREO_CAM0_COLS = slice(0, 1920)
STEREO_CAM1_COLS = slice(1920, 3840)
STEREO_CAM_ROWS = slice(0, 1200)
STEREO_CAM_W, STEREO_CAM_H = 1920, 1200
STEREO_CROP_FRACTION = 1.0   # no crop by default; pass crop_frac < 1 to trim black border


# ---------------------------------------------------------------------------
# Calibration lookup (sqlite, double-sphere)
# ---------------------------------------------------------------------------
def load_ds_intrinsics(serial=ATLAS_SERIAL, cam_index=FRONT_CAM_INDEX, db_path=_CALIB_DB):
    """Load one camera's double-sphere intrinsics from calibration_db.

    Returns dict: model='ds', fx, fy, cx, cy, xi, alpha, width, height.
    """
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f"calibration DB not found: {db_path}")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        v = conn.execute("SELECT value FROM _meta WHERE key='schema_version'").fetchone()[0]
        if str(v) != "1":
            raise AssertionError(f"calibrations.sqlite schema_version={v}, expected 1")
        row = conn.execute(
            "SELECT * FROM calibrations WHERE serial=?", (int(serial),)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise ValueError(f"No calibration row for serial {serial} (ATLASHX{serial})")
    i = cam_index
    if row[f"cam{i}_fx"] is None:
        raise ValueError(f"serial {serial} cam{i} has no calibration (status={row['status']})")
    return dict(
        model="ds",
        fx=float(row[f"cam{i}_fx"]), fy=float(row[f"cam{i}_fy"]),
        cx=float(row[f"cam{i}_cx"]), cy=float(row[f"cam{i}_cy"]),
        xi=float(row[f"cam{i}_xi"]), alpha=float(row[f"cam{i}_alpha"]),
        width=int(row[f"cam{i}_w"]), height=int(row[f"cam{i}_h"]),
    )


# ---------------------------------------------------------------------------
# Double-sphere -> pinhole undistort map (ported verbatim from rectify_6cam_videos.py)
# ---------------------------------------------------------------------------
def build_ds_undistort_map(cam, R=None, zoom_out_factor=1.0):
    """Return (map1, map2) for cv2.remap: raw double-sphere frame -> pinhole frame.

    Output is a pinhole image with the camera's native fx,fy and the principal
    point recentered to the image center. ``R`` (3x3) re-aims the virtual camera
    by rotating the OUTPUT rays before the DS forward-projection (identity = look
    straight along the true optical axis). Pixels outside the lens FOV map to -1
    (rendered black by cv2.remap).
    """
    w, h = cam["width"], cam["height"]
    cx_out, cy_out = w / 2.0, h / 2.0
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    fx_v = cam["fx"] * zoom_out_factor
    fy_v = cam["fy"] * zoom_out_factor
    x = (u - cx_out) / fx_v
    y = (v - cy_out) / fy_v
    z = np.ones_like(x)
    if R is not None:
        x, y, z = (R[0, 0] * x + R[0, 1] * y + R[0, 2] * z,
                   R[1, 0] * x + R[1, 1] * y + R[1, 2] * z,
                   R[2, 0] * x + R[2, 1] * y + R[2, 2] * z)
    xi, alpha = cam["xi"], cam["alpha"]
    d1 = np.sqrt(x * x + y * y + z * z)
    zxi = xi * d1 + z
    d2 = np.sqrt(x * x + y * y + zxi * zxi)
    div = alpha * d2 + (1 - alpha) * zxi
    map1 = (cam["fx"] * x / div + cam["cx"]).astype(np.float32)
    map2 = (cam["fy"] * y / div + cam["cy"]).astype(np.float32)
    w1 = alpha / (1 - alpha) if alpha <= 0.5 else (1 - alpha) / alpha
    w2 = (w1 + xi) / np.sqrt(2 * w1 * xi + xi * xi + 1)
    invalid = z <= -w2 * d1
    map1[invalid] = -1.0
    map2[invalid] = -1.0
    return map1, map2


def ds_valid_mask(map_pair, cam):
    """Boolean HxW mask of output pixels that sample INSIDE the source tile.

    Catches both the FoV-invalid (-1) flag AND out-of-frame sampling (the black
    border that appears when the re-aim looks past the sensor edge)."""
    m1, m2 = map_pair
    W, H = cam["width"], cam["height"]
    return (m1 >= 0) & (m1 <= W - 1) & (m2 >= 0) & (m2 <= H - 1)


# ---------------------------------------------------------------------------
# Forward-stereo re-aim + fusion pipeline (SINGLE SOURCE OF TRUTH).
# Shared by the live viewer (stream_stereo.py) AND the recorded observation
# (AtlasStereoCamera) so the saved rig_aim.json propagates identically to
# streaming, data collection, and rollout.
# ---------------------------------------------------------------------------
# Persisted per-serial rig config: re-aim angles (deg) + an ROI crop (px/edge).
DEFAULT_RIG_CONFIG = os.path.join(os.path.dirname(_CALIB_DB), "rig_aim.json")
RIG_DEFAULTS = {"pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0,
                "crop_left": 0, "crop_right": 0, "crop_top": 0, "crop_bottom": 0}
_DEG_KEYS = ("pitch_deg", "yaw_deg", "roll_deg")


def load_rig_aim(serial=ATLAS_SERIAL, path=DEFAULT_RIG_CONFIG):
    """Return the rig config (re-aim angles + ROI crop) for ``serial`` (defaults
    if the file or the serial entry is absent)."""
    out = dict(RIG_DEFAULTS)
    if not path or not os.path.isfile(path):
        return out
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return out
    entry = data.get(str(serial), {})
    for k in RIG_DEFAULTS:
        if k in entry:
            out[k] = float(entry[k]) if k in _DEG_KEYS else int(entry[k])
    return out


def save_rig_aim(serial, cfg, path=DEFAULT_RIG_CONFIG):
    """Merge ``cfg`` (any of the angle/crop keys) for ``serial`` into the JSON."""
    data = {}
    if path and os.path.isfile(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
    entry = dict(data.get(str(serial), {}))      # keep keys not being updated
    for k in RIG_DEFAULTS:
        if k in cfg:
            entry[k] = round(float(cfg[k]), 4) if k in _DEG_KEYS else int(cfg[k])
    data[str(serial)] = entry
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"[cameras] saved rig config for serial {serial} -> {path}: {entry}")
    return entry


def reaim_rotation(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0):
    """Virtual-camera re-aim rotation applied to output rays before projection.

    R = Ry(yaw) @ Rx(pitch) @ Rz(roll). +pitch aims the axis UP, -yaw aims LEFT,
    +roll rolls the image. Identity when all angles are 0."""
    p, y, r = np.deg2rad([pitch_deg, yaw_deg, roll_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def stereo_rectify_rotations(serial=ATLAS_SERIAL, db_path=_CALIB_DB):
    """Symmetric row-alignment rotations (R0, R1) for the forward stereo pair,
    from the calibrated cam0->cam1 relative rotation (split in half so both eyes
    meet at the bisector). ~0.3deg on this rig; compose with the re-aim."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT cam1_qx,cam1_qy,cam1_qz,cam1_qw FROM calibrations WHERE serial=?",
            (int(serial),)).fetchone()
    finally:
        conn.close()
    qx, qy, qz, qw = (row["cam1_qx"], row["cam1_qy"], row["cam1_qz"], row["cam1_qw"])
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R_rel = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    rvec, _ = cv2.Rodrigues(R_rel)
    R_half, _ = cv2.Rodrigues(rvec * 0.5)
    return R_half, R_rel.T @ R_half


def fuse_eyes(left, right, mask0, mask1, mode="fill"):
    """Combine the two rectified eyes (same virtual camera) into ONE image.

    'fill': cam0 base, cam1 fills only cam0's out-of-FoV border -> no parallax
    ghosting. 'blend': average where both eyes are valid (classic fused look)."""
    if mode == "fill":
        out = left.copy()
        holes = mask1 & ~mask0
        out[holes] = right[holes]
        return out
    out = np.zeros_like(left)
    out[mask0] = left[mask0]
    only1 = mask1 & ~mask0
    out[only1] = right[only1]
    both = mask0 & mask1
    out[both] = ((left[both].astype(np.uint16) + right[both]) // 2).astype(np.uint8)
    return out


def edge_crop(img, left=0, right=0, top=0, bottom=0):
    """Trim left/right/top/bottom px to isolate an ROI (e.g. the table+arms)."""
    h, w = img.shape[:2]
    x0 = min(max(left, 0), w - 1)
    x1 = max(x0 + 1, w - max(right, 0))
    y0 = min(max(top, 0), h - 1)
    y1 = max(y0 + 1, h - max(bottom, 0))
    return img[y0:y1, x0:x1]


class StereoFrontProcessor:
    """Rectify + re-aim + fuse + crop the Atlas forward stereo pair into ONE
    pinhole front image (``front_img_1``).

    The single source of truth shared by the live viewer (stream_stereo.py) and
    the recorded observation (AtlasStereoCamera). Loads the per-serial re-aim +
    crop from ``rig_aim.json`` (CLI/ctor args override), so the SAME processing
    reaches streaming, data collection, and rollout. ``intrinsics`` is the 3x4
    pinhole K valid at the cropped output resolution."""

    def __init__(self, serial=ATLAS_SERIAL, rig_config_path=DEFAULT_RIG_CONFIG,
                 pitch_deg=None, yaw_deg=None, roll_deg=None,
                 crop_left=None, crop_right=None, crop_top=None, crop_bottom=None,
                 fuse_mode="fill", stereo_rectify=False):
        if cv2 is None:
            raise ImportError("opencv (cv2) is required for stereo rectification")
        cfg = load_rig_aim(serial, rig_config_path)
        ov = {"pitch_deg": pitch_deg, "yaw_deg": yaw_deg, "roll_deg": roll_deg,
              "crop_left": crop_left, "crop_right": crop_right,
              "crop_top": crop_top, "crop_bottom": crop_bottom}
        for k, v in ov.items():
            if v is not None:
                cfg[k] = v
        self.cfg = cfg
        self.fuse_mode = fuse_mode
        self.stereo_rectify = stereo_rectify

        self.cam0 = load_ds_intrinsics(serial, 0)
        self.cam1 = load_ds_intrinsics(serial, 1)
        for i, c in ((0, self.cam0), (1, self.cam1)):
            if (c["width"], c["height"]) != (STEREO_CAM_W, STEREO_CAM_H):
                raise ValueError(
                    f"cam{i} is {c['width']}x{c['height']}, expected "
                    f"{STEREO_CAM_W}x{STEREO_CAM_H} (forward stereo)")

        R = reaim_rotation(cfg["pitch_deg"], cfg["yaw_deg"], cfg["roll_deg"])
        if stereo_rectify:
            R0r, R1r = stereo_rectify_rotations(serial)
            R0, R1 = R @ R0r, R @ R1r
        else:
            R0 = R1 = R
        self.map0 = build_ds_undistort_map(self.cam0, R0)
        self.map1 = build_ds_undistort_map(self.cam1, R1)
        self.mask0 = ds_valid_mask(self.map0, self.cam0)
        self.mask1 = ds_valid_mask(self.map1, self.cam1)

        w, h = self.cam0["width"], self.cam0["height"]
        cl, cr = int(cfg["crop_left"]), int(cfg["crop_right"])
        ct, cb = int(cfg["crop_top"]), int(cfg["crop_bottom"])
        self.crop = (cl, cr, ct, cb)
        self.out_w = max(1, w - cl - cr)
        self.out_h = max(1, h - ct - cb)
        # Pinhole K after re-aim (principal point at output center) + ROI crop.
        cx = w / 2.0 - cl
        cy = h / 2.0 - ct
        self.intrinsics = np.array([[self.cam0["fx"], 0.0, cx, 0.0],
                                    [0.0, self.cam0["fy"], cy, 0.0],
                                    [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)

    def process_split(self, left, right):
        """left/right = the two raw fisheye tiles (1920x1200 each) -> fused image."""
        left = cv2.remap(left, self.map0[0], self.map0[1], cv2.INTER_LINEAR)
        right = cv2.remap(right, self.map1[0], self.map1[1], cv2.INTER_LINEAR)
        fused = fuse_eyes(left, right, self.mask0, self.mask1, self.fuse_mode)
        return edge_crop(fused, *self.crop)

    def process_frame(self, frame):
        """frame = the full 4000x1200 main-stream frame -> fused front image."""
        left = frame[STEREO_CAM_ROWS, STEREO_CAM0_COLS]
        right = frame[STEREO_CAM_ROWS, STEREO_CAM1_COLS]
        return self.process_split(left, right)


def _crop_geometry(w, h, frac):
    """Center-crop box (x0, y0, cw, ch) for keeping `frac` of each dimension."""
    cw, ch = int(round(w * frac)), int(round(h * frac))
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return x0, y0, cw, ch


def front_output_intrinsics(cam=None, serial=ATLAS_SERIAL, crop_frac=CROP_FRACTION):
    """Pinhole K (3x4) + (out_w, out_h) of the FINAL front image (rectified cam3,
    then center-cropped to `crop_frac`).

    Rectification puts the principal point at the tile center (w/2, h/2) with the
    native fx,fy; the center crop then shifts the principal point by the crop
    offset. Returns the K valid at the cropped output resolution.
    """
    cam = cam or load_ds_intrinsics(serial, FRONT_CAM_INDEX)
    w, h = cam["width"], cam["height"]
    x0, y0, cw, ch = _crop_geometry(w, h, crop_frac)
    cx = w / 2.0 - x0
    cy = h / 2.0 - y0
    K = np.array([[cam["fx"], 0.0, cx, 0.0],
                  [0.0, cam["fy"], cy, 0.0],
                  [0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    return K, (cw, ch)


def find_atlas_node(expected_name=ATLAS_AUX_NAME):
    """Return /dev/video* nodes whose V4L2 name matches the Atlas function,
    lowest number first (the capture node is the first; later ones are metadata)."""
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


def find_atlas_main_node():
    """Return /dev/video* nodes for the Atlas main (stereo) stream, lowest first."""
    return find_atlas_node(ATLAS_MAIN_NAME)


class AtlasFrontCamera(threading.Thread):
    """Streams the Atlas aux stream and crops cam3 (down-looking) -> front_img_1
    (BGR, raw fisheye, 640x480). No rectification. Recorder contract: get_image()
    / res / wait_until_ready() / close(), shared with RealSenseRecorder."""

    NAME = "front_img_1"

    def __init__(self, serial=ATLAS_SERIAL, cam_index=FRONT_CAM_INDEX,
                 crop_frac=CROP_FRACTION, device=None):
        super().__init__(daemon=True)
        if cv2 is None:
            raise ImportError("opencv (cv2) is required for the Atlas front camera")
        self.cam = load_ds_intrinsics(serial, cam_index)
        if (self.cam["width"], self.cam["height"]) != (CAM3_W, CAM3_H):
            raise ValueError(
                f"cam{cam_index} is {self.cam['width']}x{self.cam['height']}, "
                f"expected {CAM3_W}x{CAM3_H} (the aux SLAM cameras)"
            )
        self.out_w, self.out_h = CAM3_W, CAM3_H
        self.intrinsics = None  # raw fisheye — no pinhole intrinsics

        # Device node: explicit arg > ATLAS_VIDEO_NODE env > auto-discovery.
        self.device = device or os.getenv("ATLAS_VIDEO_NODE") or self._resolve_device()
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open Atlas aux node {self.device} "
                f"(is another app/PipeWire holding it?)"
            )
        # MJPG is how the Atlas streams over USB; the default raw format can't
        # negotiate the stitched mode and falls back silently.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, AUX_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, AUX_H)
        # Verify ACTUAL delivered geometry with a warmup read; fail loud rather
        # than silently recording a black/mis-cropped front.
        ok, probe = False, None
        for _ in range(15):
            ok, probe = self.cap.read()
            if ok and probe is not None:
                break
        if not ok or probe is None:
            self.cap.release()
            raise RuntimeError(f"Atlas aux node {self.device} returned no frame at startup")
        if probe.shape[1] < CAM3_COLS.stop or probe.shape[0] < CAM3_H:
            self.cap.release()
            raise RuntimeError(
                f"Atlas aux node {self.device} delivered {probe.shape[1]}x{probe.shape[0]}, "
                f"need >= {CAM3_COLS.stop}x{CAM3_H} to crop cam3. "
                f"Check MJPG / USB3 link / another app holding the camera."
            )

        self.name = self.NAME
        self.latest_frame = None
        self.frame_count = 0
        self.error_count = 0
        self.last_error = None
        self.running = True

    def _resolve_device(self):
        nodes = find_atlas_node()
        if not nodes:
            raise RuntimeError(
                f"Atlas aux node ('{ATLAS_AUX_NAME}') not found in /dev/video*"
            )
        return nodes[0]

    def run(self):
        # Entire frame path inside one try/except so a transient read/decode
        # error can never kill the thread.
        while self.running:
            try:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue
                if frame.shape[1] < CAM3_COLS.stop or frame.shape[0] < CAM3_H:
                    self.error_count += 1
                    self.last_error = f"unexpected aux frame size {frame.shape}"
                    time.sleep(0.005)
                    continue
                cam3 = frame[CAM3_ROWS, CAM3_COLS]                       # 640x480 raw fisheye
                self.latest_frame = np.ascontiguousarray(cam3)
                self.frame_count += 1
            except Exception as e:  # noqa: BLE001 - thread must never die
                self.error_count += 1
                self.last_error = repr(e)
                time.sleep(0.005)
                continue

    def get_frame(self):
        """Latest raw front frame (BGR uint8, 640x480), or black until ready."""
        f = self.latest_frame
        if f is None:
            return np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
        return f

    # Recorder contract (shared with RealSenseRecorder) so a YamCameraRig /
    # YAMInterface can treat every camera — Atlas or RealSense — interchangeably.
    def get_image(self):
        """Alias for get_frame() to match the RealSenseRecorder recorder API."""
        return self.get_frame()

    @property
    def res(self):
        """(H, W) of the produced front image."""
        return (self.out_h, self.out_w)

    def wait_until_ready(self, timeout=10.0):
        """Block until the first frame is produced (or timeout). Never raises."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest_frame is not None:
                return True
            time.sleep(0.05)
        print(f"[cameras] WARNING: Atlas front '{self.NAME}' no frame after {timeout}s "
              f"(last_error={self.last_error}); continuing.")
        return False

    def stop(self):
        self.running = False

    def close(self):
        self.stop()
        self.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


class AtlasStereoCamera(threading.Thread):
    """The Atlas forward stereo pair (cam0 left + cam1 right) -> ONE front image.

    Opens the "Nexus2" V4L2 node at 4000x1200 MJPG, splits into cam0/cam1, and
    runs the shared ``StereoFrontProcessor`` pipeline: double-sphere -> pinhole
    rectify + re-aim (pitch/yaw/roll) + fuse + ROI crop, configured from
    ``rig_aim.json`` (keyed by serial). This is ``front_img_1`` for BOTH rollout
    (YAMInterface.get_obs) and data collection — so the saved rig config
    propagates everywhere through this one recorder.

    Pass ``raw=True`` for the legacy un-rectified side-by-side (cam0 | cam1).

    Recorder contract (shared with AtlasFrontCamera and RealSenseRecorder):
        get_image() -> latest BGR uint8 frame (black until ready, never None)
        res / intrinsics (3x4 pinhole K) / frame_count / error_count / last_error
        wait_until_ready(timeout) / close()
    """

    NAME = "front_img_1"

    def __init__(self, serial=ATLAS_SERIAL, crop_frac=STEREO_CROP_FRACTION,
                 fps=30, device=None, raw=False, rig_config_path=DEFAULT_RIG_CONFIG,
                 fuse_mode="fill", stereo_rectify=False):
        super().__init__(daemon=True)
        if cv2 is None:
            raise ImportError("opencv (cv2) is required for the Atlas stereo camera")

        self.raw = raw
        if raw:
            # Legacy: validate dims, output both eyes side-by-side, no rectify.
            cam0 = load_ds_intrinsics(serial, 0)
            cam1 = load_ds_intrinsics(serial, 1)
            for i, c in ((0, cam0), (1, cam1)):
                if (c["width"], c["height"]) != (STEREO_CAM_W, STEREO_CAM_H):
                    raise ValueError(
                        f"cam{i} is {c['width']}x{c['height']}, "
                        f"expected {STEREO_CAM_W}x{STEREO_CAM_H} (forward stereo)"
                    )
            self.proc = None
            self.out_w, self.out_h = STEREO_CAM_W * 2, STEREO_CAM_H   # 3840x1200
            self.intrinsics = None
        else:
            # Processed: rectify + re-aim + fuse + crop, per rig_aim.json.
            self.proc = StereoFrontProcessor(
                serial=serial, rig_config_path=rig_config_path,
                fuse_mode=fuse_mode, stereo_rectify=stereo_rectify)
            self.out_w, self.out_h = self.proc.out_w, self.proc.out_h
            self.intrinsics = self.proc.intrinsics

        self.device = device or os.getenv("ATLAS_MAIN_NODE") or self._resolve_device()
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"could not open Atlas main node {self.device} "
                f"(is another app/PipeWire holding it?)"
            )
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ATLAS_MAIN_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ATLAS_MAIN_H)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        ok, probe = False, None
        for _ in range(15):
            ok, probe = self.cap.read()
            if ok and probe is not None:
                break
        if not ok or probe is None:
            self.cap.release()
            raise RuntimeError(f"Atlas main node {self.device} returned no frame at startup")
        if probe.shape[1] < 3840 or probe.shape[0] < STEREO_CAM_H:
            self.cap.release()
            raise RuntimeError(
                f"Atlas main node {self.device} delivered {probe.shape[1]}x{probe.shape[0]}, "
                f"need >= 3840x{STEREO_CAM_H}. "
                f"Check MJPG / USB3 link / another app holding the camera."
            )

        self.name = self.NAME
        self.latest_frame = None
        self.frame_count = 0
        self.error_count = 0
        self.last_error = None
        self.running = True

    def _resolve_device(self):
        nodes = find_atlas_main_node()
        if not nodes:
            raise RuntimeError(
                f"Atlas main node ('{ATLAS_MAIN_NAME}') not found in /dev/video*"
            )
        return nodes[0]

    def run(self):
        while self.running:
            try:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue
                if frame.shape[1] < 3840 or frame.shape[0] < STEREO_CAM_H:
                    self.error_count += 1
                    self.last_error = f"unexpected main frame size {frame.shape}"
                    time.sleep(0.005)
                    continue
                if self.raw:
                    left = frame[STEREO_CAM_ROWS, STEREO_CAM0_COLS]
                    right = frame[STEREO_CAM_ROWS, STEREO_CAM1_COLS]
                    out = np.hstack([left, right])
                else:
                    out = self.proc.process_frame(frame)   # rectify+re-aim+fuse+crop
                self.latest_frame = np.ascontiguousarray(out)
                self.frame_count += 1
            except Exception as e:  # noqa: BLE001 - thread must never die
                self.error_count += 1
                self.last_error = repr(e)
                time.sleep(0.005)
                continue

    def get_image(self):
        f = self.latest_frame
        if f is None:
            return np.zeros((self.out_h, self.out_w, 3), dtype=np.uint8)
        return f

    # Alias so both the old (get_frame) and new (get_image) recorder APIs work.
    get_frame = get_image

    @property
    def res(self):
        return (self.out_h, self.out_w)

    def wait_until_ready(self, timeout=10.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest_frame is not None:
                return True
            time.sleep(0.05)
        print(f"[cameras] WARNING: Atlas stereo '{self.NAME}' no frame after {timeout}s "
              f"(last_error={self.last_error}); continuing.")
        return False

    def stop(self):
        self.running = False

    def close(self):
        self.stop()
        self.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


# =====================================================================
# RealSense wrist cameras.
# HARDCODED CAMERA ALIASES  ->  EDIT THIS PER MACHINE.
# Maps each RealSense serial number to the friendly name used as the obs/demo
# image key. front_img_1 comes from the Atlas camera above, NOT a RealSense — so
# only the two wrist D405s are listed here. RealSense cameras whose serial is NOT
# listed fall back to "<model>_<serial>".
# To adapt to another machine: list serials with `rs-enumerate-devices -s` (or
# read.py) and replace the serials below.
# =====================================================================
DEFAULT_CAMERA_NAMES = {
    "353322270967": "left_wrist_img",
    "323622270294": "right_wrist_img",
}

# Per-camera RealSense resolution overrides, keyed by friendly name ->
# (width, height, fps). The D405 wrists are on USB 2.0, so they stay 640x480@30
# (the only mode all wrist cams share at 30fps). Empty = all use 640x480@30.
DEFAULT_CAMERA_RESOLUTIONS = {}


def parse_camera_names(pairs):
    """Parse a list of "SERIAL=NAME" strings into {serial: name}."""
    mapping = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"camera-name expects SERIAL=NAME, got '{pair}'")
        serial, name = pair.split("=", 1)
        mapping[serial.strip()] = name.strip()
    return mapping


class RealSenseRecorder:
    """Streams one RealSense camera (D405/D435i) in the background.

    Recorder contract (shared with AtlasFrontCamera):
        get_image() -> latest BGR uint8 frame (black until ready, never None)
        res         -> (H, W)
        frame_count / error_count / last_error  -> live stall monitoring
        wait_until_ready(timeout) / close()

    Frames are BGR (RealSense native); convert to RGB at use. No spatial flip:
    the native orientation is already correct (verified — np.fliplr mirrored it).

    Default 640x480@30 is the only mode all wrist cams share at 30fps on USB 2.0.
    Wider 16:9 modes (848x480, 1280x720) need USB 3.0.
    """

    def __init__(self, serial, name, width=640, height=480, fps=30):
        if rs is None:
            raise ImportError("pyrealsense2 is not installed; cannot capture cameras.")
        self.serial = serial
        self.name = name
        self.width = width
        self.height = height
        self.fps = fps

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline.start(cfg)  # may raise; caller (factory) handles/skips

        self.latest_frame = None
        self.frame_count = 0   # monotonically increases on every new frame
        self.error_count = 0   # transient errors swallowed by the capture loop
        self.last_error = None
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def res(self):
        return (self.height, self.width)

    def _run(self):
        # The ENTIRE frame path is inside one try/except Exception so a transient
        # decode error (or any non-RuntimeError) can never kill this daemon thread
        # and freeze latest_frame into a permanent single-camera stall.
        while self.running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=200)
                color = frames.get_color_frame()
                if color:
                    self.latest_frame = np.asanyarray(color.get_data()).copy()
                    self.frame_count += 1
            except Exception as e:  # noqa: BLE001 - thread must never die
                self.error_count += 1
                self.last_error = repr(e)
                continue

    def get_image(self):
        """Latest BGR uint8 frame, or black (zeros) until the first frame lands."""
        f = self.latest_frame
        if f is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return f

    def wait_until_ready(self, timeout=10.0):
        """Block until the first frame arrives (or timeout). Never raises."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.latest_frame is not None:
                return True
            time.sleep(0.05)
        print(f"[cameras] WARNING: '{self.name}' ({self.serial}) no frame after "
              f"{timeout}s; continuing (black frames until it recovers).")
        return False

    def close(self):
        self.running = False
        try:
            self._thread.join(timeout=1.0)
        except BaseException:
            pass
        try:
            self._pipeline.stop()
        except BaseException:
            pass


# ---------------------------------------------------------------------------
# Discovery + the shared "create cam recorders" factory (cf. ARX)
# ---------------------------------------------------------------------------
def discover_realsense_recorders(serial_to_name=None, width=640, height=480,
                                 fps=30, resolutions=None):
    """Enumerate connected RealSense cameras -> {name: RealSenseRecorder}.

    Auto-discovery is the YAM convention (ARX instead lists explicit serials in a
    config). Cameras not in ``serial_to_name`` fall back to ``<model>_<serial>``.
    A camera that fails to start is warned-about and skipped (never raises).
    """
    if rs is None:
        raise ImportError("pyrealsense2 is not installed; cannot capture cameras.")
    serial_to_name = serial_to_name or {}
    resolutions = dict(resolutions) if resolutions is not None else dict(DEFAULT_CAMERA_RESOLUTIONS)

    recorders = {}
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"[cameras] Found {len(devices)} RealSense device(s).")
    for dev in devices:
        model = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        if "D405" in model:
            label = "D405"
        elif "D435" in model:
            label = "D435i"
        else:
            print(f"[cameras] Skipping unsupported device {model} ({serial}).")
            continue
        name = serial_to_name.get(serial, f"{label}_{serial}")
        w, h, f = resolutions.get(name, (width, height, fps))
        try:
            recorders[name] = RealSenseRecorder(serial, name, w, h, f)
            print(f"[cameras] Streaming {name} ({serial}) @ {w}x{h}@{f}.")
        except Exception as e:  # noqa: BLE001 - warn + skip, never crash a session
            print(f"[cameras] WARNING: failed to start {name} ({serial}) @ {w}x{h}@{f}: {e}")
    return recorders


def create_camera_recorders(cameras_cfg=None, wrist_serial_to_name=None,
                            use_front=True, front_serial=None, front_raw=False):
    """Build a ``{friendly_name: recorder}`` dict — the YAM analogue of
    ``ARXInterface.__create_cam_recorders``.

    Two modes:
      * ``cameras_cfg`` (ARX-style dict ``{name: {type, enabled, ...}}``): explicit
        per-camera config. ``type`` is "d405"/"d435"/"realsense" (needs
        ``serial_number``) or "atlas" (the front cam); an atlas entry may set
        ``raw: true`` for the un-rectified side-by-side fisheye pair.
      * otherwise auto-discovery: the Atlas front cam (``front_img_1``) plus every
        connected D405 wrist (``wrist_serial_to_name`` -> friendly names).

    ``front_raw`` (auto-discovery only): if True, build the Atlas front camera in
    RAW mode — the un-rectified side-by-side fisheye pair (cam0 | cam1, 3840x1200),
    with NO double-sphere->pinhole rectify / re-aim / fuse / crop. Use this for
    (intrinsic) camera calibration, which needs the distorted source frames; leave
    it False for normal data collection and rollout (rectified ``front_img_1``).

    Camera failures are warned-about and skipped — a bad camera never crashes the
    session (matches the "warn, continue" requirement). Callers that REQUIRE
    cameras (e.g. YAMInterface for rollout) check the returned dict for emptiness.
    """
    recorders = {}

    if cameras_cfg:
        for name, cc in cameras_cfg.items():
            if not cc.get("enabled", True):
                continue
            cam_type = cc["type"]
            try:
                if cam_type in ("d405", "d435", "realsense"):
                    recorders[name] = RealSenseRecorder(
                        str(cc["serial_number"]), name,
                        cc.get("width", 640), cc.get("height", 480), cc.get("fps", 30),
                    )
                elif cam_type in ("atlas", "atlas_stereo"):
                    rec = AtlasStereoCamera(serial=cc.get("serial", ATLAS_SERIAL),
                                            crop_frac=cc.get("crop_frac", STEREO_CROP_FRACTION),
                                            raw=cc.get("raw", False))
                    rec.start()
                    recorders[name] = rec
                elif cam_type == "atlas_slam":
                    rec = AtlasFrontCamera(serial=cc.get("serial", ATLAS_SERIAL))
                    rec.start()
                    recorders[name] = rec
                else:
                    raise ValueError(f"Unknown camera type '{cam_type}' for '{name}'")
            except Exception as e:  # noqa: BLE001
                print(f"[cameras] WARNING: failed to create '{name}' ({cam_type}): {e}")
        return recorders

    # Auto-discovery path.
    if use_front:
        try:
            front = AtlasStereoCamera(serial=front_serial or ATLAS_SERIAL, raw=front_raw)
            front.start()
            recorders[front.NAME] = front
            if front_raw:
                print("[cameras] Atlas front in RAW mode "
                      "(un-rectified side-by-side fisheye; for calibration).")
        except Exception as e:  # noqa: BLE001
            print(f"[cameras] WARNING: Atlas stereo camera unavailable ({e}); "
                  f"continuing without front_img_1.")
    recorders.update(
        discover_realsense_recorders(wrist_serial_to_name or DEFAULT_CAMERA_NAMES)
    )
    return recorders


# ---------------------------------------------------------------------------
# Thin aggregator over the recorders (for collect_yam_demo's live monitor).
# ---------------------------------------------------------------------------
class YamCameraRig:
    """Full YAM camera rig: Atlas (rectified) front + RealSense D405 wrists.

    A thin aggregator over a ``{name: recorder}`` dict built by
    ``create_camera_recorders``. Exposes the batch helpers collect_yam_demo uses
    (get_frames / wait_until_ready / frame_counts / error_info / camera_res /
    close). ``YAMInterface`` instead iterates the ``recorders`` dict directly,
    ARX-style.
    """

    def __init__(self, wrist_serial_to_name=None, use_front=True, front_serial=None,
                 cameras_cfg=None, recorders=None):
        # Either WRAP an existing {name: recorder} dict (e.g. a YAMInterface's
        # ``recorders`` — then this rig is a monitoring/aggregation VIEW and does
        # NOT own or close them), or BUILD & own a fresh set.
        if recorders is not None:
            self.recorders = recorders
            self._owns = False
        else:
            self.recorders = create_camera_recorders(
                cameras_cfg=cameras_cfg,
                wrist_serial_to_name=wrist_serial_to_name,
                use_front=use_front,
                front_serial=front_serial,
            )
            self._owns = True
        # Keep a handle to the Atlas front recorder for its rectified intrinsics.
        self.front = next(
            (r for r in self.recorders.values() if getattr(r, "NAME", None) == "front_img_1"),
            None,
        )

    @property
    def front_intrinsics(self):
        """Rectified-pinhole K of the Atlas front camera (None if no front)."""
        return None if self.front is None else self.front.intrinsics

    def wait_until_ready(self, timeout=10.0):
        print("[cameras] Waiting for first frames ...")
        for rec in self.recorders.values():
            waiter = getattr(rec, "wait_until_ready", None)
            if callable(waiter):
                waiter(timeout)

    def get_frames(self):
        """Return {name: BGR uint8 frame} for all cameras (black if not ready)."""
        return {name: rec.get_image() for name, rec in self.recorders.items()}

    def frame_counts(self):
        """{name: frames_captured_so_far} for live FPS / stall monitoring."""
        return {name: rec.frame_count for name, rec in self.recorders.items()}

    def error_info(self):
        """{name: (error_count, last_error)} for stall diagnosis."""
        return {name: (rec.error_count, rec.last_error) for name, rec in self.recorders.items()}

    @property
    def camera_res(self):
        """{name: (H, W)}."""
        return {name: rec.res for name, rec in self.recorders.items()}

    def close(self):
        if not self._owns:
            return  # recorders owned elsewhere (e.g. the YAMInterface) — leave them.
        for rec in self.recorders.values():
            try:
                rec.close()
            except BaseException:
                pass


if __name__ == "__main__":
    # Quick Atlas front-cam check.
    cam = AtlasFrontCamera()
    print("Atlas aux node:", cam.device, "| cam3 tile cols:", CAM3_COLS)
    print("cam3 DS intrinsics:", cam.cam)
    print(f"output {cam.out_w}x{cam.out_h}  rectified+{int(CROP_FRACTION*100)}%crop K:\n",
          cam.intrinsics)
    cam.close()
