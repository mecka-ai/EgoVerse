#!/usr/bin/env python3
"""Overlay EE frames + the FUTURE action chunk from a zarr episode onto front_1.

Reads the EVA/YAM zarr layout (images.* as compressed bytes, left/right
.obs_ee_pose / .cmd_ee_pose as xyz + quat(wxyz) with the R_t_e relabel baked
in), undoes the R_t_e relabel to recover the NATIVE EE frame, maps base->cam
with the chosen extrinsics and projects with the chosen intrinsics.

Per frame it draws:
  * the current obs EE frame (axes + white origin dot) per arm,
  * the future cmd_ee_pose chunk (next --horizon steps) as trajectory dots
    fading light->dark along time — mirroring the rollout's action-overlay
    (default horizon 45 = yam_rollout.DEFAULT_RESAMPLE_LENGTH),
  * the two wrist cameras flanking the front view
    [ left_wrist | front (+overlay) | right_wrist ] (--no-wrists to disable).

RUN (YAM episode):
    python debug/eva_replay/viz_eva_zarr.py --zarr zarr/yam_pick_hat/<ep>.zarr \
        --arm both --extrinsics-key yam --intrinsics-key yam
    # options: --every-k 40  --horizon 45  --axis-len 0.06  --no-wrists
    #          --out-dir /tmp/eva_viz  --video /tmp/eva_viz.mp4  --raw-relabelled
"""
import argparse
import io
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
import zarr  # noqa: E402

from egomimic.utils.egomimicUtils import INTRINSICS, EXTRINSICS  # noqa: E402

R_t_e = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
_X, _Y, _Z = (0, 0, 255), (0, 255, 0), (255, 0, 0)   # BGR


def decode(b):
    # zarr stores JPEG bytes wrapped in (possibly nested) 0-d object arrays
    while isinstance(b, np.ndarray):
        b = b.item() if b.ndim == 0 else b.tobytes()
    return cv2.cvtColor(np.asarray(Image.open(io.BytesIO(bytes(b))).convert("RGB")),
                        cv2.COLOR_RGB2BGR)


def project(K33, Tbc, pts_base):
    out = []
    for P in pts_base:
        c = (Tbc @ np.array([*P, 1.0]))[:3]
        if c[2] <= 1e-6:
            out.append(None); continue
        uv = K33 @ c
        out.append((int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))))
    return out


def draw_frame(img, K33, Tbc, p, Rm, L, label):
    px = project(K33, Tbc, [p, p + Rm[:, 0] * L, p + Rm[:, 1] * L, p + Rm[:, 2] * L])
    if px[0] is None:
        return
    for k, col in ((1, _X), (2, _Y), (3, _Z)):
        if px[k] is not None:
            cv2.arrowedLine(img, px[0], px[k], col, 2, tipLength=0.25)
    cv2.circle(img, px[0], 5, (255, 255, 255), 2)
    cv2.putText(img, label, (px[0][0] + 6, px[0][1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_chunk(img, K33, Tbc, xyz_chunk, arm):
    """Future cmd trajectory as dots fading light->dark along time (rollout's
    'Purples'-style overlay). xyz_chunk: (H,3) base-frame positions."""
    n = len(xyz_chunk)
    if n == 0:
        return
    px = project(K33, Tbc, list(xyz_chunk))
    # light -> dark purple (BGR); right arm slightly bluer to tell arms apart
    lo = np.array([246, 232, 252], float)
    hi = np.array([124, 52, 106], float) if arm == "left" else np.array([160, 60, 60], float)
    for i, pt in enumerate(px):
        if pt is None:
            continue
        c = lo + (hi - lo) * (i / max(1, n - 1))
        cv2.circle(img, pt, 3, tuple(int(v) for v in c), -1)


def wrist_panel(z, key, t, target_h):
    """Wrist frame t resized to target_h; None if the episode lacks that camera."""
    if key not in z:
        return None
    img = decode(z[key][t])
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * target_h / h)))
    img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    cv2.putText(img, key.split(".")[-1], (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
    return img


def draw_gizmo(img, Rm, anchor, L_px, label):
    """Orientation-only fallback (old episodes store zero xyz): orthographic
    gizmo of the BASE-frame rotation, viewed from behind the robot looking
    along +x_base (image right = robot right = -y_base, image up = +z_base)."""
    ax, ay = anchor
    for k, col in ((0, _X), (1, _Y), (2, _Z)):
        v = Rm[:, k]
        tip = (int(round(ax - v[1] * L_px)), int(round(ay - v[2] * L_px)))
        cv2.arrowedLine(img, (ax, ay), tip, col, 2, tipLength=0.25)
    cv2.circle(img, (ax, ay), 4, (255, 255, 255), 1)
    cv2.putText(img, label, (ax - L_px, ay + L_px + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--arm", default="both", choices=["both", "left", "right"])
    ap.add_argument("--extrinsics-key", default="x5Dec13_2")
    ap.add_argument("--intrinsics-key", default="eva")
    ap.add_argument("--every-k", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=45,
                    help="future cmd_ee_pose steps to draw (45 = rollout's "
                         "DEFAULT_RESAMPLE_LENGTH)")
    ap.add_argument("--axis-len", type=float, default=0.06)
    ap.add_argument("--wrists", default=True, action=argparse.BooleanOptionalAction,
                    help="show wrist cameras flanking the front view (--no-wrists off)")
    ap.add_argument("--out-dir", default="/tmp/eva_viz")
    ap.add_argument("--video", default=None, help="also write an mp4 here")
    ap.add_argument("--raw-relabelled", action="store_true",
                    help="draw the R_t_e-relabelled rotation as stored (skip undo)")
    args = ap.parse_args()

    arms = ["left", "right"] if args.arm == "both" else [args.arm]
    os.makedirs(args.out_dir, exist_ok=True)
    K = np.asarray(INTRINSICS[args.intrinsics_key], float)
    K33 = np.array([[K[0, 0], 0, K[0, 2]], [0, K[1, 1], K[1, 2]], [0, 0, 1]], float)
    T_cb = {a: np.asarray(EXTRINSICS[args.extrinsics_key][a], float) for a in arms}

    z = zarr.open(args.zarr, mode="r")
    imgs = z["images.front_1"]
    poses = {a: np.asarray(z[f"{a}.obs_ee_pose"]) for a in arms}
    cmds = {a: (np.asarray(z[f"{a}.cmd_ee_pose"]) if f"{a}.cmd_ee_pose" in z else None)
            for a in arms}
    T = imgs.shape[0]
    writer = None
    tiles = []
    for t in range(0, T, args.every_k):
        img = decode(imgs[t])
        for a in arms:
            Tbc = np.linalg.inv(T_cb[a])
            # Future action chunk (cmd_ee_pose[t : t+H]), rollout-style dots.
            if cmds[a] is not None and args.horizon > 0:
                chunk = cmds[a][t:t + args.horizon]
                chunk = chunk[np.linalg.norm(chunk[:, :3], axis=1) > 1e-9]
                draw_chunk(img, K33, Tbc, chunk[:, :3], a)
            q = poses[a][t]
            if np.linalg.norm(q[3:7]) < 0.5:
                continue
            Rm = R.from_quat(q[[4, 5, 6, 3]]).as_matrix()   # wxyz -> xyzw
            if not args.raw_relabelled:
                Rm = R_t_e.T @ Rm                           # undo relabel -> native EE frame
            if np.linalg.norm(q[:3]) < 1e-9:
                # Old processed_v3 episodes store ZERO xyz (orientation only):
                # fall back to a fixed-anchor orientation gizmo in BASE frame.
                ax = 90 if a == "left" else img.shape[1] - 90
                draw_gizmo(img, Rm, (ax, 100), 50, f"{a} t{t} (orient only)")
            else:
                draw_frame(img, K33, Tbc, q[:3], Rm,
                           args.axis_len, f"{a[0].upper()} t{t}")
        cv2.putText(img, "RED=x GREEN=y BLUE=z | dots=cmd chunk (light=now dark=+H)"
                    + ("  [RAW relabelled]" if args.raw_relabelled else "  [native EE frame]"),
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        if args.wrists:
            h = img.shape[0]
            lw = wrist_panel(z, "images.left_wrist", t, h)
            rw = wrist_panel(z, "images.right_wrist", t, h)
            img = np.hstack([p for p in (lw, img, rw) if p is not None])
        cv2.imwrite(os.path.join(args.out_dir, f"frame_{t:05d}.png"), img)
        tiles.append(img)
        if args.video:
            if writer is None:
                writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                         10, (img.shape[1], img.shape[0]))
            writer.write(img)
    if writer is not None:
        writer.release()
    # contact sheet of up to 9 evenly spaced tiles
    if tiles:
        pick = [tiles[i] for i in np.linspace(0, len(tiles) - 1, min(9, len(tiles))).astype(int)]
        tw = 640
        pick = [cv2.resize(t, (tw, max(1, int(round(t.shape[0] * tw / t.shape[1]))))) for t in pick]
        rows = [np.hstack(pick[i:i + 3]) for i in range(0, len(pick), 3)]
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        cv2.imwrite(os.path.join(args.out_dir, "contact_sheet.png"), np.vstack(rows))
    print(f"[viz_eva_zarr] {len(tiles)} frames -> {args.out_dir}"
          + (f" + {args.video}" if args.video else ""))
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()
