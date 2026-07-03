#!/usr/bin/env python3
"""Inspect a YAM demo HDF5 and VERIFY the camera streams were written correctly.

This reads the EgoVerse-format demos produced by collect_yam_demo.py:
    observations/images/<cam_name>   (T, H, W, 3) uint8  (stored RGB)
    observations/joints | joint_positions | eepose   (T, 14) float64
    actions/joints | eepose          (T, 14) float64
    action                           (T, 14) float64

Per-camera health checks (catch the "stalled camera" failure mode):
  * all-black frames      -> camera never delivered (got the black fallback)
  * longest identical run -> a stalled/frozen stream (the stuck-frame signature)
  * unique-frame fraction -> low means the stream repeated frames
  * brightness range      -> flat/exposure problems

Usage:
    python read.py [demo_path] [--dump-frames] [--out DIR]
Defaults to ./demos/demo_0.hdf5. Exits nonzero if any camera looks broken,
so it can be used in a quick "is this episode good?" check.
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

# consecutive-identical-frame run that we consider a stall (matches the
# collect_yam_demo save-time guard STUCK_FRAME_THRESHOLD).
STUCK_RUN_THRESHOLD = 100


def _frame_checksums(ds):
    """Per-frame fingerprint (int sum) + brightness, streamed frame-by-frame
    so we never load the whole (potentially multi-GB) image stack into RAM."""
    T = ds.shape[0]
    checksums = np.empty(T, dtype=np.int64)
    means = np.empty(T, dtype=np.float64)
    n_black = 0
    for i in range(T):
        frame = ds[i]
        checksums[i] = int(frame.sum())
        means[i] = float(frame.mean())
        if frame.max() == 0:
            n_black += 1
    return checksums, means, n_black


def _longest_identical_run(checksums):
    longest = run = 1 if len(checksums) else 0
    start = best_start = 0
    for i in range(1, len(checksums)):
        if checksums[i] == checksums[i - 1]:
            run += 1
            if run > longest:
                longest, best_start = run, start
        else:
            run = 1
            start = i
    return longest, best_start


def check_cameras(f, dump_frames=False, out_dir=Path(".")):
    imgs = f.get("observations/images")
    if imgs is None:
        print("[cameras] No observations/images group found!")
        return False
    all_ok = True
    print("\n=== CAMERA HEALTH ===")
    for cam in imgs:
        ds = imgs[cam]
        T = ds.shape[0]
        checksums, means, n_black = _frame_checksums(ds)
        n_unique = len(np.unique(checksums))
        longest_run, run_start = _longest_identical_run(checksums)
        uniq_frac = n_unique / T if T else 0.0

        problems = []
        if n_black == T:
            problems.append("ALL-BLACK (camera never delivered a frame)")
        elif n_black:
            problems.append(f"{n_black}/{T} black frames")
        if longest_run >= STUCK_RUN_THRESHOLD:
            problems.append(f"stalled: {longest_run} identical frames in a row (from idx {run_start})")
        if uniq_frac < 0.5:
            problems.append(f"low variety: only {uniq_frac:.0%} unique frames")

        status = "OK" if not problems else "BROKEN"
        if problems:
            all_ok = False
        print(f"  [{status}] {cam}: T={T} {ds.shape[1:]} | "
              f"unique={n_unique}/{T} ({uniq_frac:.0%}) | longest_identical_run={longest_run} | "
              f"brightness {means.min():.1f}-{means.max():.1f}")
        for p in problems:
            print(f"           - {p}")

        if dump_frames:
            from PIL import Image  # frames are stored RGB -> save directly
            out_dir.mkdir(parents=True, exist_ok=True)
            for tag, idx in (("first", 0), ("mid", T // 2), ("last", T - 1)):
                Image.fromarray(ds[idx]).save(out_dir / f"{cam}_{tag}.jpg")
            print(f"           dumped {cam}_first/mid/last.jpg to {out_dir}/")
    return all_ok


def show_lowdim(f):
    print("\n=== LOW-DIM (sample = frame 0) ===")
    for key in ("observations/joints", "observations/eepose",
                "actions/joints", "actions/eepose", "action"):
        ds = f.get(key)
        if ds is None:
            continue
        arr = ds[:]
        print(f"  {key:24s} shape={arr.shape}  row0={np.round(arr[0], 3).tolist()}")
    # gripper sanity: cols 6 and 13 should sit in [0, 1]
    j = f.get("observations/joints")
    if j is not None and j.shape[1] >= 14:
        g = j[:, [6, 13]]
        print(f"  gripper range L/R: [{g[:,0].min():.2f},{g[:,0].max():.2f}] "
              f"[{g[:,1].min():.2f},{g[:,1].max():.2f}]  (expect within [0,1])")


def main():
    ap = argparse.ArgumentParser(description="Inspect/verify a YAM demo HDF5.")
    ap.add_argument("demo_path", nargs="?", default="./demos/demo_0.hdf5")
    ap.add_argument("--dump-frames", action="store_true",
                    help="Save first/mid/last frame of each camera as JPG for visual inspection.")
    ap.add_argument("--out", default=".", help="Directory for dumped frames.")
    args = ap.parse_args()

    path = Path(args.demo_path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(2)

    with h5py.File(str(path), "r") as f:
        print(f"=== {path} (sim={dict(f.attrs).get('sim')}) ===")
        print("Datasets:")

        def _print_ds(name, obj):
            # NOTE: must return None — visititems STOPS if the callback returns
            # anything truthy/non-None (a bare `cond and print(...)` returns
            # False for groups and halts traversal after the first dataset).
            if isinstance(obj, h5py.Dataset):
                print(f"  {name:42s} shape={obj.shape} dtype={obj.dtype}")

        f.visititems(_print_ds)
        cams_ok = check_cameras(f, dump_frames=args.dump_frames, out_dir=Path(args.out))
        show_lowdim(f)

    print("\n=== VERDICT ===")
    if cams_ok:
        print("  All cameras look healthy. ✅")
        sys.exit(0)
    else:
        print("  One or more cameras are BROKEN (see above). ❌  Re-collect this episode.")
        sys.exit(1)


if __name__ == "__main__":
    main()
