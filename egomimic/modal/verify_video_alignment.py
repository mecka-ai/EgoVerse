"""Verify a WAM val mp4 frame-by-frame against the ORIGINAL zarr episode.

Statistical uniformity checks (inter-frame diff profiles) can tell you the
spacing looks even; they cannot tell you WHICH source frame each video frame
actually is. This does that directly: it fingerprints every raw frame of the
episode and every frame of the mp4, matches each video frame to its best source
index, and asserts the resulting sequence is strictly increasing with the
expected stride.

Intended target: the GT `validation_video_*.mp4` (its frames ARE source frames,
so matches should be near-exact). Run it on `predicted_video_*.mp4` only to see
how far the model's imagined frames drift from the GT timeline -- a predicted
frame is not expected to match its source exactly.

    MODAL_ENVIRONMENT=robotics modal run egomimic/modal/verify_video_alignment.py::verify \
        --mp4 "pksnack_wam_debug/stridefix/videos/epoch_3/MECKA_BIMANUAL/validation_video_0.mp4" \
        --episode 69b22fc5f4f4e149281a6635 --stride 6

Prints one line per video frame (`video[k] -> source[i]  delta`) plus a verdict.
Exits non-zero if the sequence is not monotonic or the stride is wrong.
"""

import sys
from pathlib import Path

import modal

# modal_setup.py sits next to this file locally (egomimic/modal/) and is baked
# into the image at /root/, so it imports before the repo is cloned. The package
# path egomimic.modal.* is NOT importable under `modal run`.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modal_setup import (  # noqa: E402
    CFG,
    image,
    training_outputs_volume,
    zarr_volume,
)

app = modal.App("egomimic-verify-video-alignment", image=image)


@app.function(
    image=image,
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
    },
    cpu=8,
    memory=32768,
    timeout=3600,
)
def _verify(mp4_rel: str, episode: str, stride: int, max_src: int, tol: float):
    import os
    import subprocess
    import tempfile

    import numpy as np
    import simplejpeg
    import zarr

    def thumb_from_jpeg(buf):
        arr = simplejpeg.decode_jpeg(bytes(buf), colorspace="GRAY")
        return _pool(arr[..., 0] if arr.ndim == 3 else arr)

    def _pool(g, hw=(18, 32)):
        """Mean-pool to a tiny fingerprint, then z-normalise so JPEG vs H.264
        exposure/compression differences don't dominate the match."""
        h, w = g.shape[:2]
        ph, pw = h // hw[0], w // hw[1]
        g = g[: ph * hw[0], : pw * hw[1]].astype(np.float32)
        g = g.reshape(hw[0], ph, hw[1], pw).mean(axis=(1, 3))
        g -= g.mean()
        s = g.std()
        return g / s if s > 1e-6 else g

    # ---- source fingerprints -------------------------------------------------
    zpath = os.path.join(CFG.volume_mount_path, f"{episode}.zarr")
    store = zarr.open(zpath, mode="r")
    key = (
        next(k for k in store.array_keys() if k.endswith("images.front_1"))
        if hasattr(store, "array_keys")
        else "images.front_1"
    )
    arr = store[key]
    n_src = min(int(arr.shape[0]), max_src)
    # arr[i] on a VariableLengthBytes array yields a 0-d wrapper that
    # simplejpeg cannot parse ("could not determine subsampling level"); the
    # reader itself documents arr[i:i+1][0] as the workaround.
    src = np.stack([thumb_from_jpeg(arr[i : i + 1][0]) for i in range(n_src)])
    print(f"[align] source: {episode} -> {n_src} raw frames fingerprinted", flush=True)

    # ---- video fingerprints --------------------------------------------------
    mp4 = os.path.join(CFG.output_mount_path, mp4_rel)
    if not os.path.exists(mp4):
        raise SystemExit(f"mp4 not found on the outputs volume: {mp4}")
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", mp4, f"{td}/f%05d.png", "-y"],
            check=True,
        )
        import PIL.Image

        fs = sorted(os.listdir(td))
        vid = np.stack(
            [
                _pool(np.asarray(PIL.Image.open(os.path.join(td, f)).convert("L")))
                for f in fs
            ]
        )
    print(f"[align] video: {len(vid)} frames from {os.path.basename(mp4)}", flush=True)

    # ---- match each video frame to its best source index --------------------
    flat_s = src.reshape(len(src), -1)
    flat_v = vid.reshape(len(vid), -1)
    matches, scores = [], []
    for k in range(len(flat_v)):
        d = np.linalg.norm(flat_s - flat_v[k], axis=1)
        i = int(np.argmin(d))
        matches.append(i)
        scores.append(float(d[i]))

    print("\n[align] video frame -> source frame (delta from previous)")
    bad_mono, bad_stride = [], []
    for k, i in enumerate(matches):
        delta = "-" if k == 0 else str(i - matches[k - 1])
        flag = ""
        if k and i <= matches[k - 1]:
            bad_mono.append(k)
            flag = "  <-- NOT INCREASING"
        elif k and abs((i - matches[k - 1]) - stride) > 0:
            bad_stride.append((k, i - matches[k - 1]))
            flag = f"  <-- step != {stride}"
        if k < 24 or flag or k >= len(matches) - 4:
            print(
                f"   video[{k:4d}] -> source[{i:5d}]  delta={delta:>4}  "
                f"dist={scores[k]:.2f}{flag}"
            )

    med = float(np.median(scores))
    print(f"\n[align] median match distance = {med:.3f} (0 = identical fingerprint)")
    steps = np.diff(matches) if len(matches) > 1 else np.array([])
    if steps.size:
        vals, cnts = np.unique(steps, return_counts=True)
        print(
            "[align] step histogram: "
            + ", ".join(f"{v}x{c}" for v, c in zip(vals.tolist(), cnts.tolist()))
        )
    ok = not bad_mono and not bad_stride and med <= tol
    print(f"\n[align] monotonic: {'OK' if not bad_mono else f'FAIL at {bad_mono[:8]}'}")
    print(
        f"[align] stride=={stride}: "
        f"{'OK' if not bad_stride else f'FAIL {bad_stride[:8]}'}"
    )
    print(
        f"[align] fingerprint match: {'OK' if med <= tol else f'WEAK (median {med:.2f} > tol {tol})'}"
    )
    print(f"\n[align] VERDICT: {'PASS' if ok else 'FAIL'}")
    return {"ok": ok, "matches": matches, "median_dist": med}


@app.local_entrypoint()
def verify(
    mp4: str,
    episode: str,
    stride: int = 6,
    max_src: int = 4000,
    tol: float = 6.0,
):
    """Check a val mp4's frames against the original episode's frame order."""
    r = _verify.remote(mp4, episode, int(stride), int(max_src), float(tol))
    if not r["ok"]:
        raise SystemExit("alignment FAILED — see the per-frame table above")
