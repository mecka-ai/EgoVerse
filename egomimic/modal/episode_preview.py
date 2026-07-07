"""Render episode image-obs to MP4 and serve them from a self-hosted web viewer.

Replaces the external "Atlas Capture" viewer: reads the per-frame JPEG image
observations (``images.front_1``) straight from the episode zarr stores on the
``mecka_data_v2`` volume, encodes each episode to an H.264 MP4 on a dedicated
previews volume, and serves a browser viewer (index + streaming) via a Modal
web endpoint.

Standalone app — does NOT import egomimic/torch. The image obs is a zarr array
of dtype VariableLengthBytes (native zarr 3.x), one JPEG per frame, so reading
needs only zarr + simplejpeg; encoding uses the ffmpeg CLI (apt-installed here).

Usage
-----
# 1) Render all 383 fourteen-task episodes to MP4 (parallel, idempotent):
MODAL_ENVIRONMENT=robotics modal run egomimic/modal/episode_preview.py::render_all

# Render a custom hash list / single episode:
MODAL_ENVIRONMENT=robotics modal run egomimic/modal/episode_preview.py::render_all --hashes 69b0a081db7a56404d0f5517

# 2) Deploy the unified viewer (latent t-SNE + MP4 streaming):
MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/latent_viz_app.py
#    → open / to pick a curation run; /episodes for the MP4 grid.
"""

import json
import os
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App / image / volumes
# ---------------------------------------------------------------------------

# Default episode set: the 383 fourteen-task hashes (same set the deminf64 /
# tokenizer runs use). Baked in so render_all needs no local file at run time.
_HASHES_FILE = Path(__file__).resolve().parents[1] / "hydra_configs/data/extra/mecka_curated_14task_all.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "zarr==3.1.5",
        "simplejpeg",
        "numpy",
        "fastapi[standard]",
    )
)

app = modal.App("egoverse-episode-preview-render", image=image)

# Source episodes (read-only) and a dedicated previews volume (read/write).
# Zarr data volume: MODAL_ZARR_VOLUME env var (default mecka_data_v2), read at submit
# time — same convention as modal_setup.py.
zarr_volume = modal.Volume.from_name(
    (__import__("os").environ.get("MODAL_ZARR_VOLUME", "mecka_data_v2").strip() or "mecka_data_v2")
)
previews_volume = modal.Volume.from_name("mecka-episode-previews", create_if_missing=True)

ZARR_MOUNT = "/mnt/zarr-data"
PREVIEW_MOUNT = "/mnt/previews"
IMAGE_KEY = "images.front_1"   # Mecka cartesian camera obs key
FPS = 30                       # obs/control rate; tune if playback looks off


# ---------------------------------------------------------------------------
# Render one episode → MP4 on the previews volume
# ---------------------------------------------------------------------------


@app.function(
    volumes={ZARR_MOUNT: zarr_volume, PREVIEW_MOUNT: previews_volume},
    gpu="L40S",
    cpu=8.0,
    memory=16384,
    timeout=1200,
)
def render_episode(episode_hash: str, fps: int = FPS, force: bool = False) -> dict:
    """Decode images.front_1 from the episode zarr and write an H.264 MP4.

    Runs on an L40S and encodes with NVENC (h264_nvenc) — hardware video
    encoding — falling back to libx264 if NVENC is unavailable. 8 CPUs cover
    the JPEG-decode feed side.

    Idempotent: skips episodes whose MP4 already exists unless force=True.
    """
    import numpy as np
    import simplejpeg
    import zarr

    out_path = Path(PREVIEW_MOUNT) / f"{episode_hash}.mp4"
    if out_path.exists() and not force:
        return {"hash": episode_hash, "status": "skipped", "frames": 0}

    # Episode dirs are stored as "<hash>.zarr" (bare "<hash>" also seen).
    store_path = None
    for cand in (f"{episode_hash}.zarr", episode_hash):
        if (Path(ZARR_MOUNT) / cand).is_dir():
            store_path = Path(ZARR_MOUNT) / cand
            break
    if store_path is None:
        return {"hash": episode_hash, "status": "missing_zarr", "frames": 0}

    store = zarr.open_group(str(store_path), mode="r")
    if IMAGE_KEY not in store:
        return {"hash": episode_hash, "status": f"no_key:{IMAGE_KEY}", "frames": 0}

    jpegs = store[IMAGE_KEY][:]  # object array of JPEG byte strings, length T
    if len(jpegs) == 0:
        return {"hash": episode_hash, "status": "empty", "frames": 0}

    # Decode first frame to get dimensions, then stream all frames into ffmpeg.
    first = simplejpeg.decode_jpeg(bytes(jpegs[0]), colorspace="RGB")
    h, w = first.shape[:2]
    outW, outH = (w // 2) - (w // 2) % 2, (h // 2) - (h // 2) % 2  # half-res, even dims

    tmp = Path("/tmp") / f"{episode_hash}.mp4"

    def _cmd(codec_args: list[str]) -> list[str]:
        return [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "-",
            "-an",
            "-vf", f"scale={outW}:{outH}",
            *codec_args,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",   # web-streamable (moov atom up front)
            str(tmp),
        ]

    # NVENC (GPU hardware encoder) first; libx264 fallback if unavailable.
    encoders = [
        ("h264_nvenc", ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23"]),
        ("libx264",    ["-c:v", "libx264", "-crf", "23", "-preset", "fast"]),
    ]
    n = 0
    last_err = ""
    for enc_name, codec_args in encoders:
        proc = subprocess.Popen(_cmd(codec_args), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        n = 0
        try:
            for jb in jpegs:
                frame = simplejpeg.decode_jpeg(bytes(jb), colorspace="RGB")
                if frame.shape[:2] != (h, w):  # guard against ragged frames
                    continue
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                n += 1
            proc.stdin.close()
            err = proc.stderr.read()
            if proc.wait() == 0:
                break  # encoded successfully
            last_err = err.decode(errors="replace")[-300:]
        except BrokenPipeError:
            last_err = proc.stderr.read().decode(errors="replace")[-300:]
        finally:
            if proc.poll() is None:
                proc.kill()
        print(f"[{episode_hash[:8]}] {enc_name} failed, trying fallback: {last_err[-160:]}")
    else:
        return {"hash": episode_hash, "status": f"ffmpeg_fail: {last_err}", "frames": n}

    out_path.write_bytes(tmp.read_bytes())
    tmp.unlink(missing_ok=True)
    previews_volume.commit()
    return {"hash": episode_hash, "status": "ok", "frames": n}


@app.local_entrypoint()
def render_all(hashes: str = "", force: bool = False) -> None:
    """Render every requested episode in parallel.

    hashes: comma-separated episode hashes; empty → the baked 383-episode set.
    """
    if hashes:
        targets = [h.strip() for h in hashes.split(",") if h.strip()]
    else:
        targets = json.loads(_HASHES_FILE.read_text())
    print(f"Rendering {len(targets)} episode(s) → mecka-episode-previews volume")

    ok = skipped = failed = 0
    for r in render_episode.map(targets, kwargs={"force": force}):
        st = r["status"]
        if st == "ok":
            ok += 1
        elif st == "skipped":
            skipped += 1
        else:
            failed += 1
            print(f"  ! {r['hash']}: {st}")
    print(f"\nDone: {ok} rendered, {skipped} already present, {failed} failed.")
    print("Deploy the viewer:  modal deploy egomimic/modal/latent_viz_app.py")
