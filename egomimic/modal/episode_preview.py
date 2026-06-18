"""Render episode image-obs to MP4 and serve them from a self-hosted web viewer.

Reads the per-frame JPEG image observations (``images.front_1``) straight from
the episode zarr stores on the ``mecka_data_v2`` volume, encodes each episode to
an H.264 MP4 on a dedicated previews volume, and serves a browser viewer (index
+ streaming) via a Modal web endpoint. The generated MSE viewer
(``egomimic/scripts/build_mse_viewer.py``) embeds ``<video>`` players that stream
from the ``/video/<hash>`` endpoint here.

Standalone app — does NOT import egomimic/torch. The image obs is a zarr array
of dtype VariableLengthBytes (native zarr 3.x), one JPEG per frame, so reading
needs only zarr + simplejpeg; encoding uses the ffmpeg CLI (apt-installed here).

Usage
-----
# 1) Render episodes to MP4 (parallel, idempotent). Pass hashes inline, as a
#    JSON list file, or straight from an mse_scores.json a scoreMseModal run wrote:
MODAL_ENVIRONMENT=robotics modal run egomimic/modal/episode_preview.py::render_all --hashes 69b0a081db7a56404d0f5517
MODAL_ENVIRONMENT=robotics modal run egomimic/modal/episode_preview.py::render_all --hashes-file /path/to/episode_hashes.json
MODAL_ENVIRONMENT=robotics modal run egomimic/modal/episode_preview.py::render_for_scores --scores-file /path/to/mse_scores.json

# 2) Deploy the viewer (persistent URL):
MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/episode_preview.py
#    → open the printed https URL; it lists every rendered episode and plays it.
"""

import json
import subprocess
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# App / image / volumes
# ---------------------------------------------------------------------------

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

app = modal.App("egoverse-episode-preview", image=image)

# Source episodes (read-only) and a dedicated previews volume (read/write).
zarr_volume = modal.Volume.from_name("mecka_data_v2")
previews_volume = modal.Volume.from_name(
    "mecka-episode-previews", create_if_missing=True
)

ZARR_MOUNT = "/mnt/zarr-data"
PREVIEW_MOUNT = "/mnt/previews"
IMAGE_KEY = "images.front_1"  # Mecka cartesian camera obs key
FPS = 30  # obs/control rate; tune if playback looks off


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
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-vf",
            f"scale={outW}:{outH}",
            *codec_args,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",  # web-streamable (moov atom up front)
            str(tmp),
        ]

    # NVENC (GPU hardware encoder) first; libx264 fallback if unavailable.
    encoders = [
        (
            "h264_nvenc",
            ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "23"],
        ),
        ("libx264", ["-c:v", "libx264", "-crf", "23", "-preset", "fast"]),
    ]
    n = 0
    last_err = ""
    for enc_name, codec_args in encoders:
        proc = subprocess.Popen(
            _cmd(codec_args), stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
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
        print(
            f"[{episode_hash[:8]}] {enc_name} failed, trying fallback: {last_err[-160:]}"
        )
    else:
        return {"hash": episode_hash, "status": f"ffmpeg_fail: {last_err}", "frames": n}

    out_path.write_bytes(tmp.read_bytes())
    tmp.unlink(missing_ok=True)
    previews_volume.commit()
    return {"hash": episode_hash, "status": "ok", "frames": n}


def _render_targets(targets: list[str], force: bool) -> None:
    targets = list(dict.fromkeys(h for h in targets if h))
    if not targets:
        raise SystemExit("No episode hashes to render.")
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
    print("Deploy the viewer:  modal deploy egomimic/modal/episode_preview.py")


@app.local_entrypoint()
def render_all(hashes: str = "", hashes_file: str = "", force: bool = False) -> None:
    """Render every requested episode in parallel.

    hashes: comma-separated episode hashes.
    hashes_file: local path to a JSON list of hashes (e.g. an
        episode_hashes.json or an exported eps_to_use list).
    """
    targets = [h.strip() for h in hashes.split(",") if h.strip()]
    if hashes_file:
        targets += json.loads(Path(hashes_file).read_text())
    _render_targets(targets, force)


@app.local_entrypoint()
def render_for_scores(scores_file: str = "", force: bool = False) -> None:
    """Render every episode referenced by an ``mse_scores.json`` file.

    scores_file: local path to an ``mse_scores.json`` (``{task: [[hash, mse], ...]}``)
        as written by ``egomimic/modal/scoreMseModal.py``. Flattens the per-task
        lists to the union of episode hashes and renders them.
    """
    if not scores_file:
        raise SystemExit("Pass --scores-file pointing at an mse_scores.json.")
    scores = json.loads(Path(scores_file).read_text())
    targets: list[str] = []
    for pairs in scores.values():
        for h, _mse in pairs:
            targets.append(h)
    _render_targets(targets, force)


# ---------------------------------------------------------------------------
# Web viewer: index page + MP4 streaming
# ---------------------------------------------------------------------------


@app.function(
    volumes={PREVIEW_MOUNT: previews_volume},
    cpu=4.0,
    memory=8192,
    min_containers=0,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def viewer():
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse, Response

    web = FastAPI(title="EgoVerse Episode Viewer")

    def _episodes() -> list[str]:
        previews_volume.reload()
        return sorted(p.stem for p in Path(PREVIEW_MOUNT).glob("*.mp4"))

    @web.get("/", response_class=HTMLResponse)
    def index():
        eps = _episodes()
        cards = "\n".join(
            f'<div class="card"><div class="h">{h}</div>'
            f'<video controls preload="metadata" src="/video/{h}"></video></div>'
            for h in eps
        )
        return f"""<!doctype html><html><head><meta charset=utf-8>
<title>EgoVerse Episodes ({len(eps)})</title>
<style>
 body{{background:#111;color:#eee;font-family:system-ui,sans-serif;margin:0;padding:16px}}
 h1{{font-size:16px;font-weight:600}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:14px}}
 .card{{background:#1c1c1c;border-radius:8px;padding:8px}}
 .card video{{width:100%;border-radius:4px;background:#000}}
 .h{{font:12px ui-monospace,monospace;color:#9ad;margin-bottom:6px;word-break:break-all}}
</style></head><body>
<h1>EgoVerse episode previews — {len(eps)} episodes (images.front_1)</h1>
<div class="grid">{cards or "<p>No MP4s yet — run <code>modal run ...::render_all</code>.</p>"}</div>
</body></html>"""

    @web.get("/video/{episode_hash}")
    def video(episode_hash: str):
        # Basename guard: no path traversal.
        safe = Path(episode_hash).name
        path = Path(PREVIEW_MOUNT) / f"{safe}.mp4"
        if not path.exists():
            previews_volume.reload()
        if not path.exists():
            return Response(status_code=404)
        return FileResponse(str(path), media_type="video/mp4")

    return web
