"""Serve the latent + episode-video viewer as a Modal website.

Builds the self-contained viewer HTML (egomimic/scripts/build_latent_viz.py)
from a run directory of tsne3d_<task>.json files on the
egoverse-training-outputs volume and serves it at a persistent URL. Works with
any run produced by egomimic/modal/latentVizModal.py (any data config) — and
with old curation runs, whose dirs have the same tsne3d/ layout.

Optional sidecar files in the run dir (used when present, ignored otherwise):
  scores_by_task.json   {task: {hash: score}} — enables score color/sort UI
  val_episodes.json     [hash, ...]           — enables VAL badges/filter

The heavy 3-D rendering itself is WebGL and runs on the *client's* GPU — this
app exists so the viewer is a launchable website (no local build step) and so
the HTML is rebuilt from the volume on each cold start (fresh data after new
export runs).

Run selection:
  LATENT_VIZ_RUN     volume-relative run dir (its tsne3d/ subdir is used; a
                     dir that itself contains tsne3d_*.json also works)
  unset              newest run dir with a tsne3d/ subdir under the roots in
                     LATENT_VIZ_ROOTS (default: latent_viz, deminf_tsne)

Deploy:
  MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/latent_viz_app.py
  → https://mecka-robotics--egoverse-latent-viz-viewer.modal.run
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

OUTPUTS_MOUNT = "/mnt/outputs"
DEFAULT_ROOTS = "latent_viz,deminf_tsne"

_HERE = Path(__file__).resolve().parent
_BUILDER = _HERE.parent / "scripts" / "build_latent_viz.py"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]")
    # Bake the builder so the app needs no repo clone.
    .add_local_file(_BUILDER, remote_path="/root/build_latent_viz.py", copy=True)
)

app = modal.App("egoverse-latent-viz", image=image)
outputs_volume = modal.Volume.from_name("egoverse-training-outputs")


def _discover_run_dir() -> Path:
    """Resolve the run dir: LATENT_VIZ_RUN, else newest tsne3d run under roots."""
    run = os.environ.get("LATENT_VIZ_RUN")
    if run:
        return Path(OUTPUTS_MOUNT) / run
    roots = os.environ.get("LATENT_VIZ_ROOTS", DEFAULT_ROOTS).split(",")
    # latent_viz runs sit at <root>/<name>/<desc_ts>/tsne3d; old curation runs
    # at <root>/<run>/tsne3d — search both depths.
    candidates = [
        t.parent
        for root in roots
        for pattern in ("*/tsne3d", "*/*/tsne3d")
        for t in (Path(OUTPUTS_MOUNT) / root.strip()).glob(pattern)
        if t.is_dir()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No run dirs with tsne3d/ under {roots} on the outputs volume. "
            "Run latentVizModal.py first, or set LATENT_VIZ_RUN."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


@app.function(
    volumes={OUTPUTS_MOUNT: outputs_volume},
    cpu=4.0,
    memory=8192,
    min_containers=0,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def viewer():
    import json
    import sys

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, PlainTextResponse

    sys.path.insert(0, "/root")
    from build_latent_viz import build_html

    run_dir = _discover_run_dir()
    tsne_dir = run_dir / "tsne3d"
    if not tsne_dir.is_dir():
        tsne_dir = run_dir  # run dir holds the tsne3d_*.json files directly

    scores = None
    scores_path = run_dir / "scores_by_task.json"
    if scores_path.is_file():
        scores = json.load(open(scores_path))

    val = None
    val_path = run_dir / "val_episodes.json"
    if val_path.is_file():
        val = json.load(open(val_path))

    html = build_html(tsne_dir, scores, val)
    rel = run_dir.relative_to(OUTPUTS_MOUNT)
    print(
        f"viewer: built {len(html)/1e6:.1f} MB HTML from {rel} "
        f"(scores={'yes' if scores else 'no'}, val={'yes' if val else 'no'})"
    )

    web = FastAPI()

    @web.get("/", response_class=HTMLResponse)
    def index():
        return html

    @web.get("/health", response_class=PlainTextResponse)
    def health():
        return f"ok ({rel}, {len(html)} bytes)"

    return web
