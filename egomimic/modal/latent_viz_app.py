"""Serve the latent + episode-video viewer as a Modal website.

Builds the self-contained viewer HTML (egomimic/scripts/build_latent_viz.py)
from the curation run's tsne3d/ JSONs + scores_by_task.json on the
egoverse-training-outputs volume, and serves it at a persistent URL.

The heavy 3-D rendering itself is WebGL and runs on the *client's* GPU — this
app exists so the viewer is a launchable website (no local build step) and so
the HTML is rebuilt from the volume on each cold start (fresh data after new
curation runs). Override the source run with the LATENT_VIZ_RUN env var.

Deploy:
  MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/latent_viz_app.py
  → https://mecka-robotics--egoverse-latent-viz-viewer.modal.run
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

# Curation run whose tsne3d/ + scores_by_task.json feed the viewer.
DEFAULT_RUN = "deminf_tsne/14task_dim64_k10_20_viz3d_2026-06-09_19-49-48"
OUTPUTS_MOUNT = "/mnt/outputs"

_HERE = Path(__file__).resolve().parent
_BUILDER = _HERE.parent / "scripts" / "build_latent_viz.py"
_VAL_JSON = _HERE.parent / "hydra_configs" / "data" / "extra" / "mecka_d64_val.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]")
    # Bake the builder + VAL list so the app needs no repo clone.
    .add_local_file(_BUILDER, remote_path="/root/build_latent_viz.py", copy=True)
    .add_local_file(_VAL_JSON, remote_path="/root/mecka_d64_val.json", copy=True)
)

app = modal.App("egoverse-latent-viz", image=image)
outputs_volume = modal.Volume.from_name("egoverse-training-outputs")


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

    run = os.environ.get("LATENT_VIZ_RUN", DEFAULT_RUN)
    run_dir = Path(OUTPUTS_MOUNT) / run
    tsne_dir = run_dir / "tsne3d"
    scores = json.load(open(run_dir / "scores_by_task.json"))
    val = json.load(open("/root/mecka_d64_val.json"))
    html = build_html(tsne_dir, scores, val)
    print(f"viewer: built {len(html)/1e6:.1f} MB HTML from {run}")

    web = FastAPI()

    @web.get("/", response_class=HTMLResponse)
    def index():
        return html

    @web.get("/health", response_class=PlainTextResponse)
    def health():
        return f"ok ({run}, {len(html)} bytes)"

    return web
