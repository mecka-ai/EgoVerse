"""Unified EgoVerse viewer on Modal: latent t-SNE + episode MP4 streaming.

One deploy serves:
  * ``/``            — run picker (choose which curation run to visualize)
  * ``/view?run=…``  — interactive latent viewer (t-SNE 3-D + video grid)
  * ``/video/{hash}``— MP4 stream for episode preview (used by the latent viewer)
  * ``/episodes``    — simple standalone episode grid
  * ``/api/runs``    — JSON list of curation runs on the outputs volume

Episode MP4s are rendered separately (GPU batch job):
  MODAL_ENVIRONMENT=robotics modal run egomimic/modal/episode_preview.py::render_all

Deploy the unified viewer:
  MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/latent_viz_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

DEFAULT_RUN = "deminf_tsne/14task_dim64_k10_20_viz3d_2026-06-09_19-49-48"
OUTPUTS_MOUNT = "/mnt/outputs"
PREVIEW_MOUNT = "/mnt/previews"

_HERE = Path(__file__).resolve().parent
_BUILDER = _HERE.parent / "scripts" / "build_latent_viz.py"
_VAL_JSON = _HERE.parent / "hydra_configs" / "data" / "extra" / "mecka_d64_val.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]")
    .add_local_file(_BUILDER, remote_path="/root/build_latent_viz.py", copy=True)
    .add_local_file(_VAL_JSON, remote_path="/root/mecka_d64_val.json", copy=True)
)

app = modal.App("egoverse-viewer", image=image)
outputs_volume = modal.Volume.from_name("egoverse-training-outputs")
previews_volume = modal.Volume.from_name("mecka-episode-previews", create_if_missing=True)


def _list_curation_runs(outputs_root: Path) -> list[str]:
    """Find runs that have both tsne3d/ and scores_by_task.json."""
    if not outputs_root.is_dir():
        return []
    runs: list[str] = []
    for scores_path in outputs_root.rglob("scores_by_task.json"):
        run_dir = scores_path.parent
        if (run_dir / "tsne3d").is_dir():
            runs.append(str(run_dir.relative_to(outputs_root)))
    return sorted(set(runs), reverse=True)


def _landing_html(runs: list[str], default_run: str, error: str = "") -> str:
    import html

    safe_default = html.escape(default_run, quote=True)
    options = "\n".join(
        f'<option value="{html.escape(r, quote=True)}"'
        f'{" selected" if r == default_run else ""}>{html.escape(r)}</option>'
        for r in runs[:80]
    )
    err = f'<p style="color:#f88">{error}</p>' if error else ""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>EgoVerse Viewer</title>
<style>
  body {{ background:#101014; color:#e8e8ea; font-family:system-ui,sans-serif; margin:0; padding:32px; }}
  h1 {{ font-size:20px; margin:0 0 8px; }}
  p {{ color:#9aa; max-width:720px; line-height:1.5; }}
  form {{ margin:24px 0; display:flex; gap:10px; flex-wrap:wrap; align-items:center; }}
  input, select, button {{ font-size:14px; padding:8px 12px; border-radius:8px; border:1px solid #333; }}
  input[type=text] {{ flex:1; min-width:320px; background:#1a1b21; color:#eee; }}
  select {{ background:#1a1b21; color:#eee; max-width:480px; }}
  button {{ background:#3b82f6; color:#fff; border:none; cursor:pointer; font-weight:600; }}
  button:hover {{ background:#2563eb; }}
  .links {{ margin-top:28px; font-size:13px; }}
  a {{ color:#7fd4ff; }}
</style></head><body>
<h1>EgoVerse latent + episode viewer</h1>
<p>Pick a curation run on the <code>egoverse-training-outputs</code> volume
(path relative to volume root, e.g. <code>deminf32/resnet_k10-20_all14_…</code>).</p>
{err}
<form action="/view" method="get">
  <select name="run" onchange="document.getElementById('runInput').value=this.value">
    <option value="">— recent runs —</option>
    {options}
  </select>
  <input id="runInput" type="text" name="run" value="{safe_default}" placeholder="name/description_timestamp" required />
  <button type="submit">Load viewer</button>
</form>
<div class="links">
  <a href="/episodes">Episode MP4 grid</a> ·
  <a href="/api/runs">/api/runs</a> ·
  <a href="/health">health</a>
</div>
</body></html>"""


@app.function(
    volumes={OUTPUTS_MOUNT: outputs_volume, PREVIEW_MOUNT: previews_volume},
    cpu=4.0,
    memory=8192,
    min_containers=0,
    scaledown_window=600,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def viewer():
    import sys

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse

    sys.path.insert(0, "/root")
    from build_latent_viz import build_html

    web = FastAPI(title="EgoVerse Viewer")
    html_cache: dict[str, str] = {}
    val_list = json.load(open("/root/mecka_d64_val.json"))

    def _reload_volumes() -> None:
        outputs_volume.reload()
        previews_volume.reload()

    def _build_latent_html(run: str) -> str:
        run = run.strip().strip("/")
        if not run:
            raise HTTPException(400, "run path is required")
        if run in html_cache:
            return html_cache[run]

        _reload_volumes()
        run_dir = Path(OUTPUTS_MOUNT) / run
        tsne_dir = run_dir / "tsne3d"
        scores_path = run_dir / "scores_by_task.json"
        if not tsne_dir.is_dir():
            raise HTTPException(404, f"tsne3d/ not found under {run!r}")
        if not scores_path.is_file():
            raise HTTPException(404, f"scores_by_task.json not found under {run!r}")

        scores = json.load(open(scores_path))
        body = build_html(tsne_dir, scores, val_list, video_base="/video/")
        bar = (
            f'<div style="position:fixed;top:0;left:0;right:0;z-index:9999;padding:6px 14px;'
            f'background:#1a1b21;border-bottom:1px solid #333;font:13px system-ui;color:#ccc">'
            f'<b style="color:#7fd4ff">run</b> {run} · '
            f'<a href="/" style="color:#9ecbff">change run</a> · '
            f'<a href="/episodes" style="color:#9ecbff">episodes</a></div>'
            f'<div style="height:36px"></div>'
        )
        html = bar + body
        html_cache[run] = html
        print(f"viewer: built {len(html)/1e6:.1f} MB HTML for run={run}")
        return html

    @web.get("/")
    def index(
        run: str | None = Query(default=None),
        error: str | None = Query(default=None),
    ):
        _reload_volumes()
        runs = _list_curation_runs(Path(OUTPUTS_MOUNT))
        default = (run or os.environ.get("LATENT_VIZ_RUN") or DEFAULT_RUN).strip()
        if run:
            return RedirectResponse(url=f"/view?run={run}", status_code=302)
        return HTMLResponse(_landing_html(runs, default, error or ""))

    @web.get("/view", response_class=HTMLResponse)
    def view(run: str = Query(..., description="Volume-relative curation run path")):
        try:
            return _build_latent_html(run)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    @web.get("/api/runs")
    def api_runs():
        _reload_volumes()
        return JSONResponse({"runs": _list_curation_runs(Path(OUTPUTS_MOUNT))})

    @web.get("/health", response_class=PlainTextResponse)
    def health():
        _reload_volumes()
        n_runs = len(_list_curation_runs(Path(OUTPUTS_MOUNT)))
        n_mp4 = len(list(Path(PREVIEW_MOUNT).glob("*.mp4")))
        return f"ok runs={n_runs} mp4s={n_mp4} cached_html={len(html_cache)}"

    @web.get("/episodes", response_class=HTMLResponse)
    def episodes():
        _reload_volumes()
        eps = sorted(p.stem for p in Path(PREVIEW_MOUNT).glob("*.mp4"))
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
 a{{color:#7fd4ff}}
</style></head><body>
<p><a href="/">← latent viewer</a></p>
<h1>EgoVerse episode previews — {len(eps)} episodes</h1>
<div class="grid">{cards or '<p>No MP4s yet — run episode_preview.py::render_all.</p>'}</div>
</body></html>"""

    @web.get("/video/{episode_hash}")
    def video(episode_hash: str):
        safe = Path(episode_hash).name
        path = Path(PREVIEW_MOUNT) / f"{safe}.mp4"
        if not path.exists():
            _reload_volumes()
        if not path.exists():
            return PlainTextResponse("not found", status_code=404)
        return FileResponse(str(path), media_type="video/mp4")

    return web
