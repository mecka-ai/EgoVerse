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
    .apt_install("ffmpeg")
    .pip_install("fastapi[standard]", "sqlalchemy", "psycopg2-binary")
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
    n_runs = len(runs)
    options = "\n".join(
        f'<option value="{html.escape(r, quote=True)}"'
        f'{" selected" if r == default_run else ""}>{html.escape(r)}</option>'
        for r in runs[:80]
    )
    err = f'<div class="err">{html.escape(error)}</div>' if error else ""
    # Parse timestamp suffix from run name for display (format: …_YYYY-MM-DD_HH-MM-SS)
    def _run_date(r: str) -> str:
        parts = r.replace("_", "-").split("-")
        for i in range(len(parts) - 5):
            try:
                return f"{parts[i+0]}-{parts[i+1]}-{parts[i+2]}"
            except Exception:
                pass
        return ""

    run_rows = "\n".join(
        f'<div class="run-row" onclick="pick(\'{html.escape(r, quote=True)}\')">'
        f'<span class="run-name">{html.escape(r)}</span>'
        f'<span class="run-date">{_run_date(r)}</span>'
        f'</div>'
        for r in runs[:60]
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Meckaverse</title>
<style>
  :root{{--bg:#101014;--bar:#1a1b21;--line:#2b2d36;--acc:#3b82f6;--txt:#e8e8ea;}}
  html,body{{height:100%;margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,Helvetica,Arial,sans-serif;}}
  body{{display:flex;flex-direction:column;overflow:hidden;}}
  #topbar{{padding:10px 18px;background:var(--bar);border-bottom:1px solid var(--line);
           display:flex;align-items:center;gap:14px;}}
  #topbar h1{{margin:0;font-size:15px;font-weight:700;}}
  #topbar .sub{{font-size:13px;color:#9aa;}}
  #topbar .links{{margin-left:auto;font-size:13px;display:flex;gap:14px;}}
  #topbar .links a{{color:#9ecbff;text-decoration:none;}}
  #topbar .links a:hover{{color:#7fd4ff;}}
  #main{{flex:1;min-height:0;display:flex;overflow:hidden;}}
  #sidebar{{width:420px;flex-shrink:0;display:flex;flex-direction:column;border-right:1px solid var(--line);overflow:hidden;}}
  #search-area{{padding:12px 14px;border-bottom:1px solid var(--line);}}
  #search-area input{{width:100%;box-sizing:border-box;font-size:13px;padding:7px 10px;
                       background:#26272f;color:var(--txt);border:1px solid var(--line);border-radius:6px;}}
  #run-list{{flex:1;overflow-y:auto;}}
  .run-row{{padding:8px 14px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;
            border-bottom:1px solid var(--line);font-size:13px;}}
  .run-row:hover{{background:#1f2027;}}
  .run-row.selected{{background:#1a2d4a;}}
  .run-name{{font-family:ui-monospace,monospace;color:#cdd;font-size:12px;}}
  .run-date{{font-size:11px;color:#777;flex-shrink:0;margin-left:8px;}}
  #detail{{flex:1;padding:24px;display:flex;flex-direction:column;gap:16px;overflow-y:auto;}}
  #detail h2{{margin:0;font-size:14px;color:#9aa;font-weight:600;}}
  #detail .run-path{{font-family:ui-monospace,monospace;font-size:13px;color:#9ecbff;word-break:break-all;}}
  #runInput{{width:100%;box-sizing:border-box;font-size:13px;padding:8px 10px;
              background:#26272f;color:var(--txt);border:1px solid var(--line);border-radius:6px;margin-top:4px;}}
  .load-btn{{padding:10px 24px;background:var(--acc);color:#fff;border:none;border-radius:8px;
              cursor:pointer;font-size:14px;font-weight:600;width:100%;margin-top:4px;}}
  .load-btn:hover{{background:#2563eb;}}
  .err{{background:#4d1d1d;color:#ff9d9d;padding:8px 12px;border-radius:6px;font-size:13px;}}
  .empty{{padding:24px;color:#666;font-size:13px;}}
</style></head>
<body>
<div id="topbar">
  <h1>Meckaverse</h1>
  <span class="sub">{n_runs} curation run{'' if n_runs == 1 else 's'} on <code>egoverse-training-outputs</code></span>
  <div class="links">
    <a href="/episodes">Episodes</a>
    <a href="/api/runs">API</a>
    <a href="/health">Health</a>
  </div>
</div>
<div id="main">
  <div id="sidebar">
    <div id="search-area">
      <input id="search" type="text" placeholder="Filter runs…" oninput="filterRuns()" autofocus />
    </div>
    <div id="run-list">
      {'<div class="empty">No curation runs found yet.</div>' if not runs else run_rows}
    </div>
  </div>
  <div id="detail">
    <div>{err}</div>
    <div>
      <h2>Selected run</h2>
      <div id="selected-label" class="run-path" style="color:#666">— none selected —</div>
    </div>
    <div>
      <h2>Or enter path manually</h2>
      <input id="runInput" type="text" value="{safe_default}" placeholder="name/description_timestamp" />
    </div>
    <button class="load-btn" onclick="load()">Load viewer →</button>
  </div>
</div>
<script>
const ALL_ROWS = Array.from(document.querySelectorAll('.run-row'));
function filterRuns() {{
  const q = document.getElementById('search').value.toLowerCase();
  ALL_ROWS.forEach(r => {{
    r.style.display = r.querySelector('.run-name').textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}
function pick(run) {{
  ALL_ROWS.forEach(r => r.classList.remove('selected'));
  const row = ALL_ROWS.find(r => r.querySelector('.run-name').textContent === run);
  if (row) row.classList.add('selected');
  document.getElementById('runInput').value = run;
  document.getElementById('selected-label').textContent = run;
  document.getElementById('selected-label').style.color = '#9ecbff';
}}
function load() {{
  const run = document.getElementById('runInput').value.trim();
  if (run) window.location.href = '/view?run=' + encodeURIComponent(run);
}}
document.getElementById('runInput').addEventListener('keydown', e => {{ if (e.key === 'Enter') load(); }});
</script>
</body></html>"""


def _fetch_episode_metadata(hashes: list[str]) -> dict:
    """Query app.episodes for the given hashes via the egoverse-sql secret."""
    if not hashes:
        return {}
    try:
        import os
        from urllib.parse import quote_plus
        from sqlalchemy import create_engine, text as sql_text

        database_url = os.environ.get("DATABASE_URL")
        if database_url:
            database_url = database_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        else:
            user = os.environ["PG_USER"]
            password = quote_plus(os.environ["PG_PASSWORD"])
            host = os.environ["PG_HOST"]
            port = os.environ.get("PG_PORT", "5432")
            database = os.environ.get("PG_DATABASE", "defaultdb")
            database_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}?sslmode=require"

        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.connect() as conn:
            rows = conn.execute(
                sql_text(
                    "SELECT episode_hash, operator, task_description, scene, objects, data_type"
                    " FROM app.episodes WHERE episode_hash = ANY(:hashes)"
                ),
                {"hashes": hashes},
            ).fetchall()
        return {
            row[0]: {
                "operator": row[1],
                "task_description": row[2],
                "scene": row[3],
                "objects": row[4],
                "data_type": row[5],
            }
            for row in rows
        }
    except Exception as exc:
        print(f"[viewer] metadata fetch failed: {exc}")
        return {}


@app.function(
    volumes={OUTPUTS_MOUNT: outputs_volume, PREVIEW_MOUNT: previews_volume},
    secrets=[modal.Secret.from_name("egoverse-sql")],
    cpu=4.0,
    memory=8192,
    min_containers=1,
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

    web = FastAPI(title="Meckaverse")
    html_cache: dict[str, str] = {}
    frame_cache: dict[str, bytes] = {}
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

        all_hashes: set[str] = set()
        for f in sorted(tsne_dir.glob("tsne3d_*.json")):
            d = json.load(open(f))
            all_hashes.update(d.get("episodes", []))
        for task_scores in scores.values():
            all_hashes.update(task_scores.keys())
        metadata = _fetch_episode_metadata(list(all_hashes))

        body = build_html(tsne_dir, scores, val_list, video_base="/video/", frame_base="/frame/", run_label=run, metadata=metadata)
        html_cache[run] = body
        print(f"viewer: built {len(body)/1e6:.1f} MB HTML for run={run} ({len(metadata)} metadata entries)")
        return body

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
<title>Meckaverse Episodes ({len(eps)})</title>
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
<h1>Meckaverse episode previews — {len(eps)} episodes</h1>
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

    @web.get("/frame/{episode_hash}/{frame_num}")
    def frame(episode_hash: str, frame_num: int):
        import subprocess
        from fastapi.responses import Response

        safe = Path(episode_hash).name
        cache_key = f"{safe}_{frame_num}"
        if cache_key in frame_cache:
            return Response(frame_cache[cache_key], media_type="image/jpeg")

        path = Path(PREVIEW_MOUNT) / f"{safe}.mp4"
        if not path.exists():
            _reload_volumes()
        if not path.exists():
            return PlainTextResponse("not found", status_code=404)

        result = subprocess.run(
            ["ffmpeg", "-ss", str(frame_num / 30.0), "-i", str(path),
             "-vframes", "1", "-q:v", "3", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
            capture_output=True,
        )
        if result.returncode != 0 or not result.stdout:
            return PlainTextResponse("frame extraction failed", status_code=500)

        frame_cache[cache_key] = result.stdout
        return Response(result.stdout, media_type="image/jpeg")

    return web
