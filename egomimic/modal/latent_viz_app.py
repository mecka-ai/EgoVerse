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

# Pre-rendered frames live at PREVIEW_MOUNT/frames/{hash}/{idx:06d}.jpg, extracted
# every FRAME_STRIDE frames (matches the t-SNE every_n sampling). File index j
# (1-based) ↔ original mp4 frame (j-1)*FRAME_STRIDE, so a requested frame f maps
# to file round(f/FRAME_STRIDE)+1. The /frame endpoint serves these as static
# files (~50ms) and only falls back to on-demand ffmpeg when one is missing.
FRAME_STRIDE = 10
FRAME_MAX_W = 640

_HERE = Path(__file__).resolve().parent
_BUILDER         = _HERE.parent / "scripts" / "build_latent_viz.py"
_SPAN_BUILDER    = _HERE.parent / "scripts" / "build_span_viz.py"
_CLUSTER_BUILDER = _HERE.parent / "scripts" / "build_cluster_viz.py"
_VAL_JSON        = _HERE.parent / "hydra_configs" / "data" / "extra" / "mecka_d64_val.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install("fastapi[standard]", "sqlalchemy", "psycopg2-binary", "scikit-learn")
    .add_local_file(_BUILDER,         remote_path="/root/build_latent_viz.py",  copy=True)
    .add_local_file(_SPAN_BUILDER,    remote_path="/root/build_span_viz.py",    copy=True)
    .add_local_file(_CLUSTER_BUILDER, remote_path="/root/build_cluster_viz.py", copy=True)
    .add_local_file(_VAL_JSON,        remote_path="/root/mecka_d64_val.json",   copy=True)
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


def _list_span_runs(outputs_root: Path) -> list[str]:
    """Find clustered runs that have tsne3d/spans_tsne3d.json."""
    if not outputs_root.is_dir():
        return []
    runs: list[str] = []
    for p in outputs_root.rglob("tsne3d/spans_tsne3d.json"):
        run_dir = p.parent.parent
        runs.append(str(run_dir.relative_to(outputs_root)))
    return sorted(set(runs), reverse=True)


def _cluster_scores_dir(run_dir: Path) -> Path | None:
    """Return the directory containing *_clustered_scores.json, checking both scores/ and scores_v2/."""
    for candidate in ("scores_v2", "scores"):
        d = run_dir / candidate
        if d.is_dir() and any(d.glob("*_clustered_scores.json")):
            return d
    return None


def _list_cluster_runs(outputs_root: Path) -> list[str]:
    """Find runs that have *_clustered_scores.json in scores/ or scores_v2/."""
    if not outputs_root.is_dir():
        return []
    runs: list[str] = []
    for candidate in ("scores_v2", "scores"):
        for p in outputs_root.rglob(f"{candidate}/*_clustered_scores.json"):
            run_dir = p.parent.parent
            runs.append(str(run_dir.relative_to(outputs_root)))
    return sorted(set(runs), reverse=True)


def _run_mtime(run_dir: Path) -> float:
    """Latest mtime of a run's output artifacts — a cache version stamp so re-scored
    runs auto-refresh (the HTML is rebuilt when scores/tsne3d change)."""
    latest = 0.0
    for sub in ("tsne3d", "scores", "scores_v2"):
        d = run_dir / sub
        if d.is_dir():
            for f in d.glob("*"):
                try:
                    latest = max(latest, f.stat().st_mtime)
                except OSError:
                    pass
    p = run_dir / "scores_by_task.json"
    if p.is_file():
        try:
            latest = max(latest, p.stat().st_mtime)
        except OSError:
            pass
    return latest


def _landing_html(runs: list[str], default_run: str, error: str = "",
                  span_runs: list[str] | None = None,
                  cluster_runs: list[str] | None = None) -> str:
    import html

    span_runs    = span_runs    or []
    cluster_runs = cluster_runs or []
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
    span_rows = "\n".join(
        f'<div class="run-row" style="border-left:3px solid #7c3aed" '
        f'onclick="window.location.href=\'/view_spans?run={html.escape(r, quote=True)}\'">'
        f'<span class="run-name" style="color:#c4b5fd">{html.escape(r)}</span>'
        f'<span class="run-date">{_run_date(r)}</span>'
        f'</div>'
        for r in span_runs[:20]
    )
    cluster_rows = "\n".join(
        f'<div class="run-row" style="border-left:3px solid #06b6d4" '
        f'onclick="window.location.href=\'/view_clusters?run={html.escape(r, quote=True)}\'">'
        f'<span class="run-name" style="color:#67e8f9">{html.escape(r)}</span>'
        f'<span class="run-date">{_run_date(r)}</span>'
        f'</div>'
        for r in cluster_runs[:20]
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
      {(f'<div style="padding:6px 14px;font-size:11px;color:#7c3aed;font-weight:600;border-top:1px solid var(--line);margin-top:4px">SPAN RUNS ({len(span_runs)})</div>' + span_rows) if span_runs else ''}
      {(f'<div style="padding:6px 14px;font-size:11px;color:#06b6d4;font-weight:600;border-top:1px solid var(--line);margin-top:4px">CLUSTER RUNS ({len(cluster_runs)})</div>' + cluster_rows) if cluster_runs else ''}
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
    image=image,
    volumes={PREVIEW_MOUNT: previews_volume},
    cpu=4.0,
    timeout=1800,
)
def _render_frames_for(episode_hash: str) -> str:
    """Extract every FRAME_STRIDE-th frame of one episode's MP4 to downscaled JPEGs
    at PREVIEW_MOUNT/frames/{hash}/%06d.jpg (one ffmpeg pass, full decode)."""
    import subprocess

    safe = Path(episode_hash).name
    mp4 = Path(PREVIEW_MOUNT) / f"{safe}.mp4"
    if not mp4.exists():
        return f"skip {safe}: no mp4"
    outdir = Path(PREVIEW_MOUNT) / "frames" / safe
    if outdir.is_dir() and any(outdir.glob("*.jpg")):
        return f"skip {safe}: already rendered"
    outdir.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(mp4),
         "-vf", f"select=not(mod(n\\,{FRAME_STRIDE})),scale='min({FRAME_MAX_W},iw)':-2",
         "-vsync", "0", "-q:v", "5", str(outdir / "%06d.jpg")],
        capture_output=True,
    )
    n = len(list(outdir.glob("*.jpg")))
    if r.returncode != 0 or n == 0:
        return f"FAIL {safe}: rc={r.returncode} {r.stderr.decode()[:120]}"
    previews_volume.commit()
    return f"{safe}: {n} frames"


@app.function(
    volumes={PREVIEW_MOUNT: previews_volume},
    timeout=7200,
)
def render_all_frames(only: list[str] | None = None):
    """Pre-render frame JPEGs for every episode MP4 (or just ``only`` hashes).
    Fan out one container per episode via .map(). Run with:
      MODAL_ENVIRONMENT=robotics modal run egomimic/modal/latent_viz_app.py::render_all_frames
    """
    previews_volume.reload()
    eps = only or sorted(p.stem for p in Path(PREVIEW_MOUNT).glob("*.mp4"))
    print(f"[frames] rendering {len(eps)} episodes (stride={FRAME_STRIDE})")
    done = failed = skipped = 0
    for msg in _render_frames_for.map(eps, order_outputs=False):
        if msg.startswith("FAIL"):
            failed += 1
            print(msg)
        elif "already" in msg or "no mp4" in msg:
            skipped += 1
        else:
            done += 1
        if (done + failed + skipped) % 50 == 0:
            print(f"[frames] {done} done · {skipped} skipped · {failed} failed")
    print(f"[frames] COMPLETE: {done} rendered · {skipped} skipped · {failed} failed")
    return {"rendered": done, "skipped": skipped, "failed": failed}


@app.function(
    volumes={OUTPUTS_MOUNT: outputs_volume, PREVIEW_MOUNT: previews_volume},
    secrets=[modal.Secret.from_name("egoverse-sql")],
    cpu=16.0,
    memory=16384,
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
    from build_latent_viz  import build_html
    from build_span_viz    import build_span_html
    from build_cluster_viz import build_cluster_html

    web = FastAPI(title="Meckaverse")
    # run path → (mtime stamp, HTML); rebuilt when the run's outputs change.
    html_cache:         dict[str, tuple[float, str]] = {}
    span_html_cache:    dict[str, tuple[float, str]] = {}
    cluster_html_cache: dict[str, tuple[float, str]] = {}
    frame_cache:        dict[str, bytes] = {}
    val_list = json.load(open("/root/mecka_d64_val.json"))

    def _reload_volumes() -> None:
        try:
            outputs_volume.reload()
        except Exception as e:
            print(f"[viewer] outputs volume reload skipped: {e}")
        try:
            previews_volume.reload()
        except Exception as e:
            print(f"[viewer] previews volume reload skipped: {e}")

    def _build_latent_html(run: str) -> str:
        run = run.strip().strip("/")
        if not run:
            raise HTTPException(400, "run path is required")
        run_dir = Path(OUTPUTS_MOUNT) / run
        stamp = _run_mtime(run_dir)
        cached = html_cache.get(run)
        if cached and cached[0] == stamp:
            return cached[1]

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
        html_cache[run] = (stamp, body)
        print(f"viewer: built {len(body)/1e6:.1f} MB HTML for run={run} ({len(metadata)} metadata entries)")
        return body

    @web.get("/")
    def index(
        run: str | None = Query(default=None),
        error: str | None = Query(default=None),
    ):
        _reload_volumes()
        runs         = _list_curation_runs(Path(OUTPUTS_MOUNT))
        span_runs    = _list_span_runs(Path(OUTPUTS_MOUNT))
        cluster_runs = _list_cluster_runs(Path(OUTPUTS_MOUNT))
        default      = (run or os.environ.get("LATENT_VIZ_RUN") or DEFAULT_RUN).strip()
        if run:
            return RedirectResponse(url=f"/view?run={run}", status_code=302)
        return HTMLResponse(_landing_html(runs, default, error or "", span_runs=span_runs, cluster_runs=cluster_runs))

    @web.get("/view", response_class=HTMLResponse)
    def view(run: str = Query(..., description="Volume-relative curation run path")):
        run_clean = run.strip().strip("/")
        _reload_volumes()
        run_dir = Path(OUTPUTS_MOUNT) / run_clean
        tsne_dir = run_dir / "tsne3d"
        # Auto-detect language-cluster organization: a run is cluster-organized if it
        # has spans_tsne3d.json or *_clustered_scores.json. Route to the cluster viewer
        # (cluster selector + per-span video grid) even if a per-task tsne3d_*.json also
        # exists — the curation pipeline emits both for clustered runs.
        has_span_tsne = (tsne_dir / "spans_tsne3d.json").is_file()
        has_cluster_scores = _cluster_scores_dir(run_dir) is not None
        if has_span_tsne or has_cluster_scores:
            return RedirectResponse(url=f"/view_clusters?run={run_clean}", status_code=302)
        try:
            return _build_latent_html(run_clean)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    def _build_span_cached(run: str) -> str:
        run = run.strip().strip("/")
        if not run:
            raise HTTPException(400, "run path is required")
        _reload_volumes()
        stamp = _run_mtime(Path(OUTPUTS_MOUNT) / run)
        cached = span_html_cache.get(run)
        if cached and cached[0] == stamp:
            return cached[1]
        json_path = Path(OUTPUTS_MOUNT) / run / "tsne3d" / "spans_tsne3d.json"
        if not json_path.is_file():
            raise HTTPException(404, f"tsne3d/spans_tsne3d.json not found under {run!r}")
        data = json.load(open(json_path))
        body = build_span_html(data, video_base="/video/", frame_base="/frame/", run_label=run)
        span_html_cache[run] = (stamp, body)
        print(f"viewer: built span HTML {len(body)/1e6:.1f} MB for run={run} ({len(data.get('spans',[]))} spans)")
        return body

    @web.get("/view_spans", response_class=HTMLResponse)
    def view_spans(run: str = Query(..., description="Volume-relative clustered run path")):
        try:
            return _build_span_cached(run)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

    def _load_cluster_tsne(run: str) -> dict:
        """Load pre-computed t-SNE from tsne3d/spans_tsne3d.json (written by curate_v2).

        Returns shared span metadata arrays (cid, score, start, end, ep, txt) plus
        per-modality coordinate arrays (state/action/language → {x, y, z}).
        All arrays are aligned: index k → the same span across all modalities.
        Returns {} if the file is absent (viewer still works from cluster scores).
        """
        tsne_path = Path(OUTPUTS_MOUNT) / run / "tsne3d" / "spans_tsne3d.json"
        if not tsne_path.is_file():
            return {}
        data = json.load(open(tsne_path))
        spans = data["spans"]
        result: dict = {
            "cid":   [s["cluster"]                        for s in spans],
            "score": [s.get("score") or 0.0               for s in spans],
            "start": [int(s.get("start", 0))              for s in spans],
            "end":   [int(s.get("end",   s.get("start", 0) + 1)) for s in spans],
            "ep":    [s.get("ep", s.get("episode", ""))   for s in spans],
            "txt":   [str(s.get("text", ""))[:60]         for s in spans],
        }
        result["method"] = str(data.get("method", "tsne"))
        result["dims"] = int(data.get("dims", 3))
        for mode in ("state", "action", "language"):
            if mode in data:
                t = data[mode]
                m = {"x": t["x"], "y": t["y"]}
                if "z" in t:
                    m["z"] = t["z"]
                result[mode] = m
        print(f"[viewer] loaded cluster t-SNE for {run} ({len(spans)} spans, "
              f"modes={[m for m in ('state','action','language') if m in result]})")
        return result

    def _build_cluster_cached(run: str) -> str:
        run = run.strip().strip("/")
        if not run:
            raise HTTPException(400, "run path is required")
        _reload_volumes()
        run_dir = Path(OUTPUTS_MOUNT) / run
        stamp = _run_mtime(run_dir)
        cached = cluster_html_cache.get(run)
        if cached and cached[0] == stamp:
            return cached[1]
        scores_dir = _cluster_scores_dir(run_dir)
        if scores_dir is None:
            raise HTTPException(404, f"No *_clustered_scores.json found under {run!r} (checked scores/ and scores_v2/)")
        clusters: dict = {}
        for sf in sorted(scores_dir.glob("*_clustered_scores.json")):
            clusters.update(json.load(open(sf)))
        tsne = _load_cluster_tsne(run)
        body = build_cluster_html(clusters, tsne, video_base="/video/", frame_base="/frame/", run_label=run)
        cluster_html_cache[run] = (stamp, body)
        n_spans = sum(len(c.get("spans", {})) for c in clusters.values())
        print(f"[viewer] cluster HTML {len(body)/1e6:.1f} MB run={run} ({len(clusters)} clusters, {n_spans} spans, scores_dir={scores_dir.name}, tsne={'yes' if tsne else 'no'})")
        return body

    @web.get("/view_clusters", response_class=HTMLResponse)
    def view_clusters(run: str = Query(..., description="Volume-relative lang-cluster run path")):
        try:
            return _build_cluster_cached(run)
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
        n_runs         = len(_list_curation_runs(Path(OUTPUTS_MOUNT)))
        n_span_runs    = len(_list_span_runs(Path(OUTPUTS_MOUNT)))
        n_cluster_runs = len(_list_cluster_runs(Path(OUTPUTS_MOUNT)))
        n_mp4          = len(list(Path(PREVIEW_MOUNT).glob("*.mp4")))
        return f"ok runs={n_runs} span_runs={n_span_runs} cluster_runs={n_cluster_runs} mp4s={n_mp4} cached_html={len(html_cache)}"

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

        # Fast path: pre-rendered static JPEG (see FRAME_STRIDE docstring).
        idx = round(frame_num / FRAME_STRIDE) + 1
        jpg = Path(PREVIEW_MOUNT) / "frames" / safe / f"{idx:06d}.jpg"
        if not jpg.exists():
            _reload_volumes()
        if jpg.exists():
            return FileResponse(
                str(jpg), media_type="image/jpeg",
                headers={"Cache-Control": "public, max-age=86400"},
            )

        # Fallback: on-demand ffmpeg extraction (runs not yet pre-rendered).
        cache_key = f"{safe}_{frame_num}"
        if cache_key in frame_cache:
            return Response(frame_cache[cache_key], media_type="image/jpeg")

        path = Path(PREVIEW_MOUNT) / f"{safe}.mp4"
        if not path.exists():
            _reload_volumes()
        if not path.exists():
            return PlainTextResponse("not found", status_code=404)

        # Input seeking (-ss before -i) is the fast keyframe-accurate path; downscale
        # to ≤640px (preview popup is 400px, thumbs ~320px) so we don't decode/encode/
        # ship a full-res frame; -nostdin/-loglevel trim per-process overhead.
        result = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error",
             "-ss", str(frame_num / 30.0), "-i", str(path),
             "-frames:v", "1", "-vf", "scale='min(640,iw)':-2",
             "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
            capture_output=True,
        )
        if result.returncode != 0 or not result.stdout:
            return PlainTextResponse("frame extraction failed", status_code=500)

        frame_cache[cache_key] = result.stdout
        return Response(
            result.stdout, media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    return web
