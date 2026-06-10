#!/usr/bin/env python3
"""
Build a self-contained interactive latent + episode-video viewer.

Two pages in one HTML file (no server needed):

  * **t-SNE 3-D** — per-task 3-D scatters of state/action latents from
    ``tsne3d_<task>.json`` (written by curateModal to ``<run>/tsne3d/``):
    task dropdown, state|action panels, same episode = same color in both,
    darker = later frame, click a point → episode hash + frame index,
    legend click isolates an episode across both panels.
  * **Video grid** — per-task grid of every scored episode (from
    ``scores_by_task.json``), sorted by MI score: rank, score, percentile bar,
    top-60% / bottom-40% badge, optional VAL badge. Each card click-loads the
    episode's MP4 inline (native <video>, served by the self-hosted Modal
    viewer egomimic/modal/episode_preview.py) and offers an "open ↗" link.

Usage (local dirs/files):
  python egomimic/scripts/build_latent_viz.py /path/to/tsne3d \\
      --scores /path/to/scores_by_task.json --out latent_viz.html

Usage (volume paths — auto-downloads tsne3d/ and the run's scores_by_task.json):
  python egomimic/scripts/build_latent_viz.py \\
      deminf_tsne/<run>/tsne3d --volume egoverse-training-outputs --out latent_viz.html
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

# Self-hosted Modal MP4 viewer (egomimic/modal/episode_preview.py) — serves
# raw H.264 MP4s with range support, so the grid embeds native <video> players.
VIDEO_BASE = "https://mecka-robotics--egoverse-episode-preview-viewer.modal.run/video/"
# Encode rate used by episode_preview.py (frame index / FPS = seek time).
FPS = 30

_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EgoVerse latent + episode viewer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; background: #111; color: #eee; }
  #bar { padding: 10px 16px; display: flex; gap: 14px; align-items: center; background: #1c1c1c; flex-wrap: wrap; }
  #bar select { font-size: 15px; padding: 4px 8px; }
  #info { font-size: 13px; padding: 6px 12px; background: #222; border-radius: 6px; min-width: 380px; }
  #info b { color: #7fd4ff; }
  .tab { padding: 6px 14px; border-radius: 6px; cursor: pointer; background: #2a2a2a; user-select: none; }
  .tab.active { background: #2f6fb3; }
  #plots { display: flex; }
  .panel { width: 50vw; height: calc(100vh - 60px); }
  h3 { margin: 0; font-weight: 600; }
  /* ---- video grid ---- */
  #gridpage { display: none; padding: 14px 18px; }
  #gridhead { margin: 4px 0 12px; color: #bbb; font-size: 14px; display:flex; gap:16px; align-items:center;}
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; }
  .card { background: #1b1b1b; border: 1px solid #2c2c2c; border-radius: 10px; overflow: hidden; }
  .card .hdr { display: flex; align-items: center; gap: 8px; padding: 8px 10px; font-size: 13px; }
  .rank { color: #888; min-width: 34px; }
  .hash { font-family: ui-monospace, monospace; color: #9ecbff; }
  .score { margin-left: auto; font-weight: 700; }
  .badge { font-size: 11px; padding: 2px 7px; border-radius: 10px; font-weight: 700; }
  .b-top { background: #1d4d2b; color: #7be495; }
  .b-bot { background: #4d1d1d; color: #ff9d9d; }
  .b-val { background: #4d3d1d; color: #ffd97b; }
  .pct { height: 4px; background: #333; }
  .pct > div { height: 100%; background: linear-gradient(90deg, #e05555, #e0c14f, #54c46c); }
  .vid { position: relative; aspect-ratio: 16/10; background: #0d0d0d; }
  .vid iframe { width: 100%; height: 100%; border: 0; }
  .vid .ph { position: absolute; inset: 0; display: flex; flex-direction: column; gap: 8px; align-items: center; justify-content: center; cursor: pointer; color: #999; }
  .vid .ph:hover { color: #fff; background: #161616; }
  .ph .play { font-size: 34px; }
  .links { padding: 7px 10px; font-size: 12px; display: flex; gap: 14px; }
  .links a { color: #7fd4ff; text-decoration: none; }
  button { background: #2a2a2a; color: #ddd; border: 1px solid #3a3a3a; border-radius: 6px; padding: 5px 10px; cursor: pointer; }
  /* ---- click-to-frame preview (t-SNE page) ---- */
  #preview { position: fixed; right: 16px; bottom: 16px; width: 440px; background: #181818;
             border: 1px solid #333; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,.6);
             display: none; z-index: 50; overflow: hidden; }
  #preview video { width: 100%; display: block; background: #000; }
  #pv-cap { font-size: 12px; padding: 7px 10px; color: #ccc; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  #pv-cap b { color: #7fd4ff; }
  #pv-cap button { padding: 2px 8px; font-size: 12px; }
  #pv-close { position: absolute; top: 6px; right: 8px; cursor: pointer; color: #aaa; font-size: 16px;
              background: rgba(0,0,0,.5); border-radius: 50%; width: 22px; height: 22px; text-align: center; line-height: 22px; }
  #pv-close:hover { color: #fff; }
</style>
</head>
<body>
<div id="bar">
  <h3>EgoVerse viewer</h3>
  <span class="tab active" id="tab-tsne" onclick="showPage('tsne')">t-SNE 3-D</span>
  <span class="tab" id="tab-grid" onclick="showPage('grid')">Video grid</span>
  <label>Task: <select id="task"></select></label>
  <div id="info">t-SNE: click a point to inspect. Grid: click ▶ to load a video.</div>
</div>
<div id="tsnepage">
  <div id="plots">
    <div id="state" class="panel"></div>
    <div id="action" class="panel"></div>
  </div>
</div>
<div id="gridpage">
  <div id="gridhead"></div>
  <div id="grid"></div>
</div>
<div id="preview">
  <div id="pv-close" onclick="hidePreview()">✕</div>
  <video id="pv-video" muted playsinline preload="metadata"></video>
  <div id="pv-cap"></div>
</div>
<script>
const DATA = __DATA__;
const SCORES = __SCORES__;
const VAL = new Set(__VAL__);
const VIDEO_BASE = "__VIDEO_BASE__";

function hsv2rgb(h, s, v) {
  const i = Math.floor(h * 6), f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  const c = [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i % 6];
  return `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;
}

/* ---------------- click-to-frame preview ---------------- */
const FPS = __FPS__;
let pvHash = null, pvTask = "", pvFrame = 0, pvTfrac = null;

function showFrame(task, hash, frame, tfrac) {
  pvTask = task; pvFrame = frame; pvTfrac = tfrac;
  const v = document.getElementById("pv-video");
  document.getElementById("preview").style.display = "block";
  const seek = () => { v.pause(); v.currentTime = pvFrame / FPS; };
  if (pvHash !== hash) {
    pvHash = hash;
    v.src = VIDEO_BASE + hash;
    v.onloadedmetadata = seek;
  } else {
    seek();
  }
  updateCap();
}

function updateCap() {
  document.getElementById("pv-cap").innerHTML =
    `<b>${pvHash ? pvHash.slice(0,12) : ""}</b> · ${pvTask} · frame <b>${pvFrame}</b>` +
    (pvTfrac != null ? ` (${Math.round(pvTfrac*100)}%)` : "") +
    ` <button onclick="stepFrame(-1)">−1f</button>` +
    ` <button onclick="stepFrame(1)">+1f</button>` +
    ` <button onclick="togglePlay()">▶/⏸</button>` +
    ` <a href="${VIDEO_BASE}${pvHash}" target="_blank" style="color:#7fd4ff">open ↗</a>` +
    ` <span style="color:#666">frame = curation-sequence index; pause-filtered eps may be offset</span>`;
}

function stepFrame(d) {
  const v = document.getElementById("pv-video");
  pvFrame = Math.max(0, pvFrame + d);
  pvTfrac = null;
  v.pause();
  v.currentTime = pvFrame / FPS;
  updateCap();
}

function togglePlay() {
  const v = document.getElementById("pv-video");
  if (v.paused) { v.play(); }
  else { v.pause(); pvFrame = Math.round(v.currentTime * FPS); pvTfrac = null; updateCap(); }
}

function hidePreview() {
  const v = document.getElementById("pv-video");
  v.pause();
  document.getElementById("preview").style.display = "none";
}

/* ---------------- t-SNE page ---------------- */
function tracesFor(mod, task) {
  const d = (DATA[task] || {})[mod];
  if (!d) return [];
  const eps = DATA[task].episodes, nEp = eps.length;
  const byEp = {};
  for (let k = 0; k < d.x.length; k++) {
    const e = d.ep[k];
    if (!byEp[e]) byEp[e] = {x:[], y:[], z:[], c:[], f:[], t:[]};
    const b = byEp[e];
    b.x.push(d.x[k]); b.y.push(d.y[k]); b.z.push(d.z[k]);
    b.c.push(hsv2rgb(e / Math.max(1, nEp), 0.85, 1.0 - 0.65 * d.t[k]));
    b.f.push(d.frame[k]); b.t.push(d.t[k]);
  }
  return Object.keys(byEp).sort((a,b)=>a-b).map(e => {
    const b = byEp[e];
    return {
      type: "scatter3d", mode: "markers",
      name: eps[e].slice(0, 10),
      meta: {hash: eps[e], mod: mod},
      x: b.x, y: b.y, z: b.z,
      customdata: b.f.map((fr, i) => [fr, b.t[i]]),
      marker: {size: 3, color: b.c},
      hovertemplate: "ep " + eps[e].slice(0,10) +
        " · frame %{customdata[0]} (t=%{customdata[1]:.0%})<extra></extra>",
    };
  });
}

const LAYOUT = (title) => ({
  title: {text: title, font: {color: "#eee", size: 15}},
  paper_bgcolor: "#111", plot_bgcolor: "#111",
  scene: { xaxis: {visible: false}, yaxis: {visible: false}, zaxis: {visible: false}, bgcolor: "#111" },
  legend: {font: {color: "#ccc", size: 10}, itemsizing: "constant"},
  margin: {l: 0, r: 0, t: 36, b: 0},
});

function renderTsne(task) {
  Plotly.react("state",  tracesFor("state", task),  LAYOUT("STATE — " + task),  {responsive: true});
  Plotly.react("action", tracesFor("action", task), LAYOUT("ACTION — " + task), {responsive: true});
  for (const div of ["state", "action"]) {
    const el = document.getElementById(div);
    el.on("plotly_click", ev => {
      const p = ev.points[0];
      const [frame, tfrac] = p.customdata;
      document.getElementById("info").innerHTML =
        `<b>${p.data.meta.mod.toUpperCase()}</b> · task <b>${task}</b> · ` +
        `episode <b>${p.data.meta.hash}</b> · frame <b>${frame}</b> ` +
        `(${Math.round(tfrac * 100)}% through) · ` +
        `<a href="${VIDEO_BASE}${p.data.meta.hash}" target="_blank" style="color:#7fd4ff">video ↗</a>`;
      showFrame(task, p.data.meta.hash, frame, tfrac);
    });
    el.on("plotly_legendclick", ev => {
      const name = ev.data[ev.curveNumber].name;
      const alreadyIsolated = ev.data.every(
        tr => tr.name === name ? tr.visible !== "legendonly" : tr.visible === "legendonly"
      );
      for (const d2 of ["state", "action"]) {
        const el2 = document.getElementById(d2);
        const vis = el2.data.map(tr => alreadyIsolated || tr.name === name ? true : "legendonly");
        Plotly.restyle(el2, {visible: vis}, el2.data.map((_, i) => i));
      }
      return false;
    });
  }
}

/* ---------------- video grid page ---------------- */
function loadVideo(cellId, hash) {
  const cell = document.getElementById(cellId);
  cell.innerHTML = `<video src="${VIDEO_BASE}${hash}" controls loop muted playsinline preload="metadata" style="width:100%;height:100%;object-fit:contain;background:#000"></video>`;
}

function renderGrid(task) {
  const entries = SCORES[task] || [];           // [[hash, score], ...] sorted desc
  const n = entries.length;
  const nTop = Math.ceil(0.6 * n);
  const scores = entries.map(e => e[1]);
  const mn = Math.min(...scores), mx = Math.max(...scores);
  const mean = scores.reduce((a,b)=>a+b,0) / Math.max(1,n);

  document.getElementById("gridhead").innerHTML =
    `<b>${task}</b> — ${n} episodes · MI mean ${mean.toFixed(3)} · range [${mn.toFixed(3)}, ${mx.toFixed(3)}] · sorted best→worst ` +
    `<button onclick="loadAll()">Load all videos</button>` +
    `<span style="color:#777">▶ streams the episode MP4 inline (self-hosted viewer)</span>`;

  const cards = entries.map(([hash, score], i) => {
    const pct = mx > mn ? (score - mn) / (mx - mn) : 0.5;
    const badge = i < nTop ? '<span class="badge b-top">TOP 60%</span>'
                           : '<span class="badge b-bot">BOT 40%</span>';
    const valb = VAL.has(hash) ? ' <span class="badge b-val">VAL</span>' : '';
    const cid = `v_${task}_${i}`;
    return `<div class="card">
      <div class="hdr"><span class="rank">#${i+1}</span>
        <span class="hash">${hash}</span>${badge}${valb}
        <span class="score">${score.toFixed(4)}</span></div>
      <div class="pct"><div style="width:${Math.round(pct*100)}%"></div></div>
      <div class="vid" id="${cid}">
        <div class="ph" onclick="loadVideo('${cid}','${hash}')">
          <div class="play">▶</div><div>load video</div></div>
      </div>
      <div class="links">
        <a href="${VIDEO_BASE}${hash}" target="_blank">open ↗</a>
        <span style="color:#666">rank ${i+1}/${n} · percentile ${(100*(1-i/Math.max(1,n-1))).toFixed(0)}</span>
      </div>
    </div>`;
  });
  document.getElementById("grid").innerHTML = cards.join("");
}

function loadAll() {
  document.querySelectorAll("#grid .ph").forEach(ph => ph.click());
}

/* ---------------- nav ---------------- */
let page = "tsne";
function showPage(p) {
  page = p;
  document.getElementById("tsnepage").style.display = p === "tsne" ? "block" : "none";
  document.getElementById("gridpage").style.display = p === "grid" ? "block" : "none";
  document.getElementById("tab-tsne").classList.toggle("active", p === "tsne");
  document.getElementById("tab-grid").classList.toggle("active", p === "grid");
  render();
}

function render() {
  const task = document.getElementById("task").value;
  if (page === "tsne") renderTsne(task); else renderGrid(task);
}

const sel = document.getElementById("task");
const taskNames = Array.from(new Set([...Object.keys(SCORES), ...Object.keys(DATA)])).sort();
taskNames.forEach(t => {
  const o = document.createElement("option"); o.value = o.textContent = t; sel.appendChild(o);
});
sel.onchange = render;
render();
</script>
</body>
</html>
"""


def _download(volume: str, env: str, remote: str, dest: Path) -> bool:
    r = subprocess.run(
        ["modal", "volume", "get", "--env", env, "--force", volume, remote, str(dest)],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _load_tsne_dir(path: str, volume: str | None, env: str) -> Path:
    p = Path(path)
    if p.is_dir():
        return p
    if volume is None:
        sys.exit(f"Directory not found locally: {path}\nPass --volume to download from Modal.")
    tmp = Path(tempfile.mkdtemp(prefix="tsne3d_"))
    print(f"Downloading {path} from volume {volume} → {tmp} ...")
    subprocess.run(
        ["modal", "volume", "get", "--env", env, "--force", volume, path.rstrip("/"), str(tmp)],
        check=True,
    )
    nested = tmp / Path(path.rstrip("/")).name
    return nested if nested.is_dir() else tmp


def _load_scores(args, tsne_local: Path) -> dict:
    """Resolve scores_by_task.json: --scores path, sibling of tsne3d dir, or volume."""
    if args.scores:
        p = Path(args.scores)
        if p.is_file():
            return json.load(open(p))
        if args.volume:
            tmp = Path(tempfile.mktemp(suffix=".json"))
            if _download(args.volume, args.env, args.scores, tmp):
                return json.load(open(tmp))
        sys.exit(f"--scores not found: {args.scores}")
    sib = tsne_local.parent / "scores_by_task.json"
    if sib.is_file():
        return json.load(open(sib))
    if args.volume:
        remote = str(Path(args.tsne3d_dir.rstrip("/")).parent / "scores_by_task.json")
        tmp = Path(tempfile.mktemp(suffix=".json"))
        if _download(args.volume, args.env, remote, tmp):
            return json.load(open(tmp))
    print("WARNING: no scores_by_task.json found — video grid will be empty.")
    return {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsne3d_dir", help="Local dir of tsne3d_*.json, or volume-relative path")
    ap.add_argument("--scores", default=None, help="scores_by_task.json (local or volume-relative); default: sibling of tsne3d dir")
    ap.add_argument("--val-json", default="egomimic/hydra_configs/data/extra/mecka_d64_val.json",
                    help="Optional episode-hash list to badge as VAL in the grid")
    ap.add_argument("--volume", default=None, help="Modal volume to download from if paths not local")
    ap.add_argument("--env", default="robotics", help="Modal environment (default: robotics)")
    ap.add_argument("--out", default="latent_viz.html", help="Output HTML path")
    args = ap.parse_args()

    src = _load_tsne_dir(args.tsne3d_dir, args.volume, args.env)
    data: dict = {}
    for f in sorted(src.glob("tsne3d_*.json")):
        d = json.load(open(f))
        data[d["task"]] = d
    print(f"t-SNE: {len(data)} task(s)")

    raw_scores = _load_scores(args, src)
    # sorted [[hash, score], ...] per task: finite scores desc, NaN last
    scores = {
        t: sorted(
            ([h, s] for h, s in v.items()),
            key=lambda kv: (not math.isfinite(kv[1]), -(kv[1] if math.isfinite(kv[1]) else 0), kv[0]),
        )
        for t, v in raw_scores.items()
    }
    print(f"scores: {len(scores)} task(s), {sum(len(v) for v in scores.values())} episodes")

    val: list = []
    vp = Path(args.val_json) if args.val_json else None
    if vp and vp.is_file():
        val = json.load(open(vp))
        print(f"VAL badges: {len(val)} episodes from {vp}")

    html = (
        _TEMPLATE
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__SCORES__", json.dumps(scores, separators=(",", ":")))
        .replace("__VAL__", json.dumps(val, separators=(",", ":")))
        .replace("__VIDEO_BASE__", VIDEO_BASE)
        .replace("__FPS__", str(FPS))
    )
    out = Path(args.out)
    out.write_text(html)
    print(f"Wrote {out.resolve()}  ({out.stat().st_size/1e6:.1f} MB)")
    print("Open it in any browser — no server needed.")


if __name__ == "__main__":
    main()
