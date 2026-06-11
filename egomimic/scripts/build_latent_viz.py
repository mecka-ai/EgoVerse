#!/usr/bin/env python3
"""
Build a self-contained interactive latent + episode-video viewer.

Only input required is a directory of ``tsne3d_<task>.json`` files (written by
``egomimic/modal/latentVizModal.py`` for any data config, or by a curation
run). Per-episode scores and a VAL-episode list are OPTIONAL extras — the
viewer renders fully without them.

Two pages in one HTML file (works as a local file or served by the Modal web
app ``egomimic/modal/latent_viz_app.py``):

  * **t-SNE 3-D** — per-task 3-D scatters of state/action latents from
    ``tsne3d_<task>.json``. Fast single-trace WebGL engine: per-task color
    tables (episode / time / score) are precomputed once and every control
    change is an in-place restyle (no scene rebuild). Tools: frame-number
    highlight (±window) across BOTH panels, time-range filter, episode
    isolate, point size, synced cameras, click → frame-seek video preview +
    gold cross-highlight of the same (episode, frame) in the other panel.
  * **Video grid** — per-task grid of every episode. With scores: sort by
    score, rank/percentile, histogram. Always: search, VAL badges (when a VAL
    list is given), Load-all / Play-all / Pause-all / playback-speed. MP4s
    stream from the self-hosted Modal viewer
    (egomimic/modal/episode_preview.py).

Usage (local dirs/files):
  python egomimic/scripts/build_latent_viz.py /path/to/tsne3d \\
      [--scores /path/to/scores_by_task.json] --out latent_viz.html

Usage (volume paths — auto-downloads tsne3d/ and, if present, the run's
scores_by_task.json):
  python egomimic/scripts/build_latent_viz.py \\
      latent_viz/<run>/tsne3d --volume egoverse-training-outputs --out latent_viz.html
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
<title>EgoVerse latent viewer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root { --bg:#101014; --bar:#1a1b21; --card:#1c1d24; --line:#2b2d36; --acc:#3b82f6; --txt:#e8e8ea; }
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; background: var(--bg); color: var(--txt); }
  #bar { padding: 10px 16px; display: flex; gap: 14px; align-items: center; background: var(--bar); flex-wrap: wrap;
         border-bottom: 1px solid var(--line); }
  select, input[type=number], input[type=text] { font-size: 13px; padding: 4px 8px; background:#26272f; color:var(--txt);
         border:1px solid var(--line); border-radius:6px; }
  input[type=number] { width: 70px; } input[type=text] { width: 130px; }
  input[type=range] { accent-color: var(--acc); }
  #info { font-size: 13px; padding: 6px 12px; background: #222; border-radius: 6px; min-width: 320px; }
  #info b { color: #7fd4ff; }
  .tab { padding: 6px 16px; border-radius: 8px; cursor: pointer; background: #2a2b33; user-select: none; font-weight:600; }
  .tab.active { background: var(--acc); }
  #tools, #gtools { padding: 8px 16px; display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
           background: #15161b; border-bottom: 1px solid var(--line); font-size: 13px; color:#bbb; }
  .tool { display: flex; gap: 7px; align-items: center; }
  .tool label { color:#9aa; }
  #plots { display: flex; }
  .panel { width: calc((100vw - 180px) / 2); height: calc(100vh - 104px); }
  #legend { width: 180px; height: calc(100vh - 104px); overflow-y: auto; background: #14151a;
            border-left: 1px solid var(--line); font-size: 12px; }
  #legend .lg-head { padding: 8px 10px; color: #9aa; position: sticky; top: 0; background: #14151a; }
  .lg-row { display: flex; align-items: center; gap: 7px; padding: 4px 10px; cursor: pointer;
            font-family: ui-monospace, monospace; color: #cdd; white-space: nowrap; }
  .lg-row:hover { background: #1f2027; }
  .lg-row.active { background: #24344f; }
  .lg-dot { width: 11px; height: 11px; border-radius: 50%; flex: none; }
  h3 { margin: 0; font-weight: 700; letter-spacing:.3px; }
  button { background: #2a2b33; color: #ddd; border: 1px solid var(--line); border-radius: 7px; padding: 5px 11px; cursor: pointer; }
  button:hover { background:#33353f; }
  button.primary { background: var(--acc); border-color: var(--acc); color:#fff; }
  .chk { display:flex; align-items:center; gap:4px; cursor:pointer; user-select:none; }
  /* ---- video grid ---- */
  #gridpage { display: none; }
  #gridwrap { padding: 14px 18px; }
  #gridhead { margin: 4px 0 12px; color: #bbb; font-size: 14px; display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 14px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
  .card .hdr { display: flex; align-items: center; gap: 8px; padding: 8px 10px; font-size: 13px; }
  .rank { color: #888; min-width: 34px; }
  .hash { font-family: ui-monospace, monospace; color: #9ecbff; }
  .score { margin-left: auto; font-weight: 700; }
  .badge { font-size: 11px; padding: 2px 7px; border-radius: 10px; font-weight: 700; }
  .b-val { background: #4d3d1d; color: #ffd97b; }
  .pct { height: 4px; background: #333; }
  .pct > div { height: 100%; background: linear-gradient(90deg, #e05555, #e0c14f, #54c46c); }
  .vid { position: relative; aspect-ratio: 16/10; background: #0a0a0c; }
  .vid video { width:100%; height:100%; object-fit:contain; background:#000; }
  .vid .ph { position: absolute; inset: 0; display: flex; flex-direction: column; gap: 8px; align-items: center; justify-content: center; cursor: pointer; color: #999; }
  .vid .ph:hover { color: #fff; background: #15161b; }
  .ph .play { font-size: 34px; }
  .links { padding: 7px 10px; font-size: 12px; display: flex; gap: 14px; }
  .links a { color: #7fd4ff; text-decoration: none; }
  #histo { height: 30px; }
  /* ---- click-to-frame preview ---- */
  #preview { position: fixed; right: 16px; bottom: 16px; width: 440px; background: #17181d;
             border: 1px solid #34363f; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,.65);
             display: none; z-index: 50; overflow: hidden; }
  #preview video { width: 100%; display: block; background: #000; }
  #pv-cap { font-size: 12px; padding: 7px 10px; color: #ccc; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  #pv-cap b { color: #7fd4ff; }
  #pv-cap button { padding: 2px 8px; font-size: 12px; }
  #pv-close { position: absolute; top: 6px; right: 8px; cursor: pointer; color: #aaa; font-size: 16px;
              background: rgba(0,0,0,.5); border-radius: 50%; width: 22px; height: 22px; text-align: center; line-height: 22px; }
  #pv-close:hover { color: #fff; }
  /* ---- cluster / prune tuning ---- */
  #ctools { padding: 8px 16px; display: flex; gap: 18px; align-items: center; flex-wrap: wrap;
            background: #131419; border-bottom: 1px solid var(--line); font-size: 13px; color: #bbb; }
  #cstats { color: #9aa; } #cstats b { color: #ffd97b; }
  #clpanel { border-bottom: 1px solid var(--line); padding-bottom: 6px; margin-bottom: 4px; }
  .cl-row { display: flex; align-items: center; gap: 6px; padding: 3px 8px; font-size: 11px;
            font-family: ui-monospace, monospace; color: #cdd; white-space: nowrap; }
  .cl-row select { font-size: 11px; padding: 1px 3px; margin-left: auto; }
  .cl-row .warn { color: #e0a14f; }
  .lg-row.removed { text-decoration: line-through; opacity: .5; }
  .b-rm { background: #4d1d1d; color: #ff9b7b; }
  .b-noemb { background: #333; color: #aaa; }
  .b-ovr { background: #1d3d4d; color: #7bd4ff; }
  .ovr-btns button { padding: 1px 6px; font-size: 11px; margin-left: 3px; }
  .ovr-btns button.on { background: var(--acc); color: #fff; border-color: var(--acc); }
  .ovr-btns button.bad.on { background: #c0392b; border-color: #c0392b; }
</style>
</head>
<body>
<div id="bar">
  <h3>EgoVerse viewer</h3>
  <span class="tab active" id="tab-tsne" onclick="showPage('tsne')">t-SNE 3-D</span>
  <span class="tab" id="tab-grid" onclick="showPage('grid')">Video grid</span>
  <label class="tool">Task <select id="task"></select></label>
  <div id="info">Click a point to inspect</div>
</div>

<div id="ctools">
  <span class="tool"><label>k</label>
    <input type="number" id="ck" value="8" min="2" max="40" style="width:52px">
    <button onclick="clusterNow()" title="k-means on the t-SNE points; re-clustering resets cluster priorities">cluster</button></span>
  <span class="tool"><label>remove %</label>
    <input type="range" id="prunePct" min="0" max="100" value="0" step="0.5" style="width:140px" onchange="pruneChanged()">
    <input type="number" id="prunePctN" value="0" min="0" max="100" step="0.5" style="width:64px" onchange="pruneChangedN()"></span>
  <span class="tool"><label>target keep h</label>
    <input type="number" id="targetH" step="0.1" min="0" style="width:70px" onchange="targetHours()"></span>
  <span class="tool"><label class="chk"><input type="checkbox" id="allowProt" onchange="pruneChanged()"> can remove protected</label></span>
  <span id="cstats">no prune selection</span>
  <button onclick="exportKeepList()" title="plain hash list — eps_to_use-ready, all tasks (untouched tasks keep everything)">⬇ keep-list</button>
  <button onclick="exportSelection()" title="full provenance: clusters, priorities, overrides, removal order">⬇ selection</button>
  <button onclick="document.getElementById('selFile').click()">import</button>
  <input type="file" id="selFile" accept=".json,application/json" style="display:none" onchange="importSelectionFile(this)">
</div>

<div id="tsnepage">
  <div id="tools">
    <span class="tool"><label>Color</label>
      <select id="colorMode" onchange="applyStyle()">
        <option value="episode">episode</option>
        <option value="time">time (light→dark)</option>
        <option value="score">score (red→green)</option>
        <option value="cluster">cluster</option>
      </select></span>
    <span class="tool"><label>Episode</label>
      <select id="epSel" onchange="applyStyle(); markLegend()"><option value="all">all</option></select></span>
    <span class="tool"><label class="chk"><input type="checkbox" id="hlOn" onchange="applyStyle()"> highlight frame</label>
      <input type="number" id="hlFrame" value="0" min="0" step="10" onchange="hlChanged()">
      <label>±</label><input type="number" id="hlWin" value="15" min="0" step="5" onchange="hlChanged()"></span>
    <span class="tool"><label>Time range %</label>
      <input type="number" id="t0" value="0" min="0" max="100" onchange="applyStyle()">–
      <input type="number" id="t1" value="100" min="0" max="100" onchange="applyStyle()"></span>
    <span class="tool"><label>Size</label>
      <input type="range" id="psize" min="1" max="8" value="3" onchange="applyStyle()"></span>
    <span class="tool"><label class="chk"><input type="checkbox" id="syncCam" checked> sync cameras</label></span>
    <button onclick="resetTools()">reset</button>
    <span id="tstats" style="color:#777"></span>
  </div>
  <div id="plots">
    <div id="state" class="panel"></div>
    <div id="action" class="panel"></div>
    <div id="legend"></div>
  </div>
</div>

<div id="gridpage">
  <div id="gtools">
    <span class="tool" id="gsortWrap"><label>Sort</label>
      <select id="gsort" onchange="render()">
        <option value="best">score (best first)</option>
        <option value="worst">score (worst first)</option>
        <option value="removal">removal order</option>
      </select></span>
    <span class="tool"><label>Show</label>
      <select id="fRemoved" onchange="render()">
        <option value="all">all</option>
        <option value="kept">kept</option>
        <option value="removed">removed</option>
        <option value="boundary">cut boundary</option>
      </select></span>
    <span class="tool" id="fValWrap">
      <label class="chk"><input type="checkbox" id="fVal" onchange="render()"> VAL only</label></span>
    <span class="tool"><label>Find</label><input type="text" id="gsearch" placeholder="hash prefix…" oninput="render()"></span>
    <button onclick="loadAll()">Load all</button>
    <button class="primary" onclick="playAll()">▶ Play all</button>
    <button onclick="pauseAll()">⏸ Pause all</button>
    <span class="tool"><label>Speed</label>
      <select id="gspeed" onchange="setSpeed()">
        <option>0.5</option><option selected>1</option><option>2</option><option>4</option>
      </select></span>
  </div>
  <div id="gridwrap">
    <div id="gridhead"></div>
    <div id="grid"></div>
  </div>
</div>

<div id="preview">
  <div id="pv-close" onclick="hidePreview()">✕</div>
  <video id="pv-video" muted playsinline preload="metadata"></video>
  <div id="pv-cap"></div>
</div>

<script>
const DATA = __DATA__;
const SCORES = __SCORES__;
const SCORES_META = __SCORES_META__;
const UNIVERSE = __UNIVERSE__;
const MANIFEST = __MANIFEST__;
const VAL = new Set(__VAL__);
const VIDEO_BASE = "__VIDEO_BASE__";
const FPS = __FPS__;
const DIM = "rgba(110,110,120,0.05)";
const REMOVED_COLOR = "rgba(225,60,60,0.30)";
// Score direction: knn grading is higher = worse; DemInf KSG is higher = better.
const HIGHER_WORSE = !!SCORES_META.higher_is_worse;
const SCORE_LABEL = (SCORES_META.metric || "score") + (HIGHER_WORSE ? " (higher = worse)" : " (higher = better)");

/* ---------------- color helpers ---------------- */
function hsv2rgb(h, s, v) {
  const i = Math.floor(h * 6), f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  const c = [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i % 6];
  return [c[0]*255, c[1]*255, c[2]*255];
}
function rgb(c) { return `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`; }
function lerp3(a, b, t) { return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t]; }
const redGreen = t => lerp3([224,85,85], [84,196,108], t);

const SCOREMAP = {};   // task -> {hash: raw score}
function scoreMap(task) {
  if (SCOREMAP[task]) return SCOREMAP[task];
  const m = {};
  (SCORES[task] || []).forEach(([h, s]) => m[h] = s);
  return (SCOREMAP[task] = m);
}

const NORM = {};       // task -> {hash: goodness in [0,1]}, 1 = best
function scoreNorm(task) {
  // Percentile (midrank) normalization, direction-corrected to GOODNESS:
  // real grading scores are heavy-tailed with many exact ties — min-max
  // compresses all resolution into one end. Non-finite scores are left out
  // and fall through to the 0.5 mid-color default.
  if (NORM[task]) return NORM[task];
  const e = (SCORES[task] || []).filter(x => Number.isFinite(x[1]));
  if (!e.length) return (NORM[task] = {});
  const sorted = e.map(x => x[1]).sort((a, b) => a - b);
  const out = {};
  for (const [h, s] of e) {
    let lo = 0, hi = sorted.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] < s) lo = mid + 1; else hi = mid; }
    let lo2 = lo, hi2 = sorted.length;
    while (lo2 < hi2) { const mid = (lo2 + hi2) >> 1; if (sorted[mid] <= s) lo2 = mid + 1; else hi2 = mid; }
    const pct = sorted.length === 1 ? 0.5 : (lo + lo2 - 1) / 2 / (sorted.length - 1);
    out[h] = HIGHER_WORSE ? 1 - pct : pct;
  }
  return (NORM[task] = out);
}
function hasScores(task) { return (SCORES[task] || []).length > 0; }
function fmtH(frames) { return (frames / FPS / 3600).toFixed(2) + " h"; }

/* ---------------- t-SNE fast engine ----------------
   One trace per panel. Per (task, mode) color tables are computed ONCE and
   cached; every control change only re-picks colors[k] = cached[k] | DIM and
   restyles in place — no scene/trace rebuild, no camera reset.            */
const CACHE = {};   // task -> {mods: {state:{d, tables:{episode,time,score}}, action:{...}}}
let curTask = null;

function buildCache(task) {
  // Tasks can exist in SCORES without a tsne3d JSON (skipped exports, old
  // curation runs) — degrade to empty panels instead of throwing.
  const eps = (DATA[task] || {}).episodes || [], nEp = eps.length;
  const sn = scoreNorm(task);
  const mods = {};
  for (const mod of ["state", "action"]) {
    const d = (DATA[task] || {})[mod];
    if (!d) continue;
    const N = d.x.length;
    const tables = {episode: new Array(N), time: new Array(N), score: new Array(N)};
    const custom = new Array(N);
    for (let k = 0; k < N; k++) {
      const e = d.ep[k], tf = d.t[k];
      tables.episode[k] = rgb(hsv2rgb(e / Math.max(1, nEp), 0.85, 1.0 - 0.65 * tf));
      tables.time[k]    = rgb(lerp3([170,215,255], [10,40,90], tf));
      tables.score[k]   = rgb(redGreen(sn[eps[e]] ?? 0.5));
      custom[k] = [d.frame[k], Math.round(tf * 100), eps[e].slice(0, 10), e];
    }
    mods[mod] = {d, tables, custom};
  }
  return {mods};
}

function uiState() {
  return {
    mode: document.getElementById("colorMode").value,
    ep: document.getElementById("epSel").value,
    hlOn: document.getElementById("hlOn").checked,
    hlF: +document.getElementById("hlFrame").value,
    hlW: +document.getElementById("hlWin").value,
    t0: +document.getElementById("t0").value / 100,
    t1: +document.getElementById("t1").value / 100,
    size: +document.getElementById("psize").value,
  };
}

function applyStyle() {
  if (!curTask || !CACHE[curTask]) return;
  const u = uiState();
  const st = CL[curTask];
  const rmEp = st && st.removedEp;               // Uint8Array per episode index
  for (const mod of ["state", "action"]) {
    const m = CACHE[curTask].mods[mod];
    if (!m) continue;
    const {d, tables} = m;
    let base = tables[u.mode];
    if (u.mode === "cluster") {
      base = (st && st.clTable && st.clTable[mod]) || tables.episode;
    }
    const N = d.x.length;
    const colors = new Array(N);
    let sizes = u.size;                          // scalar unless highlighting
    if (u.hlOn) sizes = new Array(N);
    for (let k = 0; k < N; k++) {
      const on = d.t[k] >= u.t0 && d.t[k] <= u.t1
        && (u.ep === "all" || d.ep[k] === +u.ep)
        && (!u.hlOn || Math.abs(d.frame[k] - u.hlF) <= u.hlW);
      // Filters narrow the view (DIM); the prune preview shows WITHIN the
      // view which episodes the current cut would drop (red ghost).
      colors[k] = !on ? DIM : (rmEp && rmEp[d.ep[k]] ? REMOVED_COLOR : base[k]);
      if (u.hlOn) sizes[k] = on ? u.size * 2.2 : u.size;
    }
    Plotly.restyle(mod, {"marker.color": [colors], "marker.size": Array.isArray(sizes) ? [sizes] : sizes}, [0]);
  }
}

const LAYOUT = (title) => ({
  title: {text: title, font: {color: "#eee", size: 15}},
  paper_bgcolor: "#101014", plot_bgcolor: "#101014",
  scene: { xaxis: {visible: false}, yaxis: {visible: false}, zaxis: {visible: false}, bgcolor: "#101014" },
  showlegend: false,
  margin: {l: 0, r: 0, t: 36, b: 0},
});

let camLock = false;
function renderTsne(task) {
  if (!CACHE[task]) CACHE[task] = buildCache(task);
  curTask = task;
  // score color mode only makes sense when this task has scores
  const scoreOpt = document.querySelector('#colorMode option[value="score"]');
  scoreOpt.disabled = !hasScores(task);
  const clusterOpt = document.querySelector('#colorMode option[value="cluster"]');
  clusterOpt.disabled = !(DATA[task] && (DATA[task].state || DATA[task].action));
  const cm = document.getElementById("colorMode");
  if (cm.value === "score" && scoreOpt.disabled) cm.value = "episode";
  if (cm.value === "cluster" && clusterOpt.disabled) cm.value = "episode";
  syncSelectionUI(task);
  const u = uiState();
  // episode dropdown
  const epSel = document.getElementById("epSel");
  const eps = DATA[task] ? DATA[task].episodes : [];
  epSel.innerHTML = '<option value="all">all</option>' +
    eps.map((h, i) => `<option value="${i}">${h.slice(0,10)}</option>`).join("");

  for (const mod of ["state", "action"]) {
    const m = CACHE[task].mods[mod];
    const el = document.getElementById(mod);
    if (!m) { Plotly.purge(el); el.innerHTML = ""; continue; }
    const {d, tables, custom} = m;
    Plotly.newPlot(mod, [
      { type: "scatter3d", mode: "markers",
        x: d.x, y: d.y, z: d.z, customdata: custom,
        marker: {size: u.size, color: tables[u.mode], line: {width: 0}},   // solid, no stroke
        hovertemplate: "ep %{customdata[2]} · frame %{customdata[0]} (t=%{customdata[1]}%)<extra></extra>" },
      { type: "scatter3d", mode: "markers", name: "selected",
        x: [], y: [], z: [], hoverinfo: "skip",
        marker: {size: 13, color: "rgba(255,200,0,0.95)", symbol: "diamond", line: {width: 0}} },
    ], LAYOUT((mod === "state" ? "STATE — " : "ACTION — ") + task), {responsive: true});

    el.removeAllListeners && el.removeAllListeners("plotly_click");
    // Plotly 3-D fires plotly_click when a rotate-drag happens to end on a
    // point. Track pointer travel and only accept near-stationary clicks, so
    // the video preview never loads mid-orbit.
    el.onpointerdown = e => { el._px = e.clientX; el._py = e.clientY; };
    el.onpointerup   = e => { el._drag = Math.hypot(e.clientX - (el._px ?? e.clientX),
                                                    e.clientY - (el._py ?? e.clientY)) > 5; };
    el.on("plotly_click", ev => {
      if (el._drag) return;                       // rotation release, not a click
      const p = ev.points[0];
      if (p.curveNumber !== 0) return;
      const [frame, tpct, , epIdx] = p.customdata;
      const hash = DATA[task].episodes[epIdx];
      document.getElementById("info").innerHTML =
        `<b>${mod.toUpperCase()}</b> · <b>${hash}</b> · frame <b>${frame}</b> (${tpct}%) · ` +
        `<a href="${VIDEO_BASE}${hash}" target="_blank" style="color:#7fd4ff">video ↗</a>`;
      document.getElementById("hlFrame").value = frame;
      showFrame(task, hash, frame, tpct / 100);
      crossHighlight(task, epIdx, frame);
    });
    el.on("plotly_relayout", ev => {
      if (!document.getElementById("syncCam").checked || camLock) return;
      if (ev["scene.camera"]) {
        camLock = true;
        const other = mod === "state" ? "action" : "state";
        Plotly.relayout(other, {"scene.camera": ev["scene.camera"]}).then(() => camLock = false);
      }
    });
  }
  const d = DATA[task] || {};
  const nPts = ["state","action"].reduce((a,m) => a + ((d[m]||{}).x||[]).length, 0);
  document.getElementById("tstats").textContent =
    `${(d.episodes||[]).length} episodes · ${nPts.toLocaleString()} points · every ${d.every_n||10}th frame`;
  buildLegend(task);
  applyStyle();
}

/* episode color legend sidebar — swatch = episode hue; click isolates.
   When clusters exist, a cluster panel renders above the episode list. */
function buildLegend(task) {
  const eps = DATA[task] ? DATA[task].episodes : [];
  const sm = scoreMap(task);
  const st = CL[task];
  const rows = eps.map((h, i) => {
    const c = rgb(hsv2rgb(i / Math.max(1, eps.length), 0.85, 0.85));
    const sc = sm[h];
    const cl = st && st.epDomCl ? ` · cl ${st.epDomCl[i]} (${Math.round(100 * st.epDomFrac[i])}%)` : "";
    const tip = Number.isFinite(sc) ? `${h} · ${SCORE_LABEL}: ${sc.toFixed(4)}${cl}` : `${h}${cl}`;
    const removed = st && st.removedEp && st.removedEp[i] ? " removed" : "";
    return `<div class="lg-row${removed}" id="lg_${i}" onclick="legendClick(${i})" title="${tip}">` +
           `<span class="lg-dot" style="background:${c}"></span>${h.slice(0, 10)}</div>`;
  }).join("");
  document.getElementById("legend").innerHTML =
    clusterPanelHTML(task) +
    `<div class="lg-head">episodes (click = isolate)</div>` +
    `<div class="lg-row" id="lg_all" onclick="legendClick('all')"><span class="lg-dot" style="background:#888"></span>show all</div>` + rows;
  markLegend();
}

function legendClick(i) {
  const sel = document.getElementById("epSel");
  sel.value = (String(i) === sel.value) ? "all" : String(i);   // toggle
  applyStyle();
  markLegend();
}

function markLegend() {
  const cur = document.getElementById("epSel").value;
  document.querySelectorAll("#legend .lg-row").forEach(r => r.classList.remove("active"));
  const el = document.getElementById(cur === "all" ? "lg_all" : `lg_${cur}`);
  if (el) el.classList.add("active");
}

function crossHighlight(task, epIdx, frame) {
  for (const mod of ["state", "action"]) {
    const m = (CACHE[task] || {mods:{}}).mods[mod];
    if (!m) continue;
    const d = m.d;
    let xs = [], ys = [], zs = [];
    for (let k = 0; k < d.x.length; k++) {
      if (d.ep[k] === epIdx && d.frame[k] === frame) { xs.push(d.x[k]); ys.push(d.y[k]); zs.push(d.z[k]); break; }
    }
    Plotly.restyle(mod, {x: [xs], y: [ys], z: [zs]}, [1]);
  }
}

function hlChanged() { document.getElementById("hlOn").checked = true; applyStyle(); }

function resetTools() {
  document.getElementById("colorMode").value = "episode";
  document.getElementById("epSel").value = "all";
  document.getElementById("hlOn").checked = false;
  document.getElementById("hlFrame").value = 0;
  document.getElementById("hlWin").value = 15;
  document.getElementById("t0").value = 0;
  document.getElementById("t1").value = 100;
  document.getElementById("psize").value = 3;
  applyStyle();
}

/* ---------------- click-to-frame preview ---------------- */
let pvHash = null, pvFrame = 0, pvTfrac = null;

function showFrame(task, hash, frame, tfrac) {
  pvFrame = frame; pvTfrac = tfrac;
  const v = document.getElementById("pv-video");
  document.getElementById("preview").style.display = "block";
  const seek = () => { v.pause(); v.currentTime = pvFrame / FPS; };
  if (pvHash !== hash) {
    pvHash = hash;
    v.src = VIDEO_BASE + hash;
    v.onloadedmetadata = seek;
  } else { seek(); }
  updateCap();
}

function updateCap() {
  document.getElementById("pv-cap").innerHTML =
    `<b>${pvHash ? pvHash.slice(0,12) : ""}</b> · frame <b>${pvFrame}</b>` +
    (pvTfrac != null ? ` (${Math.round(pvTfrac*100)}%)` : "") +
    ` <button onclick="stepFrame(-1)">−1f</button>` +
    ` <button onclick="stepFrame(1)">+1f</button>` +
    ` <button onclick="togglePlay()">▶/⏸</button>` +
    ` <a href="${VIDEO_BASE}${pvHash}" target="_blank" style="color:#7fd4ff">open ↗</a>`;
}

function stepFrame(d) {
  const v = document.getElementById("pv-video");
  pvFrame = Math.max(0, pvFrame + d); pvTfrac = null;
  v.pause(); v.currentTime = pvFrame / FPS;
  updateCap();
}

function togglePlay() {
  const v = document.getElementById("pv-video");
  if (v.paused) v.play();
  else { v.pause(); pvFrame = Math.round(v.currentTime * FPS); pvTfrac = null; updateCap(); }
}

function hidePreview() {
  document.getElementById("pv-video").pause();
  document.getElementById("preview").style.display = "none";
}

/* ---------------- video grid page ---------------- */
function loadVideo(cellId, hash) {
  const cell = document.getElementById(cellId);
  cell.innerHTML = `<video src="${VIDEO_BASE}${hash}" controls loop muted playsinline preload="metadata"></video>`;
}

function histoSVG(scores) {
  if (!scores.length) return "";
  const mn = Math.min(...scores), mx = Math.max(...scores), nb = 24;
  const bins = new Array(nb).fill(0);
  scores.forEach(s => bins[Math.min(nb-1, Math.floor((s-mn)/((mx-mn)||1)*nb))]++);
  const bmax = Math.max(...bins);
  // Bin color is goodness: with higher_is_worse the high-value tail is red.
  const bars = bins.map((b,i) =>
    `<rect x="${i*8}" y="${28-26*b/bmax}" width="6" height="${26*b/bmax}" fill="${rgb(redGreen(HIGHER_WORSE ? 1 - i/(nb-1) : i/(nb-1)))}"/>`).join("");
  return `<svg id="histo" width="${nb*8}" height="30" title="score distribution">${bars}</svg>`;
}

function renderGrid(task) {
  const scored = hasScores(task);
  const st = CL[task];
  const gn = scoreNorm(task);
  const dEps = new Set((DATA[task] || {}).episodes || []);
  const uni = taskUniverse(task).u;

  // Entry universe = scored ∪ embedded ∪ resolved — the spot-check filters
  // must be able to show every episode the keep-list covers, including
  // unscored and failed-embed ones. Scored entries keep the build-time
  // best-first order; the rest append hash-sorted with null scores.
  let entries = scored ? SCORES[task].slice()
                       : ((DATA[task] || {}).episodes || []).map(h => [h, null]);
  {
    const have = new Set(entries.map(e => e[0]));
    Array.from(new Set([...dEps, ...Object.keys(uni)]))
      .filter(h => !have.has(h)).sort()
      .forEach(h => entries.push([h, null]));
  }
  const n = entries.length;
  const rank = {}; entries.forEach((e, i) => rank[e[0]] = i);
  const allScores = scored ? entries.map(e => e[1]).filter(Number.isFinite) : [];
  const mn = Math.min(...allScores), mx = Math.max(...allScores);
  const mean = allScores.reduce((a,b)=>a+b,0) / Math.max(1, allScores.length);

  document.getElementById("gsortWrap").style.display = scored || (st && st.orderIdx) ? "" : "none";
  document.getElementById("fValWrap").style.display = VAL.size ? "" : "none";

  const fVal = VAL.size && document.getElementById("fVal").checked;
  const q = document.getElementById("gsearch").value.trim().toLowerCase();
  const fR = document.getElementById("fRemoved").value;
  const rs = st && st.removedSet;
  let boundarySet = null;
  if (fR === "boundary" && st && st.order && rs) {
    const cut = rs.size;
    boundarySet = new Set(st.order.slice(Math.max(0, cut - 10), cut + 10).map(x => x.hash));
  }
  entries = entries.filter(([h]) => {
    if (fVal && !VAL.has(h)) return false;
    if (q && !h.toLowerCase().startsWith(q)) return false;
    if (fR === "kept" && rs && rs.has(h)) return false;
    if (fR === "removed" && (!rs || !rs.has(h))) return false;
    if (fR === "boundary") return boundarySet ? boundarySet.has(h) : false;
    return true;
  });

  const sortMode = document.getElementById("gsort").value;
  if (sortMode === "worst" && scored) {
    const sc = entries.filter(e => Number.isFinite(e[1])).reverse();
    const un = entries.filter(e => !Number.isFinite(e[1]));
    entries = sc.concat(un);
  } else if (sortMode === "removal" && st && st.orderIdx) {
    entries = entries.slice().sort((a, b) =>
      (st.orderIdx[a[0]] ?? Infinity) - (st.orderIdx[b[0]] ?? Infinity));
  }

  const stats = st && st.stats;
  document.getElementById("gridhead").innerHTML =
    `<b>${task}</b> — showing ${entries.length}/${n} ` +
    (scored && allScores.length
      ? `· ${SCORE_LABEL} mean ${mean.toFixed(3)} · range [${mn.toFixed(3)}, ${mx.toFixed(3)}] ` +
        histoSVG(allScores)
      : "") +
    (stats && rs && rs.size
      ? ` <span style="color:#ff9b7b">cut: ${rs.size} eps / ${fmtH(stats.removedFrames)} removed</span>`
      : "") +
    `<span style="color:#777">▶ streams MP4s from the self-hosted viewer</span>`;

  const cards = entries.map(([hash, score]) => {
    const i = rank[hash];
    const removed = rs && rs.has(hash);
    const ovr = st && st.override[hash];
    const lbl = st && st.labels ? st.labels[hash] : null;
    const badges =
      (VAL.has(hash) ? ' <span class="badge b-val">VAL</span>' : '') +
      (removed ? ' <span class="badge b-rm">REMOVED</span>' : '') +
      (!dEps.has(hash) ? ' <span class="badge b-noemb">NOT EMBEDDED</span>' : '') +
      (ovr ? ` <span class="badge b-ovr">${ovr === "keep" ? "FORCE-KEEP" : "FORCE-DROP"}</span>` : '');
    const cid = `v_${task}_${i}`;
    const g = gn[hash];
    const pct = scored && Number.isFinite(g) ? g : null;
    return `<div class="card">
      <div class="hdr"><span class="rank">#${i+1}</span>
        <span class="hash">${hash}</span>${badges}
        ${scored ? `<span class="score">${Number.isFinite(score) ? score.toFixed(4) : "—"}</span>` : ""}</div>
      ${pct != null ? `<div class="pct"><div style="width:${Math.round(pct*100)}%"></div></div>` : ""}
      <div class="vid" id="${cid}">
        <div class="ph" onclick="loadVideo('${cid}','${hash}')">
          <div class="play">▶</div><div>load video</div></div>
      </div>
      <div class="links">
        <a href="${VIDEO_BASE}${hash}" target="_blank">open ↗</a>
        ${scored ? `<span style="color:#666">rank ${i+1}/${n}</span>` : ""}
        <span class="ovr-btns" style="margin-left:auto">
          <button class="${ovr === "keep" ? "on" : ""}" onclick="setOverride('${task}','${hash}','keep')" title="always keep">K</button>
          <button class="bad ${ovr === "drop" ? "on" : ""}" onclick="setOverride('${task}','${hash}','drop')" title="always drop">D</button>
          <button class="${lbl === "good" ? "on" : ""}" onclick="setLabel('${task}','${hash}','good')" title="label good (for calibrate)">✓</button>
          <button class="bad ${lbl === "bad" ? "on" : ""}" onclick="setLabel('${task}','${hash}','bad')" title="label bad (for calibrate)">✗</button>
        </span>
      </div>
    </div>`;
  });
  document.getElementById("grid").innerHTML = cards.join("");
}

function loadAll() { document.querySelectorAll("#grid .ph").forEach(ph => ph.click()); }

function playAll() {
  loadAll();
  setSpeed();
  document.querySelectorAll("#grid video").forEach(v => { v.muted = true; v.play().catch(()=>{}); });
}

function pauseAll() { document.querySelectorAll("#grid video").forEach(v => v.pause()); }

function setSpeed() {
  const r = parseFloat(document.getElementById("gspeed").value);
  document.querySelectorAll("#grid video").forEach(v => v.playbackRate = r);
}

/* ---------------- cluster + prune tuning ----------------
   Per-task selection state lives in CL[task] and survives task switches.
   Clusters are k-means over the displayed t-SNE points (deterministic:
   seeded mulberry32 PRNG, k-means++ init, squared distances, fixed loop
   order, lowest-index ties). Removal order is a total order:
   (override, priority class, scored-before-unscored, worseness desc, hash). */
const CL = {};
function clState(task) {
  if (!CL[task]) CL[task] = {
    k: 0, mod: null, assign: null, nPts: 0,
    epDomCl: null, epDomFrac: null, epFrac: null,
    clTable: null, clColors: null,
    prio: {}, override: {}, labels: {},
    pct: 0, allowProt: false,
    order: null, orderIdx: null, removedSet: new Set(), removedEp: null,
    stats: null,
  };
  return CL[task];
}

function fnv1a(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}
function mulberry32(a) {
  return function() {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function kmeans3d(X, n, k, rand) {
  // X: Float64Array(3n). Returns Int32Array(n). No transcendentals, fixed
  // summation order, strict-< nearest-center — bit-reproducible per build.
  k = Math.max(1, Math.min(k, n));
  const cx = new Float64Array(k), cy = new Float64Array(k), cz = new Float64Array(k);
  const d2 = new Float64Array(n).fill(Infinity);
  let idx = Math.floor(rand() * n);
  for (let c = 0; c < k; c++) {                       // k-means++ init
    cx[c] = X[3*idx]; cy[c] = X[3*idx+1]; cz[c] = X[3*idx+2];
    if (c === k - 1) break;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const dx = X[3*i] - cx[c], dy = X[3*i+1] - cy[c], dz = X[3*i+2] - cz[c];
      const d = dx*dx + dy*dy + dz*dz;
      if (d < d2[i]) d2[i] = d;
      sum += d2[i];
    }
    if (sum === 0) { idx = (idx + 1) % n; continue; } // k > distinct points
    let r = rand() * sum;
    idx = n - 1;
    for (let i = 0; i < n; i++) { r -= d2[i]; if (r <= 0) { idx = i; break; } }
  }
  const assign = new Int32Array(n).fill(-1);
  const counts = new Int32Array(k);
  for (let iter = 0; iter < 60; iter++) {
    let changed = 0;
    for (let i = 0; i < n; i++) {
      let best = 0, bd = Infinity;
      for (let c = 0; c < k; c++) {
        const dx = X[3*i] - cx[c], dy = X[3*i+1] - cy[c], dz = X[3*i+2] - cz[c];
        const d = dx*dx + dy*dy + dz*dz;
        if (d < bd) { bd = d; best = c; }
      }
      if (assign[i] !== best) { assign[i] = best; changed++; }
    }
    if (!changed && iter > 0) break;
    cx.fill(0); cy.fill(0); cz.fill(0); counts.fill(0);
    for (let i = 0; i < n; i++) {
      const c = assign[i];
      counts[c]++; cx[c] += X[3*i]; cy[c] += X[3*i+1]; cz[c] += X[3*i+2];
    }
    for (let c = 0; c < k; c++) {
      if (counts[c]) { cx[c] /= counts[c]; cy[c] /= counts[c]; cz[c] /= counts[c]; }
    }
    for (let c = 0; c < k; c++) {
      if (counts[c]) continue;
      let far = 0, fd = -1;                           // reseed: farthest point
      for (let i = 0; i < n; i++) {
        const a = assign[i];
        if (counts[a] <= 1) continue;
        const dx = X[3*i] - cx[a], dy = X[3*i+1] - cy[a], dz = X[3*i+2] - cz[a];
        const d = dx*dx + dy*dy + dz*dz;
        if (d > fd) { fd = d; far = i; }
      }
      counts[assign[far]]--; assign[far] = c; counts[c] = 1;
      cx[c] = X[3*far]; cy[c] = X[3*far+1]; cz[c] = X[3*far+2];
    }
  }
  return assign;
}

function taskUniverse(task) {
  // {u: {hash: frames}, basis}. Prefers the run's RESOLVED universe with raw
  // SQL frame counts (group_universe.json) — that is the repo's hours
  // convention and covers failed-embed episodes. Falls back to embedded +
  // scored hashes with a latent-row estimate (clearly flagged).
  if (UNIVERSE[task] && Object.keys(UNIVERSE[task]).length)
    return {u: UNIVERSE[task], basis: "raw"};
  const u = {};
  const d = DATA[task] || {};
  const en = d.every_n || 10;
  (d.episodes || []).forEach(h => u[h] = 0);
  const m = d.state || d.action;
  if (m) for (let i = 0; i < m.ep.length; i++) u[d.episodes[m.ep[i]]] += en;
  (SCORES[task] || []).forEach(([h]) => { if (!(h in u)) u[h] = 0; });
  return {u, basis: "latent-est"};
}

function buildClusters(task, k) {
  const d = DATA[task];
  const mod = d && d.state ? "state" : (d && d.action ? "action" : null);
  if (!mod) return false;
  const st = clState(task);
  const m = d[mod], N = m.ep.length;
  const X = new Float64Array(3 * N);
  for (let i = 0; i < N; i++) { X[3*i] = m.x[i]; X[3*i+1] = m.y[i]; X[3*i+2] = m.z[i]; }
  const assign = kmeans3d(X, N, k, mulberry32(fnv1a(task) ^ k));
  const eps = d.episodes, nEp = eps.length;
  const epFrac = new Float64Array(nEp * k);
  const epPts = new Int32Array(nEp);
  for (let i = 0; i < N; i++) { epFrac[m.ep[i] * k + assign[i]]++; epPts[m.ep[i]]++; }
  const epDomCl = new Int32Array(nEp).fill(-1);
  const epDomFrac = new Float64Array(nEp);
  for (let e = 0; e < nEp; e++) {
    if (!epPts[e]) continue;
    let best = 0;
    for (let c = 0; c < k; c++) {
      epFrac[e*k + c] /= epPts[e];
      if (epFrac[e*k + c] > epFrac[e*k + best]) best = c;
    }
    epDomCl[e] = best; epDomFrac[e] = epFrac[e*k + best];
  }
  const clColors = [];
  for (let c = 0; c < k; c++) clColors.push(rgb(hsv2rgb(c / k, 0.78, 0.95)));
  // Color table: the clustered modality per-point; the other panel by the
  // point's episode-level dominant cluster (modalities may not be aligned).
  const clTable = {};
  clTable[mod] = Array.from(assign, c => clColors[c]);
  const other = mod === "state" ? "action" : "state";
  if (d[other]) clTable[other] = Array.from(d[other].ep, e => epDomCl[e] >= 0 ? clColors[epDomCl[e]] : "#666");
  Object.assign(st, {k, mod, assign, nPts: N, epDomCl, epDomFrac, epFrac, clTable, clColors});
  st.prio = {};   // cluster ids are not comparable across re-clusterings
  return true;
}

function epClassOf(task, st, epIdx, hash) {
  // Removal classes: 0 forced-drop, 1 prune-first, 2 normal, 3 protect, 4 forced-keep.
  if (st.override[hash] === "drop") return 0;
  if (st.override[hash] === "keep") return 4;
  if (epIdx == null || epIdx < 0 || !st.assign || st.epDomCl[epIdx] < 0) return 2;
  let prot = 0, prune = 0;
  for (let c = 0; c < st.k; c++) {
    const f = st.epFrac[epIdx * st.k + c];
    if (st.prio[c] === "protect") prot += f;
    else if (st.prio[c] === "prune") prune += f;
  }
  if (prot >= 0.5) return 3;     // threshold rule, not plurality: an episode
  if (prune >= 0.5) return 1;    // is protected only if HALF its points are
  return 2;
}

function computeRemoval(task) {
  const st = clState(task);
  const {u, basis} = taskUniverse(task);
  const eps = (DATA[task] || {}).episodes || [];
  const epIdxOf = {}; eps.forEach((h, i) => epIdxOf[h] = i);
  const sm = scoreMap(task);
  const items = Object.keys(u).sort().map(h => {
    const s = sm[h];
    const scored = Number.isFinite(s);
    return {
      hash: h, frames: u[h] || 0, score: scored ? s : null, scored,
      cls: epClassOf(task, st, epIdxOf[h], h),
      worse: scored ? (HIGHER_WORSE ? s : -s) : -Infinity,
    };
  });
  // Total deterministic removal order (remove-first → remove-last):
  // class asc, scored before unscored, worst first, then hash.
  items.sort((a, b) =>
    a.cls - b.cls ||
    (b.scored ? 1 : 0) - (a.scored ? 1 : 0) ||
    b.worse - a.worse ||
    (a.hash < b.hash ? -1 : a.hash > b.hash ? 1 : 0));
  const total = items.reduce((a, x) => a + x.frames, 0);
  const maxCls = st.allowProt ? 3 : 2;
  const target = st.pct / 100 * total;
  const removed = [];
  let removedFrames = 0;
  let removable = 0;
  for (const x of items) {
    if (x.cls > maxCls) { if (x.cls === 4) continue; else break; }
    removable += x.frames;
  }
  for (const x of items) {
    if (x.cls > maxCls) break;                  // protect/forced-keep tail
    if (x.cls !== 0 && removedFrames >= target) break;
    removed.push(x); removedFrames += x.frames; // forced drops always go
  }
  // Tie-split warning: the cut lands inside a block of identical scores.
  let tieSplit = false;
  if (removed.length && removed.length < items.length) {
    const last = removed[removed.length - 1], next = items[removed.length];
    tieSplit = last.cls === next.cls && last.scored && next.scored && last.score === next.score;
  }
  st.order = items;
  st.orderIdx = {}; items.forEach((x, i) => st.orderIdx[x.hash] = i);
  st.removedSet = new Set(removed.map(x => x.hash));
  st.removedEp = new Uint8Array(eps.length);
  removed.forEach(x => { const i = epIdxOf[x.hash]; if (i != null) st.removedEp[i] = 1; });
  st.stats = {
    total, removedFrames, removedEps: removed.length,
    keptEps: items.length - removed.length, keptFrames: total - removedFrames,
    basis, tieSplit,
    maxRemovablePct: total ? 100 * removable / total : 0,
  };
  return st.stats;
}

function activeSelection(task) {
  const st = CL[task];
  return !!(st && (st.pct > 0 || st.removedSet.size || Object.keys(st.override).length));
}

function allTaskNames() {
  return Array.from(new Set([
    ...Object.keys(SCORES), ...Object.keys(DATA), ...Object.keys(UNIVERSE),
  ])).sort();
}

function clusterPanelHTML(task) {
  const st = CL[task];
  if (!st || !st.assign) return "";
  const sm = scoreMap(task);
  const eps = (DATA[task] || {}).episodes || [];
  const {u} = taskUniverse(task);
  let rows = "";
  for (let c = 0; c < st.k; c++) {
    const members = [];
    for (let e = 0; e < eps.length; e++) if (st.epDomCl[e] === c) members.push(e);
    const frames = members.reduce((a, e) => a + (u[eps[e]] || 0), 0);
    const scores = members.map(e => sm[eps[e]]).filter(Number.isFinite).sort((a, b) => a - b);
    const med = scores.length ? scores[(scores.length - 1) >> 1] : null;
    const p90i = scores.length ? Math.min(scores.length - 1, Math.floor(0.9 * scores.length)) : 0;
    const p90 = scores.length ? scores[HIGHER_WORSE ? p90i : scores.length - 1 - p90i] : null;
    const purities = members.map(e => st.epDomFrac[e]).sort((a, b) => a - b);
    const pmed = purities.length ? purities[(purities.length - 1) >> 1] : 0;
    const pmin = purities.length ? purities[0] : 0;
    const warn = pmed < 0.5 ? ' <span class="warn" title="median member has <50% of its points here — weak cluster membership">⚠</span>' : "";
    const prio = st.prio[c] || "normal";
    rows += `<div class="cl-row" title="median purity ${Math.round(100*pmed)}% / min ${Math.round(100*pmin)}%">` +
      `<span class="lg-dot" style="background:${st.clColors[c]}"></span>` +
      `c${c} · ${members.length} eps · ${fmtH(frames)}` +
      (med != null ? ` · med ${med.toFixed(3)} p90 ${p90.toFixed(3)}` : "") + warn +
      `<select onchange="setClusterPrio('${task}', ${c}, this.value)">` +
        `<option value="prune" ${prio === "prune" ? "selected" : ""}>prune-first</option>` +
        `<option value="normal" ${prio === "normal" ? "selected" : ""}>normal</option>` +
        `<option value="protect" ${prio === "protect" ? "selected" : ""}>protect</option>` +
      `</select></div>`;
  }
  return `<div id="clpanel"><div class="lg-head">clusters (k=${st.k} on ${st.mod})</div>${rows}</div>`;
}

function refreshSelection(task) {
  computeRemoval(task);
  syncSelectionUI(task);
  if (task === curTask && page === "tsne") { applyStyle(); buildLegend(task); }
  if (page === "grid") renderGrid(task);
}

function syncSelectionUI(task) {
  const st = CL[task];
  document.getElementById("prunePct").value = st ? st.pct : 0;
  document.getElementById("prunePctN").value = st ? st.pct : 0;
  document.getElementById("allowProt").checked = !!(st && st.allowProt);
  if (st && st.k) document.getElementById("ck").value = st.k;
  const el = document.getElementById("cstats");
  if (!st || !st.stats || !activeSelection(task)) { el.textContent = "no prune selection"; return; }
  const s = st.stats;
  el.innerHTML =
    `kept <b>${s.keptEps}</b> eps / ${fmtH(s.keptFrames)} · ` +
    `removed ${s.removedEps} eps / ${fmtH(s.removedFrames)}` +
    (s.basis === "latent-est" ? ` <span title="no group_universe.json — hours estimated from latent rows, NOT raw frames">(est)</span>` : "") +
    (s.tieSplit ? ` <span class="warn" title="the cut splits a block of identical scores — order within the tie is hash-arbitrary">⚠ tie split</span>` : "") +
    (st.pct > s.maxRemovablePct ? ` <span class="warn">clamped at ${s.maxRemovablePct.toFixed(1)}% (protected)</span>` : "");
}

function clusterNow() {
  if (!curTask) return;
  const k = Math.max(2, Math.min(40, +document.getElementById("ck").value || 8));
  if (!buildClusters(curTask, k)) return;
  const cm = document.getElementById("colorMode");
  if (!document.querySelector('#colorMode option[value="cluster"]').disabled) cm.value = "cluster";
  refreshSelection(curTask);
}

function pruneChanged() {
  if (!curTask) return;
  const st = clState(curTask);
  st.pct = +document.getElementById("prunePct").value;
  st.allowProt = document.getElementById("allowProt").checked;
  refreshSelection(curTask);
}
function pruneChangedN() {
  document.getElementById("prunePct").value = document.getElementById("prunePctN").value;
  pruneChanged();
}
function targetHours() {
  if (!curTask) return;
  const st = clState(curTask);
  const {u} = taskUniverse(curTask);
  const total = Object.values(u).reduce((a, b) => a + b, 0);
  const keepH = +document.getElementById("targetH").value;
  if (!total || !(keepH >= 0)) return;
  const pct = Math.max(0, Math.min(100, 100 * (1 - keepH * 3600 * FPS / total)));
  st.pct = Math.round(pct * 2) / 2;
  st.allowProt = document.getElementById("allowProt").checked;
  refreshSelection(curTask);
}

function setClusterPrio(task, c, value) {
  const st = clState(task);
  if (value === "normal") delete st.prio[c]; else st.prio[c] = value;
  refreshSelection(task);
}
function setOverride(task, hash, value) {
  const st = clState(task);
  if (st.override[hash] === value) delete st.override[hash];
  else st.override[hash] = value;
  refreshSelection(task);
}
function setLabel(task, hash, value) {
  const st = clState(task);
  if (st.labels[hash] === value) delete st.labels[hash];
  else st.labels[hash] = value;
  if (page === "grid") renderGrid(task);
}

/* ---------------- selection export / import ---------------- */
function downloadJSON(obj, name) {
  const blob = new Blob([JSON.stringify(obj)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
}
function stamp() { return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19); }

function exportKeepList() {
  // eps_to_use-ready: union over ALL tasks; tasks without a selection keep
  // every universe episode (a per-task list would silently drop the rest of
  // a multi-task config from training).
  const kept = [];
  for (const t of allTaskNames()) {
    const {u} = taskUniverse(t);
    let rs = null;
    if (activeSelection(t)) { computeRemoval(t); rs = CL[t].removedSet; }
    for (const h of Object.keys(u).sort()) if (!rs || !rs.has(h)) kept.push(h);
  }
  downloadJSON(kept, `keep_list_${stamp()}.json`);
}

function exportSelection() {
  const tasks = {};
  for (const t of allTaskNames()) {
    const st = CL[t];
    if (!st) continue;
    const touched = activeSelection(t) || Object.keys(st.labels).length || Object.keys(st.prio).length;
    if (!touched) continue;
    computeRemoval(t);
    const {u, basis} = taskUniverse(t);
    const eps = (DATA[t] || {}).episodes || [];
    const clusterOf = {};
    if (st.assign) eps.forEach((h, i) => {
      if (st.epDomCl[i] >= 0) clusterOf[h] = [st.epDomCl[i], Math.round(1000 * st.epDomFrac[i]) / 1000];
    });
    tasks[t] = {
      k: st.k || null, modality: st.mod,
      cluster_prio: st.prio, override: st.override, labels: st.labels,
      pct: st.pct, allow_protected: st.allowProt,
      frames_basis: basis, universe_n: Object.keys(u).length,
      removed: st.order.filter(x => st.removedSet.has(x.hash))
        .map(x => ({hash: x.hash, score: x.score, cls: x.cls, frames: x.frames})),
      kept: Object.keys(u).filter(h => !st.removedSet.has(h)).sort(),
      cluster_of: clusterOf,
      realized: st.stats,
    };
  }
  downloadJSON({
    schema_version: 1,
    created_at: new Date().toISOString(),
    comparator: "override,priority,scored-first,worse-desc,hash",
    scores_meta: SCORES_META,
    manifest: MANIFEST,
    tasks,
  }, `viewer_selection_${stamp()}.json`);
}

function importSelectionFile(input) {
  const file = input.files && input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    let sel;
    try { sel = JSON.parse(reader.result); } catch (e) { alert("not valid JSON: " + e); return; }
    const notes = [];
    for (const [t, b] of Object.entries(sel.tasks || {})) {
      const st = clState(t);
      st.pct = +b.pct || 0;
      st.allowProt = !!b.allow_protected;
      st.override = Object.assign({}, b.override || {});
      st.labels = Object.assign({}, b.labels || {});
      // Cluster priorities only port when the recomputed assignment matches
      // the saved one (same build, same k) — cluster ids are otherwise
      // meaningless. Falls back to overrides/pct, which are hash-grounded.
      let prioOk = false;
      if (b.k && DATA[t] && buildClusters(t, +b.k)) {
        const eps = DATA[t].episodes;
        prioOk = eps.every((h, i) => {
          const saved = (b.cluster_of || {})[h];
          return !saved || saved[0] === st.epDomCl[i];
        });
        if (prioOk) st.prio = Object.assign({}, b.cluster_prio || {});
        else notes.push(`${t}: cluster assignment drifted — priorities NOT restored (pct/overrides applied)`);
      }
      computeRemoval(t);
      const savedRemoved = new Set((b.removed || []).map(x => x.hash));
      const drift = [...st.removedSet].filter(h => !savedRemoved.has(h)).length +
                    [...savedRemoved].filter(h => !st.removedSet.has(h)).length;
      if (drift) notes.push(`${t}: removal set differs from saved by ${drift} episode(s)`);
    }
    if (curTask) refreshSelection(curTask);
    alert("selection imported" + (notes.length ? "\\n\\n" + notes.join("\\n") : ""));
  };
  reader.readAsText(file);
  input.value = "";
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


def build_html(
    tsne_dir: Path,
    scores_raw: dict | None = None,
    val: list | None = None,
    video_base: str = VIDEO_BASE,
    scores_meta: dict | None = None,
    universe: dict | None = None,
    manifest: dict | None = None,
) -> str:
    """Assemble the viewer HTML from a local tsne3d dir.

    ``scores_raw`` ({task: {hash: score}}) and ``val`` (episode-hash list) are
    optional — without them the viewer simply omits score- and VAL-based UI.

    ``scores_meta`` describes the scores' provenance and direction
    ({"higher_is_worse": bool, "metric": ..., "source": ...}); when absent the
    DemInf convention (higher = better) is assumed. ``universe``
    ({task: {hash: raw_frame_count}}) is the full RESOLVED episode set per
    group from group_universe.json — it makes the prune/keep-list math cover
    episodes that failed embedding and gives hours readouts a raw-frame basis.
    ``manifest`` is the run's viz_manifest.json, embedded for export provenance.
    """
    data: dict = {}
    for f in sorted(Path(tsne_dir).glob("tsne3d_*.json")):
        d = json.load(open(f))
        data[d["task"]] = d
    if not data:
        raise FileNotFoundError(f"No tsne3d_*.json files in {tsne_dir}")
    meta = dict(scores_meta or {})
    meta.setdefault("higher_is_worse", False)
    # Rank/grid order is best-first regardless of score direction; non-finite last.
    sign = 1.0 if meta["higher_is_worse"] else -1.0
    scores = {
        t: sorted(
            ([h, s] for h, s in v.items()),
            key=lambda kv: (
                not math.isfinite(kv[1]),
                sign * kv[1] if math.isfinite(kv[1]) else 0,
                kv[0],
            ),
        )
        for t, v in (scores_raw or {}).items()
    }
    return (
        _TEMPLATE
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__SCORES__", json.dumps(scores, separators=(",", ":")))
        .replace("__SCORES_META__", json.dumps(meta, separators=(",", ":")))
        .replace("__UNIVERSE__", json.dumps(universe or {}, separators=(",", ":")))
        .replace("__MANIFEST__", json.dumps(manifest or {}, separators=(",", ":")))
        .replace("__VAL__", json.dumps(val or [], separators=(",", ":")))
        .replace("__VIDEO_BASE__", video_base.rstrip("/") + "/")
        .replace("__FPS__", str(FPS))
    )


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
    print("No scores_by_task.json found — building score-free viewer.")
    return {}


def _load_sidecar(args, tsne_local: Path, filename: str):
    """Load an optional sibling JSON of the tsne3d dir, with a volume fallback."""
    sib = tsne_local.parent / filename
    if sib.is_file():
        return json.load(open(sib))
    if args.volume:
        remote = str(Path(args.tsne3d_dir.rstrip("/")).parent / filename)
        tmp = Path(tempfile.mktemp(suffix=".json"))
        if _download(args.volume, args.env, remote, tmp):
            return json.load(open(tmp))
    return None


def _load_val(args, tsne_local: Path) -> list:
    """Resolve val_episodes.json: --val-json path, sibling of tsne3d dir, or volume."""
    if args.val_json:
        p = Path(args.val_json)
        if p.is_file():
            return json.load(open(p))
        if args.volume:
            tmp = Path(tempfile.mktemp(suffix=".json"))
            if _download(args.volume, args.env, args.val_json, tmp):
                return json.load(open(tmp))
        sys.exit(f"--val-json not found: {args.val_json}")
    sib = tsne_local.parent / "val_episodes.json"
    if sib.is_file():
        return json.load(open(sib))
    if args.volume:
        remote = str(Path(args.tsne3d_dir.rstrip("/")).parent / "val_episodes.json")
        tmp = Path(tempfile.mktemp(suffix=".json"))
        if _download(args.volume, args.env, remote, tmp):
            return json.load(open(tmp))
    return []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsne3d_dir", help="Local dir of tsne3d_*.json, or volume-relative path")
    ap.add_argument("--scores", default=None, help="Optional scores_by_task.json (local or volume-relative); default: sibling of tsne3d dir if present")
    ap.add_argument("--val-json", default=None,
                    help="Optional episode-hash list to badge as VAL in the grid; default: sibling val_episodes.json if present")
    ap.add_argument("--video-base", default=VIDEO_BASE,
                    help="Base URL of the episode MP4 endpoint — the full /video/ route of an "
                         "episode_preview.py deployment, e.g. "
                         "https://<org>--egoverse-episode-preview-viewer.modal.run/video/")
    ap.add_argument("--volume", default=None, help="Modal volume to download from if paths not local")
    ap.add_argument("--env", default="robotics", help="Modal environment (default: robotics)")
    ap.add_argument("--out", default="latent_viz.html", help="Output HTML path")
    args = ap.parse_args()

    src = _load_tsne_dir(args.tsne3d_dir, args.volume, args.env)
    raw_scores = _load_scores(args, src)
    if not raw_scores:
        knn = _load_sidecar(args, src, "knn_scores_by_task.json")
        if knn:
            raw_scores = knn
            print("Using sibling knn_scores_by_task.json for scores.")

    val = _load_val(args, src)
    if val:
        print(f"VAL badges: {len(val)} episodes")

    scores_meta = _load_sidecar(args, src, "scores_meta.json")
    universe = _load_sidecar(args, src, "group_universe.json")
    manifest = _load_sidecar(args, src, "viz_manifest.json")
    if scores_meta:
        print(f"scores_meta: {scores_meta.get('source', '?')} · higher_is_worse={scores_meta.get('higher_is_worse')}")
    if universe:
        print(f"universe: {sum(len(v) for v in universe.values())} episodes across {len(universe)} group(s)")

    html = build_html(
        src, raw_scores, val,
        video_base=args.video_base,
        scores_meta=scores_meta,
        universe=universe,
        manifest=manifest,
    )
    out = Path(args.out)
    out.write_text(html)
    print(f"Wrote {out.resolve()}  ({out.stat().st_size/1e6:.1f} MB)")
    print("Open it in any browser — no server needed.")


if __name__ == "__main__":
    main()
