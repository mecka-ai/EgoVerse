#!/usr/bin/env python3
"""
Build a self-contained interactive latent + episode-video viewer.

Two pages in one HTML file (works as a local file or served by the Modal web
app ``egomimic/modal/latent_viz_app.py``):

  * **t-SNE 3-D** — per-task 3-D scatters of state/action latents from
    ``tsne3d_<task>.json``. Fast single-trace WebGL engine: per-task color
    tables (episode / time / MI score) are precomputed once and every control
    change is an in-place restyle (no scene rebuild). Tools: frame-number
    highlight (±window) across BOTH panels, time-range filter, episode
    isolate, point size, synced cameras, click → frame-seek video preview +
    gold cross-highlight of the same (episode, frame) in the other panel.
  * **Video grid** — per-task grid of every scored episode (from
    ``scores_by_task.json``): sort/filter/search, rank/score/percentile,
    TOP-60%/BOT-40%/VAL badges, score histogram, Load-all / Play-all /
    Pause-all / playback-speed. MP4s stream from the self-hosted Modal viewer
    (egomimic/modal/episode_preview.py).

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
  .panel { width: 50vw; height: calc(100vh - 104px); }
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
  .b-top { background: #1d4d2b; color: #7be495; }
  .b-bot { background: #4d1d1d; color: #ff9d9d; }
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

<div id="tsnepage">
  <div id="tools">
    <span class="tool"><label>Color</label>
      <select id="colorMode" onchange="applyStyle()">
        <option value="episode">episode</option>
        <option value="time">time (light→dark)</option>
        <option value="score">MI score (red→green)</option>
      </select></span>
    <span class="tool"><label>Episode</label>
      <select id="epSel" onchange="applyStyle()"><option value="all">all</option></select></span>
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
  </div>
</div>

<div id="gridpage">
  <div id="gtools">
    <span class="tool"><label>Sort</label>
      <select id="gsort" onchange="render()">
        <option value="desc">score ↓ (best first)</option>
        <option value="asc">score ↑ (worst first)</option>
      </select></span>
    <span class="tool">
      <label class="chk"><input type="checkbox" id="fTop" checked onchange="render()"> TOP 60%</label>
      <label class="chk"><input type="checkbox" id="fBot" checked onchange="render()"> BOT 40%</label>
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
const VAL = new Set(__VAL__);
const VIDEO_BASE = "__VIDEO_BASE__";
const FPS = __FPS__;
const DIM = "rgba(110,110,120,0.05)";

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

function scoreNorm(task) {
  const e = SCORES[task] || [];
  if (!e.length) return {};
  const vals = e.map(x => x[1]), mn = Math.min(...vals), mx = Math.max(...vals);
  const out = {};
  e.forEach(([h, s]) => out[h] = mx > mn ? (s - mn) / (mx - mn) : 0.5);
  return out;
}

/* ---------------- t-SNE fast engine ----------------
   One trace per panel. Per (task, mode) color tables are computed ONCE and
   cached; every control change only re-picks colors[k] = cached[k] | DIM and
   restyles in place — no scene/trace rebuild, no camera reset.            */
const CACHE = {};   // task -> {mods: {state:{d, tables:{episode,time,score}}, action:{...}}}
let curTask = null;

function buildCache(task) {
  const eps = DATA[task].episodes, nEp = eps.length;
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
  for (const mod of ["state", "action"]) {
    const m = CACHE[curTask].mods[mod];
    if (!m) continue;
    const {d, tables} = m;
    const base = tables[u.mode];
    const N = d.x.length;
    const colors = new Array(N);
    let sizes = u.size;                          // scalar unless highlighting
    if (u.hlOn) sizes = new Array(N);
    for (let k = 0; k < N; k++) {
      const on = d.t[k] >= u.t0 && d.t[k] <= u.t1
        && (u.ep === "all" || d.ep[k] === +u.ep)
        && (!u.hlOn || Math.abs(d.frame[k] - u.hlF) <= u.hlW);
      colors[k] = on ? base[k] : DIM;
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
    el.on("plotly_click", ev => {
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
  applyStyle();
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
  const bars = bins.map((b,i) =>
    `<rect x="${i*8}" y="${28-26*b/bmax}" width="6" height="${26*b/bmax}" fill="${rgb(redGreen(i/(nb-1)))}"/>`).join("");
  return `<svg id="histo" width="${nb*8}" height="30" title="score distribution">${bars}</svg>`;
}

function renderGrid(task) {
  let entries = (SCORES[task] || []).slice();
  const n = entries.length, nTop = Math.ceil(0.6 * n);
  const rank = {}; entries.forEach((e, i) => rank[e[0]] = i);
  const allScores = entries.map(e => e[1]);
  const mn = Math.min(...allScores), mx = Math.max(...allScores);
  const mean = allScores.reduce((a,b)=>a+b,0) / Math.max(1,n);

  const fTop = document.getElementById("fTop").checked;
  const fBot = document.getElementById("fBot").checked;
  const fVal = document.getElementById("fVal").checked;
  const q = document.getElementById("gsearch").value.trim().toLowerCase();
  entries = entries.filter(([h]) => {
    const isTop = rank[h] < nTop;
    if (isTop && !fTop) return false;
    if (!isTop && !fBot) return false;
    if (fVal && !VAL.has(h)) return false;
    if (q && !h.toLowerCase().startsWith(q)) return false;
    return true;
  });
  if (document.getElementById("gsort").value === "asc") entries.reverse();

  document.getElementById("gridhead").innerHTML =
    `<b>${task}</b> — showing ${entries.length}/${n} · MI mean ${mean.toFixed(3)} · range [${mn.toFixed(3)}, ${mx.toFixed(3)}] ` +
    histoSVG(allScores) +
    `<span style="color:#777">▶ streams MP4s from the self-hosted viewer</span>`;

  const cards = entries.map(([hash, score]) => {
    const i = rank[hash];
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


def build_html(tsne_dir: Path, scores_raw: dict, val: list) -> str:
    """Assemble the viewer HTML from a local tsne3d dir + raw scores dict."""
    data: dict = {}
    for f in sorted(Path(tsne_dir).glob("tsne3d_*.json")):
        d = json.load(open(f))
        data[d["task"]] = d
    scores = {
        t: sorted(
            ([h, s] for h, s in v.items()),
            key=lambda kv: (not math.isfinite(kv[1]), -(kv[1] if math.isfinite(kv[1]) else 0), kv[0]),
        )
        for t, v in scores_raw.items()
    }
    return (
        _TEMPLATE
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__SCORES__", json.dumps(scores, separators=(",", ":")))
        .replace("__VAL__", json.dumps(val, separators=(",", ":")))
        .replace("__VIDEO_BASE__", VIDEO_BASE)
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
    raw_scores = _load_scores(args, src)

    val: list = []
    vp = Path(args.val_json) if args.val_json else None
    if vp and vp.is_file():
        val = json.load(open(vp))
        print(f"VAL badges: {len(val)} episodes from {vp}")

    html = build_html(src, raw_scores, val)
    out = Path(args.out)
    out.write_text(html)
    print(f"Wrote {out.resolve()}  ({out.stat().st_size/1e6:.1f} MB)")
    print("Open it in any browser — no server needed.")


if __name__ == "__main__":
    main()
