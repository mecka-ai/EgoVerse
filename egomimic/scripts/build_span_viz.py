#!/usr/bin/env python3
"""Build a self-contained interactive span-level latent viewer for clustered DemInf runs.

Reads a single ``spans_tsne3d.json`` file (see schema below) and produces a
self-contained HTML file with:

  * **t-SNE 3-D** — one point per annotation span, colored by cluster (20 colors).
    Panels: state, action, language.  Click a point → frame preview at span midpoint.
  * **Span grid** — filterable/sortable cards per span, each card shows annotation
    text, cluster badge, score, and an inline video player that auto-seeks to the
    span's [start, end) frame range.

Schema for ``spans_tsne3d.json`` (emitted by _score_task_clustered):
::

    {
      "n_clusters": 20,
      "clusters": {
        "0": {"label": "wiping surface", "n_spans": 187},
        ...
      },
      "spans": [
        {
          "id":    "69abc..::10-55",   // "{ep_hash}::{start}-{end}"
          "ep":    "69abcdef...",       // 24-hex episode hash
          "start": 10,                  // inclusive, frame index
          "end":   55,                  // exclusive, frame index
          "text":  "wipe with cloth",
          "score": 0.823,
          "cluster": 3                  // int cluster id
        },
        ...
      ],
      "state":    {"x":[..], "y":[..], "z":[..], "span_idx":[..]},
      "action":   {"x":[..], "y":[..], "z":[..], "span_idx":[..]},
      "language": {"x":[..], "y":[..], "z":[..], "span_idx":[..]}
    }

``span_idx`` entries are indices into the ``spans`` array.  A modality can be
absent or have a different length if some spans failed to embed.
"""

from __future__ import annotations

import json
from pathlib import Path

VIDEO_BASE = "https://mecka-robotics--egoverse-viewer-viewer.modal.run/video/"
FRAME_BASE  = "https://mecka-robotics--egoverse-viewer-viewer.modal.run/frame/"
FPS = 30

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Meckaverse — span viewer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root { --bg:#101014; --bar:#1a1b21; --card:#1c1d24; --line:#2b2d36; --acc:#3b82f6; --txt:#e8e8ea; }
  html, body { height:100%; margin:0; overflow:hidden; }
  body { font-family:-apple-system,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--txt);
         display:flex; flex-direction:column; }
  #bar { flex-shrink:0; padding:7px 14px; display:flex; gap:10px; align-items:center;
         background:var(--bar); border-bottom:1px solid var(--line); flex-wrap:wrap; min-height:40px; }
  #bar h3 { margin:0; font-size:14px; font-weight:700; color:#e8e8ea; }
  .tab { padding:4px 13px; border-radius:7px; cursor:pointer; background:#2a2b33; user-select:none;
         font-weight:600; font-size:13px; }
  .tab.active { background:var(--acc); }
  #info { font-size:12px; padding:4px 10px; background:#222; border-radius:6px;
          min-width:180px; max-width:560px; overflow:hidden; text-overflow:ellipsis;
          white-space:nowrap; flex:1; }
  #info b { color:#7fd4ff; }
  .run-nav { font-size:12px; color:#9aa; white-space:nowrap; margin-left:auto; }
  .run-nav a { color:#9ecbff; text-decoration:none; }
  select, input[type=number], input[type=text] { font-size:13px; padding:4px 7px;
    background:#26272f; color:var(--txt); border:1px solid var(--line); border-radius:5px; }
  input[type=text] { width:120px; }
  button { background:#2a2b33; color:#ddd; border:1px solid var(--line); border-radius:6px;
           padding:4px 10px; cursor:pointer; font-size:13px; }
  button:hover { background:#33353f; }
  .tool { display:flex; gap:5px; align-items:center; }
  .tool > label { color:#9aa; font-size:13px; }
  .sep { width:1px; height:18px; background:var(--line); margin:0 2px; flex-shrink:0; }
  /* tsne page */
  #tsnepage { flex:1; min-height:0; display:flex; flex-direction:column; overflow:hidden; }
  #tools { flex-shrink:0; padding:5px 14px; display:flex; gap:12px; align-items:center;
           flex-wrap:wrap; background:#15161b; border-bottom:1px solid var(--line); font-size:13px; }
  #plots { flex:1; min-height:0; display:flex; overflow:hidden; }
  #plotGrid { flex:1; min-height:0; display:flex; flex-direction:column; overflow:hidden; }
  .panel-row { flex:1; min-height:0; display:flex; overflow:hidden;
               border-bottom:1px solid var(--line); }
  .panel-row:last-child { border-bottom:none; }
  .panel-wrap { flex:1; min-width:0; display:flex; flex-direction:column;
                border-right:1px solid var(--line); overflow:hidden; }
  .panel-wrap:last-child { border-right:none; }
  .panel-title { flex-shrink:0; font-size:11px; color:#9aa; padding:3px 8px; background:#14151a;
                 border-bottom:1px solid var(--line); font-weight:600; letter-spacing:.4px; }
  .panel { flex:1; min-height:0; overflow:hidden; }
  /* legend */
  #legend { flex-shrink:0; width:190px; overflow-y:auto; background:#14151a;
            border-left:1px solid var(--line); font-size:12px; }
  #legend .lg-head { padding:6px 10px; color:#9aa; position:sticky; top:0;
                     background:#14151a; font-weight:600; font-size:11px; }
  .lg-row { display:flex; align-items:center; gap:6px; padding:3px 10px; cursor:pointer;
            font-size:11px; color:#cdd; white-space:nowrap; overflow:hidden; }
  .lg-row:hover { background:#1f2027; }
  .lg-row.active { background:#24344f; }
  .lg-dot { width:9px; height:9px; border-radius:50%; flex:none; }
  .lg-label { overflow:hidden; text-overflow:ellipsis; flex:1; }
  .lg-count { color:#555; font-size:10px; flex-shrink:0; }
  /* grid page */
  #gridpage { flex:1; min-height:0; display:none; flex-direction:column; overflow:hidden; }
  #gtools { flex-shrink:0; padding:5px 14px; display:flex; gap:12px; align-items:center;
            flex-wrap:wrap; background:#15161b; border-bottom:1px solid var(--line); font-size:13px; }
  #gridwrap { flex:1; overflow-y:auto; padding:12px 16px; }
  #gridhead { margin:0 0 10px; color:#bbb; font-size:13px; }
  #span-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:12px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .card .hdr { display:flex; align-items:center; gap:6px; padding:7px 10px; font-size:13px; }
  .rank { color:#888; min-width:30px; font-size:12px; }
  .hash { font-family:ui-monospace,monospace; color:#9ecbff; font-size:12px; }
  .score { margin-left:auto; font-weight:700; font-size:12px; }
  .cbadge { font-size:10px; padding:2px 7px; border-radius:8px; font-weight:700; }
  .ann-text { padding:3px 10px 5px; font-size:11px; color:#888; font-style:italic;
              border-bottom:1px solid var(--line); }
  .pct { height:3px; background:#2a2a2a; }
  .pct > div { height:100%; background:linear-gradient(90deg,#e05555,#e0c14f,#54c46c); }
  .vid { position:relative; aspect-ratio:16/10; background:#0a0a0c; }
  .vid video { width:100%; height:100%; object-fit:contain; background:#000; }
  .vid .ph { position:absolute; inset:0; display:flex; flex-direction:column; gap:6px;
             align-items:center; justify-content:center; cursor:pointer; color:#777; }
  .vid .ph:hover { color:#fff; background:rgba(0,0,0,.4); }
  .ph .play { font-size:28px; }
  .clinks { padding:6px 10px; font-size:11px; display:flex; gap:10px; }
  .clinks a { color:#7fd4ff; text-decoration:none; }
  /* frame preview */
  #preview { position:fixed; right:14px; bottom:14px; width:400px; background:#17181d;
             border:1px solid #34363f; border-radius:10px; box-shadow:0 8px 30px rgba(0,0,0,.7);
             display:none; z-index:50; overflow:hidden; }
  #preview img { width:100%; display:block; background:#000; }
  #pv-cap { font-size:12px; padding:6px 10px; color:#ccc; display:flex; gap:6px;
            align-items:center; flex-wrap:wrap; }
  #pv-cap b { color:#7fd4ff; }
  #pv-close { position:absolute; top:5px; right:7px; cursor:pointer; color:#aaa; font-size:14px;
              background:rgba(0,0,0,.6); border-radius:50%; width:20px; height:20px;
              text-align:center; line-height:20px; }
  #pv-close:hover { color:#fff; }
</style>
</head>
<body>
<div id="bar">
  <h3>Meckaverse</h3>
  <span class="tab active" id="tab-tsne" onclick="showPage('tsne')">t-SNE 3-D</span>
  <span class="tab" id="tab-grid" onclick="showPage('grid')">Span grid</span>
  <div id="info">Click a point to inspect</div>
  <div class="run-nav" id="runNav">__RUN_LABEL__</div>
</div>

<div id="tsnepage">
  <div id="tools">
    <span class="tool"><label>Clusters</label>
      <span id="selCount" style="background:#26272f;padding:3px 8px;border-radius:5px;font-size:12px;color:#cdd">all</span>
      <button onclick="clearSel()">clear</button></span>
    <div class="sep"></div>
    <span class="tool"><label>Size</label>
      <input type="range" id="psize" min="1" max="10" value="4"
             oninput="recolor()" style="accent-color:var(--acc);width:70px"></span>
    <span class="tool"><label>Opacity</label>
      <input type="range" id="popac" min="10" max="100" value="85"
             oninput="recolor()" style="accent-color:var(--acc);width:70px"></span>
    <button onclick="resetTools()">reset</button>
  </div>
  <div id="plots">
    <div id="plotGrid"></div>
    <div id="legend"></div>
  </div>
</div>

<div id="gridpage">
  <div id="gtools">
    <span class="tool"><label>Cluster</label>
      <select id="gCluster" onchange="renderGrid()">
        <option value="">all</option>
      </select></span>
    <span class="tool"><label>Sort</label>
      <select id="gsort" onchange="renderGrid()">
        <option value="desc">score ↓</option>
        <option value="asc">score ↑</option>
      </select></span>
    <span class="tool"><label>Find</label>
      <input type="text" id="gsearch" placeholder="text or hash…" oninput="renderGrid()"></span>
    <button onclick="loadAll()">Load all</button>
    <button onclick="playAll()">▶ Play all</button>
    <button onclick="pauseAll()">⏸ Pause</button>
  </div>
  <div id="gridwrap">
    <div id="gridhead"></div>
    <div id="span-grid"></div>
  </div>
</div>

<div id="preview">
  <div id="pv-close" onclick="hidePreview()">✕</div>
  <img id="pv-img" alt="">
  <div id="pv-cap"></div>
</div>

<script>
const SPANS = __SPANS__;
const CLUSTER_META = __CLUSTER_META__;
const N_CLUSTERS = __N_CLUSTERS__;
const MODS = __MODS__;
const VIDEO_BASE = "__VIDEO_BASE__";
const FRAME_BASE = "__FRAME_BASE__";
const FPS = __FPS__;
const HIDDEN = "rgba(0,0,0,0)";
const DIM   = "rgba(110,110,120,0.06)";

/* ── cluster colors ── */
function hsv2rgb(h,s,v){
  const i=Math.floor(h*6),f=h*6-i;
  const p=v*(1-s),q=v*(1-f*s),t=v*(1-(1-f)*s);
  const c=[[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i%6];
  return [c[0]*255,c[1]*255,c[2]*255];
}
function rgb(c){return `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;}
function rgbA(c,a){return `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${a})`;}

const CLUSTER_COLORS = Array.from({length: N_CLUSTERS}, (_, i) =>
  hsv2rgb((i / Math.max(1, N_CLUSTERS)) + 0.05, 0.80, 0.92));

function clusterRgb(id){ return rgb(CLUSTER_COLORS[id % N_CLUSTERS]); }
function clusterRgbA(id, a){ return rgbA(CLUSTER_COLORS[id % N_CLUSTERS], a); }

/* ── state ── */
let selectedClusters = new Set();
let pvSpanIdx = null;

/* ── helpers ── */
function escHtml(s){ return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

/* ── legend ── */
function buildLegend(){
  const rows = Array.from({length: N_CLUSTERS}, (_, i) => {
    const meta = CLUSTER_META[String(i)] || {};
    const c = clusterRgb(i);
    const active = selectedClusters.size === 0 || selectedClusters.has(i);
    return `<div class="lg-row${active?' active':''}" id="lgr_${i}" onclick="toggleCluster(${i})">
      <span class="lg-dot" style="background:${c}"></span>
      <span class="lg-label" title="${escHtml(meta.label||'')}">C${i}: ${escHtml((meta.label||'').slice(0,22))}${(meta.label||'').length>22?'…':''}</span>
      <span class="lg-count">${meta.n_spans||0}</span>
    </div>`;
  });
  document.getElementById('legend').innerHTML =
    `<div class="lg-head">clusters (click to filter)</div>
     <div class="lg-row${selectedClusters.size===0?' active':''}" id="lgr_all" onclick="clearSel()">
       <span class="lg-dot" style="background:#888"></span>
       <span class="lg-label">show all</span>
       <span class="lg-count">${SPANS.length}</span>
     </div>` + rows.join('');
}

function toggleCluster(id){
  if(selectedClusters.has(id)) selectedClusters.delete(id);
  else selectedClusters.add(id);
  updateSelCount(); recolor(); buildLegend();
}
function clearSel(){ selectedClusters.clear(); updateSelCount(); recolor(); buildLegend(); }
function updateSelCount(){
  const el = document.getElementById('selCount');
  el.textContent = selectedClusters.size === 0 ? 'all' : selectedClusters.size+' selected';
}

/* ── t-SNE plots ── */
const LAYOUT = title => ({
  title:{text:title,font:{color:'#ddd',size:12},pad:{t:2}},
  paper_bgcolor:'#101014', plot_bgcolor:'#101014',
  scene:{xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},bgcolor:'#101014'},
  showlegend:false, margin:{l:0,r:0,t:28,b:0},
});

const MOD_LABELS = {state:'STATE', action:'ACTION', language:'LANGUAGE'};
const activeMods = Object.keys(MODS);
let camLock = false;

function ensurePanels(){
  const grid = document.getElementById('plotGrid');
  const rows = [];
  for(let i=0;i<activeMods.length;i+=2) rows.push(activeMods.slice(i,i+2));
  grid.innerHTML = rows.map(rowMods =>
    '<div class="panel-row">'+rowMods.map(mod =>
      `<div class="panel-wrap"><div class="panel-title">${MOD_LABELS[mod]||mod}</div>
       <div id="panel_${mod}" class="panel"></div></div>`
    ).join('')+'</div>'
  ).join('');
  grid.getBoundingClientRect();
}

function pointColors(mod){
  const m = MODS[mod];
  const opac = +document.getElementById('popac').value / 100;
  const size  = +document.getElementById('psize').value;
  const N = m.x.length;
  const colors = new Array(N);
  const sizes  = new Array(N);
  for(let k=0;k<N;k++){
    const s = SPANS[m.span_idx[k]];
    const cid = s ? s.cluster : 0;
    if(selectedClusters.size===0 || selectedClusters.has(cid)){
      colors[k] = clusterRgbA(cid, opac);
      sizes[k]  = size;
    } else {
      colors[k] = HIDDEN;
      sizes[k]  = 0;
    }
  }
  return {colors, sizes};
}

function renderTsne(){
  ensurePanels();
  const sz = +document.getElementById('psize').value;
  for(const mod of activeMods){
    const m = MODS[mod];
    const el = document.getElementById('panel_'+mod);
    if(!m||!el) continue;
    const {colors, sizes} = pointColors(mod);
    const custom = m.span_idx.map(si => {
      const s = SPANS[si]||{};
      return [si, s.cluster??0, (s.ep||'').slice(0,12), s.start??0, s.end??0,
              (s.text||'').slice(0,50), +(s.score||0).toFixed(3)];
    });
    Plotly.newPlot(el, [{
      type:'scatter3d', mode:'markers',
      x:m.x, y:m.y, z:m.z, customdata:custom,
      marker:{size:sizes, color:colors, line:{width:0}},
      hovertemplate:'<b>C%{customdata[1]}</b> · %{customdata[2]} · f%{customdata[3]}–%{customdata[4]}<br><i>%{customdata[5]}</i><br>score %{customdata[6]}<extra></extra>'
    },{
      type:'scatter3d', mode:'markers', name:'sel',
      x:[],y:[],z:[],hoverinfo:'skip',
      marker:{size:14,color:'rgba(255,220,0,0.95)',symbol:'diamond',line:{width:0}}
    }], LAYOUT(MOD_LABELS[mod]||mod), {responsive:true});

    el.onpointerdown = e => {el._px=e.clientX;el._py=e.clientY;};
    el.onpointerup   = e => {el._drag=Math.hypot(e.clientX-(el._px??e.clientX),e.clientY-(el._py??e.clientY))>5;};
    el.on('plotly_click', ev => {
      if(el._drag) return;
      const p = ev.points[0];
      if(p.curveNumber!==0) return;
      const [si, cid, ep, start, end, text, score] = p.customdata;
      const span = SPANS[si];
      if(!span) return;
      const mid = Math.floor((start+end)/2);
      const cColor = clusterRgb(cid);
      document.getElementById('info').innerHTML =
        `<b>${ep}</b> · f${start}–${end} · <span style="color:${cColor}">C${cid}: ${escHtml((CLUSTER_META[String(cid)]||{}).label||'')}</span> · `+
        `<i>${escHtml(text)}</i> · score ${score}`;
      showFrame(span.ep, mid, si);
      crossHighlight(mod, p.pointIndex);
    });
    el.on('plotly_relayout', ev => {
      if(!document.getElementById('syncCam')?.checked && false) return; // always sync
      if(ev['scene.camera'] && !camLock){
        camLock = true;
        const cam = ev['scene.camera'];
        Promise.all(activeMods.filter(x=>x!==mod).map(other=>
          Plotly.relayout('panel_'+other, {'scene.camera':cam})
        )).then(()=>camLock=false);
      }
    });
  }
}

function recolor(){
  for(const mod of activeMods){
    const m = MODS[mod];
    if(!m) continue;
    const {colors, sizes} = pointColors(mod);
    Plotly.restyle('panel_'+mod, {'marker.color':[colors],'marker.size':[sizes]}, [0]);
  }
}

function crossHighlight(srcMod, ptIdx){
  const m = MODS[srcMod];
  if(!m) return;
  const si = m.span_idx[ptIdx];
  for(const mod of activeMods){
    const mm = MODS[mod];
    if(!mm) continue;
    const ki = mm.span_idx.indexOf(si);
    if(ki<0){Plotly.restyle('panel_'+mod,{x:[[]],y:[[]],z:[[]]},[1]);continue;}
    Plotly.restyle('panel_'+mod,{x:[[mm.x[ki]]],y:[[mm.y[ki]]],z:[[mm.z[ki]]]},[1]);
  }
}

/* ── frame preview ── */
function showFrame(ep, frame, si){
  pvSpanIdx = si;
  document.getElementById('preview').style.display = 'block';
  document.getElementById('pv-img').src = FRAME_BASE+ep+'/'+frame;
  updateCap();
}
function updateCap(){
  const s = pvSpanIdx!=null ? SPANS[pvSpanIdx] : null;
  if(!s){document.getElementById('pv-cap').innerHTML='';return;}
  const mid = Math.floor((s.start+s.end)/2);
  document.getElementById('pv-cap').innerHTML =
    `<b>${s.ep.slice(0,12)}</b> · f<b>${mid}</b> (${s.start}–${s.end}) `+
    `<button onclick="stepSpan(-1)">−</button>`+
    `<button onclick="stepSpan(1)">+</button>`+
    `<a href="${VIDEO_BASE}${s.ep}" target="_blank" style="color:#7fd4ff">open↗</a>`;
}
let _pvFrame = null;
function stepSpan(d){
  if(pvSpanIdx==null) return;
  const s = SPANS[pvSpanIdx];
  if(_pvFrame==null) _pvFrame = Math.floor((s.start+s.end)/2);
  _pvFrame = Math.max(s.start, Math.min(s.end-1, _pvFrame+d));
  document.getElementById('pv-img').src = FRAME_BASE+s.ep+'/'+_pvFrame;
}
function hidePreview(){ document.getElementById('preview').style.display='none'; }

/* ── span grid ── */
function renderGrid(){
  const clusterFilter = document.getElementById('gCluster').value;
  const q = document.getElementById('gsearch').value.trim().toLowerCase();
  const sort = document.getElementById('gsort').value;

  let spans = SPANS.map((s,i)=>({...s,_i:i}));
  if(clusterFilter!=='') spans = spans.filter(s=>String(s.cluster)===clusterFilter);
  if(q) spans = spans.filter(s=>
    s.ep.toLowerCase().includes(q) ||
    (s.text||'').toLowerCase().includes(q)
  );
  if(sort==='asc') spans.sort((a,b)=>a.score-b.score);
  else spans.sort((a,b)=>b.score-a.score);

  const n = spans.length;
  document.getElementById('gridhead').innerHTML =
    `${n} span${n===1?'':'s'}${clusterFilter?` · C${clusterFilter}: ${escHtml((CLUSTER_META[clusterFilter]||{}).label||'')}`:''} · sorted by score`;

  const cards = spans.map((s,rank)=>{
    const c = clusterRgb(s.cluster);
    const dur = ((s.end-s.start)/FPS).toFixed(1);
    const pct = Math.max(0, Math.min(1, s.score));
    const cid = `sv_${s._i}`;
    return `<div class="card">
      <div class="hdr">
        <span class="rank">#${rank+1}</span>
        <span class="hash">${s.ep.slice(0,14)}…</span>
        <span class="cbadge" style="background:${clusterRgbA(s.cluster,.18)};color:${c}">C${s.cluster}</span>
        <span class="score">${s.score.toFixed(3)}</span>
      </div>
      <div class="ann-text" title="${escHtml(s.text)}">${escHtml((s.text||'').length>80?(s.text||'').slice(0,80)+'…':s.text||'')}</div>
      <div class="pct"><div style="width:${Math.round(pct*100)}%"></div></div>
      <div class="vid" id="${cid}" data-ep="${s.ep}" data-start="${s.start}" data-end="${s.end}">
        <div class="ph" onclick="loadSpanVideo('${cid}','${s.ep}',${s.start},${s.end})">
          <div class="play">▶</div>
          <div style="font-size:11px">f${s.start}–${s.end} (${dur}s)</div>
        </div>
      </div>
      <div class="clinks">
        <a href="${VIDEO_BASE}${s.ep}" target="_blank">open↗</a>
        <span style="color:#555">f${s.start}–${s.end} · ${dur}s</span>
      </div>
    </div>`;
  });
  document.getElementById('span-grid').innerHTML = cards.join('');
}

/* Throttled span-video loader. Each cell loads the full episode MP4 and clips to
   [start,end); browsers cap concurrent media streams (~6/host), so loading every
   cell at once stalls most to black. Load at most _MAXC at a time, starting the
   next as each finishes. _autoplay gates whether bulk-loaded clips play on ready. */
const _MAXC = 6;
let _vq = [], _vactive = 0, _autoplay = false;
function _vpump(){ while(_vactive < _MAXC && _vq.length){ _vactive++; (_vq.shift())(); } }
function _vrelease(){ _vactive = Math.max(0, _vactive - 1); _vpump(); }

function _mountSpanVideo(el, play){
  // play: () => bool — whether to start playback once metadata is ready.
  el.dataset.loaded = '1';
  const ep = el.dataset.ep, seekTo = (+el.dataset.start) / FPS, stopAt = (+el.dataset.end) / FPS;
  el.innerHTML = `<video src="${VIDEO_BASE}${ep}" controls muted playsinline preload="metadata"></video>`;
  const vid = el.querySelector('video');
  let released = false;
  const release = () => { if(!released){ released = true; _vrelease(); } };
  vid.addEventListener('loadedmetadata', () => {
    try { vid.currentTime = seekTo; } catch(e){}
    if(play()) vid.play().catch(()=>{});
    release();
  });
  vid.addEventListener('error', release);
  vid.addEventListener('timeupdate', () => {
    if(vid.currentTime >= stopAt){ vid.pause(); vid.currentTime = seekTo; }
  });
}

/* single click → load immediately (bypass queue) and play, or replay if already loaded */
function loadSpanVideo(cellId){
  const el = document.getElementById(cellId);
  if(!el) return;
  if(el.dataset.loaded){ const v = el.querySelector('video'); if(v) v.play().catch(()=>{}); return; }
  _mountSpanVideo(el, () => true);
}

function _enqueueCell(el){
  if(el.dataset.loaded){
    if(_autoplay){ const v = el.querySelector('video'); if(v) v.play().catch(()=>{}); }
    return;
  }
  el.dataset.loaded = '1';                       // claim now so it can't double-enqueue
  _vq.push(() => _mountSpanVideo(el, () => _autoplay));
  _vpump();
}
function loadAll(){ _autoplay = false; document.querySelectorAll('#span-grid .vid').forEach(_enqueueCell); }
function playAll(){ _autoplay = true;  document.querySelectorAll('#span-grid .vid').forEach(_enqueueCell); }
function pauseAll(){ _autoplay = false; _vq = []; document.querySelectorAll('#span-grid video').forEach(v=>v.pause()); }

/* ── init cluster selector ── */
(()=>{
  const sel = document.getElementById('gCluster');
  for(let i=0;i<N_CLUSTERS;i++){
    const meta = CLUSTER_META[String(i)]||{};
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = `C${i}: ${(meta.label||'').slice(0,28)}`;
    sel.appendChild(o);
  }
})();

/* ── navigation ── */
let page='tsne';
function showPage(p){
  page=p;
  document.getElementById('tsnepage').style.display = p==='tsne'?'flex':'none';
  document.getElementById('gridpage').style.display  = p==='grid'?'flex':'none';
  document.getElementById('tab-tsne').classList.toggle('active',p==='tsne');
  document.getElementById('tab-grid').classList.toggle('active',p==='grid');
  if(p==='grid') renderGrid();
}

function resetTools(){
  selectedClusters.clear(); updateSelCount();
  document.getElementById('psize').value=4;
  document.getElementById('popac').value=85;
  recolor(); buildLegend();
}

window.addEventListener('resize',()=>{
  activeMods.forEach(m=>{const el=document.getElementById('panel_'+m);if(el)Plotly.relayout(el,{autosize:true});});
});

/* ── boot ── */
buildLegend();
renderTsne();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_span_html(
    data: dict,
    *,
    video_base: str | None = None,
    frame_base: str | None = None,
    run_label: str = "",
) -> str:
    """Build the span viewer HTML from a parsed spans_tsne3d dict."""
    import html as _html

    vb = video_base if video_base is not None else VIDEO_BASE
    fb = frame_base if frame_base is not None else FRAME_BASE

    run_nav = ""
    if run_label:
        rl = _html.escape(run_label)
        run_nav = (
            f'<span style="color:#9aa">{rl}</span>'
            f' · <a href="/">change run</a>'
        )

    spans       = data.get("spans", [])
    clusters    = {str(k): v for k, v in data.get("clusters", {}).items()}
    n_clusters  = data.get("n_clusters", len(clusters))
    mods = {}
    for key in ("state", "action", "language"):
        if key in data:
            mods[key] = data[key]

    return (
        _TEMPLATE
        .replace("__SPANS__",        json.dumps(spans,      separators=(",", ":")))
        .replace("__CLUSTER_META__", json.dumps(clusters,   separators=(",", ":")))
        .replace("__N_CLUSTERS__",   str(n_clusters))
        .replace("__MODS__",         json.dumps(mods,       separators=(",", ":")))
        .replace("__VIDEO_BASE__",   vb)
        .replace("__FRAME_BASE__",   fb)
        .replace("__FPS__",          str(FPS))
        .replace("__RUN_LABEL__",    run_nav)
    )


def build_span_html_from_file(
    json_path: Path,
    *,
    video_base: str | None = None,
    frame_base: str | None = None,
    run_label: str = "",
) -> str:
    data = json.load(open(json_path))
    return build_span_html(data, video_base=video_base, frame_base=frame_base, run_label=run_label)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="spans_tsne3d.json path")
    ap.add_argument("--out", default="span_viz.html")
    ap.add_argument("--run-label", default="")
    args = ap.parse_args()

    html = build_span_html_from_file(Path(args.json), run_label=args.run_label)
    out = Path(args.out)
    out.write_text(html)
    print(f"Wrote {out.resolve()}  ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
