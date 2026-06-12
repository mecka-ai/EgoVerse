#!/usr/bin/env python3
"""
Build a self-contained interactive latent + episode-video viewer.

Two pages in one HTML file (works as a local file or served by the Modal web
app ``egomimic/modal/latent_viz_app.py``):

  * **t-SNE 3-D** — per-task 3-D scatters from ``tsne3d_<task>.json``.
    Modalities: ``state``, ``action``, and when language curation was enabled
    ``state_lang`` ([state∥language]), ``language``, ``state_by_lang`` (state
    geometry with instruction metadata). Color modes: episode, time, MI score,
    language (instruction cluster). Dynamic multi-panel grid, synced cameras,
    cross-highlight, frame-seek video preview.
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
import html as _html
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

# Default MP4 base URL for locally-built HTML. The unified Modal viewer
# (egomimic/modal/latent_viz_app.py) passes video_base="/video/" instead.
VIDEO_BASE = "https://mecka-robotics--egoverse-viewer-viewer.modal.run/video/"
FRAME_BASE = "https://mecka-robotics--egoverse-viewer-viewer.modal.run/frame/"
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
  html, body { height:100%; margin:0; overflow:hidden; }
  body { font-family:-apple-system,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--txt);
         display:flex; flex-direction:column; }
  /* top bar */
  #bar { flex-shrink:0; padding:7px 14px; display:flex; gap:10px; align-items:center; background:var(--bar);
         border-bottom:1px solid var(--line); flex-wrap:wrap; min-height:40px; }
  #bar h3 { margin:0; font-size:14px; font-weight:700; letter-spacing:.3px; color:#e8e8ea; }
  .tab { padding:4px 13px; border-radius:7px; cursor:pointer; background:#2a2b33; user-select:none;
         font-weight:600; font-size:13px; }
  .tab.active { background:var(--acc); }
  #info { font-size:12px; padding:4px 10px; background:#222; border-radius:6px; min-width:180px;
          max-width:440px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1; }
  #info b { color:#7fd4ff; }
  .run-nav { font-size:12px; color:#9aa; white-space:nowrap; margin-left:auto; }
  .run-nav a { color:#9ecbff; text-decoration:none; }
  .run-nav a:hover { color:#7fd4ff; }
  /* form controls */
  select, input[type=number], input[type=text] { font-size:13px; padding:4px 7px; background:#26272f;
         color:var(--txt); border:1px solid var(--line); border-radius:5px; }
  input[type=number] { width:66px; } input[type=text] { width:120px; }
  input[type=range] { accent-color:var(--acc); width:70px; }
  button { background:#2a2b33; color:#ddd; border:1px solid var(--line); border-radius:6px;
           padding:4px 10px; cursor:pointer; font-size:13px; }
  button:hover { background:#33353f; }
  button.primary { background:var(--acc); border-color:var(--acc); color:#fff; }
  .chk { display:flex; align-items:center; gap:4px; cursor:pointer; user-select:none; }
  .tool { display:flex; gap:5px; align-items:center; }
  .tool > label { color:#9aa; font-size:13px; }
  .sep { width:1px; height:18px; background:var(--line); margin:0 2px; flex-shrink:0; }
  /* t-SNE page */
  #tsnepage { flex:1; min-height:0; display:flex; flex-direction:column; overflow:hidden; }
  #tools { flex-shrink:0; padding:5px 14px; display:flex; gap:12px; align-items:center; flex-wrap:wrap;
           background:#15161b; border-bottom:1px solid var(--line); font-size:13px; }
  #modToggles { display:flex; gap:7px; align-items:center; flex-wrap:wrap; }
  #modToggles label.lang-mod { color:#c9b8ff; }
  #plots { flex:1; min-height:0; display:flex; overflow:hidden; }
  #plotGrid { flex:1; min-height:0; display:flex; flex-direction:column; overflow:hidden; }
  .panel-row { flex:1; min-height:0; display:flex; overflow:hidden;
               border-bottom:1px solid var(--line); }
  .panel-row:last-child { border-bottom:none; }
  .panel-wrap { flex:1; min-width:0; display:flex; flex-direction:column;
                border-right:1px solid var(--line); overflow:hidden; }
  .panel-wrap:last-child { border-right:none; }
  .panel-title { flex-shrink:0; font-size:11px; color:#9aa; padding:3px 8px; background:#14151a;
                 border-bottom:1px solid var(--line); font-weight:600; letter-spacing:.4px;
                 display:flex; align-items:center; gap:6px; }
  .pt-badge { font-size:10px; padding:1px 5px; border-radius:4px; background:#1e3a5f; color:#7fd4ff; }
  .panel { flex:1; min-height:0; overflow:hidden; }
  #legend { flex-shrink:0; width:170px; overflow-y:auto; background:#14151a;
            border-left:1px solid var(--line); font-size:12px; }
  #legend .lg-head { padding:6px 10px; color:#9aa; position:sticky; top:0; background:#14151a;
                     font-weight:600; font-size:11px; }
  .lg-row { display:flex; align-items:center; gap:6px; padding:3px 10px; cursor:pointer;
            font-family:ui-monospace,monospace; font-size:11px; color:#cdd; white-space:nowrap; }
  .lg-row:hover { background:#1f2027; }
  .lg-row.active { background:#24344f; }
  .lg-dot { width:9px; height:9px; border-radius:50%; flex:none; }
  #tstats { font-size:11px; color:#666; margin-left:auto; }
  /* grid page */
  #gridpage { flex:1; min-height:0; display:none; flex-direction:column; overflow:hidden; }
  #gtools { flex-shrink:0; padding:5px 14px; display:flex; gap:12px; align-items:center; flex-wrap:wrap;
            background:#15161b; border-bottom:1px solid var(--line); font-size:13px; }
  #gridwrap { flex:1; overflow-y:auto; padding:12px 16px; }
  #gridhead { margin:0 0 10px; color:#bbb; font-size:13px; display:flex; gap:12px;
              align-items:center; flex-wrap:wrap; }
  #grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:12px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .card .hdr { display:flex; align-items:center; gap:6px; padding:7px 10px; font-size:13px; }
  .rank { color:#888; min-width:30px; font-size:12px; }
  .hash { font-family:ui-monospace,monospace; color:#9ecbff; font-size:12px; }
  .score { margin-left:auto; font-weight:700; }
  .badge { font-size:10px; padding:2px 6px; border-radius:8px; font-weight:700; }
  .b-top { background:#1d4d2b; color:#7be495; }
  .b-bot { background:#4d1d1d; color:#ff9d9d; }
  .b-val { background:#4d3d1d; color:#ffd97b; }
  .pct { height:3px; background:#2a2a2a; }
  .pct > div { height:100%; background:linear-gradient(90deg,#e05555,#e0c14f,#54c46c); }
  .vid { position:relative; aspect-ratio:16/10; background:#0a0a0c; }
  .vid video { width:100%; height:100%; object-fit:contain; background:#000; }
  .ann { position:absolute; bottom:0; left:0; right:0; padding:4px 8px;
         background:rgba(0,0,0,0.68); color:#e8e8ea; font-size:12px; line-height:1.4;
         pointer-events:none; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
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
  #pv-cap button { padding:2px 7px; font-size:12px; }
  #pv-close { position:absolute; top:5px; right:7px; cursor:pointer; color:#aaa; font-size:14px;
              background:rgba(0,0,0,.6); border-radius:50%; width:20px; height:20px;
              text-align:center; line-height:20px; }
  #pv-close:hover { color:#fff; }
</style>
</head>
<body>
<div id="bar">
  <h3>EgoVerse</h3>
  <span class="tab active" id="tab-tsne" onclick="showPage('tsne')">t-SNE 3-D</span>
  <span class="tab" id="tab-grid" onclick="showPage('grid')">Video grid</span>
  <label class="tool" style="margin-left:4px">Task
    <select id="task"></select></label>
  <div id="info">Click a point to inspect</div>
  <div class="run-nav" id="runNav">__RUN_LABEL__</div>
</div>

<div id="tsnepage">
  <div id="tools">
    <span class="tool"><label>Color</label>
      <select id="colorMode" onchange="applyStyle();buildLegend(curTask)">
        <option value="episode">episode</option>
        <option value="time">time (light→dark)</option>
        <option value="score">MI score</option>
        <option value="language">language</option>
      </select></span>
    <span class="tool"><label>Episodes</label>
      <span id="epCount" style="background:#26272f;padding:3px 8px;border-radius:5px;font-size:12px;color:#cdd">all</span>
      <button onclick="clearEpSel()">clear</button></span>
    <div class="sep"></div>
    <span class="tool"><label class="chk"><input type="checkbox" id="hlOn" onchange="applyStyle()"> frame</label>
      <input type="number" id="hlFrame" value="0" min="0" step="10" onchange="hlChanged()">
      <label>±</label><input type="number" id="hlWin" value="15" min="0" step="5" onchange="hlChanged()"></span>
    <span class="tool"><label>Time%</label>
      <input type="number" id="t0" value="0" min="0" max="100" onchange="applyStyle()">–
      <input type="number" id="t1" value="100" min="0" max="100" onchange="applyStyle()"></span>
    <span class="tool"><label>Size</label>
      <input type="range" id="psize" min="1" max="8" value="3" onchange="applyStyle()"></span>
    <span class="tool"><label class="chk"><input type="checkbox" id="syncCam" checked> sync cameras</label></span>
    <div class="sep"></div>
    <span id="modToggles"></span>
    <button onclick="resetTools()">reset</button>
    <span id="tstats"></span>
  </div>
  <div id="plots">
    <div id="plotGrid"></div>
    <div id="legend"></div>
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
    <span class="tool"><label>Find</label>
      <input type="text" id="gsearch" placeholder="hash prefix…" oninput="render()"></span>
    <button onclick="loadAll()">Load all</button>
    <button class="primary" onclick="playAll()">▶ Play all</button>
    <button onclick="pauseAll()">⏸ Pause</button>
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
  <img id="pv-img" alt="">
  <div id="pv-cap"></div>
</div>

<script>
const DATA = __DATA__;
const SCORES = __SCORES__;
const VAL = new Set(__VAL__);
const VIDEO_BASE = "__VIDEO_BASE__";
const FRAME_BASE = "__FRAME_BASE__";
const FPS = __FPS__;
const DIM = "rgba(110,110,120,0.05)";
const HIDDEN = "rgba(0,0,0,0)";
let selectedEps = new Set();

/* color helpers */
function hsv2rgb(h,s,v){
  const i=Math.floor(h*6),f=h*6-i;
  const p=v*(1-s),q=v*(1-f*s),t=v*(1-(1-f)*s);
  const c=[[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i%6];
  return [c[0]*255,c[1]*255,c[2]*255];
}
function rgb(c){return `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;}
function lerp3(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}
const redGreen=t=>lerp3([224,85,85],[84,196,108],t);

function scoreNorm(task){
  const e=SCORES[task]||[];
  if(!e.length)return{};
  const vals=e.map(x=>x[1]),mn=Math.min(...vals),mx=Math.max(...vals);
  const out={};
  e.forEach(([h,s])=>out[h]=mx>mn?(s-mn)/(mx-mn):0.5);
  return out;
}

/* modality setup */
const MOD_ORDER=["state","state_by_lang","state_lang","language","action"];
const MOD_LABELS={
  state:"STATE (image)",
  state_by_lang:"STATE · by language",
  state_lang:"STATE ∥ LANGUAGE",
  language:"LANGUAGE",
  action:"ACTION",
};
const MOD_LANG_ONLY=new Set(["state_by_lang","state_lang","language"]);

const CACHE={};
let curTask=null;
let activeMods=[];
let hiddenMods=new Set();

function modalitiesForTask(task){
  const d=DATA[task]||{};
  return MOD_ORDER.filter(m=>d[m]&&d[m].x&&d[m].x.length);
}

function langColor(langId,nLabels){
  return rgb(hsv2rgb((langId||0)/Math.max(1,nLabels),0.8,0.9));
}

function buildCache(task){
  const eps=DATA[task].episodes,nEp=eps.length;
  const sn=scoreNorm(task);
  const langLabels=DATA[task].language_labels||[];
  const nLang=Math.max(1,langLabels.length);
  const mods={};
  for(const mod of modalitiesForTask(task)){
    const d=DATA[task][mod];
    const N=d.x.length;
    const tables={episode:new Array(N),time:new Array(N),score:new Array(N),language:new Array(N)};
    const custom=new Array(N);
    for(let k=0;k<N;k++){
      const e=d.ep[k],tf=d.t[k];
      tables.episode[k]=rgb(hsv2rgb(e/Math.max(1,nEp),0.85,1.0-0.65*tf));
      tables.time[k]=rgb(lerp3([170,215,255],[10,40,90],tf));
      tables.score[k]=rgb(redGreen(sn[eps[e]]??0.5));
      const lid=(d.lang_id&&d.lang_id[k]!=null)?d.lang_id[k]:0;
      tables.language[k]=langColor(lid,nLang);
      const langTxt=(d.lang&&d.lang[k])?String(d.lang[k]).slice(0,40):"";
      custom[k]=[d.frame[k],Math.round(tf*100),eps[e].slice(0,10),e,langTxt];
    }
    mods[mod]={d,tables,custom};
  }
  return{mods,langLabels};
}

function uiState(){
  return{
    mode:document.getElementById("colorMode").value,
    hlOn:document.getElementById("hlOn").checked,
    hlF:+document.getElementById("hlFrame").value,
    hlW:+document.getElementById("hlWin").value,
    t0:+document.getElementById("t0").value/100,
    t1:+document.getElementById("t1").value/100,
    size:+document.getElementById("psize").value,
  };
}

function colorTableForMod(m,mode){
  if(mode==="language"&&m.d.lang_id)return m.tables.language;
  if(mode==="language")return m.tables.episode;
  return m.tables[mode]||m.tables.episode;
}

function applyStyle(){
  if(!curTask||!CACHE[curTask])return;
  const u=uiState();
  for(const mod of activeMods){
    const m=CACHE[curTask].mods[mod];
    if(!m)continue;
    const{d}=m;
    const base=colorTableForMod(m,u.mode);
    const N=d.x.length;
    const colors=new Array(N);
    let sizes=u.size;
    if(u.hlOn)sizes=new Array(N);
    for(let k=0;k<N;k++){
      const on=d.t[k]>=u.t0&&d.t[k]<=u.t1
        &&(selectedEps.size===0||selectedEps.has(d.ep[k]))
        &&(!u.hlOn||Math.abs(d.frame[k]-u.hlF)<=u.hlW);
      colors[k]=on?base[k]:(selectedEps.size>0?HIDDEN:DIM);
      if(u.hlOn)sizes[k]=on?u.size*2.2:u.size;
    }
    Plotly.restyle("panel_"+mod,{"marker.color":[colors],"marker.size":Array.isArray(sizes)?[sizes]:sizes},[0]);
  }
}

const LAYOUT=title=>({
  title:{text:title,font:{color:"#ddd",size:12},pad:{t:2}},
  paper_bgcolor:"#101014",plot_bgcolor:"#101014",
  scene:{xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},bgcolor:"#101014"},
  showlegend:false,
  margin:{l:0,r:0,t:28,b:0},
});

let camLock=false;

function buildModToggles(task){
  const allMods=modalitiesForTask(task);
  const container=document.getElementById("modToggles");
  if(!allMods.length){container.innerHTML="";return;}
  const hasLang=allMods.some(m=>MOD_LANG_ONLY.has(m));
  let html='<label style="color:#9aa;font-size:13px">Panels:</label>';
  for(const m of allMods){
    const checked=!hiddenMods.has(m)?"checked":"";
    const cls=MOD_LANG_ONLY.has(m)?' class="lang-mod"':"";
    html+=`<label class="chk"${cls} title="${MOD_LABELS[m]||m}">` +
          `<input type="checkbox" ${checked} onchange="toggleMod('${m}',this.checked)"> ${m}</label>`;
  }
  container.innerHTML=html;
}

function toggleMod(mod,visible){
  if(visible)hiddenMods.delete(mod);
  else hiddenMods.add(mod);
  if(curTask)renderTsne(curTask,true);
}

function ensurePanels(mods){
  const grid=document.getElementById("plotGrid");
  // Group into rows of 2 so each row stretches to fill available height via flex.
  const rows=[];
  for(let i=0;i<mods.length;i+=2) rows.push(mods.slice(i,i+2));
  grid.innerHTML=rows.map(rowMods=>
    '<div class="panel-row">'+rowMods.map(mod=>{
      const isLang=MOD_LANG_ONLY.has(mod);
      const badge=isLang?'<span class="pt-badge">language</span>':"";
      return `<div class="panel-wrap">` +
             `<div class="panel-title">${MOD_LABELS[mod]||mod.toUpperCase()} ${badge}</div>` +
             `<div id="panel_${mod}" class="panel"></div></div>`;
    }).join("")+'</div>'
  ).join("");
  // Force synchronous layout so Plotly reads non-zero flex-computed heights.
  grid.getBoundingClientRect();
}

function renderTsne(task,preserveToggles){
  if(!CACHE[task])CACHE[task]=buildCache(task);
  curTask=task;
  const allMods=modalitiesForTask(task);
  activeMods=allMods.filter(m=>!hiddenMods.has(m));

  if(!preserveToggles)buildModToggles(task);

  const u=uiState();

  ensurePanels(activeMods);

  for(const mod of activeMods){
    const m=CACHE[task].mods[mod];
    const el=document.getElementById("panel_"+mod);
    if(!m||!el)continue;
    const{d,tables,custom}=m;
    const colors=colorTableForMod(m,u.mode);
    Plotly.newPlot(el,[
      {type:"scatter3d",mode:"markers",
       x:d.x,y:d.y,z:d.z,customdata:custom,
       marker:{size:u.size,color:colors,line:{width:0}},
       hovertemplate:"ep %{customdata[2]} · f%{customdata[0]} (%{customdata[1]}%)<br>%{customdata[4]}<extra></extra>"},
      {type:"scatter3d",mode:"markers",name:"sel",
       x:[],y:[],z:[],hoverinfo:"skip",
       marker:{size:12,color:"rgba(255,200,0,0.95)",symbol:"diamond",line:{width:0}}},
    ],LAYOUT((MOD_LABELS[mod]||mod)+" — "+task),{responsive:true});

    el.onpointerdown=e=>{el._px=e.clientX;el._py=e.clientY;};
    el.onpointerup=e=>{el._drag=Math.hypot(e.clientX-(el._px??e.clientX),e.clientY-(el._py??e.clientY))>5;};
    el.on("plotly_click",ev=>{
      if(el._drag)return;
      const p=ev.points[0];
      if(p.curveNumber!==0)return;
      const[frame,tpct,,epIdx,langTxt]=p.customdata;
      const hash=DATA[task].episodes[epIdx];
      const langLine=langTxt?` · <span style="color:#c9b8ff">${langTxt}</span>`:"";
      document.getElementById("info").innerHTML=
        `<b>${hash.slice(0,14)}</b> · f<b>${frame}</b>(${tpct}%)${langLine} · `+
        `<a href="${VIDEO_BASE}${hash}" target="_blank" style="color:#7fd4ff">video↗</a>`;
      document.getElementById("hlFrame").value=frame;
      showFrame(task,hash,frame,tpct/100);
      crossHighlight(task,epIdx,frame);
    });
    el.on("plotly_relayout",ev=>{
      if(!document.getElementById("syncCam").checked||camLock)return;
      if(ev["scene.camera"]){
        camLock=true;
        const cam=ev["scene.camera"];
        Promise.all(activeMods.filter(x=>x!==mod).map(other=>
          Plotly.relayout("panel_"+other,{"scene.camera":cam})
        )).then(()=>camLock=false);
      }
    });
  }
  const d=DATA[task]||{};
  const nPts=activeMods.reduce((a,m)=>a+((d[m]||{}).x||[]).length,0);
  const langNote=d.language_enabled?` · lang=${d.language_mode||"on"}`:"";
  const langMods=allMods.filter(m=>MOD_LANG_ONLY.has(m));
  const hidNote=hiddenMods.size?` · <span style="color:#f87">${hiddenMods.size} hidden</span>`:"";
  document.getElementById("tstats").innerHTML=
    `${(d.episodes||[]).length} eps · ${nPts.toLocaleString()} pts · every ${d.every_n||10}f`+
    `${langNote}${hidNote}`;
  buildLegend(task);
  applyStyle();
}

function buildLegend(task){
  const mode=document.getElementById("colorMode").value;
  const cache=CACHE[task]||{langLabels:[]};
  if(mode==="language"&&cache.langLabels&&cache.langLabels.length){
    const rows=cache.langLabels.map((label,i)=>{
      const c=langColor(i,cache.langLabels.length);
      return `<div class="lg-row" title="${label.replace(/"/g,"&quot;")}">` +
             `<span class="lg-dot" style="background:${c}"></span>` +
             `${String(label).slice(0,24)}${label.length>24?"…":""}</div>`;
    }).join("");
    document.getElementById("legend").innerHTML=
      `<div class="lg-head">instructions (${cache.langLabels.length})</div>`+rows;
    return;
  }
  const eps=DATA[task]?DATA[task].episodes:[];
  const rows=eps.map((h,i)=>{
    const c=rgb(hsv2rgb(i/Math.max(1,eps.length),0.85,0.85));
    const sc=SCORES[task]?(SCORES[task].find(e=>e[0]===h)||[0,NaN])[1]:NaN;
    return `<div class="lg-row" id="lg_${i}" onclick="legendClick(${i})" title="MI ${isNaN(sc)?"?":sc.toFixed(4)}">` +
           `<span class="lg-dot" style="background:${c}"></span>${h.slice(0,10)}</div>`;
  }).join("");
  document.getElementById("legend").innerHTML=
    `<div class="lg-head">episodes <span style="color:#666;font-weight:400">(click to multi-select)</span></div>` +
    `<div class="lg-row" id="lg_all" onclick="legendClick('all')">` +
    `<span class="lg-dot" style="background:#888"></span>show all</div>`+rows;
  markLegend();
}

function legendClick(i){
  if(i==="all"){selectedEps.clear();}
  else if(selectedEps.has(i)){selectedEps.delete(i);}
  else selectedEps.add(i);
  updateEpDisplay();applyStyle();markLegend();
}

function clearEpSel(){selectedEps.clear();updateEpDisplay();applyStyle();markLegend();}

function updateEpDisplay(){
  const el=document.getElementById("epCount");
  if(el)el.textContent=selectedEps.size===0?"all":selectedEps.size+" selected";
}

function markLegend(){
  document.querySelectorAll("#legend .lg-row").forEach(r=>r.classList.remove("active"));
  if(selectedEps.size===0){
    const el=document.getElementById("lg_all");if(el)el.classList.add("active");
  }else{
    selectedEps.forEach(i=>{const el=document.getElementById("lg_"+i);if(el)el.classList.add("active");});
  }
}

function crossHighlight(task,epIdx,frame){
  for(const mod of activeMods){
    const m=(CACHE[task]||{mods:{}}).mods[mod];
    if(!m)continue;
    const d=m.d;
    let xs=[],ys=[],zs=[];
    for(let k=0;k<d.x.length;k++){
      if(d.ep[k]===epIdx&&d.frame[k]===frame){xs.push(d.x[k]);ys.push(d.y[k]);zs.push(d.z[k]);break;}
    }
    Plotly.restyle("panel_"+mod,{x:[xs],y:[ys],z:[zs]},[1]);
  }
}

function hlChanged(){document.getElementById("hlOn").checked=true;applyStyle();}

function resetTools(){
  document.getElementById("colorMode").value="episode";
  selectedEps.clear();updateEpDisplay();
  document.getElementById("hlOn").checked=false;
  document.getElementById("hlFrame").value=0;
  document.getElementById("hlWin").value=15;
  document.getElementById("t0").value=0;
  document.getElementById("t1").value=100;
  document.getElementById("psize").value=3;
  hiddenMods.clear();
  if(curTask){buildModToggles(curTask);renderTsne(curTask);}
  else applyStyle();
}

/* frame preview */
let pvHash=null,pvFrame=0,pvTfrac=null;

function showFrame(task,hash,frame,tfrac){
  pvHash=hash;pvFrame=frame;pvTfrac=tfrac;
  document.getElementById("preview").style.display="block";
  document.getElementById("pv-img").src=FRAME_BASE+hash+"/"+frame;
  updateCap();
}

function updateCap(){
  document.getElementById("pv-cap").innerHTML=
    `<b>${pvHash?pvHash.slice(0,12):""}</b> · f<b>${pvFrame}</b>`+
    (pvTfrac!=null?` (${Math.round(pvTfrac*100)})%`:"")+
    ` <button onclick="stepFrame(-1)">−1f</button>`+
    ` <button onclick="stepFrame(1)">+1f</button>`+
    ` <a href="${VIDEO_BASE}${pvHash}" target="_blank" style="color:#7fd4ff">open↗</a>`;
}

function stepFrame(d){
  pvFrame=Math.max(0,pvFrame+d);pvTfrac=null;
  document.getElementById("pv-img").src=FRAME_BASE+pvHash+"/"+pvFrame;
  updateCap();
}

function hidePreview(){
  document.getElementById("preview").style.display="none";
}

/* build per-episode frame→annotation lookup from t-SNE data */
const LANG_MAP={};
(()=>{
  for(const task of Object.keys(DATA)){
    const d=DATA[task];
    const eps=d.episodes||[];
    const byHash={};
    for(const mod of ['state','state_lang','language','state_by_lang','action']){
      const md=d[mod];
      if(!md||!md.lang||!md.lang.length)continue;
      for(let k=0;k<md.ep.length;k++){
        const text=md.lang[k];
        if(!text)continue;
        const hash=eps[md.ep[k]];
        if(!byHash[hash])byHash[hash]=[];
        byHash[hash].push([md.frame[k],text]);
      }
      break;
    }
    for(const h of Object.keys(byHash))byHash[h].sort((a,b)=>a[0]-b[0]);
    LANG_MAP[task]=byHash;
  }
})();

/* video grid */
function loadVideo(cellId,hash,task){
  const el=document.getElementById(cellId);
  const anns=(LANG_MAP[task]||{})[hash]||[];
  el.innerHTML=`<video src="${VIDEO_BASE}${hash}" controls loop muted playsinline preload="metadata"></video>`+
    (anns.length?`<div class="ann" id="${cellId}_ann"></div>`:'');
  if(anns.length){
    const vid=el.querySelector('video');
    const annEl=document.getElementById(cellId+'_ann');
    vid.addEventListener('timeupdate',()=>{
      const f=Math.floor(vid.currentTime*FPS);
      let lo=0,hi=anns.length-1,best=-1;
      while(lo<=hi){const mid=(lo+hi)>>1;if(anns[mid][0]<=f){best=mid;lo=mid+1;}else hi=mid-1;}
      annEl.textContent=best>=0?anns[best][1]:'';
    });
  }
}

function histoSVG(scores){
  if(!scores.length)return"";
  const mn=Math.min(...scores),mx=Math.max(...scores),nb=24;
  const bins=new Array(nb).fill(0);
  scores.forEach(s=>bins[Math.min(nb-1,Math.floor((s-mn)/((mx-mn)||1)*nb))]++);
  const bmax=Math.max(...bins);
  const bars=bins.map((b,i)=>
    `<rect x="${i*8}" y="${28-26*b/bmax}" width="6" height="${26*b/bmax}" fill="${rgb(redGreen(i/(nb-1)))}"/>`).join("");
  return `<svg width="${nb*8}" height="30" title="score distribution">${bars}</svg>`;
}

function renderGrid(task){
  let entries=(SCORES[task]||[]).slice();
  const n=entries.length,nTop=Math.ceil(0.6*n);
  const rank={};entries.forEach((e,i)=>rank[e[0]]=i);
  const allScores=entries.map(e=>e[1]);
  const mn=Math.min(...allScores),mx=Math.max(...allScores);
  const mean=allScores.reduce((a,b)=>a+b,0)/Math.max(1,n);

  const fTop=document.getElementById("fTop").checked;
  const fBot=document.getElementById("fBot").checked;
  const fVal=document.getElementById("fVal").checked;
  const q=document.getElementById("gsearch").value.trim().toLowerCase();
  entries=entries.filter(([h])=>{
    const isTop=rank[h]<nTop;
    if(isTop&&!fTop)return false;
    if(!isTop&&!fBot)return false;
    if(fVal&&!VAL.has(h))return false;
    if(q&&!h.toLowerCase().startsWith(q))return false;
    return true;
  });
  if(document.getElementById("gsort").value==="asc")entries.reverse();

  document.getElementById("gridhead").innerHTML=
    `<b>${task}</b> — ${entries.length}/${n} episodes · MI mean ${mean.toFixed(3)} · [${mn.toFixed(3)}, ${mx.toFixed(3)}] `+
    histoSVG(allScores);

  const cards=entries.map(([hash,score])=>{
    const i=rank[hash];
    const pct=mx>mn?(score-mn)/(mx-mn):0.5;
    const badge=i<nTop?'<span class="badge b-top">TOP 60%</span>':'<span class="badge b-bot">BOT 40%</span>';
    const valb=VAL.has(hash)?' <span class="badge b-val">VAL</span>':"";
    const cid=`v_${task}_${i}`;
    return `<div class="card">
      <div class="hdr"><span class="rank">#${i+1}</span>
        <span class="hash">${hash.slice(0,16)}…</span>${badge}${valb}
        <span class="score">${score.toFixed(4)}</span></div>
      <div class="pct"><div style="width:${Math.round(pct*100)}%"></div></div>
      <div class="vid" id="${cid}">
        <div class="ph" onclick="loadVideo('${cid}','${hash}','${task}')">
          <div class="play">▶</div><div style="font-size:12px">load video</div></div>
      </div>
      <div class="clinks">
        <a href="${VIDEO_BASE}${hash}" target="_blank">open↗</a>
        <span style="color:#555">rank ${i+1}/${n} · ${(100*(1-i/Math.max(1,n-1))).toFixed(0)}th pct</span>
      </div>
    </div>`;
  });
  document.getElementById("grid").innerHTML=cards.join("");
}

function loadAll(){document.querySelectorAll("#grid .ph").forEach(ph=>ph.click());}
function playAll(){loadAll();setSpeed();document.querySelectorAll("#grid video").forEach(v=>{v.muted=true;v.play().catch(()=>{});});}
function pauseAll(){document.querySelectorAll("#grid video").forEach(v=>v.pause());}
function setSpeed(){
  const r=parseFloat(document.getElementById("gspeed").value);
  document.querySelectorAll("#grid video").forEach(v=>v.playbackRate=r);
}

/* navigation */
let page="tsne";
function showPage(p){
  page=p;
  document.getElementById("tsnepage").style.display=p==="tsne"?"flex":"none";
  document.getElementById("gridpage").style.display=p==="grid"?"flex":"none";
  document.getElementById("tab-tsne").classList.toggle("active",p==="tsne");
  document.getElementById("tab-grid").classList.toggle("active",p==="grid");
  render();
}

function render(){
  const task=document.getElementById("task").value;
  if(page==="tsne")renderTsne(task);else renderGrid(task);
}

window.addEventListener("resize",()=>{
  if(curTask&&page==="tsne")
    activeMods.forEach(m=>{ const el=document.getElementById("panel_"+m); if(el) Plotly.relayout(el,{autosize:true}); });
});

const sel=document.getElementById("task");
const taskNames=Array.from(new Set([...Object.keys(SCORES),...Object.keys(DATA)])).sort();
taskNames.forEach(t=>{const o=document.createElement("option");o.value=o.textContent=t;sel.appendChild(o);});
sel.onchange=render;
render();
</script>
</body>
</html>
"""


def build_html(
    tsne_dir: Path,
    scores_raw: dict,
    val: list,
    *,
    video_base: str | None = None,
    frame_base: str | None = None,
    run_label: str = "",
) -> str:
    """Assemble the viewer HTML from a local tsne3d dir + raw scores dict."""
    vb = video_base if video_base is not None else VIDEO_BASE
    fb = frame_base if frame_base is not None else FRAME_BASE
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
    run_nav = ""
    if run_label:
        rl = _html.escape(run_label)
        run_nav = (
            f'<span style="color:#9aa">{rl}</span>'
            f' · <a href="/">change run</a>'
            f' · <a href="/episodes">episodes</a>'
        )
    return (
        _TEMPLATE
        .replace("__DATA__", json.dumps(data, separators=(",", ":")))
        .replace("__SCORES__", json.dumps(scores, separators=(",", ":")))
        .replace("__VAL__", json.dumps(val, separators=(",", ":")))
        .replace("__VIDEO_BASE__", vb)
        .replace("__FRAME_BASE__", fb)
        .replace("__FPS__", str(FPS))
        .replace("__RUN_LABEL__", run_nav)
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
