"""Build a self-contained HTML viewer for language-cluster curation runs.

Structurally identical to build_latent_viz.py (the task viewer): all data is
embedded and the page renders synchronously on load — no async fetch, no
spinners, no failure modes. Clusters take the role that tasks/episodes play in
the task viewer.

  * Same CSS variables, top bar (tabs / dropdown / info / run-nav)
  * Same t-SNE tools: Color (cluster/score/time), Cluster multi-select, frame
    highlight, size slider, sync cameras, panel toggles, reset, tstats
  * Same 3-panel layout (STATE | ACTION | LANGUAGE) with Plotly scatter3d
  * Same legend sidebar — clusters replace episodes (clickable, multi-select)
  * Same frame-preview popup
  * Same grid page: sort, TOP/BOT filters, Find, Load-all / Play-all / Pause /
    Speed — adapted to play each span clip (looping within its [start,end])
"""
from __future__ import annotations

import json

_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Meckaverse</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root { --bg:#101014; --bar:#1a1b21; --card:#1c1d24; --line:#2b2d36; --acc:#3b82f6; --txt:#e8e8ea; }
  html, body { height:100%; margin:0; overflow:hidden; }
  body { font-family:-apple-system,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--txt);
         display:flex; flex-direction:column; }
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
  #tsnepage { flex:1; min-height:0; display:flex; flex-direction:column; overflow:hidden; }
  #tools { flex-shrink:0; padding:5px 14px; display:flex; gap:12px; align-items:center; flex-wrap:wrap;
           background:#15161b; border-bottom:1px solid var(--line); font-size:13px; }
  #modToggles { display:flex; gap:7px; align-items:center; flex-wrap:wrap; }
  #plots { flex:1; min-height:0; display:flex; overflow:hidden; }
  #plotGrid { flex:1; min-height:0; display:flex; flex-direction:column; overflow:hidden; }
  .panel-row { flex:1; min-height:0; display:flex; overflow:hidden; border-bottom:1px solid var(--line); }
  .panel-row:last-child { border-bottom:none; }
  .panel-wrap { flex:1; min-width:0; display:flex; flex-direction:column;
                border-right:1px solid var(--line); overflow:hidden; }
  .panel-wrap:last-child { border-right:none; }
  .panel-title { flex-shrink:0; font-size:11px; color:#9aa; padding:3px 8px; background:#14151a;
                 border-bottom:1px solid var(--line); font-weight:600; letter-spacing:.4px;
                 display:flex; align-items:center; gap:6px; }
  .pt-badge { font-size:10px; padding:1px 5px; border-radius:4px; background:#1e3a5f; color:#7fd4ff; }
  .panel { flex:1; min-height:0; overflow:hidden; }
  #nodata { flex:1; display:none; align-items:center; justify-content:center; color:#666; font-size:13px; }
  #legend { flex-shrink:0; width:170px; overflow-y:auto; background:#14151a;
            border-left:1px solid var(--line); font-size:12px; }
  #legend .lg-head { padding:6px 10px; color:#9aa; position:sticky; top:0; background:#14151a;
                     font-weight:600; font-size:11px; }
  .lg-row { display:flex; align-items:center; gap:6px; padding:3px 10px; cursor:pointer;
            font-size:11px; color:#cdd; white-space:nowrap; }
  .lg-row:hover { background:#1f2027; }
  .lg-row.active { background:#24344f; }
  .lg-dot { width:9px; height:9px; border-radius:50%; flex:none; }
  .lg-lbl { flex:1; overflow:hidden; text-overflow:ellipsis; font-size:10px; }
  .lg-cnt { font-size:10px; color:#555; flex-shrink:0; }
  #tstats { font-size:11px; color:#666; margin-left:auto; }
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
  .vid .ph img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain;
                 opacity:.35; z-index:-1; }
  .ph .play { font-size:28px; }
  .clinks { padding:6px 10px; font-size:11px; display:flex; gap:10px; }
  .clinks a { color:#7fd4ff; text-decoration:none; }
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
  <h3>Meckaverse</h3>
  <span class="tab active" id="tab-tsne" onclick="showPage('tsne')">t-SNE 3-D</span>
  <span class="tab" id="tab-grid" onclick="showPage('grid')">Video grid</span>
  <label class="tool" style="margin-left:4px">Cluster
    <select id="csel" onchange="onCSelDrop()"><option value="">— all —</option></select></label>
  <div id="info">Click a point to inspect</div>
  <div class="run-nav" id="runNav">__RUN_LABEL__</div>
</div>

<div id="tsnepage">
  <div id="tools">
    <span class="tool"><label>Color</label>
      <select id="colorMode" onchange="applyStyle()">
        <option value="cluster">cluster</option>
        <option value="episode">episode</option>
        <option value="score">MI score</option>
        <option value="time">time (light&rarr;dark)</option>
      </select></span>
    <span class="tool"><label>Select</label>
      <select id="selMode" onchange="onSelMode()">
        <option value="cluster">cluster</option>
        <option value="episode">episode</option>
      </select></span>
    <span class="tool"><label>Clusters</label>
      <span id="cCount" style="background:#26272f;padding:3px 8px;border-radius:5px;font-size:12px;color:#cdd">all</span>
      <button onclick="clearClusterSel()">clear</button></span>
    <div class="sep"></div>
    <span class="tool"><label class="chk"><input type="checkbox" id="hlOn" onchange="applyStyle()"> frame</label>
      <input type="number" id="hlFrame" value="0" min="0" step="10" onchange="hlChanged()">
      <label>&plusmn;</label><input type="number" id="hlWin" value="30" min="0" step="10" onchange="hlChanged()"></span>
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
    <div id="nodata">No t-SNE embeddings for this run (tsne3d/spans_tsne3d.json missing). Video grid still works.</div>
    <div id="legend"></div>
  </div>
</div>

<div id="gridpage">
  <div id="gtools">
    <span class="tool"><label>Sort</label>
      <select id="gsort" onchange="renderGrid()">
        <option value="desc">score &darr; (best first)</option>
        <option value="asc">score &uarr; (worst first)</option>
        <option value="frame">episode / frame</option>
      </select></span>
    <span class="tool">
      <label class="chk"><input type="checkbox" id="fTop" checked onchange="renderGrid()"> TOP 60%</label>
      <label class="chk"><input type="checkbox" id="fBot" checked onchange="renderGrid()"> BOT 40%</label></span>
    <span class="tool"><label>Find</label>
      <input type="text" id="gsearch" placeholder="text or hash&hellip;" oninput="renderGrid()"></span>
    <button onclick="loadAll()">Load all</button>
    <button class="primary" onclick="playAll()">&#9654; Play all</button>
    <button onclick="pauseAll()">&#9646;&#9646; Pause</button>
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
  <div id="pv-close" onclick="hidePreview()">&times;</div>
  <img id="pv-img" alt="">
  <div id="pv-cap"></div>
</div>

<script>
/* ── embedded data ── */
const CLUSTERS   = __CLUSTERS__;
const TSNE       = __TSNE__;          // {} if no spans_tsne3d.json; else {cid,score,start,end,ep,txt, state:{x,y,z}, …}
const VIDEO_BASE = "__VIDEO_BASE__";
const FRAME_BASE = "__FRAME_BASE__";
const FPS        = 30;

const HAS_TSNE = TSNE && TSNE.cid && TSNE.cid.length;
const ALL_CID = Object.keys(CLUSTERS).sort((a,b)=>
  parseInt(a.replace('cluster_','')) - parseInt(b.replace('cluster_',''))
);
const TOTAL_SPANS = ALL_CID.reduce((s,c)=>s+CLUSTERS[c].spans.length,0);

/* ── runtime state ── */
let COLOR_TABLES  = null;             // {cluster:[…], score:[…], time:[…]}
let curCluster    = null;             // string "cluster_N" or null — drives the grid
let selectedClusters = new Set();     // Set<int> — multi-highlight in scatter + grid (cluster mode)
let selectedEpisodes = new Set();     // Set<str> — multi-highlight in scatter + grid (episode mode)
let hiddenMods    = new Set();
let activeMods    = [];
let camLock       = false;
let page          = 'tsne';

const MOD_ORDER  = ['state','action','language'];
const MOD_LABELS = {state:'STATE', action:'ACTION', language:'LANGUAGE'};
const DIM        = 'rgba(110,110,120,0.05)';

/* ── color helpers ── */
function hsv2rgb(h,s,v){
  const i=Math.floor(h*6),f=h*6-i;
  const p=v*(1-s),q=v*(1-f*s),t=v*(1-(1-f)*s);
  const c=[[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i%6];
  return[c[0]*255,c[1]*255,c[2]*255];
}
function rgb(c){return`rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;}
function lerp3(a,b,t){return[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t];}
const redGreen = t=>lerp3([224,85,85],[84,196,108],Math.max(0,Math.min(1,t)));
function cidHue(ci){return(ci*137.508)%360/360;}
function cidColor(ci){return rgb(hsv2rgb(cidHue(ci),0.85,0.9));}
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}

/* ── episode index + colors (for the 'episode' color mode + episode legend) ── */
const EP_LIST = (()=>{const cnt={}; ALL_CID.forEach(cid=>CLUSTERS[cid].spans.forEach(s=>{cnt[s.ep]=(cnt[s.ep]||0)+1;})); return Object.keys(cnt).sort().map((ep,i)=>({ep,count:cnt[ep],idx:i}));})();
const EP_IDX = Object.fromEntries(EP_LIST.map(e=>[e.ep,e.idx]));
function epColor(ep){const i=EP_IDX[ep]??0; return rgb(hsv2rgb((i*137.508)%360/360,0.85,0.9));}

/* ── color tables (built once) ── */
function buildColorTables(){
  if(!HAS_TSNE) return;
  const N=TSNE.cid.length;
  const scores=TSNE.score, starts=TSNE.start;
  const finite=scores.filter(s=>isFinite(s));
  const mn=finite.length?Math.min(...finite):0, mx=finite.length?Math.max(...finite):1;
  const s0=Math.min(...starts), s1=Math.max(...starts)||1;
  const ct={cluster:new Array(N), episode:new Array(N), score:new Array(N), time:new Array(N)};
  for(let k=0;k<N;k++){
    ct.cluster[k]=cidColor(TSNE.cid[k]);
    ct.episode[k]=epColor(TSNE.ep[k]);
    ct.score[k]=rgb(redGreen(mx>mn?(scores[k]-mn)/(mx-mn):0.5));
    ct.time[k]=rgb(lerp3([170,215,255],[10,40,90],(starts[k]-s0)/Math.max(1,s1-s0)));
  }
  COLOR_TABLES=ct;
}

/* ── ui state ── */
function uiState(){
  return{
    mode: document.getElementById('colorMode').value,
    hlOn: document.getElementById('hlOn').checked,
    hlF:  +document.getElementById('hlFrame').value,
    hlW:  +document.getElementById('hlWin').value,
    size: +document.getElementById('psize').value,
  };
}

/* ── applyStyle ── */
function applyStyle(){
  if(!HAS_TSNE||!COLOR_TABLES) return;
  const u=uiState();
  const N=TSNE.cid.length;
  const base=COLOR_TABLES[u.mode]||COLOR_TABLES.cluster;
  const colors=new Array(N), sizes=new Array(N);
  const selEp = document.getElementById('selMode').value==='episode';  // selection dim (decoupled from color)
  const sel = selEp ? selectedEpisodes : selectedClusters;
  for(let k=0;k<N;k++){
    const ci=TSNE.cid[k];
    const passSel = sel.size===0 || sel.has(selEp ? TSNE.ep[k] : ci);
    const passHl = !u.hlOn || (TSNE.start[k]<=u.hlF+u.hlW && TSNE.end[k]>=u.hlF-u.hlW);
    if(passSel && passHl){ colors[k]=base[k]; sizes[k]=u.hlOn?u.size*2.2:u.size; }
    else if(!passSel){ colors[k]='rgba(0,0,0,0)'; sizes[k]=0; }   // selection active → hide non-selected entirely
    else { colors[k]=DIM; sizes[k]=u.size; }
  }
  activeMods.forEach(mod=>{
    Plotly.restyle('panel_'+mod,{'marker.color':[colors],'marker.size':[sizes]},[0]);
  });
}

/* ── panels ── */
const LAYOUT = title=>({
  title:{text:title,font:{color:'#ddd',size:12},pad:{t:2}},
  paper_bgcolor:'#101014', plot_bgcolor:'#101014',
  scene:{xaxis:{visible:false},yaxis:{visible:false},zaxis:{visible:false},bgcolor:'#101014'},
  showlegend:false, margin:{l:0,r:0,t:28,b:0},
});

function buildModToggles(){
  const avail=MOD_ORDER.filter(m=>TSNE&&TSNE[m]);
  if(!avail.length){document.getElementById('modToggles').innerHTML='';return;}
  let html='<label style="color:#9aa;font-size:13px">Panels:</label>';
  avail.forEach(m=>{
    const chk=!hiddenMods.has(m)?'checked':'';
    html+=`<label class="chk"><input type="checkbox" ${chk} onchange="toggleMod('${m}',this.checked)"> ${m}</label>`;
  });
  document.getElementById('modToggles').innerHTML=html;
}

function toggleMod(mod,vis){
  if(vis)hiddenMods.delete(mod); else hiddenMods.add(mod);
  renderTsne(true);
}

function ensurePanels(mods){
  const grid=document.getElementById('plotGrid');
  const rows=[];
  for(let i=0;i<mods.length;i+=2) rows.push(mods.slice(i,i+2));
  grid.innerHTML=rows.map(rm=>
    '<div class="panel-row">'+rm.map(mod=>`
      <div class="panel-wrap">
        <div class="panel-title">${MOD_LABELS[mod]||mod}
          <span class="pt-badge">${TOTAL_SPANS} spans</span></div>
        <div id="panel_${mod}" class="panel"></div>
      </div>`).join('')+'</div>'
  ).join('');
  grid.getBoundingClientRect();          // force synchronous layout for Plotly height
}

function renderTsne(preserveToggles){
  if(!HAS_TSNE){
    document.getElementById('plotGrid').style.display='none';
    document.getElementById('legend').style.display='none';
    document.getElementById('nodata').style.display='flex';
    document.getElementById('tstats').textContent=`${TOTAL_SPANS} spans · no embeddings`;
    return;
  }
  if(!preserveToggles) buildModToggles();
  const avail=MOD_ORDER.filter(m=>TSNE[m]);
  activeMods=avail.filter(m=>!hiddenMods.has(m));
  const u=uiState();
  const base=COLOR_TABLES[u.mode]||COLOR_TABLES.cluster;
  ensurePanels(activeMods);

  for(const mod of activeMods){
    const t=TSNE[mod];
    const el=document.getElementById('panel_'+mod);
    if(!t||!el) continue;
    const N=t.x.length;
    const custom=new Array(N);
    for(let k=0;k<N;k++){
      custom[k]=[k, TSNE.cid[k], TSNE.start[k], isFinite(TSNE.score[k])?TSNE.score[k].toFixed(3):'?', TSNE.txt[k]||''];
    }
    Plotly.newPlot(el,[
      {type:'scatter3d',mode:'markers',
       x:t.x,y:t.y,z:t.z,customdata:custom,
       marker:{size:u.size,color:base,line:{width:0}},
       hovertemplate:'cluster %{customdata[1]} · f%{customdata[2]} · %{customdata[3]}<br>%{customdata[4]}<extra></extra>'},
      {type:'scatter3d',mode:'markers',name:'sel',
       x:[],y:[],z:[],hoverinfo:'skip',
       marker:{size:12,color:'rgba(255,200,0,0.95)',symbol:'diamond',line:{width:0}}},
    ],LAYOUT(MOD_LABELS[mod]+' — cluster view'),{responsive:true});

    el.onpointerdown=e=>{el._px=e.clientX;el._py=e.clientY;};
    el.onpointerup=e=>{el._drag=Math.hypot(e.clientX-(el._px??e.clientX),e.clientY-(el._py??e.clientY))>5;};
    el.on('plotly_click',ev=>{
      if(el._drag) return;
      const p=ev.points[0]; if(!p||p.curveNumber!==0) return;
      const [k,ci,frame,sc,txt]=p.customdata;
      const ep=TSNE.ep[k]||'';
      document.getElementById('info').innerHTML=
        `<b>cluster_${ci}</b> &middot; f<b>${frame}</b> &middot; ${sc}`+
        (txt?` &middot; <span style="color:#c9b8ff">${esc(txt)}</span>`:'');
      document.getElementById('hlFrame').value=frame;
      showFrame(ep,frame);
      crossHighlight(k);
    });
    el.on('plotly_relayout',ev=>{
      if(!document.getElementById('syncCam').checked||camLock) return;
      if(ev['scene.camera']){
        camLock=true;
        const cam=ev['scene.camera'];
        Promise.all(activeMods.filter(x=>x!==mod).map(other=>
          Plotly.relayout('panel_'+other,{'scene.camera':cam})
        )).then(()=>camLock=false);
      }
    });
  }

  const hidden=hiddenMods.size?` &middot; <span style="color:#f87">${hiddenMods.size} hidden</span>`:'';
  document.getElementById('tstats').innerHTML=
    `${TOTAL_SPANS.toLocaleString()} spans &middot; ${avail.length} modes${hidden}`;
  buildLegend();
  applyStyle();
}

function crossHighlight(k){
  for(const mod of activeMods){
    const t=TSNE[mod]; if(!t) continue;
    Plotly.restyle('panel_'+mod,{x:[[t.x[k]]],y:[[t.y[k]]],z:[[t.z[k]]]},[1]);
  }
}

/* ── legend (clusters, multi-select) ── */
function buildLegend(){
  const el=document.getElementById('legend');
  const epMode=document.getElementById('selMode').value==='episode';
  if(epMode){
    el.innerHTML=
      `<div class="lg-head">episodes <span style="color:#666;font-weight:400">(click to select)</span></div>`+
      `<div class="lg-row" id="lg_all" onclick="epLegendClick(null)">
         <span class="lg-dot" style="background:#888"></span>
         <span class="lg-lbl">show all</span></div>`+
      EP_LIST.map(e=>{
        const active=selectedEpisodes.has(e.ep);
        return`<div class="lg-row${active?' active':''}" id="lg_ep_${e.idx}" onclick="epLegendClick('${e.ep}')" title="${esc(e.ep)}">
          <span class="lg-dot" style="background:${epColor(e.ep)}"></span>
          <span class="lg-lbl">${esc(e.ep.slice(0,16))}…</span>
          <span class="lg-cnt">${e.count}</span>
        </div>`;
      }).join('');
  } else {
    el.innerHTML=
      `<div class="lg-head">clusters <span style="color:#666;font-weight:400">(click to select)</span></div>`+
      `<div class="lg-row" id="lg_all" onclick="legendClick(null)">
         <span class="lg-dot" style="background:#888"></span>
         <span class="lg-lbl">show all</span></div>`+
      ALL_CID.map(cid=>{
        const ci=parseInt(cid.replace('cluster_',''));
        const c=CLUSTERS[cid];
        const lbl=c.label.length>22?c.label.slice(0,21)+'…':c.label;
        const active=selectedClusters.has(ci);
        return`<div class="lg-row${active?' active':''}" id="lg_${ci}" onclick="legendClick(${ci})" title="${esc(c.label)}">
          <span class="lg-dot" style="background:${cidColor(ci)}"></span>
          <span class="lg-lbl">${esc(lbl)}</span>
          <span class="lg-cnt">${c.spans.length}</span>
        </div>`;
      }).join('');
  }
  markLegend();
}

function legendClick(ci){
  if(ci===null){
    selectedClusters.clear(); curCluster=null;
    document.getElementById('csel').value='';
  } else if(selectedClusters.has(ci)){
    selectedClusters.delete(ci);
    if(curCluster==='cluster_'+ci) curCluster=null;
  } else {
    selectedClusters.add(ci);
    curCluster='cluster_'+ci;
    document.getElementById('csel').value='cluster_'+ci;
  }
  updateClusterDisplay(); applyStyle(); markLegend();
  if(page==='grid') renderGrid();
}

function markLegend(){
  document.querySelectorAll('#legend .lg-row').forEach(r=>r.classList.remove('active'));
  const epMode=document.getElementById('selMode').value==='episode';
  const sz = epMode ? selectedEpisodes.size : selectedClusters.size;
  if(sz===0){ const el=document.getElementById('lg_all'); if(el) el.classList.add('active'); return; }
  if(epMode) selectedEpisodes.forEach(ep=>{const el=document.getElementById('lg_ep_'+EP_IDX[ep]); if(el) el.classList.add('active');});
  else selectedClusters.forEach(ci=>{const el=document.getElementById('lg_'+ci); if(el) el.classList.add('active');});
}

function epLegendClick(ep){
  if(ep===null||ep===''){ selectedEpisodes.clear(); }
  else if(selectedEpisodes.has(ep)){ selectedEpisodes.delete(ep); }
  else { selectedEpisodes.add(ep); }
  applyStyle(); markLegend();
  if(page==='grid') renderGrid();
}

function onSelMode(){
  buildLegend();          // switch sidebar between clusters and episodes
  updateClusterDisplay();
  applyStyle();           // re-highlight by the new selection dimension
  if(page==='grid') renderGrid();
}

function updateClusterDisplay(){
  const el=document.getElementById('cCount');
  const sz=document.getElementById('selMode').value==='episode'?selectedEpisodes.size:selectedClusters.size;
  if(el) el.textContent=sz===0?'all':sz+' selected';
}

function clearClusterSel(){
  selectedClusters.clear(); curCluster=null;
  document.getElementById('csel').value='';
  updateClusterDisplay(); applyStyle(); markLegend();
  if(page==='grid') renderGrid();
}

/* ── cluster dropdown ── */
function buildCSel(){
  const sel=document.getElementById('csel');
  ALL_CID.forEach(cid=>{
    const o=document.createElement('option');
    o.value=cid;
    const l=CLUSTERS[cid].label;
    o.textContent=cid+': '+(l.length>36?l.slice(0,35)+'…':l);
    sel.appendChild(o);
  });
}

function onCSelDrop(){
  const v=document.getElementById('csel').value;
  curCluster=v||null;
  selectedClusters.clear();
  if(curCluster) selectedClusters.add(parseInt(curCluster.replace('cluster_','')));
  updateClusterDisplay(); applyStyle(); markLegend();
  if(page==='grid') renderGrid();
}

function hlChanged(){document.getElementById('hlOn').checked=true; applyStyle();}

function resetTools(){
  document.getElementById('colorMode').value='cluster';
  document.getElementById('selMode').value='cluster';
  selectedClusters.clear(); curCluster=null;
  document.getElementById('csel').value='';
  selectedEpisodes.clear();
  document.getElementById('hlOn').checked=false;
  document.getElementById('hlFrame').value=0;
  document.getElementById('hlWin').value=30;
  document.getElementById('psize').value=3;
  hiddenMods.clear();
  updateClusterDisplay();
  if(HAS_TSNE){buildModToggles(); renderTsne(false);}
  if(page==='grid') renderGrid();
}

/* ── frame-preview popup ── */
let pvHash=null, pvFrame=0;
function showFrame(ep,frame){
  pvHash=ep; pvFrame=frame;
  document.getElementById('preview').style.display='block';
  document.getElementById('pv-img').src=FRAME_BASE+ep+'/'+frame;
  updateCap();
}
function updateCap(){
  document.getElementById('pv-cap').innerHTML=
    `<b>${pvHash?pvHash.slice(0,12):''}</b> &middot; f<b>${pvFrame}</b>`+
    ` <button onclick="stepFrame(-1)">&minus;1f</button>`+
    ` <button onclick="stepFrame(1)">+1f</button>`+
    ` <a href="${VIDEO_BASE}${pvHash}" target="_blank" style="color:#7fd4ff">open&nearr;</a>`;
}
function stepFrame(d){
  pvFrame=Math.max(0,pvFrame+d);
  document.getElementById('pv-img').src=FRAME_BASE+pvHash+'/'+pvFrame;
  updateCap();
}
function hidePreview(){document.getElementById('preview').style.display='none';}

/* ── video grid ── */
function histoSVG(scores){
  if(!scores.length) return'';
  const mn=Math.min(...scores),mx=Math.max(...scores),nb=24;
  const bins=new Array(nb).fill(0);
  scores.forEach(s=>bins[Math.min(nb-1,Math.floor((s-mn)/((mx-mn)||1)*nb))]++);
  const bmax=Math.max(...bins);
  const bars=bins.map((b,i)=>
    `<rect x="${i*8}" y="${28-26*b/bmax}" width="6" height="${26*b/bmax}" fill="${rgb(redGreen(i/(nb-1)))}"/>`).join('');
  return`<svg width="${nb*8}" height="30" title="score distribution">${bars}</svg>`;
}

function renderGrid(){
  const selEp=document.getElementById('selMode').value==='episode';
  let spans=[];
  if(!selEp && curCluster && CLUSTERS[curCluster]){   // cluster-mode: one cluster's spans
    spans=CLUSTERS[curCluster].spans.map(s=>({...s,_cid:curCluster}));
  } else {
    ALL_CID.forEach(cid=>spans.push(...CLUSTERS[cid].spans.map(s=>({...s,_cid:cid}))));
  }

  const n=spans.length, nTop=Math.ceil(0.6*n);
  const allScores=spans.map(s=>s.score).filter(v=>isFinite(v));
  const mn=allScores.length?Math.min(...allScores):0;
  const mx=allScores.length?Math.max(...allScores):1;
  const mean=allScores.reduce((a,b)=>a+b,0)/Math.max(1,allScores.length);

  // rank by score desc for TOP/BOT, regardless of display sort
  const byScore=spans.slice().sort((a,b)=>b.score-a.score);
  const rank={}; byScore.forEach((s,i)=>rank[s.id]=i);

  const sort=document.getElementById('gsort').value;
  if(sort==='desc') spans.sort((a,b)=>b.score-a.score);
  else if(sort==='asc') spans.sort((a,b)=>a.score-b.score);
  else spans.sort((a,b)=>a.ep===b.ep?a.start-b.start:a.ep<b.ep?-1:1);

  const fTop=document.getElementById('fTop').checked;
  const fBot=document.getElementById('fBot').checked;
  const q=document.getElementById('gsearch').value.trim().toLowerCase();

  const filtered=spans.filter(s=>{
    if(selEp && selectedEpisodes.size && !selectedEpisodes.has(s.ep)) return false;
    const isTop=(rank[s.id]??0)<nTop;
    if(isTop&&!fTop) return false;
    if(!isTop&&!fBot) return false;
    if(q && !s.ep.toLowerCase().startsWith(q) && !s.text.toLowerCase().includes(q)) return false;
    return true;
  });

  const lbl=curCluster?`${curCluster}: ${CLUSTERS[curCluster].label}`:'all clusters';
  document.getElementById('gridhead').innerHTML=
    `<b>${esc(lbl)}</b> &mdash; ${filtered.length}/${n} spans &middot; `+
    `mean ${mean.toFixed(3)} &middot; [${mn.toFixed(3)}, ${mx.toFixed(3)}] `+
    histoSVG(allScores);

  let idc=0;
  const cards=filtered.map(s=>{
    const ri=rank[s.id]??0;
    const pct=mx>mn?(s.score-mn)/(mx-mn):0.5;
    const isTop=ri<nTop;
    const badge=isTop?'<span class="badge b-top">TOP 60%</span>':'<span class="badge b-bot">BOT 40%</span>';
    const mid=Math.round((s.start+s.end)/2);
    const cid='vc_'+(idc++);
    return`<div class="card">
      <div class="hdr">
        <span class="rank">#${ri+1}</span>
        <span class="hash">${s.ep.slice(0,14)}&hellip;</span>${badge}
        <span class="score">${isFinite(s.score)?s.score.toFixed(4):'?'}</span>
      </div>
      <div class="pct"><div style="width:${Math.round(pct*100)}%"></div></div>
      <div class="vid" id="${cid}"
           data-ep="${s.ep}" data-start="${s.start}" data-end="${s.end}" data-text="${esc(s.text)}">
        <div class="ph" onclick="loadSpan('${cid}')">
          <img src="${FRAME_BASE}${s.ep}/${mid}" loading="lazy" onerror="this.style.display='none'">
          <div class="play">&#9654;</div>
          <div style="font-size:12px">load clip · f${s.start}&ndash;${s.end}</div>
        </div>
      </div>
      <div class="clinks">
        <a href="${VIDEO_BASE}${s.ep}" target="_blank">open&nearr;</a>
        <span style="color:#555">${s._cid}</span>
      </div>
    </div>`;
  }).join('');
  document.getElementById('grid').innerHTML=cards||
    '<div style="padding:20px;color:#555">No spans match the current filters.</div>';
}

/* ── span clip playback (loops within [start,end]) ──
   Mirrors the task viewer's loadVideo: clicking the placeholder swaps in a
   <video> element; the clip is bounded by a timeupdate handler that wraps back
   to the start frame, so it loops the span like the task viewer loops episodes. */
/* Throttled clip loader. Each cell loads the full episode MP4 and clips to
   [start,end). Browsers cap concurrent media streams (~6/host), so loading every
   cell at once (the old loadAll → ph.click() on all) stalls most to black, and
   playAll's immediate play() ran before any clip had loaded/seeked. Load at most
   _MAXC concurrently, starting the next as each settles; _autoplay gates whether
   bulk-loaded clips play once ready. Single click still loads + plays instantly. */
const _MAXC=6;
let _vq=[], _vactive=0, _autoplay=false;
function _vpump(){ while(_vactive<_MAXC && _vq.length){ _vactive++; (_vq.shift())(); } }
function _vrelease(){ _vactive=Math.max(0,_vactive-1); _vpump(); }

function _mountCell(el, play, onSettled){
  // play: () => bool — whether to start playback once ready. A paused <video> after
  // a programmatic seek does not paint in most browsers (shows black), so to render
  // a still we still play() then pause; here we loop within [seekTo, stopAt].
  el.dataset.loaded='1';
  const ep=el.dataset.ep;
  const seekTo=(+el.dataset.start)/FPS, stopAt=(+el.dataset.end)/FPS;
  const text=el.dataset.text||'';
  const spd=parseFloat(document.getElementById('gspeed').value)||1;
  el.innerHTML=
    `<video src="${VIDEO_BASE}${ep}" controls loop muted playsinline preload="metadata"></video>`+
    (text?`<div class="ann">${text}</div>`:'');
  const vid=el.querySelector('video');
  vid.playbackRate=spd;
  let done=false; const settle=()=>{ if(!done){ done=true; if(onSettled) onSettled(); } };
  vid.addEventListener('loadedmetadata',()=>{ vid.currentTime=seekTo; if(play()) vid.play().catch(()=>{}); settle(); },{once:true});
  vid.addEventListener('error', settle, {once:true});
  vid.addEventListener('seeked',()=>{ if(play()) vid.play().catch(()=>{}); },{once:true});
  vid.addEventListener('timeupdate',()=>{
    if(vid.currentTime>=stopAt || vid.currentTime<seekTo-0.1){ vid.currentTime=seekTo; }
  });
}

/* single click → load now (bypass queue) + play, or replay if already loaded */
function loadSpan(cellId){
  const el=document.getElementById(cellId);
  if(!el) return;
  if(el.dataset.loaded==='1'){ const v=el.querySelector('video'); if(v) v.play().catch(()=>{}); return; }
  _mountCell(el, ()=>true, null);
}

function _enqueueCell(el){
  if(el.dataset.loaded==='1'){ if(_autoplay){ const v=el.querySelector('video'); if(v) v.play().catch(()=>{}); } return; }
  el.dataset.loaded='1';                          // claim now so it can't double-enqueue
  _vq.push(()=>_mountCell(el, ()=>_autoplay, _vrelease));
  _vpump();
}
function loadAll(){ _autoplay=false; document.querySelectorAll('#grid .vid').forEach(_enqueueCell); }
function playAll(){ _autoplay=true; setSpeed(); document.querySelectorAll('#grid .vid').forEach(_enqueueCell); }
function pauseAll(){ _autoplay=false; _vq=[]; document.querySelectorAll('#grid video').forEach(v=>v.pause()); }
function setSpeed(){
  const r=parseFloat(document.getElementById('gspeed').value);
  document.querySelectorAll('#grid video').forEach(v=>v.playbackRate=r);
}

/* ── page switch ── */
function showPage(p){
  page=p;
  document.getElementById('tsnepage').style.display=p==='tsne'?'flex':'none';
  document.getElementById('gridpage').style.display=p==='grid'?'flex':'none';
  document.getElementById('tab-tsne').classList.toggle('active',p==='tsne');
  document.getElementById('tab-grid').classList.toggle('active',p==='grid');
  if(p==='tsne'&&HAS_TSNE){
    // Plotly needs a resize once the page becomes visible again
    activeMods.forEach(m=>{const e=document.getElementById('panel_'+m);if(e)Plotly.relayout(e,{autosize:true});});
    applyStyle();
  }
  if(p==='grid') renderGrid();
}

window.addEventListener('resize',()=>{
  if(page==='tsne')
    activeMods.forEach(m=>{const el=document.getElementById('panel_'+m);if(el)Plotly.relayout(el,{autosize:true});});
});

/* ── init ── */
buildCSel();
updateClusterDisplay();
buildColorTables();
renderTsne(false);      // renders panels synchronously (or shows no-data note)
renderGrid();
</script>
</body>
</html>"""


def build_cluster_html(
    clusters_raw: dict,
    tsne: dict | None = None,
    *,
    video_base: str,
    frame_base: str,
    run_label: str,
) -> str:
    clusters_js: dict = {}
    for cluster_id, c in clusters_raw.items():
        spans_list = []
        scores = []
        for span_key, s in c.get("spans", {}).items():
            raw = s.get("score")
            score_f = float(raw) if raw is not None else float("nan")
            spans_list.append({
                "id":    span_key,
                "ep":    s["episode"],
                "start": int(s["start"]),
                "end":   int(s["end"]),
                "text":  s.get("text", ""),
                "score": score_f if score_f == score_f else 0.0,  # nan→0 for JSON
            })
            if score_f == score_f:
                scores.append(score_f)
        clusters_js[cluster_id] = {
            "label":    c.get("label", cluster_id),
            "spans":    sorted(spans_list, key=lambda x: -x["score"]),
            "minScore": min(scores) if scores else 0.0,
            "maxScore": max(scores) if scores else 1.0,
        }

    run_escaped = run_label.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    run_nav = (
        f'<span style="color:#9aa">{run_label}</span>'
        f' &middot; <a href="/">change run</a>'
        f' &middot; <a href="/episodes">episodes</a>'
    )

    return (
        _TEMPLATE
        .replace("__CLUSTERS__",          json.dumps(clusters_js, separators=(",", ":")))
        .replace("__TSNE__",               json.dumps(tsne or {}, separators=(",", ":")))
        .replace("__VIDEO_BASE__",         video_base)
        .replace("__FRAME_BASE__",         frame_base)
        .replace("__RUN_LABEL_ESCAPED__",  run_escaped)
        .replace("__RUN_LABEL__",          run_nav)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("clustered_scores_json")
    parser.add_argument("--tsne", help="path to tsne3d/spans_tsne3d.json", default=None)
    parser.add_argument("--out", default="cluster_viewer.html")
    parser.add_argument("--video-base", default="/video/")
    parser.add_argument("--frame-base", default="/frame/")
    args = parser.parse_args()
    data = json.load(open(args.clustered_scores_json))

    tsne = None
    if args.tsne:
        raw = json.load(open(args.tsne))
        spans = raw["spans"]
        tsne = {
            "cid":   [s["cluster"] for s in spans],
            "score": [s.get("score") or 0.0 for s in spans],
            "start": [int(s.get("start", 0)) for s in spans],
            "end":   [int(s.get("end", s.get("start", 0) + 1)) for s in spans],
            "ep":    [s.get("ep", s.get("episode", "")) for s in spans],
            "txt":   [str(s.get("text", ""))[:60] for s in spans],
        }
        for mode in ("state", "action", "language"):
            if mode in raw:
                tsne[mode] = {k: raw[mode][k] for k in ("x", "y", "z")}

    html = build_cluster_html(
        data, tsne,
        video_base=args.video_base,
        frame_base=args.frame_base,
        run_label=args.clustered_scores_json,
    )
    open(args.out, "w").write(html)
    print(f"wrote {args.out} ({len(html)//1024} KB)")
