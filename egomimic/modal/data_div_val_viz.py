"""Modal web viewer for D1-D5 validation videos + metric curves (epoch >= 400).

Serves a page with:
  - epoch tab buttons (>= 400)
  - two synced line charts: Valid/Loss and paired_mse_avg over global_step, with a
    marker at the selected epoch per model
  - a 5-column grid (D1..D5) of the epoch's validation videos, with synced vertical
    scrolling so the same validation sample lines up across all models.

Videos are read from the egoverse-training-outputs volume:
  data_div_val5/D{i}/videos/epoch_{N}/MECKA_BIMANUAL/validation_video_{j}.mp4
Metric curves are baked in from metrics.json (pre-fetched from W&B).

Deploy:  MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/data_div_val_viz.py
"""
import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]==0.115.*", "uvicorn")
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "data_div_metrics.json"),
        "/root/metrics.json",
    )
)
app = modal.App("data-div-val-viz", image=image)
outputs_vol = modal.Volume.from_name("egoverse-training-outputs")

BASE = "/vol/data_div_val5"
MODELS = ["D1", "D2", "D3", "D4", "D5"]
MIN_EPOCH = 400


@app.function(
    image=image,
    volumes={"/vol": outputs_vol},
    min_containers=1,
    scaledown_window=1200,
    timeout=3600,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import json
    import re
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    api = FastAPI()
    metrics = json.load(open("/root/metrics.json"))

    def scan_avail():
        """{model: {epoch:int -> num_videos}} for epochs >= MIN_EPOCH."""
        avail = {}
        for m in MODELS:
            vdir = f"{BASE}/{m}/videos"
            eps = {}
            if os.path.isdir(vdir):
                for d in os.listdir(vdir):
                    mobj = re.fullmatch(r"epoch_(\d+)", d)
                    if not mobj:
                        continue
                    ep = int(mobj.group(1))
                    if ep < MIN_EPOCH:
                        continue
                    mdir = f"{vdir}/{d}/MECKA_BIMANUAL"
                    if os.path.isdir(mdir):
                        n = len([f for f in os.listdir(mdir) if f.endswith(".mp4")])
                        if n:
                            eps[ep] = n
            avail[m] = eps
        return avail

    @api.get("/api/manifest")
    def manifest():
        avail = scan_avail()
        epochs = sorted({ep for m in MODELS for ep in avail[m]})
        max_videos = max([0] + [n for m in MODELS for n in avail[m].values()])
        return JSONResponse(
            {
                "models": MODELS,
                "epochs": epochs,
                "avail": {m: {str(k): v for k, v in avail[m].items()} for m in MODELS},
                "num_videos": max_videos,
                "metrics": metrics,
            }
        )

    @api.get("/video")
    def video(d: str, epoch: int, idx: int):
        if d not in MODELS:
            raise HTTPException(404, "bad model")
        path = f"{BASE}/{d}/videos/epoch_{epoch}/MECKA_BIMANUAL/validation_video_{idx}.mp4"
        if not os.path.isfile(path):
            raise HTTPException(404, "no video")
        return FileResponse(path, media_type="video/mp4")

    @api.get("/")
    def index():
        return HTMLResponse(PAGE)

    return api


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Data Diversity — Val Videos (D1–D5)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root{--bg:#0e1117;--panel:#161b22;--fg:#e6edf3;--mut:#8b949e;--acc:#58a6ff;--bd:#30363d;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:12px 16px;border-bottom:1px solid var(--bd)}
  h1{font-size:16px;margin:0 0 8px}
  #epochs{display:flex;flex-wrap:wrap;gap:6px}
  .ep{background:var(--panel);border:1px solid var(--bd);color:var(--fg);padding:5px 11px;border-radius:6px;cursor:pointer;font-variant-numeric:tabular-nums}
  .ep:hover{border-color:var(--acc)}
  .ep.on{background:var(--acc);color:#04101f;border-color:var(--acc);font-weight:600}
  #charts{display:grid;grid-template-columns:1fr 1fr;gap:10px;padding:10px 16px;border-bottom:1px solid var(--bd)}
  .chart{background:var(--panel);border:1px solid var(--bd);border-radius:8px;height:240px}
  #gridwrap{padding:0 16px 24px}
  #head,.row{display:grid;grid-template-columns:120px repeat(5,1fr);gap:10px;align-items:center}
  #head{position:sticky;top:0;background:var(--bg);z-index:3;padding:8px 0;border-bottom:1px solid var(--bd)}
  #head .h{font-weight:600;text-align:center}
  #head .h .mut{color:var(--mut);font-weight:400}
  .row{padding:8px 0;border-bottom:1px solid #21262d}
  .row.on{background:#0d2438}
  .rowctl{display:flex;flex-direction:column;gap:8px;align-items:center;justify-content:center}
  .rowctl .s{color:var(--fg);font-weight:600;font-size:12px}
  .rowctl button{width:46px;height:46px;font-size:18px;border-radius:50%;background:var(--panel);border:1px solid var(--bd);color:var(--fg);cursor:pointer}
  .rowctl button:hover{border-color:var(--acc);color:var(--acc)}
  .row video{width:100%;aspect-ratio:16/9;object-fit:contain;border-radius:5px;background:#000;display:block;cursor:pointer}
  .row .none{aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:12px;border:1px dashed var(--bd);border-radius:5px}
  .row .slot{aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:11px;background:#0b0f14;border:1px solid var(--bd);border-radius:5px}
  #bar{display:flex;gap:14px;align-items:center;padding:6px 16px;color:var(--mut);flex-wrap:wrap}
  #bar b{color:var(--fg)}
  button.ctl{background:var(--panel);border:1px solid var(--bd);color:var(--fg);padding:4px 10px;border-radius:6px;cursor:pointer}
  .swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:middle}
</style></head>
<body>
<header>
  <h1>Data Diversity — Validation Videos across D1–D5 · epoch ≥ 400</h1>
  <div id="epochs"></div>
</header>
<div id="bar">
  <span>selected epoch: <b id="curep">–</b></span>
  <button class="ctl" id="playall">⏸ pause all rows</button>
  <span style="color:var(--mut)">· click a row's <b style="color:var(--fg)">⬇ load</b> to fetch its 5 clips, then play/pause them together</span>
  <span id="legend"></span>
  <span id="note" style="color:#d29922"></span>
</div>
<div id="charts">
  <div class="chart" id="c_loss"></div>
  <div class="chart" id="c_mse"></div>
</div>
<div id="gridwrap">
  <div id="head"></div>
  <div id="rows"></div>
</div>
<script>
const COLORS={D1:"#58a6ff",D2:"#3fb950",D3:"#d29922",D4:"#bc8cff",D5:"#ff7b72"};
let M=null, EP=null, syncing=false;

function fmtK(v){return (v/1000).toFixed(0)+"k";}

async function boot(){
  M = await (await fetch("api/manifest")).json();
  // epoch tabs
  const eb=document.getElementById("epochs");
  M.epochs.forEach(ep=>{
    const b=document.createElement("button");b.className="ep";b.textContent=ep;b.dataset.ep=ep;
    b.onclick=()=>selectEpoch(ep); eb.appendChild(b);
  });
  // legend
  document.getElementById("legend").innerHTML = M.models.map(m=>
    `<span style="margin-right:10px"><span class="swatch" style="background:${COLORS[m]}"></span>${m}</span>`).join("");
  buildHead();
  drawCharts();
  // default: latest epoch that ALL models have; else latest overall
  let def=null;
  for(const ep of M.epochs){ if(M.models.every(m=>M.avail[m][ep]!==undefined)) def=ep; }
  if(def===null && M.epochs.length) def=M.epochs[M.epochs.length-1];
  if(def!==null) selectEpoch(def);
  document.getElementById("playall").onclick=pauseAll;
}

function buildHead(){
  document.getElementById("head").innerHTML =
    `<div class="h" style="color:var(--mut)">sample</div>` +
    M.models.map(m=>`<div class="h" style="color:${COLORS[m]}">${m}<span class="mut" id="cnt_${m}"></span></div>`).join("");
}

function selectEpoch(ep){
  EP=ep;
  document.querySelectorAll(".ep").forEach(b=>b.classList.toggle("on",+b.dataset.ep===ep));
  document.getElementById("curep").textContent=ep;
  const n=M.num_videos, missing=[];
  M.models.forEach(m=>{
    const has=M.avail[m]&&M.avail[m][ep]!==undefined;
    document.getElementById("cnt_"+m).innerHTML = has? ` · ${M.avail[m][ep]}` : ` · —`;
    if(!has) missing.push(m);
  });
  const rows=document.getElementById("rows"); rows.innerHTML="";
  for(let i=0;i<n;i++){
    const row=document.createElement("div"); row.className="row"; row.dataset.i=i;
    let html=`<div class="rowctl"><button class="load">⬇ load</button><span class="s">sample ${i}</span></div>`;
    M.models.forEach(m=>{
      const has=M.avail[m]&&M.avail[m][ep]!==undefined&&i<M.avail[m][ep];
      html += has ? `<div class="slot" data-m="${m}">not loaded</div>` : `<div class="none">—</div>`;
    });
    row.innerHTML=html;
    const btn=row.querySelector("button.load");
    btn.onclick=()=>onRowBtn(row,btn,ep,i);
    rows.appendChild(row);
  }
  document.getElementById("note").textContent = missing.length? ("no videos at this epoch for: "+missing.join(", ")) : "";
  markCharts();
}

function onRowBtn(row,btn,ep,i){
  if(row.dataset.loaded!=="1"){   // first click: load the row's videos, then play synced
    M.models.forEach(m=>{
      const slot=row.querySelector(`.slot[data-m="${m}"]`);
      if(!slot) return;
      const v=document.createElement("video");
      v.muted=true; v.loop=true; v.playsInline=true; v.preload="auto";
      v.src=`video?d=${m}&epoch=${ep}&idx=${i}`;
      v.onclick=()=>toggleRow(row,btn);
      slot.replaceWith(v);
    });
    row.dataset.loaded="1"; row.classList.add("on");
    playRow(row); btn.textContent="⏸ pause";
  } else {
    toggleRow(row,btn);
  }
}
function playRow(row){ row.querySelectorAll("video").forEach(v=>{ try{v.currentTime=0;}catch(e){} v.play().catch(()=>{}); }); }
function toggleRow(row,btn){
  const vids=row.querySelectorAll("video"); if(!vids.length) return;
  const anyPlaying=[...vids].some(v=>!v.paused);
  if(anyPlaying){ vids.forEach(v=>v.pause()); btn.textContent="▶ play"; row.classList.remove("on"); }
  else { playRow(row); btn.textContent="⏸ pause"; row.classList.add("on"); }
}
function pauseAll(){
  document.querySelectorAll(".row").forEach(r=>{
    const vids=r.querySelectorAll("video"); if(!vids.length) return;
    vids.forEach(v=>v.pause());
    const b=r.querySelector("button.load"); if(b && r.dataset.loaded==="1") b.textContent="▶ play";
    r.classList.remove("on");
  });
}

function seriesFor(key){
  return M.models.map(m=>{
    const pts=(M.metrics[m]&&M.metrics[m].points)||[];
    return {m, x:pts.map(p=>p.step), y:pts.map(p=>p[key]), ep:pts.map(p=>p.epoch)};
  });
}
function baseTraces(key){
  return seriesFor(key).map(s=>({x:s.x,y:s.y,mode:"lines+markers",name:s.m,
    line:{color:COLORS[s.m],width:2},marker:{size:4},hovertemplate:s.m+" ep%{customdata} step%{x}<br>%{y:.4f}<extra></extra>",customdata:s.ep}));
}
const LAYOUT=t=>({title:{text:t,font:{size:13,color:"#e6edf3"}},paper_bgcolor:"#161b22",plot_bgcolor:"#161b22",
  font:{color:"#8b949e",size:11},margin:{l:48,r:10,t:30,b:36},showlegend:false,
  xaxis:{title:"global_step",gridcolor:"#30363d",zeroline:false},yaxis:{gridcolor:"#30363d",zeroline:false}});
function drawCharts(){
  Plotly.newPlot("c_loss",baseTraces("val_loss"),LAYOUT("Valid/Loss over steps"),{displayModeBar:false,responsive:true});
  Plotly.newPlot("c_mse",baseTraces("paired_mse"),LAYOUT("paired_mse_avg over steps"),{displayModeBar:false,responsive:true});
}
function markCharts(){
  ["val_loss","paired_mse"].forEach((key,ci)=>{
    const div= ci===0?"c_loss":"c_mse";
    const marks=[];
    seriesFor(key).forEach(s=>{
      const j=s.ep.indexOf(EP);
      if(j>=0) marks.push({x:[s.x[j]],y:[s.y[j]],mode:"markers",name:s.m+"*",
        marker:{color:COLORS[s.m],size:14,line:{color:"#fff",width:2},symbol:"circle"},
        hovertemplate:s.m+" ep"+EP+"<br>%{y:.4f}<extra></extra>",showlegend:false});
    });
    Plotly.react(div, baseTraces(key).concat(marks), LAYOUT(ci===0?"Valid/Loss over steps":"paired_mse_avg over steps"),{displayModeBar:false,responsive:true});
  });
}
boot();
</script>
</body></html>"""
