"""Modal web viewer for the datadiv_ft_fold_ps finetune runs (D1-D5).

5 columns = data-diversity finetune models D1..D5, 6 rows = the 6 validation
videos, with epoch tabs to switch which epoch's val videos are shown. Each row has
a load-then-play control that fetches its 5 clips (one per model) and plays them in
sync so the same validation sample lines up across D1..D5.

Videos are served straight from the egoverse-training-outputs volume:
  datadiv_ft_fold_ps/D{i}/videos/epoch_{N}/EVA_BIMANUAL/validation_video_{j}.mp4
The epoch list is rescanned per page load, so new epochs appear as finetuning runs.

Deploy:  MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/datadiv_ft_val_viz.py
"""
import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]==0.115.*", "uvicorn")
)
app = modal.App("datadiv-ft-val-viz", image=image)
outputs_vol = modal.Volume.from_name("egoverse-training-outputs")

BASE = "/vol/datadiv_ft_fold_ps"
MODELS = ["D1", "D2", "D3", "D4", "D5"]
EMB = "EVA_BIMANUAL"


@app.function(
    volumes={"/vol": outputs_vol},
    min_containers=1,
    scaledown_window=1200,
    timeout=3600,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import re
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    api = FastAPI()

    def scan():
        try:
            outputs_vol.reload()
        except Exception:
            pass
        avail = {}
        for m in MODELS:
            vdir = f"{BASE}/{m}/videos"
            eps = {}
            if os.path.isdir(vdir):
                for d in os.listdir(vdir):
                    mo = re.fullmatch(r"epoch_(\d+)", d)
                    if not mo:
                        continue
                    edir = f"{vdir}/{d}/{EMB}"
                    if os.path.isdir(edir):
                        n = len([f for f in os.listdir(edir) if f.endswith(".mp4")])
                        if n:
                            eps[int(mo.group(1))] = n
            avail[m] = eps
        return avail

    @api.get("/api/manifest")
    def manifest():
        avail = scan()
        epochs = sorted({e for m in MODELS for e in avail[m]})
        nv = max([0] + [n for m in MODELS for n in avail[m].values()])
        return JSONResponse({
            "models": MODELS,
            "epochs": epochs,
            "avail": {m: {str(k): v for k, v in avail[m].items()} for m in MODELS},
            "num_videos": nv,
        })

    @api.get("/video")
    def video(d: str, epoch: int, idx: int):
        if d not in MODELS:
            raise HTTPException(404, "bad model")
        path = f"{BASE}/{d}/videos/epoch_{epoch}/{EMB}/validation_video_{idx}.mp4"
        if not os.path.isfile(path):
            raise HTTPException(404, "no video")
        return FileResponse(path, media_type="video/mp4")

    @api.get("/")
    def index():
        return HTMLResponse(PAGE)

    return api


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>datadiv_ft_fold_ps — Val Videos (D1–D5)</title>
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
  #bar{display:flex;gap:14px;align-items:center;padding:6px 16px;color:var(--mut);flex-wrap:wrap}
  #bar b{color:var(--fg)}
  button.ctl{background:var(--panel);border:1px solid var(--bd);color:var(--fg);padding:4px 10px;border-radius:6px;cursor:pointer}
  #head{position:sticky;top:0;z-index:5;background:var(--bg);display:grid;grid-template-columns:120px repeat(5,1fr);gap:10px;padding:8px 16px;border-bottom:1px solid var(--bd);font-weight:600;text-align:center;color:var(--acc)}
  #head .mut{color:var(--mut);font-weight:400}
  #rows{padding:4px 16px 24px}
  .row{display:grid;grid-template-columns:120px repeat(5,1fr);gap:10px;align-items:center;padding:8px 0;border-bottom:1px solid #21262d}
  .row.on{background:#12233b}
  .rowctl{display:flex;flex-direction:column;gap:8px;align-items:center;justify-content:center}
  .rowctl .s{color:var(--fg);font-weight:600;font-size:12px}
  .rowctl button{width:46px;height:46px;font-size:17px;border-radius:50%;background:var(--panel);border:1px solid var(--bd);color:var(--fg);cursor:pointer}
  .rowctl button:hover{border-color:var(--acc);color:var(--acc)}
  .row video{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:6px;display:block;cursor:pointer}
  .row .slot{aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:11px;background:#0b0f14;border:1px solid var(--bd);border-radius:6px}
  .row .none{aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:11px;border:1px dashed var(--bd);border-radius:6px}
</style></head>
<body>
<header>
  <h1>datadiv_ft_fold_ps · validation videos across D1–D5 (finetune of data_div D1–D5 @ ep1379)</h1>
  <div id="epochs"></div>
</header>
<div id="bar">
  <span>epoch: <b id="curep">–</b></span>
  <button class="ctl" id="pauseall">⏸ pause all</button>
  <span>click a row's <b style="color:var(--fg)">⬇ load</b> to fetch its 5 clips (D1–D5), then play them together.</span>
  <span id="note" style="color:#d29922"></span>
</div>
<div id="head"></div>
<div id="rows"></div>
<script>
let M=null, EP=null;
async function boot(){
  M=await (await fetch("api/manifest")).json();
  const eb=document.getElementById("epochs");
  M.epochs.forEach(ep=>{const b=document.createElement("button");b.className="ep";b.textContent="epoch "+ep;b.dataset.ep=ep;b.onclick=()=>selectEpoch(ep);eb.appendChild(b);});
  document.getElementById("head").innerHTML=`<div class="mut">sample</div>`+M.models.map(m=>`<div>${m}</div>`).join("");
  document.getElementById("pauseall").onclick=pauseAll;
  if(M.epochs.length) selectEpoch(M.epochs[M.epochs.length-1]);
  else document.getElementById("note").textContent="no epochs found yet";
}
function selectEpoch(ep){
  EP=ep;
  document.querySelectorAll(".ep").forEach(b=>b.classList.toggle("on",+b.dataset.ep===ep));
  document.getElementById("curep").textContent=ep;
  const n=M.num_videos, missing=[];
  M.models.forEach(m=>{ if(!(M.avail[m]&&M.avail[m][ep]!==undefined)) missing.push(m); });
  const rows=document.getElementById("rows"); rows.innerHTML="";
  for(let i=0;i<n;i++){
    const row=document.createElement("div"); row.className="row"; row.dataset.i=i;
    let html=`<div class="rowctl"><button class="load">⬇ load</button><span class="s">sample ${i}</span></div>`;
    M.models.forEach(m=>{
      const has=M.avail[m]&&M.avail[m][ep]!==undefined&&i<M.avail[m][ep];
      html += has ? `<div class="cell" data-m="${m}"><div class="slot">not loaded</div></div>` : `<div class="cell"><div class="none">—</div></div>`;
    });
    row.innerHTML=html;
    row.querySelector("button.load").onclick=(e)=>onRowBtn(row,e.target,ep,i);
    rows.appendChild(row);
  }
  document.getElementById("note").textContent = missing.length? ("no videos at this epoch for: "+missing.join(", ")) : "";
}
function onRowBtn(row,btn,ep,i){
  if(row.dataset.loaded!=="1"){
    row.querySelectorAll(".cell").forEach(cell=>{
      const m=cell.dataset.m; if(!m) return;
      const v=document.createElement("video");
      v.muted=true;v.loop=true;v.playsInline=true;v.preload="auto";
      v.src=`video?d=${m}&epoch=${ep}&idx=${i}`; v.onclick=()=>toggleRow(row,btn);
      const slot=cell.querySelector(".slot"); if(slot) slot.replaceWith(v);
    });
    row.dataset.loaded="1"; row.classList.add("on");
    playRow(row); btn.textContent="⏸ pause";
  } else { toggleRow(row,btn); }
}
function playRow(row){ row.querySelectorAll("video").forEach(v=>{try{v.currentTime=0;}catch(e){} v.play().catch(()=>{});}); }
function toggleRow(row,btn){
  const vids=row.querySelectorAll("video"); if(!vids.length) return;
  const any=[...vids].some(v=>!v.paused);
  if(any){vids.forEach(v=>v.pause()); btn.textContent="▶ play"; row.classList.remove("on");}
  else {playRow(row); btn.textContent="⏸ pause"; row.classList.add("on");}
}
function pauseAll(){
  document.querySelectorAll(".row").forEach(r=>{const vids=r.querySelectorAll("video"); if(!vids.length) return;
    vids.forEach(v=>v.pause()); const b=r.querySelector("button.load"); if(b&&r.dataset.loaded==="1")b.textContent="▶ play"; r.classList.remove("on");});
}
boot();
</script>
</body></html>"""
