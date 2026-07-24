"""Modal web viewer: 3 example episodes per repetitiveness level per task.

Layout: columns = repetitiveness level (high / medium / low); the grid is chunked
by task, and each task chunk has 3 rows (3 example episodes). Each row has a
load-then-play control that fetches its 3 clips (one per level) and plays them in
sync so you can eyeball how repetitiveness differs across the levels.

Videos are rendered on demand from each episode's images.front_1 in its zarr on the
mecka_data_v2 volume, cached as MP4 on a persistent cache volume.

Deploy:    MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/data_rep_examples_viz.py
Pre-warm:  MODAL_ENVIRONMENT=robotics modal run egomimic/modal/data_rep_examples_viz.py
"""
import os
import subprocess
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi[standard]==0.115.*", "uvicorn", "zarr>=3.0", "numcodecs", "simplejpeg", "numpy")
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "data_rep_examples.json"),
        "/root/examples.json",
    )
)
app = modal.App("data-rep-examples-viz", image=image)
zarr_vol = modal.Volume.from_name("mecka_data_v2")
cache_vol = modal.Volume.from_name("data-rep-preview-cache", create_if_missing=True)
ZARR = "/zarr"
CACHE = "/cache"
IMAGE_KEY = "images.front_1"
FPS = 30


def _render(episode_hash: str) -> "str | None":
    """Render <hash>.mp4 into the cache volume from its zarr images.front_1.
    Returns the cache path, or None if the source is unavailable. Idempotent."""
    import numpy as np
    import simplejpeg
    import zarr

    out_path = f"{CACHE}/{episode_hash}.mp4"
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    # Another container may have populated the cache after we mounted; refresh
    # our view before deciding to (expensively) re-render.
    try:
        cache_vol.reload()
    except Exception:
        pass
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        return out_path

    store_path = None
    for cand in (f"{episode_hash}.zarr", episode_hash):
        if os.path.isdir(f"{ZARR}/{cand}"):
            store_path = f"{ZARR}/{cand}"
            break
    if store_path is None:
        return None
    store = zarr.open_group(store_path, mode="r")
    if IMAGE_KEY not in store:
        return None
    jpegs = store[IMAGE_KEY][:]
    if len(jpegs) == 0:
        return None

    def to_bytes(raw):
        if isinstance(raw, np.void):
            raw = raw.item()
        if isinstance(raw, np.ndarray):
            raw = raw.item() if raw.ndim == 0 else bytes(raw)
        return raw if isinstance(raw, bytes) else bytes(raw)

    first = simplejpeg.decode_jpeg(to_bytes(jpegs[0]), colorspace="RGB")
    h, w = first.shape[:2]
    outW, outH = (w // 2) - (w // 2) % 2, (h // 2) - (h // 2) % 2
    tmp = f"/tmp/{episode_hash}.mp4"
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-an",
        "-vf", f"scale={outW}:{outH}", "-c:v", "libx264", "-crf", "24",
        "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for jb in jpegs:
            frame = simplejpeg.decode_jpeg(to_bytes(jb), colorspace="RGB")
            if frame.shape[:2] != (h, w):
                continue
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        proc.stderr.read()
        if proc.wait() != 0:
            return None
    except BrokenPipeError:
        return None
    finally:
        if proc.poll() is None:
            proc.kill()
    Path(out_path).write_bytes(Path(tmp).read_bytes())
    Path(tmp).unlink(missing_ok=True)
    cache_vol.commit()
    return out_path


@app.function(volumes={ZARR: zarr_vol, CACHE: cache_vol}, cpu=8, timeout=1800)
def render_one(episode_hash: str) -> str:
    r = _render(episode_hash)
    return f"{episode_hash}: {'ok' if r else 'FAIL'}"


@app.function(
    volumes={ZARR: zarr_vol, CACHE: cache_vol},
    cpu=4,
    timeout=1800,
    min_containers=1,
    scaledown_window=1200,
)
@modal.concurrent(max_inputs=12)
@modal.asgi_app()
def web():
    import json
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    api = FastAPI()
    manifest = json.load(open("/root/examples.json"))
    all_hashes = {h for t in manifest["examples"].values() for lst in t.values() for h in lst}

    @api.get("/api/manifest")
    def m():
        return JSONResponse(manifest)

    @api.get("/video")
    def video(ep: str):
        if ep not in all_hashes:
            raise HTTPException(404, "unknown episode")
        path = _render(ep)
        if not path:
            raise HTTPException(404, "render failed / no source")
        return FileResponse(path, media_type="video/mp4")

    @api.get("/")
    def index():
        return HTMLResponse(PAGE)

    return api


@app.local_entrypoint()
def main():
    """Pre-warm the cache: render all example clips so the UI is instant."""
    import json
    manifest = json.load(open(os.path.join(os.path.dirname(__file__), "data_rep_examples.json")))
    hashes = sorted({h for t in manifest["examples"].values() for lst in t.values() for h in lst})
    print(f"pre-rendering {len(hashes)} example clips...")
    ok = 0
    for r in render_one.map(hashes):
        ok += 1 if r.endswith("ok") else 0
    print(f"done: {ok}/{len(hashes)} rendered ok")


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Repetitiveness Examples — high / medium / low</title>
<style>
  :root{--bg:#0e1117;--panel:#161b22;--fg:#e6edf3;--mut:#8b949e;--bd:#30363d;--hi:#ff7b72;--me:#d29922;--lo:#3fb950;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif}
  header{padding:12px 16px;border-bottom:1px solid var(--bd)}
  h1{font-size:16px;margin:0}
  #bar{padding:6px 16px;color:var(--mut);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
  button.ctl{background:var(--panel);border:1px solid var(--bd);color:var(--fg);padding:4px 10px;border-radius:6px;cursor:pointer}
  #head{position:sticky;top:0;z-index:5;background:var(--bg);display:grid;grid-template-columns:130px repeat(3,1fr);gap:10px;padding:8px 16px;border-bottom:1px solid var(--bd);font-weight:600;text-align:center}
  #head .hi{color:var(--hi)} #head .me{color:var(--me)} #head .lo{color:var(--lo)}
  .task{padding:4px 16px 16px}
  .taskh{margin:14px 0 6px;font-weight:700;font-size:15px;border-left:3px solid var(--fg);padding-left:8px}
  .row{display:grid;grid-template-columns:130px repeat(3,1fr);gap:10px;align-items:center;padding:6px 0;border-bottom:1px solid #21262d}
  .row.on{background:#12233b}
  .rowctl{display:flex;flex-direction:column;gap:8px;align-items:center;justify-content:center}
  .rowctl .s{color:var(--mut);font-size:11px}
  .rowctl button{width:46px;height:46px;font-size:17px;border-radius:50%;background:var(--panel);border:1px solid var(--bd);color:var(--fg);cursor:pointer}
  .rowctl button:hover{border-color:#58a6ff;color:#58a6ff}
  .cell{position:relative}
  .cell video{width:100%;aspect-ratio:16/9;object-fit:contain;background:#000;border-radius:6px;display:block;cursor:pointer}
  .cell .slot{aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:11px;background:#0b0f14;border:1px solid var(--bd);border-radius:6px}
  .cell .none{aspect-ratio:16/9;display:flex;align-items:center;justify-content:center;color:var(--mut);font-size:11px;border:1px dashed var(--bd);border-radius:6px}
  .cell .sc{position:absolute;top:4px;left:4px;background:rgba(0,0,0,.6);padding:1px 6px;border-radius:4px;font-size:11px;font-variant-numeric:tabular-nums}
</style></head>
<body>
<header><h1>Repetitiveness examples · columns = high / medium / low · grouped by task · 3 examples each</h1></header>
<div id="bar">
  <button class="ctl" id="pauseall">⏸ pause all</button>
  <span>click a row's <b style="color:var(--fg)">⬇ load</b> to fetch its 3 clips (high/med/low), then play them together. score shown top-left of each clip.</span>
</div>
<div id="head"><div>example</div><div class="hi">HIGH</div><div class="me">MEDIUM</div><div class="lo">LOW</div></div>
<div id="tasks"></div>
<script>
let M=null;
async function boot(){
  M=await (await fetch("api/manifest")).json();
  const root=document.getElementById("tasks"); root.innerHTML="";
  M.tasks.forEach(t=>{
    const sec=document.createElement("div"); sec.className="task";
    sec.innerHTML=`<div class="taskh">${t}</div>`;
    for(let j=0;j<M.n;j++){
      const row=document.createElement("div"); row.className="row"; row.dataset.t=t; row.dataset.j=j;
      let html=`<div class="rowctl"><button class="load">⬇ load</button><span class="s">ex ${j+1}</span></div>`;
      ["high","medium","low"].forEach(lvl=>{
        const arr=M.examples[t][lvl]||[]; const sc=(M.scores[t][lvl]||[])[j];
        if(arr[j]) html+=`<div class="cell" data-ep="${arr[j]}" data-sc="${sc!==undefined?sc:''}"><div class="slot">not loaded</div></div>`;
        else html+=`<div class="cell"><div class="none">—</div></div>`;
      });
      row.innerHTML=html;
      row.querySelector("button.load").onclick=(e)=>onRowBtn(row,e.target);
      sec.appendChild(row);
    }
    root.appendChild(sec);
  });
}
function onRowBtn(row,btn){
  if(row.dataset.loaded!=="1"){
    row.querySelectorAll(".cell").forEach(cell=>{
      const ep=cell.dataset.ep; if(!ep) return;
      const sc=cell.dataset.sc;
      const v=document.createElement("video");
      v.muted=true;v.loop=true;v.playsInline=true;v.preload="auto";
      v.src=`video?ep=${ep}`; v.onclick=()=>toggleRow(row,btn);
      const slot=cell.querySelector(".slot"); if(slot) slot.replaceWith(v);
      if(sc!==""&&sc!==undefined){const tag=document.createElement("div");tag.className="sc";tag.textContent=sc;cell.appendChild(tag);}
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
document.getElementById("pauseall").onclick=()=>{
  document.querySelectorAll(".row").forEach(r=>{
    const vids=r.querySelectorAll("video"); if(!vids.length) return;
    vids.forEach(v=>v.pause()); const b=r.querySelector("button.load");
    if(b&&r.dataset.loaded==="1")b.textContent="▶ play"; r.classList.remove("on");
  });
};
boot();
</script>
</body></html>"""
