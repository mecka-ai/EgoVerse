"""Web viewer for validation videos of the 8 data_div_oss training runs.

Serves a single-page viewer that shows, for a chosen epoch, the GT-vs-pred
validation videos of all 8 runs side-by-side. Videos are read directly
(read-only) from the `egoverse-training-outputs` volume, where trainers write
them to  <run_dir>/videos/epoch_<N>/<EMBODIMENT>/validation_video_<i>.mp4
(see egomimic/eval/eval_video.py).

Deploy:
    MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/val_video_viewer.py

Routes (single ASGI web function → one URL):
    /                HTML viewer page
    /api/index       {"runs": [...], "epochs": [...]} — volume scan, cached 60 s
    /api/annotations per-run, per-video language-annotation segments in video time
    /video?path=     streams one mp4 from the volume (path relative to data_div_oss/)

Annotation sync — how video frames map to episode frames
--------------------------------------------------------
egomimic/eval/eval_video.py buffers ONE image per val sample (viz_gt_preds stacks
(B,H,W,3) per batch) and flushes the buffer to validation_video_<i>.mp4 whenever
it holds >= 1000 frames; with full batches the flush lands at 1024 frames exactly
(first multiple of the batch size >= 1000; batch 64 -> 16 steps, batch 32 -> 32
steps). The val loader is sequential (shuffle=False), the val MultiDataset is the
5 val episodes concatenated in sorted-episode-hash order (episode_table_to_df
orders by episode_hash; == the json order), one sample per raw frame (no pause
filter: pause_removal_epsilon unset in all 8 run configs), and
limit_val_batches=100 caps the pass.

  OSS runs (HPT; 1 GPU, val batch 64):  video i, frame k -> dataset sample
      s = 1024*i + k, for s < 100*64 = 6400.   (verified: videos are 6x1024+256)
  pi runs  (PI; 2 GPUs, val batch 32):  DistributedSampler(shuffle=False) gives
      rank r samples [r, r+2, ...]; both ranks write the SAME video paths
      (no rank guard in eval_video.py) so the surviving file is one rank's,
      last-writer-wins. video i, frame k -> s = 2*(1024*i + k) + r, r unknown.
      We assume r=0 -> at most 1 data frame (1/30 s) of error, but the rank
      ambiguity makes pi-run sync approximate ("~" in the UI).
      (verified: videos are 3x1024+128 = 100*32 per rank)

  sample s -> episode via cumulative total_frames [1945,2099,2155,2245,2173]
  (sum 10617), frame f = s - cum_start; annotation = segments with
  start_idx <= f < end_idx from the episode zarr's `annotations` array.
  Both data and videos are 30 fps, so video time t -> k = floor(30*t).

Known imprecision (why the bar is "~" even on OSS runs): samples that fail the
norm-stats bounds check or fail to load are silently replaced by a RANDOM other
sample of the same episode set (MultiDataset fallback), so occasional frames show
content the mapping cannot know about.
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]", "zarr>=3", "numpy"
)

app = modal.App("val-video-viewer", image=image)

outputs_volume = modal.Volume.from_name("egoverse-training-outputs")
episodes_volume = modal.Volume.from_name("mecka_data_v2")

MOUNT_PATH = "/data"
EPISODES_MOUNT = "/episodes"
BASE_PREFIX = "data_div_oss"
INDEX_TTL_S = 60

# (short label, run dir under data_div_oss/, world_size, val batch size per rank)
# world/batch read from each run's .hydra/config.yaml on the outputs volume
# (launch_params.gpus_per_node, valid_dataloader_params.batch_size).
RUNS = [
    ("300M", "300M_mm_nobc_dw48", 1, 64),
    ("600M", "600M_mm_nobc_dw48", 1, 64),
    ("1B", "1B_mm_nobc_dw48", 1, 64),
    ("1.5B", "1_5B_mm_nobc_dw48", 1, 64),
    ("pi05", "pi05_dw48", 2, 32),
    ("pali", "pali_dw48", 2, 32),
    ("pi05_lang", "pi05_lang_dw48", 2, 32),
    ("pali_lang", "pali_lang_dw48", 2, 32),
]

# Val episodes in resolver order = sorted episode_hash (== the json order), from
# egomimic/hydra_configs/data/extra/data_diversity/dishwashing_val_ophold5.json
VAL_EPISODES = [
    "69b22fc5f4f4e149281a6635",
    "69b3304abf7ebb83d8870280",
    "69b34bd305ad590410f5939f",
    "69b3c40fe1d17e38b58df9bd",
    "69b3d54ab3480508b4a67111",
]

FPS = 30
LIMIT_VAL_BATCHES = 100  # trainer.limit_val_batches in all 8 run configs
BUFFER_FLUSH_AT = 1000  # eval_video.py flushes the frame buffer at >= 1000

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>data_div_oss val videos</title>
<style>
  :root {
    --bg: #101216; --panel: #181b21; --border: #2a2e37;
    --text: #e6e8ec; --dim: #8b919d; --accent: #6ea8fe;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  header {
    position: sticky; top: 0; z-index: 10; background: var(--bg);
    border-bottom: 1px solid var(--border);
    padding: 10px 16px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }
  header h1 { font-size: 15px; margin: 0 8px 0 0; font-weight: 600; }
  select, button {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 10px; font-size: 13px; cursor: pointer;
  }
  select:hover, button:hover { border-color: var(--accent); }
  button.on { border-color: var(--accent); color: var(--accent); }
  #colctl { display: flex; align-items: center; gap: 6px; }
  #colind { color: var(--dim); font-size: 12px; min-width: 96px; text-align: center; }
  .spacer { flex: 1; }
  #status { color: var(--dim); font-size: 12px; }
  main { padding: 12px 16px 40px; }
  .run-row {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    margin-bottom: 12px; padding: 10px 12px;
  }
  .run-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px; }
  .run-label { font-size: 15px; font-weight: 600; }
  .run-dir { color: var(--dim); font-size: 12px; }
  .run-note { color: var(--dim); font-size: 12px; margin-left: auto; }
  .vids { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 4px; }
  .vid-card { flex: 0 0 auto; width: 380px; }
  .vid-card video { width: 100%; border-radius: 6px; background: #000; display: block; }
  .vid-card.active-col video { outline: 2px solid var(--accent); }
  .col-ph { flex: 0 0 auto; width: 380px; }
  .vid-cap { color: var(--dim); font-size: 11px; margin-top: 3px; text-align: center; }
  .ann-bar {
    color: #ccd3de; background: #10131a; border: 1px solid var(--border);
    border-radius: 4px; font-size: 11px; padding: 3px 6px; margin-top: 5px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    text-align: center; min-height: 20px;
  }
  .placeholder {
    color: var(--dim); border: 1px dashed var(--border); border-radius: 6px;
    padding: 28px 16px; text-align: center; font-size: 13px; width: 100%;
  }
</style>
</head>
<body>
<header>
  <h1>data_div_oss — validation videos</h1>
  <button id="prev" title="previous epoch">&#9664;</button>
  <select id="epoch"></select>
  <button id="next" title="next epoch">&#9654;</button>
  <button id="playall">Play all</button>
  <button id="pauseall">Pause all</button>
  <button id="colmode" title="load and play one video column at a time">Column mode</button>
  <span id="colctl" style="display:none">
    <button id="colprev" title="previous column">&#9664; col</button>
    <span id="colind">Column 1 of 1</span>
    <button id="colnext" title="next column">col &#9654;</button>
  </span>
  <span class="spacer"></span>
  <span id="status">loading…</span>
</header>
<main id="grid"></main>
<script>
(function () {
  let index = null;          // {runs:[{label,dir,epochs:{ep:[paths]}}], epochs:[..]}
  let annIndex = null;       // {runDir: {approx, videos: [[{s,e,t},..] per video idx]}}
  let currentEpoch = null;
  let columnMode = false;    // column mode: load/play one video column at a time
  let currentCol = 0;        // 0-based active column index
  let playToken = 0;         // guards synced play against rapid column switches

  const $ = (id) => document.getElementById(id);
  const grid = $("grid"), epochSel = $("epoch"), statusEl = $("status");

  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        const v = e.target;
        if (!v.src && v.dataset.src) { v.src = v.dataset.src; v.preload = "metadata"; }
        observer.unobserve(v);
      }
    }
  }, { rootMargin: "200px" });

  function coverage(ep) {
    return index.runs.filter((r) => r.epochs[ep] && r.epochs[ep].length).length;
  }

  // Language-annotation segments for one video, looked up by the numeric index
  // in its filename (validation_video_<i>.mp4) — robust to gaps in the listing.
  function annFor(runDir, path) {
    if (!annIndex || annIndex.error) return null;
    const a = annIndex[runDir];
    const m = path.match(/validation_video_(\d+)\.mp4$/);
    if (!a || !m) return null;
    return { segs: a.videos[Number(m[1])] || [], approx: a.approx };
  }

  function buildSelector() {
    epochSel.innerHTML = "";
    for (const ep of index.epochs) {
      const o = document.createElement("option");
      o.value = ep;
      o.textContent = "epoch " + ep + "  (" + coverage(ep) + "/" + index.runs.length + " runs)";
      epochSel.appendChild(o);
    }
  }

  function defaultEpoch() {
    // latest epoch every started run has → most comparable snapshot
    const maxes = index.runs
      .map((r) => Object.keys(r.epochs).map(Number))
      .filter((a) => a.length)
      .map((a) => Math.max.apply(null, a));
    if (!maxes.length) return null;
    const common = Math.min.apply(null, maxes);
    return index.epochs.includes(common) ? common : index.epochs[index.epochs.length - 1];
  }

  function render() {
    grid.innerHTML = "";
    for (const run of index.runs) {
      const row = document.createElement("div");
      row.className = "run-row";
      const head = document.createElement("div");
      head.className = "run-head";
      head.innerHTML =
        '<span class="run-label">' + run.label + "</span>" +
        '<span class="run-dir">' + run.dir + "</span>";
      row.appendChild(head);

      const vids = run.epochs[currentEpoch] || [];
      const body = document.createElement("div");
      body.className = "vids";
      if (!vids.length) {
        const eps = Object.keys(run.epochs).map(Number);
        const latest = eps.length ? Math.max.apply(null, eps) : null;
        const ph = document.createElement("div");
        ph.className = "placeholder";
        ph.textContent = latest === null
          ? "no videos yet for this run"
          : "no video yet for epoch " + currentEpoch + " (latest: epoch " + latest + ")";
        body.appendChild(ph);
      } else {
        vids.forEach((p, i) => {
          const card = document.createElement("div");
          card.className = "vid-card";
          card.dataset.col = i;
          const v = document.createElement("video");
          v.controls = true; v.muted = true; v.loop = true;
          v.playsInline = true; v.preload = "none";
          v.dataset.src = "video?path=" + encodeURIComponent(p);
          v.title = p;
          if (!columnMode) observer.observe(v);
          card.appendChild(v);
          const bar = document.createElement("div");
          bar.className = "ann-bar";
          bar.textContent = "—";
          const updateBar = () => {
            const ann = annFor(run.dir, p);
            if (!ann) { bar.textContent = "—"; return; }
            const t = v.currentTime;
            const seg = ann.segs.find((s) => s.s <= t && t < s.e);
            bar.textContent = (ann.approx ? "≈ " : "") + (seg ? seg.t : "—");
            bar.title = seg ? seg.t : "";
          };
          v.addEventListener("timeupdate", updateBar);
          v.addEventListener("seeked", updateBar);
          v.addEventListener("loadeddata", updateBar);
          card.appendChild(bar);
          const cap = document.createElement("div");
          cap.className = "vid-cap";
          cap.textContent = p.split("/").slice(-2).join(" / ");
          card.appendChild(cap);
          body.appendChild(card);
        });
        const n = vids.length;
        head.insertAdjacentHTML(
          "beforeend",
          '<span class="run-note">' + n + " video" + (n === 1 ? "" : "s") + "</span>"
        );
      }
      row.appendChild(body);
      grid.appendChild(row);
    }
  }

  function setEpoch(ep) {
    currentEpoch = ep;
    epochSel.value = ep;
    render();
    if (columnMode) { currentCol = 0; applyColumn(); }
  }

  // ---- column mode ----
  const colmodeBtn = $("colmode"), colctl = $("colctl"), colind = $("colind");

  function maxCols() {
    return Math.max(0, ...index.runs.map((r) => (r.epochs[currentEpoch] || []).length));
  }

  function playSynced(videos) {
    const token = ++playToken;
    const ready = videos.map((v) => new Promise((res) => {
      if (v.readyState >= 2) return res();
      const on = () => { v.removeEventListener("loadeddata", on); res(); };
      v.addEventListener("loadeddata", on);
      setTimeout(res, 5000); // don't stall the column on one slow video
    }));
    Promise.all(ready).then(() => {
      if (token !== playToken) return; // column changed meanwhile
      videos.forEach((v) => {
        try { v.currentTime = 0; } catch (e) {}
        v.play().catch(() => {});
      });
    });
  }

  function applyColumn() {
    document.querySelectorAll(".col-ph").forEach((e) => e.remove());
    const active = [];
    grid.querySelectorAll(".run-row").forEach((row) => {
      const body = row.querySelector(".vids");
      const cards = body.querySelectorAll(".vid-card");
      let found = null;
      cards.forEach((card) => {
        const v = card.querySelector("video");
        if (Number(card.dataset.col) === currentCol) {
          found = card;
          card.classList.add("active-col");
          if (!v.getAttribute("src") && v.dataset.src) {
            v.src = v.dataset.src; v.preload = "auto";
          }
          active.push(v);
        } else {
          card.classList.remove("active-col");
          v.pause();
          if (v.getAttribute("src")) { v.removeAttribute("src"); v.load(); } // unload buffer
        }
      });
      if (found) {
        found.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
      } else if (cards.length) {
        // run has videos this epoch, just fewer than currentCol+1
        const ph = document.createElement("div");
        ph.className = "placeholder col-ph";
        ph.textContent =
          "no video #" + (currentCol + 1) + " for this run at epoch " + currentEpoch;
        body.appendChild(ph);
      }
    });
    colind.textContent = "Column " + (currentCol + 1) + " of " + maxCols();
    playSynced(active);
  }

  function colStep(d) {
    const n = maxCols();
    if (!n) return;
    const next = Math.min(n - 1, Math.max(0, currentCol + d));
    if (next === currentCol) return;
    currentCol = next;
    applyColumn();
  }

  function setColumnMode(on) {
    columnMode = on;
    colmodeBtn.classList.toggle("on", on);
    colctl.style.display = on ? "" : "none";
    if (on) {
      observer.disconnect(); // column mode manages loading itself
      currentCol = 0;
      applyColumn();
    } else {
      playToken++; // cancel any pending synced play
      render();    // rebuild grid: clears highlights/placeholders, re-arms lazy-load
    }
  }

  colmodeBtn.onclick = () => { if (index) setColumnMode(!columnMode); };
  $("colprev").onclick = () => colStep(-1);
  $("colnext").onclick = () => colStep(1);
  // ---- end column mode ----

  function step(d) {
    const i = index.epochs.indexOf(currentEpoch) + d;
    if (i >= 0 && i < index.epochs.length) setEpoch(index.epochs[i]);
  }

  $("prev").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  epochSel.onchange = () => setEpoch(Number(epochSel.value));
  $("playall").onclick = () => {
    if (columnMode) { applyColumn(); return; } // reload + re-sync the active column only
    document.querySelectorAll("video").forEach((v) => {
      if (!v.src && v.dataset.src) v.src = v.dataset.src;
      v.play().catch(() => {});
    });
  };
  $("pauseall").onclick = () => {
    playToken++; // cancel any pending synced column play
    document.querySelectorAll("video").forEach((v) => v.pause());
  };
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") columnMode ? colStep(-1) : step(-1);
    if (e.key === "ArrowRight") columnMode ? colStep(1) : step(1);
  });

  fetch("api/annotations")
    .then((r) => r.json())
    .then((data) => {
      annIndex = data;
      // Refresh bars of already-rendered videos (listeners are per-video closures).
      document.querySelectorAll("video").forEach((v) =>
        v.dispatchEvent(new Event("loadeddata"))
      );
    })
    .catch(() => { annIndex = null; });

  fetch("api/index")
    .then((r) => r.json())
    .then((data) => {
      index = data;
      if (!index.epochs.length) {
        statusEl.textContent = "no validation videos found on the volume yet";
        return;
      }
      buildSelector();
      const nvids = index.runs.reduce(
        (s, r) => s + Object.values(r.epochs).reduce((t, v) => t + v.length, 0), 0);
      statusEl.textContent =
        index.epochs.length + " epochs · " + nvids + " videos · index refreshes every 60 s";
      setEpoch(defaultEpoch());
    })
    .catch((e) => { statusEl.textContent = "failed to load index: " + e; });
})();
</script>
</body>
</html>
"""


@app.function(
    volumes={
        MOUNT_PATH: outputs_volume.read_only(),
        EPISODES_MOUNT: episodes_volume.read_only(),
    },
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label="val-video-viewer")
def viewer():
    import bisect
    import json as jsonlib
    import math
    import os
    import re
    import threading
    import time

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, HTMLResponse

    api = FastAPI()
    base_dir = os.path.realpath(os.path.join(MOUNT_PATH, BASE_PREFIX))

    _cache = {"ts": 0.0, "data": None}
    _lock = threading.Lock()
    _ann_cache = {"data": None}  # annotations are static — computed once
    _ann_lock = threading.Lock()

    def _vid_sort_key(name):
        m = re.search(r"(\d+)\.mp4$", name)
        return (int(m.group(1)) if m else 1 << 30, name)

    def _scan():
        runs_out = []
        all_epochs = set()
        for label, run_dir, _world, _bs in RUNS:
            epochs = {}
            videos_dir = os.path.join(base_dir, run_dir, "videos")
            if os.path.isdir(videos_dir):
                for ep_name in os.listdir(videos_dir):
                    if not ep_name.startswith("epoch_"):
                        continue
                    try:
                        ep = int(ep_name.split("_", 1)[1])
                    except ValueError:
                        continue
                    ep_dir = os.path.join(videos_dir, ep_name)
                    vids = []
                    try:
                        embodiments = sorted(os.listdir(ep_dir))
                    except NotADirectoryError:
                        continue
                    for emb in embodiments:
                        emb_dir = os.path.join(ep_dir, emb)
                        if not os.path.isdir(emb_dir):
                            continue
                        for f in sorted(os.listdir(emb_dir), key=_vid_sort_key):
                            if f.endswith(".mp4"):
                                vids.append(f"{run_dir}/videos/{ep_name}/{emb}/{f}")
                    if vids:
                        epochs[ep] = vids
                        all_epochs.add(ep)
            runs_out.append({"label": label, "dir": run_dir, "epochs": epochs})
        return {"runs": runs_out, "epochs": sorted(all_epochs)}

    def _index():
        with _lock:
            now = time.monotonic()
            if _cache["data"] is None or now - _cache["ts"] > INDEX_TTL_S:
                try:
                    outputs_volume.reload()  # pick up freshly committed videos
                except Exception:
                    pass  # serve a possibly-stale view rather than erroring
                _cache["data"] = _scan()
                _cache["ts"] = now
            return _cache["data"]

    @api.get("/")
    def page():
        return HTMLResponse(PAGE_HTML)

    @api.get("/api/index")
    def index():
        return _index()

    # ---- language annotations, synced to video time (see module docstring) ----

    def _load_episodes():
        """Read (total_frames, [(start,end,text), ...]) per val episode, in order."""
        import zarr

        eps = []
        for h in VAL_EPISODES:
            path = next(
                (
                    p
                    for p in (
                        os.path.join(EPISODES_MOUNT, f"{h}.zarr"),
                        os.path.join(EPISODES_MOUNT, h),
                    )
                    if os.path.isdir(p)
                ),
                None,
            )
            if path is None:
                raise FileNotFoundError(f"val episode {h} not found on volume")
            group = zarr.open_group(path, mode="r")
            total_frames = int(dict(group.attrs)["total_frames"])
            segs = []
            for raw in group["annotations"][:]:
                if isinstance(raw, (bytes, bytearray, memoryview)):
                    raw = bytes(raw).decode("utf-8")
                try:
                    d = jsonlib.loads(raw)
                except (TypeError, ValueError):
                    continue
                if isinstance(d, dict) and d.get("text"):
                    segs.append(
                        (int(d.get("start_idx", -1)), int(d.get("end_idx", -1)), str(d["text"]))
                    )
            eps.append((total_frames, segs))
        return eps

    def _build_annotations():
        """{run_dir: {approx, videos: [[{s,e,t}, ...] per video index]}} — static."""
        episodes = _load_episodes()
        cum_starts = []
        total = 0
        for tf, _ in episodes:
            cum_starts.append(total)
            total += tf

        def text_at(sample_idx):
            if sample_idx >= total:
                return ""
            e = bisect.bisect_right(cum_starts, sample_idx) - 1
            f = sample_idx - cum_starts[e]
            active = [t for (s0, s1, t) in episodes[e][1] if s0 <= f < s1]
            return " · ".join(active)

        # eval_video.py flushes at >=1000 buffered frames; batches arrive whole, so
        # with full batches the flush lands at the first multiple of bs >= 1000.
        out = {}
        for _label, run_dir, world, bs in RUNS:
            flush = math.ceil(BUFFER_FLUSH_AT / bs) * bs  # 1024 for bs in {32, 64}
            per_rank_cap = min(LIMIT_VAL_BATCHES * bs, math.ceil(total / world))
            videos = []
            for i in range(math.ceil(per_rank_cap / flush)):
                p0, p1 = flush * i, min(flush * (i + 1), per_rank_cap)
                segs = []
                cur_text, cur_start = None, 0
                for p in range(p0, p1):
                    t = text_at(p * world)  # rank-0 assumption for world > 1
                    k = p - p0
                    if t != cur_text:
                        if cur_text:
                            segs.append(
                                {"s": round(cur_start / FPS, 3), "e": round(k / FPS, 3), "t": cur_text}
                            )
                        cur_text, cur_start = t, k
                if cur_text:
                    segs.append(
                        {
                            "s": round(cur_start / FPS, 3),
                            "e": round((p1 - p0) / FPS, 3),
                            "t": cur_text,
                        }
                    )
                videos.append(segs)
            out[run_dir] = {"approx": world > 1, "videos": videos}
        return out

    @api.get("/api/annotations")
    def annotations():
        with _ann_lock:
            if _ann_cache["data"] is None:
                try:
                    _ann_cache["data"] = _build_annotations()
                except Exception as exc:
                    # Don't cache failures — surface the error and retry next call.
                    return {"error": f"{type(exc).__name__}: {exc}"}
            return _ann_cache["data"]

    @api.get("/video")
    def video(path: str = Query(...)):
        # Resolve strictly under data_div_oss/ on the volume; reject traversal.
        if "\x00" in path or path.startswith(("/", "~")) or ".." in path.split("/"):
            raise HTTPException(status_code=400, detail="bad path")
        full = os.path.realpath(os.path.join(base_dir, path))
        if not full.startswith(base_dir + os.sep) or not full.endswith(".mp4"):
            raise HTTPException(status_code=400, detail="bad path")
        if not os.path.isfile(full):
            raise HTTPException(status_code=404, detail="video not found")
        return FileResponse(  # starlette FileResponse handles HTTP Range → seeking works
            full,
            media_type="video/mp4",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return api
