"""Web viewer for the SECOND offline in-distribution validation pass — the
300M / 600M / 1B HPT runs only, at their own (newer) common checkpoint epoch
(see egomimic/modal/offline_val.py; outputs under offline_val_indist_hpt3/).

The first pass validated all 8 runs at epoch 539 (the largest epoch every run
had, capped by the 1.5B laggard) — served by the `offline-val-viewer` app.
This pass skips 1.5B and re-validates 300M/600M/1B at the newest epoch all
three share (1B's latest checkpoint). Same operator, same 3 seen-in-training
episodes.

Deploy:
    MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/offline_val_viewer_hpt.py

Routes (single ASGI web function -> one URL):
    /            HTML viewer page
    /api/index   {"runs": [...], "meta": {...}} — volume scan, cached 60 s
    /video?path= streams one mp4 (path relative to offline_val_indist_hpt3/)
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]"
)

app = modal.App("offline-val-viewer-hpt", image=image)

outputs_volume = modal.Volume.from_name("egoverse-training-outputs")

MOUNT_PATH = "/data"
BASE_PREFIX = "offline_val_indist_hpt3"
INDEX_TTL_S = 60

COMMON_EPOCH = 1199  # newest epoch present for all of 300M/600M/1B (verified)
FIRST_PASS_EPOCH = 539  # all-8 common epoch used by the first pass
INDIST_OPERATOR = "68b5da0ce7c6a693e3df941c"
INDIST_EPISODES = [
    ("69b2100ed99f29421f1b4a57", 1825),
    ("69b335f8290f064f72218fab", 2083),
    ("69b37b384af76c8acce9cc65", 2245),
]

# (short label, run dir under offline_val_indist_hpt3/)
RUNS = [
    ("300M", "300M_mm_nobc_dw48"),
    ("600M", "600M_mm_nobc_dw48"),
    ("1B", "1B_mm_nobc_dw48"),
]

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>offline in-dist val — HPT @ 1199</title>
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
  button {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 10px; font-size: 13px; cursor: pointer;
  }
  button:hover { border-color: var(--accent); }
  .spacer { flex: 1; }
  #status { color: var(--dim); font-size: 12px; }
  #meta {
    padding: 10px 16px; color: var(--dim); font-size: 12.5px;
    border-bottom: 1px solid var(--border); line-height: 1.7;
  }
  #meta code { color: var(--text); background: var(--panel); padding: 1px 5px; border-radius: 4px; }
  main { padding: 12px 16px 40px; }
  .run-row {
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px;
    margin-bottom: 12px; padding: 10px 12px;
  }
  .run-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px; flex-wrap: wrap; }
  .run-label { font-size: 15px; font-weight: 600; }
  .run-dir { color: var(--dim); font-size: 12px; }
  .run-metric { color: var(--accent); font-size: 12px; }
  .run-note { color: var(--dim); font-size: 12px; margin-left: auto; }
  .vids { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 4px; }
  .vid-card { flex: 0 0 auto; width: 380px; }
  .vid-card video { width: 100%; border-radius: 6px; background: #000; display: block; }
  .vid-cap { color: var(--dim); font-size: 11px; margin-top: 3px; text-align: center; }
  .placeholder {
    color: var(--dim); border: 1px dashed var(--border); border-radius: 6px;
    padding: 28px 16px; text-align: center; font-size: 13px; width: 100%;
  }
</style>
</head>
<body>
<header>
  <h1>offline in-distribution validation — HPT 300M/600M/1B @ epoch 1199</h1>
  <button id="playall">Play all</button>
  <button id="pauseall">Pause all</button>
  <span class="spacer"></span>
  <span id="status">loading…</span>
</header>
<div id="meta"></div>
<main id="grid"></main>
<script>
(function () {
  const $ = (id) => document.getElementById(id);
  const grid = $("grid"), statusEl = $("status"), metaEl = $("meta");

  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        const v = e.target;
        if (!v.src && v.dataset.src) { v.src = v.dataset.src; v.preload = "metadata"; }
        observer.unobserve(v);
      }
    }
  }, { rootMargin: "200px" });

  function renderMeta(meta) {
    const eps = meta.episodes
      .map((e) => "<code>" + e[0] + "</code> (" + e[1] + " frames)")
      .join(" · ");
    metaEl.innerHTML =
      "SECOND pass: 300M / 600M / 1B only (1.5B skipped — it is the laggard " +
      "that capped the first pass), at checkpoint epoch <code>" + meta.epoch +
      "</code> — the newest epoch all three share (= 1B's latest checkpoint " +
      "at selection time). The first, all-8 pass used epoch <code>" +
      meta.first_pass_epoch + "</code> and is served by the offline-val-viewer " +
      "app.<br>" +
      "In-distribution operator <code>" + meta.operator + "</code> " +
      "(SEEN in training — control vs the held-out-operator val). " +
      "3 episodes, " + meta.total_frames + " frames total: " + eps + "<br>" +
      "Videos: GT vs predicted actions, 1 frame per val sample, sequential " +
      "over the 3 episodes in hash order.";
  }

  function render(index) {
    grid.innerHTML = "";
    for (const run of index.runs) {
      const row = document.createElement("div");
      row.className = "run-row";
      const head = document.createElement("div");
      head.className = "run-head";
      let metric = "";
      if (run.metrics) {
        const keys = Object.keys(run.metrics).filter((k) => /paired_mse/.test(k));
        metric = keys
          .map((k) => k.replace("Valid/", "") + " = " + run.metrics[k].toFixed(4))
          .join("   ");
      }
      head.innerHTML =
        '<span class="run-label">' + run.label + "</span>" +
        '<span class="run-dir">' + run.dir + " @ epoch " + index.meta.epoch + "</span>" +
        (metric ? '<span class="run-metric">' + metric + "</span>" : "");
      row.appendChild(head);

      const body = document.createElement("div");
      body.className = "vids";
      if (!run.videos.length) {
        const ph = document.createElement("div");
        ph.className = "placeholder";
        ph.textContent = "no offline-val videos yet for this run";
        body.appendChild(ph);
      } else {
        for (const p of run.videos) {
          const card = document.createElement("div");
          card.className = "vid-card";
          const v = document.createElement("video");
          v.controls = true; v.muted = true; v.loop = true;
          v.playsInline = true; v.preload = "none";
          v.dataset.src = "video?path=" + encodeURIComponent(p);
          v.title = p;
          observer.observe(v);
          card.appendChild(v);
          const cap = document.createElement("div");
          cap.className = "vid-cap";
          cap.textContent = p.split("/").slice(-1)[0];
          card.appendChild(cap);
          body.appendChild(card);
        }
        head.insertAdjacentHTML(
          "beforeend",
          '<span class="run-note">' + run.videos.length + " videos</span>"
        );
      }
      row.appendChild(body);
      grid.appendChild(row);
    }
  }

  $("playall").onclick = () => {
    document.querySelectorAll("video").forEach((v) => {
      if (!v.src && v.dataset.src) v.src = v.dataset.src;
      v.play().catch(() => {});
    });
  };
  $("pauseall").onclick = () => {
    document.querySelectorAll("video").forEach((v) => v.pause());
  };

  fetch("api/index")
    .then((r) => r.json())
    .then((index) => {
      renderMeta(index.meta);
      render(index);
      const nvids = index.runs.reduce((s, r) => s + r.videos.length, 0);
      const nruns = index.runs.filter((r) => r.videos.length).length;
      statusEl.textContent =
        nruns + "/" + index.runs.length + " runs · " + nvids +
        " videos · index refreshes every 60 s";
    })
    .catch((e) => { statusEl.textContent = "failed to load index: " + e; });
})();
</script>
</body>
</html>
"""


@app.function(
    volumes={MOUNT_PATH: outputs_volume.read_only()},
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label="offline-val-viewer-hpt")
def viewer():
    import json
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

    def _vid_sort_key(name):
        m = re.search(r"(\d+)\.mp4$", name)
        return (int(m.group(1)) if m else 1 << 30, name)

    def _scan():
        runs_out = []
        for label, run_dir in RUNS:
            vids = []
            videos_dir = os.path.join(base_dir, run_dir, "videos")
            if os.path.isdir(videos_dir):
                for ep_name in sorted(os.listdir(videos_dir)):
                    ep_dir = os.path.join(videos_dir, ep_name)
                    if not os.path.isdir(ep_dir):
                        continue
                    for emb in sorted(os.listdir(ep_dir)):
                        emb_dir = os.path.join(ep_dir, emb)
                        if not os.path.isdir(emb_dir):
                            continue
                        for f in sorted(os.listdir(emb_dir), key=_vid_sort_key):
                            if f.endswith(".mp4"):
                                vids.append(f"{run_dir}/videos/{ep_name}/{emb}/{f}")
            metrics = None
            manifest_path = os.path.join(base_dir, run_dir, "manifest.json")
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path) as f:
                        metrics = json.load(f).get("metrics")
                except (OSError, ValueError):
                    pass
            runs_out.append(
                {"label": label, "dir": run_dir, "videos": vids, "metrics": metrics}
            )
        return {
            "runs": runs_out,
            "meta": {
                "epoch": COMMON_EPOCH,
                "first_pass_epoch": FIRST_PASS_EPOCH,
                "operator": INDIST_OPERATOR,
                "episodes": INDIST_EPISODES,
                "total_frames": sum(n for _, n in INDIST_EPISODES),
            },
        }

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

    @api.get("/video")
    def video(path: str = Query(...)):
        # Resolve strictly under offline_val_indist_hpt3/; reject traversal.
        if "\x00" in path or path.startswith(("/", "~")) or ".." in path.split("/"):
            raise HTTPException(status_code=400, detail="bad path")
        full = os.path.realpath(os.path.join(base_dir, path))
        if not full.startswith(base_dir + os.sep) or not full.endswith(".mp4"):
            raise HTTPException(status_code=400, detail="bad path")
        if not os.path.isfile(full):
            raise HTTPException(status_code=404, detail="video not found")
        return FileResponse(  # starlette FileResponse handles HTTP Range
            full,
            media_type="video/mp4",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return api
