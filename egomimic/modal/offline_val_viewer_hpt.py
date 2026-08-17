"""Web viewer for the HPT-only offline in-distribution validation passes — the
300M / 600M / 1B runs at their own (newer) common checkpoint epochs
(see egomimic/modal/offline_val.py).

The all-8 pass is capped by the 1.5B laggard and is served by the
`offline-val-viewer` app. This page skips 1.5B and validates 300M/600M/1B at
the newest epoch all three share. Same operator, same 3 seen-in-training
episodes throughout.

Two passes are on the page, switchable with the epoch toggle in the header:

    epoch 1979  offline_val_indist_hpt_e1979/  (current — all three runs have
                                                FINISHED, so this is final)
    epoch 1199  offline_val_indist_hpt3/       (the earlier HPT-only pass)

Deploy:
    MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/offline_val_viewer_hpt.py

Routes (single ASGI web function -> one URL):
    /            HTML viewer page
    /api/index   {"sets": [...], "default_epoch": N, "meta": {...}} — volume
                 scan of every epoch set, cached 60 s
    /video?path= streams one mp4 (path relative to the volume root, restricted
                 to the prefixes listed in EPOCH_SETS)
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]"
)

app = modal.App("offline-val-viewer-hpt", image=image)

outputs_volume = modal.Volume.from_name("egoverse-training-outputs")

MOUNT_PATH = "/data"
INDEX_TTL_S = 60

# (checkpoint epoch, volume prefix holding that pass's outputs). Newest first;
# EPOCH_SETS[0] is what the page opens on. 1979 is the largest epoch all three
# HPT runs have and they are all finished, so it is their FINAL epoch.
EPOCH_SETS = [
    (1979, "offline_val_indist_hpt_e1979"),
    (1199, "offline_val_indist_hpt3"),
]
ALLOWED_PREFIXES = tuple(p for _, p in EPOCH_SETS)
COMMON_EPOCH = EPOCH_SETS[0][0]  # default / headline epoch
ALL8_EPOCH = 1499  # all-8 common epoch used by the offline-val-viewer app
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
<title>offline in-dist val — HPT 300M/600M/1B</title>
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
  button.on { border-color: var(--accent); color: var(--accent); font-weight: 600; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  #epochtabs { display: flex; gap: 6px; align-items: center; }
  #epochtabs .lbl { color: var(--dim); font-size: 12px; margin-right: 2px; }
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
  <h1>offline in-distribution validation — HPT 300M/600M/1B</h1>
  <span id="epochtabs"><span class="lbl">epoch</span></span>
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
  let index = null;      // {sets:[{epoch,prefix,runs:[...]}], default_epoch, meta}
  let current = null;    // the selected set

  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        const v = e.target;
        if (!v.src && v.dataset.src) { v.src = v.dataset.src; v.preload = "metadata"; }
        observer.unobserve(v);
      }
    }
  }, { rootMargin: "200px" });

  function renderMeta(meta, set) {
    const eps = meta.episodes
      .map((e) => "<code>" + e[0] + "</code> (" + e[1] + " frames)")
      .join(" · ");
    const others = index.sets
      .filter((s) => s.epoch !== set.epoch)
      .map((s) => "<code>" + s.epoch + "</code>")
      .join(", ");
    metaEl.innerHTML =
      "HPT-only pass: 300M / 600M / 1B (1.5B skipped — it is the laggard that " +
      "caps the all-8 pass), at checkpoint epoch <code>" + set.epoch +
      "</code> (<code>" + set.prefix + "/</code>)" +
      (set.epoch === index.newest_epoch
        ? " — the newest epoch all three share. All three runs have FINISHED " +
          "training, so this is their FINAL-epoch number."
        : " — an earlier HPT-only pass, kept for comparison.") +
      (others ? " Other pass(es) on this page: " + others + "." : "") +
      " The all-8 pass now uses epoch <code>" + meta.all8_epoch +
      "</code> and is served by the offline-val-viewer app.<br>" +
      "In-distribution operator <code>" + meta.operator + "</code> " +
      "(SEEN in training — control vs the held-out-operator val). " +
      "3 episodes, " + meta.total_frames + " frames total: " + eps + "<br>" +
      "Videos: GT vs predicted actions, 1 frame per val sample, sequential " +
      "over the 3 episodes in hash order.";
  }

  function render(set) {
    grid.innerHTML = "";
    for (const run of set.runs) {
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
        '<span class="run-dir">' + run.dir + " @ epoch " + set.epoch + "</span>" +
        (metric ? '<span class="run-metric">' + metric + "</span>" : "");
      row.appendChild(head);

      const body = document.createElement("div");
      body.className = "vids";
      if (!run.videos.length) {
        const ph = document.createElement("div");
        ph.className = "placeholder";
        ph.textContent = "no offline-val videos yet for this run at epoch " + set.epoch;
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

  function show(epoch) {
    const set = index.sets.find((s) => s.epoch === epoch) || index.sets[0];
    current = set;
    document.querySelectorAll("#epochtabs button").forEach((b) => {
      b.classList.toggle("on", Number(b.dataset.epoch) === set.epoch);
    });
    renderMeta(index.meta, set);
    render(set);
    const nvids = set.runs.reduce((s, r) => s + r.videos.length, 0);
    const nruns = set.runs.filter((r) => r.videos.length).length;
    statusEl.textContent =
      "epoch " + set.epoch + " · " + nruns + "/" + set.runs.length + " runs · " +
      nvids + " videos · index refreshes every 60 s";
  }

  function renderTabs() {
    const box = $("epochtabs");
    for (const s of index.sets) {
      const b = document.createElement("button");
      b.dataset.epoch = s.epoch;
      const n = s.runs.reduce((a, r) => a + r.videos.length, 0);
      b.textContent = s.epoch + " (" + n + ")";
      b.title = s.prefix + "/ — " + n + " videos";
      b.onclick = () => show(s.epoch);
      box.appendChild(b);
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
    .then((data) => {
      index = data;
      renderTabs();
      show(index.default_epoch);
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
    root_dir = os.path.realpath(MOUNT_PATH)

    _cache = {"ts": 0.0, "data": None}
    _lock = threading.Lock()

    def _vid_sort_key(name):
        m = re.search(r"(\d+)\.mp4$", name)
        return (int(m.group(1)) if m else 1 << 30, name)

    def _scan_set(prefix):
        """One epoch set: per-run video lists (paths relative to the volume
        root, i.e. prefixed with `prefix/`) plus that run's manifest metrics."""
        base_dir = os.path.join(root_dir, prefix)
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
                                vids.append(
                                    f"{prefix}/{run_dir}/videos/{ep_name}/{emb}/{f}"
                                )
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
        return runs_out

    def _scan():
        return {
            "sets": [
                {"epoch": epoch, "prefix": prefix, "runs": _scan_set(prefix)}
                for epoch, prefix in EPOCH_SETS
            ],
            "default_epoch": COMMON_EPOCH,
            "newest_epoch": max(e for e, _ in EPOCH_SETS),
            "meta": {
                "all8_epoch": ALL8_EPOCH,
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
        # Resolve strictly under one of the EPOCH_SETS prefixes; reject traversal.
        if "\x00" in path or path.startswith(("/", "~")) or ".." in path.split("/"):
            raise HTTPException(status_code=400, detail="bad path")
        if path.split("/")[0] not in ALLOWED_PREFIXES:
            raise HTTPException(status_code=400, detail="bad prefix")
        full = os.path.realpath(os.path.join(root_dir, path))
        if not full.startswith(root_dir + os.sep) or not full.endswith(".mp4"):
            raise HTTPException(status_code=400, detail="bad path")
        if not os.path.isfile(full):
            raise HTTPException(status_code=404, detail="video not found")
        return FileResponse(  # starlette FileResponse handles HTTP Range
            full,
            media_type="video/mp4",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return api
