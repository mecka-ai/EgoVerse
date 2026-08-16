"""Web viewer for the op_scaling TRAIN-SET offline validation videos.

Companion to opscale_viewer.py (which shows the live in-domain / held-out-
operator val videos). This one serves the OFFLINE pass written by
`egomimic/modal/offline_val.py::run_opscale`, which validates all four
operator-diversity levels at the SAME checkpoint epoch on the SAME 8 episodes:

    offline_val_opscale/L<n>/videos/epoch_0/MECKA_BIMANUAL/validation_video_<i>.mp4
    offline_val_opscale/L<n>/manifest.json   (episodes, layout, Valid/* metrics)

The 8 episodes are TRAIN-SET episodes — one per L1 operator, and each one is in
the train split of ALL FOUR levels — so every level has seen all 8 and the
comparison is a like-for-like fit measurement across diversity levels.

The offline pass runs with flush_per_episode=True, so video index i is exactly
episode i of the val set in episode-hash order (up to batch_size-1 frames of
spillover at each boundary). That makes column i the same episode/operator in
every row; the column headers are taken from each level's manifest.json.

Deploy:
    MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/opscale_trainval_viewer.py

Routes (single ASGI web function → one URL, relative paths so it works under
any mount prefix):
    /            HTML viewer page (4 rows = L1..L4, 8 columns = 8 episodes)
    /api/index   {"rows": [...], "columns": [...], "epoch": N} — cached 60 s
    /video?path= streams one mp4 (path relative to offline_val_opscale/)
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")

app = modal.App("opscale-trainval-viewer", image=image)

outputs_volume = modal.Volume.from_name("egoverse-training-outputs")

MOUNT_PATH = "/data"
BASE_PREFIX = "offline_val_opscale"
INDEX_TTL_S = 60

# (level dir, number of distinct operators in that level's training mix)
LEVELS = [
    ("L1", 8),
    ("L2", 24),
    ("L3", 72),
    ("L4", 160),
]

# episode_hash -> (operator, num_frames). Hash order == val-dataloader order ==
# video index order. Kept here so the page still labels columns if a level's
# manifest.json has not landed yet.
EPISODES = [
    ("69b083e65a299178939432ae", "696a8ab16adfd3c664a65c91", 3590),
    ("69b08bef2e8f3cdc83df98da", "6963a33b83a9fdf2d863cb6b", 3295),
    ("69b0a0624d596b45d52ba551", "6975db9bb393af9134ca5d21", 3598),
    ("69b8b2d61cf7f6f00d4364df", "6954b58920b100982d80f170", 2699),
    ("69b8b325a52e1a2126f45ffe", "6944be8574e27bfb2358061e", 2995),
    ("69b8c0beb3cc90fa8d9ac9ef", "6776bf817d12b76c8e1be433", 3601),
    ("69b92e31e749b83b1a333011", "6968000b0af001daaaad5168", 2605),
    ("69b9f1670ed8c646a6770f85", "6980e0c57c6b5a6b3c8cf16c", 3295),
]

# Headline metric to surface next to each row label.
HEADLINE_METRICS = [
    "Valid/mecka_bimanual_actions_cartesian_paired_mse_avg",
    "Valid/mecka_bimanual_actions_cartesian_final_mse_avg",
    "Valid/Loss",
]

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>op_scaling — train-set offline val</title>
<style>
  :root {
    --bg: #101216; --panel: #181b21; --border: #2a2e37;
    --text: #e6e8ec; --dim: #8b919d; --accent: #6ea8fe; --warm: #f0a868;
    --col: 340px;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  header {
    position: sticky; top: 0; z-index: 10; background: var(--bg);
    border-bottom: 1px solid var(--border); padding: 10px 16px;
  }
  .hrow { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  header h1 { font-size: 15px; margin: 0 8px 0 0; font-weight: 600; }
  .blurb { color: var(--dim); font-size: 12px; margin-top: 6px; }
  .blurb b { color: #b9c0cc; font-weight: 600; }
  .spacer { flex: 1; }
  #status { color: var(--dim); font-size: 12px; }
  button, select {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 10px; font-size: 13px; cursor: pointer;
  }
  button:hover, select:hover { border-color: var(--accent); }
  button.on { border-color: var(--accent); color: var(--accent); }
  #colctl { display: flex; align-items: center; gap: 6px; }
  #colind { color: var(--dim); font-size: 12px; min-width: 104px; text-align: center; }
  details.eps { margin-top: 6px; }
  details.eps summary { color: var(--accent); font-size: 12px; cursor: pointer; }
  table.eps { border-collapse: collapse; margin-top: 6px; font-size: 11.5px; }
  table.eps th, table.eps td {
    border: 1px solid var(--border); padding: 3px 8px; text-align: left;
  }
  table.eps th { color: var(--dim); font-weight: 500; }
  table.eps td.mono, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  main { padding: 12px 16px 40px; }
  /* one horizontally scrolling viewport shared by the column header strip and
     every level row, so column i stays vertically aligned across levels */
  .scroller { overflow-x: auto; padding-bottom: 6px; }
  .track { display: flex; gap: 10px; min-width: min-content; }
  .cell { flex: 0 0 auto; width: var(--col); }
  .colhead {
    background: #14171c; border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 7px; font-size: 11px; line-height: 1.35;
  }
  .colhead .idx { color: var(--accent); font-weight: 600; }
  .colhead .op { color: var(--dim); }
  .colhead.active { border-color: var(--accent); }
  .level-row {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    margin-bottom: 10px; padding: 8px 10px; border-left: 3px solid var(--accent);
  }
  .level-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 7px; }
  .level-label { font-size: 14px; font-weight: 600; }
  .level-sub { color: var(--dim); font-size: 11.5px; }
  .level-metric { margin-left: auto; font-size: 11.5px; color: var(--warm); }
  .cell video { width: 100%; border-radius: 6px; background: #000; display: block; }
  .cell.active-col video { outline: 2px solid var(--accent); }
  .cell .cap { color: var(--dim); font-size: 11px; margin-top: 3px; text-align: center; }
  .missing {
    border: 1px dashed var(--border); border-radius: 6px; color: var(--dim);
    font-size: 11.5px; text-align: center; padding: 30px 8px;
  }
  .placeholder {
    color: var(--dim); border: 1px dashed var(--border); border-radius: 6px;
    padding: 22px 16px; text-align: center; font-size: 12px;
  }
</style>
</head>
<body>
<header>
  <div class="hrow">
    <h1>op_scaling — train-set offline val</h1>
    <button id="playall">Play all</button>
    <button id="pauseall">Pause all</button>
    <button id="colmode" title="load and play one episode column at a time">Column mode</button>
    <span id="colctl" style="display:none">
      <button id="colprev">&#9664; ep</button>
      <span id="colind">Episode 1 of 8</span>
      <button id="colnext">ep &#9654;</button>
    </span>
    <span class="spacer"></span>
    <span id="status">loading…</span>
  </div>
  <div class="blurb" id="blurb"></div>
  <details class="eps">
    <summary>the 8 episodes (one per L1 operator, in every level's train split)</summary>
    <div id="epstable"></div>
  </details>
</header>
<main>
  <div class="scroller" id="headscroll"><div class="track" id="headtrack"></div></div>
  <div id="grid"></div>
</main>
<script>
(function () {
  let index = null;      // {epoch, columns:[{i,episode,operator,frames}], rows:[...]}
  let columnMode = false;
  let currentCol = 0;
  let playToken = 0;

  const $ = (id) => document.getElementById(id);
  const grid = $("grid"), statusEl = $("status");

  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        const v = e.target;
        if (!v.src && v.dataset.src) { v.src = v.dataset.src; v.preload = "metadata"; }
        observer.unobserve(v);
      }
    }
  }, { rootMargin: "200px" });

  // keep every row's horizontal scroll locked to the header strip, so column i
  // stays vertically aligned across all four levels
  let scrollers = [];
  let syncing = false;
  function registerScroller(el) {
    if (scrollers.indexOf(el) !== -1) return;
    scrollers.push(el);
    el.addEventListener("scroll", () => {
      if (syncing) return;
      syncing = true;
      for (const other of scrollers) {
        if (other !== el && other.isConnected) other.scrollLeft = el.scrollLeft;
      }
      syncing = false;
    });
  }

  function fmt(x) {
    if (x === null || x === undefined) return "—";
    const a = Math.abs(x);
    return (a !== 0 && (a < 1e-3 || a >= 1e4)) ? x.toExponential(3) : x.toFixed(5);
  }

  function renderHeader() {
    $("blurb").innerHTML =
      "Offline validation of all four operator-diversity levels at the <b>same " +
      "checkpoint, epoch " + index.epoch + "</b>, on the <b>same 8 TRAIN-SET " +
      "episodes</b> — one per L1 operator, each present in the train split of " +
      "L1, L2, L3 and L4, so every level has seen all 8 (this measures fit on " +
      "seen data, not transfer). Fixed 4.63 h data budget, 300M model; " +
      "L1/L2/L3/L4 spread it over 8/24/72/160 operators. GT is green, " +
      "prediction red. Column <i>i</i> is the same episode in every row.";

    let h = "<table class='eps'><tr><th>col</th><th>episode</th><th>operator</th>" +
            "<th>frames</th></tr>";
    for (const c of index.columns) {
      h += "<tr><td>" + (c.i + 1) + "</td><td class='mono'>" + c.episode +
           "</td><td class='mono'>" + c.operator + "</td><td>" + c.frames + "</td></tr>";
    }
    $("epstable").innerHTML = h + "</table>";

    const track = $("headtrack");
    track.innerHTML = "";
    for (const c of index.columns) {
      const d = document.createElement("div");
      d.className = "cell colhead";
      d.dataset.col = c.i;
      d.innerHTML = "<span class='idx'>#" + (c.i + 1) + "</span> " +
        "<span class='mono'>" + c.episode.slice(0, 8) + "…</span><br>" +
        "<span class='op'>op <span class='mono'>" + c.operator.slice(0, 8) + "…</span>" +
        " · " + c.frames + " f</span>";
      track.appendChild(d);
    }
  }

  function render() {
    grid.innerHTML = "";
    scrollers = scrollers.filter((el) => el.isConnected);  // drop removed rows
    for (const row of index.rows) {
      const el = document.createElement("div");
      el.className = "level-row";
      const head = document.createElement("div");
      head.className = "level-head";
      const metricBits = row.metrics_summary.length
        ? row.metrics_summary.map((m) => m.label + " " + fmt(m.value)).join(" · ")
        : (row.videos.filter(Boolean).length ? "metrics pending" : "not run yet");
      head.innerHTML =
        "<span class='level-label'>" + row.level + "</span>" +
        "<span class='level-sub'>" + row.ops + " operators · " +
        row.n_videos + "/" + index.columns.length + " videos</span>" +
        "<span class='level-metric'>" + metricBits + "</span>";
      el.appendChild(head);

      if (!row.n_videos) {
        const ph = document.createElement("div");
        ph.className = "placeholder";
        ph.textContent = "no offline-val videos on the volume for " + row.level + " yet";
        el.appendChild(ph);
        grid.appendChild(el);
        continue;
      }

      const sc = document.createElement("div");
      sc.className = "scroller";
      const track = document.createElement("div");
      track.className = "track";
      index.columns.forEach((c) => {
        const cell = document.createElement("div");
        cell.className = "cell";
        cell.dataset.col = c.i;
        const p = row.videos[c.i];
        if (!p) {
          cell.innerHTML = "<div class='missing'>no video for episode #" +
            (c.i + 1) + "</div>";
        } else {
          const v = document.createElement("video");
          v.controls = true; v.muted = true; v.loop = true;
          v.playsInline = true; v.preload = "none";
          v.dataset.src = "video?path=" + encodeURIComponent(p);
          v.title = p;
          if (!columnMode) observer.observe(v);
          cell.appendChild(v);
          const cap = document.createElement("div");
          cap.className = "cap";
          cap.textContent = p.split("/").pop();
          cell.appendChild(cap);
        }
        track.appendChild(cell);
      });
      sc.appendChild(track);
      el.appendChild(sc);
      grid.appendChild(el);
      registerScroller(sc);
    }
    registerScroller($("headscroll"));
  }

  // ---- column mode: same episode across all 4 levels ----
  const colmodeBtn = $("colmode"), colctl = $("colctl"), colind = $("colind");

  function playSynced(videos) {
    const token = ++playToken;
    const ready = videos.map((v) => new Promise((res) => {
      if (v.readyState >= 2) return res();
      const on = () => { v.removeEventListener("loadeddata", on); res(); };
      v.addEventListener("loadeddata", on);
      setTimeout(res, 8000);
    }));
    Promise.all(ready).then(() => {
      if (token !== playToken) return;
      videos.forEach((v) => {
        try { v.currentTime = 0; } catch (e) {}
        v.play().catch(() => {});
      });
    });
  }

  function applyColumn() {
    const active = [];
    document.querySelectorAll(".colhead").forEach((h) => {
      h.classList.toggle("active", Number(h.dataset.col) === currentCol);
    });
    grid.querySelectorAll(".level-row").forEach((row) => {
      row.querySelectorAll(".cell").forEach((cell) => {
        const v = cell.querySelector("video");
        const on = Number(cell.dataset.col) === currentCol;
        cell.classList.toggle("active-col", on);
        if (!v) return;
        if (on) {
          if (!v.getAttribute("src") && v.dataset.src) {
            v.src = v.dataset.src; v.preload = "auto";
          }
          active.push(v);
          cell.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
        } else {
          v.pause();
          if (v.getAttribute("src")) { v.removeAttribute("src"); v.load(); }
        }
      });
    });
    colind.textContent = "Episode " + (currentCol + 1) + " of " + index.columns.length;
    playSynced(active);
  }

  function colStep(d) {
    const n = index.columns.length;
    const next = Math.min(n - 1, Math.max(0, currentCol + d));
    if (next === currentCol) return;
    currentCol = next;
    applyColumn();
  }

  function setColumnMode(on) {
    columnMode = on;
    colmodeBtn.classList.toggle("on", on);
    colctl.style.display = on ? "" : "none";
    if (on) { observer.disconnect(); currentCol = 0; applyColumn(); }
    else { playToken++; render(); }
  }

  colmodeBtn.onclick = () => { if (index) setColumnMode(!columnMode); };
  $("colprev").onclick = () => colStep(-1);
  $("colnext").onclick = () => colStep(1);
  $("playall").onclick = () => {
    if (columnMode) { applyColumn(); return; }
    document.querySelectorAll("video").forEach((v) => {
      if (!v.src && v.dataset.src) v.src = v.dataset.src;
      v.play().catch(() => {});
    });
  };
  $("pauseall").onclick = () => {
    playToken++;
    document.querySelectorAll("video").forEach((v) => v.pause());
  };
  document.addEventListener("keydown", (e) => {
    if (!columnMode) return;
    if (e.key === "ArrowLeft") colStep(-1);
    if (e.key === "ArrowRight") colStep(1);
  });

  fetch("api/index")
    .then((r) => r.json())
    .then((data) => {
      index = data;
      renderHeader();
      render();
      const n = index.rows.reduce((s, r) => s + r.n_videos, 0);
      const done = index.rows.filter((r) => r.n_videos).length;
      statusEl.textContent =
        done + "/" + index.rows.length + " levels · " + n +
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
@modal.asgi_app(label="opscale-trainval-viewer")
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

    def _vid_index(name):
        m = re.search(r"validation_video_(\d+)\.mp4$", name)
        return int(m.group(1)) if m else None

    def _read_manifest(level):
        try:
            with open(os.path.join(base_dir, level, "manifest.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _columns(manifests):
        """Column i == val episode i (hash order). Prefer a manifest's own
        episode list; fall back to the checked-in EPISODES table."""
        eps = None
        for m in manifests.values():
            if m and m.get("episodes"):
                eps = list(m["episodes"])
                meta = m.get("episode_meta") or {}
                return [
                    {
                        "i": i,
                        "episode": h,
                        "operator": (meta.get(h) or {}).get("operator", "?"),
                        "frames": (meta.get(h) or {}).get("num_frames", 0),
                    }
                    for i, h in enumerate(eps)
                ]
        return [
            {"i": i, "episode": h, "operator": op, "frames": nf}
            for i, (h, op, nf) in enumerate(EPISODES)
        ]

    def _scan():
        manifests = {level: _read_manifest(level) for level, _ in LEVELS}
        columns = _columns(manifests)
        n_cols = len(columns)
        epoch = next(
            (m["epoch"] for m in manifests.values() if m and m.get("epoch") is not None),
            "?",
        )

        rows = []
        for level, n_ops in LEVELS:
            videos = [None] * n_cols
            vid_root = os.path.join(base_dir, level, "videos")
            if os.path.isdir(vid_root):
                for ep_name in sorted(os.listdir(vid_root)):
                    ep_dir = os.path.join(vid_root, ep_name)
                    if not ep_name.startswith("epoch_") or not os.path.isdir(ep_dir):
                        continue
                    for emb in sorted(os.listdir(ep_dir)):
                        emb_dir = os.path.join(ep_dir, emb)
                        if not os.path.isdir(emb_dir):
                            continue
                        for f in sorted(os.listdir(emb_dir)):
                            i = _vid_index(f) if f.endswith(".mp4") else None
                            if i is not None and 0 <= i < n_cols:
                                videos[i] = f"{level}/videos/{ep_name}/{emb}/{f}"

            man = manifests[level]
            metrics = (man or {}).get("metrics") or {}

            def _label(k):
                return (
                    k.split("/")[-1]
                    .replace("mecka_bimanual_actions_cartesian_", "")
                    .replace("_avg", "")
                )

            summary = [
                {"label": _label(k), "value": metrics[k]}
                for k in HEADLINE_METRICS
                if k in metrics
            ]
            if not summary:  # any *paired_mse_avg the run happened to log
                summary = [
                    {"label": _label(k), "value": v}
                    for k, v in sorted(metrics.items())
                    if k.endswith("paired_mse_avg")
                ][:2]
            rows.append(
                {
                    "level": level,
                    "ops": n_ops,
                    "dir": f"{BASE_PREFIX}/{level}",
                    "videos": videos,
                    "n_videos": sum(1 for v in videos if v),
                    "metrics": metrics,
                    "metrics_summary": summary,
                }
            )
        return {"epoch": epoch, "columns": columns, "rows": rows}

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
        # Resolve strictly under offline_val_opscale/ on the volume; reject traversal.
        if "\x00" in path or path.startswith(("/", "~")) or ".." in path.split("/"):
            raise HTTPException(status_code=400, detail="bad path")
        full = os.path.realpath(os.path.join(base_dir, path))
        if not full.startswith(base_dir + os.sep) or not full.endswith(".mp4"):
            raise HTTPException(status_code=400, detail="bad path")
        if not os.path.isfile(full):
            raise HTTPException(status_code=404, detail="video not found")
        return FileResponse(  # starlette FileResponse handles HTTP Range → seeking
            full,
            media_type="video/mp4",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return api
