"""Web viewer for the packaging_snacks model-size sweep (pksnack_size).

Four model sizes (300M / 600M / 1B / 1.5B) all trained on the FULL ~48 h
packaging_snacks pool. Each run does TWO validation passes, each rendering its
own GT-vs-pred video family every 30 epochs. The dir names below were read off
the runs' own .hydra/config.yaml on the volume:

    videos/           valid_datasets = pksnack_val_ophold5.json
                      held-out-operator val (5 episodes from an operator the
                      model never trained on) — the headline metric here
    videos_trainviz/  train_viz_evaluator (video_dirname: videos_trainviz,
                      metric_prefix Valid_trainviz) = pksnack_trainviz4.json —
                      train-viz (4 episodes the model DID train on; upper bound /
                      sanity check rather than generalization)

Note this differs from pksnack_ops, where videos/ is an in-domain-operator
holdout and the held-out operator lives in videos_oph/.

Full path pattern (see egomimic/eval/eval_video.py):
    pksnack_size/<size>/<family>/epoch_<N>/MECKA_BIMANUAL/validation_video_<i>.mp4

Deploy:
    MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/pksnack_size_viewer.py

Routes (single ASGI web function → one URL):
    /            HTML viewer page (up to 8 rows = 4 sizes x 2 val families)
    /api/index   {"rows": [...], "epochs": [...]} — volume scan, cached 60 s
    /video?path= streams one mp4 from the volume (path relative to pksnack_size/)
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")

app = modal.App("pksnack-size-viewer", image=image)

outputs_volume = modal.Volume.from_name("egoverse-training-outputs")

MOUNT_PATH = "/data"
BASE_PREFIX = "pksnack_size"
INDEX_TTL_S = 60

PAGE_TITLE = "pksnack_size — model size sweep val videos"
PAGE_HEADING = "pksnack_size — model size sweep"

# (group dir, group caption) — one card per model size.
GROUPS = [
    ("300M", "300M params"),
    ("600M", "600M params"),
    ("1B", "1B params"),
    ("1_5B", "1.5B params"),
]
GROUP_SUFFIX = "· full ~48 h pool"

# (subdir, short label, description, css class) — one row per (group, family).
FAMILIES = [
    ("videos", "held-out-operator val", "5 eps · operator never seen", "f1"),
    ("videos_trainviz", "train-viz", "4 eps · trained on (sanity check)", "f2"),
]

BLURB_HTML = """
    Model-size sweep on <b>packaging_snacks</b>: 300M / 600M / 1B / 1.5B, every
    run trained on the <b>full ~48 h pool</b> (all operators) — so this isolates
    capacity, not data. Two val passes per size —
    <span class="swatch sw-f1"></span><b>held-out-operator val</b>: 5 episodes
    from an operator the model never trained on (the headline metric);
    <span class="swatch sw-f2"></span><b>train-viz</b>: 4 episodes the model DID
    train on (upper bound / sanity check, not generalization).
    GT is green, prediction red.
"""

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #101216; --panel: #181b21; --border: #2a2e37;
    --text: #e6e8ec; --dim: #8b919d; --accent: #6ea8fe;
    --f0: #6ea8fe; --f1: #f0a868; --f2: #7fd1a4;
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
  .swatch {
    display: inline-block; width: 8px; height: 8px; border-radius: 2px;
    margin-right: 4px; vertical-align: middle;
  }
  .sw-f0 { background: var(--f0); }
  .sw-f1 { background: var(--f1); }
  .sw-f2 { background: var(--f2); }
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
  .level-group {
    border: 1px solid var(--border); border-radius: 10px;
    margin-bottom: 14px; padding: 8px 10px 4px; background: #14171c;
  }
  .level-head { font-size: 14px; font-weight: 600; margin: 2px 4px 8px; }
  .level-head span { color: var(--dim); font-weight: 400; font-size: 12px; }
  .run-row {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    margin-bottom: 8px; padding: 8px 10px; border-left: 3px solid var(--border);
  }
  .run-row.f0 { border-left-color: var(--f0); }
  .run-row.f1 { border-left-color: var(--f1); }
  .run-row.f2 { border-left-color: var(--f2); }
  .run-head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 7px; }
  .run-label { font-size: 13px; font-weight: 600; }
  .run-sub { color: var(--dim); font-size: 11px; }
  .run-note { color: var(--dim); font-size: 11px; margin-left: auto; }
  .vids { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 4px; }
  .vid-card { flex: 0 0 auto; width: 340px; }
  .vid-card video { width: 100%; border-radius: 6px; background: #000; display: block; }
  .vid-card.active-col video { outline: 2px solid var(--accent); }
  .vid-cap { color: var(--dim); font-size: 11px; margin-top: 3px; text-align: center; }
  .placeholder {
    color: var(--dim); border: 1px dashed var(--border); border-radius: 6px;
    padding: 22px 16px; text-align: center; font-size: 12px; width: 100%;
  }
  .col-ph { flex: 0 0 auto; width: 340px; }
</style>
</head>
<body>
<header>
  <div class="hrow">
    <h1>__HEADING__</h1>
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
  </div>
  <div class="blurb">__BLURB__</div>
</header>
<main id="grid"></main>
<script>
(function () {
  let index = null;   // {rows:[{key,group,groupCaption,groupSuffix,family,cls,label,sub,epochs}], epochs:[]}
  let currentEpoch = null;
  let columnMode = false;
  let currentCol = 0;
  let playToken = 0;

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
    return index.rows.filter((r) => r.epochs[ep] && r.epochs[ep].length).length;
  }

  function buildSelector() {
    epochSel.innerHTML = "";
    for (const ep of index.epochs) {
      const o = document.createElement("option");
      o.value = ep;
      o.textContent = "epoch " + ep + "  (" + coverage(ep) + "/" + index.rows.length + " rows)";
      epochSel.appendChild(o);
    }
    epochSel.disabled = index.epochs.length === 0;
  }

  function defaultEpoch() {
    // latest epoch every started row has → most comparable snapshot
    const maxes = index.rows
      .map((r) => Object.keys(r.epochs).map(Number))
      .filter((a) => a.length)
      .map((a) => Math.max.apply(null, a));
    if (!maxes.length) return null;
    const common = Math.min.apply(null, maxes);
    return index.epochs.includes(common) ? common : index.epochs[index.epochs.length - 1];
  }

  function render() {
    grid.innerHTML = "";
    let group = null, groupKey = null;
    for (const row of index.rows) {
      if (row.group !== groupKey) {
        groupKey = row.group;
        group = document.createElement("div");
        group.className = "level-group";
        const gh = document.createElement("div");
        gh.className = "level-head";
        gh.innerHTML = row.group + " <span>· " + row.groupCaption + " " + row.groupSuffix + "</span>";
        group.appendChild(gh);
        grid.appendChild(group);
      }

      const el = document.createElement("div");
      el.className = "run-row " + row.cls;
      const head = document.createElement("div");
      head.className = "run-head";
      head.innerHTML =
        '<span class="run-label">' + row.label + "</span>" +
        '<span class="run-sub">' + row.sub + "</span>";
      el.appendChild(head);

      const vids = (currentEpoch === null ? [] : row.epochs[currentEpoch]) || [];
      const body = document.createElement("div");
      body.className = "vids";
      if (!vids.length) {
        const eps = Object.keys(row.epochs).map(Number);
        const latest = eps.length ? Math.max.apply(null, eps) : null;
        const ph = document.createElement("div");
        ph.className = "placeholder";
        ph.textContent = latest === null
          ? "no videos yet for this row"
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
          const cap = document.createElement("div");
          cap.className = "vid-cap";
          cap.textContent = p.split("/").pop();
          card.appendChild(cap);
          body.appendChild(card);
        });
        const n = vids.length;
        head.insertAdjacentHTML(
          "beforeend",
          '<span class="run-note">' + n + " video" + (n === 1 ? "" : "s") + "</span>"
        );
      }
      el.appendChild(body);
      group.appendChild(el);
    }
  }

  function setEpoch(ep) {
    currentEpoch = ep;
    if (ep !== null) epochSel.value = ep;
    render();
    if (columnMode) { currentCol = 0; applyColumn(); }
  }

  function step(d) {
    const i = index.epochs.indexOf(currentEpoch) + d;
    if (i >= 0 && i < index.epochs.length) setEpoch(index.epochs[i]);
  }

  // ---- column mode: same episode slot across every row ----
  const colmodeBtn = $("colmode"), colctl = $("colctl"), colind = $("colind");

  function maxCols() {
    if (currentEpoch === null) return 0;
    return Math.max(0, ...index.rows.map((r) => (r.epochs[currentEpoch] || []).length));
  }

  function playSynced(videos) {
    const token = ++playToken;
    const ready = videos.map((v) => new Promise((res) => {
      if (v.readyState >= 2) return res();
      const on = () => { v.removeEventListener("loadeddata", on); res(); };
      v.addEventListener("loadeddata", on);
      setTimeout(res, 5000);
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
          if (v.getAttribute("src")) { v.removeAttribute("src"); v.load(); }
        }
      });
      if (found) {
        found.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
      } else if (cards.length) {
        const ph = document.createElement("div");
        ph.className = "placeholder col-ph";
        ph.textContent =
          "no video #" + (currentCol + 1) + " for this row at epoch " + currentEpoch;
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
      observer.disconnect();
      currentCol = 0;
      applyColumn();
    } else {
      playToken++;
      render();
    }
  }

  colmodeBtn.onclick = () => { if (index && index.epochs.length) setColumnMode(!columnMode); };
  $("colprev").onclick = () => colStep(-1);
  $("colnext").onclick = () => colStep(1);
  // ---- end column mode ----

  $("prev").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  epochSel.onchange = () => setEpoch(Number(epochSel.value));
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
    if (e.key === "ArrowLeft") columnMode ? colStep(-1) : step(-1);
    if (e.key === "ArrowRight") columnMode ? colStep(1) : step(1);
  });

  function load() {
    return fetch("api/index")
      .then((r) => r.json())
      .then((data) => {
        index = data;
        buildSelector();
        if (!index.epochs.length) {
          // Runs are young: first validation fires at epoch 29. Show the row
          // skeleton with placeholders instead of a blank page, and re-poll.
          statusEl.textContent =
            "no validation videos yet — first val runs at epoch 29 (re-checking every 60 s)";
          setEpoch(null);
          setTimeout(load, 60000);
          return;
        }
        const nvids = index.rows.reduce(
          (s, r) => s + Object.values(r.epochs).reduce((t, v) => t + v.length, 0), 0);
        statusEl.textContent =
          index.epochs.length + " epochs · " + nvids + " videos · index refreshes every 60 s";
        setEpoch(defaultEpoch());
      })
      .catch((e) => { statusEl.textContent = "failed to load index: " + e; });
  }
  load();
})();
</script>
</body>
</html>
"""

PAGE_HTML = (
    PAGE_HTML.replace("__TITLE__", PAGE_TITLE)
    .replace("__HEADING__", PAGE_HEADING)
    .replace("__BLURB__", BLURB_HTML)
)


@app.function(
    volumes={MOUNT_PATH: outputs_volume.read_only()},
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label="pksnack-size-viewer")
def viewer():
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
        rows = []
        all_epochs = set()
        for group, caption in GROUPS:
            for family, fam_label, fam_desc, cls in FAMILIES:
                epochs = {}
                fam_dir = os.path.join(base_dir, group, family)
                if os.path.isdir(fam_dir):
                    for ep_name in os.listdir(fam_dir):
                        if not ep_name.startswith("epoch_"):
                            continue
                        try:
                            ep = int(ep_name.split("_", 1)[1])
                        except ValueError:
                            continue
                        ep_dir = os.path.join(fam_dir, ep_name)
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
                                    vids.append(
                                        f"{group}/{family}/{ep_name}/{emb}/{f}"
                                    )
                        # An epoch dir can exist while the trainer is still
                        # writing it — only surface epochs that have real files.
                        if vids:
                            epochs[ep] = vids
                            all_epochs.add(ep)
                rows.append(
                    {
                        "key": f"{group}/{family}",
                        "group": group,
                        "groupCaption": caption,
                        "groupSuffix": GROUP_SUFFIX,
                        "family": family,
                        "cls": cls,
                        "label": f"{group} · {fam_label}",
                        "sub": fam_desc,
                        "dir": f"{BASE_PREFIX}/{group}/{family}",
                        "epochs": epochs,
                    }
                )
        return {"rows": rows, "epochs": sorted(all_epochs)}

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
        # Resolve strictly under the experiment prefix; reject traversal.
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
