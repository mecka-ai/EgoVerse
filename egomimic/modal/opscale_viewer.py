"""Web viewer for the operator-diversity scaling experiment (op_scaling L1..L4).

Each level writes TWO validation-video families every 30 epochs (see
egomimic/eval/eval_video.py — videos land under <root_dir>/videos/epoch_<N>/
<EMBODIMENT>/validation_video_<i>.mp4, with the held-out-operator evaluator
writing to a videos_oph/ sibling):

    op_scaling/L<n>/videos/epoch_<N>/MECKA_BIMANUAL/validation_video_<i>.mp4
        in-domain val — 8 episodes from operators seen during training
    op_scaling/L<n>/videos_oph/epoch_<N>/MECKA_BIMANUAL/validation_video_<i>.mp4
        held-out-operator val — 5 episodes from an unseen operator (the same
        held-out set the data_div_oss dw48 runs validate on)

Deploy:
    MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/opscale_viewer.py

Routes (single ASGI web function → one URL):
    /            HTML viewer page (8 rows = 4 levels x 2 val families)
    /api/index   {"rows": [...], "epochs": [...]} — volume scan, cached 60 s
    /video?path= streams one mp4 from the volume (path relative to op_scaling/)
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")

app = modal.App("opscale-viewer", image=image)

outputs_volume = modal.Volume.from_name("egoverse-training-outputs")

MOUNT_PATH = "/data"
BASE_PREFIX = "op_scaling"
INDEX_TTL_S = 60

# (level dir, number of distinct operators in that level's training mix)
LEVELS = [
    ("L1", 8),
    ("L2", 24),
    ("L3", 72),
    ("L4", 160),
]

# (subdirectory, short family label, longer description) — two rows per level.
FAMILIES = [
    ("videos", "in-domain val", "8 episodes · operators seen in training"),
    ("videos_oph", "held-out-operator val", "5 episodes · unseen operator"),
]

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>op_scaling — operator diversity val videos</title>
<style>
  :root {
    --bg: #101216; --panel: #181b21; --border: #2a2e37;
    --text: #e6e8ec; --dim: #8b919d; --accent: #6ea8fe; --oph: #f0a868;
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
  .blurb { color: var(--dim); font-size: 12px; margin-top: 6px; max-width: 100%; }
  .blurb b { color: #b9c0cc; font-weight: 600; }
  .swatch {
    display: inline-block; width: 8px; height: 8px; border-radius: 2px;
    margin-right: 4px; vertical-align: middle;
  }
  .sw-id { background: var(--accent); }
  .sw-oph { background: var(--oph); }
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
    margin-bottom: 8px; padding: 8px 10px; border-left: 3px solid var(--accent);
  }
  .run-row.oph { border-left-color: var(--oph); }
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
    <h1>op_scaling — operator diversity</h1>
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
  <div class="blurb">
    Operator-diversity scaling at a <b>fixed 4.63 h data budget</b> on the
    <b>300M</b> model: L1/L2/L3/L4 spread the same amount of data over
    <b>8 / 24 / 72 / 160 operators</b>. Two val sets per level —
    <span class="swatch sw-id"></span><b>in-domain val</b> (8 episodes, operators
    seen in training) measures fit;
    <span class="swatch sw-oph"></span><b>held-out-operator val</b> (5 episodes,
    an operator never trained on) measures transfer to a new person. GT is green,
    prediction red.
  </div>
</header>
<main id="grid"></main>
<script>
(function () {
  let index = null;          // {rows:[{key,level,ops,family,label,sub,dir,epochs}], epochs:[]}
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
    let group = null, groupLevel = null;
    for (const row of index.rows) {
      if (row.level !== groupLevel) {
        groupLevel = row.level;
        group = document.createElement("div");
        group.className = "level-group";
        const gh = document.createElement("div");
        gh.className = "level-head";
        gh.innerHTML = row.level + " <span>· " + row.ops + " operators · 4.63 h · 300M</span>";
        group.appendChild(gh);
        grid.appendChild(group);
      }

      const el = document.createElement("div");
      el.className = "run-row" + (row.family === "videos_oph" ? " oph" : "");
      const head = document.createElement("div");
      head.className = "run-head";
      head.innerHTML =
        '<span class="run-label">' + row.label + "</span>" +
        '<span class="run-sub">' + row.sub + "</span>";
      el.appendChild(head);

      const vids = row.epochs[currentEpoch] || [];
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
    epochSel.value = ep;
    render();
    if (columnMode) { currentCol = 0; applyColumn(); }
  }

  function step(d) {
    const i = index.epochs.indexOf(currentEpoch) + d;
    if (i >= 0 && i < index.epochs.length) setEpoch(index.epochs[i]);
  }

  // ---- column mode: same episode slot across all 8 rows ----
  const colmodeBtn = $("colmode"), colctl = $("colctl"), colind = $("colind");

  function maxCols() {
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

  colmodeBtn.onclick = () => { if (index) setColumnMode(!columnMode); };
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

  fetch("api/index")
    .then((r) => r.json())
    .then((data) => {
      index = data;
      if (!index.epochs.length) {
        statusEl.textContent = "no validation videos found on the volume yet";
        return;
      }
      buildSelector();
      const nvids = index.rows.reduce(
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
    volumes={MOUNT_PATH: outputs_volume.read_only()},
    scaledown_window=300,
    max_containers=1,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label="opscale-viewer")
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
        for level, n_ops in LEVELS:
            for family, fam_label, fam_desc in FAMILIES:
                epochs = {}
                fam_dir = os.path.join(base_dir, level, family)
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
                                        f"{level}/{family}/{ep_name}/{emb}/{f}"
                                    )
                        if vids:
                            epochs[ep] = vids
                            all_epochs.add(ep)
                rows.append(
                    {
                        "key": f"{level}/{family}",
                        "level": level,
                        "ops": n_ops,
                        "family": family,
                        "label": f"{level} · {fam_label}",
                        "sub": fam_desc,
                        "dir": f"{BASE_PREFIX}/{level}/{family}",
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
        # Resolve strictly under op_scaling/ on the volume; reject traversal.
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
