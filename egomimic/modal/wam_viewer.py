"""Web viewer for the WAM world-action model's validation videos.

Run under inspection: ``data_div_oss_wam22_dw48_v2`` — a Wan2.2-TI2V-5B
world-ACTION model trained on the dishwashing 48 h pool, validating on the
held-out-operator split (dishwashing_val_ophold5). Every val pass writes a
matched pair per sample:

    predicted_video_<i>.mp4   the model's IMAGINED future frames (dream), with
                              the action-trail overlay drawn on top
    validation_video_<i>.mp4  the GROUND-TRUTH clip for the same window, same
                              overlay — the reference to compare the dream to

THERE ARE TWO GENERATIONS OF VIDEO ON THE VOLUME and they must not be mixed
silently. Measured (not assumed — every number below was read out of the actual
mp4 moov atoms, cross-checked against ffprobe):

    post-fix   1936 frames @ 5 fps = 387.2 s   ~50 MB   one FULL episode
    pre-fix      32 frames @ 10 fps =  3.2 s   ~0.4 MB  one chunk, misaligned TF

Sources surfaced, in order of trustworthiness:

  1. FIXED offline pass — wam_gateA/fixcheck_*/videos/epoch_0/MECKA_BIMANUAL/
     The reference-quality pair: full episode (69b22fc5f4f4e149281a6635),
     teacher-forced rolling, latest checkpoint, verified 1936 f @ 5 fps.
  2. Offline sweep (fixed) — wam_val_sweep/ (a per-checkpoint 4xH200 sweep).
     Not present at build time; the scanner picks it up automatically the
     moment it appears (its prefix is already allow-listed), no redeploy.
  3. Live training-time val — data_div_oss/wam22_dw48_v2/videos/epoch_<N>/
     45 epochs (29 -> 1349) x ~100 pairs, ALL pre-fix at build time: the live
     run only picks up the fix on a future resubmit. Capped to the first
     TRAINING_CAP pairs per epoch to keep the page light.

Generation is detected PER EPOCH from the real file (frame count + fps parsed
from the mp4 header/trailer, no ffmpeg), so when the live run does cut over the
badge flips on its own instead of lying.

Deploy:
    MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/wam_viewer.py

Routes (single ASGI web function -> one URL, relative paths):
    /            HTML viewer page
    /api/index   {"fixed":…, "sweep":…, "training":…} — volume scan, cached 60 s
    /video?path= streams one mp4, restricted to the WAM prefixes only
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")

app = modal.App("wam-viewer", image=image)

outputs_volume = modal.Volume.from_name("egoverse-training-outputs")

MOUNT_PATH = "/data"
INDEX_TTL_S = 60

WAM_RUN_LABEL = "data_div_oss_wam22_dw48_v2"
WAM_RUN_DIR = "data_div_oss/wam22_dw48_v2"
TRAINING_CAP = 8  # pairs shown per epoch (each epoch really has ~100)

# Offline single-checkpoint passes: (key, volume prefix, label, note).
OFFLINE_PASSES = [
    (
        "gateA",
        "wam_gateA",
        "Fixed offline pass",
        "fixed pipeline · full episode · 5 fps · TF rolling",
    ),
]

# Per-checkpoint sweep prefixes — scanned if/when they show up on the volume.
SWEEP_PREFIXES = [("wam_val_sweep", "Offline sweep (fixed)")]

# /video may only serve paths under these prefixes. Anything else on the volume
# (op_scaling/, pksnack_*/, other data_div_oss/ runs, ...) is rejected. Note the
# training entry is scoped to the WAM run's videos/ subtree, NOT all of
# data_div_oss/, so sibling runs stay unreachable.
ALLOWED_PATH_PREFIXES = tuple(f"{p}/" for _, p, _, _ in OFFLINE_PASSES) + tuple(
    f"{p}/" for p, _ in SWEEP_PREFIXES
) + (f"{WAM_RUN_DIR}/videos/",)

# Generation thresholds (frames, fps) measured from the files themselves.
FULL_EPISODE_MIN_FRAMES = 500
PREFIX_FPS = 10  # the pre-fix writer used 10 fps; the fix writes 5

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WAM val videos — wam22_dw48_v2</title>
<style>
  :root {
    --bg: #101216; --panel: #181b21; --border: #2a2e37;
    --text: #e6e8ec; --dim: #8b919d; --accent: #6ea8fe;
    --good: #7fd1a4; --warn: #e0a04d; --pred: #e08a8a; --gt: #7fb3d1;
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
  .blurb { color: var(--dim); font-size: 12px; margin-top: 7px; line-height: 1.65; }
  .blurb b { color: #b9c0cc; font-weight: 600; }
  .blurb code {
    color: var(--text); background: var(--panel); padding: 1px 5px; border-radius: 4px;
  }
  select, button {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 10px; font-size: 13px; cursor: pointer;
  }
  select:hover, button:hover { border-color: var(--accent); }
  button.on { border-color: var(--accent); color: var(--accent); }
  button:disabled { opacity: .45; cursor: not-allowed; }
  .spacer { flex: 1; }
  #status { color: var(--dim); font-size: 12px; }
  main { padding: 12px 16px 40px; }
  section {
    border: 1px solid var(--border); border-radius: 12px;
    margin-bottom: 16px; padding: 10px 12px 6px; background: #14171c;
  }
  section.featured { border-color: var(--good); background: #141a17; }
  .sec-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .sec-title { font-size: 14px; font-weight: 600; }
  section.featured .sec-title { color: var(--good); }
  .sec-note { color: var(--dim); font-size: 12px; }
  .sec-sub { color: var(--dim); font-size: 12px; margin: 4px 0 8px; }
  .badge {
    font-size: 11px; padding: 2px 7px; border-radius: 999px;
    border: 1px solid var(--border); color: var(--dim); white-space: nowrap;
  }
  .badge.post { border-color: var(--good); color: var(--good); }
  .badge.pre { border-color: var(--warn); color: var(--warn); }
  .row {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    margin-bottom: 8px; padding: 8px 10px; border-left: 3px solid var(--border);
  }
  .row.pred { border-left-color: var(--pred); }
  .row.val { border-left-color: var(--gt); }
  .row-head { display: flex; align-items: baseline; gap: 8px; margin-bottom: 7px; }
  .row-label { font-size: 13px; font-weight: 600; }
  .row.pred .row-label { color: var(--pred); }
  .row.val .row-label { color: var(--gt); }
  .row-sub { color: var(--dim); font-size: 11px; }
  .row-note { color: var(--dim); font-size: 11px; margin-left: auto; }
  .vids { display: flex; gap: 10px; overflow-x: auto; padding-bottom: 4px; }
  .vid-card { flex: 0 0 auto; width: 380px; }
  .vid-card.wide { width: 620px; }
  .vid-card video { width: 100%; border-radius: 6px; background: #000; display: block; }
  .vid-card.active-col video { outline: 2px solid var(--accent); }
  .vid-cap { color: var(--dim); font-size: 11px; margin-top: 3px; text-align: center; }
  .placeholder {
    color: var(--dim); border: 1px dashed var(--border); border-radius: 6px;
    padding: 22px 16px; text-align: center; font-size: 12px; width: 100%;
  }
  .col-ph { flex: 0 0 auto; width: 380px; }
  .epctl { display: flex; align-items: center; gap: 6px; margin-left: auto; }
  .epctl .lbl { color: var(--dim); font-size: 12px; }
</style>
</head>
<body>
<header>
  <div class="hrow">
    <h1>WAM — world-action model val videos</h1>
    <button id="playall">Play all</button>
    <button id="pauseall">Pause all</button>
    <button id="colmode" title="load and play one video column at a time">Column mode</button>
    <span id="colctl" style="display:none">
      <button id="colprev">&#9664; col</button>
      <span class="lbl" id="colind" style="color:var(--dim);font-size:12px">Column 1</span>
      <button id="colnext">col &#9654;</button>
    </span>
    <span class="spacer"></span>
    <span id="status">loading…</span>
  </div>
  <div class="blurb" id="blurb"></div>
</header>
<main id="main"></main>
<script>
(function () {
  const $ = (id) => document.getElementById(id);
  const mainEl = $("main"), statusEl = $("status");
  let index = null;
  let trainEpoch = null, sweepEpoch = null;
  let columnMode = false, currentCol = 0, playToken = 0;

  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        const v = e.target;
        if (!v.src && v.dataset.src) { v.src = v.dataset.src; v.preload = "metadata"; }
        observer.unobserve(v);
      }
    }
  }, { rootMargin: "200px" });

  function fmtProbe(pr) {
    if (!pr || !pr.frames) return "";
    const d = pr.duration ? pr.duration.toFixed(1) + " s" : "?";
    return pr.frames + " f @ " + (pr.fps || "?") + " fps · " + d;
  }

  function badge(pr) {
    if (!pr || !pr.generation) return "";
    const cls = pr.generation === "post-fix" ? "post"
              : pr.generation === "pre-fix" ? "pre" : "";
    const txt = pr.generation + (fmtProbe(pr) ? " · " + fmtProbe(pr) : "");
    return '<span class="badge ' + cls + '">' + txt + "</span>";
  }

  function vidCard(p, wide) {
    const card = document.createElement("div");
    card.className = "vid-card" + (wide ? " wide" : "");
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
    return card;
  }

  // kind: "pred" | "val"
  function makeRow(kind, sub, videos, emptyMsg, note, wide) {
    const row = document.createElement("div");
    row.className = "row " + kind;
    const head = document.createElement("div");
    head.className = "row-head";
    const label = kind === "pred"
      ? "predicted — dream frames (model's imagined future)"
      : "validation — ground-truth clip";
    head.innerHTML =
      '<span class="row-label">' + label + "</span>" +
      '<span class="row-sub">' + sub + "</span>" +
      (note ? '<span class="row-note">' + note + "</span>" : "");
    row.appendChild(head);
    const body = document.createElement("div");
    body.className = "vids";
    if (!videos || !videos.length) {
      const ph = document.createElement("div");
      ph.className = "placeholder";
      ph.textContent = emptyMsg;
      body.appendChild(ph);
    } else {
      videos.forEach((p, i) => {
        const c = vidCard(p, wide);
        c.dataset.col = i;
        body.appendChild(c);
      });
    }
    row.appendChild(body);
    return row;
  }

  function section(opts) {
    const sec = document.createElement("section");
    if (opts.featured) sec.className = "featured";
    const head = document.createElement("div");
    head.className = "sec-head";
    head.innerHTML =
      '<span class="sec-title">' + opts.title + "</span>" +
      (opts.note ? '<span class="sec-note">' + opts.note + "</span>" : "") +
      (opts.badge || "");
    if (opts.ctl) head.appendChild(opts.ctl);
    sec.appendChild(head);
    if (opts.sub) {
      const s = document.createElement("div");
      s.className = "sec-sub";
      s.innerHTML = opts.sub;
      sec.appendChild(s);
    }
    for (const r of opts.rows) sec.appendChild(r);
    return sec;
  }

  function epochCtl(label, epochs, value, onChange) {
    const box = document.createElement("span");
    box.className = "epctl";
    const lbl = document.createElement("span");
    lbl.className = "lbl"; lbl.textContent = label;
    const prev = document.createElement("button"); prev.textContent = "◀";
    const sel = document.createElement("select");
    const next = document.createElement("button"); next.textContent = "▶";
    for (const e of epochs) {
      const o = document.createElement("option");
      o.value = e; o.textContent = "epoch " + e;
      sel.appendChild(o);
    }
    sel.value = value;
    sel.onchange = () => onChange(Number(sel.value));
    const step = (d) => {
      const i = epochs.indexOf(value) + d;
      if (i >= 0 && i < epochs.length) onChange(epochs[i]);
    };
    prev.onclick = () => step(-1);
    next.onclick = () => step(1);
    prev.disabled = epochs.indexOf(value) <= 0;
    next.disabled = epochs.indexOf(value) >= epochs.length - 1;
    box.append(lbl, prev, sel, next);
    return box;
  }

  function render() {
    mainEl.innerHTML = "";
    const d = index;

    // ---- 1. featured fixed offline pass(es) ----
    for (const p of d.fixed.passes) {
      const sub =
        "<code>" + p.prefix + "/" + (p.sub || "") + "</code>" +
        (p.episode ? " · episode <code>" + p.episode + "</code>" : "") +
        " · teacher-forced rolling · one full episode, compare the dream (top) " +
        "against the ground truth (bottom) over the whole 387 s.";
      const rows = [
        makeRow("pred", "epoch_0", p.predicted,
                "no predicted video in this pass", fmtProbe(p.probe), true),
        makeRow("val", "epoch_0", p.validation,
                "no validation video in this pass", fmtProbe(p.probe), true),
      ];
      mainEl.appendChild(section({
        featured: true, title: p.label, note: p.note,
        badge: badge(p.probe), sub, rows,
      }));
    }
    if (!d.fixed.passes.length) {
      mainEl.appendChild(section({
        featured: true, title: "Fixed offline pass", note: "",
        sub: "not found on the volume", rows: [],
      }));
    }

    // ---- 2. offline sweep (fixed), if it exists yet ----
    for (const sw of d.sweep) {
      if (!sw.found) {
        mainEl.appendChild(section({
          title: sw.label,
          note: "per-checkpoint sweep",
          sub: "<code>" + sw.prefix + "/</code> is not on the volume yet — this " +
               "section fills in automatically once the sweep runs (no redeploy).",
          rows: [],
        }));
        continue;
      }
      const eps = sw.epochs;
      if (sweepEpoch === null || !eps.includes(sweepEpoch)) {
        sweepEpoch = eps.length ? eps[eps.length - 1] : null;
      }
      const cur = (sw.byEpoch || {})[String(sweepEpoch)] || {};
      const pr = (sw.probe || {})[String(sweepEpoch)];
      mainEl.appendChild(section({
        title: sw.label, note: eps.length + " checkpoints",
        badge: badge(pr),
        sub: "<code>" + sw.prefix + "/</code> — one pass per checkpoint.",
        ctl: eps.length ? epochCtl("ckpt", eps, sweepEpoch,
              (e) => { sweepEpoch = e; render(); }) : null,
        rows: [
          makeRow("pred", "epoch " + sweepEpoch, cur.predicted, "no predicted videos", ""),
          makeRow("val", "epoch " + sweepEpoch, cur.validation, "no validation videos", ""),
        ],
      }));
    }

    // ---- 3. live training-time val ----
    const tr = d.training;
    const eps = tr.epochs;
    if (trainEpoch === null || !eps.includes(trainEpoch)) {
      trainEpoch = eps.length ? eps[eps.length - 1] : null;
    }
    const cur = (tr.byEpoch || {})[String(trainEpoch)] || {};
    const tot = (tr.totals || {})[String(trainEpoch)] || {};
    const pr = (tr.probe || {})[String(trainEpoch)];
    const genNote = pr && pr.generation === "pre-fix"
      ? " <b>Pre-fix output</b>: one ~3.2 s chunk at 10 fps with misaligned teacher " +
        "forcing — NOT comparable to the fixed pass above. The live run picks up " +
        "the fix only on a future resubmit."
      : pr && pr.generation === "post-fix"
        ? " <b>Post-fix output</b> — full-episode 5 fps, comparable to the fixed pass above."
        : "";
    mainEl.appendChild(section({
      title: "Live training-time val",
      note: "held-out operator · " + eps.length + " epochs",
      badge: badge(pr),
      sub: "<code>" + tr.run_dir + "/videos/epoch_" + trainEpoch + "/</code> — " +
           "first " + tr.cap + " of ~" + (tot.validation || "100") + " pairs." + genNote,
      ctl: eps.length ? epochCtl("epoch", eps, trainEpoch,
            (e) => { trainEpoch = e; render(); }) : null,
      rows: [
        makeRow("pred", "epoch " + trainEpoch, cur.predicted,
                "no predicted videos at this epoch",
                (cur.predicted || []).length + " of " + (tot.predicted || "?")),
        makeRow("val", "epoch " + trainEpoch, cur.validation,
                "no validation videos at this epoch",
                (cur.validation || []).length + " of " + (tot.validation || "?")),
      ],
    }));

    if (columnMode) applyColumn();
  }

  // ---------------- column mode ----------------
  function maxCols() {
    let n = 0;
    document.querySelectorAll(".vids").forEach((b) => {
      n = Math.max(n, b.querySelectorAll(".vid-card").length);
    });
    return n;
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
    document.querySelectorAll(".vids").forEach((body) => {
      const cards = body.querySelectorAll(".vid-card");
      if (!cards.length) return;
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
      } else {
        const ph = document.createElement("div");
        ph.className = "placeholder col-ph";
        ph.textContent = "no video #" + (currentCol + 1) + " in this row";
        body.appendChild(ph);
      }
    });
    $("colind").textContent = "Column " + (currentCol + 1) + " of " + maxCols();
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
    $("colmode").classList.toggle("on", on);
    $("colctl").style.display = on ? "" : "none";
    if (on) { observer.disconnect(); currentCol = 0; applyColumn(); }
    else { playToken++; render(); }
  }

  $("colmode").onclick = () => { if (index) setColumnMode(!columnMode); };
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
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const d = e.key === "ArrowLeft" ? -1 : 1;
    if (columnMode) { colStep(d); return; }
    const eps = index ? index.training.epochs : [];
    const i = eps.indexOf(trainEpoch) + d;
    if (i >= 0 && i < eps.length) { trainEpoch = eps[i]; render(); }
  });

  function setBlurb() {
    const f = index.fixed.passes[0];
    const fp = f ? fmtProbe(f.probe) : "n/a";
    const tp = index.training.probe[String(index.training.epochs[index.training.epochs.length - 1])];
    $("blurb").innerHTML =
      "Run <code>" + index.run + "</code> — <b>Wan2.2-TI2V-5B world-action model</b>, " +
      "dishwashing 48 h pool, validating on the <b>held-out operator</b> split. " +
      "<b>predicted</b> = the model's imagined future frames (its dream); " +
      "<b>validation</b> = the ground-truth clip for the same window. Both carry " +
      "action-trail overlays, so a good model matches the GT both in pixels and in " +
      "the drawn trajectory.<br>" +
      "Two generations of video exist on the volume and are labelled per section " +
      "from the actual files: <b>post-fix</b> = " + fp + " (one full episode, " +
      "aligned teacher forcing) vs <b>pre-fix</b> = " +
      (tp ? fmtProbe(tp) : "32 f @ 10 fps · 3.2 s") +
      " (one chunk, 10 fps, misaligned TF). Never read across the two as a " +
      "like-for-like comparison.";
  }

  function signature(d) {
    return JSON.stringify([
      d.fixed.passes.map((p) => [p.sub, p.predicted.length, p.validation.length]),
      d.sweep.map((s) => [s.found, s.epochs]),
      d.training.epochs,
    ]);
  }

  let lastSig = null;
  function load() {
    return fetch("api/index")
      .then((r) => r.json())
      .then((data) => {
        const sig = signature(data);
        if (sig === lastSig) return;  // nothing changed — don't disturb playback
        index = data; lastSig = sig;
        setBlurb();
        render();
        const nTr = data.training.epochs.length;
        statusEl.textContent =
          data.fixed.passes.length + " fixed pass · " + nTr +
          " training epochs · index refreshes every 60 s";
      })
      .catch((e) => { statusEl.textContent = "failed to load index: " + e; });
  }
  load();
  setInterval(load, 60000);  // the run is live; new val epochs land every 30
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
@modal.asgi_app(label="wam-viewer")
def viewer():
    import os
    import re
    import struct
    import threading
    import time

    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, HTMLResponse

    api = FastAPI()
    root_dir = os.path.realpath(MOUNT_PATH)

    _cache = {"ts": 0.0, "data": None}
    _lock = threading.Lock()
    # A given mp4 never changes once written, so probe results are cached
    # permanently keyed by (path, size) — only NEW epochs cost a read.
    _probe_cache = {}

    def _vid_sort_key(name):
        m = re.search(r"(\d+)\.mp4$", name)
        return (int(m.group(1)) if m else 1 << 30, name)

    def _probe(path, window=131072):
        """(duration_s, frames, fps) from the mp4 mvhd+stsz atoms.

        Reads only the first and last `window` bytes: torchvision/pyav writes
        moov at the END, other writers at the start, so both are searched.
        Verified against ffprobe on all three generations present.
        """
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        key = (path, size)
        if key in _probe_cache:
            return _probe_cache[key]
        try:
            with open(path, "rb") as fh:
                head = fh.read(window)
                if size > window:
                    fh.seek(max(0, size - window))
                    tail = fh.read(window)
                else:
                    tail = b""
        except OSError:
            return None

        dur = frames = None
        for buf in (tail, head):
            if dur is None:
                i = buf.find(b"mvhd")
                if i >= 0:
                    p = i + 4
                    ver = buf[p]
                    p += 4
                    try:
                        if ver == 1:
                            p += 16
                            ts = struct.unpack(">I", buf[p : p + 4])[0]
                            raw = struct.unpack(">Q", buf[p + 4 : p + 12])[0]
                        else:
                            p += 8
                            ts = struct.unpack(">I", buf[p : p + 4])[0]
                            raw = struct.unpack(">I", buf[p + 4 : p + 8])[0]
                        if ts:
                            dur = raw / ts
                    except (struct.error, IndexError):
                        pass
            if frames is None:
                j = buf.find(b"stsz")
                if j >= 0:
                    try:
                        frames = struct.unpack(">I", buf[j + 12 : j + 16])[0]
                    except struct.error:
                        pass

        fps = round(frames / dur, 3) if (frames and dur) else None
        if fps is not None and abs(fps - round(fps)) < 0.05:
            fps = int(round(fps))
        if frames is None:
            generation = "unknown"
        elif frames >= FULL_EPISODE_MIN_FRAMES and fps and fps < PREFIX_FPS:
            generation = "post-fix"
        elif fps == PREFIX_FPS:
            generation = "pre-fix"
        else:
            generation = "partial"
        out = {
            "duration": round(dur, 2) if dur else None,
            "frames": frames,
            "fps": fps,
            "size_mb": round(size / 1e6, 2),
            "generation": generation,
        }
        _probe_cache[key] = out
        return out

    def _split_families(emb_dir, rel_prefix, cap=None):
        val, pred = [], []
        try:
            names = sorted(os.listdir(emb_dir), key=_vid_sort_key)
        except OSError:
            return val, pred, 0, 0
        for f in names:
            if not f.endswith(".mp4"):
                continue
            rel = f"{rel_prefix}/{f}"
            if f.startswith("validation_video"):
                val.append(rel)
            elif f.startswith("predicted_video"):
                pred.append(rel)
        n_val, n_pred = len(val), len(pred)
        if cap is not None:
            val, pred = val[:cap], pred[:cap]
        return val, pred, n_val, n_pred

    def _epoch_dirs(videos_dir):
        """{epoch_int: dirname} for epoch_<N> and ckpt_epoch_<N> layouts."""
        out = {}
        try:
            names = os.listdir(videos_dir)
        except OSError:
            return out
        for name in names:
            m = re.match(r"(?:ckpt_)?epoch_(\d+)$", name)
            if m and os.path.isdir(os.path.join(videos_dir, name)):
                out[int(m.group(1))] = name
        return out

    def _episode_from_log(run_dir):
        """The episode hash an offline pass evaluated (from its eval log)."""
        try:
            names = [f for f in os.listdir(run_dir) if f.endswith(".log")]
        except OSError:
            return None
        pat = re.compile(r"restricted valid dataset to \d+ episodes[^\[]*\['([0-9a-f]{24})'")
        for name in names:
            try:
                with open(os.path.join(run_dir, name), errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            m = pat.search(text)
            if m:
                return m.group(1)
            m2 = re.search(r"\b([0-9a-f]{24})\b", text)
            if m2:
                return m2.group(1)
        return None

    def _scan_offline_pass(key, prefix, label, note):
        """One-shot offline passes: <prefix>/<sub>/videos/epoch_*/EMB/*.mp4."""
        base = os.path.join(root_dir, prefix)
        out = []
        if not os.path.isdir(base):
            return out
        for sub in sorted(os.listdir(base)):
            videos_dir = os.path.join(base, sub, "videos")
            if not os.path.isdir(videos_dir):
                continue
            val, pred, probe = [], [], None
            for ep, ep_name in sorted(_epoch_dirs(videos_dir).items()):
                ep_dir = os.path.join(videos_dir, ep_name)
                for emb in sorted(os.listdir(ep_dir)):
                    emb_dir = os.path.join(ep_dir, emb)
                    if not os.path.isdir(emb_dir):
                        continue
                    v, p, _, _ = _split_families(
                        emb_dir, f"{prefix}/{sub}/videos/{ep_name}/{emb}"
                    )
                    val += v
                    pred += p
                    if probe is None and (v or p):
                        probe = _probe(os.path.join(emb_dir, os.path.basename((v or p)[0])))
            if val or pred:
                out.append({
                    "key": key,
                    "prefix": prefix,
                    "sub": sub,
                    "label": label,
                    "note": note,
                    "validation": val,
                    "predicted": pred,
                    "probe": probe,
                    "episode": _episode_from_log(os.path.join(base, sub)),
                })
        return out

    def _scan_sweep(prefix, label):
        """Per-checkpoint sweep: <prefix>/<sub>/videos/(ckpt_)?epoch_<N>/EMB/."""
        base = os.path.join(root_dir, prefix)
        result = {
            "prefix": prefix, "label": label, "found": False,
            "epochs": [], "byEpoch": {}, "probe": {},
        }
        if not os.path.isdir(base):
            return result
        by_epoch, probes = {}, {}
        for sub in sorted(os.listdir(base)):
            videos_dir = os.path.join(base, sub, "videos")
            if not os.path.isdir(videos_dir):
                continue
            for ep, ep_name in sorted(_epoch_dirs(videos_dir).items()):
                ep_dir = os.path.join(videos_dir, ep_name)
                val, pred = [], []
                for emb in sorted(os.listdir(ep_dir)):
                    emb_dir = os.path.join(ep_dir, emb)
                    if not os.path.isdir(emb_dir):
                        continue
                    v, p, _, _ = _split_families(
                        emb_dir, f"{prefix}/{sub}/videos/{ep_name}/{emb}",
                        cap=TRAINING_CAP,
                    )
                    val += v
                    pred += p
                    if ep not in probes and (v or p):
                        probes[ep] = _probe(
                            os.path.join(emb_dir, os.path.basename((v or p)[0]))
                        )
                if val or pred:
                    by_epoch[ep] = {"validation": val, "predicted": pred}
        if by_epoch:
            eps = sorted(by_epoch)
            result.update({
                "found": True,
                "epochs": eps,
                "byEpoch": {str(e): by_epoch[e] for e in eps},
                "probe": {str(e): probes.get(e) for e in eps},
            })
        return result

    def _scan_training():
        videos_dir = os.path.join(root_dir, WAM_RUN_DIR, "videos")
        by_epoch, totals, probes = {}, {}, {}
        for ep, ep_name in sorted(_epoch_dirs(videos_dir).items()):
            ep_dir = os.path.join(videos_dir, ep_name)
            val, pred, n_val, n_pred = [], [], 0, 0
            try:
                embodiments = sorted(os.listdir(ep_dir))
            except OSError:
                continue
            for emb in embodiments:
                emb_dir = os.path.join(ep_dir, emb)
                if not os.path.isdir(emb_dir):
                    continue
                rel = f"{WAM_RUN_DIR}/videos/{ep_name}/{emb}"
                v, p, nv, np_ = _split_families(emb_dir, rel, cap=TRAINING_CAP)
                val += v
                pred += p
                n_val += nv
                n_pred += np_
                if ep not in probes and (v or p):
                    probes[ep] = _probe(
                        os.path.join(emb_dir, os.path.basename((v or p)[0]))
                    )
            if val or pred:
                by_epoch[ep] = {
                    "validation": val[:TRAINING_CAP],
                    "predicted": pred[:TRAINING_CAP],
                }
                totals[ep] = {"validation": n_val, "predicted": n_pred}
        eps = sorted(by_epoch)
        return {
            "run_dir": WAM_RUN_DIR,
            "epochs": eps,
            "byEpoch": {str(e): by_epoch[e] for e in eps},
            "totals": {str(e): totals[e] for e in eps},
            "probe": {str(e): probes.get(e) for e in eps},
            "cap": TRAINING_CAP,
        }

    def _scan():
        passes = []
        for key, prefix, label, note in OFFLINE_PASSES:
            passes += _scan_offline_pass(key, prefix, label, note)
        return {
            "run": WAM_RUN_LABEL,
            "fixed": {"passes": passes},
            "sweep": [_scan_sweep(p, lbl) for p, lbl in SWEEP_PREFIXES],
            "training": _scan_training(),
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
        # Resolve strictly under one of the WAM prefixes; reject traversal. The
        # allowlist is on the full path prefix (not the first segment) so the
        # training entry cannot reach sibling runs under data_div_oss/.
        if "\x00" in path or path.startswith(("/", "~")) or ".." in path.split("/"):
            raise HTTPException(status_code=400, detail="bad path")
        if not path.startswith(ALLOWED_PATH_PREFIXES):
            raise HTTPException(status_code=400, detail="bad prefix")
        full = os.path.realpath(os.path.join(root_dir, path))
        if not full.startswith(root_dir + os.sep) or not full.endswith(".mp4"):
            raise HTTPException(status_code=400, detail="bad path")
        if not os.path.isfile(full):
            raise HTTPException(status_code=404, detail="video not found")
        return FileResponse(  # starlette FileResponse handles HTTP Range → seeking
            full,
            media_type="video/mp4",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return api
