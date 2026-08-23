"""Web viewer for the WAM world-action model's validation videos.

Run under inspection: ``data_div_oss_wam22_dw48_v2`` — a Wan2.2-TI2V-5B
world-ACTION model trained on the dishwashing 48 h pool, validating on the
held-out-operator split (dishwashing_val_ophold5). Every val pass writes a
matched pair per sample:

    predicted_video_<i>.mp4   the model's IMAGINED future frames (dream), with
                              the action-trail overlay drawn on top
    validation_video_<i>.mp4  the GROUND-TRUTH clip for the same window, same
                              overlay — the reference to compare the dream to

FIVE GENERATIONS of video exist on the volume and must never be read as
like-for-like. Every number below was measured from the actual mp4 atoms by
``_probe`` (cross-checked against ffprobe), not assumed:

    5 fps source  ~320-352 f @ 5 fps = 64-72 s        TRUE 5 fps clips: the data
                  pipeline itself strides x6 (cam_horizon=17 now spans 3.2 s of
                  real time, not 0.567 s) and episode tiling steps 96 raw frames
                  so windows no longer overlap ~6x. Playback subsample x1.
                  Fixed in c1c4ff00 — the cure for the "repeating segments".
    real-time     ~323-326 f @ 5 fps = 64-65 s        full episode at the right
                  (30 fps       speed, but built from 17 CONSECUTIVE 30 fps frames
                  decimated)    (a 0.567 s burst per window) then decimated x6 at
                                write time (d272d135). Right duration, still the
                                repeating-burst content.
    6x slow       1936 f @ 5 fps = 387.2 s  ~50   MB   full episode, but 30 fps
                  frames written into a 5 fps container -> plays 6x too slow
    debug-capped    96 f @ 5 fps =  19.2 s   ~2.5 MB   right fps, length-capped
    chunk           32 f @ 10 fps =  3.2 s   ~0.4 MB   pre-fix: one chunk,
                  10 fps, misaligned teacher forcing

"5 fps source" and "real-time (30 fps decimated)" are INDISTINGUISHABLE from the
mp4 alone — both are ~320 frames at 5 fps. They are told apart by PATH PREFIX:
anything under ``wam_offline_5fps/`` came from the c1c4ff00 pipeline and is true
5 fps source; everything else with those dimensions is x6-decimated. That is why
``_classify`` takes a ``source_5fps`` flag instead of guessing from the file.

Sources surfaced:

  0. 5 FPS OFFLINE EVAL — wam_offline_5fps/<run>_<ts>/ (featured, first). Four
     logical runs, all at k2 checkpoint epoch 1019 over the 3 held-out-operator
     episodes: {tf,ar} x {vids,metrics}. The point of these four is the
     TF-vs-AR comparison, so the section renders GT / TF-predicted /
     AR-predicted side by side for one selected episode, with each mode's
     metrics. Run dirs are TIMESTAMPED and each logical run may have several
     attempts (the 03-50/03-51 launches crashed on the new real-time assertion
     and wrote no videos); the scanner groups by logical name, prefers the
     newest attempt that actually has videos, and reports crashed/running
     status from the attempt's log.
  1. PREVIOUS REFERENCE — wam_gate_a2/e989_realtime_*/. Full episode
     69b22fc5f4f4e149281a6635 at checkpoint epoch 989, teacher-forced rolling,
     right duration but x6-decimated content.
  2. SWEEP — wam_val_sweep/<sweep_id>/epoch_<N>_<ts>/videos/... , one offline
     pass per checkpoint (45 checkpoints, 4x H200). Sweep ids are DISCOVERED
     (never hardcoded), newest preferred, several tolerated. Per-epoch metrics
     come from metrics/epoch_<N>.json when written, else are parsed out of that
     epoch's eval_dreamzero.log so numbers show up while the sweep is still
     running.
  3. LIVE training-time val — data_div_oss/wam22_dw48_{v2,k2}/videos/epoch_<N>/,
     selectable per run, capped to TRAINING_CAP pairs. v2 is still "chunk"
     generation (32 f @ 10 fps); k2 is uniformly "real-time (30 fps decimated)"
     (326 f @ 5 fps) across every epoch 29..1079 — its .hydra config carries the
     same cam_horizon=17 with no stride, so it predates c1c4ff00 and is NOT
     5 fps source. Both relabel themselves per epoch if a resubmit changes the
     output.
  4. COMPARISON passes — wam_gateA (6x slow, superseded) and wam_gateC
     (debug-capped), kept unfeatured so the reference can't be confused.

Deploy:
    MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/wam_viewer.py

Routes (single ASGI web function -> one URL, relative paths):
    /            HTML viewer page
    /api/index   {"passes":…, "sweeps":…, "training":…} — volume scan, cached 60 s
    /video?path= streams one mp4, restricted to the WAM prefixes only
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]")

app = modal.App("wam-viewer", image=image)

outputs_volume = modal.Volume.from_name("egoverse-training-outputs")

MOUNT_PATH = "/data"
INDEX_TTL_S = 60

WAM_RUN_LABEL = "data_div_oss_wam22_dw48_v2 / _k2"
TRAINING_CAP = 8  # pairs shown per epoch (each epoch really has ~100)
SWEEP_CAP = 4  # pairs shown per sweep checkpoint

# Live training runs, selectable in the training section. (key, volume dir)
TRAINING_RUNS = [
    ("v2", "data_div_oss/wam22_dw48_v2"),
    ("k2", "data_div_oss/wam22_dw48_k2"),
]

# The c1c4ff00 pipeline: TRUE 5 fps source clips. Indistinguishable from the
# x6-decimated generation by file inspection alone (both ~320 f @ 5 fps), so
# membership is decided by this path prefix and nothing else.
FIVE_FPS_PREFIX = "wam_offline_5fps"
FIVE_FPS_LABEL = "5 fps offline eval (fixed pipeline)"
# Expected final length, from the verified plan line: 1945 frames @ 30 fps ->
# 20 windows x 17 (stride 96) -> 320 frames @ 5 fps, subsampled x1.
FIVE_FPS_EXPECTED_FRAMES = 320

# Offline single-checkpoint passes, in display order.
# (key, prefix, title, note, featured)
OFFLINE_PASSES = [
    (
        "gate_a2",
        "wam_gate_a2",
        "Reference pass",
        "fixed · full episode · real-time 5 fps (subsampled ×6)",
        True,
    ),
    (
        "gateA",
        "wam_gateA",
        "Superseded pass",
        "superseded · full episode but 6× slow — 30 fps frames in a 5 fps container",
        False,
    ),
    (
        "gateC",
        "wam_gateC",
        "Debug train-path check",
        "debug-capped · correct 5 fps, length-capped by the debug config",
        False,
    ),
]

SWEEP_PREFIX = "wam_val_sweep"
SWEEP_LABEL = "Offline sweep (fixed) — one pass per checkpoint"

# /video may only serve paths under these prefixes. Anything else on the volume
# (op_scaling/, pksnack_*/, other data_div_oss/ runs, checkpoints, ...) is
# rejected. The training entry is scoped to the WAM run's videos/ subtree, NOT
# all of data_div_oss/, so sibling runs stay unreachable.
ALLOWED_PATH_PREFIXES = (
    tuple(f"{p}/" for _, p, _, _, _ in OFFLINE_PASSES)
    + (f"{SWEEP_PREFIX}/", f"{FIVE_FPS_PREFIX}/")
    + tuple(f"{d}/videos/" for _, d in TRAINING_RUNS)
)

# Generation classification thresholds, applied to (frames, fps) measured from
# the file. The frames split between "6x slow" and "real-time" assumes episodes
# of at most a few minutes: at real-time 5 fps a 3-minute episode is 900 frames,
# while the 6x-slow generation of a 1-minute episode is already ~1936.
CHUNK_FPS_MIN = 8  # the pre-fix writer used 10 fps; every fixed one uses 5
SLOW_MIN_FRAMES = 1000
REALTIME_MIN_FRAMES = 200

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
    --good: #7fd1a4; --warn: #e0a04d; --bad: #d98b8b; --neutral: #9aa3b2;
    --pred: #e08a8a; --gt: #7fb3d1;
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
  code {
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
  section.muted { opacity: .92; background: #131519; }
  .sec-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .sec-title { font-size: 14px; font-weight: 600; }
  section.featured .sec-title { color: var(--good); }
  section.muted .sec-title { color: var(--neutral); }
  .sec-note { color: var(--dim); font-size: 12px; }
  .sec-sub { color: var(--dim); font-size: 12px; margin: 4px 0 8px; line-height: 1.6; }
  .metrics { color: var(--accent); font-size: 12px; }
  .badge {
    font-size: 11px; padding: 2px 7px; border-radius: 999px;
    border: 1px solid var(--border); color: var(--dim); white-space: nowrap;
  }
  .badge.gen-5fps { border-color: #6ee7d0; color: #6ee7d0; }
  .badge.gen-realtime { border-color: var(--good); color: var(--good); }
  .badge.status-crashed { border-color: var(--bad); color: var(--bad); }
  .badge.status-running { border-color: var(--warn); color: var(--warn); }
  .badge.status-done { border-color: var(--good); color: var(--good); }
  .cmp { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 4px; }
  .cmp-col { flex: 0 0 auto; width: 400px; }
  .cmp-col h4 {
    margin: 0 0 5px; font-size: 12px; font-weight: 600; display: flex;
    align-items: baseline; gap: 6px;
  }
  .cmp-col.gt h4 { color: var(--gt); }
  .cmp-col.tf h4 { color: #a9d97f; }
  .cmp-col.ar h4 { color: var(--pred); }
  .cmp-col .met { color: var(--accent); font-size: 11px; font-weight: 400; }
  .cmp-col video { width: 100%; border-radius: 6px; background: #000; display: block; }
  .cmp-col .cap { color: var(--dim); font-size: 11px; margin-top: 3px; text-align: center; }
  .runtable { width: 100%; border-collapse: collapse; font-size: 11.5px; margin-top: 6px; }
  .runtable th, .runtable td {
    text-align: left; padding: 3px 8px 3px 0; color: var(--dim);
    border-bottom: 1px solid #21252c; white-space: nowrap;
  }
  .runtable th { color: #9aa3b2; font-weight: 600; }
  .badge.gen-slow { border-color: var(--bad); color: var(--bad); }
  .badge.gen-chunk { border-color: var(--warn); color: var(--warn); }
  .badge.gen-capped { border-color: var(--neutral); color: var(--neutral); }
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
      <span id="colind" style="color:var(--dim);font-size:12px">Column 1</span>
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
  let trainEpoch = null, sweepEpoch = null, sweepId = null;
  let trainRun = null, fiveEp = null;
  let columnMode = false, currentCol = 0, playToken = 0;
  const SWEEP_TITLE = "Offline sweep (fixed)";
  const FIVE_FPS_TITLE = "5 fps offline eval (fixed pipeline)";

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
    const d = pr.duration != null ? pr.duration.toFixed(1) + " s" : "?";
    return pr.frames + " f @ " + (pr.fps != null ? pr.fps : "?") + " fps · " + d;
  }

  const GEN_CLASS = {
    "5 fps source": "gen-5fps",
    "real-time (30 fps decimated)": "gen-realtime",
    "6× slow": "gen-slow",
    "chunk (pre-fix)": "gen-chunk",
    "debug-capped": "gen-capped",
  };

  function badge(pr) {
    if (!pr || !pr.generation) return "";
    const g = pr.generation;
    const p = fmtProbe(pr);
    return '<span class="badge ' + (GEN_CLASS[g] || "") + '">' + g +
           (p ? " · " + p : "") + "</span>";
  }

  function statusBadge(st) {
    if (!st) return "";
    return '<span class="badge status-' + st + '">' + st + "</span>";
  }

  function fmtMetrics(m) {
    if (!m) return "";
    const order = Object.keys(m).sort((a, b) => {
      const rank = (k) => (/paired_mse/.test(k) ? 0 : /Loss$/.test(k) ? 1 : 2);
      return rank(a) - rank(b) || a.localeCompare(b);
    });
    return order.slice(0, 3)
      .map((k) => k.replace(/^Valid\//, "") + " = " + Number(m[k]).toFixed(4))
      .join("   ");
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
    else if (opts.muted) sec.className = "muted";
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
    for (const r of opts.rows || []) sec.appendChild(r);
    return sec;
  }

  function selector(label, values, value, fmt, onChange) {
    const box = document.createElement("span");
    box.className = "epctl";
    const lbl = document.createElement("span");
    lbl.className = "lbl"; lbl.textContent = label;
    const prev = document.createElement("button"); prev.textContent = "◀";
    const sel = document.createElement("select");
    const next = document.createElement("button"); next.textContent = "▶";
    for (const v of values) {
      const o = document.createElement("option");
      o.value = v; o.textContent = fmt ? fmt(v) : v;
      sel.appendChild(o);
    }
    sel.value = value;
    sel.onchange = () => onChange(sel.value);
    const step = (d) => {
      const i = values.indexOf(value) + d;
      if (i >= 0 && i < values.length) onChange(values[i]);
    };
    prev.onclick = () => step(-1);
    next.onclick = () => step(1);
    prev.disabled = values.indexOf(value) <= 0;
    next.disabled = values.indexOf(value) >= values.length - 1;
    box.append(lbl, prev, sel, next);
    return box;
  }

  function renderPass(p) {
    const wide = !!p.featured;
    const sub =
      "<code>" + p.prefix + "/" + (p.sub || "") + "</code>" +
      (p.episode ? " · episode <code>" + p.episode + "</code>" : "") +
      " · " + (p.featured
        ? "teacher-forced rolling over one full episode at real speed — compare the " +
          "dream against the ground truth over the whole " +
          (p.probe && p.probe.duration ? p.probe.duration.toFixed(1) + " s" : "clip") + "."
        : "kept for comparison only — <b>not</b> the reference.");
    return section({
      featured: p.featured, muted: !p.featured,
      title: p.title, note: p.note, badge: badge(p.probe), sub,
      rows: [
        makeRow("pred", p.epoch_label || "epoch_0", p.predicted,
                "no predicted video in this pass", fmtProbe(p.probe), wide),
        makeRow("val", p.epoch_label || "epoch_0", p.validation,
                "no validation video in this pass", fmtProbe(p.probe), wide),
      ],
    });
  }

  // ---- 5 fps offline eval: TF vs AR side by side, one episode at a time ----
  function pickRun(runs, mode) {
    // prefer the *_vids variant, fall back to *_metrics; prefer one with videos
    const cands = runs.filter((r) => r.mode === mode);
    const withVids = cands.filter((r) => (r.predicted || []).length);
    const pool = withVids.length ? withVids : cands;
    return pool.find((r) => r.variant === "vids") || pool[0] || null;
  }

  function cmpCol(kind, title, path, met, epLabel) {
    const col = document.createElement("div");
    col.className = "cmp-col " + kind;
    const h = document.createElement("h4");
    h.innerHTML = title + (met ? '<span class="met">' + met + "</span>" : "");
    col.appendChild(h);
    if (path) {
      const v = document.createElement("video");
      v.controls = true; v.muted = true; v.loop = true;
      v.playsInline = true; v.preload = "none";
      v.dataset.src = "video?path=" + encodeURIComponent(path);
      v.title = path;
      if (!columnMode) observer.observe(v);
      col.appendChild(v);
      const cap = document.createElement("div");
      cap.className = "cap";
      cap.textContent = path.split("/").pop() + " · " + epLabel;
      col.appendChild(cap);
    } else {
      const ph = document.createElement("div");
      ph.className = "placeholder";
      ph.textContent = "still running — no video for this episode yet";
      col.appendChild(ph);
    }
    return col;
  }

  function renderFiveFps() {
    const f = index.five_fps;
    if (!f || !f.runs.length) {
      return section({
        featured: true, title: FIVE_FPS_TITLE,
        sub: "<code>" + (f ? f.prefix : "wam_offline_5fps") + "/</code> is not on " +
             "the volume yet — this section fills in automatically (no redeploy).",
      });
    }
    const tf = pickRun(f.runs, "tf"), ar = pickRun(f.runs, "ar");
    const nEps = Math.max(
      tf ? (tf.predicted || []).length : 0, ar ? (ar.predicted || []).length : 0,
      tf ? (tf.validation || []).length : 0, ar ? (ar.validation || []).length : 0);
    const epList = [];
    for (let i = 0; i < Math.max(nEps, 1); i++) epList.push(i);
    if (fiveEp === null || !epList.includes(fiveEp)) fiveEp = 0;

    const hashes = (tf && tf.episodes.length ? tf.episodes
                 : (ar && ar.episodes) || []);
    const epLabel = (i) => "episode " + i + (hashes[i] ? " · " + hashes[i].slice(0, 8) : "");

    const cmp = document.createElement("div");
    cmp.className = "cmp";
    const gtPath = (tf && (tf.validation || [])[fiveEp]) ||
                   (ar && (ar.validation || [])[fiveEp]) || null;
    cmp.appendChild(cmpCol("gt", "validation — ground truth", gtPath, "", epLabel(fiveEp)));
    cmp.appendChild(cmpCol("tf", "predicted — teacher-forced (TF)",
      tf ? (tf.predicted || [])[fiveEp] : null,
      tf ? fmtMetrics(tf.metrics) : "", epLabel(fiveEp)));
    cmp.appendChild(cmpCol("ar", "predicted — autoregressive (AR)",
      ar ? (ar.predicted || [])[fiveEp] : null,
      ar ? fmtMetrics(ar.metrics) : "", epLabel(fiveEp)));

    // status table over all four logical runs
    const table = document.createElement("table");
    table.className = "runtable";
    table.innerHTML =
      "<tr><th>run</th><th>mode</th><th>status</th><th>pairs</th>" +
      "<th>video</th><th>metrics</th><th>attempts</th></tr>" +
      f.runs.map((r) => {
        const pr = r.probe;
        return "<tr><td>" + r.logical + "</td><td>" + r.mode.toUpperCase() +
          "</td><td>" + r.status + "</td><td>" +
          Math.min((r.predicted || []).length, (r.validation || []).length) +
          "</td><td>" + (pr ? fmtProbe(pr) : "—") + "</td><td>" +
          (r.metrics ? fmtMetrics(r.metrics) : "—") + "</td><td>" +
          r.n_attempts + (r.crashed_attempts.length
            ? " (" + r.crashed_attempts.length + " crashed)" : "") +
          "</td></tr>";
      }).join("");

    const probe = (tf && tf.probe) || (ar && ar.probe);
    const short = probe && probe.frames && probe.frames < f.expected_frames * 0.8;
    const allDone = f.runs.every((r) => r.status === "done");
    let sub =
      "<code>" + f.prefix + "/</code> · k2 checkpoint <b>epoch 1019</b>, 3 held-out-" +
      "operator episodes. <b>TRUE 5 fps clips</b>: the data pipeline strides ×6 so " +
      "cam_horizon=17 spans 3.2 s of real time instead of a 0.567 s burst, and " +
      "episode tiling steps 96 raw frames so windows no longer overlap ~6× — this is " +
      "the fix for the repeating segments. Compare the same episode across " +
      "<b>teacher-forced</b> (re-anchored to GT every chunk) and " +
      "<b>autoregressive</b> (free-running, drift accumulates).";
    if (short && allDone) {
      sub += "<br><b>Length caveat:</b> these passes report <i>done</i>, yet each " +
        "episode video is only " + probe.frames + " f (" +
        probe.duration.toFixed(1) + " s) against the ~" + f.expected_frames +
        " f (~64–72 s) the plan predicts — so the full-episode tiling is not " +
        "landing in the mp4s even though the fps is right. Treat the clips as a " +
        "short excerpt per episode, not a whole episode.";
    } else if (short) {
      sub += "<br><b>Still running</b> — expected ~" + f.expected_frames +
        " frames (~64–72 s) per episode, currently " + probe.frames + " f (" +
        probe.duration.toFixed(1) + " s), so lengths and metrics are provisional.";
    }
    if (!f.runs.some((r) => r.metrics)) {
      sub += "<br>No <code>metrics.json</code> written yet and the eval logs have not " +
        "flushed their metric lines, so no numbers are available for either mode yet.";
    } else {
      sub += "<br>Metrics are parsed from each pass's <code>eval_dreamzero.log</code> " +
        "(no <code>metrics.json</code> was written). Only " +
        "<code>paired_mse_avg</code> separates the two modes — the Valid/*_loss " +
        "figures are computed teacher-forced in both and come out identical.";
    }

    return section({
      featured: true, title: FIVE_FPS_TITLE,
      note: "TF vs AR · " + f.runs.length + " runs",
      badge: badge(probe) + (tf ? statusBadge(tf.status) : ""),
      sub,
      ctl: epList.length > 1
        ? selector("episode", epList, fiveEp, epLabel, (v) => { fiveEp = Number(v); render(); })
        : null,
      rows: [cmp, table],
    });
  }

  function renderSweep() {
    const sweeps = index.sweeps;
    if (!sweeps.length) {
      return section({
        title: "Offline sweep (fixed)", note: "per-checkpoint sweep",
        sub: "<code>" + index.sweep_prefix + "/</code> has no sweep dirs yet — this " +
             "section fills in automatically as the sweep writes them (no redeploy).",
      });
    }
    if (sweepId === null || !sweeps.some((s) => s.id === sweepId)) {
      sweepId = sweeps[0].id;  // newest first
    }
    const sw = sweeps.find((s) => s.id === sweepId);
    const eps = sw.epochs;
    if (sweepEpoch === null || !eps.includes(sweepEpoch)) {
      sweepEpoch = eps.length ? eps[eps.length - 1] : null;
    }
    const cur = (sw.byEpoch || {})[String(sweepEpoch)] || {};
    const pr = (sw.probe || {})[String(sweepEpoch)];
    const met = (sw.metrics || {})[String(sweepEpoch)];
    const ctlWrap = document.createElement("span");
    ctlWrap.className = "epctl";
    if (sweeps.length > 1) {
      ctlWrap.appendChild(selector("sweep", sweeps.map((s) => s.id), sweepId, null,
        (v) => { sweepId = v; sweepEpoch = null; render(); }));
    }
    if (eps.length) {
      ctlWrap.appendChild(selector("ckpt", eps, sweepEpoch,
        (e) => "epoch " + e + ((sw.metrics || {})[String(e)] ? " ✓" : ""),
        (v) => { sweepEpoch = Number(v); render(); }));
    }
    const done = Object.keys(sw.byEpoch || {}).length;
    let sub =
      "<code>" + index.sweep_prefix + "/" + sw.id + "/</code> — " + done +
      " of " + sw.n_dirs + " checkpoint dirs have videos" +
      (sw.n_expected ? " (sweep covers ~" + sw.n_expected + " checkpoints)" : "") +
      ". Real-time generation (post-fix). Videos capped to " + sw.cap + " pairs.";
    if (met) sub += '<br><span class="metrics">' + fmtMetrics(met) + "</span>";
    else if (eps.length) sub += "<br>no metrics for this checkpoint yet.";
    return section({
      title: SWEEP_TITLE, note: eps.length + " checkpoints with video",
      badge: badge(pr), sub, ctl: ctlWrap,
      rows: [
        makeRow("pred", "epoch " + sweepEpoch, cur.predicted,
                eps.length ? "no predicted videos at this checkpoint"
                           : "sweep is running — no videos committed yet",
                fmtProbe(pr)),
        makeRow("val", "epoch " + sweepEpoch, cur.validation,
                eps.length ? "no validation videos at this checkpoint"
                           : "sweep is running — no videos committed yet",
                fmtProbe(pr)),
      ],
    });
  }

  function renderTraining() {
    const runs = index.training_runs;
    if (trainRun === null || !runs.some((r) => r.key === trainRun)) {
      // default to the run with the most recent epoch
      trainRun = runs.slice().sort((a, b) =>
        (a.epochs[a.epochs.length - 1] || 0) - (b.epochs[b.epochs.length - 1] || 0)
      ).pop().key;
    }
    const tr = runs.find((r) => r.key === trainRun);
    const eps = tr.epochs;
    if (trainEpoch === null || !eps.includes(trainEpoch)) {
      trainEpoch = eps.length ? eps[eps.length - 1] : null;
    }
    const cur = (tr.byEpoch || {})[String(trainEpoch)] || {};
    const tot = (tr.totals || {})[String(trainEpoch)] || {};
    const pr = (tr.probe || {})[String(trainEpoch)];
    let genNote = "";
    if (pr && pr.generation === "chunk (pre-fix)") {
      genNote = " <b>Pre-fix output</b>: one ~3.2 s chunk at 10 fps with misaligned " +
        "teacher forcing — NOT comparable to the reference above. The live run picks " +
        "up both fixes only on its next resubmit.";
    } else if (pr && pr.generation === "real-time (30 fps decimated)") {
      genNote = " <b>Right duration, decimated content</b> — full episode at real " +
        "speed, but each window is still 17 consecutive 30 fps frames decimated ×6, " +
        "so it shows the repeating-burst behaviour rather than true 5 fps clips. " +
        "Not the same generation as the 5 fps section at the top.";
    } else if (pr && pr.generation === "5 fps source") {
      genNote = " <b>True 5 fps source</b> — comparable to the 5 fps section above.";
    } else if (pr && pr.generation === "6× slow") {
      genNote = " <b>6× slow output</b> — full episode but played 6× too slow.";
    }
    const genCounts = tr.gen_counts || {};
    const mix = Object.keys(genCounts).length > 1
      ? "<br>Epoch mix on the volume: " +
        Object.entries(genCounts).map(([g, n]) => n + " × " + g).join(", ") + "."
      : "";
    const ctlWrap = document.createElement("span");
    ctlWrap.className = "epctl";
    if (runs.length > 1) {
      ctlWrap.appendChild(selector("run", runs.map((r) => r.key), trainRun,
        (k) => k + " (" + (runs.find((r) => r.key === k).epochs.length) + " ep)",
        (v) => { trainRun = v; trainEpoch = null; render(); }));
    }
    if (eps.length) {
      ctlWrap.appendChild(selector("epoch", eps, trainEpoch, (e) => "epoch " + e,
        (v) => { trainEpoch = Number(v); render(); }));
    }
    return section({
      title: "Live training-time val",
      note: "held-out operator · " + tr.key + " · " + eps.length + " epochs",
      badge: badge(pr),
      sub: "<code>" + tr.run_dir + "/videos/epoch_" + trainEpoch + "/</code> — first " +
           tr.cap + " of ~" + (tot.validation || "100") + " pairs." + genNote + mix,
      ctl: ctlWrap,
      rows: [
        makeRow("pred", "epoch " + trainEpoch, cur.predicted,
                "no predicted videos at this epoch",
                (cur.predicted || []).length + " of " + (tot.predicted || "?")),
        makeRow("val", "epoch " + trainEpoch, cur.validation,
                "no validation videos at this epoch",
                (cur.validation || []).length + " of " + (tot.validation || "?")),
      ],
    });
  }

  function render() {
    mainEl.innerHTML = "";
    mainEl.appendChild(renderFiveFps());
    for (const p of index.passes.filter((x) => x.featured)) {
      mainEl.appendChild(renderPass(p));
    }
    mainEl.appendChild(renderSweep());
    mainEl.appendChild(renderTraining());
    const others = index.passes.filter((x) => !x.featured);
    for (const p of others) mainEl.appendChild(renderPass(p));
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
    const run = index && index.training_runs.find((r) => r.key === trainRun);
    const eps = run ? run.epochs : [];
    const i = eps.indexOf(trainEpoch) + d;
    if (i >= 0 && i < eps.length) { trainEpoch = eps[i]; render(); }
  });

  function setBlurb() {
    const ref = index.passes.find((p) => p.key === "gate_a2");
    const slow = index.passes.find((p) => p.key === "gateA");
    const v2 = index.training_runs.find((r) => r.key === "v2");
    const tp = v2 && v2.epochs.length
      ? v2.probe[String(v2.epochs[v2.epochs.length - 1])] : null;
    const f5 = index.five_fps.runs.map((r) => r.probe).find((p) => p);
    $("blurb").innerHTML =
      "Run <code>" + index.run + "</code> — <b>Wan2.2-TI2V-5B world-action model</b>, " +
      "dishwashing 48 h pool, validating on the <b>held-out operator</b> split. " +
      "<b>predicted</b> = the model's imagined future frames (its dream); " +
      "<b>validation</b> = the ground-truth clip for the same window. Both carry " +
      "action-trail overlays, so a good model matches the GT in pixels and in the " +
      "drawn trajectory.<br>" +
      "<b>Five generations</b> exist on the volume; each section is labelled from its " +
      "own files. <b>5 fps source</b> = " +
      (f5 ? fmtProbe(f5) : "~320 f @ 5 fps · ~64 s") +
      " — true 5 fps clips (pipeline strides ×6; the repeating-segments fix) · " +
      "<b>real-time (30 fps decimated)</b> = " +
      (ref && ref.probe ? fmtProbe(ref.probe) : "~323 f @ 5 fps · 64.6 s") +
      " — right duration, but 0.567 s bursts decimated ×6 · <b>6× slow</b> = " +
      (slow && slow.probe ? fmtProbe(slow.probe) : "1936 f @ 5 fps · 387.2 s") +
      " · <b>chunk (pre-fix)</b> = " +
      (tp ? fmtProbe(tp) : "32 f @ 10 fps · 3.2 s") +
      " · <b>debug-capped</b> = right fps, capped length. The first two are " +
      "byte-identical in shape and are told apart only by path prefix " +
      "(<code>wam_offline_5fps/</code>), never by the file. Never read across " +
      "generations as like-for-like.";
  }

  function signature(d) {
    return JSON.stringify([
      d.passes.map((p) => [p.key, p.sub, p.predicted.length, p.validation.length]),
      d.sweeps.map((s) => [s.id, s.epochs, Object.keys(s.metrics || {}).length]),
      d.training_runs.map((r) => [r.key, r.epochs, r.gen_counts]),
      d.five_fps.runs.map((r) => [
        r.logical, r.dir, r.status, r.predicted.length, r.validation.length,
        r.probe ? r.probe.frames : null, r.metrics ? Object.keys(r.metrics).length : 0,
      ]),
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
        const sw = data.sweeps[0];
        const f5 = data.five_fps.runs;
        const f5live = f5.filter((r) => (r.predicted || []).length).length;
        statusEl.textContent =
          "5 fps: " + f5live + "/" + f5.length + " runs with video · sweep " +
          (sw ? Object.keys(sw.byEpoch || {}).length + "/" + (sw.n_expected || sw.n_dirs) : "none") +
          " · training " +
          data.training_runs.map((r) => r.key + ":" + r.epochs.length).join(" ") +
          " · refreshes every 60 s";
      })
      .catch((e) => { statusEl.textContent = "failed to load index: " + e; });
  }
  load();
  setInterval(load, 60000);  // sweep is running; live run commits every 30 epochs
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
    import json
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
    # An mp4/log/metrics file never changes once written, so results are cached
    # permanently keyed by (path, size) — only NEW files ever cost a read.
    _probe_cache = {}
    _metrics_cache = {}

    def _vid_sort_key(name):
        m = re.search(r"(\d+)\.mp4$", name)
        return (int(m.group(1)) if m else 1 << 30, name)

    def _classify(frames, fps, source_5fps=False):
        """Generation vocabulary.

        5 fps source / real-time (30 fps decimated) / 6x slow / debug-capped /
        chunk (pre-fix). ``source_5fps`` comes from the PATH (the
        wam_offline_5fps/ prefix) because a true-5 fps clip and an x6-decimated
        one are byte-indistinguishable at ~320 f @ 5 fps.
        """
        if not frames or not fps:
            return "unknown"
        if fps >= CHUNK_FPS_MIN:
            return "chunk (pre-fix)"
        if source_5fps:
            # Length is not a reliability signal here: these passes write one
            # episode at a time while running, so short is "still going".
            return "5 fps source"
        if frames >= SLOW_MIN_FRAMES:
            return "6× slow"
        if frames >= REALTIME_MIN_FRAMES:
            return "real-time (30 fps decimated)"
        return "debug-capped"

    def _probe(path, window=131072, source_5fps=False):
        """(duration, frames, fps, generation) from the mp4 mvhd+stsz atoms.

        Reads only the first and last `window` bytes: torchvision/pyav writes
        moov at the END, other writers at the start, so both are searched.
        Verified against ffprobe on every generation present on the volume.
        """
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        key = (path, size, source_5fps)
        if key in _probe_cache:
            return _probe_cache[key]
        try:
            with open(path, "rb") as fh:
                head = fh.read(window)
                tail = b""
                if size > window:
                    fh.seek(max(0, size - window))
                    tail = fh.read(window)
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

        fps = round(frames / dur, 2) if (frames and dur) else None
        if fps is not None and abs(fps - round(fps)) < 0.05:
            fps = int(round(fps))
        out = {
            "duration": round(dur, 2) if dur else None,
            "frames": frames,
            "fps": fps,
            "size_mb": round(size / 1e6, 2),
            "generation": _classify(frames, fps, source_5fps),
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

    def _collect_videos(videos_dir, rel_videos, cap=None, source_5fps=False):
        """Walk a `videos/` tree (videos/<epoch>/<EMB>/) collecting both families."""
        val, pred, n_val, n_pred, probe = [], [], 0, 0, None
        if not os.path.isdir(videos_dir):
            return val, pred, n_val, n_pred, probe, None
        ep_label = None
        try:
            ep_names = sorted(os.listdir(videos_dir))
        except OSError:
            return val, pred, n_val, n_pred, probe, None
        for ep_name in ep_names:
            ep_dir = os.path.join(videos_dir, ep_name)
            if not os.path.isdir(ep_dir):
                continue
            ep_label = ep_label or ep_name
            try:
                embs = sorted(os.listdir(ep_dir))
            except OSError:
                continue
            for emb in embs:
                emb_dir = os.path.join(ep_dir, emb)
                if not os.path.isdir(emb_dir):
                    continue
                v, p, nv, np_ = _split_families(
                    emb_dir, f"{rel_videos}/{ep_name}/{emb}", cap=cap
                )
                val += v
                pred += p
                n_val += nv
                n_pred += np_
                if probe is None and (v or p):
                    probe = _probe(
                        os.path.join(emb_dir, os.path.basename((v or p)[0])),
                        source_5fps=source_5fps,
                    )
        if cap is not None:
            val, pred = val[:cap], pred[:cap]
        return val, pred, n_val, n_pred, probe, ep_label

    _METRIC_RE = re.compile(r"\[eval_dreamzero\] metric (\S+) = ([-\d.eE+]+)")

    def _metrics_from_log(run_dir):
        """Valid/* metrics parsed from an eval_dreamzero.log."""
        out = {}
        try:
            names = [f for f in os.listdir(run_dir) if f.endswith(".log")]
        except OSError:
            return out
        for name in names:
            path = os.path.join(run_dir, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            key = ("log", path, size)
            if key in _metrics_cache:
                out.update(_metrics_cache[key])
                continue
            found = {}
            try:
                with open(path, errors="replace") as fh:
                    for line in fh:
                        m = _METRIC_RE.search(line)
                        if m:
                            try:
                                found[m.group(1)] = float(m.group(2))
                            except ValueError:
                                pass
            except OSError:
                continue
            _metrics_cache[key] = found
            out.update(found)
        return out

    def _flatten_metrics(obj):
        """Pull scalar Valid/* style metrics out of a (possibly nested) json."""
        out = {}
        if not isinstance(obj, dict):
            return out
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = float(v)
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                        out[
                            k2 if k in ("metrics", "callback_metrics") else f"{k}/{k2}"
                        ] = float(v2)
        return out

    def _metrics_from_json(path):
        try:
            size = os.path.getsize(path)
        except OSError:
            return {}
        key = ("json", path, size)
        if key in _metrics_cache:
            return _metrics_cache[key]
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {}
        out = _flatten_metrics(data)
        _metrics_cache[key] = out
        return out

    def _pick_metrics(m):
        """Keep the headline keys: paired_mse_avg first, then Valid/Loss."""
        if not m:
            return None
        keep = {
            k: v for k, v in m.items() if re.search(r"paired_mse|_loss$|Loss$", k, re.I)
        }
        return keep or dict(list(m.items())[:3])

    def _episode_from_log(run_dir):
        """The episode hash an offline pass evaluated (from its eval log)."""
        try:
            names = [f for f in os.listdir(run_dir) if f.endswith(".log")]
        except OSError:
            return None
        pat = re.compile(
            r"restricted valid dataset to \d+ episodes[^\[]*\['([0-9a-f]{24})'"
        )
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

    _TS_SUFFIX_RE = re.compile(r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

    def _episodes_from_log(run_dir):
        """Ordered episode hashes the pass was restricted to."""
        try:
            names = [f for f in os.listdir(run_dir) if f.endswith(".log")]
        except OSError:
            return []
        pat = re.compile(r"restricted valid dataset to \d+ episodes[^\[]*\[([^\]]*)\]")
        for name in names:
            try:
                with open(os.path.join(run_dir, name), errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            m = pat.search(text)
            if m:
                return re.findall(r"[0-9a-f]{24}", m.group(1))
        return []

    def _run_status(run_dir):
        """(status, detail) for an offline attempt, read from its log."""
        log = None
        try:
            for f in os.listdir(run_dir):
                if f.endswith(".log"):
                    log = os.path.join(run_dir, f)
                    break
        except OSError:
            return "unknown", None
        if log is None:
            return "unknown", None
        try:
            with open(log, errors="replace") as fh:
                text = fh.read()
        except OSError:
            return "unknown", None
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if "[eval_dreamzero] Done." in text:
            return "done", None
        if "Traceback (most recent call last)" in text or "AssertionError" in text:
            # Surface the assertion message — that is what killed the attempt.
            detail = None
            for ln in reversed(lines):
                if len(ln) > 40 and "Traceback" not in ln:
                    detail = ln.strip()[:300]
                    break
            return "crashed", detail
        return "running", (lines[-1][:200] if lines else None)

    def _scan_5fps():
        """The c1c4ff00 5 fps-source offline evals, grouped TF vs AR.

        Run dirs are ``<logical>_<YYYY-MM-DD_HH-MM-SS>`` and a logical run may
        have several attempts; prefer the newest attempt that actually produced
        videos, else the newest overall (so a crash is still reported).
        """
        base = os.path.join(root_dir, FIVE_FPS_PREFIX)
        out = {
            "prefix": FIVE_FPS_PREFIX,
            "label": FIVE_FPS_LABEL,
            "runs": [],
            "expected_frames": FIVE_FPS_EXPECTED_FRAMES,
        }
        if not os.path.isdir(base):
            return out

        attempts = {}
        for entry in sorted(os.listdir(base)):
            run_dir = os.path.join(base, entry)
            if not os.path.isdir(run_dir):
                continue
            logical = _TS_SUFFIX_RE.sub("", entry)
            val, pred, _, _, probe, _ = _collect_videos(
                os.path.join(run_dir, "videos"),
                f"{FIVE_FPS_PREFIX}/{entry}/videos",
                source_5fps=True,
            )
            status, detail = _run_status(run_dir)
            metrics = _pick_metrics(
                _metrics_from_json(os.path.join(run_dir, "metrics.json"))
                or _metrics_from_log(run_dir)
            )
            attempts.setdefault(logical, []).append(
                {
                    "dir": entry,
                    "validation": val,
                    "predicted": pred,
                    "probe": probe,
                    "status": status,
                    "detail": detail,
                    "metrics": metrics,
                    "episodes": _episodes_from_log(run_dir),
                }
            )

        for logical in sorted(attempts):
            tries = attempts[logical]
            # newest-first; a try with videos wins over a bare crash
            tries.sort(
                key=lambda a: (bool(a["validation"] or a["predicted"]), a["dir"])
            )
            best = tries[-1]
            mode = "ar" if re.search(r"(^|_)ar(_|$)", logical) else "tf"
            variant = "metrics" if logical.endswith("metrics") else "vids"
            out["runs"].append(
                {
                    "logical": logical,
                    "mode": mode,
                    "variant": variant,
                    "n_attempts": len(tries),
                    "crashed_attempts": [
                        t["dir"] for t in tries if t["status"] == "crashed"
                    ],
                    **best,
                }
            )
        return out

    def _scan_offline_pass(key, prefix, title, note, featured):
        base = os.path.join(root_dir, prefix)
        out = []
        if not os.path.isdir(base):
            return out
        for sub in sorted(os.listdir(base)):
            run_dir = os.path.join(base, sub)
            if not os.path.isdir(run_dir):
                continue
            val, pred, _, _, probe, ep_label = _collect_videos(
                os.path.join(run_dir, "videos"), f"{prefix}/{sub}/videos"
            )
            if not (val or pred):
                continue
            out.append(
                {
                    "key": key,
                    "prefix": prefix,
                    "sub": sub,
                    "title": title,
                    "note": note,
                    "featured": featured,
                    "validation": val,
                    "predicted": pred,
                    "probe": probe,
                    "epoch_label": ep_label,
                    "episode": _episode_from_log(run_dir),
                }
            )
        return out

    _SWEEP_EP_RE = re.compile(r"^(?:ckpt_)?epoch_(\d+)(?:_.*)?$")

    def _scan_sweeps():
        """Discover sweep ids under SWEEP_PREFIX; newest (name-sorted) first."""
        base = os.path.join(root_dir, SWEEP_PREFIX)
        sweeps = []
        if not os.path.isdir(base):
            return sweeps
        for sweep_id in sorted(os.listdir(base), reverse=True):
            sweep_dir = os.path.join(base, sweep_id)
            if not os.path.isdir(sweep_dir):
                continue
            try:
                entries = sorted(os.listdir(sweep_dir))
            except OSError:
                continue

            # metrics/epoch_<N>.json written by the sweep driver
            metrics_by_ep = {}
            metrics_dir = os.path.join(sweep_dir, "metrics")
            if os.path.isdir(metrics_dir):
                for name in os.listdir(metrics_dir):
                    m = re.match(r"epoch_(\d+)\.json$", name)
                    if m:
                        got = _pick_metrics(
                            _metrics_from_json(os.path.join(metrics_dir, name))
                        )
                        if got:
                            metrics_by_ep[int(m.group(1))] = got

            summary = None
            summary_path = os.path.join(sweep_dir, "summary.json")
            if os.path.isfile(summary_path):
                try:
                    with open(summary_path) as fh:
                        summary = json.load(fh)
                except (OSError, ValueError):
                    summary = None

            by_epoch, probes, ep_dirs = {}, {}, {}
            for entry in entries:
                m = _SWEEP_EP_RE.match(entry)
                if not m:
                    continue
                ep = int(m.group(1))
                run_dir = os.path.join(sweep_dir, entry)
                if not os.path.isdir(run_dir):
                    continue
                ep_dirs[ep] = entry
                val, pred, _, _, probe, _ = _collect_videos(
                    os.path.join(run_dir, "videos"),
                    f"{SWEEP_PREFIX}/{sweep_id}/{entry}/videos",
                    cap=SWEEP_CAP,
                )
                if val or pred:
                    by_epoch[ep] = {"validation": val, "predicted": pred}
                    probes[ep] = probe
                # metrics/ may lag the videos — fall back to the epoch's own log
                # so numbers appear while the sweep is still running.
                if ep not in metrics_by_ep:
                    got = _pick_metrics(_metrics_from_log(run_dir))
                    if got:
                        metrics_by_ep[ep] = got

            eps = sorted(by_epoch)
            n_expected = None
            if isinstance(summary, dict):
                for k in ("n_checkpoints", "num_checkpoints", "total"):
                    if isinstance(summary.get(k), int):
                        n_expected = summary[k]
                        break
            sweeps.append(
                {
                    "id": sweep_id,
                    "epochs": eps,
                    "byEpoch": {str(e): by_epoch[e] for e in eps},
                    "probe": {str(e): probes.get(e) for e in eps},
                    "metrics": {
                        str(e): metrics_by_ep[e] for e in sorted(metrics_by_ep)
                    },
                    "n_dirs": len(ep_dirs),
                    "n_expected": n_expected,
                    "has_summary": summary is not None,
                    "cap": SWEEP_CAP,
                }
            )
        return sweeps

    def _scan_training_run(run_key, run_dir):
        videos_dir = os.path.join(root_dir, run_dir, "videos")
        by_epoch, totals, probes = {}, {}, {}
        try:
            names = os.listdir(videos_dir)
        except OSError:
            names = []
        for name in names:
            m = re.match(r"^epoch_(\d+)$", name)
            if not m:
                continue
            ep = int(m.group(1))
            ep_dir = os.path.join(videos_dir, name)
            if not os.path.isdir(ep_dir):
                continue
            val, pred, n_val, n_pred, probe = [], [], 0, 0, None
            try:
                embodiments = sorted(os.listdir(ep_dir))
            except OSError:
                continue
            for emb in embodiments:
                emb_dir = os.path.join(ep_dir, emb)
                if not os.path.isdir(emb_dir):
                    continue
                rel = f"{run_dir}/videos/{name}/{emb}"
                v, p, nv, np_ = _split_families(emb_dir, rel, cap=TRAINING_CAP)
                val += v
                pred += p
                n_val += nv
                n_pred += np_
                if probe is None and (v or p):
                    probe = _probe(os.path.join(emb_dir, os.path.basename((v or p)[0])))
            if val or pred:
                by_epoch[ep] = {
                    "validation": val[:TRAINING_CAP],
                    "predicted": pred[:TRAINING_CAP],
                }
                totals[ep] = {"validation": n_val, "predicted": n_pred}
                probes[ep] = probe
        eps = sorted(by_epoch)
        gen_counts = {}
        for e in eps:
            g = (probes.get(e) or {}).get("generation", "unknown")
            gen_counts[g] = gen_counts.get(g, 0) + 1
        return {
            "key": run_key,
            "run_dir": run_dir,
            "epochs": eps,
            "byEpoch": {str(e): by_epoch[e] for e in eps},
            "totals": {str(e): totals[e] for e in eps},
            "probe": {str(e): probes.get(e) for e in eps},
            "gen_counts": gen_counts,
            "cap": TRAINING_CAP,
        }

    def _scan():
        passes = []
        for key, prefix, title, note, featured in OFFLINE_PASSES:
            passes += _scan_offline_pass(key, prefix, title, note, featured)
        return {
            "run": WAM_RUN_LABEL,
            "sweep_prefix": SWEEP_PREFIX,
            "five_fps": _scan_5fps(),
            "passes": passes,
            "sweeps": _scan_sweeps(),
            "training_runs": [_scan_training_run(k, d) for k, d in TRAINING_RUNS],
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
        # training entry cannot reach sibling runs or checkpoints.
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
