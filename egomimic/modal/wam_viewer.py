"""Web viewer for the WAM world-action model's validation videos.

ONE run, deliberately: ``data_div_oss/wam22_dw48_k2_5fps`` — the only run whose
videos were produced by the verified-correct workflow. Every other WAM output
family that used to be on this page (the wam_offline_5fps TF/AR passes, the
wam22_dw48_v2 and wam22_dw48_k2 training runs, the wam_gate_a2 / wam_gateA /
wam_gateC reference passes, and the wam_val_sweep checkpoints sweep) is
known-incorrect and has been removed — not hidden. ``ALLOWED_PATH_PREFIXES``
is narrowed to this run's ``videos/`` subtree, so a stale bookmark pointing at
one of the removed families gets a 400 rather than resurrecting a wrong video.

Verified workflow, from the live run's resolved .hydra/config.yaml:

    video    cam_horizon 17, stride 6 -> 5 Hz display; spans 97 raw frames
             = 3.23 s (30 fps base, strided x6 so it appears as 5 fps)
    actions  horizon 96, stride 1 -> raw 30 Hz; spans 96 raw frames = 3.20 s
    K=2 chunk = 8 displayed frames = 1.6 s = 48 actions @ 30 Hz
    videos measure 326 f @ 5 fps = 65.2 s — a full episode

Layout: predicted (the model's dreamt frames) beside validation (the
ground-truth clip) for one selected epoch, both with action-trail overlays.

The per-video frames/fps/duration readout is measured from each mp4's own atoms
and kept deliberately: a silent double-subsample once truncated these videos to
54 frames, and that readout is what caught it. Values that deviate from the
expected 5 fps / full-episode length are flagged inline.

Deploy:
    MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/wam_viewer.py

Routes (single ASGI web function -> one URL, relative paths):
    /            HTML viewer page
    /api/index   {"epochs": [...], "byEpoch": {...}} — volume scan, cached 60 s
    /video?path= streams one mp4, restricted to this run's videos/ subtree
"""

import modal

# ffmpeg + numpy are here only for the TEMPORARY debug section's temporal
# uniformity profile (see DEBUG_SECTION); drop them when that section goes.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi[standard]", "numpy")
)

app = modal.App("wam-viewer", image=image)

outputs_volume = modal.Volume.from_name("egoverse-training-outputs")

MOUNT_PATH = "/data"
INDEX_TTL_S = 60

RUN_LABEL = "data_div_oss_wam22_dw48_k2_5fps"
RUN_DIR = "data_div_oss/wam22_dw48_k2_5fps"

# Expected shape of a correct video, used only to flag a deviating readout.
EXPECTED_FPS = 5
EXPECTED_MIN_FRAMES = 300

# ---------------------------------------------------------------------------
# TEMPORARY DEBUG SECTION -- rip-out instructions:
#   delete (a) this DEBUG_SECTION block, (b) the DEBUG_SECTION entries in
#   ALLOWED_PATH_PREFIXES, (c) _profile() / _scan_debug() and the "debug" key in
#   _scan(), (d) renderDebug() and its calls at the end of render() plus the
#   d.debug term in signature(), (e) the #debugsec CSS rules, and (f) the
#   apt_install("ffmpeg") / "numpy" additions to `image`. Nothing else refers to
#   it -- it is deliberately kept out of the epoch/default-selection logic.
# ---------------------------------------------------------------------------
DEBUG_SECTION = {
    "root": "pksnack_wam_debug",
    "title": "Temporary validation runs — read-stride before/after",
    "caveat": (
        "<b>Not models to judge.</b> Four 4-epoch throwaways "
        "(<code>trainer=debug_modal</code>, <code>num_val_episodes=1</code>, 2% norm "
        "stats, packaging_snacks) kept only to demonstrate one bug and its fix."
    ),
    "bug": (
        "The frame stride reached the keymap and the episode tiler (window anchors "
        "stepping 96 raw frames) but <b>not the actual zarr read</b>, because "
        "<code>ZarrWamDataset</code> implements its own <code>__getitem__</code>. So "
        "each clip stayed <b>17 consecutive 30 fps frames (0.567 s)</b> while the "
        "windows sat 96 frames apart: 16 output frames each advancing 1 source frame, "
        "then a <b>6× jump at the seam</b>. Intended and now fixed: take the 30 fps "
        "source, keep every 6th frame, display at 5 fps — so the 5 fps video spans the "
        "<b>same real time</b> as the 30 fps original."
    ),
    # Rendered in this order.
    "runs": [
        {
            "name": "stridefix",
            "label": "THE FIX — current / correct",
            "note": "read stride applied in ZarrWamDataset.__getitem__ too",
            "expected_frames": 336,
            "expected_duration_s": 67.2,
            "expect_uniform": True,
        },
        {
            "name": "stridecheck",
            "label": "correct duration, pre-read-fix CONTENT",
            "note": "336 f / 67.2 s, but each window holds 30 fps frames — the seam "
            "jump is the tell; geometry alone cannot separate it from the fix",
            "expected_frames": 336,
            "expected_duration_s": 67.2,
            "expect_uniform": False,
        },
        {
            "name": "walkcheck",
            "label": "×6-decimated assembly bug",
            "note": "56 f / 11.2 s — decimation applied one time too many",
            "expected_frames": 56,
            "expected_duration_s": 11.2,
            "expect_uniform": False,
        },
        {
            "name": "cfgcheck",
            "label": "original short-clip geometry",
            "note": "16 f / 3.2 s — one window only",
            "expected_frames": 16,
            "expected_duration_s": 3.2,
            "expect_uniform": False,
        },
    ],
    # Temporal-uniformity profile: mean abs inter-frame difference, averaged per
    # phase mod PHASE_PERIOD, reported as max/min ratio.
    "phase_period": 16,
    "uniform_max_ratio": 2.0,
    "seam_min_ratio": 2.5,
    "profile_size": [160, 90],
}
DEBUG_PROFILE_BUDGET = 8  # videos profiled per index scan; rest fill in on refresh

ALLOWED_PATH_PREFIXES = (f"{RUN_DIR}/videos/",) + tuple(
    # TEMPORARY -- see DEBUG_SECTION
    f"{DEBUG_SECTION['root']}/{r['name']}/videos/"
    for r in DEBUG_SECTION["runs"]
)

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WAM val videos — wam22_dw48_k2_5fps</title>
<style>
  :root {
    --bg: #101216; --panel: #181b21; --border: #2a2e37;
    --text: #e6e8ec; --dim: #8b919d; --accent: #6ea8fe;
    --pred: #e08a8a; --gt: #7fb3d1; --warn: #e0a04d; --good: #7fd1a4;
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
  code {
    color: var(--text); background: var(--panel); padding: 1px 5px; border-radius: 4px;
  }
  select, button {
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 10px; font-size: 13px; cursor: pointer;
  }
  select:hover, button:hover { border-color: var(--accent); }
  button:disabled { opacity: .45; cursor: not-allowed; }
  .lbl { color: var(--dim); font-size: 12px; }
  .spacer { flex: 1; }
  #status { color: var(--dim); font-size: 12px; }
  .workflow {
    color: var(--dim); font-size: 12px; margin-top: 7px; line-height: 1.7;
  }
  .workflow b { color: #b9c0cc; font-weight: 600; }
  main { padding: 14px 16px 40px; }
  .pair {
    display: flex; gap: 14px; margin-bottom: 16px; flex-wrap: wrap;
  }
  .col { flex: 1 1 460px; min-width: 320px; }
  .col h2 {
    margin: 0 0 6px; font-size: 13px; font-weight: 600; display: flex;
    align-items: baseline; gap: 8px; flex-wrap: wrap;
  }
  .col.pred h2 { color: var(--pred); }
  .col.val h2 { color: var(--gt); }
  .col video {
    width: 100%; border-radius: 8px; background: #000; display: block;
    border: 1px solid var(--border);
  }
  .col.pred video { border-color: #4a3236; }
  .col.val video { border-color: #2b3f4c; }
  .readout {
    color: var(--dim); font-size: 11.5px; margin-top: 5px;
    display: flex; gap: 10px; flex-wrap: wrap; align-items: baseline;
  }
  .readout .ok { color: var(--good); }
  .readout .bad { color: var(--warn); font-weight: 600; }
  .readout .fn { color: #7d8492; }
  .placeholder {
    color: var(--dim); border: 1px dashed var(--border); border-radius: 8px;
    padding: 40px 16px; text-align: center; font-size: 13px;
  }
  /* TEMPORARY debug section -- delete with the DEBUG_SECTION block */
  #debugsec {
    margin-top: 30px; border: 1px dashed #4a4433; border-radius: 10px;
    padding: 10px 12px 4px; background: #16150f;
  }
  #debugsec h2 { font-size: 13px; margin: 0 0 4px; color: var(--warn); }
  #debugsec .meta { color: var(--dim); font-size: 11.5px; line-height: 1.65; }
  #debugsec .col h3 { margin: 0 0 5px; font-size: 12px; font-weight: 600; }
  #debugsec .col.pred h3 { color: var(--pred); }
  #debugsec .col.val h3 { color: var(--gt); }
  #debugsec .ep { color: #9aa3b2; font-size: 12px; margin: 10px 0 4px; font-weight: 600; }
  #debugsec .big {
    font-size: 12.5px; font-weight: 600; padding: 2px 8px; border-radius: 999px;
    border: 1px solid var(--border);
  }
  #debugsec .big.ok { color: var(--good); border-color: var(--good); }
  #debugsec .big.bad { color: var(--warn); border-color: var(--warn); }
  #debugsec .verdict { margin: 6px 0 2px; font-size: 12.5px; }
  #debugsec .bug { color: var(--dim); font-size: 11.5px; line-height: 1.7; margin: 6px 0 2px; }
  #debugsec .run {
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px 4px;
    margin: 10px 0; background: #14140f;
  }
  #debugsec .run.fix { border-color: var(--good); background: #121710; }
  #debugsec .run h3 { margin: 0 0 3px; font-size: 12.5px; font-weight: 600; }
  #debugsec .run.fix h3 { color: var(--good); }
  #debugsec .run .rnote { color: var(--dim); font-size: 11px; line-height: 1.55; }
  #debugsec .phases {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10.5px;
    color: var(--dim); margin-top: 3px; word-break: break-all;
  }
  #debugsec .phases b { color: var(--warn); }
</style>
</head>
<body>
<header>
  <div class="hrow">
    <h1>WAM val videos — <code>wam22_dw48_k2_5fps</code></h1>
    <span class="lbl">epoch</span>
    <button id="prev" title="previous epoch">&#9664;</button>
    <select id="epoch"></select>
    <button id="next" title="next epoch">&#9654;</button>
    <button id="playboth">Play both (synced)</button>
    <button id="pauseboth">Pause</button>
    <span class="spacer"></span>
    <span id="status">loading…</span>
  </div>
  <div class="workflow" id="workflow"></div>
</header>
<main id="main"></main>
<script>
(function () {
  const $ = (id) => document.getElementById(id);
  const mainEl = $("main"), statusEl = $("status"), epochSel = $("epoch");
  let index = null, epoch = null, playToken = 0;

  const observer = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        const v = e.target;
        if (!v.src && v.dataset.src) { v.src = v.dataset.src; v.preload = "metadata"; }
        observer.unobserve(v);
      }
    }
  }, { rootMargin: "250px" });

  // Measured straight from the mp4 atoms — this readout is how a silent
  // double-subsample (326 f -> 54 f) was caught, so it stays visible.
  function readout(pr) {
    if (!pr) return '<span class="bad">no measurement</span>';
    const okFps = pr.fps === index.expected_fps;
    const okLen = pr.frames >= index.expected_min_frames;
    const cls = (ok) => (ok ? "ok" : "bad");
    return (
      '<span class="' + cls(okLen) + '">' + pr.frames + " frames</span>" +
      '<span class="' + cls(okFps) + '">' + pr.fps + " fps</span>" +
      "<span>" + (pr.duration != null ? pr.duration.toFixed(1) + " s" : "?") + "</span>" +
      "<span>" + pr.size_mb + " MB</span>" +
      (okFps && okLen ? "" : '<span class="bad">— unexpected, expected ' +
        index.expected_min_frames + "+ frames @ " + index.expected_fps + " fps</span>")
    );
  }

  function column(kind, title, item) {
    const col = document.createElement("div");
    col.className = "col " + kind;
    const h = document.createElement("h2");
    h.textContent = title;
    col.appendChild(h);
    if (!item || !item.path) {
      const ph = document.createElement("div");
      ph.className = "placeholder";
      ph.textContent = "no " + kind + " video at this epoch";
      col.appendChild(ph);
      return col;
    }
    const v = document.createElement("video");
    v.controls = true; v.muted = true; v.loop = true;
    v.playsInline = true; v.preload = "none";
    v.dataset.src = "video?path=" + encodeURIComponent(item.path);
    v.title = item.path;
    v.dataset.role = kind;
    observer.observe(v);
    col.appendChild(v);
    const r = document.createElement("div");
    r.className = "readout";
    r.innerHTML = readout(item.probe) +
      '<span class="fn">' + item.path.split("/").pop() + "</span>";
    col.appendChild(r);
    return col;
  }

  function render() {
    mainEl.innerHTML = "";
    const pairs = (index.byEpoch || {})[String(epoch)] || [];
    if (!pairs.length) {
      const ph = document.createElement("div");
      ph.className = "placeholder";
      ph.textContent = index.epochs.length
        ? "no videos at epoch " + epoch
        : "no validation videos on the volume yet for this run";
      mainEl.appendChild(ph);
      renderDebug();  // TEMPORARY -- see DEBUG_SECTION
      return;
    }
    for (const p of pairs) {
      const row = document.createElement("div");
      row.className = "pair";
      row.appendChild(column("pred", "predicted — dream frames (model's imagined future)", p.predicted));
      row.appendChild(column("val", "validation — ground-truth clip", p.validation));
      mainEl.appendChild(row);
    }
    renderDebug();  // TEMPORARY -- see DEBUG_SECTION
  }

  // ---------------------------------------------------------------------
  // TEMPORARY debug section. Self-contained: its own readout + flagging, no
  // shared state, not part of epoch/default selection. Delete this function
  // and its single call at the end of render() to remove.
  // ---------------------------------------------------------------------
  function renderDebug() {
    const d = index.debug;
    if (!d) return;
    const sec = document.createElement("section");
    sec.id = "debugsec";

    const h = document.createElement("h2");
    h.textContent = "\u26a0 " + d.title + " — secondary, temporary";
    sec.appendChild(h);

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.innerHTML = d.caveat;
    sec.appendChild(meta);

    const bug = document.createElement("div");
    bug.className = "bug";
    bug.innerHTML = "<b>The bug:</b> " + d.bug +
      "<br><b>Uniformity profile:</b> mean absolute inter-frame difference, averaged " +
      "per phase mod " + d.phase_period + ". Ratio (max/min) &lt; " +
      d.uniform_max_ratio + " = <b>uniform advance</b> (correct); \u2265 " +
      d.seam_min_ratio + " = <b>seam jump</b> (30 fps content inside the window). " +
      "Geometry alone cannot tell the fix from stridecheck — this readout can. " +
      "The <i>validation</i> clip is the clean diagnostic; predicted clips carry the " +
      "model's own frame-to-frame noise on top.";
    sec.appendChild(bug);

    const geom = (pr, spec) => {
      if (!pr || !pr.frames) return { cls: "bad", txt: "no measurement" };
      const g = pr.frames + " f @ " + pr.fps + " fps = " +
        (pr.duration != null ? pr.duration.toFixed(1) + " s" : "?");
      const ok = pr.frames === spec.expected_frames;
      return { cls: ok ? "ok" : "bad",
               txt: g + (ok ? "  \u2713" : "  \u2717 want " + spec.expected_frames + " f") };
    };

    for (const run of d.runs) {
      const box = document.createElement("div");
      box.className = "run" + (run.expect_uniform ? " fix" : "");
      const t = document.createElement("h3");
      t.textContent = run.name + " — " + run.label;
      box.appendChild(t);
      const rn = document.createElement("div");
      rn.className = "rnote";
      rn.innerHTML = run.note + " · <code>" + run.dir + "/videos/</code> · expected <b>" +
        run.expected_frames + " f @ 5 fps = " + run.expected_duration_s + " s</b>";
      box.appendChild(rn);

      if (!run.epochs.length) {
        const ph = document.createElement("div");
        ph.className = "placeholder";
        ph.style.marginTop = "8px";
        ph.textContent = run.present
          ? "still running — no videos written yet (fills in on the next 60 s refresh)"
          : "run directory not on the volume";
        box.appendChild(ph);
        sec.appendChild(box);
        continue;
      }

      for (const ep of run.epochs) {
        const lbl = document.createElement("div");
        lbl.className = "ep";
        lbl.textContent = "epoch " + ep;
        box.appendChild(lbl);
        for (const p of run.byEpoch[String(ep)]) {
          const row = document.createElement("div");
          row.className = "pair";
          for (const [kind, title, key] of [
            ["pred", "predicted", "predicted"], ["val", "validation", "validation"]]) {
            const item = p[key];
            const col = document.createElement("div");
            col.className = "col " + kind;
            const ct = document.createElement("h3");
            ct.textContent = title + (kind === "val" ? "  (diagnostic)" : "");
            col.appendChild(ct);
            if (!item || !item.path) {
              const ph = document.createElement("div");
              ph.className = "placeholder";
              ph.textContent = "no " + title + " video";
              col.appendChild(ph);
            } else {
              const v = document.createElement("video");
              v.controls = true; v.muted = true; v.loop = true;
              v.playsInline = true; v.preload = "none";
              v.dataset.src = "video?path=" + encodeURIComponent(item.path);
              v.title = item.path;
              observer.observe(v);
              col.appendChild(v);

              const g = geom(item.probe, run);
              const r = document.createElement("div");
              r.className = "readout";
              r.innerHTML = '<span class="big ' + g.cls + '">' + g.txt + "</span>" +
                '<span class="fn">' + item.path.split("/").pop() + "</span>";
              col.appendChild(r);

              const pf = item.profile;
              const u = document.createElement("div");
              u.className = "readout";
              if (!pf) {
                u.innerHTML = "<span>uniformity: not profiled yet (computed a few " +
                  "per refresh)</span>";
              } else if (pf.error) {
                u.innerHTML = "<span>uniformity: " + pf.error + "</span>";
              } else {
                const bad = pf.verdict === "seam jump";
                u.innerHTML = '<span class="big ' + (bad ? "bad" : "ok") +
                  '">ratio ' + pf.ratio + " — " + pf.verdict + "</span>";
              }
              col.appendChild(u);
              if (pf && pf.phases) {
                const ps = document.createElement("div");
                ps.className = "phases";
                ps.innerHTML = "phases " + pf.phases.map((x, i) =>
                  i === pf.peak_phase ? "<b>" + x.toFixed(1) + "</b>" : x.toFixed(1)
                ).join(" ");
                col.appendChild(ps);
              }
            }
            row.appendChild(col);
          }
          box.appendChild(row);
        }
      }
      sec.appendChild(box);
    }
    mainEl.appendChild(sec);
  }

  function setEpoch(e) {
    epoch = e;
    if (e !== null) epochSel.value = e;
    const i = index.epochs.indexOf(epoch);
    $("prev").disabled = i <= 0;
    $("next").disabled = i < 0 || i >= index.epochs.length - 1;
    render();
  }

  function step(d) {
    const i = index.epochs.indexOf(epoch) + d;
    if (i >= 0 && i < index.epochs.length) setEpoch(index.epochs[i]);
  }

  function buildSelector() {
    const sig = index.epochs.join(",");
    if (epochSel.dataset.sig === sig) return;
    epochSel.dataset.sig = sig;
    epochSel.innerHTML = "";
    for (const e of index.epochs) {
      const o = document.createElement("option");
      o.value = e; o.textContent = "epoch " + e;
      epochSel.appendChild(o);
    }
    epochSel.disabled = !index.epochs.length;
  }

  $("prev").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  epochSel.onchange = () => setEpoch(Number(epochSel.value));
  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") step(-1);
    if (e.key === "ArrowRight") step(1);
  });

  // Start predicted and validation together so they can be read against each
  // other frame for frame.
  $("playboth").onclick = () => {
    const vids = Array.from(document.querySelectorAll("video"));
    vids.forEach((v) => { if (!v.src && v.dataset.src) v.src = v.dataset.src; });
    const token = ++playToken;
    Promise.all(vids.map((v) => new Promise((res) => {
      if (v.readyState >= 2) return res();
      const on = () => { v.removeEventListener("loadeddata", on); res(); };
      v.addEventListener("loadeddata", on);
      setTimeout(res, 5000);
    }))).then(() => {
      if (token !== playToken) return;
      vids.forEach((v) => {
        try { v.currentTime = 0; } catch (err) {}
        v.play().catch(() => {});
      });
    });
  };
  $("pauseboth").onclick = () => {
    playToken++;
    document.querySelectorAll("video").forEach((v) => v.pause());
  };

  function setWorkflow() {
    const w = index.workflow;
    $("workflow").innerHTML =
      "Run <code>" + index.run + "</code> — the one run produced by the " +
      "verified-correct workflow, so it is all this page serves. " +
      "<b>predicted</b> = the model's imagined future frames; " +
      "<b>validation</b> = the ground-truth clip for the same window; both carry " +
      "action-trail overlays. <b>Note:</b> this run predates the read-stride fix, so " +
      "its clips show the same within-window artifact the debug section below " +
      "demonstrates.<br>" +
      "<b>video</b>: cam_horizon " + w.cam_horizon + ", stride " + w.video_stride +
      " → " + w.display_hz + " Hz display, spanning " + w.video_raw_frames +
      " raw frames = " + w.video_span_s + " s (30 fps base strided ×" +
      w.video_stride + " to appear as " + w.display_hz + " fps) · " +
      "<b>actions</b>: horizon " + w.action_horizon + ", stride " +
      w.action_stride + " → raw " + w.action_hz + " Hz, spanning " +
      w.action_raw_frames + " raw frames = " + w.action_span_s + " s · " +
      "<b>K=2 chunk</b> = " + w.chunk_displayed_frames + " displayed frames = " +
      w.chunk_s + " s = <b>" + w.chunk_actions + " actions @ " + w.action_hz +
      " Hz</b>. Videos measure " + w.expected_frames + " f @ " + w.display_hz +
      " fps = " + w.expected_duration_s + " s — a full episode.";
  }

  function signature(d) {
    return JSON.stringify([
      d.epochs, Object.keys(d.byEpoch || {}).length,
      d.debug ? d.debug.epochs : null,  // TEMPORARY -- see DEBUG_SECTION
    ]);
  }

  let lastSig = null;
  function load() {
    return fetch("api/index")
      .then((r) => r.json())
      .then((data) => {
        const sig = signature(data);
        const first = index === null;
        if (!first && sig === lastSig) return;  // nothing new; don't disturb playback
        const keep = epoch;
        index = data; lastSig = sig;
        if (first) setWorkflow();
        buildSelector();
        const eps = index.epochs;
        // Keep the viewer's place across refreshes; default to the newest epoch.
        setEpoch(eps.includes(keep) ? keep : (eps.length ? eps[eps.length - 1] : null));
        statusEl.textContent = eps.length
          ? eps.length + " epochs (" + eps[0] + "–" + eps[eps.length - 1] +
            ") · refreshes every 60 s"
          : "no videos yet · refreshes every 60 s";
      })
      .catch((e) => { statusEl.textContent = "failed to load index: " + e; });
  }
  load();
  setInterval(load, 60000);  // the run is live; a new val epoch lands every 30
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
    # An mp4 never changes once written, so probes are cached permanently by
    # (path, size); only newly committed epochs ever cost a read.
    _probe_cache = {}

    def _vid_sort_key(name):
        m = re.search(r"(\d+)\.mp4$", name)
        return (int(m.group(1)) if m else 1 << 30, name)

    def _probe(path, window=131072):
        """(duration, frames, fps, size) from the mp4 mvhd + stsz atoms.

        Reads only the first and last `window` bytes: torchvision/pyav writes
        moov at the END, other writers at the start, so both are searched.
        Verified against ffprobe on this run's videos (326 f, 5 fps, 65.2 s).
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
            if frames is None and dur:
                # "stsz" also occurs inside compressed frame data, so scan every
                # occurrence and keep the first whose sample count implies a
                # plausible frame rate. Without this filter a false match yields
                # a nonsense count (seen: 2.17e9 frames on one epoch).
                j = buf.find(b"stsz")
                while j >= 0:
                    try:
                        cand = struct.unpack(">I", buf[j + 12 : j + 16])[0]
                    except struct.error:
                        cand = None
                    if cand and 0.5 <= cand / dur <= 120:
                        frames = cand
                        break
                    j = buf.find(b"stsz", j + 1)

        fps = round(frames / dur, 2) if (frames and dur) else None
        if fps is not None and abs(fps - round(fps)) < 0.05:
            fps = int(round(fps))
        out = {
            "duration": round(dur, 2) if dur else None,
            "frames": frames,
            "fps": fps,
            "size_mb": round(size / 1e6, 2),
        }
        _probe_cache[key] = out
        return out

    def _item(emb_dir, rel_dir, name):
        return {
            "path": f"{rel_dir}/{name}",
            "probe": _probe(os.path.join(emb_dir, name)),
        }

    def _scan():
        """{epoch: [{predicted, validation}, ...]} for the one run."""
        videos_dir = os.path.join(root_dir, RUN_DIR, "videos")
        by_epoch = {}
        try:
            names = os.listdir(videos_dir)
        except OSError:
            names = []
        for name in names:
            m = re.match(r"^epoch_(\d+)$", name)
            if not m:
                continue
            ep_dir = os.path.join(videos_dir, name)
            if not os.path.isdir(ep_dir):
                continue
            pred, val = [], []
            try:
                embodiments = sorted(os.listdir(ep_dir))
            except OSError:
                continue
            for emb in embodiments:
                emb_dir = os.path.join(ep_dir, emb)
                if not os.path.isdir(emb_dir):
                    continue
                rel_dir = f"{RUN_DIR}/videos/{name}/{emb}"
                try:
                    files = sorted(os.listdir(emb_dir), key=_vid_sort_key)
                except OSError:
                    continue
                for f in files:
                    if not f.endswith(".mp4"):
                        continue
                    if f.startswith("predicted_video"):
                        pred.append(_item(emb_dir, rel_dir, f))
                    elif f.startswith("validation_video"):
                        val.append(_item(emb_dir, rel_dir, f))
            if pred or val:
                pairs = []
                for i in range(max(len(pred), len(val))):
                    pairs.append(
                        {
                            "predicted": pred[i] if i < len(pred) else None,
                            "validation": val[i] if i < len(val) else None,
                        }
                    )
                by_epoch[int(m.group(1))] = pairs
        epochs = sorted(by_epoch)
        return {
            "run": RUN_LABEL,
            "run_dir": RUN_DIR,
            "epochs": epochs,
            "byEpoch": {str(e): by_epoch[e] for e in epochs},
            "expected_fps": EXPECTED_FPS,
            "expected_min_frames": EXPECTED_MIN_FRAMES,
            # The verified workflow, straight from the run's resolved config.
            "workflow": {
                "cam_horizon": 17,
                "video_stride": 6,
                "display_hz": 5,
                "video_raw_frames": 97,
                "video_span_s": 3.23,
                "action_horizon": 96,
                "action_stride": 1,
                "action_hz": 30,
                "action_raw_frames": 96,
                "action_span_s": 3.20,
                "chunk_displayed_frames": 8,
                "chunk_s": 1.6,
                "chunk_actions": 48,
                "expected_frames": 326,
                "expected_duration_s": 65.2,
            },
            "debug": _scan_debug(),  # TEMPORARY -- see DEBUG_SECTION
        }

    _profile_cache = {}

    def _profile(path, budget):
        """Temporal-uniformity profile (TEMPORARY -- see DEBUG_SECTION).

        Decodes to small greyscale frames, takes the mean absolute inter-frame
        difference, then averages those per phase mod PHASE_PERIOD. Uniform
        stepping gives a flat profile; 30 fps content inside each window gives a
        big spike at the last phase (the seam). Returns None when the budget for
        this scan is used up, so a cold scan stays fast and later refreshes fill
        the rest in.
        """
        try:
            size = os.path.getsize(path)
        except OSError:
            return None, budget
        key = (path, size)
        if key in _profile_cache:
            return _profile_cache[key], budget
        if budget <= 0:
            return None, budget

        import subprocess

        period = DEBUG_SECTION["phase_period"]
        w, h = DEBUG_SECTION["profile_size"]
        out = None
        try:
            import numpy as np

            r = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    path,
                    "-vf",
                    f"scale={w}:{h},format=gray",
                    "-f",
                    "rawvideo",
                    "-",
                ],
                capture_output=True,
                timeout=180,
            )
            fsz = w * h
            n = len(r.stdout) // fsz
            if n < period + 2:
                out = {
                    "error": f"too short to profile ({n} frames vs {period}-frame period)"
                }
            else:
                arr = (
                    np.frombuffer(r.stdout[: n * fsz], dtype=np.uint8)
                    .reshape(n, fsz)
                    .astype(np.int16)
                )
                dif = np.abs(np.diff(arr, axis=0)).mean(axis=1)
                phases = [float(dif[i::period].mean()) for i in range(period)]
                lo, hi = min(phases), max(phases)
                ratio = (hi / lo) if lo > 0 else None
                if ratio is None:
                    verdict = "inconclusive"
                elif ratio < DEBUG_SECTION["uniform_max_ratio"]:
                    verdict = "uniform advance"
                elif ratio >= DEBUG_SECTION["seam_min_ratio"]:
                    verdict = "seam jump"
                else:
                    verdict = "inconclusive"
                out = {
                    "frames": n,
                    "phases": [round(x, 2) for x in phases],
                    "peak_phase": int(max(range(period), key=lambda i: phases[i])),
                    "ratio": round(ratio, 2) if ratio else None,
                    "verdict": verdict,
                }
        except Exception as exc:  # decoder missing, timeout, malformed file
            out = {"error": f"{type(exc).__name__}: {exc}"[:160]}
        _profile_cache[key] = out
        return out, budget - 1

    def _scan_debug():
        """TEMPORARY: the four read-stride debug runs (see DEBUG_SECTION)."""
        root = DEBUG_SECTION["root"]
        budget = DEBUG_PROFILE_BUDGET
        runs_out = []
        for spec in DEBUG_SECTION["runs"]:
            run_dir = os.path.join(root_dir, root, spec["name"])
            videos_dir = os.path.join(run_dir, "videos")
            by_epoch = {}
            try:
                names = os.listdir(videos_dir)
            except OSError:
                names = []
            for name in names:
                m = re.match(r"^epoch_(\d+)$", name)
                if not m:
                    continue
                ep_dir = os.path.join(videos_dir, name)
                if not os.path.isdir(ep_dir):
                    continue
                pred, val = [], []
                try:
                    embodiments = sorted(os.listdir(ep_dir))
                except OSError:
                    continue
                for emb in embodiments:
                    emb_dir = os.path.join(ep_dir, emb)
                    if not os.path.isdir(emb_dir):
                        continue
                    rel_dir = f"{root}/{spec['name']}/videos/{name}/{emb}"
                    try:
                        files = sorted(os.listdir(emb_dir), key=_vid_sort_key)
                    except OSError:
                        continue
                    for fn in files:
                        if not fn.endswith(".mp4"):
                            continue
                        item = _item(emb_dir, rel_dir, fn)
                        prof, budget = _profile(os.path.join(emb_dir, fn), budget)
                        item["profile"] = prof
                        (pred if fn.startswith("predicted_video") else val).append(item)
                if pred or val:
                    by_epoch[int(m.group(1))] = [
                        {
                            "predicted": pred[i] if i < len(pred) else None,
                            "validation": val[i] if i < len(val) else None,
                        }
                        for i in range(max(len(pred), len(val)))
                    ]
            epochs = sorted(by_epoch)
            entry = dict(spec)
            entry.update(
                {
                    "dir": f"{root}/{spec['name']}",
                    "present": os.path.isdir(run_dir),
                    "epochs": epochs,
                    "byEpoch": {str(e): by_epoch[e] for e in epochs},
                }
            )
            runs_out.append(entry)
        return {
            "title": DEBUG_SECTION["title"],
            "caveat": DEBUG_SECTION["caveat"],
            "bug": DEBUG_SECTION["bug"],
            "phase_period": DEBUG_SECTION["phase_period"],
            "uniform_max_ratio": DEBUG_SECTION["uniform_max_ratio"],
            "seam_min_ratio": DEBUG_SECTION["seam_min_ratio"],
            "runs": runs_out,
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
        # Only this run's videos/ subtree is servable: every removed family, and
        # this run's own checkpoints/ and logs, are rejected outright.
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
