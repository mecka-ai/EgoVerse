"""Web viewer for the WAM read-stride fix validation run.

ONE run, deliberately: ``pksnack_wam_debug/stridefix`` — the first output where
the frame stride actually reaches the zarr read, so each displayed frame
advances 6 source frames. Everything that used to be on this page (the
data_div_oss/wam22_dw48_k2_5fps training run and the stridecheck / walkcheck /
cfgcheck comparison debug runs) predates or demonstrates the bug and has been
removed — not hidden. ``ALLOWED_PATH_PREFIXES`` is narrowed to this run's
``videos/`` subtree, so a stale bookmark for any of them gets a 400.

The rate contract this run validates:

    30 fps source -> keep every 6th frame -> display at 5 fps, so the clip spans
    the SAME real time as the 30 fps original (336 f @ 5 fps = 67.2 s).
    One K=2 chunk = 8 displayed frames = 1.6 s carrying 48 actions at 30 Hz,
    i.e. 6 actions consumed per displayed frame; the overlay recedes
    48 -> 42 -> ... -> 6 and resets at the chunk boundary.

Two measurements are shown per video, both computed from the file itself:

  geometry    frames / fps / duration, parsed from the mp4 mvhd + stsz atoms.
              A silent double-subsample once truncated these to 54 frames and
              this readout is what caught it, so it stays.
  uniformity  mean absolute inter-frame difference, averaged per phase mod 16
              and reported as a max/min ratio. Uniform stepping is flat; 30 fps
              content inside each window spikes at the last phase (the seam).
              Reference (epoch_3 validation): ratio 1.59, phase means ~11.5.

Note the phase magnitudes ROSE from ~4.3 in the pre-fix runs to ~11.5 here.
That increase is the good signal — consecutive displayed frames are now 6
source frames apart, so they legitimately differ ~6x more. It is not a
regression.

Deploy:
    MODAL_ENVIRONMENT=robotics modal deploy egomimic/modal/wam_viewer.py

Routes (single ASGI web function -> one URL, relative paths):
    /            HTML viewer page
    /api/index   {"epochs": [...], "byEpoch": {...}} — volume scan, cached 60 s
    /video?path= streams one mp4, restricted to this run's videos/ subtree
"""

import modal

# ffmpeg + numpy back the uniformity profile; fastapi serves the page.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("fastapi[standard]", "numpy")
)

app = modal.App("wam-viewer", image=image)

outputs_volume = modal.Volume.from_name("egoverse-training-outputs")

MOUNT_PATH = "/data"
INDEX_TTL_S = 60

RUN_LABEL = "pksnack_wam_debug/stridefix"
RUN_DIR = "pksnack_wam_debug/stridefix"

# The only servable prefix. Everything else on the volume — including this run's
# own checkpoints/ and logs — is rejected.
ALLOWED_PATH_PREFIXES = (f"{RUN_DIR}/videos/",)

# Expected geometry of a correct video, used only to flag a deviating readout.
EXPECTED_FRAMES = 336
EXPECTED_FPS = 5
EXPECTED_DURATION_S = 67.2

# Uniformity profile settings.
PHASE_PERIOD = 16  # one K=2 chunk = 8 displayed frames; the window is 16
PROFILE_SIZE = (160, 90)
UNIFORM_MAX_RATIO = 2.0  # below this: uniform advance (correct)
SEAM_MIN_RATIO = 2.5  # at or above: seam jump (30 fps content in the window)

PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WAM read-stride fix — stridefix</title>
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
  .caption { color: var(--dim); font-size: 12px; margin-top: 7px; line-height: 1.7; }
  .caption b { color: #b9c0cc; font-weight: 600; }
  .caveat { color: var(--warn); }
  main { padding: 14px 16px 40px; }
  .pair { display: flex; gap: 14px; margin-bottom: 16px; flex-wrap: wrap; }
  .col { flex: 1 1 460px; min-width: 320px; }
  .col h2 {
    margin: 0 0 6px; font-size: 13px; font-weight: 600; display: flex;
    align-items: baseline; gap: 8px; flex-wrap: wrap;
  }
  .col.pred h2 { color: var(--pred); }
  .col.val h2 { color: var(--gt); }
  .col h2 .tag { color: var(--dim); font-weight: 400; font-size: 11.5px; }
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
  .readout .fn { color: #7d8492; }
  .big {
    font-size: 12.5px; font-weight: 600; padding: 2px 8px; border-radius: 999px;
    border: 1px solid var(--border);
  }
  .big.ok { color: var(--good); border-color: var(--good); }
  .big.bad { color: var(--warn); border-color: var(--warn); }
  .phases {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10.5px;
    color: var(--dim); margin-top: 4px; word-break: break-all;
  }
  .phases b { color: var(--warn); }
  .placeholder {
    color: var(--dim); border: 1px dashed var(--border); border-radius: 8px;
    padding: 40px 16px; text-align: center; font-size: 13px;
  }
</style>
</head>
<body>
<header>
  <div class="hrow">
    <h1>WAM read-stride fix — <code>stridefix</code></h1>
    <span class="lbl">epoch</span>
    <button id="prev" title="previous epoch">&#9664;</button>
    <select id="epoch"></select>
    <button id="next" title="next epoch">&#9654;</button>
    <button id="playboth">Play both (synced)</button>
    <button id="pauseboth">Pause</button>
    <span class="spacer"></span>
    <span id="status">loading…</span>
  </div>
  <div class="caption" id="caption"></div>
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

  // Geometry, straight from the mp4 atoms.
  function geometry(pr) {
    if (!pr || !pr.frames) return '<span class="big bad">no measurement</span>';
    const ok = pr.frames === index.expected_frames && pr.fps === index.expected_fps;
    const txt = pr.frames + " f @ " + pr.fps + " fps = " +
      (pr.duration != null ? pr.duration.toFixed(3) + " s" : "?");
    return '<span class="big ' + (ok ? "ok" : "bad") + '">' + txt +
      (ok ? "  ✓" : "  ✗ want " + index.expected_frames + " f @ " +
        index.expected_fps + " fps") + "</span>";
  }

  // Uniformity: flat profile = every displayed frame advances 6 source frames.
  function uniformity(pf, kind) {
    if (!pf) return '<span class="big bad">uniformity: not computed</span>';
    if (pf.error) return '<span class="big bad">uniformity: ' + pf.error + "</span>";
    const good = pf.verdict === "uniform advance";
    let s = '<span class="big ' + (good ? "ok" : "bad") + '">ratio ' + pf.ratio +
      " — " + pf.verdict + "</span>" +
      "<span>mean phase " + pf.mean_phase + "</span>";
    if (!good && kind === "pred") {
      s += "<span>— expected here: the rollout emits K=2 chunks, so predicted " +
        "frames carry their own chunk-boundary step. The validation clip is the " +
        "read-stride diagnostic.</span>";
    }
    return s;
  }

  function column(kind, title, tag, item) {
    const col = document.createElement("div");
    col.className = "col " + kind;
    const h = document.createElement("h2");
    h.innerHTML = title + '<span class="tag">' + tag + "</span>";
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
    observer.observe(v);
    col.appendChild(v);

    const g = document.createElement("div");
    g.className = "readout";
    g.innerHTML = geometry(item.probe) +
      '<span class="fn">' + item.path.split("/").pop() + "</span>";
    col.appendChild(g);

    const u = document.createElement("div");
    u.className = "readout";
    u.innerHTML = uniformity(item.profile, kind);
    col.appendChild(u);

    if (item.profile && item.profile.phases) {
      const p = document.createElement("div");
      p.className = "phases";
      p.innerHTML = "phase means mod " + index.phase_period + ": " +
        item.profile.phases.map((x, i) =>
          i === item.profile.peak_phase ? "<b>" + x.toFixed(1) + "</b>" : x.toFixed(1)
        ).join(" ");
      col.appendChild(p);
    }
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
        : "no videos on the volume yet for this run";
      mainEl.appendChild(ph);
      return;
    }
    for (const p of pairs) {
      const row = document.createElement("div");
      row.className = "pair";
      row.appendChild(column("pred", "predicted — dream frames",
        "model's imagined future", p.predicted));
      row.appendChild(column("val", "validation — ground-truth clip",
        "read-stride diagnostic", p.validation));
      mainEl.appendChild(row);
    }
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

  // Start both clips together so they can be read against each other.
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

  function setCaption() {
    const c = index.contract;
    $("caption").innerHTML =
      "Run <code>" + index.run + "</code> — the first output where the frame stride " +
      "actually reaches the <b>zarr read</b>, so each displayed frame advances " +
      c.source_stride + " source frames.<br>" +
      "<b>Rate contract:</b> " + c.source_fps + " fps source → keep every " +
      c.source_stride + "th frame → display at " + c.display_fps + " fps, so the clip " +
      "spans the <b>same real time</b> as the " + c.source_fps + " fps original (" +
      c.expected_frames + " f @ " + c.display_fps + " fps = " + c.expected_duration_s +
      " s). One <b>K=2 chunk</b> = " + c.chunk_frames + " displayed frames = " +
      c.chunk_seconds + " s carrying <b>" + c.chunk_actions + " actions at " +
      c.action_hz + " Hz</b> — " + c.actions_per_frame + " actions consumed per " +
      "displayed frame, the overlay receding " + c.overlay_sequence +
      " and resetting at the boundary.<br>" +
      "<b>Reading the uniformity profile:</b> mean absolute inter-frame difference " +
      "averaged per phase mod " + index.phase_period + "; ratio (max/min) &lt; " +
      index.uniform_max_ratio + " = <b>uniform advance</b> (correct), ≥ " +
      index.seam_min_ratio + " = <b>seam jump</b>. The phase magnitudes here (~" +
      c.phase_mean_now + ") are <b>higher</b> than the pre-fix runs (~" +
      c.phase_mean_before + "): that increase is the <b>good</b> signal — " +
      "consecutive frames are now " + c.source_stride + " source frames apart so they " +
      "legitimately differ ~" + c.source_stride + "× more. It is not a regression.<br>" +
      '<span class="caveat">Caveat: a 4-epoch throwaway on 2% norm stats. The ' +
      "imagery is meaningless — the <b>pipeline</b> is what is being judged.</span>";
  }

  function signature(d) {
    return JSON.stringify([d.epochs, Object.keys(d.byEpoch || {}).length]);
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
        if (first) setCaption();
        buildSelector();
        const eps = index.epochs;
        setEpoch(eps.includes(keep) ? keep : (eps.length ? eps[eps.length - 1] : null));
        statusEl.textContent = eps.length
          ? eps.length + " epochs (" + eps.join(", ") + ") · refreshes every 60 s"
          : "no videos yet · refreshes every 60 s";
      })
      .catch((e) => { statusEl.textContent = "failed to load index: " + e; });
  }
  load();
  setInterval(load, 60000);
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
    # An mp4 never changes once written, so both measurements are cached
    # permanently by (path, size); only new files ever cost work.
    _probe_cache = {}
    _profile_cache = {}

    def _vid_sort_key(name):
        m = re.search(r"(\d+)\.mp4$", name)
        return (int(m.group(1)) if m else 1 << 30, name)

    def _probe(path, window=131072):
        """(duration, frames, fps, size) from the mp4 mvhd + stsz atoms.

        Reads only the first and last `window` bytes: torchvision/pyav writes
        moov at the END, other writers at the start, so both are searched.
        Verified against ffprobe on this run (336 f, 5 fps, 67.200 s).
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
                # a nonsense count (seen: 2.17e9 frames on one file).
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
            "duration": round(dur, 3) if dur else None,
            "frames": frames,
            "fps": fps,
            "size_mb": round(size / 1e6, 2),
        }
        _probe_cache[key] = out
        return out

    def _profile(path):
        """Temporal-uniformity profile.

        Decodes to small greyscale frames, takes the mean absolute inter-frame
        difference, then averages those per phase mod PHASE_PERIOD. Uniform
        stepping gives a flat profile (ratio < UNIFORM_MAX_RATIO); 30 fps content
        inside each window spikes at the seam phase.
        """
        try:
            size = os.path.getsize(path)
        except OSError:
            return None
        key = (path, size)
        if key in _profile_cache:
            return _profile_cache[key]

        import subprocess

        w, h = PROFILE_SIZE
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
            if n < PHASE_PERIOD + 2:
                out = {
                    "error": f"too short to profile ({n} frames vs "
                    f"{PHASE_PERIOD}-frame period)"
                }
            else:
                arr = (
                    np.frombuffer(r.stdout[: n * fsz], dtype=np.uint8)
                    .reshape(n, fsz)
                    .astype(np.int16)
                )
                dif = np.abs(np.diff(arr, axis=0)).mean(axis=1)
                phases = [
                    float(dif[i::PHASE_PERIOD].mean()) for i in range(PHASE_PERIOD)
                ]
                lo, hi = min(phases), max(phases)
                ratio = (hi / lo) if lo > 0 else None
                if ratio is None:
                    verdict = "inconclusive"
                elif ratio < UNIFORM_MAX_RATIO:
                    verdict = "uniform advance"
                elif ratio >= SEAM_MIN_RATIO:
                    verdict = "seam jump"
                else:
                    verdict = "inconclusive"
                out = {
                    "frames": n,
                    "phases": [round(x, 2) for x in phases],
                    "peak_phase": int(
                        max(range(PHASE_PERIOD), key=lambda i: phases[i])
                    ),
                    "mean_phase": round(sum(phases) / len(phases), 2),
                    "ratio": round(ratio, 2) if ratio else None,
                    "verdict": verdict,
                }
        except Exception as exc:  # decoder missing, timeout, malformed file
            out = {"error": f"{type(exc).__name__}: {exc}"[:160]}
        _profile_cache[key] = out
        return out

    def _item(emb_dir, rel_dir, name):
        full = os.path.join(emb_dir, name)
        return {
            "path": f"{rel_dir}/{name}",
            "probe": _probe(full),
            "profile": _profile(full),
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
                by_epoch[int(m.group(1))] = [
                    {
                        "predicted": pred[i] if i < len(pred) else None,
                        "validation": val[i] if i < len(val) else None,
                    }
                    for i in range(max(len(pred), len(val)))
                ]
        epochs = sorted(by_epoch)
        return {
            "run": RUN_LABEL,
            "run_dir": RUN_DIR,
            "epochs": epochs,
            "byEpoch": {str(e): by_epoch[e] for e in epochs},
            "expected_frames": EXPECTED_FRAMES,
            "expected_fps": EXPECTED_FPS,
            "phase_period": PHASE_PERIOD,
            "uniform_max_ratio": UNIFORM_MAX_RATIO,
            "seam_min_ratio": SEAM_MIN_RATIO,
            "contract": {
                "source_fps": 30,
                "source_stride": 6,
                "display_fps": EXPECTED_FPS,
                "expected_frames": EXPECTED_FRAMES,
                "expected_duration_s": EXPECTED_DURATION_S,
                "chunk_frames": 8,
                "chunk_seconds": 1.6,
                "chunk_actions": 48,
                "action_hz": 30,
                "actions_per_frame": 6,
                "overlay_sequence": "48 → 42 → … → 6",
                "phase_mean_now": 11.5,
                "phase_mean_before": 4.3,
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
        # Only this run's videos/ subtree is servable: every removed run, and
        # this run's own checkpoints/ and logs, are rejected.
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
