#!/usr/bin/env python3
"""Build a self-contained web viewer for top-k% episode selection by MSE.

Input is an ``mse_scores.json`` (written by ``egomimic/modal/scoreMseModal.py``):

    {task: [[episode_hash, mse], ...], ...}

Optional siblings, auto-discovered next to it (or via flags):
  * ``scores_meta.json`` — {"source", "metric", "higher_is_worse"} (default
    higher_is_worse=false: low MSE = good).
  * ``episode_hashes.json`` — the full resolved episode universe; episodes that
    were not scored (skipped) default to KEEP in the exported allowlist.

The viewer (single self-contained HTML file, vanilla JS, no build step):
  * Per-task ranked grid of episode cards with MSE + percentile + lazy video
    preview (MP4 streamed from an episode_preview.py deployment).
  * A top-k% slider and a keep-direction toggle (lowest-MSE / highest-MSE,
    default lowest), an MSE histogram, search, sort, and kept/total readouts.
  * Export ``⬇ eps_to_use`` (a flat hash allowlist — drop into
    egomimic/hydra_configs/data/extra/<name>.json and point a data config's
    resolver.eps_to_use at it) or ``⬇ selection`` (a viewer_selection schema for
    egomimic/scripts/mse_apply_selection.py).

Usage (local file):
    python egomimic/scripts/build_mse_viewer.py /path/to/mse_scores.json --out mse_viewer.html

Usage (volume path — auto-downloads the run dir's siblings):
    python egomimic/scripts/build_mse_viewer.py \\
        mse_scores/<run>/<desc>_<ts>/mse_scores.json \\
        --volume egoverse-training-outputs --out mse_viewer.html
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

# Self-hosted Modal MP4 viewer (egomimic/modal/episode_preview.py) — serves raw
# H.264 MP4s, so the grid embeds native <video> players.
VIDEO_BASE = "https://mecka-robotics--egoverse-episode-preview-viewer.modal.run/video/"
FPS = 30

_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EgoVerse MSE viewer</title>
<style>
  :root { --bg:#101014; --bar:#1a1b21; --card:#1c1d24; --line:#2b2d36; --acc:#3b82f6; --txt:#e8e8ea; }
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; background: var(--bg); color: var(--txt); }
  #bar { padding: 10px 16px; display: flex; gap: 14px; align-items: center; background: var(--bar);
         flex-wrap: wrap; border-bottom: 1px solid var(--line); }
  h3 { margin: 0; font-weight: 700; letter-spacing:.3px; }
  select, input[type=number], input[type=text] { font-size: 13px; padding: 4px 8px; background:#26272f;
         color:var(--txt); border:1px solid var(--line); border-radius:6px; }
  input[type=number] { width: 64px; } input[type=text] { width: 150px; }
  input[type=range] { accent-color: var(--acc); width: 220px; vertical-align: middle; }
  label { color:#9aa; font-size:13px; }
  button { background: #2a2b33; color: #ddd; border: 1px solid var(--line); border-radius: 7px;
           padding: 5px 11px; cursor: pointer; font-size:13px; }
  button:hover { background:#33353f; }
  button.primary { background: var(--acc); border-color: var(--acc); color:#fff; }
  #stats { padding: 8px 16px; display:flex; gap:18px; align-items:center; flex-wrap:wrap;
           background:#15161b; border-bottom:1px solid var(--line); font-size:13px; color:#bbb; }
  #stats b { color:#7fd4ff; }
  #grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 14px; padding: 14px; }
  .card { background: var(--card); border-radius: 8px; padding: 8px; border:1px solid var(--line); }
  .card.dropped { opacity: 0.4; }
  .card video { width: 100%; border-radius: 4px; background:#000; aspect-ratio: 16/9; }
  .ph { width:100%; aspect-ratio:16/9; background:#000; border-radius:4px; display:flex;
        align-items:center; justify-content:center; color:#778; cursor:pointer; font-size:13px; text-align:center; }
  .meta { display:flex; justify-content:space-between; align-items:center; margin-top:6px; font-size:12px; }
  .h { font-family: ui-monospace, monospace; color:#9ad; word-break: break-all; }
  .mse { color:#cdd; font-variant-numeric: tabular-nums; }
  .badge { font-size:10px; padding:2px 6px; border-radius:5px; font-weight:700; }
  .b-keep { background:#14361f; color:#7be0a4; }
  .b-drop { background:#3a1620; color:#ff9b7b; }
  .pct { height:5px; border-radius:3px; background:#33353f; margin-top:5px; overflow:hidden; }
  .pct > div { height:100%; }
  #more { margin: 0 16px 24px; }
  svg#histo { background:#101014; border:1px solid var(--line); border-radius:4px; }
</style>
</head>
<body>
<div id="bar">
  <h3>MSE viewer</h3>
  <span><label>task</label> <select id="task"></select></span>
  <span><label>keep</label>
    <select id="keepDir">
      <option value="lowest" selected>lowest MSE</option>
      <option value="highest">highest MSE</option>
    </select>
  </span>
  <span><label>top</label>
    <input type="range" id="keepPct" min="0" max="100" step="0.5" value="100">
    <input type="number" id="keepPctN" min="0" max="100" step="0.5" value="100"><label>%</label>
  </span>
  <span><label>sort</label>
    <select id="sort">
      <option value="mse_asc" selected>MSE ↑</option>
      <option value="mse_desc">MSE ↓</option>
      <option value="kept">kept first</option>
    </select>
  </span>
  <span><label>show</label>
    <select id="show"><option value="all">all</option><option value="kept">kept</option><option value="dropped">dropped</option></select>
  </span>
  <span><input type="text" id="search" placeholder="hash prefix…"></span>
  <button class="primary" onclick="exportEpsToUse()">⬇ eps_to_use</button>
  <button onclick="exportSelection()">⬇ selection</button>
</div>
<div id="stats"></div>
<div id="grid"></div>
<button id="more" onclick="showMore()" style="display:none">show more</button>

<script>
const SCORES = __SCORES__;            // {task: [[hash, mse], ...]} ascending
const SCORES_META = __SCORES_META__;  // {higher_is_worse, metric, source}
const UNIVERSE = __UNIVERSE__;        // flat list of all resolved hashes (may be [])
const MANIFEST = __MANIFEST__;
const VIDEO_BASE = "__VIDEO_BASE__";
const FPS = __FPS__;
const HIGHER_WORSE = !!(SCORES_META && SCORES_META.higher_is_worse);
const PAGE = 300;                     // DOM cards rendered per page (selection runs over all)

/* ---------------- pure selection core (quickjs-testable, no DOM) ---------------- */
function _keepN(n, k) { return Math.max(0, Math.min(n, Math.ceil(n * k / 100))); }

function selectKept(scores, k, dir) {
  // scores: [[hash, mse], ...]; k in [0,100]; dir in {"lowest","highest"}.
  // Sort ascending by RAW mse (independent of higher_is_worse); keep the first
  // (lowest) or last (highest) ceil(n*k/100). Non-finite MSE is always kept
  // (unscored episodes are never silently dropped). Tie-break by hash.
  const finite = scores.filter(x => Number.isFinite(x[1]));
  const nonfinite = scores.filter(x => !Number.isFinite(x[1]));
  finite.sort((a, b) => (a[1] - b[1]) || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  const n = finite.length, keepN = _keepN(n, k);
  const keepArr = (dir === "highest") ? finite.slice(n - keepN) : finite.slice(0, keepN);
  const kept = new Set(keepArr.map(x => x[0]));
  for (const x of nonfinite) kept.add(x[0]);
  return kept;
}

function countKept(scores, k, dir) {
  const kept = selectKept(scores, k, dir);
  return { kept: kept.size, total: scores.length, pct: scores.length ? 100 * kept.size / scores.length : 0 };
}

function keptHashesAllTasks(allScores, k, dir, universe) {
  // Union of per-task kept sets; universe hashes not covered by any task
  // (unscored) default to KEEP. Returns a sorted flat list (eps_to_use-ready).
  const kept = new Set(), covered = new Set();
  for (const t of Object.keys(allScores)) {
    for (const pair of allScores[t]) covered.add(pair[0]);
    for (const h of selectKept(allScores[t], k, dir)) kept.add(h);
  }
  for (const h of (universe || [])) if (!covered.has(h)) kept.add(h);
  return Array.from(kept).sort();
}

function percentileNorm(scores) {
  // {hash: goodness in [0,1]} via midrank percentile; direction-corrected so
  // 1 = best (low MSE when higher_is_worse=false). Non-finite omitted.
  const e = scores.filter(x => Number.isFinite(x[1]));
  if (!e.length) return {};
  const sorted = e.map(x => x[1]).sort((a, b) => a - b);
  const out = {};
  for (const [h, s] of e) {
    let lo = 0, hi = sorted.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (sorted[m] < s) lo = m + 1; else hi = m; }
    let lo2 = lo, hi2 = sorted.length;
    while (lo2 < hi2) { const m = (lo2 + hi2) >> 1; if (sorted[m] <= s) lo2 = m + 1; else hi2 = m; }
    const pct = sorted.length === 1 ? 0.5 : (lo + lo2 - 1) / 2 / (sorted.length - 1);
    out[h] = HIGHER_WORSE ? pct : 1 - pct;  // goodness: low raw MSE -> high goodness
  }
  return out;
}

/* ---------------- DOM / rendering ---------------- */
let _page = 1, _curTask = null;

function el(id) { return document.getElementById(id); }
function uiState() {
  return {
    task: el("task").value,
    k: parseFloat(el("keepPct").value),
    dir: el("keepDir").value,
    sort: el("sort").value,
    show: el("show").value,
    q: el("search").value.trim().toLowerCase(),
  };
}

function histoSVG(vals) {
  if (!vals.length) return "";
  const mn = Math.min(...vals), mx = Math.max(...vals), nb = 24;
  const bins = new Array(nb).fill(0);
  vals.forEach(s => bins[Math.min(nb - 1, Math.floor((s - mn) / ((mx - mn) || 1) * nb))]++);
  const bmax = Math.max(...bins);
  const bars = bins.map((b, i) =>
    `<rect x="${i * 8}" y="${28 - 26 * b / bmax}" width="6" height="${26 * b / bmax}" fill="#3b82f6"/>`).join("");
  return `<svg id="histo" width="${nb * 8}" height="30" title="MSE distribution">${bars}</svg>`;
}

function goodnessColor(g) {
  // g in [0,1], 1=best -> green, 0=worst -> red
  const r = Math.round(255 * (1 - g)), gr = Math.round(180 * g);
  return `rgb(${r},${gr},90)`;
}

function loadVideo(cellId, hash) {
  const cell = el(cellId);
  const v = document.createElement("video");
  v.src = VIDEO_BASE + hash;
  v.controls = true; v.loop = true; v.muted = true; v.preload = "metadata";
  v.setAttribute("playsinline", "");
  v.onerror = () => {
    cell.outerHTML = '<div class="ph" style="cursor:default">⚠ video unavailable<br>' +
      '<span style="font-size:11px">render episode_preview.py for this run\'s episodes</span></div>';
  };
  cell.replaceWith(v);
}

function render() {
  const st = uiState();
  _curTask = st.task;
  const scores = SCORES[st.task] || [];
  const kept = selectKept(scores, st.k, st.dir);
  const gn = percentileNorm(scores);

  // stats line
  const finite = scores.filter(x => Number.isFinite(x[1])).map(x => x[1]);
  const c = countKept(scores, st.k, st.dir);
  const total = keptHashesAllTasks(SCORES, st.k, st.dir, UNIVERSE).length;
  const mean = finite.length ? finite.reduce((a, b) => a + b, 0) / finite.length : NaN;
  el("stats").innerHTML =
    `kept <b>${c.kept}</b> / ${c.total} (${c.pct.toFixed(1)}%) in <b>${st.task}</b>` +
    ` &nbsp;·&nbsp; all-tasks allowlist: <b>${total}</b> episodes` +
    ` &nbsp;·&nbsp; MSE mean <b>${isFinite(mean) ? mean.toFixed(5) : "—"}</b>` +
    (finite.length ? ` range [${Math.min(...finite).toFixed(5)}, ${Math.max(...finite).toFixed(5)}]` : "") +
    ` &nbsp; ${histoSVG(finite)}`;

  // filter + sort
  let rows = scores.map(([h, m], i) => ({ h, m, rank: i, keep: kept.has(h) }));
  if (st.q) rows = rows.filter(r => r.h.toLowerCase().startsWith(st.q));
  if (st.show === "kept") rows = rows.filter(r => r.keep);
  else if (st.show === "dropped") rows = rows.filter(r => !r.keep);
  if (st.sort === "mse_desc") rows.sort((a, b) => b.m - a.m);
  else if (st.sort === "kept") rows.sort((a, b) => (b.keep - a.keep) || (a.m - b.m));
  else rows.sort((a, b) => a.m - b.m);

  const shown = rows.slice(0, _page * PAGE);
  el("grid").innerHTML = shown.map((r, i) => {
    const cid = `cell_${i}`;
    const g = gn[r.h];
    const pctBar = (g == null) ? "" :
      `<div class="pct"><div style="width:${Math.round(g * 100)}%;background:${goodnessColor(g)}"></div></div>`;
    return `<div class="card ${r.keep ? "" : "dropped"}">
      <div class="ph" id="${cid}" onclick="loadVideo('${cid}','${r.h}')">▶ load video</div>
      <div class="meta">
        <span class="h">${r.h.slice(0, 16)}</span>
        <span class="mse">${Number.isFinite(r.m) ? r.m.toFixed(5) : "n/a"}
          <span class="badge ${r.keep ? "b-keep" : "b-drop"}">${r.keep ? "KEEP" : "DROP"}</span></span>
      </div>${pctBar}
      <div class="meta"><span style="color:#667">#${r.rank}</span>
        <a href="${VIDEO_BASE}${r.h}" target="_blank" style="color:#7fd4ff">open ↗</a></div>
    </div>`;
  }).join("");
  el("more").style.display = (rows.length > shown.length) ? "block" : "none";
  el("more").textContent = `show more (${shown.length} / ${rows.length})`;
}

function showMore() { _page++; render(); }

function downloadJSON(obj, name) {
  const blob = new Blob([JSON.stringify(obj)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
}
function stamp() { return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19); }

function exportEpsToUse() {
  const st = uiState();
  const kept = keptHashesAllTasks(SCORES, st.k, st.dir, UNIVERSE);
  downloadJSON(kept, `eps_to_use_${stamp()}.json`);
}

function exportSelection() {
  const st = uiState();
  const tasks = {};
  for (const t of Object.keys(SCORES)) {
    const kept = selectKept(SCORES[t], st.k, st.dir);
    const keptList = [], removed = [];
    for (const [h, m] of SCORES[t]) {
      if (kept.has(h)) keptList.push(h);
      else removed.push({ hash: h, score: m });
    }
    tasks[t] = { kept: keptList.sort(), removed, pct: st.k, keep_dir: st.dir };
  }
  downloadJSON({
    schema_version: 1,
    created_at: new Date().toISOString(),
    scores_meta: SCORES_META,
    manifest: MANIFEST,
    universe: UNIVERSE,
    tasks,
  }, `viewer_selection_${stamp()}.json`);
}

function syncPct(src) {
  const v = src === "n" ? el("keepPctN").value : el("keepPct").value;
  el("keepPct").value = v; el("keepPctN").value = v;
  _page = 1; render();
}

function init() {
  const sel = el("task");
  sel.innerHTML = Object.keys(SCORES).sort().map(t => `<option value="${t}">${t}</option>`).join("");
  el("keepPct").oninput = () => syncPct("r");
  el("keepPctN").oninput = () => syncPct("n");
  for (const id of ["task", "keepDir", "sort", "show"]) el(id).onchange = () => { _page = 1; render(); };
  el("search").oninput = () => { _page = 1; render(); };
  render();
}
if (typeof document !== "undefined") init();
</script>
</body>
</html>
"""


def _normalize_scores(scores_raw: dict, higher_is_worse: bool) -> dict:
    """Accept list form ({task:[[h,mse],...]}) or dict form ({task:{h:mse}}).

    Returns {task: [[hash, mse], ...]} sorted ascending by raw MSE (non-finite
    last, tie-break by hash) — best-first for the default higher_is_worse=false.
    """
    out: dict[str, list] = {}
    for task, v in (scores_raw or {}).items():
        items = list(v.items()) if isinstance(v, dict) else [(h, m) for h, m in v]
        items = [(h, float(m)) for h, m in items]
        items.sort(
            key=lambda kv: (
                not math.isfinite(kv[1]),
                kv[1] if math.isfinite(kv[1]) else 0.0,
                kv[0],
            )
        )
        out[task] = [[h, m] for h, m in items]
    return out


def build_html(
    scores_raw: dict,
    scores_meta: dict | None = None,
    universe: list | dict | None = None,
    manifest: dict | None = None,
    video_base: str = VIDEO_BASE,
) -> str:
    """Assemble the viewer HTML from an mse_scores mapping.

    ``scores_raw`` is ``{task: [[hash, mse], ...]}`` (or the dict form). ``universe``
    is the full resolved episode list (flat list, or {task: [hash,...]} which is
    flattened); unscored universe episodes default to KEEP in exports.
    """
    meta = dict(scores_meta or {})
    meta.setdefault("higher_is_worse", False)
    scores = _normalize_scores(scores_raw, bool(meta["higher_is_worse"]))

    uni = universe or []
    if isinstance(uni, dict):
        uni = sorted({h for hs in uni.values() for h in hs})
    else:
        uni = sorted(set(uni))

    return (
        _TEMPLATE.replace("__SCORES__", json.dumps(scores, separators=(",", ":")))
        .replace("__SCORES_META__", json.dumps(meta, separators=(",", ":")))
        .replace("__UNIVERSE__", json.dumps(uni, separators=(",", ":")))
        .replace("__MANIFEST__", json.dumps(manifest or {}, separators=(",", ":")))
        .replace("__VIDEO_BASE__", video_base.rstrip("/") + "/")
        .replace("__FPS__", str(FPS))
    )


def _download(volume: str, env: str, remote: str, dest: Path) -> bool:
    r = subprocess.run(
        ["modal", "volume", "get", "--env", env, "--force", volume, remote, str(dest)],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _resolve_scores(path: str, volume: str | None, env: str) -> tuple[dict, Path]:
    """Return (scores dict, local dir holding it) — downloading from a volume if needed."""
    p = Path(path)
    if p.is_file():
        return json.load(open(p)), p.parent
    if volume is None:
        sys.exit(f"Not found locally: {path}\nPass --volume to download from Modal.")
    tmp = Path(tempfile.mkdtemp(prefix="mse_scores_"))
    dest = tmp / "mse_scores.json"
    if not _download(volume, env, path, dest):
        sys.exit(f"Could not download {path} from volume {volume}")
    # Pull the run dir's siblings too (scores_meta / episode_hashes) into tmp.
    parent = str(Path(path).parent)
    for sib in ("scores_meta.json", "episode_hashes.json"):
        _download(volume, env, f"{parent}/{sib}", tmp / sib)
    return json.load(open(dest)), tmp


def _load_sibling(local_dir: Path, explicit: str | None, filename: str):
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return json.load(open(p))
        sys.exit(f"--{filename} not found: {explicit}")
    sib = local_dir / filename
    return json.load(open(sib)) if sib.is_file() else None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scores", help="Local mse_scores.json, or a volume-relative path")
    ap.add_argument(
        "--scores-meta",
        default=None,
        help="Optional scores_meta.json (default: sibling)",
    )
    ap.add_argument(
        "--universe",
        default=None,
        help="Optional episode_hashes.json (default: sibling)",
    )
    ap.add_argument(
        "--video-base",
        default=VIDEO_BASE,
        help="Base /video/ URL of an episode_preview.py deploy",
    )
    ap.add_argument(
        "--volume",
        default=None,
        help="Modal volume to download from if paths are not local",
    )
    ap.add_argument(
        "--env", default="robotics", help="Modal environment (default: robotics)"
    )
    ap.add_argument("--out", default="mse_viewer.html", help="Output HTML path")
    args = ap.parse_args()

    scores_raw, local_dir = _resolve_scores(args.scores, args.volume, args.env)
    scores_meta = _load_sibling(local_dir, args.scores_meta, "scores_meta.json")
    universe = _load_sibling(local_dir, args.universe, "episode_hashes.json")
    if scores_meta:
        print(
            f"scores_meta: {scores_meta.get('source', '?')} · "
            f"higher_is_worse={scores_meta.get('higher_is_worse')}"
        )
    if universe:
        print(f"universe: {len(universe)} episodes")

    html = build_html(
        scores_raw,
        scores_meta=scores_meta,
        universe=universe,
        video_base=args.video_base,
    )
    out = Path(args.out)
    out.write_text(html)
    print(f"Wrote {out.resolve()}  ({out.stat().st_size / 1e6:.2f} MB)")
    print("Open it in any browser — no server needed.")


if __name__ == "__main__":
    main()
