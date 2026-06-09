#!/usr/bin/env python3
"""
Build a self-contained interactive 3-D latent viewer from tsne3d_*.json files.

The curation run (curateModal.py) writes one ``tsne3d_<task>.json`` per task to
``<run>/tsne3d/`` on the training-outputs volume. This script bundles them into
a single HTML app (Plotly, no server needed):

  * task dropdown — one view per task
  * two linked 3-D scatters: state latents (left) and action latents (right)
  * one color per episode (same episode = same color in both panels)
  * points darken with frame index (early = light, late = dark)
  * click any point → exact episode hash + frame index + time-%
  * click a legend entry → isolate that episode in BOTH panels

Usage (local dir):
  python egomimic/scripts/build_latent_viz.py /path/to/tsne3d --out latent_viz.html

Usage (volume path — auto-downloads):
  python egomimic/scripts/build_latent_viz.py \\
      deminf_tsne/14task_dim64_k10_20_viz3d_<ts>/tsne3d \\
      --volume egoverse-training-outputs --env robotics --out latent_viz.html
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>EgoVerse latent viewer — 3-D t-SNE</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; background: #111; color: #eee; }
  #bar { padding: 10px 16px; display: flex; gap: 16px; align-items: center; background: #1c1c1c; }
  #bar select { font-size: 15px; padding: 4px 8px; }
  #info { font-size: 14px; padding: 6px 12px; background: #222; border-radius: 6px; min-width: 420px; }
  #info b { color: #7fd4ff; }
  #plots { display: flex; }
  .panel { width: 50vw; height: calc(100vh - 60px); }
  h3 { margin: 0; font-weight: 600; }
</style>
</head>
<body>
<div id="bar">
  <h3>Latent t-SNE (3-D)</h3>
  <label>Task: <select id="task"></select></label>
  <div id="info">Click a point to inspect it.</div>
</div>
<div id="plots">
  <div id="state" class="panel"></div>
  <div id="action" class="panel"></div>
</div>
<script>
const DATA = __DATA__;

function hsv2rgb(h, s, v) {
  const i = Math.floor(h * 6), f = h * 6 - i;
  const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  const c = [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i % 6];
  return `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`;
}

function tracesFor(mod, task) {
  const d = DATA[task][mod];
  if (!d) return [];
  const eps = DATA[task].episodes, nEp = eps.length;
  // bucket points by episode
  const byEp = {};
  for (let k = 0; k < d.x.length; k++) {
    const e = d.ep[k];
    if (!byEp[e]) byEp[e] = {x:[], y:[], z:[], c:[], f:[], t:[]};
    const b = byEp[e];
    b.x.push(d.x[k]); b.y.push(d.y[k]); b.z.push(d.z[k]);
    b.c.push(hsv2rgb(e / Math.max(1, nEp), 0.85, 1.0 - 0.65 * d.t[k]));
    b.f.push(d.frame[k]); b.t.push(d.t[k]);
  }
  return Object.keys(byEp).sort((a,b)=>a-b).map(e => {
    const b = byEp[e];
    return {
      type: "scatter3d", mode: "markers",
      name: eps[e].slice(0, 10),
      meta: {hash: eps[e], mod: mod},
      x: b.x, y: b.y, z: b.z,
      customdata: b.f.map((fr, i) => [fr, b.t[i]]),
      marker: {size: 3, color: b.c},
      hovertemplate: "ep " + eps[e].slice(0,10) +
        " · frame %{customdata[0]} (t=%{customdata[1]:.0%})<extra></extra>",
    };
  });
}

const LAYOUT = (title) => ({
  title: {text: title, font: {color: "#eee", size: 15}},
  paper_bgcolor: "#111", plot_bgcolor: "#111",
  scene: {
    xaxis: {visible: false}, yaxis: {visible: false}, zaxis: {visible: false},
    bgcolor: "#111",
  },
  legend: {font: {color: "#ccc", size: 10}, itemsizing: "constant"},
  margin: {l: 0, r: 0, t: 36, b: 0},
});

function render(task) {
  Plotly.react("state",  tracesFor("state", task),  LAYOUT("STATE — " + task),  {responsive: true});
  Plotly.react("action", tracesFor("action", task), LAYOUT("ACTION — " + task), {responsive: true});
  for (const div of ["state", "action"]) {
    const el = document.getElementById(div);
    el.removeAllListeners && el.removeAllListeners("plotly_click");
    el.on("plotly_click", ev => {
      const p = ev.points[0];
      const [frame, tfrac] = p.customdata;
      document.getElementById("info").innerHTML =
        `<b>${p.data.meta.mod.toUpperCase()}</b> · task <b>${task}</b> · ` +
        `episode <b>${p.data.meta.hash}</b> · frame <b>${frame}</b> ` +
        `(${Math.round(tfrac * 100)}% through episode)`;
    });
    // legend click → isolate the same episode in BOTH panels (click again to show all)
    el.on("plotly_legendclick", ev => {
      const name = ev.data[ev.curveNumber].name;
      const alreadyIsolated = ev.data.every(
        tr => tr.name === name ? tr.visible !== "legendonly" : tr.visible === "legendonly"
      );
      for (const d2 of ["state", "action"]) {
        const el2 = document.getElementById(d2);
        const vis = el2.data.map(tr => alreadyIsolated || tr.name === name ? true : "legendonly");
        Plotly.restyle(el2, {visible: vis}, el2.data.map((_, i) => i));
      }
      return false;
    });
  }
}

const sel = document.getElementById("task");
Object.keys(DATA).sort().forEach(t => {
  const o = document.createElement("option"); o.value = o.textContent = t; sel.appendChild(o);
});
sel.onchange = () => render(sel.value);
render(sel.value);
</script>
</body>
</html>
"""


def _load_dir(path: str, volume: str | None, env: str) -> Path:
    p = Path(path)
    if p.is_dir():
        return p
    if volume is None:
        sys.exit(f"Directory not found locally: {path}\nPass --volume to download from Modal.")
    tmp = Path(tempfile.mkdtemp(prefix="tsne3d_"))
    print(f"Downloading {path} from volume {volume} → {tmp} ...")
    subprocess.run(
        ["modal", "volume", "get", "--env", env, "--force", volume, path.rstrip("/"), str(tmp)],
        check=True,
    )
    # modal volume get may nest the dir name under tmp
    nested = tmp / Path(path.rstrip("/")).name
    return nested if nested.is_dir() else tmp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tsne3d_dir", help="Local dir of tsne3d_*.json, or volume-relative path")
    ap.add_argument("--volume", default=None, help="Modal volume to download from if not local")
    ap.add_argument("--env", default="robotics", help="Modal environment (default: robotics)")
    ap.add_argument("--out", default="latent_viz.html", help="Output HTML path")
    args = ap.parse_args()

    src = _load_dir(args.tsne3d_dir, args.volume, args.env)
    files = sorted(src.glob("tsne3d_*.json"))
    if not files:
        sys.exit(f"No tsne3d_*.json files in {src}")

    data: dict = {}
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        data[d["task"]] = d
    print(f"Loaded {len(data)} task(s): {', '.join(sorted(data))}")

    html = _TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = Path(args.out)
    out.write_text(html)
    n_pts = sum(
        len(d[m]["x"]) for d in data.values() for m in ("state", "action") if m in d
    )
    print(f"Wrote {out.resolve()}  ({n_pts:,} points, {out.stat().st_size/1e6:.1f} MB)")
    print("Open it in any browser — no server needed.")


if __name__ == "__main__":
    main()
