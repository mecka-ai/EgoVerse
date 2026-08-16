"""
Build a self-contained HTML val-comparison report from WAM offline-eval dirs.

Each ``--eval-dir label=path`` names one eval run (e.g. a TF and an AR pass of
``egomimic/modal/offline_val_wam.py``, downloaded locally with
``modal volume get --env robotics egoverse-training-outputs <run_dir> <dest>``).
Videos are discovered under ``<path>/videos/**/<EMBODIMENT>/
{predicted,validation}_video_<i>.mp4`` and COPIED (or symlinked with
``--symlink``) into the report directory:

    <REPORT_DIR>/
        index.html
        manifest.json
        videos/<label>/<embodiment>/{predicted,validation}_<i>.mp4

The HTML shows the selected evals side-by-side with a single Play/Pause +
seek bar and dropdowns for embodiment / video kind / episode. Playback is
time-synced: a video that runs ahead is pulled back if drift exceeds a
threshold. Zip the report dir for a portable handoff.

Usage:
    python egomimic/scripts/build_val_report.py \
        --eval-dir tf=modal-outputs/wam_offline_eval/wam22_dw48_tf_2026-08-15_12-00-00 \
        --eval-dir ar=modal-outputs/wam_offline_eval/wam22_dw48_ar_2026-08-15_12-30-00 \
        --out wam22_dw48_report [--symlink] [--zip]
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import zipfile
from pathlib import Path

_VIDEO_RE = re.compile(r"^(predicted|validation)_video_(\d+)\.mp4$")


def _collect_videos(src_root: Path, label: str, report_dir: Path, symlink: bool):
    """Copy/symlink one eval dir's mp4s into the report layout; return a nested
    ``{embodiment: {kind: {episode: rel_path}}}`` index (rel to report_dir)."""
    index: dict[str, dict[str, dict[str, str]]] = {}
    for mp4 in sorted(src_root.rglob("*.mp4")):
        m = _VIDEO_RE.match(mp4.name)
        if not m:
            continue
        kind, episode = m.group(1), m.group(2)
        embodiment = mp4.parent.name  # e.g. MECKA_BIMANUAL
        rel = Path("videos") / label / embodiment / f"{kind}_{episode}.mp4"
        dst = report_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        if symlink:
            dst.symlink_to(mp4.resolve())
        else:
            shutil.copy2(mp4, dst)
        index.setdefault(embodiment, {}).setdefault(kind, {})[episode] = str(rel)
    return index


_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>WAM val comparison</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #111; color: #eee; }}
  .controls {{ display: flex; gap: 1rem; align-items: center; flex-wrap: wrap; margin-bottom: 1rem; }}
  select, button {{ font-size: 1rem; padding: 0.3rem 0.6rem; }}
  input[type=range] {{ width: 22rem; }}
  .grid {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .cell {{ flex: 1 1 28rem; min-width: 20rem; }}
  .cell h3 {{ margin: 0 0 0.4rem 0; font-weight: 600; }}
  video {{ width: 100%; background: #000; }}
  .missing {{ color: #f66; padding: 2rem 0; }}
</style>
</head>
<body>
<h2>WAM val comparison</h2>
<div class="controls">
  <label>Embodiment <select id="emb">{emb_options}</select></label>
  <label>Kind <select id="kind">
    <option value="predicted">predicted (dream)</option>
    <option value="validation">validation (GT overlay)</option>
  </select></label>
  <label>Episode <select id="ep">{ep_options}</select></label>
  <button id="play">Play</button>
  <input type="range" id="seek" min="0" max="1000" value="0">
</div>
<div class="grid" id="grid">{cells}</div>
<script>
const INDEX = {index_json};
const LABELS = {labels_json};
let playing = false;
const vids = () => Array.from(document.querySelectorAll("video"));

function currentSel() {{
  return [document.getElementById("emb").value,
          document.getElementById("kind").value,
          document.getElementById("ep").value];
}}
function refresh() {{
  const [emb, kind, ep] = currentSel();
  for (const label of LABELS) {{
    const holder = document.getElementById("cell_" + label);
    const rel = (((INDEX[label] || {{}})[emb] || {{}})[kind] || {{}})[ep];
    holder.innerHTML = rel
      ? '<video src="' + rel + '" preload="auto" muted loop></video>'
      : '<div class="missing">missing</div>';
  }}
  playing = false;
  document.getElementById("play").textContent = "Play";
}}
document.getElementById("emb").onchange = refresh;
document.getElementById("kind").onchange = refresh;
document.getElementById("ep").onchange = refresh;
document.getElementById("play").onclick = () => {{
  playing = !playing;
  document.getElementById("play").textContent = playing ? "Pause" : "Play";
  vids().forEach(v => playing ? v.play() : v.pause());
}};
document.getElementById("seek").oninput = (e) => {{
  const frac = e.target.value / 1000;
  vids().forEach(v => {{ if (v.duration) v.currentTime = frac * v.duration; }});
}};
// Time-sync: pull the leader back if drift exceeds threshold.
setInterval(() => {{
  const vs = vids().filter(v => v.duration);
  if (vs.length < 2 || !playing) return;
  const tmin = Math.min(...vs.map(v => v.currentTime));
  vs.forEach(v => {{ if (v.currentTime - tmin > 0.15) v.currentTime = tmin; }});
}}, 500);
refresh();
</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-dir",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="one offline-eval output dir per flag, e.g. tf=/path/to/eval_tf_run",
    )
    ap.add_argument("--out", default="wam_val_report", help="report output dir")
    ap.add_argument(
        "--symlink",
        action="store_true",
        help="symlink mp4s instead of copying (not portable across machines)",
    )
    ap.add_argument("--zip", action="store_true", help="also write <out>.zip")
    args = ap.parse_args()

    report_dir = Path(args.out)
    report_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {}
    labels: list[str] = []
    for spec in args.eval_dir:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--eval-dir must be LABEL=PATH, got {spec!r}")
        src = Path(path)
        if not src.exists():
            raise SystemExit(f"eval dir not found: {src}")
        labels.append(label)
        index[label] = _collect_videos(src, label, report_dir, args.symlink)
        n = sum(
            len(eps) for kinds in index[label].values() for eps in kinds.values()
        )
        print(f"[report] {label}: {n} videos from {src}")

    embodiments = sorted({e for idx in index.values() for e in idx})
    episodes = sorted(
        {
            ep
            for idx in index.values()
            for kinds in idx.values()
            for eps in kinds.values()
            for ep in eps
        },
        key=int,
    )
    if not embodiments or not episodes:
        raise SystemExit("no {predicted,validation}_video_*.mp4 found in any eval dir")

    cells = "\n".join(
        f'<div class="cell"><h3>{html.escape(lb)}</h3>'
        f'<div id="cell_{html.escape(lb)}"></div></div>'
        for lb in labels
    )
    page = _HTML.format(
        emb_options="".join(f'<option value="{e}">{e}</option>' for e in embodiments),
        ep_options="".join(f'<option value="{e}">{e}</option>' for e in episodes),
        cells=cells,
        index_json=json.dumps(index),
        labels_json=json.dumps(labels),
    )
    (report_dir / "index.html").write_text(page)
    (report_dir / "manifest.json").write_text(
        json.dumps({"labels": labels, "index": index}, indent=2)
    )
    print(f"[report] wrote {report_dir / 'index.html'}")

    if args.zip:
        zip_path = report_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in report_dir.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(report_dir.parent))
        print(f"[report] wrote {zip_path}")


if __name__ == "__main__":
    main()
