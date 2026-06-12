"""End-to-end repro of the viewer-driven prune flow (run with the project venv).

fake knn run + fake viz run -> review_tools pair -> build_html (writes
/tmp/prune_viewer.html for the quickjs harness) -> apply -> calibrate.
"""

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

WT = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, WT)

FAIL = []


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), name, detail if not cond else "")
    if not cond:
        FAIL.append(name)


root = Path(tempfile.mkdtemp(prefix="prune_flow_"))
viz = root / "viz_run"
knn = root / "knn_run"
(viz / "tsne3d").mkdir(parents=True)
knn.mkdir()

# Universe: 8 episodes in group g1 (raw frames), 2 in g2. One g1 episode
# (h07) failed embedding -> in universe but NOT in tsne3d.
g1 = [f"h{i:02d}" for i in range(8)]
g2 = ["x00", "x01"]
universe = {
    "g1": {h: 3000 + 100 * i for i, h in enumerate(g1)},
    "g2": {h: 2000 for h in g2},
}
json.dump(universe, open(viz / "group_universe.json", "w"))
json.dump(sorted(g1 + g2), open(viz / "episode_hashes.json", "w"))
json.dump({"name": "t", "groups": {"g1": {}}, "group_by": "task"}, open(viz / "viz_manifest.json", "w"))

# tsne3d for g1: 7 embedded episodes (h07 missing), 2 well-separated blobs of
# episodes -> k=2 clustering is unambiguous.
emb = [h for h in g1 if h != "h07"]
pts_per = 30
state = {"x": [], "y": [], "z": [], "ep": [], "frame": [], "t": []}
for e, h in enumerate(emb):
    blob = 0 if e < 4 else 1          # h00-h03 cluster A, h04-h06 cluster B
    for j in range(pts_per):
        state["x"].append(round(blob * 50 + (j % 5) * 0.1, 3))
        state["y"].append(round((j // 5) * 0.1, 3))
        state["z"].append(round(e * 0.01, 3))
        state["ep"].append(e)
        state["frame"].append(j * 10)
        state["t"].append(round(j / (pts_per - 1), 4))
json.dump(
    # action mirrors state so dual-panel paths (cross-highlight, cluster
    # color tables for the other modality) are exercised.
    {"task": "g1", "episodes": emb, "every_n": 10, "state": state, "action": state},
    open(viz / "tsne3d" / "tsne3d_g1.json", "w"),
)

# knn grading run: raw SQL task name with a SPACE ("task one") to prove the
# hash-rekey path; h00 worst, descending; h05/h06 unscored; h07 scored 0.05.
raw_scores = {"task one": {h: round(0.15 - 0.02 * i, 4) for i, h in enumerate(g1[:5] + ["h07"])},
              "other task": {g2[0]: 0.5}}
json.dump(raw_scores, open(knn / "knn_scores_by_task.json", "w"))
json.dump({"source": "knn_grading", "metric": "primary_score", "higher_is_worse": True},
          open(knn / "scores_meta.json", "w"))

# ---- pair (local-dir mode) -------------------------------------------------
r = subprocess.run(
    [sys.executable, f"{WT}/egomimic/scripts/knn_grading/review_tools.py", "pair",
     "--viz-run", str(viz), "--knn-run", str(knn)],
    capture_output=True, text=True,
)
check("pair exits 0", r.returncode == 0, r.stderr[-300:])
paired = json.load(open(viz / "scores_by_task.json"))
check("pair rekeys by hash into viz groups",
      set(paired) == {"g1", "g2"} and set(paired["g1"]) == {"h00", "h01", "h02", "h03", "h04", "h07"}
      and paired["g2"] == {"x00": 0.5},
      json.dumps(paired)[:200])
meta = json.load(open(viz / "scores_meta.json"))
check("pair meta direction + counts", meta["higher_is_worse"] is True and meta["n_matched"] == 7)

# ---- build_html with all sidecars -------------------------------------------
from egomimic.scripts.build_latent_viz import build_html

html = build_html(
    viz / "tsne3d",
    scores_raw=paired,
    val=None,
    scores_meta=meta,
    universe=universe,
    manifest=json.load(open(viz / "viz_manifest.json")),
)
Path("/tmp/prune_viewer.html").write_text(html)
check("html embeds direction meta", '"higher_is_worse":true' in html)
check("html embeds universe", '"h07":3700' in html)

# Best-first python sort under higher_is_worse: h07 (0.05) is the lowest =
# BEST scored g1 episode -> first entry; h00 (0.15, worst) -> last scored.
sc_g1 = json.loads(html.split("const SCORES = ")[1].split(";\n")[0])["g1"]
check("python rank best-first under higher_is_worse",
      sc_g1[0][0] == "h07" and sc_g1[-1][0] == "h00", str(sc_g1))

# ---- apply on a synthetic viewer selection ----------------------------------
sel = {
    "schema_version": 1,
    "created_at": "2026-06-11T00:00:00Z",
    "scores_meta": meta,
    "tasks": {
        "g1": {
            "kept": sorted(set(g1) - {"h00", "h01"}),
            "removed": [{"hash": "h00", "score": 0.15, "cls": 2, "frames": 3000},
                         {"hash": "h01", "score": 0.13, "cls": 2, "frames": 3100}],
            "labels": {"h00": "bad", "h04": "good"},
        }
    },
}
sel_path = root / "viewer_selection_test.json"
json.dump(sel, open(sel_path, "w"))
out_path = root / "extra" / "subset.json"
r = subprocess.run(
    [sys.executable, f"{WT}/egomimic/scripts/knn_grading/review_tools.py", "apply",
     "--selection", str(sel_path), "--universe", str(viz / "episode_hashes.json"),
     "--out", str(out_path)],
    capture_output=True, text=True,
)
check("apply exits 0", r.returncode == 0, r.stderr[-300:])
allow = json.load(open(out_path))
check("apply: union + default-keep uncovered (g2)",
      set(allow) == set(g1 + g2) - {"h00", "h01"}, str(allow))
prov = json.loads((root / "extra" / "subset.provenance.json").read_text())
check("apply provenance sidecar", prov["n_kept"] == len(allow) and prov["n_default_kept"] == 2
      and prov["sources"][0]["sha256"])

# calibrate from selection labels
report = {"task one": {"per_episode": {h: {"primary_score": raw_scores["task one"].get(h, math.nan)} for h in g1}}}
rp = root / "report.json"
json.dump(report, open(rp, "w"))
r = subprocess.run(
    [sys.executable, f"{WT}/egomimic/scripts/knn_grading/review_tools.py", "calibrate",
     "--report", str(rp), "--selection-labels", str(sel_path)],
    capture_output=True, text=True,
)
check("calibrate accepts selection labels", r.returncode == 0 and "Recommended" in r.stdout,
      (r.stdout + r.stderr)[-300:])

print()
print("FAILURES:" if FAIL else "ALL PRUNE-FLOW REPROS PASSED", FAIL or "")
sys.exit(1 if FAIL else 0)
