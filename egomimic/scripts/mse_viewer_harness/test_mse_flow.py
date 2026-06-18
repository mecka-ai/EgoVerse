"""End-to-end flow test for the MSE viewer (no node, no GPU, stdlib only).

Builds the viewer HTML from fake mse_scores via ``build_mse_viewer.build_html``,
writes ``/tmp/mse_viewer.html`` (consumed by ``test_mse_viewer_js.py``), and
round-trips a synthetic viewer selection through ``mse_apply_selection.py`` to
assert the union + keep-all-uncovered + provenance behavior.

Run:  python egomimic/scripts/mse_viewer_harness/test_mse_flow.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "egomimic" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_mse_viewer as bmv  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), name, "" if cond else f"-> {detail}")
    if not cond:
        FAIL.append(name)


# --- fake scoring artifacts ---------------------------------------------------
# Two tasks; "g" is in the universe but unscored (should default to KEEP).
SCORES = {
    "t1": [["a", 0.1], ["b", 0.2], ["c", 0.3], ["d", 0.4]],
    "t2": [["e", 1.0], ["f", 2.0]],
}
META = {"source": "test_ckpt", "metric": "paired_mse_unnorm", "higher_is_worse": False}
UNIVERSE = ["a", "b", "c", "d", "e", "f", "g"]

html = bmv.build_html(SCORES, scores_meta=META, universe=UNIVERSE)
out_html = Path("/tmp/mse_viewer.html")
out_html.write_text(html)
print(f"Wrote {out_html} ({len(html)} bytes)")

check("meta embedded (higher_is_worse=false)", '"higher_is_worse":false' in html)
check("scores embedded", '"t1":[["a",0.1]' in html)
check("universe embedded incl unscored g", '"g"' in html)
check("no plotly dependency (focused viewer)", "plotly" not in html.lower())
check(
    "video base placeholder filled", "__VIDEO_BASE__" not in html and "/video/" in html
)

# --- build_html accepts the dict form too -------------------------------------
html_dictform = bmv.build_html({"t1": {"a": 0.1, "b": 0.2}}, scores_meta=META)
check(
    "dict-form scores normalized to list", '"t1":[["a",0.1],["b",0.2]]' in html_dictform
)

# --- apply round-trip: synthetic selection -> eps_to_use ----------------------
selection = {
    "schema_version": 1,
    "created_at": "2026-01-01T00:00:00Z",
    "scores_meta": META,
    "universe": UNIVERSE,
    "tasks": {
        "t1": {
            "kept": ["a", "b"],
            "removed": [{"hash": "c", "score": 0.3}, {"hash": "d", "score": 0.4}],
            "pct": 50,
            "keep_dir": "lowest",
        },
        "t2": {
            "kept": ["e"],
            "removed": [{"hash": "f", "score": 2.0}],
            "pct": 50,
            "keep_dir": "lowest",
        },
    },
}
sel_path = Path(tempfile.mktemp(suffix=".json"))
sel_path.write_text(json.dumps(selection))
allow_path = Path(tempfile.mktemp(suffix=".json"))

proc = subprocess.run(
    [
        sys.executable,
        str(SCRIPTS / "mse_apply_selection.py"),
        "--selection",
        str(sel_path),
        "--out",
        str(allow_path),
    ],
    capture_output=True,
    text=True,
)
if proc.returncode != 0:
    print(proc.stdout)
    print(proc.stderr)
check("apply exits 0", proc.returncode == 0)

allow = json.load(open(allow_path))
# kept = union of t1{a,b} + t2{e} + uncovered universe {g}
check("allowlist union + keep-all-uncovered", allow == ["a", "b", "e", "g"], allow)
prov = Path(str(allow_path).removesuffix(".json") + ".provenance.json")
check("provenance sidecar written", prov.exists())
if prov.exists():
    pj = json.load(open(prov))
    check("provenance n_kept correct", pj["n_kept"] == 4, pj.get("n_kept"))
    check(
        "provenance uses embedded universe",
        pj["universe"] == "(embedded)",
        pj.get("universe"),
    )

print()
print("FAILURES:" if FAIL else "ALL FLOW CHECKS PASSED", FAIL or "")
sys.exit(1 if FAIL else 0)
