"""Execute the MSE viewer JS under quickjs — run with /usr/bin/python3.12.

Needs /tmp/mse_viewer.html (from test_mse_flow.py) and quickjs in /tmp/jsengine2
(pip install --target /tmp/jsengine2 quickjs).

Asserts the pure selection core (selectKept / countKept / keptHashesAllTasks /
percentileNorm) for both keep directions and edge cases, that render() and the
exports run without throwing, and that the exported eps_to_use / selection match
the embedded scores.
"""

import json
import re
import sys

sys.path.insert(0, "/tmp/jsengine2")
import quickjs

html = open("/tmp/mse_viewer.html").read()
script = re.search(r"<script>\n(.*?)</script>", html, re.S).group(1)

STUBS = r"""
var DOWNLOADS = [];
function setTimeout(f, t) { f(); }
function Blob(parts, opts) { this.text = parts.join(""); }
var URL = { createObjectURL: function(b) { return {__blob: b}; }, revokeObjectURL: function() {} };
function makeEl(id) {
  return {
    id: id, value: "", checked: false, innerHTML: "", textContent: "",
    style: {}, download: "", href: null, src: "",
    classList: {add:function(){}, remove:function(){}, toggle:function(){}},
    appendChild: function(){}, removeChild: function(){}, replaceWith: function(){},
    setAttribute: function(){}, click: function(){ if (this.download) DOWNLOADS.push({name:this.download, text:this.href.__blob.text}); },
    onchange: null, oninput: null, onclick: null, onerror: null,
  };
}
var ELS = {};
function getEl(id){ if (!ELS[id]) ELS[id] = makeEl(id); return ELS[id]; }
var document = {
  body: makeEl("body"),
  getElementById: function(id){ return getEl(id); },
  querySelectorAll: function(){ return []; },
  createElement: function(tag){ return makeEl("tag:"+tag); },
};
"""

CHECKS = r"""
var out = {};

// --- pure selection core ---
out.lowHalf  = Array.from(selectKept(SCORES["t1"], 50, "lowest")).sort().join(",");   // a,b
out.highHalf = Array.from(selectKept(SCORES["t1"], 50, "highest")).sort().join(",");  // c,d
out.all100   = selectKept(SCORES["t1"], 100, "lowest").size;                          // 4
out.none0    = selectKept(SCORES["t1"], 0, "lowest").size;                            // 0

// non-finite (unscored) is ALWAYS kept, even at k=0
var nf = selectKept([["x", 0.5], ["y", null]], 0, "lowest");
out.nonFiniteKept = nf.has("y") && !nf.has("x");

// deterministic tie-break by hash
out.tieBreak = Array.from(selectKept([["b",0.5],["a",0.5],["c",0.1]], 50, "lowest")).sort().join(",");  // a,c

// count
var c = countKept(SCORES["t1"], 50, "lowest");
out.countPct = c.kept + "/" + c.total;   // 2/4

// all-tasks union + keep-all-uncovered (g unscored)
out.allTasks = keptHashesAllTasks(SCORES, 50, "lowest", UNIVERSE).join(",");  // a,b,e,g
out.allTasksHigh = keptHashesAllTasks(SCORES, 50, "highest", UNIVERSE).join(","); // c,d,f,g

// percentile goodness: lowest MSE 'a' is best (higher_is_worse=false)
var gn = percentileNorm(SCORES["t1"]);
out.goodnessAGtD = gn["a"] > gn["d"];

// --- render + DOM (no throw) ---
getEl("task").value = "t1";
getEl("keepPct").value = "50"; getEl("keepPctN").value = "50";
getEl("keepDir").value = "lowest"; getEl("sort").value = "mse_asc";
getEl("show").value = "all"; getEl("search").value = "";
render();
out.renderNoThrow = true;
out.gridHasKeep = getEl("grid").innerHTML.indexOf("KEEP") >= 0;
out.gridHasDrop = getEl("grid").innerHTML.indexOf("DROP") >= 0;
out.statsNaNFree = getEl("stats").innerHTML.indexOf("NaN") < 0;

// --- exports ---
exportEpsToUse();
out.epsList = JSON.parse(DOWNLOADS[DOWNLOADS.length-1].text).join(",");  // a,b,e,g

exportSelection();
var sel = JSON.parse(DOWNLOADS[DOWNLOADS.length-1].text);
out.selSchema = sel.schema_version === 1 && !!sel.tasks.t1 && !!sel.tasks.t2;
out.selMetaFalse = sel.scores_meta.higher_is_worse === false;
out.selKeptT1 = sel.tasks.t1.kept.join(",");                       // a,b
out.selRemovedT1 = sel.tasks.t1.removed.map(function(x){return x.hash;}).join(","); // c,d
out.selKeepDir = sel.tasks.t1.keep_dir;                            // lowest

JSON.stringify(out);
"""

ctx = quickjs.Context()
try:
    result = ctx.eval(STUBS + script + CHECKS)
except quickjs.JSException as e:
    print("FAIL: viewer JS threw:", e)
    sys.exit(1)

out = json.loads(result)
FAIL = []


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), name, "" if cond else f"-> {detail}")
    if not cond:
        FAIL.append(name)


check("selectKept lowest 50% = a,b", out["lowHalf"] == "a,b", out["lowHalf"])
check("selectKept highest 50% = c,d", out["highHalf"] == "c,d", out["highHalf"])
check("selectKept 100% keeps all 4", out["all100"] == 4, out["all100"])
check("selectKept 0% keeps none (finite)", out["none0"] == 0, out["none0"])
check("non-finite always kept (even at 0%)", out["nonFiniteKept"])
check("deterministic tie-break by hash", out["tieBreak"] == "a,c", out["tieBreak"])
check("countKept 2/4", out["countPct"] == "2/4", out["countPct"])
check(
    "all-tasks lowest union + keep-uncovered",
    out["allTasks"] == "a,b,e,g",
    out["allTasks"],
)
check(
    "all-tasks highest union + keep-uncovered",
    out["allTasksHigh"] == "c,d,f,g",
    out["allTasksHigh"],
)
check("percentile goodness low-MSE is best", out["goodnessAGtD"])
check("render() no throw", out["renderNoThrow"])
check("grid shows KEEP + DROP badges", out["gridHasKeep"] and out["gridHasDrop"])
check("stats line NaN-free", out["statsNaNFree"])
check("exportEpsToUse = sorted union", out["epsList"] == "a,b,e,g", out["epsList"])
check("selection schema_version 1 + tasks", out["selSchema"])
check("selection carries scores_meta", out["selMetaFalse"])
check("selection kept t1 = a,b", out["selKeptT1"] == "a,b", out["selKeptT1"])
check("selection removed t1 = c,d", out["selRemovedT1"] == "c,d", out["selRemovedT1"])
check("selection keep_dir recorded", out["selKeepDir"] == "lowest", out["selKeepDir"])

print()
print("FAILURES:" if FAIL else "ALL MSE-VIEWER JS CHECKS PASSED", FAIL or "")
sys.exit(1 if FAIL else 0)
