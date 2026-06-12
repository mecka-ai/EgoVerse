"""Execute the prune-tuning viewer JS (quickjs) — run with /usr/bin/python3.12.

Needs /tmp/prune_viewer.html from test_prune_flow.py and quickjs in
/tmp/jsengine2 (pip install --target /tmp/jsengine2 quickjs).
"""

import json
import re
import sys

sys.path.insert(0, "/tmp/jsengine2")
import quickjs

html = open("/tmp/prune_viewer.html").read()
script = re.search(r"<script>\n(.*?)</script>", html, re.S).group(1)

STUBS = r"""
var __errors = [];
var DOWNLOADS = [];
var ALERTS = [];
function alert(m) { ALERTS.push(String(m)); }
function setTimeout(f, t) { f(); }
function Blob(parts, opts) { this.text = parts.join(""); }
var URL = { createObjectURL: function(b) { return {__blob: b}; }, revokeObjectURL: function() {} };
function makeEl(id) {
  var el = {
    id: id, value: "", checked: false, innerHTML: "", textContent: "",
    style: {}, disabled: false, files: null, download: "", href: null, src: "",
    classList: {add: function(){}, remove: function(){}, toggle: function(){}},
    children: [],
    appendChild: function(c){ this.children.push(c); if (this.value === "") this.value = c.value; },
    removeChild: function(){},
    _handlers: {},
    on: function(name, cb){ this._handlers[name] = cb; },
    removeAllListeners: function(){},
    click: function(){ if (this.download) DOWNLOADS.push({name: this.download, text: this.href.__blob.text}); },
    pause: function(){}, play: function(){ return {catch: function(){}}; },
    setAttribute: function(){},
  };
  return el;
}
var ELS = {};
function el(id){ if (!ELS[id]) ELS[id] = makeEl(id); return ELS[id]; }
el("colorMode").value = "episode"; el("epSel").value = "all";
el("hlFrame").value = "0"; el("hlWin").value = "15";
el("t0").value = "0"; el("t1").value = "100"; el("psize").value = "3";
el("gsort").value = "best"; el("gspeed").value = "1"; el("gsearch").value = "";
el("fRemoved").value = "all";
el("hlOn").checked = false; el("fVal").checked = false; el("syncCam").checked = true;
el("ck").value = "2"; el("prunePct").value = "0"; el("prunePctN").value = "0";
el("targetH").value = ""; el("allowProt").checked = false;
var OPTSTORE = {};
var document = {
  body: makeEl("body"),
  getElementById: function(id){ return el(id); },
  querySelector: function(sel){ if (!OPTSTORE[sel]) OPTSTORE[sel] = makeEl(sel); return OPTSTORE[sel]; },
  querySelectorAll: function(){ return []; },
  createElement: function(tag){ return makeEl("tag:" + tag); },
};
var RESTYLES = [];
var Plotly = {
  newPlot: function(){}, purge: function(){},
  restyle: function(div, props, traces) { RESTYLES.push({div: div, props: props, traces: traces}); },
  relayout: function(){ return {then: function(f){ f(); }}; },
};
function lastTrace0(div) {
  for (var i = RESTYLES.length - 1; i >= 0; i--) {
    var r = RESTYLES[i];
    if (r.div === div && r.traces && r.traces[0] === 0) return r;
  }
  return null;
}
"""

CHECKS = r"""
var out = {};
out.initialTask = el("task").value;                       // g1 sorts first

// --- clustering: two well-separated blobs, k=2, deterministic ---
el("ck").value = "2";
clusterNow();
var st = CL["g1"];
out.k = st.k;
var a1 = Array.from(st.assign);
buildClusters("g1", 2);                                   // rebuild — must be identical
out.deterministic = a1.every(function(v, i){ return v === CL["g1"].assign[i]; });
st = CL["g1"];
var clA = st.epDomCl[0];
out.blobSplit = [0,1,2,3].every(function(e){ return st.epDomCl[e] === clA; }) &&
                [4,5,6].every(function(e){ return st.epDomCl[e] !== clA; });
out.purity = Array.from(st.epDomFrac).every(function(f){ return f === 1; });

// --- protect the second blob, prune 30% ---
var clB = st.epDomCl[4];
setClusterPrio("g1", clB, "protect");
el("prunePct").value = "30"; pruneChanged();
st = CL["g1"];
out.removed30 = Array.from(st.removedSet).sort().join(",");   // h00,h01,h02
out.stats30 = st.stats.removedFrames;

// --- protect clamps at 100% ---
el("prunePct").value = "100"; pruneChanged();
st = CL["g1"];
out.protectedSurvive = !st.removedSet.has("h05") && !st.removedSet.has("h06");
out.maxRemovablePct = Math.round(CL["g1"].stats.maxRemovablePct);

// --- overrides ---
el("prunePct").value = "30"; pruneChanged();
setOverride("g1", "h00", "keep");                          // rescue the worst episode
out.keepOverride = !CL["g1"].removedSet.has("h00") && CL["g1"].removedSet.has("h03");
setOverride("g1", "h00", "keep");                          // toggle off
el("prunePct").value = "0"; pruneChanged();
setOverride("g1", "h03", "drop");                          // forced drop at 0%
out.forcedDrop = CL["g1"].removedSet.has("h03") && CL["g1"].removedSet.size === 1;
setOverride("g1", "h03", "drop");                          // toggle off
el("prunePct").value = "30"; pruneChanged();

// --- removal order determinism ---
var o1 = CL["g1"].order.map(function(x){ return x.hash; }).join(",");
computeRemoval("g1");
out.orderStable = o1 === CL["g1"].order.map(function(x){ return x.hash; }).join(",");
out.orderHead = o1.split(",").slice(0, 4).join(",");
out.unscoredLast = o1.split(",").indexOf("h07") < o1.split(",").indexOf("h05");

// --- grid: filters + badges ---
setLabel("g1", "h00", "bad");
showPage("grid");
el("fRemoved").value = "removed"; render();
out.gridRemoved = (el("grid").innerHTML.match(/REMOVED/g) || []).length;
el("fRemoved").value = "all"; render();
out.gridNotEmbedded = el("grid").innerHTML.indexOf("NOT EMBEDDED") >= 0;   // h07
el("gsort").value = "removal"; render();
out.gridRemovalSortOk = el("grid").innerHTML.indexOf("h00") < el("grid").innerHTML.indexOf("h07");
out.gridheadNaNFree = el("gridhead").innerHTML.indexOf("NaN") < 0;
out.directionShown = el("gridhead").innerHTML.indexOf("higher = worse") >= 0;
out.gridTaskPinned = _gridTask === "g1";

// --- SCORES-only task (g2): no DATA, prune still works over universe ---
el("task").value = "g2"; showPage("tsne");
out.g2NoThrow = el("tstats").textContent.indexOf("0 episodes") >= 0;
el("prunePct").value = "50"; pruneChanged();
out.g2Removed = Array.from(CL["g2"].removedSet).join(",");  // x00 first

// --- exports ---
el("task").value = "g1"; render();
exportKeepList();
var keep = JSON.parse(DOWNLOADS[DOWNLOADS.length - 1].text);
out.keepHasG2Kept = keep.indexOf("x01") >= 0 && keep.indexOf("x00") < 0;
out.keepExcludesRemoved = keep.indexOf("h00") < 0 && keep.indexOf("h07") >= 0;
exportSelection();
var selJson = JSON.parse(DOWNLOADS[DOWNLOADS.length - 1].text);
out.selSchema = selJson.schema_version === 1 && !!selJson.tasks.g1 && !!selJson.tasks.g2;
out.selRemovedOrdered = selJson.tasks.g1.removed.length === CL["g1"].removedSet.size &&
                        selJson.tasks.g1.removed[0].hash === "h00";
out.selLabels = selJson.tasks.g1.labels.h00 === "bad";
out.selClusterOf = selJson.tasks.g1.cluster_of.h04 && selJson.tasks.g1.cluster_of.h04[1] === 1;
out.selMeta = selJson.scores_meta.higher_is_worse === true;

// --- target hours drives the slider inversely ---
el("targetH").value = String((26800 - 6100) / 30 / 3600); // keep all but h00+h01
targetHours();
out.targetHoursRemoved = Array.from(CL["g1"].removedSet).sort().join(",");

// --- preview overrides pin the clicked task ---
showFrame("g1", "h02", 40, 0.5);
out.pvTaskPinned = pvTask === "g1";

// --- removed points are dropped from the trace (not just recolored) ---
el("prunePct").value = "30"; pruneChanged();    // removes h00,h01,h02 (3 of 7 embedded eps)
var r30 = lastTrace0("state");
out.hiddenPts = r30 && r30.props.x ? r30.props.x[0].length : -1;          // 4 eps x 30 pts
out.hiddenCustomAligned = !!(r30 && r30.props.customdata &&
  r30.props.customdata[0].length === out.hiddenPts);
el("prunePct").value = "0"; pruneChanged();     // selection cleared -> full restore
var rFull = lastTrace0("state");
out.restoredPts = rFull && rFull.props.x ? rFull.props.x[0].length : -1;  // 7 x 30
el("prunePct").value = "30"; pruneChanged();

// --- click: cheap cross-highlight (only the OTHER panel restyles trace 1) ---
var h = ELS["state"]._handlers["plotly_click"];
out.hasClickHandler = !!h;
var before = RESTYLES.length;
ELS["state"]._drag = false;
h({points: [{curveNumber: 0, customdata: [40, 50, "h03_______", 3]}]});
var t1s = RESTYLES.slice(before).filter(function(r){ return r.traces && r.traces[0] === 1; });
out.clickRestyles = t1s.length;                                            // 1: action panel only
out.clickTargets = t1s.map(function(r){ return r.div; }).join(",");
h({points: [{curveNumber: 0, customdata: [40, 50, "h03_______", 3]}]});    // same point again
out.repeatClickRestyles = RESTYLES.filter(function(r){ return r.traces && r.traces[0] === 1; }).length - (before >= 0 ? t1s.length : 0) - RESTYLES.slice(0, before).filter(function(r){ return r.traces && r.traces[0] === 1; }).length;
out.clickNoThrow = true;

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
    print(("PASS" if cond else "FAIL"), name, detail if not cond else "")
    if not cond:
        FAIL.append(name)


check("initial task g1, no throw", out["initialTask"] == "g1", out["initialTask"])
check("k-means deterministic", out["deterministic"])
check("two blobs split 4/3", out["blobSplit"])
check("purity 1.0 on clean blobs", out["purity"])
check("30% prune removes worst-first from unprotected", out["removed30"] == "h00,h01,h02",
      out["removed30"])
check("prefix rule overshoot (>=8040 frames)", out["stats30"] == 9300, out["stats30"])
check("protect survives 100%", out["protectedSurvive"])
check("max removable reflects protected pool", out["maxRemovablePct"] < 100, out["maxRemovablePct"])
check("force-keep rescues + next-worst slides in", out["keepOverride"])
check("force-drop removes at 0%", out["forcedDrop"])
check("removal order stable across recompute", out["orderStable"])
check("order head = scored worst-first", out["orderHead"] == "h00,h01,h02,h03", out["orderHead"])
check("unscored after scored in class", out["unscoredLast"])
check("grid removed filter + badges", out["gridRemoved"] >= 3, out["gridRemoved"])
check("NOT EMBEDDED badge for failed-embed ep", out["gridNotEmbedded"])
check("grid removal-order sort", out["gridRemovalSortOk"])
check("grid head NaN-free", out["gridheadNaNFree"])
check("direction string rendered", out["directionShown"])
check("grid task pinned for handlers", out["gridTaskPinned"])
check("SCORES-only task degrades, no throw", out["g2NoThrow"])
check("score-only prune without clusters", out["g2Removed"] == "x00", out["g2Removed"])
check("keep-list spans tasks + respects selections", out["keepHasG2Kept"] and out["keepExcludesRemoved"])
check("selection schema + ordered removed", out["selSchema"] and out["selRemovedOrdered"])
check("selection carries labels + cluster_of + meta", out["selLabels"] and out["selClusterOf"] and out["selMeta"])
check("target-hours inverse drive", out["targetHoursRemoved"] == "h00,h01", out["targetHoursRemoved"])
check("preview override task pinned", out["pvTaskPinned"])
check("removed points dropped from trace", out["hiddenPts"] == 120, out["hiddenPts"])
check("customdata realigned with filtered points", out["hiddenCustomAligned"])
check("full data restored when selection clears", out["restoredPts"] == 210, out["restoredPts"])
check("click restyles only the other panel", out["clickRestyles"] == 1 and out["clickTargets"] == "action",
      f"{out['clickRestyles']} -> {out['clickTargets']}")
check("repeat click skips redundant restyle", out["repeatClickRestyles"] == 0, out["repeatClickRestyles"])
check("click path no throw", out["clickNoThrow"])

print()
print("FAILURES:" if FAIL else "ALL PRUNE-JS CHECKS PASSED", FAIL or "")
sys.exit(1 if FAIL else 0)
