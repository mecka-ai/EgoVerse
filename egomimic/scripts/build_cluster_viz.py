"""Build a self-contained HTML viewer for language-cluster curation runs.

Each cluster is a page in the sidebar; the main area shows a video grid of
annotation spans for that cluster, sorted by MI score descending.
"""
from __future__ import annotations

import json

_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Meckaverse — Cluster Viewer</title>
<style>
:root{--bg:#101014;--bar:#1a1b21;--line:#2b2d36;--acc:#06b6d4;--txt:#e8e8ea;}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;background:var(--bg);color:var(--txt);font-family:-apple-system,Helvetica,Arial,sans-serif;}
body{display:flex;flex-direction:column;overflow:hidden;}
#topbar{flex-shrink:0;padding:10px 18px;background:var(--bar);border-bottom:1px solid var(--line);
        display:flex;align-items:center;gap:12px;}
#topbar h3{font-size:15px;font-weight:700;}
#topbar .run{font-size:12px;color:#9aa;font-family:ui-monospace,monospace;
             white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:600px;}
#topbar a{margin-left:auto;font-size:13px;color:#9ecbff;text-decoration:none;flex-shrink:0;}
#body{flex:1;min-height:0;display:flex;overflow:hidden;}
#sidebar{width:310px;flex-shrink:0;display:flex;flex-direction:column;border-right:1px solid var(--line);overflow:hidden;}
#cluster-search{padding:10px 12px;border-bottom:1px solid var(--line);}
#cluster-search input{width:100%;font-size:13px;padding:6px 9px;background:#26272f;
                       color:var(--txt);border:1px solid var(--line);border-radius:6px;}
#cluster-list{flex:1;overflow-y:auto;}
.cl-row{padding:10px 14px;cursor:pointer;border-bottom:1px solid var(--line);
        display:flex;flex-direction:column;gap:3px;border-left:3px solid transparent;}
.cl-row:hover{background:#1c1d24;}
.cl-row.active{background:#0d2430;border-left-color:var(--acc);}
.cl-id{font-size:10px;color:#5a6a7a;font-family:ui-monospace,monospace;}
.cl-label{font-size:12px;color:#dde;line-height:1.4;}
.cl-count{font-size:11px;color:#06b6d4;}
#content{flex:1;display:flex;flex-direction:column;overflow:hidden;}
#gridbar{flex-shrink:0;padding:8px 14px;background:var(--bar);border-bottom:1px solid var(--line);
         display:flex;align-items:center;gap:12px;font-size:13px;}
#grid-cluster-id{font-family:ui-monospace,monospace;font-size:11px;color:#5a6a7a;}
#grid-cluster-label{font-weight:600;color:#06b6d4;font-size:13px;}
#grid-count{color:#666;font-size:12px;}
.sort-btn{margin-left:auto;font-size:12px;color:#9aa;cursor:pointer;user-select:none;}
.sort-btn:hover{color:#dde;}
#grid-wrap{flex:1;overflow-y:auto;padding:14px;}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;}
.span-card{background:#1a1b21;border:1px solid var(--line);border-radius:8px;overflow:hidden;
           cursor:pointer;transition:border-color 0.15s;}
.span-card:hover{border-color:#06b6d4;}
.span-card.playing{border-color:#06b6d4;}
.card-thumb{position:relative;width:100%;aspect-ratio:16/9;background:#000;overflow:hidden;}
.card-thumb img{width:100%;height:100%;object-fit:cover;display:block;}
.card-thumb video{width:100%;height:100%;object-fit:cover;display:none;position:absolute;inset:0;}
.card-thumb .score-badge{position:absolute;top:6px;right:6px;background:rgba(0,0,0,.75);
                          font-size:11px;font-weight:700;padding:2px 7px;border-radius:4px;}
.card-thumb .play-icon{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
                        font-size:32px;opacity:0;transition:opacity 0.15s;pointer-events:none;}
.span-card:hover .play-icon{opacity:1;}
.span-card.playing .play-icon{opacity:0;}
.span-card.playing .card-thumb video{display:block;}
.card-body{padding:8px 10px 10px;}
.card-text{font-size:12px;color:#bbb;line-height:1.5;}
.card-meta{font-size:10px;color:#555;margin-top:4px;font-family:ui-monospace,monospace;}
.empty{padding:32px;color:#555;font-size:14px;}
</style>
</head>
<body>
<div id="topbar">
  <h3>Meckaverse</h3>
  <span class="run">__RUN_LABEL__</span>
  <a href="/">&#8592; back</a>
</div>
<div id="body">
  <div id="sidebar">
    <div id="cluster-search">
      <input id="search" placeholder="Filter clusters&#x2026;" oninput="filterClusters()">
    </div>
    <div id="cluster-list"></div>
  </div>
  <div id="content">
    <div id="gridbar">
      <span id="grid-cluster-id"></span>
      <span id="grid-cluster-label"></span>
      <span id="grid-count"></span>
      <span class="sort-btn" onclick="toggleSort()">sort: <span id="sort-label">score &#8595;</span></span>
    </div>
    <div id="grid-wrap"><div id="grid"></div></div>
  </div>
</div>
<script>
const CLUSTERS = __CLUSTERS__;
const VIDEO_BASE = "__VIDEO_BASE__";
const FRAME_BASE = "__FRAME_BASE__";
const FPS = 30;

let curCluster = null;
let sortMode = "score";
const ALL_IDS = Object.keys(CLUSTERS).sort((a,b)=>{
  const ai=parseInt(a.split("_")[1]||"0"), bi=parseInt(b.split("_")[1]||"0");
  return ai-bi;
});

function buildSidebar(){
  const list=document.getElementById("cluster-list");
  list.innerHTML=ALL_IDS.map(id=>{
    const c=CLUSTERS[id];
    return `<div class="cl-row" id="cl-${id}" onclick="selectCluster('${id}')">
      <span class="cl-id">${id}</span>
      <span class="cl-label">${esc(c.label)}</span>
      <span class="cl-count">${c.spans.length} spans</span>
    </div>`;
  }).join("");
}

function filterClusters(){
  const q=document.getElementById("search").value.toLowerCase();
  ALL_IDS.forEach(id=>{
    const c=CLUSTERS[id];
    const row=document.getElementById("cl-"+id);
    row.style.display=(c.label.toLowerCase().includes(q)||id.toLowerCase().includes(q))?"":"none";
  });
}

function selectCluster(id){
  if(curCluster){
    const p=document.getElementById("cl-"+curCluster);
    if(p)p.classList.remove("active");
    // stop any playing video
    document.querySelectorAll(".span-card.playing").forEach(card=>{
      const v=card.querySelector("video"); if(v)v.pause();
      card.classList.remove("playing");
    });
  }
  curCluster=id;
  const row=document.getElementById("cl-"+id);
  if(row){row.classList.add("active");row.scrollIntoView({block:"nearest"});}
  renderGrid();
}

function toggleSort(){
  sortMode=sortMode==="score"?"frame":"score";
  document.getElementById("sort-label").textContent=sortMode==="score"?"score ↓":"ep/frame";
  renderGrid();
}

function sortedSpans(spans){
  const arr=[...spans];
  if(sortMode==="score") arr.sort((a,b)=>b.score-a.score);
  else arr.sort((a,b)=>a.ep===b.ep?a.start-b.start:a.ep<b.ep?-1:1);
  return arr;
}

function scoreColor(score, minS, maxS){
  if(minS===maxS||maxS===minS)return"#aaa";
  const t=Math.max(0,Math.min(1,(score-minS)/(maxS-minS)));
  const r=Math.round(220*(1-t)), g=Math.round(200*t);
  return`rgb(${r},${g},80)`;
}

function renderGrid(){
  const id=curCluster;
  const c=CLUSTERS[id];
  document.getElementById("grid-cluster-id").textContent=id;
  document.getElementById("grid-cluster-label").textContent=c.label;
  const spans=sortedSpans(c.spans);
  document.getElementById("grid-count").textContent=spans.length+" spans";
  const grid=document.getElementById("grid");
  if(!spans.length){
    grid.innerHTML='<div class="empty">No spans in this cluster.</div>';
    return;
  }
  grid.innerHTML=spans.map((s,i)=>{
    const mid=Math.round((s.start+s.end)/2);
    const col=scoreColor(s.score,c.minScore,c.maxScore);
    const scoreStr=isNaN(s.score)?"-":s.score.toFixed(3);
    return`<div class="span-card" id="card-${i}" onclick="playSpan(${i},'${s.ep}',${s.start},${s.end})">
      <div class="card-thumb">
        <img id="img-${i}" src="${FRAME_BASE}${s.ep}/${mid}" loading="lazy" onerror="this.style.opacity=0.15">
        <video id="vid-${i}" muted playsinline preload="none"></video>
        <div class="score-badge" style="color:${col}">${scoreStr}</div>
        <div class="play-icon">&#9654;</div>
      </div>
      <div class="card-body">
        <div class="card-text">${esc(s.text)}</div>
        <div class="card-meta">${s.ep.slice(0,10)}&hellip; &middot; frames&nbsp;${s.start}&ndash;${s.end}</div>
      </div>
    </div>`;
  }).join("");
}

function playSpan(i, ep, start, end){
  const card=document.getElementById("card-"+i);
  const vid=document.getElementById("vid-"+i);
  if(card.classList.contains("playing")){
    vid.pause(); card.classList.remove("playing"); return;
  }
  document.querySelectorAll(".span-card.playing").forEach(c=>{
    const v=c.querySelector("video"); if(v)v.pause(); c.classList.remove("playing");
  });
  card.classList.add("playing");
  vid.src=VIDEO_BASE+ep;
  const seekTo=start/FPS;
  const stopAt=end/FPS;
  const doSeek=()=>{vid.currentTime=seekTo; vid.play(); vid.removeEventListener("loadedmetadata",doSeek);};
  vid.addEventListener("loadedmetadata",doSeek);
  vid.addEventListener("timeupdate",function stopper(){
    if(vid.currentTime>=stopAt){
      vid.pause(); vid.removeEventListener("timeupdate",stopper);
      card.classList.remove("playing");
    }
  });
  vid.load();
}

function esc(s){
  return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

buildSidebar();
if(ALL_IDS.length) selectCluster(ALL_IDS[0]);
</script>
</body>
</html>"""


def build_cluster_html(
    clusters_raw: dict,
    *,
    video_base: str,
    frame_base: str,
    run_label: str,
) -> str:
    clusters_js: dict = {}
    for cluster_id, c in clusters_raw.items():
        spans_list = []
        scores = []
        for span_key, s in c.get("spans", {}).items():
            score = float(s.get("score", 0))
            spans_list.append({
                "id": span_key,
                "ep": s["episode"],
                "start": int(s["start"]),
                "end": int(s["end"]),
                "text": s.get("text", ""),
                "score": score,
            })
            scores.append(score)
        clusters_js[cluster_id] = {
            "label": c.get("label", cluster_id),
            "spans": spans_list,
            "minScore": min(scores) if scores else 0.0,
            "maxScore": max(scores) if scores else 1.0,
        }

    return (
        _TEMPLATE
        .replace("__CLUSTERS__", json.dumps(clusters_js, separators=(",", ":")))
        .replace("__VIDEO_BASE__", video_base)
        .replace("__FRAME_BASE__", frame_base)
        .replace("__RUN_LABEL__", run_label)
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("clustered_scores_json")
    parser.add_argument("--out", default="cluster_viewer.html")
    parser.add_argument("--video-base", default="/video/")
    parser.add_argument("--frame-base", default="/frame/")
    args = parser.parse_args()
    data = json.load(open(args.clustered_scores_json))
    html = build_cluster_html(
        data,
        video_base=args.video_base,
        frame_base=args.frame_base,
        run_label=args.clustered_scores_json,
    )
    open(args.out, "w").write(html)
    print(f"wrote {args.out} ({len(html)//1024} KB)")
