"""What does quality_score look like, and does it relate to the existing splits?"""
import csv
from collections import defaultdict

import numpy as np

rows = []
for name in ("business", "home"):
    p = f"/home/mecka/Delivery/QA_exp/{name}_task_family_samples.csv"
    for r in csv.DictReader(open(p)):
        rows.append({
            "domain": r["task_domain"], "split": r["split"],
            "eid": r["episode_id"], "score": float(r["quality_score"]),
            "hours": float(r["hours"]),
        })
print(f"{len(rows):,} episodes total\n")

# does the existing split label track quality_score?
by_split = defaultdict(list)
for r in rows:
    by_split[r["split"]].append(r["score"])
print("quality_score by existing split label:")
for s in ("top", "uniform", "bottom"):
    v = np.array(by_split[s])
    print(f"  {s:8s} n={len(v):6,}  min={v.min():.3f} p25={np.percentile(v,25):.3f} "
          f"med={np.median(v):.3f} p75={np.percentile(v,75):.3f} max={v.max():.3f}")

# duplicates? the same episode can appear under more than one split
seen = defaultdict(set)
for r in rows:
    seen[r["eid"]].add(r["split"])
multi = {e: s for e, s in seen.items() if len(s) > 1}
print(f"\nepisodes appearing in >1 split: {len(multi):,}")
print(f"distinct episodes: {len(seen):,}")

# per-domain score spread, and what equal-size halves would give
print(f"\n{'domain':34s} {'eps':>7s} {'hours':>8s} {'half':>8s} "
      f"{'top-half med':>13s} {'bot-half med':>13s}")
bydom = defaultdict(list)
for r in rows:
    bydom[r["domain"]].append(r)
for d in sorted(bydom):
    rs = sorted({r["eid"]: r for r in bydom[d]}.values(), key=lambda x: -x["score"])
    h = sum(x["hours"] for x in rs)
    half = len(rs) // 2
    tm = np.median([x["score"] for x in rs[:half]])
    bm = np.median([x["score"] for x in rs[half:]])
    print(f"{d[:34]:34s} {len(rs):7,} {h:7.1f}h {sum(x['hours'] for x in rs[:half]):7.1f}h "
          f"{tm:13.3f} {bm:13.3f}")
