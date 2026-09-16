"""Score distribution of the top vs bottom arms for the three lead domains.

A median cut makes the arms adjacent: the lowest-scoring episode in `top` and
the highest in `bottom` sit on either side of the same boundary and are nearly
identical. Contrast comes only from the tails. Compare against a tertile cut,
which discards the ambiguous middle and widens the gap.
"""
import csv
from collections import defaultdict

import numpy as np

LEAD = ["cleaning-sanitation", "organization-stocking", "dish-handling"]

by_eid = {}
for name in ("business", "home"):
    p = f"/home/mecka/Delivery/QA_exp/{name}_task_family_samples.csv"
    for r in csv.DictReader(open(p)):
        if r["episode_id"] not in by_eid:
            by_eid[r["episode_id"]] = {
                "domain": r["task_domain"], "score": float(r["quality_score"]),
                "hours": float(r["hours"]),
            }

bydom = defaultdict(list)
for r in by_eid.values():
    bydom[r["domain"]].append(r)


def describe(vals):
    v = np.array(vals)
    return (f"min={v.min():.3f} p25={np.percentile(v,25):.3f} med={np.median(v):.3f} "
            f"p75={np.percentile(v,75):.3f} max={v.max():.3f}")


for d in LEAD:
    rs = sorted(bydom[d], key=lambda x: -x["score"])
    n = len(rs)
    print(f"\n=== {d}  ({n:,} episodes, {sum(x['hours'] for x in rs):.1f}h) ===")

    half = n // 2
    top, bot = rs[:half], rs[-half:]
    print(f"  MEDIAN CUT ({half:,} eps/arm)")
    print(f"    top    {describe([x['score'] for x in top])}  {sum(x['hours'] for x in top):.1f}h")
    print(f"    bottom {describe([x['score'] for x in bot])}  {sum(x['hours'] for x in bot):.1f}h")
    print(f"    boundary: top.min={min(x['score'] for x in top):.3f} "
          f"vs bottom.max={max(x['score'] for x in bot):.3f}  (arms touch)")

    third = n // 3
    t3, b3 = rs[:third], rs[-third:]
    mid_lo = min(x["score"] for x in t3)
    mid_hi = max(x["score"] for x in b3)
    print(f"  TERTILE CUT ({third:,} eps/arm, middle third discarded)")
    print(f"    top    {describe([x['score'] for x in t3])}  {sum(x['hours'] for x in t3):.1f}h")
    print(f"    bottom {describe([x['score'] for x in b3])}  {sum(x['hours'] for x in b3):.1f}h")
    print(f"    gap between arms: {mid_lo:.3f} down to {mid_hi:.3f} "
          f"= {mid_lo - mid_hi:.3f} of clear air")

    # how much of each arm sits at the score ceiling
    ceil = max(x["score"] for x in rs)
    at_ceil = sum(1 for x in t3 if x["score"] >= ceil - 1e-6)
    print(f"    note: {at_ceil:,}/{third:,} of the top tertile sit at the "
          f"score ceiling {ceil:.3f}")
