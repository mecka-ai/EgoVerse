"""Sample the two training pools for the pi0.5 selection experiment.

Pool A = the 2,600-hour selection, Pool B = the 6,000-hour selection, both built
by the same greedy quality composite with a 15%-per-family cap. Draw N episodes
RANDOMLY from each -- not top-N -- so each sample reflects its pool's own
distribution. Top-N would make both runs draw the same best episodes and the
comparison would collapse.

Equal N is what makes the comparison about selectivity rather than volume.

    python sample_training_pools.py --n 20000
"""

import argparse
import csv
import json
import random
from collections import defaultdict

import numpy as np

SRC = "/home/mecka/.claude/jobs/75e13470/tmp/metrics_all.json"
OUT_DIR = "/home/mecka/Delivery/6k"
FPS = 30.0
CAP_PCT = 15.0


def build_pool(rows, rig, jrk, arm, dom, hrs, fam, score, order, valid, target):
    cap_h = target * CAP_PCT / 100.0
    used = defaultdict(float)
    total = 0.0
    chosen = []
    for i in order:
        if not valid[i]:
            continue
        f = fam[i]
        if used[f] + hrs[i] > cap_h:
            continue
        chosen.append(i)
        used[f] += hrs[i]
        total += hrs[i]
        if total >= target:
            break
    return chosen, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rows = [r for r in json.load(open(SRC)) if not r.get("unusable")]
    rows = [r for r in rows if all(r.get(k) is not None for k in
            ("rigid_cv", "jerk_p99", "arm_motion", "arm_dominance", "n_frames"))]
    rig = np.array([r["rigid_cv"] for r in rows])
    jrk = np.array([r["jerk_p99"] for r in rows])
    arm = np.array([r["arm_motion"] for r in rows])
    dom = np.array([r["arm_dominance"] for r in rows])
    hrs = np.array([r["n_frames"] for r in rows]) / FPS / 3600.0
    fam = np.array([r["group_id"] for r in rows])

    valid = (rig <= np.percentile(rig, 98)) & (jrk <= np.percentile(jrk, 98))

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-12)
    score = z(arm) + z(dom) - z(rig)
    order = np.argsort(-score)

    rng = random.Random(a.seed)
    pools = {}
    for label, target in (("2600h", 2600.0), ("6000h", 6000.0)):
        idx, total = build_pool(rows, rig, jrk, arm, dom, hrs, fam, score, order, valid, target)
        samp = rng.sample(idx, min(a.n, len(idx)))
        pools[label] = samp
        print(f"pool {label}: {len(idx):,} episodes ({total:,.0f} h) "
              f"-> sampled {len(samp):,} ({hrs[samp].sum():,.0f} h, "
              f"{len(set(fam[samp])):,} families)")
        print(f"   sampled means  arm_motion={arm[samp].mean():.5f}  "
              f"arm_dom={dom[samp].mean():.4f}  rigid_cv={rig[samp].mean():.5f}")

        with open(f"{OUT_DIR}/train_pool_{label}.json", "w") as f:
            json.dump([rows[i]["episode_id"] for i in samp], f)

    a_set = {rows[i]["episode_id"] for i in pools["2600h"]}
    b_set = {rows[i]["episode_id"] for i in pools["6000h"]}
    union = a_set | b_set
    print(f"\noverlap between samples: {len(a_set & b_set):,}")
    print(f"UNION to convert: {len(union):,} episodes "
          f"(~{len(union) * 48 / 1e3:.1f} GB at 48 MB/ep)")

    with open(f"{OUT_DIR}/train_union.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode_id", "group_id"])
        by_id = {r["episode_id"]: r["group_id"] for r in rows}
        for e in sorted(union):
            w.writerow([e, by_id[e]])
    print(f"wrote {OUT_DIR}/train_union.csv and train_pool_*.json")


if __name__ == "__main__":
    main()
