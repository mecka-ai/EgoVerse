"""Build the three training arms per task domain from quality_score.

The pre-made `split` column is not usable as-is: its cells overlap (9,703
episodes appear under more than one label) and `top` falls short of the others
in 10 of 13 domains, so comparing them would confound quality with data volume.

Instead, within each domain: dedupe, sort by quality_score, and cut at the
median. top-half / bottom-half / a random draw of the same size. Equal episode
count and near-equal hours across all three arms by construction, so the only
thing that differs is where on the quality ranking the data came from.

Emits one JSON of episode hashes per arm, consumable as `eps_to_use`.

    python qaexp_build_arms.py
"""

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path("/home/mecka/Delivery/QA_exp/arms")
SEED = 42


def main():
    by_eid = {}
    for name in ("business", "home"):
        p = f"/home/mecka/Delivery/QA_exp/{name}_task_family_samples.csv"
        for r in csv.DictReader(open(p)):
            eid = r["episode_id"]
            # an episode can be listed under several splits; the score and
            # domain are per-episode, so first occurrence is authoritative
            if eid not in by_eid:
                by_eid[eid] = {
                    "eid": eid,
                    "domain": r["task_domain"],
                    "score": float(r["quality_score"]),
                    "hours": float(r["hours"]),
                }
    print(f"{len(by_eid):,} distinct episodes")

    bydom = defaultdict(list)
    for r in by_eid.values():
        bydom[r["domain"]].append(r)

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print(f"\n{'domain':34s} {'n/arm':>7s} {'top h':>8s} {'bot h':>8s} {'rand h':>8s} "
          f"{'top med':>8s} {'bot med':>8s} {'gap':>6s}")
    summary = []
    for d in sorted(bydom):
        rs = sorted(bydom[d], key=lambda x: -x["score"])
        half = len(rs) // 2
        top = rs[:half]
        bot = rs[-half:]
        rand = rng.sample(rs, half)

        arms = {"top": top, "bottom": bot, "random": rand}
        for arm, sel in arms.items():
            with open(OUT / f"{d}__{arm}.json", "w") as f:
                json.dump([x["eid"] for x in sel], f)

        tm = float(np.median([x["score"] for x in top]))
        bm = float(np.median([x["score"] for x in bot]))
        print(f"{d[:34]:34s} {half:7,} {sum(x['hours'] for x in top):7.1f}h "
              f"{sum(x['hours'] for x in bot):7.1f}h {sum(x['hours'] for x in rand):7.1f}h "
              f"{tm:8.3f} {bm:8.3f} {tm-bm:6.2f}")
        summary.append({"domain": d, "n_per_arm": half, "gap": tm - bm,
                        "top_h": sum(x["hours"] for x in top)})

    summary.sort(key=lambda s: -s["gap"])
    print(f"\nranked by score separation (a domain with a small gap cannot test "
          f"the scoring method -- its arms are nearly the same data):")
    for s in summary:
        mark = "  <-- too little separation to test" if s["gap"] < 1.0 else ""
        print(f"   gap {s['gap']:5.2f}  {s['n_per_arm']:5,} eps/arm  {s['domain']}{mark}")

    with open(OUT / "_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\nwrote {len(bydom)*3} arm files to {OUT}")


if __name__ == "__main__":
    main()
