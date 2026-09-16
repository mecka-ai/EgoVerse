"""Hours available per (task_domain, split) in the QA_exp manifests."""
import csv
from collections import defaultdict

hrs = defaultdict(float)
eps = defaultdict(int)
for name in ("business", "home"):
    p = f"/home/mecka/Delivery/QA_exp/{name}_task_family_samples.csv"
    for r in csv.DictReader(open(p)):
        k = (r["task_domain"], r["split"])
        hrs[k] += float(r["hours"])
        eps[k] += 1

domains = sorted({d for d, _ in hrs})
splits = ["top", "uniform", "bottom"]
print(f"{'task_domain':34s} " + " ".join(f"{s:>16s}" for s in splits) + f" {'total':>8s}")
print("-" * (35 + 17 * len(splits) + 9))
short = []
for d in domains:
    row = []
    for s in splits:
        row.append(f"{hrs[(d,s)]:7.1f}h/{eps[(d,s)]:5d}")
        if hrs[(d, s)] < 50:
            short.append((d, s, hrs[(d, s)]))
    print(f"{d[:34]:34s} " + " ".join(f"{c:>16s}" for c in row)
          + f" {sum(hrs[(d,s)] for s in splits):7.1f}h")

print(f"\ntotal: {sum(hrs.values()):,.0f} h across {len(domains)} domains, "
      f"{len(domains)*3} (domain, split) cells")
print(f"cells with < 50 h: {len(short)} of {len(domains)*3}")
for d, s, h in sorted(short, key=lambda x: x[2])[:12]:
    print(f"   {h:6.1f}h  {d} / {s}")
