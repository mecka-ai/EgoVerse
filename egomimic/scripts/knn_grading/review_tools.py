#!/usr/bin/env python3
"""Human review + threshold calibration for k-NN action-consistency grading.

Workflow
--------
1. Pull a run's report from the Modal outputs volume:

       modal volume get egoverse-training-outputs \\
           knn_grading/<name>/<desc>_<ts>/knn_report.json scratch/knn_report.json

2. Build a labeling sheet (head/tail/random episodes per task), watch the
   episodes, fill the ``label`` column with ``bad`` / ``good``:

       python egomimic/scripts/knn_grading/review_tools.py sheet \\
           --report scratch/knn_report.json --out scratch/knn_labels.csv

3. Sweep prune thresholds against your labels:

       python egomimic/scripts/knn_grading/review_tools.py calibrate \\
           --report scratch/knn_report.json --labels scratch/knn_labels.csv

The calibrate step prints precision/recall/F1 per candidate cutoff on the
chosen metric (default ``primary_score``) and recommends the best-F1 cutoff.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

SHEET_COLS = [
    "task",
    "bucket",
    "rank",
    "hash",
    "primary_score",
    "frac_flagged_spatial",
    "frac_flagged_velocity",
    "longest_flagged_run",
    "coverage_frac",
    "mean_ambiguity_pctile",
    "label",
]

BAD_LABELS = {"bad", "1", "true", "prune", "incoherent"}
GOOD_LABELS = {"good", "0", "false", "keep", "clean"}


def _load_report(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _ranked_episodes(report: dict) -> dict[str, list[tuple[str, dict]]]:
    """{task: [(hash, metrics) sorted worst-first]} skipping skipped tasks."""
    out = {}
    for task, block in report.items():
        per_ep = block.get("per_episode", {})
        if not per_ep:
            continue

        def _key(kv):
            v = kv[1].get("primary_score")
            return v if isinstance(v, (int, float)) and math.isfinite(v) else -1.0

        out[task] = sorted(per_ep.items(), key=_key, reverse=True)
    return out


def cmd_sheet(args: argparse.Namespace) -> None:
    report = _load_report(Path(args.report))
    rng = random.Random(args.seed)
    rows = []
    for task, ranked in _ranked_episodes(report).items():
        n = len(ranked)
        head = ranked[: args.per_tail]
        tail = ranked[max(n - args.per_tail, args.per_tail) :]
        tail = tail[-args.per_tail :] if tail else []
        middle_pool = ranked[args.per_tail : max(n - args.per_tail, args.per_tail)]
        mid = rng.sample(middle_pool, min(args.per_random, len(middle_pool)))
        for bucket, items in (("worst", head), ("random", mid), ("best", tail)):
            for h, m in items:
                rank = next(i for i, (hh, _) in enumerate(ranked) if hh == h)
                rows.append(
                    {
                        "task": task,
                        "bucket": bucket,
                        "rank": rank,
                        "hash": h,
                        "label": "",
                        **{k: m.get(k) for k in SHEET_COLS if k in m},
                    }
                )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    n_tasks = len({r["task"] for r in rows})
    print(f"Wrote {len(rows)} episodes across {n_tasks} tasks → {out}")
    print("Watch each episode and fill the `label` column with bad/good.")


def cmd_calibrate(args: argparse.Namespace) -> None:
    report = _load_report(Path(args.report))
    metric = args.metric

    labels: dict[str, bool] = {}
    with open(args.labels, newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("label") or "").strip().lower()
            if raw in BAD_LABELS:
                labels[row["hash"]] = True
            elif raw in GOOD_LABELS:
                labels[row["hash"]] = False
    if not labels:
        raise SystemExit("No labeled rows found — fill the `label` column first.")

    scored: list[tuple[float, bool]] = []
    n_unscored = 0
    for block in report.values():
        for h, m in block.get("per_episode", {}).items():
            if h not in labels:
                continue
            v = m.get(metric)
            if isinstance(v, (int, float)) and math.isfinite(v):
                scored.append((float(v), labels[h]))
            else:
                n_unscored += 1

    n_bad = sum(1 for _, b in scored if b)
    print(
        f"{len(scored)} labeled episodes matched ({n_bad} bad, "
        f"{len(scored) - n_bad} good, {n_unscored} without finite {metric})\n"
    )
    if not scored or n_bad == 0 or n_bad == len(scored):
        raise SystemExit("Need both bad and good labels with finite scores.")

    candidates = sorted({round(v, 4) for v, _ in scored})
    print(f"{'cutoff':>8}  {'pruned':>6}  {'precision':>9}  {'recall':>7}  {'F1':>6}")
    best = (0.0, None)
    for cut in candidates:
        tp = sum(1 for v, b in scored if v >= cut and b)
        fp = sum(1 for v, b in scored if v >= cut and not b)
        fn = sum(1 for v, b in scored if v < cut and b)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{cut:8.4f}  {tp + fp:6d}  {prec:9.3f}  {rec:7.3f}  {f1:6.3f}")
        if f1 > best[0]:
            best = (f1, cut)

    print(
        f"\nRecommended: prune episodes with {metric} >= {best[1]:.4f} "
        f"(F1={best[0]:.3f} on your labels)."
    )
    print(
        "Validate with a fixed-hours A/B retrain (filtered vs unfiltered at "
        "equal hours) before pruning at scale."
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sheet = sub.add_parser("sheet", help="build a labeling sheet from a report")
    sheet.add_argument("--report", required=True, help="knn_report.json path")
    sheet.add_argument("--out", default="scratch/knn_labels.csv")
    sheet.add_argument(
        "--per-tail",
        type=int,
        default=10,
        help="worst + best episodes per task (default 10 each)",
    )
    sheet.add_argument(
        "--per-random",
        type=int,
        default=5,
        help="random mid-ranked episodes per task (default 5)",
    )
    sheet.add_argument("--seed", type=int, default=42)
    sheet.set_defaults(func=cmd_sheet)

    cal = sub.add_parser("calibrate", help="sweep prune cutoffs against labels")
    cal.add_argument("--report", required=True, help="knn_report.json path")
    cal.add_argument("--labels", required=True, help="filled labeling sheet CSV")
    cal.add_argument(
        "--metric",
        default="primary_score",
        help="per-episode metric to threshold (default primary_score)",
    )
    cal.set_defaults(func=cmd_calibrate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
