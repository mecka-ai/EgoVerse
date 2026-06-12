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

Viewer-driven flow (preferred — no CSV watching)
------------------------------------------------
Tune the prune in the latent viewer (cluster the t-SNE, set cluster
priorities, drag the remove-% slider, spot-check the removed grid) and export
its selection, then:

4. Turn one or more exported viewer selections into a training allowlist:

       python egomimic/scripts/knn_grading/review_tools.py apply \\
           --selection viewer_selection_<ts>.json \\
           --out egomimic/hydra_configs/data/extra/my_subset.json

   ``apply`` unions kept episodes across all tasks in all selections,
   defaults uncovered tasks to keep-all (given --universe), writes the plain
   hash list + a .provenance.json sidecar, and prints the resolver snippet.
   Labels captured in the viewer (✓/✗ on grid cards) are exported inside the
   selection and can be fed to ``calibrate`` via --selection-labels.

5. Refresh scores on an existing viz run after re-grading (CPU-only — no
   GPU re-embed; reads/writes the Modal volume):

       python egomimic/scripts/knn_grading/review_tools.py pair \\
           --viz-run latent_viz/<name>/<desc>_<ts> \\
           --knn-run knn_grading/<name>/<desc>_<ts>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import tempfile
from datetime import datetime, timezone
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


def _labels_from_selection(path: Path) -> dict[str, bool]:
    """Extract {hash: is_bad} from a viewer selection's per-task ✓/✗ labels."""
    sel = json.load(open(path))
    out: dict[str, bool] = {}
    for block in (sel.get("tasks") or {}).values():
        for h, lbl in (block.get("labels") or {}).items():
            raw = str(lbl).strip().lower()
            if raw in BAD_LABELS:
                out[h] = True
            elif raw in GOOD_LABELS:
                out[h] = False
    return out


def cmd_calibrate(args: argparse.Namespace) -> None:
    report = _load_report(Path(args.report))
    metric = args.metric

    labels: dict[str, bool] = {}
    if args.labels:
        with open(args.labels, newline="") as f:
            for row in csv.DictReader(f):
                raw = (row.get("label") or "").strip().lower()
                if raw in BAD_LABELS:
                    labels[row["hash"]] = True
                elif raw in GOOD_LABELS:
                    labels[row["hash"]] = False
    if args.selection_labels:
        labels.update(_labels_from_selection(Path(args.selection_labels)))
    if not labels:
        raise SystemExit(
            "No labels found — fill the CSV `label` column or pass "
            "--selection-labels with a viewer selection containing ✓/✗ marks."
        )

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


# ---------------------------------------------------------------------------
# apply: viewer selection(s) → eps_to_use allowlist
# ---------------------------------------------------------------------------


def cmd_apply(args: argparse.Namespace) -> None:
    """Union kept episodes from viewer selections into a training allowlist."""
    kept: set[str] = set()
    covered: set[str] = set()
    sources = []
    per_task: dict[str, dict] = {}

    for sel_path in args.selection:
        p = Path(sel_path)
        raw = p.read_bytes()
        sel = json.loads(raw)
        tasks = sel.get("tasks") or {}
        if not tasks:
            raise SystemExit(f"{p}: no tasks in selection — is this a viewer selection export?")
        for task, block in tasks.items():
            k = set(block.get("kept") or [])
            r = set(x["hash"] for x in (block.get("removed") or []))
            if task in per_task:
                print(f"WARNING: task {task!r} appears in multiple selections — intersecting kept sets")
                prev = per_task[task]
                k &= set(prev["kept"])
                r |= set(prev["removed"])
            per_task[task] = {"kept": sorted(k), "removed": sorted(r), "source": str(p)}
            kept |= k
            covered |= k | r
        sources.append(
            {
                "file": str(p),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "created_at": sel.get("created_at"),
                "tasks": {t: {"kept": len(b.get("kept") or []), "removed": len(b.get("removed") or [])} for t, b in tasks.items()},
            }
        )

    # Re-intersect after possible multi-selection merges
    kept = set()
    for block in per_task.values():
        kept |= set(block["kept"])

    n_default = 0
    if args.universe:
        uni = set(json.load(open(args.universe)))
        default_kept = uni - covered
        n_default = len(default_kept)
        kept |= default_kept

    print(f"{'task':<40} {'kept':>6} {'removed':>8}")
    for t, b in sorted(per_task.items()):
        print(f"{t:<40} {len(b['kept']):>6} {len(b['removed']):>8}")
    if args.universe:
        print(f"{'(not covered by any selection — kept)':<40} {n_default:>6}")
    elif n_default == 0:
        print(
            "NOTE: no --universe given — episodes outside the selections' tasks "
            "are NOT in the allowlist. Pass the viz run's episode_hashes.json "
            "to default uncovered episodes to keep."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(kept)))
    prov = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "universe": str(args.universe) if args.universe else None,
        "n_kept": len(kept),
        "n_default_kept": n_default,
        "allowlist_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    Path(str(out).removesuffix(".json") + ".provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"\nWrote {len(kept)} hashes → {out} (+ .provenance.json sidecar)")
    print("Use it in a data config resolver:\n")
    print(f"    resolver:\n      eps_to_use: {out}\n")
    print("(Commit + push the JSON — Modal containers read it from the repo clone.)")


# ---------------------------------------------------------------------------
# pair: refresh scores on an existing viz run from a grading run (no re-embed)
# ---------------------------------------------------------------------------


def _vol_get(volume: str, env: str, remote: str, dest: Path) -> bool:
    r = subprocess.run(
        ["modal", "volume", "get", "--env", env, "--force", volume, remote, str(dest)],
        capture_output=True, text=True,
    )
    return r.returncode == 0


def _vol_put(volume: str, env: str, src: Path, remote: str) -> None:
    subprocess.run(
        ["modal", "volume", "put", "--env", env, "--force", volume, str(src), remote],
        check=True,
    )


def cmd_pair(args: argparse.Namespace) -> None:
    """Hash-rekey a grading run's scores into an existing viz run dir (CPU-only).

    Mirrors latentVizModal's scores_from logic, but against the volume after
    the fact — the knn feature-cache design makes re-grading cheap, and this
    makes re-pairing cheap too (the hosted viewer picks the new scores up on
    its next cold start).
    """
    tmp = Path(tempfile.mkdtemp(prefix="pair_"))

    def _fetch(run: str, name: str) -> Path | None:
        local = Path(run) / name
        if local.is_file():
            return local
        dest = tmp / f"{name}"
        return dest if _vol_get(args.volume, args.env, f"{run}/{name}", dest) else None

    uni_path = _fetch(args.viz_run, "group_universe.json")
    if uni_path is None:
        raise SystemExit(
            f"{args.viz_run}/group_universe.json not found — the viz run predates "
            "universe export; re-export it (or pass a run produced by the current code)."
        )
    groups = {g: list(d) for g, d in json.load(open(uni_path)).items()}

    scores_path = _fetch(args.knn_run, "scores_by_task.json") or _fetch(args.knn_run, "knn_scores_by_task.json")
    if scores_path is None:
        raise SystemExit(f"no scores_by_task.json / knn_scores_by_task.json under {args.knn_run}")
    src_scores = json.load(open(scores_path))

    flat: dict[str, float] = {}
    for task, sc in src_scores.items():
        for h, v in sc.items():
            flat.setdefault(h, v)
    by_group = {g: {h: flat[h] for h in hs if h in flat} for g, hs in groups.items()}
    by_group = {g: d for g, d in by_group.items() if d}
    n_matched = sum(len(d) for d in by_group.values())
    all_hashes = {h for hs in groups.values() for h in hs}
    unmatched = sorted(set(flat) - all_hashes)
    if n_matched == 0:
        raise SystemExit("zero scored hashes overlap the viz run's universe — wrong run pair?")

    meta_path = _fetch(args.knn_run, "scores_meta.json")
    meta = json.load(open(meta_path)) if meta_path else {}
    meta.setdefault("higher_is_worse", False)
    meta.update({"scores_from": args.knn_run, "n_matched": n_matched, "n_unmatched": len(unmatched),
                 "paired_at": datetime.now(timezone.utc).isoformat()})

    out_scores = tmp / "scores_by_task.json"
    out_scores.write_text(json.dumps(by_group, indent=2))
    out_meta = tmp / "scores_meta.json"
    out_meta.write_text(json.dumps(meta, indent=2))
    if Path(args.viz_run).is_dir():
        (Path(args.viz_run) / "scores_by_task.json").write_text(out_scores.read_text())
        (Path(args.viz_run) / "scores_meta.json").write_text(out_meta.read_text())
    else:
        _vol_put(args.volume, args.env, out_scores, f"{args.viz_run}/scores_by_task.json")
        _vol_put(args.volume, args.env, out_meta, f"{args.viz_run}/scores_meta.json")
    print(
        f"Paired {n_matched} scores into {len(by_group)} group(s) of {args.viz_run} "
        f"({len(unmatched)} scored hashes unmatched)."
    )
    print("The hosted viewer serves the new scores on its next cold start "
          "(or redeploy latent_viz_app to force it).")


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
    cal.add_argument("--labels", default=None, help="filled labeling sheet CSV")
    cal.add_argument(
        "--selection-labels",
        default=None,
        help="viewer selection JSON whose per-episode ✓/✗ labels to use (combinable with --labels)",
    )
    cal.add_argument(
        "--metric",
        default="primary_score",
        help="per-episode metric to threshold (default primary_score)",
    )
    cal.set_defaults(func=cmd_calibrate)

    apply_ = sub.add_parser(
        "apply", help="viewer selection(s) → eps_to_use allowlist + provenance sidecar"
    )
    apply_.add_argument(
        "--selection", nargs="+", required=True,
        help="one or more viewer_selection_*.json exports",
    )
    apply_.add_argument(
        "--universe", default=None,
        help="episode_hashes.json of the viz run — episodes not covered by any selection default to KEEP",
    )
    apply_.add_argument(
        "--out", required=True,
        help="output allowlist path (egomimic/hydra_configs/data/extra/<name>.json)",
    )
    apply_.set_defaults(func=cmd_apply)

    pair = sub.add_parser(
        "pair", help="re-pair a grading run's scores into an existing viz run (no re-embed)"
    )
    pair.add_argument("--viz-run", required=True, help="volume-relative (or local) viz run dir")
    pair.add_argument("--knn-run", required=True, help="volume-relative (or local) grading run dir")
    pair.add_argument("--volume", default="egoverse-training-outputs")
    pair.add_argument("--env", default="robotics")
    pair.set_defaults(func=cmd_pair)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
