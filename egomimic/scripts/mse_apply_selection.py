#!/usr/bin/env python3
"""Turn MSE-viewer selection export(s) into a training ``eps_to_use`` allowlist.

The MSE viewer (``build_mse_viewer.py``) has two export buttons:

  * ``⬇ eps_to_use`` — already a flat hash list; drop it straight into
    ``egomimic/hydra_configs/data/extra/<name>.json`` and point a data config's
    ``resolver.eps_to_use`` at it. No further step needed.
  * ``⬇ selection`` — a richer ``viewer_selection_*.json`` (per-task kept/removed
    + provenance). This tool converts one or more of those into the same flat
    allowlist plus a ``.provenance.json`` sidecar (reproducible runs).

Usage
-----
    python egomimic/scripts/mse_apply_selection.py \\
        --selection viewer_selection_<ts>.json \\
        --out egomimic/hydra_configs/data/extra/my_subset.json

``--universe`` (the run's episode_hashes.json, or any hash list) makes episodes
not covered by any selection's tasks default to KEEP — matching the viewer's
"unscored episodes are kept" behavior. The selection's embedded ``universe`` is
used automatically when ``--universe`` is omitted.

Then in a data config:

    resolver:
      eps_to_use: egomimic/hydra_configs/data/extra/my_subset.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def cmd_apply(args: argparse.Namespace) -> None:
    """Union kept episodes from viewer selection(s) into a training allowlist."""
    per_task: dict[str, dict] = {}
    sources = []
    embedded_universe: set[str] = set()

    for sel_path in args.selection:
        p = Path(sel_path)
        raw = p.read_bytes()
        sel = json.loads(raw)
        tasks = sel.get("tasks") or {}
        if not tasks:
            raise SystemExit(f"{p}: no tasks — is this an MSE-viewer selection export?")
        for uni_h in sel.get("universe") or []:
            embedded_universe.add(uni_h)
        for task, block in tasks.items():
            k = set(block.get("kept") or [])
            r = set(x["hash"] for x in (block.get("removed") or []))
            if task in per_task:
                print(
                    f"WARNING: task {task!r} in multiple selections — intersecting kept sets"
                )
                k &= set(per_task[task]["kept"])
                r |= set(per_task[task]["removed"])
            per_task[task] = {"kept": sorted(k), "removed": sorted(r), "source": str(p)}
        sources.append(
            {
                "file": str(p),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "created_at": sel.get("created_at"),
                "tasks": {
                    t: {
                        "kept": len(b.get("kept") or []),
                        "removed": len(b.get("removed") or []),
                    }
                    for t, b in tasks.items()
                },
            }
        )

    kept: set[str] = set()
    covered: set[str] = set()
    for block in per_task.values():
        kept |= set(block["kept"])
        covered |= set(block["kept"]) | set(block["removed"])

    n_default = 0
    universe: set[str] = set()
    if args.universe:
        universe = set(json.load(open(args.universe)))
    elif embedded_universe:
        universe = embedded_universe
        print(f"Using selection's embedded universe ({len(universe)} episodes).")
    if universe:
        default_kept = universe - covered
        n_default = len(default_kept)
        kept |= default_kept

    print(f"{'task':<40} {'kept':>6} {'removed':>8}")
    for t, b in sorted(per_task.items()):
        print(f"{t:<40} {len(b['kept']):>6} {len(b['removed']):>8}")
    if universe:
        print(f"{'(uncovered by any selection — kept)':<40} {n_default:>6}")
    else:
        print(
            "NOTE: no universe — episodes outside the selections' tasks are NOT in "
            "the allowlist. Pass --universe (the run's episode_hashes.json) to keep them."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(kept)))
    prov = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "universe": str(args.universe)
        if args.universe
        else "(embedded)"
        if universe
        else None,
        "n_kept": len(kept),
        "n_default_kept": n_default,
        "allowlist_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    Path(str(out).removesuffix(".json") + ".provenance.json").write_text(
        json.dumps(prov, indent=2)
    )
    print(f"\nWrote {len(kept)} hashes → {out} (+ .provenance.json sidecar)")
    print("Use it in a data config resolver:\n")
    print(f"    resolver:\n      eps_to_use: {out}\n")
    print("(Commit + push the JSON — Modal containers read it from the repo clone.)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--selection",
        nargs="+",
        required=True,
        help="one or more viewer_selection_*.json",
    )
    ap.add_argument(
        "--universe",
        default=None,
        help="episode_hashes.json — uncovered episodes default to KEEP (else the "
        "selection's embedded universe is used)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="output allowlist path (egomimic/hydra_configs/data/extra/<name>.json)",
    )
    ap.set_defaults(func=cmd_apply)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
