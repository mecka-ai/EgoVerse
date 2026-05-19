#!/usr/bin/env python3
"""
Extract episode hash lists from a DemInf scores_by_task.json.

Writes two JSON files to egomimic/hydra_configs/data/extra/:
  <name>_top<K>pct.json  — top K% from each task (NaN episodes excluded)
  <name>_all.json        — all episodes in the curation run (scored + NaN)

Use the resulting files as eps_to_use in a data config.

Usage (local file):
  python egomimic/scripts/extract_curated_hashes.py \\
      /tmp/scores_by_task.json --top-k 0.7 --name mecka_curated

Usage (Modal volume path — auto-downloads):
  python egomimic/scripts/extract_curated_hashes.py \\
      deminf_test/per_task_v3_2026-05-18_22-56-55/scores_by_task.json \\
      --volume egoverse-training-outputs --env robotics \\
      --top-k 0.7 --name mecka_curated
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EXTRA_DIR = REPO_ROOT / "egomimic" / "hydra_configs" / "data" / "extra"


def load_scores(path: str, volume: str | None, env: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)

    if volume is None:
        sys.exit(f"File not found locally: {path}\nPass --volume to download from Modal.")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name

    print(f"Downloading {path} from volume {volume}...")
    subprocess.run(
        ["modal", "volume", "get", "--env", env, volume, path, tmp_path],
        check=True,
    )
    with open(tmp_path) as f:
        return json.load(f)


def extract(scores_by_task: dict, top_k: float) -> tuple[list[str], list[str]]:
    top_hashes: list[str] = []
    all_hashes: list[str] = []

    for task_scores in scores_by_task.values():
        scored = [(h, s) for h, s in task_scores.items() if math.isfinite(s)]
        nan_eps = [h for h, s in task_scores.items() if not math.isfinite(s)]

        all_hashes.extend(h for h, _ in scored)
        all_hashes.extend(nan_eps)

        if nan_eps:
            # Task has unscored episodes — include everything (can't fairly rank)
            top_hashes.extend(h for h, _ in scored)
            top_hashes.extend(nan_eps)
        else:
            # All episodes scored — keep top k%
            # Scores are already sorted highest→lowest by curateModal.py
            n_keep = max(1, math.ceil(len(scored) * top_k)) if scored else 0
            top_hashes.extend(h for h, _ in scored[:n_keep])

    return top_hashes, all_hashes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scores_path", help="Path to scores_by_task.json (local or volume-relative)")
    ap.add_argument("--top-k", type=float, default=0.7, help="Fraction to keep per task (default: 0.7)")
    ap.add_argument("--name", default="mecka_curated", help="Output file prefix (default: mecka_curated)")
    ap.add_argument("--volume", default=None, help="Modal volume name to download from if path is not local")
    ap.add_argument("--env", default="robotics", help="Modal environment (default: robotics)")
    args = ap.parse_args()

    scores_by_task = load_scores(args.scores_path, args.volume, args.env)

    top_hashes, all_hashes = extract(scores_by_task, args.top_k)

    k_pct = int(args.top_k * 100)
    top_path = EXTRA_DIR / f"{args.name}_top{k_pct}pct.json"
    all_path = EXTRA_DIR / f"{args.name}_all.json"

    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    with open(top_path, "w") as f:
        json.dump(top_hashes, f, indent=2)
    with open(all_path, "w") as f:
        json.dump(all_hashes, f, indent=2)

    total_scored = sum(
        sum(1 for s in t.values() if math.isfinite(s)) for t in scores_by_task.values()
    )
    n_tasks = len(scores_by_task)

    print(f"Tasks:          {n_tasks}")
    print(f"Total scored:   {total_scored}")
    print(f"All episodes:   {len(all_hashes)}  → {all_path.relative_to(REPO_ROOT)}")
    print(f"Top {k_pct}% kept:    {len(top_hashes)}  → {top_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
