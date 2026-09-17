"""Pull whatever the 9 arm runs actually logged before they died.

No checkpoints survived any round, so there is no model to evaluate. What does
exist is the W&B summary each run wrote while training, plus the per-run counts
of episodes the dataloader discarded. Gather both so the scoring question can be
answered with real numbers where possible -- and so the limits of those numbers
are visible rather than assumed.

CPU-only and tiny; this costs cents, not dollars.
"""
import json
from pathlib import Path

import modal

OUT = "/root/EgoVerse/logs"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("collect-results", image=image)
out_vol = modal.Volume.from_name("egoverse-training-outputs", create_if_missing=False)

DOMAINS = ["cleaning-sanitation", "organization-stocking", "dish-handling"]
ARMS = ["top", "bottom", "random"]


@app.function(volumes={OUT: out_vol}, cpu=2, memory=4096, timeout=1800)
def collect():
    out_vol.reload()
    root = Path(OUT)
    rows = []
    for dom in DOMAINS:
        for arm in ARMS:
            run_root = root / f"qaexp-{dom}-{arm}"
            if not run_root.exists():
                continue
            for run in sorted(run_root.iterdir()):
                summaries = list(run.glob("wandb/*/files/wandb-summary.json"))
                ckpts = list(run.glob("checkpoints/*.ckpt"))
                best = None
                for s in summaries:
                    try:
                        d = json.loads(s.read_text())
                    except Exception:
                        continue
                    if best is None or d.get("epoch", -1) >= best.get("epoch", -1):
                        best = d
                if best is None and not ckpts:
                    continue
                rows.append({
                    "domain": dom, "arm": arm, "run": run.name,
                    "epoch": best.get("epoch") if best else None,
                    "train_loss": best.get("Train/Loss") if best else None,
                    "action_loss": best.get("Train/action_loss") if best else None,
                    "valid_loss": (best or {}).get("Valid/Loss"),
                    "grad_norm": (best or {}).get("Train/policy_grad_norms_raw"),
                    "n_checkpoints": len(ckpts),
                })

    rows.sort(key=lambda r: (r["domain"], r["arm"], r["run"]))
    for r in rows:
        print(f"{r['domain']:22s} {r['arm']:7s} {r['run'][-19:]} "
              f"epoch={str(r['epoch']):>6s} train_loss={str(r['train_loss'])[:9]:>9s} "
              f"valid={str(r['valid_loss'])[:9]:>9s} ckpts={r['n_checkpoints']}")
    print(f"\n{len(rows)} run records; "
          f"{sum(r['n_checkpoints'] for r in rows)} checkpoints in total")
    return rows


@app.local_entrypoint()
def main():
    print(json.dumps(collect.remote(), indent=2))
