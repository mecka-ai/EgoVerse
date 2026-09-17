"""What metric keys did a round-2 run actually write?

Determines whether any Valid/* metric reached W&B, or whether the validation
loss was computed but never emitted.
"""
import json
from pathlib import Path

import modal

OUT = "/root/EgoVerse/logs"
image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("dump-wandb-keys", image=image)
out_vol = modal.Volume.from_name("egoverse-training-outputs", create_if_missing=False)

RUN = ("qaexp-cleaning-sanitation-top/"
       "qaexp-scoring-cleaning-sanitation-top_2026-09-17_01-04-10")


@app.function(volumes={OUT: out_vol}, cpu=2, memory=4096, timeout=1800)
def dump():
    out_vol.reload()
    run = Path(OUT) / RUN
    for s in sorted(run.glob("wandb/*/files/wandb-summary.json")):
        d = json.loads(s.read_text())
        keys = sorted(d.keys())
        print(f"\n{s.parent.parent.name}: {len(keys)} keys")
        print("  Train/*:", [k for k in keys if k.startswith("Train/")])
        print("  Valid/*:", [k for k in keys if k.startswith("Valid/")] or "NONE")
        print("  other  :", [k for k in keys
                             if not k.startswith(("Train/", "Valid/"))][:12])
    return "ok"


@app.local_entrypoint()
def main():
    print(dump.remote())
