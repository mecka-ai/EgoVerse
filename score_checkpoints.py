"""Score every arm's checkpoints on the domain's common eval set.

The 9 runs each validate on their own arm, so their train/val curves cannot be
compared with each other. This scores all of them on one set per domain that no
arm in that domain trained on (_eval_common__<domain>.json, built by
build_common_eval.py), which is what actually answers whether the quality score
predicts model performance.

Each checkpoint keeps its OWN norm stats -- they ride along in the checkpoint's
hyper_parameters and are part of the model -- and only the data is held common.

Batches are drawn with a fixed seed and the same count for every checkpoint, so
the numbers are comparable run to run rather than merely indicative.

    modal run -e robotics score_checkpoints.py --domain cleaning-sanitation
"""
import json
import os
from pathlib import Path

import modal

from egomimic.modal.modal_setup import CFG, image as train_image  # noqa: F401

ZARR_MOUNT = "/mnt/zarr-data"
OUT_MOUNT = "/root/EgoVerse/logs"

app = modal.App("score-checkpoints")
zarr_vol = modal.Volume.from_name("mecka-zarr-qaexp", create_if_missing=False)
out_vol = modal.Volume.from_name("egoverse-training-outputs", create_if_missing=False)

N_EVAL_BATCHES = int(os.environ.get("SCORE_N_BATCHES", "60"))
BATCH_SIZE = 48


@app.function(
    image=train_image,
    gpu="H200:1",
    cpu=16,
    memory=131072,
    timeout=14400,
    ephemeral_disk=600 * 1024,
    volumes={ZARR_MOUNT: zarr_vol, OUT_MOUNT: out_vol},
)
def score(domain: str, run_globs: list[str]) -> dict:
    import torch
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.rldb.zarr.prefetch_dataset import (
        PrefetchedMapDataset,
        ZarrDirEpisodeResolver,
    )
    from egomimic.rldb.embodiment.human import Mecka

    eval_manifest = f"{ZARR_MOUNT}/_arms/_eval_common__{domain}.json"
    n_eval = len(json.load(open(eval_manifest)))
    print(f"domain={domain} common eval episodes={n_eval}")

    results = {}
    for pattern in run_globs:
        ckpts = sorted(Path(OUT_MOUNT).glob(pattern))
        if not ckpts:
            print(f"  no checkpoints yet for {pattern}")
            continue
        for ckpt in ckpts:
            arm = ckpt.parts[len(Path(OUT_MOUNT).parts)]
            print(f"\n=== {arm} :: {ckpt.name} ===")

            model = ModelWrapper.load_from_checkpoint(str(ckpt), map_location="cuda")
            model.eval().cuda()

            resolver = ZarrDirEpisodeResolver(
                ZARR_MOUNT,
                catalog_cache=f"{ZARR_MOUNT}/_catalog_cache.json",
                eps_to_use=eval_manifest,
                valid_ratio=0.0,          # evaluate on all of it
                key_map=Mecka.get_keymap(mode="cartesian"),
                transform_list=Mecka.get_transform_list(mode="cartesian_6d"),
                norm_stats=model.model.data_schematic.norm_stats,
            )
            ds = PrefetchedMapDataset(
                resolver=resolver,
                mode="train",              # "train" = use the whole catalog, no split
                episodes_per_epoch=n_eval,
                cache_dir="/cache/eval_cache",
                pool_size_gb=200,
                lookahead_epochs=1.0,
                n_copy_threads=48,
                prepare_timeout_s=7200,
            )
            ds.prepare_epoch(0)
            loader = torch.utils.data.DataLoader(
                ds, batch_size=BATCH_SIZE, num_workers=12,
                shuffle=False, pin_memory=True, persistent_workers=False,
            )

            tot, n = 0.0, 0
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= N_EVAL_BATCHES:
                        break
                    batch = model.model.process_batch_for_training(batch)
                    pred = model.model.forward_training(batch)
                    losses = model.model.compute_losses(pred, batch)
                    tot += float(losses["action_loss"])
                    n += 1
            mean = tot / max(n, 1)
            results.setdefault(arm, {})[ckpt.name] = round(mean, 6)
            print(f"  {arm} {ckpt.name}: action_loss={mean:.6f} over {n} batches")
            del model
            torch.cuda.empty_cache()

    print("\n=== summary ===")
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def main(domain: str = "cleaning-sanitation"):
    globs = [f"qaexp-{domain}-{arm}/*/checkpoints/*.ckpt"
             for arm in ("top", "bottom", "random")]
    print(json.dumps(score.remote(domain, globs), indent=2))
