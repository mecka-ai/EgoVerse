"""Score one or more checkpoints on a task family's shared held-out eval set.

Runs as a SUBPROCESS inside the prepared repo, exactly like trainHydra does.
That matters: _prepare_repo pip-installs openpi's patched transformers 4.53.2
and overlays its gemma/paligemma modeling files, and those only take effect in a
fresh interpreter. Doing this in-process instead builds the model from the old
modeling code, and the checkpoint then fails to load with mismatched keys like
`paligemma_with_expert.gemma_expert...input_layernorm.dense.weight`.

Each checkpoint keeps its OWN norm stats -- they are part of the model, stored
in the checkpoint's hyper_parameters -- and only the eval data is held common,
so the arms of a task family are compared on identical, never-trained-on
episodes.

    python egomimic/scripts/score_common_eval.py \
        --ckpt <path.ckpt> [--ckpt ...] \
        --eval-manifest /mnt/zarr-data/_arms/_eval_common__dish-handling.json \
        --zarr-dir /mnt/zarr-data --n-batches 60 --out results.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--eval-manifest", required=True)
    ap.add_argument("--zarr-dir", default="/mnt/zarr-data")
    ap.add_argument("--n-batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--cache-dir", default="/cache/eval_cache")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import torch

    # torch>=2.6 defaults torch.load to weights_only=True, which refuses the
    # omegaconf containers Lightning stores in hyper_parameters. These are our
    # own artifacts, so allow the full unpickle.
    _torch_load = torch.load

    def _load_trusted(*a, **kw):
        kw["weights_only"] = False
        return _torch_load(*a, **kw)

    torch.load = _load_trusted

    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.rldb.embodiment.human import Mecka
    from egomimic.rldb.zarr.prefetch_dataset import (
        PrefetchedMapDataset,
        ZarrDirEpisodeResolver,
    )

    n_eval = len(json.load(open(args.eval_manifest)))
    print(f"shared eval set: {n_eval} episodes from {args.eval_manifest}",
          flush=True)

    results: dict[str, float] = {}
    for ckpt in args.ckpt:
        name = "/".join(Path(ckpt).parts[-4:])
        print(f"\n=== {name} ===", flush=True)

        model = ModelWrapper.load_from_checkpoint(ckpt, map_location="cuda")
        model.eval().cuda()

        resolver = ZarrDirEpisodeResolver(
            args.zarr_dir,
            catalog_cache=f"{args.zarr_dir}/_catalog_cache.json",
            eps_to_use=args.eval_manifest,
            valid_ratio=0.0,
            key_map=Mecka.get_keymap(mode="cartesian"),
            transform_list=Mecka.get_transform_list(mode="cartesian_6d"),
            norm_stats=model.model.data_schematic.norm_stats,
        )
        ds = PrefetchedMapDataset(
            resolver=resolver,
            mode="train",          # whole catalog, no further split
            episodes_per_epoch=n_eval,
            cache_dir=args.cache_dir,
            pool_size_gb=200,
            lookahead_epochs=1.0,
            n_copy_threads=48,
            prepare_timeout_s=7200,
        )
        ds.prepare_epoch(0)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False, pin_memory=True, persistent_workers=False,
        )

        tot, n = 0.0, 0
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= args.n_batches:
                    break
                batch = model.model.process_batch_for_training(batch)
                pred = model.model.forward_training(batch)
                losses = model.model.compute_losses(pred, batch)
                tot += float(losses["action_loss"])
                n += 1

        mean = tot / max(n, 1)
        results[name] = round(mean, 6)
        print(f"  action_loss={mean:.6f} over {n} batches", flush=True)

        del model
        torch.cuda.empty_cache()

    print("\n=== RESULTS_JSON ===", flush=True)
    print(json.dumps(results), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
