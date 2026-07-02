#!/usr/bin/env python
"""Build a one-time metadata manifest for a local zarr folder so LocalEpisodeResolver
can use the DEFERRED fast-path (skip the per-worker eager zarr open of every store).

Reads each store's root .zattrs ONCE (parallel) and writes {hash: {num_frames, embodiment}}.
The resolver then constructs ZarrDataset with _total_frames/_embodiment (len known, zarr
opened lazily on first __getitem__) -> fast startup, and validation no longer storms.

Re-run only when the set of episodes changes.

    ./emimic/bin/python egomimic/scripts/abc_process/build_meta_manifest.py \
        --folder /workspace/eva/abc130k_zarr --out /workspace/eva/abc130k_meta.json
"""

import argparse
import glob
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import zarr


def read_one(store_path):
    ep = os.path.basename(store_path)
    if ep.endswith(".zarr"):
        ep = ep[: -len(".zarr")]
    try:
        a = dict(zarr.open_group(store_path, mode="r").attrs)
    except Exception as e:
        return (ep, None, f"open failed: {e}")
    tf = a.get("total_frames")
    if tf is None:
        return (ep, None, "no total_frames")
    return (
        ep,
        {"num_frames": int(tf), "embodiment": a.get("embodiment", "eva_bimanual")},
        None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="/workspace/eva/abc130k_zarr")
    ap.add_argument("--out", default="/workspace/eva/abc130k_meta.json")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    stores = sorted(glob.glob(os.path.join(args.folder, "*.zarr")))
    print(f"[manifest] {len(stores)} stores under {args.folder}")
    manifest, errs = {}, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, fut in enumerate(
            as_completed([ex.submit(read_one, s) for s in stores]), 1
        ):
            ep, meta, err = fut.result()
            if meta is None:
                errs.append((ep, err))
            else:
                manifest[ep] = meta
            if i % 4000 == 0:
                print(f"  {i}/{len(stores)} (ok={len(manifest)} err={len(errs)})")
    json.dump(manifest, open(args.out, "w"))
    print(
        f"[manifest] wrote {len(manifest)} entries to {args.out} in {time.time()-t0:.1f}s; errors={len(errs)}"
    )
    for ep, e in errs[:10]:
        print(f"  ERR {ep}: {e}")


if __name__ == "__main__":
    main()
