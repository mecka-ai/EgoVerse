"""Dataset indexer + WebDataset shard materializer, Modal-native.

Builds the per-sample tar shards (<key>.jpg + <key>.npy) the loaders consume — loader-agnostic, so
the same shards feed data=mecka_all_energon (after `energon prepare`) or data=mecka_all_sb. Every
volume is mounted on every function (read anything directly). Pipeline:

  build_index   one threaded container: scan the dataset → manifest + norm stats. The expensive,
                once-per-dataset step; reads the mounted volume directly, never pulls to local.
  plan          cheap: partition the manifest's valid anchors into a random-EPISODE shard
                collection sized to ~target_shard_mb. Returns a human-readable collection id
                like "2gib-a3f9c1d2" — every collection is uniquely named, so nothing collides.
  materialize   one container per shard (Modal load-balances): pull each episode to local NVMe
                in parallel (plain byte copies — never zarr over the network), read it locally,
                bake each valid anchor into a self-contained WebDataset record (<key>.jpg + .npy).
  wipe          delete a collection (plans + shards), or 'all'.

Energon: run `energon prepare <shard_dir> --sample-type CrudeWebdataset` once, then
data=mecka_all_energon (see egomimic/rldb/energon/mecka_energon.py).

    modal run egomimic/modal/build_sb_shards.py::main                        # index + a 1 GiB collection
    modal run egomimic/modal/build_sb_shards.py::main --target-shard-mb 512  # different shard size
    modal run egomimic/modal/build_sb_shards.py::main --do status --cid <id>
    modal run egomimic/modal/build_sb_shards.py::main --do wipe   --cid all
"""

from __future__ import annotations

import os

import modal

os.environ.setdefault("MODAL_ENVIRONMENT", "robotics")

# ============================ CONFIG — edit for your workspace ============================
APP_NAME = "egoverse-sb-shards"

# Modal volumes (created if missing).
DATA_VOLUME = "mecka_data_v2"                # source: one <episode>.zarr directory per episode
INDEX_VOLUME = "egoverse-mecka-index"        # manifest.json, norm_stats.json, plans/<cid>/
SHARDS_VOLUME = "mecka-energon"              # output: materialized/<cid>/<shard>.tar
DATA_MOUNT, INDEX_MOUNT, SHARDS_MOUNT = "/mnt/zarr-data", "/index", "/mnt/shards"

# Dataset schema — the raw zarr keys baked into each record. MUST match the loader's decode()
# (egomimic/rldb/zarr/sb_shard_dataset.py): the npy stores action_l/_r + proprio_l/_r/_head from
# these pose keys, and the jpg is IMAGE_KEY.
IMAGE_KEY = "images.front_1"
POSE_KEYS = ("left.obs_ee_pose", "right.obs_ee_pose", "obs_head_pose")
BYTES_PER_SAMPLE = 62 * 1024   # rough; only used to size shards to ~target_shard_mb

# Materialize fan-out. 96×128 threads once saturated the volume net link → heartbeat starvation;
# 64 containers × 24 copy-threads (= ~1536 concurrent reads) was the safe ceiling.
MAX_SHARD_CONTAINERS = 64
COPY_THREADS = 24
# ==========================================================================================

app = modal.App(APP_NAME)
image = modal.Image.debian_slim().pip_install("numpy", "zarr==3.1.5")

data = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
index = modal.Volume.from_name(INDEX_VOLUME, create_if_missing=True)
shards = modal.Volume.from_name(SHARDS_VOLUME, create_if_missing=True)
DATA, INDEX, SHARDS = DATA_MOUNT, INDEX_MOUNT, SHARDS_MOUNT
VOLUMES = {DATA: data, INDEX: index, SHARDS: shards}  # mounted on every function


def scan_episode(g, horizon):
    """(n_frames, valid_anchor_indices, {key: norm-partials}) for one open zarr group.
    An anchor is valid iff its [i, i+horizon) window has no zero-norm-quaternion frame."""
    import numpy as np

    bad, partials = None, {}
    for k in POSE_KEYS:
        if k not in g:
            continue
        a = np.asarray(g[k][:], dtype=np.float64)
        partials[k] = (a.shape[0], a.sum(0), (a * a).sum(0), a.min(0), a.max(0))
        q = np.linalg.norm(a[:, 3:7], axis=1) <= 1e-6
        bad = q if bad is None else (bad | q)
    n = 0 if bad is None else len(bad)
    cs = np.concatenate([[0], np.cumsum(~bad)]) if n else None
    valid = [
        i for i in range(n) if cs[min(i + horizon, n)] - cs[i] == min(horizon, n - i)
    ]
    return n, valid, partials


@app.function(image=image, volumes=VOLUMES, cpu=16.0, memory=32768, timeout=7200)
def build_index(
    n_episodes: int = 0, horizon: int = 30, threads: int = 32, force: bool = False
) -> dict:
    """Scan every episode (threaded, reading the mounted volume directly) → manifest + norm stats.
    Idempotent: reuses an existing manifest unless force=True, so re-runs never re-scan."""
    import json
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import zarr

    index.reload()
    if not force and os.path.exists(f"{INDEX}/manifest.json"):
        m = json.load(open(f"{INDEX}/manifest.json"))
        return {
            "episodes": m["n_episodes"],
            "valid_anchors": m["total_valid_anchors"],
            "reused": True,
        }
    data.reload()
    eps = sorted(d for d in os.listdir(DATA) if d.endswith(".zarr"))[
        : n_episodes or None
    ]
    norm, lock = {}, threading.Lock()

    def scan(ep):
        try:
            n, valid, partials = scan_episode(
                zarr.open_group(f"{DATA}/{ep}", mode="r"), horizon
            )
        except Exception as e:
            return ep, {
                "frame_count": -1,
                "n_valid": 0,
                "valid_anchors": [],
                "error": str(e)[:120],
            }
        with lock:
            for k, (c, s, sq, mn, mx) in partials.items():
                p = norm.setdefault(
                    k, dict(count=0, sum=0.0, sumsq=0.0, min=np.inf, max=-np.inf)
                )
                p["count"] += c
                p["sum"] = p["sum"] + s
                p["sumsq"] = p["sumsq"] + sq
                p["min"] = np.minimum(p["min"], mn)
                p["max"] = np.maximum(p["max"], mx)
        return ep, {"frame_count": n, "n_valid": len(valid), "valid_anchors": valid}

    with ThreadPoolExecutor(max_workers=threads) as ex:
        manifest = dict(ex.map(scan, eps))

    norm_stats = {}
    for k, p in norm.items():
        c = max(p["count"], 1)
        mean = p["sum"] / c
        std = np.sqrt(np.maximum(p["sumsq"] / c - mean * mean, 0.0))
        norm_stats[k] = dict(
            count=int(p["count"]),
            mean=mean.tolist(),
            std=std.tolist(),
            min=p["min"].tolist(),
            max=p["max"].tolist(),
        )

    def write(path, obj):
        with open(path + ".tmp", "w") as f:
            json.dump(obj, f)
        os.replace(path + ".tmp", path)

    total_valid = sum(m["n_valid"] for m in manifest.values())
    write(
        f"{INDEX}/manifest.json",
        {
            "horizon": horizon,
            "n_episodes": len(manifest),
            "total_valid_anchors": total_valid,
            "episodes": manifest,
        },
    )
    write(f"{INDEX}/norm_stats.json", norm_stats)
    index.commit()
    return {
        "episodes": len(manifest),
        "valid_anchors": total_valid,
        "norm_keys": sorted(norm_stats),
    }


@app.function(image=image, volumes=VOLUMES, cpu=8.0, memory=16384, timeout=86400)
def pipeline(
    cid: str,
    target_shard_mb: float = 1024.0,
    n_episodes: int = 0,
    seed: int = 42,
    force_index: bool = False,
    one_per_episode: bool = False,
) -> dict:
    """The whole job, server-side (so the local caller can disconnect): ensure the manifest
    exists (idempotent — only scans if missing/force), partition it into this collection's
    random-episode shard plans (instant, skipped if `cid` is already planned → resumable), then
    spawn_map one container per shard. Returns once dispatched; track with do=status."""
    import glob
    import json
    import os
    import random

    index.reload()
    if force_index or not os.path.exists(f"{INDEX}/manifest.json"):
        build_index.remote(n_episodes=n_episodes, force=force_index)
        index.reload()

    pdir = f"{INDEX}/plans/{cid}"
    if not os.path.isdir(pdir):  # plan from the manifest (instant)
        man = json.load(open(f"{INDEX}/manifest.json"))
        H = man["horizon"]
        pool = [
            (ep, m["valid_anchors"])
            for ep, m in man["episodes"].items()
            if m["n_valid"]
        ]
        random.Random(seed).shuffle(pool)
        if n_episodes:                       # subset collection (e.g. shard-size sweep): cap the
            pool = pool[:n_episodes]         # planned episodes so the same N land in every size
        os.makedirs(pdir, exist_ok=True)
        if one_per_episode:
            # One tar per episode, named after the episode (stem without .zarr).
            for ep, valid in pool:
                name = ep[:-5] if ep.endswith(".zarr") else ep
                with open(f"{pdir}/{name}.json.tmp", "w") as f:
                    json.dump({"horizon": H, "episodes": [{"episode": ep, "valid_anchors": valid}]}, f)
                os.replace(f"{pdir}/{name}.json.tmp", f"{pdir}/{name}.json")
        else:
            target = max(1, int(target_shard_mb * 1024 * 1024 / BYTES_PER_SAMPLE))
            plans, cur, cur_n = [], [], 0
            for ep, valid in pool:
                cur.append({"episode": ep, "valid_anchors": valid})
                cur_n += len(valid)
                if cur_n >= target:
                    plans.append(cur)
                    cur, cur_n = [], 0
            if cur:
                plans.append(cur)
            for sid, episodes in enumerate(plans):
                with open(f"{pdir}/{sid:05d}.json.tmp", "w") as f:
                    json.dump({"horizon": H, "episodes": episodes}, f)
                os.replace(f"{pdir}/{sid:05d}.json.tmp", f"{pdir}/{sid:05d}.json")
        index.commit()

    names = sorted(
        os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{pdir}/*.json")
    )
    # Fan out one container per shard and BLOCK here. `pipeline` is itself spawned (server-side),
    # so this .map runs under it — it survives the local caller disconnecting, and keeping pipeline
    # alive until the shards finish is what stops the app (and the children) from being torn down.
    ok = [
        r
        for r in materialize_shard.map([cid] * len(names), names)
        if r.get("status") == "ok"
    ]
    return {
        "cid": cid,
        "n_shards": len(names),
        "materialized": len(ok),
        "GB": round(sum(r.get("MB", 0) for r in ok) / 1024, 1),
    }


@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=8.0,
    memory=16384,
    timeout=3600,
    retries=2,
    max_containers=MAX_SHARD_CONTAINERS,
)
def materialize_shard(cid: str, shard_name: str) -> dict:
    import io
    import json
    import os
    import shutil
    import tarfile
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import zarr

    tar_path = f"{SHARDS}/materialized/{cid}/{shard_name}.tar"
    if os.path.exists(tar_path):  # resume within a collection
        return {"shard_id": shard_name, "status": "skip"}
    os.makedirs(os.path.dirname(tar_path), exist_ok=True)
    index.reload()
    plan = json.load(open(f"{INDEX}/plans/{cid}/{shard_name}.json"))
    H, t0 = plan["horizon"], time.time()

    def pull(ep):  # bulk-copy ONE episode to local NVMe
        src, dst = f"{DATA}/{ep}", f"/tmp/{ep}"  # plain threaded byte copies (no zarr)
        jobs = [
            (s := os.path.join(dp, f), os.path.join(dst, os.path.relpath(s, src)))
            for dp, _, fs in os.walk(src)
            for f in fs
        ]

        def cp(sd):
            s, d = sd
            os.makedirs(os.path.dirname(d), exist_ok=True)
            with open(s, "rb") as r, open(d, "wb") as w:
                w.write(r.read())

        with ThreadPoolExecutor(max_workers=COPY_THREADS) as ex:
            list(ex.map(cp, jobs))
        return dst

    pad = (
        lambda a: a
        if len(a) >= H
        else np.concatenate([a, np.repeat(a[-1:], H - len(a), 0)])
    )
    n, tmp = 0, tar_path + ".tmp"
    with tarfile.open(tmp, "w") as tar:

        def add(name, blob):
            ti = tarfile.TarInfo(name)
            ti.size = len(blob)
            tar.addfile(ti, io.BytesIO(blob))

        for e in plan["episodes"]:  # one episode at a time → /tmp bounded
            ep, local = e["episode"], None
            try:
                local = pull(ep)  # threaded pull, then read LOCALLY
                g = zarr.open_group(local, mode="r")
                imgs, pose = (
                    g[IMAGE_KEY][:],
                    {k: np.asarray(g[k][:]) for k in POSE_KEYS},
                )
                key = ep[:-5] if ep.endswith(".zarr") else ep
                for a in e["valid_anchors"]:
                    meta = {
                        "episode": ep,
                        "anchor": a,
                        "action_l": pad(pose["left.obs_ee_pose"][a : a + H]),
                        "action_r": pad(pose["right.obs_ee_pose"][a : a + H]),
                        "proprio_l": pose["left.obs_ee_pose"][a],
                        "proprio_r": pose["right.obs_ee_pose"][a],
                        "proprio_head": pose["obs_head_pose"][a],
                    }
                    buf = io.BytesIO()
                    np.save(buf, np.array(meta, dtype=object))
                    add(f"{key}_{a:06d}.jpg", bytes(imgs[a]))
                    add(f"{key}_{a:06d}.npy", buf.getvalue())
                    n += 1
            except Exception:
                continue
            finally:
                if local:
                    shutil.rmtree(local, ignore_errors=True)
    os.replace(tmp, tar_path)
    shards.commit()
    return {
        "shard_id": shard_name,
        "status": "ok",
        "samples": n,
        "MB": round(os.path.getsize(tar_path) / 1e6, 1),
        "seconds": round(time.time() - t0, 1),
    }


@app.function(image=image, volumes=VOLUMES, timeout=300)
def status(cid: str) -> dict:
    """Materialization progress for a collection (materialized tars vs planned shards)."""
    import glob

    index.reload()
    shards.reload()
    planned = len(glob.glob(f"{INDEX}/plans/{cid}/*.json"))
    done = len(glob.glob(f"{SHARDS}/materialized/{cid}/*.tar"))
    return {"collection": cid, "planned": planned, "materialized": done}


@app.function(
    image=image.pip_install("megatron-energon"),
    volumes=VOLUMES,
    cpu=4.0,
    memory=8192,
    timeout=1800,
)
def energon_prepare(cid: str) -> dict:
    """Write the .nv-meta sidecar beside a materialized shard collection (one-time step).

    Must be run once per collection before data=mecka_all_energon can read it.
    Equivalent to: energon prepare <shard_dir> --non-interactive --split-ratio 1.0,0,0
                                   --sample-type CrudeWebdataset
    """
    import subprocess

    shards.reload()
    shard_dir = f"{SHARDS}/materialized/{cid}"
    result = subprocess.run(
        [
            "energon", "prepare", shard_dir,
            "--non-interactive",
            "--split-ratio", "1.0,0,0",
            "--sample-type", "CrudeWebdataset",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"energon prepare failed:\n{result.stderr}")
    shards.commit()
    return {"cid": cid, "shard_dir": shard_dir, "stdout": result.stdout[:500]}


@app.function(image=image, volumes=VOLUMES, timeout=900)
def wipe(cid: str = "all") -> dict:
    """Delete a shard collection (its plans + materialized shards), or everything with 'all'."""
    import shutil

    targets = (
        [f"{INDEX}/plans", f"{SHARDS}/materialized"]
        if cid == "all"
        else [f"{INDEX}/plans/{cid}", f"{SHARDS}/materialized/{cid}"]
    )
    for t in targets:
        shutil.rmtree(t, ignore_errors=True)
    index.commit()
    shards.commit()
    return {"wiped": cid}


@app.local_entrypoint()
def main(
    do: str = "make",
    target_shard_mb: float = 1024.0,
    cid: str = "",
    n_episodes: int = 0,
    force: bool = False,
    one_per_episode: bool = False,
):
    """One entrypoint. Examples:
        modal run egomimic/modal/build_sb_shards.py::main                               # build + 1 GiB collection
        modal run egomimic/modal/build_sb_shards.py::main --target-shard-mb 512         # different shard size
        modal run egomimic/modal/build_sb_shards.py::main --one-per-episode --n-episodes 1000  # 1 ep/shard
        modal run egomimic/modal/build_sb_shards.py::main --do make --cid 1gib-abc123   # resume
        modal run egomimic/modal/build_sb_shards.py::main --do prepare --cid 1gib-abc123  # write .nv-meta
        modal run egomimic/modal/build_sb_shards.py::main --do status --cid 1gib-abc123
        modal run egomimic/modal/build_sb_shards.py::main --do wipe   --cid all
    Spawns a server-side function and waits on the handle, so a local disconnect doesn't cancel it."""
    import uuid

    if do == "make":
        prefix = "1ep" if one_per_episode else f"{target_shard_mb / 1024:g}gib"
        cid = cid or f"{prefix}-{uuid.uuid4().hex[:8]}"
        fc = pipeline.spawn(
            cid,
            target_shard_mb=target_shard_mb,
            n_episodes=n_episodes,
            force_index=force,
            one_per_episode=one_per_episode,
        )
        print(fc.get())
    elif do == "prepare":
        if not cid:
            print("--cid required for prepare")
            return
        print(energon_prepare.remote(cid))
    elif do == "status":
        print(status.spawn(cid).get())
    elif do == "wipe":
        print(wipe.spawn(cid or "all").get())
    else:
        print(f"unknown do={do!r}; use make | prepare | status | wipe")
