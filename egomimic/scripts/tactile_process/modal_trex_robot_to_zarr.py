"""
Full-scale T-Rex robot tactile -> Zarr conversion, run on Modal.

Scales up `trex_robot_to_zarr.py` (the local pilot, 3 episodes from a single
shard) to the whole open-sourced slice of `zekaiwang/trex_dataset` (5464
episodes). Unlike the pilot, this does NOT restrict itself to episodes that
happen to fit in shard 0 -- it covers every episode.

Sharding strategy: grouping by the exact (episode -> file) tuple across all
11 relevant files (1 data parquet + 10 raw-tactile-video streams) fragments
into ~3000 tiny groups with heavy redundant downloads. Instead this groups
episodes by their *data* shard only (48 groups covering all 5464 episodes),
and within each group processes video files one at a time -- for each
distinct (stream, chunk, file) touched by that group's episodes, download
once, decode only the frame ranges needed, immediately free the file, and
write out any episode whose all 10 streams are now collected. That bounds
per-container disk to ~1 data parquet + 1 video file at a time (~300MB),
rather than all ~128 video files a group can touch (~25GB) at once.

Only raw grayscale tactile images + the tactile_force wrench + state/action
are pulled (same scope as the local pilot) -- deform maps and RGB video are
skipped entirely. Output lands in the `tactile-zarr-data` Modal volume under
`tactile_robot/`, not local disk.

Size: measured directly from `meta/episodes/chunk-000/file-000.parquet`
(the full 5464-row index), this is ~5762 physical video shard files at
~200MB target size each -- on the order of 1-1.2TB of video downloaded
through Modal, well beyond the ~400-500GB rough estimate given before this
was measured precisely. Confirm before running at full scale.

Usage:
    # small test: only the first N data-groups
    modal run modal_trex_robot_to_zarr.py --max-groups 1

    # full run, detached
    modal run --detach modal_trex_robot_to_zarr.py
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict

import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ID = "zekaiwang/trex_dataset"
VOLUME_NAME = "tactile-zarr-data"
VOLUME_MOUNT = "/vol/tactile_zarr"
OUT_SUBDIR = "tactile_robot"

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
RAW_STREAMS = [f"tactile_{side}_raw_{finger}" for side in ("left", "right") for finger in FINGERS]

image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0"
).pip_install(
    "numpy", "pandas", "pyarrow", "opencv-python-headless", "zarr>=3.0", "huggingface_hub"
)
app = modal.App("tactile-robot-trex-zarr", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)


@app.function(
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
    memory=8192,
    cpu=2,
    max_containers=24,
    retries=modal.Retries(max_retries=1, initial_delay=10.0, backoff_coefficient=2.0),
)
def process_data_group(data_chunk: int, data_file: int, episodes_meta: list[dict], fps: float) -> dict:
    import shutil
    import tempfile

    import numpy as np
    import pandas as pd
    import zarr
    from huggingface_hub import hf_hub_download

    scratch = tempfile.mkdtemp(prefix=f"trex_g{data_chunk}_{data_file}_")
    os.environ["HF_HOME"] = os.path.join(scratch, "hf_cache")

    def hf_dl(filename: str) -> str:
        return hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset", filename=filename, cache_dir=os.path.join(scratch, "hf_cache")
        )

    ok, skipped, failed = 0, 0, []
    try:
        # Filter out episodes whose output already exists (idempotent reruns).
        pending = []
        for ep in episodes_meta:
            dest = f"{VOLUME_MOUNT}/{OUT_SUBDIR}/{ep['episode_index']:06d}.zarr"
            if os.path.isdir(dest):
                skipped += 1
            else:
                pending.append(ep)
        if not pending:
            return {"data_chunk": data_chunk, "data_file": data_file, "ok": 0, "skipped": skipped, "failed": []}

        data_path = hf_dl(f"data/chunk-{data_chunk:03d}/file-{data_file:03d}.parquet")
        data_df = pd.read_parquet(data_path)
        # `dataset_from_index`/`dataset_to_index` in the episode metadata are GLOBAL
        # row indices across the whole dataset, not local offsets into this one
        # shard's parquet file. Convert to local by subtracting this group's
        # minimum -- the first episode in the group starts at local row 0.
        # (Only chunk 0/file 0 has global == local, which is why the very first
        # shard silently "worked" before this fix and every other shard produced
        # empty slices.)
        row_offset = min(ep["dataset_from_index"] for ep in episodes_meta)

        # file_key -> list of (episode_index, stream, from_ts, to_ts)
        file_needs = defaultdict(list)
        for ep in pending:
            for stream in RAW_STREAMS:
                c = ep[f"{stream}_chunk"]
                fi = ep[f"{stream}_file"]
                from_ts = ep[f"{stream}_from_ts"]
                to_ts = ep[f"{stream}_to_ts"]
                file_needs[(stream, c, fi)].append((ep["episode_index"], from_ts, to_ts))

        collected: dict[int, dict] = {ep["episode_index"]: {} for ep in pending}
        by_index = {ep["episode_index"]: ep for ep in pending}

        def maybe_finalize(ep_idx: int):
            streams = collected[ep_idx]
            if len(streams) < len(RAW_STREAMS):
                return
            nonlocal ok
            ep = by_index[ep_idx]
            try:
                t_min = min(a.shape[0] for a in streams.values())
                if t_min == 0:
                    raise RuntimeError("zero decoded frames on one or more streams")
                tactile_raw = np.stack(
                    [streams[s][:t_min] for s in RAW_STREAMS], axis=1
                )  # (T, 10, H, W)

                i0 = ep["dataset_from_index"] - row_offset
                i1 = ep["dataset_to_index"] - row_offset
                rows = data_df.iloc[i0:i1]
                if len(rows) == 0:
                    raise RuntimeError(
                        f"empty data-row slice [{i0}:{i1}] out of {len(data_df)} rows "
                        f"(row_offset={row_offset})"
                    )
                state = np.stack(rows["observation.state"].to_numpy()).astype(np.float32)[:t_min]
                action = np.stack(rows["action"].to_numpy()).astype(np.float32)[:t_min]
                tactile_force = np.stack(rows["observation.tactile_force"].to_numpy()).astype(
                    np.float32
                )[:t_min]
                timestamp = rows["timestamp"].to_numpy().astype(np.float32).reshape(-1)[:t_min]

                dest = f"{VOLUME_MOUNT}/{OUT_SUBDIR}/{ep_idx:06d}.zarr"
                tmp_dest = dest + ".tmp"
                if os.path.isdir(tmp_dest):
                    shutil.rmtree(tmp_dest)
                root = zarr.open_group(tmp_dest, mode="w")
                # chunks=<full shape> => exactly one chunk file per array. Modal
                # Volumes have a hard 500K-inode cap; zarr's default auto-chunking
                # (which chunks along every axis, not just time) blew through that
                # in well under 150 episodes before this fix.
                tactile_raw = np.ascontiguousarray(tactile_raw, dtype=np.uint8)
                root.create_array("tactile_raw", data=tactile_raw, chunks=tactile_raw.shape)
                root.create_array("tactile_force", data=tactile_force, chunks=tactile_force.shape)
                root.create_array("state", data=state, chunks=state.shape)
                root.create_array("action", data=action, chunks=action.shape)
                root.create_array("timestamp", data=timestamp, chunks=timestamp.shape)
                root.attrs["episode_index"] = ep_idx
                root.attrs["tasks"] = ep.get("tasks", [])
                root.attrs["motor_primitive"] = ep.get("motor_primitive", "")
                root.attrs["object"] = ep.get("object", "")
                root.attrs["fps"] = fps
                root.attrs["tactile_raw_channel_order"] = RAW_STREAMS
                root.attrs["source"] = f"huggingface:{REPO_ID} (robot tactile, full pull)"
                os.rename(tmp_dest, dest)
                ok += 1
            except Exception as e:  # noqa: BLE001
                failed.append({"episode_index": ep_idx, "error": f"{type(e).__name__}: {e}"})
            finally:
                collected.pop(ep_idx, None)

        import cv2

        # Process one physical video file at a time, in a stable order, to
        # bound peak disk to ~1 data parquet + 1 video file.
        for (stream, c, fi), needs in file_needs.items():
            try:
                video_path = hf_dl(f"videos/observation.images.{stream}/chunk-{c:03d}/file-{fi:03d}.mp4")
            except Exception as e:  # noqa: BLE001
                for ep_idx, _, _ in needs:
                    failed.append(
                        {"episode_index": ep_idx, "error": f"video download failed ({stream}): {e}"}
                    )
                continue

            cap = cv2.VideoCapture(video_path)
            for ep_idx, from_ts, to_ts in sorted(needs, key=lambda x: x[1]):
                start_frame = int(round(from_ts * fps))
                end_frame = int(round(to_ts * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frames = []
                for _ in range(end_frame - start_frame):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame.ndim == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frames.append(frame)
                collected[ep_idx][stream] = (
                    np.stack(frames, axis=0) if frames else np.zeros((0, 0, 0), dtype=np.uint8)
                )
                maybe_finalize(ep_idx)
            cap.release()
            if os.path.exists(video_path):
                os.remove(video_path)  # free disk immediately; container is ephemeral anyway

        # Anything left uncollected means a stream never got a chance to finalize.
        for ep_idx in list(collected.keys()):
            failed.append(
                {"episode_index": ep_idx, "error": f"missing streams: {set(RAW_STREAMS) - set(collected[ep_idx].keys())}"}
            )

        volume.commit()
        return {"data_chunk": data_chunk, "data_file": data_file, "ok": ok, "skipped": skipped, "failed": failed}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# The dispatch loop (submitting groups, tallying results, writing the
# manifest) used to live in `@app.local_entrypoint()`, which runs on THIS
# machine. `--detach` only keeps already-dispatched remote work alive if the
# local process dies -- it does NOT keep the *dispatch loop* running, so
# this box's local `modal run` process getting OOM-killed (it also runs eval
# workloads that spike memory) would silently stall the whole job partway
# through even with `--detach` set. Running the loop in this remote function
# instead means submit + starmap + tally + manifest all run on Modal's
# infrastructure, independent of this machine.
@app.function(volumes={VOLUME_MOUNT: volume}, timeout=21600)
def run_all(group_items: list, fps: float) -> dict:
    args = [(c, f, meta, fps) for (c, f), meta in group_items]
    total_ok, total_skipped, total_failed = 0, 0, []
    for i, result in enumerate(process_data_group.starmap(args, return_exceptions=True)):
        if isinstance(result, Exception):
            total_failed.append({"group": group_items[i][0], "error": str(result)})
            continue
        total_ok += result["ok"]
        total_skipped += result["skipped"]
        total_failed.extend(result["failed"])
        summary = {
            "done_groups": i + 1,
            "total_groups": len(group_items),
            "ok": total_ok,
            "skipped": total_skipped,
            "failed": len(total_failed),
        }
        logger.info(f"progress: {summary}")
        with open(f"{VOLUME_MOUNT}/{OUT_SUBDIR}/_progress.json", "w") as f:
            json.dump(summary, f)
        volume.commit()

    manifest = {"ok": total_ok, "skipped": total_skipped, "failed": total_failed}
    logger.info(f"DONE. ok={total_ok} skipped={total_skipped} failed={len(total_failed)}")
    with open(f"{VOLUME_MOUNT}/{OUT_SUBDIR}/_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()
    return manifest


@app.local_entrypoint()
def main(max_groups: int = 0):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    info = json.loads(
        open(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename="meta/info.json")).read()
    )
    fps = float(info["fps"])

    episodes_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename="meta/episodes/chunk-000/file-000.parquet"
    )
    df = pd.read_parquet(episodes_path)

    def row_to_meta(row) -> dict:
        d = {
            "episode_index": int(row["episode_index"]),
            "dataset_from_index": int(row["dataset_from_index"]),
            "dataset_to_index": int(row["dataset_to_index"]),
            "tasks": list(row["tasks"]) if row["tasks"] is not None else [],
            "motor_primitive": row.get("motor_primitive", "") or "",
            "object": row.get("object", "") or "",
        }
        for stream in RAW_STREAMS:
            d[f"{stream}_chunk"] = int(row[f"videos/observation.images.{stream}/chunk_index"])
            d[f"{stream}_file"] = int(row[f"videos/observation.images.{stream}/file_index"])
            d[f"{stream}_from_ts"] = float(row[f"videos/observation.images.{stream}/from_timestamp"])
            d[f"{stream}_to_ts"] = float(row[f"videos/observation.images.{stream}/to_timestamp"])
        return d

    groups = defaultdict(list)
    for _, row in df.iterrows():
        key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
        groups[key].append(row_to_meta(row))

    group_items = sorted(groups.items())
    if max_groups > 0:
        group_items = group_items[:max_groups]

    logger.info(f"{len(df)} episodes total, dispatching {len(group_items)} data-shard groups")

    call = run_all.spawn(group_items, fps)
    logger.info(
        f"Dispatched run_all as a remote Modal function call (id={call.object_id}). "
        f"This survives local disconnection/OOM -- check progress via "
        f"{VOLUME_MOUNT}/{OUT_SUBDIR}/_progress.json on the volume, not this process."
    )
