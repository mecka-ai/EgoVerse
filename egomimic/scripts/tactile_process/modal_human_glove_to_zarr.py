"""
Full-scale human tactile-glove -> Zarr conversion, run on Modal.

Same conversion logic as `human_glove_to_zarr.py` (the local pilot), scaled
to all 7733 episodes in `tactile_data/tactile_200h_episodes.csv` (~200
hours, ~296GB of raw glove CSVs). Output lands in the Modal volume
`tactile-zarr-data`, not local disk, since the full pull doesn't fit on
this machine.

v2: a first version (one plain `@app.function` call per episode, opening a
fresh MongoClient + boto3 client every call) only managed ~750 episodes in
43 minutes -- each call was paying a full new MongoDB TLS handshake, which
dominated runtime. This version uses a `@app.cls` container with
`@modal.enter()` so each container opens ONE MongoDB connection and ONE R2
client and reuses them across every episode it handles, and batches several
episodes per remote call to cut down Modal's own per-call dispatch
overhead too.

Requires (already created for this run):
  - Modal secret `egoverse-r2`: R2_ACCESS_KEY, R2_SECRET_KEY, R2_ENDPOINT, BUCKET
  - Modal secret `egoverse-mongodb`: MONGODB_URI
  - Modal volume `tactile-zarr-data`

Usage:
    modal run --detach modal_human_glove_to_zarr.py \\
        --csv-path /home/mecka/EgoVerse/tactile_data/tactile_200h_episodes.csv

    # smaller test run first:
    modal run modal_human_glove_to_zarr.py --limit 20
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os

import modal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MONGODB_DB = "mecka-ai"
MONGODB_EPISODES_COLLECTION = "episodes"
VOLUME_NAME = "tactile-zarr-data"
VOLUME_MOUNT = "/vol/tactile_zarr"
OUT_SUBDIR = "tactile_human"
BATCH_SIZE = 20

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy", "pymongo", "boto3", "zarr>=3.0"
)
app = modal.App("tactile-human-glove-zarr", image=image)
volume = modal.Volume.from_name(VOLUME_NAME)


def _parse_storage_key(storage_key: str, default_bucket: str) -> tuple[str, str]:
    if storage_key.startswith("r2://"):
        without_scheme = storage_key[len("r2://"):]
        bucket, _, key = without_scheme.partition("/")
        return bucket, key
    return default_bucket, storage_key


@app.cls(
    secrets=[modal.Secret.from_name("egoverse-r2"), modal.Secret.from_name("egoverse-mongodb")],
    volumes={VOLUME_MOUNT: volume},
    timeout=600,
    memory=1024,
    max_containers=100,
    retries=modal.Retries(max_retries=2, initial_delay=5.0, backoff_coefficient=2.0),
)
class GloveConverter:
    @modal.enter()
    def setup(self):
        import boto3
        from pymongo import MongoClient

        self.mongo_client = MongoClient(
            os.environ["MONGODB_URI"], serverSelectionTimeoutMS=15000, maxPoolSize=10
        )
        self.db = self.mongo_client[MONGODB_DB]
        self.bucket_default = os.environ.get("BUCKET", "data")
        self.r2 = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["R2_SECRET_KEY"],
            region_name="auto",
            config=boto3.session.Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )

    def _fetch(self, storage_key: str):
        import numpy as np

        b, key = _parse_storage_key(storage_key, self.bucket_default)
        body = self.r2.get_object(Bucket=b, Key=key)["Body"].read()
        return np.loadtxt(io.StringIO(body.decode("utf-8")), delimiter=",", skiprows=1).astype(
            np.float32
        )

    def _convert_one(self, episode_id: str) -> dict:
        import shutil

        import numpy as np
        import zarr
        from bson import ObjectId

        dest = f"{VOLUME_MOUNT}/{OUT_SUBDIR}/{episode_id}.zarr"
        if os.path.isdir(dest):
            return {"episode_id": episode_id, "status": "skipped_exists"}

        try:
            doc = self.db[MONGODB_EPISODES_COLLECTION].find_one({"_id": ObjectId(episode_id)})
            if doc is None:
                raise ValueError("episode not found in MongoDB")

            left_key = doc.get("tactileLeftKey")
            right_key = doc.get("tactileRightKey")
            if not left_key or not right_key:
                raise ValueError("missing tactileLeftKey/tactileRightKey")

            left = self._fetch(left_key)
            right = self._fetch(right_key)

            tmp_dest = dest + ".tmp"
            if os.path.isdir(tmp_dest):
                shutil.rmtree(tmp_dest)
            root = zarr.open_group(tmp_dest, mode="w")
            # chunks=<full shape> => one chunk file per array (Modal Volumes have a
            # hard 500K-inode cap; see modal_trex_robot_to_zarr.py for where this bit).
            tleft = np.ascontiguousarray(left[:, 2:], dtype=np.float32)
            tright = np.ascontiguousarray(right[:, 2:], dtype=np.float32)
            root.create_array("tactile_left", data=tleft, chunks=tleft.shape)
            root.create_array(
                "tactile_left_frame_index", data=left[:, 0].astype(np.int64), chunks=(left.shape[0],)
            )
            root.create_array(
                "tactile_left_time_ns", data=left[:, 1].astype(np.int64), chunks=(left.shape[0],)
            )
            root.create_array("tactile_right", data=tright, chunks=tright.shape)
            root.create_array(
                "tactile_right_frame_index", data=right[:, 0].astype(np.int64), chunks=(right.shape[0],)
            )
            root.create_array(
                "tactile_right_time_ns", data=right[:, 1].astype(np.int64), chunks=(right.shape[0],)
            )
            root.attrs["episode_id"] = episode_id
            root.attrs["task"] = doc.get("task_desc", "")
            root.attrs["scene"] = str(doc.get("scene_id", ""))
            root.attrs["environment"] = str(doc.get("environment_id", ""))
            root.attrs["duration"] = doc.get("duration")
            root.attrs["num_taxels_per_hand"] = 460
            root.attrs["source"] = "mecka-ai.episodes (human tactile glove, full pull)"
            root.attrs["note"] = (
                "tactile_{left,right} are raw (T, 460) taxel readings, NOT reshaped "
                "to a spatial grid -- physical taxel layout unknown. left/right are "
                "sampled independently and are not frame-aligned."
            )
            os.rename(tmp_dest, dest)
            return {"episode_id": episode_id, "status": "ok"}

        except Exception as e:  # noqa: BLE001
            return {"episode_id": episode_id, "status": "error", "error": f"{type(e).__name__}: {e}"}

    @modal.method()
    def convert_batch(self, episode_ids: list[str]) -> list[dict]:
        results = [self._convert_one(eid) for eid in episode_ids]
        volume.commit()
        return results


def _batched(seq: list, size: int) -> list[list]:
    return [seq[i : i + size] for i in range(0, len(seq), size)]


# The dispatch loop (submitting batches, tallying results, writing the
# manifest) used to live in `@app.local_entrypoint()`, which runs on THIS
# machine. `--detach` only keeps already-dispatched remote batches alive if
# the local process dies -- it does NOT keep the *dispatch loop* itself
# running, so when the local `modal run` process got OOM-killed twice (this
# box also runs eval workloads that spike memory), the whole job silently
# stalled partway through even though `--detach` was set. Moving the loop
# into this remote function means the entire pipeline -- submit, map, tally,
# write manifest -- runs on Modal's infrastructure and survives regardless
# of what happens on the local machine.
@app.function(volumes={VOLUME_MOUNT: volume}, timeout=7200)
def run_all(episode_ids: list[str]) -> dict:
    batches = _batched(episode_ids, BATCH_SIZE)
    logger.info(f"{len(episode_ids)} episodes queued in {len(batches)} batches of up to {BATCH_SIZE}")

    converter = GloveConverter()
    ok, skipped, failed = 0, 0, []
    done = 0
    for batch_result in converter.convert_batch.map(batches, return_exceptions=True):
        if isinstance(batch_result, Exception):
            failed.append({"episode_id": "unknown_batch", "error": str(batch_result)})
            continue
        for result in batch_result:
            status = result.get("status")
            if status == "ok":
                ok += 1
            elif status == "skipped_exists":
                skipped += 1
            else:
                failed.append(result)
        done += 1
        summary = {"done_batches": done, "total_batches": len(batches), "ok": ok, "skipped": skipped, "failed": len(failed)}
        if done % 10 == 0 or done == len(batches):
            logger.info(f"progress: {summary}")
            with open(f"{VOLUME_MOUNT}/{OUT_SUBDIR}/_progress.json", "w") as f:
                json.dump(summary, f)
            volume.commit()

    logger.info(f"DONE. ok={ok} skipped={skipped} failed={len(failed)} total={len(episode_ids)}")
    manifest = {"ok": ok, "skipped": skipped, "failed": failed, "total": len(episode_ids)}
    with open(f"{VOLUME_MOUNT}/{OUT_SUBDIR}/_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    csv_path: str = "/home/mecka/EgoVerse/tactile_data/tactile_200h_episodes.csv",
    limit: int = 0,
):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    episode_ids = [r["episode_id"] for r in rows]
    if limit > 0:
        episode_ids = episode_ids[:limit]

    call = run_all.spawn(episode_ids)
    logger.info(
        f"Dispatched run_all as a remote Modal function call (id={call.object_id}). "
        f"This survives local disconnection/OOM -- check progress via "
        f"{VOLUME_MOUNT}/{OUT_SUBDIR}/_progress.json on the volume, not this process."
    )
