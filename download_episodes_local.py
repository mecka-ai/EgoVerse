"""
Local batch download + conversion of Mecka episodes to Zarr.

For each episode hash this:
  1. resolves the MongoDB doc + presigned R2 asset URLs,
  2. fetches VLM annotations from `episode_vlm_segments`,
  3. runs the base `MeckaDatasetConverter` (egomimic/.../mecka_to_zarr.py)
     to produce `<hash>.zarr` (+ preview `<hash>.mp4`),
  4. copies the result into `--output-dir`.

It reuses the proven resolve/convert path (`_prepare_episode`) from
`modal_mecka_to_zarr.py` but runs entirely locally — no Modal account, no
Modal Volume. Credentials come from the environment (MONGODB_URI, R2_*).

Usage:
    set -a; . /root/.egoverse_env; set +a
    python download_episodes_local.py \
        --ids-file /workspace/episode_hashes.txt \
        --output-dir /workspace/zarr_episodes
"""

import argparse
import importlib.util
import json
import logging
import os
import shutil
import tempfile
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_episodes_local")


def _quiet_logging():
    """Trim batch-log noise so it shows mainly our per-episode results.

    Keeps OUR logger at INFO (progress / done / failed / runner markers) but
    drops the converter's per-file INFO spam ("Downloading data files",
    "Extracted N frames", ...), the per-video torchvision->ffmpeg fallback
    WARNING (the ffmpeg path always succeeds), and zarr's
    UnstableSpecificationWarning. Genuine converter errors still surface.

    Called in the main process and in every worker (converter runs in workers).
    """
    logging.getLogger().setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)

    class _DropNoise(logging.Filter):
        def filter(self, rec):  # noqa: A003
            return "torchvision failed" not in rec.getMessage()

    for h in logging.getLogger().handlers:
        h.addFilter(_DropNoise())
    warnings.filterwarnings("ignore")


_MODAL_SCRIPT = Path(__file__).resolve().parent / "modal_mecka_to_zarr.py"

# Per-worker state for parallel mode: loading the modal helpers and opening a
# MongoDB connection is expensive, so do it once per process (in the pool
# initializer) rather than once per episode. pymongo clients are NOT fork-safe,
# so each worker must create its own — never share a parent's db handle.
_MM = None
_DB = None


class _ModalStub:
    """No-op stand-in for the ``modal`` package.

    ``modal_mecka_to_zarr.py`` builds a Modal App / Image / Volume and decorates
    its remote entrypoints at import time, but the resolve+convert helpers we use
    here (``_prepare_episode``, ``_get_mongo_db``, ``_is_freeform``) are plain
    functions that never touch Modal. This stub lets that module import locally
    WITHOUT the real ``modal`` dependency: every attribute access returns the
    stub, builder calls keep chaining, and a decorator call (a single callable
    arg) returns the wrapped function unchanged.
    """

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        if len(a) == 1 and not k and callable(a[0]):
            return a[0]  # @app.function(...) / @app.local_entrypoint() -> identity
        return self  # Image.debian_slim(...).pip_install(...) etc. -> keep chaining

    def __getattr__(self, name):
        return self


def _load_modal_helpers():
    """Import the resolve/convert helpers from modal_mecka_to_zarr.py.

    Local-first: if the real ``modal`` package is installed we use it, otherwise
    we register a no-op stub so the module imports without that dependency. The
    helpers we call run entirely locally (MongoDB + R2 + the egomimic converter).
    """
    import sys

    try:
        import modal  # noqa: F401
    except ImportError:
        sys.modules["modal"] = _ModalStub()

    spec = importlib.util.spec_from_file_location("mecka_modal_helpers", _MODAL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_env():
    missing = []
    if not os.environ.get("MONGODB_URI"):
        missing.append("MONGODB_URI")
    if not (os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("R2_ACCESS_KEY")):
        missing.append("R2_ACCESS_KEY_ID")
    if not (os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("R2_SECRET_KEY")):
        missing.append("R2_SECRET_ACCESS_KEY")
    if not (os.environ.get("R2_BUCKET") or os.environ.get("BUCKET")):
        missing.append("R2_BUCKET (or BUCKET)")
    if not (
        os.environ.get("AWS_ENDPOINT_URL_S3")
        or os.environ.get("R2_ENDPOINT_URL")
        or os.environ.get("R2_ENDPOINT")
        or os.environ.get("R2_ACCOUNT_ID")
    ):
        missing.append("AWS_ENDPOINT_URL_S3 (or R2_ACCOUNT_ID)")
    if missing:
        raise SystemExit(
            "Missing required credentials in environment: "
            + ", ".join(missing)
            + "\nLoad them first, e.g.:  set -a; . /root/.egoverse_env; set +a"
        )


def _read_ids(path: str) -> list[str]:
    """Read episode hashes from a newline-delimited .txt or a JSON list file.

    De-duplicates while preserving order.
    """
    text = Path(path).read_text()
    ids: list[str] = []
    if path.endswith(".json"):
        data = json.loads(text)
        if not isinstance(data, list):
            raise SystemExit(
                f"{path}: expected a JSON list of hashes, got {type(data)}"
            )
        ids = [str(h).strip() for h in data if str(h).strip()]
    else:
        for line in text.splitlines():
            h = line.strip().strip(",").strip('"')
            if h and not h.startswith("#"):
                ids.append(h)
    seen, deduped = set(), []
    for h in ids:
        if h not in seen:
            seen.add(h)
            deduped.append(h)
    return deduped


def _convert_one(mm, db, h, out_dir, force_task_type, overwrite):
    """Resolve + convert a single episode hash to ``<hash>.zarr`` in out_dir.

    Returns (status, detail) where status is one of "ok"/"skipped"/"failed".
    Mirrors the original sequential loop body so both paths behave identically.
    """
    dest_zarr = out_dir / f"{h}.zarr"
    if dest_zarr.exists() and not overwrite:
        return "skipped", None

    if force_task_type:
        task_type = force_task_type
    else:
        try:
            task_type = "freeform" if mm._is_freeform(db, h) else "flagship"
        except Exception as e:
            return "failed", f"classify: {e}"

    tmp_dir = tempfile.mkdtemp(prefix=f"zarr_local_{h}_")
    try:
        result = mm._prepare_episode(h, task_type, tmp_dir)
        if dest_zarr.exists():
            shutil.rmtree(dest_zarr)
        shutil.copytree(result["zarr_dir"], dest_zarr)
        if result.get("mp4_path") and os.path.exists(result["mp4_path"]):
            shutil.copy2(result["mp4_path"], out_dir / f"{h}.mp4")
        return "ok", result.get("num_annotations")
    except Exception as e:
        logger.debug(traceback.format_exc())
        return "failed", str(e)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _worker_init():
    """Pool initializer: load modal helpers + open a per-process Mongo handle."""
    global _MM, _DB
    _quiet_logging()
    if not os.environ.get("R2_BUCKET") and os.environ.get("BUCKET"):
        os.environ["R2_BUCKET"] = os.environ["BUCKET"]
    _MM = _load_modal_helpers()
    _DB = _MM._get_mongo_db()


def _process_one(task):
    """Pool task wrapper: (i, total, h, out_dir, force_task_type, overwrite)."""
    i, total, h, out_dir, force_task_type, overwrite = task
    status, detail = _convert_one(
        _MM, _DB, h, Path(out_dir), force_task_type, overwrite
    )
    return h, status, detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Mecka -> Zarr batch download")
    parser.add_argument("--ids-file", default="/workspace/episode_hashes.txt")
    parser.add_argument("--output-dir", default="/workspace/zarr_episodes")
    parser.add_argument(
        "--force-task-type",
        choices=["flagship", "freeform"],
        default="",
        help="Skip MongoDB classification and force this task type.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-convert episodes even if <hash>.zarr already exists in output-dir.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel conversion processes (1 = original sequential).",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=0,
        help="Recycle each worker process after this many episodes to release "
        "leaked native (PyAV/OpenCV) memory (0 = never). WARNING: leave at 0 — "
        "with the fork start method, recycling all workers at once deadlocks "
        "ProcessPoolExecutor. Control peak RSS via --workers instead.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of nodes splitting this ids-file (default 1 = no "
        "sharding). Every node must pass the SAME --ids-file and --num-shards.",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="This node's shard index in [0, num-shards). Each node processes a "
        "disjoint round-robin slice (ids[shard::num_shards]), so no two nodes "
        "ever touch the same episode.",
    )
    args = parser.parse_args()
    _quiet_logging()

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"--shard must be in [0, {args.num_shards}); got {args.shard}")

    _check_env()
    # _parse_storage_key reads R2_BUCKET; mirror BUCKET into it for convenience.
    if not os.environ.get("R2_BUCKET") and os.environ.get("BUCKET"):
        os.environ["R2_BUCKET"] = os.environ["BUCKET"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = _read_ids(args.ids_file)
    total_ids = len(ids)
    if args.num_shards > 1:
        # Deterministic round-robin split: _read_ids preserves file order and
        # de-dups identically on every node, so ids[k::N] yields N disjoint,
        # exhaustive slices. Each node must use the same ids-file + num-shards.
        ids = ids[args.shard :: args.num_shards]
        logger.info(
            "Loaded %d hashes from %s -> shard %d/%d = %d episodes for this node",
            total_ids,
            args.ids_file,
            args.shard,
            args.num_shards,
            len(ids),
        )
    else:
        logger.info("Loaded %d episode hashes from %s", total_ids, args.ids_file)
    logger.info("Output dir: %s  |  workers: %d", out_dir, args.workers)

    ok, skipped, failed = [], [], []

    def _record(i, h, status, detail):
        if status == "ok":
            logger.info("[%d/%d] %s — done (%s annotations)", i, len(ids), h, detail)
            ok.append(h)
        elif status == "skipped":
            # Demoted to DEBUG: on resume this fires thousands of times. The
            # end-of-run summary reports the skipped count.
            logger.debug("[%d/%d] %s — already present, skipping", i, len(ids), h)
            skipped.append(h)
        else:
            logger.error("[%d/%d] %s — FAILED: %s", i, len(ids), h, detail)
            failed.append((h, detail))

    if args.workers <= 1:
        # Original sequential path.
        mm = _load_modal_helpers()
        db = mm._get_mongo_db()
        for i, h in enumerate(ids, 1):
            logger.info("[%d/%d] %s — converting", i, len(ids), h)
            status, detail = _convert_one(
                mm, db, h, out_dir, args.force_task_type, args.overwrite
            )
            _record(i, h, status, detail)
    else:
        # Parallel path: one process per worker, each with its own modal helpers
        # + Mongo connection (created in the pool initializer). Episodes are
        # independent (each writes its own <hash>.zarr), so no locking needed.
        tasks = [
            (i, len(ids), h, str(out_dir), args.force_task_type, args.overwrite)
            for i, h in enumerate(ids, 1)
        ]
        done = 0
        pool_kwargs = dict(max_workers=args.workers, initializer=_worker_init)
        if args.max_tasks_per_child and args.max_tasks_per_child > 0:
            # Python 3.11+: replace each worker after N tasks so leaked native
            # memory (video decoders) is reclaimed and peak RSS stays bounded
            # under the container's cgroup memory limit.
            pool_kwargs["max_tasks_per_child"] = args.max_tasks_per_child
        with ProcessPoolExecutor(**pool_kwargs) as ex:
            futures = {ex.submit(_process_one, t): t for t in tasks}
            for fut in as_completed(futures):
                i = futures[fut][0]
                h, status, detail = fut.result()
                done += 1
                _record(i, h, status, detail)
                if done % 100 == 0 or done == len(ids):
                    logger.info(
                        "progress: %d/%d processed (ok=%d skipped=%d failed=%d)",
                        done,
                        len(ids),
                        len(ok),
                        len(skipped),
                        len(failed),
                    )

    logger.info("=" * 60)
    logger.info(
        "DONE  ok=%d  skipped=%d  failed=%d", len(ok), len(skipped), len(failed)
    )
    if failed:
        logger.info("Failures:")
        for h, err in failed:
            logger.info("  %s: %s", h, err)
        # Write a retry list of just the failed hashes.
        retry = out_dir / "failed_hashes.txt"
        retry.write_text("\n".join(h for h, _ in failed) + "\n")
        logger.info("Wrote failed hashes -> %s", retry)


if __name__ == "__main__":
    main()
