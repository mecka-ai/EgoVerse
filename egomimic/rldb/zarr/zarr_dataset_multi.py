"""
ZarrDataset implementation for EgoVerse.

Mirrors the LeRobotDataset API while reading data from Zarr arrays
instead of parquet/HF datasets.

Directory structure (per-episode metadata):
    dataset_root/
    └── episode_{ep_idx}.zarr/
        ├── observations.images.{cam}  (JPEG compressed)
        ├── observations.state
        ├── actions_joints
        └── ...

Each episode is self-contained with its own metadata, enabling:
- Independent episode uploads to S3
- Parallel processing without global coordination
- Easy episode-level data management
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import numpy as np
import pandas as pd
import simplejpeg
import torch
import zarr

from egomimic.rldb.embodiment.embodiment import get_embodiment_id
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.zarr import store_handle_cache as hc
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.aws.aws_sql import (
    create_default_engine,
    episode_table_to_df,
)
from egomimic.utils.pose_utils import bimanual_cartesian_layout

if TYPE_CHECKING:
    # Annotation-only import — avoids a runtime circular import with
    # zarr_dataset_action_expert (which itself imports MultiDataset from here).
    from egomimic.rldb.zarr.zarr_dataset_action_expert import (
        ZarrActionExpertDataset,
    )

logger = logging.getLogger(__name__)

SEED = 42

PAUSE_DETECT_KEYS: tuple[str, str] = ("left.obs_ee_pose", "right.obs_ee_pose")
PAUSE_DETECT_KEYPOINT_KEYS: tuple[str, str] = (
    "left.obs_keypoints",
    "right.obs_keypoints",
)


def _keypoint_max_delta(kp: np.ndarray) -> np.ndarray:
    """Per-frame max landmark delta for a flattened (T, K*3) keypoint array.

    Reshapes to (T, K, 3), takes per-landmark L2 of the time-diff, and
    returns max over landmarks → (T-1,). This catches finger flexion even
    when the wrist (centroid) is stationary.
    """
    T = kp.shape[0]
    if T < 2 or kp.ndim != 2 or kp.shape[1] % 3 != 0 or kp.shape[1] == 0:
        return np.zeros(max(T - 1, 0))
    n_landmarks = kp.shape[1] // 3
    diff = np.diff(kp.reshape(T, n_landmarks, 3), axis=0)
    per_landmark_norm = np.linalg.norm(diff, axis=-1)
    return per_landmark_norm.max(axis=-1)


def _build_pause_keep_mask(
    *,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    epsilon: float,
    left_keypoints: np.ndarray | None = None,
    right_keypoints: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean keep-mask over frames.

    A frame at index t is a "pause step" iff EVERY available motion signal
    moved less than ``epsilon`` between t-1 and t:
      - left/right ee_pose L2 delta < eps (wrist movement), and
      - if keypoints are provided, max-over-landmarks keypoint delta < eps
        on the corresponding hand (catches finger flexion when the wrist
        is stationary).

    Missing keypoint arrays are simply skipped — the test reduces to
    ee_pose-only. The first frame of each pause run is kept (transition
    into pause); subsequent in-pause frames are dropped until motion
    resumes.
    """
    T = len(left_pose)
    if T < 2:
        return np.ones(T, dtype=bool)

    left_d = np.linalg.norm(np.diff(left_pose, axis=0), axis=-1)
    right_d = np.linalg.norm(np.diff(right_pose, axis=0), axis=-1)
    is_paused_step = (left_d < epsilon) & (right_d < epsilon)

    if left_keypoints is not None and len(left_keypoints) == T:
        is_paused_step = is_paused_step & (
            _keypoint_max_delta(np.asarray(left_keypoints)) < epsilon
        )
    if right_keypoints is not None and len(right_keypoints) == T:
        is_paused_step = is_paused_step & (
            _keypoint_max_delta(np.asarray(right_keypoints)) < epsilon
        )

    keep = np.ones(T, dtype=bool)
    in_pause = False
    for t in range(1, T):
        if is_paused_step[t - 1]:
            if in_pause:
                keep[t] = False
            else:
                in_pause = True
        else:
            in_pause = False
    return keep


PAUSE_PRECOMPUTE_CACHE_ENV = "EGOMIMIC_PAUSE_PRECOMPUTE_CACHE"


def split_dataset_names(dataset_names, valid_ratio=0.2, seed=SEED):
    """
    Split a list of dataset names into train/valid sets.
    Args:
        dataset_names (Iterable[str])
        valid_ratio (float): fraction of datasets to put in valid.
        seed (int): for deterministic shuffling.


    Returns:
        train_set (set[str]), valid_set (set[str])
    """
    names = sorted(dataset_names)
    if not names:
        return set(), set()

    rng = random.Random(seed)
    rng.shuffle(names)

    if not (0.0 <= valid_ratio <= 1.0):
        raise ValueError(f"valid_ratio must be in [0,1], got {valid_ratio}")

    n_valid = int(len(names) * valid_ratio)
    if valid_ratio > 0.0:
        n_valid = max(1, n_valid)

    valid = set(names[:n_valid])
    train = set(names[n_valid:])
    return train, valid


def _ensure_dataset_filter(filters: DatasetFilter | None) -> DatasetFilter:
    if filters is None:
        return DatasetFilter()
    if isinstance(filters, DatasetFilter):
        return filters
    raise TypeError(
        "filters must be a DatasetFilter or None in the zarr resolver path. "
        "Plain dict filters are no longer supported."
    )


def _is_missing_filter_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""

    try:
        missing = pd.isna(value)
    except Exception:
        return False

    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _first_present(*values: object) -> object | None:
    for value in values:
        if not _is_missing_filter_value(value):
            return value
    return None


def _normalize_filter_row(
    row: Mapping[str, Any],
    *,
    episode_hash: str | None = None,
) -> dict[str, Any]:
    normalized = dict(row)
    normalized["episode_hash"] = (
        episode_hash if episode_hash is not None else normalized.get("episode_hash")
    )

    if _is_missing_filter_value(normalized.get("is_deleted")):
        normalized["is_deleted"] = False

    robot_name = _first_present(
        normalized.get("robot_name"),
        normalized.get("robot_type"),
        normalized.get("embodiment"),
    )
    if robot_name is not None:
        normalized["robot_name"] = robot_name

    return normalized


def get_fallback_idx(
    idx: int,
    candidates: Iterable[int],
    _attempts: int | None,
    max_attempts: int,
    exhausted_error: str,
) -> tuple[int, int]:
    attempts = (_attempts or 0) + 1
    valid_candidates = [
        candidate_idx for candidate_idx in candidates if candidate_idx != idx
    ]
    if attempts >= max_attempts or not valid_candidates:
        raise RuntimeError(exhausted_error)
    return random.choice(valid_candidates), attempts


class EpisodeResolver:
    """
    Base class for episode resolution utilities.
    Provides shared static/class helpers; subclasses implement resolve().
    """

    _dataset_class = None  # set to ZarrDataset after that class is defined

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
        norm_stats: dict | None = None,
        pause_removal_epsilon: float | None = None,
    ):
        self.folder_path = Path(folder_path)
        self.key_map = key_map
        self.transform_list = transform_list
        self.norm_stats = norm_stats
        self.pause_removal_epsilon = pause_removal_epsilon

    def _load_zarr_datasets(self, search_path: Path, valid_folder_names: set[str]):
        """
        Loads multiple Zarr datasets from the specified folder path, filtering only those whose hashes
        are present in the valid_folder_names set.

        Args:
            search_path (Path): The root directory to search for Zarr datasets.
            valid_folder_names (set[str]): A set of valid folder names (episode hashes without ".zarr") to filter datasets.
        Returns:
            dict[str, ZarrDataset]: a dictionary mapping string keys to constructed zarr datasets from valid filters.
        """
        import time

        dataset_class = self._dataset_class or ZarrDataset

        all_paths = sorted(search_path.iterdir())
        datasets: dict[str, ZarrDataset] = {}
        skipped: list[str] = []
        total = len(all_paths)
        log_every = max(1, total // 20)
        start_time = time.monotonic()
        logger.info(f"Loading {total} Zarr datasets from {search_path}...")
        for i, p in enumerate(all_paths, start=1):
            if not p.is_dir():
                logger.info(f"{p} is not a valid directory")
                skipped.append(p.name)
                continue
            name = p.name
            if name.endswith(".zarr"):
                name = name[: -len(".zarr")]
            if name not in valid_folder_names:
                skipped.append(p.name)
                continue
            try:
                ds_obj = dataset_class(
                    p,
                    key_map=self.key_map,
                    transform_list=self.transform_list,
                    norm_stats=self.norm_stats,
                    pause_removal_epsilon=self.pause_removal_epsilon,
                )
                datasets[name] = ds_obj
            except Exception as e:
                logger.error(f"Failed to load dataset at {p}: {e}")
                skipped.append(p.name)

            if i % log_every == 0 or i == total:
                elapsed = time.monotonic() - start_time
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (total - i) / rate if rate > 0 else float("inf")
                logger.info(
                    f"Loading Zarr datasets: {i}/{total} ({100 * i / total:.1f}%) "
                    f"| loaded={len(datasets)} skipped={len(skipped)} "
                    f"| {rate:.1f} ep/s | ETA {eta:.0f}s"
                )

        logger.info(
            f"Loaded {len(datasets)} datasets, skipped {len(skipped)} "
            f"(total scanned {total}) in {time.monotonic() - start_time:.1f}s"
        )
        return datasets

    def _run_pause_precompute(self, datasets: dict) -> None:
        """Run per-episode pause precompute; no-op if epsilon is None.

        Two paths:
          - If $EGOMIMIC_PAUSE_PRECOMPUTE_CACHE points at a JSON file produced
            by trainModal.run_hydra_train (which fans out to the Modal worker
            before launching this subprocess), load keep_indices from there.
            Episodes missing from the cache fall through to the in-process
            thread pool.
          - Otherwise (local dev, or no pre-train fan-out happened), do the
            whole thing in-process.
        """
        if self.pause_removal_epsilon is None or not datasets:
            return

        cache_path = os.environ.get(PAUSE_PRECOMPUTE_CACHE_ENV)
        if cache_path and Path(cache_path).is_file():
            remaining = self._apply_pause_precompute_cache(cache_path, datasets)
            if remaining:
                self._inprocess_pause_precompute(remaining)
            return

        self._inprocess_pause_precompute(datasets)

    def _apply_pause_precompute_cache(self, cache_path: str, datasets: dict) -> dict:
        """Populate keep_indices from the cache JSON; return cache-miss subset.

        Cache JSON shape: ``{episode_hash: {"raw_total": int, "keep_indices": [int]}}``.
        Entries with raw_total == 0 (worker had an error reading the episode)
        are treated as misses so the in-process fallback can retry.
        """
        import json
        import time

        t0 = time.monotonic()
        with open(cache_path) as f:
            cache = json.load(f)

        n_hit = 0
        n_total = 0
        n_kept = 0
        miss: dict = {}
        for episode_hash, ds in datasets.items():
            entry = cache.get(episode_hash)
            if entry is None or int(entry.get("raw_total", 0)) == 0:
                miss[episode_hash] = ds
                continue
            ds._raw_total_frames = int(entry["raw_total"])
            ds.keep_indices = np.asarray(entry["keep_indices"], dtype=np.int64)
            n_hit += 1
            n_total += int(entry["raw_total"])
            n_kept += len(entry["keep_indices"])

        elapsed = time.monotonic() - t0
        pct = (100.0 * n_kept / n_total) if n_total else 100.0
        logger.info(
            "Pause precompute via cache (%s): %d hits, %d misses | "
            "kept %d/%d frames (%.1f%%) in %.2fs",
            cache_path,
            n_hit,
            len(miss),
            n_kept,
            n_total,
            pct,
            elapsed,
        )
        return miss

    def _inprocess_pause_precompute(self, datasets: dict) -> None:
        import time
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(32, max(2, len(datasets)))
        t0 = time.monotonic()
        results: list[tuple[str, int, int, str | None]] = []

        def _one(item):
            name, ds = item
            try:
                n_raw, n_kept = ds.precompute_pause_filter()
                return (name, n_raw, n_kept, None)
            except Exception as e:
                return (name, 0, 0, f"{type(e).__name__}: {e}")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for r in ex.map(_one, datasets.items()):
                results.append(r)

        n_total = sum(r[1] for r in results)
        n_kept = sum(r[2] for r in results)
        n_errs = sum(1 for r in results if r[3] is not None)
        elapsed = time.monotonic() - t0
        pct = (100.0 * n_kept / n_total) if n_total else 100.0
        logger.info(
            "Pause precompute via in-process thread pool (epsilon=%s): kept %d/%d frames "
            "(%.1f%%) across %d episodes in %.1fs (errors=%d)",
            self.pause_removal_epsilon,
            n_kept,
            n_total,
            pct,
            len(datasets),
            elapsed,
            n_errs,
        )
        if n_errs:
            sample = [r for r in results if r[3] is not None][:5]
            for name, _, _, err in sample:
                logger.warning("Pause precompute failed for %s: %s", name, err)

    @classmethod
    def _episode_already_present(cls, local_dir: Path, episode_hash: str) -> bool:
        direct = local_dir / episode_hash
        if direct.is_dir():
            return True
        return False


class S3EpisodeResolver(EpisodeResolver):
    """
    Resolves episodes via SQL table and optionally syncs from S3.
    """

    def __init__(
        self,
        folder_path: Path,
        bucket_name: str = "rldb",
        main_prefix: str = "processed_v3",
        key_map: dict | None = None,
        transform_list: list | None = None,
        debug: int | bool | None = None,
        norm_stats: dict | None = None,
        pause_removal_epsilon: float | None = None,
    ):
        self.bucket_name = bucket_name
        self.main_prefix = main_prefix
        self.debug = debug
        super().__init__(
            folder_path,
            key_map=key_map,
            transform_list=transform_list,
            norm_stats=norm_stats,
            pause_removal_epsilon=pause_removal_epsilon,
        )

    def resolve(
        self,
        filters: DatasetFilter | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters.
        Syncs S3 paths to local_root before indexing.
        """
        filters = _ensure_dataset_filter(filters)

        if self.folder_path.is_dir():
            logger.info(f"Using existing directory: {self.folder_path}")
        if not self.folder_path.is_dir():
            self.folder_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Filters: {filters}")

        filtered_paths = self.sync_from_filters(
            bucket_name=self.bucket_name,
            filters=filters,
            local_dir=self.folder_path,
            debug=self.debug,
        )

        valid_hashes = {hashes for _, hashes in filtered_paths}
        if not valid_hashes:
            raise ValueError(
                "No valid collection names from _get_filtered_paths: "
                "filters matched no episodes in the SQL table."
            )

        datasets = self._load_zarr_datasets(
            search_path=self.folder_path,
            valid_folder_names=valid_hashes,
        )
        self._run_pause_precompute(datasets)

        return datasets

    @staticmethod
    def _get_filtered_paths(
        filters: DatasetFilter | None = None, debug: int | bool | None = None
    ) -> list[tuple[str, str]]:
        """
        Filters episodes from the SQL episode table according to the criteria specified in `filters`
        and returns a list of (zarr_processed_path, episode_hash) tuples for episodes that match and
        have a non-null zarr_processed_path.

        Args:
            filters (DatasetFilter | None): Filter object applied row-by-row to the
                episode table.

        Returns:
            list[tuple[str, str]]: List of tuples, each containing (zarr_processed_path, episode_hash)
                                   for episodes passing the filter criteria.
        """
        filters = _ensure_dataset_filter(filters)
        engine = create_default_engine()
        df = episode_table_to_df(engine)
        if df.empty:
            logger.info("Episode table is empty.")
            return []

        df = filters.filter_df(df)
        mask = df.apply(
            lambda row: filters.matches(_normalize_filter_row(row.to_dict())),
            axis=1,
        )
        output = df.loc[mask, ["zarr_processed_path", "episode_hash"]]
        n_matched_sql = len(output)

        output = output[
            output["zarr_processed_path"].fillna("").astype(str).str.strip() != ""
        ]
        n_skipped_null = n_matched_sql - len(output)
        if n_skipped_null:
            logger.info(
                "Skipped %d episodes with null/empty zarr_processed_path.",
                n_skipped_null,
            )

        if debug is not None and debug is not False:
            k = min(10 if debug is True else int(debug), len(output))
            if k < len(output):
                logger.info("Debug mode: limiting to %d datasets.", k)
            output = output.iloc[:k]

        paths = list(output.itertuples(index=False, name=None))
        logger.info(f"Paths: {paths}")
        return paths

    @classmethod
    def _sync_s3_to_local(
        cls,
        bucket_name: str,
        s3_paths: list[tuple[str, str]],
        local_dir: Path,
        numworkers: int = 10,
    ):
        if not s3_paths:
            return

        # 0) Skip episodes already present locally
        to_sync = []
        already = []
        for processed_path, episode_hash in s3_paths:
            if cls._episode_already_present(local_dir, episode_hash):
                already.append(episode_hash)
            else:
                to_sync.append((processed_path, episode_hash))

        if already:
            logger.info("Skipping %d episodes already present locally.", len(already))

        if not to_sync:
            logger.info("Nothing to sync from S3 (all episodes already present).")
            return

        # 1) Build s5cmd batch script (one line per episode)
        local_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="_s5cmd_sync_",
            suffix=".txt",
            delete=False,
        ) as tmp_file:
            batch_path = Path(tmp_file.name)

        lines = []
        for processed_path, episode_hash in to_sync:
            # processed_path like: s3://rldb/processed_v2/eva/<hash>/
            if processed_path.startswith("s3://"):
                src_prefix = processed_path.rstrip("/") + "/*"
            else:
                src_prefix = (
                    f"s3://{bucket_name}/{processed_path.lstrip('/').rstrip('/')}"
                    + "/*"
                )

            # Destination is the root local_dir; s5cmd will preserve <hash>/... under it
            dst = local_dir / episode_hash
            lines.append(f'sync "{src_prefix}" "{str(dst)}/"')

        try:
            batch_path.write_text("\n".join(lines) + "\n")

            load_env()
            rl2_endpoint_url = os.environ.get("R2_ENDPOINT_URL")
            access_key_id = os.environ.get("R2_ACCESS_KEY_ID")
            secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY")
            if not all([rl2_endpoint_url, access_key_id, secret_access_key]):
                raise ValueError(
                    "R2 credentials missing. Ensure ~/.egoverse_env has "
                    "R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY."
                )
            s5cmd_env = os.environ.copy()
            s5cmd_env["AWS_ACCESS_KEY_ID"] = access_key_id
            s5cmd_env["AWS_SECRET_ACCESS_KEY"] = secret_access_key
            s5cmd_env["AWS_DEFAULT_REGION"] = "auto"
            s5cmd_env["AWS_REGION"] = "auto"
            cmd = [
                "s5cmd",
                "--endpoint-url",
                rl2_endpoint_url,
                "--numworkers",
                str(numworkers),
                "run",
                str(batch_path),
            ]
            logger.info("Running s5cmd batch (%d lines): %s", len(lines), " ".join(cmd))
            subprocess.run(cmd, check=True, env=s5cmd_env)

        finally:
            try:
                batch_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Failed to delete batch file %s: %s", batch_path, e)

    @classmethod
    def sync_from_filters(
        cls,
        *,
        bucket_name: str,
        filters: DatasetFilter | None = None,
        local_dir: Path,
        numworkers: int = 10,
        debug: int | bool | None = None,
    ):
        """
        Public API:
        - resolves episodes from DB using filters
        - runs a single aws s3 sync with includes
        - downloads into local_dir

        Args:
            numworkers: Number of parallel workers for s5cmd.

        Returns:
            List[(processed_path, episode_hash)]
        """
        filters = _ensure_dataset_filter(filters)

        # 1) Resolve episodes from DB
        filtered_paths = cls._get_filtered_paths(filters, debug=debug)
        if not filtered_paths:
            logger.warning("No episodes matched filters.")
            return []

        # 2) Logging
        logger.info(
            f"Syncing S3 datasets with filters {filters} to local directory {local_dir}..."
        )

        # 3) Sync
        cls._sync_s3_to_local(
            bucket_name=bucket_name,
            s3_paths=filtered_paths,
            local_dir=local_dir,
            numworkers=numworkers,
        )

        return filtered_paths


class LocalEpisodeResolver(EpisodeResolver):
    """
    Resolves episodes from local Zarr stores, filtering via local metadata.
    No Modal-specific logic — for local or non-Modal use only.
    """

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
        norm_stats: dict | None = None,
        debug: int | bool | None = None,
        allowed_episode_ids: list[str] | None = None,
        pause_removal_epsilon: float | None = None,
    ):
        super().__init__(
            folder_path,
            key_map,
            transform_list,
            norm_stats=norm_stats,
            pause_removal_epsilon=pause_removal_epsilon,
        )
        self.debug = debug
        self.allowed_episode_ids = (
            set(allowed_episode_ids) if allowed_episode_ids else None
        )

    @staticmethod
    def _local_filters_match(
        metadata: dict,
        episode_hash: str,
        filters: DatasetFilter,
    ) -> bool:
        return filters.matches(
            _normalize_filter_row(metadata, episode_hash=episode_hash)
        )

    @classmethod
    def _get_local_filtered_paths(
        cls,
        search_path: Path,
        filters: DatasetFilter | None = None,
        debug: int | bool | None = None,
    ):
        filters = _ensure_dataset_filter(filters)
        if not search_path.is_dir():
            logger.warning("Local path does not exist: %s", search_path)
            return []

        if filters.is_empty():
            filtered = []
            for p in search_path.iterdir():
                if not p.is_dir():
                    continue
                episode_hash = p.name[:-5] if p.name.endswith(".zarr") else p.name
                filtered.append((str(p), episode_hash))
            logger.info("Local paths (no filter): %d episodes", len(filtered))
        else:
            filtered = []
            for p in search_path.iterdir():
                if not p.is_dir():
                    continue

                episode_hash = p.name[:-5] if p.name.endswith(".zarr") else p.name

                try:
                    store = zarr.open_group(str(p), mode="r")
                    metadata = dict(store.attrs)
                except Exception as e:
                    logger.warning("Failed to read metadata for %s: %s", p, e)
                    continue

                if cls._local_filters_match(metadata, episode_hash, filters):
                    filtered.append((str(p), episode_hash))

            logger.info("Local filtered paths: %d episodes", len(filtered))

        if debug:
            k = 10 if debug is True else int(debug)
            filtered = filtered[:k]
            logger.info("Debug mode: using first %d episodes", len(filtered))

        return filtered

    def resolve(
        self,
        sync_from_s3=False,
        filters: DatasetFilter | None = None,
    ) -> dict[str, "ZarrDataset"]:
        """
        Outputs a dict of ZarrDatasets with relevant filters from local data.
        """
        if sync_from_s3:
            logger.warning(
                "LocalEpisodeResolver does not sync from S3; ignoring sync_from_s3=True."
            )

        filters = _ensure_dataset_filter(filters)

        if self.allowed_episode_ids is not None:
            # Skip the full directory scan — directly resolve and load only the allowed paths.
            dataset_class = self._dataset_class or ZarrDataset
            datasets = {}
            for episode_hash in self.allowed_episode_ids:
                for candidate in (
                    self.folder_path / f"{episode_hash}.zarr",
                    self.folder_path / episode_hash,
                ):
                    if candidate.is_dir():
                        try:
                            datasets[episode_hash] = dataset_class(
                                candidate,
                                key_map=self.key_map,
                                transform_list=self.transform_list,
                                norm_stats=self.norm_stats,
                                pause_removal_epsilon=self.pause_removal_epsilon,
                            )
                        except Exception as e:
                            logger.error(
                                "Failed to load dataset at %s: %s", candidate, e
                            )
                        break
                else:
                    logger.warning(
                        "allowed_episode_id not found on disk: %s", episode_hash
                    )
            if not datasets:
                raise ValueError(
                    "No valid episodes found for allowed_episode_ids in the local directory."
                )
            self._run_pause_precompute(datasets)
            return datasets

        filtered_paths = self._get_local_filtered_paths(
            self.folder_path, filters, debug=self.debug
        )

        valid_folder_names = {folder_name for _, folder_name in filtered_paths}
        logger.info(f"Valid folder names: {valid_folder_names}")
        if not valid_folder_names:
            raise ValueError(
                "No valid collection names from local filtering: "
                "filters matched no episodes in the local directory."
            )
        datasets = self._load_zarr_datasets(
            search_path=self.folder_path, valid_folder_names=valid_folder_names
        )
        self._run_pause_precompute(datasets)
        return datasets


class ModalEpisodeResolver(EpisodeResolver):
    """
    Resolves episodes via SQL and loads from a locally-mounted Modal volume.

    Filtering happens on the SQL table (fast — no per-file zarr reads).
    Episodes present in the table but absent from the local volume are skipped
    with a warning, so a partially-synced volume still works.

    This class is also the home for Modal-specific fan-out parallel container
    methods (_modal_fanout_scan, _modal_fanout_load) used by other Modal code.
    """

    def __init__(
        self,
        folder_path: Path,
        key_map: dict | None = None,
        transform_list: list | None = None,
        debug: int | bool | None = None,
        norm_stats: dict | None = None,
        exclude_hashes: list[str] | None = None,
        pause_removal_epsilon: float | None = None,
        eps_to_ignore: str | None = None,
        eps_to_use: str | None = None,
        allowed_episode_ids: list[str] | None = None,
    ):
        super().__init__(
            folder_path,
            key_map,
            transform_list,
            norm_stats=norm_stats,
            pause_removal_epsilon=pause_removal_epsilon,
        )
        self.debug = debug
        self.exclude_hashes: set[str] = set(exclude_hashes) if exclude_hashes else set()
        if eps_to_ignore:
            with open(eps_to_ignore) as f:
                self.exclude_hashes.update(json.load(f))
            logger.info(
                "eps_to_ignore: %d hashes from %s",
                len(self.exclude_hashes),
                eps_to_ignore,
            )
        self.include_hashes: set[str] | None = None
        if eps_to_use:
            with open(eps_to_use) as f:
                self.include_hashes = set(json.load(f))
            logger.info(
                "eps_to_use: %d hashes from %s", len(self.include_hashes), eps_to_use
            )
        # allowed_episode_ids: restrict to exactly these hashes (used by curation per-task scoping)
        if allowed_episode_ids is not None:
            allowed_set = set(allowed_episode_ids)
            self.include_hashes = (
                allowed_set
                if self.include_hashes is None
                else self.include_hashes & allowed_set
            )
            logger.info(
                "allowed_episode_ids: restricted to %d episodes",
                len(self.include_hashes),
            )

    def resolve(
        self,
        filters: DatasetFilter | None = None,
        **kwargs,
    ) -> dict[str, "ZarrDataset"]:
        filters = _ensure_dataset_filter(filters)

        engine = create_default_engine()
        df = episode_table_to_df(engine)

        if df.empty:
            raise ValueError("SQL episode table is empty.")

        # Always exclude deleted episodes before any other filtering
        if "is_deleted" in df.columns:
            df = df[df["is_deleted"] != True]  # noqa: E712 — handles non-bool True values

        if self.exclude_hashes:
            before = len(df)
            df = df[~df["episode_hash"].isin(self.exclude_hashes)]
            logger.info("eps_to_ignore: excluded %d episodes", before - len(df))

        if self.include_hashes is not None:
            before = len(df)
            df = df[df["episode_hash"].isin(self.include_hashes)]
            logger.info("eps_to_use: restricted to %d / %d episodes", len(df), before)

        df = filters.filter_df(df)
        mask = df.apply(
            lambda row: filters.matches(_normalize_filter_row(row.to_dict())),
            axis=1,
        )
        matched = df.loc[mask, ["episode_hash", "num_frames", "robot_name"]]
        episode_hashes = matched["episode_hash"].tolist()
        logger.info("SQL filter matched %d episodes", len(episode_hashes))

        if not episode_hashes:
            raise ValueError("SQL filter matched no episodes.")

        if self.debug:
            k = 10 if self.debug is True else int(self.debug)
            matched = matched.iloc[:k]
            episode_hashes = matched["episode_hash"].tolist()
            logger.info("Debug mode: using first %d episodes", len(episode_hashes))

        dataset_class = self._dataset_class or ZarrDataset

        # Build a lookup from episode_hash → (num_frames, robot_name) from SQL
        meta_lookup = {
            row["episode_hash"]: (int(row["num_frames"]), str(row["robot_name"]))
            for _, row in matched.iterrows()
        }

        datasets: dict[str, ZarrDataset] = {}
        n_missing = 0
        for episode_hash, (num_frames, robot_name) in meta_lookup.items():
            local_path = next(
                (
                    p
                    for p in (
                        self.folder_path / episode_hash,
                        self.folder_path / f"{episode_hash}.zarr",
                    )
                    if p.is_dir()
                ),
                None,
            )
            if local_path is None:
                n_missing += 1
                logger.warning("Episode not found locally, skipping: %s", episode_hash)
                continue
            datasets[episode_hash] = dataset_class(
                local_path,
                key_map=self.key_map,
                transform_list=self.transform_list,
                norm_stats=self.norm_stats,
                pause_removal_epsilon=self.pause_removal_epsilon,
                _total_frames=num_frames,
                _embodiment=robot_name,
            )

        if n_missing:
            logger.info("Skipped %d episodes not present in local volume", n_missing)

        if not datasets:
            raise ValueError(
                "No episodes loaded — all SQL-matched episodes are missing from the local volume."
            )

        logger.info("Loaded %d episodes from local volume", len(datasets))
        self._run_pause_precompute(datasets)
        return datasets

    # ------------------------------------------------------------------
    # Modal SDK import helper
    # ------------------------------------------------------------------

    @staticmethod
    def _import_real_modal():
        """Return the installed Modal SDK, working around egomimic.modal shadowing.

        When trainHydra.py runs as `python /root/EgoVerse/egomimic/trainHydra.py`,
        Python puts /root/EgoVerse/egomimic on sys.path[0], which makes a bare
        `import modal` resolve to the egomimic.modal subpackage. This helper:
          1. Returns the cached real SDK if one is already in sys.modules.
          2. Otherwise strips egomimic from sys.path and re-imports.
        """
        import sys

        existing = sys.modules.get("modal")
        if existing is not None and hasattr(existing, "Function"):
            return existing

        _orig_path = list(sys.path)
        sys.path = [p for p in sys.path if Path(p).name != "egomimic"]
        sys.modules.pop("modal", None)
        try:
            import modal as _modal
        finally:
            sys.path = _orig_path

        if not hasattr(_modal, "Function"):
            raise RuntimeError(
                "Imported `modal` has no `Function` attribute — "
                "egomimic.modal subpackage is still shadowing the installed Modal SDK"
            )
        return _modal

    # ------------------------------------------------------------------
    # Modal fan-out helpers — parallel container code for Modal volume
    # ------------------------------------------------------------------

    @classmethod
    def _modal_fanout_scan(
        cls,
        search_path: Path,
        filters: DatasetFilter,
        start_time: float,
    ) -> list[tuple[str, str]]:
        """Fan out a zarr metadata filter scan across Modal containers via scan_shard.

        Lists candidate dir names locally (one fast readdir), shards them, and
        invokes egomimic-scan::scan_shard in parallel.  Each worker mounts the
        zarr volume read-only and runs a thread-pooled .zattrs scan.
        """
        import time

        modal = cls._import_real_modal()

        logger.info(f"Modal fan-out scan: listing {search_path} (single readdir)...")
        t0 = time.monotonic()
        names = [n for n in os.listdir(search_path) if not n.startswith(".")]
        total = len(names)
        logger.info(
            f"Modal fan-out scan: listed {total} entries in {time.monotonic() - t0:.1f}s"
        )
        if total == 0:
            return []

        n_shards = min(int(os.environ.get("EGOMIMIC_SCAN_SHARDS", "100")), total)
        shards = [names[i::n_shards] for i in range(n_shards)]
        shards = [s for s in shards if s]
        total_shards = len(shards)

        logger.info(
            f"Modal fan-out scan: {total} entries across {total_shards} shards "
            f"(~{total // total_shards} per shard). Looking up scan_shard..."
        )

        fn = modal.Function.from_name(
            "egomimic-scan", "scan_shard", environment_name="robotics"
        )
        logger.info("Modal fan-out scan: function lookup OK; launching .map()...")
        filter_lambdas = list(filters.filter_lambdas)

        matched: list[tuple[str, str]] = []
        completed = 0
        log_every = max(1, total_shards // 20)
        for shard_result in fn.map(shards, [filter_lambdas] * total_shards):
            completed += 1
            matched.extend(shard_result)
            if completed % log_every == 0 or completed == total_shards:
                elapsed = time.monotonic() - start_time
                logger.info(
                    f"Modal scan: shard {completed}/{total_shards} done "
                    f"| matched={len(matched)} | elapsed {elapsed:.0f}s"
                )

        logger.info(
            f"Modal fan-out scan complete: {len(matched)} matches from {total} "
            f"entries in {time.monotonic() - start_time:.1f}s "
            f"({total_shards} shards)"
        )
        return matched

    def _modal_fanout_load(
        self,
        search_path: Path,
        valid_folder_names: set[str],
        dataset_class,
    ) -> dict:
        """Fan out per-episode .zattrs reads across Modal containers via load_shard.

        Worker returns picklable (path, hash, metadata) tuples; ZarrDataset is
        constructed locally with precomputed_metadata so its zarr open is
        deferred until first __getitem__.
        """
        import time

        modal = self._import_real_modal()

        start_time = time.monotonic()
        logger.info(f"Modal fan-out load: listing {search_path} (single readdir)...")
        t0 = time.monotonic()
        on_disk_names = []
        for raw_name in os.listdir(search_path):
            if raw_name.startswith("."):
                continue
            hash_name = raw_name[:-5] if raw_name.endswith(".zarr") else raw_name
            if hash_name in valid_folder_names:
                on_disk_names.append(raw_name)
        total = len(on_disk_names)
        logger.info(
            f"Modal fan-out load: {total} target entries (of {len(valid_folder_names)} requested) "
            f"in {time.monotonic() - t0:.1f}s"
        )
        if total == 0:
            return {}

        n_shards = min(int(os.environ.get("EGOMIMIC_LOAD_SHARDS", "100")), total)
        shards = [on_disk_names[i::n_shards] for i in range(n_shards)]
        shards = [s for s in shards if s]
        total_shards = len(shards)

        logger.info(
            f"Modal fan-out load: {total} entries across {total_shards} shards "
            f"(~{total // total_shards} per shard). Looking up load_shard..."
        )

        fn = modal.Function.from_name(
            "egomimic-scan", "load_shard", environment_name="robotics"
        )
        logger.info("Modal fan-out load: function lookup OK; launching .map()...")

        datasets: dict = {}
        skipped: list[str] = []
        completed = 0
        log_every = max(1, total_shards // 20)
        for shard_result in fn.map(shards):
            completed += 1
            for path_str, episode_hash, metadata in shard_result:
                if episode_hash not in valid_folder_names:
                    continue
                try:
                    datasets[episode_hash] = dataset_class(
                        Path(path_str),
                        key_map=self.key_map,
                        transform_list=self.transform_list,
                        norm_stats=self.norm_stats,
                        precomputed_metadata=metadata,
                    )
                except Exception as e:
                    logger.error("Failed to construct dataset for %s: %s", path_str, e)
                    skipped.append(episode_hash)
            if completed % log_every == 0 or completed == total_shards:
                elapsed = time.monotonic() - start_time
                logger.info(
                    f"Modal load: shard {completed}/{total_shards} done "
                    f"| loaded={len(datasets)} skipped={len(skipped)} "
                    f"| elapsed {elapsed:.0f}s"
                )

        logger.info(
            f"Modal fan-out load complete: {len(datasets)} datasets "
            f"(skipped {len(skipped)}) in {time.monotonic() - start_time:.1f}s "
            f"({total_shards} shards)"
        )
        return datasets

    # ------------------------------------------------------------------
    # Pause precompute — Modal fan-out override
    # ------------------------------------------------------------------

    def _run_pause_precompute(self, datasets: dict) -> None:
        """Run per-episode pause precompute; no-op if epsilon is None.

        Priority:
          1. Pre-built cache JSON ($EGOMIMIC_PAUSE_PRECOMPUTE_CACHE) — fastest.
          2. Modal fan-out (inside Modal container with volume mount).
          3. In-process thread pool (local dev fallback).
        """
        if self.pause_removal_epsilon is None or not datasets:
            return

        cache_path = os.environ.get(PAUSE_PRECOMPUTE_CACHE_ENV)
        if cache_path and Path(cache_path).is_file():
            remaining = self._apply_pause_precompute_cache(cache_path, datasets)
            if remaining:
                self._inprocess_pause_precompute(remaining)
            return

        if self._should_use_modal_pause_precompute(datasets):
            try:
                self._modal_fanout_pause_precompute(datasets)
                return
            except Exception as e:
                logger.warning(
                    "Modal pause precompute failed (%s) — falling back to in-process thread pool",
                    e,
                )

        self._inprocess_pause_precompute(datasets)

    @staticmethod
    def _should_use_modal_pause_precompute(datasets: dict) -> bool:
        inside_modal = os.environ.get("MODAL_IS_REMOTE") == "1" or bool(
            os.environ.get("MODAL_TASK_ID")
        )
        if not inside_modal:
            return False
        if os.environ.get("EGOMIMIC_DISABLE_MODAL_PAUSE_PRECOMPUTE") == "1":
            return False
        return any(
            str(getattr(ds, "episode_path", "")).startswith("/mnt/zarr-data")
            for ds in datasets.values()
        )

    def _modal_fanout_pause_precompute(self, datasets: dict) -> None:
        """Compute keep_indices in parallel across egomimic-scan::pause_precompute_shard workers."""
        import time

        modal = self._import_real_modal()
        t0 = time.monotonic()

        work = [(name, str(ds.episode_path)) for name, ds in datasets.items()]
        n = len(work)
        n_shards = min(
            int(os.environ.get("EGOMIMIC_PAUSE_PRECOMPUTE_SHARDS", "100")), n
        )
        shards = [work[i::n_shards] for i in range(n_shards)]
        shards = [s for s in shards if s]
        total_shards = len(shards)

        logger.info(
            "Pause precompute via Modal fan-out: %d episodes across %d shards — "
            "looking up pause_precompute_shard function...",
            n,
            total_shards,
        )
        fn = modal.Function.from_name(
            "egomimic-scan", "pause_precompute_shard", environment_name="robotics"
        )
        epsilons = [self.pause_removal_epsilon] * total_shards

        n_total = 0
        n_kept = 0
        n_errs = 0
        completed = 0
        log_every = max(1, total_shards // 20)
        for shard_result in fn.map(shards, epsilons):
            completed += 1
            for episode_hash, raw_total, indices in shard_result:
                ds = datasets.get(episode_hash)
                if ds is None:
                    n_errs += 1
                    continue
                if raw_total == 0:
                    n_errs += 1
                    continue
                ds._raw_total_frames = int(raw_total)
                ds.keep_indices = np.asarray(indices, dtype=np.int64)
                n_total += raw_total
                n_kept += len(indices)
            if completed % log_every == 0 or completed == total_shards:
                elapsed = time.monotonic() - t0
                logger.info(
                    "Pause precompute via Modal: shard %d/%d done | kept=%d/%d "
                    "errors=%d | elapsed %.0fs",
                    completed,
                    total_shards,
                    n_kept,
                    n_total,
                    n_errs,
                    elapsed,
                )

        elapsed = time.monotonic() - t0
        pct = (100.0 * n_kept / n_total) if n_total else 100.0
        logger.info(
            "Pause precompute via Modal fan-out complete (epsilon=%s): kept %d/%d "
            "frames (%.1f%%) across %d episodes in %.1fs (errors=%d, shards=%d)",
            self.pause_removal_epsilon,
            n_kept,
            n_total,
            pct,
            n,
            elapsed,
            n_errs,
            total_shards,
        )


# Backward-compat alias — YAML configs that reference LocalSQLEpisodeResolver continue to work.
LocalSQLEpisodeResolver = ModalEpisodeResolver


class MultiDataset(torch.utils.data.Dataset):
    """
    Self wrapping MultiDataset, can wrap zarr or multi dataset.

    """

    def __init__(
        self,
        datasets: dict[str, MultiDataset | ZarrDataset | ZarrActionExpertDataset],
        mode="train",
        percent=0.1,
        valid_ratio=0.2,
        **kwargs,
    ):
        """
        Args:
            datasets (dict): Dictionary mapping unique dataset hashes (str) to dataset objects. Datasets can be individual Zarr datasets or other multi-datasets; mixing different types is supported.
            mode (str, optional): Split mode to use (e.g., "train", "valid"). Defaults to "train".
            percent (float, optional): Fraction of the dataset to use from each underlying dataset. Defaults to 0.1.
            valid_ratio (float, optional): Validation split ratio for datasets that support a train/valid split.
            **kwargs: Additional keyword arguments passed to underlying dataset constructors if needed.
        """
        self.train_collections, self.valid_collections = split_dataset_names(
            datasets.keys(), valid_ratio=valid_ratio, seed=SEED
        )

        if mode == "train":
            chosen = self.train_collections
        elif mode == "valid":
            chosen = self.valid_collections
        elif mode == "total":
            chosen = set(datasets.keys())
        elif mode == "percent":
            all_names = sorted(datasets.keys())
            rng = random.Random(SEED)
            rng.shuffle(all_names)

            n_keep = int(len(all_names) * percent)
            if percent > 0.0:
                n_keep = max(1, n_keep)
            chosen = set(all_names[:n_keep])
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.datasets = {rid: ds for rid, ds in datasets.items() if rid in chosen}
        assert self.datasets, "No datasets left after applying mode split."

        self.index_map = []
        self._global_indices_by_dataset: dict[str, list[int]] = {
            dataset_name: [] for dataset_name in self.datasets
        }
        for dataset_name, dataset in self.datasets.items():
            for local_idx in range(len(dataset)):
                global_idx = len(self.index_map)
                self.index_map.append((dataset_name, local_idx))
                self._global_indices_by_dataset[dataset_name].append(global_idx)

        self.data_schematic = None
        self._n_samples_checked = 0
        self._n_violation_samples = 0
        self._violation_log_every = 100

        super().__init__()

    def __len__(self) -> int:
        return len(self.index_map)

    @staticmethod
    def _episode_name_for_dataset(dataset, dataset_name: str) -> str:
        episode_path = getattr(dataset, "episode_path", None)
        if episode_path is None:
            return dataset_name
        return Path(episode_path).name

    def _check_bounds(
        self, data: dict, dataset, idx: int, dataset_name: str
    ) -> str | None:
        if self.data_schematic is None:
            return None

        embodiment_id = data.get("embodiment")
        if embodiment_id is None:
            raise ValueError("data has no embodiment metadata")

        norm_stats = self.data_schematic.norm_stats.get(embodiment_id, {})
        if not norm_stats:
            return None

        self._n_samples_checked += 1
        prefix: str | None = None

        for key_name, stats in norm_stats.items():
            zarr_key = self.data_schematic.keyname_to_zarr_key(key_name, embodiment_id)
            if zarr_key is None or zarr_key not in data:
                continue

            v = data[zarr_key]
            if isinstance(v, torch.Tensor):
                arr = v.float()
            elif isinstance(v, np.ndarray):
                arr = torch.from_numpy(v).float()
            else:
                continue

            q_low = stats.get(
                "quantile_0_01",
                stats.get("quantile_0_1", stats["quantile_1"]),
            )
            q_high = stats.get(
                "quantile_99_99",
                stats.get("quantile_99_9", stats["quantile_99"]),
            )
            q_low = torch.as_tensor(q_low, device=arr.device, dtype=torch.float32)
            q_high = torch.as_tensor(q_high, device=arr.device, dtype=torch.float32)

            try:
                q_low = torch.broadcast_to(q_low, arr.shape)
                q_high = torch.broadcast_to(q_high, arr.shape)
            except RuntimeError:
                shape_warn_key = (str(zarr_key), tuple(arr.shape), tuple(q_low.shape))
                if shape_warn_key not in getattr(self, "_shape_mismatch_warned", set()):
                    if not hasattr(self, "_shape_mismatch_warned"):
                        self._shape_mismatch_warned = set()
                    self._shape_mismatch_warned.add(shape_warn_key)
                    logger.warning(
                        "Skipping bounds check for key=%s due to incompatible shapes: value=%s q_low=%s",
                        zarr_key,
                        tuple(arr.shape),
                        tuple(q_low.shape),
                    )
                continue

            if torch.any(torch.isnan(arr)) or torch.any(torch.isinf(arr)):
                episode_name = self._episode_name_for_dataset(dataset, dataset_name)
                prefix = (
                    f"NaN/Inf violation ep={episode_name} frame={idx} key={zarr_key}"
                )
                break

            # The bimanual cartesian action chunk and the ee_pose proprio share a
            # [L | R] layout whose rotation channels are either Euler ypr (wraps
            # at ±π) or continuous 6D columns. In both cases quantile bounds on
            # the rotation channels are meaningless and reject otherwise-valid
            # frames, so we only bounds-check the translation (and gripper)
            # channels. ``bimanual_cartesian_layout`` recognizes widths 12/14
            # (ypr) and 18/20 (6D); any other width falls through to a
            # full-vector check. NaN/Inf above still covers the full vector.
            cartesian_layout = None
            if zarr_key in ("actions_cartesian", "observations.state.ee_pose"):
                cartesian_layout = bimanual_cartesian_layout(arr.shape[-1])
            if cartesian_layout is not None:
                check_idx = list(cartesian_layout["xyz"]) + list(
                    cartesian_layout["grip"]
                )
                arr_q = arr[..., check_idx]
                q_low_c = q_low[..., check_idx]
                q_high_c = q_high[..., check_idx]
            else:
                arr_q = arr
                q_low_c = q_low
                q_high_c = q_high

            if torch.any(arr_q < q_low_c) or torch.any(arr_q > q_high_c):
                episode_name = self._episode_name_for_dataset(dataset, dataset_name)
                prefix = (
                    f"Bounds violation ep={episode_name} frame={idx} key={zarr_key}"
                )
                break

        if prefix is not None:
            self._n_violation_samples += 1

        if (
            self._n_samples_checked == 1
            or self._n_samples_checked % self._violation_log_every == 0
        ):
            pct = 100 * self._n_violation_samples / max(self._n_samples_checked, 1)
            logger.info(
                "[BoundsCheck] %d samples checked: %d violations (%.1f%%)",
                self._n_samples_checked,
                self._n_violation_samples,
                pct,
            )

        return prefix

    def __getitem__(self, idx, _attempts: int | None = None):
        """
        Multidataset handles outlier rejection so that you don't need to propagate the norm stats down to every sub dataset.
        """
        dataset_name, local_idx = self.index_map[idx]
        dataset = self.datasets[dataset_name]
        data = dataset[local_idx]

        if isinstance(dataset, MultiDataset):
            return data

        violation = self._check_bounds(data, dataset, local_idx, dataset_name)
        if violation is not None:
            next_idx, attempts = get_fallback_idx(
                idx=idx,
                candidates=self._global_indices_by_dataset[dataset_name],
                _attempts=_attempts,
                max_attempts=len(self._global_indices_by_dataset[dataset_name]),
                exhausted_error=(
                    f"Entire dataset bad (no valid indices): dataset={dataset_name}"
                ),
            )
            return self.__getitem__(next_idx, _attempts=attempts)

        return data

    def set_data_schematic(self, data_schematic, bounds_slack: float = 0.0) -> None:
        """
        Set the data schematic used for top-level bounds checking.

        When child datasets are themselves MultiDatasets, recursively assign the
        same schematic so each wrapper can validate its own returned samples.

        bounds_slack: absolute tolerance added to each side of the quantile bounds
            before rejecting a frame. Use a small value (e.g. 0.01) to suppress
            spurious rejections at circular-domain boundaries such as ±π.
        """
        self.data_schematic = data_schematic
        self.bounds_slack = bounds_slack
        for ds in self.datasets.values():
            if isinstance(ds, MultiDataset):
                ds.set_data_schematic(data_schematic, bounds_slack=bounds_slack)
        logger.info(
            f"Set data_schematic on MultiDataset with {len(self.datasets)} child datasets"
        )

    @classmethod
    def _from_resolver(cls, resolver: EpisodeResolver, **kwargs):
        """
        create a MultiDataset from an EpisodeResolver.

        Args:
            resolver (EpisodeResolver): The resolver instance to use for loading datasets.
            **kwargs: Keyword args forwarded to resolver (e.g., filters,
                sync_from_s3) and MultiDataset constructor (e.g., mode, percent,
                key_map, valid_ratio).
        Returns:
            MultiDataset: The constructed multi-dataset.
        """
        sync_from_s3 = kwargs.pop("sync_from_s3", False)
        filters = kwargs.pop("filters", None)

        if isinstance(resolver, LocalEpisodeResolver):
            resolved = resolver.resolve(
                sync_from_s3=sync_from_s3,
                filters=filters,
            )
        else:
            resolved = resolver.resolve(filters=filters)

        return cls(datasets=resolved, **kwargs)


class ZarrDataset(torch.utils.data.Dataset):
    """
    Base Zarr Dataset object — reads frames from a single zarr episode directory.
    """

    def __init__(
        self,
        Episode_path: Path,
        key_map: dict,
        transform_list: list | None = None,
        norm_stats: dict | None = None,
        pause_removal_epsilon: float | None = None,
        _total_frames: int | None = None,
        _embodiment: str | None = None,
        precomputed_metadata: dict | None = None,
        defer_open: bool = False,
    ):
        """
        Args:
            episode_path: just a path to the designated zarr episode
            key_map: dict mapping from dataset keys to zarr keys and horizon info, e.g. {"obs/image/front": {"zarr_key": "observations.images.front", "horizon": 4}, ...}
            transform_list: list of Transform objects to apply to the data after loading, e.g. for action chunk transformations. Should be in order of application.
            norm_stats: optional dict mapping dataset key names (same keys as key_map) to
                {"quantile_1": tensor, "quantile_99": tensor} bounds. When provided, any
                loaded sample whose values fall outside [quantile_1, quantile_99] for any
                tracked key triggers the random index fallback.
            pause_removal_epsilon: when set, the resolver calls
                ``precompute_pause_filter()`` after construction, which sets
                ``keep_indices`` from left/right obs_ee_pose deltas. ``__len__``
                then returns the kept count and ``__getitem__`` translates
                through keep_indices. None leaves the episode unchanged.
            _total_frames: if provided (e.g. from SQL), skip zarr open at init and defer to first __getitem__.
            _embodiment: if provided alongside _total_frames, also skip zarr open at init.
            precomputed_metadata: full .zattrs dict pre-loaded externally (e.g. from _modal_fanout_load).
                When provided, skips zarr open at init and populates all metadata fields immediately.
            defer_open: if True, do not mmap the zarr store until the first ``__getitem__``
                (used by PrefetchedIterableDataset workers).
        """
        self.episode_path = Episode_path
        self.metadata = None
        self._image_keys = None
        self._json_keys = None
        self._annotations = None
        self.episode_reader = None

        if precomputed_metadata is not None:
            # Full metadata pre-loaded externally (e.g. from Modal fan-out load_shard)
            self._init_from_metadata(precomputed_metadata)
        elif defer_open:
            # Prefetch workers: open the store inside __getitem__, not at construction.
            self.total_frames = 0
            self.embodiment = ""
            self.keys_dict = {}
        elif _total_frames is not None and _embodiment is not None:
            # Lightweight fast path from SQL — defer zarr open to first access
            self.total_frames = _total_frames
            self.embodiment = _embodiment
            self.keys_dict = {}
        else:
            self.init_episode()

        self.key_map = key_map
        self.transform = transform_list
        self.norm_stats = norm_stats or {}
        self.pause_removal_epsilon = pause_removal_epsilon
        self.keep_indices: np.ndarray | None = None
        self._raw_total_frames: int | None = None
        self._zarr_bulk_cache: dict[str, np.ndarray] | None = None
        super().__init__()

    def _ensure_episode_reader(self):
        """Open the zarr store on first access if it was deferred at init."""
        if self.episode_reader is None:
            self.init_episode()

    def init_episode(self):
        """Eagerly open the zarr group and populate all metadata-derived fields."""
        self.episode_reader = ZarrEpisode(self.episode_path)
        self._init_from_metadata(self.episode_reader.metadata)

    def close(self) -> None:
        """Release zarr mmap handles and in-memory caches (for lazy episode loading)."""
        if self.episode_reader is not None:
            self.episode_reader.close()
            self.episode_reader = None
        self._zarr_bulk_cache = None
        self._annotations = None

    def _init_from_metadata(self, metadata: dict) -> None:
        """Populate metadata-derived fields from a pre-loaded `.zattrs` dict."""
        self.metadata = metadata
        self.total_frames = metadata["total_frames"]
        self.embodiment = metadata["embodiment"]
        features = metadata.get("features", {})
        keys_list = (
            list(features.keys()) if isinstance(features, dict) else list(features)
        )
        self.keys_dict = {k: (0, None) for k in keys_list}
        self._image_keys = self._detect_image_keys()
        self._json_keys = self._detect_json_keys()

    def _detect_image_keys(self) -> set[str]:
        """
        Detect which keys contain JPEG-encoded image data from metadata.

        Returns:
            Set of keys containing JPEG data
        """
        features = self.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "jpeg"}

    def _detect_json_keys(self) -> set[str]:
        """
        Detect keys containing JSON-encoded bytes from metadata.

        Returns:
            Set of keys containing JSON payloads.
        """
        features = self.metadata.get("features", {})
        return {key for key, info in features.items() if info.get("dtype") == "json"}

    @staticmethod
    def _decode_json_entry(value):
        if isinstance(value, np.void):
            value = value.item()
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bytes):
            return json.loads(value.decode("utf-8"))
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _load_annotations(self) -> list[dict]:
        """
        Load and cache decoded language annotations.

        Expected format per entry:
            {"text": str, "start_idx": int, "end_idx": int}
        """
        if self._annotations is not None:
            return self._annotations

        raw = self.episode_reader._store["annotations"][:]

        decoded = [self._decode_json_entry(x) for x in raw]
        self._annotations = [d for d in decoded if isinstance(d, dict)]
        return self._annotations

    def _annotation_text_for_frame(self, frame_idx: int) -> str:
        """
        Resolve language annotation text for a frame from span annotations.
        """
        annotations = self._load_annotations()
        valid_annotations = []
        for ann in annotations:
            start_idx = int(ann.get("start_idx", -1))
            end_idx = int(ann.get("end_idx", -1))
            if start_idx <= frame_idx < end_idx:
                valid_annotations.append(ann.get("text", ""))
        return valid_annotations

    def __len__(self) -> int:
        if self.keep_indices is not None:
            return int(len(self.keep_indices))
        return self.total_frames

    def actions_only_view(self) -> "ZarrDataset":
        """
        Shallow copy without camera/annotation keys (same transforms and pause mask).

        Used by the curation batched transform path; images are bulk-read separately
        in ``collect_curation_episode``.
        """
        import copy

        view = copy.copy(self)
        view.key_map = {
            k: v
            for k, v in self.key_map.items()
            if v.get("key_type") not in ("camera_keys", "annotation_keys")
        }
        view._zarr_bulk_cache = None
        return view

    def preload_zarr_arrays(self) -> None:
        """Load every zarr array in ``key_map`` once (``store[key][:]`` into RAM)."""
        self._ensure_episode_reader()
        store = self.episode_reader._store
        cache: dict[str, np.ndarray] = {}
        for spec in self.key_map.values():
            zarr_key = spec.get("zarr_key")
            if not zarr_key or zarr_key in cache:
                continue
            cache[zarr_key] = np.asarray(store[zarr_key][:])
        self._zarr_bulk_cache = cache

    def _read_key_slice(self, zarr_key: str, start: int, end: int | None) -> np.ndarray:
        if self._zarr_bulk_cache is not None and zarr_key in self._zarr_bulk_cache:
            arr = self._zarr_bulk_cache[zarr_key]
            if end is not None:
                return arr[start:end]
            return arr[start : start + 1][0]
        read_dict = {zarr_key: (start, end)}
        return self.episode_reader.read(read_dict)[zarr_key]

    _CURATION_POSE_SENTINEL = 1e8
    _CURATION_MIN_QUAT_NORM = 1e-6

    def _logical_to_real_index(self, logical_idx: int) -> int:
        if self.keep_indices is not None:
            return int(self.keep_indices[logical_idx])
        return int(logical_idx)

    @staticmethod
    def _pose_rows_valid(
        pose: np.ndarray,
        *,
        min_quat_norm: float = 1e-6,
        sentinel: float = 1e8,
    ) -> np.ndarray:
        """Per-row validity for ``(T, 7)`` xyz+wxyz pose arrays."""
        pose = np.asarray(pose)
        if pose.ndim != 2 or pose.shape[-1] != 7:
            raise ValueError(f"Expected (T, 7) pose, got {pose.shape}")
        xyz_ok = np.all(np.abs(pose[:, :3]) < sentinel, axis=1)
        quat_norms = np.linalg.norm(pose[:, 3:7], axis=1)
        return xyz_ok & (quat_norms > min_quat_norm)

    def _action_chunk_horizon(self) -> int:
        for spec in self.key_map.values():
            horizon = spec.get("horizon")
            if horizon is not None:
                return int(horizon)
        return 1

    def _curation_valid_logical_indices(self) -> np.ndarray:
        """Logical timestep indices with valid pose windows (skip zero-quat / cut rows)."""
        n_logical = len(self)
        if n_logical == 0:
            return np.array([], dtype=np.int64)

        horizon = self._action_chunk_horizon()
        view = self.actions_only_view()
        cache = view._zarr_bulk_cache or self._zarr_bulk_cache
        if cache is None:
            return np.array([], dtype=np.int64)

        required = {
            "obs_head_pose",
            "left.obs_ee_pose",
            "right.obs_ee_pose",
        }
        if not required.issubset(cache.keys()):
            return np.array([], dtype=np.int64)

        head = np.asarray(cache["obs_head_pose"])
        left = np.asarray(cache["left.obs_ee_pose"])
        right = np.asarray(cache["right.obs_ee_pose"])
        head_ok = self._pose_rows_valid(
            head,
            min_quat_norm=self._CURATION_MIN_QUAT_NORM,
            sentinel=self._CURATION_POSE_SENTINEL,
        )
        left_ok = self._pose_rows_valid(
            left,
            min_quat_norm=self._CURATION_MIN_QUAT_NORM,
            sentinel=self._CURATION_POSE_SENTINEL,
        )
        right_ok = self._pose_rows_valid(
            right,
            min_quat_norm=self._CURATION_MIN_QUAT_NORM,
            sentinel=self._CURATION_POSE_SENTINEL,
        )

        valid: list[int] = []
        for logical_idx in range(n_logical):
            real_idx = self._logical_to_real_index(logical_idx)
            end_idx = min(real_idx + horizon, left.shape[0])
            if real_idx >= end_idx:
                continue
            if not head_ok[real_idx]:
                continue
            if (
                not left_ok[real_idx:end_idx].all()
                or not right_ok[real_idx:end_idx].all()
            ):
                continue
            if not left_ok[real_idx] or not right_ok[real_idx]:
                continue
            valid.append(logical_idx)

        return np.asarray(valid, dtype=np.int64)

    def _build_batched_transform_batch(
        self, logical_indices: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Stack per-timestep raw/transform inputs for ``apply_transform``-style batching.

        Uses vectorised sliding-window indexing when the instance's ``_chunk_end_idx``
        is the base implementation (simple ``min(start+H, total_frames)`` clamping).
        Subclasses that override ``_chunk_end_idx`` (e.g. ``ZarrAnnotationCutoffDataset``)
        fall back to the per-sample loop to preserve their clamping semantics.
        """
        view = self.actions_only_view()
        view.keep_indices = self.keep_indices
        if self._zarr_bulk_cache is not None:
            needed = {
                spec["zarr_key"]
                for spec in view.key_map.values()
                if spec.get("zarr_key")
            }
            view._zarr_bulk_cache = {
                k: self._zarr_bulk_cache[k]
                for k in needed
                if k in self._zarr_bulk_cache
            }

        n = len(logical_indices)
        # Vectorised path: safe only when _chunk_end_idx is the base implementation.
        use_vec = type(self)._chunk_end_idx is ZarrDataset._chunk_end_idx
        if use_vec and self._zarr_bulk_cache is not None:
            real_indices = (
                self.keep_indices[logical_indices]
                if self.keep_indices is not None
                else logical_indices.astype(np.int64)
            )
            batch: dict[str, np.ndarray] = {}
            for key, spec in view.key_map.items():
                zarr_key = spec["zarr_key"]
                horizon = spec.get("horizon")
                arr = np.asarray(view._zarr_bulk_cache[zarr_key])
                if horizon is None:
                    # Simple gather: (n, D)
                    batch[key] = arr[real_indices]
                else:
                    H = int(horizon)
                    # Pad the tail with the last frame so every window is length H.
                    padding = np.repeat(arr[-1:], H - 1, axis=0)
                    arr_padded = np.concatenate([arr, padding], axis=0)
                    # Build (n, H) index matrix and gather in one shot.
                    step_indices = real_indices[:, None] + np.arange(H, dtype=np.int64)
                    batch[key] = arr_padded[step_indices]  # (n, H, D) or (n, H)
            return batch

        # Per-sample fallback (subclass overrides _chunk_end_idx, or no bulk cache).
        batch = {}
        for key, spec in view.key_map.items():
            zarr_key = spec["zarr_key"]
            horizon = spec.get("horizon")
            key_type = spec.get("key_type")
            rows: list[np.ndarray] = []
            for logical_idx in logical_indices:
                real_idx = self._logical_to_real_index(int(logical_idx))
                if horizon is not None:
                    end_idx = self._chunk_end_idx(real_idx, int(horizon), key_type)
                    window = self._read_key_slice(zarr_key, real_idx, end_idx)
                    window = self._pad_sequences({zarr_key: window}, int(horizon))[
                        zarr_key
                    ]
                    rows.append(np.asarray(window))
                else:
                    rows.append(
                        np.asarray(self._read_key_slice(zarr_key, real_idx, None))
                    )
            batch[key] = np.stack(rows, axis=0) if n > 1 else np.expand_dims(rows[0], 0)
        return batch

    def _apply_transform_list_batched(
        self, batch: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray] | None, np.ndarray]:
        """
        Run ``transform_list`` on a stacked batch; drop rows that still fail.

        Returns ``(output_batch, ok_positions)`` where ``ok_positions`` index into
        the leading batch dimension.

        Fast path: if every transform has ``transform_batch``, run the whole batch
        in one vectorized pass.  Falls back to the per-sample loop on any exception.
        """
        if not self.transform:
            return batch, np.arange(next(iter(batch.values())).shape[0], dtype=np.int64)

        batch_size = next(iter(batch.values())).shape[0]

        if all(hasattr(t, "transform_batch") for t in self.transform):
            try:
                data = batch
                for transform in self.transform:
                    data = transform.transform_batch(data)
                out = {
                    k: np.asarray(v)
                    for k, v in data.items()
                    if isinstance(v, np.ndarray)
                }
                return out, np.arange(batch_size, dtype=np.int64)
            except Exception as exc:
                logger.debug(
                    "Vectorized transform_batch failed for ep=%s (%s: %s), falling back to per-sample",
                    Path(self.episode_path).name,
                    type(exc).__name__,
                    exc,
                )

        results: list[dict[str, np.ndarray]] = []
        ok_positions: list[int] = []

        for i in range(batch_size):
            sample = {
                k: v[i]
                for k, v in batch.items()
                if isinstance(v, np.ndarray) and v.shape[0] == batch_size
            }
            try:
                data = sample
                for transform in self.transform:
                    data = transform.transform(data)
                results.append(
                    {
                        k: np.asarray(v)
                        for k, v in data.items()
                        if isinstance(v, np.ndarray)
                    }
                )
                ok_positions.append(i)
            except Exception as exc:
                logger.debug(
                    "Skipping curation row ep=%s batch_pos=%d: %s: %s",
                    Path(self.episode_path).name,
                    i,
                    type(exc).__name__,
                    exc,
                )

        if not results:
            return None, np.array([], dtype=np.int64)

        out: dict[str, np.ndarray] = {}
        for key in results[0]:
            out[key] = np.stack([r[key] for r in results], axis=0)
        return out, np.asarray(ok_positions, dtype=np.int64)

    def _camera_image_key(self) -> str:
        for key, spec in self.key_map.items():
            if spec.get("key_type") == "camera_keys":
                return key
        return next(iter(self.key_map))

    def _collect_curation_batched(
        self,
        action_key: str,
        image_key: str,
        image_decode_workers: int,
        *,
        load_images: bool,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if load_images and image_key not in self.key_map:
            return None, None

        if self.pause_removal_epsilon is not None and self.keep_indices is None:
            self.precompute_pause_filter()
        self.preload_zarr_arrays()

        n_logical = len(self)
        if n_logical == 0:
            return None, None

        logical_valid = self._curation_valid_logical_indices()
        n_skipped_pose = n_logical - len(logical_valid)
        if n_skipped_pose > 0:
            logger.info(
                "curation ep=%s: skipped %d/%d timesteps (invalid/zero pose rows)",
                Path(self.episode_path).name,
                n_skipped_pose,
                n_logical,
            )
        if len(logical_valid) == 0:
            return None, None

        batched_in = self._build_batched_transform_batch(logical_valid)
        batched_out, ok_pos = self._apply_transform_list_batched(batched_in)
        if batched_out is None or action_key not in batched_out:
            return None, None

        if len(ok_pos) < len(logical_valid):
            logger.info(
                "curation ep=%s: dropped %d/%d timesteps after transform failures",
                Path(self.episode_path).name,
                len(logical_valid) - len(ok_pos),
                len(logical_valid),
            )
            logical_valid = logical_valid[ok_pos]

        actions = np.asarray(batched_out[action_key], dtype=np.float32)
        if actions.ndim < 2:
            return None, None

        if not load_images:
            return actions, None

        img_zarr_key = self.key_map[image_key]["zarr_key"]
        if self._zarr_bulk_cache is None or img_zarr_key not in self._zarr_bulk_cache:
            return None, None

        jpeg_frames = self._zarr_bulk_cache[img_zarr_key]
        if self.keep_indices is not None:
            jpeg_frames = jpeg_frames[self.keep_indices]
        jpeg_frames = jpeg_frames[logical_valid]

        workers = max(0, int(image_decode_workers))
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                images = np.stack(
                    list(pool.map(self._decode_jpeg_to_chw, jpeg_frames)),
                    axis=0,
                )
        else:
            images = np.stack(
                [self._decode_jpeg_to_chw(b) for b in jpeg_frames],
                axis=0,
            )

        if len(images) != len(actions):
            raise RuntimeError(
                f"action/image length mismatch ep={self.episode_path.name}: "
                f"{len(actions)} vs {len(images)}"
            )
        return actions, images

    @staticmethod
    def _decode_jpeg_to_chw(jpeg_bytes: object) -> np.ndarray:
        """Decode one JPEG payload to float32 CHW in [0, 1]."""
        if isinstance(jpeg_bytes, np.ndarray):
            if jpeg_bytes.dtype == np.uint8 and jpeg_bytes.ndim >= 2:
                return np.transpose(jpeg_bytes, (2, 0, 1)).astype(np.float32) / 255.0
            jpeg_bytes = jpeg_bytes.item() if jpeg_bytes.ndim == 0 else jpeg_bytes[0]
        decoded = simplejpeg.decode_jpeg(jpeg_bytes, colorspace="RGB")
        return np.transpose(decoded, (2, 0, 1)).astype(np.float32) / 255.0

    def collect_curation_episode(
        self,
        action_key: str = "actions_cartesian",
        image_key: str = "observations.images.front_img_1",
        image_decode_workers: int = 8,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        DemInf Pass 2 per episode: bulk zarr read, batched transforms, aligned decode.

        1. ``preload_zarr_arrays()`` — one ``[:]`` per zarr key.
        2. Filter timesteps with invalid/zero quaternions (pose cut rows).
        3. Stack valid rows → run ``transform_list`` in batch (no random fallback).
        4. Decode JPEGs only for surviving logical indices.

        Returns:
            ``(actions, images)`` as ``(T, chunk, D)`` and ``(T, C, H, W)``, or
            ``(None, None)`` if empty / keys missing.
        """
        return self._collect_curation_batched(
            action_key=action_key,
            image_key=image_key,
            image_decode_workers=image_decode_workers,
            load_images=True,
        )

    def precompute_pause_filter(self) -> tuple[int, int]:
        """Build the per-episode pause keep-mask from raw obs_ee_pose deltas.

        Idempotent; caches on ``self.keep_indices``. Reads the pose arrays so
        the zarr store is opened on first call. Returns
        ``(raw_total_frames, kept_frames)``. If the episode does not have both
        PAUSE_DETECT_KEYS, keeps all frames.
        """
        if self.pause_removal_epsilon is None:
            return (self.total_frames, self.total_frames)
        if self.keep_indices is not None:
            return (self._raw_total_frames or self.total_frames, len(self.keep_indices))

        self._ensure_episode_reader()
        store = self.episode_reader._store
        left_key, right_key = PAUSE_DETECT_KEYS
        try:
            left_pose = np.asarray(store[left_key][:])
            right_pose = np.asarray(store[right_key][:])
        except KeyError:
            self._raw_total_frames = int(self.total_frames)
            self.keep_indices = np.arange(self.total_frames, dtype=np.int64)
            return (self.total_frames, self.total_frames)

        # Keypoints are optional — some embodiments / older episodes don't
        # have them; in that case the detector falls back to ee_pose-only.
        left_kp_key, right_kp_key = PAUSE_DETECT_KEYPOINT_KEYS
        left_kp = right_kp = None
        try:
            left_kp = np.asarray(store[left_kp_key][:])
        except KeyError:
            pass
        try:
            right_kp = np.asarray(store[right_kp_key][:])
        except KeyError:
            pass

        keep = _build_pause_keep_mask(
            left_pose=left_pose,
            right_pose=right_pose,
            epsilon=self.pause_removal_epsilon,
            left_keypoints=left_kp,
            right_keypoints=right_kp,
        )
        self._raw_total_frames = int(self.total_frames)
        self.keep_indices = np.flatnonzero(keep).astype(np.int64)
        return (self._raw_total_frames, int(len(self.keep_indices)))

    def _chunk_end_idx(self, start_idx: int, horizon: int, key_type: str | None) -> int:
        """End index (exclusive) for a windowed read starting at ``start_idx``.

        Subclasses can override to add per-key-type clamping (e.g., annotation EOS).
        """
        return min(start_idx + horizon, self.total_frames)

    def _pad_sequences(self, data, horizon: int | None) -> dict:
        if horizon is None:
            return data

        # Note that k is zarr key
        for k in data:
            if isinstance(data[k], np.ndarray):
                seq_len = data[k].shape[0]
                if seq_len < horizon:
                    # Pad by repeating the last frame
                    pad_len = horizon - seq_len
                    last_frame = data[k][-1:]  # Keep dims: (1, action_dim)
                    padding = np.repeat(last_frame, pad_len, axis=0)
                    data[k] = np.concatenate([data[k], padding], axis=0)

        return data

    def __getitem__(
        self,
        idx: int,
        _fallback_origin: int | None = None,
        _attempts: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        ZarrDataset handles jpeg decoding and transform function errors, and triggers resample on dataset level.
        """
        self._ensure_episode_reader()
        if self.keep_indices is not None:
            real_idx = int(self.keep_indices[idx])
        else:
            real_idx = idx
        data = {}
        for k in self.key_map:
            zarr_key = self.key_map[k]["zarr_key"]
            key_type = self.key_map[k].get("key_type", None)
            horizon = self.key_map[k].get("horizon", None)

            if key_type == "annotation_keys":
                data[k] = self._annotation_text_for_frame(real_idx)
                continue

            if horizon is not None:
                if self.keep_indices is not None:
                    # Fully-filtered chunk: take H consecutive *filtered* frames
                    # by fancy-indexing keep_indices, so paused raw frames are
                    # skipped inside the chunk too (not just at the anchor).
                    end_filtered = min(idx + horizon, len(self.keep_indices))
                    read_interval = self.keep_indices[idx:end_filtered]
                else:
                    end_idx = self._chunk_end_idx(real_idx, horizon, key_type)
                    read_interval = (real_idx, end_idx)
            else:
                read_interval = (real_idx, None)
            try:
                if isinstance(read_interval, np.ndarray):
                    # Fancy-index path (fully-filtered chunk)
                    read_dict = {zarr_key: read_interval}
                    raw_data = self.episode_reader.read(read_dict)
                else:
                    raw_data = {
                        zarr_key: self._read_key_slice(
                            zarr_key, read_interval[0], read_interval[1]
                        )
                    }
            except (IndexError, KeyError) as e:
                origin = _fallback_origin if _fallback_origin is not None else idx
                next_idx, attempts = get_fallback_idx(
                    idx=idx,
                    candidates=range(self.total_frames),
                    _attempts=_attempts,
                    max_attempts=self.total_frames,
                    exhausted_error=(
                        f"Entire episode bad (no valid indices): ep={Path(self.episode_path).name}"
                    ),
                )
                logger.warning(
                    f"Zarr read failed ep={Path(self.episode_path).name} frame={idx} key={k} "
                    f"({type(e).__name__}: {e}) | attempt {attempts}, trying random idx {next_idx}"
                )
                return self.__getitem__(
                    next_idx, _fallback_origin=origin, _attempts=attempts
                )
            self._pad_sequences(raw_data, horizon)  # should be able to pad images
            data[k] = raw_data[zarr_key]

            if zarr_key in self._image_keys:
                jpeg_bytes = data[k]
                try:
                    data[k] = self._decode_jpeg_to_chw(jpeg_bytes)
                except Exception:
                    origin = _fallback_origin if _fallback_origin is not None else idx
                    next_idx, attempts = get_fallback_idx(
                        idx=idx,
                        candidates=range(len(self)),
                        _attempts=_attempts,
                        max_attempts=len(self),
                        exhausted_error=(
                            f"Entire episode bad (no valid indices): ep={Path(self.episode_path).name}"
                        ),
                    )
                    logger.warning(
                        f"JPEG decode failed ep={Path(self.episode_path).name} frame={idx} key={k} | "
                        f"attempt {attempts}, trying random idx {next_idx}"
                    )
                    result = self.__getitem__(
                        next_idx, _fallback_origin=origin, _attempts=attempts
                    )
                    return result
            elif zarr_key in self._json_keys:
                if isinstance(data[k], np.ndarray):
                    data[k] = [self._decode_json_entry(v) for v in data[k]]
                else:
                    data[k] = self._decode_json_entry(data[k])

        if self.transform:
            for transform in self.transform:
                try:
                    data = transform.transform(data)
                except Exception:
                    origin = _fallback_origin if _fallback_origin is not None else idx
                    next_idx, attempts = get_fallback_idx(
                        idx=idx,
                        candidates=range(len(self)),
                        _attempts=_attempts,
                        max_attempts=len(self),
                        exhausted_error=(
                            f"Entire episode bad (no valid indices): ep={Path(self.episode_path).name}"
                        ),
                    )
                    # logger.warning(
                    #     f"Transform failed ep={Path(self.episode_path).name} frame={idx} ({type(e).__name__}: {e}) | "
                    #     f"attempt {attempts}, trying random idx {next_idx}"
                    # )
                    result = self.__getitem__(
                        next_idx, _fallback_origin=origin, _attempts=attempts
                    )
                    return result

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).to(torch.float32)

        data["metadata.robot_name"] = get_embodiment_id(self.embodiment)
        data["embodiment"] = get_embodiment_id(self.embodiment)
        return data


class ZarrAnnotationCutoffDataset(ZarrDataset):
    """ZarrDataset that clamps action chunks at the end of the enclosing annotation.

    Standard chunking from the start frame, but action reads stop at EOS+1 of the
    annotation span containing the start frame. The chunk is then padded out to
    ``horizon`` via the base ``_pad_sequences`` (repeat-last), so frames beyond
    EOS become the last action of the interval rather than crossing into the
    next annotation.

    If the start frame is not inside any annotation, behaves like the base class.
    """

    def __init__(self, *args, **kwargs):
        # Set before super().__init__ so it exists regardless of whether the
        # parent goes through init_episode (eager) or _init_from_metadata (lazy).
        self._frame_to_ann_end: dict[int, int] | None = None
        super().__init__(*args, **kwargs)

    def _build_frame_to_ann_end(self) -> dict[int, int]:
        """Map ``frame_idx -> ann_end`` (exclusive) for every frame inside an
        annotation span. Annotations use half-open ``[start_idx, end_idx)``.
        """
        mapping: dict[int, int] = {}
        for ann in self._load_annotations():
            start_idx = int(ann.get("start_idx", -1))
            end_idx = int(ann.get("end_idx", -1))
            if start_idx < 0 or end_idx <= start_idx:
                continue
            for idx in range(start_idx, end_idx):
                mapping[idx] = end_idx
        return mapping

    def _chunk_end_idx(self, start_idx: int, horizon: int, key_type: str | None) -> int:
        end_idx = super()._chunk_end_idx(start_idx, horizon, key_type)
        if key_type != "action_keys":
            return end_idx
        if self._frame_to_ann_end is None:
            self._frame_to_ann_end = self._build_frame_to_ann_end()
        ann_end = self._frame_to_ann_end.get(start_idx)
        if ann_end is None:
            return end_idx
        return min(end_idx, ann_end)


class S3AnnotationCutoffEpisodeResolver(S3EpisodeResolver):
    """S3EpisodeResolver that loads ZarrAnnotationCutoffDataset instances."""

    _dataset_class = ZarrAnnotationCutoffDataset


class LocalAnnotationCutoffEpisodeResolver(LocalEpisodeResolver):
    """LocalEpisodeResolver that loads ZarrAnnotationCutoffDataset instances."""

    _dataset_class = ZarrAnnotationCutoffDataset


class ZarrEpisode:
    """
    Lightweight wrapper around a single Zarr episode store.
    Designed for efficient PyTorch DataLoader usage with direct store access.
    """

    __slots__ = (
        "_path",
        "_store",
        "metadata",
        "keys",
    )

    def __init__(self, path: str | Path):
        """
        Initialize ZarrEpisode wrapper.
        Args:
            path: Path to the .zarr episode directory
        """
        self._path = Path(path)
        # Reuse fds + cache shard indices on the per-frame NFS read path. Idempotent,
        # env-gated, and runs in every fork/spawn worker since each opens its own
        # episodes. See egomimic.rldb.zarr.store_handle_cache.
        hc.ensure_enabled()
        self._store = zarr.open_group(str(self._path), mode="r")
        self.metadata = dict(self._store.attrs)
        self.keys = self.metadata["features"]

    def close(self) -> None:
        """Close the underlying zarr store to drop mmap mappings."""
        store = getattr(self, "_store", None)
        if store is not None:
            # Close cached fds + drop the shard-index cache so they don't
            # accumulate as the working set slides over a large dataset.
            try:
                hc.close_store(store)
            except Exception:
                pass
            try:
                store.close()
            except Exception:
                pass
            self._store = None

    def read(
        self, keys_with_ranges: dict[str, tuple[int, int | None]]
    ) -> dict[str, np.ndarray]:
        """
        Read data for specified keys, each with their own index or range.
        Args:
            keys_with_ranges: Dictionary mapping keys to (start, end) tuples.
                - start: Starting frame index
                - end: Ending frame index (exclusive). If None, reads single frame at start.
        Returns:
            Dictionary mapping keys to numpy arrays
        Example:
            >>> episode.read({
            ...     "obs/image": (0, 10),      # Read frames 0-10
            ...     "actions": (5, 15),        # Read frames 5-15
            ...     "rewards": (20, None),     # Read single frame at index 20
            ... })
        """
        result = {}
        for key, (start, end) in keys_with_ranges.items():
            arr = self._store[key]
            if end is not None:
                data = arr[start:end]
            else:
                # Single frame read - use slicing to avoid 0D array issues with VariableLengthBytes
                # arr[start:start+1] gives us a 1D array, then [0] extracts the actual object
                data = arr[start : start + 1][0]
            result[key] = data
        return result

    def _collect_keys(self) -> list[str]:
        """
        Collect all array keys from the store.
        Returns:
            List of array keys (flat structure with dot-separated names)
        """
        if isinstance(self.keys, dict):
            return list(self.keys.keys())
        return list(self.keys)

    def __len__(self) -> int:
        """
        Get total number of frames in the episode.
        Returns:
            Number of frames
        """
        return self.metadata["total_frames"]

    def __repr__(self) -> str:
        """String representation of the episode."""
        return f"ZarrEpisode(path={self._path}, frames={len(self)})"
