"""Multi-GPU offline val sweep over EVERY checkpoint of a live WAM run.

Evaluates every ``epoch_epoch=<N>.ckpt`` of ``data_div_oss/wam22_dw48_v2`` with
``egomimic.eval.eval_dreamzero`` (TF rolling) and folds the per-epoch metrics
into ONE W&B run whose x-axis is the checkpoint epoch.

Design
------
1. **Epoch enumeration runs remotely.** The ``egoverse-training-outputs`` volume
   is only mounted inside containers, so ``list_checkpoint_epochs`` (CPU only)
   does the ``ls`` + filename parse and returns ``[29, 59, ...]``. The source run
   is still training, so that list is a SNAPSHOT taken at sweep start — anything
   written after it is simply not in this sweep (re-run to pick it up).

2. **Sharding, not DDP.** The epochs are split across ``num_shards`` containers,
   each with ONE H200, and every eval is a plain single-process
   ``python -m egomimic.eval.eval_dreamzero`` subprocess. Never run DDP inside a
   shard: ``WAMEvalVideo`` writes its mp4s from rank 0 only, so a multi-rank
   container silently drops the episodes that landed on the other ranks. Modal
   caps concurrency at 4 containers via ``max_containers`` on ``run_shard``
   (Modal 1.x's name for the old ``concurrency_limit``).

3. **Aggregate, then log.** Shard containers write ``epoch_<N>.json`` metric
   files to the volume and log NOTHING to W&B — parallel writers into one run
   produce out-of-order/conflicting history. A final CPU-only container reads
   all the jsons, sorts by epoch, and logs them in ascending order into a single
   fresh run with ``epoch`` as the step metric.

Outputs on the volume (mounted at ``<repo>/logs``)
    logs/wam_val_sweep/<sweep_id>/epoch_<N>_<hydra ts>/videos/...   (per-epoch mp4s)
    logs/wam_val_sweep/<sweep_id>/metrics/epoch_<N>.json            (per-epoch metrics)
    logs/wam_val_sweep/<sweep_id>/summary.json                      (aggregated table)

Launch (Modal turns local-entrypoint params into ``--kebab-case`` CLI flags)
    MODAL_ENVIRONMENT=robotics modal run --detach \
        egomimic/modal/wam_val_sweep.py::sweep \
        --gpu-type H200 --num-shards 4 --num-val-episodes 2

    # non-default GPU: Modal 1.x has no Function.with_options, so the decorator
    # reads MODAL_GPU at import time — pass BOTH (they must agree):
    MODAL_GPU=B200 MODAL_ENVIRONMENT=robotics modal run --detach \
        egomimic/modal/wam_val_sweep.py::sweep --gpu-type B200

    # extra/overriding hydra args (appended to the fixed set, later wins):
    MODAL_ENVIRONMENT=robotics modal run --detach \
        egomimic/modal/wam_val_sweep.py::sweep \
        --hydra-args "evaluator=eval_dreamzero_ar seed=7"

    # just print the checkpoint snapshot:
    MODAL_ENVIRONMENT=robotics modal run \
        egomimic/modal/wam_val_sweep.py::list_epochs

    # re-log an interrupted / partially-finished sweep (no evals re-run):
    MODAL_ENVIRONMENT=robotics modal run \
        egomimic/modal/wam_val_sweep.py::aggregate --sweep-id 2026-08-18_11-22-33
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from pathlib import Path

import modal

# modal_setup.py lives next to this file locally (egomimic/modal/) and is baked
# into the image at /root/ so it is importable before the repo is cloned.
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from modal_setup import (  # noqa: E402
    CFG,
    WAN_CKPT_MOUNT,
    _local_wandb_key,
    _prepare_repo,
    _resolve_git_state,
    image,
    training_outputs_volume,
    wan_checkpoints_volume,
    zarr_volume,
)

app = modal.App("egomimic-wam-val-sweep", image=image)

# ---------------------------------------------------------------------------
# Sweep constants
# ---------------------------------------------------------------------------

# Run being swept. Its checkpoints/ + norm_stats/ live under the outputs volume.
SOURCE_RUN = "data_div_oss/wam22_dw48_v2"
DEFAULT_CHECKPOINTS_DIR = f"{CFG.output_mount_path}/{SOURCE_RUN}/checkpoints"

# hydra ``name=`` for every eval AND the volume prefix the sweep writes under, so
# the run dirs (videos) and the metrics dir sit side by side.
SWEEP_NAME = "wam_val_sweep"

# Concurrency cap. Fixed at decoration time (Modal 1.x dropped
# Function.with_options), so --num-shards > this only queues shards; it never
# runs more than this many H200s at once.
MAX_CONCURRENT_SHARDS = 4

# Per-shard GPU. Must be resolved at import time for the decorator; the
# ``--gpu-type`` flag is validated against it in the local entrypoints.
_SHARD_GPU = os.environ.get("MODAL_GPU", "H200")

DEFAULT_WANDB_PROJECT = "mecka-robotics"
DEFAULT_WANDB_ENTITY = "kevin_yam1_-mecka-ai"
DEFAULT_WANDB_RUN_NAME = "wam22_dw48_v2_valsweep"

# Fixed hydra args shared by every epoch's eval. Callers extend/override these
# through --hydra-args (see _merge_hydra_args).
#
# * norm_stats.precomputed_norm_path reuses the source run's cached stats —
#   without it each of the ~35 evals recomputes norm stats over the full 48h
#   train split (minutes of dataloading per checkpoint, all identical).
#   Relative -> _resolve_volume_paths prefixes the outputs-volume mount.
# * data=data_dishwashing_48h_wam already points the VALID split at
#   data_diversity/dishwashing_val_ophold5.json (the 5 held-out-OPERATOR
#   episodes) — that is the split this sweep scores. Only a caller-supplied
#   ``...eps_to_use=`` override in --hydra-args changes it.
# * batch_size=1 on both loaders: one val step per episode -> one mp4 pair per
#   episode (eval_dreamzero forces the valid side too, we set it explicitly so
#   the printed command is self-describing).
_BASE_HYDRA_ARGS: tuple[str, ...] = (
    "--config-name=train_zarr_human_wam_wan22_5b",
    "data=data_dishwashing_48h_wam",
    "evaluator=eval_dreamzero_tf",
    "data_schematic.norm_mode=minmax",
    "reject_outliers=false",
    f"norm_stats.precomputed_norm_path={SOURCE_RUN}/norm_stats/norm_stats.json",
    f"name={SWEEP_NAME}",
    "launch_params.gpus_per_node=1",
    "launch_params.nodes=1",
    "data.train_dataloader_params.mecka_bimanual.batch_size=1",
    "data.valid_dataloader_params.mecka_bimanual.batch_size=1",
)

# Hydra keys whose values are paths on the outputs volume; relative values get
# prefixed with the volume mount (same convention as offline_val_wam/trainModal).
_PATH_KEYS = {
    "ckpt_path",
    "checkpoints_dir",
    "norm_stats.precomputed_norm_path",
}

_COMMON = dict(
    cpu=12.0,
    # 160 GB (vs offline_val_wam's 128 GB): the WAM valid loader keeps 10
    # persistent workers holding 17-frame ~53MB clips, and a shard walks ~9
    # checkpoints back to back in ONE container, so /dev/shm + host RAM pressure
    # accumulates across evals. Under-provisioning this is what OOM/SIGBUS-killed
    # the wam22_dw48 training run around epoch 21.
    memory=163840,
    # 24 h (Modal max): a shard runs len(epochs)/num_shards evals sequentially,
    # each re-loading a 5B checkpoint from the volume.
    timeout=86400,
    secrets=[modal.Secret.from_name(name) for name in CFG.secret_names],
    volumes={
        CFG.volume_mount_path: zarr_volume,
        CFG.output_mount_path: training_outputs_volume,
        WAN_CKPT_MOUNT: wan_checkpoints_volume,
    },
)

_LIGHT = dict(
    cpu=2.0,
    memory=8192,
    volumes={CFG.output_mount_path: training_outputs_volume},
)


# ---------------------------------------------------------------------------
# Hydra-arg helpers
# ---------------------------------------------------------------------------


def _resolve_volume_paths(hydra_args: list[str]) -> list[str]:
    fixed = []
    for arg in hydra_args:
        key, sep, val = arg.partition("=")
        if (
            sep
            and key.lstrip("+") in _PATH_KEYS
            and val
            and val != "null"
            and not val.startswith("/")
        ):
            arg = f"{key}={CFG.output_mount_path}/{val}"
        fixed.append(arg)
    return fixed


def _split_args(hydra_args: str) -> list[str]:
    """Shell-split the --hydra-args passthrough (empty is fine — unlike
    offline_val_wam this script builds a complete arg set itself)."""
    import shlex

    return shlex.split(hydra_args or "")


def _override_key(arg: str) -> str:
    """Dedup key for a hydra arg: everything left of the first ``=``, with any
    leading ``+`` stripped so ``+num_val_episodes`` and ``num_val_episodes``
    collide (hydra rejects a ``+`` add for a key that already exists)."""
    return arg.partition("=")[0].lstrip("+")


def _merge_hydra_args(base: tuple[str, ...] | list[str], extra: list[str]) -> list[str]:
    """Merge arg lists so later entries REPLACE earlier ones for the same key.

    Hydra composition is unhappy about the same key appearing twice, so the
    passthrough overrides are folded in by key rather than appended blindly.
    """
    merged: dict[str, str] = {}
    for arg in (*base, *extra):
        merged[_override_key(arg)] = arg
    return list(merged.values())


def _epoch_hydra_args(
    epoch: int,
    sweep_id: str,
    base_args: list[str],
    checkpoints_dir: str,
    num_val_episodes: int,
) -> list[str]:
    """Return the full hydra argv for one checkpoint's eval."""
    # ckpt_path must be ABSOLUTE (_resolve_volume_paths would otherwise prefix
    # the volume mount) and the '=' inside "epoch_epoch=<N>.ckpt" must be
    # escaped, because hydra's override grammar treats a bare '=' in a value as
    # a syntax error. THIS CASE: the arg list is built here and handed straight
    # to subprocess.run() as a list — no shell, and it is never shlex-split — so
    # exactly ONE backslash must reach hydra, i.e. "\\=" in this source line.
    # (The doubled-backslash form is only needed for args that survive a
    # shlex.split round-trip, which these do not.)
    ckpt_arg = f"{checkpoints_dir}/epoch_epoch\\={epoch}.ckpt"
    metrics_path = (
        f"{CFG.output_mount_path}/{SWEEP_NAME}/{sweep_id}/metrics/epoch_{epoch}.json"
    )
    epoch_args = [
        f"ckpt_path={ckpt_arg}",
        # description carries the sweep id + epoch, so hydra's run dir becomes
        # logs/wam_val_sweep/<sweep_id>/epoch_<N>_<timestamp>/ (videos land there).
        f"description={sweep_id}/epoch_{epoch}",
        f"+metrics_out={metrics_path}",
    ]
    if not any(_override_key(a) == "num_val_episodes" for a in base_args):
        epoch_args.append(f"+num_val_episodes={num_val_episodes}")
    # Per-epoch args win over anything the caller passed through.
    return _merge_hydra_args(base_args, epoch_args)


def ckpt_file_for_epoch(checkpoints_dir: str, epoch: int) -> str:
    """On-disk (UNescaped) checkpoint path for *epoch*."""
    return f"{checkpoints_dir}/epoch_epoch={epoch}.ckpt"


def make_shards(epochs: list[int], num_shards: int, shard_mode: str) -> list[list[int]]:
    """Split *epochs* into at most *num_shards* work lists.

    ``roundrobin`` (default) interleaves, so every shard spans the whole training
    timeline — if one container is preempted the surviving curve still covers
    early→late epochs instead of losing a contiguous block. ``contiguous`` keeps
    neighbouring epochs together.
    """
    if num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")
    if shard_mode == "roundrobin":
        shards = [epochs[i::num_shards] for i in range(num_shards)]
    elif shard_mode == "contiguous":
        size = max(1, math.ceil(len(epochs) / num_shards))
        shards = [epochs[i : i + size] for i in range(0, len(epochs), size)]
    else:
        raise SystemExit(
            f"unknown --shard-mode {shard_mode!r} "
            "(expected 'roundrobin' or 'contiguous')"
        )
    return [s for s in shards if s]


def _check_gpu_type(gpu_type: str) -> None:
    """Modal 1.x fixes the GPU at decoration time (no Function.with_options), so
    --gpu-type can only confirm what MODAL_GPU already selected."""
    if gpu_type != _SHARD_GPU:
        raise SystemExit(
            f"--gpu-type {gpu_type!r} != the decorated GPU {_SHARD_GPU!r}. "
            f"Modal 1.x cannot change a function's GPU per call — relaunch with "
            f"MODAL_GPU={gpu_type} in the environment."
        )


# ---------------------------------------------------------------------------
# Remote: checkpoint enumeration
# ---------------------------------------------------------------------------


@app.function(timeout=600, **_LIGHT)
def list_checkpoint_epochs(checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR) -> list[int]:
    """Sorted epochs of every ``epoch_epoch=<N>.ckpt`` in *checkpoints_dir*.

    ``last.ckpt`` and ``modal_auto_restart.ckpt`` are skipped: both are moving
    aliases rewritten by the live run, not stable points in time.
    """
    import re

    # The source run is still writing; reload so this container sees the newest
    # committed checkpoints rather than its mount-time snapshot.
    training_outputs_volume.reload()

    if not os.path.isdir(checkpoints_dir):
        raise FileNotFoundError(
            f"checkpoints dir not found on volume: {checkpoints_dir}"
        )

    epochs: list[int] = []
    for fname in os.listdir(checkpoints_dir):
        match = re.fullmatch(r"epoch_epoch=(\d+)\.ckpt", fname)
        if match:
            epochs.append(int(match.group(1)))
    epochs.sort()
    print(f"[wam_val_sweep] {len(epochs)} checkpoints in {checkpoints_dir}: {epochs}")
    return epochs


# ---------------------------------------------------------------------------
# Remote: one shard of epochs, evaluated sequentially on one GPU
# ---------------------------------------------------------------------------


@app.function(gpu=_SHARD_GPU, max_containers=MAX_CONCURRENT_SHARDS, **_COMMON)
def run_shard(
    shard: list[int],
    sweep_id: str,
    git_remote: str,
    git_commit: str,
    checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR,
    extra_hydra_args: tuple[str, ...] = (),
    num_val_episodes: int = 2,
    wandb_api_key: str = "",
) -> list[dict]:
    """Run ``eval_dreamzero`` once per epoch in *shard*, sequentially, 1 GPU.

    One process per eval — no DDP (see module docstring: the video evaluator
    writes mp4s from rank 0 only). Returns one result dict per epoch instead of
    raising, so a single bad checkpoint neither aborts its shard's remaining
    epochs nor hides the per-epoch detail behind a map-level exception.
    """
    import shlex
    import subprocess

    # See the newest committed checkpoints of the live source run, then clone the
    # repo ONCE per container (not once per epoch — it is the same commit).
    training_outputs_volume.reload()
    _prepare_repo(git_remote=git_remote, git_commit=git_commit)

    metrics_dir = f"{CFG.output_mount_path}/{SWEEP_NAME}/{sweep_id}/metrics"
    os.makedirs(metrics_dir, exist_ok=True)

    base_args = _merge_hydra_args(_BASE_HYDRA_ARGS, list(extra_hydra_args))

    env = os.environ.copy()
    env["MODAL_IS_REMOTE"] = "1"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    # Prevent thread explosion: many DataLoader workers x many CPU cores.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if wandb_api_key:
        env["WANDB_API_KEY"] = wandb_api_key
        env.setdefault("WANDB_START_METHOD", "thread")

    results: list[dict] = []
    print(f"[wam_val_sweep] shard of {len(shard)} epochs: {shard}")
    for epoch in shard:
        ckpt_file = ckpt_file_for_epoch(checkpoints_dir, epoch)
        metrics_path = f"{metrics_dir}/epoch_{epoch}.json"
        if not os.path.exists(ckpt_file):
            print(f"[wam_val_sweep] epoch {epoch}: MISSING {ckpt_file} — skipped")
            results.append(
                {"epoch": epoch, "returncode": -1, "metrics_path": metrics_path}
            )
            continue

        hydra_args = _resolve_volume_paths(
            _epoch_hydra_args(
                epoch, sweep_id, base_args, checkpoints_dir, num_val_episodes
            )
        )
        cmd = [CFG.python_bin, "-m", "egomimic.eval.eval_dreamzero", *hydra_args]
        # shlex.join is display only — cmd is passed as a list, so the single
        # backslash before '=' in ckpt_path reaches hydra verbatim.
        print(f"[wam_val_sweep] epoch {epoch}: {shlex.join(cmd)}")
        proc = subprocess.run(cmd, cwd=CFG.remote_repo_dir, env=env, check=False)

        # Commit after EVERY epoch: videos + metrics of finished epochs survive a
        # preemption or a later epoch crashing mid-shard.
        training_outputs_volume.commit()
        results.append(
            {
                "epoch": epoch,
                "returncode": proc.returncode,
                "metrics_path": metrics_path,
            }
        )
        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"[wam_val_sweep] epoch {epoch}: {status}; outputs committed")

    failed = [r["epoch"] for r in results if r["returncode"] != 0]
    print(
        f"[wam_val_sweep] shard done: {len(results) - len(failed)}/{len(results)} ok"
        + (f", failed epochs: {failed}" if failed else "")
    )
    return results


# ---------------------------------------------------------------------------
# Remote: aggregate the per-epoch metric jsons into ONE W&B run
# ---------------------------------------------------------------------------


@app.function(timeout=1800, **_LIGHT)
def aggregate_to_wandb(
    sweep_id: str,
    wandb_api_key: str = "",
    project: str = DEFAULT_WANDB_PROJECT,
    entity: str = DEFAULT_WANDB_ENTITY,
    run_name: str = DEFAULT_WANDB_RUN_NAME,
    source_run: str = SOURCE_RUN,
    checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR,
    epochs: list[int] | None = None,
) -> str:
    """Log every ``metrics/epoch_*.json`` of *sweep_id* to one new W&B run.

    Runs only after the shard map finishes: a single writer logging in ascending
    epoch order is what keeps the run's history monotonic in ``epoch``.
    Returns the W&B run URL ("" when no API key was available).
    """
    import glob
    import json
    import re

    # Pick up everything the shard containers committed.
    training_outputs_volume.reload()

    sweep_dir = f"{CFG.output_mount_path}/{SWEEP_NAME}/{sweep_id}"
    metrics_dir = f"{sweep_dir}/metrics"
    rows: list[dict] = []
    for path in glob.glob(f"{metrics_dir}/epoch_*.json"):
        match = re.search(r"epoch_(\d+)\.json$", os.path.basename(path))
        if not match:
            continue
        with open(path) as f:
            payload = json.load(f)
        # Every key the evaluator emitted is logged — no whitelist. Numeric-ish
        # values are coerced to float so W&B charts them; anything else is passed
        # through as-is. The filename is authoritative for the epoch.
        row: dict = {}
        for key, val in payload.items():
            if key == "epoch":
                continue
            try:
                row[key] = float(val)
            except (TypeError, ValueError):
                row[key] = val
        row["epoch"] = int(match.group(1))
        rows.append(row)

    rows.sort(key=lambda r: r["epoch"])
    if not rows:
        raise FileNotFoundError(f"no epoch_*.json metrics found under {metrics_dir}")

    metric_keys = sorted({k for r in rows for k in r if k != "epoch"})
    print(
        f"[wam_val_sweep] aggregating {len(rows)} epochs "
        f"({[r['epoch'] for r in rows]}) x {len(metric_keys)} metrics: {metric_keys}"
    )

    run_url = ""
    run_id = ""
    if wandb_api_key:
        import wandb

        os.environ["WANDB_API_KEY"] = wandb_api_key
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=["wam-val-sweep"],
            config={
                "sweep_id": sweep_id,
                "source_run": source_run,
                "checkpoints_dir": checkpoints_dir,
                "checkpoint_epochs": epochs if epochs is not None else [],
                "evaluated_epochs": [r["epoch"] for r in rows],
                "num_checkpoints_evaluated": len(rows),
                "metric_keys": metric_keys,
            },
        )
        # Checkpoint epoch is the x-axis for every metric (W&B's default step is
        # a log counter, and the source run's global_step restarts per ckpt load).
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        for row in rows:
            wandb.log(row)
        run_url = run.url or ""
        run_id = run.id
        wandb.finish()
    else:
        print(
            "[wam_val_sweep] WANDB_API_KEY missing — skipped W&B logging "
            "(summary.json still written)."
        )

    summary_path = f"{sweep_dir}/summary.json"
    os.makedirs(sweep_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "sweep_id": sweep_id,
                "source_run": source_run,
                "checkpoints_dir": checkpoints_dir,
                "checkpoint_epochs": epochs if epochs is not None else [],
                "metric_keys": metric_keys,
                "wandb": {
                    "project": project,
                    "entity": entity,
                    "name": run_name,
                    "id": run_id,
                    "url": run_url,
                },
                "rows": rows,
            },
            f,
            indent=2,
        )
    training_outputs_volume.commit()
    print(f"[wam_val_sweep] wrote {summary_path}")
    if run_url:
        print(f"[wam_val_sweep] W&B run: {run_url}")
    return run_url


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def list_epochs(checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR) -> None:
    """Print the checkpoint-epoch snapshot of the source run."""
    epochs = list_checkpoint_epochs.remote(checkpoints_dir)
    print(f"{len(epochs)} checkpoints in {checkpoints_dir}")
    print(f"epochs: {epochs}")


@app.local_entrypoint()
def sweep(
    gpu_type: str = "H200",
    num_shards: int = 4,
    shard_mode: str = "roundrobin",
    num_val_episodes: int = 2,
    checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR,
    hydra_args: str = "",
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    wandb_run_name: str = DEFAULT_WANDB_RUN_NAME,
) -> None:
    """Evaluate every checkpoint of the source run, then log one W&B curve.

    Launch with ``modal run --detach`` so the shard containers (and the metric
    files they commit epoch by epoch) survive a local disconnect. If the local
    driver does die mid-sweep, nothing is lost: re-run ``::aggregate`` with the
    printed sweep id to log whatever finished.
    """
    _check_gpu_type(gpu_type)

    git_remote, git_commit, is_dirty = _resolve_git_state()
    if is_dirty:
        print("Warning: local repo dirty; the containers run the last pushed commit.")

    epochs = list_checkpoint_epochs.remote(checkpoints_dir)
    if not epochs:
        raise SystemExit(f"no epoch_epoch=*.ckpt found in {checkpoints_dir}")
    # Snapshot: the source run is live and may write more checkpoints while this
    # sweep is in flight — those are not evaluated here.
    print(f"Checkpoint snapshot ({len(epochs)}): {epochs}")

    shards = make_shards(epochs, num_shards, shard_mode)
    if len(shards) > MAX_CONCURRENT_SHARDS:
        print(
            f"Note: {len(shards)} shards > max_containers={MAX_CONCURRENT_SHARDS} — "
            "extra shards queue until a container frees up."
        )
    print(f"Sharding {len(epochs)} epochs into {len(shards)} shards ({shard_mode}):")
    for idx, shard in enumerate(shards):
        print(f"  shard {idx}: {len(shard)} epochs -> {shard}")

    # Same timestamp format hydra's run dir uses in train_zarr_cartesian.yaml.
    sweep_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra = tuple(_split_args(hydra_args))
    wandb_key = _local_wandb_key()
    print(
        f"sweep_id={sweep_id}  gpu={_SHARD_GPU}  commit={git_commit[:12]}  "
        f"num_val_episodes={num_val_episodes}"
    )
    if extra:
        print(f"extra hydra args: {list(extra)}")
    print(f"metrics -> {SWEEP_NAME}/{sweep_id}/metrics/epoch_<N>.json (volume)")

    results = list(
        run_shard.map(
            shards,
            kwargs=dict(
                sweep_id=sweep_id,
                git_remote=git_remote,
                git_commit=git_commit,
                checkpoints_dir=checkpoints_dir,
                extra_hydra_args=extra,
                num_val_episodes=num_val_episodes,
                # Passed only so the eval subprocess has a key if a config ever
                # enables a logger; the sweep itself logs from the aggregator.
                wandb_api_key=wandb_key,
            ),
            return_exceptions=True,
        )
    )

    ok, failed = 0, []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"shard {idx} raised: {result}")
            continue
        for entry in result:
            if entry["returncode"] == 0:
                ok += 1
            else:
                failed.append(entry["epoch"])
    print(
        f"Evals finished: {ok} ok"
        + (f", failed epochs: {sorted(failed)}" if failed else "")
    )

    aggregate_to_wandb.remote(
        sweep_id,
        wandb_api_key=wandb_key,
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        checkpoints_dir=checkpoints_dir,
        epochs=epochs,
    )
    print(f"Done. Re-log any time with: ::aggregate --sweep-id {sweep_id}")


@app.local_entrypoint()
def aggregate(
    sweep_id: str,
    checkpoints_dir: str = DEFAULT_CHECKPOINTS_DIR,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str = DEFAULT_WANDB_ENTITY,
    wandb_run_name: str = DEFAULT_WANDB_RUN_NAME,
) -> None:
    """Aggregate an existing sweep's metric jsons into one W&B run (no evals)."""
    aggregate_to_wandb.remote(
        sweep_id,
        wandb_api_key=_local_wandb_key(),
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        checkpoints_dir=checkpoints_dir,
    )
