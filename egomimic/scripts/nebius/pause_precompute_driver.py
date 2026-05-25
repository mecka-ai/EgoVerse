"""SLURM/Nebius driver for pause-filter precompute.

Mirrors the Modal-based ``_precompute_pause_to_cache`` fan-out, but submits
a SLURM array job instead of dispatching Modal containers. Designed to run
on the cluster's login node before training.

Pipeline
--------
1. Hydra-compose the training config and walk ``train_datasets`` +
   ``valid_datasets`` to find every resolver with ``pause_removal_epsilon``.
2. For each resolver, call ``discover_episode_paths(filters)`` (added to
   ModalEpisodeResolver for this purpose) to enumerate ``[(hash, path)]``
   without building ZarrDatasets and without triggering in-process precompute.
3. Group by ``epsilon``, partition into N shards, write ``manifest.jsonl``.
4. ``sbatch --array=0-(N-1)%C pause_precompute.sbatch <manifest> <out-dir>``
   — one array task = one shard = one ``shard_<i>.json``.
5. Poll ``sacct`` (or ``squeue``) until the array finishes.
6. Merge per-shard JSONs into ``cache.json``.

The training process is unchanged: set
``EGOMIMIC_PAUSE_PRECOMPUTE_CACHE=<out-dir>/cache.json`` before launching
``trainHydra.py``. The existing ``_apply_pause_precompute_cache`` consumer
reads this file and populates ``keep_indices`` on each ZarrDataset.
Action-chunk reads in ``ZarrDataset.__getitem__`` already fancy-index
``keep_indices``, so chunks contain only filtered frames as well.

Usage
-----
    python -m egomimic.scripts.nebius.pause_precompute_driver \\
        --config-name train_zarr_cartesian \\
        --overrides data=mecka_50k_20k \\
        --out-dir /shared/pause/run-$(date +%Y%m%d_%H%M%S) \\
        [--shards 100] [--concurrency 50] \\
        [--partition cpu] [--time 00:30:00] [--mem 16G] [--cpus-per-task 8] \\
        [--dry-run] [--no-wait]

After completion::

    export EGOMIMIC_PAUSE_PRECOMPUTE_CACHE=/shared/pause/run-.../cache.json
    sbatch your-training-job.sbatch
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / "egomimic" / "hydra_configs"
SBATCH_TEMPLATE = Path(__file__).with_name("pause_precompute.sbatch")


def _hydra_compose(config_name: str, overrides: list[str]):
    from hydra import compose, initialize_config_dir

    if not CONFIG_DIR.is_dir():
        raise FileNotFoundError(f"Hydra config dir missing: {CONFIG_DIR}")
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


def _discover_work(cfg) -> dict[float, dict[str, str]]:
    """Walk train_datasets + valid_datasets → {epsilon: {hash: path}}.

    Dedups across train/valid: same episode_hash + same epsilon = same work.
    """
    from hydra.utils import instantiate

    from egomimic.rldb.zarr.zarr_dataset_multi import ModalEpisodeResolver

    work_by_eps: dict[float, dict[str, str]] = {}
    seen_blocks = 0
    for block_name in ("train_datasets", "valid_datasets"):
        block = cfg.data.get(block_name)
        if block is None:
            continue
        for ds_name, ds_cfg in block.items():
            resolver_cfg = ds_cfg.get("resolver")
            if resolver_cfg is None:
                continue
            seen_blocks += 1
            try:
                resolver = instantiate(resolver_cfg)
            except Exception as e:
                print(
                    f"[pause-precompute] {block_name}.{ds_name}: resolver "
                    f"instantiation failed: {e}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(resolver, ModalEpisodeResolver):
                continue
            eps = resolver.pause_removal_epsilon
            if eps is None:
                continue

            filters_cfg = ds_cfg.get("filters")
            filters = instantiate(filters_cfg) if filters_cfg is not None else None

            try:
                pairs = resolver.discover_episode_paths(filters)
            except Exception as e:
                print(
                    f"[pause-precompute] {block_name}.{ds_name}: "
                    f"discover_episode_paths failed: {e}",
                    file=sys.stderr,
                )
                continue

            bucket = work_by_eps.setdefault(float(eps), {})
            for episode_hash, local_path in pairs:
                bucket[episode_hash] = local_path
            print(
                f"[pause-precompute] {block_name}.{ds_name}: "
                f"epsilon={eps}, {len(pairs)} episodes resolved"
            )

    print(f"[pause-precompute] walked {seen_blocks} dataset block(s)")
    return work_by_eps


def _write_manifest(
    work_by_eps: dict[float, dict[str, str]],
    n_shards: int,
    manifest_path: Path,
) -> int:
    """Partition work into shards (round-robin per epsilon) and emit JSONL.

    Returns the actual number of shards written (may be < n_shards if total
    episodes is small or some shards end up empty).
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[dict[str, Any]] = []
    for eps, hash_to_path in work_by_eps.items():
        episodes = list(hash_to_path.items())
        if not episodes:
            continue
        # Round-robin so episodes with similar paths/storage layout get
        # spread across shards rather than clumped — keeps tail latency tight.
        eps_shards = min(n_shards, len(episodes))
        for i in range(eps_shards):
            shard_episodes = episodes[i::eps_shards]
            if not shard_episodes:
                continue
            lines.append(
                {
                    "shard_id": len(lines),
                    "epsilon": float(eps),
                    "episodes": [[h, p] for h, p in shard_episodes],
                }
            )

    with manifest_path.open("w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    print(
        f"[pause-precompute] wrote manifest with {len(lines)} shards → {manifest_path}"
    )
    return len(lines)


def _sbatch_submit(
    *,
    sbatch_template: Path,
    manifest_path: Path,
    out_dir: Path,
    n_shards: int,
    concurrency: int,
    partition: str | None,
    time_limit: str | None,
    mem: str | None,
    cpus_per_task: int | None,
    job_name: str,
    extra_sbatch: list[str],
) -> str:
    """Submit the array job. Returns the SLURM job id (e.g. ``"123456"``).

    Builds sbatch CLI overrides for ``--array``, ``--partition``, etc. so the
    template only carries defaults. ``extra_sbatch`` is passed through verbatim
    (e.g. ``["--account=foo", "--qos=normal"]``).
    """
    cmd: list[str] = [
        "sbatch",
        "--parsable",
        f"--job-name={job_name}",
        f"--array=0-{n_shards - 1}%{concurrency}",
    ]
    if partition:
        cmd += [f"--partition={partition}"]
    if time_limit:
        cmd += [f"--time={time_limit}"]
    if mem:
        cmd += [f"--mem={mem}"]
    if cpus_per_task is not None:
        cmd += [f"--cpus-per-task={cpus_per_task}"]
    cmd += list(extra_sbatch)
    cmd += [
        str(sbatch_template),
        str(manifest_path),
        str(out_dir),
    ]

    print(f"[pause-precompute] submitting: {shlex.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = proc.stdout.strip().split(";")[0]  # --parsable: "jobid[;cluster]"
    print(f"[pause-precompute] submitted array job {job_id}")
    return job_id


def _poll_until_done(job_id: str, poll_seconds: int = 30) -> dict[str, int]:
    """Poll ``sacct`` until no tasks remain in a pending/running state.

    Returns counts ``{"COMPLETED": x, "FAILED": y, "TIMEOUT": z, ...}``
    over the array tasks (the parent + steps). Loops indefinitely — the user
    can ctrl-C; the array continues running and re-running the driver is safe.
    """
    pending_states = {"PENDING", "RUNNING", "REQUEUED", "RESIZING", "SUSPENDED"}
    while True:
        proc = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--format=JobID,State",
                "--parsable2",
                "--noheader",
                "-X",  # parent job only — one row per array task
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(
                f"[pause-precompute] sacct failed: {proc.stderr.strip()}; "
                f"falling back to squeue check",
                file=sys.stderr,
            )
            squeue = subprocess.run(
                ["squeue", "-j", job_id, "-h"],
                capture_output=True,
                text=True,
            )
            if squeue.returncode == 0 and not squeue.stdout.strip():
                # Nothing queued/running anymore → assume done.
                return {"COMPLETED": -1}
            time.sleep(poll_seconds)
            continue

        counts: dict[str, int] = {}
        any_pending = False
        for row in proc.stdout.strip().splitlines():
            try:
                _jid, state = row.split("|", 1)
            except ValueError:
                continue
            state = state.split()[0].strip()  # strip "CANCELLED by NN"
            counts[state] = counts.get(state, 0) + 1
            if state in pending_states:
                any_pending = True

        status = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"[pause-precompute] job {job_id}: {status}")
        if not any_pending and counts:
            return counts
        time.sleep(poll_seconds)


def _aggregate(out_dir: Path) -> Path:
    """Merge ``<out_dir>/shards/shard_*.json`` into ``<out_dir>/cache.json``.

    Last-write-wins on conflicting hashes (shouldn't happen unless two
    epsilons cover the same episode; the original Modal flow had the same
    semantics).
    """
    shards_dir = out_dir / "shards"
    cache: dict[str, dict] = {}
    files = sorted(shards_dir.glob("shard_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No shard outputs under {shards_dir} — array job produced nothing"
        )
    for f in files:
        try:
            with f.open() as fh:
                cache.update(json.load(fh))
        except Exception as e:
            print(
                f"[pause-precompute] WARN: failed to load {f.name}: {e}",
                file=sys.stderr,
            )

    cache_path = out_dir / "cache.json"
    tmp = cache_path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(cache, f)
    tmp.replace(cache_path)

    n_total = sum(v["raw_total"] for v in cache.values())
    n_kept = sum(len(v["keep_indices"]) for v in cache.values())
    n_miss = sum(1 for v in cache.values() if v["raw_total"] == 0)
    pct = (100.0 * n_kept / n_total) if n_total else 100.0
    print(
        f"[pause-precompute] aggregated {len(cache)} episodes from "
        f"{len(files)} shards: kept {n_kept}/{n_total} ({pct:.1f}%), "
        f"misses={n_miss} → {cache_path}"
    )
    return cache_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config-name",
        required=True,
        help="Hydra config name (e.g. train_zarr_cartesian)",
    )
    p.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Hydra overrides (e.g. data=mecka_50k_20k model=foo)",
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--shards",
        type=int,
        default=int(os.environ.get("EGOMIMIC_PAUSE_PRECOMPUTE_SHARDS", "100")),
    )
    p.add_argument(
        "--concurrency", type=int, default=50, help="sbatch --array %% throttle"
    )
    p.add_argument("--partition", default=None)
    p.add_argument("--time", dest="time_limit", default="00:30:00")
    p.add_argument("--mem", default="16G")
    p.add_argument("--cpus-per-task", type=int, default=8)
    p.add_argument("--job-name", default="pause-precompute")
    p.add_argument(
        "--sbatch-template",
        type=Path,
        default=SBATCH_TEMPLATE,
        help="Override the sbatch script (defaults to bundled template).",
    )
    p.add_argument(
        "--extra-sbatch",
        nargs=argparse.REMAINDER,
        default=[],
        help="Pass-through sbatch flags (use after --: -- --account=foo --qos=normal)",
    )
    p.add_argument(
        "--no-wait", action="store_true", help="Submit and exit without polling."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build manifest + print sbatch command, but don't submit.",
    )
    p.add_argument("--poll-seconds", type=int, default=30)
    args = p.parse_args(argv)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _hydra_compose(args.config_name, list(args.overrides))
    work_by_eps = _discover_work(cfg)
    if not work_by_eps:
        print(
            "[pause-precompute] no resolvers with pause_removal_epsilon found; nothing to do",
            file=sys.stderr,
        )
        return 1

    total_eps = sum(len(v) for v in work_by_eps.values())
    print(
        f"[pause-precompute] {total_eps} episodes across {len(work_by_eps)} epsilon group(s): "
        + ", ".join(f"eps={e}({len(v)})" for e, v in work_by_eps.items())
    )

    manifest_path = out_dir / "manifest.jsonl"
    n_shards = _write_manifest(work_by_eps, args.shards, manifest_path)

    if args.dry_run:
        # Show the sbatch command that would run
        preview = [
            "sbatch",
            "--parsable",
            f"--job-name={args.job_name}",
            f"--array=0-{n_shards - 1}%{args.concurrency}",
        ]
        if args.partition:
            preview.append(f"--partition={args.partition}")
        preview += [
            f"--time={args.time_limit}",
            f"--mem={args.mem}",
            f"--cpus-per-task={args.cpus_per_task}",
            *list(args.extra_sbatch),
            str(args.sbatch_template),
            str(manifest_path),
            str(out_dir),
        ]
        print(f"[pause-precompute] DRY-RUN sbatch command:\n  {shlex.join(preview)}")
        return 0

    job_id = _sbatch_submit(
        sbatch_template=args.sbatch_template,
        manifest_path=manifest_path,
        out_dir=out_dir,
        n_shards=n_shards,
        concurrency=args.concurrency,
        partition=args.partition,
        time_limit=args.time_limit,
        mem=args.mem,
        cpus_per_task=args.cpus_per_task,
        job_name=args.job_name,
        extra_sbatch=list(args.extra_sbatch),
    )

    # Stash job id alongside outputs for later inspection
    (out_dir / "slurm_job_id").write_text(job_id + "\n")

    if args.no_wait:
        print(
            f"[pause-precompute] submitted job {job_id}. Re-run with --aggregate-only "
            "(or just `python -c 'from egomimic.scripts.nebius.pause_precompute_driver "
            f"import _aggregate; _aggregate(Path({str(out_dir)!r}))'`) once it completes."
        )
        return 0

    counts = _poll_until_done(job_id, poll_seconds=args.poll_seconds)
    failed = sum(
        v
        for k, v in counts.items()
        if k not in {"COMPLETED", "CANCELLED"} and not re.match(r"^COMPLETED", k)
    )
    if failed:
        print(
            f"[pause-precompute] WARNING: {failed} array tasks did not complete cleanly. "
            "Inspect slurm-*.out files. Aggregation will still run on whatever shards "
            "did write.",
            file=sys.stderr,
        )

    cache_path = _aggregate(out_dir)
    print(f"\n=== DONE ===\nexport EGOMIMIC_PAUSE_PRECOMPUTE_CACHE={cache_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
