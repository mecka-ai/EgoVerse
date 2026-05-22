#!/usr/bin/env python3
"""EgoVerse Modal run manager.

Monitors the local logs/ directory produced by trainHydra.py (via Hydra).
Each subdirectory under logs/<group>/<name>_<timestamp>/ is a run.

Usage
-----
# Print status of all runs
python egomimic/modal/run_manager.py

# Watch logs/ continuously (re-scans every N seconds)
python egomimic/modal/run_manager.py --watch [--interval 30]

# Delete logs for failed/stale runs (interactive confirmation)
python egomimic/modal/run_manager.py --clean

# Auto-fix known Modal error patterns and re-submit
python egomimic/modal/run_manager.py --fix [--dry-run]
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = REPO_ROOT / "logs"
LOG_FILE_NAME = "trainHydra.log"


# ---------------------------------------------------------------------------
# Error patterns (ordered: most-specific first)
# ---------------------------------------------------------------------------

class FailureKind(str, Enum):
    CUDA_OOM       = "cuda_oom"
    MODAL_TIMEOUT  = "modal_timeout"
    GIT_PUSH       = "git_push_failed"
    SECRET_MISSING = "secret_missing"
    VOLUME_MOUNT   = "volume_mount"
    ZARR_MISSING   = "zarr_missing"
    NORM_STATS     = "norm_stats_missing"
    CHECKPOINT     = "checkpoint_missing"
    HYDRA_CONFIG   = "hydra_config_error"
    IMPORT_ERROR   = "import_error"
    CUDA_INIT      = "cuda_init"
    GENERIC        = "generic_error"


_PATTERNS: list[tuple[re.Pattern, FailureKind]] = [
    (re.compile(r"CUDA out of memory", re.I),                      FailureKind.CUDA_OOM),
    (re.compile(r"torch\.cuda\.OutOfMemoryError", re.I),           FailureKind.CUDA_OOM),
    (re.compile(r"(Modal.*timeout|container.*timed out)", re.I),   FailureKind.MODAL_TIMEOUT),
    (re.compile(r"git push failed", re.I),                         FailureKind.GIT_PUSH),
    (re.compile(r"secret.*not found|MISSING.*secret", re.I),       FailureKind.SECRET_MISSING),
    (re.compile(r"volume.*mount|mount.*failed", re.I),             FailureKind.VOLUME_MOUNT),
    (re.compile(r"zarr.*not found|No such zarr", re.I),            FailureKind.ZARR_MISSING),
    (re.compile(r"(precomputed_norm_path|norm_stats).*not.*found|FileNotFoundError.*norm", re.I),
                                                                    FailureKind.NORM_STATS),
    (re.compile(r"(ckpt_path|checkpoint).*not.*found|FileNotFoundError.*\.ckpt", re.I),
                                                                    FailureKind.CHECKPOINT),
    (re.compile(r"(hydra|omegaconf).*(error|missing|not found)", re.I),
                                                                    FailureKind.HYDRA_CONFIG),
    (re.compile(r"ImportError|ModuleNotFoundError", re.I),         FailureKind.IMPORT_ERROR),
    (re.compile(r"cuda.*driver|CUDA driver", re.I),                FailureKind.CUDA_INIT),
    (re.compile(r"(Traceback|ERROR|Exception|Error:)", re.M),      FailureKind.GENERIC),
]

# Lines that look like errors but are not failures
_NOISE_PATTERNS = [
    re.compile(r"\[rank: 0\] Extras config not found", re.I),
    re.compile(r"UserWarning", re.I),
    re.compile(r"FutureWarning", re.I),
    re.compile(r"DeprecationWarning", re.I),
]

# Lines that indicate a successful run
_SUCCESS_PATTERNS = [
    re.compile(r"`Trainer.fit` stopped", re.I),
    re.compile(r"Submitted Modal job:", re.I),
    re.compile(r"Training finished", re.I),
    re.compile(r"best_model_path", re.I),
]

# Lines that carry useful metrics
_METRIC_PATTERNS = [
    re.compile(r"(val/|train/|valid/)[\w/]+=[\d.eE+\-]+", re.I),
    re.compile(r"Epoch \d+", re.I),
    re.compile(r"loss[\s=:]+[\d.eE+\-]+", re.I),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class RunStatus(str, Enum):
    SUBMITTED   = "submitted"   # only a stub log (Hydra wrote, Modal spawned)
    RUNNING     = "running"     # log is growing; no terminal marker yet
    SUCCESS     = "success"
    FAILED      = "failed"
    UNKNOWN     = "unknown"


@dataclass
class RunInfo:
    path: Path                               # logs/<group>/<name>_<ts>/
    log_file: Path                           # path to trainHydra.log
    group: str                               # e.g. "mecka_modal"
    name: str                               # e.g. "test_2026-05-12_09-59-21"
    status: RunStatus = RunStatus.UNKNOWN
    failure_kind: Optional[FailureKind] = None
    error_lines: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    overrides: list[str] = field(default_factory=list)     # from .hydra/overrides.yaml
    log_size_bytes: int = 0
    last_modified: float = 0.0

    @property
    def short_name(self) -> str:
        return f"{self.group}/{self.name}"

    @property
    def age_hours(self) -> float:
        return (time.time() - self.last_modified) / 3600


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _is_noise(line: str) -> bool:
    return any(p.search(line) for p in _NOISE_PATTERNS)


def parse_log(log_path: Path) -> tuple[RunStatus, Optional[FailureKind], list[str], list[str]]:
    """Return (status, failure_kind, error_lines, metric_lines)."""
    if not log_path.exists():
        return RunStatus.UNKNOWN, None, [], []

    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return RunStatus.UNKNOWN, None, [], []

    lines = text.splitlines()
    error_lines: list[str] = []
    metric_lines: list[str] = []
    failure_kind: Optional[FailureKind] = None

    # --- check for success markers first ---
    for line in lines:
        for pat in _SUCCESS_PATTERNS:
            if pat.search(line):
                # Still collect metrics
                for ml in lines:
                    for mp in _METRIC_PATTERNS:
                        if mp.search(ml) and ml not in metric_lines:
                            metric_lines.append(ml.strip())
                return RunStatus.SUCCESS, None, [], metric_lines[:20]

    # --- scan for errors ---
    in_traceback = False
    traceback_buf: list[str] = []

    for line in lines:
        if _is_noise(line):
            continue

        if line.strip().startswith("Traceback (most recent call last)"):
            in_traceback = True
            traceback_buf = [line]
            continue

        if in_traceback:
            traceback_buf.append(line)
            # End of traceback: a non-indented non-empty line after the first
            if line and not line.startswith(" ") and not line.startswith("\t") and len(traceback_buf) > 1:
                in_traceback = False
                error_lines.extend(traceback_buf)
                traceback_buf = []
            continue

        for pattern, kind in _PATTERNS:
            if pattern.search(line):
                if kind != FailureKind.GENERIC or not _is_noise(line):
                    error_lines.append(line.strip())
                    if failure_kind is None:
                        failure_kind = kind
                break

        for mp in _METRIC_PATTERNS:
            if mp.search(line) and line.strip() not in metric_lines:
                metric_lines.append(line.strip())

    # Flush any open traceback
    if traceback_buf:
        error_lines.extend(traceback_buf)
        if failure_kind is None:
            failure_kind = FailureKind.GENERIC

    if error_lines:
        return RunStatus.FAILED, failure_kind or FailureKind.GENERIC, error_lines[:40], metric_lines[:20]

    # A log that has only the WARNING line is a "submitted" stub
    if len(lines) <= 3:
        return RunStatus.SUBMITTED, None, [], []

    return RunStatus.RUNNING, None, [], metric_lines[:20]


def _read_overrides(run_path: Path) -> list[str]:
    overrides_file = run_path / ".hydra" / "overrides.yaml"
    if not overrides_file.exists():
        return []
    try:
        return [
            line.strip().lstrip("- ")
            for line in overrides_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    except OSError:
        return []


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------

def discover_runs(logs_dir: Path = LOGS_DIR) -> list[RunInfo]:
    """Walk logs/<group>/<run>/ and parse each run."""
    runs: list[RunInfo] = []
    if not logs_dir.exists():
        return runs

    for group_dir in sorted(logs_dir.iterdir()):
        if not group_dir.is_dir():
            continue
        for run_dir in sorted(group_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            log_file = run_dir / LOG_FILE_NAME
            stat = log_file.stat() if log_file.exists() else run_dir.stat()
            status, failure_kind, error_lines, metrics = parse_log(log_file)
            overrides = _read_overrides(run_dir)
            runs.append(RunInfo(
                path=run_dir,
                log_file=log_file,
                group=group_dir.name,
                name=run_dir.name,
                status=status,
                failure_kind=failure_kind,
                error_lines=error_lines,
                metrics=metrics,
                overrides=overrides,
                log_size_bytes=log_file.stat().st_size if log_file.exists() else 0,
                last_modified=stat.st_mtime,
            ))

    # Sort newest first
    runs.sort(key=lambda r: r.last_modified, reverse=True)
    return runs


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_STATUS_COLOR = {
    RunStatus.SUCCESS:   "\033[32m",   # green
    RunStatus.FAILED:    "\033[31m",   # red
    RunStatus.RUNNING:   "\033[33m",   # yellow
    RunStatus.SUBMITTED: "\033[36m",   # cyan
    RunStatus.UNKNOWN:   "\033[90m",   # gray
}
_RESET = "\033[0m"


def _colorize(text: str, color: str) -> str:
    if sys.stdout.isatty():
        return f"{color}{text}{_RESET}"
    return text


def print_run_summary(runs: list[RunInfo]) -> None:
    """Print a compact status table."""
    if not runs:
        print("No runs found in", LOGS_DIR)
        return

    counts = {s: 0 for s in RunStatus}
    for r in runs:
        counts[r.status] += 1

    print(f"\n{'='*70}")
    print(f"  EgoVerse Run Manager  —  {LOGS_DIR}")
    print(f"{'='*70}")
    parts = []
    for s in [RunStatus.SUCCESS, RunStatus.RUNNING, RunStatus.FAILED, RunStatus.SUBMITTED]:
        if counts[s]:
            parts.append(_colorize(f"{counts[s]} {s.value}", _STATUS_COLOR[s]))
    print("  " + "  |  ".join(parts))
    print(f"{'='*70}")

    for run in runs:
        color = _STATUS_COLOR[run.status]
        status_str = _colorize(f"[{run.status.value:>10}]", color)
        age = f"{run.age_hours:.1f}h ago"
        print(f"\n  {status_str}  {run.short_name}  ({age})")

        if run.overrides:
            key_overrides = [o for o in run.overrides if any(
                o.startswith(k) for k in ("model=", "data=", "trainer=", "name=", "description=")
            )]
            if key_overrides:
                print(f"             overrides: {' '.join(key_overrides[:5])}")

        if run.status == RunStatus.FAILED:
            kind_str = run.failure_kind.value if run.failure_kind else "unknown"
            print(f"             failure:  {_colorize(kind_str, _STATUS_COLOR[RunStatus.FAILED])}")
            for line in run.error_lines[:5]:
                print(f"               {line[:100]}")

        if run.metrics:
            print(f"             metrics:  {run.metrics[-1][:100]}")

    print(f"\n{'='*70}\n")


def print_run_detail(run: RunInfo) -> None:
    """Print full error for a single failed run."""
    color = _STATUS_COLOR[run.status]
    print(f"\n{'='*70}")
    print(f"  {_colorize(run.short_name, color)}")
    print(f"  Status : {run.status.value}")
    print(f"  Log    : {run.log_file}")
    print(f"  Age    : {run.age_hours:.1f}h ago")
    if run.overrides:
        print(f"  Overrides:")
        for o in run.overrides:
            print(f"    {o}")
    if run.error_lines:
        print(f"\n  {'='*30} ERROR {'='*30}")
        for line in run.error_lines:
            print(f"  {line}")
    if run.metrics:
        print(f"\n  {'='*30} METRICS {'='*28}")
        for m in run.metrics[-10:]:
            print(f"  {m}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Auto-fix logic
# ---------------------------------------------------------------------------

def suggest_fix(run: RunInfo) -> Optional[str]:
    """Return a human-readable fix suggestion for a failed run."""
    kind = run.failure_kind
    if kind is None:
        return None

    if kind == FailureKind.CUDA_OOM:
        return (
            "CUDA OOM: reduce batch size or switch to a larger GPU.\n"
            "  Re-submit with: +modal_gpu=A100:2  (multi-GPU)\n"
            "  or add trainer.accumulate_grad_batches=4 to overrides."
        )
    if kind == FailureKind.MODAL_TIMEOUT:
        return (
            "Modal timeout: the container ran out of time.\n"
            "  ModalAutoRestartCallback should have saved a checkpoint.\n"
            "  Re-submit with ckpt_path=<last_modal_auto_restart.ckpt>."
        )
    if kind == FailureKind.GIT_PUSH:
        return (
            "git push failed before Modal submission.\n"
            "  Check: git push --set-upstream origin <branch>\n"
            "  Then re-run trainHydra.py."
        )
    if kind == FailureKind.SECRET_MISSING:
        return (
            "A Modal secret is missing.\n"
            "  Check modal.com/apps → Secrets for: egoverse-r2, egoverse-mongodb, egoverse-db, egoverse-sql\n"
            "  Run: modal run --env robotics egomimic/modal/trainModal.py::verify"
        )
    if kind == FailureKind.VOLUME_MOUNT:
        return (
            "Volume mount failed in the Modal container.\n"
            "  Run: modal run --env robotics egomimic/modal/trainModal.py::verify\n"
            "  Check that 'mecka_data_v2' and 'egoverse-training-outputs' volumes exist."
        )
    if kind == FailureKind.NORM_STATS:
        # Try to extract the bad path from error lines
        bad_path = None
        for line in run.error_lines:
            m = re.search(r"norm_stats[^\s,\"']*", line)
            if m:
                bad_path = m.group()
                break
        fix = (
            "precomputed_norm_path not found in the training-outputs volume.\n"
            "  Either remove norm_stats.precomputed_norm_path from your overrides,\n"
            "  or run offline_norm_stats.py first and verify the path exists."
        )
        if bad_path:
            fix += f"\n  Bad path: {bad_path}"
        return fix
    if kind == FailureKind.CHECKPOINT:
        bad_ckpt = None
        for line in run.error_lines:
            m = re.search(r"[\w./\-]+\.ckpt", line)
            if m:
                bad_ckpt = m.group()
                break
        fix = (
            "Checkpoint file not found in the training-outputs volume.\n"
            "  Use: modal volume ls --env robotics egoverse-training-outputs  to verify.\n"
            "  Remove ckpt_path= override if you want to start fresh."
        )
        if bad_ckpt:
            fix += f"\n  Missing: {bad_ckpt}"
        return fix
    if kind == FailureKind.HYDRA_CONFIG:
        return (
            "Hydra/OmegaConf configuration error.\n"
            "  Check the config override values and HYDRA_FULL_ERROR=1 output.\n"
            "  Common cause: mistyped config group name or missing key in yaml."
        )
    if kind == FailureKind.IMPORT_ERROR:
        bad_mod = None
        for line in run.error_lines:
            m = re.search(r"No module named '([^']+)'", line)
            if m:
                bad_mod = m.group(1)
                break
        fix = (
            "Python ImportError inside the Modal container.\n"
            "  Check that the package is listed in modal_setup.py image.pip_install(...)."
        )
        if bad_mod:
            fix += f"\n  Missing module: {bad_mod}"
        return fix
    if kind == FailureKind.CUDA_INIT:
        return (
            "CUDA driver initialization failed.\n"
            "  Likely the requested GPU type is unavailable in the Modal robotics environment.\n"
            "  Try a different GPU: +modal_gpu=A100 or +modal_gpu=L40S"
        )
    return None


def build_resubmit_cmd(run: RunInfo) -> Optional[list[str]]:
    """Build a re-submit command from a failed run's overrides.

    Returns the argv list to pass to subprocess, or None if we can't reconstruct it.
    """
    if not run.overrides:
        return None

    # Base command
    cmd = [sys.executable, str(REPO_ROOT / "egomimic" / "modal" / "trainModal.py")]

    # Strip problematic overrides that caused the failure
    skip_prefixes: list[str] = []
    if run.failure_kind == FailureKind.NORM_STATS:
        skip_prefixes.append("norm_stats.precomputed_norm_path=")
    if run.failure_kind == FailureKind.CHECKPOINT:
        skip_prefixes.append("ckpt_path=")

    for override in run.overrides:
        if any(override.startswith(p) for p in skip_prefixes):
            print(f"  [fix] Dropping problematic override: {override}")
            continue
        cmd.append(override)

    # For CUDA OOM: suggest bumping memory (can't auto-apply without user confirmation)
    if run.failure_kind == FailureKind.CUDA_OOM:
        print("  [fix] Adding +modal_memory_gb=128 to address CUDA OOM")
        cmd.append("+modal_memory_gb=128")

    return cmd


# ---------------------------------------------------------------------------
# Clean-up
# ---------------------------------------------------------------------------

def clean_failed_runs(runs: list[RunInfo], dry_run: bool = False) -> int:
    """Delete log directories for failed runs. Returns number deleted."""
    failed = [r for r in runs if r.status == RunStatus.FAILED]
    if not failed:
        print("No failed runs to clean up.")
        return 0

    print(f"\nFound {len(failed)} failed run(s):\n")
    for r in failed:
        print(f"  {r.short_name}  ({r.age_hours:.1f}h ago)  [{r.failure_kind}]")
        print(f"    {r.path}")

    if dry_run:
        print("\n[dry-run] No files deleted.")
        return 0

    print()
    answer = input(f"Delete all {len(failed)} failed run directories? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return 0

    deleted = 0
    for r in failed:
        try:
            shutil.rmtree(r.path)
            print(f"  Deleted: {r.path}")
            deleted += 1
        except OSError as e:
            print(f"  ERROR deleting {r.path}: {e}")
    print(f"\nDeleted {deleted} run directories.")
    return deleted


# ---------------------------------------------------------------------------
# Auto-fix entrypoint
# ---------------------------------------------------------------------------

def auto_fix_failed_runs(runs: list[RunInfo], dry_run: bool = False) -> None:
    """Print fixes and optionally re-submit failed runs."""
    failed = [r for r in runs if r.status == RunStatus.FAILED]
    if not failed:
        print("No failed runs found.")
        return

    for run in failed:
        print_run_detail(run)
        fix = suggest_fix(run)
        if fix:
            print(f"  SUGGESTED FIX:\n")
            for line in fix.splitlines():
                print(f"    {line}")
        cmd = build_resubmit_cmd(run)
        if cmd:
            print(f"\n  RE-SUBMIT COMMAND:")
            print(f"    {shlex.join(cmd)}\n")
            if not dry_run:
                answer = input("  Re-submit this run? [y/N] ").strip().lower()
                if answer == "y":
                    print(f"  Submitting...")
                    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
                    if result.returncode == 0:
                        print("  Submitted successfully.")
                    else:
                        print(f"  Submission failed (exit {result.returncode}).")
        print()


# ---------------------------------------------------------------------------
# Watch mode
# ---------------------------------------------------------------------------

def watch_logs(interval: int = 30) -> None:
    """Continuously re-scan logs/ and print status. Ctrl-C to stop."""
    print(f"Watching {LOGS_DIR} every {interval}s  (Ctrl-C to stop)\n")
    seen_failures: set[str] = set()
    try:
        while True:
            runs = discover_runs()
            print_run_summary(runs)

            # Alert on newly-detected failures
            for run in runs:
                if run.status == RunStatus.FAILED and run.short_name not in seen_failures:
                    seen_failures.add(run.short_name)
                    print(f"  *** NEW FAILURE: {run.short_name} [{run.failure_kind}] ***")
                    fix = suggest_fix(run)
                    if fix:
                        print("  FIX:")
                        for line in fix.splitlines():
                            print(f"    {line}")
                    print()

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EgoVerse Modal run manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Watch logs/ continuously and alert on new failures.",
    )
    parser.add_argument(
        "--interval", type=int, default=30,
        help="Polling interval in seconds for --watch (default: 30).",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Interactively delete log directories for failed runs.",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Print auto-fix suggestions and optionally re-submit failed runs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="With --clean or --fix: print actions without executing them.",
    )
    parser.add_argument(
        "--detail", metavar="RUN_NAME",
        help="Print full error detail for a specific run (partial name match).",
    )
    parser.add_argument(
        "--logs-dir", default=str(LOGS_DIR),
        help=f"Path to logs directory (default: {LOGS_DIR}).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logs_dir = Path(args.logs_dir)

    if args.watch:
        watch_logs(interval=args.interval)
        return

    runs = discover_runs(logs_dir)

    if args.detail:
        query = args.detail.lower()
        matches = [r for r in runs if query in r.short_name.lower()]
        if not matches:
            print(f"No run matching '{args.detail}'")
            return
        for r in matches:
            print_run_detail(r)
        return

    print_run_summary(runs)

    if args.fix:
        auto_fix_failed_runs(runs, dry_run=args.dry_run)
        return

    if args.clean:
        clean_failed_runs(runs, dry_run=args.dry_run)
        return

    # Default: if any failures, print brief fix hints
    failed = [r for r in runs if r.status == RunStatus.FAILED]
    if failed:
        print(f"  {len(failed)} failed run(s). Run with --fix for auto-fix suggestions,")
        print(f"  or --clean to remove their log directories.\n")


if __name__ == "__main__":
    main()
