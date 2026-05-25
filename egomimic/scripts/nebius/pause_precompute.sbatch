#!/bin/bash
# Array-job wrapper for the Nebius/SLURM pause-precompute worker.
#
# Submitted by egomimic.scripts.nebius.pause_precompute_driver, which fills in
# the --array range and per-job sizing via CLI overrides. Defaults below are
# safe for ~700 episodes/shard on a CPU partition.
#
# Manual invocation:
#   sbatch --array=0-99%50 \
#          pause_precompute.sbatch <manifest_path> <out_dir>
#
# Required positional args (passed by the driver):
#   $1 = manifest path (jsonl, one shard per line)
#   $2 = out dir (will contain shards/, cache.json once aggregated)

#SBATCH --job-name=pause-precompute
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --ntasks=1

set -euo pipefail

MANIFEST="${1:?manifest path required}"
OUT_DIR="${2:?out dir required}"

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set — this script is meant to be sbatch'd as an array job" >&2
    exit 2
fi

# Cluster-specific environment hookup. Override these in the driver via
# --extra-sbatch if your site uses different module/conda/container patterns.
#
# Nebius managed-Slurm typically wants enroot+pyxis with a baked image:
#   #SBATCH --container-image=/shared/images/egomimic.sqsh
#   #SBATCH --container-mounts=/shared:/shared,/scratch:/scratch
# In that case the python below just runs inside the container.
#
# For a venv-based setup, source the project venv:
if [[ -f "${EGOMIMIC_VENV_ACTIVATE:-}" ]]; then
    # shellcheck disable=SC1090
    source "${EGOMIMIC_VENV_ACTIVATE}"
fi

# Make sure the repo (or its installed wheel) is importable. EGOMIMIC_REPO_ROOT
# lets you point at a checked-out repo without needing it on PYTHONPATH via
# system config.
if [[ -n "${EGOMIMIC_REPO_ROOT:-}" ]]; then
    export PYTHONPATH="${EGOMIMIC_REPO_ROOT}:${PYTHONPATH:-}"
fi

echo "=== pause_precompute.sbatch ==="
echo "host=$(hostname) | task=${SLURM_ARRAY_TASK_ID} | array_job=${SLURM_ARRAY_JOB_ID}"
echo "manifest=${MANIFEST}"
echo "out_dir=${OUT_DIR}"
echo "python=$(command -v python) | $(python --version 2>&1)"
echo "==============================="

exec python -m egomimic.scripts.nebius.pause_precompute_worker \
    --manifest "${MANIFEST}" \
    --out-dir "${OUT_DIR}" \
    --shard-id "${SLURM_ARRAY_TASK_ID}"
