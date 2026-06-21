#!/usr/bin/env bash
# Launch HPT (hpt_bc_flow_mecka) training on the assembling-booklets subset inside
# tmux so it survives SSH disconnects AND the launching shell being torn down.
#
# Defaults: assembling_blookets_test data config (48 workers, bs=64), full epochs
#           (limit_train_batches=null so the per-epoch refill stall amortizes),
#           step-based checkpoint/val cadence, online wandb, fresh run.
#
# Usage:
#   ./run_hpt.sh                          # best config, online wandb
#   NUM_WORKERS=64 ./run_hpt.sh           # override dataloader workers
#   DESC=my_run ./run_hpt.sh              # change run name/description
#   MAX_STEPS=20000 ./run_hpt.sh          # add a hard stop (default: none)
#   SESSION=hptb ./run_hpt.sh             # use a different tmux session name
#
set -euo pipefail

REPO=/workspace/EgoVerse
SESSION="${SESSION:-hpt}"
DESC="${DESC:-hpt_blookets_bs64_48w}"
VAL_INTERVAL="${VAL_INTERVAL:-2000}"
MAX_STEPS="${MAX_STEPS:-}"
NUM_WORKERS="${NUM_WORKERS:-48}"        # train dataloader workers
BATCH_SIZE="${BATCH_SIZE:-64}"          # train dataloader batch size (per GPU/rank)
GPUS="${GPUS:-1}"                        # GPUs per node (DDP devices = GPUS * nodes)
DATA="${DATA:-assembling_blookets_test}" # hydra data config (e.g. fold_clothes)
NAME="${NAME:-blookets_hpt}"             # experiment name -> logs/<NAME>/<DESC>_<ts>
NORM_PATH="${NORM_PATH:-}"               # precomputed norm_stats.json -> skip recompute; empty = compute fresh
SAVE_TOP_K="${SAVE_TOP_K:-3}"            # keep only the N most recent ckpts; -1 = keep all
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"
LOG="/tmp/hpt_run_${DESC}.log"
EXTRA_ARGS=("$@")                        # extra hydra overrides passed through (e.g. reject_outliers=false)

# ---------------------------------------------------------------------------
# Phase 1: not yet inside tmux -> spawn a detached tmux session that re-runs us.
# ---------------------------------------------------------------------------
if [[ "${_HPT_IN_TMUX:-}" != "1" ]]; then
  command -v tmux >/dev/null || { echo "ERROR: tmux is not installed." >&2; exit 1; }
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' is already running." >&2
    echo "  attach: tmux attach -t $SESSION     kill: tmux kill-session -t $SESSION" >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    -e _HPT_IN_TMUX=1 -e DESC="$DESC" -e VAL_INTERVAL="$VAL_INTERVAL" \
    -e MAX_STEPS="$MAX_STEPS" -e NUM_WORKERS="$NUM_WORKERS" \
    -e BATCH_SIZE="$BATCH_SIZE" -e GPUS="$GPUS" \
    -e DATA="$DATA" -e NAME="$NAME" -e NORM_PATH="$NORM_PATH" \
    -e EGOMIMIC_ZARR_FDCACHE_FADV="${EGOMIMIC_ZARR_FDCACHE_FADV:-}" \
    -e EGOMIMIC_ZARR_FDCACHE_MAX="${EGOMIMIC_ZARR_FDCACHE_MAX:-}" \
    -e SAVE_TOP_K="$SAVE_TOP_K" -e LIMIT_VAL_BATCHES="$LIMIT_VAL_BATCHES" \
    -e SESSION="$SESSION" \
    "bash '$(readlink -f "$0")' ${EXTRA_ARGS[*]}"
  echo "Launched '$DESC' in tmux session '$SESSION' (eval every $VAL_INTERVAL steps)."
  [[ -n "$MAX_STEPS" ]] && echo "  max_steps     : $MAX_STEPS"
  echo "  workers : $NUM_WORKERS"
  echo "  batch   : $BATCH_SIZE per GPU x $GPUS GPUs = $((BATCH_SIZE * GPUS)) global"
  echo "  gpus    : $GPUS"
  echo "  attach  : tmux attach -t $SESSION    (detach: Ctrl-b then d)"
  echo "  log     : tail -f $LOG"
  echo "  stop    : tmux kill-session -t $SESSION"
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2: inside tmux -> set up env and run training.
# ---------------------------------------------------------------------------
cd "$REPO"
# shellcheck disable=SC1091
source emimic/bin/activate
[[ -f apikey.txt ]] && export HF_TOKEN="$(grep -oP 'HF_TOKEN=\K.*' apikey.txt || true)"
export WANDB_MODE=online
# CRITICAL (multi-GPU): NCCL's NVLS (NVLink SHARP multicast) transport tries to
# create CUDA multicast objects (transport/nvls.cc) which fail here with
# `Cuda failure 401 'the operation cannot be performed in the present state'`
# because the MNNVL fabric/IMEX state is uninitialized in this container. NVLS
# only engages with >=3 ranks, so 1-2 GPU runs worked but 4-GPU DDP died at
# init_process_group. Disable NVLS; NCCL falls back to ring/tree over NVLink P2P.
export NCCL_NVLS_ENABLE=0
# Keep checkpoint temp writes off the small overlay '/'; use the big volume.
export TMPDIR=/workspace/tmp TEMP=/workspace/tmp TMP=/workspace/tmp
mkdir -p "$TMPDIR"

ARGS=(
  --config-name=train_zarr_cartesian
  "data=$DATA"
  model=hpt_bc_flow_mecka
  trainer.limit_train_batches=null
  # Step-based validation: epoch is ~1136 batches with limit_train_batches=null,
  # so gate val on step count. Lightning rejects val_check_interval > batches/epoch
  # unless check_val_every_n_epoch is null (plain ddp.yaml still sets it to 200).
  trainer.check_val_every_n_epoch=null
  # '+' append: plain ddp.yaml (unlike ddp_pi.yaml) has no val_check_interval key.
  "+trainer.val_check_interval=$VAL_INTERVAL"
  "trainer.limit_val_batches=$LIMIT_VAL_BATCHES"
  "data.train_dataloader_params.mecka_bimanual.num_workers=$NUM_WORKERS"
  "data.train_dataloader_params.mecka_bimanual.batch_size=$BATCH_SIZE"
  "launch_params.gpus_per_node=$GPUS"
  "callbacks.model_checkpoint.save_top_k=$SAVE_TOP_K"
  "name=$NAME"
  "description=$DESC"
)
[[ -n "$MAX_STEPS" ]] && ARGS+=("+trainer.max_steps=$MAX_STEPS")
[[ -n "$NORM_PATH" ]] && ARGS+=("++norm_stats.precomputed_norm_path=$NORM_PATH")
# top_k>0 needs a quantity to rank by (Lightning rejects save_top_k>1 with monitor=None).
[[ "$SAVE_TOP_K" -gt 0 ]] && ARGS+=("+callbacks.model_checkpoint.monitor=step" "+callbacks.model_checkpoint.mode=max")

echo "[run_hpt] python -u egomimic/trainHydra.py ${ARGS[*]}"
python -u egomimic/trainHydra.py "${ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
