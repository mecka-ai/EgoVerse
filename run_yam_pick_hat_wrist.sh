#!/usr/bin/env bash
# YAM pick-hat finetune in WRISTFRAME 6D from the stock pi0.5 base release.
#
# Single-embodiment (yam_bimanual) pi0.5 finetune-init from pi05_base_pytorch on
# the local 42-episode yam_pick_hat set (mode: cartesian_wristframe_6d, 20-dim
# [L: xyz(3) 6d(6) grip(1) | R: same]; the cam<-base extrinsic cancels out of
# the wrist-frame action target). Mirrors run_eva_foldclothes_wrist.sh:
#   - DATA          = yam_pick_hat_wrist_pi  (local resolver, /workspace/yam_pick_hat)
#   - MODEL         = pi0.5_bc_yam           (RobotBimanualCartesian6D, 20-dim)
#   - visualization = pi_cartesian_lang_wrist_6d (wrist->cam revert; yam entry)
#   - NORM_PATH unset = compute fresh WRISTFRAME norm stats at startup (the set
#                       is tiny, so default sample_frac=1.0 -> exact stats).
#   - step-based cadence (val_check_interval / viz_every_n_steps /
#     every_n_train_steps), matching the step-logging training regime.
#
# Usage:
#   ./run_yam_pick_hat_wrist.sh                # full run, 8 GPUs, tmux
#   SMOKE=1 ./run_yam_pick_hat_wrist.sh        # inline pipeline smoke (trainer=debug, 1 GPU)
#   SMOKE=1 SMOKE_GPUS=4 ./run_yam_pick_hat_wrist.sh  # smoke with DDP across 4 GPUs
#   GPUS=4 LR=1e-4 ./run_yam_pick_hat_wrist.sh # tweak resources
#   CKPT_PATH=/abs/last.ckpt ./run_yam_pick_hat_wrist.sh   # resume an interrupted run
set -euo pipefail

# Anchor to the checkout this script lives in (NOT a hardcoded path): the venv's
# editable egomimic install may point at a different checkout/worktree, so we
# also export PYTHONPATH so `import egomimic` resolves to THIS repo.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${VENV:-$REPO/emimic}"
[[ -d "$VENV" ]] || VENV=/workspace/EgoVerse/emimic
KEYFILE="$REPO/apikey.txt"
[[ -f "$KEYFILE" ]] || KEYFILE=/workspace/EgoVerse/apikey.txt

SESSION="${SESSION:-yam_wrist}"
DESC="${DESC:-pi05_yam_pick_hat_wrist}"
NAME="${NAME:-yam_pick_hat_wrist}"
DATA="${DATA:-yam_pick_hat_wrist_pi}"
MODEL="${MODEL:-pi0.5_bc_yam}"
VIZ="${VIZ:-pi_cartesian_lang_wrist_6d}"
WEIGHT_PATH="${WEIGHT_PATH:-$REPO/egomimic/algo/pi_checkpoints/pi05_base_pytorch}"
[[ -f "$WEIGHT_PATH/model.safetensors" ]] || WEIGHT_PATH=/workspace/EgoVerse/egomimic/algo/pi_checkpoints/pi05_base_pytorch
LR="${LR:-5e-5}"
# Norm stats: empty => compute FRESH wristframe stats at startup (correct frame;
# 42 episodes -> exact full-pass stats in ~a minute). Set NORM_PATH only if you
# precomputed for THIS wristframe config.
NORM_PATH="${NORM_PATH:-}"
NORM_SAMPLE_FRAC="${NORM_SAMPLE_FRAC:-1.0}"
GPUS="${GPUS:-8}"
# Step-based cadence: val every 1000 steps, ckpt every 2000, viz every 5000
# (every 5th val). VIZ must be a multiple of VAL_INTERVAL for the step-modulo
# gate to line up with a validation pass.
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
VIZ_EVERY="${VIZ_EVERY:-5000}"
CKPT_EVERY="${CKPT_EVERY:-2000}"
CKPT_PATH="${CKPT_PATH:-}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-50}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-4}"
WANDB_ID="${WANDB_ID:-}"
RUN_DIR="${RUN_DIR:-}"
SMOKE="${SMOKE:-}"                              # SMOKE=1 -> trainer=debug pipeline check
LOG="/tmp/yam_wrist_${DESC}.log"

_common_env() {
  cd "$REPO"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
  export NCCL_NVLS_ENABLE=0
  export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
  # /tmp (local + executable): NFS TMPDIR deadlocks the val dataloader teardown
  # and noexec tmpfs breaks the val torch.compile dlopen (see eva run notes).
  export TMPDIR=/tmp/yam_tmp TEMP=/tmp/yam_tmp TMP=/tmp/yam_tmp
  mkdir -p "$TMPDIR"
}

# ---------------------------------------------------------------------------
# SMOKE: run INLINE (no tmux) with trainer=debug for a fast end-to-end pipeline
# check: loads data -> wristframe transform -> forward/loss -> validates (incl.
# the wrist->cam viz revert, forced every step) -> checkpoints -> exits, on ONE
# GPU. trainer=debug caps limit_train_batches=2 / limit_val_batches=3 /
# max_epochs=4, so do NOT pass limit_train_batches=null here.
# ---------------------------------------------------------------------------
if [[ -n "$SMOKE" ]]; then
  _common_env
  export HF_TOKEN="$(grep -oP 'HF_TOKEN=\K.*' "$KEYFILE" || true)"
  export WANDB_MODE=disabled
  SMOKE_GPUS="${SMOKE_GPUS:-1}"
  [[ -f "$WEIGHT_PATH/model.safetensors" ]] || {
    echo "ERROR: base weights not found: $WEIGHT_PATH/model.safetensors" >&2; exit 1; }

  SMOKE_ARGS=(
    --config-name=train_zarr_cartesian_pi
    "data=$DATA"
    "model=$MODEL"
    "visualization=$VIZ"
    "model.robomimic_model.config.pytorch_weight_path=$WEIGHT_PATH"
    trainer=debug
    "launch_params.gpus_per_node=$SMOKE_GPUS"
    # Fire viz every step so the wrist->cam revert path is exercised in the smoke.
    evaluator.viz_every_n_steps=1
    train_viz_evaluator.viz_every_n_steps=1
    norm_stats.precomputed_norm_path=null
    norm_stats.sample_frac=0.05
    name="${NAME}_smoke"
    description="${DESC}_smoke"
  )
  echo "[smoke] python -u egomimic/trainHydra.py ${SMOKE_ARGS[*]} $*"
  python -u egomimic/trainHydra.py "${SMOKE_ARGS[@]}" "$@" 2>&1 | tee "/tmp/yam_wrist_smoke.log"
  exit "${PIPESTATUS[0]}"
fi

# ---------------------------------------------------------------------------
# Phase 1: not yet inside tmux -> spawn a detached tmux session that re-runs us.
# ---------------------------------------------------------------------------
EXTRA_ARGS=("$@")
if [[ "${_YAM_IN_TMUX:-}" != "1" ]]; then
  command -v tmux >/dev/null || { echo "ERROR: tmux is not installed." >&2; exit 1; }
  if [[ -z "$CKPT_PATH" && ! -f "$WEIGHT_PATH/model.safetensors" ]]; then
    echo "ERROR: finetune-init weights not found: $WEIGHT_PATH/model.safetensors" >&2
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' is already running." >&2
    echo "  attach: tmux attach -t $SESSION     kill: tmux kill-session -t $SESSION" >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    -e _YAM_IN_TMUX=1 -e DESC="$DESC" -e NAME="$NAME" -e DATA="$DATA" -e MODEL="$MODEL" \
    -e VIZ="$VIZ" -e WEIGHT_PATH="$WEIGHT_PATH" -e LR="$LR" -e VENV="$VENV" \
    -e NORM_PATH="$NORM_PATH" -e NORM_SAMPLE_FRAC="$NORM_SAMPLE_FRAC" \
    -e GPUS="$GPUS" -e VAL_INTERVAL="$VAL_INTERVAL" -e CKPT_PATH="$CKPT_PATH" \
    -e VIZ_EVERY="$VIZ_EVERY" -e CKPT_EVERY="$CKPT_EVERY" \
    -e MAX_STEPS="$MAX_STEPS" -e SAVE_TOP_K="$SAVE_TOP_K" \
    -e LIMIT_VAL_BATCHES="$LIMIT_VAL_BATCHES" -e VAL_NUM_WORKERS="$VAL_NUM_WORKERS" \
    -e WANDB_ID="$WANDB_ID" -e RUN_DIR="$RUN_DIR" -e SESSION="$SESSION" \
    "bash '$(readlink -f "$0")' ${EXTRA_ARGS[*]}"
  echo "Launched '$DESC' in tmux session '$SESSION' (val every $VAL_INTERVAL steps, $GPUS GPUs)."
  if [[ -n "$CKPT_PATH" ]]; then
    echo "  RESUMING from : $CKPT_PATH"
  else
    echo "  base-init weights : $WEIGHT_PATH/model.safetensors"
  fi
  echo "  attach : tmux attach -t $SESSION    (detach: Ctrl-b then d)"
  echo "  log    : tail -f $LOG"
  echo "  stop   : tmux kill-session -t $SESSION"
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2: inside tmux -> set up env and run training.
# ---------------------------------------------------------------------------
_common_env
export HF_TOKEN="$(grep -oP 'HF_TOKEN=\K.*' "$KEYFILE")"
export WANDB_API_KEY="$(grep -oP 'WANDB_API_KEY=\K.*' "$KEYFILE")"
export WANDB_MODE=online

ARGS=(
  --config-name=train_zarr_cartesian_pi
  "data=$DATA"
  "model=$MODEL"
  "visualization=$VIZ"
  "model.optimizer.lr=$LR"
  "launch_params.gpus_per_node=$GPUS"
  trainer.limit_train_batches=null
  "trainer.val_check_interval=$VAL_INTERVAL"
  "trainer.limit_val_batches=$LIMIT_VAL_BATCHES"
  +trainer.log_every_n_steps=100
  "data.valid_dataloader_params.yam_bimanual.num_workers=$VAL_NUM_WORKERS"
  data.valid_dataloader_params.yam_bimanual.persistent_workers=true
  data.valid_dataloader_params.yam_bimanual.shuffle=false
  "callbacks.model_checkpoint.save_top_k=$SAVE_TOP_K"
  "callbacks.model_checkpoint.every_n_train_steps=$CKPT_EVERY"
  # Render eval + train-viz videos every VIZ_EVERY steps (metrics still every val).
  "evaluator.viz_every_n_steps=$VIZ_EVERY"
  "train_viz_evaluator.viz_every_n_steps=$VIZ_EVERY"
  "name=$NAME"
  "description=$DESC"
)
if [[ -n "$NORM_PATH" ]]; then
  ARGS+=("norm_stats.precomputed_norm_path=$NORM_PATH")
else
  ARGS+=("norm_stats.sample_frac=$NORM_SAMPLE_FRAC" "norm_stats.precomputed_norm_path=null")
fi
if [[ -n "$CKPT_PATH" ]]; then
  # Single-quote: Lightning ckpt names contain '=' which breaks Hydra's grammar.
  ARGS+=("ckpt_path='$CKPT_PATH'" "model.robomimic_model.config.pytorch_weight_path=null")
else
  ARGS+=("model.robomimic_model.config.pytorch_weight_path=$WEIGHT_PATH")
fi
[[ -n "$MAX_STEPS" ]] && ARGS+=(+"trainer.max_steps=$MAX_STEPS")
[[ -n "$WANDB_ID" ]] && ARGS+=("logger.wandb.id=$WANDB_ID" "+logger.wandb.resume=allow")
[[ -n "$RUN_DIR" ]] && ARGS+=("hydra.run.dir=$RUN_DIR")
[[ "$SAVE_TOP_K" -gt 0 ]] && ARGS+=("+callbacks.model_checkpoint.monitor=step" "+callbacks.model_checkpoint.mode=max")

echo "[run_yam_pick_hat_wrist] python -u egomimic/trainHydra.py ${ARGS[*]} ${EXTRA_ARGS[*]}"
python -u egomimic/trainHydra.py "${ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
