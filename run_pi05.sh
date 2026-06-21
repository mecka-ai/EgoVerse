
#!/usr/bin/env bash
# Launch pi0.5 training (best bs=64 / 32-worker config) inside tmux so it
# survives SSH disconnects AND the launching shell/session being torn down.
#
# Defaults: best dataloader config (baked into data=assembling_blookets_test_pi),
#           full epochs, eval every 2000 steps, online wandb, fresh run.
#
# Usage:
#   ./run_pi05.sh                                   # best config, eval every 2000 steps
#   VAL_INTERVAL=1000 ./run_pi05.sh                 # change eval cadence
#   DESC=my_run ./run_pi05.sh                       # change run name/description
#   CKPT_PATH=/abs/path/step_step=4000.ckpt ./run_pi05.sh   # resume from a checkpoint
#   MAX_STEPS=60000 ./run_pi05.sh                   # add a hard stop (default: none)
#   SESSION=pi05b ./run_pi05.sh                     # use a different tmux session name
#
set -euo pipefail

REPO=/workspace/EgoVerse
SESSION="${SESSION:-pi05}"
DESC="${DESC:-pi05_foldclothes_all_bs64_32w_4gpu}"
VAL_INTERVAL="${VAL_INTERVAL:-2000}"
CKPT_PATH="${CKPT_PATH:-}"
MAX_STEPS="${MAX_STEPS:-}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"   # keep only the N most recent ckpts (~21GB each); -1 = keep all (fills disk!)
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-20}"  # val batches per eval (was 80); fewer = much shorter eval pause
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-16}"      # val dataloader workers (was 8); also made persistent below
WANDB_ID="${WANDB_ID:-}"                       # set to a run id to CONTINUE/resume that wandb run
RUN_DIR="${RUN_DIR:-}"                          # pin hydra output dir (resume into same dir so ckpt mgmt + norm_stats are reused)
LOG="/tmp/pi_run_${DESC}.log"

# ---------------------------------------------------------------------------
# Phase 1: not yet inside tmux -> spawn a detached tmux session that re-runs us.
# ---------------------------------------------------------------------------
EXTRA_ARGS=("$@")
if [[ "${_PI_IN_TMUX:-}" != "1" ]]; then
  command -v tmux >/dev/null || { echo "ERROR: tmux is not installed." >&2; exit 1; }
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session '$SESSION' is already running." >&2
    echo "  attach: tmux attach -t $SESSION     kill: tmux kill-session -t $SESSION" >&2
    exit 1
  fi
  tmux new-session -d -s "$SESSION" \
    -e _PI_IN_TMUX=1 -e DESC="$DESC" -e VAL_INTERVAL="$VAL_INTERVAL" \
    -e CKPT_PATH="$CKPT_PATH" -e MAX_STEPS="$MAX_STEPS" -e SAVE_TOP_K="$SAVE_TOP_K" \
    -e LIMIT_VAL_BATCHES="$LIMIT_VAL_BATCHES" -e VAL_NUM_WORKERS="$VAL_NUM_WORKERS" \
    -e WANDB_ID="$WANDB_ID" -e RUN_DIR="$RUN_DIR" -e SESSION="$SESSION" \
    "bash '$(readlink -f "$0")' ${EXTRA_ARGS[*]}"
  echo "Launched '$DESC' in tmux session '$SESSION' (eval every $VAL_INTERVAL steps)."
  [[ -n "$CKPT_PATH" ]] && echo "  resuming from : $CKPT_PATH"
  [[ -n "$MAX_STEPS" ]] && echo "  max_steps     : $MAX_STEPS"
  echo "  attach : tmux attach -t $SESSION    (detach: Ctrl-b then d)"
  echo "  log    : tail -f $LOG"
  echo "  stop   : tmux kill-session -t $SESSION"
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase 2: inside tmux -> set up env and run training.
# ---------------------------------------------------------------------------
cd "$REPO"
# shellcheck disable=SC1091
source emimic/bin/activate
export HF_TOKEN="$(grep -oP 'HF_TOKEN=\K.*' apikey.txt)"
export WANDB_API_KEY="$(grep -oP 'WANDB_API_KEY=\K.*' apikey.txt)"
export WANDB_MODE=online
# CRITICAL (multi-GPU): NCCL's NVLS (NVLink SHARP multicast) transport tries to
# create CUDA multicast objects (transport/nvls.cc) which fail here with
# `Cuda failure 401 'the operation cannot be performed in the present state'`
# because the MNNVL fabric/IMEX state is uninitialized in this container (NCCL
# logs `fabric UUID 0.0 ... state 3`). NVLS only engages with >=3 ranks, so 1-2
# GPU runs worked but 4-GPU DDP died at init_process_group. Disable NVLS; NCCL
# falls back to ring/tree over NVLink P2P. (Verified: 4-rank all_reduce OK.)
export NCCL_NVLS_ENABLE=0
# fd-cache config: the zarr fd-cache (store_handle_cache) is ESSENTIAL for NFS
# throughput — disabling it caused NFS thrash (202s first batch, 12% GPU util).
# But the OLD cache was PER-STORE, so total fds = open-stores x cap, UNBOUNDED as
# the working set slides -> ~33.6k fds/proc -> system fs.file-max=5M exhausted ->
# "OSError [Errno 23] Too many open files" crash at ~step 1000. FIXED in
# store_handle_cache.py: the fd cache is now a single PER-PROCESS LRU, so total
# fds/proc are capped at EGOMIMIC_ZARR_FDCACHE_MAX regardless of store count.
# 256 workers x 4096 = ~1M fds (fs.file-max is read-only 5M in this container).
export EGOMIMIC_ZARR_FDCACHE=1
export EGOMIMIC_ZARR_FDCACHE_MAX=4096
ulimit -n 1048576 2>/dev/null || true
# CRITICAL: Lightning's _atomic_save writes the ~21GB checkpoint to a tempfile in $TMPDIR
# (fsspec transaction, autocommit=False) and then moves it to the target. Default TMPDIR=/tmp
# is the 9GB overlay '/', so the write ENOSPCs every time (and is a cross-device move to the
# NFS volume). Point TMPDIR at the big /workspace volume: room to write + same-device atomic rename.
export TMPDIR=/workspace/tmp TEMP=/workspace/tmp TMP=/workspace/tmp
mkdir -p "$TMPDIR"

ARGS=(
  --config-name=train_zarr_cartesian_pi
  data=fold_clothes_all_pi
  model=pi0.5_bc_mecka
  "model.robomimic_model.config.pytorch_weight_path=$REPO/egomimic/algo/pi_checkpoints/pi05_base_pytorch"
  trainer.limit_train_batches=null
  "trainer.val_check_interval=$VAL_INTERVAL"
  "trainer.limit_val_batches=$LIMIT_VAL_BATCHES"
  "data.valid_dataloader_params.mecka_bimanual.num_workers=$VAL_NUM_WORKERS"
  data.valid_dataloader_params.mecka_bimanual.persistent_workers=true
  "callbacks.model_checkpoint.save_top_k=$SAVE_TOP_K"
  name=fold_clothes_all_pi
  "description=$DESC"
)
[[ -n "$CKPT_PATH" ]] && ARGS+=("ckpt_path=$CKPT_PATH")
[[ -n "$MAX_STEPS" ]]  && ARGS+=("trainer.max_steps=$MAX_STEPS")
# Continue an existing wandb run (resume not in the config struct -> needs '+').
[[ -n "$WANDB_ID" ]] && ARGS+=("logger.wandb.id=$WANDB_ID" "+logger.wandb.resume=allow")
# Pin the hydra output dir (e.g. resume into the same run dir).
[[ -n "$RUN_DIR" ]] && ARGS+=("hydra.run.dir=$RUN_DIR")
# top_k>0 needs a quantity to rank by (Lightning rejects save_top_k>1 with monitor=None).
# Rank by global step => "keep the N most recent checkpoints". (-1=all, 0=none need no monitor.)
[[ "$SAVE_TOP_K" -gt 0 ]] && ARGS+=("+callbacks.model_checkpoint.monitor=step" "+callbacks.model_checkpoint.mode=max")

echo "[run_pi05] python -u egomimic/trainHydra.py ${ARGS[*]} ${EXTRA_ARGS[*]}"
python -u egomimic/trainHydra.py "${ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "$LOG"
