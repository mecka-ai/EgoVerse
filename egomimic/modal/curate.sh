#!/usr/bin/env bash
# Launch a DemInf curation run on Modal — always detached (nohup).
# All args are passed as Hydra overrides to run_curate_cmd.
#
# Usage:
#   egomimic/modal/curate.sh name=my_run description=test
#   egomimic/modal/curate.sh name=my_run description=test model.scorer.k_range=[5,9]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="/tmp/curate_$(date +%Y%m%d_%H%M%S).log"

nohup modal run --env robotics "$SCRIPT_DIR/curateModal.py::run_curate_cmd" -- "$@" > "$LOG" 2>&1 &
PID=$!

echo "Curation launched (PID $PID)"
echo "Log:     $LOG"
echo "Monitor: https://modal.com/apps/mecka/robotics"
echo ""
echo "Tail logs:  tail -f $LOG"
