#!/usr/bin/env bash
set -euo pipefail

# LOX Overnight Autonomous Policy Synthesis Runner
SESSION_NAME="lox-synthesis"
LOG_FILE="synthesis.log"

FRESH_ARG=""
if [[ "${1:-}" == "--fresh" ]]; then
    FRESH_ARG="--fresh"
    echo "Starting FRESH campaign with baseline: data/modular_starter_policy.py"
    echo "Clearing previous synthesis log: $LOG_FILE"
    > "$LOG_FILE"
elif [[ -f "data/latest_policy.py" ]]; then
    echo "Resuming campaign from active checkpoint: data/latest_policy.py"
    echo "(To start completely fresh, run: $0 --fresh)"
    echo "Appending output to: $LOG_FILE"
else
    FRESH_ARG="--fresh"
    echo "No existing checkpoint found. Starting fresh from modular baseline."
    > "$LOG_FILE"
fi

# Start detached tmux session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Warning: tmux session '$SESSION_NAME' is already running."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

echo "Launching tmux session '$SESSION_NAME'..."
WORKERS="${WORKERS:-20}"
tmux new-session -d -s "$SESSION_NAME" \
  "uv run python -u scripts/run_synthesis.py \
     --provider openrouter \
     --model qwen/qwen3.5-9b \
     --generations 1000 \
     --eval-episodes 100 \
     --max-turns 25000 \
     --target-depth 50.0 \
     --min-delta 0.40 \
     --min-improved 20 \
     $FRESH_ARG \
     --policy-path data/latest_policy.py \
     --starter-policy data/modular_starter_policy.py \
     --twin-test \
     --workers $WORKERS 2>&1 | tee -a $LOG_FILE"

echo "Synthesis successfully running detached in tmux!"
echo "------------------------------------------------"
echo "To monitor live:       tmux attach -t $SESSION_NAME"
echo "To detach from tmux:   Press Ctrl+B then D"
echo "To view log file:      tail -f $LOG_FILE"
echo "To check if running:   tmux ls"
echo "To stop synthesis:     tmux kill-session -t $SESSION_NAME"
echo "------------------------------------------------"
