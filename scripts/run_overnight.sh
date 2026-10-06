#!/usr/bin/env bash
set -euo pipefail

# LOX Overnight Autonomous Policy Synthesis Runner
SESSION_NAME="lox-synthesis"
LOG_FILE="synthesis.log"

echo "=== LOX Overnight Policy Synthesis ==="
echo "Clearing previous synthesis log: $LOG_FILE"
> "$LOG_FILE"

echo "Starting fresh campaign with modular baseline: data/modular_starter_policy.py"
echo "Logging output to: $LOG_FILE"

# Start detached tmux session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Warning: tmux session '$SESSION_NAME' is already running."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

echo "Launching tmux session '$SESSION_NAME'..."
tmux new-session -d -s "$SESSION_NAME" \
  "uv run python scripts/run_synthesis.py \
     --provider openrouter \
     --model google/gemma-4-31b-it \
     --eval-episodes 20 \
     --max-turns 25000 \
     --target-depth 20.0 \
     --min-delta 0.25 \
     --min-improved 2 \
     --fresh \
     --policy-path data/latest_policy.py \
     --starter-policy data/modular_starter_policy.py \
     --twin-test \
     --workers 10 2>&1 | tee $LOG_FILE"

echo "Synthesis successfully running detached in tmux!"
echo "------------------------------------------------"
echo "To monitor live:       tmux attach -t $SESSION_NAME"
echo "To detach from tmux:   Press Ctrl+B then D"
echo "To view log file:      tail -f $LOG_FILE"
echo "To check if running:   tmux ls"
echo "To stop synthesis:     tmux kill-session -t $SESSION_NAME"
echo "------------------------------------------------"
