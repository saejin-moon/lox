#!/usr/bin/env bash
set -e

# LOX-ψ Fast Preliminary Evaluation (~15-30s Sanity Benchmark)
# Usage: ./scripts/run_eval_fast.sh [episodes] [steps]

EPISODES=${1:-10}
STEPS=${2:-1000}

# Generate 6-character Base-62 run ID from date + time
RUN_ID=$(uv run python scripts/gen_run_id.py)

echo "================================================================================"
echo "Starting LOX-ψ Fast Preliminary Evaluation: $EPISODES Episodes (Max $STEPS Steps)"
echo "Run ID: $RUN_ID"
echo "================================================================================"

uv run python scripts/run_benchmark.py \
    --mode random \
    --eval-type fast_prelim \
    --run-id "$RUN_ID" \
    --episodes "$EPISODES" \
    --max-steps "$STEPS" \
    --provider mock

echo ""
echo "Fast evaluation complete."
