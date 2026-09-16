#!/usr/bin/env bash
set -e

# CORP Fast Preliminary Evaluation (~15-30s Sanity Benchmark)
# Bash frontend

EPISODES=${1:-10}
STEPS=${2:-1000}
RUN_ID="fast_prelim_$(date +%s)"

echo "================================================================================"
echo "Starting CORP Fast Preliminary Evaluation: $EPISODES Episodes (Max $STEPS Steps)"
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
