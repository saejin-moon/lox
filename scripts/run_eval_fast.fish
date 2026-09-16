#!/usr/bin/env fish

# CORP Fast Preliminary Evaluation (~15-30s Sanity Benchmark)
# Fish shell frontend

set -l EPISODES 10
if test (count $argv) -ge 1
    set EPISODES $argv[1]
end

set -l STEPS 1000
if test (count $argv) -ge 2
    set STEPS $argv[2]
end

set -l RUN_ID "fast_prelim_"(date +%s)

echo "================================================================================"
echo "Starting CORP Fast Preliminary Evaluation: $EPISODES Episodes (Max $STEPS Steps)"
echo "Run ID: $RUN_ID"
echo "================================================================================"

uv run python scripts/run_benchmark.py \
    --mode random \
    --eval-type fast_prelim \
    --run-id $RUN_ID \
    --episodes $EPISODES \
    --max-steps $STEPS \
    --provider mock

echo ""
echo "Fast evaluation complete."
