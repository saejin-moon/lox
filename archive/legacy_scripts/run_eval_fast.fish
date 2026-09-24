#!/usr/bin/env fish

# LOX-ψ Fast Preliminary Evaluation (~15-30s Sanity Benchmark)
# Usage: ./scripts/run_eval_fast.fish [episodes] [steps]

set -l EPISODES 10
if test (count $argv) -ge 1
    set EPISODES $argv[1]
end

set -l STEPS 1000
if test (count $argv) -ge 2
    set STEPS $argv[2]
end

# Generate 6-character Base-62 run ID from date + time
set -l RUN_ID (uv run python scripts/gen_run_id.py)

echo "================================================================================"
echo "Starting LOX-ψ Fast Preliminary Evaluation: $EPISODES Episodes (Max $STEPS Steps)"
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
