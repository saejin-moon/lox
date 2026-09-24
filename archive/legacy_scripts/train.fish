#!/usr/bin/env fish

# LOX-ψ Autonomous Training & CDCL Nogood Extraction Pipeline
# Usage: ./scripts/train.fish [episodes] [steps] [model]

set -l EPISODES 20
if test (count $argv) -ge 1
    set EPISODES $argv[1]
end

set -l STEPS 5000
if test (count $argv) -ge 2
    set STEPS $argv[2]
end

set -l MODEL ""
if test (count $argv) -ge 3
    set MODEL $argv[3]
else if set -q MODEL
    set MODEL $MODEL
end

# Generate 6-character Base-62 run ID from date + time
set -l RUN_ID (uv run python scripts/gen_run_id.py)

echo "================================================================================"
echo "Starting LOX-ψ Autonomous Training: $EPISODES Episodes (Max $STEPS Steps/Ep)"
echo "Run ID: $RUN_ID"
echo "================================================================================"

set -l PROVIDER_ARGS "--provider" "mock"
if set -q OPENROUTER_API_KEY
    echo "[Info] OPENROUTER_API_KEY detected. Enabling live open-weight thinking autopsies."
    set PROVIDER_ARGS "--provider" "openrouter" "--enable-autopsy"
else if set -q GEMINI_API_KEY
    echo "[Info] GEMINI_API_KEY detected. Enabling live Gemini Flash autopsies."
    set PROVIDER_ARGS "--provider" "gemini" "--enable-autopsy"
else
    echo "[Notice] No API keys detected in environment. Running offline training with mock autopsies."
    set PROVIDER_ARGS "--provider" "mock" "--enable-autopsy"
end

set -l MODEL_ARGS
if test -n "$MODEL"
    echo "[Info] Selected model override: $MODEL"
    set MODEL_ARGS "--model" "$MODEL"
end

uv run python scripts/run_benchmark.py \
    --mode random \
    --eval-type training \
    --run-id $RUN_ID \
    --episodes $EPISODES \
    --max-steps $STEPS \
    $PROVIDER_ARGS \
    $MODEL_ARGS

echo ""
echo "Training complete. Telemetry saved to data/lox_telemetry.duckdb"
