#!/usr/bin/env fish

# CORP Autonomous Training & CDCL Nogood Extraction Pipeline
# Fish shell frontend

set -l EPISODES 20
if test (count $argv) -ge 1
    set EPISODES $argv[1]
end

set -l STEPS 5000
if test (count $argv) -ge 2
    set STEPS $argv[2]
end

set -l RUN_ID "train_"(date +%s)

echo "================================================================================"
echo "Starting CORP Autonomous Training: $EPISODES Episodes (Max $STEPS Steps/Ep)"
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

uv run python scripts/run_benchmark.py \
    --mode random \
    --eval-type training \
    --run-id $RUN_ID \
    --episodes $EPISODES \
    --max-steps $STEPS \
    $PROVIDER_ARGS

echo ""
echo "Training complete. Telemetry saved to data/corp_telemetry.duckdb"
