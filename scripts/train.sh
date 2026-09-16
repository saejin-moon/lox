#!/usr/bin/env bash
set -e

# CORP Autonomous Training & CDCL Nogood Extraction Pipeline
# Bash frontend

EPISODES=${1:-20}
STEPS=${2:-5000}
RUN_ID="train_$(date +%s)"

echo "================================================================================"
echo "Starting CORP Autonomous Training: $EPISODES Episodes (Max $STEPS Steps/Ep)"
echo "Run ID: $RUN_ID"
echo "================================================================================"

# Provider auto-detection
PROVIDER_ARGS="--provider mock"
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "[Info] OPENROUTER_API_KEY detected. Enabling live open-weight thinking autopsies."
    PROVIDER_ARGS="--provider openrouter --enable-autopsy"
elif [ -n "$GEMINI_API_KEY" ]; then
    echo "[Info] GEMINI_API_KEY detected. Enabling live Gemini Flash autopsies."
    PROVIDER_ARGS="--provider openrouter --enable-autopsy"
else
    echo "[Notice] No API keys detected in environment. Running offline training with mock autopsies."
    PROVIDER_ARGS="--provider mock --enable-autopsy"
fi

uv run python scripts/run_benchmark.py \
    --mode random \
    --eval-type training \
    --run-id "$RUN_ID" \
    --episodes "$EPISODES" \
    --max-steps "$STEPS" \
    $PROVIDER_ARGS

echo ""
echo "Training complete. Telemetry saved to data/corp_telemetry.duckdb"
