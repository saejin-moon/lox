#!/usr/bin/env bash
set -e

# CORP Research-Grade Definitive Evaluation
# Evaluates against AutoAscend baselines across 100+ seeds
# Bash frontend

MODE=${1:-competence}
EPISODES=${2:-100}
STEPS=${3:-50000}
RUN_ID="research_grade_${MODE}_$(date +%s)"

echo "================================================================================"
echo "Starting CORP Research-Grade Evaluation: Mode=$MODE | $EPISODES Episodes"
echo "Step Budget: $STEPS | Run ID: $RUN_ID"
echo "================================================================================"

PROVIDER_ARGS="--provider mock"
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "[Info] Live OpenRouter API key detected."
    PROVIDER_ARGS="--provider openrouter --enable-autopsy"
elif [ -n "$GEMINI_API_KEY" ]; then
    echo "[Info] Live Gemini API key detected."
    PROVIDER_ARGS="--provider openrouter --enable-autopsy"
else
    echo "[Notice] Running with offline deterministic autopsies."
    PROVIDER_ARGS="--provider mock --enable-autopsy"
fi

uv run python scripts/run_benchmark.py \
    --mode "$MODE" \
    --eval-type research_grade \
    --run-id "$RUN_ID" \
    --episodes "$EPISODES" \
    --max-steps "$STEPS" \
    $PROVIDER_ARGS

echo ""
echo "Research evaluation complete. Results consolidated into data/corp_telemetry.duckdb"
