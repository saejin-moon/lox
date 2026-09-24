#!/usr/bin/env bash
set -e

# LOX-ψ Research-Grade Definitive Evaluation
# Evaluates against AutoAscend baselines across 100+ seeds
# Usage: ./scripts/run_eval_full.sh [mode] [episodes] [steps] [model]

MODE=${1:-competence}
EPISODES=${2:-100}
STEPS=${3:-50000}
MODEL=${4:-${MODEL:-}}

# Generate 6-character Base-62 run ID from date + time
RUN_ID=$(uv run python scripts/gen_run_id.py)

echo "================================================================================"
echo "Starting LOX-ψ Research-Grade Evaluation: Mode=$MODE | $EPISODES Episodes"
echo "Step Budget: $STEPS | Run ID: $RUN_ID"
echo "================================================================================"

PROVIDER_ARGS="--provider mock"
if [ -n "$OPENROUTER_API_KEY" ]; then
    echo "[Info] Live OpenRouter API key detected."
    PROVIDER_ARGS="--provider openrouter --enable-autopsy"
elif [ -n "$GEMINI_API_KEY" ]; then
    echo "[Info] Live Gemini API key detected."
    PROVIDER_ARGS="--provider gemini --enable-autopsy"
else
    echo "[Notice] Running with offline deterministic autopsies."
    PROVIDER_ARGS="--provider mock --enable-autopsy"
fi

MODEL_ARGS=""
if [ -n "$MODEL" ]; then
    echo "[Info] Selected model override: $MODEL"
    MODEL_ARGS="--model $MODEL"
fi

uv run python scripts/run_benchmark.py \
    --mode "$MODE" \
    --eval-type research_grade \
    --run-id "$RUN_ID" \
    --episodes "$EPISODES" \
    --max-steps "$STEPS" \
    $PROVIDER_ARGS \
    $MODEL_ARGS

echo ""
echo "Research evaluation complete. Results consolidated into data/lox_telemetry.duckdb"
