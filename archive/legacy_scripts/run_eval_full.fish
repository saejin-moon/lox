#!/usr/bin/env fish

# LOX-ψ Research-Grade Definitive Evaluation
# Evaluates against AutoAscend baselines across 100+ seeds
# Usage: ./scripts/run_eval_full.fish [mode] [episodes] [steps] [model]

set -l MODE "competence"
if test (count $argv) -ge 1
    set MODE $argv[1]
end

set -l EPISODES 100
if test (count $argv) -ge 2
    set EPISODES $argv[2]
end

set -l STEPS 50000
if test (count $argv) -ge 3
    set STEPS $argv[3]
end

set -l MODEL ""
if test (count $argv) -ge 4
    set MODEL $argv[4]
else if set -q MODEL
    set MODEL $MODEL
end

# Generate 6-character Base-62 run ID from date + time
set -l RUN_ID (uv run python scripts/gen_run_id.py)

echo "================================================================================"
echo "Starting LOX-ψ Research-Grade Evaluation: Mode=$MODE | $EPISODES Episodes"
echo "Step Budget: $STEPS | Run ID: $RUN_ID"
echo "================================================================================"

set -l PROVIDER_ARGS "--provider" "mock"
if set -q OPENROUTER_API_KEY
    echo "[Info] Live OpenRouter API key detected."
    set PROVIDER_ARGS "--provider" "openrouter" "--enable-autopsy"
else if set -q GEMINI_API_KEY
    echo "[Info] Live Gemini API key detected."
    set PROVIDER_ARGS "--provider" "gemini" "--enable-autopsy"
else
    echo "[Notice] Running with offline deterministic autopsies."
    set PROVIDER_ARGS "--provider" "mock" "--enable-autopsy"
end

set -l MODEL_ARGS
if test -n "$MODEL"
    echo "[Info] Selected model override: $MODEL"
    set MODEL_ARGS "--model" "$MODEL"
end

uv run python scripts/run_benchmark.py \
    --mode $MODE \
    --eval-type research_grade \
    --run-id $RUN_ID \
    --episodes $EPISODES \
    --max-steps $STEPS \
    $PROVIDER_ARGS \
    $MODEL_ARGS

echo ""
echo "Research evaluation complete. Results consolidated into data/lox_telemetry.duckdb"
