#!/usr/bin/env fish

# CORP Research-Grade Definitive Evaluation
# Evaluates against AutoAscend baselines across 100+ seeds
# Fish shell frontend

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

set -l RUN_ID "research_grade_"$MODE"_" (date +%s)

echo "================================================================================"
echo "Starting CORP Research-Grade Evaluation: Mode=$MODE | $EPISODES Episodes"
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

uv run python scripts/run_benchmark.py \
    --mode $MODE \
    --eval-type research_grade \
    --run-id $RUN_ID \
    --episodes $EPISODES \
    --max-steps $STEPS \
    $PROVIDER_ARGS

echo ""
echo "Research evaluation complete. Results consolidated into data/corp_telemetry.duckdb"
