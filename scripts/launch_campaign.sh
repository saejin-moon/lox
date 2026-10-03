#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

echo "=================================================================" > synthesis_benchmark.log
echo "Starting LOX Batched Empirical Synthesis Campaign at $(date)" >> synthesis_benchmark.log
echo "Config: 15 generations | 20 episodes/gen | 25000 max turns" >> synthesis_benchmark.log
echo "=================================================================" >> synthesis_benchmark.log

uv run python -u -m scripts.run_synthesis --provider openrouter --model google/gemma-4-31b-it --generations 15 --eval-episodes 20 --max-turns 25000 --target-depth 10.0 --policy-path data/latest_policy.py >> synthesis_benchmark.log 2>&1

echo "=================================================================" >> synthesis_benchmark.log
echo "Starting LOX Autonomous NetHack Benchmark at $(date)" >> synthesis_benchmark.log
echo "Config: 20 episodes | 25000 max turns | Policy: data/latest_policy.py" >> synthesis_benchmark.log
echo "=================================================================" >> synthesis_benchmark.log

uv run python -u -m scripts.run_nethack --episodes 20 --max-turns 25000 --policy-path data/latest_policy.py >> synthesis_benchmark.log 2>&1

echo "=================================================================" >> synthesis_benchmark.log
echo "Campaign and benchmark fully completed at $(date)" >> synthesis_benchmark.log
echo "=================================================================" >> synthesis_benchmark.log
