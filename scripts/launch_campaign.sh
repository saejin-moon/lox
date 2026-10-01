#!/usr/bin/env bash
set -euo pipefail

cd /home/bae/lox
export PYTHONUNBUFFERED=1

echo "=================================================================" > synthesis_benchmark.log
echo "Starting LOX Batched Empirical Synthesis Campaign at $(date)" >> synthesis_benchmark.log
echo "Config: 50 generations | 50 episodes/gen | 10000 max turns" >> synthesis_benchmark.log
echo "=================================================================" >> synthesis_benchmark.log

/home/bae/lox/.venv/bin/python -u scripts/run_synthesis.py --provider openrouter --model google/gemma-4-31b-it --generations 50 --eval-episodes 50 --max-turns 10000 --policy-path data/latest_policy.py >> synthesis_benchmark.log 2>&1

echo "=================================================================" >> synthesis_benchmark.log
echo "Starting LOX 100-episode Autonomous NetHack Benchmark at $(date)" >> synthesis_benchmark.log
echo "Config: 100 episodes | 10000 max turns | Policy: data/latest_policy.py" >> synthesis_benchmark.log
echo "=================================================================" >> synthesis_benchmark.log

/home/bae/lox/.venv/bin/python -u scripts/run_nethack.py --episodes 100 --max-turns 10000 --policy-path data/latest_policy.py >> synthesis_benchmark.log 2>&1

echo "=================================================================" >> synthesis_benchmark.log
echo "Campaign and benchmark fully completed at $(date)" >> synthesis_benchmark.log
echo "=================================================================" >> synthesis_benchmark.log
