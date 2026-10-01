#!/usr/bin/env bash
set -euo pipefail

cd /home/bae/lox
export PYTHONUNBUFFERED=1

echo "=================================================================" > synthesis_benchmark.log
echo "Starting LOX 500-generation synthesis campaign at $(date)" >> synthesis_benchmark.log
echo "=================================================================" >> synthesis_benchmark.log

/home/bae/lox/.venv/bin/python -u scripts/run_synthesis.py --provider openrouter --model google/gemma-4-31b-it --generations 500 --eval-episodes 2 >> synthesis_benchmark.log 2>&1

echo "=================================================================" >> synthesis_benchmark.log
echo "Starting LOX 100-episode benchmark at $(date)" >> synthesis_benchmark.log
echo "=================================================================" >> synthesis_benchmark.log

/home/bae/lox/.venv/bin/python -u scripts/run_nethack.py --episodes 100 >> synthesis_benchmark.log 2>&1

echo "=================================================================" >> synthesis_benchmark.log
echo "Campaign and benchmark fully completed at $(date)" >> synthesis_benchmark.log
echo "=================================================================" >> synthesis_benchmark.log
