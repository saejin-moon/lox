#!/usr/bin/env python3
"""
LOX-ψ: Depth-10 Autonomous Campaign Driver.

Iteratively drives policy synthesis and parallel evaluation until the agent
achieves a median dungeon depth >= 10.0 (matching/beating AutoAscend).

Each iteration:
1. Evaluates current policy program across 30 CPU workers in parallel.
2. Consolidates DuckDB telemetry (autopsies, lethal taxonomy, failure counters).
3. Evaluates stopping criterion (median_depth >= 10.0).
4. Invokes Gemma 4 31B author agent to synthesize pure Pythonic policy diffs
   addressing the specific death taxonomy observed.
5. Gates and compiles accepted diffs into the live policy program.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.deliberative.providers.openrouter import OpenRouterProvider
from lox.deliberative.providers.gemini import GeminiProvider
from lox.policy.author_agent import author_session
from lox.policy.ledger import RevisionLedger, DEFAULT_LEDGER_PATH
from lox.policy.manifest import build_manifest
from lox.policy.program import PolicyProgram, DEFAULT_PROGRAM_PATH
from lox.policy.report import build_report_bundle, DEFAULT_DB_PATH
from lox.policy.validator import ValidatorHooks


def run_campaign(
    target_depth: float = 10.0,
    episodes: int = 30,
    max_steps: int = 15000,
    jobs: int = 30,
    role: str = "valkyrie",
    provider_name: str = "openrouter",
    model_name: str = "google/gemma-4-31b-it",
    max_iterations: int = 50,
    output_log: str = "data/depth10_campaign_log.jsonl",
) -> None:
    os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.dirname(output_log) or ".", exist_ok=True)

    print("=" * 80)
    print(f"LOX-ψ: Depth-10 Campaign Loop Starting")
    print(f"Target: Median Depth >= {target_depth} | Model: {model_name} ({provider_name})")
    print(f"Batch: {episodes} eps @ {max_steps} steps | Jobs: {jobs} workers | Role: {role}")
    print("=" * 80)

    # Initialize provider
    if provider_name == "gemini":
        provider = GeminiProvider(model=model_name, timeout=180.0)
    elif provider_name == "openrouter":
        provider = OpenRouterProvider(model=model_name, timeout=180.0)
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    start_wall = time.perf_counter()

    for iteration in range(1, max_iterations + 1):
        program_path = "data/compiled/nethack.json"
        if not os.path.exists(program_path):
            program_path = DEFAULT_PROGRAM_PATH

        live_prog = PolicyProgram.load(program_path)
        print(f"\n>>> [CAMPAIGN ITERATION {iteration}/{max_iterations}] Evaluating Program v{live_prog.version}...")

        # 1. Run Parallel Batch Evaluation
        batch_output = f"data/campaign_batch_iter{iteration}_v{live_prog.version}.json"
        batch_cmd = [
            sys.executable, "scripts/run_parallel_batch.py",
            "--role", role,
            "--episodes", str(episodes),
            "--max-steps", str(max_steps),
            "--jobs", str(jobs),
            "--output", batch_output,
            "--program-path", program_path,
        ]

        t0_eval = time.perf_counter()
        ret = subprocess.run(batch_cmd)
        eval_sec = time.perf_counter() - t0_eval

        if ret.returncode != 0 or not os.path.exists(batch_output):
            print(f"[campaign] Batch evaluation failed (exit {ret.returncode}). Retrying iteration.")
            time.sleep(5)
            continue

        with open(batch_output, "r", encoding="utf-8") as f:
            batch_data = json.load(f)

        med_depth = batch_data.get("median_depth", 0.0)
        mean_score = batch_data.get("mean_score", 0.0)
        mean_turns = batch_data.get("mean_turns", 0.0)
        asc_rate = batch_data.get("ascension_rate", 0.0)
        ok_eps = [r for r in batch_data.get("detailed_episodes", []) if "error" not in r]
        max_depth = max([r.get("max_depth", 0) for r in ok_eps] or [0])
        survivors = sum(1 for r in ok_eps if r.get("death_message") == "Survived")
        surv_rate = (survivors / len(ok_eps)) if ok_eps else 0.0

        # Log iteration record
        iter_record = {
            "iteration": iteration,
            "timestamp": time.time(),
            "program_version": live_prog.version,
            "median_depth": med_depth,
            "max_depth": max_depth,
            "mean_score": mean_score,
            "mean_turns": mean_turns,
            "survival_rate": surv_rate,
            "ascension_rate": asc_rate,
            "eval_wall_sec": round(eval_sec, 2),
            "total_campaign_sec": round(time.perf_counter() - start_wall, 2),
        }

        with open(output_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(iter_record) + "\n")

        print(f"\n[CAMPAIGN METRICS Iteration {iteration} | Program v{live_prog.version}]")
        print(f"  -> Median Depth:  {med_depth:.1f} (Target: >= {target_depth})")
        print(f"  -> Max Depth:     {max_depth}")
        print(f"  -> Mean Score:    {mean_score:.1f}")
        print(f"  -> Survival Rate: {surv_rate * 100:.1f}% ({survivors}/{len(ok_eps)})")
        print(f"  -> Eval Throughput: {eval_sec:.1f}s ({episodes / eval_sec * 3600:.0f} eps/hr)")

        # Check Termination Goal
        if med_depth >= target_depth:
            print("=" * 80)
            print(f"🏆 SUCCESS: TARGET REACHED! Median Depth = {med_depth:.1f} >= {target_depth}!")
            print(f"Winning Program Version: v{live_prog.version}")
            print(f"Total Campaign Time: {(time.perf_counter() - start_wall)/60:.1f} minutes across {iteration} iterations.")
            print("=" * 80)
            break

        # 2. Consolidate Telemetry
        subprocess.run([sys.executable, "scripts/clean_telemetry.py"], capture_output=True)

        # 3. Invoke LLM Author Agent (Gemma 4 31B) to synthesize next policy revision
        print(f"\n>>> [CAMPAIGN ITERATION {iteration}] Invoking {model_name} to synthesize Policy v{live_prog.version + 1}...")
        author_cmd = [
            sys.executable, "scripts/run_author_session.py",
            "--provider", provider_name,
            "--model", model_name,
            "--domain", "nethack",
            "--commit",
            "--out", f"data/campaign_author_iter{iteration}.json",
        ]

        t0_author = time.perf_counter()
        auth_ret = subprocess.run(author_cmd)
        auth_sec = time.perf_counter() - t0_author

        if auth_ret.returncode == 0:
            new_prog = PolicyProgram.load(program_path)
            print(f"  -> Policy Diff ACCEPTED! Advanced to Program v{new_prog.version} in {auth_sec:.1f}s.")
        else:
            print(f"  -> Policy Diff REJECTED in {auth_sec:.1f}s. Performing mutational parameter perturbation on survival/descent knobs...")
            from scripts.run_evolution import mutate_program_params  # noqa: PLC0415
            mutated_prog = mutate_program_params(live_prog, "nethack")
            mutated_prog.save(program_path)
            mutated_prog.save("data/compiled/nethack.json")
            mutated_prog.save(f"data/programs/archive/policy_program_v{mutated_prog.version}.json")
            print(f"  -> Mutational fallback created and compiled Program v{mutated_prog.version}.")


def main():
    parser = argparse.ArgumentParser(description="LOX-ψ: Depth-10 Autonomous Campaign Loop")
    parser.add_argument("--target-depth", type=float, default=10.0)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=15000)
    parser.add_argument("--jobs", type=int, default=30)
    parser.add_argument("--role", type=str, default="valkyrie")
    parser.add_argument("--provider", type=str, default="openrouter", choices=["openrouter", "gemini"])
    parser.add_argument("--model", type=str, default="google/gemma-4-31b-it")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--output-log", type=str, default="data/depth10_campaign_log.jsonl")
    args = parser.parse_args()

    run_campaign(
        target_depth=args.target_depth,
        episodes=args.episodes,
        max_steps=args.max_steps,
        jobs=args.jobs,
        role=args.role,
        provider_name=args.provider,
        model_name=args.model,
        max_iterations=args.max_iterations,
        output_log=args.output_log,
    )


if __name__ == "__main__":
    main()
