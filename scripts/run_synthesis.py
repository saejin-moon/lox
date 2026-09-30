#!/usr/bin/env python3
"""
LOX 2.0 Policy Synthesis Engine: Dynamic Trigger Outer Loop.
Executes episodes, detects stalls and fatalities, invokes the Author Agent,
queries DuckDB for empirical validation, logs token usage, and evolves the Behavior Tree.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.author.agent import AuthorAgent
from lox.telemetry.recorder import FlightRecorder
from lox.telemetry.triggers import DynamicTriggerEngine, TriggerType
from lox.telemetry.tokens import get_token_usage_summary
from lox.envs.minihack import MiniHackAdapter


def run_synthesis_loop(
    provider: str = "mock",
    model: str | None = None,
    task: str = "MiniHack-ExploreMaze-Easy-Mapped-v0",
    max_generations: int = 2,
    db_path: str = "data/lox.duckdb",
):
    print("=" * 65)
    print("LOX 2.0 Dynamic Policy Synthesis Engine")
    print(f"Provider: {provider} | Task: {task}")
    print(f"Database: {db_path}")
    print("=" * 65)

    run_id = f"synth_{provider}_{int(time.time())}"
    author = AuthorAgent(provider=provider, model=model, db_path=db_path)
    trigger_engine = DynamicTriggerEngine(stall_threshold=25, cluster_threshold=2)
    recorder = FlightRecorder(capacity=50)

    # Initial minimal policy
    current_policy = """
def explore():
    step_to_frontier()

plan = [
    explore,
]
"""
    print("\n[Generation 0] Initial Seed Policy:")
    print(current_policy.strip())

    for gen in range(1, max_generations + 1):
        print(f"\n--- Running Generation {gen} Evaluation ---")
        adapter = MiniHackAdapter(task=task)
        obs = adapter.reset(seed=gen * 100)

        trigger_fired = False
        trigger_reason = ""

        for step in range(30):
            recorder.record_turn(
                turn=step,
                depth=1,
                hp=obs.hero.hp,
                max_hp=obs.hero.max_hp,
                hunger=obs.hero.hunger_state.name,
                pos=(obs.hero.y, obs.hero.x),
                action_name="step_to_frontier",
                message=obs.message,
            )

            trig, reason = trigger_engine.check_turn(turns_on_level=step, depth=1, has_frontier=False)
            if trig == TriggerType.STALL:
                trigger_fired = True
                trigger_reason = reason
                print(f"[Trigger Fired] {reason}")
                break

        adapter.close()

        if trigger_fired:
            print("\n[Author Agent] Initiating synthesis session (querying DuckDB & evaluating)...")
            status_rep = recorder.generate_compact_status_report(trigger_reason=trigger_reason)
            new_code, tree, error = author.synthesize_policy(
                current_policy=current_policy,
                trigger_reason=trigger_reason,
                status_report=status_rep,
                run_id=run_id,
            )

            if error:
                print(f"[Validation Failed] {error}")
            else:
                print(f"[Policy Verified & Compiled! Generation {gen} accepted]")
                current_policy = new_code
                print("\nEvolved Policy Program:")
                print(new_code.strip())

    # Report token expenditure for the synthesis run
    token_stats = get_token_usage_summary(run_id=run_id, db_path=db_path)
    print("\n" + "=" * 65)
    print("Synthesis Session Token & Cost Accounting:")
    print(f"- LLM Synthesis Sessions: {token_stats['total_calls']}")
    print(f"- Prompt Tokens:          {token_stats['prompt_tokens']:,}")
    print(f"- Completion Tokens:      {token_stats['completion_tokens']:,}")
    print(f"- Total Tokens:           {token_stats['total_tokens']:,}")
    print(f"- Estimated Cost:         ${token_stats['total_cost_usd']:.4f} USD")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="mock", choices=["mock", "gemini", "openrouter"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default="MiniHack-ExploreMaze-Easy-Mapped-v0")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--db-path", default="data/lox.duckdb")
    args = parser.parse_args()

    run_synthesis_loop(
        provider=args.provider,
        model=args.model,
        task=args.task,
        max_generations=args.generations,
        db_path=args.db_path,
    )
