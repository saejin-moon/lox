#!/usr/bin/env python3
"""
LOX 2.0 Policy Synthesis Engine: Dynamic Trigger Outer Loop.
Executes episodes, detects stalls and fatalities, invokes the Author Agent,
and evolves the Pythonic Behavior Tree in real-time.
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
from lox.envs.minihack import MiniHackAdapter


def run_synthesis_loop(
    provider: str = "mock",
    model: str | None = None,
    task: str = "MiniHack-ExploreMaze-Easy-Mapped-v0",
    max_generations: int = 3,
):
    print("=" * 60)
    print("LOX 2.0 Dynamic Policy Synthesis Engine Initialized")
    print(f"Provider: {provider} | Task: {task}")
    print("=" * 60)

    author = AuthorAgent(provider=provider, model=model)
    trigger_engine = DynamicTriggerEngine(stall_threshold=30, cluster_threshold=2)
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

        # Simulate turn execution and monitor triggers
        trigger_fired = False
        trigger_reason = ""

        for step in range(35):
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

            trig, reason = trigger_engine.check_turn(turns_on_level=step, depth=1)
            if trig == TriggerType.STALL:
                trigger_fired = True
                trigger_reason = reason
                print(f"[Trigger Fired] {reason}")
                break

        adapter.close()

        if trigger_fired:
            print("\n[Author Agent] Initiating synthesis session...")
            autopsy = recorder.generate_autopsy_report(death_reason=trigger_reason)
            new_code, tree, error = author.synthesize_policy(
                current_policy=current_policy,
                trigger_reason=trigger_reason,
                autopsy_report=autopsy,
            )

            if error:
                print(f"[Validation Failed] {error}")
            else:
                print(f"[Policy Verified & Compiled Successfully! Generation {gen} accepted]")
                current_policy = new_code
                print("\nEvolved Policy Program:")
                print(new_code.strip())

    print("\n" + "=" * 60)
    print("Synthesis Campaign Completed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="mock", choices=["mock", "gemini", "openrouter"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", default="MiniHack-ExploreMaze-Easy-Mapped-v0")
    parser.add_argument("--generations", type=int, default=2)
    args = parser.parse_args()

    run_synthesis_loop(
        provider=args.provider,
        model=args.model,
        task=args.task,
        max_generations=args.generations,
    )
