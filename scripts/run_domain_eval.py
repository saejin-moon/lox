#!/usr/bin/env python3
"""
CORP-Ω R5: transfer-domain batch evaluator (adapter-aware benchmark).

Runs N seeded episodes of a transfer domain (minihack today; craftax when its adapter
lands) against a policy program, writing benchmark-compatible JSON. This is the
transfer-domain analogue of scripts/run_benchmark.py.

  uv run python scripts/run_domain_eval.py --domain minihack --episodes 10 --seed-base 0
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from corp.executor import get_adapter, available_domains  # noqa: E402
from corp.policy.program import PolicyProgram  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="CORP R5 transfer-domain batch evaluator")
    p.add_argument("--domain", type=str, default="minihack", choices=available_domains() or ["minihack"])
    p.add_argument("--task", type=str, default=None, help="override the adapter's default task")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed-base", type=int, default=0, help="seeds are seed_base + i")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--program-path", type=str, default=None)
    p.add_argument("--output", type=str, default="data/domain_eval.json")
    args = p.parse_args()

    adapter = get_adapter(args.domain)
    program_path = args.program_path or f"data/policy_program_{args.domain}.json"
    program = PolicyProgram.load(program_path)
    max_steps = args.max_steps or adapter.spec.step_limit_default
    print(f"[domain-eval] {args.domain} | program v{program.version} | "
          f"{args.episodes} eps × {max_steps} steps (seeded)")

    results = []
    for i in range(args.episodes):
        seed = args.seed_base + i
        env = adapter.make_env(seed=seed, task=args.task)
        t0 = time.perf_counter()
        r = adapter.run_episode(env, program, seed=seed, max_steps=max_steps)
        env.close()
        r["seed"] = seed
        r["wall_sec"] = time.perf_counter() - t0
        results.append(r)
        print(f"  Ep {i + 1:3d}/{args.episodes} [seed {seed}] "
              f"reward={r['reward']:.2f} steps={r['steps']} "
              f"success={r['success']} coverage={r.get('coverage', '-')}")

    rewards = [r["reward"] for r in results]
    success_steps = sorted(r["steps"] for r in results if r["success"])
    summary = {
        "domain": args.domain,
        "program_version": program.version,
        "episodes": args.episodes,
        "max_steps": max_steps,
        "mean_reward": statistics.mean(rewards) if rewards else 0.0,
        "success_rate": sum(1 for r in results if r["success"]) / max(1, len(results)),
        "mean_steps": statistics.mean(r["steps"] for r in results),
        # R5 metric-gap fix: saturated rewards need step-efficiency to compare revisions
        "median_steps_on_success": statistics.median(success_steps) if success_steps else None,
        "mean_steps_on_success": statistics.mean(success_steps) if success_steps else None,
        "detailed_episodes": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[domain-eval] mean_reward={summary['mean_reward']:.3f} "
          f"success_rate={summary['success_rate']:.2f} "
          f"median_steps_on_success={summary['median_steps_on_success']} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())