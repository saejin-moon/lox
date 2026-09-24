#!/usr/bin/env python3
"""
LOX-ψ: MiniHack Parallel Batch Evaluator.

Runs N seeded episodes of MiniHack across M worker processes in parallel,
aggregating rewards, success rates, step efficiency, and exploration coverage.
Sustains high-throughput evaluation (10k–50k eps/hr) across 30 cores.

Usage:
  uv run python scripts/run_minihack_batch.py --task MiniHack-ExploreMaze-Easy-Mapped-v0 --episodes 30 --jobs 10
  uv run python scripts/run_minihack_batch.py --task MiniHack-ExploreMaze-Hard-v0 --episodes 100 --jobs 30 --output data/minihack_batch.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lox.executor import get_adapter
from lox.policy.program import PolicyProgram


def _minihack_worker(worker_cfg: dict[str, Any]) -> dict[str, Any]:
    """Runs a slice of seeded MiniHack episodes within an isolated worker process."""
    seeds = worker_cfg["seeds"]
    task = worker_cfg["task"]
    max_steps = worker_cfg["max_steps"]
    program_dict = worker_cfg["program_dict"]

    adapter = get_adapter("minihack")
    program = PolicyProgram.from_dict(program_dict)

    results = []
    t0 = time.perf_counter()

    for seed in seeds:
        env = adapter.make_env(seed=seed, task=task)
        ep_t0 = time.perf_counter()
        res = adapter.run_episode(env, program, seed=seed, max_steps=max_steps)
        env.close()
        res["seed"] = seed
        res["wall_sec"] = time.perf_counter() - ep_t0
        results.append(res)

    return {
        "worker_id": worker_cfg["worker_id"],
        "results": results,
        "total_wall_sec": time.perf_counter() - t0,
    }


def run_minihack_batch(
    task: str = "MiniHack-ExploreMaze-Easy-Mapped-v0",
    episodes: int = 30,
    jobs: int = 8,
    max_steps: int = 1000,
    seed_base: int = 42,
    program_path: str | None = None,
    output: str | None = None,
) -> dict[str, Any]:
    adapter = get_adapter("minihack")

    # Load policy program
    if program_path is None:
        if os.path.exists("data/compiled/minihack.json"):
            program_path = "data/compiled/minihack.json"
        elif os.path.exists("data/policy_program_minihack.json"):
            program_path = "data/policy_program_minihack.json"
        else:
            raise FileNotFoundError("No MiniHack program found. Run compiler or provide --program-path.")

    program = PolicyProgram.load(program_path)
    program_dict = program.to_dict()

    num_workers = min(jobs, episodes)
    seeds = [seed_base + i for i in range(episodes)]
    seed_chunks = [seeds[i::num_workers] for i in range(num_workers)]

    worker_configs = [
        {
            "worker_id": i,
            "seeds": chunk,
            "task": task,
            "max_steps": max_steps,
            "program_dict": program_dict,
        }
        for i, chunk in enumerate(seed_chunks)
        if len(chunk) > 0
    ]

    print(
        f"[minihack-batch] Task: {task} | Program v{program.version} | "
        f"{episodes} eps across {len(worker_configs)} workers (step_limit={max_steps})"
    )

    t_start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(worker_configs)) as pool:
        worker_outputs = pool.map(_minihack_worker, worker_configs)
    total_wall_time = time.perf_counter() - t_start

    all_results = []
    for out in worker_outputs:
        all_results.extend(out["results"])

    all_results.sort(key=lambda r: r["seed"])

    rewards = [r["reward"] for r in all_results]
    success_steps = sorted(r["steps"] for r in all_results if r["success"])
    coverages = [r.get("coverage", 0) for r in all_results]

    success_rate = sum(1 for r in all_results if r["success"]) / max(1, len(all_results))
    eps_per_hr = (len(all_results) / total_wall_time) * 3600 if total_wall_time > 0 else 0.0

    summary = {
        "task": task,
        "program_version": program.version,
        "program_path": program_path,
        "episodes": len(all_results),
        "workers": len(worker_configs),
        "max_steps": max_steps,
        "total_wall_sec": round(total_wall_time, 2),
        "episodes_per_hour": round(eps_per_hr, 1),
        "success_rate": round(success_rate, 4),
        "mean_reward": round(statistics.mean(rewards), 4) if rewards else 0.0,
        "mean_steps": round(statistics.mean(r["steps"] for r in all_results), 2) if all_results else 0.0,
        "median_steps_on_success": statistics.median(success_steps) if success_steps else None,
        "mean_coverage": round(statistics.mean(coverages), 2) if coverages else 0.0,
        "results": all_results,
    }

    if output:
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[minihack-batch] Saved detailed summary to {output}")

    print(
        f"[minihack-batch] Done in {total_wall_time:.2f}s | Success Rate: {success_rate*100:.1f}% | "
        f"Mean Reward: {summary['mean_reward']} | Throughput: {eps_per_hr:.0f} eps/hr"
    )
    return summary


def main() -> int:
    p = argparse.ArgumentParser(description="LOX-ψ: MiniHack Parallel Batch Evaluator")
    p.add_argument("--task", type=str, default="MiniHack-ExploreMaze-Easy-Mapped-v0", help="MiniHack gym task ID")
    p.add_argument("--episodes", type=int, default=30, help="Total episodes to evaluate")
    p.add_argument("--jobs", type=int, default=8, help="Number of parallel worker processes")
    p.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    p.add_argument("--seed-base", type=int, default=42, help="Starting integer seed")
    p.add_argument("--program-path", type=str, default=None, help="Path to compiled policy program")
    p.add_argument("--output", type=str, default="data/minihack_batch.json", help="Path to write JSON results")
    args = p.parse_args()

    run_minihack_batch(
        task=args.task,
        episodes=args.episodes,
        jobs=args.jobs,
        max_steps=args.max_steps,
        seed_base=args.seed_base,
        program_path=args.program_path,
        output=args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
