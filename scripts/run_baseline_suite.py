"""
Unified AutoAscend vs CORP Baseline Comparison Suite.

Runs both agents on identical seeds and step budgets for apples-to-apples comparison,
producing a unified comparison table and saving results to data/baseline_comparison.json.

Usage:
    uv run python scripts/run_baseline_suite.py --episodes 100 --step-limit 50000 --role val
    uv run python scripts/run_baseline_suite.py --episodes 10 --step-limit 20000 --role val --quick
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_autoascend(episodes: int, role: str, seed: int, step_limit: int, duckdb_path: str) -> dict:
    """Run AutoAscend episodes via the in-process harness and return aggregate stats."""
    from run_autoascend_eval import run_autoascend_episode

    summaries = []
    for i in range(episodes):
        ep_seed = seed + i
        print(f"[AutoAscend] Episode {i + 1}/{episodes} (seed={ep_seed}) ...", flush=True)
        summaries.append(run_autoascend_episode(role=role, seed=ep_seed, step_limit=step_limit))
        s = summaries[-1]
        print(
            f"  Turns: {int(s.get('turns', 0)):6d} | Depth: {int(s.get('level_num', 1)):2d} | "
            f"Score: {int(s.get('score', 0)):6d} | {s.get('end_reason', '?')[:40]}",
            flush=True,
        )
    return {
        "agent": "autoascend",
        "role": role,
        "episodes": episodes,
        "step_limit": step_limit,
        "median_depth": float(np.median([s.get('level_num', 1) for s in summaries])),
        "max_depth": int(max(s.get('level_num', 1) for s in summaries)),
        "mean_score": float(np.mean([s.get('score', 0) for s in summaries])),
        "mean_turns": float(np.mean([s.get('turns', 0) for s in summaries])),
        "ascension_rate": float(sum(1 for s in summaries if s.get('ascended')) / max(1, len(summaries))),
        "detailed_episodes": summaries,
    }


def run_corp(episodes: int, role: str, seed: int, step_limit: int, duckdb_path: str) -> dict:
    """Run CORP episodes via run_benchmark's episode runner and return aggregate stats."""
    from corp.env.nle_wrapper import make_env
    from corp.agent.corp_agent import CORPAgent
    from corp.planner.nogood import NogoodStore
    from corp.deliberative.providers.mock_provider import MockProvider
    from corp.telemetry import ParquetLogger

    ROLE_CHARACTER_MAP = {
        "val": "val-hum-law-fem",
        "valkyrie": "val-hum-law-fem",
        "bar": "bar-hum-neu-mal",
        "barbarian": "bar-hum-neu-mal",
        "sam": "sam-hum-law-mal",
        "samurai": "sam-hum-law-mal",
    }
    character = ROLE_CHARACTER_MAP.get(role.lower(), role)

    nogood_store = NogoodStore()
    parquet_logger = ParquetLogger(base_dir="logs/parquet", run_id=f"cmp_{int(time.time())}")
    results = []
    for i in range(episodes):
        print(f"[CORP] Episode {i + 1}/{episodes} (seed={seed + i}) ...", flush=True)
        env = make_env(character=character)
        agent = CORPAgent(
            env=env,
            nogood_store=nogood_store,
            llm_provider=MockProvider(),
            parquet_logger=parquet_logger,
            eval_type="baseline_comparison",
            mode="random",
            seed=seed + i,
            role=role,
        )
        result = agent.run_episode(max_steps=step_limit)
        results.append(result)
        print(
            f"  Turns: {result.turns:6d} | Depth: {result.max_depth:2d} | "
            f"Score: {result.final_score:6d} | {result.death_message[:40] if result.death_message else 'Survived'}",
            flush=True,
        )
    parquet_logger.flush_all()

    depths = [r.max_depth for r in results]
    return {
        "agent": "corp",
        "role": role,
        "episodes": episodes,
        "step_limit": step_limit,
        "median_depth": float(np.median(depths)),
        "max_depth": int(max(depths)),
        "mean_score": float(np.mean([r.final_score for r in results])),
        "mean_turns": float(np.mean([r.turns for r in results])),
        "ascension_rate": float(sum(1 for r in results if r.is_ascended) / max(1, len(results))),
        "mean_sps": float(np.mean([r.steps_per_second for r in results])),
        "detailed_episodes": [
            {
                "seed": seed + i,
                "turns": r.turns,
                "max_depth": r.max_depth,
                "score": r.final_score,
                "ascended": r.is_ascended,
                "death_message": r.death_message,
            }
            for i, r in enumerate(results)
        ],
    }


def print_comparison(autoascend: dict, corp: dict) -> None:
    print("\n" + "=" * 84)
    print("UNIFIED BASELINE COMPARISON: AutoAscend vs CORP (identical seeds & step budget)")
    print("=" * 84)
    print(f"{'Metric':<28} | {'AutoAscend':>26} | {'CORP':>26}")
    print("-" * 84)
    rows = [
        ("Episodes", f"{autoascend['episodes']} @ {autoascend['step_limit']} steps", f"{corp['episodes']} @ {corp['step_limit']} steps"),
        ("Median Dungeon Depth", f"{autoascend['median_depth']:.1f}", f"{corp['median_depth']:.1f}"),
        ("Max Dungeon Depth", f"{autoascend['max_depth']}", f"{corp['max_depth']}"),
        ("Mean Score", f"{autoascend['mean_score']:.1f}", f"{corp['mean_score']:.1f}"),
        ("Mean Turns", f"{autoascend['mean_turns']:.1f}", f"{corp['mean_turns']:.1f}"),
        ("Ascension Rate", f"{autoascend['ascension_rate']*100:.1f}%", f"{corp['ascension_rate']*100:.1f}%"),
    ]
    if "mean_sps" in corp:
        rows.append(("Mean Throughput (SPS)", "N/A", f"{corp['mean_sps']:.1f}"))
    for name, a, c in rows:
        print(f"{name:<28} | {a:>26} | {c:>26}")
    print("=" * 84)


def main():
    parser = argparse.ArgumentParser(description="Unified AutoAscend vs CORP baseline comparison")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per agent")
    parser.add_argument("--step-limit", type=int, default=20000, help="Step budget per episode (AutoAscend default: 50000)")
    parser.add_argument("--role", type=str, default="val", help="Role for both agents (val, bar, sam, pri, mon)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--agents", choices=["both", "autoascend", "corp"], default="both", help="Which agent(s) to run")
    parser.add_argument("--output", type=str, default="data/baseline_comparison.json", help="Path for the comparison JSON")
    args = parser.parse_args()

    started = time.time()
    autoascend = corp = None
    if args.agents in ("both", "autoascend"):
        autoascend = run_autoascend(args.episodes, args.role, args.seed, args.step_limit, args.output)
    if args.agents in ("both", "corp"):
        corp = run_corp(args.episodes, args.role, args.seed, args.step_limit, args.output)

    if autoascend and corp:
        print_comparison(autoascend, corp)
    elif autoascend:
        print_comparison(autoascend, {"episodes": 0, "step_limit": args.step_limit, "median_depth": 0, "max_depth": 0, "mean_score": 0, "mean_turns": 0, "ascension_rate": 0})
    elif corp:
        print_comparison({"episodes": 0, "step_limit": args.step_limit, "median_depth": 0, "max_depth": 0, "mean_score": 0, "mean_turns": 0, "ascension_rate": 0}, corp)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": time.time() - started,
        "step_limit": args.step_limit,
        "role": args.role,
        "base_seed": args.seed,
        "autoascend": autoascend,
        "corp": corp,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Comparison written to {args.output}")


if __name__ == "__main__":
    main()
