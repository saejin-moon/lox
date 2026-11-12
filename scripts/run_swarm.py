#!/usr/bin/env python3
"""
CLI Entrypoint for the LOX Hierarchical Multi-Agent Swarm Campaign.
Coordinates dynamic macro-category discovery, concurrent worker exploration on frontier checkpoints,
and Behavior Tree policy synthesis.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.db.schema import DEFAULT_DB_PATH
from lox.engine.swarm_orchestrator import (
    DEFAULT_FRONTIER_DIR,
    DEFAULT_WORKSPACE_DIR,
    SwarmCampaignOrchestrator,
)
from lox.ui.console import ui


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LOX Hierarchical Multi-Agent Swarm Campaign Entrypoint."
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=5,
        help="Number of swarm synthesis generations to execute (default: 5).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker processes / category specialists (default: 4).",
    )
    parser.add_argument(
        "--turns-per-worker",
        type=int,
        default=3,
        help="Number of exploration & theory turns per category worker (default: 3).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=8,
        help="Number of seeds to evaluate per generation master batch (default: 8).",
    )
    parser.add_argument(
        "--provider",
        choices=["openrouter", "deepseek", "anthropic", "openai", "gemini", "mock"],
        default="openrouter",
        help="LLM API provider for Author Agents (default: openrouter).",
    )
    parser.add_argument(
        "--model",
        default="z-ai/glm-5.3-flash:nitro",
        help="LLM model identifier (default: z-ai/glm-5.3-flash:nitro).",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"Path to DuckDB database (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--workspace-dir",
        default=DEFAULT_WORKSPACE_DIR,
        help=f"Path to workspace directory (default: {DEFAULT_WORKSPACE_DIR}).",
    )
    parser.add_argument(
        "--frontier-dir",
        default=DEFAULT_FRONTIER_DIR,
        help=f"Path to frontier checkpoint directory (default: {DEFAULT_FRONTIER_DIR}).",
    )
    parser.add_argument(
        "--benchmark",
        default="nethack",
        help="Environment benchmark (default: nethack).",
    )
    parser.add_argument(
        "--role",
        default="val-hum-law-fem",
        help="Character role/race/alignment/gender (default: val-hum-law-fem).",
    )

    args = parser.parse_args()

    ui.header("LOX Hierarchical Swarm Synthesis Engine")
    ui.info(
        f"Config: {args.generations} generations | {args.workers} workers | "
        f"{args.turns_per_worker} turns/worker | Provider: {args.provider} ({args.model})"
    )

    orchestrator = SwarmCampaignOrchestrator(
        db_path=args.db_path,
        workspace_dir=args.workspace_dir,
        frontier_dir=args.frontier_dir,
        provider=args.provider,
        model=args.model,
        num_workers=args.workers,
        turns_per_worker=args.turns_per_worker,
        seeds_per_batch=args.seeds,
        benchmark=args.benchmark,
        role=args.role,
    )

    try:
        results = orchestrator.run(generations=args.generations)
        ui.header("Campaign Results Summary")
        for r in results:
            ui.info(
                f"Gen {r['generation']}: Avg Depth={r['avg_depth']:.2f} (Max={r['max_depth']}) | "
                f"Avg Score={r['avg_score']:.1f} | New Frontier CPs={r['new_frontier_checkpoints']} | "
                f"Time={r['elapsed_sec']:.1f}s"
            )
    except KeyboardInterrupt:
        ui.warn("\nInterrupted by user. Shutting down cleanly...")
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
