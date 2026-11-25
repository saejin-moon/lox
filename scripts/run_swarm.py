#!/usr/bin/env python3
"""
CLI Entrypoint for the LOX Layout A (`2R + 2P`) Option-A Peer Swarm Campaign.
Coordinates orthogonal `DeficitAnalyzer` + `SwarmDirectivePlanner` scoping,
concurrent Peer Agent exploration on frontier checkpoints & `expert_ticks`,
and Split-Sample Paired CRN tournament promotion (`SwarmReferee`).
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LOX Option-A Peer Swarm Synthesis Campaign Entrypoint."
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
        help="Number of concurrent LLM peer agents (`reactive_1..2`, `proactive_1..2`, default: 4).",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=None,
        help="Number of NLE CPU rollout worker processes (default: auto up to 16 based on seeds & CPU cores).",
    )
    parser.add_argument(
        "--turns-per-worker",
        "--hypotheses-per-worker",
        dest="turns_per_worker",
        type=int,
        default=3,
        help="Number of synthesis hypotheses / micro-pilots budget per peer agent (default: 3).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=64,
        help="Full CRN tournament seeds per generation (default: 64, two-stage raced when --race-seeds < --seeds).",
    )
    parser.add_argument(
        "--race-seeds",
        type=int,
        default=32,
        help="Stage-1 CRN racing seed count before extending finalists to --seeds (default: 32).",
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
        "--refactor-model",
        default=None,
        help="Optional stronger LLM model identifier for refactor/plateau slots (defaults to --model).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "max", "none"],
        default="medium",
        help="OpenRouter reasoning effort parameter (default: medium).",
    )
    parser.add_argument(
        "--routing",
        choices=["nitro", "floor", "exacto", "default"],
        default="nitro",
        help="OpenRouter provider routing strategy suffix (default: nitro).",
    )
    parser.add_argument(
        "--scenario-gate",
        choices=["off", "advisory", "strict"],
        default="off",
        help="Optional synthetic scenario gate mode (default: off; organic FrontierArchive is always preferred).",
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
        default=None,
        help="Path to frontier checkpoint directory (defaults to <workspace-dir>/checkpoints when custom workspace is set, else data/frontier).",
    )
    parser.add_argument(
        "--role",
        default="val-hum-law-fem",
        help="Character role/race/alignment/gender (default: val-hum-law-fem).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Reset campaign tables in DuckDB and workspace artifacts (policy.py, skills/) "
            "to Generation 0 clean state before running, while preserving expert_runs and wiki."
        ),
    )

    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if not getattr(args, "refactor_model", None):
        args.refactor_model = args.model
    return args


def main() -> None:
    args = parse_args()
    if not args.frontier_dir:
        args.frontier_dir = (
            os.path.join(args.workspace_dir, "checkpoints")
            if os.path.abspath(args.workspace_dir)
            != os.path.abspath(DEFAULT_WORKSPACE_DIR)
            else DEFAULT_FRONTIER_DIR
        )

    if args.clean:
        from scripts.reset_clean import reset_clean

        ui.warn(
            f"--clean specified: Resetting workspace `{args.workspace_dir}` and campaign tables in `{args.db_path}`..."
        )
        reset_clean(
            db_path=args.db_path,
            policy_path=os.path.join(args.workspace_dir, "policy.py"),
            frontier_dir=args.frontier_dir,
        )

    ui.header("LOX Peer Swarm Synthesis Engine (Layout A: 2 Reactive + 2 Proactive)")
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
        role=args.role,
        eval_workers=args.eval_workers,
        reasoning_effort=args.reasoning_effort,
        routing=args.routing,
        refactor_model=args.refactor_model,
        race_seeds=args.race_seeds,
        scenario_gate=args.scenario_gate,
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
