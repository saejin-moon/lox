#!/usr/bin/env python3
"""
LOX — High-Throughput Empirical Policy Synthesis Engine CLI ("Hard Code Nothing").
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.db.schema import DEFAULT_DB_PATH
from lox.engine.orchestrator import DEFAULT_POLICY_PATH, CampaignOrchestrator
from lox.lang.macros import DEFAULT_MACROS_PATH
from lox.ui.console import ui


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LOX — High-Throughput Empirical Policy Synthesis Engine ('Hard Code Nothing')"
    )
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=["openrouter", "openai", "mock"],
        help="LLM API provider (default: openrouter)",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-4-31b-it",
        help="Target LLM model identifier (default: google/gemma-4-31b-it)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key (defaults to OPENROUTER_API_KEY / OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to DuckDB database (default: data/lox.duckdb)",
    )
    parser.add_argument(
        "--policy-path",
        default=DEFAULT_POLICY_PATH,
        help="Path to active policy file (default: data/workspace/policy.py)",
    )
    parser.add_argument(
        "--macros-path",
        default=DEFAULT_MACROS_PATH,
        help="Path to reusable macros file (default: data/workspace/macros.py)",
    )
    parser.add_argument(
        "--role",
        default="val-hum-law-fem",
        help="NetHack character role/race/align/gender (default: val-hum-law-fem)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) - 2),
        help="Number of persistent NLE worker processes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2500,
        help="Max steps per episode (default: 2500)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=50,
        help="Number of synthesis generations to run (default: 50)",
    )
    parser.add_argument(
        "--target-depth",
        type=float,
        default=20.0,
        help="Campaign target average depth (default: 20.0)",
    )
    parser.add_argument(
        "--full-write-episodes",
        type=int,
        default=250,
        help="Episodes for FULL_WRITE mode (default: 250)",
    )
    parser.add_argument(
        "--func-mod-episodes",
        type=int,
        default=150,
        help="Triggered episodes per modified/added function in FUNC_MOD_ADD mode (default: 150)",
    )
    parser.add_argument(
        "--param-tune-episodes",
        type=int,
        default=100,
        help="Episodes per SPSA validation batch in PARAM_TUNE mode (default: 100)",
    )
    parser.add_argument(
        "--param-tune-every",
        type=int,
        default=3,
        help="Run PARAM_TUNE every N generations (default: 3)",
    )
    parser.add_argument(
        "--macro-every",
        type=int,
        default=5,
        help="Run MACRO_WRITE every N generations (default: 5)",
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["FULL_WRITE", "MACRO_WRITE", "FUNC_MOD_ADD", "PARAM_TUNE"],
        help="Optional override to force a specific modification mode",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable sequential futility early stopping",
    )
    parser.add_argument(
        "--pilot-revisions",
        type=int,
        default=5,
        help="Max self-iterative pilot revision runs per generation (default: 5)",
    )
    parser.add_argument(
        "--pilot-episodes",
        type=int,
        default=20,
        help="Episodes per parallel pilot batch test (default: 20)",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=5,
        help="Max in-turn sandbox repair attempts per generation (default: 5)",
    )
    args = parser.parse_args()

    orchestrator = CampaignOrchestrator(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        db_path=args.db_path,
        policy_path=args.policy_path,
        macros_path=args.macros_path,
        role=args.role,
        workers=args.workers,
        max_steps=args.max_steps,
        full_write_episodes=args.full_write_episodes,
        func_mod_episodes_per_fn=args.func_mod_episodes,
        param_tune_episodes=args.param_tune_episodes,
        param_tune_every=args.param_tune_every,
        macro_every=args.macro_every,
        early_stopping=not args.no_early_stop,
        max_repair_turns=args.max_repairs,
        pilot_revisions=args.pilot_revisions,
        pilot_episodes=args.pilot_episodes,
    )
    try:
        orchestrator.run_campaign(
            max_generations=args.generations,
            target_depth=args.target_depth,
            forced_mode=args.mode,
        )
    except KeyboardInterrupt:
        ui.warn("Campaign interrupted by user (Ctrl+C). Exiting cleanly.")
    finally:
        orchestrator.close()


if __name__ == "__main__":
    main()
