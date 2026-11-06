#!/usr/bin/env python3
"""
scripts/calibrate_pilot.py — Calibrates the Pilot Revision Budget for LOX.

Analyzes DuckDB synthesis logs and experiment outcomes to determine:
1. Attempt-by-attempt pilot pass rates and full-rollout transfer rates.
2. Cumulative promotion yields as a function of allowed pilot revisions K.
3. Fallback promotion yields (candidates promoted despite failing pilot sign test).
4. Data-driven recommendation for --pilot-revisions.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from typing import Any

import duckdb


def parse_pilot_delta(tool_results_json: str | None) -> float | None:
    """Extracts numerical pilot delta from tool results json or summary text."""
    if not tool_results_json:
        return None
    m = re.search(r"[Δ\\]Delta\s*=\s*([+-]?\d+\.?\d*)", tool_results_json)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def calibrate_pilot(db_path: str, min_generations: int = 5) -> None:
    if not os.path.exists(db_path):
        print(f"Error: Database file `{db_path}` does not exist.", file=sys.stderr)
        sys.exit(1)

    con = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "synthesis_logs" not in tables:
        print(
            f"Error: `synthesis_logs` table not found in `{db_path}`.", file=sys.stderr
        )
        sys.exit(1)

    pilot_rows = con.execute("""
        SELECT generation, turn_index, status, tool_results_json, prompt_tokens, completion_tokens, wall_ms
        FROM synthesis_logs
        WHERE turn_type = 'PILOT_EXECUTION'
        ORDER BY generation, turn_index
    """).fetchall()

    if not pilot_rows:
        print(f"No pilot execution records found in `{db_path}`.")
        return

    # Map promotions per generation
    promoted_gens: set[int] = set()
    if "batches" in tables:
        promoted_gens = set(
            r[0]
            for r in con.execute(
                "SELECT DISTINCT generation FROM batches WHERE hypothesis_passed = TRUE AND mode != 'PILOT'"
            ).fetchall()
        )
    elif "experiments" in tables:
        promoted_gens = set(
            r[0]
            for r in con.execute(
                "SELECT DISTINCT generation FROM experiments WHERE passed = TRUE AND mode != 'PILOT'"
            ).fetchall()
        )

    # Group attempts by generation
    gen_attempts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for gen, turn_idx, status, res_json, p_tok, c_tok, wall in pilot_rows:
        gen_attempts[gen].append(
            {
                "turn_index": turn_idx,
                "status": status,
                "res_json": res_json,
                "delta": parse_pilot_delta(res_json),
                "tokens": (p_tok or 0) + (c_tok or 0),
                "wall_ms": wall or 0.0,
            }
        )

    total_gens = len(gen_attempts)
    if total_gens < min_generations:
        print(
            f"Warning: Only {total_gens} generations found with pilot logs (recommended >= {min_generations})."
        )

    attempt_counts: dict[int, int] = defaultdict(int)
    attempt_passes: dict[int, int] = defaultdict(int)
    attempt_promoted: dict[int, int] = defaultdict(int)
    fallback_promoted: dict[int, int] = defaultdict(int)
    first_pass_attempts: dict[int, int] = defaultdict(int)

    for gen, attempts in gen_attempts.items():
        first_pass_idx: int | None = None
        for idx, att in enumerate(attempts, 1):
            attempt_counts[idx] += 1
            if att["status"] == "PILOT_IMPROVED":
                attempt_passes[idx] += 1
                if first_pass_idx is None:
                    first_pass_idx = idx

        if first_pass_idx is not None:
            first_pass_attempts[first_pass_idx] += 1
            if gen in promoted_gens:
                attempt_promoted[first_pass_idx] += 1
        elif gen in promoted_gens:
            # Fallback promotion: all pilots failed sign test, but candidate promoted in full rollout
            last_attempt_idx = len(attempts)
            fallback_promoted[last_attempt_idx] += 1

    max_attempt = max(attempt_counts.keys()) if attempt_counts else 0

    print("=" * 78)
    print(" LOX PILOT REVISION CALIBRATION REPORT")
    print(
        f" Database: {db_path} | Analyzed Generations: {total_gens} | Promoted Generations: {len(promoted_gens)}"
    )
    print("=" * 78)
    print()

    # Table 1: Per-Attempt Performance Breakdown
    print(
        "┌"
        + "─" * 8
        + "┬"
        + "─" * 12
        + "┬"
        + "─" * 14
        + "┬"
        + "─" * 14
        + "┬"
        + "─" * 14
        + "┬"
        + "─" * 10
        + "┐"
    )
    print(
        "│ Attempt│ Gen Count  │ Pilot Passes │ Full Promos  │ Pass Rate %  │ Transfer%│"
    )
    print(
        "├"
        + "─" * 8
        + "┼"
        + "─" * 12
        + "┼"
        + "─" * 14
        + "┼"
        + "─" * 14
        + "┼"
        + "─" * 14
        + "┼"
        + "─" * 10
        + "┤"
    )

    for k in range(1, max_attempt + 1):
        cnt = attempt_counts[k]
        pas = attempt_passes[k]
        pro = attempt_promoted[k] + fallback_promoted[k]
        pass_rate = f"{(pas / cnt * 100):.1f}%" if cnt > 0 else "N/A"
        transfer_rate = (
            f"{(pro / pas * 100):.1f}%"
            if pas > 0
            else ("(fallback)" if pro > 0 else "N/A")
        )
        print(
            f"│   {k:4d} │ {cnt:10d} │ {pas:12d} │ {pro:12d} │ {pass_rate:>12s} │ {transfer_rate:>8s} │"
        )

    print(
        "└"
        + "─" * 8
        + "┴"
        + "─" * 12
        + "┴"
        + "─" * 14
        + "┴"
        + "─" * 14
        + "┴"
        + "─" * 14
        + "┴"
        + "─" * 10
        + "┘"
    )
    print()

    # Table 2: Cumulative Pipeline Efficiency vs. Pilot Budget K
    print("Cumulative Yield by Allowed Revision Budget (K):")
    print(
        "┌"
        + "─" * 10
        + "┬"
        + "─" * 16
        + "┬"
        + "─" * 16
        + "┬"
        + "─" * 16
        + "┬"
        + "─" * 14
        + "┐"
    )
    print(
        "│ Budget (K)│ Cumul. Passes  │ Cumul. Promoted│ Gen Success %  │ Marginal Adds│"
    )
    print(
        "├"
        + "─" * 10
        + "┼"
        + "─" * 16
        + "┼"
        + "─" * 16
        + "┼"
        + "─" * 16
        + "┼"
        + "─" * 14
        + "┤"
    )

    cumul_pass = 0
    cumul_promo = 0
    recommended_k = 3
    diminishing_k: int | None = None

    for k in range(1, max_attempt + 1):
        new_passes = first_pass_attempts[k]
        new_promos = attempt_promoted[k] + fallback_promoted[k]
        cumul_pass += new_passes
        cumul_promo += new_promos
        succ_rate = (
            f"{(cumul_promo / total_gens * 100):.1f}%" if total_gens > 0 else "N/A"
        )
        print(
            f"│    K = {k:2d} │ {cumul_pass:14d} │ {cumul_promo:14d} │ {succ_rate:>14s} │ +{new_promos:11d} │"
        )

        if new_promos > 0:
            recommended_k = max(recommended_k, k)
        elif diminishing_k is None and k > 2:
            diminishing_k = k

    print(
        "└"
        + "─" * 10
        + "┴"
        + "─" * 16
        + "┴"
        + "─" * 16
        + "┴"
        + "─" * 16
        + "┴"
        + "─" * 14
        + "┘"
    )
    print()

    # Analysis & Actionable Recommendation
    print("Strategic Recommendations:")
    if (
        max_attempt <= 3
        and attempt_counts.get(3, 0) > 0
        and attempt_passes.get(3, 0) > 0
    ):
        print(
            f" • Attempt 3 produced {attempt_passes.get(3, 0)} pilot passes and {attempt_promoted.get(3, 0)} promotions."
        )
        print(" • In previous runs, generations were auto-aborted after attempt 3.")
        print(
            f" • Recommended setting: `--pilot-revisions {max(8, recommended_k)}` to explore higher-order revisions."
        )
    elif diminishing_k is not None:
        print(f" • Marginal promotion gains level off after K={diminishing_k - 1}.")
        print(
            f" • Recommended setting: `--pilot-revisions {diminishing_k - 1}` for optimal compute-to-progress ratio."
        )
    else:
        print(f" • Recommended setting: `--pilot-revisions {recommended_k}`.")

    total_fallbacks = sum(fallback_promoted.values())
    if total_fallbacks > 0:
        print(
            f" • Best-candidate fallback yielded {total_fallbacks} promotion(s) that would otherwise have been discarded."
        )
    else:
        print(
            " • With `--pilot-fallback best`, near-neutral candidates will be rescued in full rollout with early stopping."
        )
    print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrates the pilot revision budget and evaluates attempt promotion transfer rates."
    )
    parser.add_argument(
        "--db-path",
        default="data/lox.duckdb",
        help="Path to DuckDB database (default: data/lox.duckdb)",
    )
    parser.add_argument(
        "--min-generations",
        type=int,
        default=5,
        help="Minimum analyzed generations for confident recommendation (default: 5)",
    )
    args = parser.parse_args()
    calibrate_pilot(db_path=args.db_path, min_generations=args.min_generations)


if __name__ == "__main__":
    main()
