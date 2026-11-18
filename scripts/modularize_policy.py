#!/usr/bin/env python3
"""
One-time CLI wrapper around `lox.lang.modularize` for decomposing a monolithic `policy.py`.

Reads `policy.py` from `--workspace-dir` (or the latest promoted policy from `--db-path`),
splits top-level definitions into balanced `skills/*.py` modules + a thin `policy.py`,
verifies 100% bit-identical NetHack trajectories, and writes the modularized package.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.db.schema import DEFAULT_DB_PATH, connect_db
from lox.lang.modularize import modularize_policy_source, verify_bit_identical
from lox.ui.console import ui


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time deterministic AST modularizer for monolithic policy.py"
    )
    parser.add_argument(
        "--workspace-dir",
        default="data/workspace",
        help="Target workspace directory (default: data/workspace).",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help=f"DuckDB path for --from-db (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Load the latest promoted policy from --db-path instead of --workspace-dir/policy.py.",
    )
    parser.add_argument(
        "--num-modules",
        type=int,
        default=4,
        help="Target number of subtree skill modules (plus shared module, default: 4).",
    )
    parser.add_argument(
        "--verify-seeds",
        type=int,
        default=6,
        help="Number of seeds to verify bit-identical behavior before writing (default: 6).",
    )
    parser.add_argument(
        "--verify-steps",
        type=int,
        default=300,
        help="Steps per verification seed (default: 300).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run split and parity verification without writing files to disk.",
    )
    args = parser.parse_args()

    ws_dir = Path(args.workspace_dir)
    pol_path = ws_dir / "policy.py"
    skills_dir = ws_dir / "skills"

    if args.from_db or not pol_path.exists():
        con = connect_db(args.db_path, read_only=True)
        try:
            row = con.execute(
                "SELECT policy_id, generation, policy_code, skills_json FROM policies "
                "WHERE is_promoted = TRUE ORDER BY generation DESC, created_at DESC LIMIT 1"
            ).fetchone()
        finally:
            con.close()
        if not row or not row[2]:
            raise SystemExit(f"No promoted policy found in `{args.db_path}`.")
        orig_policy = str(row[2])
        orig_skills = dict(json.loads(row[3]) if row[3] else {})
        ui.info(f"Loaded promoted `{row[0]}` (Gen {row[1]}) from `{args.db_path}`.")
    else:
        from lox.lang.bundle import load_skills_dir

        orig_policy = pol_path.read_text(encoding="utf-8")
        orig_skills = load_skills_dir(skills_dir)
        ui.info(f"Loaded `{pol_path}` ({len(orig_policy.splitlines())} LOC).")

    if orig_skills:
        ui.warn(
            f"Policy already has {len(orig_skills)} skill module(s): {sorted(orig_skills)}. "
            "Proceeding splits only functions inside policy.py while preserving existing skills."
        )

    new_policy, final_skills = modularize_policy_source(
        orig_policy,
        num_modules=max(1, int(args.num_modules)),
        existing_skills=orig_skills,
    )

    ui.info(
        f"Planned split: policy.py {len(orig_policy.splitlines())} -> "
        f"{len(new_policy.splitlines())} LOC across {len(final_skills)} skill module(s):"
    )
    for k, src in sorted(final_skills.items()):
        ui.info(f"  * skills/{k}.py: {len(src.splitlines())} LOC")

    seeds = [1000 + i for i in range(max(1, int(args.verify_seeds)))]
    verify_bit_identical(
        orig_policy,
        orig_skills,
        new_policy,
        final_skills,
        seeds=seeds,
        steps=max(50, int(args.verify_steps)),
    )
    ui.info(
        f"Verified 100% bit-identical action trajectories across {len(seeds)} seeds x {args.verify_steps} steps."
    )

    if args.dry_run:
        ui.info("--dry-run set: no files written.")
        return

    from lox.lang.bundle import write_skills_dir

    ws_dir.mkdir(parents=True, exist_ok=True)
    pol_path.write_text(new_policy, encoding="utf-8")
    write_skills_dir(skills_dir, final_skills, sync=True, ensure_trailing_newline=False)
    ui.info(f"Wrote modularized package cleanly to `{ws_dir}`.")


if __name__ == "__main__":
    main()
