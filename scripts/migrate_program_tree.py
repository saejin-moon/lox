#!/usr/bin/env python3
"""
LOX-ψ S1: migrate the monolithic policy programs into per-file authoring trees
(AGENT_PLAN §1.1, decision-ledger "Storage: per-file .sexpr authoring tree").

For each domain with a live program:
  data/policy_program.json          (nethack)
  data/policy_program_minihack.json (minihack)
  → data/program/<domain>/          (the authoring tree; the LLM's workspace)
  → data/compiled/<domain>.json     (the canonical compiled artifact)
  → data/programs/archive/          (archived copy, provenance)
  → live file rewritten in canonical compiled form (executor contract unchanged;
    PolicyProgram.load consumes it with zero rewrites)

Byte-equivalence is enforced at every step (compile_and_commit fails loud).

Usage:
  uv run python scripts/migrate_program_tree.py            # migrate all domains
  uv run python scripts/migrate_program_tree.py --check    # verify trees compile
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.policy.compiler import (
    compile_and_commit, compile_authoring_tree, tree_dir_for,
)
from lox.policy.program import PolicyProgram

LIVE_PROGRAM_PATHS = {
    "nethack": "data/policy_program.json",
    "minihack": "data/policy_program_minihack.json",
}


def migrate_domain(domain: str, live_path: str) -> None:
    if not os.path.exists(live_path):
        print(f"[migrate] {domain}: no live program at {live_path} — skipped")
        return
    program = PolicyProgram.load(live_path)
    tree = tree_dir_for(domain)
    before = json.dumps(program.to_dict(), indent=2)
    compiled_path = compile_and_commit(program, tree, live_path)
    after = open(live_path, encoding="utf-8").read()
    n_rules = len(program.tactic_rules)
    n_goals = len(program.strategy_plan)
    print(f"[migrate] {domain}: v{program.version} → {tree} "
          f"({n_rules} rules, {n_goals} goals, {len(program.params)} params)")
    print(f"[migrate] {domain}: compiled artifact {compiled_path}; "
          f"live file canonical={'changed' if before + '\n' != after else 'unchanged'}")


def check_domain(domain: str, live_path: str) -> bool:
    tree = tree_dir_for(domain)
    if not os.path.exists(os.path.join(tree, "program.json")):
        print(f"[check] {domain}: no tree at {tree}")
        return False
    compiled = compile_authoring_tree(tree)
    ok = True
    if os.path.exists(live_path):
        live = PolicyProgram.load(live_path)
        if live.to_dict() != compiled.to_dict():
            print(f"[check] {domain}: MISMATCH compiled tree vs live file")
            ok = False
    out = os.path.join("data", "compiled", f"{domain}.json")
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            if json.load(f) != compiled.to_dict():
                print(f"[check] {domain}: MISMATCH compiled tree vs {out}")
                ok = False
    print(f"[check] {domain}: {'OK' if ok else 'FAILED'} "
          f"(v{compiled.version}, {len(compiled.tactic_rules)} rules, "
          f"{len(compiled.strategy_plan)} goals)")
    return ok


def main() -> int:
    p = argparse.ArgumentParser(description="S1: monolith → authoring-tree migration")
    p.add_argument("--check", action="store_true",
                   help="verify existing trees compile byte-equal; no writes")
    p.add_argument("--domain", type=str, default=None,
                   help="migrate/check a single domain")
    args = p.parse_args()
    domains = ([args.domain] if args.domain else list(LIVE_PROGRAM_PATHS))
    ok = True
    for domain in domains:
        live_path = LIVE_PROGRAM_PATHS.get(domain, f"data/policy_program_{domain}.json")
        if args.check:
            ok = check_domain(domain, live_path) and ok
        else:
            migrate_domain(domain, live_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
