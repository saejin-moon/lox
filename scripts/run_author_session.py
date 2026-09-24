#!/usr/bin/env python3
"""
LOX-ψ S0: ONE agentic author session (AGENT_PLAN §2-S0).

The author LLM gathers evidence through the tool loop (query_duckdb, wiki_search,
read_trajectory, read_env_schema, read_manifest, read_program_tree), emits ONE
policy diff, and that diff flows through the validator gate pipeline. The
validator is the only program writer: accepted candidates are committed +
ledgered, rejected diffs are ledgered with their error code.

Usage (mock dry-run):
  uv run python scripts/run_author_session.py --provider mock --no-commit

Live author with fixture-shadow validation + program commit:
  uv run python scripts/run_author_session.py --provider gemini \
      --model gemma-4-26b-a4b-it --live-gates
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env (GEMINI_API_KEY etc.) before provider construction
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

from lox.deliberative.providers.gemini import GeminiProvider
from lox.deliberative.providers.llama_cpp import LlamaCppProvider
from lox.deliberative.providers.mock_provider import MockProvider
from lox.deliberative.providers.openrouter import OpenRouterProvider
from lox.policy.author_agent import author_session
from lox.policy.ledger import RevisionLedger
from lox.policy.manifest import build_manifest
from lox.policy.predicates import default_ctx
from lox.policy.program import PolicyProgram, DEFAULT_PROGRAM_PATH
from lox.policy.report import build_report_bundle, DEFAULT_DB_PATH
from lox.policy.validator import ValidatorHooks


def get_provider(provider_type: str, model: str | None = None):
    if provider_type == "gemini":
        return GeminiProvider(model=model, timeout=180.0)
    if provider_type == "openrouter":
        return OpenRouterProvider(model=model, timeout=180.0)
    if provider_type == "llama_cpp":
        return LlamaCppProvider(model=model, timeout=180.0)
    return MockProvider()


def make_quick_batch_gate(args) -> callable:
    """Tiered quick-batch gate (v6 lesson): rule/handler adds gate on 10ep×10k,
    param-only on 3ep×5k."""
    def quick_batch(candidate: PolicyProgram) -> dict:
        parent = PolicyProgram.load(args.program_path)
        ep, steps = args.quick_batch_episodes, args.quick_batch_steps
        if len(candidate.tactic_rules) > len(parent.tactic_rules):
            ep, steps = max(ep, 10), max(steps, 10000)
        backup_path = args.program_path + ".pre_quick_batch"
        had_backup = os.path.exists(args.program_path)
        if had_backup:
            import shutil
            shutil.copy2(args.program_path, backup_path)
        candidate.save(args.program_path)
        try:
            out_path = os.path.join("logs", f"author_session_batch_{candidate.version}.json")
            subprocess.run(
                ["uv", "run", "python", "scripts/run_benchmark.py",
                 "--episodes", str(ep), "--max-steps", str(steps),
                 "--role", "valkyrie", "--output", out_path],
                check=True, capture_output=True, timeout=3600)
            with open(out_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            return {"mean_score": stats.get("mean_score", 0.0)}
        finally:
            if had_backup:
                import shutil
                shutil.copy2(backup_path, args.program_path)
                os.unlink(backup_path)
    return quick_batch


def build_hooks(args, program: PolicyProgram) -> ValidatorHooks:
    hooks_kwargs: dict = {}
    if args.fixture_shadow:
        hooks_kwargs["fixture_states"] = [
            default_ctx(), default_ctx(depth=6, hp=10, max_hp=50, hunger_state=3, is_fighting=True),
            default_ctx(dnum=2, depth=3, adjacent_hostiles=1, has_healing=False),
            default_ctx(dnum=3, depth=7),
            # failure-counter fixture: the S0 predicate path must evaluate
            {**default_ctx(), "failures": {"descend": 3}, "failures_total": 3},
        ]
    if args.live_gates and args.domain == "nethack":
        hooks_kwargs["quick_batch"] = make_quick_batch_gate(args)
        baseline = make_quick_batch_gate(args)(program)
        hooks_kwargs["baseline_score"] = baseline.get("mean_score")
        print(f"[session] quick-batch baseline (v{program.version}): "
              f"mean_score={hooks_kwargs['baseline_score']}")
    return ValidatorHooks(**hooks_kwargs)


async def run_session(args) -> int:
    program = PolicyProgram.load(args.program_path)
    print(f"[session] program v{program.version} from {args.program_path} | "
          f"provider={args.provider} mode={args.author_mode} "
          f"prompt={args.prompt_version or 'default'}")

    manifest = build_manifest(program.version, program.domain,
                              live_macros={m["name"]: m["body"] for m in program.macros})
    bundle = build_report_bundle(args.db_path, args.ledger_path, program.domain)
    hooks = build_hooks(args, program)
    provider = get_provider(args.provider, args.model)
    author_label = args.model or getattr(provider, "model", args.provider)
    ledger = RevisionLedger(args.ledger_path)

    out = await author_session(
        provider, program, args.program_path,
        ledger=ledger, domain=args.domain,
        author_mode=args.author_mode, prompt_version=args.prompt_version,
        db_path=args.db_path, ledger_path=args.ledger_path,
        max_turns=args.max_turns, hooks=hooks, manifest=manifest, bundle=bundle,
        commit=args.commit, author_label=author_label,
    )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({
                "accepted": out.accepted, "error_code": out.error_code,
                "gate": out.gate, "detail": out.detail, "wall_sec": out.wall_sec,
                "diff_text": out.diff_text[:4000], "meta": out.meta,
            }, f, indent=2, default=str)
        print(f"[session] transcript written to {args.out}")

    if out.accepted:
        print(f"  ACCEPT v{out.candidate.version}: "
              f"{(out.candidate.provenance.get('reason') or '')[:100]}")
        return 0
    print(f"  REJECT {out.error_code} @ {out.gate}: {out.detail[:200]}")
    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOX-ψ S0: one agentic author session")
    p.add_argument("--provider", choices=["mock", "gemini", "openrouter", "llama_cpp"],
                   default="mock")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--author-mode", choices=["agentic", "bundle"], default="agentic",
                   help="agentic = tool-loop author (S0); bundle = single-shot Reviser baseline")
    p.add_argument("--prompt-version", type=str, default=None,
                   help="prompt version id (data/prompts/<v>.md) or path; default = newest")
    p.add_argument("--program-path", type=str, default=DEFAULT_PROGRAM_PATH)
    p.add_argument("--domain", type=str, default="nethack")
    p.add_argument("--ledger-path", type=str, default="data/revision_ledger.jsonl")
    p.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH)
    p.add_argument("--max-turns", type=int, default=6, help="tool-loop turn budget")
    p.add_argument("--live-gates", action="store_true",
                   help="run the tiered quick-batch gate (real episodes; slow)")
    p.add_argument("--fixture-shadow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--commit", action=argparse.BooleanOptionalAction, default=True,
                   help="save the accepted candidate to --program-path (default on)")
    p.add_argument("--out", type=str, default=None, help="write the session transcript JSON")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run_session(parse_args())))
