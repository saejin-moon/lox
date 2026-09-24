#!/usr/bin/env python3
"""
LOX-ψ S0: prompt-program A/B harness (AGENT_PLAN §1.6, §2-S0).

The author system prompt is a first-class versioned artifact. This harness
evaluates prompt candidates under the same gated discipline as the policy
program: per candidate × author-arm, N isolated author sessions are run from a
shared parent program; each session's diff goes through the SAME validator
gates. Metrics per arm:

  (a) author-acceptance rate,
  (b) validator-reject taxonomy (error code → count),
  (c) cost (tokens, wall time, tool-call counts),
  (d) downstream batch mean score on the accepted program (--downstream).

Mock dry-run (offline, deterministic):
  uv run python scripts/run_prompt_ab.py --candidates v1,v2,v5 --sessions 2

Live campaign (agentic author, all discovered versions):
  uv run python scripts/run_prompt_ab.py --provider gemini \
      --model gemma-4-26b-a4b-it --sessions 3 --downstream
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lox.env_loader import load_env  # noqa: E402

load_env()

from lox.deliberative.providers.gemini import GeminiProvider
from lox.deliberative.providers.llama_cpp import LlamaCppProvider
from lox.deliberative.providers.mock_provider import MockProvider
from lox.deliberative.providers.openrouter import OpenRouterProvider
from lox.policy.author_agent import author_session
from lox.policy.manifest import build_manifest
from lox.policy.predicates import default_ctx
from lox.policy.program import PolicyProgram, DEFAULT_PROGRAM_PATH
from lox.policy.prompts import PROMPTS_DIR, list_prompt_versions
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


def build_hooks() -> ValidatorHooks:
    return ValidatorHooks(fixture_states=[
        default_ctx(), default_ctx(depth=6, hp=10, max_hp=50, hunger_state=3, is_fighting=True),
        default_ctx(dnum=2, depth=3, adjacent_hostiles=1, has_healing=False),
        default_ctx(dnum=3, depth=7),
        {**default_ctx(), "failures": {"descend": 3}, "failures_total": 3},
    ])


async def run_ab(args) -> int:
    from lox.policy import compiler  # noqa: PLC0415
    base_program = PolicyProgram.load(args.program_path)
    manifest = build_manifest(base_program.version, base_program.domain,
                              live_macros={m["name"]: m["body"] for m in base_program.macros})
    bundle = build_report_bundle(args.db_path, args.ledger_path, base_program.domain)
    hooks = build_hooks()
    candidates = ([c.strip() for c in args.candidates.split(",") if c.strip()]
                  if args.candidates else list_prompt_versions())
    if not candidates:
        print("no prompt candidates found (data/prompts/v*.md empty) — aborting")
        return 2
    arms = [a.strip() for a in args.author_mode.split(",") if a.strip()]
    workdir = args.workdir or os.path.join("data", "prompt_ab_work", str(int(time.time())))
    os.makedirs(workdir, exist_ok=True)
    provider = get_provider(args.provider, args.model)
    # An A/B campaign is an experiment: never overwrite the live authoring tree,
    # compiled artifact, or archive with candidate sessions.
    compiler_ctx = compiler.output_dirs(
        program_dir=os.path.join(workdir, "tree"),
        compiled_dir=os.path.join(workdir, "compiled"),
        archive_dir=os.path.join(workdir, "archive"))

    print(f"[prompt_ab] candidates={candidates} arms={arms} "
          f"sessions={args.sessions} provider={args.provider} "
          f"base=v{base_program.version}")
    with compiler_ctx:
        results = await _run_campaign(args, base_program, manifest, bundle, hooks,
                                      candidates, arms, workdir, provider)

    out_path = args.out or os.path.join("data", f"prompt_ab_{int(time.time())}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[prompt_ab] results written to {out_path}\n")
    print(f"{'arm':<28} {'accept':>6} {'tokens':>9} {'tools':>6} {'wall_s':>7} "
          f"{'down_mean':>10}  taxonomy")
    for key, r in results["arms"].items():
        down = (r.get("downstream") or {}).get("mean_score", "")
        print(f"{key:<28} {r['accept_rate']:>6.2f} {r['mean_tokens']:>9.0f} "
              f"{r['mean_tool_calls']:>6.1f} {r['mean_wall_sec']:>7.1f} "
              f"{down if down == '' else round(down, 1):>10}  {r['reject_taxonomy']}")
    return 0


async def _run_campaign(args, base_program, manifest, bundle, hooks,
                        candidates, arms, workdir, provider) -> dict:
    """Runs candidate × arm sessions from the SAME parent program; returns the
    results dict. Must be called inside compiler.output_dirs(...)."""
    results = {"base_version": base_program.version, "provider": args.provider,
               "model": args.model or getattr(provider, "model", args.provider),
               "episodes_downstream": args.downstream_episodes,
               "arms": {}}
    for arm in arms:
        for cand in candidates:
            arm_key = f"{arm}:{cand}"
            sessions = []
            for s in range(args.sessions):
                session_dir = os.path.join(workdir, arm_key.replace(":", "_"), f"s{s}")
                os.makedirs(session_dir, exist_ok=True)
                program_path = os.path.join(session_dir, "program.json")
                base_program.save(program_path)
                ledger_path = os.path.join(session_dir, "ledger.jsonl")
                out = await author_session(
                    provider, base_program, program_path,
                    ledger_path=ledger_path, domain=base_program.domain,
                    author_mode=arm, prompt_version=cand,
                    db_path=args.db_path,
                    max_turns=args.max_turns, hooks=hooks,
                    manifest=manifest, bundle=bundle, commit=True,
                )
                sessions.append({
                    "accepted": out.accepted,
                    "error_code": out.error_code,
                    "gate": out.gate,
                    "detail": out.detail[:200],
                    "tokens_in": out.meta.get("tokens_in", 0),
                    "tokens_out": out.meta.get("tokens_out", 0),
                    "tokens_thought": out.meta.get("tokens_thought", 0),
                    "tool_calls": len(out.meta.get("tool_log", [])),
                    "wall_sec": out.wall_sec,
                    "candidate_version": (out.candidate.version if out.candidate else None),
                    "program_path": program_path if out.accepted else None,
                })
                status = "ACCEPT" if out.accepted else f"REJECT {out.error_code}@{out.gate}"
                print(f"  [{arm_key}] session {s + 1}/{args.sessions}: {status}")

            accepted = [s for s in sessions if s["accepted"]]
            taxonomy = Counter(s["error_code"] or "accepted" for s in sessions)
            arm_result = {
                "accept_rate": len(accepted) / max(1, len(sessions)),
                "reject_taxonomy": dict(taxonomy),
                "mean_tokens": (sum(s["tokens_in"] + s["tokens_out"] + s["tokens_thought"]
                                    for s in sessions) / len(sessions)),
                "mean_tool_calls": sum(s["tool_calls"] for s in sessions) / len(sessions),
                "mean_wall_sec": sum(s["wall_sec"] for s in sessions) / len(sessions),
                "sessions": sessions,
            }
            # (d) downstream batch on the first accepted program
            accepted_paths = [s["program_path"] for s in accepted if s["program_path"]]
            if args.downstream and accepted_paths:
                arm_result["downstream"] = run_downstream(accepted_paths[0], args)
            results["arms"][arm_key] = arm_result
    return results


def run_downstream(program_path: str, args) -> dict:
    """Quick benchmark of an accepted candidate program (swapped onto the live
    program path, restored afterward)."""
    backup = DEFAULT_PROGRAM_PATH + ".prompt_ab_backup"
    had = os.path.exists(DEFAULT_PROGRAM_PATH)
    if had:
        shutil.copy2(DEFAULT_PROGRAM_PATH, backup)
    shutil.copy2(program_path, DEFAULT_PROGRAM_PATH)
    try:
        out_path = os.path.join("logs", f"prompt_ab_downstream_{int(time.time())}.json")
        subprocess.run(
            ["uv", "run", "python", "scripts/run_benchmark.py",
             "--episodes", str(args.downstream_episodes),
             "--max-steps", str(args.downstream_steps),
             "--role", "valkyrie", "--output", out_path],
            check=True, capture_output=True, timeout=7200)
        with open(out_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return {"mean_score": stats.get("mean_score"),
                "median_depth": stats.get("median_depth"),
                "episodes": args.downstream_episodes}
    except Exception as e:  # noqa: BLE001 — downstream is optional reporting
        return {"error": str(e)[:200]}
    finally:
        if had:
            shutil.copy2(backup, DEFAULT_PROGRAM_PATH)
            os.unlink(backup)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOX-ψ S0: prompt-program A/B harness")
    p.add_argument("--candidates", type=str, default=None,
                   help="comma-separated prompt versions (default: all data/prompts/v*.md)")
    p.add_argument("--author-mode", type=str, default="agentic",
                   help="comma-separated subset of agentic,bundle (bundle = S0 baseline)")
    p.add_argument("--sessions", type=int, default=2,
                   help="author sessions per candidate × arm")
    p.add_argument("--provider", choices=["mock", "gemini", "openrouter", "llama_cpp"],
                   default="mock")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--program-path", type=str, default=DEFAULT_PROGRAM_PATH,
                   help="shared parent program for every session")
    p.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH)
    p.add_argument("--ledger-path", type=str, default="data/revision_ledger.jsonl")
    p.add_argument("--max-turns", type=int, default=6)
    p.add_argument("--downstream", action="store_true",
                   help="run a quick benchmark batch per candidate's first accepted program")
    p.add_argument("--downstream-episodes", type=int, default=3)
    p.add_argument("--downstream-steps", type=int, default=5000)
    p.add_argument("--workdir", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run_ab(parse_args())))
