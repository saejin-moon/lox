#!/usr/bin/env python3
"""
CORP-Ω R3: unattended gated revision loop (AGENT_PLAN refactor steps 6–8).

Cycle: load program → build run-report bundle → author LLM proposes ONE policy diff →
validator gate pipeline (parse→header→vocab→bounds→budgets→macros→expansion→mount→
invariants→shadow) → [optional: certification + 3-ep quick batch] → commit program bump
+ git + ledger, or ledger the rejection with its error code. Rejection is first-class.

Usage (mock dry-run — R3 acceptance criterion 1):
  uv run python scripts/run_revision_loop.py --provider mock --max-revisions 10

Live author (R3 acceptance criterion 2 — 3-ep quick batch gates acceptance):
  uv run python scripts/run_revision_loop.py --provider gemini --model gemma-3-27b-it \
      --max-revisions 1 --live-gates
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

from corp.deliberative.providers.gemini import GeminiProvider
from corp.deliberative.providers.llama_cpp import LlamaCppProvider
from corp.deliberative.providers.mock_provider import MockProvider
from corp.deliberative.providers.openrouter import OpenRouterProvider
from corp.policy.ledger import RevisionLedger, DEFAULT_LEDGER_PATH
from corp.policy.manifest import build_manifest
from corp.policy.program import PolicyProgram, DEFAULT_PROGRAM_PATH
from corp.policy.report import build_report_bundle, DEFAULT_DB_PATH
from corp.policy.reviser import Reviser
from corp.policy.validator import validate_diff, ValidatorHooks

PROGRAM_ARCHIVE_DIR = "data/policy_program"


def get_provider(provider_type: str, model: str | None = None):
    if provider_type == "gemini":
        return GeminiProvider(model=model, timeout=180.0)
    if provider_type == "openrouter":
        return OpenRouterProvider(model=model, timeout=180.0)
    if provider_type == "llama_cpp":
        return LlamaCppProvider(model=model, timeout=180.0)
    return MockProvider()


# ---------------------------------------------------------------------------
# Heavy gates (§6 steps 11–12) — subprocess-based, run only with --live-gates
# ---------------------------------------------------------------------------

def make_quick_batch_gate(args) -> callable:
    def quick_batch(candidate: PolicyProgram) -> dict:
        """3 episodes × 5k steps on the CANDIDATE program (R3 addendum thresholds).
        Swaps the candidate onto the live program path for the batch, then restores
        whatever was there (commit_program re-saves the candidate on acceptance)."""
        program_path = args.program_path
        backup_path = program_path + ".pre_quick_batch"
        had_backup = os.path.exists(program_path)
        if had_backup:
            shutil.copy2(program_path, backup_path)
        candidate.save(program_path)
        try:
            out_path = os.path.join("logs", f"quick_batch_rev{candidate.version}.json")
            cmd = [
                "uv", "run", "python", "scripts/run_benchmark.py",
                "--episodes", str(args.quick_batch_episodes),
                "--max-steps", str(args.quick_batch_steps),
                "--role", "valkyrie",
                "--output", out_path,
            ]
            print(f"  [quick-batch] running {args.quick_batch_episodes}ep × {args.quick_batch_steps} steps ...")
            subprocess.run(cmd, check=True, capture_output=True, timeout=3600)
            with open(out_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            return {"mean_score": stats.get("mean_score", 0.0),
                    "median_depth": stats.get("median_depth")}
        finally:
            if had_backup:
                shutil.copy2(backup_path, program_path)
                os.unlink(backup_path)
    return quick_batch


def make_certification_gate(args) -> callable:
    def certification(candidate: PolicyProgram) -> None:
        print("  [certification] running skill certification suite ...")
        proc = subprocess.run(
            ["uv", "run", "python", "scripts/run_skill_certifications.py"],
            capture_output=True, timeout=3600,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"certification suite failed (rc={proc.returncode})")
    return certification


def commit_program(program: PolicyProgram, program_path: str, no_git: bool) -> None:
    """§6 step 13: archive previous, save new, git commit."""
    os.makedirs(PROGRAM_ARCHIVE_DIR, exist_ok=True)
    if os.path.exists(program_path):
        shutil.copy2(program_path, os.path.join(
            PROGRAM_ARCHIVE_DIR, f"policy_program_v{program.version}.json"))
    program.save(program_path)
    if not no_git:
        try:
            subprocess.run(["git", "add", program_path, PROGRAM_ARCHIVE_DIR,
                            DEFAULT_LEDGER_PATH], check=False, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m",
                 f"policy: program v{program.version} via revision loop ({program.provenance.get('author', 'unknown')})"],
                check=False, capture_output=True)
        except Exception:  # noqa: BLE001 — git is best-effort
            pass


async def run_loop(args) -> int:
    ledger = RevisionLedger(args.ledger_path)
    if args.program_path is None:
        args.program_path = (DEFAULT_PROGRAM_PATH if args.domain == "nethack"
                             else f"data/policy_program_{args.domain}.json")
    program_path = args.program_path
    program = PolicyProgram.load(program_path)
    if args.domain != "nethack" and program.domain != args.domain:
        # cold-start: seed the file with the adapter's default program on first run
        from corp.executor import get_adapter  # noqa: PLC0415
        PolicyProgram.from_dict(get_adapter(args.domain).default_program()).save(program_path)
        program = PolicyProgram.load(program_path)
    print(f"[loop] program v{program.version} from {program_path} | provider={args.provider}")

    adapter = None
    if args.domain != "nethack":
        from corp.executor import get_adapter  # noqa: PLC0415
        adapter = get_adapter(args.domain)

    provider = get_provider(args.provider, args.model)
    model_name = args.model or getattr(provider, "model", args.provider)
    reviser = Reviser(provider, ledger)

    hooks_kwargs: dict = {}
    if args.live_gates:
        hooks_kwargs["certification"] = make_certification_gate(args)
        hooks_kwargs["quick_batch"] = make_quick_batch_gate(args)
        # §6 step 12 baseline: the current program's own quick-batch mean. A revision
        # is rejected if the candidate drops > 50% below this.
        baseline_stats = make_quick_batch_gate(args)(program)
        # (the gate restores the pre-batch program file, so v{program.version} is intact)
        hooks_kwargs["baseline_score"] = baseline_stats.get("mean_score")
        print(f"[loop] quick-batch baseline (v{program.version}): "
              f"mean_score={hooks_kwargs['baseline_score']}")
    if args.fixture_shadow:
        from corp.policy.predicates import default_ctx  # noqa: PLC0415
        if adapter is not None:
            # domain shadow fixtures: the domain ctx over the shared vocabulary
            hooks_kwargs["fixture_states"] = [
                {"stairs_known": False, "hp": 10, "max_hp": 10, "adjacent_hostiles": 0,
                 "depth": 1, "turn": 1},
                {"stairs_known": True, "hp": 3, "max_hp": 10, "adjacent_hostiles": 1,
                 "monster_name": "jackal", "depth": 1, "turn": 200},
            ]
        else:
            hooks_kwargs["fixture_states"] = [
                default_ctx(), default_ctx(depth=6, hp=10, max_hp=50, hunger_state=3, is_fighting=True),
                default_ctx(dnum=2, depth=3, adjacent_hostiles=1, has_healing=False),
                default_ctx(dnum=3, depth=7),
            ]

    n_accepted = 0
    for rev_idx in range(1, args.max_revisions + 1):
        if adapter is not None:
            manifest = adapter.manifest(program.version)
            # transfer domains: the batch report = a seeded run of the CURRENT program
            batch = [adapter.run_episode(adapter.make_env(seed=s), program, s,
                                         adapter.spec.step_limit_default)
                     for s in range(args.report_episodes)]
            bundle = adapter.report_bundle(batch)
        else:
            manifest = build_manifest(program.version, program.domain,
                                      live_macros={m["name"]: m["body"] for m in program.macros})
            bundle = build_report_bundle(args.db_path, args.ledger_path, program.domain)
        print(f"\n=== revision {rev_idx}/{args.max_revisions} | program v{program.version} ===")

        t0 = time.perf_counter()
        try:
            diff_text = await reviser.propose_diff(program, manifest, bundle)
        except Exception as e:  # noqa: BLE001 — author path failure = parse-level rejection
            ledger.append(ledger.revision_entry(
                revision=program.version + 1, parent_version=program.version,
                author_model=model_name, provider=args.provider, accepted=False,
                reject_code="ERR_PARSE", reject_detail=f"author output unusable: {e}",
                gate="parse", wall_sec=time.perf_counter() - t0))
            print(f"  REJECT ERR_PARSE: author output unusable: {e}")
            continue
        meta = dict(reviser.last_meta)

        result = validate_diff(diff_text, program, manifest,
                               hooks=ValidatorHooks(**hooks_kwargs))
        if result.ok:
            candidate = result.candidate
            delta = {}
            if hooks_kwargs.get("quick_batch"):
                delta = {"metric": "quick_batch_mean_score",
                         "after": candidate.provenance}
            commit_program(candidate, program_path, args.no_git)
            ledger.append(ledger.revision_entry(
                revision=candidate.version, parent_version=program.version,
                author_model=model_name, provider=args.provider, accepted=True,
                reason=result.diff.header.reason, diff_text=diff_text,
                tokens_in=meta.get("tokens_in", 0), tokens_out=meta.get("tokens_out", 0),
                tokens_thought=meta.get("tokens_thought", 0),
                latency_ms=meta.get("latency_ms", 0.0),
                thinking=meta.get("thinking", ""), response_text=meta.get("response_text", ""),
                wall_sec=meta.get("wall_sec", 0.0), delta=delta))
            print(f"  ACCEPT v{candidate.version}: {result.diff.header.reason[:90]}")
            print(f"  ops: {len(result.diff.sets)} set, {len(result.diff.rules)} rule, "
                  f"{len(result.diff.goals)} goal, {len(result.diff.defmacros)} macro, "
                  f"{len(result.diff.nogoods)} nogood")
            program = candidate
            n_accepted += 1
        else:
            ledger.append(ledger.revision_entry(
                revision=result.diff.header.revision if result.diff else program.version + 1,
                parent_version=program.version,
                author_model=model_name, provider=args.provider, accepted=False,
                reject_code=result.error_code, reject_detail=result.detail[:300],
                gate=result.gate, reason=result.diff.header.reason if result.diff else "",
                diff_text=diff_text[:2000], tokens_in=meta.get("tokens_in", 0),
                tokens_out=meta.get("tokens_out", 0),
                tokens_thought=meta.get("tokens_thought", 0),
                latency_ms=meta.get("latency_ms", 0.0),
                thinking=meta.get("thinking", ""), response_text=meta.get("response_text", ""),
                wall_sec=meta.get("wall_sec", 0.0)))
            print(f"  REJECT {result.error_code} @ {result.gate}: {result.detail[:120]}")

    print(f"\n[loop] done: {n_accepted}/{args.max_revisions} accepted | "
          f"acceptance rate: {ledger.acceptance_rate(model_name):.2f}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CORP R3 unattended gated revision loop")
    p.add_argument("--max-revisions", type=int, default=10)
    p.add_argument("--provider", choices=["mock", "gemini", "openrouter", "llama_cpp"], default="mock")
    p.add_argument("--model", type=str, default=None,
                   help="author model id (e.g. 'gemma-3-27b-it')")
    p.add_argument("--program-path", type=str, default=None,
                   help="defaults to data/policy_program.json (nethack) or data/policy_program_<domain>.json")
    p.add_argument("--domain", type=str, default="nethack",
                   help="adapter domain: nethack | minihack")
    p.add_argument("--report-episodes", type=int, default=5,
                   help="transfer domains: seeded episodes run per revision for the batch report")
    p.add_argument("--ledger-path", type=str, default=DEFAULT_LEDGER_PATH)
    p.add_argument("--db-path", type=str, default=DEFAULT_DB_PATH)
    p.add_argument("--live-gates", action="store_true",
                   help="run certification + 3-ep quick batch gates (slow; real episodes)")
    p.add_argument("--fixture-shadow", action=argparse.BooleanOptionalAction, default=True,
                   help="run shadow tests on synthetic fixture states (default: on)")
    p.add_argument("--quick-batch-episodes", type=int, default=3)
    p.add_argument("--quick-batch-steps", type=int, default=5000)
    p.add_argument("--no-git", action="store_true", help="skip git commit of program bumps")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(run_loop(parse_args())))