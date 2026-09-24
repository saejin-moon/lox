#!/usr/bin/env python3
"""
LOX-ψ: three-arm ablation harness (AGENT_PLAN §8 — the falsifiability control).

Arms:
  llm    — the revision loop runs live (author LLM gated revisions), then the resulting
           program is benchmarked.
  frozen — the unmodified program is benchmarked (control for batch variance).
  random — random perturbation of the SAME param leaves within declared bounds, gated
           through the SAME validator pipeline (proves it's not the gate doing the work).

Protocol (§8): per arm, per domain — 100-ep batches on NetHack (unseedable, σ
acknowledged); seeded domains 3 seeds × 34 eps (R5). Statistics: median depth, mean
score with bootstrap 95% CI (10k resamples); arm-vs-frozen Mann-Whitney U (p < 0.05)
plus Cliff's delta. Results: data/ablation_<ts>.json + stdout table.

Smoke test (mock author, fast):
  uv run python scripts/run_ablation.py --arms frozen,random --episodes 2 --max-steps 1000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# .env for the llm arm's author provider
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

from lox.policy.manifest import build_manifest, param_leaves_from_config
from lox.policy.program import PolicyProgram, DEFAULT_PROGRAM, DEFAULT_PROGRAM_PATH
from lox.policy.validator import validate_diff, ValidatorHooks


# ---------------------------------------------------------------------------
# Arm program preparation
# ---------------------------------------------------------------------------

def prepare_frozen_arm(workdir: str) -> str:
    """Arm 2: the unmodified program."""
    path = os.path.join(workdir, "program_frozen.json")
    PolicyProgram.from_dict(DEFAULT_PROGRAM).save(path)
    return path


def prepare_random_arm(workdir: str, n_revisions: int = 5, ops_per_rev: int = 3,
                       seed: int = 7, domain: str = "nethack") -> str:
    """Arm 3: random perturbation WITHIN declared bounds, gated through the SAME
    validator pipeline. Identical acceptance machinery to the llm arm — only the
    proposal distribution differs (this is the control that makes the claim
    'the LLM adds value beyond gated search' falsifiable)."""
    program = PolicyProgram.from_dict(
        DEFAULT_PROGRAM if domain == "nethack"
        else __import__("lox.executor", fromlist=["get_adapter"]).get_adapter(domain).default_program()
    )
    rng = random.Random(seed)
    if domain == "nethack":
        numeric = {p: s for p, s in param_leaves_from_config().items()
                   if s["type"] in ("int", "float")}
    else:
        from lox.executor import get_adapter  # noqa: PLC0415
        numeric = {p: {"type": s.type, "min": s.min, "max": s.max}
                   for p, s in get_adapter(domain).param_leaves().items()
                   if s.type in ("int", "float")}
    for _ in range(n_revisions):
        ops = []
        for path in rng.sample(sorted(numeric), min(ops_per_rev, len(numeric))):
            spec = numeric[path]
            value = rng.uniform(float(spec["min"]), float(spec["max"]))
            value = round(value) if spec["type"] == "int" else round(value, 4)
            ops.append(f"(set policy_params.{path} {value})")
        diff_text = (
            f'(revision {program.version + 1} (parent {program.version}) (author "random-control") '
            f'(domain {program.domain}) (reason "random perturbation within bounds (control arm)"))\n'
            + "\n".join(ops)
        )
        manifest = build_manifest(program.version, program.domain,
                                  live_macros={m["name"]: m["body"] for m in program.macros})
        result = validate_diff(diff_text, program, manifest, hooks=ValidatorHooks())
        if result.ok:
            program = result.candidate  # same gates, same acceptance semantics
    path = os.path.join(workdir, "program_random.json")
    program.save(path)
    return path


def prepare_llm_arm(workdir: str, args) -> str:
    """Arm 1: run the gated revision loop (mock or live author), then benchmark the
    resulting program."""
    program_path = os.path.join(workdir, f"program_llm_{args.domain}.json")
    PolicyProgram.from_dict(
        DEFAULT_PROGRAM if args.domain == "nethack"
        else __import__("lox.executor", fromlist=["get_adapter"]).get_adapter(args.domain).default_program()
    ).save(program_path)
    from run_revision_loop import run_loop  # scripts/ is on sys.path
    from argparse import Namespace
    from lox.policy import compiler  # noqa: PLC0415
    loop_args = Namespace(
        max_revisions=args.llm_revisions, provider=args.llm_provider, model=args.llm_model,
        author_mode=args.author_mode, prompt_version=args.prompt_version,
        max_turns=args.max_turns,
        program_path=program_path, ledger_path=os.path.join(workdir, "ledger_llm.jsonl"),
        db_path=args.db_path, domain=args.domain, report_episodes=args.report_episodes,
        live_gates=False,   # gates inside the loop are the dry-run set;
        fixture_shadow=True,  # the ablation batch itself is the gate
        quick_batch_episodes=3, quick_batch_steps=5000, no_git=True,
    )
    # experiment harnesses must never overwrite the live authoring tree/compiled/archive
    with compiler.output_dirs(
            program_dir=os.path.join(workdir, "tree"),
            compiled_dir=os.path.join(workdir, "compiled"),
            archive_dir=os.path.join(workdir, "archive")):
        asyncio.run(run_loop(loop_args))
    return program_path


# ---------------------------------------------------------------------------
# Benchmark + statistics
# ---------------------------------------------------------------------------

def run_domain_batch(program_path: str, out_path: str, args) -> dict:
    """Transfer-domain batch (R5): in-process seeded episodes via the adapter.
    Seeds: 3 bases × episodes (§8: 3 seeds × 34 eps on seeded domains)."""
    from lox.executor import get_adapter  # noqa: PLC0415
    adapter = get_adapter(args.domain)
    program = PolicyProgram.load(program_path)
    max_steps = adapter.spec.step_limit_default
    seed_bases = [int(s) for s in args.seed_bases.split(",")]
    results = []
    t0 = time.perf_counter()
    for base in seed_bases:
        for i in range(args.episodes):
            seed = base + i
            env = adapter.make_env(seed=seed, task=args.task)
            r = adapter.run_episode(env, program, seed=seed, max_steps=max_steps)
            env.close()
            r["seed"] = seed
            results.append(r)
    wall = time.perf_counter() - t0
    summary = adapter.report_bundle(results)
    summary["detailed_episodes"] = results
    summary["wall_sec"] = wall
    summary["program_version"] = program.version
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_benchmark(program_path: str, out_path: str, args) -> dict:
    """Runs the benchmark subprocess against the given program (swapped onto the live
    program path), restoring the original afterward."""
    backup = None
    had = os.path.exists(DEFAULT_PROGRAM_PATH)
    if had:
        backup = DEFAULT_PROGRAM_PATH + ".ablation_backup"
        shutil.copy2(DEFAULT_PROGRAM_PATH, backup)
    shutil.copy2(program_path, DEFAULT_PROGRAM_PATH)
    try:
        cmd = [
            "uv", "run", "python", "scripts/run_benchmark.py",
            "--episodes", str(args.episodes), "--max-steps", str(args.max_steps),
            "--role", args.role, "--output", out_path,
        ]
        t0 = time.perf_counter()
        subprocess.run(cmd, check=True, capture_output=True, timeout=args.benchmark_timeout)
        wall = time.perf_counter() - t0
        with open(out_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        stats["wall_sec"] = wall
        return stats
    finally:
        if had:
            shutil.copy2(backup, DEFAULT_PROGRAM_PATH)
            os.unlink(backup)


def bootstrap_ci(values, n_resamples: int = 10000, alpha: float = 0.05, seed: int = 13):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    means = rng.choice(arr, size=(n_resamples, arr.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def cliffs_delta(a, b) -> float:
    """Nonparametric effect size: P(a > b) - P(a < b)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return 0.0
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return float((gt - lt) / (a.size * b.size))


def compare(arm_scores, frozen_scores):
    from scipy.stats import mannwhitneyu
    out = {}
    try:
        u, p = mannwhitneyu(arm_scores, frozen_scores, alternative="two-sided")
        out["mannwhitney_p"] = float(p)
    except Exception:
        out["mannwhitney_p"] = None
    out["cliffs_delta"] = cliffs_delta(arm_scores, frozen_scores)
    out["bootstrap_ci_95"] = bootstrap_ci(arm_scores)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="LOX-ψ R4 three-arm ablation harness")
    p.add_argument("--arms", type=str, default="frozen,random,llm",
                   help="comma-separated subset of frozen,random,llm")
    p.add_argument("--domain", type=str, default="nethack",
                   help="nethack (subprocess benchmark) or a transfer domain (adapter episodes)")
    p.add_argument("--task", type=str, default=None, help="transfer-domain task override")
    p.add_argument("--seed-bases", type=str, default="0,100,200",
                   help="transfer domains: 3 seed bases × --episodes each (§8)")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--role", type=str, default="valkyrie")
    p.add_argument("--db-path", type=str, default="data/lox_telemetry.duckdb")
    p.add_argument("--llm-provider", type=str, default="mock")
    p.add_argument("--llm-model", type=str, default=None)
    p.add_argument("--llm-revisions", type=int, default=5)
    p.add_argument("--author-mode", choices=["bundle", "agentic"], default="agentic",
                   help="llm-arm author: agentic tool-loop (S0) vs bundle-only Reviser — "
                        "the S0 exit-criterion comparison")
    p.add_argument("--prompt-version", type=str, default=None,
                   help="prompt version (data/prompts/<v>.md) for the llm arm")
    p.add_argument("--max-turns", type=int, default=6, help="agentic author tool-loop budget")
    p.add_argument("--report-episodes", type=int, default=10,
                   help="transfer domains: seeded episodes per revision for the author report")
    p.add_argument("--benchmark-timeout", type=int, default=14400)
    p.add_argument("--workdir", type=str, default="data/ablation_work")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print(f"[ablation] arms={arms} episodes={args.episodes} role={args.role}")
    results: dict = {"domain": args.domain, "role": args.role,
                     "episodes": args.episodes, "max_steps": args.max_steps,
                     "arms": {}, "comparisons": {}}

    programs = {}
    for arm in arms:
        if arm == "frozen":
            programs[arm] = prepare_frozen_arm(args.workdir)
        elif arm == "random":
            programs[arm] = prepare_random_arm(args.workdir, domain=args.domain)
        elif arm == "llm":
            programs[arm] = prepare_llm_arm(args.workdir, args)
        else:
            raise SystemExit(f"unknown arm {arm!r}")
        prog = PolicyProgram.load(programs[arm])
        print(f"  [{arm}] program v{prog.version} "
              f"({len(prog.tactic_rules)} rules, {len(prog.params)} tuned params)")

    for arm in arms:
        out_path = os.path.join("logs", f"ablation_{arm}_{int(time.time())}.json")
        print(f"[ablation] benchmarking arm={arm} ...")
        if args.domain == "nethack":
            stats = run_benchmark(programs[arm], out_path, args)
            episodes = stats.get("detailed_episodes", [])
            scores = [e.get("score", 0) for e in episodes]
            depths = [e.get("depth", 0) for e in episodes]
            results["arms"][arm] = {
                "mean_score": stats.get("mean_score"),
                "median_depth": stats.get("median_depth"),
                "scores": scores, "depths": depths,
                "wall_sec": stats.get("wall_sec"),
                "program_version": PolicyProgram.load(programs[arm]).version,
            }
            print(f"  [{arm}] mean_score={stats.get('mean_score')} "
                  f"median_depth={stats.get('median_depth')}")
        else:
            stats = run_domain_batch(programs[arm], out_path, args)
            episodes = stats.get("detailed_episodes", [])
            batch = stats.get("batch", {})
            success_steps = sorted(e["steps"] for e in episodes if e.get("success"))
            results["arms"][arm] = {
                "success_rate": batch.get("success_rate"),
                "mean_reward": batch.get("mean_reward"),
                "steps_on_success": success_steps,
                "median_steps_on_success": batch.get("median_steps_on_success"),
                "mean_coverage": batch.get("mean_coverage"),
                "wall_sec": stats.get("wall_sec"),
                "program_version": stats.get("program_version"),
            }
            print(f"  [{arm}] success_rate={batch.get('success_rate')} "
                  f"median_steps_on_success={batch.get('median_steps_on_success')}")

    if "frozen" in results["arms"]:
        frozen = results["arms"]["frozen"]
        frozen_metric = frozen.get("scores") or frozen.get("steps_on_success") or []
        for arm in arms:
            if arm == "frozen":
                continue
            arm_metric = results["arms"][arm].get("scores") or \
                results["arms"][arm].get("steps_on_success") or []
            results["comparisons"][arm] = compare(arm_metric, frozen_metric)
            c = results["comparisons"][arm]
            print(f"  [{arm} vs frozen] p={c['mannwhitney_p']} "
                  f"cliffs_delta={c['cliffs_delta']:.3f} ci95={c['bootstrap_ci_95']}")

    out = args.out or f"data/ablation_{int(time.time())}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[ablation] results written to {out}")

    # §8 acceptance for the paper (reported, not enforced — needs 100-ep batches)
    if "frozen" in results["comparisons"] or results["comparisons"]:
        for arm, c in results["comparisons"].items():
            sig = c["mannwhitney_p"] is not None and c["mannwhitney_p"] < 0.05
            print(f"  [{arm}] significant vs frozen: {sig} "
                  f"(δ={c['cliffs_delta']:.3f}) — requires 100-ep batches for the paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())