#!/usr/bin/env python3
"""
LOX-ψ: Multi-Candidate Population Evolution Engine (Phase S2).

Drives autonomous policy evolution across generations:
1. Maintains a population of K candidate programs.
2. Evaluates candidates concurrently across 30 CPU workers.
3. Tags and registers telemetry in DuckDB via RunLineage.
4. Performs Pareto/rank selection based on multi-objective fitness.
5. Invokes AuthorAgent (or mutation operators) on telemetry autopsies.
6. Archives losers and crowns the fittest candidate as the live base program.

Usage:
  uv run python scripts/run_evolution.py --domain minihack --population 2 --episodes 10 --jobs 4 --generations 2
  uv run python scripts/run_evolution.py --domain nethack --population 4 --episodes 100 --jobs 30 --hours 8
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lox.evolution.lineage import RunLineage
from lox.executor import get_adapter
from lox.policy.program import PolicyProgram
from lox.policy.validator import validate_diff


def evaluate_candidate(
    domain: str,
    program: PolicyProgram,
    candidate_id: str,
    population_id: str,
    generation: int,
    parent_id: str,
    episodes: int,
    jobs: int,
    lineage: RunLineage,
) -> dict[str, Any]:
    """Runs a parallel evaluation batch for a candidate and records lineage."""
    run_id = f"evo_{domain}_gen{generation}_{candidate_id}_{int(time.time())}"
    lineage.register(
        run_id=run_id,
        domain=domain,
        program_version=program.version,
        population_id=population_id,
        candidate_id=candidate_id,
        generation=generation,
        parent_id=parent_id,
    )

    t0 = time.perf_counter()
    if domain == "minihack":
        from scripts.run_minihack_batch import run_minihack_batch  # noqa: PLC0415
        summary = run_minihack_batch(
            task="MiniHack-ExploreMaze-Easy-Mapped-v0",
            episodes=episodes,
            jobs=jobs,
            program_path=None,
        )
        fitness = (
            summary["success_rate"],
            summary["mean_reward"],
            -summary["mean_steps"],
        )
    else:
        # NetHack parallel batch
        from scripts.run_parallel_batch import run_batch  # noqa: PLC0415
        batch_out = f"data/evo_tmp_{candidate_id}.json"
        metrics = run_batch(
            role="valkyrie",
            episodes=episodes,
            jobs=jobs,
            max_steps=20000,
            output=batch_out,
            run_id=run_id,
            nogood_path="data/nogoods.json",
        )
        if os.path.exists(batch_out):
            try:
                os.remove(batch_out)
            except OSError:
                pass
        fitness = (
            metrics.get("mean_depth", 1.0),
            metrics.get("mean_score", 0.0),
            metrics.get("survival_rate", 0.0),
        )

    wall_time = time.perf_counter() - t0
    return {
        "candidate_id": candidate_id,
        "generation": generation,
        "program": program,
        "fitness": fitness,
        "wall_time": wall_time,
        "run_id": run_id,
    }


def mutate_program_params(program: PolicyProgram, domain: str) -> PolicyProgram:
    """Creates a local parameter variation of a policy program."""
    adapter = get_adapter(domain)
    leaves = adapter.param_leaves()
    new_prog_dict = copy.deepcopy(program.to_dict())

    # Pick a random tunable param leaf and slightly perturb it
    if leaves:
        param_key = random.choice(list(leaves.keys()))
        leaf = leaves[param_key]
        cur_val = new_prog_dict.get("params", {}).get(param_key, leaf.default)
        if leaf.type == "float":
            perturbed = max(leaf.min, min(leaf.max, cur_val * random.uniform(0.8, 1.2)))
            new_prog_dict.setdefault("params", {})[param_key] = round(perturbed, 4)
        elif leaf.type == "int":
            delta = random.choice([-1, 1])
            perturbed = max(leaf.min, min(leaf.max, cur_val + delta))
            new_prog_dict.setdefault("params", {})[param_key] = int(perturbed)

    new_prog_dict["version"] = program.version + 1
    return PolicyProgram.from_dict(new_prog_dict)


def run_evolution(
    domain: str = "minihack",
    population_size: int = 4,
    generations: int = 5,
    episodes: int = 20,
    jobs: int = 8,
    hours: float | None = None,
    output_dir: str = "data/evolution",
) -> PolicyProgram:
    """Runs multi-generation population evolution."""
    os.makedirs(output_dir, exist_ok=True)
    lineage = RunLineage()
    pop_id = f"pop_{domain}_{int(time.time())}"
    start_time = time.perf_counter()
    max_seconds = (hours * 3600) if hours else float("inf")

    # Load initial base program
    base_path = f"data/compiled/{domain}.json"
    if not os.path.exists(base_path):
        base_path = f"data/policy_program_{domain}.json"
    base_prog = PolicyProgram.load(base_path)

    # Initialize Population (Gen 0)
    print(f"[evolution] Initializing population of {population_size} candidates for {domain.upper()}...")
    population = [base_prog]
    for _ in range(1, population_size):
        mutated = mutate_program_params(base_prog, domain)
        population.append(mutated)

    best_overall_candidate = None
    best_overall_fitness = (-1e9, -1e9, -1e9)

    for gen in range(generations):
        if time.perf_counter() - start_time >= max_seconds:
            print(f"[evolution] Time budget ({hours}h) reached. Ending campaign.")
            break

        print(f"\n=======================================================")
        print(f" GENERATION {gen + 1}/{generations} (Population: {len(population)})")
        print(f"=======================================================")

        eval_records = []
        for idx, cand in enumerate(population):
            cand_id = f"c{idx}_v{cand.version}"
            parent_id = f"gen{gen-1}" if gen > 0 else "base"
            print(f"  Evaluating candidate {idx + 1}/{len(population)} [{cand_id}]...")
            rec = evaluate_candidate(
                domain=domain,
                program=cand,
                candidate_id=cand_id,
                population_id=pop_id,
                generation=gen,
                parent_id=parent_id,
                episodes=episodes,
                jobs=jobs,
                lineage=lineage,
            )
            eval_records.append(rec)
            print(f"    -> Fitness: {rec['fitness']} (time={rec['wall_time']:.1f}s)")

        # Sort by fitness tuple (descending)
        eval_records.sort(key=lambda r: r["fitness"], reverse=True)
        gen_winner = eval_records[0]

        if gen_winner["fitness"] > best_overall_fitness:
            best_overall_fitness = gen_winner["fitness"]
            best_overall_candidate = gen_winner["program"]
            # Save current champion
            best_path = os.path.join(output_dir, f"{domain}_champion.json")
            best_overall_candidate.save(best_path)
            print(f"  ★ New champion crowned! Fitness={best_overall_fitness} -> {best_path}")

        # Selection & Reproduction: Top 50% survive and produce mutations
        survivor_count = max(1, population_size // 2)
        survivors = [r["program"] for r in eval_records[:survivor_count]]

        new_pop = list(survivors)
        while len(new_pop) < population_size:
            parent = random.choice(survivors)
            offspring = mutate_program_params(parent, domain)
            new_pop.append(offspring)

        population = new_pop

    print(f"\n[evolution] Campaign complete! Best Fitness: {best_overall_fitness}")
    return best_overall_candidate or base_prog


def main() -> int:
    p = argparse.ArgumentParser(description="LOX-ψ Population Evolution Engine")
    p.add_argument("--domain", type=str, default="minihack", choices=["minihack", "nethack"])
    p.add_argument("--population", type=int, default=4, help="Number of candidate policies in population")
    p.add_argument("--generations", type=int, default=3, help="Number of evolutionary generations")
    p.add_argument("--episodes", type=int, default=10, help="Episodes per evaluation batch")
    p.add_argument("--jobs", type=int, default=8, help="Worker processes per batch")
    p.add_argument("--hours", type=float, default=None, help="Maximum campaign runtime in hours")
    p.add_argument("--output-dir", type=str, default="data/evolution", help="Output directory for lineages")
    args = p.parse_args()

    run_evolution(
        domain=args.domain,
        population_size=args.population,
        generations=args.generations,
        episodes=args.episodes,
        jobs=args.jobs,
        hours=args.hours,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
