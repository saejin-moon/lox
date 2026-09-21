"""
Multiprocessing Batch Runner: N worker processes, each building its own env+agent
and running a slice of episodes; per-episode dicts are merged in the parent.

Rationale (R5→R6 handoff item 2): NetHack SPS is dominated by NLE's C-side env
stepping, not Python — parallelism is the statistics-throughput lever. Verified:
NLE is process-safe (NOT thread-safe); use multiprocessing, never threads.

Nogood semantics: workers start from the same initial nogood store (no mid-batch
cross-worker learning — that would be order-dependent); the parent merges all
learned nogoods by union at the end, which is exactly the serial store's content
modulo episode ordering.
"""

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from corp.telemetry import DuckDBConsolidator, generate_base62_id  # noqa: E402

ROLE_CHARACTER_MAP = {
    "valkyrie": "val-hum-law-fem",
    "barbarian": "bar-hum-neu-mal",
    "samurai": "sam-hum-law-mal",
    "wizard": "wiz-hum-neu-mal",
    "monk": "mon-hum-neu-mal",
    "rogue": "rog-hum-cha-mal",
    "tourist": "tou-hum-neu-mal",
}


def _worker_main(worker_cfg: dict[str, Any]) -> dict[str, Any]:
    """Runs this worker's slice of episodes in a fresh process (own env+agent)."""
    # Deferred imports: every worker process builds its own NLE/env/agent state.
    from corp.agent.corp_agent import CORPAgent  # noqa: PLC0415
    from corp.env.nle_wrapper import make_env  # noqa: PLC0415
    from corp.planner.nogood import NogoodStore  # noqa: PLC0415
    from corp.telemetry import ParquetLogger  # noqa: PLC0415

    wid = worker_cfg["worker_id"]
    episode_ids = worker_cfg["episode_ids"]
    run_base = worker_cfg["run_id"]
    parquet_dir = worker_cfg["parquet_dir"]

    nogood_store = NogoodStore()
    if os.path.exists(worker_cfg["nogood_path"]):
        nogood_store.load_from_json(worker_cfg["nogood_path"])

    # Per-worker parquet logger: distinct run_id per worker keeps file ownership
    # exclusive; the parent consolidates the whole parquet dir at the end.
    logger = ParquetLogger(base_dir=parquet_dir, run_id=f"{run_base}_w{wid}")

    results: list[dict[str, Any]] = []
    for ep_no in episode_ids:
        character = (
            ROLE_CHARACTER_MAP.get(worker_cfg["role"].lower(), worker_cfg["role"])
            if worker_cfg["role"] != "all"
            else "*"
        )
        env = make_env(character=character)
        agent = CORPAgent(
            env=env,
            nogood_store=nogood_store,
            llm_provider=None,  # parallel batches run without deliberative LLM calls
            enable_deliberative_autopsy=False,
            enable_in_game_deliberation=False,
            parquet_logger=logger,
            eval_type=worker_cfg["eval_type"],
            mode="random",
            seed=worker_cfg["seed"] + ep_no,
            role=worker_cfg["role"] if worker_cfg["role"] != "all" else "unknown",
        )
        ep_start = time.perf_counter()
        try:
            result = asyncio.run(agent.run_episode_async(max_steps=worker_cfg["max_steps"]))
        except Exception as exc:  # a worker must never take down the batch
            results.append({
                "episode": ep_no,
                "worker": wid,
                "character": character,
                "error": f"{type(exc).__name__}: {exc}",
            })
            env.close()
            continue
        results.append({
            "episode": ep_no,
            "worker": wid,
            "character": character,
            "turns": result.turns,
            "max_depth": result.max_depth,
            "score": result.final_score,
            "hp": result.final_hp,
            "ascended": result.is_ascended,
            "death_message": result.death_message,
            "active_nogoods": result.active_nogoods_count,
            "sps": result.steps_per_second,
            "duration_sec": time.perf_counter() - ep_start,
        })
        env.close()

    logger.flush_all()

    from dataclasses import asdict  # noqa: PLC0415
    return {
        "worker": wid,
        "episodes": results,
        # entries are plain dicts; parent unions them across workers
        "nogoods": [asdict(e) for e in nogood_store.entries],
    }


def _merge_nogoods(worker_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Union of nogood entries across workers, canonical-first (dedup by full tuple)."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for payload in sorted(worker_payloads, key=lambda p: p["worker"]):
        for entry in payload["nogoods"]:
            key = json.dumps(entry, sort_keys=True)
            if key not in seen:
                seen.add(key)
                merged.append(entry)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="CORP Parallel Multiprocessing Batch Runner")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--role", type=str, default="valkyrie")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 4)),
                        help="Worker processes (each owns its env+agent)")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--eval-type", choices=["fast_prelim", "research_grade", "training"],
                        default="research_grade")
    parser.add_argument("--nogood-path", type=str, default="data/nogoods.json")
    parser.add_argument("--parquet-dir", type=str, default="logs/parquet")
    parser.add_argument("--duckdb-path", type=str, default="data/corp_telemetry.duckdb")
    parser.add_argument("--clean-parquet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-consolidate", action="store_true",
                        help="Skip DuckDB consolidation (parquet stays on disk)")
    parser.add_argument("--output", type=str, default="data/parallel_batch_results.json")
    args = parser.parse_args()

    run_id = args.run_id or generate_base62_id()
    # Round-robin episode assignment: workers finish roughly together (episode
    # durations vary, but assignment cost is trivial either way).
    assignments: list[list[int]] = [[] for _ in range(args.jobs)]
    for ep in range(1, args.episodes + 1):
        assignments[ep % args.jobs].append(ep)

    worker_cfgs = [{
        "worker_id": w,
        "episode_ids": assignments[w],
        "run_id": run_id,
        "role": args.role,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "eval_type": args.eval_type,
        "nogood_path": args.nogood_path,
        "parquet_dir": args.parquet_dir,
    } for w in range(args.jobs) if assignments[w]]

    print("=" * 80)
    print(f"CORP Parallel Batch Runner | Run {run_id}")
    print(f"Episodes: {args.episodes} @ {args.max_steps} steps | Jobs: {len(worker_cfgs)} | Role: {args.role}")
    print("=" * 80)

    ctx = mp.get_context("spawn")  # NLE is process-safe, not fork-state-safe
    start_total = time.perf_counter()
    with ctx.Pool(processes=len(worker_cfgs)) as pool:
        payloads = pool.map(_worker_main, worker_cfgs)

    results: list[dict[str, Any]] = []
    for p in payloads:
        results.extend(p["episodes"])
    results.sort(key=lambda r: r["episode"])

    merged_nogoods = _merge_nogoods(payloads)

    # Persist merged nogoods (union across workers).
    if not os.path.exists(args.nogood_path):
        os.makedirs(os.path.dirname(args.nogood_path) or ".", exist_ok=True)
        json.dump([], open(args.nogood_path, "w"))
    from corp.planner.nogood import NogoodEntry, NogoodStore  # noqa: PLC0415
    from dataclasses import asdict  # noqa: PLC0415
    store = NogoodStore()
    if os.path.exists(args.nogood_path):
        store.load_from_json(args.nogood_path)
    existing = {json.dumps(asdict(e), sort_keys=True) for e in store.entries}
    for entry in merged_nogoods:
        if json.dumps(entry, sort_keys=True) not in existing:
            store.entries.append(NogoodEntry(**entry))
    store.save_to_json(args.nogood_path)

    ok = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    total_elapsed = time.perf_counter() - start_total

    depths = [r["max_depth"] for r in ok] or [0]
    scores = [r["score"] for r in ok] or [0]
    turns = [r["turns"] for r in ok] or [0]
    sps_vals = [r["sps"] for r in ok] or [0]

    summary = {
        "run_id": run_id,
        "episodes": args.episodes,
        "jobs": len(worker_cfgs),
        "total_elapsed_sec": total_elapsed,
        "worker_sps_equivalent": float(sum(turns)) / max(total_elapsed, 1e-9),
        "median_depth": float(np.median(depths)),
        "mean_score": float(np.mean(scores)),
        "mean_turns": float(np.mean(turns)),
        "mean_sps": float(np.mean(sps_vals)),
        "ascension_rate": (sum(1 for r in ok if r["ascended"]) / len(ok)) if ok else 0.0,
        "error_count": len(errors),
        "errors": errors,
        "detailed_episodes": results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    for r in results:
        if "error" in r:
            print(f"Ep {r['episode']:3d} [w{r['worker']}] ERROR: {r['error'][:90]}")
            continue
        print(
            f"Ep {r['episode']:3d} [w{r['worker']}] [{r['character']:8s}] "
            f"Turns: {r['turns']:5d} | Depth: {r['max_depth']:2d} | Score: {r['score']:5d} | "
            f"SPS: {r['sps']:7.1f} | Death: {(r['death_message'] or 'Survived')[:30]}"
        )
    print("=" * 80)
    print(f"Median Depth: {summary['median_depth']:.1f} | Mean Score: {summary['mean_score']:.1f} | "
          f"Mean Turns: {summary['mean_turns']:.1f} | Wall SPS-equiv: {summary['worker_sps_equivalent']:.1f}")
    print(f"Nogoods merged: {len(merged_nogoods)} entries | Errors: {len(errors)}")
    print(f"Results written to {args.output} ({total_elapsed:.1f}s wall)")

    if not args.no_consolidate:
        consolidator = DuckDBConsolidator(db_path=args.duckdb_path, parquet_dir=args.parquet_dir)
        c_counts = consolidator.consolidate(clean_parquet=args.clean_parquet)
        print(f"DuckDB: consolidated {c_counts['episodes']} episodes & {c_counts['ticks']} ticks")


if __name__ == "__main__":
    main()
