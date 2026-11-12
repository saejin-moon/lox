"""
LOX Swarm Campaign Orchestrator.
Coordinates the Hierarchical Multi-Agent Swarm:
1. Dynamic Macro-Category Discovery by the Meta-Agent based on empirical DuckDB bottlenecks & frontier progression.
2. Concurrent Category Workers exploring assigned domains on tier checkpoints and logging to LanceDB.
3. Meta-Agent Policy Synthesis assembling verified subtrees into Behavior Tree policies.
4. Multi-Core Worker Pool evaluating master policies on NetHack and logging to DuckDB.
5. Automated Frontier Checkpoint creation and MAP-Elites archive maintenance.
"""

from __future__ import annotations

import concurrent.futures
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from lox.author.swarm_meta import SwarmMetaAgent
from lox.author.swarm_worker import SwarmWorker
from lox.core.contracts import BatchEvaluation
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import DEFAULT_DB_PATH, init_db
from lox.db.theories import TheoryDocumentStore
from lox.engine.frontier import FrontierArchive
from lox.engine.map_elites import MAPElitesArchive
from lox.runner.pool import PersistentEvalPool
from lox.ui.console import ui

DEFAULT_WORKSPACE_DIR = "data/workspace"
DEFAULT_FRONTIER_DIR = "data/frontier"


class SwarmCampaignOrchestrator:
    """End-to-End Orchestrator for the LOX Multi-Agent Swarm."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        workspace_dir: str = DEFAULT_WORKSPACE_DIR,
        frontier_dir: str = DEFAULT_FRONTIER_DIR,
        provider: str = "openrouter",
        model: str = "z-ai/glm-5.3-flash:nitro",
        num_workers: int = 4,
        turns_per_worker: int = 3,
        seeds_per_batch: int = 8,
        benchmark: str = "nethack",
        role: str = "val-hum-law-fem",
    ) -> None:
        self.db_path = db_path
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.frontier_dir = frontier_dir
        self.provider = provider
        self.model = model
        self.num_workers = num_workers
        self.turns_per_worker = turns_per_worker
        self.seeds_per_batch = seeds_per_batch
        self.benchmark = benchmark
        self.role = role

        # Initialize DuckDB tables and logger
        init_db(self.db_path)
        self.db_logger = BatchDatabaseLogger(db_path=self.db_path)

        # Initialize LanceDB Theory Store & Frontier Archive
        theories_path = os.path.join(self.workspace_dir, "theories.lance")
        self.theories = TheoryDocumentStore(db_path=theories_path)
        self.frontier = FrontierArchive(frontier_dir=self.frontier_dir)

        # Initialize MAP-Elites Diversity Archive
        self.map_elites = MAPElitesArchive()

        # Initialize Meta-Agent (share live FrontierArchive so depth tiers stay in sync)
        self.meta_agent = SwarmMetaAgent(
            provider=self.provider,
            model=self.model,
            db_path=self.db_path,
            theories_path=theories_path,
            frontier_dir=self.frontier_dir,
        )
        self.meta_agent.frontier = self.frontier

        # Initialize Multi-Core Persistent Eval Pool
        self.eval_pool = PersistentEvalPool(
            workers=max(1, min(self.num_workers, 16)),
            role=self.role,
            max_steps=5000,
            benchmark=self.benchmark,
        )

    def close(self) -> None:
        """Shuts down worker pool and logs completion."""
        try:
            self.eval_pool.close()
        except Exception:
            pass

    def run_generation(self, generation: int) -> dict[str, Any]:
        """
        Executes a single generation of the Hierarchical Swarm loop:
        1. Meta-Agent discovers dynamic macro-categories from empirical DuckDB bottlenecks.
        2. Dispatches concurrent workers to formulate, code, and test on frontier checkpoints.
        3. Meta-Agent integrates worker reflection reports into unified Behavior Tree policy.
        4. PersistentEvalPool evaluates master policy across seeds.
        5. Logs batch to DuckDB and extracts new frontier checkpoints and MAP-Elites niches.
        """
        t0 = time.time()
        ui.header(f"LOX Swarm Campaign - Generation {generation}")

        # 1. Dynamic Macro-Category Discovery
        categories = self.meta_agent.discover_macro_categories(
            generation=generation, n_categories=max(1, int(self.num_workers))
        )
        cat_names = [c["category_id"] for c in categories]
        ui.info(f"Gen {generation}: Active Categories: {cat_names}")

        # 2. Concurrent Worker Exploration
        worker_reports: dict[str, dict[str, Any]] = {}
        theories_path = os.path.join(self.workspace_dir, "theories.lance")

        def _run_worker(cat: dict[str, Any]) -> dict[str, Any]:
            worker = SwarmWorker(
                role=cat["category_id"],
                description=cat.get("description", ""),
                target_metrics=cat.get("target_metrics", []),
                provider=self.provider,
                model=self.model,
                db_path=self.db_path,
                theories_path=theories_path,
                frontier_dir=self.frontier_dir,
                max_turns=self.turns_per_worker,
            )
            # Sample relevant checkpoint matching target depth tier
            target_depth = cat.get("target_depth_tier", (1, 2))[0]
            cp = self.frontier.sample_checkpoint(min_depth=target_depth)
            return worker.run_cycle(generation=generation, checkpoint=cp)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(categories)
        ) as executor:
            future_to_cat = {
                executor.submit(_run_worker, cat): cat["category_id"]
                for cat in categories
            }
            for fut in concurrent.futures.as_completed(future_to_cat):
                c_id = future_to_cat[fut]
                try:
                    res = fut.result()
                    worker_reports[c_id] = res
                except Exception as e:
                    ui.warn(f"Worker [{c_id}] failed: {e}")

        # 3. Meta-Agent Policy Synthesis & Assembly
        policy_path = self.workspace_dir / "policy.py"
        current_policy = (
            policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
        )
        master_policy_code = self.meta_agent.assemble_policy(
            generation=generation,
            worker_reports=worker_reports,
            current_policy=current_policy,
        )
        policy_path.write_text(master_policy_code, encoding="utf-8")
        ui.info(f"Gen {generation}: Master policy saved to `{policy_path}`.")

        # 4. Multi-Core Evaluation Batch on PersistentEvalPool
        ui.info(
            f"Gen {generation}: Evaluating master policy on {self.seeds_per_batch} seeds..."
        )
        seeds = [1000 + generation * 100 + i for i in range(self.seeds_per_batch)]
        batch_eval: BatchEvaluation = self.eval_pool.evaluate_batch(
            batch_id=f"batch_swarm_g{generation}_{int(time.time())}",
            policy_code=master_policy_code,
            macros_code="",
            target_episodes=self.seeds_per_batch,
            priority_seeds=seeds,
            record_ticks=True,
        )

        # 5. Log Batch and Telemetry to DuckDB
        runs = batch_eval.get("runs", [])
        batch_id = str(batch_eval.get("batch_id", f"batch_swarm_g{generation}"))
        batch_metrics = {
            "avg_depth": float(batch_eval.get("avg_depth", 1.0)),
            "p90_depth": float(batch_eval.get("p90_depth", 1.0)),
            "max_depth": int(batch_eval.get("max_depth", 1)),
            "avg_score": float(batch_eval.get("avg_score", 0.0)),
            "max_score": int(batch_eval.get("max_score", 0)),
            "avg_turns": float(batch_eval.get("avg_turns", 0.0)),
            "seeds_evaluated": len(runs),
            "is_early_stopped": bool(batch_eval.get("early_stopped", False)),
            "pilot_best_delta": 0.0,
            "total_cost_usd": 0.0,
        }
        policy_id = f"policy_swarm_g{generation}"
        self.db_logger.save_policy(
            policy_id=policy_id,
            generation=generation,
            mode="SWARM",
            policy_code=master_policy_code,
            macros_code="",
            params={},
            avg_depth=float(batch_metrics["avg_depth"]),
            max_depth=int(batch_metrics["max_depth"]),
            is_promoted=True,
        )
        log_res = self.db_logger.save_batch(
            batch_id=batch_id,
            generation=generation,
            mode="SWARM",
            parent_policy_id=f"policy_swarm_g{max(0, generation - 1)}",
            candidate_policy_id=policy_id,
            target_functions=[],
            runs=runs,
            episodes_discarded=int(batch_eval.get("episodes_discarded", 0)),
            batch_metrics=batch_metrics,
            verdict={"passed": True, "reason": f"Swarm Generation {generation} Master"},
        )

        # 6. Extract New Frontier Checkpoints from High-Performing Runs
        new_cps = 0
        for ep in runs:
            ep_dict = dict(ep) if isinstance(ep, dict) else ep.__dict__
            depth = max(int(ep_dict.get("depth", 1)), int(ep_dict.get("max_depth", 1)))
            act_bytes = bytes(ep_dict.get("action_bytes") or b"")
            if depth >= 2 and len(act_bytes) > 0:
                seed = int(ep_dict.get("seed", 0))
                cp_id = f"fc_dlvl_{depth}_seed{seed}_g{generation}"
                if cp_id not in self.frontier.checkpoints:
                    prefix_bytes = (
                        act_bytes[:-25]
                        if len(act_bytes) > 35
                        else (act_bytes[:-5] or act_bytes)
                    )
                    self.frontier.add_checkpoint(
                        seed=seed,
                        depth=depth,
                        milestone=f"dlvl_{depth}",
                        turns=int(ep_dict.get("turns", 0)),
                        action_bytes=prefix_bytes,
                        game_log=str(ep_dict.get("game_log", "")),
                        checkpoint_id=cp_id,
                    )
                    new_cps += 1

        # 7. Ingest into MAP-Elites Diversity Archive
        avg_depth = float(batch_eval.get("avg_depth", 1.0))
        max_d = int(batch_eval.get("max_depth", 1))
        avg_score = float(batch_eval.get("avg_score", 0.0))

        metrics = {
            "avg_depth": avg_depth,
            "max_depth": max_d,
            "avg_score": avg_score,
            "pct_wait": float(np.mean([r.get("pct_wait", 0.0) for r in runs]))
            if runs
            else 0.0,
            "pct_search": float(np.mean([r.get("pct_search", 0.0) for r in runs]))
            if runs
            else 0.0,
            "avg_turns": float(np.mean([r.get("turns", 0.0) for r in runs]))
            if runs
            else 100.0,
        }
        self.map_elites.add_candidate(
            policy_code=master_policy_code,
            metrics=metrics,
            batch_id=batch_id,
            generation=generation,
        )

        elapsed = time.time() - t0
        ui.info(
            f"Gen {generation} Complete in {elapsed:.1f}s: "
            f"Avg Depth: {avg_depth:.2f} (Max: {max_d}) | Avg Score: {avg_score:.1f} | "
            f"New Frontier CPs: {new_cps} | FunctionStats: {len(log_res)}"
        )

        return {
            "generation": generation,
            "avg_depth": avg_depth,
            "max_depth": max_d,
            "avg_score": avg_score,
            "new_frontier_checkpoints": new_cps,
            "elapsed_sec": elapsed,
            "worker_reports": len(worker_reports),
        }

    def run(self, generations: int = 5) -> list[dict[str, Any]]:
        """Runs the full multi-generation swarm campaign."""
        ui.info(
            f"Starting LOX Swarm Campaign: {generations} generations, {self.num_workers} workers, "
            f"provider={self.provider}, model={self.model}"
        )
        history: list[dict[str, Any]] = []
        try:
            for g in range(1, generations + 1):
                res = self.run_generation(generation=g)
                history.append(res)
        finally:
            self.close()

        ui.header("LOX Swarm Campaign Completed")
        return history
