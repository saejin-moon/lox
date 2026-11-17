"""
LOX Swarm Campaign Orchestrator ("Hard Code Nothing").
Coordinates the Hierarchical Multi-Agent Swarm with a Two-Pass Advisory Critic
and ZERO hardcoded rollbacks:
1. Critic Pass 1 (`guide_before_generation`): Runs as the very first agent every
   generation to audit DuckDB telemetry, branch turn shares, and game mechanics,
   providing a high-level strategic guide to the Meta-Agent.
2. Meta-Agent Dynamic Macro-Category Discovery (`discover_macro_categories`).
3. Concurrent Category Workers (`SwarmWorker.run_cycle`) exploring hypotheses on
   frontier checkpoints and failure fixtures.
4. Meta-Agent Initial Policy Assembly (`assemble_policy`) + Paired Seed Evaluation.
5. Critic Pass 2 (`review_after_assembly`): Compares the Meta-Agent's evaluated draft
   and actions against previous generations and informs the Meta-Agent.
6. Meta-Agent Post-Critic Revision (`revise_policy_with_critic`): The Meta-Agent
   decides for itself whether to keep its draft or revise/restore capabilities.
7. Frontier Checkpoint extraction and MAP-Elites diversity archive maintenance.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from lox.author.swarm_critic import SwarmCriticAgent
from lox.author.swarm_deficits import DeficitAnalyzer
from lox.author.swarm_meta import SwarmMetaAgent
from lox.author.swarm_peer import PeerCandidate, SwarmPeerAgent
from lox.author.swarm_worker import SwarmWorker
from lox.core.contracts import BatchEvaluation
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import DEFAULT_DB_PATH, connect_db, init_db
from lox.db.shards import mine_organic_scenarios
from lox.db.theories import TheoryDocumentStore
from lox.engine.frontier import FrontierArchive
from lox.engine.map_elites import MAPElitesArchive
from lox.engine.orchestrator import compute_adaptive_max_steps, write_status_report
from lox.engine.swarm_referee import SwarmReferee, _package_fingerprint
from lox.lang.modularize import maybe_auto_modularize
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
        max_steps: int = 5000,
        frontier_min_depth: int = 2,
        swarm_architecture: str = "peer",
        eval_workers: int | None = None,
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
        self.max_steps = int(max_steps)
        self.frontier_min_depth = int(frontier_min_depth)
        if eval_workers is not None and int(eval_workers) > 0:
            self.eval_workers = max(1, min(int(eval_workers), 32))
        else:
            cpu_cap = max(4, (os.cpu_count() or 8) - 2)
            self.eval_workers = max(
                self.num_workers, min(int(self.seeds_per_batch), cpu_cap)
            )
        self.swarm_architecture = (
            str(swarm_architecture or "peer").strip().lower()
            if str(swarm_architecture or "peer").strip().lower() in ("peer", "legacy")
            else "peer"
        )

        # Initialize DuckDB tables and logger
        init_db(self.db_path)
        self.db_logger = BatchDatabaseLogger(db_path=self.db_path)

        # Initialize Unified DuckDB Theory Store & Frontier Archive
        theories_path = self.db_path
        self.theories = TheoryDocumentStore(db_path=theories_path)
        self.frontier = FrontierArchive(frontier_dir=self.frontier_dir)

        # Initialize MAP-Elites Diversity Archive and hydrate persisted elites/plateau state
        self.map_elites = MAPElitesArchive()
        try:
            _con_init = connect_db(self.db_path, read_only=True)
            try:
                self.map_elites.load_from_db(_con_init)
            finally:
                _con_init.close()
        except Exception:
            pass

        # Initialize Deficit Analyzer & Deterministic Referee (for peer mode)
        self.deficit_analyzer = DeficitAnalyzer(db_path=self.db_path)
        self.referee = SwarmReferee(
            max_tournament_peers=max(2, min(4, self.num_workers))
        )
        self.campaign_start_time = time.time()
        self._plateau_streak: int = self._compute_depth_plateau_streak()
        self._attempted_ast_modular_fps: set[str] = set()

        # Legacy Meta-Agent & Two-Pass Advisory Critic (only used when swarm_architecture='legacy')
        self.meta_agent = SwarmMetaAgent(
            provider=self.provider,
            model=self.model,
            db_path=self.db_path,
            theories_path=theories_path,
            frontier_dir=self.frontier_dir,
        )
        self.meta_agent.frontier = self.frontier
        self.meta_agent.author.tools.theories_store = self.meta_agent.theories
        self.critic = SwarmCriticAgent(
            provider=self.provider,
            model=self.model,
            db_path=self.db_path,
            theories_store=self.theories,
        )

        # Initialize Multi-Core Persistent Eval Pool
        self.eval_pool = PersistentEvalPool(
            workers=max(1, self.eval_workers),
            role=self.role,
            max_steps=self.max_steps,
            benchmark=self.benchmark,
        )

    def close(self) -> None:
        """Shuts down worker pool and logs completion."""
        try:
            self.eval_pool.close()
        except Exception:
            pass

    def _delete_batch_artifacts(self, batch_id: str) -> None:
        """Removes draft run/tick/function_stats rows when replacing a batch after Meta-Agent revision."""
        try:
            con = connect_db(self.db_path, read_only=False)
            try:
                con.execute(
                    "DELETE FROM ticks WHERE run_id IN (SELECT run_id FROM runs WHERE batch_id = ?)",
                    [batch_id],
                )
                con.execute("DELETE FROM runs WHERE batch_id = ?", [batch_id])
                con.execute("DELETE FROM function_stats WHERE batch_id = ?", [batch_id])
            finally:
                con.close()
        except Exception:
            pass

    def _sync_batch_cost(self, batch_id: str, generation: int) -> None:
        """Updates `batches.total_cost_usd` and `total_llm_tokens` after Pass 2 and Meta-Agent revision."""
        try:
            con = connect_db(self.db_path, read_only=False)
            try:
                row_c = con.execute(
                    "SELECT COALESCE(SUM(cost_usd), 0.0), COALESCE(SUM(prompt_tokens + completion_tokens), 0) "
                    "FROM synthesis_logs WHERE generation = ? OR batch_id = ?",
                    [generation, batch_id],
                ).fetchone()
                if row_c:
                    con.execute(
                        "UPDATE batches SET total_cost_usd = ?, total_llm_tokens = ? WHERE batch_id = ?",
                        [float(row_c[0] or 0.0), int(row_c[1] or 0), batch_id],
                    )
            finally:
                con.close()
        except Exception:
            pass

    def _get_start_generation(self) -> int:
        """Returns `MAX(generation) + 1` from DuckDB `batches` so restarted overnight runs resume cleanly."""
        try:
            con = connect_db(self.db_path, read_only=True)
            try:
                row = con.execute(
                    "SELECT COALESCE(MAX(generation), 0) FROM batches"
                ).fetchone()
                return int(row[0] or 0) + 1
            finally:
                con.close()
        except Exception:
            return 1

    @staticmethod
    def _extract_batch_metrics(
        batch_eval: BatchEvaluation, runs: list[Any]
    ) -> dict[str, Any]:
        return {
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

    def run_generation(self, generation: int) -> dict[str, Any]:
        """Dispatches generation execution to the selected swarm architecture (`peer` or `legacy`)."""
        if self.swarm_architecture == "peer":
            return self._run_generation_peer(generation)
        return self._run_generation_legacy(generation)

    def _run_generation_peer(self, generation: int) -> dict[str, Any]:
        """
        Democratized Peer Swarm loop:
        1. Deterministic SQL `DeficitAnalyzer` partitions `D_succ` vs `D_fail`, plateau signal, and dormant branches.
        2. Adaptive N symmetric `SwarmPeerAgent`s claim distinct empirical clusters in isolated sandboxes
           and incubate candidates on a paired micro-shard (`tau_iso`).
        3. Deterministic `SwarmReferee` builds optional AST-disjoint orthogonal pair and runs full-CRN
           head-to-head tournament vs Incumbent (`delta >= 0` promotion gate).
        4. Updates `workspace/`, DuckDB telemetry, `FrontierArchive`, and `MAPElitesArchive`.
        """
        t0 = time.time()
        ui.header(f"LOX Democratized Peer Swarm - Generation {generation}")

        policy_path = self.workspace_dir / "policy.py"
        skills_dir = self.workspace_dir / "skills"
        current_policy = (
            policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
        )
        initial_skills = (
            {
                p.stem: p.read_text(encoding="utf-8")
                for p in sorted(skills_dir.glob("*.py"))
                if p.stem != "__init__"
            }
            if skills_dir.exists()
            else {}
        )

        # 1. Deterministic Deficit Discovery + Organic Scenario Bank + Adaptive Horizon
        active_max_steps = self._compute_active_max_steps()
        self._plateau_streak = self._compute_depth_plateau_streak()
        deficit_report = self.deficit_analyzer.analyze(
            generation=generation,
            current_policy=current_policy,
            skills=initial_skills,
        )
        try:
            deficit_report.organic_shards = mine_organic_scenarios(
                self.db_path,
                frontier_dir=self.frontier_dir,
                max_total=8,
            )
        except Exception:
            deficit_report.organic_shards = []

        if self._plateau_streak >= 2 and deficit_report.regime != "BOOTSTRAP":
            deficit_report.regime = "PLATEAU"
        if self._plateau_streak >= 2 and getattr(self.map_elites, "cells", None):
            top_elites = sorted(
                self.map_elites.cells.values(),
                key=lambda e: float(e.fitness),
                reverse=True,
            )[:3]
            deficit_report.plateau_hints = [
                f"{el.cell_id} (batch={el.batch_id}, gen={el.generation}, D={el.metrics.get('avg_depth', 1.0):.2f}/AC={el.metrics.get('avg_ac', 10.0):.1f}/S={el.metrics.get('avg_score', 0.0):.0f})"
                for el in top_elites
            ]
        if deficit_report.regime == "CLIMBING":
            active_peers = max(1, min(self.num_workers, max(2, self.num_workers // 2)))
        else:
            active_peers = max(1, int(self.num_workers))

        # Select orthogonal clusters with disjoint dominant_fns when possible
        clusters = deficit_report.fail_clusters
        orthogonal_clusters: list[Any] = []
        used_dom_fns: set[str] = set()
        for cl in clusters:
            dom = str((cl.dominant_fns or [""])[0] or "").strip()
            if not dom or dom not in used_dom_fns:
                orthogonal_clusters.append(cl)
                if dom:
                    used_dom_fns.add(dom)
        if not orthogonal_clusters:
            orthogonal_clusters = list(clusters)

        ui.info(
            f"Gen {generation}: Regime={deficit_report.regime} "
            f"(Δ={deficit_report.rolling_depth_delta:+.2f}, streak={self._plateau_streak}) | "
            f"Clusters={len(clusters)} (ortho={len(orthogonal_clusters)}) | Peers={active_peers} | MaxSteps={active_max_steps}"
        )

        # 2. Shared Paired CRN Seeds (100% Fresh Per Generation: S_g != S_{g-1}) + Hidden Holdout Seeds
        seeds = [
            10_000 + int(generation) * 100 + i for i in range(self.seeds_per_batch)
        ]
        holdout_seeds = [
            100_000 + i for i in range(max(4, min(8, int(self.seeds_per_batch))))
        ]
        raw_shard_seeds = list(
            dict.fromkeys(
                deficit_report.d_succ_seeds[:2]
                + [sd for cl in clusters for sd in cl.sample_seeds[:2]]
                + seeds[:2]
            )
        )
        max_shard = min(6, max(2, int(self.seeds_per_batch)))
        shard_seeds = raw_shard_seeds[:max_shard]
        incumbent_shard_runs: dict[int, dict[str, Any]] = {}
        if current_policy.strip() and shard_seeds:
            full_seeds_key = tuple(int(s) for s in seeds)
            inc_fp = _package_fingerprint(current_policy, initial_skills)
            cache_k = (
                id(self.eval_pool),
                inc_fp,
                full_seeds_key,
                int(active_max_steps),
            )
            cached_pair = self.referee._incumbent_cache.get(cache_k)
            if cached_pair is not None:
                for r in cached_pair[0].get("runs", []) or []:
                    sd_i = int(r.get("seed", 0))
                    if sd_i in shard_seeds:
                        incumbent_shard_runs[sd_i] = dict(r)
            missing_shard_seeds = [
                s for s in shard_seeds if s not in incumbent_shard_runs
            ]
            if missing_shard_seeds:
                inc_shard = self.eval_pool.evaluate_batch(
                    batch_id=f"incubate_inc_g{generation}_{int(time.time())}",
                    policy_code=current_policy,
                    macros_code="",
                    skills=initial_skills,
                    target_episodes=len(missing_shard_seeds),
                    priority_seeds=missing_shard_seeds,
                    max_steps=active_max_steps,
                    record_ticks=False,
                )
                for r in inc_shard.get("runs", []) or []:
                    incumbent_shard_runs[int(r.get("seed", 0))] = dict(r)

        # 3. Concurrent Symmetric Peer Agents + Plateau MAP-Elites UCB Branching
        peer_assignments: list[tuple[str, Any, str, dict[str, str], str | None]] = []
        sampled_elite_ids: list[str] = []
        use_elite_reseed = (
            self._plateau_streak >= 2
            and len(getattr(self.map_elites, "cells", {})) >= 2
            and bool(current_policy.strip())
        )
        reseed_slots = max(1, active_peers // 2) if use_elite_reseed else 0
        for i in range(active_peers):
            cl = orthogonal_clusters[(generation + i) % len(orthogonal_clusters)]
            start_pol = current_policy
            start_sk = initial_skills
            elite_cid: str | None = None
            if i < reseed_slots:
                el = self.map_elites.sample_elite(strategy="ucb_curiosity")
                if el is not None and str(el.policy_code or "").strip():
                    start_pol = el.policy_code
                    start_sk = dict(el.skills or initial_skills)
                    elite_cid = el.cell_id
                    sampled_elite_ids.append(elite_cid)
            peer_assignments.append(
                (f"peer_{i + 1}", cl, start_pol, start_sk, elite_cid)
            )

        peer_tool_turns = max(6, int(self.turns_per_worker) * 3)
        peer_candidates: list[PeerCandidate] = []

        def _run_peer(
            peer_id: str,
            cluster: Any,
            p_policy: str,
            p_skills: dict[str, str],
        ) -> PeerCandidate:
            peer = SwarmPeerAgent(
                peer_id=peer_id,
                provider=self.provider,
                model=self.model,
                db_path=self.db_path,
                theories_store=self.theories,
                max_tool_turns=peer_tool_turns,
            )
            return peer.run_cycle(
                generation=generation,
                deficit_report=deficit_report,
                claimed_deficit=cluster,
                current_policy=p_policy,
                current_skills=p_skills,
                eval_pool=self.eval_pool,
                incumbent_shard_runs=incumbent_shard_runs,
                shard_seeds=shard_seeds,
                frontier=self.frontier,
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(peer_assignments)
        ) as executor:
            fut_map = {
                executor.submit(_run_peer, pid, cl, sp, ss): pid
                for pid, cl, sp, ss, _ecid in peer_assignments
            }
            for fut in concurrent.futures.as_completed(fut_map):
                pid = fut_map[fut]
                try:
                    peer_candidates.append(fut.result())
                except Exception as exc:
                    ui.warn(f"Peer [{pid}] failed: {exc}")

        # 4. Deterministic CRN Tournament + AST 3-Way Composition + Hidden Holdout Audit
        verdict = self.referee.evaluate_tournament(
            generation=generation,
            candidates=peer_candidates,
            base_policy=current_policy,
            base_skills=initial_skills,
            eval_pool=self.eval_pool,
            seeds=seeds,
            max_steps=active_max_steps,
            holdout_seeds=holdout_seeds,
        )

        # 4b. Sparse Hysteresis Circuit Breaker: deterministic bit-identical AST modularization
        active_pol = verdict.winner_policy_code
        active_skills = dict(verdict.winner_skills or {})
        active_eval = (
            verdict.winner_batch_eval
            if verdict.promoted
            else verdict.incumbent_batch_eval
        )
        active_fp = (
            _package_fingerprint(active_pol, active_skills)
            if active_pol.strip()
            else ""
        )
        if active_pol.strip() and active_fp not in self._attempted_ast_modular_fps:
            self._attempted_ast_modular_fps.add(active_fp)
            changed, mod_pol, mod_skills, mod_msg = maybe_auto_modularize(
                active_pol,
                active_skills,
                verify_seeds=seeds[:6],
                verify_steps=300,
            )
            if changed:
                ui.info(f"Gen {generation} AST Auto-Modularizer: {mod_msg}")
                verdict.winner_policy_code = mod_pol
                verdict.winner_skills = mod_skills
                if active_eval is not None:
                    verdict.winner_batch_eval = active_eval
                verdict.promoted = True
                verdict.reason = f"{verdict.reason} | {mod_msg}"
                if active_eval is not None:
                    self.referee.cache_incumbent(
                        eval_pool=self.eval_pool,
                        policy_code=mod_pol,
                        skills=mod_skills,
                        seeds=seeds,
                        max_steps=active_max_steps,
                        batch_eval=active_eval,
                    )
                self._attempted_ast_modular_fps.add(
                    _package_fingerprint(mod_pol, mod_skills)
                )

        # Sync workspace only if promoted (or bootstrapping / auto-modularized)
        if verdict.promoted:
            policy_path.write_text(verdict.winner_policy_code, encoding="utf-8")
            skills_dir.mkdir(parents=True, exist_ok=True)
            for s_name, s_src in verdict.winner_skills.items():
                (skills_dir / f"{s_name}.py").write_text(s_src, encoding="utf-8")
            # Prune removed skill files cleanly
            SwarmMetaAgent.prune_unused_skills(
                verdict.winner_policy_code,
                verdict.winner_skills,
                skills_dir=skills_dir,
            )

        batch_eval = verdict.winner_batch_eval
        runs = batch_eval.get("runs", [])
        batch_id = str(
            batch_eval.get("batch_id", f"batch_swarm_g{generation}_{int(time.time())}")
        )
        batch_metrics = self._extract_batch_metrics(batch_eval, runs)
        policy_id = f"policy_swarm_g{generation}"
        parent_id = f"policy_swarm_g{max(0, generation - 1)}"

        self.db_logger.save_policy(
            policy_id=policy_id,
            generation=generation,
            mode="SWARM",
            policy_code=verdict.winner_policy_code,
            macros_code="",
            params={},
            avg_depth=float(batch_metrics["avg_depth"]),
            max_depth=int(batch_metrics["max_depth"]),
            is_promoted=verdict.promoted,
            skills=verdict.winner_skills,
        )
        log_res = self.db_logger.save_batch(
            batch_id=batch_id,
            generation=generation,
            mode="SWARM",
            parent_policy_id=parent_id,
            candidate_policy_id=policy_id,
            target_functions=verdict.target_functions,
            runs=runs,
            episodes_discarded=int(batch_eval.get("episodes_discarded", 0)),
            batch_metrics=batch_metrics,
            verdict={
                "passed": verdict.promoted,
                "reason": verdict.reason,
                "scoreboard": verdict.scoreboard,
            },
        )

        # Persist challenger tournament attempts (including rejected peers + tracebacks)
        try:
            self.db_logger.save_tournament_attempts(verdict.attempts_summary)
        except Exception:
            pass

        # 5. Extract Frontier Checkpoints (all deep/valid runs) & Ingest into MAP-Elites v2 Archive
        new_cps = self._ingest_frontier_and_map_elites(
            generation=generation,
            runs=runs,
            batch_eval=batch_eval,
            policy_code=verdict.winner_policy_code,
            skills=verdict.winner_skills,
            batch_id=batch_id,
            runner_ups=verdict.all_runner_up_batches,
            allow_frontier=bool(verdict.promoted or generation <= 2 or runs),
        )
        if sampled_elite_ids:
            try:
                _con_u = connect_db(self.db_path, read_only=False)
                try:
                    for ecid in sampled_elite_ids:
                        self.map_elites.record_outcome(
                            ecid, bool(verdict.promoted), con=_con_u
                        )
                finally:
                    _con_u.close()
            except Exception:
                pass
        self._sync_batch_cost(batch_id, generation)
        self._plateau_streak = self._compute_depth_plateau_streak()

        if self.theories is not None and hasattr(self.theories, "save_document"):
            try:
                sb_lines = [
                    f"{row.get('peer_id')} ({row.get('deficit')}): D={row.get('avg_depth')}, maxD={row.get('max_depth')}, S={row.get('avg_score')}, stalls={row.get('stalls')}"
                    for row in (verdict.scoreboard or [])
                ]
                status_word = "PROMOTED" if verdict.promoted else "KEPT_INCUMBENT"
                self.theories.save_document(
                    generation=generation,
                    agent_role="referee",
                    doc_type="META_SYNTHESIS",
                    title=f"Gen {generation} Tournament {status_word}: winner={verdict.winner_id}",
                    content=(
                        f"Generation {generation} CRN Tournament ({status_word})\n"
                        f"Reason: {verdict.reason}\n"
                        f"Target Functions: {verdict.target_functions}\n"
                        f"Incumbent Depth={verdict.incumbent_avg_depth:.2f}, Score={verdict.incumbent_avg_score:.1f} -> "
                        f"Winner Depth={verdict.winner_avg_depth:.2f}, Score={verdict.winner_avg_score:.1f}\n"
                        + (
                            "Scoreboard:\n- " + "\n- ".join(sb_lines)
                            if sb_lines
                            else ""
                        )
                    ),
                    tags=[verdict.winner_id, status_word.lower()]
                    + [str(t) for t in (verdict.target_functions or [])[:5]],
                    metrics={
                        "promoted": bool(verdict.promoted),
                        "incumbent_avg_depth": float(verdict.incumbent_avg_depth),
                        "winner_avg_depth": float(verdict.winner_avg_depth),
                        "winner_avg_score": float(verdict.winner_avg_score),
                    },
                )
            except Exception:
                pass

        try:
            write_status_report(
                workspace_dir=str(self.workspace_dir),
                db_path=self.db_path,
                gen=generation,
                start_time=self.campaign_start_time,
                baseline_runs=runs,
                active_max_steps=active_max_steps,
                current_policy=verdict.winner_policy_code,
                macro_code="",
                skills_count=len(verdict.winner_skills),
                niches=sorted(getattr(self.map_elites, "cells", {}).keys()) or ["peer"],
            )
        except Exception:
            pass

        if generation % 5 == 0:
            try:
                self.db_logger.prune_old_telemetry(
                    current_generation=generation, keep_generations=20
                )
            except Exception:
                pass

        avg_depth = float(batch_eval.get("avg_depth", 1.0))
        max_d = int(batch_eval.get("max_depth", 1))
        avg_score = float(batch_eval.get("avg_score", 0.0))
        elapsed = time.time() - t0
        ui.info(
            f"Gen {generation} Complete in {elapsed:.1f}s ({'PROMOTED' if verdict.promoted else 'KEPT INCUMBENT'}): "
            f"Winner={verdict.winner_id} | Avg Depth={avg_depth:.2f} (Max={max_d}) | "
            f"Avg Score={avg_score:.1f} | New Frontier CPs={new_cps} | FunctionStats={len(log_res)}"
        )
        return {
            "generation": generation,
            "promoted": verdict.promoted,
            "winner_id": verdict.winner_id,
            "avg_depth": avg_depth,
            "max_depth": max_d,
            "avg_score": avg_score,
            "active_max_steps": active_max_steps,
            "new_frontier_checkpoints": new_cps,
            "elapsed_sec": elapsed,
            "peers_run": len(peer_candidates),
            "peers_incubated": sum(1 for c in peer_candidates if c.sandbox_passed),
            "scoreboard": verdict.scoreboard,
        }

    def _compute_depth_plateau_streak(self) -> int:
        """
        Counts consecutive recent SWARM generations without a promoted breakthrough or
        paired progression gain (immune to across-generation random seed difficulty variance).
        """
        try:
            con = connect_db(self.db_path, read_only=True)
            try:
                rows = con.execute(
                    """
                    SELECT generation, avg_depth, max_depth, hypothesis_passed
                    FROM batches
                    WHERE UPPER(mode) = 'SWARM'
                    ORDER BY generation DESC, created_at DESC
                    LIMIT 12
                    """
                ).fetchall()
            finally:
                con.close()
            if len(rows) < 2:
                return 0
            streak = 0
            for gen_i, _ad, _md, passed in rows:
                if bool(passed):
                    break
                streak += 1
            return streak
        except Exception:
            return 0

    def _compute_active_max_steps(self) -> int:
        """Uses `compute_adaptive_max_steps` from recent promoted batches to lift the 5k horizon on deep runs."""
        try:
            con = connect_db(self.db_path, read_only=True)
            try:
                row = con.execute(
                    """
                    SELECT p90_depth, max_depth
                    FROM batches
                    WHERE UPPER(mode) = 'SWARM' AND hypothesis_passed = TRUE
                    ORDER BY generation DESC, created_at DESC
                    LIMIT 1
                    """
                ).fetchone()
                if row:
                    p90_d = float(row[0] or row[1] or 1.0)
                    return compute_adaptive_max_steps(
                        p90_d, base_max_steps=self.max_steps
                    )
            finally:
                con.close()
        except Exception:
            pass
        return self.max_steps

    def _ingest_frontier_and_map_elites(
        self,
        generation: int,
        runs: list[Any],
        batch_eval: BatchEvaluation,
        policy_code: str,
        batch_id: str,
        runner_ups: list[dict[str, Any]] | None = None,
        skills: dict[str, str] | None = None,
        allow_frontier: bool = True,
    ) -> int:
        new_cps = 0
        if allow_frontier:
            depth_tier_counts: dict[int, int] = {}
            for cp in self.frontier.checkpoints.values():
                depth_tier_counts[int(cp.depth)] = (
                    depth_tier_counts.get(int(cp.depth), 0) + 1
                )
            for ep in runs:
                ep_dict = dict(ep) if isinstance(ep, dict) else ep.__dict__
                depth = max(
                    int(ep_dict.get("depth", 1)), int(ep_dict.get("max_depth", 1))
                )
                act_bytes = bytes(ep_dict.get("action_bytes") or b"")
                # Cap checkpoints per depth tier so deep mid-game states aren't diluted by Dlvl 2-3 noise
                if (
                    depth >= self.frontier_min_depth
                    and len(act_bytes) > 0
                    and depth_tier_counts.get(depth, 0) < 12
                ):
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
                        depth_tier_counts[depth] = depth_tier_counts.get(depth, 0) + 1
                        new_cps += 1

        metrics = {
            "avg_depth": float(batch_eval.get("avg_depth", 1.0)),
            "max_depth": int(batch_eval.get("max_depth", 1)),
            "avg_score": float(batch_eval.get("avg_score", 0.0)),
            "avg_ac": float(
                np.mean(
                    [
                        r.get("final_ac", 10) if r.get("final_ac") is not None else 10
                        for r in runs
                    ]
                )
            )
            if runs
            else 10.0,
            "avg_xp_level": float(np.mean([r.get("xp_level", 1) or 1 for r in runs]))
            if runs
            else 1.0,
            "avg_unique_tiles": float(
                np.mean([r.get("unique_tiles", 0) or 0 for r in runs])
            )
            if runs
            else 0.0,
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
        con = None
        try:
            con = connect_db(self.db_path, read_only=False)
            self.map_elites.add_candidate(
                policy_code=policy_code,
                metrics=metrics,
                batch_id=batch_id,
                generation=generation,
                con=con,
                skills=skills,
            )
            for ru in runner_ups or []:
                ru_code = str(ru.get("policy_code", "") or "")
                ru_metrics = ru.get("metrics") or {}
                if ru_code.strip() and ru_metrics:
                    self.map_elites.add_candidate(
                        policy_code=ru_code,
                        metrics=ru_metrics,
                        batch_id=str(ru.get("batch_id", batch_id)),
                        generation=generation,
                        con=con,
                        skills=ru.get("skills"),
                    )
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass
        return new_cps

    def _run_generation_legacy(self, generation: int) -> dict[str, Any]:
        """
        Executes a single generation of the Legacy Hierarchical Swarm loop
        (`SwarmCriticAgent` + `SwarmMetaAgent` + thematic `SwarmWorker`s).
        """
        t0 = time.time()
        ui.header(f"LOX Swarm Campaign - Generation {generation}")

        policy_path = self.workspace_dir / "policy.py"
        skills_dir = self.workspace_dir / "skills"
        current_policy = (
            policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
        )
        initial_skills = (
            {
                p.stem: p.read_text(encoding="utf-8")
                for p in sorted(skills_dir.glob("*.py"))
                if p.stem != "__init__"
            }
            if skills_dir.exists()
            else {}
        )

        # 1. Critic Pass 1: Pre-Generation High-Level Guide (very first agent called)
        pre_gen_guide = self.critic.guide_before_generation(
            generation=generation,
            current_policy=current_policy,
            skills=initial_skills,
        )

        # 2. Dynamic Macro-Category Discovery (informed by Critic Pass 1)
        categories = self.meta_agent.discover_macro_categories(
            generation=generation,
            n_categories=max(1, int(self.num_workers)),
            current_policy=current_policy,
            skills=initial_skills,
            pre_gen_guide=pre_gen_guide,
        )
        cat_names = [c["category_id"] for c in categories]
        ui.info(f"Gen {generation}: Active Categories: {cat_names}")

        # 3. Concurrent Worker Exploration
        worker_reports: dict[str, dict[str, Any]] = {}
        theories_path = self.db_path

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
            target_depth = cat.get("target_depth_tier", (1, 2))[0]
            cp = self.frontier.sample_checkpoint(min_depth=target_depth)
            return worker.run_cycle(
                generation=generation,
                checkpoint=cp,
                scaffold_code=current_policy,
                skills=initial_skills,
            )

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

        # 4. Meta-Agent Initial Policy Assembly
        master_policy_code = self.meta_agent.assemble_policy(
            generation=generation,
            worker_reports=worker_reports,
            current_policy=current_policy,
            skills_dir=skills_dir,
            pre_gen_guide=pre_gen_guide,
        )
        policy_path.write_text(master_policy_code, encoding="utf-8")
        skills_dict = (
            {
                p.stem: p.read_text(encoding="utf-8")
                for p in sorted(skills_dir.glob("*.py"))
                if p.stem != "__init__"
            }
            if skills_dir.exists()
            else {}
        )
        skills_dict = self.meta_agent.prune_unused_skills(
            master_policy_code, skills_dict, skills_dir=skills_dir
        )
        ui.info(
            f"Gen {generation}: Draft master policy saved to `{policy_path}` ({len(skills_dict)} active skill module(s) in `{skills_dir}`)."
        )

        # 5. Evaluate Draft Policy on Paired Seeds & Persist to DuckDB
        ui.info(
            f"Gen {generation}: Evaluating master policy on {self.seeds_per_batch} paired seeds..."
        )
        seeds = [1000 + i for i in range(self.seeds_per_batch)]
        batch_id = f"batch_swarm_g{generation}_{int(time.time())}"
        batch_eval: BatchEvaluation = self.eval_pool.evaluate_batch(
            batch_id=batch_id,
            policy_code=master_policy_code,
            macros_code="",
            skills=skills_dict,
            target_episodes=self.seeds_per_batch,
            priority_seeds=seeds,
            record_ticks=True,
        )

        runs = batch_eval.get("runs", [])
        batch_id = str(batch_eval.get("batch_id", batch_id))
        batch_metrics = self._extract_batch_metrics(batch_eval, runs)
        policy_id = f"policy_swarm_g{generation}"
        parent_id = f"policy_swarm_g{max(0, generation - 1)}"

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
            skills=skills_dict,
        )
        log_res = self.db_logger.save_batch(
            batch_id=batch_id,
            generation=generation,
            mode="SWARM",
            parent_policy_id=parent_id,
            candidate_policy_id=policy_id,
            target_functions=[],
            runs=runs,
            episodes_discarded=int(batch_eval.get("episodes_discarded", 0)),
            batch_metrics=batch_metrics,
            verdict={"passed": True, "reason": "Swarm Generation Evaluation"},
        )

        # 6. Critic Pass 2: Post-Assembly Comparative Review
        post_critique = self.critic.review_after_assembly(
            generation=generation,
            batch_id=batch_id,
            candidate_policy_id=policy_id,
            current_policy=master_policy_code,
            skills=skills_dict,
        )

        # 7. Meta-Agent Post-Critic Revision (Meta-Agent decides; zero hardcoded rollbacks)
        revised_code, revised_skills, was_revised = (
            self.meta_agent.revise_policy_with_critic(
                generation=generation,
                draft_policy=master_policy_code,
                skills_dir=skills_dir,
                skills=skills_dict,
                post_critique=post_critique,
                pre_gen_guide=pre_gen_guide,
            )
        )
        if was_revised:
            master_policy_code = revised_code
            skills_dict = self.meta_agent.prune_unused_skills(
                master_policy_code, revised_skills, skills_dir=skills_dir
            )
            policy_path.write_text(master_policy_code, encoding="utf-8")
            ui.info(
                f"Gen {generation}: Re-evaluating Post-Critic revised policy on {self.seeds_per_batch} paired seeds..."
            )
            batch_eval = self.eval_pool.evaluate_batch(
                batch_id=batch_id,
                policy_code=master_policy_code,
                macros_code="",
                skills=skills_dict,
                target_episodes=self.seeds_per_batch,
                priority_seeds=seeds,
                record_ticks=True,
            )
            runs = batch_eval.get("runs", [])
            batch_metrics = self._extract_batch_metrics(batch_eval, runs)
            self._delete_batch_artifacts(batch_id)
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
                skills=skills_dict,
            )
            log_res = self.db_logger.save_batch(
                batch_id=batch_id,
                generation=generation,
                mode="SWARM",
                parent_policy_id=parent_id,
                candidate_policy_id=policy_id,
                target_functions=[],
                runs=runs,
                episodes_discarded=int(batch_eval.get("episodes_discarded", 0)),
                batch_metrics=batch_metrics,
                verdict={
                    "passed": True,
                    "reason": "Swarm Post-Critic Revised Evaluation",
                },
            )

        # 8. Extract New Frontier Checkpoints from High-Performing Runs
        new_cps = 0
        for ep in runs:
            ep_dict = dict(ep) if isinstance(ep, dict) else ep.__dict__
            depth = max(int(ep_dict.get("depth", 1)), int(ep_dict.get("max_depth", 1)))
            act_bytes = bytes(ep_dict.get("action_bytes") or b"")
            if depth >= self.frontier_min_depth and len(act_bytes) > 0:
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

        # 9. Ingest into MAP-Elites Diversity Archive
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
        self._sync_batch_cost(batch_id, generation)

        if generation % 5 == 0:
            try:
                self.db_logger.prune_old_telemetry(
                    current_generation=generation, keep_generations=20
                )
            except Exception:
                pass

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
            "revised_after_critic": was_revised,
        }

    def run(self, generations: int = 5) -> list[dict[str, Any]]:
        """Runs the multi-generation swarm campaign, resuming from the latest DB generation if present."""
        start_gen = self._get_start_generation()
        end_gen = start_gen + max(1, int(generations)) - 1
        ui.info(
            f"Starting LOX Swarm Campaign: generations {start_gen}..{end_gen} ({generations} total), "
            f"{self.num_workers} workers, provider={self.provider}, model={self.model}"
        )
        history: list[dict[str, Any]] = []
        err_log_path = self.workspace_dir / "campaign_errors.log"
        try:
            for g in range(start_gen, end_gen + 1):
                try:
                    res = self.run_generation(generation=g)
                    history.append(res)
                except (KeyboardInterrupt, PermissionError):
                    raise
                except Exception as exc:
                    tb = traceback.format_exc()
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ui.warn(
                        f"Generation {g} encountered an error ({type(exc).__name__}: {exc}); logged to `{err_log_path}` and continuing..."
                    )
                    try:
                        with err_log_path.open("a", encoding="utf-8") as ef:
                            ef.write(f"\n[{ts}] Generation {g} Error:\n{tb}\n")
                    except Exception:
                        pass
        finally:
            self.close()

        ui.header("LOX Swarm Campaign Completed")
        return history
