"""
LOX Democratized Peer Swarm Campaign Orchestrator ("Hard Code Nothing").
Coordinates the Layout A (`2R + 2P`) Option-A Peer Swarm:
1. Pre-generation Side Cohesion Refactor when any module >= 500 LOC.
2. Deterministic `DeficitAnalyzer` + organic scenario mining (`D_fail` + `D_succ`).
3. 1-shot `SwarmDirectivePlanner` partitioning orthogonal scopes across parallel peers.
4. Concurrent `SwarmPeerAgent` execution (`reactive_1`, `reactive_2`, `proactive_1`, `proactive_2`).
5. Split-Sample CRN Tournament (`SwarmReferee`) + O'Brien-Fleming / Triggered Stage-3b Confirmation.
6. Lifecycle persistence (`stepping_stones`, `proactive_branches`, `FrontierArchive`, `MAPElitesArchive`).
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

from lox.author.directives import SwarmDirectivePlanner, prune_unused_skills
from lox.author.swarm_deficits import DeficitAnalyzer
from lox.author.swarm_peer import PeerCandidate, SwarmPeerAgent
from lox.core.contracts import BatchEvaluation
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import DEFAULT_DB_PATH, connect_db, init_db
from lox.db.shards import mine_organic_scenarios
from lox.db.theories import TheoryDocumentStore
from lox.engine.frontier import FrontierArchive
from lox.engine.lifecycle import (
    build_layout_a_assignments,
    update_near_misses,
    update_proactive_branch_lifecycle,
    update_stepping_stone_lifecycle,
)
from lox.engine.map_elites import MAPElitesArchive
from lox.engine.referee_gates import (
    _assemble_batch_eval,
    _package_fingerprint,
    opt_float,
    opt_int,
)
from lox.engine.reporting import compute_adaptive_max_steps, write_status_report
from lox.engine.swarm_referee import SwarmReferee
from lox.lang.bundle import load_skills_dir, write_skills_dir
from lox.lang.modularize import run_side_cohesion_refactor
from lox.lang.starter import build_clean_option_a_starter
from lox.runner.pool import PersistentEvalPool
from lox.ui.console import ui

DEFAULT_WORKSPACE_DIR = "data/workspace"
DEFAULT_FRONTIER_DIR = "data/frontier"
DEFAULT_EXPERT_MODEL = "thinkingmachines/inkling-2-small:floor"


class SwarmCampaignOrchestrator:
    """End-to-End Orchestrator for the LOX Layout A Peer Swarm."""

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
        role: str = "val-hum-law-fem",
        max_steps: int = 5000,
        frontier_min_depth: int = 2,
        eval_workers: int | None = None,
        reasoning_effort: str = "medium",
        routing: str = "nitro",
        expert_model: str | None = None,
        race_seeds: int | None = None,
        scenario_gate: str = "off",
    ) -> None:
        self.db_path = db_path
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.frontier_dir = frontier_dir
        self.provider = provider
        self.model = model
        self.reasoning_effort = str(reasoning_effort or "medium")
        self.routing = str(routing or "nitro")
        if expert_model and str(expert_model).strip():
            self.expert_model = str(expert_model).strip()
        elif str(provider or "").strip().lower() == "openrouter":
            self.expert_model = DEFAULT_EXPERT_MODEL
        else:
            self.expert_model = str(model)
        self.race_seeds = max(4, int(race_seeds)) if race_seeds is not None else None
        self.scenario_gate = (
            str(scenario_gate or "off").strip().lower()
            if str(scenario_gate or "off").strip().lower()
            in ("off", "advisory", "strict")
            else "off"
        )
        self.num_workers = num_workers
        self.turns_per_worker = turns_per_worker
        self.seeds_per_batch = seeds_per_batch
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

        # Initialize DuckDB tables and close the init handle cleanly (Fix L-3)
        _init_con = init_db(self.db_path)
        try:
            _init_con.close()
        except Exception as exc:
            ui.warn(f"Initial DB close warning: {exc}")
        self.db_logger = BatchDatabaseLogger(db_path=self.db_path)

        # Initialize Unified DuckDB Theory Store & Frontier Archive
        self.theories = TheoryDocumentStore(db_path=self.db_path)
        self.frontier = FrontierArchive(frontier_dir=self.frontier_dir)

        # Initialize MAP-Elites Diversity Archive
        self.map_elites = MAPElitesArchive()
        try:
            _con_init = connect_db(self.db_path, read_only=True)
            try:
                self.map_elites.load_from_db(_con_init)
            finally:
                _con_init.close()
        except Exception as exc:
            ui.warn(f"Initial MAP-Elites load warning: {exc}")

        # Initialize Deficit Analyzer, Directive Planner, & Deterministic Referee
        self.deficit_analyzer = DeficitAnalyzer(db_path=self.db_path)
        self.referee = SwarmReferee(max_tournament_peers=max(4, int(self.num_workers)))
        self.directive_planner = SwarmDirectivePlanner(
            provider=self.provider,
            model=self.model,
            db_path=self.db_path,
            theories_store=self.theories,
            reasoning_effort=self.reasoning_effort,
            routing=self.routing,
        )
        self.campaign_start_time = time.time()
        self._plateau_streak: int = self._compute_depth_plateau_streak()
        self._attempted_ast_modular_fps: set[str] = set()
        self._near_misses: list[dict[str, Any]] = []

        # Initialize Multi-Core Persistent Eval Pool
        self.eval_pool = PersistentEvalPool(
            workers=max(1, self.eval_workers),
            role=self.role,
            max_steps=self.max_steps,
        )

    def close(self) -> None:
        """Shuts down worker pool cleanly."""
        try:
            self.eval_pool.close()
        except Exception as exc:
            ui.warn(f"Eval pool close warning: {exc}")

    def _sync_batch_cost(self, batch_id: str, generation: int) -> None:
        """Updates `batches.total_cost_usd` and `total_llm_tokens` from `synthesis_logs`."""
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
        except Exception as exc:
            ui.warn(f"Gen {generation}: batch cost sync warning ({exc})")

    def _get_start_generation(self) -> int:
        """Returns `MAX(generation) + 1` from DuckDB `batches` so restarted runs resume cleanly."""
        try:
            con = connect_db(self.db_path, read_only=True)
            try:
                row = con.execute(
                    "SELECT COALESCE(MAX(generation), 0) FROM batches"
                ).fetchone()
                return (int(row[0] or 0) + 1) if row else 1
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

    def _load_workspace_skills(self, skills_dir: Path) -> dict[str, str]:
        return load_skills_dir(skills_dir)

    def _write_workspace_package(
        self,
        policy_path: Path,
        skills_dir: Path,
        policy_code: str,
        skills: dict[str, str],
    ) -> None:
        policy_path.write_text(policy_code, encoding="utf-8")
        write_skills_dir(
            skills_dir, skills or {}, sync=False, ensure_trailing_newline=False
        )
        prune_unused_skills(policy_code, skills, skills_dir=skills_dir)

    def run_generation(self, generation: int) -> dict[str, Any]:
        """Runs a single generation of the Layout A (`2R+2P`) Democratized Peer Swarm."""
        t0 = time.time()
        ui.header(f"LOX Democratized Peer Swarm - Generation {generation}")

        policy_path = self.workspace_dir / "policy.py"
        skills_dir = self.workspace_dir / "skills"
        current_policy = (
            policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
        )
        initial_skills = self._load_workspace_skills(skills_dir)

        if not current_policy.strip():
            current_policy, initial_skills = build_clean_option_a_starter()
            self._write_workspace_package(
                policy_path, skills_dir, current_policy, initial_skills
            )

        # 0. Mandatory Pre-Gen Side Cohesion Refactor (outside promotion gating)
        cur_fp = _package_fingerprint(current_policy, initial_skills)
        pol_loc_now = len(current_policy.splitlines())
        needs_side_refactor = (
            (not initial_skills and pol_loc_now >= 200)
            or pol_loc_now >= 500
            or any(len(v.splitlines()) >= 500 for v in initial_skills.values())
        )
        if cur_fp not in self._attempted_ast_modular_fps and needs_side_refactor:
            self._attempted_ast_modular_fps.add(cur_fp)
            ref_changed, ref_pol, ref_skills, ref_msg = run_side_cohesion_refactor(
                current_policy,
                initial_skills,
                max_loc=500,
                verify_seeds=[10_000 + int(generation) * 100 + i for i in range(6)],
                verify_steps=250,
            )
            if ref_changed:
                old_fp = cur_fp
                ui.info(f"Gen {generation} Pre-Gen Side Refactor: {ref_msg}")
                current_policy = ref_pol
                initial_skills = ref_skills
                self._write_workspace_package(
                    policy_path, skills_dir, current_policy, initial_skills
                )
                # Fix H-12: Recompute `cur_fp` immediately after pre-gen side cohesion refactor!
                cur_fp = _package_fingerprint(current_policy, initial_skills)
                self._attempted_ast_modular_fps.add(cur_fp)
                if cur_fp != old_fp:
                    try:
                        self.db_logger.retire_stepping_stones_for_parent(
                            old_fp, new_status="superseded"
                        )
                        self.db_logger.retire_proactive_branches_for_champion(
                            old_fp, new_status="superseded"
                        )
                    except Exception as ret_exc:
                        ui.warn(
                            f"Pre-gen refactor lifecycle retirement warning: {ret_exc}"
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

        deficit_report.plateau_streak = int(self._plateau_streak)
        if self._plateau_streak >= 2 and deficit_report.regime != "BOOTSTRAP":
            deficit_report.regime = "PLATEAU"
        if self._plateau_streak >= 2 and self._near_misses:
            deficit_report.plateau_hints = [
                f"{nm['peer_id']} (gen={nm['generation']}, D={nm['avg_depth']:.2f}/maxD={nm['max_depth']}/S={nm['avg_score']:.0f}, "
                f"gained={len(nm['seeds_gained'])}, lost={len(nm['seeds_lost'])}, touched={sorted(nm['touched_skills'])[:3]})"
                for nm in self._near_misses[:3]
            ]

        clusters = deficit_report.fail_clusters
        orthogonal_clusters: list[Any] = []
        seen_cids: set[str] = set()
        for cl in clusters:
            cid = str(getattr(cl, "cluster_id", "") or "").strip()
            if not cid or cid not in seen_cids:
                orthogonal_clusters.append(cl)
                if cid:
                    seen_cids.add(cid)
        if not orthogonal_clusters:
            orthogonal_clusters = list(clusters)

        target_peers = max(min(4, int(self.num_workers)), len(orthogonal_clusters))
        active_peers = max(1, min(int(self.num_workers), target_peers))
        self.eval_pool.max_steps = int(active_max_steps)

        ui.info(
            f"Gen {generation}: Regime={deficit_report.regime} "
            f"(Δ={deficit_report.rolling_depth_delta:+.2f}, streak={self._plateau_streak}) | "
            f"Clusters={len(clusters)} (ortho={len(orthogonal_clusters)}) | Peers={active_peers} | MaxSteps={active_max_steps}"
        )

        # 2. Strictly Disjoint Paired CRN Seed Partitions (stride 10,000 -> zero cross-gen collision)
        seeds = [
            10_000_000 + int(generation) * 10_000 + i
            for i in range(self.seeds_per_batch)
        ]
        if self.race_seeds is not None and int(self.seeds_per_batch) >= 8:
            n_select = max(4, min(int(self.race_seeds), int(self.seeds_per_batch)))
        elif self.race_seeds is None and int(self.seeds_per_batch) >= 16:
            n_select = max(4, min(32, int(self.seeds_per_batch) // 2))
        else:
            n_select = 0
        select_seeds: list[int] | None = (
            [100_000 + int(generation) * 10_000 + i for i in range(n_select)]
            if n_select >= 4
            else None
        )
        used_eval_seeds = set(seeds) | set(select_seeds or [])
        probe_seeds: list[int] = []
        if int(self.seeds_per_batch) >= 32:
            target_probes = 12
        elif int(self.seeds_per_batch) >= 12:
            target_probes = 6
        elif int(self.seeds_per_batch) >= 6:
            target_probes = 2
        else:
            target_probes = 0
        if target_probes > 0:
            p_cand = 5_000_000 + int(generation) * 10_000
            while len(probe_seeds) < target_probes:
                if p_cand not in used_eval_seeds:
                    probe_seeds.append(p_cand)
                    used_eval_seeds.add(p_cand)
                p_cand += 1

        raw_shard_seeds = list(
            dict.fromkeys(
                deficit_report.d_succ_seeds[:4]
                + [sd for cl in clusters for sd in cl.sample_seeds[:4]]
            )
        ) or [1, 2]
        max_shard = min(32, max(2, int(self.seeds_per_batch)))
        hist_cap = min(20, max(2, max_shard - len(probe_seeds)))
        shard_seeds = list(dict.fromkeys(raw_shard_seeds[:hist_cap] + probe_seeds))[
            :max_shard
        ]
        incumbent_shard_runs: dict[int, dict[str, Any]] = {}
        if current_policy.strip() and shard_seeds:
            inc_shard_bid = f"incubate_inc_g{generation}_{int(time.time())}"
            inc_shard = self.eval_pool.evaluate_batch(
                batch_id=inc_shard_bid,
                policy_code=current_policy,
                macros_code="",
                skills=initial_skills,
                target_episodes=len(shard_seeds),
                priority_seeds=shard_seeds,
                max_steps=active_max_steps,
                record_ticks=True,
            )
            inc_shard_list = [dict(r) for r in (inc_shard.get("runs", []) or [])]
            for r in inc_shard_list:
                incumbent_shard_runs[int(r.get("seed", 0))] = r
            if inc_shard_list:
                try:
                    self.db_logger.save_runs_and_ticks(
                        batch_id=inc_shard_bid,
                        generation=generation,
                        run_kind="incubate_baseline",
                        peer_id="incumbent",
                        runs=inc_shard_list,
                        run_id_prefix=f"g{generation}_inc_base",
                    )
                except Exception as exc:
                    ui.warn(
                        f"Gen {generation}: incubate_baseline persistence failed ({exc})"
                    )

        # 3. Layout A — Fixed 4-Peer Balanced Team (`2 Reactive + 2 Proactive`)
        active_stone: dict[str, Any] | None = None
        active_branch: dict[str, Any] | None = None
        if current_policy.strip():
            active_stone = self.db_logger.load_active_stepping_stone(
                parent_policy_id=cur_fp, max_revisions=2
            )
            active_branch = self.db_logger.load_active_proactive_branch(
                base_champion_fp=cur_fp
            )

        if active_peers >= 4:
            roster_specs: list[tuple[str, str]] = [
                ("reactive_1", "revision" if active_stone else "reactive"),
                ("reactive_2", "reactive"),
                ("proactive_1", "proactive_capability"),
                ("proactive_2", "proactive_branch"),
            ]
            for extra_i in range(4, active_peers):
                roster_specs.append((f"reactive_{extra_i - 1}", "reactive"))
        elif active_peers == 3:
            roster_specs = [
                ("reactive_1", "revision" if active_stone else "reactive"),
                ("reactive_2", "reactive"),
                ("proactive_1", "proactive_capability"),
            ]
        elif active_peers == 2:
            roster_specs = [
                ("reactive_1", "revision" if active_stone else "reactive"),
                ("proactive_1", "proactive_capability"),
            ]
        else:
            roster_specs = [
                ("reactive_1", "revision" if active_stone else "reactive"),
            ]

        roster_peer_ids = [pid for pid, _ in roster_specs]
        roster_roles = {pid: rtrack for pid, rtrack in roster_specs}

        arch_summary = ""
        try:
            from lox.author.tools import AuthorToolRegistry

            _tmp_tools = AuthorToolRegistry(db_path=self.db_path)
            _tmp_tools.current_policy_code = current_policy
            _tmp_tools.skills = dict(initial_skills)
            arch_summary, _ = _tmp_tools.architecture_map()
        except Exception:
            arch_summary = ""

        planned_directives: dict[str, dict[str, Any]] = {}
        try:
            dir_list = self.directive_planner.plan_peer_directives(
                generation=generation,
                fresh_peer_ids=roster_peer_ids,
                deficit_report=deficit_report,
                architecture_map=arch_summary,
                active_stepping_stone=active_stone,
                peer_roles=roster_roles,
                active_proactive_branch=active_branch,
            )
            planned_directives = {
                str(d["peer_id"]): d
                for d in dir_list
                if isinstance(d, dict) and d.get("peer_id")
            }
            if planned_directives and self.provider != "mock":
                dir_summary = " | ".join(
                    f"{pid}->{d.get('focus_subsystem', d.get('target_cluster_id', '?'))}"
                    for pid, d in planned_directives.items()
                )
                ui.info(f"Gen {generation} Meta-Directives (--model): {dir_summary}")
        except Exception as exc:
            ui.warn(f"Gen {generation}: Directive planner skipped ({exc})")

        peer_assignments = build_layout_a_assignments(
            generation=generation,
            roster_specs=roster_specs,
            clusters=clusters,
            orthogonal_clusters=orthogonal_clusters,
            deficit_report=deficit_report,
            current_policy=current_policy,
            initial_skills=initial_skills,
            probe_seeds=probe_seeds,
            planned_directives=planned_directives,
            active_stone=active_stone,
            active_branch=active_branch,
            plateau_streak=self._plateau_streak,
            near_misses=self._near_misses,
        )

        peer_tool_turns = max(22, int(self.turns_per_worker) * 6)
        peer_candidates: list[PeerCandidate] = []
        use_expert_model_for_p2 = bool(
            self.expert_model and self.expert_model != self.model
        )
        if use_expert_model_for_p2 and self.provider != "mock":
            ui.info(
                f"Gen {generation}: Routing sheltered branch `proactive_2` to `--expert-model` `{self.expert_model}`."
            )

        def _run_peer(
            peer_id: str,
            cluster: Any,
            p_policy: str,
            p_skills: dict[str, str],
            p_directive: dict[str, Any] | None,
            p_stone: dict[str, Any] | None,
            p_track: str,
            p_branch: dict[str, Any] | None,
        ) -> PeerCandidate:
            peer_model = (
                self.expert_model
                if (use_expert_model_for_p2 and peer_id == "proactive_2")
                else self.model
            )
            peer = SwarmPeerAgent(
                peer_id=peer_id,
                provider=self.provider,
                model=peer_model,
                db_path=self.db_path,
                theories_store=self.theories,
                max_tool_turns=peer_tool_turns,
                reasoning_effort=self.reasoning_effort,
                routing=self.routing,
                scenario_gate=self.scenario_gate,
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
                probe_seeds=probe_seeds,
                directive=p_directive,
                stepping_stone=p_stone,
                incumbent_policy=current_policy,
                incumbent_skills=initial_skills,
                track=p_track,
                proactive_branch=p_branch,
            )

        peer_errors: list[tuple[str, BaseException]] = []
        err_log_path = self.workspace_dir / "campaign_errors.log"
        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=len(peer_assignments)
        )
        try:
            fut_map = {
                executor.submit(
                    _run_peer, pid, cl, sp, ss, pdir, pstone, ptrk, pbr
                ): pid
                for pid, cl, sp, ss, pdir, pstone, ptrk, pbr in peer_assignments
            }
            for fut in concurrent.futures.as_completed(fut_map):
                pid = fut_map[fut]
                try:
                    peer_candidates.append(fut.result())
                except PermissionError as exc:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    try:
                        with err_log_path.open("a", encoding="utf-8") as ef:
                            ef.write(
                                f"\n[{ts}] Generation {generation} Peer [{pid}] Fatal PermissionError: {exc}\n"
                            )
                    except Exception:
                        pass
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as exc:
                    msg_l = str(exc).lower()
                    if "402" in msg_l or "insufficient credits" in msg_l:
                        executor.shutdown(wait=False, cancel_futures=True)
                        raise PermissionError(str(exc)) from exc
                    peer_errors.append((pid, exc))
                    ui.warn(f"Peer [{pid}] failed: {type(exc).__name__}: {exc}")
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    tb_txt = "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    )
                    try:
                        with err_log_path.open("a", encoding="utf-8") as ef:
                            ef.write(
                                f"\n[{ts}] Generation {generation} Peer [{pid}] Error:\n{tb_txt}\n"
                            )
                    except Exception:
                        pass
        finally:
            executor.shutdown(wait=False)

        if not peer_candidates and peer_errors:
            first_pid, first_exc = peer_errors[0]
            raise RuntimeError(
                f"Gen {generation}: All {len(peer_errors)} peers failed "
                f"({first_pid}: {type(first_exc).__name__}: {first_exc}); "
                "aborting generation to avoid zombie incumbent rollouts."
            )

        # 4. Strict Split-Sample Paired CRN Tournament
        verdict = self.referee.evaluate_tournament(
            generation=generation,
            candidates=peer_candidates,
            base_policy=current_policy,
            base_skills=initial_skills,
            eval_pool=self.eval_pool,
            seeds=seeds,
            select_seeds=select_seeds,
            max_steps=active_max_steps,
            race_seeds=self.race_seeds,
            scenario_gate=self.scenario_gate,
        )

        # 4b. Post-Promotion Side Cohesion Refactor (Fix M-13: filter OBF runs to `set(seeds)` before caching!)
        win_pol_loc = len(str(verdict.winner_policy_code or "").splitlines())
        win_skills_dict = dict(verdict.winner_skills or {})
        needs_post_refactor = (
            (not win_skills_dict and win_pol_loc >= 200)
            or win_pol_loc >= 500
            or any(len(v.splitlines()) >= 500 for v in win_skills_dict.values())
        )
        if verdict.promoted and needs_post_refactor:
            ref_changed, ref_pol, ref_skills, ref_msg = run_side_cohesion_refactor(
                verdict.winner_policy_code,
                win_skills_dict,
                max_loc=500,
                verify_seeds=seeds[:6],
                verify_steps=250,
            )
            if ref_changed:
                ui.info(f"Gen {generation} Post-Promotion Side Refactor: {ref_msg}")
                verdict.winner_policy_code = ref_pol
                verdict.winner_skills = ref_skills
                verdict.reason = f"{verdict.reason} | {ref_msg}"
                if verdict.winner_batch_eval is not None:
                    seed_set = {int(s) for s in seeds}
                    all_w_runs = list(verdict.winner_batch_eval.get("runs", []) or [])
                    filtered_runs = [
                        r for r in all_w_runs if int(r.get("seed", -1)) in seed_set
                    ]
                    cache_be = (
                        _assemble_batch_eval(
                            str(verdict.winner_batch_eval.get("batch_id", "")),
                            filtered_runs,
                        )
                        if len(filtered_runs) == len(seeds)
                        else verdict.winner_batch_eval
                    )
                    if len(filtered_runs) == len(seeds) or len(all_w_runs) == len(
                        seeds
                    ):
                        self.referee.cache_incumbent(
                            eval_pool=self.eval_pool,
                            policy_code=ref_pol,
                            skills=ref_skills,
                            seeds=seeds,
                            max_steps=active_max_steps,
                            batch_eval=cache_be,
                        )

        # Update Near-Miss Crossover buffer (Fix M-9: cleared immediately on promotion)
        update_near_misses(
            self._near_misses,
            generation=generation,
            verdict=verdict,
            peer_candidates=peer_candidates,
            current_policy=current_policy,
            initial_skills=initial_skills,
        )

        # Sync workspace only if promoted
        if verdict.promoted:
            self._write_workspace_package(
                policy_path,
                skills_dir,
                verdict.winner_policy_code,
                dict(verdict.winner_skills or {}),
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
        bp_stats = dict(getattr(verdict, "best_pair", {}) or {})
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
                "wilcoxon_p": opt_float(bp_stats, "wilcoxon_p", 1.0),
                "ci_lower": opt_float(bp_stats, "ci_lower", 0.0),
                "ci_upper": opt_float(bp_stats, "ci_upper", 0.0),
            },
        )

        try:
            self.db_logger.save_tournament_attempts(verdict.attempts_summary)
        except Exception as exc:
            ui.warn(f"Gen {generation}: save_tournament_attempts failed ({exc})")

        try:
            for sel_pid, sel_be in (verdict.select_batches or {}).items():
                sel_runs = list((sel_be or {}).get("runs", []) or [])
                if not sel_runs:
                    continue
                is_inc_sel = sel_pid == "inc"
                self.db_logger.save_runs_and_ticks(
                    batch_id=str(
                        sel_be.get(
                            "batch_id",
                            f"sel_{sel_pid}_g{generation}_{int(time.time() * 1000)}",
                        )
                    ),
                    generation=generation,
                    run_kind="select_incumbent" if is_inc_sel else "select",
                    peer_id="incumbent" if is_inc_sel else str(sel_pid),
                    runs=sel_runs,
                    run_id_prefix=f"g{generation}_{'inc' if is_inc_sel else sel_pid}_sel",
                )
            if verdict.confirm_challenger_eval is not None:
                c_pid, c_be = verdict.confirm_challenger_eval
                c_runs = list((c_be or {}).get("runs", []) or [])
                if c_runs:
                    self.db_logger.save_runs_and_ticks(
                        batch_id=str(
                            c_be.get(
                                "batch_id",
                                f"conf_{c_pid}_g{generation}_{int(time.time() * 1000)}",
                            )
                        ),
                        generation=generation,
                        run_kind="confirm_challenger",
                        peer_id=str(c_pid),
                        runs=c_runs,
                        run_id_prefix=f"g{generation}_{c_pid}_conf",
                    )
        except Exception as exc:
            ui.warn(f"Gen {generation}: select/confirm tick persistence failed ({exc})")

        # Update `stepping_stones` & `proactive_branches` lifecycles (Fix M-8)
        try:
            update_stepping_stone_lifecycle(
                db_logger=self.db_logger,
                generation=generation,
                cur_fp=cur_fp,
                active_stone=active_stone,
                verdict=verdict,
                max_revisions=2,
            )
        except Exception as exc:
            ui.warn(f"Gen {generation}: stepping_stone lifecycle update failed ({exc})")

        try:
            update_proactive_branch_lifecycle(
                db_logger=self.db_logger,
                generation=generation,
                cur_fp=cur_fp,
                active_branch=active_branch,
                verdict=verdict,
                peer_candidates=peer_candidates,
            )
        except Exception as exc:
            ui.warn(
                f"Gen {generation}: proactive_branch lifecycle update failed ({exc})"
            )

        # Persist generation outcome into `experiments` and promotions into `discoveries`
        try:
            exp_verdict = {
                "passed": bool(verdict.promoted),
                "mean_delta": opt_float(
                    bp_stats,
                    "mean_delta",
                    verdict.winner_avg_depth - verdict.incumbent_avg_depth,
                ),
                "ci_lower": opt_float(bp_stats, "ci_lower", 0.0),
                "ci_upper": opt_float(bp_stats, "ci_upper", 0.0),
                "wilcoxon_p": opt_float(bp_stats, "wilcoxon_p", 1.0),
                "wins": int(bp_stats.get("wins", 0) or 0),
                "losses": int(bp_stats.get("losses", 0) or 0),
                "ties": int(bp_stats.get("ties", 0) or 0),
                "seeds_evaluated": int(bp_stats.get("paired_count", len(runs)) or 0),
                "is_early_stopped": False,
                "pilot_best_delta": opt_float(bp_stats, "iqm_delta", 0.0),
                "reason": str(verdict.reason or ""),
            }
            exp_hyp = {
                "observation_from_db": f"Swarm targets: {', '.join(verdict.target_functions[:4])}",
                "mechanism": str(verdict.reason or ""),
                "winner_id": str(verdict.winner_id),
                "holdout_iqm_delta": verdict.holdout_iqm_delta,
                "holdout_rejected": bool(getattr(verdict, "holdout_rejected", False)),
            }
            self.db_logger.save_experiment(
                experiment_id=f"exp_swarm_g{generation}",
                generation=generation,
                batch_id=batch_id,
                mode="SWARM",
                target_functions=verdict.target_functions,
                hypothesis=exp_hyp,
                verdict=exp_verdict,
            )
            if verdict.promoted and verdict.winner_id != "incumbent":
                self.db_logger.save_discovery(
                    discovery_id=f"disc_swarm_g{generation}",
                    generation=generation,
                    batch_id=batch_id,
                    mode="SWARM",
                    target_functions=verdict.target_functions,
                    binding_constraint=f"Deficits: {', '.join(verdict.target_functions[:3])}",
                    verified_c_rule=(
                        f"{verdict.winner_id} CRN Δdepth="
                        f"{verdict.winner_avg_depth - verdict.incumbent_avg_depth:+.2f}, "
                        f"IQMΔ={opt_float(bp_stats, 'iqm_delta', 0.0):+.3f}"
                    ),
                    mechanism=str(verdict.reason or ""),
                    mean_delta=opt_float(
                        bp_stats,
                        "iqm_delta",
                        verdict.winner_avg_depth - verdict.incumbent_avg_depth,
                    ),
                    avg_depth=float(batch_metrics["avg_depth"]),
                    p90_depth=float(batch_metrics["p90_depth"]),
                    max_depth=int(batch_metrics["max_depth"]),
                )
        except Exception as exc:
            ui.warn(
                f"Gen {generation}: experiment/discovery persistence failed ({exc})"
            )

        # 5. Extract Frontier Checkpoints & Ingest into MAP-Elites v2 Archive
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
        self._sync_batch_cost(batch_id, generation)
        self._plateau_streak = self._compute_depth_plateau_streak()

        if self.theories is not None and hasattr(self.theories, "save_document"):
            try:
                sb_lines = [
                    f"{row.get('peer_id')} ({row.get('deficit')}): D={row.get('avg_depth')}, maxD={row.get('max_depth')}, S={row.get('avg_score')}"
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
            except Exception as exc:
                ui.warn(f"Gen {generation}: outcome theory persistence warning ({exc})")

        try:
            write_status_report(
                workspace_dir=str(self.workspace_dir),
                db_path=self.db_path,
                gen=generation,
                start_time=self.campaign_start_time,
                baseline_runs=runs,
                active_max_steps=active_max_steps,
                current_policy=verdict.winner_policy_code,
                skills_count=len(verdict.winner_skills),
                niches=sorted(getattr(self.map_elites, "cells", {}).keys()) or ["peer"],
            )
        except Exception as exc:
            ui.warn(f"Gen {generation}: status report write failed ({exc})")

        if self.db_logger.prune_ticks and generation % 5 == 0:
            try:
                self.db_logger.prune_old_telemetry(
                    current_generation=generation, keep_generations=20
                )
            except Exception as exc:
                ui.warn(f"Gen {generation}: telemetry prune warning ({exc})")

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
        """Counts consecutive recent SWARM generations without a promoted breakthrough."""
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
            for _gen_i, _ad, _md, passed in rows:
                if bool(passed):
                    break
                streak += 1
            return streak
        except Exception:
            return 0

    def _compute_active_max_steps(self) -> int:
        """Uses `compute_adaptive_max_steps` from recent promoted batches."""
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
        except Exception as exc:
            ui.warn(f"Adaptive max_steps lookup warning: {exc}")
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
                    1 if ep_dict.get("depth") is None else int(ep_dict["depth"]),
                    1
                    if ep_dict.get("max_depth") is None
                    else int(ep_dict["max_depth"]),
                )
                act_bytes = bytes(ep_dict.get("action_bytes") or b"")
                if (
                    depth >= self.frontier_min_depth
                    and len(act_bytes) > 0
                    and depth_tier_counts.get(depth, 0) < 12
                ):
                    seed = 0 if ep_dict.get("seed") is None else int(ep_dict["seed"])
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
                            turns=0
                            if ep_dict.get("turns") is None
                            else int(ep_dict["turns"]),
                            action_bytes=prefix_bytes,
                            game_log=str(ep_dict.get("game_log") or ""),
                            checkpoint_id=cp_id,
                        )
                        depth_tier_counts[depth] = depth_tier_counts.get(depth, 0) + 1
                        new_cps += 1

        metrics = {
            "avg_depth": opt_float(batch_eval, "avg_depth", 1.0),
            "max_depth": opt_int(batch_eval, "max_depth", 1),
            "avg_score": opt_float(batch_eval, "avg_score", 0.0),
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
            "avg_xp_level": float(
                np.mean([opt_float(r, "xp_level", 1.0) for r in runs])
            )
            if runs
            else 1.0,
            "avg_unique_tiles": float(
                np.mean([opt_float(r, "unique_tiles", 0.0) for r in runs])
            )
            if runs
            else 0.0,
            "avg_branches": float(
                np.mean([opt_float(r, "dungeons_visited", 1.0) for r in runs])
            )
            if runs
            else 1.0,
            "pct_wait": float(np.mean([opt_float(r, "pct_wait", 0.0) for r in runs]))
            if runs
            else 0.0,
            "pct_search": float(
                np.mean([opt_float(r, "pct_search", 0.0) for r in runs])
            )
            if runs
            else 0.0,
            "avg_turns": float(np.mean([opt_float(r, "turns", 0.0) for r in runs]))
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
                except Exception as exc:
                    ui.warn(
                        f"Gen {generation}: incubate_baseline persistence failed ({exc})"
                    )
        return new_cps

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
        consec_failures = 0
        try:
            for g in range(start_gen, end_gen + 1):
                try:
                    res = self.run_generation(generation=g)
                    history.append(res)
                    consec_failures = 0
                except KeyboardInterrupt:
                    raise
                except PermissionError as exc:
                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ui.error(
                        f"Campaign halted on Generation {g}: PermissionError ({exc}). "
                        f"Logged to `{err_log_path}`."
                    )
                    try:
                        with err_log_path.open("a", encoding="utf-8") as ef:
                            ef.write(
                                f"\n[{ts}] Generation {g} Fatal PermissionError:\n{traceback.format_exc()}\n"
                            )
                    except Exception:
                        pass
                    raise
                except Exception as exc:
                    consec_failures += 1
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
                    if consec_failures >= 3:
                        ui.error(
                            f"Campaign halting after {consec_failures} consecutive failed generations."
                        )
                        break
        finally:
            self.close()

        ui.header("LOX Swarm Campaign Completed")
        return history
