"""
LOX Campaign Orchestrator.
Coordinates the 4 Modification Modes (`FULL_WRITE`, `MACRO_WRITE`, `FUNC_MOD_ADD`, `PARAM_TUNE`),
the Persistent NLE Worker Pool, the Multi-Table DuckDB Logger & Outlier Summarizer,
and the Stateless Author Agent.
"""

from __future__ import annotations

import datetime
import json
import os
from typing import Any

from lox.author.agent import AuthorAgent
from lox.datasets.download_experts import ingest_expert_runs_into_duckdb
from lox.db.logger import BatchDatabaseLogger
from lox.db.schema import DEFAULT_DB_PATH, connect_db, init_db
from lox.db.summarizer import build_batch_contrast_digest
from lox.engine.hypothesis import (
    evaluate_paired_hypothesis,
    should_stop_early_for_futility,
)
from lox.engine.modes import run_spsa_param_tuning
from lox.engine.scheduler import ModeScheduler
from lox.lang.macros import DEFAULT_MACROS_PATH, MacroLibrary
from lox.runner.pool import PersistentEvalPool
from lox.ui.console import ui

DEFAULT_POLICY_PATH = "data/workspace/policy.py"


class CampaignOrchestrator:
    """Drives the end-to-end evolutionary policy synthesis loop."""

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "google/gemma-4-31b-it",
        api_key: str | None = None,
        db_path: str = DEFAULT_DB_PATH,
        policy_path: str = DEFAULT_POLICY_PATH,
        macros_path: str = DEFAULT_MACROS_PATH,
        role: str = "val-hum-law-fem",
        workers: int = 8,
        max_steps: int = 25000,
        full_write_episodes: int = 10000,
        func_mod_episodes_per_fn: int = 1000,
        param_tune_episodes: int = 100,
        param_tune_every: int = 3,
        macro_every: int = 5,
        early_stopping: bool = True,
        max_repair_turns: int = 5,
        pilot_revisions: int = 5,
        pilot_episodes: int = 20,
    ):
        self.provider = provider
        self.model = model
        self.db_path = db_path
        self.policy_path = policy_path
        self.macros_path = macros_path
        self.role = role
        self.workers = workers
        self.max_steps = max_steps
        self.full_write_episodes = full_write_episodes
        self.func_mod_episodes_per_fn = func_mod_episodes_per_fn
        self.param_tune_episodes = param_tune_episodes
        self.early_stopping = early_stopping
        self.pilot_revisions = pilot_revisions
        self.pilot_episodes = pilot_episodes

        os.makedirs(
            os.path.dirname(os.path.abspath(self.policy_path)) or ".", exist_ok=True
        )
        os.makedirs(
            os.path.dirname(os.path.abspath(self.macros_path)) or ".", exist_ok=True
        )
        con = init_db(self.db_path)
        con.close()

        self.db_logger = BatchDatabaseLogger(db_path=self.db_path)
        self.macro_lib = MacroLibrary(path=self.macros_path)
        self.author = AuthorAgent(
            provider=provider,
            model=model,
            api_key=api_key,
            db_path=db_path,
            macros_path=macros_path,
            max_repair_turns=max_repair_turns,
        )
        self.scheduler = ModeScheduler(
            param_tune_every=param_tune_every,
            macro_every=macro_every,
        )
        self.pool = PersistentEvalPool(
            workers=workers,
            role=role,
            max_steps=max_steps,
        )

    def load_active_policy(self) -> str:
        if os.path.exists(self.policy_path):
            with open(self.policy_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ""

    def save_active_policy(self, code: str) -> None:
        with open(self.policy_path, "w", encoding="utf-8") as f:
            f.write(code.strip() + "\n" if code.strip() else "")

    def get_recent_experiments_markdown(self, limit: int = 5) -> str:
        con = connect_db(self.db_path, read_only=True)
        try:
            rows = con.execute(
                """
                SELECT generation, mode, target_functions, passed, mean_delta, ci_lower, ci_upper, wilcoxon_p, verdict_reason, mechanism
                FROM experiments
                ORDER BY generation DESC
                LIMIT ?
                """,
                [limit],
            ).fetchall()
            if not rows:
                return ""
            lines = []
            for r in rows:
                gen, mode, tf, passed, md, cil, ciu, wp, reason, mech = r
                status = "PROMOTED" if passed else "REJECTED"
                lines.append(
                    f"- **Gen {gen} ({mode} on {tf}) -> {status}**: "
                    f"Δ={md:+.3f} (95% CS [{cil:+.3f}, {ciu:+.3f}], p={wp:.4f}) | {reason} | Mechanism: {str(mech)[:90]}"
                )
            return "\n".join(lines)
        except Exception:
            return ""
        finally:
            con.close()

    def close(self) -> None:
        """Terminates and closes the persistent worker pool cleanly."""
        if hasattr(self, "pool") and self.pool is not None:
            self.pool.close()

    def run_campaign(
        self,
        max_generations: int = 50,
        target_depth: float = 20.0,
        forced_mode: str | None = None,
        seed_base: int = 10000,
    ) -> dict[str, Any]:
        """Executes the multi-generation policy synthesis campaign."""
        ui.banner(
            model=self.model,
            provider=self.provider,
            db_path=self.db_path,
            workers=self.workers,
            role=self.role,
        )

        # Ensure expert_runs table is populated for the Author Agent to query
        exp_info = ingest_expert_runs_into_duckdb(
            self.db_path, max_ascensions=30, fetch_dumplogs=5
        )
        ui.info(
            f"Expert NetHack corpus ready in DuckDB (`expert_runs`: {exp_info['expert_runs']} runs)."
        )

        current_policy = self.load_active_policy()
        current_policy_id = "pol_gen000" if current_policy else "none"
        baseline_by_seed: dict[int, dict[str, Any]] = {}
        baseline_runs: list[dict[str, Any]] = []
        registered_params: dict[str, dict[str, Any]] = {}
        last_batch_digest = ""

        campaign_id = datetime.datetime.now().strftime("c%Y%m%d_%H%M%S")
        start_gen = 0
        if current_policy and forced_mode != "FULL_WRITE":
            start_gen = 1
            # Check DuckDB to see if prior generations were completed, and load baseline state
            con = connect_db(self.db_path, read_only=True)
            try:
                max_gen_row = con.execute(
                    "SELECT MAX(generation) FROM batches"
                ).fetchone()
                last_db_gen = (
                    max_gen_row[0]
                    if (max_gen_row and max_gen_row[0] is not None)
                    else None
                )
                if last_db_gen is not None:
                    start_gen = int(last_db_gen) + 1

                    # Load the most recent promoted policy's baseline runs from DuckDB
                    prom_row = con.execute(
                        """
                        SELECT p.candidate_policy_id, p.params_json, b.batch_id
                        FROM policies p
                        JOIN batches b ON b.candidate_policy_id = p.candidate_policy_id
                        WHERE p.is_promoted = TRUE
                        ORDER BY p.generation DESC LIMIT 1
                        """
                    ).fetchone()
                    if prom_row:
                        prom_pol_id, prom_params_json, prom_batch_id = prom_row
                        current_policy_id = prom_pol_id
                        if prom_params_json:
                            try:
                                registered_params.update(json.loads(prom_params_json))
                            except Exception:
                                pass

                        run_cur = con.execute(
                            """
                            SELECT run_id, seed, depth, max_depth, score, turns, xp_level,
                                   final_hp, max_hp, final_ac, gold, end_reason,
                                   is_outlier_good, outlier_z_score, batch_percentile,
                                   summary_digest, game_log, triggered_functions, last_active_function
                            FROM runs WHERE batch_id = ?
                            """,
                            (prom_batch_id,),
                        )
                        run_cols = [d[0] for d in run_cur.description]
                        for r_tup in run_cur.fetchall():
                            r_dict = dict(zip(run_cols, r_tup))
                            baseline_by_seed[int(r_dict["seed"])] = r_dict
                        baseline_runs = list(baseline_by_seed.values())

                        fs_cur = con.execute(
                            """
                            SELECT batch_id, generation, function_name, episodes_triggered,
                                   trigger_rate, total_calls, total_turns_active,
                                   avg_depth_when_triggered, death_rate_while_active,
                                   ochiai_good_run_score, liblit_failure_score
                            FROM function_stats WHERE batch_id = ?
                            """,
                            (prom_batch_id,),
                        )
                        fs_cols = [d[0] for d in fs_cur.description]
                        fn_stats = [dict(zip(fs_cols, t)) for t in fs_cur.fetchall()]

                        b_cur = con.execute(
                            "SELECT * FROM batches WHERE batch_id = ?",
                            (prom_batch_id,),
                        )
                        b_cols = [d[0] for d in b_cur.description]
                        b_row = b_cur.fetchone()
                        batch_stats = dict(zip(b_cols, b_row)) if b_row else {}

                        last_batch_digest = build_batch_contrast_digest(
                            batch_stats, baseline_runs, fn_stats
                        )
                        ui.info(
                            f"Resuming campaign at Generation {start_gen:03d} (loaded {len(baseline_runs)} baseline runs from promoted batch `{prom_batch_id}`)."
                        )
            except Exception:
                pass
            finally:
                con.close()

        try:
            # If warm-starting from an existing policy.py without prior DB runs, evaluate a baseline batch first
            if current_policy and start_gen >= 1 and not baseline_runs:
                warm_batch_id = f"{campaign_id}_g000_warm"
                ui.info(
                    f"Evaluating existing `{self.policy_path}` on {self.func_mod_episodes_per_fn} seeds to establish baseline..."
                )
                warm_res = self.pool.evaluate_batch(
                    batch_id=warm_batch_id,
                    policy_code=current_policy,
                    macros_code=self.macro_lib.load_code(),
                    target_episodes=self.func_mod_episodes_per_fn,
                    seed_start=seed_base,
                    record_ticks=True,
                    show_progress=True,
                )
                if warm_res.get("registered_params"):
                    registered_params.update(warm_res["registered_params"])
                for r in warm_res["runs"]:
                    baseline_by_seed[int(r["seed"])] = r
                baseline_runs = list(baseline_by_seed.values())
                warm_verdict = {
                    "passed": True,
                    "mean_delta": warm_res["avg_depth"],
                    "ci_lower": warm_res["avg_depth"],
                    "ci_upper": warm_res["avg_depth"],
                    "wilcoxon_p": 0.0,
                    "wins": warm_res["episodes_recorded"],
                    "losses": 0,
                    "ties": 0,
                    "reason": "Warm-start baseline established from existing policy.py",
                }
                warm_fn_stats = self.db_logger.save_batch(
                    batch_id=warm_batch_id,
                    generation=0,
                    mode="FULL_WRITE",
                    parent_policy_id="none",
                    candidate_policy_id=current_policy_id,
                    target_functions=[],
                    runs=warm_res["runs"],
                    episodes_discarded=0,
                    batch_metrics=warm_res,
                    verdict=warm_verdict,
                )
                last_batch_digest = build_batch_contrast_digest(
                    warm_res, baseline_runs, warm_fn_stats
                )

            for gen in range(start_gen, start_gen + max_generations):
                mode = self.scheduler.select_mode(
                    generation=gen,
                    has_policy=bool(current_policy),
                    has_tunable_params=bool(
                        registered_params or "param(" in current_policy
                    ),
                    forced_mode=forced_mode,
                )
                batch_id = f"{campaign_id}_g{gen:03d}"

                # --- MODE 4: ZERO-LLM PARAMETER FINETUNING (CRN-SPSA) ---
                if mode == "PARAM_TUNE":
                    ui.gen_header(
                        gen=gen,
                        mode="PARAM_TUNE",
                        target_episodes=self.param_tune_episodes,
                        target_fns=list(registered_params.keys()) or ["all_params"],
                    )
                    # If registered_params not yet populated in memory, run 1 quick episode to collect them
                    if not registered_params and current_policy:
                        probe = self.pool.evaluate_batch(
                            batch_id=f"{batch_id}_probe",
                            policy_code=current_policy,
                            macros_code=self.macro_lib.load_code(),
                            target_episodes=1,
                            record_ticks=False,
                            show_progress=False,
                        )
                        registered_params.update(probe.get("registered_params", {}))

                    spsa_res = run_spsa_param_tuning(
                        pool=self.pool,
                        batch_id=batch_id,
                        policy_code=current_policy,
                        macros_code=self.macro_lib.load_code(),
                        registered_params=registered_params,
                        baseline_runs=baseline_runs,
                        validation_episodes=self.param_tune_episodes,
                        seed=seed_base + gen * 100,
                    )
                    verdict = spsa_res["verdict"]
                    val_batch = spsa_res["val_result"]
                    if val_batch is not None:
                        cand_id = f"pol_{batch_id}"
                        fn_stats = self.db_logger.save_batch(
                            batch_id=batch_id,
                            generation=gen,
                            mode="PARAM_TUNE",
                            parent_policy_id=current_policy_id,
                            candidate_policy_id=cand_id,
                            target_functions=list(registered_params.keys()),
                            runs=val_batch["runs"],
                            episodes_discarded=val_batch["episodes_discarded"],
                            batch_metrics=val_batch,
                            verdict=verdict,
                        )
                        ui.batch_summary(val_batch, verdict)
                        self.scheduler.record_outcome(
                            "PARAM_TUNE",
                            verdict.get("mean_delta", 0.0),
                            val_batch.get("wall_sec", 1.0),
                        )
                        self.db_logger.save_experiment(
                            experiment_id=f"exp_{batch_id}",
                            generation=gen,
                            batch_id=batch_id,
                            mode="PARAM_TUNE",
                            target_functions=list(registered_params.keys()),
                            hypothesis={
                                "mode": "PARAM_TUNE",
                                "target_functions": list(registered_params.keys()),
                                "observation_from_db": "CRN-SPSA parameter optimization",
                                "mechanism": f"Optimized params: {spsa_res['optimized_params']}",
                            },
                            verdict=verdict,
                        )
                        if verdict["passed"]:
                            current_policy = spsa_res["new_policy_code"]
                            current_policy_id = cand_id
                            self.save_active_policy(current_policy)
                            if spsa_res.get("new_macros_code"):
                                self.macro_lib.save_code(spsa_res["new_macros_code"])
                            for pk, pval in spsa_res["optimized_params"].items():
                                if pk in registered_params:
                                    if isinstance(registered_params[pk], dict):
                                        registered_params[pk]["default"] = float(pval)
                                    else:
                                        registered_params[pk].default = float(pval)
                            for r in val_batch["runs"]:
                                baseline_by_seed[int(r["seed"])] = r
                            baseline_runs = list(baseline_by_seed.values())
                            last_batch_digest = build_batch_contrast_digest(
                                val_batch, val_batch["runs"], fn_stats
                            )
                            self.db_logger.save_policy(
                                policy_id=cand_id,
                                generation=gen,
                                mode="PARAM_TUNE",
                                policy_code=current_policy,
                                macros_code=self.macro_lib.load_code(),
                                params=spsa_res["optimized_params"],
                                avg_depth=val_batch["avg_depth"],
                                max_depth=val_batch["max_depth"],
                                is_promoted=True,
                            )
                    else:
                        ui.warn(verdict["reason"])
                    continue

                # --- MODES 1, 2, 3: STATELESS AUTHOR AGENT SYNTHESIS ---
                recent_exps = self.get_recent_experiments_markdown(limit=5)
                if mode == "MACRO_WRITE":
                    ui.gen_header(gen=gen, mode="MACRO_WRITE", target_episodes=0)
                elif mode == "FULL_WRITE":
                    ui.gen_header(
                        gen=gen,
                        mode="FULL_WRITE",
                        target_episodes=self.full_write_episodes,
                    )
                else:
                    ui.gen_header(
                        gen=gen,
                        mode="FUNC_MOD_ADD",
                        target_episodes=self.func_mod_episodes_per_fn,
                    )

                try:
                    proposal = self.author.synthesize(
                        mode=mode,
                        generation=gen,
                        current_policy=current_policy,
                        batch_digest=last_batch_digest,
                        recent_experiments=recent_exps,
                        pool=self.pool,
                        baseline_runs=baseline_runs,
                        max_pilot_runs=self.pilot_revisions,
                        pilot_episodes=self.pilot_episodes,
                    )
                except Exception as synth_err:
                    ui.warn(
                        f"Gen {gen:03d} synthesis failed after all repair turns ({synth_err}). Retrying next generation with fresh context."
                    )
                    continue
                ui.hypothesis_box(proposal.hypothesis)

                # --- MODE 2: MACRO_WRITE (Skip Rollout & Advance Immediately) ---
                if proposal.mode == "MACRO_WRITE":
                    ui.info(
                        f"Committed macro(s) {proposal.target_functions} to `{self.macros_path}`. "
                        "Skipping rollout and starting next generation."
                    )
                    self.scheduler.record_outcome("MACRO_WRITE", 0.05, 1.0)
                    self.db_logger.save_experiment(
                        experiment_id=f"exp_{batch_id}",
                        generation=gen,
                        batch_id=batch_id,
                        mode="MACRO_WRITE",
                        target_functions=proposal.target_functions,
                        hypothesis=proposal.hypothesis,
                        verdict={
                            "passed": True,
                            "mean_delta": 0.0,
                            "ci_lower": 0.0,
                            "ci_upper": 0.0,
                            "wilcoxon_p": 1.0,
                            "wins": 0,
                            "losses": 0,
                            "ties": 0,
                            "reason": "Macro committed directly to macros.py",
                        },
                    )
                    continue

                # --- MODE 1 (`FULL_WRITE`) & MODE 3 (`FUNC_MOD_ADD`) ---
                raw_targets = [
                    f for f in proposal.target_functions if f and f != "__init__"
                ]
                non_root_targets = [
                    f for f in raw_targets if f not in ("run", "step", "act")
                ]
                target_fns = (
                    []
                    if proposal.mode == "FULL_WRITE"
                    else (non_root_targets or raw_targets)
                )
                num_funcs = max(1, len(target_fns))
                target_eps = (
                    self.full_write_episodes
                    if proposal.mode == "FULL_WRITE"
                    else self.func_mod_episodes_per_fn * num_funcs
                )

                # Prioritize seeds from baseline_runs so paired comparison is 1-to-1 on identical seeds
                priority_seeds = (
                    [int(r["seed"]) for r in baseline_runs] if baseline_runs else None
                )

                early_stop_cb = None
                if (
                    self.early_stopping
                    and proposal.mode == "FUNC_MOD_ADD"
                    and baseline_by_seed
                ):
                    min_p = min(60, max(20, target_eps // 4))

                    def _check_futility(
                        cand_runs: list[dict[str, Any]],
                        _b=baseline_by_seed,
                        _mp=min_p,
                    ) -> bool:
                        return should_stop_early_for_futility(
                            cand_runs, _b, min_pairs=_mp
                        )

                    early_stop_cb = _check_futility

                eval_res = self.pool.evaluate_batch(
                    batch_id=batch_id,
                    policy_code=proposal.full_policy_code,
                    macros_code=proposal.full_macros_code,
                    target_episodes=target_eps,
                    target_functions=target_fns
                    if proposal.mode == "FUNC_MOD_ADD"
                    else None,
                    priority_seeds=priority_seeds,
                    seed_start=seed_base + gen * 10000,
                    record_ticks=True,
                    show_progress=True,
                    early_stop_fn=early_stop_cb,
                )

                if eval_res.get("registered_params"):
                    registered_params.update(eval_res["registered_params"])

                cand_id = f"pol_{batch_id}"

                if proposal.mode == "FULL_WRITE" or not baseline_runs:
                    verdict = {
                        "passed": bool(eval_res["episodes_recorded"] > 0),
                        "mean_delta": eval_res["avg_depth"],
                        "ci_lower": eval_res["avg_depth"],
                        "ci_upper": eval_res["avg_depth"],
                        "wilcoxon_p": 0.0,
                        "wins": eval_res["episodes_recorded"],
                        "losses": 0,
                        "ties": 0,
                        "reason": "Initial FULL_WRITE baseline established",
                    }
                else:
                    # Backfill any triggered seeds not yet in `baseline_by_seed` so 100% of triggered episodes
                    # are paired 1-to-1 by exact seed against `current_policy` (zero survivorship bias!)
                    missing_seeds = [
                        int(r["seed"])
                        for r in eval_res["runs"]
                        if int(r["seed"]) not in baseline_by_seed
                    ]
                    if (
                        missing_seeds
                        and current_policy
                        and not eval_res.get("early_stopped")
                        and not eval_res.get("unreachable_abort")
                    ):
                        backfill_res = self.pool.evaluate_batch(
                            batch_id=f"{batch_id}_base_backfill",
                            policy_code=current_policy,
                            macros_code=self.macro_lib.load_code(),
                            target_episodes=len(missing_seeds),
                            priority_seeds=missing_seeds,
                            record_ticks=False,
                            show_progress=False,
                        )
                        for br in backfill_res["runs"]:
                            baseline_by_seed[int(br["seed"])] = br
                        baseline_runs = list(baseline_by_seed.values())

                    verdict = evaluate_paired_hypothesis(
                        eval_res["runs"], baseline_runs
                    )
                    if eval_res.get("early_stopped"):
                        verdict["reason"] = (
                            f"Early Futility Stop ({len(eval_res['runs'])} eps): "
                            + verdict["reason"]
                        )
                    elif eval_res.get("unreachable_abort"):
                        verdict["passed"] = False
                        verdict["reason"] = (
                            f"Unreachable Function Abort: {target_fns} triggered 0 times in "
                            f"{eval_res.get('raw_episodes_run', 0)} episodes!"
                        )

                fn_stats = self.db_logger.save_batch(
                    batch_id=batch_id,
                    generation=gen,
                    mode=proposal.mode,
                    parent_policy_id=current_policy_id,
                    candidate_policy_id=cand_id,
                    target_functions=target_fns,
                    runs=eval_res["runs"],
                    episodes_discarded=eval_res["episodes_discarded"],
                    batch_metrics=eval_res,
                    verdict=verdict,
                )
                ui.batch_summary(eval_res, verdict)

                self.scheduler.record_outcome(
                    proposal.mode,
                    verdict.get("mean_delta", 0.0),
                    eval_res.get("wall_sec", 1.0),
                )
                self.db_logger.save_experiment(
                    experiment_id=f"exp_{batch_id}",
                    generation=gen,
                    batch_id=batch_id,
                    mode=proposal.mode,
                    target_functions=target_fns,
                    hypothesis=proposal.hypothesis,
                    verdict=verdict,
                )
                self.db_logger.save_policy(
                    policy_id=cand_id,
                    generation=gen,
                    mode=proposal.mode,
                    policy_code=proposal.full_policy_code,
                    macros_code=proposal.full_macros_code,
                    params=registered_params,
                    avg_depth=eval_res["avg_depth"],
                    max_depth=eval_res["max_depth"],
                    is_promoted=verdict["passed"],
                )

                if verdict["passed"]:
                    current_policy = proposal.full_policy_code
                    current_policy_id = cand_id
                    self.save_active_policy(current_policy)
                    if proposal.mode == "FULL_WRITE":
                        baseline_by_seed = {int(r["seed"]): r for r in eval_res["runs"]}
                    else:
                        for r in eval_res["runs"]:
                            baseline_by_seed[int(r["seed"])] = r
                    baseline_runs = list(baseline_by_seed.values())
                    last_batch_digest = build_batch_contrast_digest(
                        eval_res, eval_res["runs"], fn_stats
                    )
                elif not last_batch_digest and eval_res["runs"]:
                    last_batch_digest = build_batch_contrast_digest(
                        eval_res, eval_res["runs"], fn_stats
                    )

                uncond_avg_depth = (
                    float(
                        sum(r["max_depth"] for r in baseline_runs) / len(baseline_runs)
                    )
                    if baseline_runs
                    else eval_res["avg_depth"]
                )
                if uncond_avg_depth >= target_depth and verdict["passed"]:
                    ui.info(
                        f"Campaign Target Depth ({target_depth:.1f}) achieved! Avg Depth: {uncond_avg_depth:.2f}"
                    )
                    break

            return {
                "active_policy_id": current_policy_id,
                "active_policy_code": current_policy,
                "baseline_runs_count": len(baseline_runs),
            }
        finally:
            self.pool.close()
