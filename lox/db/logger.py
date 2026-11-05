"""
High-Throughput DuckDB Batch Logger with Smart Tail-Based Tick Sampling.
Persists:
- 100% of recorded runs in `runs` (with full `summary_digest`, `game_log`, stats, `is_outlier_good`, and 4KB `action_bytes`)
- Full tick-level traces in `ticks` for all `is_outlier_good = TRUE` runs, worst/median exemplars, and up to `max_tick_runs_per_batch` triggered runs.
"""

from __future__ import annotations

import datetime
import json
from typing import Any

import pyarrow as pa

from lox.db.schema import DEFAULT_DB_PATH, connect_db, init_db
from lox.db.summarizer import compute_function_stats, mark_outlier_good_runs
from lox.ui.console import ui


class BatchDatabaseLogger:
    """Writes batch metadata, runs, tail-sampled ticks, function stats, policies, and experiments to DuckDB."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        max_tick_runs_per_batch: int = 25,
    ):
        self.db_path = db_path
        self.max_tick_runs_per_batch = max_tick_runs_per_batch
        con = init_db(self.db_path)
        con.close()

    def save_batch(
        self,
        batch_id: str,
        generation: int,
        mode: str,
        parent_policy_id: str,
        candidate_policy_id: str,
        target_functions: list[str],
        runs: list[dict[str, Any]],
        episodes_discarded: int,
        batch_metrics: dict[str, Any],
        verdict: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Marks statistical outlier good runs, computes function stats, and writes
        `batches`, `runs`, `ticks` (tail-sampled), and `function_stats` to DuckDB.
        Returns the computed `function_stats` rows.
        """
        mark_outlier_good_runs(runs)
        fn_stats = compute_function_stats(batch_id, generation, runs)
        outlier_count = sum(1 for r in runs if r.get("is_outlier_good"))
        batch_metrics["outlier_count"] = outlier_count

        con = connect_db(self.db_path, read_only=False)
        now = datetime.datetime.now()

        try:
            con.execute("BEGIN TRANSACTION")
            con.execute(
                """
                INSERT OR REPLACE INTO batches VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    batch_id,
                    generation,
                    mode,
                    parent_policy_id,
                    candidate_policy_id,
                    json.dumps(target_functions),
                    len(runs),
                    int(episodes_discarded),
                    float(batch_metrics.get("avg_depth", 0.0)),
                    float(batch_metrics.get("p90_depth", 0.0)),
                    int(batch_metrics.get("max_depth", 0)),
                    float(batch_metrics.get("avg_score", 0.0)),
                    int(batch_metrics.get("max_score", 0)),
                    float(batch_metrics.get("avg_turns", 0.0)),
                    int(outlier_count),
                    bool(verdict["passed"]) if verdict else True,
                    float(verdict.get("wilcoxon_p", 1.0)) if verdict else 1.0,
                    float(verdict.get("ci_lower", 0.0)) if verdict else 0.0,
                    float(verdict.get("ci_upper", 0.0)) if verdict else 0.0,
                    now,
                ],
            )

            if runs:
                ui.status(f"Saving {len(runs):,} episode runs & telemetry to DuckDB...")
                run_cols = {
                    "run_id": [str(r["run_id"]) for r in runs],
                    "batch_id": [str(batch_id)] * len(runs),
                    "generation": [int(generation)] * len(runs),
                    "seed": [int(r["seed"]) for r in runs],
                    "role": [str(r.get("role", "val-hum-law-fem")) for r in runs],
                    "depth": [int(r.get("depth", 1)) for r in runs],
                    "max_depth": [int(r.get("max_depth", 1)) for r in runs],
                    "score": [int(r.get("score", 0)) for r in runs],
                    "turns": [int(r.get("turns", 0)) for r in runs],
                    "steps": [int(r.get("steps", 0)) for r in runs],
                    "xp_level": [int(r.get("xp_level", 1)) for r in runs],
                    "final_hp": [int(r.get("final_hp", 0)) for r in runs],
                    "max_hp": [int(r.get("max_hp", 0)) for r in runs],
                    "final_ac": [int(r.get("final_ac", 10)) for r in runs],
                    "gold": [int(r.get("gold", 0)) for r in runs],
                    "end_reason": [str(r.get("end_reason", "ended")) for r in runs],
                    "death_message": [str(r.get("death_message", "")) for r in runs],
                    "is_outlier_good": [
                        bool(r.get("is_outlier_good", False)) for r in runs
                    ],
                    "outlier_z_score": [
                        float(r.get("outlier_z_score", 0.0)) for r in runs
                    ],
                    "batch_percentile": [
                        float(r.get("batch_percentile", 0.0)) for r in runs
                    ],
                    "summary_digest": [str(r.get("summary_digest", "")) for r in runs],
                    "game_log": [str(r.get("game_log", "")) for r in runs],
                    "milestones_json": [
                        (
                            r.get("milestones_json")
                            if isinstance(r.get("milestones_json"), str)
                            and not r.get("milestones")
                            else json.dumps(r.get("milestones") or {})
                        )
                        for r in runs
                    ],
                    "triggered_functions": [
                        (
                            r.get("triggered_functions")
                            if isinstance(r.get("triggered_functions"), str)
                            else json.dumps(
                                r.get("triggered_functions")
                                if r.get("triggered_functions") is not None
                                else r.get("functions_triggered", {})
                            )
                        )
                        for r in runs
                    ],
                    "last_active_function": [
                        str(r.get("last_active_function", "root")) for r in runs
                    ],
                    "action_bytes": [bytes(r.get("action_bytes", b"")) for r in runs],
                    "wall_ms": [float(r.get("wall_ms", 0.0)) for r in runs],
                }
                run_tbl = pa.Table.from_pydict(run_cols)
                con.register("_temp_runs", run_tbl)
                con.execute("INSERT OR REPLACE INTO runs SELECT * FROM _temp_runs")
                con.unregister("_temp_runs")

                # Smart Tail-Based Tick Sampling:
                # Always keep ALL outlier good runs + sample of normal/median/failed runs up to max_tick_runs_per_batch
                selected_for_ticks = [
                    r for r in runs if r.get("is_outlier_good") and r.get("ticks")
                ]
                remaining_quota = max(
                    0, self.max_tick_runs_per_batch - len(selected_for_ticks)
                )
                if remaining_quota > 0:
                    others = [
                        r
                        for r in runs
                        if not r.get("is_outlier_good") and r.get("ticks")
                    ]
                    if len(others) <= remaining_quota:
                        selected_for_ticks.extend(others)
                    else:
                        step_stride = max(1, len(others) // remaining_quota)
                        selected_for_ticks.extend(
                            others[::step_stride][:remaining_quota]
                        )

                tick_cols: dict[str, list[Any]] = {
                    "run_id": [],
                    "turn": [],
                    "step_idx": [],
                    "depth": [],
                    "dungeon_num": [],
                    "hp": [],
                    "max_hp": [],
                    "energy": [],
                    "ac": [],
                    "xp_level": [],
                    "hunger_state": [],
                    "condition_mask": [],
                    "y": [],
                    "x": [],
                    "action_name": [],
                    "raw_key": [],
                    "active_function": [],
                    "message": [],
                    "visible_monsters": [],
                    "inventory_summary": [],
                }
                for r in selected_for_ticks:
                    rid = str(r["run_id"])
                    for t in r.get("ticks", []):
                        tick_cols["run_id"].append(rid)
                        tick_cols["turn"].append(int(t.get("turn", 0)))
                        tick_cols["step_idx"].append(int(t.get("step_idx", 0)))
                        tick_cols["depth"].append(int(t.get("depth", 1)))
                        tick_cols["dungeon_num"].append(int(t.get("dungeon_num", 0)))
                        tick_cols["hp"].append(int(t.get("hp", 0)))
                        tick_cols["max_hp"].append(int(t.get("max_hp", 0)))
                        tick_cols["energy"].append(int(t.get("energy", 0)))
                        tick_cols["ac"].append(int(t.get("ac", 10)))
                        tick_cols["xp_level"].append(int(t.get("xp_level", 1)))
                        tick_cols["hunger_state"].append(int(t.get("hunger_state", 1)))
                        tick_cols["condition_mask"].append(
                            int(t.get("condition_mask", 0))
                        )
                        tick_cols["y"].append(int(t.get("y", 0)))
                        tick_cols["x"].append(int(t.get("x", 0)))
                        tick_cols["action_name"].append(str(t.get("action_name", "")))
                        tick_cols["raw_key"].append(int(t.get("raw_key", 0)))
                        tick_cols["active_function"].append(
                            str(t.get("active_function", "root"))
                        )
                        tick_cols["message"].append(str(t.get("message", "")))
                        tick_cols["visible_monsters"].append(
                            str(t.get("visible_monsters", ""))
                        )
                        tick_cols["inventory_summary"].append(
                            str(t.get("inventory_summary", ""))
                        )

                if tick_cols["run_id"]:
                    tick_tbl = pa.Table.from_pydict(tick_cols)
                    con.register("_temp_ticks", tick_tbl)
                    con.execute("INSERT INTO ticks SELECT * FROM _temp_ticks")
                    con.unregister("_temp_ticks")

            if fn_stats:
                con.execute(
                    "DELETE FROM function_stats WHERE batch_id = ?", (batch_id,)
                )
                fn_cols = {
                    "batch_id": [str(fs["batch_id"]) for fs in fn_stats],
                    "generation": [int(fs["generation"]) for fs in fn_stats],
                    "function_name": [str(fs["function_name"]) for fs in fn_stats],
                    "episodes_triggered": [
                        int(fs["episodes_triggered"]) for fs in fn_stats
                    ],
                    "trigger_rate": [float(fs["trigger_rate"]) for fs in fn_stats],
                    "total_calls": [int(fs["total_calls"]) for fs in fn_stats],
                    "total_turns_active": [
                        int(fs["total_turns_active"]) for fs in fn_stats
                    ],
                    "avg_depth_when_triggered": [
                        float(fs["avg_depth_when_triggered"]) for fs in fn_stats
                    ],
                    "death_rate_while_active": [
                        float(fs["death_rate_while_active"]) for fs in fn_stats
                    ],
                    "ochiai_good_run_score": [
                        float(fs["ochiai_good_run_score"]) for fs in fn_stats
                    ],
                    "liblit_failure_score": [
                        float(fs["liblit_failure_score"]) for fs in fn_stats
                    ],
                    "stagnation_turns": [
                        int(fs.get("stagnation_turns", 0)) for fs in fn_stats
                    ],
                    "stagnation_share": [
                        float(fs.get("stagnation_share", 0.0)) for fs in fn_stats
                    ],
                    "livelock_score": [
                        float(fs.get("livelock_score", 0.0)) for fs in fn_stats
                    ],
                }
                fn_tbl = pa.Table.from_pydict(fn_cols)
                con.register("_temp_fn_stats", fn_tbl)
                con.execute("INSERT INTO function_stats SELECT * FROM _temp_fn_stats")
                con.unregister("_temp_fn_stats")

            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()

        return fn_stats

    def save_policy(
        self,
        policy_id: str,
        generation: int,
        mode: str,
        policy_code: str,
        macros_code: str,
        params: dict[str, Any],
        avg_depth: float,
        max_depth: int,
        is_promoted: bool,
    ) -> None:
        con = connect_db(self.db_path, read_only=False)
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO policies VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    policy_id,
                    generation,
                    mode,
                    policy_code,
                    macros_code,
                    json.dumps(params),
                    float(avg_depth),
                    int(max_depth),
                    bool(is_promoted),
                    datetime.datetime.now(),
                ],
            )
        finally:
            con.close()

    def save_experiment(
        self,
        experiment_id: str,
        generation: int,
        batch_id: str,
        mode: str,
        target_functions: list[str],
        hypothesis: dict[str, Any],
        verdict: dict[str, Any],
    ) -> None:
        import yaml

        con = connect_db(self.db_path, read_only=False)
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO experiments VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    experiment_id,
                    generation,
                    batch_id,
                    mode,
                    json.dumps(target_functions),
                    yaml.safe_dump(hypothesis, sort_keys=False),
                    str(
                        hypothesis.get("observation_from_db")
                        or hypothesis.get("causal_finding")
                        or hypothesis.get("binding_constraint")
                        or hypothesis.get("verified_c_rule")
                        or ""
                    ),
                    str(hypothesis.get("mechanism") or ""),
                    bool(verdict.get("passed", False)),
                    float(verdict.get("mean_delta", 0.0)),
                    float(verdict.get("ci_lower", 0.0)),
                    float(verdict.get("ci_upper", 0.0)),
                    float(verdict.get("wilcoxon_p", 1.0)),
                    int(verdict.get("wins", 0)),
                    int(verdict.get("losses", 0)),
                    int(verdict.get("ties", 0)),
                    str(verdict.get("reason", "")),
                    datetime.datetime.now(),
                ],
            )
            if (
                verdict.get("passed")
                and mode != "PILOT"
                and (hypothesis.get("verified_c_rule") or hypothesis.get("mechanism"))
            ):
                try:
                    self.save_discovery(
                        discovery_id=f"disc_{batch_id}",
                        generation=generation,
                        batch_id=batch_id,
                        mode=mode,
                        target_functions=target_functions,
                        binding_constraint=str(
                            hypothesis.get("binding_constraint") or ""
                        ),
                        verified_c_rule=str(hypothesis.get("verified_c_rule") or ""),
                        mechanism=str(hypothesis.get("mechanism") or ""),
                        mean_delta=float(verdict.get("mean_delta", 0.0)),
                        avg_depth=float(verdict.get("avg_depth", 1.0)),
                        p90_depth=float(verdict.get("p90_depth", 1.0)),
                        max_depth=int(verdict.get("max_depth", 1)),
                    )
                except Exception:
                    pass
        finally:
            con.close()

    def save_discovery(
        self,
        discovery_id: str,
        generation: int,
        batch_id: str,
        mode: str,
        target_functions: list[str],
        binding_constraint: str,
        verified_c_rule: str,
        mechanism: str,
        mean_delta: float,
        avg_depth: float = 1.0,
        p90_depth: float = 1.0,
        max_depth: int = 1,
    ) -> None:
        con = connect_db(self.db_path, read_only=False)
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO discoveries (
                    discovery_id, generation, batch_id, mode, target_functions,
                    binding_constraint, verified_c_rule, mechanism, mean_delta,
                    avg_depth, p90_depth, max_depth, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    discovery_id,
                    generation,
                    batch_id,
                    mode,
                    json.dumps(target_functions),
                    str(binding_constraint or ""),
                    str(verified_c_rule or ""),
                    str(mechanism or ""),
                    float(mean_delta),
                    float(avg_depth),
                    float(p90_depth),
                    int(max_depth),
                    datetime.datetime.now(),
                ],
            )
        finally:
            con.close()

    def get_recent_discoveries(self, limit: int = 12) -> list[dict[str, Any]]:
        con = connect_db(self.db_path, read_only=True)
        try:
            cur = con.execute(
                """
                SELECT discovery_id, generation, batch_id, mode, target_functions,
                       binding_constraint, verified_c_rule, mechanism, mean_delta,
                       avg_depth, p90_depth, max_depth, created_at
                FROM discoveries
                ORDER BY generation DESC, created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception:
            return []
        finally:
            con.close()

    def get_recent_failed_experiments(self, limit: int = 8) -> list[dict[str, Any]]:
        con = connect_db(self.db_path, read_only=True)
        try:
            cur = con.execute(
                """
                SELECT experiment_id, generation, batch_id, mode, target_functions,
                       observation_from_db, mechanism, verdict_reason, mean_delta, created_at
                FROM experiments
                WHERE passed = FALSE
                ORDER BY generation DESC, created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception:
            return []
        finally:
            con.close()

    def save_synthesis_log(
        self,
        log_id: str,
        generation: int,
        batch_id: str,
        mode: str,
        turn_index: int,
        turn_type: str,
        repair_idx: int = 0,
        model: str = "",
        thinking_content: str = "",
        message_content: str = "",
        tool_calls_json: str = "",
        tool_results_json: str = "",
        candidate_code: str = "",
        skill_name: str = "",
        skill_code: str = "",
        hypothesis_yaml: str = "",
        status: str = "",
        error_diagnostic: str = "",
        prompt_sent: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        reasoning_tokens: int = 0,
        wall_ms: float = 0.0,
    ) -> None:
        """Saves a rich turn-level synthesis record (reasoning tokens, candidate code, errors, tools) to DuckDB."""
        con = connect_db(self.db_path, read_only=False)
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO synthesis_logs (
                    log_id, generation, batch_id, mode, turn_index, turn_type, repair_idx,
                    model, thinking_content, message_content, tool_calls_json, tool_results_json,
                    candidate_code, skill_name, skill_code, hypothesis_yaml, status,
                    error_diagnostic, prompt_sent, prompt_tokens, completion_tokens,
                    reasoning_tokens, wall_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    str(log_id),
                    int(generation),
                    str(batch_id or ""),
                    str(mode or ""),
                    int(turn_index),
                    str(turn_type or ""),
                    int(repair_idx),
                    str(model or ""),
                    str(thinking_content or ""),
                    str(message_content or ""),
                    str(tool_calls_json or "[]"),
                    str(tool_results_json or "[]"),
                    str(candidate_code or ""),
                    str(skill_name or ""),
                    str(skill_code or ""),
                    str(hypothesis_yaml or ""),
                    str(status or ""),
                    str(error_diagnostic or ""),
                    str(prompt_sent or ""),
                    int(prompt_tokens),
                    int(completion_tokens),
                    int(reasoning_tokens),
                    float(wall_ms),
                    datetime.datetime.now(),
                ],
            )
        except Exception:
            pass
        finally:
            con.close()

    def get_recent_synthesis_logs(
        self,
        generation: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieves recent synthesis logs ordered by generation and turn sequence."""
        con = connect_db(self.db_path, read_only=True)
        try:
            if generation is not None:
                cur = con.execute(
                    """
                    SELECT log_id, generation, batch_id, mode, turn_index, turn_type, repair_idx,
                           model, thinking_content, message_content, tool_calls_json, tool_results_json,
                           candidate_code, skill_name, skill_code, hypothesis_yaml, status,
                           error_diagnostic, prompt_sent, prompt_tokens, completion_tokens,
                           reasoning_tokens, wall_ms, created_at
                    FROM synthesis_logs
                    WHERE generation = ?
                    ORDER BY turn_index ASC, repair_idx ASC, created_at ASC
                    LIMIT ?
                    """,
                    (generation, limit),
                )
            else:
                cur = con.execute(
                    """
                    SELECT log_id, generation, batch_id, mode, turn_index, turn_type, repair_idx,
                           model, thinking_content, message_content, tool_calls_json, tool_results_json,
                           candidate_code, skill_name, skill_code, hypothesis_yaml, status,
                           error_diagnostic, prompt_sent, prompt_tokens, completion_tokens,
                           reasoning_tokens, wall_ms, created_at
                    FROM synthesis_logs
                    ORDER BY generation DESC, turn_index DESC, repair_idx DESC, created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception:
            return []
        finally:
            con.close()

    def prune_old_telemetry(
        self,
        current_generation: int,
        keep_generations: int = 20,
        keep_recent_generations: int | None = None,
    ) -> dict[str, int]:
        keep = (
            keep_recent_generations
            if keep_recent_generations is not None
            else keep_generations
        )
        cutoff_gen = current_generation - keep
        pilot_cutoff_gen = current_generation - 5
        con = connect_db(self.db_path, read_only=False)
        try:
            t_before = con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            con.execute(
                """
                DELETE FROM ticks
                WHERE run_id IN (
                    SELECT r.run_id FROM runs r
                    JOIN batches b ON r.batch_id = b.batch_id
                    WHERE b.generation < ?
                      AND (b.hypothesis_passed = FALSE OR b.hypothesis_passed IS NULL)
                      AND (r.is_outlier_good = FALSE OR r.is_outlier_good IS NULL)
                )
                """,
                (cutoff_gen,),
            )
            t_after = con.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
            ticks_deleted = t_before - t_after

            r_before = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            con.execute(
                """
                DELETE FROM runs
                WHERE batch_id LIKE 'pilot_%'
                  AND generation < ?
                """,
                (pilot_cutoff_gen,),
            )
            r_after = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            runs_deleted = r_before - r_after

            con.execute("CHECKPOINT")
            return {"ticks_pruned": ticks_deleted, "runs_pruned": runs_deleted}
        except Exception:
            return {"ticks_pruned": 0, "runs_pruned": 0}
        finally:
            con.close()
