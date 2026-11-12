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

from lox.core.contracts import (
    EpisodeRun,
    normalize_episode_run,
    split_full_and_shard_runs,
)
from lox.db.schema import DEFAULT_DB_PATH, compute_llm_cost, connect_db, init_db
from lox.db.summarizer import compute_function_stats, mark_outlier_good_runs
from lox.ui.console import ui


class BatchDatabaseLogger:
    """Writes batch metadata, runs, tail-sampled ticks, function stats, policies, and experiments to DuckDB."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        max_tick_runs_per_batch: int = 1000,
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
        norm_runs: list[EpisodeRun] = [normalize_episode_run(r) for r in runs]
        full_runs, _ = split_full_and_shard_runs(norm_runs)
        stat_runs = full_runs if full_runs else norm_runs
        mark_outlier_good_runs(stat_runs)
        for r, er in zip(runs, norm_runs):
            if r is not er and isinstance(r, dict):
                r["is_outlier_good"] = er.get("is_outlier_good", False)
                r["outlier_z_score"] = er.get("outlier_z_score", 0.0)
                r["batch_percentile"] = er.get("batch_percentile", 0.0)
        runs = norm_runs
        fn_stats = compute_function_stats(batch_id, generation, runs)
        outlier_count = sum(1 for r in stat_runs if r.get("is_outlier_good"))
        batch_metrics["outlier_count"] = outlier_count

        con = connect_db(self.db_path, read_only=False)
        now = datetime.datetime.now()

        total_cost_usd = 0.0
        total_tokens = 0
        try:
            row_c = con.execute(
                "SELECT SUM(cost_usd), SUM(prompt_tokens + completion_tokens) FROM synthesis_logs WHERE batch_id = ?",
                [batch_id],
            ).fetchone()
            if row_c:
                total_cost_usd = float(row_c[0] or 0.0)
                total_tokens = int(row_c[1] or 0)
        except Exception:
            pass

        try:
            con.execute("BEGIN TRANSACTION")
            con.execute(
                """
                INSERT OR REPLACE INTO batches (
                    batch_id, generation, mode, parent_policy_id, candidate_policy_id,
                    target_functions, episodes_recorded, episodes_discarded,
                    avg_depth, p90_depth, max_depth, avg_score, max_score, avg_turns,
                    outlier_count, hypothesis_passed, wilcoxon_p, ci_lower, ci_upper,
                    seeds_evaluated, is_early_stopped, pilot_best_delta,
                    total_cost_usd, total_llm_tokens, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
                    int(batch_metrics.get("seeds_evaluated", len(runs))),
                    bool(batch_metrics.get("is_early_stopped", False)),
                    float(batch_metrics.get("pilot_best_delta", 0.0)),
                    float(total_cost_usd),
                    int(total_tokens),
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
                    "auto_esc_count": [int(r.get("auto_esc_count", 0)) for r in runs],
                    "subgen_preemptions": [
                        int(r.get("subgen_preemptions", 0)) for r in runs
                    ],
                    "policy_exceptions": [
                        int(r.get("policy_exceptions", 0)) for r in runs
                    ],
                    "unique_tiles": [int(r.get("unique_tiles", 0)) for r in runs],
                    "pct_wait": [float(r.get("pct_wait", 0.0)) for r in runs],
                    "pct_search": [float(r.get("pct_search", 0.0)) for r in runs],
                    "starved_with_food": [
                        bool(r.get("starved_with_food", False)) for r in runs
                    ],
                    "died_unarmored": [
                        bool(r.get("died_unarmored", False)) for r in runs
                    ],
                    "is_shard": [bool(r.get("is_shard", False)) for r in runs],
                    "shard_id": [str(r.get("shard_id", "")) for r in runs],
                }
                run_tbl = pa.Table.from_pydict(run_cols)
                run_cols_str = ", ".join(run_cols.keys())
                con.register("_temp_runs", run_tbl)
                con.execute(
                    f"INSERT OR REPLACE INTO runs ({run_cols_str}) SELECT {run_cols_str} FROM _temp_runs"
                )
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
                    "byte_offset": [],
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
                        tick_cols["byte_offset"].append(int(t.get("byte_offset", 0)))
                        tick_cols["message"].append(str(t.get("message", "")))
                        tick_cols["visible_monsters"].append(
                            str(t.get("visible_monsters", ""))
                        )
                        tick_cols["inventory_summary"].append(
                            str(t.get("inventory_summary", ""))
                        )

                if tick_cols["run_id"]:
                    tick_tbl = pa.Table.from_pydict(tick_cols)
                    tick_cols_str = ", ".join(tick_cols.keys())
                    con.register("_temp_ticks", tick_tbl)
                    con.execute(
                        f"INSERT INTO ticks ({tick_cols_str}) SELECT {tick_cols_str} FROM _temp_ticks"
                    )
                    con.unregister("_temp_ticks")

                    # Prune ticks from old, superseded non-champion, non-pilot, non-outlier batches
                    # so `ticks` stays fast while preserving 100% full ticks for champion, recent, pilot, and outlier runs.
                    con.execute(
                        """
                        DELETE FROM ticks
                        WHERE run_id NOT IN (SELECT run_id FROM champion_runs)
                          AND run_id NOT IN (SELECT run_id FROM latest_pilot_runs)
                          AND run_id NOT IN (
                              SELECT run_id FROM runs
                              WHERE is_outlier_good = TRUE
                                 OR batch_id IN (
                                     SELECT batch_id FROM batches
                                     ORDER BY created_at DESC LIMIT 4
                                 )
                          )
                        """
                    )

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
                fn_cols_str = ", ".join(fn_cols.keys())
                con.register("_temp_fn_stats", fn_tbl)
                con.execute(
                    f"INSERT INTO function_stats ({fn_cols_str}) SELECT {fn_cols_str} FROM _temp_fn_stats"
                )
                con.unregister("_temp_fn_stats")

            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.close()

        return fn_stats

    def load_champion_baseline(
        self, batch_id: str | None = None
    ) -> dict[str, Any] | None:
        """
        Loads the champion baseline package from DuckDB:
        - Resolves latest promoted batch if `batch_id` is None
        - Loads policy code, macros code, and registered params
        - Loads 100% full-episode baseline runs (zero shards) into normalized `EpisodeRun` objects
        - Populates `ticks` (or step byte offsets) for all runs
        - Loads function stats and batch stats
        Returns None if no promoted batches exist.
        """
        con = connect_db(self.db_path, read_only=True)
        try:
            target_batch_id = batch_id
            target_gen = 0
            prom_pol_id = ""
            prom_policy_code = ""
            prom_macros_code = ""
            registered_params: dict[str, Any] = {}

            if target_batch_id is None:
                prom_row = con.execute(
                    """
                    SELECT p.policy_id, p.policy_code, p.macros_code, p.params_json, b.batch_id, b.generation
                    FROM policies p
                    JOIN batches b ON b.candidate_policy_id = p.policy_id
                    WHERE p.is_promoted = TRUE
                      AND b.mode NOT IN ('PILOT', 'BASELINE_BACKFILL', 'MACRO_WRITE')
                    ORDER BY b.generation DESC, b.created_at DESC LIMIT 1
                    """
                ).fetchone()
                if not prom_row:
                    return None
                (
                    prom_pol_id,
                    prom_policy_code,
                    prom_macros_code,
                    prom_params_json,
                    target_batch_id,
                    target_gen,
                ) = prom_row
                if prom_params_json:
                    try:
                        registered_params = json.loads(prom_params_json)
                    except Exception:
                        registered_params = {}
            else:
                pol_row = con.execute(
                    """
                    SELECT p.policy_id, p.policy_code, p.macros_code, p.params_json, b.generation
                    FROM policies p
                    JOIN batches b ON b.candidate_policy_id = p.policy_id
                    WHERE b.batch_id = ?
                    LIMIT 1
                    """,
                    (target_batch_id,),
                ).fetchone()
                if pol_row:
                    (
                        prom_pol_id,
                        prom_policy_code,
                        prom_macros_code,
                        prom_params_json,
                        target_gen,
                    ) = pol_row
                    if prom_params_json:
                        try:
                            registered_params = json.loads(prom_params_json)
                        except Exception:
                            registered_params = {}

            run_cur = con.execute(
                """
                SELECT run_id, seed, depth, max_depth, score, turns, steps, xp_level,
                       final_hp, max_hp, final_ac, gold, end_reason, death_message,
                       is_outlier_good, outlier_z_score, batch_percentile,
                       summary_digest, game_log, triggered_functions, last_active_function,
                       action_bytes, milestones_json, unique_tiles, pct_wait, pct_search,
                       starved_with_food, died_unarmored, auto_esc_count,
                       subgen_preemptions, policy_exceptions
                FROM runs
                WHERE batch_id = ? AND run_id NOT LIKE 'shard_%' AND (is_shard IS NULL OR is_shard = FALSE)
                """,
                (target_batch_id,),
            )
            run_cols = [d[0] for d in run_cur.description]
            baseline_by_seed: dict[int, EpisodeRun] = {}
            run_by_id: dict[str, EpisodeRun] = {}
            for r_tup in run_cur.fetchall():
                r_dict = dict(zip(run_cols, r_tup))
                er = normalize_episode_run(r_dict)
                er["ticks"] = []
                baseline_by_seed[int(er["seed"])] = er
                run_by_id[str(er["run_id"])] = er

            if run_by_id:
                try:
                    tick_cur = con.execute(
                        """
                        SELECT t.run_id, t.step_idx, t.turn, t.depth, t.byte_offset, t.active_function
                        FROM ticks t
                        JOIN runs r ON t.run_id = r.run_id
                        WHERE r.batch_id = ? AND r.run_id NOT LIKE 'shard_%' AND (r.is_shard IS NULL OR r.is_shard = FALSE)
                        ORDER BY t.run_id, t.step_idx
                        """,
                        (target_batch_id,),
                    )
                    for (
                        tr_id,
                        t_step,
                        t_turn,
                        t_depth,
                        t_boff,
                        t_fn,
                    ) in tick_cur.fetchall():
                        rd = run_by_id.get(str(tr_id))
                        if rd is not None:
                            rd["ticks"].append(
                                {
                                    "step_idx": int(t_step or 0),
                                    "turn": int(t_turn or 0),
                                    "depth": int(t_depth or 1),
                                    "byte_offset": int(t_boff or 0),
                                    "active_function": str(t_fn or ""),
                                }
                            )
                except Exception:
                    pass

            baseline_runs = list(baseline_by_seed.values())

            fs_cur = con.execute(
                """
                SELECT batch_id, generation, function_name, episodes_triggered,
                       trigger_rate, total_calls, total_turns_active,
                       avg_depth_when_triggered, death_rate_while_active,
                       ochiai_good_run_score, liblit_failure_score,
                       stagnation_turns, stagnation_share, livelock_score
                FROM function_stats WHERE batch_id = ?
                """,
                (target_batch_id,),
            )
            fs_cols = [d[0] for d in fs_cur.description]
            fn_stats = []
            for t in fs_cur.fetchall():
                fs_dict = dict(zip(fs_cols, t))
                fs_dict["stagnation_turns"] = int(fs_dict.get("stagnation_turns") or 0)
                fs_dict["stagnation_share"] = float(
                    fs_dict.get("stagnation_share") or 0.0
                )
                fs_dict["livelock_score"] = float(fs_dict.get("livelock_score") or 0.0)
                fn_stats.append(fs_dict)

            b_cur = con.execute(
                "SELECT * FROM batches WHERE batch_id = ?",
                (target_batch_id,),
            )
            b_cols = [d[0] for d in b_cur.description]
            b_row = b_cur.fetchone()
            batch_stats = dict(zip(b_cols, b_row)) if b_row else {}

            max_gen_row = con.execute("SELECT MAX(generation) FROM batches").fetchone()
            max_gen = (
                int(max_gen_row[0])
                if (max_gen_row and max_gen_row[0] is not None)
                else int(target_gen or 0)
            )

            return {
                "batch_id": target_batch_id,
                "generation": int(target_gen or 0),
                "max_generation": max_gen,
                "policy_id": prom_pol_id,
                "policy_code": prom_policy_code,
                "macros_code": prom_macros_code,
                "params": registered_params,
                "baseline_runs": baseline_runs,
                "fn_stats": fn_stats,
                "batch_stats": batch_stats,
            }
        finally:
            con.close()

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
                INSERT OR REPLACE INTO policies (
                    policy_id, generation, mode, policy_code, macros_code,
                    params_json, avg_depth, max_depth, is_promoted, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        total_cost_usd = 0.0
        try:
            row_c = con.execute(
                "SELECT SUM(cost_usd) FROM synthesis_logs WHERE batch_id = ?",
                [batch_id],
            ).fetchone()
            if row_c and row_c[0] is not None:
                total_cost_usd = float(row_c[0])
        except Exception:
            pass

        try:
            con.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, generation, batch_id, mode, target_functions,
                    hypothesis_yaml, observation_from_db, mechanism, passed, mean_delta,
                    ci_lower, ci_upper, wilcoxon_p, wins, losses, ties,
                    seeds_evaluated, is_early_stopped, pilot_best_delta, total_cost_usd,
                    verdict_reason, created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
                    int(verdict.get("seeds_evaluated", 0)),
                    bool(verdict.get("is_early_stopped", False)),
                    float(verdict.get("pilot_best_delta", 0.0)),
                    float(total_cost_usd),
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
        agent_role: str = "author",
        cost_usd: float | None = None,
        wall_ms: float = 0.0,
    ) -> None:
        """Saves a rich turn-level synthesis record (reasoning tokens, candidate code, errors, tools, cost) to DuckDB."""
        if cost_usd is None:
            cost_usd = compute_llm_cost(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
            )
        con = connect_db(self.db_path, read_only=False)
        try:
            con.execute(
                """
                INSERT OR REPLACE INTO synthesis_logs (
                    log_id, generation, batch_id, mode, turn_index, turn_type, repair_idx,
                    model, thinking_content, message_content, tool_calls_json, tool_results_json,
                    candidate_code, skill_name, skill_code, hypothesis_yaml, status,
                    error_diagnostic, prompt_sent, prompt_tokens, completion_tokens,
                    reasoning_tokens, agent_role, cost_usd, wall_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    str(agent_role or "author"),
                    float(cost_usd or 0.0),
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
