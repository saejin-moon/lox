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

from lox.db.schema import DEFAULT_DB_PATH, init_db
from lox.db.summarizer import compute_function_stats, mark_outlier_good_runs


class BatchDatabaseLogger:
    """Writes batch metadata, runs, tail-sampled ticks, function stats, policies, and experiments to DuckDB."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        max_tick_runs_per_batch: int = 150,
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

        con = init_db(self.db_path)
        now = datetime.datetime.now()

        try:
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
                run_rows = [
                    (
                        r["run_id"],
                        batch_id,
                        generation,
                        int(r["seed"]),
                        str(r.get("role", "val-hum-law-fem")),
                        int(r.get("depth", 1)),
                        int(r.get("max_depth", 1)),
                        int(r.get("score", 0)),
                        int(r.get("turns", 0)),
                        int(r.get("steps", 0)),
                        int(r.get("xp_level", 1)),
                        int(r.get("final_hp", 0)),
                        int(r.get("max_hp", 0)),
                        int(r.get("final_ac", 10)),
                        int(r.get("gold", 0)),
                        str(r.get("end_reason", "ended")),
                        str(r.get("death_message", "")),
                        bool(r.get("is_outlier_good", False)),
                        float(r.get("outlier_z_score", 0.0)),
                        float(r.get("batch_percentile", 0.0)),
                        str(r.get("summary_digest", "")),
                        str(r.get("game_log", "")),
                        json.dumps(r.get("milestones", {})),
                        json.dumps(r.get("triggered_functions", {})),
                        str(r.get("last_active_function", "root")),
                        bytes(r.get("action_bytes", b"")),
                        float(r.get("wall_ms", 0.0)),
                    )
                    for r in runs
                ]
                con.executemany(
                    """
                    INSERT OR REPLACE INTO runs VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    run_rows,
                )

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

                tick_rows = []
                for r in selected_for_ticks:
                    rid = r["run_id"]
                    for t in r.get("ticks", []):
                        tick_rows.append(
                            (
                                rid,
                                int(t.get("turn", 0)),
                                int(t.get("step_idx", 0)),
                                int(t.get("depth", 1)),
                                int(t.get("dungeon_num", 0)),
                                int(t.get("hp", 0)),
                                int(t.get("max_hp", 0)),
                                int(t.get("energy", 0)),
                                int(t.get("ac", 10)),
                                int(t.get("xp_level", 1)),
                                int(t.get("hunger_state", 1)),
                                int(t.get("condition_mask", 0)),
                                int(t.get("y", 0)),
                                int(t.get("x", 0)),
                                str(t.get("action_name", "")),
                                int(t.get("raw_key", 0)),
                                str(t.get("active_function", "root")),
                                str(t.get("message", "")),
                                str(t.get("visible_monsters", "")),
                                str(t.get("inventory_summary", "")),
                            )
                        )
                if tick_rows:
                    con.executemany(
                        """
                        INSERT INTO ticks VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                        """,
                        tick_rows,
                    )

            if fn_stats:
                fn_rows = [
                    (
                        fs["batch_id"],
                        fs["generation"],
                        fs["function_name"],
                        fs["episodes_triggered"],
                        fs["trigger_rate"],
                        fs["total_calls"],
                        fs["total_turns_active"],
                        fs["avg_depth_when_triggered"],
                        fs["death_rate_while_active"],
                        fs["ochiai_good_run_score"],
                        fs["liblit_failure_score"],
                    )
                    for fs in fn_stats
                ]
                con.executemany(
                    """
                    INSERT INTO function_stats VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    fn_rows,
                )
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
        con = init_db(self.db_path)
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

        con = init_db(self.db_path)
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
        finally:
            con.close()
