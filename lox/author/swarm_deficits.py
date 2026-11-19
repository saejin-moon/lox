"""
Deterministic SQL & AST Deficit Analyzer for the Democratized LOX Peer Swarm.
("Hard Code Nothing": zero hardcoded NetHack heuristics or thematic silos.)

Partitions recent rollout telemetry into:
- Plateau signal (`CLIMBING` vs `PLATEAU`) for adaptive peer concurrency
- `D_succ`: outlier-good / top-depth trajectories and protected high-Ochiai branches
- `D_fail`: empirical failure clusters (confinement stalls, mortality modes, errors)
- Dormancy & bloat metrics (`tau_cull`): 0-turn branches and overlapping skill footprint
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lox.author.swarm_critic import extract_priority_selector_branches
from lox.db.schema import connect_db


@dataclass
class DeficitCluster:
    """Empirical failure or cleanup cluster claimed autonomously by a Peer Agent."""

    cluster_id: str
    kind: str  # e.g. 'confinement_stall', 'early_mortality', 'blunder_flag', 'dormant_cleanup', 'initial_synthesis'
    label: str
    run_count: int = 0
    avg_depth: float = 1.0
    avg_turns: float = 0.0
    sample_run_ids: list[str] = field(default_factory=list)
    sample_seeds: list[int] = field(default_factory=list)
    dominant_fns: list[str] = field(default_factory=list)
    replay_fixtures: list[dict[str, Any]] = field(default_factory=list)
    evidence: str = ""


@dataclass
class DeficitReport:
    """Complete deterministic telemetry partition for a generation."""

    generation: int
    regime: str  # 'CLIMBING' | 'PLATEAU' | 'BOOTSTRAP'
    rolling_depth_delta: float = 0.0
    incumbent_avg_depth: float = 1.0
    incumbent_max_depth: int = 1
    latest_completed_gen: int = 0
    ticks_min_gen: int = 0
    d_succ_seeds: list[int] = field(default_factory=list)
    d_succ_run_ids: list[str] = field(default_factory=list)
    protected_branches: list[dict[str, Any]] = field(default_factory=list)
    fail_clusters: list[DeficitCluster] = field(default_factory=list)
    end_reason_histogram: list[tuple[str, int, float]] = field(default_factory=list)
    dormant_branches: list[str] = field(default_factory=list)
    plateau_hints: list[str] = field(default_factory=list)
    policy_loc: int = 0
    skill_loc: int = 0
    skill_count: int = 0
    bloat_warning: bool = False
    skill_bloat_notes: list[str] = field(default_factory=list)
    vital_summary: str = ""
    recent_attempts: list[str] = field(default_factory=list)
    recent_theories: list[str] = field(default_factory=list)
    organic_shards: list[Any] = field(default_factory=list)

    def format_brief(self) -> str:
        lines = [
            f"### Deterministic Telemetry Partition (Gen {self.generation} | Regime: {self.regime})",
            f"- Incumbent Progression: avg_depth={self.incumbent_avg_depth:.2f}, max_depth=Dlvl {self.incumbent_max_depth}, rolling_delta={self.rolling_depth_delta:+.2f}",
            f"- Code Footprint: policy={self.policy_loc} LOC, skills={self.skill_count} module(s) ({self.skill_loc} LOC)"
            + (
                " [DITCH MONOLITH: extract the subsystem you touch into `skills/<domain>.py` so `policy.py` stays <= 250 LOC]"
                if self.bloat_warning
                else ""
            ),
        ]
        if self.skill_bloat_notes:
            lines.append(
                f"- Large Skill Modules (split or keep surgical): {', '.join(self.skill_bloat_notes)}"
            )
        if self.vital_summary:
            lines.append(
                f"- Incumbent Vital & Capability Telemetry: {self.vital_summary}"
            )
        if self.latest_completed_gen > 0:
            ticks_note = (
                f", ticks stored for gen >= {self.ticks_min_gen}"
                if self.ticks_min_gen > 0
                else ""
            )
            lines.append(
                f"- DB Telemetry Window: latest_completed_gen={self.latest_completed_gen} "
                f"(Gen {self.generation} is currently running and NOT yet in `batches`/`runs`{ticks_note})"
            )
        if self.protected_branches:
            prot_str = ", ".join(
                f"{b['fn']} (ochiai={b['ochiai']:.2f}, turns={b['turns']})"
                for b in self.protected_branches[:6]
            )
            lines.append(
                f"- D_succ Protected High-Ochiai Branches (DO NOT regress): {prot_str}"
            )
        if self.dormant_branches:
            lines.append(
                f"- Dormant BT Branches (0 active turns recently — prune or consolidate): {', '.join(self.dormant_branches[:10])}"
            )
        if self.end_reason_histogram:
            hist_str = " | ".join(
                f"`{er}` x{cnt} (D={ad:.1f})"
                for er, cnt, ad in self.end_reason_histogram[:8]
            )
            lines.append(f"- Recent Outcome Distribution (diagnose freely): {hist_str}")
        if self.plateau_hints:
            lines.append(
                "- Near-Miss Stepping Stones (recent frontier challengers; inspect code/diffs to synthesize improvements): "
                + " | ".join(self.plateau_hints[:4])
            )
        if self.recent_attempts:
            lines.append(
                "### Recent Experiment & Rejection Memory (DO NOT Repeat Rejected Mechanisms):"
            )
            for ra in self.recent_attempts[:6]:
                lines.append(f"  * {ra}")
        if self.recent_theories:
            lines.append("### Relevant DuckDB Theory Notes:")
            for rt in self.recent_theories[:3]:
                lines.append(f"  * {rt}")
        if self.fail_clusters:
            lines.append("- Objective Trajectory Outcome Slices (D_fail):")
            for c in self.fail_clusters:
                r_ids = f", run_ids={c.sample_run_ids[:1]}" if c.sample_run_ids else ""
                lines.append(
                    f"  * [{c.cluster_id}] {c.label} (runs={c.run_count}, avg_depth={c.avg_depth:.2f}, avg_turns={c.avg_turns:.0f}{r_ids}): {c.evidence}"
                )
        return "\n".join(lines)


class DeficitAnalyzer:
    """Computes domain-agnostic `DeficitReport` from DuckDB + workspace AST."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def analyze(
        self,
        generation: int,
        current_policy: str = "",
        skills: dict[str, str] | None = None,
    ) -> DeficitReport:
        skills_map = dict(skills or {})
        policy_loc = len(current_policy.splitlines()) if current_policy.strip() else 0
        skill_lens = [len(v.splitlines()) for v in skills_map.values() if v]
        skill_loc = sum(skill_lens)
        max_skill_loc = max(skill_lens) if skill_lens else 0
        skill_count = len(skills_map)
        bloat_warning = (
            (policy_loc >= 220 and skill_count == 0)
            or (policy_loc >= 250)
            or (max_skill_loc >= 300)
            or (policy_loc + skill_loc >= 800)
        )
        skill_bloat_notes = [
            f"{k} ({len(v.splitlines())} LOC)"
            for k, v in sorted(skills_map.items())
            if v and len(v.splitlines()) >= 250
        ]

        report = DeficitReport(
            generation=generation,
            regime="BOOTSTRAP" if not current_policy.strip() else "CLIMBING",
            policy_loc=policy_loc,
            skill_loc=skill_loc,
            skill_count=skill_count,
            bloat_warning=bloat_warning,
            skill_bloat_notes=skill_bloat_notes,
        )

        raw_branches = (
            extract_priority_selector_branches(current_policy)
            if current_policy.strip()
            else []
        )

        if not self.db_path or not Path(self.db_path).exists():
            report.fail_clusters = self._bootstrap_clusters()
            return report

        con = None
        try:
            con = connect_db(self.db_path, read_only=True)
            self._populate_telemetry_window(con, report)
            self._populate_plateau_signal(con, report)
            self._populate_vital_summary(con, report)
            self._populate_recent_attempts(con, report)
            self._populate_d_succ_and_branches(con, report, raw_branches)
            self._populate_d_fail_clusters(con, report)
            self._attach_premortem_and_wiki_context(con, report)
            try:
                th_rows = con.execute(
                    """
                    SELECT generation, agent_role, title, verified_c_rule
                    FROM theories
                    WHERE doc_type != 'META_SYNTHESIS' OR generation > 0
                    ORDER BY
                        CASE WHEN UPPER(COALESCE(doc_type, '')) = 'THEORY' THEN 0 ELSE 1 END,
                        generation DESC,
                        created_at DESC
                    LIMIT 3
                    """
                ).fetchall()
                report.recent_theories = [
                    f"[Gen {tg} | {tr}] {tt}" + (f" ({trule})" if trule else "")
                    for tg, tr, tt, trule in th_rows
                ]
            except Exception:
                pass
        except Exception:
            if not report.fail_clusters:
                report.fail_clusters = self._bootstrap_clusters()
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

        if not report.fail_clusters:
            report.fail_clusters = self._bootstrap_clusters()

        return report

    @staticmethod
    def _bootstrap_clusters() -> list[DeficitCluster]:
        return [
            DeficitCluster(
                cluster_id="bootstrap_survival",
                kind="initial_synthesis",
                label="Initial BehaviorTree survival & progression scaffold",
                evidence="No prior empirical failures recorded yet; build clean modular BehaviorTree Policy.",
            ),
            DeficitCluster(
                cluster_id="bootstrap_exploration",
                kind="initial_synthesis",
                label="Initial spatial exploration & prompt/menu safety",
                evidence="Ensure valid actions, prompt dismissal safety, and positive tile coverage.",
            ),
        ]

    @staticmethod
    def _populate_telemetry_window(con: Any, report: DeficitReport) -> None:
        from lox.db.schema import NON_SHARD_RUNS_SQL

        try:
            row = con.execute(
                f"SELECT COALESCE(MAX(generation), 0) FROM runs WHERE {NON_SHARD_RUNS_SQL}"
            ).fetchone()
            if row and row[0] is not None:
                report.latest_completed_gen = int(row[0])
        except Exception:
            pass
        try:
            t_row = con.execute(
                "SELECT COALESCE(MIN(r.generation), 0) "
                "FROM ticks t JOIN runs r ON t.run_id = r.run_id "
                "WHERE COALESCE(r.is_shard, FALSE) = FALSE AND COALESCE(r.run_id, '') NOT LIKE 'shard_%'"
            ).fetchone()
            if t_row and t_row[0] is not None:
                report.ticks_min_gen = int(t_row[0])
        except Exception:
            pass

    @staticmethod
    def _has_verified_batches(con: Any) -> bool:
        try:
            row = con.execute(
                "SELECT 1 FROM batches WHERE hypothesis_passed = TRUE OR batch_id LIKE '%_inc_%' LIMIT 1"
            ).fetchone()
            return row is not None
        except Exception:
            return False

    @classmethod
    def _valid_runs_clause(cls, con: Any) -> str:
        from lox.db.schema import NON_SHARD_RUNS_SQL

        if cls._has_verified_batches(con):
            return (
                f"generation > 0 AND {NON_SHARD_RUNS_SQL} AND ("
                "batch_id LIKE '%_inc_%' OR EXISTS ("
                "SELECT 1 FROM batches b WHERE b.batch_id = runs.batch_id AND b.hypothesis_passed = TRUE"
                "))"
            )
        return f"generation > 0 AND {NON_SHARD_RUNS_SQL}"

    @classmethod
    def _populate_vital_summary(cls, con: Any, report: DeficitReport) -> None:
        try:
            where_sql = cls._valid_runs_clause(con)
            latest_row = con.execute(
                f"SELECT MAX(generation) FROM runs WHERE {where_sql}"
            ).fetchone()
            lat_gen = int(latest_row[0] or 0) if latest_row else 0
            if lat_gen <= 0:
                return
            row = con.execute(
                f"""
                SELECT COUNT(*),
                       ROUND(AVG(COALESCE(final_ac, 10)), 2),
                       ROUND(AVG(COALESCE(xp_level, 1)), 2),
                       ROUND(AVG(COALESCE(unique_tiles, 0)), 0),
                       ROUND(AVG(COALESCE(pct_wait, 0.0)), 1),
                       ROUND(AVG(COALESCE(pct_search, 0.0)), 1),
                       SUM(CASE WHEN COALESCE(starved_with_food, FALSE) THEN 1 ELSE 0 END),
                       SUM(CASE WHEN COALESCE(died_unarmored, FALSE) THEN 1 ELSE 0 END),
                       SUM(CASE WHEN LOWER(COALESCE(milestones_json, '')) LIKE '%\"unworn_armor_in_pack\": true%' THEN 1 ELSE 0 END),
                       SUM(COALESCE(auto_esc_count, 0)),
                       SUM(COALESCE(policy_exceptions, 0))
                FROM runs
                WHERE {where_sql} AND generation = ?
                """,
                [lat_gen],
            ).fetchone()
            if row and int(row[0] or 0) > 0:
                n_r = int(row[0])
                report.vital_summary = (
                    f"gen={lat_gen} (N={n_r}): avg_ac={row[1]}, avg_xl={row[2]}, "
                    f"avg_tiles={int(row[3] or 0)}, pct_wait={row[4]}%, pct_search={row[5]}%, "
                    f"starved_with_food={int(row[6] or 0)}, unworn_armor_deaths={int(row[7] or 0) or int(row[8] or 0)}, "
                    f"auto_esc={int(row[9] or 0)}, policy_exc={int(row[10] or 0)}"
                )
        except Exception:
            pass

    @staticmethod
    def _populate_recent_attempts(con: Any, report: DeficitReport) -> None:
        attempts: list[str] = []
        min_g = max(1, int(report.generation) - 4)
        # 1. Query exact challenger records from `tournament_attempts` (including rejected peers & tracebacks)
        try:
            ta_rows = con.execute(
                """
                SELECT generation, peer_id, deficit_id, promoted, sandbox_passed,
                       ROUND(avg_depth, 2), max_depth, ROUND(avg_ac, 1),
                       policy_exceptions, seeds_gained_json, seeds_lost_json,
                       top_tracebacks_json, sandbox_error,
                       COALESCE(holdout_rejected, FALSE), holdout_iqm_delta,
                       COALESCE(reject_reason, ''), wins, losses
                FROM tournament_attempts
                WHERE generation >= ? AND generation < ?
                ORDER BY generation DESC, promoted DESC, avg_depth DESC
                LIMIT 8
                """,
                [min_g, int(report.generation)],
            ).fetchall()
            for (
                gen_i,
                pid,
                did,
                promoted,
                sb_ok,
                ad,
                md,
                aac,
                excs,
                sg_json,
                sl_json,
                tb_json,
                sb_err,
                holdout_rej,
                holdout_iqm,
                rej_reason,
                wins_i,
                losses_i,
            ) in ta_rows:
                tag = (
                    "PROMOTED"
                    if promoted
                    else (
                        "HOLD_REJECTED"
                        if holdout_rej
                        else ("CRN_REJECTED" if sb_ok else "SANDBOX_FAIL")
                    )
                )
                gained = json.loads(sg_json) if sg_json else []
                lost = json.loads(sl_json) if sl_json else []
                tbs = json.loads(tb_json) if tb_json else []
                extra = []
                if rej_reason and not promoted:
                    extra.append(f"reason={rej_reason}")
                if wins_i is not None and losses_i is not None:
                    extra.append(f"W/L={int(wins_i)}/{int(losses_i)}")
                if holdout_iqm is not None:
                    extra.append(f"holdout_iqm={float(holdout_iqm):+.2f}")
                if gained:
                    extra.append(f"+seeds={gained[:3]}")
                if lost:
                    extra.append(f"-seeds={lost[:3]}")
                if int(excs or 0) > 0:
                    extra.append(f"exc={excs}")
                if tbs:
                    extra.append(f"tb={str(tbs[0])[:240]}")
                elif sb_err and not sb_ok:
                    extra.append(f"err={str(sb_err)[:240]}")
                ext_s = f" | {'; '.join(extra)}" if extra else ""
                attempts.append(
                    f"Gen {gen_i} [{tag} | {pid}@{did} | D={ad} (Max={md}), AC={aac}]{ext_s}"
                )
        except Exception:
            pass

        if attempts:
            report.recent_attempts = attempts
            return

        # 2. Fallback to `batches` + `synthesis_logs` when `tournament_attempts` is not yet populated
        try:
            b_rows = con.execute(
                """
                SELECT generation, candidate_policy_id, hypothesis_passed,
                       ROUND(avg_depth, 2), max_depth, ROUND(avg_score, 1)
                FROM batches
                WHERE generation >= ? AND generation < ? AND UPPER(mode) = 'SWARM'
                ORDER BY generation DESC, created_at DESC
                LIMIT 5
                """,
                [min_g, int(report.generation)],
            ).fetchall()
            if not b_rows:
                return
            syn_by_gen: dict[int, list[str]] = {}
            try:
                s_rows = con.execute(
                    """
                    SELECT generation, agent_role, hypothesis_yaml
                    FROM synthesis_logs
                    WHERE generation >= ? AND generation < ?
                      AND turn_type = 'SMOKE_CHECK'
                      AND LENGTH(COALESCE(hypothesis_yaml, '')) > 5
                    ORDER BY generation DESC, agent_role, turn_index DESC
                    """,
                    [min_g, int(report.generation)],
                ).fetchall()
                seen_role_gen: set[tuple[int, str]] = set()
                for sg, srole, shyp in s_rows:
                    key = (int(sg), str(srole))
                    if key in seen_role_gen:
                        continue
                    seen_role_gen.add(key)
                    raw_h = str(shyp or "")
                    summary = ""
                    try:
                        import yaml

                        parsed = yaml.safe_load(raw_h)
                        if isinstance(parsed, dict):
                            cl = str(parsed.get("cluster", "") or "").strip()
                            hy = str(
                                parsed.get("hypothesis")
                                or parsed.get("mechanism")
                                or parsed.get("expected_effect")
                                or ""
                            ).strip()
                            prefix = f"[{cl}] " if cl else ""
                            summary = f"{prefix}{hy}".strip()
                    except Exception:
                        summary = ""
                    if not summary:
                        summary = " ".join(raw_h.split())
                    if len(summary) > 110:
                        summary = summary[:107] + "..."
                    syn_by_gen.setdefault(int(sg), []).append(f"{srole}: {summary}")
            except Exception:
                pass
            attempts: list[str] = []
            for gen_i, pid, passed, ad, md, asc in b_rows:
                tag = "PROMOTED" if passed else "REJECTED"
                hyps = "; ".join(syn_by_gen.get(int(gen_i), [])[:3])
                hyp_str = f" | {hyps}" if hyps else ""
                attempts.append(
                    f"Gen {gen_i} [{tag} | {pid} | D={ad} (Max={md}), S={asc}]{hyp_str}"
                )
            report.recent_attempts = attempts
        except Exception:
            pass

    @classmethod
    def _populate_plateau_signal(cls, con: Any, report: DeficitReport) -> None:
        where_extra = (
            " AND (hypothesis_passed = TRUE OR batch_id LIKE '%_inc_%')"
            if cls._has_verified_batches(con)
            else ""
        )
        rows = con.execute(
            f"""
            SELECT generation, avg_depth, max_depth
            FROM batches
            WHERE UPPER(mode) = 'SWARM'{where_extra}
            ORDER BY generation DESC, created_at DESC
            LIMIT 4
            """
        ).fetchall()
        if not rows:
            return
        depths = [float(r[1] or 1.0) for r in rows]
        report.incumbent_avg_depth = depths[0]
        report.incumbent_max_depth = int(rows[0][2] or 1)
        if len(depths) >= 2:
            # Compare latest batch against best of previous window
            prev_best = max(depths[1:])
            delta = depths[0] - prev_best
            report.rolling_depth_delta = round(delta, 4)
            if len(depths) >= 3 and delta < 0.15:
                report.regime = "PLATEAU"
            elif delta < 0.0:
                report.regime = "PLATEAU"
            else:
                report.regime = "CLIMBING"

    @classmethod
    def _populate_d_succ_and_branches(
        cls,
        con: Any,
        report: DeficitReport,
        bt_branches: list[dict[str, str]],
    ) -> None:
        where_runs = cls._valid_runs_clause(con)
        succ_rows = con.execute(
            f"""
            SELECT run_id, seed, max_depth, score, turns
            FROM runs
            WHERE {where_runs}
            ORDER BY is_outlier_good DESC, max_depth DESC, score DESC, turns DESC
            LIMIT 4
            """
        ).fetchall()
        report.d_succ_run_ids = [str(r[0]) for r in succ_rows]
        report.d_succ_seeds = list(
            dict.fromkeys(int(r[1]) for r in succ_rows if r[1] is not None)
        )

        fn_rows = []
        # Scope Ochiai/turns to latest 1-3 promoted SWARM champion batches & exclude chronically stagnating functions
        try:
            fn_rows = con.execute(
                """
                WITH recent_champs AS (
                    SELECT batch_id FROM batches
                    WHERE UPPER(mode) = 'SWARM' AND hypothesis_passed = TRUE
                    ORDER BY generation DESC, created_at DESC
                    LIMIT 3
                )
                SELECT fs.function_name,
                       MAX(COALESCE(fs.ochiai_good_run_score, 0.0) * (1.0 - LEAST(0.85, COALESCE(fs.stagnation_share, 0.0)))) AS net_ochiai,
                       SUM(COALESCE(fs.total_turns_active, 0)) AS sum_turns
                FROM function_stats fs
                JOIN recent_champs rc ON fs.batch_id = rc.batch_id
                GROUP BY fs.function_name
                HAVING AVG(COALESCE(fs.stagnation_share, 0.0)) < 0.35
                ORDER BY net_ochiai DESC, sum_turns DESC
                """
            ).fetchall()
        except Exception:
            fn_rows = []
        if not fn_rows:
            fn_rows = con.execute(
                """
                SELECT function_name,
                       MAX(COALESCE(ochiai_good_run_score, 0.0) * (1.0 - LEAST(0.85, COALESCE(stagnation_share, 0.0)))) AS max_ochiai,
                       SUM(COALESCE(total_turns_active, 0)) AS sum_turns
                FROM function_stats
                GROUP BY function_name
                ORDER BY max_ochiai DESC, sum_turns DESC
                """
            ).fetchall()
        active_turns_by_fn = {str(r[0]): int(r[2] or 0) for r in fn_rows}
        protected: list[dict[str, Any]] = []
        for fn_name, max_och, sum_turns in fn_rows:
            fn_s = str(fn_name or "")
            if not fn_s or fn_s in ("root", "wrapped", "_osc_gate"):
                continue
            if float(max_och or 0.0) >= 0.40 and int(sum_turns or 0) > 0:
                protected.append(
                    {
                        "fn": fn_s,
                        "ochiai": round(float(max_och), 3),
                        "turns": int(sum_turns),
                    }
                )
        report.protected_branches = protected[:8]

        if bt_branches:
            dormant: list[str] = []
            for br in bt_branches:
                if isinstance(br, dict):
                    l_name = str(br.get("leaf_name", "") or "").strip()
                    f_name = str(br.get("fn_name", "") or "").strip()
                    turns_l = active_turns_by_fn.get(l_name, 0) if l_name else 0
                    turns_f = active_turns_by_fn.get(f_name, 0) if f_name else 0
                    label = l_name or f_name
                    if label and turns_l == 0 and turns_f == 0:
                        dormant.append(label)
                else:
                    s_br = str(br or "").strip()
                    if s_br and active_turns_by_fn.get(s_br, 0) == 0:
                        dormant.append(s_br)
            report.dormant_branches = dormant

    @classmethod
    def _populate_d_fail_clusters(cls, con: Any, report: DeficitReport) -> None:
        where_runs = cls._valid_runs_clause(con)
        # Prefer the complete latest generation batch (0 dropped max_steps / high-turn runs!).
        # Fall back to the latest 64 runs across recent generations only when the latest batch is tiny (<16 runs).
        raw_rows = con.execute(
            f"""
            SELECT run_id, seed, depth, max_depth, turns, end_reason,
                   unique_tiles, pct_wait, pct_search,
                   COALESCE(starved_with_food, FALSE) AS starved_with_food,
                   COALESCE(died_unarmored, FALSE) AS died_unarmored,
                   COALESCE(auto_esc_count, 0) AS auto_esc_count,
                   COALESCE(policy_exceptions, 0) AS policy_exceptions,
                   COALESCE(last_active_function, '') AS last_active_fn,
                   COALESCE(summary_digest, '') AS summary_digest,
                   action_bytes,
                   milestones_json,
                   COALESCE(RIGHT(game_log, 600), '') AS game_log_tail
            FROM runs
            WHERE {where_runs}
              AND generation = (SELECT MAX(generation) FROM runs WHERE {where_runs})
            ORDER BY run_id ASC
            """
        ).fetchall()
        if len(raw_rows) < 16:
            raw_rows = con.execute(
                f"""
                SELECT run_id, seed, depth, max_depth, turns, end_reason,
                       unique_tiles, pct_wait, pct_search,
                       COALESCE(starved_with_food, FALSE) AS starved_with_food,
                       COALESCE(died_unarmored, FALSE) AS died_unarmored,
                       COALESCE(auto_esc_count, 0) AS auto_esc_count,
                       COALESCE(policy_exceptions, 0) AS policy_exceptions,
                       COALESCE(last_active_function, '') AS last_active_fn,
                       COALESCE(summary_digest, '') AS summary_digest,
                       action_bytes,
                       milestones_json,
                       COALESCE(RIGHT(game_log, 600), '') AS game_log_tail
                FROM runs
                WHERE {where_runs}
                ORDER BY generation DESC, run_id ASC
                LIMIT 64
                """
            ).fetchall()
        if not raw_rows:
            return
        if len(raw_rows) <= 64:
            rows = list(raw_rows)
        else:
            # Symmetrically stratify across all 4 tails (shortest, longest/max_steps, deepest, median)
            by_turns = sorted(raw_rows, key=lambda x: int(x[4] or 0))
            by_depth = sorted(raw_rows, key=lambda x: int(x[3] or 1), reverse=True)
            mid_start = max(0, (len(by_turns) // 2) - 8)
            picked: dict[str, tuple[Any, ...]] = {}
            for r in (
                by_turns[:16]
                + by_turns[-16:]
                + by_depth[:16]
                + by_turns[mid_start : mid_start + 16]
            ):
                picked[str(r[0])] = r
            for r in raw_rows:
                if len(picked) >= 64:
                    break
                picked.setdefault(str(r[0]), r)
            rows = list(picked.values())

        # Build raw end_reason histogram across recent runs so peers freely diagnose root causes
        end_counts: dict[str, list[float]] = {}
        for r in rows:
            er = str(r[5] or "unknown").strip() or "unknown"
            end_counts.setdefault(er, []).append(float(r[3] or 1.0))
        report.end_reason_histogram = [
            (er, len(ds), sum(ds) / len(ds))
            for er, ds in sorted(
                end_counts.items(), key=lambda kv: len(kv[1]), reverse=True
            )
        ][:10]

        clusters: dict[str, dict[str, Any]] = {}

        def _add_to_cluster(
            cid: str,
            kind: str,
            label: str,
            row: tuple[Any, ...],
            evidence_note: str,
        ) -> None:
            c = clusters.setdefault(
                cid,
                {
                    "kind": kind,
                    "label": label,
                    "rows": [],
                    "fns": {},
                    "notes": [],
                },
            )
            c["rows"].append(row)
            fn = str(row[13] or "")
            if fn:
                c["fns"][fn] = c["fns"].get(fn, 0) + 1
            if evidence_note and evidence_note not in c["notes"]:
                c["notes"].append(evidence_note)

        depths_all = sorted(float(r[3] or 1.0) for r in rows)
        turns_all = sorted(int(r[4] or 0) for r in rows)
        median_turn = turns_all[len(turns_all) // 2] if turns_all else 300
        high_depth_cut = depths_all[int(len(depths_all) * 0.70)] if depths_all else 2.0

        # Domain-agnostic recurring terminal states (zero hardcoded game strings):
        # any specific non-generic end_reason repeating on >= max(4, n//16) runs gets its own clean slice.
        _generic_ends = {
            "",
            "unknown",
            "killed",
            "max_steps",
            "stalled_zero_turn",
            "ascended",
            "escaped",
            "ended",
        }
        recurring_end_min = max(4, len(rows) // 16)
        recurring_end_ranks: dict[str, int] = {}
        for er_label, er_ds in sorted(
            end_counts.items(), key=lambda kv: len(kv[1]), reverse=True
        ):
            er_low = er_label.strip().lower()
            if (
                er_low not in _generic_ends
                and "stalled" not in er_low
                and len(er_ds) >= recurring_end_min
                and len(recurring_end_ranks) < 3
            ):
                recurring_end_ranks[er_low] = len(recurring_end_ranks) + 1

        for row in rows:
            (
                run_id,
                seed,
                depth,
                max_depth,
                turns,
                end_reason,
                unique_tiles,
                pct_wait,
                pct_search,
                _starved,
                _unarmored,
                auto_esc,
                policy_excs,
                last_fn,
                digest,
                _abytes,
                m_json,
                game_log_tail,
            ) = row

            end_str = str(end_reason or "unknown").strip() or "unknown"
            end_low = end_str.lower()
            if end_low in ("ascended", "escaped"):
                continue
            digest_s = str(digest or "")
            gl_lines = [
                ln.strip() for ln in str(game_log_tail or "").splitlines() if ln.strip()
            ]
            gl_tail = " | ".join(gl_lines[-3:])[-160:] if gl_lines else ""
            stall_count = 0
            unworn_armor_pack = False
            if m_json:
                try:
                    m_obj = json.loads(m_json)
                    stall_count = int(m_obj.get("stall_count", 0) or 0)
                    unworn_armor_pack = bool(m_obj.get("unworn_armor_in_pack", False))
                except Exception:
                    stall_count = 0

            # Purely domain-agnostic trajectory-outcome & capability slices (zero NetHack keywords)
            if end_low == "max_steps" and int(max_depth or 1) <= 2:
                _add_to_cluster(
                    "slice_max_steps_confinement",
                    "confinement_horizon",
                    "Horizon limit reached at shallow depth (Dlvl <= 2)",
                    row,
                    f"end=`{end_str}`, turns={turns}, Dlvl={max_depth}, pct_search={float(pct_search or 0):.1f}%, tiles={unique_tiles}"
                    + (f", last_msg=`{gl_tail}`" if gl_tail else ""),
                )
            elif end_low in recurring_end_ranks and int(turns or 0) >= 400:
                r_idx = recurring_end_ranks[end_low]
                _add_to_cluster(
                    f"slice_recurring_outcome_{r_idx}",
                    "recurring_terminal_state",
                    f"Recurring terminal outcome: `{end_str[:64]}`",
                    row,
                    f"end=`{end_str}`, Dlvl={max_depth}, turns={turns}, pct_search={float(pct_search or 0):.1f}%, tiles={unique_tiles}"
                    + (f", last_msg=`{gl_tail}`" if gl_tail else ""),
                )
            elif (float(pct_search or 0.0) >= 35.0 and int(turns or 0) >= 1500) or (
                end_low == "max_steps" and int(turns or 0) >= 2000
            ):
                _add_to_cluster(
                    "slice_high_search_or_horizon_burn",
                    "search_or_horizon_loop",
                    "High search-action share or step-horizon burn without advancing floors",
                    row,
                    f"end=`{end_str}`, Dlvl={max_depth}, turns={turns}, pct_search={float(pct_search or 0):.1f}%, tiles={unique_tiles}"
                    + (f", last_msg=`{gl_tail}`" if gl_tail else ""),
                )
            elif (
                stall_count >= 25
                or "stalled" in end_low
                or "max_steps" in end_low
                or (float(pct_wait or 0.0) + float(pct_search or 0.0) >= 65.0)
                or (
                    "CONFINEMENT_STALL" in digest_s
                    and int(max_depth or 1) <= 2
                    and (
                        float(pct_wait or 0.0) + float(pct_search or 0.0) >= 35.0
                        or int(unique_tiles or 0) <= 120
                    )
                )
                or (
                    int(turns or 0) >= 400
                    and int(max_depth or 1) <= 1
                    and int(unique_tiles or 0) <= 60
                )
            ):
                _add_to_cluster(
                    "slice_stalled",
                    "stalled_trajectory",
                    f"Stalled / low-progress trajectories (Dlvl {int(max_depth or 1)})",
                    row,
                    f"end=`{end_str}`, tiles={unique_tiles}, pct_wait={float(pct_wait or 0):.1f}%"
                    + (f", last_msg=`{gl_tail}`" if gl_tail else ""),
                )
            elif int(policy_excs or 0) > 0 or int(auto_esc or 0) > 2:
                _add_to_cluster(
                    "slice_safety_traps",
                    "runtime_or_prompt_trap",
                    "Runtime exceptions or unhandled prompt traps",
                    row,
                    f"policy_exceptions={policy_excs}, auto_esc={auto_esc}"
                    + (f", last_msg=`{gl_tail}`" if gl_tail else ""),
                )
            elif bool(_unarmored) or unworn_armor_pack:
                _add_to_cluster(
                    "slice_unequipped_progression",
                    "capability_gap",
                    "Terminated carrying unequipped gear / stagnant protection",
                    row,
                    f"end=`{end_str}`, Dlvl={max_depth}, unworn_armor={bool(_unarmored) or unworn_armor_pack}",
                )
            elif float(max_depth or 1) >= max(2.0, high_depth_cut):
                _add_to_cluster(
                    "slice_frontier_failures",
                    "frontier_trajectory",
                    f"Deep-frontier terminations (Dlvl >= {int(max(2.0, high_depth_cut))})",
                    row,
                    f"end=`{end_str}`, turns={turns}",
                )
            elif int(turns or 0) <= max(250, int(median_turn)):
                _add_to_cluster(
                    "slice_early_terminations",
                    "early_trajectory",
                    "Short-horizon terminations before mid-run progression",
                    row,
                    f"end=`{end_str}`, turns={turns}, Dlvl={max_depth}",
                )
            else:
                _add_to_cluster(
                    "slice_median_outcomes",
                    "median_trajectory",
                    "Mid-horizon terminations (diagnose latent bottlenecks)",
                    row,
                    f"end=`{end_str}`, turns={turns}, Dlvl={max_depth}",
                )

        def _build_cluster(
            cid: str,
            kind: str,
            label: str,
            c_rows: list[tuple[Any, ...]],
            dom_fns: list[str],
            notes: list[str],
        ) -> DeficitCluster:
            avg_d = sum(float(r[3] or 1.0) for r in c_rows) / len(c_rows)
            avg_t = sum(float(r[4] or 0.0) for r in c_rows) / len(c_rows)
            fixtures: list[dict[str, Any]] = []
            for r in c_rows[:2]:
                ab = bytes(r[15] or b"") if r[15] is not None else b""
                if len(ab) > 8:
                    stop_idx = (
                        max(1, len(ab) - 35) if len(ab) > 45 else max(1, len(ab) - 10)
                    )
                    fixtures.append(
                        {
                            "run_id": str(r[0]),
                            "seed": int(r[1]),
                            "depth": int(r[2] or 1),
                            "end_reason": str(r[5] or ""),
                            "action_bytes": ab,
                            "stop_at_byte_idx": stop_idx,
                        }
                    )
            return DeficitCluster(
                cluster_id=cid,
                kind=kind,
                label=label,
                run_count=len(c_rows),
                avg_depth=avg_d,
                avg_turns=avg_t,
                sample_run_ids=[str(r[0]) for r in c_rows[:4]],
                sample_seeds=list(
                    dict.fromkeys(int(r[1]) for r in c_rows[:4] if r[1] is not None)
                ),
                dominant_fns=dom_fns[:5],
                replay_fixtures=fixtures,
                evidence="; ".join(notes[:2]),
            )

        built: list[DeficitCluster] = []
        for cid, data in sorted(
            clusters.items(),
            key=lambda kv: (
                len(kv[1]["rows"])
                * (
                    1.5
                    if kv[1]["kind"]
                    in ("confinement_horizon", "search_or_horizon_loop")
                    else (1.2 if kv[1]["kind"] == "recurring_terminal_state" else 1.0)
                ),
                len(kv[1]["rows"]),
            ),
            reverse=True,
        ):
            dom_fns = [
                fn
                for fn, _ in sorted(
                    data["fns"].items(), key=lambda kv: kv[1], reverse=True
                )
            ]
            built.append(
                _build_cluster(
                    cid,
                    data["kind"],
                    data["label"],
                    data["rows"],
                    dom_fns,
                    data["notes"],
                )
            )

        # Sub-split oversized failure clusters (> 8 runs) or narrow cluster pools by objective
        # outcome signature (`end_reason` / stall / depth band) rather than `last_active_function`
        # to avoid blaming symptom dispatchers (e.g. combat vs prompt handoff bugs).
        def _outcome_subbucket(row: tuple[Any, ...]) -> tuple[str, str]:
            end_s = str(row[5] or "unknown").strip().lower()
            max_d = int(row[3] or 1)
            auto_e = int(row[11] or 0)
            excs = int(row[12] or 0)
            gl_tail = str(row[17] or "").lower()
            if excs > 0:
                return "exceptions", "runtime exceptions"
            if (
                auto_e > 0
                or "[" in end_s
                or "]" in end_s
                or "?" in end_s
                or "[yn" in gl_tail
            ):
                return "prompt_interaction", "interactive prompt/direction loop"
            if "stalled" in end_s:
                band = "dlvl1" if max_d <= 1 else "dlvl2plus"
                return (
                    f"zero_turn_{band}",
                    f"zero-turn stall (Dlvl {'>=2' if max_d > 1 else '1'})",
                )
            if "max_steps" in end_s:
                return "step_horizon", "step horizon exhaustion"
            if max_d <= 1:
                return "early_dlvl1", "early Dlvl-1 termination"
            return "deep_dlvl2plus", "Dlvl-2+ termination"

        if built:
            expanded: list[DeficitCluster] = list(built)
            split_targets = [c for c in built if c.run_count > 8] or (
                [built[0]] if len(built) < 4 else []
            )
            for target_c in split_targets:
                if len(expanded) >= 6:
                    break
                cid_src = target_c.cluster_id
                if cid_src not in clusters:
                    continue
                by_outcome: dict[str, tuple[str, list[tuple[Any, ...]]]] = {}
                for r in clusters[cid_src]["rows"]:
                    o_key, o_desc = _outcome_subbucket(r)
                    if o_key not in by_outcome:
                        by_outcome[o_key] = (o_desc, [])
                    by_outcome[o_key][1].append(r)
                if len(by_outcome) >= 2:
                    ordered_outcomes = sorted(
                        by_outcome.items(),
                        key=lambda kv: len(kv[1][1]),
                        reverse=True,
                    )
                    for o_key, (o_desc, sub_rows) in ordered_outcomes[1:]:
                        if len(expanded) >= 6 or len(sub_rows) < 2:
                            break
                        sub_cid = f"{cid_src}__{o_key}"
                        if any(ec.cluster_id == sub_cid for ec in expanded):
                            continue
                        sub_notes = list(
                            dict.fromkeys(
                                f"end=`{str(r[5] or 'unknown').strip()}`, tiles={r[6]}"
                                for r in sub_rows[:2]
                            )
                        )
                        expanded.append(
                            _build_cluster(
                                sub_cid,
                                f"{clusters[cid_src]['kind']}:{o_key}",
                                f"{clusters[cid_src]['label']} — {o_desc}",
                                sub_rows,
                                [],
                                sub_notes,
                            )
                        )
            built = expanded

        report.fail_clusters = built[:6]

    @staticmethod
    def _attach_premortem_and_wiki_context(con: Any, report: DeficitReport) -> None:
        """Attaches pre-mortem tick slices and dynamic Wiki excerpts to top failure clusters."""
        if not report.fail_clusters:
            return
        wiki_engine = None
        try:
            from lox.datasets.wiki import DEFAULT_WIKI_DB, WikiEngine

            if os.path.exists(DEFAULT_WIKI_DB):
                wiki_engine = WikiEngine(DEFAULT_WIKI_DB)
        except Exception:
            wiki_engine = None

        for cl in report.fail_clusters[:2]:
            extras: list[str] = []
            if cl.sample_run_ids:
                rid = str(cl.sample_run_ids[0])
                try:
                    t_rows = con.execute(
                        """
                        SELECT step_idx, depth, hp, max_hp, action_name, active_function, message
                        FROM ticks
                        WHERE run_id = ?
                        ORDER BY step_idx DESC
                        LIMIT 5
                        """,
                        [rid],
                    ).fetchall()
                    if t_rows:
                        rev = list(reversed(t_rows))
                        seq_str = " -> ".join(
                            f"t{tr[0]}(D{tr[1]} HP{tr[2]}/{tr[3]} {tr[5]}:{tr[4]}"
                            + (
                                f" msg={str(tr[6])[:40]!r})"
                                if str(tr[6] or "").strip()
                                else ")"
                            )
                            for tr in rev
                        )
                        extras.append(f"pre_mortem=[{seq_str}]")
                        if wiki_engine is not None:
                            last_msg = next(
                                (
                                    str(tr[6] or "").strip()
                                    for tr in t_rows
                                    if str(tr[6] or "").strip()
                                ),
                                "",
                            )
                            if last_msg:
                                hits = wiki_engine.search_wiki(last_msg, top_k=1)
                                if hits:
                                    snip = (
                                        str(hits[0].get("content", "") or "")
                                        .replace("\n", " ")[:180]
                                        .strip()
                                    )
                                    if snip:
                                        extras.append(
                                            f"wiki[{hits[0].get('title', '')}]: {snip}"
                                        )
                except Exception:
                    pass
            if extras:
                cl.evidence = (
                    f"{cl.evidence}; " + "; ".join(extras)
                    if cl.evidence
                    else "; ".join(extras)
                )
