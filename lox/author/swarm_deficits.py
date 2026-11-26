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
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lox.author.directives import extract_policy_branches
from lox.db.schema import connect_db

_LOG = logging.getLogger(__name__)


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
    efficiency_summary: str = ""
    expert_parity_gaps: str = ""
    plateau_streak: int = 0
    recent_attempts: list[str] = field(default_factory=list)
    recent_theories: list[str] = field(default_factory=list)
    organic_shards: list[Any] = field(default_factory=list)
    cooled_down_deficits: dict[str, int] = field(default_factory=dict)

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
        if self.efficiency_summary:
            lines.append(
                f"- Floor Residence & Action Concentration Telemetry: {self.efficiency_summary}"
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
                f"- Dormant Branches (0 active turns recently — prune or consolidate): {', '.join(self.dormant_branches[:10])}"
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
            for ra in self.recent_attempts[:8]:
                lines.append(f"  * {ra}")
        if self.recent_theories:
            lines.append("### Relevant DuckDB Theory Notes:")
            for rt in self.recent_theories[:3]:
                lines.append(f"  * {rt}")
        if self.fail_clusters:
            lines.append("- Objective Trajectory Outcome Slices (D_fail):")
            for c in self.fail_clusters:
                r_ids = f", run_ids={c.sample_run_ids[:1]}" if c.sample_run_ids else ""
                cd_n = int((self.cooled_down_deficits or {}).get(c.cluster_id, 0) or 0)
                cd_tag = (
                    f" [COOLDOWN: failed {cd_n}x in last 4 gens — prefer untouched slices]"
                    if cd_n >= 2
                    else ""
                )
                lines.append(
                    f"  * [{c.cluster_id}] {c.label}{cd_tag} (runs={c.run_count}, avg_depth={c.avg_depth:.2f}, avg_turns={c.avg_turns:.0f}{r_ids}): {c.evidence}"
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
            extract_policy_branches(current_policy, skills_map)
            if current_policy.strip() or skills_map
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
            self._populate_efficiency_telemetry(con, report)
            self._populate_expert_parity_gaps(con, report)
            self._populate_recent_attempts(con, report)
            self._populate_d_succ_and_branches(con, report, raw_branches)
            self._populate_d_fail_clusters(con, report)
            # Exclude guard seeds that overlap target cluster sample_seeds (preserving up to 4 clean guards)
            if report.d_succ_seeds and report.fail_clusters:
                target_seeds_union = {
                    int(sd)
                    for cl in report.fail_clusters
                    for sd in (getattr(cl, "sample_seeds", []) or [])
                }
                non_overlap = [
                    sd
                    for sd in report.d_succ_seeds
                    if int(sd) not in target_seeds_union
                ]
                if non_overlap:
                    report.d_succ_seeds = non_overlap[:4]
                else:
                    report.d_succ_seeds = report.d_succ_seeds[:4]
            elif report.d_succ_seeds:
                report.d_succ_seeds = report.d_succ_seeds[:4]
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
            except Exception as exc:
                _LOG.warning(
                    "Failed querying recent theories in DeficitAnalyzer: %s", exc
                )
        except Exception as exc:
            _LOG.warning("DeficitAnalyzer DB analysis error: %s", exc)
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
                label="Initial Option-A survival & progression scaffold",
                evidence="No prior empirical failures recorded yet; build clean modular Option-A Policy.",
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
        except Exception as exc:
            _LOG.warning("Failed querying latest_completed_gen: %s", exc)
        try:
            t_row = con.execute(
                "SELECT COALESCE(MIN(r.generation), 0) "
                "FROM ticks t JOIN runs r ON t.run_id = r.run_id "
                "WHERE COALESCE(r.is_shard, FALSE) = FALSE "
                "AND COALESCE(r.run_id, '') NOT LIKE 'shard_%' "
                "AND COALESCE(r.run_kind, 'champion') = 'champion'"
            ).fetchone()
            if t_row and t_row[0] is not None:
                report.ticks_min_gen = int(t_row[0])
        except Exception as exc:
            _LOG.warning("Failed querying ticks_min_gen: %s", exc)

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
                       SUM(CASE WHEN COALESCE(died_unarmored, FALSE) OR LOWER(COALESCE(milestones_json, '')) LIKE '%\"unworn_armor_in_pack\": true%' THEN 1 ELSE 0 END),
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
                    f"starved_with_food={int(row[6] or 0)}, unworn_armor_deaths={int(row[7] or 0)}, "
                    f"auto_esc={int(row[8] or 0)}, policy_exc={int(row[9] or 0)}"
                )
        except Exception as exc:
            _LOG.warning("Failed populating vital_summary: %s", exc)

    @classmethod
    def _populate_efficiency_telemetry(cls, con: Any, report: DeficitReport) -> None:
        """
        Domain-agnostic per-depth residence & action-concentration summary from `ticks`
        (and optional `expert_ticks` residence parity when available). Zero hardcoded action names.
        """
        try:
            where_sql = cls._valid_runs_clause(con)
            latest_row = con.execute(
                f"SELECT MAX(generation) FROM runs WHERE {where_sql}"
            ).fetchone()
            lat_gen = int(latest_row[0] or 0) if latest_row else 0
            if lat_gen <= 0:
                return

            depth_rows = con.execute(
                f"""
                WITH gen_runs AS (
                    SELECT run_id FROM runs
                    WHERE {where_sql} AND generation = ?
                ),
                per_run_depth AS (
                    SELECT t.run_id, t.depth, COUNT(*) AS n_ticks
                    FROM ticks t
                    JOIN gen_runs r ON t.run_id = r.run_id
                    WHERE t.depth IS NOT NULL AND t.depth BETWEEN 1 AND 5
                    GROUP BY t.run_id, t.depth
                )
                SELECT depth, COUNT(*) AS n_runs, ROUND(AVG(n_ticks), 0) AS avg_ticks
                FROM per_run_depth
                GROUP BY depth
                ORDER BY depth ASC
                LIMIT 4
                """,
                [lat_gen],
            ).fetchall()
            if not depth_rows:
                return

            act_rows = con.execute(
                f"""
                WITH gen_runs AS (
                    SELECT run_id FROM runs
                    WHERE {where_sql} AND generation = ?
                ),
                counts AS (
                    SELECT t.depth, COALESCE(t.action_name, 'UNKNOWN') AS act, COUNT(*) AS cnt
                    FROM ticks t
                    JOIN gen_runs r ON t.run_id = r.run_id
                    WHERE t.depth IS NOT NULL AND t.depth BETWEEN 1 AND 5
                    GROUP BY t.depth, act
                ),
                totals AS (
                    SELECT depth, SUM(cnt) AS tot FROM counts GROUP BY depth
                ),
                ranked AS (
                    SELECT c.depth, c.act, 100.0 * c.cnt / NULLIF(t.tot, 0) AS pct,
                           ROW_NUMBER() OVER (PARTITION BY c.depth ORDER BY c.cnt DESC) AS rn
                    FROM counts c
                    JOIN totals t ON c.depth = t.depth
                )
                SELECT depth, act, ROUND(pct, 1)
                FROM ranked
                WHERE rn <= 2
                ORDER BY depth ASC, rn ASC
                """,
                [lat_gen],
            ).fetchall()
            top_acts: dict[int, list[str]] = {}
            for d_val, act_nm, pct_val in act_rows:
                top_acts.setdefault(int(d_val), []).append(
                    f"{act_nm} {float(pct_val or 0.0):.1f}%"
                )

            expert_res: dict[int, float] = {}
            try:
                cov_row = con.execute(
                    """
                    WITH per_dep AS (
                        SELECT run_id,
                               depth,
                               MAX(turn) - MIN(turn) AS dspan,
                               MAX(turn) AS max_t,
                               MIN(turn) AS min_t
                        FROM expert_ticks
                        WHERE depth IS NOT NULL AND turn IS NOT NULL
                        GROUP BY run_id, depth
                    ),
                    per_run AS (
                        SELECT run_id,
                               SUM(dspan) AS sum_dspan,
                               MAX(max_t) - MIN(min_t) + 1 AS total_span
                        FROM per_dep
                        GROUP BY run_id
                    )
                    SELECT AVG(CAST(sum_dspan AS DOUBLE) / GREATEST(total_span, 1))
                    FROM per_run
                    """
                ).fetchone()
                exp_span_cov = (
                    float(cov_row[0]) if (cov_row and cov_row[0] is not None) else 0.0
                )
                if exp_span_cov >= 0.25:
                    exp_rows = con.execute(
                        """
                        WITH per_exp AS (
                            SELECT run_id, depth, MAX(turn) - MIN(turn) AS span
                            FROM expert_ticks
                            WHERE depth IS NOT NULL AND depth BETWEEN 1 AND 5
                            GROUP BY run_id, depth
                        )
                        SELECT depth, ROUND(AVG(span), 0)
                        FROM per_exp
                        GROUP BY depth
                        """
                    ).fetchall()
                    for ed, es in exp_rows:
                        if es is not None and float(es) > 0:
                            expert_res[int(ed)] = float(es)
            except Exception as exp_exc:
                _LOG.warning("Failed querying expert residence spans: %s", exp_exc)

            parts: list[str] = []
            for d_val, n_r, avg_t in depth_rows:
                d_int = int(d_val)
                acts_str = ", ".join(top_acts.get(d_int, []))
                exp_note = (
                    f" vs expert ~{int(expert_res[d_int])}t"
                    if d_int in expert_res
                    else ""
                )
                act_part = f" [top: {acts_str}]" if acts_str else ""
                parts.append(
                    f"D{d_int} (n={int(n_r)}): {int(avg_t or 0)}t/run{act_part}{exp_note}"
                )
            if parts:
                report.efficiency_summary = " | ".join(parts)
        except Exception as exc:
            _LOG.warning("Failed populating efficiency_summary: %s", exc)

    @classmethod
    def _populate_expert_parity_gaps(cls, con: Any, report: DeficitReport) -> None:
        """
        Domain-agnostic Champion-vs-Expert capability & efficiency parity digest
        computed purely via SQL across `Dlvl 1 .. min(10, p90_depth + 2)`.
        Zero hardcoded game actions or strategies.
        """
        try:
            where_sql = cls._valid_runs_clause(con)
            latest_row = con.execute(
                f"SELECT MAX(generation) FROM runs WHERE {where_sql}"
            ).fetchone()
            lat_gen = int(latest_row[0] or 0) if latest_row else 0
            if lat_gen <= 0:
                return

            p90_row = con.execute(
                f"""
                SELECT QUANTILE_CONT(max_depth, 0.90),
                       ROUND(AVG(COALESCE(final_ac, 10)), 2),
                       ROUND(AVG(COALESCE(xp_level, 1)), 2),
                       ROUND(AVG(COALESCE(dungeons_visited, 1)), 2)
                FROM runs
                WHERE {where_sql} AND generation = ?
                """,
                [lat_gen],
            ).fetchone()
            p90_d = float(p90_row[0]) if (p90_row and p90_row[0] is not None) else 2.0
            c_ac = float(p90_row[1]) if (p90_row and p90_row[1] is not None) else 10.0
            c_xl = float(p90_row[2]) if (p90_row and p90_row[2] is not None) else 1.0
            c_br = float(p90_row[3]) if (p90_row and p90_row[3] is not None) else 1.0
            tier_max = min(10, max(3, int(round(p90_d)) + 2))

            exp_cnt_row = con.execute("SELECT COUNT(*) FROM expert_ticks").fetchone()
            if not exp_cnt_row or int(exp_cnt_row[0] or 0) == 0:
                return

            dens_row = con.execute(
                """
                SELECT AVG(cnt * 1.0 / NULLIF(max_t - min_t + 1, 0))
                FROM (
                    SELECT run_id, COUNT(*) AS cnt, MIN(turn) AS min_t, MAX(turn) AS max_t
                    FROM expert_ticks
                    GROUP BY run_id
                )
                """
            ).fetchone()
            exp_density = (
                float(dens_row[0])
                if (dens_row is not None and dens_row[0] is not None)
                else 1.0
            )
            is_sparse_milestones = exp_density < 0.25

            _norm_act_expr = (
                "CASE WHEN action_name LIKE 'seq(%' "
                "THEN RTRIM(SPLIT_PART(SUBSTR(action_name, 5), ',', 1), ')') "
                "ELSE COALESCE(action_name, '') END"
            )
            exp_acts = con.execute(
                f"""
                WITH filtered AS (
                    SELECT {_norm_act_expr} AS act
                    FROM expert_ticks
                    WHERE depth BETWEEN 1 AND ?
                      AND COALESCE(action_name, '') != ''
                ),
                tot AS (SELECT COUNT(*) AS n FROM filtered)
                SELECT f.act, 100.0 * COUNT(*) / NULLIF(t.n, 0) AS pct
                FROM filtered f, tot t
                GROUP BY f.act, t.n
                HAVING COUNT(*) >= 5
                ORDER BY pct DESC
                """,
                [tier_max],
            ).fetchall()

            _norm_t_act_expr = (
                "CASE WHEN t.action_name LIKE 'seq(%' "
                "THEN RTRIM(SPLIT_PART(SUBSTR(t.action_name, 5), ',', 1), ')') "
                "ELSE COALESCE(t.action_name, '') END"
            )
            champ_acts = con.execute(
                f"""
                WITH gen_runs AS (
                    SELECT run_id FROM runs WHERE {where_sql} AND generation = ?
                ),
                filtered AS (
                    SELECT {_norm_t_act_expr} AS act
                    FROM ticks t
                    JOIN gen_runs r ON t.run_id = r.run_id
                    WHERE t.depth BETWEEN 1 AND ?
                      AND COALESCE(t.action_name, '') != ''
                ),
                tot AS (SELECT COUNT(*) AS n FROM filtered)
                SELECT f.act, 100.0 * COUNT(*) / NULLIF(t.n, 0) AS pct
                FROM filtered f, tot t
                GROUP BY f.act, t.n
                ORDER BY pct DESC
                """,
                [lat_gen, tier_max],
            ).fetchall()

            exp_map = {str(a): float(p or 0.0) for a, p in exp_acts}
            champ_map = {str(a): float(p or 0.0) for a, p in champ_acts}

            missing_gaps: list[str] = []
            excess_gaps: list[str] = []
            unused_verbs: list[str] = []
            if is_sparse_milestones:
                _cardinal = {"N", "S", "E", "W", "NE", "NW", "SE", "SW"}
                for act in sorted(exp_map.keys()):
                    if act in _cardinal:
                        continue
                    cpct = champ_map.get(act, 0.0)
                    if cpct < 0.005:
                        unused_verbs.append(f"`{act}` (champ {cpct:.4f}%)")
            else:
                for act, epct in sorted(
                    exp_map.items(), key=lambda kv: kv[1], reverse=True
                ):
                    if epct < 0.5:
                        continue
                    cpct = champ_map.get(act, 0.0)
                    if cpct <= 0.01 or cpct < 0.2 * epct:
                        missing_gaps.append(
                            f"`{act}` (expert {epct:.1f}% vs champ {cpct:.2f}%)"
                        )
                    if len(missing_gaps) >= 8:
                        break

                for act, cpct in sorted(
                    champ_map.items(), key=lambda kv: kv[1], reverse=True
                ):
                    if cpct < 5.0:
                        continue
                    epct = exp_map.get(act, 0.0)
                    if cpct > max(5.0, 3.0 * epct):
                        excess_gaps.append(
                            f"`{act}` (champ {cpct:.1f}% vs expert {epct:.1f}%)"
                        )
                    if len(excess_gaps) >= 4:
                        break

            exp_st_row = con.execute(
                """
                SELECT ROUND(AVG(ac), 2), ROUND(AVG(xp_level), 2)
                FROM expert_ticks
                WHERE depth BETWEEN 1 AND ?
                """,
                [tier_max],
            ).fetchone()
            exp_pace_row = (
                None
                if is_sparse_milestones
                else con.execute(
                    """
                    SELECT ROUND(AVG(span_t), 0)
                    FROM (
                        SELECT run_id, depth, MAX(turn) - MIN(turn) + 1 AS span_t
                        FROM expert_ticks
                        WHERE depth BETWEEN 1 AND ?
                        GROUP BY run_id, depth
                    )
                    """,
                    [tier_max],
                ).fetchone()
            )
            champ_pace_row = con.execute(
                f"""
                WITH gen_runs AS (
                    SELECT run_id FROM runs WHERE {where_sql} AND generation = ?
                )
                SELECT ROUND(AVG(span_t), 0)
                FROM (
                    SELECT t.run_id, t.depth, MAX(t.turn) - MIN(t.turn) + 1 AS span_t
                    FROM ticks t
                    JOIN gen_runs r ON t.run_id = r.run_id
                    WHERE t.depth BETWEEN 1 AND ?
                    GROUP BY t.run_id, t.depth
                )
                """,
                [lat_gen, tier_max],
            ).fetchone()
            parts: list[str] = [
                f"Scope: Dlvl 1..{tier_max} (p90_depth={p90_d:.1f} + 2)"
            ]
            if exp_st_row and exp_st_row[0] is not None:
                pace_s = ""
                if champ_pace_row and champ_pace_row[0] is not None:
                    if exp_pace_row and exp_pace_row[0] is not None:
                        pace_s = f", turns/floor={int(champ_pace_row[0])}t vs expert {int(exp_pace_row[0])}t"
                    else:
                        pace_s = f", turns/floor={int(champ_pace_row[0])}t"
                parts.append(
                    f"State Parity: champ(AC={c_ac:.1f}, XL={c_xl:.1f}, branches={c_br:.2f}{pace_s}) "
                    f"vs expert@D1..{tier_max}(AC={float(exp_st_row[0]):.1f}, XL={float(exp_st_row[1] or 1.0):.1f})"
                )
            if is_sparse_milestones:
                if unused_verbs:
                    parts.append(
                        "Near-Zero Champion Action Verbs: "
                        + ", ".join(unused_verbs[:8])
                    )
                parts.append(
                    f"Expert Data Note: `expert_ticks` are sparse progression milestones (density={exp_density * 100:.1f}%), "
                    "NOT per-turn action frequencies—do NOT chase action-share % or synthetic turns/floor ratios; "
                    "diagnose real gaps from `champion_runs`/`champion_ticks` and NetHack Wiki rules."
                )
            else:
                if missing_gaps:
                    parts.append(
                        "Under-Used/Missing Expert Actions: " + ", ".join(missing_gaps)
                    )
                if excess_gaps:
                    parts.append(
                        "Over-Concentrated Champion Actions: " + ", ".join(excess_gaps)
                    )
            report.expert_parity_gaps = "\n- ".join(parts)
        except Exception as exc:
            _LOG.warning("Failed computing expert_parity_gaps: %s", exc)

    @staticmethod
    def _populate_recent_attempts(con: Any, report: DeficitReport) -> None:
        attempts: list[str] = []
        min_g = max(1, int(report.generation) - 12)
        # Track recently rejected deficit_ids since last promotion (4-gen window) for rotation cooldown
        try:
            lp_row = con.execute(
                "SELECT COALESCE(MAX(generation), 0) FROM batches WHERE hypothesis_passed = TRUE AND generation < ?",
                [int(report.generation)],
            ).fetchone()
            last_prom_g = int(lp_row[0] or 0) if lp_row else 0
            cd_min_g = max(last_prom_g + 1, int(report.generation) - 4, 1)
            cd_rows = con.execute(
                """
                SELECT deficit_id, COUNT(*) AS fails
                FROM tournament_attempts
                WHERE generation >= ? AND generation < ?
                  AND COALESCE(promoted, FALSE) = FALSE
                  AND COALESCE(deficit_id, '') != ''
                GROUP BY deficit_id
                HAVING COUNT(*) >= 2
                """,
                [cd_min_g, int(report.generation)],
            ).fetchall()
            _skip_cd = {
                "proactive_expert_capability",
                "proactive_sheltered_branch",
            }
            report.cooled_down_deficits = {
                str(did): int(cnt)
                for did, cnt in cd_rows
                if str(did) not in _skip_cd and not str(did).startswith("revision_")
            }
        except Exception as exc:
            _LOG.warning("Failed computing cooled_down_deficits: %s", exc)
        # 1. Query exact challenger records from `tournament_attempts` (including rejected peers & tracebacks)
        try:
            mech_by_gen_peer: dict[tuple[int, str], str] = {}
            try:
                th_rows = con.execute(
                    """
                    SELECT generation, agent_role, content
                    FROM theories
                    WHERE generation >= ? AND generation < ?
                      AND agent_role != 'referee'
                    ORDER BY created_at DESC
                    """,
                    [min_g, int(report.generation)],
                ).fetchall()
                for tg, trole, tcontent in th_rows:
                    key = (int(tg), str(trole or ""))
                    if key in mech_by_gen_peer:
                        continue
                    raw_c = str(tcontent or "")
                    mech_snip = ""
                    m_idx = raw_c.find("Mechanism:")
                    if m_idx >= 0:
                        tail = raw_c[m_idx + len("Mechanism:") :].strip()
                        for stop_tok in ("\nC/Wiki Rule:", "\nTouched Branches:"):
                            s_i = tail.find(stop_tok)
                            if s_i >= 0:
                                tail = tail[:s_i]
                        mech_snip = " ".join(tail.split())[:140]
                    if mech_snip:
                        mech_by_gen_peer[key] = mech_snip
            except Exception as exc:
                _LOG.warning("Failed loading theory mechanism snippets: %s", exc)

            ta_cols = {
                str(r[1])
                for r in con.execute(
                    "PRAGMA table_info('tournament_attempts')"
                ).fetchall()
            }
            wd_expr = "winsor_delta" if "winsor_delta" in ta_cols else "NULL"
            sp_expr = "sign_perm_p" if "sign_perm_p" in ta_cols else "NULL"
            ta_rows = con.execute(
                f"""
                SELECT generation, peer_id, deficit_id, promoted, sandbox_passed,
                       ROUND(avg_depth, 2), max_depth, ROUND(avg_ac, 1),
                       policy_exceptions, seeds_gained_json, seeds_lost_json,
                       top_tracebacks_json, sandbox_error,
                       COALESCE(holdout_rejected, FALSE), holdout_iqm_delta,
                       COALESCE(reject_reason, ''), wins, losses,
                       {wd_expr}, {sp_expr}
                FROM tournament_attempts
                WHERE generation >= ? AND generation < ?
                ORDER BY generation DESC, promoted DESC, avg_depth DESC
                LIMIT 16
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
                winsor_d,
                sign_p,
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
                if winsor_d is not None:
                    extra.append(f"WΔ={float(winsor_d):+.2f}")
                if sign_p is not None:
                    extra.append(f"p={float(sign_p):.3f}")
                if holdout_iqm is not None:
                    extra.append(f"holdout_iqm={float(holdout_iqm):+.2f}")
                if gained:
                    extra.append(f"+seeds={gained[:3]}")
                if lost:
                    extra.append(f"-seeds={lost[:3]}")
                if int(excs or 0) > 0:
                    extra.append(f"exc={excs}")
                if tbs:
                    extra.append(f"tb={str(tbs[0])[:150]}")
                elif sb_err and not sb_ok:
                    extra.append(f"err={str(sb_err)[:150]}")
                mech_txt = mech_by_gen_peer.get((int(gen_i), str(pid or "")), "")
                if mech_txt and not promoted:
                    extra.append(f"mech={mech_txt}")
                ext_s = f" | {'; '.join(extra)}" if extra else ""
                attempts.append(
                    f"Gen {gen_i} [{tag} | {pid}@{did} | D={ad} (Max={md}), AC={aac}]{ext_s}"
                )
        except Exception as exc:
            _LOG.warning("Failed populating recent tournament_attempts: %s", exc)

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
            except Exception as exc:
                _LOG.warning("Failed querying synthesis_logs summaries: %s", exc)
            attempts: list[str] = []
            for gen_i, pid, passed, ad, md, asc in b_rows:
                tag = "PROMOTED" if passed else "REJECTED"
                hyps = "; ".join(syn_by_gen.get(int(gen_i), [])[:3])
                hyp_str = f" | {hyps}" if hyps else ""
                attempts.append(
                    f"Gen {gen_i} [{tag} | {pid} | D={ad} (Max={md}), S={asc}]{hyp_str}"
                )
            report.recent_attempts = attempts
        except Exception as exc:
            _LOG.warning("Failed populating recent_attempts fallback: %s", exc)

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
        recent_min_g = max(
            1, int(report.latest_completed_gen or report.generation or 1) - 10
        )
        succ_rows = con.execute(
            f"""
            SELECT run_id, seed, max_depth, score, turns
            FROM runs
            WHERE {where_runs}
              AND generation >= ?
            ORDER BY is_outlier_good DESC, max_depth DESC, score DESC, turns DESC
            LIMIT 12
            """,
            [recent_min_g],
        ).fetchall()
        if not succ_rows:
            succ_rows = con.execute(
                f"""
                SELECT run_id, seed, max_depth, score, turns
                FROM runs
                WHERE {where_runs}
                ORDER BY is_outlier_good DESC, max_depth DESC, score DESC, turns DESC
                LIMIT 12
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
                   COALESCE(RIGHT(game_log, 600), '') AS game_log_tail,
                   COALESCE(final_ac, 10) AS final_ac
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
                       COALESCE(RIGHT(game_log, 600), '') AS game_log_tail,
                       COALESCE(final_ac, 10) AS final_ac
                FROM runs
                WHERE {where_runs}
                ORDER BY generation DESC, run_id ASC
                LIMIT 64
                """
            ).fetchall()
        if not raw_rows:
            return
        if len(raw_rows) <= 512:
            rows = list(raw_rows)
        else:
            # Symmetrically stratify across all 4 tails (shortest, longest/max_steps, deepest, median)
            by_turns = sorted(raw_rows, key=lambda x: int(x[4] or 0))
            by_depth = sorted(raw_rows, key=lambda x: int(x[3] or 1), reverse=True)
            mid_start = max(0, (len(by_turns) // 2) - 64)
            picked: dict[str, tuple[Any, ...]] = {}
            for r in (
                by_turns[:128]
                + by_turns[-128:]
                + by_depth[:128]
                + by_turns[mid_start : mid_start + 128]
            ):
                picked[str(r[0])] = r
            for r in raw_rows:
                if len(picked) >= 512:
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
        # 1) Exact non-generic end_reason repeating on >= max(4, n//16) runs.
        # 2) Generic phrase grouping (e.g. `You miss the <X>` + `You hit the <X>! ...`)
        #    when >= 2 distinct phrasings share a recurring `the <1-2 word phrase>` across >= max(4, n//16) runs.
        import re as _re

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
        recurring_end_min = max(4, len(rows) // 36)
        recurring_end_ranks: dict[str, int] = {}
        recurring_end_labels: dict[int, str] = {}
        for er_label, er_ds in sorted(
            end_counts.items(), key=lambda kv: len(kv[1]), reverse=True
        ):
            er_low = er_label.strip().lower()
            if (
                er_low not in _generic_ends
                and "stalled" not in er_low
                and len(er_ds) >= recurring_end_min
                and len(recurring_end_ranks) < 4
            ):
                r_idx = len(recurring_end_ranks) + 1
                recurring_end_ranks[er_low] = r_idx
                recurring_end_labels[r_idx] = er_label.strip()[:64]

        # Group split phrasings sharing a recurring `the <phrase>` subject when slots remain
        if len(recurring_end_labels) < 4:
            phrase_to_ers: dict[str, list[tuple[str, int]]] = {}
            for er_label, er_ds in end_counts.items():
                er_low = er_label.strip().lower()
                if (
                    er_low in _generic_ends
                    or "stalled" in er_low
                    or er_low in recurring_end_ranks
                ):
                    continue
                first_clause = _re.split(r"[|!.?]", er_low, maxsplit=1)[0].strip()
                m_ph = _re.search(r"\bthe\s+([a-z]+(?:\s+[a-z]+)?)", first_clause)
                if m_ph:
                    ph = m_ph.group(1).strip()
                    if len(ph) >= 4:
                        phrase_to_ers.setdefault(ph, []).append((er_low, len(er_ds)))
            for ph, er_list in sorted(
                phrase_to_ers.items(),
                key=lambda kv: sum(cnt for _, cnt in kv[1]),
                reverse=True,
            ):
                if len(recurring_end_labels) >= 4:
                    break
                tot_cnt = sum(cnt for _, cnt in er_list)
                if len(er_list) >= 2 and tot_cnt >= recurring_end_min:
                    r_idx = len(recurring_end_labels) + 1
                    recurring_end_labels[r_idx] = f"recurring phrase `{ph}`"
                    for er_low, _ in er_list:
                        recurring_end_ranks[er_low] = r_idx

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
                final_ac_val,
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

            # Cross-cutting upstream efficiency slice: collect high action-burn runs NON-exclusively
            # so recurring terminal-state symptoms (e.g. starvation/exhaustion) never hide the
            # upstream repetitive action loop, while still allowing distinct peers to tackle both.
            is_high_burn = (
                (float(pct_search or 0.0) >= 35.0 and int(turns or 0) >= 1500)
                or (
                    float(pct_search or 0.0) + float(pct_wait or 0.0) >= 50.0
                    and int(turns or 0) >= 1200
                )
                or (end_low == "max_steps" and int(turns or 0) >= 2000)
            )
            if is_high_burn and not (
                end_low == "max_steps" and int(max_depth or 1) <= 2
            ):
                _add_to_cluster(
                    "slice_high_search_or_horizon_burn",
                    "search_or_horizon_loop",
                    "High repetitive action-burn share or step-horizon burn without advancing floors",
                    row,
                    f"end=`{end_str}`, Dlvl={max_depth}, turns={turns}, pct_search={float(pct_search or 0):.1f}%, pct_wait={float(pct_wait or 0):.1f}%, tiles={unique_tiles}"
                    + (f", last_msg=`{gl_tail}`" if gl_tail else ""),
                )

            # Purely domain-agnostic trajectory-outcome & capability slices (zero NetHack keywords)
            if "stalled" in end_low or stall_count >= 50:
                _add_to_cluster(
                    "slice_stalled",
                    "stalled_trajectory",
                    f"Zero-turn / repeated-action stall terminations (avg Dlvl {int(max_depth or 1)})",
                    row,
                    f"end=`{end_str}`, fn=`{last_fn}`, Dlvl={max_depth}, turns={turns}"
                    + (f", last_msg=`{gl_tail}`" if gl_tail else ""),
                )
            elif end_low == "max_steps" and int(max_depth or 1) <= 2:
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
                rec_lbl = recurring_end_labels.get(r_idx, end_str[:64])
                _add_to_cluster(
                    f"slice_recurring_outcome_{r_idx}",
                    "recurring_terminal_state",
                    f"Recurring terminal outcome: `{rec_lbl}`",
                    row,
                    f"end=`{end_str}`, Dlvl={max_depth}, turns={turns}, pct_search={float(pct_search or 0):.1f}%, tiles={unique_tiles}"
                    + (f", last_msg=`{gl_tail}`" if gl_tail else ""),
                )
            elif is_high_burn:
                # Already recorded above in `slice_high_search_or_horizon_burn`
                pass
            elif (
                stall_count >= 25
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
            elif bool(_unarmored) or (
                unworn_armor_pack
                and int(final_ac_val if final_ac_val is not None else 10) >= 7
            ):
                _add_to_cluster(
                    "slice_unequipped_progression",
                    "capability_gap",
                    "Terminated carrying unequipped gear / stagnant protection",
                    row,
                    f"end=`{end_str}`, Dlvl={max_depth}, ac={final_ac_val}, unworn_armor={bool(_unarmored) or unworn_armor_pack}",
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
            from collections import Counter

            n_c = len(c_rows)
            depths_c = [int(r[3] or 1) for r in c_rows]
            avg_d = sum(float(d) for d in depths_c) / n_c
            avg_t = sum(float(r[4] or 0.0) for r in c_rows) / n_c
            tiles_c = sorted(int(r[6] or 0) for r in c_rows)
            med_tiles = tiles_c[n_c // 2] if tiles_c else 0
            avg_wait = sum(float(r[7] or 0.0) for r in c_rows) / n_c
            avg_srch = sum(float(r[8] or 0.0) for r in c_rows) / n_c
            starved_cnt = sum(1 for r in c_rows if bool(r[9]))
            unarmored_cnt = sum(1 for r in c_rows if bool(r[10]))
            end_hist = Counter(
                str(r[5] or "unknown").strip()[:36] for r in c_rows
            ).most_common(2)
            top_ends = ", ".join(f"`{er}` x{cnt}" for er, cnt in end_hist)
            cluster_stats = (
                f"cluster_stats[N={n_c}]: depth={min(depths_c)}-{max(depths_c)} (avg={avg_d:.2f}), "
                f"avg_turns={avg_t:.0f}, med_tiles={med_tiles}, "
                f"avg_search={avg_srch:.1f}%, avg_wait={avg_wait:.1f}%"
            )
            if starved_cnt:
                cluster_stats += f", starved_with_food={starved_cnt}/{n_c}"
            if unarmored_cnt:
                cluster_stats += f", unworn_armor={unarmored_cnt}/{n_c}"
            if top_ends:
                cluster_stats += f", top_ends=[{top_ends}]"
            if cid == "slice_stalled" and dom_fns:
                cluster_stats += f", top_fns=[{', '.join(dom_fns[:4])}]"
            rep_note = f"; ex1: {notes[0]}" if notes else ""
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
                run_count=n_c,
                avg_depth=avg_d,
                avg_turns=avg_t,
                sample_run_ids=[str(r[0]) for r in c_rows[:4]],
                sample_seeds=list(
                    dict.fromkeys(int(r[1]) for r in c_rows[:4] if r[1] is not None)
                ),
                dominant_fns=dom_fns[:5],
                replay_fixtures=fixtures,
                evidence=f"{cluster_stats}{rep_note}",
            )

        built: list[DeficitCluster] = []
        _kind_weights = {
            "stalled_trajectory": 1.6,
            "confinement_horizon": 1.5,
            "search_or_horizon_loop": 1.5,
            "recurring_terminal_state": 1.3,
        }
        for cid, data in sorted(
            clusters.items(),
            key=lambda kv: (
                len(kv[1]["rows"])
                * _kind_weights.get(str(kv[1]["kind"]), 1.0)
                * (0.35 if kv[0] in (report.cooled_down_deficits or {}) else 1.0),
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

        report.fail_clusters = built[:8]

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
                        LIMIT 3
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
                        extras.append(f"sample1_pre_mortem=[{seq_str}]")
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
                                        str(
                                            hits[0].get("snippet")
                                            or hits[0].get("text")
                                            or ""
                                        )
                                        .replace("\n", " ")[:180]
                                        .strip()
                                    )
                                    if snip:
                                        extras.append(
                                            f"wiki[{hits[0].get('title', '')}]: {snip}"
                                        )
                except Exception as exc:
                    _LOG.warning("Failed attaching pre-mortem/wiki context: %s", exc)
            if extras:
                cl.evidence = (
                    f"{cl.evidence}; " + "; ".join(extras)
                    if cl.evidence
                    else "; ".join(extras)
                )
