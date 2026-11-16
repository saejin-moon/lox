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
                "- MAP-Elites Diversity Stepping Stones (inspect via `read_policy` to escape plateau): "
                + " | ".join(self.plateau_hints[:4])
            )
        if self.fail_clusters:
            lines.append("- Domain-Agnostic Trajectory Slices (D_fail):")
            for c in self.fail_clusters:
                r_ids = f", run_ids={c.sample_run_ids[:2]}" if c.sample_run_ids else ""
                lines.append(
                    f"  * [{c.cluster_id}] {c.label} (runs={c.run_count}, avg_depth={c.avg_depth:.2f}, avg_turns={c.avg_turns:.0f}, fns={c.dominant_fns[:3]}{r_ids}): {c.evidence}"
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
        skill_loc = sum(len(v.splitlines()) for v in skills_map.values() if v)
        skill_count = len(skills_map)
        bloat_warning = (
            (policy_loc >= 300 and skill_count == 0)
            or (policy_loc >= 400)
            or (skill_count >= 6)
            or ((policy_loc + skill_loc) >= 500)
        )

        report = DeficitReport(
            generation=generation,
            regime="BOOTSTRAP" if not current_policy.strip() else "CLIMBING",
            policy_loc=policy_loc,
            skill_loc=skill_loc,
            skill_count=skill_count,
            bloat_warning=bloat_warning,
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
            self._populate_d_succ_and_branches(con, report, raw_branches)
            self._populate_d_fail_clusters(con, report)
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
        try:
            row = con.execute(
                "SELECT COALESCE(MAX(generation), 0) FROM runs"
            ).fetchone()
            if row and row[0] is not None:
                report.latest_completed_gen = int(row[0])
        except Exception:
            pass
        try:
            t_row = con.execute(
                "SELECT COALESCE(MIN(r.generation), 0) "
                "FROM ticks t JOIN runs r ON t.run_id = r.run_id"
            ).fetchone()
            if t_row and t_row[0] is not None:
                report.ticks_min_gen = int(t_row[0])
        except Exception:
            pass

    @staticmethod
    def _populate_plateau_signal(con: Any, report: DeficitReport) -> None:
        rows = con.execute(
            """
            SELECT generation, avg_depth, max_depth
            FROM batches
            WHERE UPPER(mode) = 'SWARM'
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

    @staticmethod
    def _populate_d_succ_and_branches(
        con: Any,
        report: DeficitReport,
        bt_branches: list[dict[str, str]],
    ) -> None:
        succ_rows = con.execute(
            """
            SELECT run_id, seed, max_depth, score, turns
            FROM runs
            WHERE generation > 0
            ORDER BY is_outlier_good DESC, max_depth DESC, score DESC, turns DESC
            LIMIT 4
            """
        ).fetchall()
        report.d_succ_run_ids = [str(r[0]) for r in succ_rows]
        report.d_succ_seeds = list(
            dict.fromkeys(int(r[1]) for r in succ_rows if r[1] is not None)
        )

        fn_rows = con.execute(
            """
            SELECT function_name,
                   MAX(COALESCE(ochiai_good_run_score, 0.0)) AS max_ochiai,
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

    @staticmethod
    def _populate_d_fail_clusters(con: Any, report: DeficitReport) -> None:
        rows = con.execute(
            """
            SELECT run_id, seed, depth, max_depth, turns, end_reason,
                   unique_tiles, pct_wait, pct_search,
                   COALESCE(starved_with_food, FALSE) AS starved_with_food,
                   COALESCE(died_unarmored, FALSE) AS died_unarmored,
                   COALESCE(auto_esc_count, 0) AS auto_esc_count,
                   COALESCE(policy_exceptions, 0) AS policy_exceptions,
                   COALESCE(last_active_function, '') AS last_active_fn,
                   COALESCE(summary_digest, '') AS summary_digest,
                   action_bytes,
                   milestones_json
            FROM runs
            WHERE generation > 0
            ORDER BY generation DESC, turns DESC
            LIMIT 48
            """
        ).fetchall()
        if not rows:
            return

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
            ) = row

            end_str = str(end_reason or "unknown").strip() or "unknown"
            if end_str in ("ascended", "escaped"):
                continue
            digest_s = str(digest or "")
            stall_count = 0
            if m_json:
                try:
                    stall_count = int(json.loads(m_json).get("stall_count", 0) or 0)
                except Exception:
                    stall_count = 0

            # Purely domain-agnostic trajectory-outcome slices (zero NetHack keywords)
            if (
                stall_count >= 25
                or "CONFINEMENT_STALL" in digest_s
                or "stalled" in end_str.lower()
                or "max_steps" in end_str.lower()
                or (float(pct_wait or 0.0) + float(pct_search or 0.0) >= 65.0)
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
                    f"end=`{end_str}`, tiles={unique_tiles}, pct_wait={float(pct_wait or 0):.1f}%",
                )
            elif int(policy_excs or 0) > 0 or int(auto_esc or 0) > 2:
                _add_to_cluster(
                    "slice_safety_traps",
                    "runtime_or_prompt_trap",
                    "Runtime exceptions or unhandled prompt traps",
                    row,
                    f"policy_exceptions={policy_excs}, auto_esc={auto_esc}",
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
            clusters.items(), key=lambda kv: len(kv[1]["rows"]), reverse=True
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
        # If fewer than 4 distinct clusters exist while the top cluster bundles multiple
        # distinct `last_active_function` cohorts, split out secondary cohorts (domain-agnostic)
        # so concurrent peers claim genuinely distinct empirical slices instead of duplicating.
        if len(built) < 4 and built:
            top_cid = built[0].cluster_id
            by_fn: dict[str, list[tuple[Any, ...]]] = {}
            for r in clusters[top_cid]["rows"]:
                fn_key = str(r[13] or "unknown").strip() or "unknown"
                by_fn.setdefault(fn_key, []).append(r)
            if len(by_fn) >= 2:
                ordered_fns = sorted(
                    by_fn.items(), key=lambda kv: len(kv[1]), reverse=True
                )
                for fn_key, sub_rows in ordered_fns[1:]:
                    if len(built) >= 4 or len(sub_rows) < 2:
                        break
                    sub_notes = list(
                        dict.fromkeys(
                            f"end=`{str(r[5] or 'unknown').strip()}`, tiles={r[6]}"
                            for r in sub_rows[:2]
                        )
                    )
                    built.append(
                        _build_cluster(
                            f"{top_cid}_{fn_key}",
                            clusters[top_cid]["kind"],
                            f"{clusters[top_cid]['label']} [{fn_key}]",
                            sub_rows,
                            [fn_key],
                            sub_notes,
                        )
                    )

        report.fail_clusters = built[:6]
