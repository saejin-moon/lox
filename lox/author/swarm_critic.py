"""
Two-Pass Read-Only Advisory Critic Agent for LOX Swarm Synthesis ("Hard Code Nothing").

Strictly separated from code implementation (ZERO code-writing, ZERO `dry_run_policy`,
and ZERO hardcoded rollback gates or metric thresholds):
1. Pass 1 — Pre-Generation High-Level Guide (`guide_before_generation`):
   Called as the very first agent every generation. Investigates DuckDB telemetry,
   cross-generation progression, `PrioritySelector` branch attribution, policy diffs,
   and NetHack Wiki mechanics to provide the Meta-Agent with a high-level strategic guide.
2. Pass 2 — Post-Assembly Comparative Review (`review_after_assembly`):
   Called once after the Meta-Agent finishes its initial policy assembly and evaluation.
   Compares the candidate's empirical results, actions, and branch turn shares against
   previous generations and informs the Meta-Agent so the Meta-Agent can decide whether
   and how to revise its policy.
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, field
from typing import Any

from lox.author.agent import AuthorAgent
from lox.author.prompts import build_system_prompt
from lox.db.schema import connect_db
from lox.db.theories import TheoryDocumentStore
from lox.ui.console import ui

READ_ONLY_CRITIC_TOOLS: set[str] = {
    "query_db",
    "read_policy",
    "diff_policies",
    "read_skill",
    "query_policy_docs",
    "query_wiki",
    "query_monster",
    "query_item",
    "query_theories",
}


def extract_priority_selector_branches(policy_code: str) -> list[dict[str, str]]:
    """
    Parses `policy_code` AST to extract the ordered list of children in the root
    `PrioritySelector([...])` (priority `#1..#K`, `leaf_name`, and `fn_name`).
    """
    if not policy_code or not policy_code.strip():
        return []
    try:
        tree = ast.parse(policy_code)
    except Exception:
        return []

    def _expr_name(expr: ast.AST) -> str:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            base = _expr_name(expr.value)
            return f"{base}.{expr.attr}" if base else expr.attr
        return ""

    selector_calls: list[ast.Call] = []
    scan_roots: list[ast.AST] = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "build_tree"
    ] or [tree]

    for root_node in scan_roots:
        for node in ast.walk(root_node):
            if isinstance(node, ast.Call):
                func_id = ""
                if isinstance(node.func, ast.Name):
                    func_id = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_id = node.func.attr
                if func_id == "PrioritySelector" and (
                    node.args or any(kw.arg == "children" for kw in node.keywords)
                ):
                    if (
                        node.args and isinstance(node.args[0], (ast.List, ast.Tuple))
                    ) or any(
                        kw.arg == "children"
                        and isinstance(kw.value, (ast.List, ast.Tuple))
                        for kw in node.keywords
                    ):
                        selector_calls.append(node)
                elif (
                    func_id == "from_modules"
                    and isinstance(node.func, ast.Attribute)
                    and _expr_name(node.func.value) == "PrioritySelector"
                ):
                    selector_calls.append(node)

    if not selector_calls:
        # Option-A fallback: inspect registered `register_goal(...)` branches
        goal_branches: list[dict[str, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fid = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (
                        node.func.attr if isinstance(node.func, ast.Attribute) else ""
                    )
                )
                if fid == "register_goal":
                    g_name = ""
                    fn_ref = ""
                    if node.args and isinstance(node.args[0], ast.Constant):
                        g_name = str(node.args[0].value)
                    for kw in node.keywords:
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                            g_name = str(kw.value.value)
                        elif kw.arg in ("fn", "handler", "action_fn"):
                            fn_ref = _expr_name(kw.value).split(".")[-1]
                    if len(node.args) >= 2 and not fn_ref:
                        fn_ref = _expr_name(node.args[1]).split(".")[-1]
                    if g_name:
                        idx_g = len(goal_branches) + 1
                        goal_branches.append(
                            {
                                "priority": str(idx_g),
                                "leaf_name": g_name,
                                "fn_name": fn_ref or g_name,
                            }
                        )
        return goal_branches

    # Breadth-first ast.walk() yields the outermost PrioritySelector first; pick first unless named "master".
    target_call = selector_calls[0]
    for c in selector_calls:
        for kw in c.keywords:
            if (
                kw.arg == "name"
                and isinstance(kw.value, ast.Constant)
                and "master" in str(kw.value.value).lower()
            ):
                target_call = c
                break

    if (
        isinstance(target_call.func, ast.Attribute)
        and target_call.func.attr == "from_modules"
    ):
        seq_node = (
            target_call.args[0]
            if target_call.args
            else next(
                (kw.value for kw in target_call.keywords if kw.arg == "modules"),
                None,
            )
        )
        mod_elts = (
            getattr(seq_node, "elts", [])
            if isinstance(seq_node, (ast.List, ast.Tuple))
            else []
        )
        mod_branches: list[dict[str, str]] = []
        for idx_m, elt_m in enumerate(mod_elts, start=1):
            nm = (
                str(elt_m.value)
                if isinstance(elt_m, ast.Constant) and isinstance(elt_m.value, str)
                else _expr_name(elt_m)
            )
            if nm:
                short_nm = nm.split(".")[-1]
                mod_branches.append(
                    {"priority": str(idx_m), "leaf_name": nm, "fn_name": short_nm}
                )
        return mod_branches

    seq_node = (
        target_call.args[0]
        if target_call.args
        else next(
            (kw.value for kw in target_call.keywords if kw.arg == "children"),
            None,
        )
    )
    elts = getattr(seq_node, "elts", [])
    branches: list[dict[str, str]] = []

    def _collect_call_fns(call_node: ast.Call) -> list[str]:
        sub_fns: list[str] = []
        target_args = list(call_node.args[:1])
        for kw in call_node.keywords:
            if kw.arg in ("action_fn", "cond_fn", "children", "child"):
                target_args.append(kw.value)
        for first in target_args:
            if isinstance(first, (ast.List, ast.Tuple)):
                for sub in first.elts:
                    if isinstance(sub, ast.Call):
                        sub_fns.extend(_collect_call_fns(sub))
                    else:
                        s_fn = _expr_name(sub).split(".")[-1]
                        if s_fn:
                            sub_fns.append(s_fn)
            elif isinstance(first, ast.Call):
                sub_fns.extend(_collect_call_fns(first))
            else:
                s_fn = _expr_name(first).split(".")[-1]
                if s_fn:
                    sub_fns.append(s_fn)
        return sub_fns

    for idx, elt in enumerate(elts, start=1):
        leaf_name = ""
        fn_name = ""
        if isinstance(elt, ast.Call):
            for kw in elt.keywords:
                if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                    leaf_name = str(kw.value.value)
            fns = _collect_call_fns(elt)
            if fns:
                fn_name = "+".join(fns)
            if not leaf_name:
                leaf_name = fn_name or _expr_name(elt.func)
        else:
            raw_ref = _expr_name(elt)
            leaf_name = raw_ref
            fn_name = raw_ref.split(".")[-1]

        branches.append(
            {
                "priority": str(idx),
                "leaf_name": leaf_name or f"branch_{idx}",
                "fn_name": fn_name or leaf_name or f"branch_{idx}",
            }
        )
    return branches


def build_swarm_diagnostic_digest(
    db_path: str,
    current_policy: str = "",
    target_generation: int | None = None,
) -> str:
    """
    Constructs an objective, cross-generation diagnostic digest from DuckDB and `current_policy`:
    1. Cross-Generation Progression Table (`batches` + `runs` aggregates, highlighting historical bests).
    2. Ordered `PrioritySelector` Branch Telemetry Table (`#1..#K` joined with `function_stats`).
    3. Active Function Shifts (functions active in prior/best generation missing in current generation).
    4. Swallowed `Blackboard.leaf_errors` & Top Termination Causes.
    """
    if not db_path or not os.path.exists(db_path):
        return ""

    con = connect_db(db_path, read_only=True)
    sections: list[str] = []
    try:
        batch_rows = con.execute(
            """
            SELECT b.generation,
                   b.candidate_policy_id,
                   b.avg_depth,
                   b.p90_depth,
                   b.max_depth,
                   b.avg_score,
                   b.max_score,
                   b.avg_turns,
                   COALESCE(AVG(r.unique_tiles), 0.0) AS avg_tiles,
                   COALESCE(SUM(r.policy_exceptions), 0) AS total_exc
            FROM batches b
            LEFT JOIN runs r ON r.batch_id = b.batch_id AND COALESCE(r.is_shard, FALSE) = FALSE
            GROUP BY b.generation, b.batch_id, b.candidate_policy_id, b.avg_depth, b.p90_depth,
                     b.max_depth, b.avg_score, b.max_score, b.avg_turns, b.created_at
            ORDER BY b.generation ASC, b.created_at ASC
            """
        ).fetchall()

        if not batch_rows:
            return ""

        best_depth_gen = max(
            batch_rows,
            key=lambda r: (float(r[2] or 0.0), int(r[4] or 0), float(r[5] or 0.0)),
        )
        best_gen_num = int(best_depth_gen[0])
        latest_gen_num = (
            int(target_generation)
            if target_generation is not None
            else int(batch_rows[-1][0])
        )
        latest_batch_row = con.execute(
            "SELECT batch_id FROM batches WHERE generation = ? ORDER BY created_at DESC LIMIT 1",
            [latest_gen_num],
        ).fetchone()
        latest_batch_id = str(latest_batch_row[0]) if latest_batch_row else None

        # 1. Cross-Generation Progression Table
        table_lines = [
            "### Cross-Generation Progression Overview (`batches` & `runs`):",
            "| Gen | Policy ID | Avg Depth | P90 Depth | Max Depth | Avg Score | Max Score | Avg Turns | Avg Tiles | Exceptions | Notes |",
            "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
        ]
        # Show up to last 8 generations plus best generation if earlier
        display_rows = batch_rows[-8:]
        if best_depth_gen not in display_rows:
            display_rows = [best_depth_gen, *display_rows]

        for r in display_rows:
            g_num = int(r[0])
            p_id = str(r[1] or f"policy_swarm_g{g_num}")
            avg_d = float(r[2]) if r[2] is not None else 1.0
            p90_d = float(r[3]) if r[3] is not None else 1.0
            max_d = int(r[4]) if r[4] is not None else 1
            avg_s = float(r[5]) if r[5] is not None else 0.0
            max_s = int(r[6]) if r[6] is not None else 0
            avg_t = float(r[7]) if r[7] is not None else 0.0
            avg_tiles = float(r[8]) if r[8] is not None else 0.0
            tot_exc = int(r[9]) if r[9] is not None else 0
            tags = []
            if g_num == best_gen_num:
                tags.append("PEAK_DEPTH")
            if g_num == latest_gen_num:
                tags.append("LATEST")
            note_str = ", ".join(tags)
            table_lines.append(
                f"| {g_num} | `{p_id}` | {avg_d:.2f} | {p90_d:.1f} | {max_d} | "
                f"{avg_s:.1f} | {max_s} | {avg_t:.0f} | {avg_tiles:.1f} | {tot_exc} | {note_str} |"
            )
        sections.append("\n".join(table_lines))

        # 2. Termination Breakdown & Leaf Errors for latest_gen_num
        scope_col = "batch_id" if latest_batch_id is not None else "generation"
        scope_val = latest_batch_id if latest_batch_id is not None else latest_gen_num
        end_rows = con.execute(
            f"""
            SELECT end_reason, COUNT(*) AS cnt, ROUND(AVG(max_depth), 2) AS avg_d, ROUND(AVG(turns), 0) AS avg_t
            FROM runs
            WHERE {scope_col} = ? AND COALESCE(is_shard, FALSE) = FALSE
            GROUP BY end_reason
            ORDER BY cnt DESC
            LIMIT 6
            """,
            [scope_val],
        ).fetchall()
        if end_rows:
            end_bullets = [
                f"- `{r[0]}`: {r[1]} run(s) (avg_max_depth={r[2]}, avg_turns={int(r[3] or 0)})"
                for r in end_rows
            ]
            sections.append(
                f"### Generation {latest_gen_num} Top Termination Causes:\n"
                + "\n".join(end_bullets)
            )

        # Check for swallowed leaf_errors in milestones_json
        milestone_rows = con.execute(
            f"SELECT run_id, milestones_json FROM runs WHERE {scope_col} = ? AND COALESCE(is_shard, FALSE) = FALSE AND policy_exceptions > 0 LIMIT 10",
            [scope_val],
        ).fetchall()
        leaf_err_samples: list[str] = []
        for r_id, m_json in milestone_rows:
            try:
                m_dict = json.loads(m_json or "{}")
                for le in m_dict.get("leaf_errors") or []:
                    sample = f"`{r_id}` -> {le}"
                    if sample not in leaf_err_samples:
                        leaf_err_samples.append(sample)
            except Exception:
                continue
        if leaf_err_samples:
            sections.append(
                f"### Generation {latest_gen_num} Swallowed BehaviorTree Leaf Errors (`policy_exceptions`):\n"
                + "\n".join(f"- {s}" for s in leaf_err_samples[:6])
            )

        # 3. Function Stats & Ordered PrioritySelector Branch Table
        fn_rows = con.execute(
            f"""
            SELECT function_name, episodes_triggered, total_calls, total_turns_active,
                   COALESCE(stagnation_share, 0.0),
                   COALESCE(livelock_score, 0.0),
                   COALESCE(liblit_failure_score, 0.0),
                   COALESCE(ochiai_good_run_score, 0.0)
            FROM function_stats
            WHERE {scope_col} = ?
            ORDER BY total_turns_active DESC, total_calls DESC
            """,
            [scope_val],
        ).fetchall()
        fn_map: dict[str, dict[str, Any]] = {}
        total_active_turns = sum(int(r[3] or 0) for r in fn_rows) or 1
        for r in fn_rows:
            fn_map[str(r[0])] = {
                "episodes": int(r[1] or 0),
                "calls": int(r[2] or 0),
                "turns": int(r[3] or 0),
                "share_pct": 100.0 * int(r[3] or 0) / total_active_turns,
                "stag": float(r[4] or 0.0),
                "livelock": float(r[5] or 0.0),
                "liblit": float(r[6] or 0.0),
                "ochiai": float(r[7] or 0.0),
            }

        branches = extract_priority_selector_branches(current_policy)
        if branches:
            b_lines = [
                f"### Ordered `PrioritySelector` Branch Telemetry (Gen {latest_gen_num}):",
                "| Priority | Leaf Name | Function | Episodes | Calls | Active Turns (% Share) | Stagnation | Livelock | Liblit Fail | Status |",
                "|---:|:---|:---|---:|---:|---:|---:|---:|---:|:---|",
            ]
            for b in branches:
                fn_key = b["fn_name"]
                part_keys = [k.strip() for k in fn_key.split("+") if k.strip()]
                stats = fn_map.get(fn_key)
                if stats is None and part_keys:
                    for pk in part_keys:
                        if pk in fn_map:
                            stats = fn_map[pk]
                            break
                if stats is None:
                    b_lines.append(
                        f"| #{b['priority']} | `{b['leaf_name']}` | `{fn_key}` | 0 | 0 | 0 (0.0%) | 0.00 | 0.00 | 0.00 | 0-TURN (UNTRIGGERED/PREEMPTED) |"
                    )
                else:
                    status_flags = []
                    if stats["turns"] == 0:
                        status_flags.append("0-TURN (PREEMPTED/COND_FALSE)")
                    if stats["stag"] >= 0.40:
                        status_flags.append("HIGH_STAGNATION")
                    if stats["livelock"] >= 0.35:
                        status_flags.append("HIGH_LIVELOCK")
                    status_str = ", ".join(status_flags) or "ACTIVE"
                    b_lines.append(
                        f"| #{b['priority']} | `{b['leaf_name']}` | `{fn_key}` | {stats['episodes']} | {stats['calls']} | "
                        f"{stats['turns']} ({stats['share_pct']:.1f}%) | {stats['stag']:.2f} | {stats['livelock']:.2f} | "
                        f"{stats['liblit']:.2f} | {status_str} |"
                    )
            sections.append("\n".join(b_lines))
        elif fn_rows:
            f_lines = [
                f"### Top Active Functions in Generation {latest_gen_num}:",
            ]
            for r in fn_rows[:12]:
                if int(r[3] or 0) > 0 or int(r[2] or 0) > 0:
                    share = 100.0 * int(r[3] or 0) / total_active_turns
                    f_lines.append(
                        f"- `{r[0]}`: turns={r[3]} ({share:.1f}%), calls={r[2]}, eps={r[1]}, "
                        f"stag={float(r[4]):.2f}, livelock={float(r[5]):.2f}, liblit={float(r[6]):.2f}"
                    )
            sections.append("\n".join(f_lines))

        # 4. Compare active functions against previous generation and peak generation
        compare_gens = sorted(
            {
                g
                for g in (latest_gen_num - 1, best_gen_num)
                if g >= 1 and g != latest_gen_num
            }
        )
        dropped_notes: list[str] = []
        for cg in compare_gens:
            prior_fns = con.execute(
                """
                SELECT function_name, total_turns_active
                FROM function_stats
                WHERE generation = ? AND total_turns_active > 0
                ORDER BY total_turns_active DESC
                """,
                [cg],
            ).fetchall()
            missing = [
                f"`{pf[0]}` ({pf[1]} turns in Gen {cg})"
                for pf in prior_fns
                if str(pf[0]) not in fn_map or fn_map[str(pf[0])]["turns"] == 0
            ]
            if missing:
                label = "Peak" if cg == best_gen_num else "Previous"
                dropped_notes.append(
                    f"- Active in {label} Gen {cg} (`policy_swarm_g{cg}`) but 0 active turns in Gen {latest_gen_num}: {', '.join(missing[:8])}"
                )
        if dropped_notes:
            sections.append(
                "### Branch Activity Shifts vs. Prior Generations:\n"
                + "\n".join(dropped_notes)
            )
    except Exception as exc:
        ui.warn(f"Critic diagnostic digest query warning: {exc}")
    finally:
        con.close()

    return "\n\n".join(s for s in sections if s.strip())


@dataclass
class CriticGuidance:
    """High-level strategic guide produced by Critic Pass 1 at the start of a generation."""

    generation: int
    reference_policy_id: str = "none"
    high_level_summary: str = ""
    root_causes: list[dict[str, Any]] = field(default_factory=list)
    working_capabilities: list[str] = field(default_factory=list)
    high_level_priorities: list[str] = field(default_factory=list)
    diagnostic_digest: str = ""
    raw_summary: str = ""

    @property
    def champion_policy_id(self) -> str:
        return self.reference_policy_id

    @property
    def protected_invariants(self) -> list[str]:
        return self.working_capabilities

    @property
    def next_gen_focus(self) -> list[str]:
        return self.high_level_priorities

    def to_prompt_block(self) -> str:
        parts = [
            f"### Critic Pre-Generation High-Level Guide (Generation {self.generation})"
        ]
        if self.high_level_summary:
            parts.append(f"**High-Level Assessment:** {self.high_level_summary}")
        if self.root_causes:
            parts.append("**Empirical Root Causes & Bottlenecks:**")
            for rc in self.root_causes:
                sym = rc.get("symptom", "")
                fn = rc.get("offending_function_or_skill", "")
                rids = rc.get("evidence_run_ids", [])
                mech = rc.get("mechanism", "")
                parts.append(f"- [{fn}] {sym} (runs={rids}): {mech}")
        if self.working_capabilities:
            parts.append("**Working Capabilities Observed in History:**")
            for cap in self.working_capabilities:
                parts.append(f"- {cap}")
        if self.high_level_priorities:
            parts.append("**High-Level Priorities for This Generation:**")
            for pri in self.high_level_priorities:
                parts.append(f"- {pri}")
        if len(parts) == 1 and self.raw_summary:
            parts.append(self.raw_summary)
        if self.diagnostic_digest:
            parts.append(self.diagnostic_digest)
        return "\n\n".join(parts)


# Backward-compatible alias
DiagnosticReport = CriticGuidance


@dataclass
class PostAssemblyCritique:
    """Comparative review produced by Critic Pass 2 after Meta-Agent finishes its initial assembly."""

    generation: int
    candidate_policy_id: str
    needs_revision: bool = False
    comparison_to_prev_gens: str = ""
    identified_issues: list[str] = field(default_factory=list)
    recommendations_for_meta: list[str] = field(default_factory=list)
    diagnostic_digest: str = ""
    raw_summary: str = ""

    def to_prompt_block(self) -> str:
        parts = [
            f"### Critic Post-Assembly Comparative Review (Generation {self.generation} Draft: `{self.candidate_policy_id}`)",
            f"- **Advisory Revision Recommended**: `{self.needs_revision}`",
        ]
        if self.comparison_to_prev_gens:
            parts.append(
                f"**Comparison vs. Previous Generations:**\n{self.comparison_to_prev_gens}"
            )
        if self.identified_issues:
            parts.append("**Identified Issues / Regressions / Branch Conflicts:**")
            for iss in self.identified_issues:
                parts.append(f"- {iss}")
        if self.recommendations_for_meta:
            parts.append("**High-Level Recommendations for Meta-Agent:**")
            for rec in self.recommendations_for_meta:
                parts.append(f"- {rec}")
        if (
            not self.comparison_to_prev_gens
            and not self.identified_issues
            and not self.recommendations_for_meta
            and self.raw_summary
        ):
            parts.append(self.raw_summary)
        if self.diagnostic_digest:
            parts.append(self.diagnostic_digest)
        return "\n\n".join(parts)


class SwarmCriticAgent:
    """
    Two-Pass Read-Only Empirical Critic & High-Level Guide for the Meta-Agent.
    Contains ZERO hardcoded rollbacks or threshold gates; empowers the Meta-Agent
    with structured cross-generation diagnostics before and after assembly.
    """

    def __init__(
        self,
        provider: str = "openrouter",
        model: str | None = None,
        db_path: str = "data/lox.duckdb",
        theories_store: TheoryDocumentStore | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.db_path = db_path
        self.theories = theories_store or TheoryDocumentStore(db_path=db_path)
        self.author = AuthorAgent(
            provider=provider,
            model=model or "z-ai/glm-5.3-flash:nitro",
            db_path=db_path,
            agent_role="critic",
        )
        self.author.tools.theories_store = self.theories

    @staticmethod
    def _parse_json_object(raw_txt: str) -> dict[str, Any]:
        from lox.lang.bundle import extract_json_payload

        parsed = extract_json_payload(raw_txt)
        return parsed if isinstance(parsed, dict) else {}

    def guide_before_generation(
        self,
        generation: int,
        reference_policy_id: str | None = None,
        current_policy: str = "",
        skills: dict[str, str] | None = None,
        champion_policy_id: str | None = None,
    ) -> CriticGuidance:
        """
        Pass 1 (Pre-Generation Guide): Called as the very first agent in Generation `generation`.
        Audits DuckDB history, `PrioritySelector` branch telemetry, policy diffs, and NetHack Wiki
        mechanics to give the Meta-Agent a high-level strategic guide for the generation.
        """
        ref_id = (
            reference_policy_id
            or champion_policy_id
            or (f"policy_swarm_g{generation - 1}" if generation > 1 else "none")
        )
        digest = build_swarm_diagnostic_digest(
            self.db_path,
            current_policy=current_policy,
            target_generation=generation - 1 if generation > 1 else None,
        )

        if self.provider == "mock":
            return CriticGuidance(
                generation=generation,
                reference_policy_id=ref_id,
                high_level_summary=f"Pre-generation high-level guide for Generation {generation}.",
                root_causes=[
                    {
                        "symptom": "Early exploration and progression baseline",
                        "offending_function_or_skill": "none",
                        "evidence_run_ids": [],
                        "mechanism": "Mock pre-generation diagnostic baseline.",
                    }
                ],
                working_capabilities=[
                    "Preserve working navigation and prompt handling branches"
                ],
                high_level_priorities=[
                    "progression_and_exploration",
                    "threat_and_survival",
                ],
                diagnostic_digest=digest,
                raw_summary="Mock CriticGuidance",
            )

        ui.info(
            f"Critic Pass 1 (Pre-Gen Guide): Auditing campaign state for Generation {generation}..."
        )
        self.author.tools.generation = int(generation)
        self.author.tools.current_policy_code = str(current_policy or "")
        if skills is not None:
            self.author.tools.skills = dict(skills)

        empty_campaign = self.author.tools._campaign_tables_empty()
        gen_instruction = (
            "Campaign tables (`runs`, `batches`, `ticks`, `function_stats`, `policies`) are currently empty. "
            "Call BOTH `query_wiki` and `query_db` on `expert_runs` / `expert_ticks` to study how human ascensions progress."
            if empty_campaign
            else (
                "Review the Cross-Generation & Branch Telemetry below and use your read-only tools "
                "(`query_db`, `read_policy`, `diff_policies`, `read_skill`, `query_wiki`, `query_monster`, `query_item`) "
                "to compare recent generations against the best historical generation in `batches` and `policies`."
            )
        )
        digest_section = f"\n{digest}\n" if digest else ""
        prompt = f"""You are the High-Level Critic & Strategic Guide (Pass 1 — Start of Generation {generation}).
You are the very first agent called in Generation {generation}. Your role is to investigate what happened in previous generations (or expert runs on Gen 1) and provide a high-level strategic guide to the Meta-Agent. You DO NOT write code.
{digest_section}
{gen_instruction}

Determine:
1. `high_level_summary`: Concise synthesis of how the policy is performing across generations (including whether recent generations improved or regressed compared to earlier policies, and why).
2. `root_causes`: Top empirical bottlenecks holding the agent back (cite concrete `run_id`s, `function_stats`, `end_reason`s, or C mechanics from `query_wiki`).
3. `working_capabilities`: Existing branches, skills, or historical policy components that are working well and should be preserved or restored by the Meta-Agent.
4. `high_level_priorities`: 2–4 high-level problem areas for the Meta-Agent and its specialist workers to focus on this generation.

Return ONLY a valid JSON object:
{{
  "high_level_summary": "...",
  "root_causes": [
    {{
      "symptom": "...",
      "offending_function_or_skill": "...",
      "evidence_run_ids": ["..."],
      "mechanism": "..."
    }}
  ],
  "working_capabilities": ["..."],
  "high_level_priorities": ["..."]
}}
"""
        guidance = CriticGuidance(
            generation=generation,
            reference_policy_id=ref_id,
            diagnostic_digest=digest,
        )
        try:
            raw = self.author.run_tool_session(
                system_prompt=build_system_prompt(self.db_path, is_swarm=True),
                user_prompt=prompt,
                max_tool_turns=5,
                required_tools=(
                    ["query_wiki", "query_db:expert_runs"] if empty_campaign else None
                ),
                allowed_tools=READ_ONLY_CRITIC_TOOLS,
                generation=generation,
                session_mode="CRITIC_PRE_GUIDE",
                agent_role_override="critic",
                require_json=True,
                expected_json_type="object",
            )
            guidance.raw_summary = raw
            parsed = self._parse_json_object(raw)
            guidance.high_level_summary = str(parsed.get("high_level_summary") or "")
            if isinstance(parsed.get("root_causes"), list):
                guidance.root_causes = [
                    rc for rc in parsed["root_causes"] if isinstance(rc, dict)
                ]
            caps = parsed.get("working_capabilities") or parsed.get(
                "protected_invariants"
            )
            if isinstance(caps, list):
                guidance.working_capabilities = [str(x) for x in caps if str(x).strip()]
            pris = parsed.get("high_level_priorities") or parsed.get("next_gen_focus")
            if isinstance(pris, list):
                guidance.high_level_priorities = [
                    str(x) for x in pris if str(x).strip()
                ]
        except Exception as exc:
            ui.warn(f"Critic Pass 1 fallback ({exc})")

        try:
            self.theories.save_document(
                generation=generation,
                agent_role="critic",
                doc_type="CRITIC_PRE_GUIDE",
                title=f"Critic Pre-Gen Guide G{generation}",
                content=guidance.to_prompt_block(),
                tags=["critic", f"gen_{generation}", "pre_guide"],
            )
        except Exception:
            pass
        return guidance

    # Alias for callers using the previous method name
    diagnose_for_generation = guide_before_generation

    def review_after_assembly(
        self,
        generation: int,
        batch_id: str,
        candidate_policy_id: str,
        current_policy: str = "",
        skills: dict[str, str] | None = None,
    ) -> PostAssemblyCritique:
        """
        Pass 2 (Post-Assembly Comparative Review): Called once after the Meta-Agent finishes
        its initial policy assembly and the candidate batch (`batch_id`) has been evaluated.
        Compares the candidate's empirical results, actions, and `PrioritySelector` branch
        behavior against previous generations, and informs the Meta-Agent.
        """
        digest = build_swarm_diagnostic_digest(
            self.db_path,
            current_policy=current_policy,
            target_generation=generation,
        )

        if self.provider == "mock":
            return PostAssemblyCritique(
                generation=generation,
                candidate_policy_id=candidate_policy_id,
                needs_revision=False,
                comparison_to_prev_gens=(
                    f"Mock post-assembly comparison for `{candidate_policy_id}` (`{batch_id}`)."
                ),
                identified_issues=[],
                recommendations_for_meta=[
                    "Draft policy executed cleanly in mock evaluation."
                ],
                diagnostic_digest=digest,
                raw_summary="Mock PostAssemblyCritique",
            )

        ui.info(
            f"Critic Pass 2 (Post-Assembly Review): Comparing `{candidate_policy_id}` (`{batch_id}`) against previous generations..."
        )
        self.author.tools.generation = int(generation)
        self.author.tools.current_policy_code = str(current_policy or "")
        if skills is not None:
            self.author.tools.skills = dict(skills)

        digest_section = f"\n{digest}\n" if digest else ""
        prompt = f"""You are the High-Level Critic & Advisor (Pass 2 — Post-Assembly Review for Generation {generation}).
The Meta-Agent has assembled its candidate policy `{candidate_policy_id}` and evaluated it in batch `{batch_id}`.
Your job is to compare `{candidate_policy_id}`'s empirical results, actions, and `PrioritySelector` branch telemetry against previous generations (and `expert_runs` if Gen 1), and advise the Meta-Agent on whether any revision is needed before Generation {generation} concludes.
There are NO automatic rollbacks—the Meta-Agent relies on your analysis to decide whether to keep `{candidate_policy_id}` as-is, fix a bug/hijacking branch, or restore code from an earlier policy in `policies`.
{digest_section}
Use your read-only tools (`query_db`, `diff_policies`, `read_policy`, `read_skill`, `query_wiki`) to inspect:
1. How `{candidate_policy_id}` (`{batch_id}`) compares to previous generations (`batches`, `runs.summary_digest`, `end_reason`, `avg_depth`, `max_depth`, `avg_score`, `unique_tiles`, `policy_exceptions`).
2. Whether any newly wired `PrioritySelector` branch is monopolizing turns with high stagnation/livelock, throwing `leaf_errors`, or starving lower-priority branches (`0-TURN`), or whether a working branch from an earlier policy (`diff_policies`) was accidentally dropped.
3. Concrete, high-level recommendations for the Meta-Agent (`needs_revision`: `true` if the Meta-Agent should revise `{candidate_policy_id}` or restore a capability from an earlier `policy_id`; `false` if `{candidate_policy_id}` is healthy and ready).

Return ONLY a valid JSON object:
{{
  "needs_revision": true or false,
  "comparison_to_prev_gens": "Empirical comparison of depth, score, turns, actions, and branch behavior vs previous generations",
  "identified_issues": ["concrete regressions, hijacking branches, leaf errors, or dropped capabilities (or empty list if none)"],
  "recommendations_for_meta": ["actionable guidance for the Meta-Agent (e.g. which branch to fix/reorder or which policy_id/skill to inspect/restore)"]
}}
"""
        critique = PostAssemblyCritique(
            generation=generation,
            candidate_policy_id=candidate_policy_id,
            diagnostic_digest=digest,
        )
        try:
            raw = self.author.run_tool_session(
                system_prompt=build_system_prompt(self.db_path, is_swarm=True),
                user_prompt=prompt,
                max_tool_turns=4,
                allowed_tools=READ_ONLY_CRITIC_TOOLS,
                generation=generation,
                session_mode="CRITIC_POST_REVIEW",
                agent_role_override="critic",
                require_json=True,
                expected_json_type="object",
            )
            critique.raw_summary = raw
            parsed = self._parse_json_object(raw)
            if "needs_revision" in parsed:
                critique.needs_revision = bool(parsed["needs_revision"])
            critique.comparison_to_prev_gens = str(
                parsed.get("comparison_to_prev_gens") or ""
            )
            if isinstance(parsed.get("identified_issues"), list):
                critique.identified_issues = [
                    str(x) for x in parsed["identified_issues"] if str(x).strip()
                ]
            if isinstance(parsed.get("recommendations_for_meta"), list):
                critique.recommendations_for_meta = [
                    str(x) for x in parsed["recommendations_for_meta"] if str(x).strip()
                ]
        except Exception as exc:
            ui.warn(f"Critic Pass 2 fallback ({exc})")

        try:
            self.theories.save_document(
                generation=generation,
                agent_role="critic",
                doc_type="CRITIC_POST_REVIEW",
                title=f"Critic Post-Assembly Review G{generation} ({candidate_policy_id})",
                content=critique.to_prompt_block(),
                tags=["critic", f"gen_{generation}", "post_review"],
            )
        except Exception:
            pass
        return critique
