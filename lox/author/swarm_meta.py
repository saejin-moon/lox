"""
Meta-Agent Coordinator for the LOX Multi-Agent Swarm.
Integrates reflection reports from specialized category workers (dialog, threat, survival, navigation),
queries DuckDB for causal metrics, and assembles the unified Behavior Tree policy.
"""

from __future__ import annotations

import json
import traceback
from typing import Any

from lox.author.agent import AuthorAgent
from lox.db.schema import connect_db
from lox.db.theories import TheoryDocumentStore
from lox.engine.frontier import FrontierArchive
from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.lang.compiler import compile_policy_package
from lox.lang.validator import validate_python_code
from lox.ui.console import ui


class SwarmMetaAgent:
    """Lead Architect agent that coordinates the swarm and integrates subtrees into master BehaviorTree."""

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "z-ai/glm-5.3-flash:nitro",
        db_path: str = "data/lox.duckdb",
        theories_path: str = "data/workspace/theories.lance",
        frontier_dir: str = "data/frontier",
    ) -> None:
        self.provider = provider
        self.model = model
        self.db_path = db_path
        self.theories = TheoryDocumentStore(db_path=theories_path)
        self.frontier = FrontierArchive(frontier_dir=frontier_dir)
        self.author = AuthorAgent(
            provider=provider,
            model=model,
            db_path=db_path,
            agent_role="meta_agent",
        )

    def discover_macro_categories(
        self,
        generation: int,
        n_categories: int = 4,
    ) -> list[dict[str, Any]]:
        """
        Dynamically derives macro-categories for the upcoming generation by querying DuckDB
        for current failure bottlenecks, frontier depth progression, and expert run references.
        'Hard Code Nothing': categories change dynamically as LOX penetrates deeper dungeon tiers.
        """
        ui.info(
            f"Meta-Agent: Dynamically analyzing DuckDB bottlenecks for Generation {generation}..."
        )

        avg_depth = 1.0
        max_depth = 1
        avg_turns = 0.0
        avg_tiles = 0.0
        mortality_causes: list[tuple[str, str, int]] = []
        top_fn_turns: list[tuple[str, int, int, float]] = []
        auto_escs = 0
        died_unarmored = 0
        starved_with_food = 0
        latest_gen: int | None = None

        # Query DuckDB for recent empirical performance & function turn distribution
        try:
            con = connect_db(self.db_path, read_only=True)
            try:
                max_gen_row = con.execute("SELECT max(generation) FROM runs").fetchone()
                latest_gen = (
                    max_gen_row[0]
                    if max_gen_row and max_gen_row[0] is not None
                    else None
                )

                if latest_gen is not None:
                    mort_rows = con.execute(
                        """
                        SELECT end_reason, death_message, count(*) as cnt
                        FROM runs
                        WHERE generation = ?
                        GROUP BY end_reason, death_message
                        ORDER BY cnt DESC
                        LIMIT 6
                        """,
                        [latest_gen],
                    ).fetchall()
                    mortality_causes = [
                        (str(r[0]), str(r[1]), int(r[2])) for r in mort_rows
                    ]

                    metrics_row = con.execute(
                        """
                        SELECT
                            coalesce(avg(depth), 1.0),
                            coalesce(max(depth), 1),
                            coalesce(avg(turns), 0.0),
                            coalesce(avg(unique_tiles), 0.0),
                            coalesce(sum(case when auto_esc_count > 0 then 1 else 0 end), 0),
                            coalesce(sum(case when died_unarmored then 1 else 0 end), 0),
                            coalesce(sum(case when starved_with_food then 1 else 0 end), 0)
                        FROM runs
                        WHERE generation = ?
                        """,
                        [latest_gen],
                    ).fetchone()
                    if metrics_row:
                        avg_depth = float(metrics_row[0])
                        max_depth = int(metrics_row[1])
                        avg_turns = float(metrics_row[2])
                        avg_tiles = float(metrics_row[3])
                        auto_escs = int(metrics_row[4])
                        died_unarmored = int(metrics_row[5])
                        starved_with_food = int(metrics_row[6])

                    fn_rows = con.execute(
                        """
                        SELECT function_name, episodes_triggered, total_turns_active,
                               coalesce(liblit_failure_score, 0.0)
                        FROM function_stats
                        WHERE generation = ? AND total_turns_active > 0
                        ORDER BY total_turns_active DESC
                        LIMIT 10
                        """,
                        [latest_gen],
                    ).fetchall()
                    top_fn_turns = [
                        (str(r[0]), int(r[1]), int(r[2]), float(r[3])) for r in fn_rows
                    ]
            finally:
                con.close()
        except Exception:
            pass

        frontier_depths = (
            sorted({cp.depth for cp in self.frontier.checkpoints.values()})
            if self.frontier.checkpoints
            else [1]
        )
        active_frontier_depth = max(
            max_depth, max(frontier_depths) if frontier_depths else 1
        )

        categories: list[dict[str, Any]] = []
        if self.provider != "mock":
            # Dynamically synthesize orthogonal specialist categories via Meta-Agent LLM
            recent_decomps = self.theories.search_theories(
                "CATEGORY_DECOMPOSITION", limit=2, doc_type="CATEGORY_DECOMPOSITION"
            )
            prev_cats_note = (
                "\n".join(
                    f"- Gen {d.get('generation')}: {str(d.get('content', ''))[:400]}"
                    for d in recent_decomps
                )
                if recent_decomps
                else "None yet."
            )
            prompt = f"""Analyze the empirical telemetry from Generation {latest_gen if latest_gen is not None else 0} and design {n_categories} orthogonal specialist worker categories for Generation {generation}.

### Empirical Telemetry:
- avg_depth={avg_depth:.2f}, max_depth={max_depth}, active_frontier_depth={active_frontier_depth}
- avg_turns={avg_turns:.1f}, avg_unique_tiles={avg_tiles:.1f}
- diagnostic_flags: auto_escs={auto_escs}, died_unarmored={died_unarmored}, starved_with_food={starved_with_food}
- top_end_reasons: {mortality_causes}
- top_active_functions (name, episodes, turns_active, liblit_fail): {top_fn_turns}

### Previous Category Decompositions (avoid repeating stale categories if bottlenecks shifted):
{prev_cats_note}

### Output Format:
Return ONLY a valid JSON array of {n_categories} objects, each with:
- "category_id": short_snake_case_identifier (unique, descriptive of the specific empirical bottleneck)
- "name": Human-readable title
- "description": Concrete empirical problem & scope for this specialist (without hardcoding unverified game rules)
- "target_depth_tier": [min_depth, max_depth]
- "target_metrics": list of telemetry column names (e.g. ["max_depth", "unique_tiles", "turns"])
- "blackboard_keys": list of 2-3 Blackboard variable names this branch should coordinate
"""
            try:
                resp = self.author._call_llm_api(
                    [
                        {
                            "role": "system",
                            "content": "You are the LOX Meta-Agent Architect. Output strictly valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    allow_tools=False,
                )
                raw_txt = str(resp.get("content") or "").strip()
                if "```" in raw_txt:
                    import re

                    m = re.search(
                        r"```(?:json)?\s*\n(.*?)```", raw_txt, flags=re.DOTALL
                    )
                    if m:
                        raw_txt = m.group(1).strip()
                parsed = json.loads(raw_txt)
                if isinstance(parsed, list):
                    for item in parsed[:n_categories]:
                        if isinstance(item, dict) and item.get("category_id"):
                            tier = item.get("target_depth_tier") or [
                                1,
                                active_frontier_depth,
                            ]
                            if isinstance(tier, list) and len(tier) >= 2:
                                tier_tup = (int(tier[0]), int(tier[1]))
                            else:
                                tier_tup = (1, active_frontier_depth)
                            categories.append(
                                {
                                    "category_id": str(item["category_id"])
                                    .strip()
                                    .lower()
                                    .replace(" ", "_"),
                                    "name": str(
                                        item.get("name") or item["category_id"]
                                    ),
                                    "description": str(item.get("description") or ""),
                                    "target_depth_tier": tier_tup,
                                    "target_metrics": [
                                        str(m)
                                        for m in (
                                            item.get("target_metrics")
                                            or ["max_depth", "unique_tiles"]
                                        )
                                    ],
                                    "blackboard_keys": [
                                        str(k)
                                        for k in (item.get("blackboard_keys") or [])
                                    ],
                                }
                            )
            except Exception as exc:
                ui.warn(f"Meta-Agent LLM category discovery fallback ({exc})")

        if not categories:
            # Domain-agnostic structural fallback (mock mode or LLM parse failure)
            generic_specs = [
                (
                    "progression_and_exploration",
                    "Progression & Spatial Exploration",
                    f"Maximizes map exploration and level transitions (current avg_tiles={avg_tiles:.1f}, avg_depth={avg_depth:.2f}).",
                    (1, active_frontier_depth + 1),
                    ["unique_tiles", "max_depth", "depth"],
                    ["target_tile", "frontier_path"],
                ),
                (
                    "risk_and_survival",
                    "Risk & Threat Response",
                    f"Mitigates top termination causes ({mortality_causes[:2]}) and preserves HP under hostile encounters.",
                    (1, active_frontier_depth),
                    ["final_hp", "turns", "killed_monsters"],
                    ["danger_level", "last_target"],
                ),
                (
                    "resource_management",
                    "Resource & State Optimization",
                    f"Improves resource utilization and endurance (avg_turns={avg_turns:.1f}, starved_with_food={starved_with_food}, died_unarmored={died_unarmored}).",
                    (1, active_frontier_depth),
                    ["turns", "final_ac", "xp_level"],
                    ["resource_needed", "emergency_action"],
                ),
                (
                    "interactive_state_handling",
                    "Interactive State & Stall Recovery",
                    f"Resolves blocking prompts and zero-progress stalls cleanly (auto_escs={auto_escs}).",
                    (1, max(2, active_frontier_depth)),
                    ["auto_esc_count", "stagnation_turns"],
                    ["in_prompt", "stall_recovery"],
                ),
            ]
            for cid, cname, cdesc, ctier, cmetrics, cbkeys in generic_specs[
                :n_categories
            ]:
                categories.append(
                    {
                        "category_id": cid,
                        "name": cname,
                        "description": cdesc,
                        "target_depth_tier": ctier,
                        "target_metrics": cmetrics,
                        "blackboard_keys": cbkeys,
                    }
                )

        categories = categories[:n_categories]

        # Save decomposition to LanceDB
        decomp_json = json.dumps(
            {
                "generation": generation,
                "avg_depth": avg_depth,
                "max_depth": max_depth,
                "active_frontier_depth": active_frontier_depth,
                "mortality_causes": mortality_causes,
                "categories": categories,
            },
            indent=2,
        )
        content = f"# Dynamic Category Decomposition Plan (Gen {generation})\n\n```json\n{decomp_json}\n```"
        doc_id = self.theories.save_document(
            generation=generation,
            agent_role="meta_agent",
            doc_type="CATEGORY_DECOMPOSITION",
            title=f"Meta-Agent Dynamic Category Decomposition G{generation}",
            content=content,
            tags=["meta_agent", f"gen_{generation}", "decomposition"],
        )
        ui.info(
            f"Meta-Agent: Discovered {len(categories)} dynamic categories for Gen {generation} (Doc: {doc_id}): "
            f"{[c['category_id'] for c in categories]}"
        )
        return categories

    def verify_policy(self, code: str, steps: int = 10) -> tuple[bool, str]:
        """
        Validates Python syntax/AST, compiles the policy, and executes a smoke test
        in NetHackHarness for `steps` ticks to verify runtime stability.
        Returns (is_valid, diagnostic_error_string).
        """
        try:
            validate_python_code(code)
            compiled = compile_policy_package(code)
            runner = compiled.create_runner()
            harness = NetHackHarness(max_steps=steps + 5)
            try:
                st = harness.reset(seed=42)
                for _ in range(steps):
                    act, _ = runner.next_action(st)
                    if act is None:
                        act = Action.SEARCH
                    st, done, _ = harness.step(act)
                    if done or harness.done:
                        break
                return True, ""
            finally:
                harness.close()
        except Exception as e:
            tb = traceback.format_exc()
            return False, f"{type(e).__name__}: {e}\n{tb}"

    def assemble_policy(
        self,
        generation: int,
        worker_reports: dict[str, dict[str, Any]] | None = None,
        current_policy: str = "",
        max_repair_turns: int = 3,
    ) -> str:
        """
        Gathers worker reports from LanceDB, checks DuckDB metrics, and synthesizes the master Policy code.
        Runs compiler-in-the-loop and live NetHackHarness verification with up to `max_repair_turns`
        self-repair turns if candidate code fails.
        """
        if worker_reports is None:
            worker_reports = self.theories.get_latest_reports(generation=generation)

        ui.info(
            f"Meta-Agent: Assembling Generation {generation} master policy from "
            f"{len(worker_reports)} worker report(s): {list(worker_reports.keys())}"
        )

        # Standard clean Behavior Tree master template
        default_template = f"""# Master Behavior Tree Policy - Synthesized by Meta-Agent (Gen {generation})
import numpy as np
from lox.env.actions import Action, DIR_DELTAS, seq
from lox.lang.bt import BehaviorTree, PrioritySelector, Sequence, ActionLeaf, ConditionLeaf, Blackboard

class Policy:
    def __init__(self):
        self.blackboard = Blackboard()

        # Priority 1: Interactive Prompts & --More-- Pagination
        def dialog_action(state):
            if getattr(state, "waiting_for_space", False):
                return Action.SPACE
            if getattr(state, "in_text_prompt", False) or getattr(state, "in_yn_prompt", False):
                return Action.ESC
            return None
        self.node_dialog = ActionLeaf(dialog_action, name="DialogLeaf")

        # Priority 2: Tactical Threat & Melee Combat
        def threat_action(state, bb):
            if hasattr(state, "adjacent_monsters"):
                hostiles = state.adjacent_monsters(exclude_pets=True)
                if hostiles:
                    m = hostiles[0]
                    bb.set_var("last_target", (m.y, m.x))
                    return DIR_DELTAS.get((m.y - state.y, m.x - state.x), Action.WAIT)
            return None
        self.node_threat = ActionLeaf(threat_action, name="ThreatLeaf")

        # Priority 3: Survival & Nutrition
        def survival_action(state, bb):
            return None
        self.node_survival = ActionLeaf(survival_action, name="SurvivalLeaf")

        # Priority 4: Spatial Exploration & Stair Descent
        def navigation_action(state, bb):
            if state.char_at(state.y, state.x) == ">":
                return Action.DOWN
            stairs = np.argwhere(state.chars == ord(">"))
            if len(stairs) > 0:
                ty, tx = stairs[0]
                dy = int(np.sign(ty - state.y))
                dx = int(np.sign(tx - state.x))
                return DIR_DELTAS.get((dy, dx), Action.SEARCH)
            return Action.SEARCH
        self.node_navigation = ActionLeaf(navigation_action, name="NavigationLeaf")

        # Assemble Master Root Priority Selector
        root = PrioritySelector([
            self.node_dialog,
            self.node_threat,
            self.node_survival,
            self.node_navigation,
        ], name="MasterSelector")

        self.tree = BehaviorTree(root=root, blackboard=self.blackboard)

    def step(self, state):
        return self.tree.tick(state)
"""
        if self.provider == "mock":
            master_policy_code = default_template
            is_valid, diag = self.verify_policy(master_policy_code)
            if not is_valid:
                raise RuntimeError(
                    f"Default master template failed verification: {diag}"
                )
        else:
            # Build synthesis prompt summarizing worker discoveries & comparative hypotheses
            reports_summary = []
            for role, rep in worker_reports.items():
                code_snip = rep.get("best_code", "")
                obs = rep.get("observation")
                obs_text = (
                    obs.summary() if hasattr(obs, "summary") else str(obs or "verified")
                )
                rep_doc = str(rep.get("report_content") or "").strip()
                if rep_doc:
                    if len(rep_doc) > 3500:
                        rep_doc = rep_doc[:3500] + "\n... [truncated]"
                    reports_summary.append(
                        f"### Specialist Worker: {role}\n{rep_doc}\n"
                    )
                else:
                    reports_summary.append(
                        f"### Specialist Worker: {role}\n"
                        f"**Empirical Verification**: {obs_text}\n"
                        f"```python\n{code_snip}\n```\n"
                    )
            fn_telemetry_block = ""
            try:
                con = connect_db(self.db_path, read_only=True)
                try:
                    prev_gen_row = con.execute(
                        "SELECT max(generation) FROM function_stats"
                    ).fetchone()
                    prev_gen = (
                        prev_gen_row[0]
                        if prev_gen_row and prev_gen_row[0] is not None
                        else None
                    )
                    if prev_gen is not None:
                        fn_rows = con.execute(
                            """
                            SELECT function_name, episodes_triggered, total_turns_active,
                                   coalesce(liblit_failure_score, 0.0)
                            FROM function_stats
                            WHERE generation = ? AND total_turns_active > 0
                            ORDER BY total_turns_active DESC
                            LIMIT 12
                            """,
                            [prev_gen],
                        ).fetchall()
                        tot_turns = sum(int(r[2]) for r in fn_rows) or 1
                        if fn_rows:
                            fn_lines = "\n".join(
                                f"- `{r[0]}`: {r[2]} turns ({100.0 * int(r[2]) / tot_turns:.1f}% share), "
                                f"episodes={r[1]}, liblit_fail={float(r[3]):.3f}"
                                for r in fn_rows
                            )
                            fn_telemetry_block = (
                                f"\n### Function Turn-Share Telemetry (Gen {prev_gen} — prune/refactor nodes >30% if hijacking or looping):\n"
                                f"{fn_lines}\n"
                            )
                finally:
                    con.close()
            except Exception:
                pass

            context = "\n".join(reports_summary)
            meta_prompt = (
                f"You are the Meta-Agent Architect. Synthesize a unified `class Policy` utilizing Behavior Trees "
                f"(`BehaviorTree`, `PrioritySelector`, `ActionLeaf`, `ConditionLeaf`, `Blackboard`) for Generation {generation}.\n"
                f"{fn_telemetry_block}\n"
                f"Integrate the specialized subtrees and findings from these category workers, while pruning any dead or hijacking branches:\n\n{context}\n\n"
                f"Ensure the master policy defines `class Policy` with `def __init__(self)` and `def step(self, state): return self.tree.tick(state)`."
            )

            proposal = self.author.synthesize(
                mode="SWARM_META",
                generation=generation,
                current_policy=current_policy or default_template,
                batch_digest=meta_prompt,
                max_tool_turns=6,
            )
            candidate_code = proposal.full_policy_code or proposal.raw_code_patch

            # Live verification gate with self-repair turns
            is_valid, diag = self.verify_policy(candidate_code)
            repair_turns = 0
            while not is_valid and repair_turns < max_repair_turns:
                repair_turns += 1
                ui.warn(
                    f"Meta-Agent: Policy verification failed; executing self-repair turn ({repair_turns}/{max_repair_turns})...\nDiagnostic: {diag[:300]}"
                )
                repair_proposal = self.author.synthesize(
                    mode="SWARM_META",
                    generation=generation,
                    current_policy=candidate_code,
                    batch_digest=f"Self-Repair Required: Master policy failed verification:\n{diag}",
                    max_tool_turns=2,
                )
                candidate_code = (
                    repair_proposal.full_policy_code or repair_proposal.raw_code_patch
                )
                is_valid, diag = self.verify_policy(candidate_code)

            if is_valid:
                master_policy_code = candidate_code
                ui.info(
                    f"Meta-Agent: Successfully assembled and verified master Behavior Tree policy (Repairs: {repair_turns})."
                )
            else:
                ui.warn(
                    f"Meta-Agent: Repairs exhausted; falling back to default Behavior Tree template. Error was: {diag[:200]}"
                )
                master_policy_code = default_template

        # Save synthesis reflection into LanceDB
        self.theories.save_document(
            generation=generation,
            agent_role="meta_agent",
            doc_type="META_SYNTHESIS",
            title=f"Meta-Agent Policy Assembly G{generation}",
            content=f"### Master Policy Code\n```python\n{master_policy_code}\n```",
            tags=["meta_agent", f"gen_{generation}", "assembly"],
        )

        return master_policy_code
