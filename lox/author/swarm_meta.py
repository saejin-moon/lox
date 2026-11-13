"""
Meta-Agent Coordinator for the LOX Multi-Agent Swarm.
Integrates reflection reports from specialized category workers (dialog, threat, survival, navigation),
queries DuckDB for causal metrics, and assembles the unified Behavior Tree policy.
"""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path
from typing import Any

from lox.author.agent import AuthorAgent
from lox.author.prompts import build_system_prompt
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
        self.author.tools.theories_store = self.theories

    def discover_macro_categories(
        self,
        generation: int,
        n_categories: int = 4,
        current_policy: str = "",
        skills: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Dynamically derives macro-categories for the upcoming generation via tool-enabled
        investigation of DuckDB, LanceDB, policy.py, and skills/ (zero pre-fed summaries).
        """
        ui.info(
            f"Meta-Agent: Dynamically analyzing DuckDB & codebase for Generation {generation}..."
        )
        self.author.tools.current_policy_code = str(current_policy or "")
        if skills is not None:
            self.author.tools.skills = dict(skills)
        self.author.tools.theories_store = self.theories

        frontier_depths = (
            sorted({cp.depth for cp in self.frontier.checkpoints.values()})
            if self.frontier.checkpoints
            else [1]
        )
        active_frontier_depth = max(frontier_depths) if frontier_depths else 1

        categories: list[dict[str, Any]] = []
        if self.provider != "mock":
            prompt = f"""Design {n_categories} specialist worker categories for Generation {generation}.

Do NOT assume what the bottlenecks or regressions are. Use your tools (`query_db`, `query_policy_docs`, `read_skill`, `query_theories`) to inspect the database and codebase yourself:
- Check `batches` across generations to see the full progression/regression trajectory.
- Check `runs` (including `summary_digest`, `end_reason`, `unique_tiles`, `pct_wait`, `pct_search`), `function_stats`, and `policies` to diagnose what is actually happening (if `runs` has 0 rows on Gen 1, check `expert_runs`, `query_wiki`, and `query_policy_docs` instead).
- Check `policy.py` (`query_policy_docs('policy')`) and `skills/` (`query_policy_docs('skills')` / `read_skill`) to see what already exists and where branches are failing or conflicting.
- Reuse an existing `category_id` (skill module name) when a worker should refine or fix an existing skill, or introduce a new `category_id` when a distinct capability is missing.

### Output Format:
After finishing your tool calls, return ONLY a valid JSON array of {n_categories} objects, each with:
- "category_id": short_snake_case_identifier
- "name": Human-readable title
- "description": Concrete empirical problem & scope for this specialist
- "target_depth_tier": [min_depth, max_depth]
- "target_metrics": list of telemetry column names (e.g. ["max_depth", "unique_tiles", "turns"])
- "blackboard_keys": list of 2-3 Blackboard variable names this branch should coordinate
"""
            try:
                raw_txt = self.author.run_tool_session(
                    system_prompt=build_system_prompt(self.db_path),
                    user_prompt=prompt,
                    max_tool_turns=5,
                )
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
                    "Maximizes map exploration and level transitions based on DuckDB telemetry.",
                    (1, active_frontier_depth + 1),
                    ["unique_tiles", "max_depth", "depth"],
                    ["target_tile", "frontier_path"],
                ),
                (
                    "risk_and_survival",
                    "Risk & Threat Response",
                    "Mitigates top termination causes and preserves HP under hostile encounters.",
                    (1, active_frontier_depth),
                    ["final_hp", "turns", "killed_monsters"],
                    ["danger_level", "last_target"],
                ),
                (
                    "resource_management",
                    "Resource & State Optimization",
                    "Improves resource utilization and endurance across recorded episodes.",
                    (1, active_frontier_depth),
                    ["turns", "final_ac", "xp_level"],
                    ["resource_needed", "emergency_action"],
                ),
                (
                    "interactive_state_handling",
                    "Interactive State & Stall Recovery",
                    "Resolves blocking prompts and zero-progress stalls cleanly.",
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
                "active_frontier_depth": active_frontier_depth,
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

    def verify_policy(
        self,
        code: str,
        steps: int = 10,
        skills: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """
        Validates Python syntax/AST, compiles the policy (with optional skill modules),
        and executes a smoke test in NetHackHarness for `steps` ticks.
        """
        try:
            validate_python_code(code)
            compiled = compile_policy_package(code, skills=skills or {})
            runner = compiled.create_runner()
            harness = NetHackHarness(max_steps=steps + 5)
            try:
                st = harness.reset(seed=42)
                non_wait_actions = 0
                steps_run = 0
                for _ in range(steps):
                    act, _ = runner.next_action(st)
                    if act is None:
                        act = Action.WAIT
                    if act != Action.WAIT:
                        non_wait_actions += 1
                    steps_run += 1
                    st, done, _ = harness.step(act)
                    if done or harness.done:
                        break
                if steps_run >= 10 and non_wait_actions == 0 and not harness.done:
                    return (
                        False,
                        f"Policy returned only None/Action.WAIT for all {steps_run} smoke-check steps (no active branch fired).",
                    )
                return True, ""
            finally:
                harness.close()
        except Exception as e:
            tb = traceback.format_exc()
            return False, f"{type(e).__name__}: {e}\n{tb}"

    @staticmethod
    def _build_skill_modules(
        worker_reports: dict[str, dict[str, Any]],
        skills_dir: str | Path | None = None,
    ) -> dict[str, str]:
        """
        Extracts each worker's winning subtree into a standalone skill module under `skills/<role>.py`
        so `policy.py` stays a lean Behavior Tree coordinator importing `skills.<role>`.
        """
        from pathlib import Path

        skill_modules: dict[str, str] = {}
        out_dir = Path(skills_dir) if skills_dir else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            for existing in out_dir.glob("*.py"):
                if existing.stem != "__init__":
                    skill_modules[existing.stem] = existing.read_text(encoding="utf-8")

        header = (
            "import bisect\n"
            "import collections\n"
            "import copy\n"
            "import heapq\n"
            "import itertools\n"
            "import math\n"
            "import re\n"
            "import numpy as np\n"
            "from lox.env.actions import Action, DIR_DELTAS, seq, key, repeat\n"
            "from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, ConditionLeaf, Inverter, PrioritySelector, Sequence\n\n"
        )
        for role, rep in worker_reports.items():
            raw_code = str(rep.get("best_code") or rep.get("content") or "").strip()
            if not raw_code or "class Policy" in raw_code:
                continue
            safe_name = (
                re.sub(r"[^a-zA-Z0-9_]", "_", str(role).lower()).strip("_")
                or "skill_node"
            )
            mod_code = f"# Auto-generated modular skill for specialist `{role}`\n{header}{raw_code}\n"
            try:
                validate_python_code(mod_code)
            except Exception:
                continue
            skill_modules[safe_name] = mod_code
            if out_dir is not None:
                (out_dir / f"{safe_name}.py").write_text(mod_code, encoding="utf-8")
        return skill_modules

    def _query_function_telemetry(self) -> str:
        """Queries DuckDB for the previous generation's function turn shares."""
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
                if prev_gen is None:
                    return ""
                fn_rows = con.execute(
                    """
                    SELECT function_name, episodes_triggered, total_turns_active,
                           coalesce(liblit_failure_score, 0.0),
                           coalesce(stagnation_share, 0.0),
                           coalesce(livelock_score, 0.0)
                    FROM function_stats
                    WHERE generation = ? AND total_turns_active > 0
                    ORDER BY total_turns_active DESC
                    LIMIT 12
                    """,
                    [prev_gen],
                ).fetchall()
                tot_turns = sum(int(r[2]) for r in fn_rows) or 1
                if not fn_rows:
                    return ""
                fn_lines = "\n".join(
                    f"- `{r[0]}`: {r[2]} turns ({100.0 * int(r[2]) / tot_turns:.1f}% share), "
                    f"episodes={r[1]}, stagnation_share={float(r[4]):.2f}, "
                    f"livelock={float(r[5]):.2f}, liblit_fail={float(r[3]):.3f}"
                    for r in fn_rows
                )
                return (
                    f"\n### Function Telemetry (Gen {prev_gen} — separate healthy workhorse share from stagnation/livelock/failure):\n"
                    f"{fn_lines}\n"
                )
            finally:
                con.close()
        except Exception:
            return ""

    def assemble_policy(
        self,
        generation: int,
        worker_reports: dict[str, dict[str, Any]] | None = None,
        current_policy: str = "",
        max_repair_turns: int = 3,
        skills_dir: str | Any = None,
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
        skill_modules = self._build_skill_modules(worker_reports, skills_dir=skills_dir)
        if self.provider == "mock":
            master_policy_code = default_template
            is_valid, diag = self.verify_policy(
                master_policy_code, skills=skill_modules
            )
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
                    reports_summary.append(
                        f"### Specialist Worker: {role}\n{rep_doc}\n"
                    )
                else:
                    reports_summary.append(
                        f"### Specialist Worker: {role}\n"
                        f"**Empirical Verification**: {obs_text}\n"
                        f"```python\n{code_snip}\n```\n"
                    )

            context = "\n".join(reports_summary)
            meta_prompt = (
                f"You are the Meta-Agent Architect synthesizing the Generation {generation} master `class Policy`.\n"
                f"Do NOT rely solely on worker proposals—use `query_db`, `read_skill`, `query_policy_docs`, and `query_theories` "
                f"to inspect `batches`, `runs` (including `summary_digest`), `function_stats`, `policies`, and `skills/` yourself "
                f"to see what is working, what regressed, and how branches interact.\n\n"
                f"Specialist subtrees are saved in `skills/` (see Hierarchical Skill Registry):\n\n{context}\n\n"
                f"Ensure the master policy defines `class Policy` with `def __init__(self)` and `def step(self, state): return self.tree.tick(state)`."
            )

            proposal = self.author.synthesize(
                mode="SWARM_META",
                generation=generation,
                current_policy=current_policy or default_template,
                batch_digest=meta_prompt,
                skills=skill_modules,
                max_tool_turns=10,
            )
            candidate_code = proposal.full_policy_code or proposal.raw_code_patch

            # Live verification gate with self-repair turns
            is_valid, diag = self.verify_policy(candidate_code, skills=skill_modules)
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
                    batch_digest=f"Self-Repair Required: Master policy failed verification:\n{diag}\n\n---\n{meta_prompt}",
                    skills=skill_modules,
                    max_tool_turns=4,
                )
                candidate_code = (
                    repair_proposal.full_policy_code or repair_proposal.raw_code_patch
                )
                is_valid, diag = self.verify_policy(
                    candidate_code, skills=skill_modules
                )

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
