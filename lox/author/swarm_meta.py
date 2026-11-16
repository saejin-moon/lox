"""
Meta-Agent Coordinator for the LOX Multi-Agent Swarm ("Hard Code Nothing").
Receives Pass 1 high-level guidance from SwarmCriticAgent, dynamically discovers
specialist worker categories, integrates worker subtrees into `skills/` and `policy.py`,
and receives Pass 2 comparative feedback from SwarmCriticAgent to finalize the policy.
"""

from __future__ import annotations

import ast
import json
import re
import traceback
from pathlib import Path
from typing import Any

from lox.author.agent import AuthorAgent
from lox.author.prompts import build_system_prompt
from lox.author.swarm_critic import build_swarm_diagnostic_digest
from lox.db.theories import TheoryDocumentStore
from lox.engine.frontier import FrontierArchive
from lox.env.actions import Action
from lox.env.harness import NetHackHarness
from lox.lang.bt import extract_runner_leaf_errors
from lox.lang.compiler import compile_policy_package
from lox.lang.validator import is_ast_equivalent, validate_python_code
from lox.ui.console import ui


def _minimal_bt_stub(generation: int) -> str:
    """Minimal domain-agnostic BehaviorTree coordinator stub (used for mock mode or empty bootstrap fallback)."""
    return f"""# Master Behavior Tree Policy (Gen {generation})
from lox.env.actions import Action
from lox.lang.bt import ActionLeaf, BehaviorTree, Blackboard, PrioritySelector


class Policy:
    def __init__(self):
        self.blackboard = Blackboard()

        def explore_step(state, bb):
            return Action.SEARCH

        root = PrioritySelector(
            [ActionLeaf(explore_step, name="ExploreLeaf")],
            name="MasterSelector",
        )
        self.tree = BehaviorTree(root=root, blackboard=self.blackboard)

    def step(self, state):
        return self.tree.tick(state) or Action.SEARCH
"""


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

    @staticmethod
    def _parse_categories_json(raw_txt: str) -> list[dict[str, Any]]:
        """Robustly extracts a JSON array of category dicts from LLM output."""
        txt = re.sub(
            r"<think>.*?</think>", "", str(raw_txt or ""), flags=re.DOTALL
        ).strip()
        if "```" in txt:
            m = re.search(r"```(?:json)?\s*\n(.*?)```", txt, flags=re.DOTALL)
            if m:
                txt = m.group(1).strip()
        for candidate in (txt,):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return [x for x in parsed if isinstance(x, dict)]
                if isinstance(parsed, dict) and isinstance(
                    parsed.get("categories"), list
                ):
                    return [x for x in parsed["categories"] if isinstance(x, dict)]
            except Exception:
                pass
        m_arr = re.search(r"\[.*\]", txt, flags=re.DOTALL)
        if m_arr:
            try:
                parsed = json.loads(m_arr.group(0))
                if isinstance(parsed, list):
                    return [x for x in parsed if isinstance(x, dict)]
            except Exception:
                pass
        return []

    def discover_macro_categories(
        self,
        generation: int,
        n_categories: int = 4,
        current_policy: str = "",
        skills: dict[str, str] | None = None,
        pre_gen_guide: Any = None,
        diagnostic_report: Any = None,
    ) -> list[dict[str, Any]]:
        """
        Dynamically derives macro-categories for the upcoming generation via tool-enabled
        investigation of DuckDB, LanceDB, policy.py, and skills/ informed by Critic Pass 1.
        """
        guide = pre_gen_guide if pre_gen_guide is not None else diagnostic_report
        ui.info(
            f"Meta-Agent: Dynamically analyzing DuckDB & codebase for Generation {generation}..."
        )
        self.author.tools.current_policy_code = str(current_policy or "")
        self.author.tools.generation = int(generation)
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
            guide_block = (
                f"\n{guide.to_prompt_block()}\n"
                if guide is not None and hasattr(guide, "to_prompt_block")
                else ""
            )
            empty_campaign = self.author.tools._campaign_tables_empty()
            db_guidance = (
                "Campaign tables (`runs`, `batches`, `ticks`, `function_stats`, `policies`) are currently empty — "
                "call BOTH `query_wiki` and `query_db` on `expert_runs` / `expert_ticks` (real human ascensions)."
                if empty_campaign
                else "Inspect `batches`, `runs`, `function_stats`, `policies`, `expert_runs`, and `query_wiki` / `query_monster` / `query_item`."
            )
            prompt = f"""Design {n_categories} specialist worker categories for Generation {generation}.
{guide_block}
Do NOT assume what the bottlenecks or regressions are. Follow this investigation protocol:
1. Call `query_db` (pass ONE `SELECT` statement per call — never multiple `;`-separated queries) and `query_wiki` / `query_monster` / `query_item`:
   - {db_guidance}
2. Check `policy.py` (`query_policy_docs('policy')`) and `skills/` (`query_policy_docs('skills')` / `read_skill`) to see what already exists and where branches are failing, starving, or conflicting.
3. Reuse an existing `category_id` (skill module name) when a worker should refine or fix an existing skill, or introduce a new `category_id` when a distinct capability is missing.

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
                    system_prompt=build_system_prompt(self.db_path, is_swarm=True),
                    user_prompt=prompt,
                    max_tool_turns=5,
                    required_tools=(
                        ["query_wiki", "query_db:expert_runs"]
                        if empty_campaign
                        else None
                    ),
                    generation=generation,
                    session_mode="META_DISCOVERY",
                    agent_role_override="meta_agent",
                    require_json=True,
                    expected_json_type="array",
                )
                parsed_items = self._parse_categories_json(raw_txt)
                for item in parsed_items[:n_categories]:
                    if item.get("category_id"):
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
                                "name": str(item.get("name") or item["category_id"]),
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
                                    str(k) for k in (item.get("blackboard_keys") or [])
                                ],
                            }
                        )
            except Exception as exc:
                ui.warn(f"Meta-Agent LLM category discovery fallback ({exc})")

        if not categories:
            # Neutral domain-agnostic structural fallback (mock mode or LLM parse failure)
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
                    "threat_and_survival",
                    "Threat Response & Survival",
                    "Mitigates top termination causes and preserves HP under encounters.",
                    (1, active_frontier_depth),
                    ["final_hp", "turns", "max_depth"],
                    ["active_threat", "last_target"],
                ),
                (
                    "resource_and_equipment",
                    "Resource & Equipment Management",
                    "Improves resource utilization, nutrition, and equipment state across episodes.",
                    (1, active_frontier_depth),
                    ["turns", "final_ac", "xp_level"],
                    ["resource_state", "equipped_state"],
                ),
                (
                    "environment_interaction",
                    "Environment & Feature Interaction",
                    "Handles environmental features, transitions, and interactive prompts cleanly.",
                    (1, max(2, active_frontier_depth)),
                    ["max_depth", "unique_tiles", "auto_esc_count"],
                    ["feature_target", "prompt_state"],
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
        executes a smoke test in NetHackHarness for `steps` ticks, and verifies that
        no BehaviorTree leaf threw a swallowed runtime exception on `blackboard.leaf_errors`.
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
                leaf_errors = extract_runner_leaf_errors(runner)
                if leaf_errors:
                    err_fmt = "; ".join(f"{n}: {e}" for n, e in leaf_errors[:5])
                    return (
                        False,
                        f"BehaviorTree leaf raised runtime exception during verification: {err_fmt}",
                    )
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
                probe_policy = (
                    "from lox.env.actions import Action\n"
                    "class Policy:\n"
                    "    def step(self, state):\n"
                    "        return Action.WAIT\n"
                )
                compile_policy_package(
                    probe_policy, skills={**skill_modules, safe_name: mod_code}
                ).create_runner()
            except Exception as exc:
                ui.warn(
                    f"Meta-Agent: Skipping broken skill module `{safe_name}` ({type(exc).__name__}: {exc})"
                )
                continue
            skill_modules[safe_name] = mod_code
            if out_dir is not None:
                (out_dir / f"{safe_name}.py").write_text(mod_code, encoding="utf-8")
        return skill_modules

    @staticmethod
    def _apply_proposal_skills(
        proposal: Any,
        skill_modules: dict[str, str],
        skills_dir: str | Path | None = None,
    ) -> None:
        """Applies any `<skill_code module="...">` updates emitted by the Meta-Agent."""
        updates: dict[str, str] = {}
        if getattr(proposal, "skill_modules", None):
            updates.update(proposal.skill_modules)
        s_name_single = getattr(proposal, "skill_name", None) or (
            (getattr(proposal, "hypothesis", None) or {}).get("skill_module")
        )
        s_code_single = getattr(proposal, "skill_code", "")
        if s_name_single and s_code_single:
            clean_s = re.sub(r"[^a-zA-Z0-9_]", "_", str(s_name_single).lower()).strip(
                "_"
            )
            if clean_s:
                updates[clean_s] = s_code_single

        for s_name, s_code in updates.items():
            try:
                validate_python_code(s_code, filename=f"skills/{s_name}.py")
            except Exception as exc:
                ui.warn(f"Meta-Agent: Skipping invalid skill update `{s_name}` ({exc})")
                continue
            skill_modules[s_name] = s_code
            if skills_dir is not None:
                Path(skills_dir).mkdir(parents=True, exist_ok=True)
                (Path(skills_dir) / f"{s_name}.py").write_text(s_code, encoding="utf-8")

    def assemble_policy(
        self,
        generation: int,
        worker_reports: dict[str, dict[str, Any]] | None = None,
        current_policy: str = "",
        max_repair_turns: int = 3,
        skills_dir: str | Any = None,
        eval_pool: Any = None,
        baseline_runs: list[dict[str, Any]] | None = None,
        pre_gen_guide: Any = None,
        diagnostic_report: Any = None,
    ) -> str:
        """
        Gathers worker reports from LanceDB, checks DuckDB metrics & Critic Pass 1 guide,
        and synthesizes the draft master Policy code.
        """
        guide = pre_gen_guide if pre_gen_guide is not None else diagnostic_report
        if worker_reports is None:
            worker_reports = self.theories.get_latest_reports(generation=generation)

        ui.info(
            f"Meta-Agent: Assembling Generation {generation} master policy from "
            f"{len(worker_reports)} worker report(s): {list(worker_reports.keys())}"
        )

        fallback_code = (
            current_policy.strip()
            if current_policy and current_policy.strip()
            else _minimal_bt_stub(generation)
        )
        skill_modules = self._build_skill_modules(worker_reports, skills_dir=skills_dir)
        if self.provider == "mock":
            master_policy_code = fallback_code
            is_valid, diag = self.verify_policy(
                master_policy_code, skills=skill_modules
            )
            if not is_valid:
                master_policy_code = _minimal_bt_stub(generation)
        else:
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
            guide_block = (
                f"\n{guide.to_prompt_block()}\n\n"
                if guide is not None and hasattr(guide, "to_prompt_block")
                else ""
            )
            if not guide_block:
                auto_digest = build_swarm_diagnostic_digest(
                    self.db_path,
                    current_policy=current_policy,
                    target_generation=generation - 1 if generation > 1 else None,
                )
                if auto_digest:
                    guide_block = f"\n{auto_digest}\n\n"

            meta_prompt = (
                f"You are the Meta-Agent Architect synthesizing the Generation {generation} master `class Policy`.\n"
                f"{guide_block}"
                f"Do NOT rely solely on worker proposals—use `query_db`, `read_policy`, `diff_policies`, `read_skill(skill_name, policy_id=...)`, "
                f"`query_wiki`, `query_monster`, `query_item`, and `query_policy_docs` (over `query_theories`) to inspect "
                f"`batches`, `runs` (`summary_digest`), `function_stats`, `policies`, `expert_runs`, and `skills/` yourself "
                f"to see what is working, what regressed, and how branches interact.\n"
                f'You can also emit `<skill_code module="name">...</skill_code>` if you want to directly repair or restore a skill module.\n\n'
                f"Specialist subtrees are saved in `skills/` (see Hierarchical Skill Registry):\n\n{context}\n\n"
                f"Ensure the master policy defines `class Policy` with `def __init__(self)` and `def step(self, state): return self.tree.tick(state)`."
            )

            proposal = self.author.synthesize(
                mode="SWARM_META",
                generation=generation,
                current_policy=current_policy,
                batch_digest=meta_prompt,
                skills=skill_modules,
                pool=None,
                baseline_runs=None,
                max_tool_turns=10,
            )
            self._apply_proposal_skills(proposal, skill_modules, skills_dir=skills_dir)
            candidate_code = proposal.full_policy_code or proposal.raw_code_patch

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
                    pool=None,
                    baseline_runs=None,
                    max_tool_turns=4,
                )
                self._apply_proposal_skills(
                    repair_proposal, skill_modules, skills_dir=skills_dir
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
                    f"Meta-Agent: Successfully assembled and verified draft Behavior Tree policy (Repairs: {repair_turns})."
                )
            else:
                ui.warn(
                    f"Meta-Agent: Repairs exhausted; retaining prior/fallback policy. Error was: {diag[:200]}"
                )
                master_policy_code = fallback_code

        self.theories.save_document(
            generation=generation,
            agent_role="meta_agent",
            doc_type="META_SYNTHESIS",
            title=f"Meta-Agent Policy Assembly G{generation}",
            content=f"### Master Policy Code\n```python\n{master_policy_code}\n```",
            tags=["meta_agent", f"gen_{generation}", "assembly"],
        )

        return master_policy_code

    @staticmethod
    def prune_unused_skills(
        policy_code: str,
        skills: dict[str, str],
        skills_dir: str | Path | None = None,
    ) -> dict[str, str]:
        """
        Finds the exact transitive set of `skills.<name>` modules imported by `policy_code`
        (including skill-to-skill imports) and removes orphaned `.py` files from `skills_dir`
        and `skills` so workspace prompts and `sys.modules` stay lean across generations.
        """
        if not skills:
            if skills_dir is not None:
                s_dir = Path(skills_dir)
                if s_dir.exists():
                    removed_disk = []
                    for p in sorted(s_dir.glob("*.py")):
                        if p.stem != "__init__":
                            try:
                                p.unlink()
                                removed_disk.append(p.stem)
                            except Exception:
                                pass
                    if removed_disk:
                        ui.info(
                            f"Meta-Agent: Pruned {len(removed_disk)} unused skill module(s) from workspace "
                            f"(kept 0 active): {removed_disk[:8]}"
                            + ("..." if len(removed_disk) > 8 else "")
                        )
            return {}

        def _extract_skill_imports(src: str) -> set[str]:
            refs: set[str] = set()
            try:
                tree = ast.parse(src)
            except Exception:
                return refs
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module == "skills":
                        for alias in node.names:
                            refs.add(alias.name.split(".")[0])
                    elif node.module.startswith("skills."):
                        refs.add(node.module.split(".")[1])
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("skills."):
                            refs.add(alias.name.split(".")[1])
            return refs

        keep: set[str] = set()
        queue = list(_extract_skill_imports(policy_code))
        while queue:
            s_name = queue.pop(0)
            if s_name in keep or s_name not in skills:
                continue
            keep.add(s_name)
            queue.extend(_extract_skill_imports(skills[s_name]))

        pruned = {k: v for k, v in skills.items() if k in keep}
        removed = sorted(set(skills.keys()) - keep)
        if skills_dir is not None:
            s_dir = Path(skills_dir)
            if s_dir.exists():
                for p in s_dir.glob("*.py"):
                    if p.stem != "__init__" and p.stem not in keep:
                        try:
                            p.unlink()
                        except Exception:
                            pass
        if removed and skills_dir is not None:
            ui.info(
                f"Meta-Agent: Pruned {len(removed)} unused skill module(s) from workspace "
                f"(kept {len(pruned)} active): {removed[:8]}"
                + ("..." if len(removed) > 8 else "")
            )
        return pruned

    def _get_recent_failing_run_ids(self, generation: int, limit: int = 3) -> list[str]:
        """Queries DuckDB for recent failing run_ids in `generation` so Meta-Agent can replay-test fixes."""
        if not self.db_path or not Path(self.db_path).exists():
            return []
        try:
            from lox.db.schema import connect_db

            con = connect_db(self.db_path, read_only=True)
            try:
                rows = con.execute(
                    """
                    SELECT run_id, end_reason, max_depth, turns
                    FROM runs
                    WHERE generation = ? AND end_reason NOT IN ('ascended', 'max_steps')
                    ORDER BY turns DESC, max_depth ASC
                    LIMIT ?
                    """,
                    [generation, limit],
                ).fetchall()
                return [
                    f"`{r[0]}` (end=`{r[1]}`, max_depth={r[2]}, turns={r[3]})"
                    for r in rows
                ]
            finally:
                con.close()
        except Exception:
            return []

    def revise_policy_with_critic(
        self,
        generation: int,
        draft_policy: str,
        skills_dir: str | Path | None,
        skills: dict[str, str],
        post_critique: Any,
        pre_gen_guide: Any = None,
        max_repair_turns: int = 3,
    ) -> tuple[str, dict[str, str], bool]:
        """
        Receives Critic Pass 2's comparative review (`post_critique`) after `draft_policy`
        has been evaluated against previous generations, allowing the Meta-Agent to decide
        whether to keep `draft_policy` as-is or revise `policy.py` / `skills/*.py`.
        Returns `(final_policy_code, final_skills_dict, was_modified)`.
        """
        updated_skills = dict(skills or {})
        if self.provider == "mock" or post_critique is None:
            return draft_policy, updated_skills, False

        ui.info(
            f"Meta-Agent: Reviewing Critic Pass 2 feedback for Generation {generation} "
            f"(advisory needs_revision={getattr(post_critique, 'needs_revision', False)})..."
        )
        critique_block = (
            post_critique.to_prompt_block()
            if hasattr(post_critique, "to_prompt_block")
            else str(post_critique)
        )
        pre_block = (
            f"\n{pre_gen_guide.to_prompt_block()}\n"
            if pre_gen_guide is not None and hasattr(pre_gen_guide, "to_prompt_block")
            else ""
        )
        failing_runs = self._get_recent_failing_run_ids(generation=generation, limit=3)
        replay_hint = (
            "- **Failure-Replay Verification**: You can verify any fix directly on the actual mid/late-game failing states from this generation via `dry_run_policy(code=..., replay_run_id='<run_id>')`:\n  "
            + "\n  ".join(f"* {fr}" for fr in failing_runs)
            + "\n"
            if failing_runs
            else ""
        )
        revision_prompt = (
            f"You are the Meta-Agent Architect completing the Post-Critic Review for Generation {generation}.\n"
            f"Your initial draft policy (`policy_swarm_g{generation}`) was evaluated on the batch seeds, and the Read-Only Critic "
            f"has compared its results, actions, and `PrioritySelector` branch telemetry against previous generations:\n\n"
            f"{critique_block}\n"
            f"{pre_block}\n"
            f"### Your Decision & Authority (Zero Hardcoded Rollbacks):\n"
            f'- There are NO automatic rollbacks in the harness—YOU decide the final `class Policy` (and optional `<skill_code module="name">...</skill_code>` updates) for Generation {generation}.\n'
            f"- If your draft policy `policy_swarm_g{generation}` improved or performed well and needs no changes, simply output the current `class Policy` unchanged.\n"
            f"- If the Critic identified a regression, hijacking branch, `0-TURN` starved branch, or dropped capability compared to an earlier generation, use `query_db`, `diff_policies`, `read_policy`, `read_skill(skill_name, policy_id=...)`, and `dry_run_policy` to fix the issue or restore the working component from an earlier `policy_id`.\n"
            f"{replay_hint}"
            f'- Output your final `class Policy` inside `<policy_code>...</policy_code>` (or a ```python ... ``` block), and any updated skill modules inside `<skill_code module="name">...</skill_code>`.'
        )

        try:
            proposal = self.author.synthesize(
                mode="SWARM_META",
                generation=generation,
                current_policy=draft_policy,
                batch_digest=revision_prompt,
                skills=updated_skills,
                pool=None,
                baseline_runs=None,
                max_tool_turns=6,
            )
        except Exception as exc:
            ui.warn(
                f"Meta-Agent post-critic revision session failed ({exc}); keeping verified draft policy."
            )
            return draft_policy, updated_skills, False

        cand_skills = dict(updated_skills)
        self._apply_proposal_skills(proposal, cand_skills, skills_dir=None)
        candidate_code = (
            proposal.full_policy_code or proposal.raw_code_patch or draft_policy
        )

        skills_changed = set(cand_skills.keys()) != set(updated_skills.keys()) or any(
            not is_ast_equivalent(cand_skills[k], updated_skills[k])
            for k in cand_skills
            if k in updated_skills
        )
        policy_changed = not is_ast_equivalent(candidate_code, draft_policy)
        if not policy_changed and not skills_changed:
            ui.info(
                f"Meta-Agent: Confirmed Generation {generation} draft policy with no changes required."
            )
            return draft_policy, updated_skills, False

        is_valid, diag = self.verify_policy(candidate_code, skills=cand_skills)
        repair_turns = 0
        while not is_valid and repair_turns < max_repair_turns:
            repair_turns += 1
            ui.warn(
                f"Meta-Agent: Post-Critic revised policy failed verification; repairing ({repair_turns}/{max_repair_turns})...\nDiagnostic: {diag[:300]}"
            )
            rep_proposal = self.author.synthesize(
                mode="SWARM_META",
                generation=generation,
                current_policy=candidate_code,
                batch_digest=f"Self-Repair Required: Post-Critic revised policy failed verification:\n{diag}\n\n---\n{revision_prompt}",
                skills=cand_skills,
                pool=None,
                baseline_runs=None,
                max_tool_turns=3,
            )
            self._apply_proposal_skills(rep_proposal, cand_skills, skills_dir=None)
            candidate_code = (
                rep_proposal.full_policy_code
                or rep_proposal.raw_code_patch
                or candidate_code
            )
            is_valid, diag = self.verify_policy(candidate_code, skills=cand_skills)

        if not is_valid:
            ui.warn(
                f"Meta-Agent: Post-Critic revision failed verification after {max_repair_turns} repair(s); keeping verified draft policy."
            )
            return draft_policy, updated_skills, False

        if skills_dir is not None:
            out_dir = Path(skills_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for k, v in cand_skills.items():
                (out_dir / f"{k}.py").write_text(v, encoding="utf-8")

        self.theories.save_document(
            generation=generation,
            agent_role="meta_agent",
            doc_type="META_REVISION",
            title=f"Meta-Agent Post-Critic Revision G{generation}",
            content=f"### Revised Master Policy Code\n```python\n{candidate_code}\n```",
            tags=["meta_agent", f"gen_{generation}", "post_critic_revision"],
        )
        ui.info(
            f"Meta-Agent: Applied Post-Critic revision for Generation {generation} (Repairs: {repair_turns})."
        )
        return candidate_code, cand_skills, True
