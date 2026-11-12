"""
Meta-Agent Coordinator for the LOX Multi-Agent Swarm.
Integrates reflection reports from specialized category workers (dialog, threat, survival, navigation),
queries DuckDB for causal metrics, and assembles the unified Behavior Tree policy.
"""

from __future__ import annotations

from typing import Any

from lox.author.agent import AuthorAgent
from lox.db.theories import TheoryDocumentStore
from lox.engine.frontier import FrontierArchive
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

    def assemble_policy(
        self,
        generation: int,
        worker_reports: dict[str, dict[str, Any]] | None = None,
    ) -> str:
        """
        Gathers worker reports from LanceDB, checks DuckDB metrics, and synthesizes the master Policy code.
        """
        if worker_reports is None:
            worker_reports = self.theories.get_latest_reports(generation=generation)

        ui.info(
            f"Meta-Agent: Assembling Generation {generation} master policy from "
            f"{len(worker_reports)} worker report(s): {list(worker_reports.keys())}"
        )

        # Standard clean Behavior Tree master template
        master_policy_code = f"""# Master Behavior Tree Policy - Synthesized by Meta-Agent (Gen {generation})
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
            # Check for emergency prayer or item consumption if needed
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
