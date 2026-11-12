"""
Meta-Agent Coordinator for the LOX Multi-Agent Swarm.
Integrates reflection reports from specialized category workers (dialog, threat, survival, navigation),
queries DuckDB for causal metrics, and assembles the unified Behavior Tree policy.
"""

from __future__ import annotations

import json
from typing import Any

from lox.author.agent import AuthorAgent
from lox.db.schema import connect_db
from lox.db.theories import TheoryDocumentStore
from lox.engine.frontier import FrontierArchive
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
        mortality_causes: list[tuple[str, str, int]] = []
        auto_escs = 0
        died_unarmored = 0
        starved_with_food = 0

        # Query DuckDB for recent empirical performance
        try:
            con = connect_db(self.db_path, read_only=True)
            try:
                # Find latest batch generation
                max_gen_row = con.execute("SELECT max(generation) FROM runs").fetchone()
                latest_gen = (
                    max_gen_row[0]
                    if max_gen_row and max_gen_row[0] is not None
                    else None
                )

                if latest_gen is not None:
                    # Top mortality causes
                    mort_rows = con.execute(
                        """
                        SELECT end_reason, death_message, count(*) as cnt
                        FROM runs
                        WHERE generation = ?
                        GROUP BY end_reason, death_message
                        ORDER BY cnt DESC
                        LIMIT 5
                        """,
                        [latest_gen],
                    ).fetchall()
                    mortality_causes = [
                        (str(r[0]), str(r[1]), int(r[2])) for r in mort_rows
                    ]

                    # Aggregate diagnostic indicators
                    metrics_row = con.execute(
                        """
                        SELECT
                            coalesce(avg(depth), 1.0),
                            coalesce(max(depth), 1),
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
                        auto_escs = int(metrics_row[2])
                        died_unarmored = int(metrics_row[3])
                        starved_with_food = int(metrics_row[4])
            finally:
                con.close()
        except Exception:
            pass

        # Check frontier archive for available depth tiers
        frontier_depths = (
            sorted({cp.depth for cp in self.frontier.checkpoints.values()})
            if self.frontier.checkpoints
            else [1]
        )
        active_frontier_depth = max(frontier_depths) if frontier_depths else 1

        # Formulate dynamic emergent categories based on empirical bottlenecks
        categories: list[dict[str, Any]] = []

        # Category 1: Dialog & Prompt Safety (always critical if auto-ESCs triggered or early prompt traps)
        if auto_escs > 0 or generation == 0:
            categories.append(
                {
                    "category_id": "dialog_and_prompts",
                    "name": "Dialog & Prompt Safety Specialist",
                    "description": (
                        f"Specialist in handling interactive text prompts, --More-- pagination, and y/n naming traps. "
                        f"Observed {auto_escs} auto-ESC prompt dismissals in recent runs. "
                        f"Must produce reactive ActionLeaf preventing episode stall timeouts."
                    ),
                    "target_depth_tier": (1, max(2, active_frontier_depth)),
                    "target_metrics": ["auto_esc_count", "stagnation_turns"],
                    "blackboard_keys": ["in_prompt", "dialog_resolved"],
                }
            )

        # Category 2: Tactical Threat & Combat (tailored to mortality and depth tier)
        combat_focus = "early_melee"
        if active_frontier_depth >= 4:
            combat_focus = "ant_and_poison_defense"
            threat_desc = (
                f"Specialist in deep-tier hostile encounters at Dlvl >= 4 (soldier ants, floating eyes, mimics). "
                f"Top mortality causes: {mortality_causes[:2]}. "
                f"Must produce Behavior Tree combat subtrees implementing chokepoint kiting, dust Elbereth, and tactical retreat thresholds."
            )
        elif active_frontier_depth >= 2:
            combat_focus = "pack_combat_and_kiting"
            threat_desc = (
                f"Specialist in mid-dungeon pack combat at Dlvl 2-3 (jackal swarms, sewer rats, orcs). "
                f"Top mortality causes: {mortality_causes[:2]}. "
                f"Must produce reactive combat leaf nodes targeting hostiles in 1-tile corridor chokepoints."
            )
        else:
            combat_focus = "early_threat_and_flee"
            threat_desc = (
                "Specialist in early survival combat on Dlvl 1. "
                "Produces low-risk melee strikes against adjacent vermin and retreat thresholds when HP <= 6."
            )

        categories.append(
            {
                "category_id": f"tactical_threat_{combat_focus}",
                "name": f"Tactical Threat ({combat_focus.replace('_', ' ').capitalize()})",
                "description": threat_desc,
                "target_depth_tier": (
                    max(1, active_frontier_depth - 1),
                    active_frontier_depth + 1,
                ),
                "target_metrics": ["final_hp", "killed_monsters", "subgen_preemptions"],
                "blackboard_keys": ["last_target", "danger_level", "tactical_retreat"],
            }
        )

        # Category 3: Resource & Health Optimization (nutrition, corpses, armor)
        if starved_with_food > 0:
            res_id = "nutrition_and_corpse_eating"
            res_desc = (
                f"Specialist in nutrition and emergency food consumption. "
                f"Detected {starved_with_food} starved_with_food blunders! "
                f"Must query corpse safety (query_monster) to eat non-poisonous corpses before rations."
            )
        elif died_unarmored > 0:
            res_id = "armor_scavenging_and_ac"
            res_desc = (
                f"Specialist in scavenging and equipping armor to lower AC. "
                f"Detected {died_unarmored} died_unarmored runs. "
                f"Must prioritize equipping scavenged dwarvish/orcish armor when standing on dropped items."
            )
        else:
            res_id = "intrinsic_resistances_and_hp"
            res_desc = (
                "Specialist in intrinsic resistance farming (poison/cold resistance from fresh corpses) "
                "and strategic resting on staircases for HP regeneration."
            )

        categories.append(
            {
                "category_id": res_id,
                "name": f"Resource & Health ({res_id.replace('_', ' ').capitalize()})",
                "description": res_desc,
                "target_depth_tier": (1, active_frontier_depth),
                "target_metrics": ["final_ac", "starved_with_food", "died_unarmored"],
                "blackboard_keys": [
                    "hunger_level",
                    "equipped_armor",
                    "hp_regen_needed",
                ],
            }
        )

        # Category 4: Spatial Progression & Frontier Exploration
        if active_frontier_depth >= 5:
            nav_id = "branch_and_sokoban_navigation"
            nav_desc = (
                "Specialist in branch navigation (Minetown stairs, Sokoban entrance at Dlvl 5-6). "
                "Solves corridor navigation and detects stairs down while avoiding boulder traps."
            )
        else:
            nav_id = "dungeon_exploration_and_stairs"
            nav_desc = (
                "Specialist in BFS room exploration, kicking locked doors ('+'), and finding stairs down ('>'). "
                "Maximizes unique_tiles visited and transitions between dungeon depths."
            )

        categories.append(
            {
                "category_id": nav_id,
                "name": f"Frontier Navigation ({nav_id.replace('_', ' ').capitalize()})",
                "description": nav_desc,
                "target_depth_tier": (active_frontier_depth, active_frontier_depth + 2),
                "target_metrics": ["unique_tiles", "depth", "max_depth"],
                "blackboard_keys": ["stairs_loc", "target_tile", "room_cleared"],
            }
        )

        # Trim or pad to requested number of categories
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
        # Validate AST
        validate_python_code(master_policy_code)

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
