"""
Agent Layer: CORPAgent.
Top-level cognitive operating system integrating microsecond fast spine
(HTN + A* + CDCL Nogoods + Domain Managers) with deliberative LLM supervision.
"""

from dataclasses import dataclass, field
from typing import Any
import asyncio
import json
import time
import uuid
import re
import numpy as np
import random
import gymnasium as gym
from nle import nethack

from corp.telemetry.parquet_logger import TickRecord, EpisodeRecord

from corp.env.auto_more import AutoMoreWrapper
from corp.env.blstats import BottomLineStats, HungerState
from corp.env.inventory_tracker import InventoryNormalizer
from corp.planner.persona import PersonaProfiler, PersonaTraitVector, parse_character_metadata
from corp.planner.nogood import NogoodStore, NogoodEntry
from corp.planner.predicates import compile_predicate_mask, PredicateBit
from corp.planner.htn import (
    HTNPlanner,
    CycleDetector,
    PlanSignature,
    Task,
    PrimitiveTask,
    CompoundTask,
    Method,
    HTNDeadlockException,
)
from corp.domain.combat_manager import TacticalCombatManager
from corp.domain.navigation_manager import NavigationManager
from corp.domain.inventory_manager import InventoryManager
from corp.domain.skill_worker import SkillWorker
from corp.domain.medusa_handler import MedusaHandler
from corp.policy.config import PolicyConfig, default_config
from corp.workers.dispatcher import ActionDispatcher
from corp.epistemic.epistemic_manager import EpistemicManager
from corp.epistemic.entropy_gates import ShannonSafeGate
from corp.domain.epistemic_worker import EpistemicWorker
from corp.domain.dungeon_graph import DungeonGraph
from corp.domain.shop_manager import ShopManager
from corp.domain.macro_director import MacroAscensionDirector, AscensionPhase
from corp.policy.goal_interpreter import GoalInterpreter
from corp.deliberative.autopsy_engine import AutopsyEngine
from corp.deliberative.deadlock_resolver import DeadlockResolver
from corp.deliberative.providers.mock_provider import MockProvider
from corp.deliberative.throttler import LLMRateThrottler, ThrottlerConfig
from corp.deliberative.impasse_evaluator import HolisticImpasseEvaluator, ImpasseEvaluation


@dataclass
class LockedIntent:
    """Represents a multi-turn committed intent to prevent heuristic fluttering."""
    name: str  # e.g. "DOOR_KICK", "RETREAT_FUNNEL", "ALTAR_TEST", "DIP_FOUNTAIN"
    target_pos: tuple[int, int]
    subtasks: list[Task] = field(default_factory=list)
    remaining_steps: int = 0
    abort_on_emergency: bool = True


@dataclass
class EpisodeResult:
    """Summary metrics of a completed NetHack episode."""
    turns: int
    max_depth: int
    final_score: int
    final_hp: int
    is_ascended: bool
    death_message: str
    active_nogoods_count: int
    steps_per_second: float


class CORPAgent:
    """
    Cognitive Operating System for NetHack (CORP).
    Integrates sub-millisecond symbolic decision spine with an asynchronous,
    strictly gated deliberative reasoning engine for deadlocks and autopsies.
    """

    def __init__(
        self,
        env: AutoMoreWrapper,
        nogood_store: NogoodStore | None = None,
        llm_provider: Any = None,
        enable_deliberative_autopsy: bool = False,
        enable_in_game_deliberation: bool = False,
        parquet_logger: Any = None,
        eval_type: str = "fast_prelim",
        mode: str = "random",
        seed: int = 42,
        role: str = "unknown",
        throttler_config: ThrottlerConfig | None = None,
        policy_config: PolicyConfig | None = None,
    ):
        self.env = env
        self.nogood_store = nogood_store or NogoodStore()
        self.llm_provider = llm_provider or MockProvider()
        self.enable_deliberative_autopsy = enable_deliberative_autopsy
        self.enable_in_game_deliberation = enable_in_game_deliberation
        self.parquet_logger = parquet_logger
        self.eval_type = eval_type
        self.mode = mode
        self.seed = seed
        self.throttler = LLMRateThrottler(
            throttler_config or ThrottlerConfig(
                min_wall_seconds=0.0,
                min_turn_gap=0,
                max_in_game_per_episode=10,
                escalation_threshold=1,
            )
        )
        self.impasse_evaluator = HolisticImpasseEvaluator(threshold=1.0)

        # Fast spine subsystems
        self.profiler = PersonaProfiler()
        self.epistemic = EpistemicManager()
        self.epistemic_worker = EpistemicWorker(self.epistemic)
        self.policy_config = policy_config or default_config()
        self.combat_mgr = TacticalCombatManager(self.policy_config)
        self.nav_mgr = NavigationManager(self.policy_config)
        self.inv_mgr = InventoryManager(self.policy_config)
        self.skill_mgr = SkillWorker()
        self.medusa_handler = MedusaHandler()
        self.dungeon_graph = DungeonGraph()
        self.shop_mgr = ShopManager()
        self.goals = GoalInterpreter(self.policy_config)
        self.dispatcher = ActionDispatcher(env)
        self.cycle_detector = CycleDetector(window_size=10, max_repetitions=3)
        self.planner = HTNPlanner(nogood_store=self.nogood_store)
        self._setup_htn_methods()

        # Deliberative subsystems
        self.deadlock_resolver = DeadlockResolver(self.llm_provider)
        self.autopsy_engine = AutopsyEngine(self.llm_provider, self.nogood_store)

        # Active episode state
        self.current_role: str = role
        self.episode_id: str = f"ep_{self._role_prefix(self.current_role)}_{str(uuid.uuid4())[:8]}"
        self.step_counter: int = 0
        self.persona: PersonaTraitVector | None = None
        self.current_blstats: BottomLineStats | None = None
        self.current_chars: np.ndarray | None = None
        self.current_glyphs: np.ndarray | None = None
        self.max_depth_reached: int = 1
        self.max_turn_reached: int = 1
        self.is_terminal: bool = False
        self.terminal_message: str = ""
        self.current_message: str = ""
        self.current_race: str = "unknown"
        self.current_gender: str = "unknown"
        self.current_alignment: str = "unknown"
        self.intrinsics: set[str] = set()
        self.locked_intent: LockedIntent | None = None

        # LLM query telemetry & in-game plan injection
        self.last_llm_turn: int = -1
        self.last_llm_timestamp: float = 0.0
        self.total_llm_queries: int = 0
        self.llm_turn_intervals: list[int] = []
        self.plan_queue: list[Task] = []
        self._deadlock_trigger: tuple[str, str] | None = None
        self.failed_step_counts: dict[tuple[int, int], int] = {}
        self.consecutive_zero_turn_steps: int = 0

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Initializes a new episode and derives continuous Turn-0 persona."""
        self.step_counter = 0
        self.failed_step_counts.clear()
        self.consecutive_zero_turn_steps = 0
        obs, info = self.env.reset(**kwargs)
        self.current_blstats = info["blstats"]
        self.current_chars = obs["chars"]
        self.current_glyphs = obs["glyphs"]
        self.max_depth_reached = self.current_blstats.depth
        self.max_turn_reached = self.current_blstats.turn
        self.is_terminal = False
        self.terminal_message = ""
        self.current_message = info.get("full_message", "")

        # Extract character metadata from startup banner
        meta = parse_character_metadata(self.current_message)
        if meta["role"] != "unknown":
            self.current_role = meta["role"]
        self.current_race = meta["race"]
        self.current_gender = meta["gender"]
        self.current_alignment = meta["alignment"]

        # Format unique episode_id with role prefix
        self.episode_id = f"ep_{self._role_prefix(self.current_role)}_{str(uuid.uuid4())[:8]}"

        # Reset subsystems
        self.dungeon_graph.reset()
        self.shop_mgr.reset()
        self.goals.reset()
        self.locked_intent = None

        # Track intrinsics (Barbarians, Healers, and Orcs start with intrinsic poison resistance)
        self.intrinsics = set()
        if self.current_role.lower() in ("barbarian", "healer") or self.current_race.lower() == "orc":
            self.intrinsics.add("poison_res")

        self.cycle_detector.reset()
        self.nav_mgr.reset()
        self.inv_mgr = InventoryManager(self.policy_config)
        self.combat_mgr.reset()
        self.skill_mgr.reset()
        self.medusa_handler.reset()
        self.epistemic_worker.reset()
        self.throttler.reset_episode()
        self.impasse_evaluator.reset()
        if hasattr(self.env, "inventory_tracker"):
            self.env.inventory_tracker.reset(role=self.current_role)

        self.last_llm_turn = -1
        self.last_llm_timestamp = 0.0
        self.total_llm_queries = 0
        self.llm_turn_intervals.clear()
        self.plan_queue.clear()
        self._deadlock_trigger = None

        # Turn-0 Continuous Persona Profiling
        self.persona = self.profiler.derive_persona(
            blstats=self.current_blstats,
            inventory=self.env.inventory_tracker,
        )

        return obs, info

    def has_poison_resistance(self, inv_tracker: Any = None) -> bool:
        """Determines if the hero has intrinsic or extrinsic poison resistance."""
        if "poison_res" in self.intrinsics:
            return True
        if self.current_role.lower() in ("barbarian", "healer"):
            return True
        if self.current_race.lower() == "orc":
            return True
        tracker = inv_tracker or (self.env.inventory_tracker if hasattr(self.env, "inventory_tracker") else None)
        if tracker is not None:
            active = tracker.get_active_items() if hasattr(tracker, "get_active_items") else [it for it in getattr(tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
            for it in active:
                desc = it.raw_str.lower()
                if "(being worn)" in desc or "on left hand" in desc or "on right hand" in desc:
                    if any(k in desc for k in ("amulet versus poison", "ring of poison resistance", "alchemy smock", "green dragon scale")):
                        return True
        return False

    def has_reflection(self, inv_tracker: Any = None) -> bool:
        """Determines if the hero has extrinsic reflection (reflects ray wands and strikes)."""
        tracker = inv_tracker or (self.env.inventory_tracker if hasattr(self.env, "inventory_tracker") else None)
        if tracker is not None:
            active = tracker.get_active_items() if hasattr(tracker, "get_active_items") else [it for it in getattr(tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
            for it in active:
                desc = it.raw_str.lower()
                if "(being worn)" in desc or "shield in hand" in desc or ("(wielded)" in desc and "shield" in desc):
                    if any(k in desc for k in ("shield of reflection", "amulet of reflection", "silver dragon scale")):
                        return True
        return False

    def _setup_htn_methods(self):
        """
        Formalizes the HTN compound hierarchy and registers domain capability methods
        governed by continuous persona prioritization and CDCL Nogood branch cuts.
        """
        # Tier 1: Root Compound SURVIVE_AND_ASCEND decompositions
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_NUTRITION",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_emergency_eat,
            subtasks_fn=lambda s: [CompoundTask("EMERGENCY_NUTRITION")],
            base_utility=1000.0,
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_TRIAGE",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_emergency_triage,
            subtasks_fn=lambda s: [CompoundTask("EMERGENCY_TRIAGE")],
            base_utility=900.0,
            priority_weights=(2.0, 0.0, 0.0, 3.0, 1.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_COMBAT_ADJ",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_adjacent_combat,
            subtasks_fn=lambda s: [CompoundTask("TACTICAL_COMBAT")],
            base_utility=800.0,
            priority_weights=(4.0, -1.0, 0.0, -2.0, 0.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_RESOURCE",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=lambda s: True,
            subtasks_fn=lambda s: [CompoundTask("RESOURCE_MANAGEMENT")],
            base_utility=700.0,
            priority_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_SHOP",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_shop_action,
            subtasks_fn=lambda s: [CompoundTask("SHOP_ACTION")],
            base_utility=680.0,
            priority_weights=(0.0, 1.0, 3.0, 0.0, 3.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_EPISTEMIC",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_epistemic_action,
            subtasks_fn=lambda s: [CompoundTask("EPISTEMIC_ACTION")],
            base_utility=650.0,
            priority_weights=(0.0, 0.0, 2.0, 0.0, 4.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_MEDUSA_PREP",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_medusa_prep,
            subtasks_fn=lambda s: [CompoundTask("MEDUSA_PREP")],
            base_utility=660.0,
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_SKILL",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=lambda s: True,
            subtasks_fn=lambda s: [CompoundTask("SKILL_ENHANCEMENT")],
            base_utility=600.0,
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_COMBAT_RANGED",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_ranged_combat,
            subtasks_fn=lambda s: [CompoundTask("TACTICAL_COMBAT")],
            base_utility=550.0,
            priority_weights=(-2.0, 5.0, 3.0, 2.0, 0.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_NAV",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=lambda s: True,
            subtasks_fn=lambda s: [CompoundTask("SPATIAL_NAVIGATION")],
            base_utility=500.0,
            priority_weights=(0.0, 0.0, 0.0, 1.0, 0.0),
        ))

        # Tier 2: Domain-specific compound decompositions
        # EMERGENCY_NUTRITION
        self.planner.register_method(Method(
            name="EMERGENCY_NUTRITION_DECOMP",
            target_task="EMERGENCY_NUTRITION",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_emergency_eat,
        ))
        # EMERGENCY_TRIAGE
        self.planner.register_method(Method(
            name="EMERGENCY_TRIAGE_DECOMP",
            target_task="EMERGENCY_TRIAGE",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_emergency_triage,
        ))
        # TACTICAL_COMBAT (Melee vs Ranged methods)
        self.planner.register_method(Method(
            name="MELEE_COMBAT_DECOMP",
            target_task="TACTICAL_COMBAT",
            preconditions=self._htn_can_adjacent_combat,
            subtasks_fn=self._htn_do_adjacent_combat,
            priority_weights=(4.0, -2.0, 0.0, 0.0, 0.0),
        ))
        self.planner.register_method(Method(
            name="RANGED_COMBAT_DECOMP",
            target_task="TACTICAL_COMBAT",
            preconditions=self._htn_can_ranged_combat,
            subtasks_fn=self._htn_do_ranged_combat,
            priority_weights=(-2.0, 4.0, 2.0, 1.0, 0.0),
        ))
        # RESOURCE_MANAGEMENT
        self.planner.register_method(Method(
            name="RESOURCE_MANAGEMENT_DECOMP",
            target_task="RESOURCE_MANAGEMENT",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_resource_mgmt,
        ))
        # SHOP_ACTION
        self.planner.register_method(Method(
            name="SHOP_ACTION_DECOMP",
            target_task="SHOP_ACTION",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_shop_action,
        ))
        # EPISTEMIC_ACTION
        self.planner.register_method(Method(
            name="EPISTEMIC_ACTION_DECOMP",
            target_task="EPISTEMIC_ACTION",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_epistemic_action,
        ))
        # SKILL_ENHANCEMENT
        self.planner.register_method(Method(
            name="SKILL_ENHANCEMENT_DECOMP",
            target_task="SKILL_ENHANCEMENT",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_skill_enhance,
        ))
        # MEDUSA_PREP (Phase 4: DL 19-25 gaze protection equipping)
        self.planner.register_method(Method(
            name="MEDUSA_PREP_DECOMP",
            target_task="MEDUSA_PREP",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_medusa_prep,
        ))
        # SPATIAL_NAVIGATION
        self.planner.register_method(Method(
            name="SPATIAL_NAVIGATION_DECOMP",
            target_task="SPATIAL_NAVIGATION",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_navigate,
        ))

    def _htn_can_emergency_eat(self, agent: Any) -> bool:
        return agent.current_blstats is not None and agent.current_blstats.hunger_state >= HungerState.WEAK

    def _htn_do_emergency_eat(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        lvl_map = agent.nav_mgr.get_or_create_level(blstats)
        msg_lower = agent.current_message.lower()
        is_on_fountain = (blstats.y, blstats.x) in lvl_map.fountains or "fountain" in msg_lower
        has_poison_res = agent.has_poison_resistance(agent.env.inventory_tracker)
        monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, blstats, has_poison_res=has_poison_res) if agent.current_glyphs is not None else []
        has_adj = any(m.is_adjacent for m in monsters)
        faint_task = agent.inv_mgr.evaluate_resource_turn(
            blstats,
            agent.env.inventory_tracker,
            message=agent.current_message,
            is_on_fountain=is_on_fountain,
            role=agent.current_role,
            race=agent.current_race,
            has_poison_res=has_poison_res,
            epistemic_mgr=agent.epistemic,
            has_adjacent_hostiles=has_adj,
            lvl_map=lvl_map,
        )
        if faint_task is not None:
            return [faint_task]
        # If no carried food and starving, emergency prayer immediately restores nutrition to 900
        if agent.inv_mgr.can_safely_pray(blstats):
            agent.inv_mgr.prayer_state.last_prayer_turn = blstats.turn
            agent.inv_mgr.prayer_state.prayer_count += 1
            return [Task("PRAY", is_primitive=True)]
        return []

    def _htn_can_emergency_triage(self, agent: Any) -> bool:
        blstats = agent.current_blstats
        if blstats is None or agent.current_glyphs is None:
            return False
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        has_adj = any(m.is_adjacent for m in agent.combat_mgr.scan_monsters(agent.current_glyphs, blstats, has_poison_res=has_pr))
        s = agent.policy_config.strategy
        return (
            blstats.hp <= max(s.triage_min_hp, int(blstats.max_hp * s.triage_hp_frac))
            or (has_adj and blstats.hp <= max(s.triage_adj_min_hp, int(blstats.max_hp * s.triage_adj_hp_frac)))
        )

    def _htn_do_emergency_triage(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        inv_tracker = agent.env.inventory_tracker
        # 1. Healing potion
        if inv_tracker is not None:
            active_items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
            for item in active_items:
                if item.buc_state == "CURSED":
                    continue
                if agent.epistemic is not None:
                    belief = agent.epistemic.get_or_create_belief(item)
                    safe, _ = ShannonSafeGate.can_safely_quaff(belief)
                    if not safe:
                        continue
                desc = item.raw_str.lower()
                if any(p in desc for p in ("potion of full healing", "potion of extra healing", "potion of healing")):
                    return [Task("QUAFF", is_primitive=True, args={"slot": item.current_letter})]

        # 2. Teleport scroll
        if inv_tracker is not None and blstats.hp <= max(5, int(blstats.max_hp * 0.30)):
            active_items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
            for item in active_items:
                if item.buc_state in ("BLESSED", "UNCURSED") and "scroll of teleportation" in item.raw_str.lower():
                    if agent.epistemic is not None:
                        belief = agent.epistemic.get_or_create_belief(item)
                        safe, _ = ShannonSafeGate.can_safely_read(belief)
                        if not safe:
                            continue
                    return [Task("READ", is_primitive=True, args={"slot": item.current_letter})]

        # 3. Divine prayer
        if blstats.hp <= max(5, int(blstats.max_hp * self.policy_config.survival.critical_retreat_frac)):
            if agent.inv_mgr.can_safely_pray(blstats):
                agent.inv_mgr.prayer_state.last_prayer_turn = blstats.turn
                agent.inv_mgr.prayer_state.prayer_count += 1
                return [Task("PRAY", is_primitive=True)]
        return []

    def _htn_can_adjacent_combat(self, agent: Any) -> bool:
        if agent.current_glyphs is None or agent.current_blstats is None:
            return False
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        return any(m.is_adjacent for m in agent.combat_mgr.scan_monsters(agent.current_glyphs, agent.current_blstats, has_poison_res=has_pr))

    def _htn_do_adjacent_combat(self, agent: Any) -> list[Task]:
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        has_refl = agent.has_reflection(agent.env.inventory_tracker)
        task = agent.combat_mgr.evaluate_combat_turn(
            agent.current_glyphs,
            agent.current_chars,
            agent.current_blstats,
            inv_tracker=agent.env.inventory_tracker,
            role=agent.current_role,
            message=agent.current_message,
            has_poison_res=has_pr,
            has_reflection=has_refl,
        )
        return [task] if task is not None else []

    def _htn_do_resource_mgmt(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        lvl_map = agent.nav_mgr.get_or_create_level(blstats)
        msg_lower = agent.current_message.lower()
        is_on_fountain = (blstats.y, blstats.x) in lvl_map.fountains or "fountain" in msg_lower
        has_poison_res = agent.has_poison_resistance(agent.env.inventory_tracker)
        monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, blstats, has_poison_res=has_poison_res) if agent.current_glyphs is not None else []
        has_adj = any(m.is_adjacent for m in monsters)
        task = agent.inv_mgr.evaluate_resource_turn(
            blstats,
            agent.env.inventory_tracker,
            message=agent.current_message,
            is_on_fountain=is_on_fountain,
            role=agent.current_role,
            race=agent.current_race,
            has_poison_res=has_poison_res,
            epistemic_mgr=agent.epistemic,
            has_adjacent_hostiles=has_adj,
            lvl_map=lvl_map,
        )
        return [task] if task is not None else []


    def _htn_can_epistemic_action(self, agent: Any) -> bool:
        if agent.current_blstats is None:
            return False
        lvl_map = agent.nav_mgr.get_or_create_level(agent.current_blstats)
        if not lvl_map.altars or agent.epistemic_worker.altar_test_count >= 10:
            return False
        inv_tracker = agent.env.inventory_tracker
        return (
            agent.epistemic_worker.has_untested_items(inv_tracker)
            or agent.epistemic_worker.get_sacrificable_corpse(inv_tracker, agent.current_blstats.turn) is not None
        )

    def _htn_do_epistemic_action(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        lvl_map = agent.nav_mgr.get_or_create_level(blstats)
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, blstats, has_poison_res=has_pr)
        has_adj = any(m.is_adjacent for m in monsters)
        task = agent.epistemic_worker.evaluate_epistemic_turn(
            blstats=blstats,
            chars=agent.current_chars,
            inv_tracker=agent.env.inventory_tracker,
            lvl_altars=lvl_map.altars,
            has_adjacent_hostiles=has_adj,
            visible_monster_count=len(monsters),
        )
        return [task] if task is not None else []

    def _htn_do_skill_enhance(self, agent: Any) -> list[Task]:
        task = agent.skill_mgr.evaluate_skill_turn(agent.current_blstats, message=agent.current_message)
        return [task] if task is not None else []

    def _htn_can_medusa_prep(self, agent: Any) -> bool:
        return (
            agent.medusa_handler.is_medusa_approach(agent.current_blstats)
            and not agent.has_reflection(agent.env.inventory_tracker)
        )

    def _htn_do_medusa_prep(self, agent: Any) -> list[Task]:
        task = agent.medusa_handler.evaluate_medusa_prep(
            agent.current_blstats,
            inv_tracker=agent.env.inventory_tracker,
            has_reflection=agent.has_reflection(agent.env.inventory_tracker),
        )
        return [task] if task is not None else []

    def _htn_can_ranged_combat(self, agent: Any) -> bool:
        if agent.current_glyphs is None or agent.current_blstats is None:
            return False
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        return len(agent.combat_mgr.scan_monsters(agent.current_glyphs, agent.current_blstats, has_poison_res=has_pr)) > 0

    def _htn_do_ranged_combat(self, agent: Any) -> list[Task]:
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        has_refl = agent.has_reflection(agent.env.inventory_tracker)
        task = agent.combat_mgr.evaluate_combat_turn(
            agent.current_glyphs,
            agent.current_chars,
            agent.current_blstats,
            inv_tracker=agent.env.inventory_tracker,
            role=agent.current_role,
            message=agent.current_message,
            has_poison_res=has_pr,
            has_reflection=has_refl,
        )
        return [task] if task is not None else []

    def _htn_can_shop_action(self, agent: Any) -> bool:
        if agent.current_blstats is None or agent.current_chars is None:
            return False
        # 1. Unpaid item resolution (Top commercial priority to avoid shopkeeper rage)
        inv = agent.env.inventory_tracker if hasattr(agent.env, "inventory_tracker") else None
        if inv is not None and agent.shop_mgr.has_unpaid_items(inv):
            return True
        # 2. Temple priest donation
        if agent.shop_mgr.priest_pos is not None:
            monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, agent.current_blstats)
            has_adj = any(m.is_adjacent for m in monsters)
            if agent.shop_mgr.evaluate_temple_donation(agent.current_blstats, agent.current_chars, has_adjacent_hostiles=has_adj) is not None:
                return True
        # 3. Shop price probing
        if agent.shop_mgr.in_shop:
            monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, agent.current_blstats)
            has_adj = any(m.is_adjacent for m in monsters)
            if agent.shop_mgr.evaluate_shop_probing(agent.current_blstats, inv, agent.current_chars, has_adjacent_hostiles=has_adj) is not None:
                return True
        return False

    def _htn_do_shop_action(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        chars = agent.current_chars
        monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, blstats)
        has_adj = any(m.is_adjacent for m in monsters)
        inv = agent.env.inventory_tracker if hasattr(agent.env, "inventory_tracker") else None
        # 1. Priority: Unpaid item payment or debt drop
        if inv is not None and agent.shop_mgr.has_unpaid_items(inv):
            task = agent.shop_mgr.evaluate_shop_payment(blstats, inv, agent.current_message)
            if task is not None:
                return [task]
        # 2. Priority: Temple donation (divine protection for -10 AC)
        if agent.shop_mgr.priest_pos is not None:
            task = agent.shop_mgr.evaluate_temple_donation(blstats, chars, has_adjacent_hostiles=has_adj)
            if task is not None:
                return [task]
        # 3. Priority: Shop price probing
        if agent.shop_mgr.in_shop:
            task = agent.shop_mgr.evaluate_shop_probing(blstats, inv, chars, has_adjacent_hostiles=has_adj)
            if task is not None:
                return [task]
        return []

    def _htn_do_navigate(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        lvl_map = agent.nav_mgr.get_or_create_level(blstats)
        inv_tracker = agent.env.inventory_tracker
        target_fountain = False
        long_sword_slot = None
        is_lawful = (blstats.alignment == 1) or (agent.persona is not None and agent.persona.alignment >= 0.8) or (agent.current_role.lower() in ("valkyrie", "samurai", "knight"))
        if blstats.experience >= 5 and is_lawful and inv_tracker is not None:
            active_items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
            has_excalibur = any("excalibur" in it.raw_str.lower() for it in active_items)
            if not has_excalibur:
                for it in active_items:
                    raw = it.raw_str.lower()
                    if "long sword" in raw and "cursed" not in raw:
                        target_fountain = True
                        long_sword_slot = it.current_letter
                        break

        target_altar = False
        if lvl_map.altars and lvl_map.turns_spent < 50 and agent.epistemic_worker.altar_test_count < 10 and (
            agent.epistemic_worker.has_untested_items(inv_tracker)
            or agent.epistemic_worker.get_sacrificable_corpse(inv_tracker, blstats.turn) is not None
        ):
            target_altar = True

        has_poison_res = agent.has_poison_resistance(inv_tracker)
        task = agent.nav_mgr.evaluate_navigation_turn(
            agent.current_chars,
            blstats,
            target_fountain=target_fountain,
            target_altar=target_altar,
            message=agent.current_message,
            inv_tracker=inv_tracker,
            glyphs=agent.current_glyphs,
            has_poison_res=has_poison_res,
            long_sword_slot=long_sword_slot,
            dungeon_graph=agent.dungeon_graph,
            macro_director=agent.goals,
        )
        if task is not None:
            if task.name == "KICK" and "delta" in task.args:
                dr, dc = task.args["delta"]
                door_pos = (blstats.y + dr, blstats.x + dc)
                lvl_map = agent.nav_mgr.get_or_create_level(blstats)
                if (
                    door_pos not in lvl_map.shop_doors
                    and not lvl_map.has_shops
                    and (agent.current_chars == ord("@")).sum() <= 1
                ):
                    agent.locked_intent = LockedIntent(
                        name="DOOR_KICK",
                        target_pos=door_pos,
                        remaining_steps=29,
                    )
            return [task]
        return []

    def select_action(self) -> Task:
        """
        Microsecond decision loop. Cascades through prioritized tactical managers
        with 64-bit Nogood bitmask validation.
        """
        assert self.current_blstats is not None
        assert self.current_chars is not None
        assert self.current_glyphs is not None

        blstats = self.current_blstats
        chars = self.current_chars
        glyphs = self.current_glyphs
        inv_tracker = self.env.inventory_tracker

        # 0. Anti-Stall Guard: If 4+ consecutive steps yielded 0 turns, force escape/wait to break deadlock
        if self.consecutive_zero_turn_steps >= self.policy_config.strategy.zero_turn_limit:
            if self.consecutive_zero_turn_steps % 2 == 1:
                return Task("ESCAPE", is_primitive=True)
            else:
                return Task("WAIT", is_primitive=True)

        # 0. Update cross-level graph and commercial perceptions
        self.dungeon_graph.record_step(blstats, self.current_message)
        self.shop_mgr.update_perception(blstats, self.current_message, chars=chars, glyphs=glyphs)
        lvl_map = self.nav_mgr.get_or_create_level(blstats)
        if self.shop_mgr.in_shop:
            lvl_map.has_shops = True
        self.nav_mgr.unlock_tool_slot = self.inv_mgr.get_unlock_tool(inv_tracker)

        # 0.05 Update macro ascension director state
        has_pr = self.has_poison_resistance(inv_tracker)
        has_refl = self.has_reflection(inv_tracker)
        self.goals.update_state(
            blstats=blstats,
            message=self.current_message,
            inv_tracker=inv_tracker,
            dungeon_graph=self.dungeon_graph,
            role=self.current_role,
            has_poison_res=has_pr,
            has_reflection=has_refl,
            temple_donations=self.shop_mgr.priest_donations_count,
        )

        # 0.1 Check multi-turn locked intent commitment
        msg_lower = self.current_message.lower()
        if self.locked_intent is not None:
            # Immediate emergency interrupt: health critical (<= 55%) or starving
            if blstats.hp <= int(blstats.max_hp * self.policy_config.strategy.triage_hp_frac) or blstats.hunger_state >= HungerState.WEAK:
                self.locked_intent = None
            elif self.locked_intent.name == "DOOR_KICK":
                t_pos = self.locked_intent.target_pos
                is_shop_door = (
                    t_pos in lvl_map.shop_doors
                    or (chars == ord("@")).sum() > 1
                    or any(w in msg_lower for w in ("shop", "store", "closed for inventory", "how dare you", "angry", "chime"))
                )
                if is_shop_door:
                    self.locked_intent = None
                    lvl_map.shop_doors.add(t_pos)
                    lvl_map.walkable[t_pos[0], t_pos[1]] = False
                elif chars[t_pos[0], t_pos[1]] != ord("+"):
                    self.locked_intent = None
                elif "hurt your leg" in msg_lower or "hurt your foot" in msg_lower:
                    return Task("WAIT", is_primitive=True)
                elif self.locked_intent.remaining_steps > 0:
                    self.locked_intent.remaining_steps -= 1
                    dr, dc = t_pos[0] - blstats.y, t_pos[1] - blstats.x
                    return Task("KICK", is_primitive=True, args={"delta": (dr, dc)})
                else:
                    self.locked_intent = None
            elif self.locked_intent.subtasks:
                next_task = self.locked_intent.subtasks.pop(0)
                if not self.locked_intent.subtasks:
                    self.locked_intent = None
                return next_task

        # 0.2 High-level deliberative plan queue overrides
        if self.plan_queue:
            next_task = self.plan_queue.pop(0)
            if not next_task.is_primitive and self.persona is not None:
                try:
                    subplan = self.planner.plan(next_task, state=self, persona=self.persona)
                    if subplan:
                        self.plan_queue = subplan[1:] + self.plan_queue
                        return subplan[0]
                except Exception:
                    pass
            return next_task

        # Record confirmed peaceful entities from prompt messages (NARROWED: only the
        # specific monster type named by the prompt is pacified — previously ANY adjacent
        # monster got position-poisoned as peaceful whenever any message mentioned
        # 'peaceful', leaving hostile attackers permanently unattackable)
        msg_lower = self.current_message.lower()
        if "really attack" in msg_lower or "peaceful" in msg_lower:
            match = re.search(r"really attack (?:the )?([a-z\s]+)\?", msg_lower)
            if match:
                extracted_name = match.group(1).strip()
                if extracted_name:
                    self.combat_mgr.PEACEFUL_NAMES.add(extracted_name)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = blstats.y + dr, blstats.x + dc
                    if 0 <= nr < 21 and 0 <= nc < 79:
                        is_mon = glyphs is not None and nethack.glyph_is_monster(int(glyphs[nr, nc]))
                        is_human = chars[nr, nc] == ord("@") and (nr, nc) != (blstats.y, blstats.x)
                        if is_human:
                            self.combat_mgr.record_peaceful((nr, nc))
                            lvl_map.walkable[nr, nc] = False
                        elif is_mon:
                            # Only pacify monsters whose species is a known peaceful type
                            mname = nethack.permonst(nethack.glyph_to_mon(int(glyphs[nr, nc]))).mname.lower()
                            is_peaceful_type = any(p in mname for p in self.combat_mgr.PEACEFUL_NAMES)
                            is_known_hostile = any(h in mname for h in self.combat_mgr.hostile_names)
                            if is_peaceful_type and not is_known_hostile:
                                self.combat_mgr.record_peaceful((nr, nc))
                                lvl_map.walkable[nr, nc] = False

        # Update intrinsic resistances from message feedback
        if "feel healthy" in msg_lower or "feel especially healthy" in msg_lower:
            self.intrinsics.add("poison_res")
        if "feels amplified" in msg_lower or "grounded in reality" in msg_lower:
            self.intrinsics.add("shock_res")
        if "feel cool" in msg_lower or "cold chill" in msg_lower:
            self.intrinsics.add("fire_res")
        if "hot air" in msg_lower or "feel warm" in msg_lower:
            self.intrinsics.add("cold_res")
        if "feel totally together" in msg_lower:
            self.intrinsics.add("telepathy")

        # Check for adjacent hostile monsters
        monsters = self.combat_mgr.scan_monsters(glyphs, blstats)
        has_adjacent_hostile = any(m.is_adjacent for m in monsters)
        lvl_map = self.nav_mgr.get_or_create_level(blstats)

        # 1. Compile 64-bit state predicate mask with spatial and monster awareness
        state_mask = compile_predicate_mask(
            blstats=blstats,
            inventory=inv_tracker,
            monsters=monsters,
            chars=chars,
            lvl_map=lvl_map,
        )

        # 2. Bicameral HTN Decomposition: select tactical primitive action via persona prioritization & Nogoods
        chosen_task = None
        try:
            plan = self.planner.plan(
                root_task=Task("SURVIVE_AND_ASCEND", is_primitive=False),
                state=self,
                persona=self.persona or PersonaTraitVector(),
                state_predicate_mask=state_mask,
            )
            if plan:
                chosen_task = plan[0]
        except HTNDeadlockException:
            chosen_task = None

        if chosen_task is None:
            chosen_task = Task("SEARCH", is_primitive=True)

        # 5. Cycle Detection & Oscillation Interlock across ALL tactical domains
        py, px = blstats.y, blstats.x
        if self.cycle_detector.record_and_check((py, px), chosen_task.name):
            stall_count = self.impasse_evaluator.record_stall(chosen_task.name)
            lvl_map = self.nav_mgr.get_or_create_level(blstats)
            inv_items = self.env.inventory_tracker.get_active_items() if hasattr(self.env, "inventory_tracker") else []

            # Extract names of visible adjacent monsters
            adj_monsters = self.combat_mgr.scan_monsters(glyphs, blstats)
            adj_names = [m.name for m in adj_monsters if m.is_adjacent]

            # Calculate remaining prayer cooldown turns
            prayer_cooldown = max(0, 850 - (blstats.turn - self.inv_mgr.prayer_state.last_prayer_turn))

            # Multi-dimensional holistic evaluation
            eval_res = self.impasse_evaluator.evaluate(
                blstats=blstats,
                chars=chars,
                lvl_map=lvl_map,
                inv_items=inv_items,
                failed_task=chosen_task.name,
                message=self.current_message,
                adjacent_monster_names=adj_names,
                prayer_cooldown_remaining=prayer_cooldown,
            )

            if self.enable_in_game_deliberation and eval_res.should_trigger:
                self._deadlock_trigger = ("deadlock_macro_impasse", chosen_task.name)

            # System 1 Hierarchical symbolic fallback:
            # Immediate random perturbation step into adjacent walkable tile to shatter oscillation cycles
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = py + dr, px + dc
                if 0 <= nr < 21 and 0 <= nc < 79 and lvl_map.walkable[nr, nc]:
                    if (nr, nc) in lvl_map.blocked_tiles:
                        continue
                    # Never step into another human (@), monster, pet, or peaceful entity during perturbation!
                    if chars[nr, nc] == ord("@") and (nr, nc) != (py, px):
                        continue
                    if glyphs is not None:
                        g = int(glyphs[nr, nc])
                        if nethack.glyph_is_monster(g) or nethack.glyph_is_pet(g):
                            continue
                    if (nr, nc) in self.combat_mgr.peaceful_positions:
                        continue
                    # Corner clipping and doorway interlocks: prevent illegal diagonal perturbations
                    if dr != 0 and dc != 0:
                        if not lvl_map.walkable[py + dr, px] or not lvl_map.walkable[py, px + dc]:
                            continue
                        if (py, px) in lvl_map.doors or (nr, nc) in lvl_map.doors:
                            continue
                        if chars[py, px] in (ord("+"), ord("'")) or chars[nr, nc] in (ord("+"), ord("'")):
                            continue
                    self.nav_mgr.last_step_delta = (dr, dc)
                    return Task("STEP", is_primitive=True, args={"delta": (dr, dc)})
            return Task("SEARCH", is_primitive=True)
        else:
            self.impasse_evaluator.clear_stall(chosen_task.name)

        if chosen_task.name in ("DESCEND", "ASCEND"):
            self.dungeon_graph.record_stair_usage((py, px))

        return chosen_task

    def _micro_step_callback(
        self,
        obs: dict[str, Any],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> bool:
        """
        Step-level callback executed after each micro-action in a fiber.
        Returns False to interrupt the fiber immediately if a critical hazard or emergency arises.
        """
        if terminated or truncated:
            return False

        new_blstats = info.get("blstats")
        if new_blstats is not None and self.current_blstats is not None:
            # Check for critical burst damage or lethal status
            if new_blstats.hp <= max(4, int(new_blstats.max_hp * self.policy_config.survival.critical_retreat_frac)):
                return False
            if new_blstats.hp < self.current_blstats.hp:
                hp_loss = self.current_blstats.hp - new_blstats.hp
                if hp_loss >= max(3, int(new_blstats.max_hp * self.policy_config.nutrition.moderate_emergency_hp_frac)):
                    return False

        msg = info.get("full_message", "").lower()
        if any(w in msg for w in ("can't engrave", "cannot engrave", "large object", "never mind", "don't have that", "open the tin", "can't write", "cannot write")):
            return False

        return True

    def step(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Executes one full cognitive step: selects tactical task, dispatches keystroke,
        and updates telemetry.
        """
        if self.is_terminal:
            return {}, 0.0, True, False, {}
        t_start = time.perf_counter()
        task = self.select_action()
        latency_us = (time.perf_counter() - t_start) * 1e6

        prev_pos = (self.current_blstats.y, self.current_blstats.x) if self.current_blstats else None

        obs, reward, terminated, truncated, info = self.dispatcher.dispatch(
            task,
            step_callback=self._micro_step_callback,
        )
        self.step_counter += 1

        new_blstats = info["blstats"]
        # Track 0-turn steps for anti-stall deadlock resolution
        if self.current_blstats is not None and new_blstats.turn == self.current_blstats.turn:
            self.consecutive_zero_turn_steps += 1
        else:
            self.consecutive_zero_turn_steps = 0

        # Detect unmovable obstacle collisions (walls, boulders, locked doors)
        if prev_pos is not None and task.name == "STEP" and "delta" in task.args:
            dr, dc = task.args["delta"]
            target_pos = (prev_pos[0] + dr, prev_pos[1] + dc)
            new_pos = (new_blstats.y, new_blstats.x)
            if new_pos == prev_pos and 0 <= target_pos[0] < 21 and 0 <= target_pos[1] < 79:
                self.failed_step_counts[target_pos] = self.failed_step_counts.get(target_pos, 0) + 1
                if self.failed_step_counts[target_pos] >= 2:
                    lvl_map = self.nav_mgr.get_or_create_level(new_blstats)
                    lvl_map.blocked_tiles.add(target_pos)
                    lvl_map.walkable[target_pos[0], target_pos[1]] = False
            else:
                self.failed_step_counts.pop(target_pos, None)

        if new_blstats.turn > 0:
            self.current_blstats = new_blstats
            if new_blstats.turn > self.max_turn_reached:
                self.max_turn_reached = new_blstats.turn
            if new_blstats.depth > self.max_depth_reached:
                self.max_depth_reached = new_blstats.depth
        elif self.current_blstats is None:
            self.current_blstats = new_blstats

        # Update holistic progress history for sliding-window macro-stagnation
        if self.current_blstats is not None:
            lvl_map = self.nav_mgr.get_or_create_level(self.current_blstats)
            self.impasse_evaluator.record_step(self.current_blstats, lvl_map)

        self.current_chars = obs["chars"]
        self.current_glyphs = obs["glyphs"]
        self.current_message = info.get("full_message", "")

        # Invalidate phantom items if NetHack rejects item interaction
        msg_lower = self.current_message.lower()
        if "don't have that object" in msg_lower:
            failed_slot = task.args.get("slot")
            if failed_slot and hasattr(self.env, "inventory_tracker"):
                for it in list(self.env.inventory_tracker.active_items.values()):
                    if it.current_letter == failed_slot:
                        it.is_active = False
                        self.env.inventory_tracker.active_items.pop(it.uid, None)

        if task.name == "EAT":
            if any(k in msg_lower for k in ("don't have that object", "open the tin", "can't eat that", "cannot eat", "never heard of")):
                failed_slot = task.args.get("slot")
                if failed_slot:
                    self.inv_mgr.failed_eat_slots.add(failed_slot)

        # Fountain dried up tracking
        if "dries up" in self.current_message.lower() or "dried up" in self.current_message.lower():
            if self.current_blstats is not None:
                lvl_map = self.nav_mgr.get_or_create_level(self.current_blstats)
                lvl_map.fountains.discard((self.current_blstats.y, self.current_blstats.x))

        # Epistemic observation feedback
        if task.name == "ALTAR_TEST":
            flash_msg = info.get("altar_flash", self.current_message)
            self.epistemic_worker.on_altar_test_result(
                task.args.get("uid", ""),
                flash_msg,
                self.current_blstats or new_blstats,
                self.env.inventory_tracker,
            )
        elif task.name == "ENGRAVE_WAND":
            self.epistemic_worker.on_wand_engrave_result(
                task.args.get("uid", ""),
                self.current_message,
                self.env.inventory_tracker,
            )

        if terminated or truncated:
            self.is_terminal = True
            if terminated or (self.current_blstats and self.current_blstats.hp <= 0):
                self.terminal_message = self.current_message
            else:
                self.terminal_message = "Survived"

        # Log tick frame if ParquetLogger attached
        if self.parquet_logger is not None:
            monsters = self.combat_mgr.scan_monsters(obs["glyphs"], self.current_blstats)
            closest_m = monsters[0] if monsters else None
            effective_turn = self.current_blstats.turn if self.current_blstats.turn > 0 else self.max_turn_reached
            effective_depth = self.current_blstats.depth if self.current_blstats.depth > 0 else self.max_depth_reached
            tick = TickRecord(
                episode_id=self.episode_id,
                run_id=self.parquet_logger.run_id,
                step=self.step_counter,
                turn=effective_turn,
                timestamp=time.time(),
                depth=effective_depth,
                dungeon_number=self.current_blstats.dungeon_number,
                level_number=self.current_blstats.level_number,
                x=self.current_blstats.x,
                y=self.current_blstats.y,
                hp=self.current_blstats.hp,
                max_hp=self.current_blstats.max_hp,
                energy=self.current_blstats.energy,
                max_energy=self.current_blstats.max_energy,
                ac=self.current_blstats.ac,
                xp_level=self.current_blstats.experience,
                xp_points=self.current_blstats.score,
                gold=self.current_blstats.gold,
                hunger_state=self.current_blstats.hunger_state,
                encumbrance=self.current_blstats.encumbrance,
                condition_bits=self.current_blstats.condition_bits,
                predicate_mask=0,
                active_nogoods_count=len(self.nogood_store.entries),
                action_name=task.name,
                action_index=0,
                action_key_char="",
                action_args_json=json.dumps(task.args),
                decision_latency_us=latency_us,
                visible_hostiles_count=len(monsters),
                closest_monster_name=closest_m.name if closest_m else "",
                closest_monster_dist=closest_m.distance if closest_m else -1,
                closest_monster_speed=closest_m.speed if closest_m else 0,
                closest_monster_threat=closest_m.threat_score if closest_m else 0.0,
                cycle_detected=False,
                llm_invoked=False,
                reward=float(reward),
            )
            self.parquet_logger.log_tick(tick)

        return obs, reward, terminated, truncated, info

    async def run_episode_async(self, max_steps: int = 50000) -> EpisodeResult:
        """
        Runs the full autonomous game loop until game over or step budget exhaustion.
        Triggers post-mortem autopsy upon death.
        """
        self.reset()
        goal_flush_target = getattr(self.parquet_logger, "run_id", "norun") if self.parquet_logger else "norun"
        start_time = time.perf_counter()
        steps = 0

        while not self.is_terminal and steps < max_steps:
            # In-game deliberative intervention on deadlock / stall
            if self.enable_in_game_deliberation and self._deadlock_trigger is not None:
                trigger_type, failed_task = self._deadlock_trigger
                self._deadlock_trigger = None

                state_hash = hash((
                    self.current_blstats.depth if self.current_blstats else 0,
                    self.current_blstats.y if self.current_blstats else 0,
                    self.current_blstats.x if self.current_blstats else 0,
                    failed_task,
                ))

                # Fast Path: Check LRU patch cache
                cached_patch = self.throttler.get_cached_patch(state_hash)
                if cached_patch is not None:
                    injected = DeadlockResolver.patch_to_tasks(cached_patch)
                    self.plan_queue.extend(injected)
                else:
                    turn_now = self.current_blstats.turn if self.current_blstats else self.max_turn_reached
                    allowed, reason = self.throttler.should_allow_query(
                        trigger_type=trigger_type,
                        current_turn=turn_now,
                        task_name=failed_task,
                        state_hash=state_hash,
                    )
                    if allowed:
                        t_llm_start = time.perf_counter()
                        try:
                            patch = await self.deadlock_resolver.resolve_deadlock(
                                failed_task=failed_task,
                                blstats=self.current_blstats,
                                cycle_detected=True,
                                extra_context=f"Dlvl {self.max_depth_reached} | Turns {turn_now}",
                            )
                            llm_latency = (time.perf_counter() - t_llm_start) * 1000.0
                            time_now = time.time()
                            turns_gap = turn_now - self.last_llm_turn if self.last_llm_turn >= 0 else -1
                            wall_gap = time_now - self.last_llm_timestamp if self.last_llm_timestamp > 0 else -1.0
                            self.last_llm_turn = turn_now
                            self.last_llm_timestamp = time_now
                            self.total_llm_queries += 1
                            if turns_gap >= 0:
                                self.llm_turn_intervals.append(turns_gap)

                            self.throttler.record_query_dispatched(trigger_type, turn_now)
                            self.throttler.store_cached_patch(state_hash, patch)

                            injected = DeadlockResolver.patch_to_tasks(patch)
                            self.plan_queue.extend(injected)

                            if self.parquet_logger is not None:
                                import uuid
                                from corp.telemetry.parquet_logger import LLMQueryRecord
                                q_rec = LLMQueryRecord(
                                    query_id=str(uuid.uuid4())[:8],
                                    episode_id=self.episode_id,
                                    run_id=self.parquet_logger.run_id,
                                    step=self.step_counter,
                                    turn=turn_now,
                                    timestamp=time_now,
                                    trigger_type=trigger_type,
                                    provider=self.llm_provider.__class__.__name__,
                                    model=getattr(self.llm_provider, "model", "N/A"),
                                    prompt_preview=f"Deadlock in {failed_task}",
                                    tokens_consumed=0,
                                    latency_ms=llm_latency,
                                    turns_since_last_query=turns_gap,
                                    wall_seconds_since_last_query=wall_gap,
                                )
                                self.parquet_logger.log_llm_query(q_rec)
                        except Exception:
                            pass

            self.step()
            steps += 1

        # R2: persist goal transitions to the DuckDB goal_events table
        try:
            self.goals.flush_events("data/corp_telemetry.duckdb", goal_flush_target, self.episode_id)
        except Exception:
            pass

        elapsed = max(1e-4, time.perf_counter() - start_time)
        sps = float(steps) / elapsed

        is_ascended = (
            self.current_blstats is not None
            and self.current_blstats.depth >= 50
            and "ascended" in self.terminal_message.lower()
        )

        # Post-mortem autopsy for cross-generational Nogood synthesis
        if self.is_terminal and not is_ascended and self.enable_deliberative_autopsy:
            turn_now = self.current_blstats.turn if self.current_blstats else self.max_turn_reached
            allowed, reason = self.throttler.should_allow_query(
                trigger_type="autopsy",
                current_turn=turn_now,
            )
            if allowed:
                try:
                    t_llm_start = time.perf_counter()
                    report, entry = await self.autopsy_engine.conduct_autopsy(
                        recorder=self.env.flight_recorder,
                        death_message=self.terminal_message,
                        generation=1,
                    )
                    llm_latency = (time.perf_counter() - t_llm_start) * 1000.0
                    time_now = time.time()
                    turns_gap = turn_now - self.last_llm_turn if self.last_llm_turn >= 0 else -1
                    wall_gap = time_now - self.last_llm_timestamp if self.last_llm_timestamp > 0 else -1.0
                    self.last_llm_turn = turn_now
                    self.last_llm_timestamp = time_now
                    self.total_llm_queries += 1
                    if turns_gap >= 0:
                        self.llm_turn_intervals.append(turns_gap)

                    self.throttler.record_query_dispatched("autopsy", turn_now)

                    if self.parquet_logger is not None:
                        import uuid
                        from corp.telemetry.parquet_logger import LLMQueryRecord
                        q_rec = LLMQueryRecord(
                            query_id=str(uuid.uuid4())[:8],
                            episode_id=self.episode_id,
                            run_id=self.parquet_logger.run_id,
                            step=self.step_counter,
                            turn=turn_now,
                            timestamp=time_now,
                            trigger_type="autopsy",
                            provider=self.llm_provider.__class__.__name__,
                            model=getattr(self.llm_provider, "model", "N/A"),
                            prompt_preview=f"Autopsy on death: {self.terminal_message[:60]}",
                            tokens_consumed=0,
                            latency_ms=llm_latency,
                            turns_since_last_query=turns_gap,
                            wall_seconds_since_last_query=wall_gap,
                        )
                        self.parquet_logger.log_llm_query(q_rec)
                except Exception:
                    pass # Autopsy failure must not crash benchmark runner

        # Log episode frame if ParquetLogger attached
        if self.parquet_logger is not None:
            death_cat = "none"
            msg_lower = self.terminal_message.lower()
            if "starv" in msg_lower or "faint" in msg_lower:
                death_cat = "starvation"
            elif "poison" in msg_lower:
                death_cat = "poison"
            elif "drown" in msg_lower:
                death_cat = "drowning"
            elif "petrif" in msg_lower or "turned to stone" in msg_lower:
                death_cat = "instakill"
            elif "killed" in msg_lower or "died" in msg_lower:
                death_cat = "combat"

            ep_rec = EpisodeRecord(
                episode_id=self.episode_id,
                run_id=self.parquet_logger.run_id,
                eval_type=self.eval_type,
                seed=self.seed,
                mode=self.mode,
                character="*",
                role=self.current_role or "unknown",
                race=self.current_race or "unknown",
                gender=self.current_gender or "unknown",
                alignment=self.current_alignment or "unknown",
                persona_resilience=self.persona.resilience if self.persona else 0.5,
                persona_ranged=self.persona.ranged if self.persona else 0.5,
                persona_mana=self.persona.mana if self.persona else 0.5,
                persona_stealth=self.persona.stealth if self.persona else 0.5,
                persona_alignment=self.persona.alignment if self.persona else 0.5,
                total_turns=self.max_turn_reached if self.max_turn_reached > 0 else steps,
                total_steps=steps,
                max_depth=self.max_depth_reached,
                final_score=self.current_blstats.score if self.current_blstats else 0,
                final_hp=self.current_blstats.hp if self.current_blstats else 0,
                final_gold=self.current_blstats.gold if self.current_blstats else 0,
                is_ascended=is_ascended,
                death_message=self.terminal_message or "Survived",
                death_category=death_cat,
                nogoods_active=len(self.nogood_store.entries),
                nogoods_synthesized=0,
                mean_sps=sps,
                wall_duration_sec=elapsed,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            )
            self.parquet_logger.log_episode(ep_rec)
            self.parquet_logger.flush_all()

        return EpisodeResult(
            turns=self.max_turn_reached if self.max_turn_reached > 0 else steps,
            max_depth=self.max_depth_reached,
            final_score=self.current_blstats.score if self.current_blstats else 0,
            final_hp=self.current_blstats.hp if self.current_blstats else 0,
            is_ascended=is_ascended,
            death_message=self.terminal_message,
            active_nogoods_count=len(self.nogood_store.entries),
            steps_per_second=sps,
        )

    @staticmethod
    def _role_prefix(role: str) -> str:
        """Maps a character role string to a 3-4 char prefix for episode IDs."""
        if role == "unknown":
            return "gen"
        r_lower = role.lower()
        _PREFIX_MAP = {
            "valk": "valk", "barb": "barb", "samu": "samu", "wiza": "wiza",
            "monk": "monk", "tour": "tour", "rogu": "rogu",
        }
        for key, pfx in _PREFIX_MAP.items():
            if key in r_lower:
                return pfx
        return r_lower[:4]

    def run_episode(self, max_steps: int = 50000) -> EpisodeResult:
        """Synchronous wrapper for running an autonomous episode."""
        return asyncio.run(self.run_episode_async(max_steps=max_steps))
