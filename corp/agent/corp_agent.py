"""
Agent Layer: CORPAgent.
Top-level cognitive operating system integrating microsecond fast spine
(HTN + A* + CDCL Nogoods + Domain Managers) with deliberative LLM supervision.

Behavior-preserving split: HTN methods -> corp/agent/htn_methods.py, episode runner ->
corp/agent/episode_runner.py, dataclasses -> corp/agent/episode.py.
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
from corp.policy.goal_interpreter import GoalInterpreter
from corp.deliberative.autopsy_engine import AutopsyEngine
from corp.deliberative.deadlock_resolver import DeadlockResolver, emit_deadlock_diff
from corp.deliberative.providers.mock_provider import MockProvider
from corp.deliberative.throttler import LLMRateThrottler, ThrottlerConfig
from corp.deliberative.impasse_evaluator import HolisticImpasseEvaluator, ImpasseEvaluation

from corp.agent.episode import EpisodeResult, LockedIntent
from corp.agent.htn_methods import HTNMethodsMixin
from corp.agent.episode_runner import EpisodeRunnerMixin


class CORPAgent(HTNMethodsMixin, EpisodeRunnerMixin):
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
        # R4: the combat manager's tactic engine runs the LIVE program's rules
        # (program v4+ carries the interlock rules alongside LLM-authored ones)
        self.combat_mgr.tactic_engine = TacticalCombatManager.tactic_engine_from(
            self.goals.program.tactic_rules)
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
            # R4: role profile overlay (defaults ← domain ← role ← params)
            if hasattr(self.goals, "apply_role_profile"):
                self.goals.apply_role_profile(self.current_role)
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
