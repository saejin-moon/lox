"""
Agent Layer: CORPAgent.
Top-level cognitive operating system integrating microsecond fast spine
(HTN + A* + CDCL Nogoods + Domain Managers) with deliberative LLM supervision.
"""

from dataclasses import dataclass, field
from typing import Any
import asyncio
import numpy as np
import gymnasium as gym
from nle import nethack

from corp.env.auto_more import AutoMoreWrapper
from corp.env.blstats import BottomLineStats, HungerState
from corp.env.inventory_tracker import InventoryNormalizer
from corp.planner.persona import PersonaProfiler, PersonaTraitVector
from corp.planner.nogood import NogoodStore, NogoodEntry
from corp.planner.predicates import compile_predicate_mask, PredicateBit
from corp.planner.htn import HTNPlanner, CycleDetector, PlanSignature, Task, PrimitiveTask
from corp.domain.combat_manager import TacticalCombatManager
from corp.domain.navigation_manager import NavigationManager
from corp.domain.inventory_manager import InventoryManager
from corp.workers.dispatcher import ActionDispatcher
from corp.epistemic.epistemic_manager import EpistemicManager
from corp.deliberative.autopsy_engine import AutopsyEngine
from corp.deliberative.deadlock_resolver import DeadlockResolver
from corp.deliberative.providers.mock_provider import MockProvider


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
    ):
        self.env = env
        self.nogood_store = nogood_store or NogoodStore()
        self.llm_provider = llm_provider or MockProvider()
        self.enable_deliberative_autopsy = enable_deliberative_autopsy

        # Fast spine subsystems
        self.profiler = PersonaProfiler()
        self.epistemic = EpistemicManager()
        self.combat_mgr = TacticalCombatManager()
        self.nav_mgr = NavigationManager()
        self.inv_mgr = InventoryManager()
        self.dispatcher = ActionDispatcher(env)
        self.cycle_detector = CycleDetector(window_size=10, max_repetitions=3)

        # Deliberative subsystems
        self.deadlock_resolver = DeadlockResolver(self.llm_provider)
        self.autopsy_engine = AutopsyEngine(self.llm_provider, self.nogood_store)

        # Active episode state
        self.persona: PersonaTraitVector | None = None
        self.current_blstats: BottomLineStats | None = None
        self.current_chars: np.ndarray | None = None
        self.current_glyphs: np.ndarray | None = None
        self.max_depth_reached: int = 1
        self.is_terminal: bool = False
        self.terminal_message: str = ""

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """Initializes a new episode and derives continuous Turn-0 persona."""
        obs, info = self.env.reset(**kwargs)
        self.current_blstats = info["blstats"]
        self.current_chars = obs["chars"]
        self.current_glyphs = obs["glyphs"]
        self.max_depth_reached = self.current_blstats.depth
        self.is_terminal = False
        self.terminal_message = ""
        self.cycle_detector.reset()

        # Turn-0 Continuous Persona Profiling
        self.persona = self.profiler.derive_persona(
            blstats=self.current_blstats,
            inventory=self.env.inventory_tracker,
        )

        return obs, info

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

        # 1. Compile 64-bit state predicate mask
        state_mask = compile_predicate_mask(
            blstats=blstats,
            inventory=inv_tracker,
        )

        # 2. Resource & Emergency Management (Prayer, Hunger, Equipment)
        resource_task = self.inv_mgr.evaluate_resource_turn(blstats, inv_tracker)
        if resource_task is not None:
            if not self._is_nogood(resource_task, state_mask):
                return resource_task

        # 3. Tactical Combat Management (Hostiles, Kiting, Corridor Funneling, Melee)
        combat_task = self.combat_mgr.evaluate_combat_turn(glyphs, chars, blstats)
        if combat_task is not None:
            if not self._is_nogood(combat_task, state_mask):
                return combat_task

        # 4. Spatial Navigation & Exploration (Stairs, Frontiers, Secret Doors)
        nav_task = self.nav_mgr.evaluate_navigation_turn(chars, blstats)

        # 5. Cycle Detection & Oscillation Interlock
        py, px = blstats.y, blstats.x
        sig = PlanSignature(task_name=nav_task.name, target_pos=(py, px), turn=blstats.turn)
        if self.cycle_detector.record_and_check(sig):
            # Oscillation loop detected! Perform localized secret door search to break cycle
            return Task("SEARCH", is_primitive=True)

        return nav_task

    def step(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        Executes one full cognitive step: selects tactical task, dispatches keystroke,
        and updates telemetry.
        """
        task = self.select_action()
        obs, reward, terminated, truncated, info = self.dispatcher.dispatch(task)

        self.current_blstats = info["blstats"]
        self.current_chars = obs["chars"]
        self.current_glyphs = obs["glyphs"]
        if self.current_blstats.depth > self.max_depth_reached:
            self.max_depth_reached = self.current_blstats.depth

        if terminated or truncated:
            self.is_terminal = True
            self.terminal_message = info.get("full_message", "")

        return obs, reward, terminated, truncated, info

    async def run_episode_async(self, max_steps: int = 50000) -> EpisodeResult:
        """
        Runs the full autonomous game loop until game over or step budget exhaustion.
        Triggers post-mortem autopsy upon death.
        """
        import time

        self.reset()
        start_time = time.perf_counter()
        steps = 0

        while not self.is_terminal and steps < max_steps:
            self.step()
            steps += 1

        elapsed = max(1e-4, time.perf_counter() - start_time)
        sps = float(steps) / elapsed

        is_ascended = (
            self.current_blstats is not None
            and self.current_blstats.depth >= 50
            and "ascended" in self.terminal_message.lower()
        )

        # Post-mortem autopsy for cross-generational Nogood synthesis
        if self.is_terminal and not is_ascended and self.enable_deliberative_autopsy:
            try:
                trajectory = self.env.flight_recorder.get_recent_frames(50)
                await self.autopsy_engine.conduct_autopsy(
                    flight_trajectory=trajectory,
                    death_message=self.terminal_message,
                    revealed_inventory=[],
                    dlvl=self.max_depth_reached,
                    turn=self.current_blstats.turn if self.current_blstats else 0,
                )
            except Exception:
                pass # Autopsy failure must not crash benchmark runner

        return EpisodeResult(
            turns=self.current_blstats.turn if self.current_blstats else steps,
            max_depth=self.max_depth_reached,
            final_score=self.current_blstats.score if self.current_blstats else 0,
            final_hp=self.current_blstats.hp if self.current_blstats else 0,
            is_ascended=is_ascended,
            death_message=self.terminal_message,
            active_nogoods_count=len(self.nogood_store.entries),
            steps_per_second=sps,
        )

    def run_episode(self, max_steps: int = 50000) -> EpisodeResult:
        """Synchronous wrapper for running an autonomous episode."""
        return asyncio.run(self.run_episode_async(max_steps=max_steps))

    def _is_nogood(self, task: Task, state_mask: int) -> bool:
        """Checks if the proposed task violates any active CDCL Nogood constraints."""
        forbidden, _ = self.nogood_store.is_forbidden(state_mask, task.name)
        return forbidden
