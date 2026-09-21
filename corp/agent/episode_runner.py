"""Episode runner + env step plumbing — extracted from CORPAgent (behavior-preserving split).

Mixed into CORPAgent; references agent state via self.*"""

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


class EpisodeRunnerMixin:
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

    def _emit_deadlock_diff(self, patch, failed_task: str) -> None:
        """R3 (AGENT_PLAN step 9): converts a DeadlockResolver patch into a policy diff,
        validates it through the gate pipeline, and persists the bumped program on
        acceptance (next-episode effect). Ledger records accept AND reject."""
        try:
            program = self.goals.program
            diff_text = DeadlockResolver.patch_to_diff(
                patch,
                parent_version=program.version,
                failed_task=failed_task,
                depth=self.current_blstats.depth if self.current_blstats else None,
                turn=self.current_blstats.turn if self.current_blstats else 0,
            )
            accepted = emit_deadlock_diff(
                diff_text, program,
                program_path=getattr(self.goals, "program_path", "data/policy_program.json"),
            )
            if accepted:
                # Re-load so the CURRENT episode keeps running on its compiled config;
                # the next episode's GoalInterpreter picks up the bumped program.
                from corp.policy.program import PolicyProgram
                self.goals.program = PolicyProgram.load(
                    getattr(self.goals, "program_path", "data/policy_program.json"))
            from corp.policy.ledger import RevisionLedger
            RevisionLedger().append({
                "type": "deadlock_revision",
                "accepted": accepted,
                "failed_task": failed_task,
                "diff": diff_text[:500],
            })
        except Exception as e:  # noqa: BLE001 — deliberation must never kill the loop
            print(f"[agent] deadlock diff emission failed: {e}")

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
                    # R3: cached deadlock verdicts re-emit as policy diffs (idempotent
                    # — the nogood already lives in the program if it was accepted).
                    self._emit_deadlock_diff(cached_patch, failed_task)
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

                            # R3: deadlock → policy diff through the validator (next-
                            # episode effect). The transient plan_queue injection path
                            # is removed per MACRO.md §10 — the LLM may never bypass
                            # the executor's priority cascade mid-episode.
                            self._emit_deadlock_diff(patch, failed_task)

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
