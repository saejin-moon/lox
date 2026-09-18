"""
CORP-Ω Policy Layer: GoalInterpreter (R2) — evaluates the policy program's strategy_plan
and emits navigation directives. Drop-in replacement for MacroAscensionDirector's phase
machine: same public surface (update_state / get_navigation_directive /
should_defer_stairs_for_farming / state / reset), same default behavior.

Safety-critical branch pre-passes (mines retreat, sokoban exit) run BEFORE the goal walk —
they are policy-independent interlocks, not goals the LLM may reorder (documented in
MACRO.md §10 non-goals: goals enter/leave the vocabulary by certification, not by diff).
"""
from __future__ import annotations

import os
import time
from typing import Any

from corp.domain.macro_director import AscensionPhase, MacroDirectorState
from corp.env.blstats import BottomLineStats
from corp.policy.config import PolicyConfig, default_config
from corp.policy.predicates import eval_condition, nethack_bindings
from corp.policy.program import PolicyProgram, DEFAULT_PROGRAM_PATH


class GoalInterpreter:
    def __init__(self, config: PolicyConfig | None = None, program_path: str = DEFAULT_PROGRAM_PATH):
        self.cfg = config or default_config()
        self.program_path = program_path
        self.program = PolicyProgram.load(program_path)
        # R2: apply the params overlay onto a private copy of the config so the executor
        # reads program-tuned values while defaults stay untouched for other consumers.
        if self.program.params:
            self.program.apply_overlay(self.cfg)
        self.state = MacroDirectorState()
        self.reset()

    def apply_role_profile(self, role: str) -> list[str]:
        """R4: applies the program's role_profile overlay onto the live config.
        Precedence: defaults ← domain_profile ← role_profile ← program.params (the
        params overlay already ran at construction, so a role profile value for a path
        also present in params is overridden here — role is the more specific scope)."""
        from corp.policy.profiles import apply_profile_overlay  # noqa: PLC0415
        rp = (self.program.role_profiles or {}).get((role or "").lower())
        if not rp:
            return []
        return apply_profile_overlay(self.cfg, rp, f"role_profile[{role}]")

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.state = MacroDirectorState()
        self._goal_index: int = 0
        self._goal_steps: int = 0
        self._skipped: set[int] = set()
        self._completed: set[int] = set()
        self._events: list[dict] = []
        self._last_active: str | None = None

    # ------------------------------------------------------------------
    # Public surface 1: state update (drop-in for MacroAscensionDirector.update_state)
    # ------------------------------------------------------------------
    def update_state(
        self,
        blstats: BottomLineStats,
        message: str = "",
        inv_tracker: Any = None,  # noqa: ANN401
        dungeon_graph: Any = None,  # noqa: ANN401
        role: str = "",
        has_poison_res: bool = False,
        has_reflection: bool = False,
        temple_donations: int = 0,
    ) -> AscensionPhase:
        st = self.state
        st.total_steps_in_phase += 1
        msg_lower = (message or "").lower()

        if temple_donations > st.temple_donations_count:
            st.temple_donations_count = temple_donations
        if has_poison_res or st.poison_res_obtained:
            st.poison_res_obtained = True
        if "feel healthy" in msg_lower or "feel especially healthy" in msg_lower:
            st.poison_res_obtained = True
        if has_reflection or st.reflection_obtained:
            st.reflection_obtained = True
        if blstats.depth > st.max_dlevel_reached:
            st.max_dlevel_reached = blstats.depth
        if "welcome to minetown" in msg_lower:
            st.minetown_visited = True

        if inv_tracker is not None:
            active = (
                inv_tracker.get_active_items()
                if hasattr(inv_tracker, "get_active_items")
                else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
            )
            for it in active:
                desc = it.raw_str.lower()
                if "excalibur" in desc:
                    st.excalibur_obtained = True
                if any(k in desc for k in ("shield of reflection", "amulet of reflection", "silver dragon")):
                    st.reflection_obtained = True

        # Sokoban prize: reflection obtained while inside Sokoban → exit the branch
        if getattr(blstats, "dungeon_number", 0) == 3 and st.reflection_obtained:
            st.sokoban_prize_collected = True

        self._advance_goals(blstats, role)
        return self.state.current_phase

    # ------------------------------------------------------------------
    # Goal walk
    # ------------------------------------------------------------------
    def _ctx(self, blstats: BottomLineStats, role: str) -> dict:
        st = self.state
        return {
            "xl": blstats.experience_level,
            "depth": blstats.depth,
            "dnum": blstats.dungeon_number,
            "lawful": (blstats.alignment == 1) or (role.lower() in ("valkyrie", "samurai", "knight")),
            "has_poison_res": st.poison_res_obtained,
            "has_reflection": st.reflection_obtained,
            "has_excalibur": st.excalibur_obtained,
            "sokoban_completed": st.sokoban_completed,
            "minetown_visited": st.minetown_visited,
            "donations_count": st.temple_donations_count,
            "ac": blstats.ac,
            "steps_in_goal": self._goal_steps,
            "turn": blstats.turn,
        }

    def _advance_goals(self, blstats: BottomLineStats, role: str) -> None:
        plan = self.program.strategy_plan
        if self._goal_index >= len(plan):
            self._set_phase(AscensionPhase.DEEP_DESCENT)
            return

        goal = plan[self._goal_index]
        self._goal_steps += 1
        ctx = self._ctx(blstats, role)

        # `until` completion → advance (cascading over skipped/completed successors)
        try:
            done = eval_condition(goal.until, nethack_bindings(ctx))
        except ValueError as e:
            print(f"[policy] goal {goal.goal!r} until-condition error: {e}")
            done = False
        over_budget = goal.max_steps is not None and self._goal_steps > goal.max_steps

        if done or over_budget:
            self._emit_event(blstats, goal.goal, "completed" if done else "skipped_budget", self._goal_steps)
            self._completed.add(self._goal_index)
            self._advance_to_next_active(blstats, role, ctx)
        else:
            # Emit "activated" on goal change (dedup inside _emit_event) so every episode
            # records at least its initial goal activation for telemetry
            self._emit_event(blstats, goal.goal, "activated", self._goal_steps)
            self._set_phase_for_goal(goal)

    def _advance_to_next_active(self, blstats: BottomLineStats, role: str, ctx: dict) -> None:
        plan = self.program.strategy_plan
        idx = self._goal_index + 1
        while idx < len(plan):
            goal = plan[idx]
            if idx in self._skipped or idx in self._completed:
                idx += 1
                continue
            try:
                when_ok = eval_condition(goal.when, nethack_bindings(ctx))
            except ValueError as e:
                print(f"[policy] goal {goal.goal!r} when-condition error: {e}")
                when_ok = False
            if when_ok:
                self._goal_index = idx
                self._goal_steps = 0
                self._emit_event(blstats, goal.goal, "activated", 0)
                self._set_phase_for_goal(goal)
                return
            # when false at activation time → skipped permanently (mirrors the elif chain)
            self._skipped.add(idx)
            self._emit_event(blstats, goal.goal, "skipped_when", 0)
            idx += 1
        self._goal_index = len(plan)
        self._goal_steps = 0
        self._set_phase(AscensionPhase.DEEP_DESCENT)

    def _set_phase_for_goal(self, goal) -> None:
        phase_map = {
            "explore_floor": AscensionPhase.EARLY_EXPLORATION,
            "forge_excalibur": AscensionPhase.EXCALIBUR_FORGE,
            "hunt_poison_res": AscensionPhase.POISON_RES_HUNT,
            "enter_sokoban": AscensionPhase.SOKOBAN_PROGRESSION,
            "goto_minetown": AscensionPhase.MINETOWN_PROTECTION,
            "descend": AscensionPhase.DEEP_DESCENT,
        }
        self._set_phase(phase_map.get(goal.goal, self.state.current_phase))

    def _set_phase(self, phase: AscensionPhase) -> None:
        if phase != self.state.current_phase:
            self.state.current_phase = phase
            self.state.total_steps_in_phase = 0

    # ------------------------------------------------------------------
    # Public surface 2: navigation directives
    # ------------------------------------------------------------------
    def get_navigation_directive(self, blstats: BottomLineStats, dungeon_graph: Any = None) -> str | None:
        dnum = int(getattr(blstats, "dungeon_number", 0))
        depth = int(getattr(blstats, "depth", 1))
        dlevel = int(getattr(blstats, "dlevel", 0) or 0)
        d = self.cfg.descent

        # --- Safety-critical pre-passes (policy-independent interlocks) ---
        if dnum == 2:
            if dlevel >= d.mines_end_dlevel:
                return "ASCEND_FROM_MINES"
            if self.state.minetown_visited and self.state.temple_donations_count >= d.minetown_donations_done:
                return "ASCEND_FROM_MINES"
            if dungeon_graph is not None:
                node = dungeon_graph.nodes.get((dnum, dlevel))
                if node is not None and node.fully_explored and node.stairs_down is None:
                    return "ASCEND_FROM_MINES"
            return None

        if dnum == 3:
            return "EXIT_SOKOBAN" if self.state.sokoban_prize_collected else None

        # --- Goal-directed directives ---
        plan = self.program.strategy_plan
        if self._goal_index >= len(plan):
            return "DESCEND" if dnum == 0 else None
        goal = plan[self._goal_index]

        if goal.goal == "descend":
            return "DESCEND" if dnum == 0 else None
        if goal.goal == "enter_sokoban":
            return "ENTER_SOKOBAN" if dnum == 0 else None
        if goal.goal == "goto_minetown":
            if (
                dnum == 0
                and not self.state.minetown_visited
                and d.minetown_depth_lo <= depth <= d.minetown_depth_hi
            ):
                return "GOTO_MINETOWN"
            return None
        return None

    # ------------------------------------------------------------------
    # Public surface 3: stair-farming deferral (phase-independent, as pre-R2)
    # ------------------------------------------------------------------
    def should_defer_stairs_for_farming(
        self,
        blstats: BottomLineStats,
        unvisited_count: int,
        turns_spent: int,
    ) -> bool:
        if blstats.hunger_state >= 3:  # HungerState.WEAK — never defer when starving
            return False
        s = self.cfg.strategy
        if blstats.depth == 1:
            if unvisited_count > s.farming_unvisited and turns_spent < s.farming_turns_d1 and blstats.experience_level < 2:
                return True
        if blstats.depth in (2, 3):
            if unvisited_count > s.farming_unvisited and turns_spent < s.farming_turns_d23 and blstats.experience_level < 3:
                return True
        return False

    # ------------------------------------------------------------------
    # Telemetry: goal transitions → DuckDB goal_events
    # ------------------------------------------------------------------
    def _emit_event(self, blstats: BottomLineStats, goal: str, event: str, steps: int) -> None:
        if self._last_active != goal or event != "activated":
            self._events.append({
                "ts": time.time(),
                "depth": int(blstats.depth),
                "dnum": int(blstats.dungeon_number),
                "turn": int(blstats.turn),
                "goal": goal,
                "event": event,
                "steps_in_goal": steps,
            })
        self._last_active = goal

    def flush_events(self, db_path: str, run_id: str, episode_id: str) -> int:
        """Appends buffered goal transitions to the DuckDB goal_events table. Returns rows written."""
        if not self._events:
            return 0
        try:
            import duckdb
            con = duckdb.connect(db_path)
            con.execute(
                """CREATE TABLE IF NOT EXISTS goal_events (
                       ts DOUBLE, run_id VARCHAR, episode_id VARCHAR,
                       depth INTEGER, dnum INTEGER, turn INTEGER,
                       goal VARCHAR, event VARCHAR, steps_in_goal INTEGER)"""
            )
            rows = [
                (e["ts"], run_id, episode_id, e["depth"], e["dnum"], e["turn"],
                 e["goal"], e["event"], e["steps_in_goal"])
                for e in self._events
            ]
            con.executemany(
                "INSERT INTO goal_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
            )
            con.close()
            n = len(rows)
            self._events = []
            return n
        except Exception as e:
            print(f"[policy] goal_events flush failed: {e}")
            return 0
