"""
LOX-ψ Policy Layer (LOX-ψ): GoalInterpreter (R2) — evaluates the policy program's strategy_plan
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

from lox.policy.goal_state import AscensionPhase, MacroDirectorState
from lox.env.blstats import BottomLineStats
from lox.policy.config import PolicyConfig, default_config
from lox.policy.predicates import eval_condition, nethack_bindings
from lox.policy.program import PolicyProgram, DEFAULT_PROGRAM_PATH


def _program_uses_failure_predicate(program: PolicyProgram) -> bool:
    """True when any strategy goal / tactic rule / nogood condition references
    failures_in_10_episodes_ge — the only case where failure counters must be loaded."""
    needle = "failures_in_10_episodes_ge"
    for g in program.strategy_plan:
        if needle in g.when or needle in g.until:
            return True
    for r in program.tactic_rules:
        if needle in (r.get("when") or "") or needle in (r.get("unless") or ""):
            return True
    for ng in program.nogoods:
        if needle in (ng.get("when") or ""):
            return True
    return False


class GoalInterpreter:
    def __init__(self, config: PolicyConfig | None = None, program_path: str = DEFAULT_PROGRAM_PATH,
                 failure_counters: dict[str, int] | None = None):
        self.cfg = config or default_config()
        self.program_path = program_path
        self.program = PolicyProgram.load(program_path)
        # R2: apply the params overlay onto a private copy of the config so the executor
        # reads program-tuned values while defaults stay untouched for other consumers.
        if self.program.params:
            self.program.apply_overlay(self.cfg)
        # S0 (AGENT_PLAN §1.3): cross-episode goal-failure counters from DuckDB
        # goal_events → the failures_in_10_episodes_ge condition predicate. Loaded
        # lazily — and ONLY when the program actually references that predicate
        # (the live program does not, so the per-process DuckDB query is skipped).
        self.failure_counters = failure_counters
        self._uses_failure_predicate = _program_uses_failure_predicate(self.program)
        self.state = MacroDirectorState()
        self.reset()

    def apply_role_profile(self, role: str) -> list[str]:
        """R4: applies the program's role_profile overlay onto the live config.
        Precedence: defaults ← domain_profile ← role_profile ← program.params (the
        params overlay already ran at construction, so a role profile value for a path
        also present in params is overridden here — role is the more specific scope)."""
        from lox.policy.profiles import apply_profile_overlay  # noqa: PLC0415
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
        turns_on_current_level: int = 0,
        closest_monster_dist: int = 99,
    ) -> AscensionPhase:
        st = self.state
        st.turns_on_current_level = turns_on_current_level
        st.closest_monster_dist = closest_monster_dist
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

        # --- R6 milestone flag detection (inventory scan + message hooks) ---
        if inv_tracker is not None:
            for it in active:
                desc = it.raw_str.lower()
                if "gray dragon scale" in desc or "ring of magic resistance" in desc:
                    st.mr_obtained = True
                if any(k in desc for k in ("lamp", "lantern", "candle", "torch")):
                    st.light_source_carried = True
                if "wand of wishing" in desc:
                    st.wand_of_wishing_carried = True
                if "amulet of yendor" in desc:
                    st.amulet_obtained = True
                if "candelabrum" in desc:
                    st.candelabrum_obtained = True
                    st.vlad_defeated = True  # the Candelabrum drops from Vlad
                if "bell of opening" in desc or "silver bell" in desc:
                    st.bell_of_opening_obtained = True
                if "book of the dead" in desc:
                    st.book_of_the_dead_obtained = True
                if "bag of holding" in desc or "dragon scale mail" in desc:
                    # Castle prize acquired — the wishing protocol worked
                    st.castle_wishing_done = True
        if "vibrating square" in msg_lower:
            st.invocation_done = True

        self._advance_goals(blstats, role)
        return self.state.current_phase

    # ------------------------------------------------------------------
    # Goal walk
    # ------------------------------------------------------------------
    def _ctx(self, blstats: BottomLineStats, role: str) -> dict:
        st = self.state
        if self.failure_counters is None:
            if self._uses_failure_predicate:
                from lox.policy.failure_counters import load_failure_counters  # noqa: PLC0415
                self.failure_counters = load_failure_counters()
            else:
                self.failure_counters = {}  # predicate unused: never touch DuckDB
        return {
            # S0: cross-episode goal-failure counters (goal_events → vocabulary)
            "failures": dict(self.failure_counters),
            "failures_total": sum(self.failure_counters.values()),
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
            "turns_on_current_level": getattr(st, "turns_on_current_level", 0),
            "closest_monster_dist": getattr(st, "closest_monster_dist", 99),
            # R6 milestone context
            "has_mr": st.mr_obtained,
            "has_light": st.light_source_carried,
            "has_wishing_wand": st.wand_of_wishing_carried,
            "has_amulet": st.amulet_obtained,
            "has_candelabrum": st.candelabrum_obtained,
            "castle_wishing_done": st.castle_wishing_done,
            "vlad_defeated": st.vlad_defeated,
            "invocation_done": st.invocation_done,
        }

    def _advance_goals(self, blstats: BottomLineStats, role: str) -> None:
        plan = self.program.strategy_plan
        if self._goal_index >= len(plan):
            self._set_phase(AscensionPhase.DEEP_DESCENT)
            return

        goal = plan[self._goal_index]
        self._goal_steps += 1
        ctx = self._ctx(blstats, role)

        # `until` completion → advance (cascading over skipped/completed successors).
        # R6 milestone goals additionally honor policy-param thresholds (the
        # set_threshold goal op tunes the config leaves → apply_overlay → here).
        try:
            done = eval_condition(goal.until, nethack_bindings(ctx))
        except ValueError as e:
            print(f"[policy] goal {goal.goal!r} until-condition error: {e}")
            done = False
        done = done or self._r6_until(goal, blstats)
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
                when_ok = eval_condition(goal.when, nethack_bindings(ctx)) and self._r6_when(goal, blstats)
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
            "early_survival_stack": AscensionPhase.SURVIVAL_STACK,
            "forge_excalibur": AscensionPhase.EXCALIBUR_FORGE,
            "hunt_poison_res": AscensionPhase.POISON_RES_HUNT,
            "enter_sokoban": AscensionPhase.SOKOBAN_PROGRESSION,
            "equip_upgrade": AscensionPhase.EQUIP_UPGRADE,
            "goto_minetown": AscensionPhase.MINETOWN_PROTECTION,
            "survival_intrinsics": AscensionPhase.INTRINSICS,
            "descend": AscensionPhase.DEEP_DESCENT,
            "enter_gehennom": AscensionPhase.GEHENNOM,
            "castle_wishing": AscensionPhase.CASTLE,
            "vlad_invocation": AscensionPhase.VLAD_INVOCATION,
            "ascension_run": AscensionPhase.ASCENSION_RUN,
        }
        self._set_phase(phase_map.get(goal.goal, self.state.current_phase))

    # ------------------------------------------------------------------
    # R6 milestone gating: policy-param-driven when/until for the ascension
    # stack goals. Thresholds live in PolicyConfig.strategy (tunable via the
    # set_threshold goal op through owned_params); the plan's literal until
    # strings mirror the defaults for report/inspection readability only.
    # ------------------------------------------------------------------
    def _r6_until(self, goal, blstats: BottomLineStats) -> bool:
        g = goal.goal
        s = self.cfg.strategy
        if g == "early_survival_stack":
            return (blstats.ac <= s.survival_ac_target
                    or blstats.depth >= s.survival_exit_depth
                    or self._goal_steps >= s.survival_timeout_steps)
        if g == "equip_upgrade":
            return (blstats.ac <= s.equip_ac_target
                    or blstats.depth >= s.equip_exit_depth
                    or self._goal_steps >= s.equip_timeout_steps)
        if g == "survival_intrinsics":
            return (self.state.mr_obtained
                    or blstats.depth >= s.intrinsics_exit_depth
                    or self._goal_steps >= s.intrinsics_timeout_steps)
        return False

    def _r6_when(self, goal, blstats: BottomLineStats) -> bool:
        """Milestone activation gates: no milestone may be skipped-to."""
        g = goal.goal
        s = self.cfg.strategy
        if g == "survival_intrinsics":
            return blstats.depth >= s.intrinsics_min_depth
        if g == "enter_gehennom":
            return blstats.depth >= s.gehennom_min_depth
        if g == "castle_wishing":
            return blstats.depth >= s.castle_min_depth
        if g == "vlad_invocation":
            return blstats.depth >= s.vlad_min_depth
        return True

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
        if goal.goal == "enter_gehennom":
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
        if goal.goal == "ascension_run":
            # Amulet carried: climb the Planes back up; before the Invocation,
            # keep pushing down toward the Vibrating Square.
            if dnum == 0:
                return "ASCEND" if self.state.amulet_obtained else "DESCEND"
            return "ASCEND" if self.state.amulet_obtained else None
        return None

    def activate_goal(self, goal_name: str) -> bool:
        """Programmatically activates a strategy_plan goal by name (fixture/\
        tooling seam: tests, certifications, and the deadlock resolver's restart
        path). Returns False if the goal is not in the live program's plan."""
        plan = self.program.strategy_plan
        for i, g in enumerate(plan):
            if g.goal == goal_name:
                self._goal_index = i
                self._goal_steps = 0
                self._set_phase_for_goal(g)
                return True
        return False

    def get_minetown_donation_target(self, blstats: BottomLineStats) -> int:
        """Full divine protection requires ~5 donations of 400 * XL gold each
        (2-4 AC points per donation, targeting AC <= -5)."""
        return 400 * max(1, blstats.experience_level) * 5

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
        import time as _time
        for attempt in range(5):
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
                err_str = str(e).lower()
                if attempt < 4 and ("lock" in err_str or "conflict" in err_str):
                    _time.sleep(0.05 * (2 ** attempt))
                    continue
                if "lock" not in err_str and "conflict" not in err_str:
                    print(f"[policy] goal_events flush failed: {e}")
                return 0
        return 0

