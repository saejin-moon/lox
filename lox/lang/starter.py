"""
Clean Option-A Bare Policy Starter (`WorldState` + Goal/Task State Machine).

Strictly adheres to LOX's "Hard Code Nothing" principle (`AGENTS.md`):
- Zero hardcoded monster/item/corpse lists, HP thresholds, prayer cooldowns, or gameplay heuristics.
- Provides only pure UI prompt safety, NLE C-terrain doorway geometry (`no_diag_mask`, `path_step`),
  typed `WorldState` with domain-owned `ws.domain(...)` sub-states, and a plugin `GoalRunner`.
"""

from __future__ import annotations

WORLD_PY = '''"""Typed per-episode & per-floor observation cache with domain sub-state registry."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class FloorCache:
    """Per-(dungeon_num, level_num) geometric and blocked-coordinate cache."""

    visited: set[tuple[int, int]] = field(default_factory=set)
    blocked_until: dict[tuple[int, int], int] = field(default_factory=dict)

    def blacklist(self, pos: tuple[int, int], until_turn: int) -> None:
        self.blocked_until[(int(pos[0]), int(pos[1]))] = int(until_turn)

    def is_blocked(self, pos: tuple[int, int], turn_now: int) -> bool:
        exp = self.blocked_until.get((int(pos[0]), int(pos[1])))
        if exp is None:
            return False
        if int(turn_now) >= exp:
            self.blocked_until.pop((int(pos[0]), int(pos[1])), None)
            return False
        return True


@dataclass
class WorldState:
    """Shared typed episode state updated once per turn via `observe(state)`."""

    turn: int = 0
    step_idx: int = 0
    last_pos: tuple[int, int] | None = None
    last_floor_key: tuple[int, int] = (0, 1)
    pending_move_target: tuple[int, int] | None = None
    active_goal: str = ""
    last_active_goal: str = ""
    floors: dict[tuple[int, int], FloorCache] = field(default_factory=dict)
    domains: dict[str, Any] = field(default_factory=dict)
    # Single-pass per-turn cached masks (21x79 bool arrays)
    passable_mask: Any = None
    safe_passable_mask: Any = None
    no_diag_mask: Any = None

    def floor(self, state: Any) -> FloorCache:
        fk = (
            int(getattr(state, "dungeon_num", 0) or 0),
            int(getattr(state, "level_num", 1) or 1),
        )
        fc = self.floors.get(fk)
        if fc is None:
            fc = FloorCache()
            self.floors[fk] = fc
        return fc

    def domain(self, name: str, factory: Any) -> Any:
        """Returns (or initializes via `factory()`) a typed domain-specific state object."""
        obj = self.domains.get(name)
        if obj is None:
            obj = factory() if callable(factory) else factory
            self.domains[name] = obj
        return obj

    def observe(self, state: Any) -> None:
        """Updates per-turn coordinates, runtime stall feedback, and cached pathing masks."""
        self.step_idx += 1
        self.turn = int(getattr(state, "turn", 0) or 0)
        self.last_active_goal = ""
        fk = (
            int(getattr(state, "dungeon_num", 0) or 0),
            int(getattr(state, "level_num", 1) or 1),
        )
        pos = (int(getattr(state, "y", 0) or 0), int(getattr(state, "x", 0) or 0))
        fc = self.floor(state)
        fc.visited.add(pos)

        # Record runtime movement obstruction strictly from zero-turn feedback (no hardcoded tile heuristics)
        if (
            self.pending_move_target is not None
            and fk == self.last_floor_key
            and not bool(getattr(state, "turn_advanced", True))
            and not bool(getattr(state, "in_menu", False))
            and not bool(getattr(state, "in_text_prompt", False))
            and not bool(getattr(state, "in_yn_prompt", False))
        ):
            fc.blacklist(self.pending_move_target, self.turn + 25)
        self.pending_move_target = None
        self.last_floor_key = fk
        self.last_pos = pos

        # Single-pass terrain & C-doorway no_diag masks per step
        try:
            self.passable_mask = state.passable_mask
            self.safe_passable_mask = state.safe_passable_mask
            no_diag = np.zeros((21, 79), dtype=bool)
            for ry in range(21):
                for rx in range(79):
                    if state.is_closed_door(ry, rx) or state.is_open_door(ry, rx):
                        no_diag[ry, rx] = True
            self.no_diag_mask = no_diag
        except Exception:
            self.passable_mask = None
            self.safe_passable_mask = None
            self.no_diag_mask = None
'''

PROMPTS_PY = '''"""Pure mechanical UI/prompt dismissal plumbing (zero hardcoded gameplay choices)."""

from lox.env.actions import Action, key


def handle_prompts(state, ws):
    """Unblocks pending TTY UI prompts safely without assuming strategy."""
    if getattr(state, "waiting_for_space", False):
        return Action.MORE
    if getattr(state, "in_text_prompt", False) or getattr(state, "in_naming_prompt", False):
        return Action.ESC
    if getattr(state, "in_menu", False):
        return Action.ESC
    if getattr(state, "in_yn_prompt", False):
        return key("n")
    return None
'''

SURVIVAL_PY = '''"""Survival interrupts and resource management hooks (to be synthesized)."""


def check_interrupt(state, ws):
    """High-priority survival check before goal execution."""
    return None
'''

COMBAT_PY = '''"""Combat & threat handling hooks (to be synthesized)."""


def check_threat(state, ws):
    """Tactical combat interrupt before non-combat goals."""
    return None
'''

NAVIGATION_PY = '''"""Pure geometric pathing helpers over GameState C masks and DIR_DELTAS."""

from lox.env.actions import DIR_DELTAS


_DELTA_TO_ACTION = {delta: act for act, delta in DIR_DELTAS.items()}


def step_toward(state, ws, target, allow_diagonal=True):
    """Computes one pure A* step toward `target` honoring NLE doorway `no_diag` geometry."""
    mask = ws.safe_passable_mask if ws.safe_passable_mask is not None else ws.passable_mask
    if mask is None or target is None:
        return None
    dy, dx = state.path_step(
        mask,
        (int(target[0]), int(target[1])),
        allow_diagonal=bool(allow_diagonal),
        no_diag=ws.no_diag_mask,
    )
    if dy == 0 and dx == 0:
        return None
    ws.pending_move_target = (int(state.y) + dy, int(state.x) + dx)
    return _DELTA_TO_ACTION.get((dy, dx))
'''

GOALS_PY = '''"""Plugin Goal registry and committed task dispatcher."""

from dataclasses import dataclass
from typing import Any, Callable
from lox.env.actions import Action


@dataclass
class GoalSpec:
    name: str
    priority: int
    trigger: Callable[[Any, Any], bool]
    step: Callable[[Any, Any], Any]
    preempt: Callable[[Any, Any], bool] | None = None


_GOAL_REGISTRY: dict[str, GoalSpec] = {}


def register_goal(
    name: str,
    priority: int = 50,
    trigger: Callable[[Any, Any], bool] | None = None,
    step: Callable[[Any, Any], Any] | None = None,
    preempt: Callable[[Any, Any], bool] | None = None,
) -> None:
    """Registers or replaces a named Goal handler without touching `policy.py`."""
    if trigger is None or step is None:
        return
    _GOAL_REGISTRY[str(name)] = GoalSpec(
        name=str(name),
        priority=int(priority),
        trigger=trigger,
        step=step,
        preempt=preempt,
    )


class GoalRunner:
    """Runs active committed goals, switches cleanly on completion/preemption, and attributes turns."""

    def tick(self, state: Any, ws: Any) -> Any:
        specs = sorted(_GOAL_REGISTRY.values(), key=lambda g: g.priority)
        if ws.active_goal:
            cur = _GOAL_REGISTRY.get(ws.active_goal)
            if cur is None or (cur.preempt is not None and cur.preempt(state, ws)):
                ws.active_goal = ""
            else:
                act = cur.step(state, ws)
                if act is not None:
                    ws.last_active_goal = cur.name
                    return act
                ws.active_goal = ""

        for spec in specs:
            if spec.trigger(state, ws):
                act = spec.step(state, ws)
                if act is not None:
                    ws.active_goal = spec.name
                    ws.last_active_goal = spec.name
                    return act
        return None


def fallback_step(state: Any, ws: Any) -> Any:
    """Bare Gen-0 fallback stub until exploration/movement goals are synthesized."""
    ws.last_active_goal = "fallback_step"
    return Action.SEARCH
'''

POLICY_PY = '''"""Option-A Goal/Task Policy Orchestrator (thin loop over typed WorldState + Skills)."""

import skills.world as world
import skills.prompts as prompts
import skills.survival.core as survival
import skills.combat.core as combat
import skills.navigation.core as navigation  # noqa: F401
import skills.goals.core as goals


class Policy:
    def __init__(self):
        self.ws = world.WorldState()
        self.goals = goals.GoalRunner()

    def step(self, state):
        self.ws.observe(state)

        act = prompts.handle_prompts(state, self.ws)
        if act is not None:
            self.ws.last_active_goal = "handle_prompts"
            return act

        act = survival.check_interrupt(state, self.ws)
        if act is not None:
            self.ws.last_active_goal = "survival_interrupt"
            return act

        act = combat.check_threat(state, self.ws)
        if act is not None:
            self.ws.last_active_goal = "combat_threat"
            return act

        act = self.goals.tick(state, self.ws)
        if act is not None:
            return act

        return goals.fallback_step(state, self.ws)
'''


def build_clean_option_a_starter() -> tuple[str, dict[str, str]]:
    """Returns `(policy_code, skills_dict)` for a bare, zero-heuristic Option-A Gen 0 start."""
    skills = {
        "world": WORLD_PY.strip() + "\n",
        "prompts": PROMPTS_PY.strip() + "\n",
        "survival/core": SURVIVAL_PY.strip() + "\n",
        "combat/core": COMBAT_PY.strip() + "\n",
        "navigation/core": NAVIGATION_PY.strip() + "\n",
        "goals/core": GOALS_PY.strip() + "\n",
    }
    return POLICY_PY.strip() + "\n", skills
