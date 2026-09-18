"""
CORP-Ω MiniHack DomainAdapter (R5, refactor step 11 — first transfer domain).

MiniHack is built ON NLE: same observation dict (glyphs/chars/blstats/message), same
21×79 grid — the adapter is a thin wrapper around a compact goal-driven navigator.
Task: MiniHack-ExploreMaze-Hard-v0 (descent/reach-stairs loop; the plan's original
task names don't exist in minihack 1.0.2 — ExploreMaze is the closest family, noted
in spec.notes). Fully seeded.

Certified goals: explore_floor (frontier BFS), reach_stairs (A* to '>' — the descend
analogue). Predicates: the same closed vocabulary evaluated on the minihack context.
Params: domain leaves (explore/nav/combat sections) tunable via policy diffs.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from corp.executor.interface import (
    CertCase, DomainAdapter, DomainSpec, GoalHandler, ParamLeaf, PredicateBinding,
)

# ---------------------------------------------------------------------------
# Domain params (the minihack PolicyConfig analogue — diff-tunable leaves)
# ---------------------------------------------------------------------------

@dataclass
class ExplorePolicy:
    stuck_patience: int = 30          # steps without progress → re-target frontier
    revisit_cap: int = 6              # per-tile visit cap before tiles count as blocked
    frontier_bias: float = 1.2        # prefer frontiers away from explored center

@dataclass
class MhNavPolicy:
    unknown_penalty: float = 2.0      # A* cost on unknown tiles
    monster_cost: float = 200.0       # A* cost on hostile-adjacent tiles

@dataclass
class MhCombatPolicy:
    retreat_hp_frac: float = 0.35     # below this, avoid adjacent hostiles instead of fighting
    attack_range: int = 1             # melee only when Chebyshev distance <= this

@dataclass
class MiniHackConfig:
    explore: ExplorePolicy = None
    nav: MhNavPolicy = None
    combat: MhCombatPolicy = None

    def __post_init__(self):
        self.explore = self.explore or ExplorePolicy()
        self.nav = self.nav or MhNavPolicy()
        self.combat = self.combat or MhCombatPolicy()

    def copy(self):
        import copy  # noqa: PLC0415
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Context + predicates
# ---------------------------------------------------------------------------

def minihack_ctx(state, cfg: MiniHackConfig) -> dict:
    """Evaluation context for the closed vocabulary on minihack."""
    return {
        "true": True, "false": False,
        "stairs_known": state.get("stairs_pos") is not None,
        "depth": 1, "dnum": 0,
        "hp": state.get("hp", 10), "max_hp": state.get("max_hp", 10),
        "hp_frac": state.get("hp", 10) / max(1, state.get("max_hp", 10)),
        "xl": 1, "hunger_state": 1,
        "adjacent_hostiles": state.get("adjacent_hostiles", 0),
        "has_healing": False, "is_fighting": state.get("adjacent_hostiles", 0) > 0,
        "has_poison_res": False, "has_reflection": False,
        "monster_name": state.get("monster_name", ""),
        "item_name": state.get("item_name", ""),
        "turn": state.get("steps", 0),
        "encumbrance": 0, "lawful": False,
        "donations_count": 0, "steps_in_goal": state.get("steps_in_goal", 0),
        "failures": {}, "failures_total": 0,
    }


def evaluate_condition(text: str, state: dict, cfg: MiniHackConfig) -> bool:
    from corp.policy.predicates import eval_condition, nethack_bindings  # noqa: PLC0415
    ctx = minihack_ctx(state, cfg)
    bindings = nethack_bindings(ctx)
    # minihack-specific: monster/item match against the observed names
    bindings["monster"] = lambda name: str(name).lower() in ctx["monster_name"].lower()
    bindings["item"] = lambda name: str(name).lower() in ctx["item_name"].lower()
    return eval_condition(text, bindings)


# ---------------------------------------------------------------------------
# The navigator (the domain agent)
# ---------------------------------------------------------------------------

WALKABLE = set(".#+><}_")
STAIRS = ord(">")
HERO = ord("@")

MOVE_DELTAS = {
    (-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3,      # N E S W
    (-1, 1): 4, (1, 1): 5, (1, -1): 6, (-1, -1): 7,    # NE SE SW NW
}


class MiniHackAgent:
    """Goal-driven navigator: reach_stairs when known, else explore_floor (frontier
    BFS). Attacks adjacent hostiles unless wounded below the retreat gate. The
    strategy_plan's active goal selects the handler; conditions are evaluated with
    the same closed vocabulary as NetHack."""

    def __init__(self, env: Any, program: Any, max_steps: int = 4000):
        self.env = env
        self.program = program
        self.max_steps = max_steps
        self.cfg = MiniHackConfig()
        self._apply_params(program)
        self._action_idx = self._build_action_index()
        self._plan = (program.get("strategy_plan", []) if isinstance(program, dict)
                      else [g.raw for g in program.strategy_plan])

    def _apply_params(self, program: Any) -> None:
        """Applies the program's params overlay onto the minihack config (dotted
        paths, same conventions as PolicyProgram.apply_overlay)."""
        params = program.get("params", {}) if isinstance(program, dict) else (program.params or {})
        for dotted, value in params.items():
            parts = dotted.split(".")
            obj = self.cfg
            for p in parts[:-1]:
                obj = getattr(obj, p)
            cur = getattr(obj, parts[-1])
            if isinstance(cur, bool):
                setattr(obj, parts[-1], bool(value))
            elif isinstance(cur, int):
                setattr(obj, parts[-1], int(round(float(value))))
            elif isinstance(cur, float):
                setattr(obj, parts[-1], float(value))

    # ------------------------------------------------------------------
    def run(self, seed: int | None = None) -> dict:
        obs, _info = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        total_reward = 0.0
        visited: set = set()
        goal_events: list[dict] = []
        active_goal = None
        steps = 0

        for steps in range(1, self.max_steps + 1):
            chars = obs["chars"]
            blstats = obs["blstats"]
            hero = (int(blstats[1]), int(blstats[0]))
            visited.add(hero)

            stairs_pos = self._find_stairs(chars)
            state = self._state(obs, hero, chars, stairs_pos, steps)
            goal = self._select_goal(state, goal_events, active_goal)
            if goal != active_goal:
                goal_events.append({"goal": goal, "event": "activated", "step": steps})
                active_goal = goal
                state["steps_in_goal"] = 0

            action = self._decide(obs, state, hero, chars, visited)
            if action is None:
                break
            obs, rew, term, trunc, _info = self.env.step(action)
            total_reward += float(rew)
            if rew > 0:
                last_progress_step = steps
            if term or trunc:
                break

        success = total_reward > 0 or self._on_stairs(obs if steps else None)
        return {
            "reward": total_reward,
            "steps": steps,
            "success": bool(success),
            "coverage": len(visited),
            "goal_events": goal_events,
        }

    # ------------------------------------------------------------------
    def _build_action_index(self) -> dict[tuple[int, int], int]:
        if self.env is None:
            return {}
        from nle.nethack import CompassDirection  # noqa: PLC0415
        actions = list(self.env.unwrapped.actions)
        out = {}
        comp_map = {
            (-1, 0): CompassDirection.N, (0, 1): CompassDirection.E,
            (1, 0): CompassDirection.S, (0, -1): CompassDirection.W,
            (-1, 1): CompassDirection.NE, (1, 1): CompassDirection.SE,
            (1, -1): CompassDirection.SW, (-1, -1): CompassDirection.NW,
        }
        for delta in ((-1, 0), (0, 1), (1, 0), (0, -1),
                      (-1, 1), (1, 1), (1, -1), (-1, -1)):
            out[delta] = actions.index(comp_map[delta])
        return out

    def _find_stairs(self, chars) -> tuple | None:
        pos = np.argwhere(chars == STAIRS)
        return (int(pos[0][0]), int(pos[0][1])) if len(pos) else None

    def _on_stairs(self, obs) -> bool:
        if obs is None:
            return False
        try:
            b = obs["blstats"]
            return int(obs["chars"][int(b[1]), int(b[0])]) == STAIRS
        except Exception:
            return False

    def _state(self, obs, hero, chars, stairs_pos, steps) -> dict:
        glyphs = obs.get("glyphs")
        hostiles = 0
        monster_name = ""
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = hero[0] + dr, hero[1] + dc
                if 0 <= r < chars.shape[0] and 0 <= c < chars.shape[1]:
                    g = int(glyphs[r, c]) if glyphs is not None else 0
                    ch = int(chars[r, c])
                    # letters are monsters; non-pet monsters adjacent = hostile
                    if 65 <= ch <= 90 or 97 <= ch <= 122:
                        hostiles += 1
                        monster_name += chr(ch)
        return {"stairs_pos": stairs_pos, "hp": int(obs["blstats"][10]),
                "max_hp": int(obs["blstats"][11]), "adjacent_hostiles": hostiles,
                "monster_name": monster_name, "item_name": "", "steps": steps,
                "steps_in_goal": getattr(self, "_goal_steps", 0)}

    def _select_goal(self, state: dict, goal_events, active_goal) -> str:
        """Evaluates the program's strategy_plan top-down (same semantics as the
        NetHack GoalInterpreter: first goal whose `when` holds)."""
        for g in self._plan:
            try:
                if evaluate_condition(g["when"], state, self.cfg):
                    return g["goal"]
            except (ValueError, KeyError):
                continue
        return "explore_floor"

    def _decide(self, obs, state, hero, chars, visited):
        hp_frac = state["hp"] / max(1, state["max_hp"])
        # Combat override: attack adjacent hostiles unless below the retreat gate
        if state["adjacent_hostiles"] > 0:
            if hp_frac > self.cfg.combat.retreat_hp_frac:
                return self._step_toward(hero, self._nearest_hostile(hero, chars), visited, chars)
            return self._step_away(hero, chars, visited)

        walkable = self._walkable_mask(chars)
        if state["stairs_pos"] is not None:
            goal = state["stairs_pos"]
        else:
            goal = self._nearest_frontier(hero, chars, visited, walkable)
            if goal is None:
                return None  # nothing left to explore
        nxt = self._next_step(hero, goal, walkable, chars)
        if nxt is None:
            # blocked: bias around by marking current tile heavily visited
            visited.add((hero[0], hero[1], len(visited)))
            return self._explore_fallback(hero, chars, visited)
        return nxt

    def _nearest_hostile(self, hero, chars):
        best, bd = None, 99
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r, c = hero[0] + dr, hero[1] + dc
                if (dr or dc) and 0 <= r < chars.shape[0] and 0 <= c < chars.shape[1]:
                    ch = int(chars[r, c])
                    if 65 <= ch <= 90 or 97 <= ch <= 122:
                        if max(abs(dr), abs(dc)) < bd:
                            best, bd = (r, c), max(abs(dr), abs(dc))
        return best

    def _step_toward(self, hero, target, visited, chars):
        if target is None:
            return None
        dr = (target[0] > hero[0]) - (target[0] < hero[0])
        dc = (target[1] > hero[1]) - (target[1] < hero[1])
        return self._action_for((dr, dc), hero, chars)

    def _step_away(self, hero, chars, visited):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
            r, c = hero[0] + dr, hero[1] + dc
            if 0 <= r < chars.shape[0] and 0 <= c < chars.shape[1] and int(chars[r, c]) in WALKABLE_ORDS:
                return self._action_for((dr, dc), hero, chars)
        return None

    def _walkable_mask(self, chars):
        mask = np.zeros_like(chars, dtype=bool)
        for ch, val in WALKABLE_ORDS.items():
            mask[chars == ch] = True
        return mask

    def _nearest_frontier(self, hero, chars, visited, walkable):
        """BFS from hero over walkable tiles to the nearest tile adjacent to an
        unexplored (never-seen) tile."""
        from collections import deque  # noqa: PLC0415
        seen = {hero}
        q = deque([hero])
        shape = chars.shape
        while q:
            r, c = q.popleft()
            for dr, dc in MOVE_DELTAS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < shape[0] and 0 <= nc < shape[1]):
                    continue
                if (nr, nc) in seen:
                    continue
                if not walkable[nr, nc]:
                    continue
                seen.add((nr, nc))
                # frontier: some neighbor unexplored
                frontier = False
                for dr2, dc2 in MOVE_DELTAS:
                    ur, uc = nr + dr2, nc + dc2
                    if 0 <= ur < shape[0] and 0 <= uc < shape[1]:
                        if (ur, uc) not in visited and walkable[ur, uc] is False and \
                           int(chars[ur, uc]) in (ord(" "), ord("#")):
                            frontier = True
                            break
                if frontier or (nr, nc) not in visited and (nr, nc) != hero:
                    if (nr, nc) not in visited:
                        return (nr, nc)
                q.append((nr, nc))
        return None

    def _next_step(self, hero, goal, walkable, chars):
        from corp.navigation.astar import GridAStar  # noqa: PLC0415
        path = GridAStar.find_path(hero, goal, walkable)
        if not path:
            return None
        node = path[0]
        dr = node.row - hero[0]
        dc = node.col - hero[1]
        return self._action_for((dr, dc), hero, chars)

    def _explore_fallback(self, hero, chars, visited):
        for (dr, dc) in MOVE_DELTAS:
            r, c = hero[0] + dr, hero[1] + dc
            if 0 <= r < chars.shape[0] and 0 <= c < chars.shape[1] and \
               int(chars[r, c]) in WALKABLE_ORDS:
                return self._action_for((dr, dc), hero, chars)
        return None

    def _action_for(self, delta: tuple[int, int], hero, chars):
        idx = self._action_idx.get(delta)
        if idx is None:
            return None
        r, c = hero[0] + delta[0], hero[1] + delta[1]
        if 0 <= r < chars.shape[0] and 0 <= c < chars.shape[1]:
            if int(chars[r, c]) == ord("#"):  # corridor wall → blocked (mazes have doors)
                return None
        return idx


# Walkable char set as {ord: True} for mask building
WALKABLE_ORDS = {ord(ch): True for ch in ".#+><}"}


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

DEFAULT_TASK = "MiniHack-ExploreMaze-Hard-v0"
CERT_TASK = "MiniHack-ExploreMaze-Easy-Mapped-v0"   # fully mapped → pure A* skill case


class MiniHackAdapter(DomainAdapter):
    spec = DomainSpec(
        name="minihack",
        obs_layout="NLE dict: glyphs/chars/blstats/message (identical interface to NetHack)",
        grid_shape=(21, 79),
        n_actions=12,
        seeded=True,
        step_limit_default=4000,
        notes="ExploreMaze family (plan's 'Explore-HardFixed/Maze-HardReach' do not "
              "exist in minihack 1.0.2); reward on reaching the staircase.",
    )

    def predicates(self) -> dict[str, PredicateBinding]:
        sigs = {
            "stairs_known": ([], "the staircase down has been observed"),
            "hp_frac": ([("op",), ("float", 0.0, 1.0)], "hp/max_hp compared to v"),
            "adjacent_hostiles": ([], "a hostile monster is adjacent"),
            "monster": ([("str",)], "observed adjacent monster letters contain substring"),
            "item": ([("str",)], "tile item name contains substring"),
            "depth_between": ([("int",), ("int",)], "floor depth in [lo, hi] (always 1)"),
            "turn_ge": ([("int",)], "episode step >= t"),
            "true": ([], "always true"),
            "false": ([], "always false"),
        }
        return {n: PredicateBinding(args=a, desc=d,
                                    fn=lambda *args, _n=n: True)  # bound at eval time
                for n, (a, d) in sigs.items()}

    def param_leaves(self) -> dict[str, ParamLeaf]:
        out = {}
        for section, obj in (("explore", ExplorePolicy()), ("nav", MhNavPolicy()),
                             ("combat", MhCombatPolicy())):
            for f in fields(obj):
                v = getattr(obj, f.name)
                path = f"{section}.{f.name}"
                if isinstance(v, bool):
                    mn, mx, typ = False, True, "bool"
                elif isinstance(v, float):
                    mn, mx, typ = v / 2.0, v * 2.0, "float"
                else:
                    mn, mx, typ = max(0, v // 2), v * 2, "int"
                out[path] = ParamLeaf(path=path, type=typ, default=v, min=mn, max=mx)
        return out

    def goal_handlers(self) -> dict[str, GoalHandler]:
        return {
            "explore_floor": GoalHandler(name="explore_floor", desc="frontier BFS exploration"),
            "reach_stairs": GoalHandler(name="reach_stairs",
                                        owned_params=["explore.stuck_patience"],
                                        desc="A* to the staircase down (descend analogue)"),
        }

    def verbs(self) -> dict[str, str]:
        return {"retreat": "step away from adjacent hostiles",
                "avoid": "route around without engaging"}

    def make_env(self, seed: int | None = None, task: str | None = None, **cfg: Any) -> Any:
        import gymnasium  # noqa: PLC0415
        import minihack  # noqa: F401,PLC0415
        return gymnasium.make(
            task or DEFAULT_TASK,
            observation_keys=["glyphs", "chars", "blstats", "message"], **cfg)

    def run_episode(self, env: Any, program: Any, seed: int | None,
                    max_steps: int) -> dict:
        agent = MiniHackAgent(env, program, max_steps=max_steps)
        return agent.run(seed=seed)

    def certification_suite(self) -> list[CertCase]:
        def stairs_reached(r: dict) -> bool:
            return bool(r.get("success"))

        def progress(r: dict) -> bool:
            return (r.get("success") and r.get("coverage", 0) >= 15) \
                or r.get("coverage", 0) >= 40

        return [
            CertCase(name="mapped_easy_astar", seed=7, max_steps=300,
                     check=stairs_reached,
                     desc="mapped Easy maze: pure A* must reach the stairs"),
            CertCase(name="unmapped_explore_progress", seed=11, max_steps=600,
                     check=progress,
                     desc="unmapped maze: frontier exploration makes coverage progress"),
        ]

    def rag_context(self, query: str = "", k: int = 4) -> list[str]:
        from corp.policy.corpus import rag_slices  # noqa: PLC0415
        return rag_slices("minihack", query=query or "ExploreMaze staircase navigation", k=k)

    def default_program(self) -> dict:
        import copy  # noqa: PLC0415
        return copy.deepcopy(MINIHACK_PROGRAM)


MINIHACK_PROGRAM: dict = {
    "version": 1,
    "domain": "minihack",
    "params": {},
    "macros": [],
    "tactic_rules": [],
    "nogoods": [],
    "role_profiles": {},
    "domain_profiles": {},
    "strategy_plan": [
        {"goal": "reach_stairs", "when": "(stairs_known)", "until": "(false)"},
        {"goal": "explore_floor", "when": "(true)", "until": "(false)"},
    ],
    "provenance": {"author": "r5-cold-start",
                   "note": "minihack cold-start: reach stairs when known, else explore"},
}