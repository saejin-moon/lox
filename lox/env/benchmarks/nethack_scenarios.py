"""
Parameterized Noisy MiniHack `.des` Scenarios & Real NLE `prompt_torture` Suites.
("Hard Code Nothing": zero NetHack strategy heuristics or hints in harness code.)

Provides:
- 7 procedural noisy mid/late-game scenario archetypes (locked vault doors,
  boulder choke, altar + clutter pile, trap corridor, fountain/sink room,
  multi-threat melee, dark labyrinth search).
- 4 real NLE UI `prompt_torture` trap suites (text call/naming prompt,
  Y/N confirmation, directional prompt, multi-item menu/pager).
- `NetHackScenarioHarness` (subclassing `NetHackHarness` so `GameState`,
  `WorldMemory`, prompt-linger, and `PolicyRunner` work 1-to-1).
- `NetHackScenariosAdapter` registered as `"nethack_scenarios"`.
- `run_scenario_gate(...)` for fast candidate validation (`<0.4s`).
"""

from __future__ import annotations

import random
from typing import Any

from minihack import LevelGenerator, MiniHackSkill

from lox.env.harness import NetHackHarness
from lox.env.nethack.adapter import NetHackAdapter
from lox.env.state import GameState

SCENARIO_ARCHETYPES: tuple[str, ...] = (
    "locked_vault_door",
    "boulder_choke",
    "altar_and_pile",
    "trap_corridor",
    "fountain_sink_room",
    "multi_threat_melee",
    "dark_labyrinth_search",
)

PROMPT_TORTURE_SUITES: tuple[str, ...] = (
    "prompt_torture_call_name",
    "prompt_torture_yn_confirm",
    "prompt_torture_direction_count",
    "prompt_torture_multipage_menu",
)

ALL_SCENARIOS: tuple[str, ...] = SCENARIO_ARCHETYPES + PROMPT_TORTURE_SUITES

_DISTRACTOR_OBJECTS: tuple[tuple[str, str], ...] = (
    ("apple", "%"),
    ("slime mold", "%"),
    ("arrow", ")"),
    ("dagger", ")"),
    ("rock", "*"),
    ("leather gloves", "["),
)


def _rand_free_pos(
    rng: random.Random,
    w: int,
    h: int,
    used: set[tuple[int, int]],
    *,
    x_min: int = 1,
    x_max: int | None = None,
    y_min: int = 1,
    y_max: int | None = None,
) -> tuple[int, int]:
    hi_x = (w - 2) if x_max is None else min(w - 2, x_max)
    hi_y = (h - 2) if y_max is None else min(h - 2, y_max)
    for _ in range(80):
        px = rng.randint(max(1, x_min), max(max(1, x_min), hi_x))
        py = rng.randint(max(1, y_min), max(max(1, y_min), hi_y))
        if (px, py) not in used:
            used.add((px, py))
            return (px, py)
    for px in range(1, w - 1):
        for py in range(1, h - 1):
            if (px, py) not in used:
                used.add((px, py))
                return (px, py)
    return (1, 1)


def _add_wall_column(
    lg: LevelGenerator, x: int, h: int, used: set[tuple[int, int]]
) -> None:
    for yy in range(h):
        lg.add_terrain((x, yy), "|")
        used.add((x, yy))


def build_scenario_des(scenario: str, seed: int = 42) -> tuple[str, bytes]:
    """
    Generates a parameterized noisy MiniHack `.des` level script and optional
    initial NLE C trigger bytes (non-empty only for `prompt_torture_*` traps).
    """
    rng = random.Random(int(seed))
    name = str(scenario or "").strip().lower()
    if name not in ALL_SCENARIOS:
        name = ALL_SCENARIOS[abs(int(seed)) % len(ALL_SCENARIOS)]

    w = rng.randint(14, 18)
    h = rng.randint(7, 10)
    lit = name != "dark_labyrinth_search"
    lg = LevelGenerator(w=w, h=h, lit=lit)
    used: set[tuple[int, int]] = set()
    mid_x = w // 2
    mid_y = h // 2
    pre_trigger = b""

    # Place downstair in right zone & reserve center-left spawn band
    used.add((2, mid_y))
    stair_pos = _rand_free_pos(rng, w, h, used, x_min=mid_x + 2)
    lg.add_stair_down(place=stair_pos)

    if name == "locked_vault_door":
        _add_wall_column(lg, mid_x, h, used)
        door_y = rng.randint(2, h - 3)
        lg.add_terrain((mid_x, door_y), ".")
        lg.add_door("locked" if rng.random() < 0.5 else "closed", place=(mid_x, door_y))
        for _ in range(3):
            obj_n, obj_sym = rng.choice(_DISTRACTOR_OBJECTS)
            lg.add_object(
                name=obj_n, symbol=obj_sym, place=_rand_free_pos(rng, w, h, used)
            )
        lg.add_monster(
            name=rng.choice(["newt", "kobold", "grid bug"]),
            place=_rand_free_pos(rng, w, h, used),
        )

    elif name == "boulder_choke":
        _add_wall_column(lg, mid_x, h, used)
        gap_y = rng.randint(2, h - 3)
        lg.add_terrain((mid_x, gap_y), ".")
        lg.add_boulder(place=(mid_x, gap_y))
        if h >= 8:
            alt_y = gap_y + 1 if gap_y + 1 < h - 1 else gap_y - 1
            lg.add_terrain((mid_x, alt_y), ".")
        for _ in range(3):
            obj_n, obj_sym = rng.choice(_DISTRACTOR_OBJECTS)
            lg.add_object(
                name=obj_n, symbol=obj_sym, place=_rand_free_pos(rng, w, h, used)
            )

    elif name == "altar_and_pile":
        altar_pos = _rand_free_pos(rng, w, h, used, x_max=mid_x + 1)
        lg.add_altar(place=altar_pos)
        for _ in range(rng.randint(5, 8)):
            obj_n, obj_sym = rng.choice(_DISTRACTOR_OBJECTS)
            lg.add_object(
                name=obj_n, symbol=obj_sym, place=_rand_free_pos(rng, w, h, used)
            )
        lg.add_monster(
            name=rng.choice(["jackal", "newt", "goblin"]),
            place=_rand_free_pos(rng, w, h, used),
        )

    elif name == "trap_corridor":
        for _ in range(rng.randint(4, 7)):
            trap_t = rng.choice(["arrow", "pit", "dart", "bear", "sleep gas"])
            lg.add_trap(name=trap_t, place=_rand_free_pos(rng, w, h, used))
        lg.add_monster(
            name=rng.choice(["grid bug", "newt", "gecko"]),
            place=_rand_free_pos(rng, w, h, used),
        )
        for _ in range(2):
            obj_n, obj_sym = rng.choice(_DISTRACTOR_OBJECTS)
            lg.add_object(
                name=obj_n, symbol=obj_sym, place=_rand_free_pos(rng, w, h, used)
            )

    elif name == "fountain_sink_room":
        lg.add_fountain(place=_rand_free_pos(rng, w, h, used, x_max=mid_x))
        lg.add_sink(place=_rand_free_pos(rng, w, h, used, x_min=mid_x))
        for _ in range(4):
            obj_n, obj_sym = rng.choice(_DISTRACTOR_OBJECTS)
            lg.add_object(
                name=obj_n, symbol=obj_sym, place=_rand_free_pos(rng, w, h, used)
            )
        lg.add_monster(
            name=rng.choice(["newt", "kobold", "sewer rat"]),
            place=_rand_free_pos(rng, w, h, used),
        )

    elif name == "multi_threat_melee":
        # Pillar cover + multiple fast/low enemies
        pillar_x = rng.randint(4, mid_x + 1)
        lg.add_terrain((pillar_x, mid_y), "|")
        used.add((pillar_x, mid_y))
        for m_name in rng.sample(
            ["jackal", "kobold", "goblin", "sewer rat", "newt"], k=3
        ):
            lg.add_monster(name=m_name, place=_rand_free_pos(rng, w, h, used, x_min=4))

    elif name == "dark_labyrinth_search":
        _add_wall_column(lg, mid_x - 1, h, used)
        dy = rng.randint(2, h - 3)
        lg.add_terrain((mid_x - 1, dy), ".")
        lg.add_door("closed", place=(mid_x - 1, dy))
        for _ in range(3):
            obj_n, obj_sym = rng.choice(_DISTRACTOR_OBJECTS)
            lg.add_object(
                name=obj_n, symbol=obj_sym, place=_rand_free_pos(rng, w, h, used)
            )

    elif name == "prompt_torture_call_name":
        lg.add_object(name="apple", symbol="%", place=(3, mid_y))
        # '#' opens NetHack extended command text prompt (`in_text_prompt=True`)
        pre_trigger = b"#"

    elif name == "prompt_torture_yn_confirm":
        lg.add_fountain(place=(3, mid_y))
        # 'p' (#pray) triggers real C `[yn]` confirmation prompt (`in_yn_prompt=True`)
        pre_trigger = b"p"

    elif name == "prompt_torture_direction_count":
        _add_wall_column(lg, 5, h, used)
        lg.add_terrain((5, mid_y), ".")
        lg.add_door("closed", place=(5, mid_y))
        # 'o' (open) prompts for direction (`In what direction?`)
        pre_trigger = b"o"

    elif name == "prompt_torture_multipage_menu":
        # Rich pile on floor near player triggers drop / pickup multi-item interaction
        for _ in range(6):
            obj_n, obj_sym = rng.choice(_DISTRACTOR_OBJECTS)
            lg.add_object(name=obj_n, symbol=obj_sym, place=(3, mid_y))
        # 'd' (drop) triggers genuine C inventory drop prompt / menu
        pre_trigger = b"d"

    return lg.get_des(), pre_trigger


class NetHackScenarioHarness(NetHackHarness):
    """
    Deterministic MiniHack `.des` scenario harness wrapping NLE C via `MiniHackSkill`
    while exposing 100% identical `NetHackHarness` step/GameState/WorldMemory semantics.
    """

    def __init__(
        self,
        scenario: str = "locked_vault_door",
        role: str = "val-hum-law-fem",
        max_steps: int = 200,
        allow_menu_linger: bool = True,
        **kwargs: Any,
    ) -> None:
        self.scenario = str(scenario or "locked_vault_door").strip().lower()
        self._mh_env: Any = None
        self._last_pre_trigger: bytes = b""
        super().__init__(
            role=role,
            max_steps=max_steps,
            allow_menu_linger=allow_menu_linger,
            **kwargs,
        )

    def reset(self, seed: int = 42, scenario: str | None = None) -> GameState:
        target_sc = str(scenario or self.scenario).strip().lower()
        self.scenario = target_sc
        des_str, pre_trigger = build_scenario_des(target_sc, seed=int(seed))
        self._last_pre_trigger = pre_trigger

        if self._mh_env is not None:
            try:
                self._mh_env.close()
            except Exception:
                pass
            self._mh_env = None
            self.nh = None

        mh = MiniHackSkill(
            des_file=des_str,
            character=self.role,
            savedir=None,
            max_episode_steps=max(50, int(self.max_steps)),
        )
        mh.reset(seed=int(seed))
        self._mh_env = mh

        # Bind directly onto the MiniHack-compiled Nethack C engine + full key table
        unwrapped = mh.unwrapped
        self.nh = unwrapped.nethack
        self._obs_keys = list(unwrapped._observation_keys)
        self._idx_map = {k: i for i, k in enumerate(self._obs_keys)}
        self._prog_state_idx = self._idx_map["program_state"]
        self._msg_idx = self._idx_map["message"]
        self._internal_idx = self._idx_map["internal"]

        self.current_seed = int(seed)
        self.action_bytes.clear()
        self.step_count = 0
        self.prompt_aborts = 0
        self.prompt_linger_steps = 0
        self.consecutive_zero_turn_steps = 0
        self._prev_turn = -1
        self.done = False
        self.mem.reset()

        obs_tuple = unwrapped.last_observation
        for tb in pre_trigger:
            if self.done:
                break
            self.action_bytes.append(tb)
            try:
                obs_tuple, self.done = self.nh.step(tb)
            except RuntimeError:
                self.done = True
                break

        self._last_raw_tuple = obs_tuple
        self.step_byte_offsets = [len(self.action_bytes)]
        return self.wrap_state(obs_tuple)

    def replay_bytes(
        self,
        seed: int,
        recorded_bytes: bytes | bytearray | None = None,
        stop_at_byte_idx: int | None = None,
    ) -> GameState:
        self.reset(seed=seed)
        buf = bytes(recorded_bytes or b"")
        pre_len = len(self._last_pre_trigger)
        if stop_at_byte_idx is not None:
            buf = buf[: max(0, int(stop_at_byte_idx))]
        for b in buf[pre_len:]:
            if self.done:
                break
            self.action_bytes.append(b)
            obs_tuple, self.done = self.nh.step(b)
            self.step_count += 1
            self.step_byte_offsets.append(len(self.action_bytes))
            self._last_raw_tuple = obs_tuple
            st_step = GameState(
                obs_tuple,
                step_count=self.step_count,
                mem=self.mem,
                obs_keys=self._obs_keys,
                obs_idx_map=self._idx_map,
            )
            self.mem.update(st_step)
        return self.wrap_state()

    def close(self) -> None:
        if self._mh_env is not None:
            try:
                self._mh_env.close()
            except Exception:
                pass
            self._mh_env = None
            self.nh = None
        super().close()


class NetHackScenariosAdapter(NetHackAdapter):
    """BenchmarkAdapter registered under `'nethack_scenarios'`."""

    @property
    def name(self) -> str:
        return "nethack_scenarios"

    def create_harness(self, **kwargs: Any) -> NetHackScenarioHarness:
        return NetHackScenarioHarness(**kwargs)


def run_scenario_gate(
    policy_code: str,
    skills: dict[str, str] | None = None,
    *,
    seed: int = 42,
    max_steps: int = 80,
    scenarios: list[str] | tuple[str, ...] | None = None,
    include_archetypes: bool = True,
    include_prompt_torture: bool = True,
) -> dict[str, Any]:
    """
    Compiles candidate `(policy_code, skills)` once and executes it across requested
    MiniHack scenario archetypes and/or NLE `prompt_torture` suites.
    Returns structured results (`passed_count`, `total_count`, `exceptions`,
    `prompt_livelocks`, `scenario_results`).
    """
    from lox.env.harness import policy_handles_menus
    from lox.lang.bt import extract_runner_leaf_errors
    from lox.lang.compiler import compile_policy_package

    compiled = compile_policy_package(policy_code, "", skills=dict(skills or {}))
    handles_menus = policy_handles_menus(policy_code, dict(skills or {}))

    if scenarios is not None:
        target_list = list(scenarios)
    else:
        target_list = []
        if include_archetypes:
            target_list.extend(SCENARIO_ARCHETYPES)
        if include_prompt_torture:
            target_list.extend(PROMPT_TORTURE_SUITES)

    results: list[dict[str, Any]] = []
    total_exc = 0
    total_livelocks = 0
    passed_cnt = 0

    for idx, sc_name in enumerate(target_list):
        sc_seed = int(seed) + idx * 17
        harness = NetHackScenarioHarness(
            scenario=sc_name,
            max_steps=max_steps,
            allow_menu_linger=handles_menus,
        )
        try:
            state = harness.reset(seed=sc_seed, scenario=sc_name)
            runner = compiled.create_runner()
            prompt_repeats = 0
            last_prompt = ""
            livelocked = False
            steps_done = 0
            last_end_reason = "running"
            for _ in range(max(10, int(max_steps))):
                raw_act, _fn = runner.next_action(state)
                state, done, info = harness.step(raw_act)
                steps_done += 1
                if info.get("end_reason"):
                    last_end_reason = str(info["end_reason"])
                elif done:
                    if info.get("is_ascended"):
                        last_end_reason = "ascended"
                    elif info.get("truncated"):
                        last_end_reason = "max_steps"
                    elif int(getattr(state, "hp", 1) or 0) <= 0:
                        msg_s = str(getattr(state, "message", "") or "").strip()
                        last_end_reason = msg_s.split(".")[0][:60] if msg_s else "died"
                    else:
                        last_end_reason = "ended"
                is_prompt = getattr(state, "in_yn_prompt", False) or getattr(
                    state, "in_text_prompt", False
                )
                msg_l = (state.message or "").lower()
                if is_prompt:
                    if msg_l == last_prompt:
                        prompt_repeats += 1
                        if prompt_repeats >= 6:
                            livelocked = True
                    else:
                        prompt_repeats = 1
                        last_prompt = msg_l
                else:
                    prompt_repeats = 0
                    last_prompt = ""
                if done:
                    break
            if last_end_reason == "running" and steps_done >= max(10, int(max_steps)):
                last_end_reason = "max_steps"

            leaf_errs = extract_runner_leaf_errors(runner)
            exc_cnt = int(getattr(runner, "policy_exceptions", 0) or 0) + len(leaf_errs)
            resolved_prompt = (not livelocked) and (
                not sc_name.startswith("prompt_torture_")
                or (
                    int(harness.prompt_aborts) == 0
                    and not (
                        getattr(state, "in_yn_prompt", False)
                        or getattr(state, "in_text_prompt", False)
                    )
                )
            )
            ok = exc_cnt == 0 and resolved_prompt
            if ok:
                passed_cnt += 1
            total_exc += exc_cnt
            if livelocked:
                total_livelocks += 1
            results.append(
                {
                    "scenario": sc_name,
                    "seed": sc_seed,
                    "passed": ok,
                    "steps": steps_done,
                    "depth": int(state.depth),
                    "exceptions": exc_cnt,
                    "livelocked": livelocked,
                    "prompt_aborts": int(harness.prompt_aborts),
                    "end_reason": last_end_reason,
                }
            )
        finally:
            harness.close()

    total_n = len(target_list)
    return {
        "passed_count": passed_cnt,
        "total_count": total_n,
        "pass_rate": round(passed_cnt / max(1, total_n), 4),
        "exceptions": total_exc,
        "prompt_livelocks": total_livelocks,
        "scenario_results": results,
    }


__all__ = [
    "ALL_SCENARIOS",
    "PROMPT_TORTURE_SUITES",
    "SCENARIO_ARCHETYPES",
    "NetHackScenarioHarness",
    "NetHackScenariosAdapter",
    "build_scenario_des",
    "run_scenario_gate",
]
