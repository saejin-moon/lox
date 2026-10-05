"""
Unit tests for LOX Ascension Milestones & Advanced Primitives:
1. Altar Sacrificing (sacrifice_on_altar)
2. Temple Priest Intrinsic AC Protection Donation (donate_to_priest)
3. Blindfold / Towel Telepathy Scouting (apply_blindfold)
4. Semi-Permanent Athame & Burned Wand Engraving
5. Sokoban Branch Boulder Solver (step_solve_sokoban)
6. Castle Drawbridge Breaching (breach_drawbridge)
7. Gehennom Straight-Line Tunneling (dig_tunnel)
8. Endgame Invocation Ritual State Machine (InvocationSolver & perform_invocation_step)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lox.core.types import (
    Action,
    DungeonView,
    HeroState,
    InventoryView,
    Item,
    Observation,
)
from lox.envs.nethack import NetHackAdapter
from lox.envs.solvers.invocation_solver import InvocationSolver


class DummyEnv:
    """Mock gymnasium environment for testing NetHackAdapter action sequences."""

    def __init__(self):
        self.step_history: list[int] = []
        self.last_action: int | None = None

    def step(self, action: int):
        self.step_history.append(action)
        self.last_action = action
        raw_obs = {
            "blstats": np.array(
                [
                    10,
                    10,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    20,
                    20,
                    1,
                    1000,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ),
            "chars": np.full((21, 79), ord("."), dtype=np.uint8),
            "glyphs": np.zeros((21, 79), dtype=np.int32),
            "message": np.zeros(256, dtype=np.uint8),
            "inv_letters": np.zeros(55, dtype=np.uint8),
            "inv_strs": np.zeros((55, 80), dtype=np.uint8),
            "inv_oclasses": np.zeros(55, dtype=np.uint8),
            "misc": np.zeros(3, dtype=np.int32),
        }
        return raw_obs, 0.0, False, False, {}

    def reset(self, seed=None):
        raw_obs = {
            "blstats": np.array(
                [
                    10,
                    10,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    20,
                    20,
                    1,
                    1000,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            ),
            "chars": np.full((21, 79), ord("."), dtype=np.uint8),
            "glyphs": np.zeros((21, 79), dtype=np.int32),
            "message": np.zeros(256, dtype=np.uint8),
            "inv_letters": np.zeros(55, dtype=np.uint8),
            "inv_strs": np.zeros((55, 80), dtype=np.uint8),
            "inv_oclasses": np.zeros(55, dtype=np.uint8),
            "misc": np.zeros(3, dtype=np.int32),
        }
        return raw_obs, {}

    def close(self):
        pass


def make_obs(
    hero: HeroState | None = None,
    inventory: InventoryView | None = None,
    dungeon: DungeonView | None = None,
    chars: np.ndarray | None = None,
    glyphs: np.ndarray | None = None,
    message: str = "",
    raw_obs: Any = None,
) -> Observation:
    if chars is None:
        chars = np.full((21, 79), ord("."), dtype=np.uint8)
    if glyphs is None:
        glyphs = np.zeros((21, 79), dtype=np.int32)
    if hero is None:
        hero = HeroState(y=10, x=10)
    if inventory is None:
        inventory = InventoryView([])
    if dungeon is None:
        dungeon = DungeonView()
    return Observation(
        chars=chars,
        glyphs=glyphs,
        hero=hero,
        inventory=inventory,
        dungeon=dungeon,
        message=message,
        raw_obs=raw_obs or {"chars": chars, "glyphs": glyphs},
    )


def test_sacrifice_on_altar_with_floor_corpse():
    """Verify sacrifice_on_altar executes #offer and confirms floor corpse."""
    adapter = NetHackAdapter()
    adapter.env = DummyEnv()
    hero = HeroState(y=10, x=10, level=1)
    inv = InventoryView([])
    dungeon = DungeonView(standing_on_altar=True)
    obs = make_obs(hero=hero, inventory=inv, dungeon=dungeon)
    adapter._last_obs = obs

    # Place a corpse on the altar tile
    adapter.floor_corpses[(10, 10)] = ("goblin corpse", 10, False)

    act = Action(name="sacrifice_on_altar")
    new_obs, reward, term, trunc, info = adapter.step(act)

    executed_chars = [
        chr(
            adapter.actions[idx].value
            if hasattr(adapter.actions[idx], "value")
            else adapter.actions[idx]
        )
        for idx in adapter.env.step_history
        if idx in adapter.char_to_act.values()
    ]
    seq_str = "".join(executed_chars)
    assert "#offer" in seq_str
    # Floor corpse on altar tile should be removed
    assert (10, 10) not in adapter.floor_corpses


def test_sacrifice_on_altar_with_inventory_corpse():
    """Verify sacrifice_on_altar selects inventory corpse slot when no floor corpse exists."""
    adapter = NetHackAdapter()
    adapter.env = DummyEnv()
    hero = HeroState(y=10, x=10, level=1)
    corpse_item = Item(slot="c", name="fresh cave spider corpse", category="food")
    inv = InventoryView([corpse_item])
    dungeon = DungeonView(standing_on_altar=True)
    obs = make_obs(hero=hero, inventory=inv, dungeon=dungeon)
    adapter._last_obs = obs

    act = Action(name="sacrifice_on_altar")
    new_obs, reward, term, trunc, info = adapter.step(act)

    executed_chars = [
        chr(
            adapter.actions[idx].value
            if hasattr(adapter.actions[idx], "value")
            else adapter.actions[idx]
        )
        for idx in adapter.env.step_history
        if idx in adapter.char_to_act.values()
    ]
    seq_str = "".join(executed_chars)
    assert "#offer" in seq_str
    assert "c" in seq_str  # Corpse slot selected


def test_donate_to_priest():
    """Verify donate_to_priest chats with priest and enters 400 * XL gold."""
    adapter = NetHackAdapter()
    adapter.env = DummyEnv()
    hero = HeroState(y=10, x=10, level=3, gold=2000)
    inv = InventoryView([])
    dungeon = DungeonView(
        in_temple=True, has_priest=True, adjacent_priest=True, priest_pos=(10, 11)
    )
    obs = make_obs(hero=hero, inventory=inv, dungeon=dungeon)
    adapter._last_obs = obs
    adapter.known_priest_pos = (10, 11)  # East of hero ('l')

    act = Action(name="donate_to_priest")
    new_obs, reward, term, trunc, info = adapter.step(act)

    executed_chars = [
        chr(
            adapter.actions[idx].value
            if hasattr(adapter.actions[idx], "value")
            else adapter.actions[idx]
        )
        for idx in adapter.env.step_history
        if idx in adapter.char_to_act.values()
    ]
    seq_str = "".join(executed_chars)
    assert "#chat" in seq_str
    assert "l" in seq_str  # Direction toward priest (East)
    assert "1200" in seq_str  # 400 * Level 3 = 1200 gold donation


def test_apply_blindfold():
    """Verify apply_blindfold executes 'a' + blindfold slot."""
    adapter = NetHackAdapter()
    adapter.env = DummyEnv()
    blindfold_item = Item(slot="b", name="a blindfold", category="tool")
    inv = InventoryView([blindfold_item])
    obs = make_obs(inventory=inv)
    adapter._last_obs = obs

    act = Action(name="apply_blindfold")
    new_obs, reward, term, trunc, info = adapter.step(act)

    executed_chars = [
        chr(
            adapter.actions[idx].value
            if hasattr(adapter.actions[idx], "value")
            else adapter.actions[idx]
        )
        for idx in adapter.env.step_history
        if idx in adapter.char_to_act.values()
    ]
    seq_str = "".join(executed_chars)
    assert "ab" in seq_str


def test_semi_permanent_engraving_with_burn_wand():
    """Verify engrave_dust_elbereth uses wand of fire/lightning/digging for permanent burned Elbereth."""
    adapter = NetHackAdapter()
    adapter.env = DummyEnv()
    hero = HeroState(y=5, x=5)
    fire_wand = Item(slot="w", name="a wand of fire (0:3)", category="wand")
    inv = InventoryView([fire_wand])
    obs = make_obs(hero=hero, inventory=inv)
    adapter._last_obs = obs

    act = Action(name="engrave_dust_elbereth")
    new_obs, reward, term, trunc, info = adapter.step(act)

    executed_chars = [
        chr(
            adapter.actions[idx].value
            if hasattr(adapter.actions[idx], "value")
            else adapter.actions[idx]
        )
        for idx in adapter.env.step_history
        if idx in adapter.char_to_act.values()
    ]
    seq_str = "".join(executed_chars)
    # Must use wand slot 'w' instead of fingers '-'
    assert "EwElbereth" in seq_str


def test_semi_permanent_engraving_with_athame():
    """Verify engrave_dust_elbereth uses athame slot for permanent carved Elbereth when no burn wand."""
    adapter = NetHackAdapter()
    adapter.env = DummyEnv()
    hero = HeroState(y=5, x=5)
    athame = Item(slot="a", name="a blessed silver athame", category="weapon")
    inv = InventoryView([athame])
    obs = make_obs(hero=hero, inventory=inv)
    adapter._last_obs = obs

    act = Action(name="engrave_dust_elbereth")
    new_obs, reward, term, trunc, info = adapter.step(act)

    executed_chars = [
        chr(
            adapter.actions[idx].value
            if hasattr(adapter.actions[idx], "value")
            else adapter.actions[idx]
        )
        for idx in adapter.env.step_history
        if idx in adapter.char_to_act.values()
    ]
    seq_str = "".join(executed_chars)
    assert "EaElbereth" in seq_str


def test_sokoban_boulder_solver_step():
    """Verify step_solve_sokoban advances hero to push boulder."""
    adapter = NetHackAdapter()
    adapter.env = DummyEnv()
    hero = HeroState(y=10, x=10)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    chars[10, 11] = ord("0")  # Boulder East of hero
    chars[10, 12] = ord("^")  # Pit East of boulder
    inv = InventoryView([])
    dungeon = DungeonView(is_sokoban=True, has_boulders=True)
    obs = make_obs(hero=hero, inventory=inv, dungeon=dungeon, chars=chars)
    adapter._last_obs = obs

    act = Action(name="step_solve_sokoban")
    new_obs, reward, term, trunc, info = adapter.step(act)

    # Hero is adjacent to boulder at (10, 11) and behind it relative to pit at (10, 12);
    # pushing east means stepping East ('l')
    assert adapter.env.last_action == adapter.char_to_act.get("l")


def test_castle_drawbridge_breach_step():
    """Verify breach_drawbridge steps to safe distance and fires wand of striking."""
    adapter = NetHackAdapter()
    adapter.env = DummyEnv()
    # Hero is at distance 2 orthogonal to closed drawbridge: hero (10, 8), drawbridge (10, 10)
    hero = HeroState(y=10, x=8)
    striking_wand = Item(slot="z", name="a wand of striking (0:4)", category="wand")
    inv = InventoryView([striking_wand])
    dungeon = DungeonView(drawbridge_in_fov=True, closest_drawbridge_pos=(10, 10))
    obs = make_obs(hero=hero, inventory=inv, dungeon=dungeon)
    adapter._last_obs = obs

    act = Action(name="breach_drawbridge")
    new_obs, reward, term, trunc, info = adapter.step(act)

    executed_chars = [
        chr(
            adapter.actions[idx].value
            if hasattr(adapter.actions[idx], "value")
            else adapter.actions[idx]
        )
        for idx in adapter.env.step_history
        if idx in adapter.char_to_act.values()
    ]
    seq_str = "".join(executed_chars)
    # Zaps wand of striking directly at drawbridge (East 'l')
    assert "zzl" in seq_str or "zl" in seq_str


def test_dig_tunnel():
    """Verify dig_tunnel zaps wand of digging cardinal vector through stone."""
    adapter = NetHackAdapter()
    adapter.env = DummyEnv()
    hero = HeroState(y=10, x=10)
    dig_wand = Item(slot="d", name="a wand of digging (0:5)", category="wand")
    inv = InventoryView([dig_wand])
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    chars[11, 10] = ord("-")  # Stone wall South of hero
    raw_obs = {"chars": chars, "glyphs": np.zeros((21, 79), dtype=np.int32)}
    obs = make_obs(hero=hero, inventory=inv, chars=chars, raw_obs=raw_obs)
    adapter._last_obs = obs
    adapter.known_stairs_down = (15, 10)  # Downstairs is south

    act = Action(name="dig_tunnel")
    new_obs, reward, term, trunc, info = adapter.step(act)

    executed_chars = [
        chr(
            adapter.actions[idx].value
            if hasattr(adapter.actions[idx], "value")
            else adapter.actions[idx]
        )
        for idx in adapter.env.step_history
        if idx in adapter.char_to_act.values()
    ]
    seq_str = "".join(executed_chars)
    # Zaps wand of digging south ('j')
    assert "zdj" in seq_str


def test_invocation_solver_full_sequence():
    """Verify InvocationSolver executes the 3 mandatory ritual steps at Vibrating Square."""
    solver = InvocationSolver()
    candelabrum = Item(
        slot="c", name="the Candelabrum of Abernathy (7 candles)", category="tool"
    )
    bell = Item(slot="b", name="the Bell of Opening", category="tool")
    book = Item(slot="B", name="the Book of the Dead", category="scroll")
    inv = InventoryView([candelabrum, bell, book])

    hero = HeroState(y=12, x=14)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    chars[12, 14] = ord("~")  # Standing on Vibrating Square

    obs = make_obs(
        hero=hero,
        inventory=inv,
        chars=chars,
        message="You feel a strange vibration beneath your feet.",
    )

    # Step 1: Light Candelabrum
    act1 = solver.plan_step(obs)
    assert act1 is not None
    assert act1.slot == "c"

    # Simulate message that candelabrum glows
    obs_glow = make_obs(
        hero=hero,
        inventory=inv,
        chars=chars,
        message="The candelabrum burns with a pure flame.",
    )
    # Step 2: Ring Bell of Opening
    act2 = solver.plan_step(obs_glow)
    assert act2 is not None
    assert act2.slot == "b"

    # Simulate message that bell rings
    obs_bell = make_obs(
        hero=hero,
        inventory=inv,
        chars=chars,
        message="The bell rings with a resonant chime.",
    )
    # Step 3: Read Book of the Dead
    act3 = solver.plan_step(obs_bell)
    assert act3 is not None
    assert act3.slot == "B"

    # Simulate message that stairway opens
    obs_open = make_obs(
        hero=hero,
        inventory=inv,
        chars=chars,
        message="A stairway opens into Moloch's Sanctum!",
    )
    act4 = solver.plan_step(obs_open)
    assert act4 is not None
    assert act4.name == "descend"
