import numpy as np
import pytest
from lox.envs.nethack import NetHackAdapter
from lox.core.types import Action


def test_tactical_primitives_execution():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=101)

    initial_turn = obs.hero.turn
    assert initial_turn >= 1

    # 1. Test engrave_elbereth sequence (advances turn)
    obs, reward, term, trunc, _ = adapter.step(Action(name="engrave_elbereth"))
    assert obs.hero.turn > initial_turn

    # 2. Test kick closed door (in east direction)
    obs, reward, term, trunc, _ = adapter.step(Action(name="kick_closed_door", direction=(0, 1)))
    assert obs.hero.hp > 0

    # 3. Test open door (in east direction)
    obs, reward, term, trunc, _ = adapter.step(Action(name="open_door", direction=(0, 1)))
    assert obs.hero.turn >= initial_turn

    # 4. Test eat floor corpse (safe fallback when empty)
    obs, reward, term, trunc, _ = adapter.step(Action(name="eat_floor_corpse"))
    assert obs is not None

    # 5. Test step_to coordinate navigation
    obs, reward, term, trunc, _ = adapter.step(Action(name="step_to", target_pos=(obs.hero.y, obs.hero.x + 1)))
    assert obs is not None

    # 6. Test peaceful positions and blocked tiles memory tracking
    adapter.peaceful_positions.add((obs.hero.y, obs.hero.x + 1))
    adapter.blocked_tiles.add((obs.hero.y, obs.hero.x - 1))
    obs_test = adapter._extract_obs(adapter._last_raw_obs)
    assert obs_test.combat.adjacent_peaceful is True
    assert (obs.hero.y, obs.hero.x - 1) in adapter.blocked_tiles

    adapter.close()


def test_wear_armor_failed_slot_tracking():
    from lox.core.types import Item, InventoryView

    item1 = Item(slot="b", name="scale mail", category="armor", is_equipped=True)
    item2 = Item(slot="f", name="ring mail", category="armor", is_equipped=False)
    item3 = Item(slot="g", name="helmet", category="armor", is_equipped=False)

    inv = InventoryView([item1, item2, item3])
    assert inv.has_unworn_armor is True
    assert inv.get_unworn_armor_slot() == "f"

    # Simulate slot "f" failing to be worn (e.g. redundant body armor)
    inv.failed_armor_slots.add("f")
    assert inv.has_unworn_armor is True
    assert inv.get_unworn_armor_slot() == "g"  # Falls back to helmet!

    # Simulate slot "g" failing too
    inv.failed_armor_slots.add("g")
    assert inv.has_unworn_armor is False
    assert inv.get_unworn_armor_slot() is None


def test_throw_dagger_and_elbereth_tactics():
    from lox.core.types import Item, InventoryView

    # 1. Test InventoryView daggers queries
    sword = Item(slot="a", name="long sword", category="weapon", is_equipped=True)
    dagger = Item(slot="c", name="+0 dagger", category="weapon", is_equipped=False)
    inv = InventoryView([sword, dagger])
    assert inv.has_daggers is True
    assert inv.get_dagger_slot() == "c"

    # 2. Test NetHackAdapter throw_dagger and engrave_dust_elbereth actions
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=105)

    # Test engrave_dust_elbereth
    t_start = obs.hero.turn
    obs, reward, term, trunc, _ = adapter.step(Action(name="engrave_dust_elbereth"))
    assert obs.hero.turn >= t_start
    assert (obs.hero.y, obs.hero.x) in adapter.elbereth_positions or obs.combat.standing_on_elbereth

    # Test throw_dagger execution
    obs, reward, term, trunc, _ = adapter.step(Action(name="throw_dagger", target_pos=(obs.hero.y, obs.hero.x + 2)))
    assert obs is not None
    assert obs.hero.hp > 0

    adapter.close()


def test_peaceful_prompt_recording_and_zero_turn_shield():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=106)

    # Set up previous attempted direction
    adapter._last_attempted_dir = (0, 1)
    adapter._prev_hero_pos = (obs.hero.y, obs.hero.x)

    # Simulate peaceful encounter prompt in _dismiss_more
    raw_obs_mock = {"message": list(b"Really attack the hobbit? [yn] (n)")}
    adapter._dismiss_more(raw_obs_mock, False, False)

    # Verify peaceful position was immediately recorded
    expected_pos = (obs.hero.y, obs.hero.x + 1)
    assert expected_pos in adapter.peaceful_positions

    adapter.close()


def test_fountain_fov_and_step_to_fountain():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=107)

    # Artificially inject a fountain in FOV on chars grid
    obs.chars[obs.hero.y, obs.hero.x + 2] = ord("{")
    raw_obs_mock = obs.raw_obs.copy()
    raw_obs_mock["chars"] = obs.chars
    obs_fountain = adapter._extract_obs(raw_obs_mock)

    assert obs_fountain.dungeon.fountain_in_fov is True
    assert obs_fountain.dungeon.closest_fountain_pos == (obs.hero.y, obs.hero.x + 2)

    # Test step_to_fountain execution
    obs_after, _, _, _, _ = adapter.step(Action(name="step_to_fountain"))
    assert obs_after is not None

    adapter.close()


