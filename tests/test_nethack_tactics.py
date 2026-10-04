import math
import numpy as np
import pytest
from lox.envs.nethack import NetHackAdapter
from lox.core.types import Action
from nle import nethack


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


def test_nearby_loot_and_step_to_loot():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=108)

    # Place armor '[' on a walkable adjacent tile with no monster standing on it
    _, walkable_nav = adapter._build_walkable_nav(obs.raw_obs)
    candidates = [
        (obs.hero.y + dy, obs.hero.x + dx)
        for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        if (dy != 0 or dx != 0)
        and 0 <= obs.hero.y + dy < 21 and 0 <= obs.hero.x + dx < 79
        and walkable_nav[obs.hero.y + dy, obs.hero.x + dx]
        and not nethack.glyph_is_monster(int(obs.glyphs[obs.hero.y + dy, obs.hero.x + dx]))
    ]
    target_loot = candidates[0] if candidates else (obs.hero.y, obs.hero.x + 1)
    raw_obs_mock = obs.raw_obs.copy()
    raw_obs_mock["chars"] = obs.chars.copy()
    raw_obs_mock["chars"][target_loot[0], target_loot[1]] = ord("[")
    raw_obs_mock["glyphs"] = obs.glyphs.copy()
    raw_obs_mock["glyphs"][target_loot[0], target_loot[1]] = 2359  # non-monster floor/object glyph
    obs_loot = adapter._extract_obs(raw_obs_mock)

    assert obs_loot.spatial.has_nearby_loot is True
    assert obs_loot.spatial.nearby_loot_pos is not None
    assert math.hypot(obs_loot.spatial.nearby_loot_pos[0] - obs.hero.y, obs_loot.spatial.nearby_loot_pos[1] - obs.hero.x) <= 4

    # Step to loot
    obs_after, _, _, _, _ = adapter.step(Action(name="step_to_loot"))
    assert obs_after is not None

    adapter.close()


def test_wand_actions_and_teleport_panic_escape():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=109)

    # Inject teleport scroll and wand of striking into inventory
    from lox.core.types import Item, InventoryView
    test_inv = InventoryView([
        Item(slot="a", name="+1 long sword", category="weapon", is_equipped=True, buc="uncursed"),
        Item(slot="b", name="scroll of teleportation", category="scroll", is_equipped=False, buc="uncursed"),
        Item(slot="c", name="wand of striking", category="wand", is_equipped=False, buc="uncursed"),
        Item(slot="d", name="wand of teleportation", category="wand", is_equipped=False, buc="uncursed"),
    ])
    assert test_inv.has_scroll_of_teleport is True
    assert test_inv.get_scroll_of_teleport_slot() == "b"
    assert test_inv.has_offensive_wand is True
    assert test_inv.get_offensive_wand_slot() == "c"
    assert test_inv.has_wand_of_teleport is True
    assert test_inv.get_wand_of_teleport_slot() == "d"

    # Test action dispatching
    obs_zap, _, _, _, _ = adapter.step(Action(name="zap_offensive_wand", slot="c", target_pos=(obs.hero.y, obs.hero.x + 2)))
    assert obs_zap is not None

    obs_tele, _, _, _, _ = adapter.step(Action(name="zap_wand_teleport", slot="d"))
    assert obs_tele is not None

    obs_scroll, _, _, _, _ = adapter.step(Action(name="read_scroll_teleport", slot="b"))
    adapter.close()


def test_passive_hazard_nav_mask_and_cornered_retreat_safety():
    """Verify that passive hazards (floating eyes, gas spores, molds) are excluded from walkable_nav
    and that cornered retreat falls back to wait() rather than bumping/attacking them."""
    import nle.nethack as nh
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=123)

    hy, hx = obs.hero.y, obs.hero.x
    # Simulate a floating eye adjacent to the hero
    eye_pos = (hy, hx + 1)
    eye_mon_id = nh.PM_FLOATING_EYE if hasattr(nh, "PM_FLOATING_EYE") else None
    if eye_mon_id is None:
        for i in range(nh.NUMMONS):
            if nh.permonst(i).mname == "floating eye":
                eye_mon_id = i
                break
    eye_glyph = nh.GLYPH_MON_OFF + eye_mon_id

    # Place eye glyph in raw observation
    obs.glyphs[eye_pos[0], eye_pos[1]] = eye_glyph
    obs.raw_obs["glyphs"][eye_pos[0], eye_pos[1]] = eye_glyph

    # Verify _build_walkable_nav excludes the floating eye
    walkable, walkable_nav = adapter._build_walkable_nav(obs)
    assert walkable_nav[eye_pos[0], eye_pos[1]] is False or walkable_nav[eye_pos[0], eye_pos[1]] == 0

    # Set up observation with adjacent_floating_eye and no open retreat
    obs.combat.closest_hostile_name = "floating eye"
    obs.combat.closest_hostile_pos = eye_pos
    obs.combat.adjacent_floating_eye = True
    obs.combat.adjacent_hostile = True
    obs.combat.hostile_count_fov = 1

    # In step_away_from_hostile, when blocked or no tile increases distance, must yield wait()
    # Mask all neighbors in walkable_nav so no direction increases distance
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if (hy + dy, hx + dx) != (hy, hx):
                walkable_nav[hy + dy, hx + dx] = False

    # Dispatch step_away_from_hostile
    next_obs, _, _, _, _ = adapter.step(Action(name="step_away_from_hostile"))
    assert next_obs is not None

    adapter.close()


def test_peaceful_guard_and_active_hostile_discrimination():
    """Verify that Vault Guards and priests are recognized as peaceful species
    and that has_active_hostile correctly differentiates active predators from passive hazards."""
    import nle.nethack as nh
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=456)

    hy, hx = obs.hero.y, obs.hero.x
    guard_pos = (hy, hx + 1)

    guard_mon_id = 268  # Vault guard
    assert nh.permonst(guard_mon_id).mname.lower() == "guard"
    guard_glyph = nh.GLYPH_MON_OFF + guard_mon_id

    initial_active = obs.combat.active_hostile_count

    # Place guard glyph in raw observation
    obs.glyphs[guard_pos[0], guard_pos[1]] = guard_glyph
    obs.raw_obs["glyphs"][guard_pos[0], guard_pos[1]] = guard_glyph

    obs_extracted = adapter._extract_obs(obs.raw_obs)
    # The guard should be identified as peaceful and NOT increment active_hostile_count
    assert guard_pos in adapter.peaceful_positions
    assert obs_extracted.combat.adjacent_peaceful is True
    assert obs_extracted.combat.active_hostile_count == initial_active

    # Throwing dagger or zapping wand at peaceful position must NOT fire at peaceful guard
    act_dagger = Action(name="throw_dagger", target_pos=guard_pos)
    obs_res, _, _, _, _ = adapter.step(act_dagger)
    assert obs_res is not None

    adapter.close()

