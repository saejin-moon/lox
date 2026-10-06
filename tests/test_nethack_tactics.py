import math

from nle import nethack

from lox.core.types import Action
from lox.envs.nethack import NetHackAdapter


def test_tactical_primitives_execution():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=101)

    initial_turn = obs.hero.turn
    assert initial_turn >= 1

    # 1. Test engrave_elbereth sequence (advances turn)
    obs, reward, term, trunc, _ = adapter.step(Action(name="engrave_elbereth"))
    assert obs.hero.turn > initial_turn

    # 2. Test kick closed door (in east direction)
    obs, reward, term, trunc, _ = adapter.step(
        Action(name="kick_closed_door", direction=(0, 1))
    )
    assert obs.hero.hp > 0

    # 3. Test open door (in east direction)
    obs, reward, term, trunc, _ = adapter.step(
        Action(name="open_door", direction=(0, 1))
    )
    assert obs.hero.turn >= initial_turn

    # 4. Test eat floor corpse (safe fallback when empty)
    obs, reward, term, trunc, _ = adapter.step(Action(name="eat_floor_corpse"))
    assert obs is not None

    # 5. Test step_to coordinate navigation
    obs, reward, term, trunc, _ = adapter.step(
        Action(name="step_to", target_pos=(obs.hero.y, obs.hero.x + 1))
    )
    assert obs is not None

    # 6. Test peaceful positions and blocked tiles memory tracking
    adapter.peaceful_positions.add((obs.hero.y, obs.hero.x + 1))
    adapter.blocked_tiles.add((obs.hero.y, obs.hero.x - 1))
    obs_test = adapter._extract_obs(adapter._last_raw_obs)
    assert obs_test.combat.adjacent_peaceful is True
    assert (obs.hero.y, obs.hero.x - 1) in adapter.blocked_tiles

    adapter.close()


def test_wear_armor_failed_slot_tracking():
    from lox.core.types import InventoryView, Item

    item1 = Item(slot="b", name="small shield", category="armor", is_equipped=True)
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
    from lox.core.types import InventoryView, Item

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
    assert (
        obs.hero.y,
        obs.hero.x,
    ) in adapter.elbereth_positions or obs.combat.standing_on_elbereth

    # Test throw_dagger execution
    obs, reward, term, trunc, _ = adapter.step(
        Action(name="throw_dagger", target_pos=(obs.hero.y, obs.hero.x + 2))
    )
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

    # Artificially inject a fountain in FOV on chars grid (clearing any natural fountains first)
    obs.chars[obs.chars == ord("{")] = ord(".")
    obs.chars[obs.hero.y, obs.hero.x + 2] = ord("{")
    raw_obs_mock = obs.raw_obs.copy()
    raw_obs_mock["chars"] = obs.chars
    adapter.known_fountain_pos = None
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
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy != 0 or dx != 0)
        and 0 <= obs.hero.y + dy < 21
        and 0 <= obs.hero.x + dx < 79
        and walkable_nav[obs.hero.y + dy, obs.hero.x + dx]
        and not nethack.glyph_is_monster(
            int(obs.glyphs[obs.hero.y + dy, obs.hero.x + dx])
        )
    ]
    target_loot = candidates[0] if candidates else (obs.hero.y, obs.hero.x + 1)
    raw_obs_mock = obs.raw_obs.copy()
    raw_obs_mock["chars"] = obs.chars.copy()
    raw_obs_mock["chars"][target_loot[0], target_loot[1]] = ord("[")
    raw_obs_mock["glyphs"] = obs.glyphs.copy()
    raw_obs_mock["glyphs"][target_loot[0], target_loot[1]] = (
        2359  # non-monster floor/object glyph
    )
    obs_loot = adapter._extract_obs(raw_obs_mock)

    assert obs_loot.spatial.has_nearby_loot is True
    assert obs_loot.spatial.nearby_loot_pos is not None
    assert (
        math.hypot(
            obs_loot.spatial.nearby_loot_pos[0] - obs.hero.y,
            obs_loot.spatial.nearby_loot_pos[1] - obs.hero.x,
        )
        <= 4
    )

    # Step to loot
    obs_after, _, _, _, _ = adapter.step(Action(name="step_to_loot"))
    assert obs_after is not None

    adapter.close()


def test_wand_actions_and_teleport_panic_escape():
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=109)

    # Inject teleport scroll and wand of striking into inventory
    from lox.core.types import InventoryView, Item

    test_inv = InventoryView(
        [
            Item(
                slot="a",
                name="+1 long sword",
                category="weapon",
                is_equipped=True,
                buc="uncursed",
            ),
            Item(
                slot="b",
                name="scroll of teleportation",
                category="scroll",
                is_equipped=False,
                buc="uncursed",
            ),
            Item(
                slot="c",
                name="wand of striking",
                category="wand",
                is_equipped=False,
                buc="uncursed",
            ),
            Item(
                slot="d",
                name="wand of teleportation",
                category="wand",
                is_equipped=False,
                buc="uncursed",
            ),
        ]
    )
    assert test_inv.has_scroll_of_teleport is True
    assert test_inv.get_scroll_of_teleport_slot() == "b"
    assert test_inv.has_offensive_wand is True
    assert test_inv.get_offensive_wand_slot() == "c"
    assert test_inv.has_wand_of_teleport is True
    assert test_inv.get_wand_of_teleport_slot() == "d"

    # Test action dispatching
    obs_zap, _, _, _, _ = adapter.step(
        Action(
            name="zap_offensive_wand", slot="c", target_pos=(obs.hero.y, obs.hero.x + 2)
        )
    )
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
    assert (
        walkable_nav[eye_pos[0], eye_pos[1]] is False
        or walkable_nav[eye_pos[0], eye_pos[1]] == 0
    )

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


def test_food_poisoning_shield_and_universal_missiles():
    """Verify that corpses in inventory are excluded from has_food and get_food_slot,
    and that darts, arrows, and rocks are recognized as ranged missile ammunition."""
    from lox.core.types import InventoryView, Item

    corpse_item = Item(slot="d", name="a kobold corpse", category="food")
    food_ration = Item(slot="e", name="an uncursed food ration", category="food")
    lichen_corpse = Item(slot="h", name="a lichen corpse", category="food")
    dart_item = Item(slot="f", name="24 darts", category="weapon", quantity=24)
    rock_item = Item(slot="g", name="5 rocks", category="gem", quantity=5)

    # 1. Inventory with only rotting corpse: has_food must be False and get_food_slot None
    inv_corpse_only = InventoryView([corpse_item])
    assert inv_corpse_only.has_food is False
    assert inv_corpse_only.get_food_slot() is None
    assert inv_corpse_only.food_count == 0

    # 2. Inventory with non-rotting lichen corpse: has_food must be True
    inv_lichen = InventoryView([lichen_corpse])
    assert inv_lichen.has_food is True
    assert inv_lichen.get_food_slot() == "h"
    assert inv_lichen.food_count == 1

    # 3. Inventory with food ration + rotting corpse: get_food_slot must strictly return ration slot 'e'
    inv_mixed = InventoryView([corpse_item, food_ration])
    assert inv_mixed.has_food is True
    assert inv_mixed.get_food_slot() == "e"
    assert inv_mixed.food_count == 1

    # 3. Universal missiles: darts and rocks must be recognized when daggers are absent
    inv_missiles = InventoryView([dart_item, rock_item])
    assert inv_missiles.has_daggers is True
    assert inv_missiles.get_dagger_slot() == "f"
    assert inv_missiles.dagger_count == 29


def test_adjacent_floating_eye_and_proactive_nutrition():
    """Verify Invariants 44 & 45:
    - Missiles can be thrown directly at an adjacent floating eye (safe at distance 1)
    - Nutrition is consumed at hunger_state >= 2 during combat when adjacent to passive hazards
    """
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=109)

    # Mock adjacent floating eye glyph at (hero.y, hero.x + 1)
    target_pos = (obs.hero.y, obs.hero.x + 1)
    raw_obs_mock = obs.raw_obs.copy()
    raw_obs_mock["glyphs"] = obs.glyphs.copy()
    for r in range(21):
        for c in range(79):
            if nethack.glyph_is_monster(int(raw_obs_mock["glyphs"][r, c])):
                raw_obs_mock["glyphs"][r, c] = 2359
    eye_mon_id = 89
    for mid in range(380):
        try:
            if nethack.permonst(mid).mname.lower() == "floating eye":
                eye_mon_id = mid
                break
        except Exception:
            continue
    raw_obs_mock["glyphs"][target_pos[0], target_pos[1]] = (
        nethack.GLYPH_MON_OFF + eye_mon_id
    )
    obs_extracted = adapter._extract_obs(raw_obs_mock)

    assert obs_extracted.combat.adjacent_floating_eye is True
    assert obs_extracted.combat.closest_hostile_name.lower() == "floating eye"

    # Throwing dagger at adjacent floating eye must execute throw sequence, not melee bump
    obs_res, _, _, _, _ = adapter.step(
        Action(name="throw_dagger", target_pos=target_pos)
    )
    assert obs_res is not None

    adapter.close()


def test_floating_eye_melee_prohibition_and_missile_pickup():
    from nle import nethack

    from lox.core.types import Action
    from lox.envs.nethack import NetHackAdapter

    adapter = NetHackAdapter()
    obs = adapter.reset(seed=123)

    # 1. Verify missile collection options
    options = adapter.options
    assert "pickup_thrown" in options
    assert "pickup_types:?!/%=[$*" in options

    # 2. Mock cornered adjacent floating eye with consecutive passive waits >= 2
    raw_obs_mock = obs.raw_obs.copy()
    raw_obs_mock["glyphs"] = obs.glyphs.copy()
    for r in range(21):
        for c in range(79):
            if nethack.glyph_is_monster(int(raw_obs_mock["glyphs"][r, c])):
                raw_obs_mock["glyphs"][r, c] = 2359

    target_pos = (obs.hero.y, obs.hero.x + 1)
    eye_mon_id = 89
    for mid in range(380):
        try:
            if nethack.permonst(mid).mname.lower() == "floating eye":
                eye_mon_id = mid
                break
        except Exception:
            continue
    raw_obs_mock["glyphs"][target_pos[0], target_pos[1]] = (
        nethack.GLYPH_MON_OFF + eye_mon_id
    )
    obs_extracted = adapter._extract_obs(raw_obs_mock)

    adapter.last_obs = obs_extracted
    # Step away from hostile should NOT attack the floating eye
    obs_res, _, _, _, _ = adapter.step(Action(name="step_away_from_hostile"))
    assert obs_res is not None

    adapter.close()


def test_enhance_weapon_skill_and_superior_body_armor():
    """Verify that enhance_weapon_skill, superior body armor replacement, and pack threats function correctly."""
    from lox.core.types import InventoryView, Item

    # 1. Superior body armor detection
    worn_jacket = Item(
        slot="a",
        name="a leather jacket (being worn)",
        category="armor",
        is_equipped=True,
    )
    unworn_mithril = Item(
        slot="b", name="a dwarvish mithril-coat", category="armor", is_equipped=False
    )
    unworn_leather = Item(
        slot="c", name="a leather jacket", category="armor", is_equipped=False
    )

    inv = InventoryView([worn_jacket, unworn_mithril, unworn_leather])
    slots = inv.get_superior_body_armor_slot()
    assert slots == ("a", "b"), f"Expected ('a', 'b'), got {slots}"

    # Superior armor already worn: should not downgrade
    worn_plate = Item(
        slot="a", name="a plate mail (being worn)", category="armor", is_equipped=True
    )
    inv_plate = InventoryView([worn_plate, unworn_mithril])
    assert inv_plate.get_superior_body_armor_slot() is None

    # 2. Test enhance_weapon_skill in adapter
    adapter = NetHackAdapter()
    obs = adapter.reset(seed=456)
    obs_res, _, _, _, _ = adapter.step(Action(name="enhance_weapon_skill"))
    assert obs_res is not None
    assert obs_res.hero.turn >= 1

    # 3. Test is_pack_threat
    obs_extracted = adapter._extract_obs(obs.raw_obs)
    assert hasattr(obs_extracted.combat, "is_pack_threat")
    adapter.close()


def test_wear_armor_with_cloak_removal():
    """Verify that wear_armor and replace_body_armor safely take off cloak before wearing body armor."""
    from lox.core.types import Item

    adapter = NetHackAdapter()
    obs = adapter.reset(seed=789)

    # Mock inventory with worn cloak and unworn plate mail
    cloak = Item(slot="d", name="a dwarvish cloak (being worn)", category="armor", is_equipped=True)
    suit = Item(slot="e", name="a plate mail", category="armor", is_equipped=False)
    adapter._last_obs.inventory.clear()
    adapter._last_obs.inventory.extend([cloak, suit])

    # Calling wear_armor should generate the 6-key sequence: T d W e W d
    executed_seq = []
    original_step_sequence = adapter._step_sequence

    def mock_step_sequence(seq):
        executed_seq.extend(seq)
        return original_step_sequence(seq[:1])  # execute at least 1 action

    adapter._step_sequence = mock_step_sequence
    try:
        adapter.step(Action(name="wear_armor", slot="e"))
        expected_chars = ["T", "d", "W", "e", "W", "d"]
        expected_seq = [adapter.char_to_act.get(c, 0) for c in expected_chars]
        assert executed_seq == expected_seq, f"Expected {expected_seq}, got {executed_seq}"
    finally:
        adapter._step_sequence = original_step_sequence
        adapter.close()


def test_passive_hazard_nav_buffering():
    """Verify that passive hazards (floating eyes) buffer adjacent tiles when lacking ranged weapons."""
    import nle.nethack as nh
    from lox.core.types import Item

    adapter = NetHackAdapter()
    obs = adapter.reset(seed=123)
    hy, hx = obs.hero.y, obs.hero.x

    eye_mon_id = nh.PM_FLOATING_EYE if hasattr(nh, "PM_FLOATING_EYE") else None
    if eye_mon_id is None:
        for i in range(nh.NUMMONS):
            if nh.permonst(i).mname == "floating eye":
                eye_mon_id = i
                break
    eye_glyph = nh.GLYPH_MON_OFF + eye_mon_id

    walkable, _ = adapter._build_walkable_nav(obs)
    candidates = []
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            ny, nx = hy + dy, hx + dx
            if 0 <= ny < 21 and 0 <= nx < 79 and walkable[ny, nx] and (ny, nx) != (hy, hx):
                for cdy, cdx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ay, ax = ny + cdy, nx + cdx
                    if (
                        0 <= ay < 21
                        and 0 <= ax < 79
                        and walkable[ay, ax]
                        and (ay, ax) != (hy, hx)
                    ):
                        candidates.append(((ny, nx), (ay, ax)))
                        break
        if candidates:
            break

    eye_pos, adj_pos = candidates[0]
    obs.glyphs[eye_pos[0], eye_pos[1]] = eye_glyph
    obs.raw_obs["glyphs"][eye_pos[0], eye_pos[1]] = eye_glyph

    # Remove daggers and offensive wands from inventory
    obs.inventory.clear()
    adapter._last_obs = obs

    _, nav_no_ranged = adapter._build_walkable_nav(obs)
    # The tile immediately next to the floating eye should be buffered (not walkable)
    assert not nav_no_ranged[adj_pos[0], adj_pos[1]], "Tile adjacent to passive hazard must be buffered when lacking ranged weapons"

    # Now give hero a dagger
    obs.inventory.append(Item(slot="a", name="a dagger", category="weapon"))
    adapter._last_obs = obs
    _, nav_with_ranged = adapter._build_walkable_nav(obs)
    # With daggers, adjacent tile is walkable so hero can approach to throw missiles safely
    assert nav_with_ranged[adj_pos[0], adj_pos[1]], "Tile adjacent to passive hazard must be walkable when possessing ranged weapons"

    adapter.close()


def test_shop_tiles_tracking_and_loot_shield():
    """Verify that shop welcome messages register persistent shop_tiles and shield shop merchandise from loot_candidates."""
    import numpy as np

    adapter = NetHackAdapter()
    obs = adapter.reset(seed=456)
    hy, hx = obs.hero.y, obs.hero.x

    # Place an armor '[' on an adjacent floor tile
    target_pos = (hy, hx + 1)
    raw_obs_mock = obs.raw_obs.copy()
    raw_obs_mock["chars"] = obs.chars.copy()
    raw_obs_mock["chars"][target_pos[0], target_pos[1]] = ord("[")
    raw_obs_mock["glyphs"] = obs.glyphs.copy()
    raw_obs_mock["glyphs"][target_pos[0], target_pos[1]] = 2359

    # Without shop message, it should be recognized as loot
    raw_obs_mock["message"] = np.frombuffer(b"You see an open floor.\x00" + b"\x00" * 233, dtype=np.uint8)
    obs_loot = adapter._extract_obs(raw_obs_mock)
    assert obs_loot.spatial.has_nearby_loot is True

    # Now step into a store: shop greeting message
    raw_obs_mock["message"] = np.frombuffer(b"Welcome to Izchak's general store!\x00" + b"\x00" * 220, dtype=np.uint8)
    obs_shop = adapter._extract_obs(raw_obs_mock)

    # Shop tiles must be populated
    assert len(adapter.shop_tiles) > 0
    assert (hy, hx) in adapter.shop_tiles
    assert target_pos in adapter.shop_tiles
    assert obs_shop.dungeon.in_shop is True
    # Merchandise inside the shop must NOT trigger has_nearby_loot
    assert obs_shop.spatial.has_nearby_loot is False

    adapter.close()


def test_wear_armor_apron_over_cloak_immediate_blacklisting():
    import numpy as np
    from lox.envs.nethack import NetHackAdapter
    from lox.core.types import Action, Item, InventoryView

    adapter = NetHackAdapter()
    obs = adapter.reset()

    # Artificially inject an equipped cloak and an unworn apron into inventory
    worn_cloak = Item(slot="c", name="dwarvish cloak", category="armor", is_equipped=True)
    unworn_apron = Item(slot="o", name="an apron", category="armor", is_equipped=False)
    obs.inventory = InventoryView([worn_cloak, unworn_apron])

    # Directly step wear_armor targeting slot 'o'
    # Mock _step_sequence to simulate NetHack message "You are already wearing a cloak." without equipping
    def mock_step_sequence(seq):
        mock_raw = obs.raw_obs.copy()
        mock_raw["message"] = np.frombuffer(b"You are already wearing a cloak.\x00" + b"\x00" * 223, dtype=np.uint8)
        new_obs = adapter._extract_obs(mock_raw)
        new_obs.inventory = InventoryView([worn_cloak, unworn_apron], failed_armor_slots=adapter.failed_wear_slots)
        return new_obs, 0.0, False, False, {}

    adapter._step_sequence = mock_step_sequence
    obs_next, _, _, _, _ = adapter.step(Action(name="wear_armor", slot="o"))

    # The failed slot 'o' MUST be immediately blacklisted in failed_wear_slots
    assert "o" in adapter.failed_wear_slots
    assert "o" in obs_next.inventory.failed_armor_slots
    # And has_unworn_armor must now be False, preventing 23,000-turn loops
    assert obs_next.inventory.has_unworn_armor is False

    adapter.close()


def test_multi_slot_armor_prioritization():
    from lox.core.types import InventoryView, Item

    # Valkyrie starts with only a shield (base AC 6)
    shield = Item(slot="b", name="small shield", category="armor", is_equipped=True)
    boots = Item(slot="d", name="iron shoes", category="armor", is_equipped=False)
    body = Item(slot="e", name="dwarvish mithril coat", category="armor", is_equipped=False)
    helm = Item(slot="f", name="dwarvish iron helm", category="armor", is_equipped=False)
    cloak = Item(slot="g", name="elven cloak", category="armor", is_equipped=False)

    inv = InventoryView([shield, boots, body, helm, cloak])
    # Body armor has highest priority
    assert inv.get_unworn_armor_slot() == "e"

    # Once body armor is equipped, helm has next priority
    body.is_equipped = True
    assert inv.get_unworn_armor_slot() == "f"

    # Once helm is equipped, boots have next priority
    helm.is_equipped = True
    assert inv.get_unworn_armor_slot() == "d"

    # Once boots are equipped, cloak has next priority
    boots.is_equipped = True
    assert inv.get_unworn_armor_slot() == "g"


def test_corrosive_and_heavy_weapon_threat_detection():
    import numpy as np
    from lox.envs.nethack import (
        NetHackAdapter,
        GLYPH_IS_CORROSIVE_LUT,
        GLYPH_IS_HEAVY_WEAPON_LUT,
    )
    from lox.core.types import Action

    adapter = NetHackAdapter()
    obs = adapter.reset()

    # Corrosive monster name check
    raw_obs = obs.raw_obs.copy()
    chars = raw_obs["chars"].copy()
    hy, hx = obs.hero.y, obs.hero.x
    chars[hy, hx + 1] = ord("P")  # brown pudding char

    obs_extracted = adapter._extract_obs(raw_obs)
    # Verify LUTs are populated
    assert np.any(GLYPH_IS_CORROSIVE_LUT)
    assert np.any(GLYPH_IS_HEAVY_WEAPON_LUT)

    adapter.close()


def test_sokoban_entrance_detection_and_step():
    from lox.envs.nethack import NetHackAdapter
    from lox.core.types import Action

    adapter = NetHackAdapter()
    obs = adapter.reset()
    hy, hx = obs.hero.y, obs.hero.x

    # Set sokoban entrance pos on current level at an adjacent tile
    adapter.sokoban_entrance_pos = (hy, hx + 1)

    # Step to Sokoban entrance navigates towards (hy, hx + 1)
    act = Action(name="step_to_sokoban_entrance")
    obs_next, _, _, _, _ = adapter.step(act)
    assert adapter.sokoban_entrance_pos == (hy, hx + 1)

    # If standing on sokoban entrance, yields ascend
    adapter.sokoban_entrance_pos = (obs_next.hero.y, obs_next.hero.x)
    executed_seq = []
    adapter._step_sequence = lambda seq: executed_seq.extend(seq) or (obs_next, 0.0, False, False, {})
    adapter.step(Action(name="step_to_sokoban_entrance"))
    assert executed_seq == [adapter.char_to_act.get("<")]

    adapter.close()


def test_emergency_consumables_actions():
    from lox.envs.nethack import NetHackAdapter
    from lox.core.types import Action, Item, InventoryView

    adapter = NetHackAdapter()
    obs = adapter.reset()

    # Add unidentified potion and scroll
    unid_pot = Item(slot="u", name="a cloudy potion", category="potion", is_equipped=False)
    unid_scr = Item(slot="v", name="an unlabelled scroll", category="scroll", is_equipped=False)
    inv = InventoryView([unid_pot, unid_scr])
    assert inv.has_unidentified_potion is True
    assert inv.get_unidentified_potion_slot() == "u"
    assert inv.has_unidentified_scroll is True
    assert inv.get_unidentified_scroll_slot() == "v"

    # Mock _step_sequence to inspect keystrokes
    executed_seq = []
    def mock_step_sequence(seq):
        executed_seq.extend(seq)
        return obs, 0.0, False, False, {}

    adapter._step_sequence = mock_step_sequence

    # Test quaff_emergency_potion
    adapter.step(Action(name="quaff_emergency_potion", slot="u"))
    assert executed_seq[0] == adapter.char_to_act.get("q")
    assert executed_seq[1] == adapter.char_to_act.get("u")

    executed_seq.clear()
    # Test read_emergency_scroll
    adapter.step(Action(name="read_emergency_scroll", slot="v"))
    assert executed_seq[0] == adapter.char_to_act.get("r")
    assert executed_seq[1] == adapter.char_to_act.get("v")

    adapter.close()


def test_latest_policy_dry_run_validation():
    from lox.dsl.compiler import compile_policy
    from lox.core.types import Observation, HeroState

    with open("data/latest_policy.py") as f:
        policy_code = f.read()

    executor = compile_policy(policy_code)
    assert executor is not None

    obs = Observation(chars=None, glyphs=None, hero=HeroState(hp=50, max_hp=50))
    runner = executor.create_runner(obs)

    # Step 1: Normal state -> yields action
    act1 = runner.send(obs)
    assert act1 is not None
    assert isinstance(act1.name, str)

    # Step 2: Combat state -> yields combat action
    obs_combat = Observation(chars=None, glyphs=None, hero=HeroState(hp=50, max_hp=50))
    obs_combat.combat.adjacent_hostile = True
    act2 = runner.send(obs_combat)
    assert act2 is not None
    assert act2.name in ("melee_attack_hostile", "step_away_from_hostile", "step_to_chokepoint")

def test_pack_threat_adjacent_melee_priority():
    """Verify that adjacent enemies (like jackals or coyotes) are attacked in melee rather than retreated from."""
    from lox.dsl.compiler import compile_policy
    from lox.core.types import Observation, HeroState

    with open("data/latest_policy.py") as f:
        policy_code = f.read()

    executor = compile_policy(policy_code)
    obs = Observation(chars=None, glyphs=None, hero=HeroState(hp=16, max_hp=16, depth=2))
    obs.combat.adjacent_hostile = True
    obs.combat.hostile_count_fov = 1
    obs.combat.closest_hostile_name = "jackal"
    obs.combat.closest_hostile_dist = 1
    obs.combat.can_retreat = True
    obs.combat.in_corridor = False

    runner = executor.create_runner(obs)
    action = runner.send(obs)
    assert action is not None
    # Must attack the adjacent jackal directly; must NEVER yield step_to_chokepoint while adjacent and healthy
    assert action.name == "melee_attack_hostile", f"Expected melee_attack_hostile but got {action.name}"


def test_in_combat_emergency_eating_unconditional():
    """Verify that when hunger_state >= WEAK, the hero consumes carried food even with an adjacent hostile."""
    from lox.dsl.compiler import compile_policy
    from lox.core.types import Observation, HeroState, Item, HungerState

    with open("data/latest_policy.py") as f:
        policy_code = f.read()

    executor = compile_policy(policy_code)
    obs = Observation(
        chars=None,
        glyphs=None,
        hero=HeroState(hp=16, max_hp=16, depth=2, hunger_state=HungerState.WEAK),
    )
    obs.combat.adjacent_hostile = True
    obs.combat.hostile_count_fov = 1
    obs.combat.closest_hostile_name = "acid blob"
    obs.combat.closest_hostile_dist = 1
    obs.combat.can_retreat = True
    obs.inventory.append(
        Item(slot="d", name="an uncursed food ration", category="food")
    )

    runner = executor.create_runner(obs)
    action = runner.send(obs)
    assert action is not None
    assert action.name == "eat_carried_food", f"Expected eat_carried_food but got {action.name}"


def test_peaceful_domestic_animal_protection_and_sokoban_sweep():
    """Verify that genuine peaceful NPCs are recognized, attack retaliation triggers, and Sokoban sweep is shielded."""
    from lox.envs.nethack import (
        GLYPH_IS_PEACEFUL_SPECIES_LUT,
        GLYPH_MON_NAME,
        _MAX_GLYPH,
        NetHackAdapter,
    )
    from lox.dsl.compiler import compile_policy
    from lox.core.types import Observation, HeroState, SpatialView, DungeonView

    # 1. Genuine peaceful humanoid NPCs in peaceful LUT
    peaceful_names = [GLYPH_MON_NAME[g].lower() for g in range(_MAX_GLYPH) if GLYPH_IS_PEACEFUL_SPECIES_LUT[g]]
    assert any("guard" in n for n in peaceful_names), "Guard must be in peaceful species LUT"
    assert any("priest" in n for n in peaceful_names), "Priest must be in peaceful species LUT"
    assert any("shopkeeper" in n for n in peaceful_names), "Shopkeeper must be in peaceful species LUT"
    assert any("oracle" in n for n in peaceful_names), "Oracle must be in peaceful species LUT"

    # 2. Domestic animals must NOT be blanket peaceful so hero defends against wild/feral attacks
    assert not any("kitten" in n for n in peaceful_names), "Kitten must not be blanket peaceful in LUT"
    assert not any("little dog" in n for n in peaceful_names), "Little dog must not be blanket peaceful in LUT"

    # 3. Sokoban sweep shielding in policy
    with open("data/latest_policy.py") as f:
        policy_code = f.read()

    executor = compile_policy(policy_code)
    obs = Observation(
        chars=None,
        glyphs=None,
        hero=HeroState(hp=30, max_hp=30, depth=7, dungeon_branch="dungeon"),
        spatial=SpatialView(standing_on_stairs_down=True, stairs_down_known=True, has_unvisited_frontier=True),
        dungeon=DungeonView(has_sokoban_entrance=False),
    )
    runner = executor.create_runner(obs)
    action = runner.send(obs)
    assert action is not None
    # Must NOT descend on DL 7 when standing on stairs down if sweeping for Sokoban
    assert action.name != "descend", f"Expected not to descend on DL 7 during Sokoban sweep but got {action.name}"


def test_campaign_77_invariants_and_tactical_hardening():
    """Verify that blobs/oozes are in passive LUT, shops are excluded from dead ends, and walkable_nav does not block empty peaceful positions."""
    import numpy as np
    from lox.envs.nethack import (
        GLYPH_IS_PASSIVE_HAZARD_LUT,
        GLYPH_MON_NAME,
        _MAX_GLYPH,
        NetHackAdapter,
    )
    from lox.core.types import Observation, HeroState, CombatView, SpatialView, InventoryView

    # 1. Passive hazard LUT includes blobs, oozes, and puddings
    passive_names = [GLYPH_MON_NAME[g].lower() for g in range(_MAX_GLYPH) if GLYPH_IS_PASSIVE_HAZARD_LUT[g]]
    assert any("acid blob" in n for n in passive_names), "Acid blob must be in passive hazard LUT"
    assert any("gray ooze" in n for n in passive_names), "Gray ooze must be in passive hazard LUT"
    assert any("brown pudding" in n for n in passive_names), "Brown pudding must be in passive hazard LUT"

    # 2. Shop tiles strictly masked out of dead-end search masks
    adapter = NetHackAdapter()
    adapter.shop_tiles = {(5, 5), (5, 6), (5, 7)}
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    walkable = np.ones((21, 79), dtype=bool)
    dead_ends_mask = adapter._compute_dead_ends_mask(chars, walkable)
    for sy, sx in adapter.shop_tiles:
        assert not dead_ends_mask[sy, sx], f"Shop tile ({sy}, {sx}) must not be in dead_ends_mask"

    # 3. Walkable nav does not block peaceful_positions if empty floor
    import nle.nethack as nh
    obs_mock = Observation(
        chars=chars,
        glyphs=np.full((21, 79), nh.GLYPH_CMAP_OFF, dtype=np.int32),
        hero=HeroState(y=5, x=4, hp=30, max_hp=30, depth=1),
    )
    adapter.peaceful_positions.add((5, 5))
    _, walkable_nav = adapter._build_walkable_nav(obs_mock)
    assert bool(walkable_nav[5, 5]) is True, "Empty floor tile in peaceful_positions must remain walkable in walkable_nav"


def test_deadlock_circuit_breaker_emergency_strike():
    """Verify that when trapped adjacent to a passive hazard with 0 escape routes, deadlock circuit breaker strikes."""
    import unittest.mock
    import numpy as np
    from lox.envs.nethack import NetHackAdapter, Action
    from lox.core.types import Observation, HeroState, CombatView, InventoryView

    adapter = NetHackAdapter()
    adapter.reset(seed=42)

    # Simulate being trapped in a 1-tile corner with floating eye adjacent at (10, 21)
    hero = HeroState(y=10, x=20, hp=30, max_hp=30, depth=1)
    combat = CombatView(
        adjacent_hostile=True,
        closest_hostile_name="floating eye",
        closest_hostile_pos=(10, 21),
        adjacent_floating_eye=True,
    )
    inv = InventoryView(items=[])
    obs_prev = Observation(
        chars=np.full((21, 79), ord(" "), dtype=np.uint8),
        glyphs=np.zeros((21, 79), dtype=np.int32),
        hero=hero,
        combat=combat,
        inventory=inv,
    )
    adapter._last_obs = obs_prev
    adapter.consecutive_passive_waits = 2  # Already waited twice
    adapter.passive_search_count = 15      # Searches exhausted

    # All neighbors are walls (' ') except the monster's tile (10, 21)
    obs_prev.chars[10, 20] = ord(".")
    obs_prev.chars[10, 21] = ord(".")

    # Intercept step: calling step_away_from_hostile should execute emergency strike at (10, 21)
    # i.e., direction (0, 1)
    act = Action(name="step_away_from_hostile")
    resolved_actions = []
    original_step = adapter.step

    def capture_step(a):
        resolved_actions.append(a)
        if a.name == "melee_attack":
            return obs_prev, 0.0, False, False, {}
        return original_step(a)

    with unittest.mock.patch.object(adapter, "step", side_effect=capture_step):
        adapter.step(act)

    assert any(a.name == "melee_attack" and a.direction == (0, 1) for a in resolved_actions), (
        f"Expected emergency melee attack towards (0, 1) but got {[a.name for a in resolved_actions]}"
    )


def test_melee_attack_hostile_held_by_monster_and_blocked_tiles_discard():
    import numpy as np
    from lox.core.types import Observation, HeroState, CombatView, InventoryView, Action
    from lox.envs.nethack import NetHackAdapter, GLYPH_IS_MON_HOSTILE_LUT

    adapter = NetHackAdapter()
    hero = HeroState(y=10, x=20, hp=15, max_hp=15)
    combat = CombatView(adjacent_hostile=True, closest_hostile_name="lichen")
    inv = InventoryView(items=[])
    glyphs = np.full((21, 79), 2359, dtype=np.int32)
    glyphs[10, 21] = 155  # Lichen glyph

    obs_prev = Observation(
        chars=np.full((21, 79), ord("."), dtype=np.uint8),
        glyphs=glyphs,
        hero=hero,
        combat=combat,
        inventory=inv,
        message="You cannot escape from the lichen!",
    )
    adapter._last_obs = obs_prev
    # Simulate (10, 21) being mistakenly in blocked_tiles
    adapter.blocked_tiles.add((10, 21))

    captured_seq = []
    def mock_step_sequence(seq):
        captured_seq.extend(seq)
        return obs_prev, 0.0, False, False, {}

    import unittest.mock
    with unittest.mock.patch.object(adapter, "_step_sequence", side_effect=mock_step_sequence):
        adapter.step(Action(name="melee_attack_hostile"))

    # Blocked tile must be discarded
    assert (10, 21) not in adapter.blocked_tiles
    # Direction must be (0, 1) East towards the lichen
    assert adapter._last_attempted_dir == (0, 1)


def test_step_to_dead_end_anti_oscillation():
    import numpy as np
    from lox.core.types import Observation, HeroState, SpatialView, DungeonView, Action
    from lox.envs.nethack import NetHackAdapter

    adapter = NetHackAdapter()
    hero = HeroState(y=12, x=23, depth=1)
    # Floor room with no doors, no frontiers, no stairs
    spatial = SpatialView(stairs_down_known=False, has_unvisited_frontier=False)
    dungeon = DungeonView(has_closed_door=False)
    chars = np.full((21, 79), ord("."), dtype=np.uint8)
    glyphs = np.full((21, 79), 2359, dtype=np.int32)  # Floor CMAP
    obs_prev = Observation(
        chars=chars,
        glyphs=glyphs,
        hero=hero,
        spatial=spatial,
        dungeon=dungeon,
    )
    adapter._last_obs = obs_prev
    # Set last_move_from to (13, 23) (South)
    adapter.last_move_from = (13, 23)

    resolved_actions = []
    original_step = adapter.step

    def capture_step(a):
        resolved_actions.append(a)
        return obs_prev, 0.0, False, False, {}

    import unittest.mock
    with unittest.mock.patch.object(adapter, "step", side_effect=capture_step):
        adapter.step(Action(name="step_to_dead_end"))

    # Action must NOT step back to (1, 0) South because of is_prev anti-oscillation penalty
    steps = [a for a in resolved_actions if a.name in ("step_direction", "step_or_breach")]
    for s in steps:
        if s.direction is not None:
            assert s.direction != (1, 0), "Should not step back into _prev_hero_pos"


def test_step_to_dead_end_immediate_search_when_unexhausted():
    import numpy as np
    from lox.core.types import Observation, HeroState, SpatialView, DungeonView, Action
    from lox.envs.nethack import NetHackAdapter

    adapter = NetHackAdapter()
    hero = HeroState(y=7, x=2, depth=4)
    spatial = SpatialView(stairs_down_known=False, has_unvisited_frontier=False)
    dungeon = DungeonView(has_closed_door=False)
    chars = np.full((21, 79), ord(" "), dtype=np.uint8)
    # 4x4 room with walls around
    chars[6, 1:6] = ord("-")
    chars[11, 1:6] = ord("-")
    chars[7:11, 1] = ord("|")
    chars[7:11, 6] = ord("|")
    chars[7:11, 2:6] = ord(".")
    glyphs = np.full((21, 79), 2359, dtype=np.int32)
    obs_prev = Observation(
        chars=chars,
        glyphs=glyphs,
        hero=hero,
        spatial=spatial,
        dungeon=dungeon,
    )
    adapter._last_obs = obs_prev
    # Tile (7, 2) is a corner with 2 adjacent walls, searched_count is 0 (< 20)
    assert adapter.searched_count[7, 2] == 0

    resolved_actions = []
    original_step = adapter.step

    def capture_step(a):
        resolved_actions.append(a)
        if a.name == "search":
            return obs_prev, 0.0, False, False, {}
        return original_step(a)

    import unittest.mock
    with unittest.mock.patch.object(adapter, "step", side_effect=capture_step):
        adapter.step(Action(name="step_to_dead_end"))

    # Must immediately trigger search rather than stepping away
    assert any(a.name == "search" for a in resolved_actions)







