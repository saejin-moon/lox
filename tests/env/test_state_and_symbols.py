"""Domain-scoped tests for LOX."""

from __future__ import annotations

import numpy as np

from lox.datasets.symbols import SYMBOL_REGISTRY, NetHackSymbolRegistry
from lox.env.actions import (
    Action,
)
from lox.env.harness import NetHackHarness
from lox.env.state import FloorMemory, WorldMemory
from lox.lang.compiler import compile_policy_package


def test_gamestate_queries_and_pathfinding():
    harness = NetHackHarness(max_steps=200)
    try:
        state = harness.reset(seed=42)
        assert state.depth == 1
        assert state.deepest_level_reached == 1
        assert state.internal_hunger >= 0
        assert state.is_in_moveloop is True
        assert state.hp > 0
        assert state.max_hp >= state.hp
        assert 0 <= state.y < 21
        assert 0 <= state.x < 79
        assert state.char_at(state.y, state.x) != ""
        assert len(state.ascii_map().splitlines()) == 21
        assert len(state.tty_lines) == 24
        assert len(state.tty_text.splitlines()) == 24
        assert len(state.description_at(state.y, state.x)) > 0
        assert len(state.terrain_at(state.y, state.x)) > 0

        # Inventory should be populated from initial reset with typed properties
        inv = state.inventory()
        assert len(inv) > 0
        assert any(it.letter == "a" for it in inv)
        # Verify item.char is not '?' and typed properties work
        assert any(it.is_weapon and it.char == ")" for it in inv)
        assert any(it.is_armor and it.char == "[" for it in inv)
        assert any(it.is_food and it.char == "%" for it in inv)

        # Monster and object queries
        mons = state.visible_monsters()
        objs = state.visible_objects()
        assert isinstance(mons, list)
        assert isinstance(objs, list)
        if len(mons) > 0:
            m = mons[0]
            assert hasattr(m, "is_pet")
            assert hasattr(m, "name")
            assert len(m.name) > 0

        # BFS distances and A* path step
        passable = (
            (state.chars == ord("."))
            | (state.chars == ord("#"))
            | (state.chars == ord(">"))
            | (state.chars == ord("<"))
            | (state.chars == ord("+"))
        )
        dists = state.bfs_distances(passable)
        assert dists.shape == (21, 79)
        assert dists[state.y, state.x] == 0

        # Find any reachable tile at distance >= 1 and verify path_step returns a valid step
        reachable = np.argwhere(dists > 0)
        if len(reachable) > 0:
            ty, tx = int(reachable[0][0]), int(reachable[0][1])
            dy, dx = state.path_step(passable, (ty, tx))
            assert (dy, dx) != (0, 0)
            assert abs(dy) <= 1 and abs(dx) <= 1
    finally:
        harness.close()


def test_gamestate_c_tables_and_status_properties():
    harness = NetHackHarness(max_steps=20)
    try:
        st = harness.reset(seed=99)
        assert isinstance(st.hunger_name, str)
        assert isinstance(st.encumbrance_name, str)
        assert st.is_blind is False
        assert st.is_confused is False
        assert st.is_stunned is False
        assert st.is_hallucinating is False
        assert st.is_food_poisoned is False
        assert st.is_levitating is False

        # Test static glyph lookup helpers
        hero_g = st.glyph_at(st.y, st.x)
        assert st.is_monster_glyph(hero_g) or st.is_cmap_glyph(hero_g) or hero_g >= 0
        minfo = st.monster_info(hero_g)
        if minfo is not None:
            assert minfo.name != ""
            assert minfo.speed >= 0
    finally:
        harness.close()


def test_symbols_registry_real_fields():
    food = SYMBOL_REGISTRY.query_item("food ration")
    assert food is not None
    assert "nutrition" not in food
    assert food["delay"] > 0
    assert food["weight"] > 0
    assert "oprop" in food and "descr" in food

    eye = SYMBOL_REGISTRY.query_monster("floating eye")
    assert eye is not None
    assert "attacks" not in eye
    assert "flags1" in eye and "flags2" in eye and "flags3" in eye
    assert "geno" in eye and "msound" in eye and "mlet" in eye


def test_state_masks_readonly_and_object_occlusion_cleared():
    fm = FloorMemory()
    assert fm.passable_mask.flags.writeable is False
    assert fm.safe_passable_mask.flags.writeable is False

    h = NetHackHarness(max_steps=50)
    try:
        st = h.reset(seed=42)

        class FakeMsgState:
            y = st.y
            x = st.x
            depth = st.depth
            level_num = st.level_num
            dungeon_num = st.dungeon_num
            turn = st.turn + 1
            step_count = st.step_count + 1
            chars = st.chars
            glyphs = st.glyphs
            message = "There is nothing here to pick up."

        wm2 = WorldMemory()
        wm2.remembered_obj_glyphs[5, 5] = 123
        fms = FakeMsgState()
        fms.y = 5
        fms.x = 5
        fms.chars = np.full((21, 79), ord("."), dtype=np.uint8)
        fms.glyphs = np.zeros((21, 79), dtype=np.int16)
        wm2.update(fms, action_name="PICKUP")
        assert int(wm2.remembered_obj_glyphs[5, 5]) == 0
    finally:
        h.close()


# ---------------------------------------------------------------------------
# 2. C Engine Symbol Ground Truth Tests
# ---------------------------------------------------------------------------
def test_c_symbol_ground_truth_registry():
    reg = NetHackSymbolRegistry()

    # Query monster
    jackal = reg.query_monster("jackal")
    assert jackal is not None
    assert jackal["name"] == "jackal"
    assert "level" in jackal
    assert "ac" in jackal
    assert "speed" in jackal

    # Query item
    food = reg.query_item("food ration")
    assert food is not None
    assert food["name"] == "food ration"
    assert food["class"] == "FOOD"
    assert "cost" in food
    assert "weight" in food

    # Search items
    potions = reg.search_items("potion", limit=3)
    assert len(potions) >= 1
    assert any("potion" in p["name"] for p in potions)


def test_world_memory_multi_floor_bookkeeping_and_oscillation():
    mem = WorldMemory()
    mem.reset()

    # Create dummy observation tuple
    harness = NetHackHarness(max_steps=50)
    state = harness.reset(seed=123)

    assert hasattr(state, "mem")
    assert isinstance(state.mem, WorldMemory)

    # Initial state should be visited
    assert bool(state.mem.visited[state.y, state.x]) is True
    assert state.mem.visit_count[state.y, state.x] >= 1
    assert not state.mem.is_oscillating

    # Check that remembered chars contains terrain
    rem_chars = state.mem.remembered_chars
    assert isinstance(rem_chars, np.ndarray)
    assert rem_chars.shape == (21, 79)

    # Simulate multi-step oscillation on the same floor
    f0 = (state.dungeon_num, state.level_num)
    for i in range(14):
        y, x = (10, 20) if i % 2 == 0 else (10, 21)
        state.mem.pos_history.append((f0[0], f0[1], y, x))

    assert state.mem.is_oscillating is True

    # Switching floors should maintain separate FloorMemory
    fl1 = state.mem.get_floor(0, 1)
    fl2 = state.mem.get_floor(0, 2)
    fl1.visited[5, 5] = True
    assert bool(fl2.visited[5, 5]) is False


def test_gamestate_cursed_bug_traps_menu_and_guarded_range():
    from lox.env.harness import NetHackHarness
    from lox.env.state import _FLOOR_GLYPH_FLAGS_LUT, InventoryItem
    from lox.lang.instrument import PolicyLoopError

    # 1. "uncursed" must NEVER match is_cursed
    itm_un = InventoryItem(
        letter="c",
        name="an uncursed +3 small shield (being worn)",
        glyph=1,
        oclass=3,
        char="[",
        quantity=1,
        info=None,
        corpse_monster=None,
    )
    assert itm_un.is_uncursed is True
    assert itm_un.is_cursed is False
    assert itm_un.is_worn is True
    assert itm_un.enchantment == 3

    itm_cu = InventoryItem(
        letter="d",
        name="a cursed ring of hunger (on right hand)",
        glyph=2,
        oclass=4,
        char="=",
        quantity=1,
        info=None,
        corpse_monster=None,
    )
    assert itm_cu.is_cursed is True
    assert itm_cu.is_worn is True

    # 2. Interactive menu is preserved for at least 1 step instead of auto-dismissed
    h = NetHackHarness(max_steps=30)
    try:
        st0 = h.reset(42)
        assert st0.mem.remembered_trap_glyphs.shape == (21, 79)
        assert st0.mem.remembered_obj_glyphs.shape == (21, 79)
        assert st0.trap_mask.shape == (21, 79)
        assert st0.is_trap(0, 0) is False
        assert st0.trap_at(0, 0) is None
        st1, done1, _ = h.step(Action.INVENTORY)
        assert not done1
        assert st1.in_menu is True
        st2, done2, _ = h.step(Action.ESC)
        assert not done2
        assert st2.in_menu is False
        assert _FLOOR_GLYPH_FLAGS_LUT.shape[0] > 0
    finally:
        h.close()

    # 3. guarded_range supports huge range with early break AND catches real zero-yield loops
    ok_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        for i in range(10**9):\n"
        "            if i >= 5:\n"
        "                break\n"
        "        return Action.WAIT\n"
    )
    compiled_ok = compile_policy_package(ok_pol)
    runner_ok = compiled_ok.create_runner()
    act, _ = runner_ok.next_action(st0)
    assert act == Action.WAIT

    bad_pol = (
        "from lox.env.actions import Action\n"
        "class Policy:\n"
        "    def step(self, state):\n"
        "        tot = 0\n"
        "        for i in range(300000):\n"
        "            tot += i\n"
        "        return Action.WAIT\n"
    )
    compiled_bad = compile_policy_package(bad_pol, max_loop_ticks=50000)
    runner_bad = compiled_bad.create_runner()
    try:
        runner_bad.next_action(st0)
        raise AssertionError("Expected PolicyLoopError")
    except PolicyLoopError:
        pass


def test_v3_gamestate_hotpath_caches_and_trap_mask_parity():
    """Verifies Component 5 cached GameState slots, WorldMemory._cached_floor, cached trap_mask/is_trap, and stairs_down_positions."""
    from lox.env.state import _IS_TRAP_LUT

    h = NetHackHarness(max_steps=40)
    try:
        st = h.reset(seed=77)
        bl = st.blstats.tolist()
        assert st.x == bl[0]
        assert st.y == bl[1]
        assert st.pos == (bl[1], bl[0])
        assert st.depth == bl[12]
        assert st.ac == bl[16]
        assert st.xp_level == bl[18]
        assert st.turn == bl[20]
        assert st.dungeon_num == bl[23]
        assert st.level_num == bl[24]
        assert st.condition_mask == (bl[25] if len(bl) > 25 else 0)
        assert st.mem._cached_floor is st.mem.current_floor

        # Inject a known trap glyph and verify trap_mask & is_trap agree 100%
        trap_glyphs = np.where(_IS_TRAP_LUT)[0]
        assert len(trap_glyphs) > 0
        tg = int(trap_glyphs[0])
        st._trap_mask_cache = None
        st._trap_grid_cache = None
        st.glyphs[5, 10] = tg
        tm = st.trap_mask
        assert tm.dtype == np.bool_ and tm.shape == (21, 79)
        assert bool(tm[5, 10]) is True
        assert st.is_trap(5, 10) is True
        assert st.is_trap(0, 0) is False
        # Cached object identity on repeat access
        assert st.trap_mask is tm
        sd1 = st.stairs_down_positions
        assert st.stairs_down_positions is sd1
    finally:
        h.close()
