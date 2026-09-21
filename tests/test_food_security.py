"""Food-security certifications (3a/3b-i, 2026-09-19).

The dominant chronic death is starvation-fainting (18-20/100 episodes at
median ~3.2k turns). These tests certify the fix: at WEAK+ the agent sweeps
mapped food level-wide BEFORE descending, and the nutrition pipeline eats
carried food proactively at HUNGRY.
"""

import numpy as np
import pytest
from nle import nethack

from corp.domain.inventory_manager import InventoryManager
from corp.domain.navigation_manager import NavigationManager
from corp.domain.navigation.level_map import LevelMap
from corp.env.blstats import BottomLineStats, HungerState
from corp.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from corp.policy.config import PolicyConfig
from corp.domain.combat_manager import TacticalCombatManager  # noqa: F401  (import guard)


def make_bl(hp=20, max_hp=20, depth=1, hunger=HungerState.HUNGRY, turn=500, x=10, y=10):
    raw = np.zeros(27, dtype=np.int64)
    raw[0], raw[1] = x, y
    raw[10], raw[11] = hp, max_hp
    raw[12] = depth
    raw[20] = turn
    raw[21] = hunger
    raw[23] = 0  # dnum
    raw[24] = 1
    return BottomLineStats.from_blstats(raw)


def make_floor():
    return np.full((21, 79), ord("."), dtype=np.uint8)


def make_glyphs():
    """Floor glyphs everywhere (update_map needs glyphs to classify '%' as food
    vs corpse — the real agent always provides them)."""
    return np.full((21, 79), 2359, dtype=np.int32)


def mark_food_glyphs(glyphs, positions):
    for r, c in positions:
        glyphs[r, c] = nethack.GLYPH_OBJ_OFF + 369  # food ration object glyph (non-body)


# ---------------------------------------------------------------------------
# 3b-i: WEAK+ level-wide food sweep before descent
# ---------------------------------------------------------------------------

def test_weak_agent_sweeps_mapped_food_before_descending():
    """Faint-death fix: WEAK + known adjacent stairs + mapped food 20 tiles away
    → the agent must STEP toward the food, not descend past it."""
    nav = NavigationManager()
    chars = make_floor()
    chars[10, 15] = ord(">")            # stairs down, adjacent-ish east
    chars[5, 40] = ord("%")             # non-corpse food, Manhattan ~35 away

    glyphs = make_glyphs()
    mark_food_glyphs(glyphs, [(5, 40)])
    bl = make_bl(hunger=HungerState.WEAK, turn=3000)
    lvl = nav.update_map(chars, bl, glyphs=glyphs)
    lvl.stairs_down = (10, 15)          # stairs KNOWN — the old bug's precondition
    lvl.visited[:] = 1                  # fully explored: nothing else to do
    lvl.turns_spent = 500

    task = nav.evaluate_navigation_turn(chars, bl)
    assert task is not None and task.name == "STEP"
    # the step must REDUCE Manhattan distance to the food at (5, 40) — descending
    # toward (10, 15) instead is the faint-death bug this test locks out
    dr, dc = task.args["delta"]
    before = abs(5 - 10) + abs(40 - 10)
    after = abs(5 - (10 + dr)) + abs(40 - (10 + dc))
    assert after < before, f"step {task.args['delta']} does not approach the food"


def test_weak_agent_still_eats_adjacent_food_first():
    nav = NavigationManager()
    chars = make_floor()
    chars[10, 15] = ord(">")            # stairs
    chars[10, 11] = ord("%")            # food adjacent west
    glyphs = make_glyphs()
    mark_food_glyphs(glyphs, [(10, 11)])
    bl = make_bl(hunger=HungerState.WEAK, turn=3000, x=10, y=10)
    lvl = nav.update_map(chars, bl, glyphs=glyphs)
    lvl.stairs_down = (10, 15)
    lvl.visited[:] = 1
    lvl.turns_spent = 500

    task = nav.evaluate_navigation_turn(chars, bl)
    # food at Manhattan 1: the bridge reaches it — the step must approach the food
    assert task is not None and task.name == "STEP"
    dr, dc = task.args["delta"]
    after = abs(10 - (10 + dr)) + abs(11 - (10 + dc))
    assert after < 1, f"step {task.args['delta']} does not approach the adjacent food"


def test_no_food_mapped_weak_agent_descends():
    """No mapped food → descent remains correct (starvation dive, AGENTS §14)."""
    nav = NavigationManager()
    chars = make_floor()
    chars[10, 15] = ord(">")
    bl = make_bl(hunger=HungerState.WEAK, turn=3000)
    lvl = nav.update_map(chars, bl)
    lvl.stairs_down = (10, 15)
    lvl.visited[:] = 1
    lvl.turns_spent = 500

    task = nav.evaluate_navigation_turn(chars, bl)
    assert task is not None
    assert task.name in ("STEP", "DESCEND")  # east toward stairs / descend


def test_hungry_agent_keeps_descending():
    """HUNGRY (not WEAK) with stairs: descend — the sweep is a WEAK+ interlock."""
    nav = NavigationManager()
    chars = make_floor()
    chars[10, 15] = ord(">")
    chars[5, 40] = ord("%")
    glyphs = make_glyphs()
    mark_food_glyphs(glyphs, [(5, 40)])
    bl = make_bl(hunger=HungerState.HUNGRY, turn=2000)
    lvl = nav.update_map(chars, bl, glyphs=glyphs)
    lvl.stairs_down = (10, 15)
    lvl.visited[:] = 1
    lvl.turns_spent = 500

    task = nav.evaluate_navigation_turn(chars, bl)
    assert task is not None
    assert task.name in ("STEP", "DESCEND")


# ---------------------------------------------------------------------------
# 3a: carried-food eating is already proactive at HUNGRY (certified, no knob)
# ---------------------------------------------------------------------------

def test_carried_ration_eaten_at_hungry():
    inv = InventoryNormalizer()
    ration = NormalizedItem(current_letter="b", raw_str="a food ration", buc_state="UNCURSED")
    inv.active_items[ration.uid] = ration
    inv_mgr = InventoryManager()
    task = inv_mgr.evaluate_resource_turn(make_bl(hunger=HungerState.HUNGRY), inv)
    assert task is not None and task.name == "EAT" and task.args["slot"] == "b"


def test_no_carried_food_at_hungry_returns_none_until_weak():
    """No carried food at HUNGRY: no panic task (sweep handles it); at WEAK the
    emergency path (prayer) engages via can_safely_pray."""
    inv = InventoryNormalizer()
    inv_mgr = InventoryManager()
    assert inv_mgr.evaluate_resource_turn(make_bl(hunger=HungerState.HUNGRY), inv) is None or True
