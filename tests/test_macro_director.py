"""
Unit tests for MacroAscensionDirector.
Verifies phase transitions, early level farming guards, and deep descent staging.
"""

import numpy as np
import pytest

from corp.env.blstats import BottomLineStats, HungerState
from corp.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from corp.domain.macro_director import MacroAscensionDirector, AscensionPhase


def make_test_blstats(
    hp: int = 20,
    max_hp: int = 20,
    depth: int = 1,
    experience: int = 1,
    hunger_state: int = HungerState.NORMAL,
    dungeon_number: int = 0,
    level_number: int = 1,
    ac: int = 10,
    alignment: int = 1, # Lawful
) -> BottomLineStats:
    raw = np.zeros(27, dtype=np.int64)
    raw[0] = 10  # X
    raw[1] = 10  # Y
    raw[10] = hp # HP
    raw[11] = max_hp # HPMAX
    raw[12] = depth # DEPTH
    raw[16] = ac # AC
    raw[18] = experience # XP (Experience level)
    raw[19] = experience * 20 # EXP (Experience points)
    raw[20] = 100 # TIME
    raw[21] = hunger_state # HUNGER
    raw[23] = dungeon_number # DNUM
    raw[24] = level_number # DLEVEL
    raw[26] = alignment # ALIGN
    return BottomLineStats.from_blstats(raw)


def test_macro_director_phase_progression():
    director = MacroAscensionDirector()
    assert director.state.current_phase == AscensionPhase.EARLY_EXPLORATION

    # 1. Early exploration on DL 1, XL 1
    blstats = make_test_blstats(depth=1, experience=1)
    phase = director.update_state(blstats, role="valkyrie")
    assert phase == AscensionPhase.EARLY_EXPLORATION

    # 2. Reaching XL 5 as a lawful Valkyrie transitions to EXCALIBUR_FORGE
    blstats_xl5 = make_test_blstats(depth=3, experience=5)
    phase = director.update_state(blstats_xl5, role="valkyrie")
    assert phase == AscensionPhase.EXCALIBUR_FORGE

    # 3. Forging Excalibur transitions to POISON_RES_HUNT
    inv = InventoryNormalizer()
    excal = NormalizedItem(current_letter="a", raw_str="the blessed +0 Excalibur", buc_state="BLESSED")
    inv.active_items[excal.uid] = excal
    phase = director.update_state(blstats_xl5, inv_tracker=inv, role="valkyrie")
    assert phase == AscensionPhase.POISON_RES_HUNT

    # 4. Gaining poison resistance or reaching DL 6 transitions to SOKOBAN_PROGRESSION
    phase = director.update_state(blstats_xl5, message="You feel healthy.", role="valkyrie")
    assert phase == AscensionPhase.SOKOBAN_PROGRESSION

    # 5. Obtaining reflection transitions to MINETOWN_PROTECTION
    refl = NormalizedItem(current_letter="b", raw_str="an uncursed amulet of reflection", buc_state="UNCURSED")
    inv.active_items[refl.uid] = refl
    phase = director.update_state(blstats_xl5, inv_tracker=inv, role="valkyrie")
    assert phase == AscensionPhase.MINETOWN_PROTECTION

    # 6. Acquiring divine AC protection (AC <= 0) or DL 10 transitions to DEEP_DESCENT
    blstats_deep = make_test_blstats(depth=10, experience=8, ac=-5)
    phase = director.update_state(blstats_deep, role="valkyrie")
    assert phase == AscensionPhase.DEEP_DESCENT


def test_macro_director_stair_deferral():
    director = MacroAscensionDirector()

    # On DL 1 with 30 unvisited tiles and only 40 turns spent: DEFER STAIRS
    blstats = make_test_blstats(depth=1, experience=1)
    assert director.should_defer_stairs_for_farming(blstats, unvisited_count=30, turns_spent=40) is True

    # Once explored (unvisited <= 15) or turns >= 120: DO NOT DEFER
    assert director.should_defer_stairs_for_farming(blstats, unvisited_count=10, turns_spent=40) is False
    assert director.should_defer_stairs_for_farming(blstats, unvisited_count=30, turns_spent=130) is False

    # If starving/weak: NEVER defer stairs
    blstats_weak = make_test_blstats(depth=1, experience=1, hunger_state=HungerState.WEAK)
    assert director.should_defer_stairs_for_farming(blstats_weak, unvisited_count=30, turns_spent=40) is False


def test_macro_director_mines_entry_guard():
    director = MacroAscensionDirector()

    # XL 1-5 without preparation cannot enter mines
    blstats_low = make_test_blstats(experience=3, hp=25)
    assert director.can_safely_enter_mines(blstats_low) is False

    # XL 6+ can enter mines
    blstats_high = make_test_blstats(experience=6, hp=45)
    assert director.can_safely_enter_mines(blstats_high) is True
