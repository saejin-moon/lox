"""Ascension goal-state primitives (extracted from corp/domain/macro_director.py).

R5→R6 checklist item 6: the legacy `MacroAscensionDirector` phase machine was
deleted after the GoalInterpreter's decision-trace equivalence was proven (R2)
and green since. These two primitives survive because the GoalInterpreter's
public surface (`state`, phase telemetry, goal_events) is built on them.
"""

from dataclasses import dataclass
from enum import IntEnum


class AscensionPhase(IntEnum):
    EARLY_EXPLORATION = 1    # DL 1-3: thorough room exploration, item pickup, leveling to XL >= 3-4
    EXCALIBUR_FORGE = 2      # XL >= 5 & Lawful: dip uncursed long sword in fountains for Excalibur
    POISON_RES_HUNT = 3      # Acquire permanent intrinsic poison resistance from conveyor corpses
    SOKOBAN_PROGRESSION = 4  # DL 6-9 upward branch: retrieve guaranteed Reflection from Sokoban floor 4
    MINETOWN_PROTECTION = 5  # Gnomish Mines: visit Minetown temple, donate 400*XL gold for divine AC <= -5
    DEEP_DESCENT = 6         # DL 10 to DL 20+: push down through Dungeons of Doom with artifact & divine armor
    # --- R6 Ascension Knowledge Stack milestones (mortality-driven order) ---
    SURVIVAL_STACK = 7       # early_survival_stack: AC ladder + food security + wand counterplay (DL 1-6)
    EQUIP_UPGRADE = 8        # equip_upgrade: weapon enchantment + armor AC <= -10 by DL 8
    INTRINSICS = 9           # survival_intrinsics: MR acquisition + drain counters (DL 8-12)
    GEHENNOM = 10            # gehennom_survival: light logistics + maze mapping (DL 14+)
    CASTLE = 11              # castle_wishing: drawbridge + wand-of-wishing protocol (DL 22+)
    VLAD_INVOCATION = 12     # vlad_invocation: Vlad's Tower -> Candelabrum -> Invocation (DL 30+)
    ASCENSION_RUN = 13       # ascension_run: Planes handling -> Astral -> ascend


@dataclass
class MacroDirectorState:
    current_phase: AscensionPhase = AscensionPhase.EARLY_EXPLORATION
    excalibur_obtained: bool = False
    reflection_obtained: bool = False
    poison_res_obtained: bool = False
    minetown_visited: bool = False
    temple_donations_count: int = 0
    sokoban_completed: bool = False
    sokoban_prize_collected: bool = False
    max_dlevel_reached: int = 1
    total_steps_in_phase: int = 0
    # --- R6 milestone flags (inventory/message-derived each turn) ---
    mr_obtained: bool = False                    # magic resistance (gray dragon gear / MR ring)
    light_source_carried: bool = False           # lamp / lantern / candles (gehennom logistics)
    wand_of_wishing_carried: bool = False        # identified wand of wishing
    amulet_obtained: bool = False                # the Amulet of Yendor
    candelabrum_obtained: bool = False           # Candelabrum of Invocation (implies Vlad defeated)
    bell_of_opening_obtained: bool = False       # Bell of Opening
    book_of_the_dead_obtained: bool = False      # Book of the Dead
    castle_wishing_done: bool = False            # BoH / dragon scale mail acquired past the Castle
    vlad_defeated: bool = False
    invocation_done: bool = False                # performed at the Vibrating Square
