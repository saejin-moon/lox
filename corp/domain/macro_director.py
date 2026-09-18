"""
Macro Ascension Director: Strategic phase coordinator for deep dungeon ascent.
Guides the agent through staged ascension milestones from Early Exploration to Deep Descent (DL 20+).
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Any

from corp.env.blstats import BottomLineStats, HungerState
from corp.policy.config import PolicyConfig, default_config


class AscensionPhase(IntEnum):
    EARLY_EXPLORATION = 1    # DL 1-3: thorough room exploration, item pickup, leveling to XL >= 3-4
    EXCALIBUR_FORGE = 2      # XL >= 5 & Lawful: dip uncursed long sword in fountains for Excalibur
    POISON_RES_HUNT = 3      # Acquire permanent intrinsic poison resistance from conveyor corpses
    SOKOBAN_PROGRESSION = 4  # DL 6-9 upward branch: retrieve guaranteed Reflection from Sokoban floor 4
    MINETOWN_PROTECTION = 5  # Gnomish Mines: visit Minetown temple, donate 400*XL gold for divine AC <= -5
    DEEP_DESCENT = 6         # DL 10 to DL 20+: push down through Dungeons of Doom with artifact & divine armor


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


class MacroAscensionDirector:
    """
    Coordinates macro-level goals and prevents catastrophic early blunders:
    1. Early Farming: Prevents premature stair diving on DL 1-3.
    2. Excalibur Milestones: Ensures lawful melee fighters prioritize dipping once XL >= 5.
    3. Mines Steering: Strictly prevents entering Gnomish Mines until XL >= 6 and prepared.
    4. Sokoban Reflection: Directs the agent to Sokoban for guaranteed reflection.
    5. Divine AC Protection: Coordinates Minetown temple donations to lower AC <= -5.
    6. Deep Ascent: Drives aggressive depth progression past DL 10 once fully geared.
    """

    def __init__(self, config: PolicyConfig | None = None):
        self.cfg = config or default_config()
        self.state = MacroDirectorState()

    def reset(self):
        self.state = MacroDirectorState()

    def update_state(
        self,
        blstats: BottomLineStats,
        message: str = "",
        inv_tracker: Any = None,
        dungeon_graph: Any = None,
        role: str = "",
        has_poison_res: bool = False,
        has_reflection: bool = False,
        temple_donations: int = 0,
    ) -> AscensionPhase:
        """
        Updates ascension milestones based on current observations and returns the active phase.
        """
        self.state.total_steps_in_phase += 1
        msg_lower = message.lower()

        # Sync temple donation progress from ShopManager
        if temple_donations > self.state.temple_donations_count:
            self.state.temple_donations_count = temple_donations

        # Sokoban prize tracking: once reflection is obtained inside Sokoban, the branch is done;
        # exit by descending back to the Dungeons of Doom (prevents floor 3/4 oscillation)
        if getattr(blstats, "dungeon_number", 0) == 3 and self.state.reflection_obtained:
            self.state.sokoban_prize_collected = True

        # Update gear / intrinsic status
        if has_poison_res or self.state.poison_res_obtained:
            self.state.poison_res_obtained = True
        if "feel healthy" in msg_lower or "feel especially healthy" in msg_lower:
            self.state.poison_res_obtained = True

        if has_reflection or self.state.reflection_obtained:
            self.state.reflection_obtained = True

        if inv_tracker is not None:
            active = (
                inv_tracker.get_active_items()
                if hasattr(inv_tracker, "get_active_items")
                else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
            )
            for it in active:
                desc = it.raw_str.lower()
                if "excalibur" in desc:
                    self.state.excalibur_obtained = True
                if any(k in desc for k in ("shield of reflection", "amulet of reflection", "silver dragon")):
                    self.state.reflection_obtained = True

        # Track dungeon progress
        if blstats.depth > self.state.max_dlevel_reached:
            self.state.max_dlevel_reached = blstats.depth

        if "welcome to minetown" in msg_lower:
            self.state.minetown_visited = True

        # Sokoban prize tracking (after reflection flag updates — R2 ordering fix):
        # once reflection is obtained inside Sokoban, the branch is done; exit by descending
        # back to the Dungeons of Doom (prevents floor 3/4 oscillation)
        if getattr(blstats, "dungeon_number", 0) == 3 and self.state.reflection_obtained:
            self.state.sokoban_prize_collected = True

        # Phase transition machine
        r_lower = role.lower()
        is_lawful_fighter = (blstats.alignment == 1) or (r_lower in ("valkyrie", "samurai", "knight"))

        # Phase 1: Early Exploration (Floors 1-3)
        if self.state.current_phase == AscensionPhase.EARLY_EXPLORATION:
            if is_lawful_fighter and blstats.experience_level >= self.cfg.strategy.excalibur_xl and not self.state.excalibur_obtained:
                self.state.current_phase = AscensionPhase.EXCALIBUR_FORGE
                self.state.total_steps_in_phase = 0
            elif blstats.experience_level >= self.cfg.strategy.early_exit_xl or blstats.depth >= self.cfg.strategy.early_exit_depth:
                self.state.current_phase = AscensionPhase.POISON_RES_HUNT
                self.state.total_steps_in_phase = 0

        # Phase 2: Excalibur Forging
        elif self.state.current_phase == AscensionPhase.EXCALIBUR_FORGE:
            if self.state.excalibur_obtained or self.state.total_steps_in_phase > self.cfg.strategy.excalibur_timeout_steps:
                self.state.current_phase = AscensionPhase.POISON_RES_HUNT
                self.state.total_steps_in_phase = 0

        # Phase 3: Poison Resistance Hunt
        elif self.state.current_phase == AscensionPhase.POISON_RES_HUNT:
            if self.state.poison_res_obtained or blstats.depth >= self.cfg.strategy.poison_exit_depth:
                self.state.current_phase = AscensionPhase.SOKOBAN_PROGRESSION
                self.state.total_steps_in_phase = 0

        # Phase 4: Sokoban Progression (Reflection)
        elif self.state.current_phase == AscensionPhase.SOKOBAN_PROGRESSION:
            if self.state.reflection_obtained or blstats.depth >= self.cfg.strategy.sokoban_exit_depth or self.state.sokoban_completed:
                self.state.current_phase = AscensionPhase.MINETOWN_PROTECTION
                self.state.total_steps_in_phase = 0

        # Phase 5: Minetown Divine Protection
        elif self.state.current_phase == AscensionPhase.MINETOWN_PROTECTION:
            # Transition to deep descent once divine protection acquired (AC <= 0) or explored
            if blstats.ac <= self.cfg.strategy.minetown_ac_target or self.state.total_steps_in_phase > self.cfg.strategy.minetown_timeout_steps or blstats.depth >= self.cfg.strategy.minetown_exit_depth:
                self.state.current_phase = AscensionPhase.DEEP_DESCENT
                self.state.total_steps_in_phase = 0

        return self.state.current_phase

    def should_defer_stairs_for_farming(
        self,
        blstats: BottomLineStats,
        unvisited_count: int,
        turns_spent: int,
    ) -> bool:
        """
        Determines if the hero should stay and explore/farm the current floor
        rather than rushing down the staircase.
        """
        # If starving or weak, never defer stairs: find food immediately!
        if blstats.hunger_state >= HungerState.WEAK:
            return False

        s = self.cfg.strategy
        # On DL 1: Explore until unvisited tiles <= 15 or turns >= 120
        if blstats.depth == 1:
            if unvisited_count > s.farming_unvisited and turns_spent < s.farming_turns_d1 and blstats.experience_level < 2:
                return True

        # On DL 2-3: Explore until unvisited tiles <= 15 or turns >= 150
        if blstats.depth in (2, 3):
            if unvisited_count > s.farming_unvisited and turns_spent < s.farming_turns_d23 and blstats.experience_level < 3:
                return True

        return False

    def get_navigation_directive(self, blstats: BottomLineStats, dungeon_graph: Any = None) -> str | None:
        """
        Returns macro navigation override steering commands for the NavigationManager:
        - 'DESCEND':            Aggressively prioritize stair descent (DEEP_DESCENT phase)
        - 'ENTER_SOKOBAN':      Hunt the Sokoban branch entry (dnum 0, DL 5-9) via upstairs '<'
        - 'ASCEND_FROM_MINES':  Retreat from Gnomish Mines back to the Dungeons of Doom
        - 'EXIT_SOKOBAN':       Prize collected inside Sokoban: descend back to dnum 0
        Returns None to let normal navigation decide.
        """
        dnum = int(getattr(blstats, "dungeon_number", 0))
        depth = int(getattr(blstats, "depth", 1))
        dlevel = int(getattr(blstats, "dlevel", 0) or 0)

        # 1. Mines retreat policy (branch dead-end: no downstairs exist below Mine's End)
        if dnum == 2:
            if dlevel >= self.cfg.descent.mines_end_dlevel:
                return "ASCEND_FROM_MINES"
            if self.state.minetown_visited and self.state.temple_donations_count >= self.cfg.descent.minetown_donations_done:
                return "ASCEND_FROM_MINES"
            if dungeon_graph is not None:
                node = dungeon_graph.nodes.get((dnum, dlevel))
                if node is not None and node.fully_explored and node.stairs_down is None:
                    return "ASCEND_FROM_MINES"
            return None

        # 2. Sokoban exit after prize collection (prevents floor 3/4 oscillation)
        if dnum == 3:
            return "EXIT_SOKOBAN" if self.state.sokoban_prize_collected else None

        # 2.5 Minetown divine protection: route into the Gnomish Mines branch (entry DL 2-4)
        if (
            self.state.current_phase == AscensionPhase.MINETOWN_PROTECTION
            and dnum == 0
            and not self.state.minetown_visited
            and self.cfg.descent.minetown_depth_lo <= depth <= self.cfg.descent.minetown_depth_hi
        ):
            return "GOTO_MINETOWN"

        # 3. Deep descent override (drive depth progression once geared)
        if self.state.current_phase == AscensionPhase.DEEP_DESCENT and dnum == 0:
            return "DESCEND"

        # 4. Sokoban entry hunt (reflection push): DL 5-9 branch via upstairs '<'
        if (
            self.state.current_phase == AscensionPhase.SOKOBAN_PROGRESSION
            and dnum == 0
            and not self.state.sokoban_prize_collected
        ):
            # Cap the branch hunt so a fruitless search resumes normal descent progression
            if self.state.total_steps_in_phase <= self.cfg.descent.sokoban_hunt_step_cap:
                return "ENTER_SOKOBAN"

        return None

    def get_minetown_donation_target(self, blstats: BottomLineStats) -> int:
        """
        Full divine protection requires ~5 donations of 400 * XL gold each
        (2-4 AC points per donation, targeting AC <= -5).
        """
        return 400 * max(1, blstats.experience_level) * 5

    def can_safely_enter_mines(self, blstats: BottomLineStats, has_light: bool = False) -> bool:
        """
        NetHack Rule: Entering the Gnomish Mines under-leveled (XL < 6)
        leads to fatal wand of striking/lightning zaps from gnome lords.
        """
        s = self.cfg.strategy
        if blstats.experience_level >= s.mines_enter_xl:
            return True
        if self.state.excalibur_obtained and blstats.experience_level >= s.mines_alt_xl and blstats.hp >= s.mines_alt_hp:
            return True
        return False
