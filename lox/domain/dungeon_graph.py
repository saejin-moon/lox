"""
Domain Layer: Macro Dungeon Branch Graph.
Tracks cross-level topologies, branch transitions (Dungeons of Doom, Gnomish Mines, Sokoban),
and safe branch routing policies.
"""

from dataclasses import dataclass, field
from typing import Any
from lox.env.blstats import BottomLineStats


@dataclass
class DungeonNode:
    """Represents a discrete dungeon level identified by (dnum, dlevel)."""
    dnum: int
    dlevel: int
    branch_name: str
    stairs_down: tuple[int, int] | None = None
    stairs_up: tuple[int, int] | None = None
    has_shops: bool = False
    has_altar: bool = False
    has_priest: bool = False
    is_minetown: bool = False
    turns_visited: int = 0
    fully_explored: bool = False
    # Connected edges: stair_pos -> (target_dnum, target_dlevel)
    stair_connections: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)


class DungeonGraph:
    """
    Maintains the cross-level macro dungeon graph and governs branch policy:
    1. Dungeons of Doom (dnum=0): Primary progression backbone.
    2. Gnomish Mines (dnum=2): Entry DL 2-4. Guarded: enter only if XL >= 5 or light/infravision present.
    3. Sokoban (dnum=3): Entry DL 5-9 via upward staircase. High reward: food & reflection/BoH.
    """

    BRANCH_NAMES = {
        0: "Dungeons of Doom",
        1: "Gehennom / Quest",
        2: "Gnomish Mines",
        3: "Sokoban",
        4: "Fort Ludios",
    }

    def __init__(self):
        self.nodes: dict[tuple[int, int], DungeonNode] = {}
        self.current_key: tuple[int, int] = (0, 1)
        self.previous_key: tuple[int, int] | None = None
        self.last_stair_taken: tuple[int, int] | None = None

    def reset(self):
        self.nodes.clear()
        self.current_key = (0, 1)
        self.previous_key = None
        self.last_stair_taken = None

    def get_or_create_node(self, dnum: int, dlevel: int) -> DungeonNode:
        key = (dnum, dlevel)
        if key not in self.nodes:
            branch = self.BRANCH_NAMES.get(dnum, f"Branch_{dnum}")
            self.nodes[key] = DungeonNode(dnum=dnum, dlevel=dlevel, branch_name=branch)
        return self.nodes[key]

    def record_step(self, blstats: BottomLineStats, message: str = "") -> DungeonNode:
        """Updates graph state upon each step, detecting level transitions and branch changes."""
        dnum = int(blstats.dungeon_number)
        dlevel = int(blstats.dlevel)
        key = (dnum, dlevel)

        node = self.get_or_create_node(dnum, dlevel)
        node.turns_visited += 1

        # Detect level transition
        if key != self.current_key:
            self.previous_key = self.current_key
            self.current_key = key

            # If we just took a staircase, link the previous node's stair to this new level
            if self.previous_key in self.nodes and self.last_stair_taken is not None:
                prev_node = self.nodes[self.previous_key]
                prev_node.stair_connections[self.last_stair_taken] = key

        msg_lower = message.lower()
        if "welcome to minetown" in msg_lower or (dnum == 2 and 3 <= dlevel <= 5 and "shop" in msg_lower):
            node.is_minetown = True

        if "temple" in msg_lower or "shrine" in msg_lower:
            node.has_priest = True

        return node

    def record_stair_usage(self, stair_pos: tuple[int, int]):
        """Records that the agent took the staircase at stair_pos."""
        self.last_stair_taken = stair_pos

    def should_enter_mines(
        self,
        blstats: BottomLineStats,
        has_light: bool = False,
        has_infravision: bool = False,
    ) -> bool:
        """
        Policy guard for descending into Gnomish Mines (dnum=2).
        Gnomish Mines are dark and gnomes carry wands of striking/lightning.
        Entering under-leveled (XL < 5) without light or infravision leads to instant death.
        """
        if blstats.experience_level >= 5:
            return True
        if has_light or has_infravision:
            return blstats.experience_level >= 3 and blstats.hp >= int(blstats.max_hp * 0.75)
        return False

    def should_ascend_from_mines(
        self,
        blstats: BottomLineStats,
        has_light: bool = False,
        has_infravision: bool = False,
    ) -> bool:
        """
        Policy guard for retreating from Gnomish Mines back to Dungeons of Doom.
        If under-leveled (XL < 5) and without light, or HP is critically low, ascend immediately!
        """
        if blstats.dungeon_number != 2:
            return False
        if blstats.experience_level < 5 and not (has_light or has_infravision):
            return True
        if blstats.hp < int(blstats.max_hp * 0.45):
            return True
        return False

    def is_in_sokoban(self, blstats: BottomLineStats) -> bool:
        """Returns True if the hero is in Sokoban (dnum=3)."""
        return blstats.dungeon_number == 3

    def is_in_mines(self, blstats: BottomLineStats) -> bool:
        """Returns True if the hero is in Gnomish Mines (dnum=2)."""
        return blstats.dungeon_number == 2
