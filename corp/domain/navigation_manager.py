"""
Domain Layer: Spatial Navigation Manager.
Governs cost-weighted Grid A* pathfinding, unvisited frontier exploration, corridor dead-end
secret door searches, and stairs navigation.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from corp.env.blstats import BottomLineStats, HungerState
from nle import nethack
from corp.navigation.astar import GridAStar, PathNode
from corp.navigation.frontier import FrontierExplorer
from corp.planner.htn import Task, PrimitiveTask
from corp.planner.guards import HTNGuards
from corp.policy.config import PolicyConfig, default_config
from corp.domain.navigation.level_map import LevelMap, LevelStore
from corp.domain.navigation.stepping import SteppingMixin
from corp.domain.navigation.exploration import ExplorationMixin


class NavigationManager(ExplorationMixin, SteppingMixin):
    """
    Directs spatial traversal and exploration across NetHack's 80x21 grid.
    Strictly enforces:
    1. Grid corner-clipping and doorway interlocks (via GridAStar).
    2. Hazard cost weighting (traps, lava, water).
    3. Unvisited floor and corridor exploration before stair descent.
    4. Corridor dead-end and perimeter wall secret door search bursts.
    5. Autonomous stair descent once floor frontiers are exhausted.
    """

    WALKABLE_CHARS = {
        ord("."),  # Room floor
        ord("#"),  # Corridor
        ord("+"),  # Closed door
        ord("'"),  # Open door
        ord("@"),  # Player glyph
        ord("<"),  # Stairs up
        ord(">"),  # Stairs down
        ord("_"),  # Altar
        ord("{"),  # Fountain
        ord("\\"), # Throne
        ord("^"),  # Trap
        ord("$"),  # Gold
        ord(")"),  # Weapon
        ord("["),  # Armor
        ord("?"),  # Scroll
        ord("!"),  # Potion
        ord("/"),  # Wand
        ord("="),  # Ring
        ord("\""), # Amulet
        ord("("),  # Tool
        ord("%"),  # Food
        ord("*"),  # Gem
    }

    # Precomputed 256-element boolean lookup table for zero-allocation C-speed masking
    WALKABLE_LUT = np.zeros(256, dtype=bool)
    for _c in WALKABLE_CHARS:
        WALKABLE_LUT[_c] = True

    ROWS = 21
    COLS = 79
    SIZE = ROWS * COLS
    ALL_NEIGHBORS = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    NEIGHBOR_DELTAS = [dr * 79 + dc for dr, dc in ALL_NEIGHBORS]
    DIR_DR = [dr for dr, _ in ALL_NEIGHBORS]
    DIR_DC = [dc for _, dc in ALL_NEIGHBORS]

    _bfs_epoch = 0
    _bfs_visited = [0] * SIZE

    def __init__(self, config: PolicyConfig | None = None):
        self.cfg = config or default_config()
        self.levels: LevelStore[tuple[int, int], LevelMap] = LevelStore()
        self.astar = GridAStar()
        self.frontier_explorer = FrontierExplorer()
        self.last_step_delta: tuple[int, int] | None = None
        self.unlock_tool_slot: str | None = None
        self._current_search_spot: tuple[int, int] | None = None
        self._current_search_burst: int = 0
        self._last_dnum: int = 0
        self._last_dlevel: int = 1
        self._prev_hp: int = 0
        self._rest_active: bool = False

    @property
    def visited_thrones(self) -> set[tuple[int, int]]:
        res = set()
        for lvl in self.levels.values():
            res.update(lvl.visited_thrones)
        return res

    @property
    def looted_containers(self) -> set[tuple[int, int]]:
        res = set()
        for lvl in self.levels.values():
            res.update(lvl.looted_containers)
        return res

    @property
    def collected_gold(self) -> set[tuple[int, int]]:
        res = set()
        for lvl in self.levels.values():
            res.update(lvl.collected_gold)
        return res

    @property
    def disarmed_traps(self) -> set[tuple[int, int]]:
        res = set()
        for lvl in self.levels.values():
            res.update(lvl.disarmed_traps)
        return res

    @property
    def dead_end_searches(self) -> dict[tuple[int, int], int]:
        res = {}
        for lvl in self.levels.values():
            res.update(lvl.dead_end_searches)
        return res

    @property
    def door_attempts(self) -> dict[tuple[int, int], int]:
        res = {}
        for lvl in self.levels.values():
            res.update(lvl.door_attempts)
        return res

    def reset(self):
        """Clears all level maps and exploration caches for a new episode."""
        self.levels.clear()
        self.last_step_delta = None
        self.unlock_tool_slot = None
        self._current_search_spot = None
        self._current_search_burst = 0
        self._last_dnum = 0
        self._last_dlevel = 1
        self._rest_active = False

    def get_or_create_level(
        self,
        dnum_or_depth: int | tuple[int, int] | BottomLineStats,
        dlevel: int | None = None,
        shape: tuple[int, int] = (21, 79),
        depth: int | None = None,
    ) -> LevelMap:
        """
        Retrieves or initializes a branch-aware LevelMap.
        Supports (dnum, dlevel), BottomLineStats, or single-integer depth for backward compatibility.
        """
        if isinstance(dnum_or_depth, BottomLineStats):
            dnum = dnum_or_depth.dungeon_number
            dlevel = dnum_or_depth.level_number if dnum_or_depth.level_number > 0 else dnum_or_depth.depth
            depth = dnum_or_depth.depth
        elif isinstance(dnum_or_depth, tuple):
            dnum, dlevel = dnum_or_depth
            if depth is None:
                depth = dlevel
        elif dlevel is not None:
            dnum = dnum_or_depth
            if depth is None:
                depth = dlevel
        else:
            dnum = 0
            dlevel = dnum_or_depth
            depth = dnum_or_depth

        key = (dnum, dlevel)
        if key not in self.levels:
            self.levels[key] = LevelMap(
                depth=depth,
                dnum=dnum,
                dlevel=dlevel,
                walkable=np.zeros(shape, dtype=bool),
                mapped=np.zeros(shape, dtype=bool),
                walls=np.zeros(shape, dtype=bool),
                visited=np.zeros(shape, dtype=np.int32),
                searched=np.zeros(shape, dtype=np.int32),
            )
        return self.levels[key]

    def update_map(
        self,
        chars: np.ndarray,
        blstats: BottomLineStats,
        message: str = "",
        glyphs: np.ndarray | None = None,
        dungeon_graph: Any = None,
    ) -> LevelMap:
        """
        Updates the internal topological map using current sensory chars via vectorized LUT.
        """
        dnum = getattr(blstats, "dungeon_number", 0)
        dlevel = getattr(blstats, "level_number", 0)
        if dlevel <= 0:
            dlevel = getattr(blstats, "depth", 1)
        if (dnum, dlevel) != (self._last_dnum, self._last_dlevel):
            self._current_search_spot = None
            self._current_search_burst = 0
            self._last_dnum = dnum
            self._last_dlevel = dlevel
        lvl = self.get_or_create_level(dnum, dlevel, chars.shape, depth=blstats.depth)
        py, px = blstats.y, blstats.x
        lvl.turns_spent += 1

        # Vectorized walkable update (60x faster than scalar 21x79 loop)
        # NetHack Interlock: Hallucination completely randomizes monster, wall, and item glyphs;
        # forbid updating topological walkable/walls while hallucinating to prevent permanent map corruption!
        if not blstats.is_hallucinating:
            lvl.walkable |= self.WALKABLE_LUT[chars]
            # Water Crossing (Phase 4): levitation/flying makes water tiles traversable
            if blstats.is_levitating or blstats.is_flying:
                lvl.walkable |= chars == ord("}")
            # Vectorized mapped update (all non-null, non-space characters seen)
            lvl.mapped |= (chars != 0) & (chars != ord(" "))
            # Vectorized walls update
            lvl.walls |= (chars == ord("|")) | (chars == ord("-"))

        # Permanently enforce blocked/unwalkable obstacles on this level
        for br, bc in lvl.blocked_tiles:
            if 0 <= br < lvl.walkable.shape[0] and 0 <= bc < lvl.walkable.shape[1]:
                lvl.walkable[br, bc] = False
        # Mark all other '@' (shopkeepers, priests, watchmen, guards) as strictly UNWALKABLE
        other_humans = np.argwhere(chars == ord("@"))
        for pt in other_humans:
            hr, hc = int(pt[0]), int(pt[1])
            if (hr, hc) != (py, px):
                lvl.walkable[hr, hc] = False

        # Vectorized stairs detection
        down_pts = np.argwhere(chars == ord(">"))
        if len(down_pts) > 0:
            for pt in down_pts:
                sp = (int(pt[0]), int(pt[1]))
                lvl.all_stairs_down.add(sp)

            # Filter out staircases that lead to dangerous branches (e.g. Gnomish Mines when under-leveled)
            current_node = dungeon_graph.nodes.get((dnum, dlevel)) if dungeon_graph is not None else None
            safe_down_stairs = []
            for pt in down_pts:
                sp = (int(pt[0]), int(pt[1]))
                lvl.all_stairs_down.add(sp)
                if current_node is not None:
                    target_branch = current_node.stair_connections.get(sp)
                    if target_branch is not None and target_branch[0] == 2:  # Leads to Mines
                        if not dungeon_graph.should_enter_mines(blstats):
                            lvl.blocked_tiles.add(sp)  # Block the Mines staircase
                            continue
                safe_down_stairs.append(sp)

            if safe_down_stairs:
                lvl.stairs_down = safe_down_stairs[0]
            elif lvl.stairs_down in lvl.blocked_tiles:
                lvl.stairs_down = None

        up_pts = np.argwhere(chars == ord("<"))
        if len(up_pts) > 0:
            for pt in up_pts:
                lvl.all_stairs_up.add((int(pt[0]), int(pt[1])))
            lvl.stairs_up = (int(up_pts[0, 0]), int(up_pts[0, 1]))
        elif lvl.stairs_up is None and lvl.turns_spent <= 2 and (dnum > 0 or dlevel > 1):
            lvl.stairs_up = (py, px)
            lvl.all_stairs_up.add((py, px))

        # Message-based stairs detection (when item or monster obscures the '>' on floor)
        msg_lower = message.lower()
        if any(w in msg_lower for w in ("staircase down", "ladder down", "stairs down", "stair down", "down staircase")):
            if (py, px) not in lvl.blocked_tiles:
                lvl.stairs_down = (py, px)
                lvl.all_stairs_down.add((py, px))
        if any(w in msg_lower for w in ("staircase up", "ladder up", "stairs up", "stair up", "up staircase")):
            lvl.stairs_up = (py, px)
            lvl.all_stairs_up.add((py, px))

        # Vectorized doors detection (closed '+' and open '\'')
        door_pts = np.argwhere((chars == ord("+")) | (chars == ord("'")))
        for pt in door_pts:
            lvl.doors.add((int(pt[0]), int(pt[1])))

        # Shop perception & shop door interlock
        if "closed for inventory" in msg_lower or ("welcome to" in msg_lower and "shop" in msg_lower) or "(unpaid" in msg_lower or "shopkeeper" in msg_lower:
            lvl.has_shops = True
            if "closed for inventory" in msg_lower:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = py + dr, px + dc
                        if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                            if chars[nr, nc] == ord("+") or (nr, nc) in lvl.doors:
                                lvl.shop_doors.add((nr, nc))
            if "welcome to" in msg_lower and "shop" in msg_lower and self.last_step_delta is not None:
                dr, dc = self.last_step_delta
                prev_tile = (py - dr, px - dc)
                lvl.shop_doors.add(prev_tile)

        # Vectorized corridors detection ('#')
        corridor_pts = np.argwhere(chars == ord("#"))
        for pt in corridor_pts:
            lvl.corridors.add((int(pt[0]), int(pt[1])))

        # Vectorized fountains detection
        fountain_pts = np.argwhere(chars == ord("{"))
        for pt in fountain_pts:
            lvl.fountains.add((int(pt[0]), int(pt[1])))

        # Vectorized altars detection ('_')
        altar_pts = np.argwhere(chars == ord("_"))
        for pt in altar_pts:
            lvl.altars.add((int(pt[0]), int(pt[1])))

        # Vectorized thrones detection ('\\')
        throne_pts = np.argwhere(chars == ord("\\"))
        for pt in throne_pts:
            lvl.thrones.add((int(pt[0]), int(pt[1])))

        # Vectorized traps detection ('^')
        trap_pts = np.argwhere(chars == ord("^"))
        for pt in trap_pts:
            lvl.traps.add((int(pt[0]), int(pt[1])))

        # Vectorized containers detection ('(')
        container_pts = np.argwhere(chars == ord("("))
        for pt in container_pts:
            lvl.containers.add((int(pt[0]), int(pt[1])))

        # Vectorized gold detection ('$')
        gold_pts = np.argwhere(chars == ord("$"))
        for pt in gold_pts:
            lvl.gold_piles.add((int(pt[0]), int(pt[1])))

        if (py, px) in lvl.gold_piles and chars[py, px] != ord("$"):
            lvl.gold_piles.discard((py, px))
            lvl.collected_gold.add((py, px))

        # Vectorized boulder detection ('0')
        boulder_pts = np.argwhere(chars == ord("0"))
        lvl.boulders = {(int(pt[0]), int(pt[1])) for pt in boulder_pts}

        # Vectorized food/corpse detection ('%')
        food_pts = np.argwhere(chars == ord("%"))
        for pt in food_pts:
            fr, fc = int(pt[0]), int(pt[1])
            if glyphs is not None:
                g = int(glyphs[fr, fc])
                if nethack.glyph_is_body(g):
                    # Verified corpse body glyph
                    mon_id = g - nethack.GLYPH_BODY_OFF
                    if 0 <= mon_id < 381:
                        pm = nethack.permonst(mon_id)
                        mname = pm.mname.lower()
                        # Fatal stoning corpses: NEVER touch or eat!
                        if "cockatrice" in mname or "chickatrice" in mname or "medusa" in mname:
                            lvl.consumed_corpses.add((fr, fc))
                            lvl.floor_corpses.pop((fr, fc), None)
                            lvl.conveyor_corpses.pop((fr, fc), None)
                            continue
                        # Lichen corpses: NEVER rot in NetHack! (Safe indefinitely)
                        if "lichen" in mname:
                            if (fr, fc) not in lvl.floor_corpses and (fr, fc) not in lvl.consumed_corpses:
                                lvl.floor_corpses[(fr, fc)] = blstats.turn
                            continue
                        # Poison resistance conveyor corpses: track to prioritize consumption!
                        if any(c in mname for c in ("killer bee", "soldier ant", "cave spider", "centipede", "giant beetle", "naga hatchling", "shrieker", "blue jelly", "quivering blob")):
                            if (fr, fc) not in lvl.consumed_corpses:
                                lvl.conveyor_corpses[(fr, fc)] = (blstats.turn, mname)
                    was_mapped = bool(lvl.mapped[fr, fc])
                    is_kill_msg = any(k in message.lower() for k in ("kill", "destroy", "dies", "smit", "vaporiz", "fell", "hit!"))
                    if (fr, fc) not in lvl.floor_corpses and (fr, fc) not in lvl.consumed_corpses:
                        if was_mapped or is_kill_msg:
                            lvl.floor_corpses[(fr, fc)] = blstats.turn
                        else:
                            # Pre-existing ancient corpse from level generation (tainted/rotten)
                            lvl.consumed_corpses.add((fr, fc))
                else:
                    # Non-corpse food (rations, lembas, pancakes, fruit): 100% safe, never rots!
                    if (fr, fc) not in lvl.collected_food:
                        lvl.floor_food.add((fr, fc))
            else:
                # Fallback when glyphs are unavailable (e.g. synthetic unit tests)
                was_mapped = bool(lvl.mapped[fr, fc])
                is_kill_msg = any(k in message.lower() for k in ("kill", "destroy", "dies", "smit", "vaporiz", "fell", "hit!"))
                if (fr, fc) not in lvl.floor_corpses and (fr, fc) not in lvl.consumed_corpses:
                    if was_mapped or is_kill_msg:
                        lvl.floor_corpses[(fr, fc)] = blstats.turn
                    else:
                        lvl.consumed_corpses.add((fr, fc))

        if (py, px) in lvl.floor_corpses and chars[py, px] != ord("%"):
            lvl.floor_corpses.pop((py, px), None)
            lvl.conveyor_corpses.pop((py, px), None)
            lvl.consumed_corpses.add((py, px))

        if (py, px) in lvl.floor_food and chars[py, px] != ord("%"):
            lvl.floor_food.discard((py, px))
            lvl.collected_food.add((py, px))

        # Prune rotten corpses (> 25 turns old)
        rotten_corpses = [pos for pos, spawn_turn in lvl.floor_corpses.items() if blstats.turn - spawn_turn > 25]
        for r_pos in rotten_corpses:
            lvl.floor_corpses.pop(r_pos, None)
            lvl.conveyor_corpses.pop(r_pos, None)

        # Discard fountains that dried up if tile is visible and not '{' and not '@'
        to_remove = set()
        for fr, fc in lvl.fountains:
            if (fr, fc) != (py, px) and chars[fr, fc] != ord("{") and chars[fr, fc] != 0 and chars[fr, fc] != ord(" "):
                to_remove.add((fr, fc))
        lvl.fountains -= to_remove

        lvl.visited[py, px] += 1
        return lvl

    def _compute_doorway_mask(self, lvl: LevelMap) -> np.ndarray:
        door_mask = np.zeros(lvl.walkable.shape, dtype=bool)
        for dr, dc in lvl.doors:
            if 0 <= dr < door_mask.shape[0] and 0 <= dc < door_mask.shape[1]:
                door_mask[dr, dc] = True
        return door_mask

    def _get_sokoban_walkable_and_costs(
        self,
        chars: np.ndarray,
        lvl: LevelMap,
        hazard_costs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        In Sokoban (dnum == 3), allows pathfinding through pushable boulders.
        A boulder at (br, bc) is pushable if there exists an adjacent cardinal direction
        where the landing tile beyond is open floor (.) or pit (^).
        """
        soko_walkable = lvl.walkable.copy()
        soko_costs = hazard_costs.copy()

        boulder_pts = np.argwhere(chars == ord("0"))
        for pt in boulder_pts:
            br, bc = int(pt[0]), int(pt[1])
            is_pushable = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                lr, lc = br + dr, bc + dc
                if 0 <= lr < chars.shape[0] and 0 <= lc < chars.shape[1]:
                    if chars[lr, lc] in (ord("."), ord("#"), ord("^")) and chars[lr, lc] != ord("0"):
                        is_pushable = True
                        break
            if is_pushable:
                soko_walkable[br, bc] = True
                soko_costs[br, bc] += self.cfg.navigation.boulder_push_cost

        return soko_walkable, soko_costs



    def evaluate_navigation_turn(
        self,
        chars: np.ndarray,
        blstats: BottomLineStats,
        force_descend: bool = False,
        target_fountain: bool = False,
        target_altar: bool = False,
        message: str = "",
        inv_tracker: Any = None,
        glyphs: np.ndarray | None = None,
        has_poison_res: bool = False,
        long_sword_slot: str | None = None,
        dungeon_graph: Any = None,
        macro_director: Any = None,
    ) -> Task:
        """
        Calculates the next navigation action:
        1. On stairs down check: descend if ready.
        2. Adjacent closed door interaction (OPEN, APPLY tool lockpick, defer KICK until explored).
        3. Excalibur fountain navigation (XL >= 5, long sword, no Excalibur).
        3.5. Altar navigation (BUC testing / corpse offering).
        4. Target nearest unvisited reachable tile (rooms, corridors, doors).
        5. Stairs down navigation (when floor explored or turns high).
        6. Target unopened door (+) on the level.
        7. Corridor dead-end secret door search burst (up to 20 searches).
        8. Perimeter wall secret door search.
        9. Fallback: least visited step or search.
        """
        lvl = self.update_map(chars, blstats, message=message, glyphs=glyphs, dungeon_graph=dungeon_graph)
        py, px = blstats.y, blstats.x
        lvl.walkable[py, px] = True

        # Check if hero took damage since previous turn or received hit message:
        # immediately abort secret door search burst to avoid being beaten to death by unseen attackers
        took_damage = (self._prev_hp > 0 and blstats.hp < self._prev_hp) or (
            any(w in message.lower() for w in ("hits!", "bites!", "stings!", "strikes!", "slams!"))
        )
        if took_damage:
            self._current_search_spot = None
            self._current_search_burst = 0
        self._prev_hp = blstats.hp

        doorway_mask = self._compute_doorway_mask(lvl)
        hazard_costs = self._compute_hazard_costs(chars, lvl, glyphs=glyphs)

        # 0.9 Macro Director navigation directive (Phase 2 wiring: strategic branch steering)
        nav_directive = None
        if macro_director is not None and hasattr(macro_director, "get_navigation_directive"):
            nav_directive = macro_director.get_navigation_directive(blstats, dungeon_graph)

        # 1. On stairs down check:
        if (py, px) == lvl.stairs_down and (py, px) not in lvl.blocked_tiles:
            defer_for_poi = (
                ((target_fountain and lvl.fountains) or (target_altar and lvl.altars))
                and lvl.turns_spent < 50
                and blstats.hunger_state < HungerState.HUNGRY
            )
            defer_for_macro = (
                macro_director is not None
                and macro_director.should_defer_stairs_for_farming(
                    blstats,
                    int((lvl.walkable & (lvl.visited == 0)).sum()),
                    lvl.turns_spent,
                )
            )
            can_descend = HTNGuards.can_descend(blstats, has_poison_res=has_poison_res, config=self.cfg)
            if not defer_for_poi and not defer_for_macro and can_descend:
                return Task("DESCEND", is_primitive=True)
            # Rest-on-stairs interlock (Phase 2): if HP is below the descent threshold but not
            # critical, WAIT here to regenerate instead of stepping off the stairs and wandering
            # away (the historic 1,000+ turn floor-stall loop)
            if HTNGuards.should_rest_on_stairs(blstats, on_stairs_down=True, config=self.cfg):
                return Task("WAIT", is_primitive=True)

        # 1.00 Step off blocked staircase / tile (e.g. freshly retreated from Gnomish Mines)
        if (py, px) in lvl.blocked_tiles:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = py + dr, px + dc
                if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                    if lvl.walkable[nr, nc] and (nr, nc) not in lvl.blocked_tiles:
                        is_mon = False
                        if glyphs is not None:
                            g = int(glyphs[nr, nc])
                            if nethack.glyph_is_monster(g) and not nethack.glyph_is_pet(g):
                                is_mon = True
                        if chars[nr, nc] == ord("@") and (nr, nc) != (py, px):
                            is_mon = True
                        if not is_mon:
                            return self._step_or_open(
                                py, px, PathNode(row=nr, col=nc),
                                chars, lvl, message=message, glyphs=glyphs, inv_tracker=inv_tracker
                            )

        # 1.01 Safe retreat from Gnomish Mines if under-leveled (XL < 6 without poison resistance)
        if blstats.dungeon_number == 2 and blstats.experience_level < self.cfg.descent.mines_retreat_xl and not has_poison_res:
            if lvl.stairs_up is None or lvl.stairs_up == (py, px) or lvl.turns_spent <= 2:
                lvl.stairs_up = (py, px)
                lvl.all_stairs_up.add((py, px))
                return Task("ASCEND", is_primitive=True)
            elif lvl.stairs_up:
                path = self.astar.find_path(
                    (py, px),
                    lvl.stairs_up,
                    lvl.walkable,
                    hazard_costs=hazard_costs,
                    doorway_mask=doorway_mask,
                )
                if path:
                    return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 1.015 General Mines retreat policy (Phase 2): Gnomish Mines are a dead-end branch.
        # Ascend back to the Dungeons of Doom when: (a) MacroDirector policy directs it
        # (Mine's End reached, temple donations complete, fully-explored dead-end node),
        # (b) the level is fully explored with no downstairs, or (c) lingering > 50 turns
        # with no downstairs discovered.
        if blstats.dungeon_number == 2:
            mines_retreat = nav_directive == "ASCEND_FROM_MINES"
            if not mines_retreat and dungeon_graph is not None:
                mines_retreat = dungeon_graph.should_ascend_from_mines(blstats, has_light=False, has_infravision=False)
            if not mines_retreat and lvl.stairs_down is None:
                unvisited_left = int((lvl.mapped & lvl.walkable & (lvl.visited == 0)).sum())
                if unvisited_left == 0 and lvl.turns_spent > self.cfg.descent.mines_linger_turns:
                    mines_retreat = True
            if mines_retreat:
                if lvl.stairs_up is None or lvl.stairs_up == (py, px) or lvl.turns_spent <= 2:
                    lvl.stairs_up = (py, px)
                    lvl.all_stairs_up.add((py, px))
                    return Task("ASCEND", is_primitive=True)
                elif lvl.stairs_up:
                    path = self.astar.find_path(
                        (py, px),
                        lvl.stairs_up,
                        lvl.walkable,
                        hazard_costs=hazard_costs,
                        doorway_mask=doorway_mask,
                    )
                    if path:
                        return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 1.02 Sokoban Branch Progression (dnum == 3): In Sokoban, '<' ascends deeper towards the prize
        if blstats.dungeon_number == 3:
            soko_lvl = getattr(blstats, "level_number", 1)
            # Prize collected (Phase 2 fix): exit Sokoban by descending ('>' chain) back to the
            # Dungeons of Doom. Prevents the historic floor 3/4 oscillation loop.
            if nav_directive == "EXIT_SOKOBAN":
                if lvl.stairs_down == (py, px) and (py, px) not in lvl.blocked_tiles:
                    return Task("DESCEND", is_primitive=True)
                elif lvl.stairs_down:
                    path = self.astar.find_path(
                        (py, px),
                        lvl.stairs_down,
                        lvl.walkable,
                        hazard_costs=hazard_costs,
                        doorway_mask=doorway_mask,
                    )
                    if not path:
                        soko_w, soko_c = self._get_sokoban_walkable_and_costs(chars, lvl, hazard_costs)
                        path = self.astar.find_path(
                            (py, px),
                            lvl.stairs_down,
                            soko_w,
                            hazard_costs=soko_c,
                            doorway_mask=doorway_mask,
                        )
                    if path:
                        return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs, inv_tracker=inv_tracker)
            if soko_lvl < self.cfg.descent.sokoban_prize_floor:
                if lvl.stairs_up == (py, px):
                    return Task("ASCEND", is_primitive=True)
                elif lvl.stairs_up:
                    path = self.astar.find_path(
                        (py, px),
                        lvl.stairs_up,
                        lvl.walkable,
                        hazard_costs=hazard_costs,
                        doorway_mask=doorway_mask,
                    )
                    if not path:
                        soko_w, soko_c = self._get_sokoban_walkable_and_costs(chars, lvl, hazard_costs)
                        path = self.astar.find_path(
                            (py, px),
                            lvl.stairs_up,
                            soko_w,
                            hazard_costs=soko_c,
                            doorway_mask=doorway_mask,
                        )
                    if path:
                        return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs, inv_tracker=inv_tracker)
            else:
                # Top floor: descend back down to Dungeons of Doom after exploration
                if lvl.stairs_down == (py, px) and lvl.turns_spent >= self.cfg.descent.sokoban_exit_min_turns:
                    return Task("DESCEND", is_primitive=True)
                elif lvl.stairs_down and lvl.turns_spent >= self.cfg.descent.sokoban_exit_min_turns:
                    path = self.astar.find_path(
                        (py, px),
                        lvl.stairs_down,
                        lvl.walkable,
                        hazard_costs=hazard_costs,
                        doorway_mask=doorway_mask,
                    )
                    if not path:
                        soko_w, soko_c = self._get_sokoban_walkable_and_costs(chars, lvl, hazard_costs)
                        path = self.astar.find_path(
                            (py, px),
                            lvl.stairs_down,
                            soko_w,
                            hazard_costs=soko_c,
                            doorway_mask=doorway_mask,
                        )
                    if path:
                        return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs, inv_tracker=inv_tracker)

        # 1.03 Sokoban Branch Entry Hunt (macro directive): route to '<' upstairs on DL 5-9.
        # Sokoban (dnum 3) branches off main dungeon levels 5-9 via upward staircases. Stairs
        # already recorded as leading to the Dungeons of Doom are ruled out, and the agent
        # descends to hunt the branch from other levels instead of ping-ponging.
        if nav_directive == "ENTER_SOKOBAN" and blstats.dungeon_number == 0:
            cur_node = (
                dungeon_graph.nodes.get((int(blstats.dungeon_number), int(blstats.dlevel)))
                if dungeon_graph is not None
                else None
            )
            su = lvl.stairs_up
            ruled_out = (
                su is not None
                and cur_node is not None
                and cur_node.stair_connections.get(su) is not None
                and cur_node.stair_connections.get(su)[0] == 0
            )
            if su is not None and su not in lvl.blocked_tiles and not ruled_out:
                if (py, px) == su:
                    return Task("ASCEND", is_primitive=True)
                path = self.astar.find_path(
                    (py, px),
                    su,
                    lvl.walkable,
                    hazard_costs=hazard_costs,
                    doorway_mask=doorway_mask,
                )
                if path:
                    return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs, inv_tracker=inv_tracker)
            # This level's upstairs cannot reach Sokoban: descend to hunt the branch entry
            # from other dungeon levels.
            if lvl.stairs_down and lvl.stairs_down not in lvl.blocked_tiles:
                if (py, px) == lvl.stairs_down and (py, px) not in lvl.blocked_tiles:
                    return Task("DESCEND", is_primitive=True)
                path = self.astar.find_path(
                    (py, px),
                    lvl.stairs_down,
                    lvl.walkable,
                    hazard_costs=hazard_costs,
                    doorway_mask=doorway_mask,
                )
                if path:
                    return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs, inv_tracker=inv_tracker)

        # 1.04 TACTICAL HP RESTING (Phase 3): when wounded (HP < 60%) with no hostiles
        # nearby, WAIT in place to regenerate until HP >= 85% instead of wandering into
        # deeper danger. Abort immediately on damage taken, a nearby monster appearing,
        # or hunger kicking in (resting burns nutrition).
        if not took_damage and blstats.hunger_state < HungerState.HUNGRY:
            visible_hostile_near = False
            if glyphs is not None:
                rad = self.cfg.survival.rest_hostile_radius
                for rr in range(max(0, py - rad), min(21, py + rad + 1)):
                    for cc in range(max(0, px - rad), min(79, px + rad + 1)):
                        g = int(glyphs[rr, cc])
                        if nethack.glyph_is_monster(g) and not nethack.glyph_is_pet(g):
                            visible_hostile_near = True
                            break
                if visible_hostile_near:
                    self._rest_active = False
            hp_frac = blstats.hp / max(1, blstats.max_hp)
            if self._rest_active:
                if visible_hostile_near or blstats.hp >= int(blstats.max_hp * self.cfg.survival.rest_until_frac):
                    self._rest_active = False
                else:
                    return Task("WAIT", is_primitive=True)
            elif blstats.hp < int(blstats.max_hp * self.cfg.survival.rest_below_frac) and not visible_hostile_near:
                self._rest_active = True
                return Task("WAIT", is_primitive=True)

        # 1.035 Minetown Temple Routing (macro directive): descend into the Gnomish Mines
        # branch to reach Minetown (Mines DL 3-5) for divine protection donations.
        if nav_directive == "GOTO_MINETOWN" and blstats.dungeon_number == 0:
            cur_node = (
                dungeon_graph.nodes.get((int(blstats.dungeon_number), int(blstats.dlevel)))
                if dungeon_graph is not None
                else None
            )
            mines_stairs: list[tuple[int, int]] = []
            if cur_node is not None:
                mines_stairs = [
                    sp
                    for sp, tgt in cur_node.stair_connections.items()
                    if tgt[0] == 2 and sp not in lvl.blocked_tiles
                ]
            if mines_stairs:
                target = mines_stairs[0]
                if (py, px) == target and (py, px) not in lvl.blocked_tiles:
                    return Task("DESCEND", is_primitive=True)
                path = self.astar.find_path(
                    (py, px),
                    target,
                    lvl.walkable,
                    hazard_costs=hazard_costs,
                    doorway_mask=doorway_mask,
                )
                if path:
                    return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs, inv_tracker=inv_tracker)
            # Fall through: normal exploration will discover the Mines branch stairs

        # 1.06 Water Crossing (Phase 4): activate levitation when standing adjacent to a
        # wide water body (pools, moats, Medusa's island). Isolated '}' tiles are sinks
        # (harmless floor features), so require >= 2 water tiles before levitation prep.
        if not blstats.is_levitating and not blstats.is_flying and inv_tracker is not None:
            water_tiles = [
                (py + dr, px + dc)
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if 0 <= py + dr < 21 and 0 <= px + dc < 79 and chars[py + dr, px + dc] == ord("}")
            ]
            if len(water_tiles) >= self.cfg.navigation.water_body_min_tiles:
                lev_tool = self._find_levitation_tool(inv_tracker)
                if lev_tool is not None:
                    kind, slot = lev_tool
                    if kind == "ZAP":
                        # Freeze an adjacent water tile with a wand of cold
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            nr, nc = py + dr, px + dc
                            if 0 <= nr < 21 and 0 <= nc < 79 and chars[nr, nc] == ord("}"):
                                return Task("ZAP", is_primitive=True, args={"slot": slot, "delta": (dr, dc)})
                    return Task(kind, is_primitive=True, args={"slot": slot})

        # 1.05 Excalibur Fountain Dipping: Standing on a fountain with a long sword
        if target_fountain and (py, px) in lvl.fountains and long_sword_slot:
            return Task("DIP", is_primitive=True, args={"slot": long_sword_slot})

        # 1.1 Throne Interaction: DISABLED. NetHack 3.6.6 throne effects include electric
        # shocks ("You get zapped!", ~6d6) and hostile summonings — repeatedly fatal at
        # Valkyrie power levels. The throne tile is recorded as visited and walked over.
        if (py, px) in lvl.thrones:
            lvl.visited_thrones.add((py, px))

        # 1.2 Container Looting: Loot chest or box on current floor tile
        if (py, px) in lvl.containers and (py, px) not in lvl.looted_containers:
            lvl.looted_containers.add((py, px))
            return Task("LOOT", is_primitive=True, args={"delta": (0, 0)})

        # 1.3 Gold Pickup on current floor tile
        if (py, px) in lvl.gold_piles or chars[py, px] == ord("$"):
            lvl.collected_gold.add((py, px))
            lvl.gold_piles.discard((py, px))
            return Task("PICKUP", is_primitive=True)

        # 2. Check immediately adjacent closed doors
        unlocking_tool = self._find_unlocking_tool(inv_tracker)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = py + dr, px + dc
            if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                if (
                    chars[nr, nc] == ord("+")
                    and (nr, nc) not in lvl.blocked_tiles
                    and (nr, nc) not in lvl.shop_doors
                ):
                    attempts = lvl.door_attempts.get((nr, nc), 0)
                    if attempts < self.cfg.navigation.door_attempt_cap:
                        lvl.door_attempts[(nr, nc)] = attempts + 1
                        if attempts == 0:
                            return Task("OPEN", is_primitive=True, args={"delta": (dr, dc)})
                        
                        # Door is locked (attempts >= 1):
                        # If player has an unlocking tool, apply it
                        if unlocking_tool is not None and attempts < 10:
                            return Task("APPLY", is_primitive=True, args={"slot": unlocking_tool, "delta": (dr, dc)})

                        # Guard shop doors: on Depth >= 2, never kick locked doors if alternatives exist!
                        is_shop = (
                            (nr, nc) in lvl.shop_doors
                            or any(w in message.lower() for w in ("shop", "store", "closed for inventory", "how dare you"))
                        )
                        has_peaceful_human = (chars == ord("@")).sum() > 1
                        # Reachability-aware alternatives (stall fix): unvisited tiles behind
                        # the locked door itself must NOT count as alternatives, otherwise the
                        # door is never kicked and the stairs behind it stay hidden forever.
                        dist_grid_l = self._compute_bfs_distances(py, px, lvl)
                        reach_unvis = bool(np.any((dist_grid_l >= 0) & (lvl.walkable & (lvl.visited == 0))))
                        has_alternatives = (lvl.stairs_down is not None) or reach_unvis
                        if is_shop or has_peaceful_human or (lvl.depth >= 2 and has_alternatives):
                            # Only genuine shop doors are recorded in shop_doors; doors blocked
                            # merely for depth/alternatives stay siegable later (stall fix)
                            if is_shop or has_peaceful_human:
                                lvl.shop_doors.add((nr, nc))
                            lvl.blocked_tiles.add((nr, nc))
                            lvl.walkable[nr, nc] = False
                            continue

                        if "hurt your leg" in message.lower() or "hurt your foot" in message.lower():
                            return Task("WAIT", is_primitive=True)

                        # Only kick if on Depth 1 or no alternative path/stair exists to progress
                        return Task("KICK", is_primitive=True, args={"delta": (dr, dc)})
                    else:
                        # Mandatory door revival (stall fix): no stairs known and no reachable
                        # frontier left — this locked door guards the only progress route.
                        # Keep kicking (NetHack kicks eventually break the door) instead of
                        # permanently blocking the route and stalling for thousands of turns.
                        dist_grid_l = self._compute_bfs_distances(py, px, lvl)
                        reach_unvis = bool(np.any((dist_grid_l >= 0) & (lvl.walkable & (lvl.visited == 0))))
                        if (
                            lvl.stairs_down is None
                            and not reach_unvis
                            and (chars == ord("@")).sum() <= 1
                            and (nr, nc) not in lvl.shop_doors
                        ):
                            lvl.door_attempts[(nr, nc)] = 1
                            return Task("KICK", is_primitive=True, args={"delta": (dr, dc)})
                        lvl.blocked_tiles.add((nr, nc))
                        lvl.walkable[nr, nc] = False

        # 2.1 Align orthogonally to diagonally adjacent closed doors
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = py + dr, px + dc
            if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                if (
                    chars[nr, nc] == ord("+")
                    and (nr, nc) not in lvl.blocked_tiles
                    and (nr, nc) not in lvl.shop_doors
                    and lvl.door_attempts.get((nr, nc), 0) < 30
                ):
                    if 0 <= py + dr < lvl.walkable.shape[0] and lvl.walkable[py + dr, px]:
                        cand_pos = (py + dr, px)
                        is_mon = (glyphs is not None and nethack.glyph_is_monster(int(glyphs[cand_pos]))) or chars[cand_pos] == ord("@")
                        if not is_mon and cand_pos not in lvl.blocked_tiles:
                            self.last_step_delta = (dr, 0)
                            return Task("STEP", is_primitive=True, args={"delta": (dr, 0)})
                    if 0 <= px + dc < lvl.walkable.shape[1] and lvl.walkable[py, px + dc]:
                        cand_pos = (py, px + dc)
                        is_mon = (glyphs is not None and nethack.glyph_is_monster(int(glyphs[cand_pos]))) or chars[cand_pos] == ord("@")
                        if not is_mon and cand_pos not in lvl.blocked_tiles:
                            self.last_step_delta = (0, dc)
                            return Task("STEP", is_primitive=True, args={"delta": (0, dc)})

        # 2.2 Pet Swapping Interlock: If we just swapped places with our pet in a corridor,
        # pause with WAIT for 1 turn to let the pet step forward and clear the path
        msg_l = message.lower()
        if "swap places with your" in msg_l or "is in the way" in msg_l:
            return Task("WAIT", is_primitive=True)

        # 2.5 Disarm immediately adjacent discovered traps
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = py + dr, px + dc
            if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                if (nr, nc) in lvl.traps and (nr, nc) not in lvl.disarmed_traps:
                    lvl.disarmed_traps.add((nr, nc))
                    return Task("UNTRAP", is_primitive=True, args={"delta": (dr, dc)})

        # 2.85 ACTIVE POISON RESISTANCE FARMING (Phase 3): fresh conveyor corpses (killer bee,
        # soldier ant, cave spider, ...) convey permanent poison resistance — the single most
        # critical mid-game intrinsic. Without it, DL 8+ is a death sentence. Interrupt even
        # descent to grab a fresh conveyor within reach.
        if not has_poison_res and lvl.conveyor_corpses:
            fresh_conveyors = [
                pos for pos, (sturn, mname) in lvl.conveyor_corpses.items()
                if blstats.turn - sturn <= 25 and pos != (py, px) and pos not in lvl.consumed_corpses
            ]
            if fresh_conveyors:
                fresh_conveyors.sort(key=lambda c: abs(c[0] - py) + abs(c[1] - px))
                closest_conv = fresh_conveyors[0]
                if abs(closest_conv[0] - py) + abs(closest_conv[1] - px) <= self.cfg.nutrition.conveyor_radius:
                    path = self.astar.find_path(
                        (py, px),
                        closest_conv,
                        lvl.walkable,
                        hazard_costs=hazard_costs,
                        doorway_mask=doorway_mask,
                    )
                    if path:
                        return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 2.9 AGGRESSIVE DESCENT (Phase 2): stair descent is evaluated BEFORE POI routing,
        # container looting, gold collection, and food collection so the agent stops
        # stalling on shallow floors for thousands of turns.
        unvisited_count = int((lvl.walkable & (lvl.visited == 0)).sum())
        defer_for_poi = (
            ((target_fountain and lvl.fountains) or (target_altar and lvl.altars))
            and lvl.turns_spent < self.cfg.navigation.poi_defer_turns
            and blstats.hunger_state < HungerState.HUNGRY
        )
        defer_for_macro = (
            macro_director is not None
            and macro_director.should_defer_stairs_for_farming(blstats, unvisited_count, lvl.turns_spent)
        )
        should_descend = (
            force_descend
            or (
                (nav_directive == "DESCEND" or (not defer_for_poi and not defer_for_macro))
                and HTNGuards.should_descend(
                    blstats=blstats,
                    stairs_down_known=lvl.stairs_down is not None,
                    unvisited_count=unvisited_count,
                    turns_spent=lvl.turns_spent,
                    has_poison_res=has_poison_res,
                    config=self.cfg,
                )
            )
        )
        # 2.95 Starvation bridge: when WEAK or worse with a fresh corpse/food within
        # reach, eat it BEFORE descending — one corpse buys ~600 nutrition (survival
        # bridge across the ~850-turn prayer cooldown window).
        if blstats.hunger_state >= HungerState.WEAK and should_descend and lvl.stairs_down:
            food_targets = [
                pos for pos in lvl.floor_food
                if pos != (py, px) and pos not in lvl.collected_food
            ]
            food_targets.extend([
                pos for pos, spawn_turn in lvl.floor_corpses.items()
                if blstats.turn - spawn_turn <= 25 and pos != (py, px) and pos not in lvl.consumed_corpses
            ])
            if food_targets:
                food_targets.sort(key=lambda f: abs(f[0] - py) + abs(f[1] - px))
                closest_food = food_targets[0]
                if abs(closest_food[0] - py) + abs(closest_food[1] - px) <= self.cfg.nutrition.food_bridge_radius:
                    path = self.astar.find_path(
                        (py, px),
                        closest_food,
                        lvl.walkable,
                        hazard_costs=hazard_costs,
                        doorway_mask=doorway_mask,
                    )
                    if path:
                        return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        if should_descend and lvl.stairs_down:
            stairs_task = self.step_towards_stairs(
                py, px, lvl, chars, hazard_costs=hazard_costs, doorway_mask=doorway_mask, message=message, glyphs=glyphs
            )
            if stairs_task is not None:
                return stairs_task

        # 3. Excalibur Fountain Navigation (Valkyrie XL >= 5, long sword, no Excalibur)
        if target_fountain and lvl.fountains:
            sorted_fountains = sorted(lvl.fountains, key=lambda f: abs(f[0] - py) + abs(f[1] - px))
            for f_pos in sorted_fountains:
                if f_pos == (py, px):
                    continue
                path = self.astar.find_path(
                    (py, px),
                    f_pos,
                    lvl.walkable,
                    hazard_costs=hazard_costs,
                    doorway_mask=doorway_mask,
                )
                if path:
                    return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 3.5. Altar Navigation (when hero has untested items to BUC test or corpse to offer)
        if target_altar and lvl.altars:
            sorted_altars = sorted(lvl.altars, key=lambda a: abs(a[0] - py) + abs(a[1] - px))
            for a_pos in sorted_altars:
                if a_pos == (py, px):
                    continue
                path = self.astar.find_path(
                    (py, px),
                    a_pos,
                    lvl.walkable,
                    hazard_costs=hazard_costs,
                    doorway_mask=doorway_mask,
                )
                if path:
                    return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 4.1 Nearby Container Looting: Path to unlooted containers within distance <= 10
        unlooted_containers = [c for c in lvl.containers if c not in lvl.looted_containers and c != (py, px)]
        if unlooted_containers:
            unlooted_containers.sort(key=lambda c: abs(c[0] - py) + abs(c[1] - px))
            closest_container = unlooted_containers[0]
            if abs(closest_container[0] - py) + abs(closest_container[1] - px) <= self.cfg.navigation.container_radius:
                path = self.astar.find_path(
                    (py, px),
                    closest_container,
                    lvl.walkable,
                    hazard_costs=hazard_costs,
                    doorway_mask=doorway_mask,
                )
                if path:
                    return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 4.2 Nearby Gold Collection: Path to uncollected gold within distance <= 12
        uncollected_gold = [g for g in lvl.gold_piles if g not in lvl.collected_gold and g != (py, px)]
        if uncollected_gold and blstats.encumbrance == 0:
            uncollected_gold.sort(key=lambda g: abs(g[0] - py) + abs(g[1] - px))
            closest_gold = uncollected_gold[0]
            if abs(closest_gold[0] - py) + abs(closest_gold[1] - px) <= self.cfg.navigation.gold_radius:
                path = self.astar.find_path(
                    (py, px),
                    closest_gold,
                    lvl.walkable,
                    hazard_costs=hazard_costs,
                    doorway_mask=doorway_mask,
                )
                if path:
                    return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 4.3 Nearby Floor Food / Fresh Floor Corpse Collection: Path to food within distance <= 8
        if blstats.hunger_state != HungerState.SATIATED:
            # 1. Non-corpse food (rations, lembas, pancakes, fruit): 100% safe, never rots!
            safe_targets = [
                pos for pos in lvl.floor_food
                if pos != (py, px) and pos not in lvl.collected_food
            ]
            # 2. Verified fresh corpses (<= 25 turns old)
            if lvl.floor_corpses:
                safe_targets.extend([
                    pos for pos, spawn_turn in lvl.floor_corpses.items()
                    if blstats.turn - spawn_turn <= 25 and pos != (py, px) and pos not in lvl.consumed_corpses
                ])
            if safe_targets:
                safe_targets.sort(key=lambda c: abs(c[0] - py) + abs(c[1] - px))
                closest_target = safe_targets[0]
                # Starvation fix: while hungry or worse, chase fresh corpses/food across the
                # whole mapped level — kills leave edible corpses exactly when needed
                max_food_dist = self.cfg.nutrition.food_radius_hungry if blstats.hunger_state >= HungerState.HUNGRY else self.cfg.nutrition.food_radius
                if abs(closest_target[0] - py) + abs(closest_target[1] - px) <= max_food_dist:
                    path = self.astar.find_path(
                        (py, px),
                        closest_target,
                        lvl.walkable,
                        hazard_costs=hazard_costs,
                        doorway_mask=doorway_mask,
                    )
                    if path:
                        return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 4.5 Target nearest unvisited reachable tile (rooms, corridors, doors)
        unvisited_target = self.find_nearest_unvisited(py, px, lvl)
        if unvisited_target is not None:
            path = self.astar.find_path(
                (py, px),
                unvisited_target,
                lvl.walkable,
                hazard_costs=hazard_costs,
                doorway_mask=doorway_mask,
            )
            if path:
                return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 5. Path to stairs down if known
        if lvl.stairs_down:
            path = self.astar.find_path(
                (py, px),
                lvl.stairs_down,
                lvl.walkable,
                hazard_costs=hazard_costs,
                doorway_mask=doorway_mask,
            )
            if path:
                return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 6. Target unopened door (+) on the level
        door_mask = (chars == ord("+")) & (lvl.visited == 0)
        unopened_door_pts = np.argwhere(door_mask)
        if len(unopened_door_pts) > 0:
            dists = np.abs(unopened_door_pts[:, 0] - py) + np.abs(unopened_door_pts[:, 1] - px)
            sorted_indices = np.argsort(dists)
            for idx in sorted_indices:
                door_pos = (int(unopened_door_pts[idx, 0]), int(unopened_door_pts[idx, 1]))
                path = self.astar.find_path(
                    (py, px),
                    door_pos,
                    lvl.walkable,
                    hazard_costs=hazard_costs,
                    doorway_mask=doorway_mask,
                )
                if path:
                    return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 7. AutoAscend-Inspired Prioritized Secret Door Search:
        # If stairs down not yet discovered, systematically patrol and search dead ends and perimeter walls
        if not lvl.stairs_down:
            # Adaptive burst cap (Phase 3): maze levels on DL 4+ hide stairs behind secret
            # doors requiring 15-25 searches to reveal.
            burst_cap = self.cfg.search.burst_cap_deep if lvl.depth >= 4 else self.cfg.search.burst_cap_early
            spot_searches = int(lvl.searched[py, px]) + int(lvl.dead_end_searches.get((py, px), 0))
            # Burst search: if currently at the search spot, continue searching up to the cap!
            # (Stall fix: the burst counter is cumulative — re-selecting the same spot must
            # NOT reset it, otherwise a lone dead-end candidate loops forever)
            if self._current_search_spot == (py, px) and spot_searches < burst_cap:
                self._current_search_burst = spot_searches + 1
                lvl.searched[py, px] += 1
                if (py, px) in lvl.corridors:
                    lvl.dead_end_searches[(py, px)] = lvl.dead_end_searches.get((py, px), 0) + 1
                return Task("SEARCH", is_primitive=True)

            dist_grid = self._compute_bfs_distances(py, px, lvl)
            best_search_pos = self.find_best_search_tile(py, px, lvl, chars, dist_grid=dist_grid)
            if best_search_pos is not None:
                if best_search_pos == (py, px):
                    self._current_search_spot = (py, px)
                    # Cumulative burst: continue where this spot's search count left off
                    spot_searches = int(lvl.searched[py, px]) + int(lvl.dead_end_searches.get((py, px), 0))
                    self._current_search_burst = spot_searches + 1
                    lvl.searched[py, px] += 1
                    if (py, px) in lvl.corridors:
                        lvl.dead_end_searches[(py, px)] = lvl.dead_end_searches.get((py, px), 0) + 1
                    return Task("SEARCH", is_primitive=True)
                else:
                    self._current_search_spot = None
                    self._current_search_burst = 0
                    path = self.astar.find_path(
                        (py, px),
                        best_search_pos,
                        lvl.walkable,
                        hazard_costs=hazard_costs,
                        doorway_mask=doorway_mask,
                    )
                    if path:
                        return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)
            else:
                # 7.5 DOOR SIEGE (stall fix): all secret-door search candidates are exhausted,
                # no stairs found, and no reachable frontiers remain. The stairs are almost
                # certainly behind a locked door we previously abandoned. Re-attack the
                # nearest non-shop blocked door (kicks eventually break it open).
                reach_unvis_count = int(((lvl.walkable & (lvl.visited == 0)) & (dist_grid >= 0)).sum())
                if reach_unvis_count == 0:
                    siege_doors = [
                        (r, c)
                        for (r, c) in lvl.blocked_tiles
                        if (r, c) in lvl.doors
                        and (r, c) not in lvl.shop_doors
                        and 0 <= r < 21 and 0 <= c < 79
                    ]
                    siege_doors.sort(key=lambda d: abs(d[0] - py) + abs(d[1] - px))
                    for d_pos in siege_doors:
                        siege_walkable = lvl.walkable.copy()
                        siege_walkable[d_pos] = True
                        path = self.astar.find_path(
                            (py, px),
                            d_pos,
                            siege_walkable,
                            hazard_costs=hazard_costs,
                            doorway_mask=doorway_mask,
                        )
                        if path:
                            return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)

        # 8. Fallback: Step to least visited or search
        self._current_search_spot = None
        self._current_search_burst = 0
        least_visited_delta = self._find_least_visited_step(py, px, lvl, chars, glyphs=glyphs)
        if least_visited_delta is not None:
            return Task("STEP", is_primitive=True, args={"delta": least_visited_delta})

        lvl.searched[py, px] += 1
        return Task("SEARCH", is_primitive=True)

