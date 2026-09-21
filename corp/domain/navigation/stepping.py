"""Stepping/door/stairs primitives — extracted from navigation_manager."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from corp.env.blstats import BottomLineStats, HungerState
from nle import nethack
from corp.navigation.astar import GridAStar, PathNode
from corp.navigation.frontier import FrontierExplorer
from corp.planner.htn import Task, PrimitiveTask
from corp.planner.guards import HTNGuards
from corp.domain.navigation.level_map import LevelMap


UNLOCKING_TOOLS = ("skeleton key", "lock pick", "credit card", "key")


class SteppingMixin:
    def _step_or_open(
        self,
        py: int,
        px: int,
        next_node: Any,
        chars: np.ndarray,
        lvl: LevelMap,
        message: str = "",
        glyphs: np.ndarray | None = None,
        inv_tracker: Any = None,
    ) -> Task:
        """Emits OPEN/KICK if next tile is a closed door (+), otherwise STEP."""
        nr, nc = next_node.row, next_node.col
        dr, dc = nr - py, nc - px

        # Interlock: diagonal movement into or out of doorways is illegal in NetHack
        if dr != 0 and dc != 0:
            is_door_transition = (
                chars[nr, nc] in (ord("+"), ord("'"))
                or chars[py, px] in (ord("+"), ord("'"))
                or (nr, nc) in lvl.doors
                or (py, px) in lvl.doors
            )
            if is_door_transition:
                # Decompose diagonal into orthogonal step to approach or leave door safely
                if 0 <= py + dr < lvl.walkable.shape[0] and lvl.walkable[py + dr, px]:
                    dr, dc = dr, 0
                    nr, nc = py + dr, px
                elif 0 <= px + dc < lvl.walkable.shape[1] and lvl.walkable[py, px + dc]:
                    dr, dc = 0, dc
                    nr, nc = py, px + dc
                else:
                    return Task("SEARCH", is_primitive=True)

        if chars[nr, nc] == ord("+"):
            attempts = lvl.door_attempts.get((nr, nc), 0)
            if attempts < self.cfg.navigation.door_attempt_cap:
                lvl.door_attempts[(nr, nc)] = attempts + 1
                if attempts == 0:
                    return Task("OPEN", is_primitive=True, args={"delta": (dr, dc)})

                # Check for unlocking tool (skeleton key, lock pick, credit card)
                if attempts == 1 and getattr(self, "unlock_tool_slot", None):
                    return Task("APPLY", is_primitive=True, args={"slot": self.unlock_tool_slot, "delta": (dr, dc)})
                if attempts == 2 and getattr(self, "unlock_tool_slot", None):
                    return Task("OPEN", is_primitive=True, args={"delta": (dr, dc)})

                # Guard shop doors:
                # 1. If shop door recorded, shop message, level has shops, or multiple @ seen: NEVER KICK!
                # 2. If depth >= 2 (shops common) AND (stairs_down is already found OR unvisited frontiers exist):
                #    NEVER kick a locked door when stairs are known or other rooms exist to explore!
                is_shop = (
                    (nr, nc) in lvl.shop_doors
                    or any(w in message.lower() for w in ("shop", "store", "closed for inventory", "how dare you"))
                )
                has_peaceful_human = (chars == ord("@")).sum() > 1
                has_alternatives = (lvl.stairs_down is not None) or bool(np.any(lvl.walkable & (lvl.visited == 0)))
                if is_shop or has_peaceful_human or (getattr(lvl, "depth", 1) >= 2 and has_alternatives):
                    lvl.shop_doors.add((nr, nc))
                    lvl.blocked_tiles.add((nr, nc))
                    lvl.walkable[nr, nc] = False
                    return Task("SEARCH", is_primitive=True)
                if "hurt your leg" in message.lower() or "hurt your foot" in message.lower():
                    return Task("WAIT", is_primitive=True)
                return Task("KICK", is_primitive=True, args={"delta": (dr, dc)})
            else:
                # Abandon door to prevent infinite loop — unless it is mandatory:
                # no stairs known and no reachable frontier left (the door guards the
                # only progress route), in which case keep kicking to break it open.
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

        # Guard against stepping into any non-pet monster or peaceful entity from navigation
        if glyphs is not None and 0 <= nr < glyphs.shape[0] and 0 <= nc < glyphs.shape[1]:
            g = int(glyphs[nr, nc])
            if nethack.glyph_is_monster(g) and not nethack.glyph_is_pet(g):
                # Never bump-attack non-pet monsters from the navigation layer!
                # If combat attack was desired, CombatManager would have executed it.
                # Stepping into a non-pet attacks them (e.g. peaceful ponies, domestic dogs, gas spores).
                return Task("WAIT", is_primitive=True)

        # Guard against stepping into peaceful humans (@) like shopkeepers, priests, watchmen
        if chars[nr, nc] == ord("@") and (nr, nc) != (py, px):
            lvl.blocked_tiles.add((nr, nc))
            lvl.walkable[nr, nc] = False
            return Task("WAIT", is_primitive=True)

        # Castle Drawbridge Blasting: If closed drawbridge encountered and hero carries wand of striking
        if getattr(lvl, "depth", 1) >= 24 and getattr(lvl, "dnum", 0) == 0:
            msg_l = message.lower()
            if "drawbridge is closed" in msg_l or "portcullis" in msg_l:
                if inv_tracker is not None:
                    active_items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
                    for it in active_items:
                        if "wand of striking" in it.raw_str.lower() and it.buc_state != "CURSED":
                            return Task("ZAP", is_primitive=True, args={"slot": it.current_letter, "delta": (dr, dc)})

        # Stepping into a boulder pushes it forward
        if chars[nr, nc] == ord("0"):
            self.last_step_delta = (dr, dc)
            return Task("STEP", is_primitive=True, args={"delta": (dr, dc)})

        self.last_step_delta = (dr, dc)
        return Task("STEP", is_primitive=True, args={"delta": (dr, dc)})



    def step_towards_stairs(
        self,
        py: int,
        px: int,
        lvl: LevelMap,
        chars: np.ndarray,
        hazard_costs: np.ndarray | None = None,
        doorway_mask: np.ndarray | None = None,
        message: str = "",
        glyphs: np.ndarray | None = None,
    ) -> Task | None:
        """
        Routes directly to stairs down. If already standing on stairs down, issues DESCEND.
        Returns None if stairs down position is unknown or unreachable.
        """
        if lvl.stairs_down is None or lvl.stairs_down in lvl.blocked_tiles:
            return None

        if (py, px) == lvl.stairs_down and (py, px) not in lvl.blocked_tiles:
            return Task("DESCEND", is_primitive=True)

        if hazard_costs is None:
            hazard_costs = self._compute_hazard_costs(chars, lvl, glyphs=glyphs)
        if doorway_mask is None:
            doorway_mask = self._compute_doorway_mask(lvl)

        path = self.astar.find_path(
            (py, px),
            lvl.stairs_down,
            lvl.walkable,
            hazard_costs=hazard_costs,
            doorway_mask=doorway_mask,
        )
        if path:
            return self._step_or_open(py, px, path[0], chars, lvl, message=message, glyphs=glyphs)
        return None

    UNLOCKING_TOOLS = ("skeleton key", "lock pick", "credit card", "key")




    def _find_levitation_tool(self, inv_tracker: Any) -> tuple[str, str] | None:
        """
        Finds the safest available levitation source for water crossing (Phase 4).
        Priority: boots of levitation (wear) > ring of levitation (puton) >
        potion of levitation (quaff, uncursed/blessed only) > wand of cold (freeze water).
        Returns (task_kind, inventory_letter) or None.
        """
        if not inv_tracker:
            return None
        items = (
            inv_tracker.get_active_items()
            if hasattr(inv_tracker, "get_active_items")
            else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
        )
        fallback: tuple[str, str] | None = None
        for item in items:
            desc = item.raw_str.lower()
            is_cursed = desc.startswith("cursed") or " cursed" in desc
            if "levitation boots" in desc or ("levitation" in desc and "boots" in desc):
                if "(being worn)" in desc:
                    continue  # already equipped: levitation condition would be active
                if not is_cursed:
                    return ("WEAR", item.current_letter)
            elif "ring of levitation" in desc:
                if not is_cursed:
                    return ("PUTON", item.current_letter)
            elif "potion of levitation" in desc:
                if not is_cursed:
                    fallback = fallback or ("QUAFF", item.current_letter)
            elif "wand of cold" in desc:
                if not is_cursed:
                    fallback = fallback or ("ZAP", item.current_letter)
        return fallback

    def _find_unlocking_tool(self, inv_tracker: Any) -> str | None:
        if not inv_tracker:
            return None
        items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [
            it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)
        ]
        for item in items:
            desc = item.raw_str.lower()
            if any(k in desc for k in self.UNLOCKING_TOOLS):
                return item.current_letter
        return None

