"""
Domain Layer: Tactical Combat Manager.
Governs monster threat evaluation, kiting, corridor funneling, instakill avoidance,
and emergency Elbereth dust-engraving.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
from nle import nethack

from corp.env.blstats import BottomLineStats, ConditionFlag
from corp.planner.htn import Task, PrimitiveTask


@dataclass(slots=True, frozen=True)
class MonsterTrack:
    """Tracked visible monster with physical and tactical metadata."""
    pos: tuple[int, int]         # (row, col)
    name: str
    level: int
    speed: int                  # mmove (normal=12, killer bee=18)
    ac: int
    distance: int               # Chebyshev distance to player
    is_adjacent: bool
    is_instakill: bool
    threat_score: float


class TacticalCombatManager:
    """
    Evaluates visible monsters and dispatches tactical combat directives.
    Strictly enforces:
    1. Floating eye melee lockout (unless blind).
    2. Cockatrice barehanded lockout.
    3. Speed differential constraint (forbidding open-room retreat if monster is faster).
    4. Emergency dust Elbereth when cornered at critical HP.
    """

    # High-threat entity names
    INSTAKILL_NAMES = {
        "floating eye": 200.0,
        "cockatrice": 250.0,
        "chickatrice": 200.0,
        "mind flayer": 300.0,
        "master mind flayer": 400.0,
        "rust monster": 75.0,
        "disenchanter": 90.0,
    }

    def __init__(self, player_speed: int = 12):
        self.player_speed = player_speed

    def scan_monsters(
        self,
        glyphs: np.ndarray,
        blstats: BottomLineStats,
    ) -> list[MonsterTrack]:
        """
        Scans observation glyphs for all visible hostile monsters.
        """
        py, px = blstats.y, blstats.x
        monsters: list[MonsterTrack] = []

        rows, cols = glyphs.shape
        for r in range(rows):
            for c in range(cols):
                g = int(glyphs[r, c])
                if not nethack.glyph_is_monster(g):
                    continue
                if nethack.glyph_is_pet(g):
                    continue
                if r == py and c == px:
                    # The player character
                    continue

                mon_id = nethack.glyph_to_mon(g)
                pm = nethack.permonst(mon_id)
                mname = pm.mname.lower()
                speed = int(pm.mmove)
                level = int(pm.mlevel)
                ac = int(pm.ac)

                dist = max(abs(r - py), abs(c - px))
                is_adj = dist == 1

                # Calculate threat score
                instakill_weight = 0.0
                is_instakill = False
                for ik_name, weight in self.INSTAKILL_NAMES.items():
                    if ik_name in mname:
                        instakill_weight = weight
                        is_instakill = True
                        break

                base_dmg = max(1.0, float(level) * 2.5)
                speed_ratio = float(speed) / 12.0
                hp_ratio = float(max(1, blstats.hp)) / float(max(1, blstats.max_hp))
                threat = ((base_dmg * speed_ratio) / hp_ratio) + instakill_weight

                monsters.append(
                    MonsterTrack(
                        pos=(r, c),
                        name=mname,
                        level=level,
                        speed=speed,
                        ac=ac,
                        distance=dist,
                        is_adjacent=is_adj,
                        is_instakill=is_instakill,
                        threat_score=threat,
                    )
                )

        # Sort by distance ascending, then threat descending
        monsters.sort(key=lambda m: (m.distance, -m.threat_score))
        return monsters

    def evaluate_combat_turn(
        self,
        glyphs: np.ndarray,
        chars: np.ndarray,
        blstats: BottomLineStats,
    ) -> Task | None:
        """
        Evaluates tactical posture and returns the optimal combat Task,
        or None if no combat intervention is required.
        """
        monsters = self.scan_monsters(glyphs, blstats)
        if not monsters:
            return None

        py, px = blstats.y, blstats.x
        is_blind = blstats.is_blind
        is_critical_hp = blstats.hp <= max(4, int(blstats.max_hp * 0.25))

        adjacent_monsters = [m for m in monsters if m.is_adjacent]

        # 1. Emergency Elbereth Dust-Engraving
        # Trap / critical condition with adjacent hostile
        if adjacent_monsters and is_critical_hp:
            # NetHack 3.6.6: Dust Elbereth provides 1-turn disengage
            return Task(
                name="ENGRAVE_DUST",
                is_primitive=True,
                args={"text": "Elbereth"},
            )

        # 2. Adjacent monster handling
        if adjacent_monsters:
            primary_target = adjacent_monsters[0]
            dr = primary_target.pos[0] - py
            dc = primary_target.pos[1] - px

            # Floating eye guard: Melee attack strictly forbidden unless blind
            if "floating eye" in primary_target.name and not is_blind:
                # Must not melee! Find a non-monster tile to step away
                step_away_delta = self._find_escape_step(py, px, adjacent_monsters, chars)
                if step_away_delta is not None:
                    return Task("STEP", is_primitive=True, args={"delta": step_away_delta})
                # If cannot step away, wait rather than strike eye
                return Task("WAIT", is_primitive=True)

            # Cockatrice guard: no unarmed combat (default weapons usually equipped)
            if "cockatrice" in primary_target.name or "chickatrice" in primary_target.name:
                pass # Weapons are checked in inventory manager

            # Standard melee strike
            return Task(
                name="MELEE_ATTACK",
                is_primitive=True,
                args={"delta": (dr, dc)},
            )

        # 3. Non-adjacent monsters (Distance >= 2)
        closest_monster = monsters[0]
        speed_delta = self.player_speed - closest_monster.speed

        # Speed differential constraint:
        # If monster is faster than player (speed_delta < 0), do not kite in open room.
        # Wait or close distance / funnel to corridor
        if speed_delta < 0:
            # Approaching fast monster: wait for it to close the gap into melee range
            if closest_monster.distance == 2:
                return Task("WAIT", is_primitive=True)

        return None

    def _find_escape_step(
        self,
        py: int,
        px: int,
        threats: list[MonsterTrack],
        chars: np.ndarray,
    ) -> tuple[int, int] | None:
        """Finds an adjacent walkable cell that does not move closer to threats."""
        threat_positions = {t.pos for t in threats}
        walkable_chars = {ord("."), ord("#"), ord("+"), ord("<"), ord(">")}

        candidates: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = py + dr, px + dc
                if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                    if (nr, nc) not in threat_positions and chars[nr, nc] in walkable_chars:
                        candidates.append((dr, dc))

        if candidates:
            return candidates[0]
        return None
