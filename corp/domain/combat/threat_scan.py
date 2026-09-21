"""Extracted from corp/domain/combat_manager.py — behavior-preserving split."""

from dataclasses import dataclass
from typing import Any
import numpy as np
from nle import nethack

from corp.env.blstats import BottomLineStats
from corp.policy.config import PolicyConfig, default_config
from corp.env.blstats import ConditionFlag, HungerState
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




class ThreatScanMixin:
    def scan_monsters(
        self,
        glyphs: np.ndarray,
        blstats: BottomLineStats,
        has_poison_res: bool = False,
    ) -> list[MonsterTrack]:
        """
        Scans observation glyphs for all visible hostile monsters.
        """
        py, px = blstats.y, blstats.x
        monsters: list[MonsterTrack] = []

        # Vectorized glyph range mask for hostile / non-pet monsters:
        # Hostile normal monsters: [GLYPH_MON_OFF, GLYPH_PET_OFF)
        # Detected monsters:       [GLYPH_DETECT_OFF, GLYPH_BODY_OFF)
        # Ridden monsters:         [GLYPH_RIDDEN_OFF, GLYPH_OBJ_OFF)
        mon_mask = (
            ((glyphs >= nethack.GLYPH_MON_OFF) & (glyphs < nethack.GLYPH_PET_OFF)) |
            ((glyphs >= nethack.GLYPH_DETECT_OFF) & (glyphs < nethack.GLYPH_BODY_OFF)) |
            ((glyphs >= nethack.GLYPH_RIDDEN_OFF) & (glyphs < nethack.GLYPH_OBJ_OFF))
        )
        if 0 <= py < glyphs.shape[0] and 0 <= px < glyphs.shape[1]:
            mon_mask[py, px] = False

        # Prune stale peaceful coordinates that no longer contain a monster
        if self.peaceful_positions:
            self.peaceful_positions = {
                p for p in self.peaceful_positions
                if 0 <= p[0] < glyphs.shape[0] and 0 <= p[1] < glyphs.shape[1]
                and nethack.glyph_is_monster(int(glyphs[p[0], p[1]]))
            }

        mon_coords = np.argwhere(mon_mask)
        if len(mon_coords) == 0:
            return monsters

        for idx in range(len(mon_coords)):
            r, c = int(mon_coords[idx, 0]), int(mon_coords[idx, 1])
            g = int(glyphs[r, c])

            mon_id = nethack.glyph_to_mon(g)
            pm = nethack.permonst(mon_id)
            mname = pm.mname.lower()
            if (r, c) in self.peaceful_positions:
                continue
            is_peaceful_type = any(p in mname for p in self.PEACEFUL_NAMES)
            is_known_hostile = any(h in mname for h in self.hostile_names)
            # Peaceful humans (shopkeepers, priests, guards) are NEVER attackable.
            # Peaceful ANIMALS (dogs, cats, ponies) are attackable when they park
            # adjacent and the hero is under attack — an attacking animal is by
            # definition already angry, and the heroes were observed dying helplessly
            # to pony kicks they were forbidden to answer (attrition-death fix).
            # The navigation layer separately forbids bump-attacks (unprovoked).
            under_attack = self.turns_since_damaged < self.cfg.survival.escape_grace_turns and max(abs(r - py), abs(c - px)) == 1
            if is_peaceful_type and not is_known_hostile:
                is_peaceful_human = any(h in mname for h in self.PEACEFUL_HUMANS)
                if is_peaceful_human or not under_attack:
                    self.peaceful_positions.add((r, c))
                    continue

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

            # Lethal poison biters (Giant spiders, queen bees, scorpions) deal fatal poison without resistance
            if not has_poison_res and not is_instakill:
                for lp_name, weight in self.LETHAL_POISON_NAMES.items():
                    if lp_name in mname:
                        if lp_name == "snake" and "garter snake" in mname:
                            continue
                        instakill_weight = weight
                        is_instakill = True
                        break

            base_dmg = max(1.0, float(level) * self.cfg.combat.threat_damage_per_level)
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

