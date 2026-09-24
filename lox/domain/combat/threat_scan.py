"""Extracted from lox/domain/combat_manager.py — behavior-preserving split."""

from dataclasses import dataclass
from typing import Any
import numpy as np
from nle import nethack

from lox.env.blstats import BottomLineStats
from lox.policy.config import PolicyConfig, default_config
from lox.env.blstats import ConditionFlag, HungerState
from lox.planner.htn import Task, PrimitiveTask


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




# Precomputed static monster metadata LUT (R0 acceleration)
# Avoids per-turn nethack.permonst C-API allocations & string loops
_MAX_MON = getattr(nethack, "NUMMONS", 381) + 32
_PM_NAME: list[str] = [""] * _MAX_MON
_PM_SPEED = np.zeros(_MAX_MON, dtype=np.int16)
_PM_LEVEL = np.zeros(_MAX_MON, dtype=np.int16)
_PM_AC = np.zeros(_MAX_MON, dtype=np.int16)
_PM_IS_PEACEFUL_TYPE = np.zeros(_MAX_MON, dtype=bool)
_PM_IS_PEACEFUL_HUMAN = np.zeros(_MAX_MON, dtype=bool)
_PM_INSTAKILL_WEIGHT = np.zeros(_MAX_MON, dtype=np.float32)
_PM_IS_INSTAKILL = np.zeros(_MAX_MON, dtype=bool)
_PM_LETHAL_POISON_WEIGHT = np.zeros(_MAX_MON, dtype=np.float32)
_PM_IS_LETHAL_POISON = np.zeros(_MAX_MON, dtype=bool)

_INSTAKILL_NAMES = {
    "floating eye": 200.0,
    "gas spore": 300.0,
    "cockatrice": 250.0,
    "chickatrice": 200.0,
    "mind flayer": 300.0,
    "master mind flayer": 400.0,
    "homunculus": 100.0,
    "rust monster": 75.0,
    "disenchanter": 90.0,
}

_LETHAL_POISON_NAMES = {
    "giant spider": 350.0,
    "killer bee": 300.0,
    "queen bee": 300.0,
    "soldier ant": 280.0,
    "scorpion": 250.0,
    "pit viper": 220.0,
    "water moccasin": 200.0,
    "snake": 150.0,
}

_PEACEFUL_HUMANS = {
    "shopkeeper", "priest", "priestess", "watchman", "watch captain",
    "guard", "vault guard", "aligned priest", "aligned cleric", "cleric",
    "high priest", "oracle",
}

_PEACEFUL_NAMES = {
    "shopkeeper", "priest", "priestess", "watchman", "watch captain",
    "aligned priest", "aligned cleric", "cleric", "high priest", "oracle",
    "guard", "vault guard", "horse", "saddled horse", "pony", "warhorse",
    "cat", "kitten", "housecat", "large cat", "dog", "little dog",
    "large dog", "peaceful",
}

for _mid in range(getattr(nethack, "NUMMONS", 381)):
    try:
        _pm = nethack.permonst(_mid)
        _mname = _pm.mname.lower()
        _PM_NAME[_mid] = _mname
        _PM_SPEED[_mid] = _pm.mmove
        _PM_LEVEL[_mid] = _pm.mlevel
        _PM_AC[_mid] = _pm.ac

        if any(_p in _mname for _p in _PEACEFUL_NAMES):
            _PM_IS_PEACEFUL_TYPE[_mid] = True
        if any(_h in _mname for _h in _PEACEFUL_HUMANS):
            _PM_IS_PEACEFUL_HUMAN[_mid] = True

        for _ik_name, _weight in _INSTAKILL_NAMES.items():
            if _ik_name in _mname:
                _PM_INSTAKILL_WEIGHT[_mid] = _weight
                _PM_IS_INSTAKILL[_mid] = True
                break

        for _lp_name, _weight in _LETHAL_POISON_NAMES.items():
            if _lp_name in _mname:
                if _lp_name == "snake" and "garter snake" in _mname:
                    continue
                _PM_LETHAL_POISON_WEIGHT[_mid] = _weight
                _PM_IS_LETHAL_POISON[_mid] = True
                break
    except Exception:
        pass


class ThreatScanMixin:
    # -- per-perception scan memo -------------------------------------------------
    def _scan_key(self, glyphs, blstats, has_poison_res):
        try:
            # Fast perceptual signature avoiding 3.3KB per-call string allocation
            sig = (
                int(glyphs[0, 0]),
                int(glyphs[10, 39]),
                int(glyphs[20, 78]),
                int(glyphs[getattr(blstats, "y", 0), getattr(blstats, "x", 0)]),
                int(glyphs.sum()),
            )
        except Exception:  # noqa: BLE001 — non-array glyphs: never cache
            return None
        return (sig, int(getattr(blstats, "turn", 0)), bool(has_poison_res),
                int(getattr(blstats, "y", -1)), int(getattr(blstats, "x", -1)),
                int(getattr(blstats, "hp", 0)), int(getattr(blstats, "max_hp", 0)),
                int(getattr(self, "turns_since_damaged", 0)))

    def _scan_cache_set(self, key, monsters):
        self._scan_cache = (key, monsters)

    def _scan_monsters_cached(self, key):
        if key is None:
            return None
        cached = getattr(self, "_scan_cache", None)
        if cached is not None and cached[0] == key:
            return list(cached[1])
        return None

    def scan_monsters(
        self,
        glyphs: np.ndarray,
        blstats: BottomLineStats,
        has_poison_res: bool = False,
    ) -> list[MonsterTrack]:
        """
        Scans observation glyphs for all visible hostile monsters using precomputed LUT.
        """
        _cache_key = self._scan_key(glyphs, blstats, has_poison_res)
        cached = self._scan_monsters_cached(_cache_key)
        if cached is not None:
            return cached
        py, px = blstats.y, blstats.x
        monsters: list[MonsterTrack] = []

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
            if _cache_key is not None:
                self._scan_cache_set(_cache_key, list(monsters))
            return monsters

        for idx in range(len(mon_coords)):
            r, c = int(mon_coords[idx, 0]), int(mon_coords[idx, 1])
            g = int(glyphs[r, c])

            mon_id = nethack.glyph_to_mon(g)
            if (r, c) in self.peaceful_positions:
                continue

            if 0 <= mon_id < _MAX_MON:
                mname = _PM_NAME[mon_id]
                speed = int(_PM_SPEED[mon_id])
                level = int(_PM_LEVEL[mon_id])
                ac = int(_PM_AC[mon_id])
                is_peaceful_type = bool(_PM_IS_PEACEFUL_TYPE[mon_id]) or any(p in mname for p in self.PEACEFUL_NAMES)
                is_peaceful_human = bool(_PM_IS_PEACEFUL_HUMAN[mon_id])
                is_instakill = bool(_PM_IS_INSTAKILL[mon_id])
                instakill_weight = float(_PM_INSTAKILL_WEIGHT[mon_id])
                is_lethal_poison = bool(_PM_IS_LETHAL_POISON[mon_id])
                poison_weight = float(_PM_LETHAL_POISON_WEIGHT[mon_id])
            else:
                pm = nethack.permonst(mon_id)
                mname = pm.mname.lower()
                speed = int(pm.mmove)
                level = int(pm.mlevel)
                ac = int(pm.ac)
                is_peaceful_type = any(p in mname for p in self.PEACEFUL_NAMES)
                is_peaceful_human = any(h in mname for h in self.PEACEFUL_HUMANS)
                is_instakill = False
                instakill_weight = 0.0
                for ik_name, weight in self.INSTAKILL_NAMES.items():
                    if ik_name in mname:
                        instakill_weight = weight
                        is_instakill = True
                        break
                is_lethal_poison = False
                poison_weight = 0.0
                for lp_name, weight in self.LETHAL_POISON_NAMES.items():
                    if lp_name in mname and not (lp_name == "snake" and "garter snake" in mname):
                        poison_weight = weight
                        is_lethal_poison = True
                        break

            is_known_hostile = any(h in mname for h in self.hostile_names)
            under_attack = self.turns_since_damaged < self.cfg.survival.escape_grace_turns and max(abs(r - py), abs(c - px)) == 1
            if is_peaceful_type and not is_known_hostile:
                if is_peaceful_human or not under_attack:
                    self.peaceful_positions.add((r, c))
                    continue

            dist = max(abs(r - py), abs(c - px))
            is_adj = dist == 1

            if not has_poison_res and not is_instakill and is_lethal_poison:
                instakill_weight = poison_weight
                is_instakill = True

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
                    is_adjacent=bool(is_adj),
                    is_instakill=bool(is_instakill),
                    threat_score=threat,
                )
            )

        monsters.sort(key=lambda m: (m.distance, -m.threat_score))
        if _cache_key is not None:
            self._scan_cache_set(_cache_key, list(monsters))
        return monsters

