"""
Offline C Symbol Table Queries over NetHack 3.6.6 C Structs via NLE.
Provides zero-latency, ground-truth queries over `nle.nethack.permonst` (monsters)
and `nle.nethack.objclass` (items) without wiki scraping or guessing.
"""

from __future__ import annotations

from typing import Any

RESISTANCE_FLAGS: dict[int, str] = {
    0x01: "fire_resistance",
    0x02: "cold_resistance",
    0x04: "sleep_resistance",
    0x08: "disintegration_resistance",
    0x10: "shock_resistance",
    0x20: "poison_resistance",
    0x40: "acid_resistance",
    0x80: "petrification_resistance",
}

OCLASS_NAMES: dict[int, str] = {
    1: "ILLOBJ",
    2: "WEAPON",
    3: "ARMOR",
    4: "RING",
    5: "AMULET",
    6: "TOOL",
    7: "FOOD",
    8: "POTION",
    9: "SCROLL",
    10: "SPELLBOOK",
    11: "WAND",
    12: "COIN",
    13: "GEM",
    14: "ROCK",
    15: "BALL",
    16: "CHAIN",
    17: "VENOM",
}


CLASS_PREFIXES: dict[int, str] = {
    4: "ring of",
    5: "amulet of",
    8: "potion of",
    9: "scroll of",
    10: "spellbook of",
    11: "wand of",
}


def _decode_resistances(mask: int) -> list[str]:
    res = []
    for bit, name in RESISTANCE_FLAGS.items():
        if mask & bit:
            res.append(name)
    return res


class NetHackSymbolRegistry:
    """Pre-indexed registry of all native NetHack 3.6.6 permonst and objclass entries."""

    def __init__(self) -> None:
        self._monsters_by_name: dict[str, dict[str, Any]] = {}
        self._objects_by_name: dict[str, dict[str, Any]] = {}
        self._load_symbols()

    def _load_symbols(self) -> None:
        try:
            import nle.nethack as nh
        except Exception:
            return

        # Index monsters from nh.permonst
        for i in range(1000):
            try:
                pm = nh.permonst(i)
                mname = str(pm.mname).strip().lower()
                if not mname or mname in self._monsters_by_name:
                    continue
                self._monsters_by_name[mname] = {
                    "id": i,
                    "name": mname,
                    "level": int(pm.mlevel),
                    "speed": int(pm.mmove),
                    "ac": int(pm.ac),
                    "magic_resistance": int(pm.mr),
                    "nutrition": int(pm.cnutrit),
                    "corpse_weight": int(pm.cwt),
                    "conveys": _decode_resistances(int(pm.mconveys)),
                    "resists": _decode_resistances(int(pm.mresists)),
                    "difficulty": int(getattr(pm, "difficulty", pm.mlevel)),
                    "size": int(getattr(pm, "msize", 0)),
                }
            except Exception:
                break

        # Index objects from nh.objclass
        num_objs = int(getattr(nh, "NUM_OBJECTS", 500))
        for i in range(num_objs):
            try:
                oc = nh.objclass(i)
                raw_name = str(nh.OBJ_NAME(oc)).strip().lower()
                if not raw_name:
                    continue
                oc_class_int = (
                    int(oc.oc_class)
                    if isinstance(oc.oc_class, int)
                    else ord(str(oc.oc_class or " "))
                )
                cls_name = OCLASS_NAMES.get(oc_class_int, f"CLASS_{oc_class_int}")
                prefix = CLASS_PREFIXES.get(oc_class_int)
                full_name = f"{prefix} {raw_name}" if prefix else raw_name
                obj_data = {
                    "id": i,
                    "name": full_name,
                    "raw_name": raw_name,
                    "class": cls_name,
                    "cost": int(oc.oc_cost),
                    "weight": int(oc.oc_weight),
                    "nutrition": int(getattr(oc, "oc_nutrition", 0)),
                }
                if full_name not in self._objects_by_name:
                    self._objects_by_name[full_name] = obj_data
                if raw_name not in self._objects_by_name:
                    self._objects_by_name[raw_name] = obj_data
            except Exception:
                break

    def query_monster(self, name: str) -> dict[str, Any] | None:
        q = str(name).strip().lower()
        if q in self._monsters_by_name:
            return self._monsters_by_name[q]
        # Partial prefix or substring match
        for mname, data in self._monsters_by_name.items():
            if q == mname or q in mname.split():
                return data
        for mname, data in self._monsters_by_name.items():
            if q in mname:
                return data
        return None

    def query_item(self, name: str) -> dict[str, Any] | None:
        q = str(name).strip().lower()
        if q in self._objects_by_name:
            return self._objects_by_name[q]
        for oname, data in self._objects_by_name.items():
            if q == oname or q in oname.split():
                return data
        for oname, data in self._objects_by_name.items():
            if q in oname:
                return data
        return None

    def search_monsters(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        q = str(query).strip().lower()
        matches = []
        for mname, data in self._monsters_by_name.items():
            if q in mname:
                matches.append(data)
                if len(matches) >= limit:
                    break
        return matches

    def search_items(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        q = str(query).strip().lower()
        matches = []
        for oname, data in self._objects_by_name.items():
            if q in oname:
                matches.append(data)
                if len(matches) >= limit:
                    break
        return matches


# Global singleton instance
SYMBOL_REGISTRY = NetHackSymbolRegistry()
