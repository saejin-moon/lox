"""
Domain Layer: Tactical Combat Manager.
Governs monster threat evaluation, kiting, corridor funneling, instakill avoidance,
and emergency Elbereth dust-engraving.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
from nle import nethack

from corp.env.blstats import BottomLineStats, ConditionFlag, HungerState
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
        "gas spore": 300.0,
        "cockatrice": 250.0,
        "chickatrice": 200.0,
        "mind flayer": 300.0,
        "master mind flayer": 400.0,
        "homunculus": 100.0,
        "rust monster": 75.0,
        "disenchanter": 90.0,
    }

    # Lethal poison biters in NetHack 3.6.6 (instakill / fatal strength drain without poison resistance)
    LETHAL_POISON_NAMES = {
        "giant spider": 350.0,
        "killer bee": 300.0,
        "queen bee": 300.0,
        "soldier ant": 280.0,
        "scorpion": 250.0,
        "pit viper": 220.0,
        "water moccasin": 200.0,
        "snake": 150.0,
    }

    # Monsters frequently generating with or zapping offensive wands (striking/lightning/fire/cold)
    WAND_WIELDER_NAMES = (
        "gnome lord",
        "gnome king",
        "gnomish wizard",
        "elf-lord",
        "elvenking",
        "lieutenant",
        "captain",
        "wizard",
    )

    # Peaceful / shopkeeper / domestic entities to never provoke unless they attack us
    PEACEFUL_NAMES = {
        "shopkeeper",
        "priest",
        "priestess",
        "watchman",
        "watch captain",
        "aligned priest",
        "aligned cleric",
        "cleric",
        "high priest",
        "oracle",
        "guard",
        "vault guard",
        "horse",
        "saddled horse",
        "pony",
        "warhorse",
        "cat",
        "kitten",
        "housecat",
        "large cat",
        "dog",
        "little dog",
        "large dog",
        "peaceful",
    }

    # Peaceful humans: provoking them (attacking) is instantly fatal (shopkeeper wands,
    # priest smiting). Never attackable regardless of circumstances.
    PEACEFUL_HUMANS = {
        "shopkeeper",
        "priest",
        "priestess",
        "watchman",
        "watch captain",
        "aligned priest",
        "aligned cleric",
        "cleric",
        "high priest",
        "oracle",
        "guard",
        "vault guard",
        "shopkeeper's",
    }

    # Mounts (ponies, horses): provoking them provokes 2d7 kick retaliation — the AGENTS.md
    # domestic-animal interlock. Never attackable even when adjacent under attack.
    PEACEFUL_MOUNTS = {
        "pony",
        "horse",
        "saddled horse",
        "warhorse",
    }

    # Throwable projectile items
    THROWABLE_KEYWORDS = (
        "dagger",
        "dart",
        "shuriken",
        "boomerang",
        "spear",
        "javelin",
        "rock",
    )

    # Offensive wand keywords
    OFFENSIVE_WAND_KEYWORDS = (
        "wand of striking",
        "wand of magic missile",
        "wand of fire",
        "wand of cold",
        "wand of lightning",
        "wand of sleep",
        "wand of death",
    )

    PERMANENT_ENGRAVERS = ("athame", "wand of digging", "wand of fire", "wand of lightning")

    # Heavy hitter monster classes and names that hit for 10-30+ HP per turn
    HEAVY_HITTERS = (
        "ogre",
        "ogre lord",
        "ogre king",
        "gnome lord",
        "gnome king",
        "dwarf lord",
        "dwarf king",
        "soldier ant",
        "rothe",
        "mumak",
        "leocrotta",
        "stalker",
        "giant",
        "ettin",
        "minotaur",
    )

    def __init__(self, player_speed: int = 12):
        self.player_speed = player_speed
        self.elbereth_pos: tuple[int, int, int] | None = None  # (depth, py, px)
        self.elbereth_turns: int = 0
        self.elbereth_waited_turns: int = 0
        self.elbereth_cooldown: int = 0
        self._prev_hp: int = 0
        self.turns_since_damaged: int = 100
        self.peaceful_positions: set[tuple[int, int]] = set()
        self.hostile_names: set[str] = set()

    @property
    def on_elbereth_turns(self) -> int:
        return self.elbereth_turns

    @on_elbereth_turns.setter
    def on_elbereth_turns(self, val: int) -> None:
        self.elbereth_turns = val

    def reset(self) -> None:
        """Resets combat manager state for a new episode."""
        self.elbereth_pos = None
        self.elbereth_turns = 0
        self.elbereth_waited_turns = 0
        self.elbereth_cooldown = 0
        self._prev_hp = 0
        self.turns_since_damaged = 100
        self.peaceful_positions.clear()
        self.hostile_names.clear()

    def record_peaceful(self, pos: tuple[int, int]) -> None:
        """Marks a coordinate as housing a confirmed peaceful entity."""
        self.peaceful_positions.add(pos)

    def _find_athame_or_permanent_engraver(self, inv_tracker: Any) -> str | None:
        """Finds an athame or permanent engraving wand if available in inventory."""
        if not inv_tracker:
            return None
        items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [
            it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)
        ]
        for item in items:
            if item.buc_state == "CURSED":
                continue
            desc = item.raw_str.lower()
            if any(eng in desc for eng in self.PERMANENT_ENGRAVERS):
                if "wand" in desc and ("(0:0)" in desc or "(0)" in desc):
                    continue
                return item.current_letter
        return None

    def _find_throwable_slot(self, inv_tracker: Any) -> str | None:
        """Finds an inventory slot containing throwable projectiles."""
        if not inv_tracker:
            return None
        items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [
            it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)
        ]
        for item in items:
            desc = item.raw_str.lower()
            if any(k in desc for k in self.THROWABLE_KEYWORDS):
                return item.current_letter
        return None

    def _find_quivered_or_missile(self, inv_tracker: Any) -> tuple[str | None, bool]:
        if getattr(self, "_ranged_disabled", False) or getattr(self, "_ranged_fail_streak", 0) >= 3:
            return None, False  # ranged temporarily disabled (stale-slot "Never mind" storm)
        """
        Returns (slot, is_quivered).
        If an item is in quiver (marked 'in quiver' in raw_str), returns (slot, True).
        Otherwise returns the first throwable missile slot (slot, False).
        """
        if not inv_tracker:
            return None, False
        items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [
            it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)
        ]
        first_missile = None
        for item in items:
            desc = item.raw_str.lower()
            if "in quiver" in desc or "at the ready" in desc:
                return item.current_letter, True
            if first_missile is None and any(k in desc for k in self.THROWABLE_KEYWORDS):
                first_missile = item.current_letter
        return first_missile, False

    def _has_usable_healing(self, inv_tracker: Any) -> bool:
        """
        True if the hero carries a usable healing resource: potion of healing / extra
        healing / full healing, or a healing spellbook (uncursed/blessed only).
        Used by the critical-HP universal retreat (cursed items are undrinkable).
        """
        if not inv_tracker:
            return False
        items = (
            inv_tracker.get_active_items()
            if hasattr(inv_tracker, "get_active_items")
            else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
        )
        for item in items:
            desc = item.raw_str.lower()
            is_cursed = desc.startswith("cursed") or " cursed" in desc
            if is_cursed:
                continue
            if ("healing" in desc or "heal" in desc) and ("potion" in desc or "spellbook" in desc):
                return True
        return False

    def _find_offensive_wand_slot(self, inv_tracker: Any) -> str | None:
        if getattr(self, "_ranged_disabled", False) or getattr(self, "_ranged_fail_streak", 0) >= 3:
            return None  # ranged temporarily disabled (stale-slot "Never mind" storm)
        """Finds an uncursed directional offensive wand with available charges."""
        if not inv_tracker:
            return None
        items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [
            it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)
        ]
        for item in items:
            if item.buc_state == "CURSED":
                continue
            desc = item.raw_str.lower()
            if any(w in desc for w in self.OFFENSIVE_WAND_KEYWORDS):
                if "(0:0)" not in desc and "(0)" not in desc:
                    return item.current_letter
        return None

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
            under_attack = self.turns_since_damaged < 3 and max(abs(r - py), abs(c - px)) == 1
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
        inv_tracker: Any = None,
        role: str = "",
        message: str = "",
        has_poison_res: bool = False,
        has_reflection: bool = False,
    ) -> Task | None:
        """
        Evaluates tactical posture and returns the optimal combat Task,
        or None if no combat intervention is required.
        """
        # 0. Emergency Status: Wipe face when blinded by venom, cream pie, or mud
        msg_lower = message.lower()

        # Ranged-failure streak guard: stale inventory slot letters cause "Never mind." /
        # "don't have that object" storms — 1,000+ zero-turn decisions were observed.
        # After 3 consecutive failures, ranged attacks are temporarily disabled (the
        # streak self-heals as soon as the message stops showing the failure).
        if "never mind" in msg_lower or "don't have that object" in msg_lower:
            self._ranged_fail_streak = getattr(self, "_ranged_fail_streak", 0) + 1
            if self._ranged_fail_streak >= 3:
                # Timed cooldown: stale-slot failures persist until the inventory tracker
                # realigns, so a single clean turn must not re-enable the storm
                self._ranged_cooldown_until = int(getattr(blstats, "turn", 0)) + 50
                self._ranged_fail_streak = 0
        else:
            self._ranged_fail_streak = 0
        self._ranged_disabled = int(getattr(blstats, "turn", 0)) < getattr(
            self, "_ranged_cooldown_until", -1
        )
        if blstats.is_blind and any(w in msg_lower for w in ("can't see", "cream", "venom", "mud", "slime")):
            return Task("WIPE", is_primitive=True)

        is_attacked_msg = any(atk in msg_lower for atk in ("bites", "hits", "claws", "kicks", "strikes", "shoots", "casts", "zaps", "touches", "stung"))
        if (self._prev_hp > 0 and blstats.hp < self._prev_hp) or is_attacked_msg:
            self.turns_since_damaged = 0
            self.peaceful_positions.clear()
            for p in self.PEACEFUL_NAMES:
                if p in msg_lower and p != "peaceful":
                    self.hostile_names.add(p)
        else:
            self.turns_since_damaged += 1

        try:
            monsters = self.scan_monsters(glyphs, blstats, has_poison_res=has_poison_res)
        except TypeError:
            monsters = self.scan_monsters(glyphs, blstats)
        if not monsters:
            # Tactical Healing Rest: ONLY rest if safe, well-fed, and NOT recently damaged!
            # If hero took damage in the last 5 turns, resting is strictly forbidden (an unseen or dark attacker is present!)
            if (
                blstats.hp < max(int(blstats.max_hp * 0.85), blstats.max_hp - 2)
                and blstats.hunger_state <= HungerState.NORMAL
                and self.turns_since_damaged >= 5
            ):
                self._prev_hp = blstats.hp
                return Task("WAIT", is_primitive=True)
            self._prev_hp = blstats.hp
            return None

        py, px = blstats.y, blstats.x
        is_blind = blstats.is_blind
        cur_pos = (blstats.depth, py, px)

        if self.elbereth_cooldown > 0:
            self.elbereth_cooldown -= 1

        # Invalidate Elbereth if hero has changed position or depth
        if self.elbereth_pos is not None and cur_pos != self.elbereth_pos:
            self.elbereth_pos = None
            self.elbereth_turns = 0
            self.elbereth_waited_turns = 0

        adjacent_monsters = [m for m in monsters if m.is_adjacent]
        active_adjacent = [
            m for m in adjacent_monsters
            if "floating eye" not in m.name and "gas spore" not in m.name
        ]

        # Turn Undead for Priests/Knights when facing multiple undead monsters
        # (Must NOT trigger if living adjacent threats like grid bugs, newts, or jackals are attacking!)
        if role.lower() in ("priest", "knight"):
            is_undead = lambda m: any(u in m.name.lower() for u in ("zombie", "skeleton", "vampire", "ghoul", "wraith", "mummy", "lich", "ghost", "shade"))
            if adjacent_monsters:
                adjacent_undead = [m for m in adjacent_monsters if is_undead(m)]
                if len(adjacent_undead) >= 2 and len(adjacent_monsters) == len(adjacent_undead):
                    return Task("TURN_UNDEAD", is_primitive=True)
            elif len(monsters) >= 2 and all(is_undead(m) for m in monsters):
                return Task("TURN_UNDEAD", is_primitive=True)

        # 0. Active Elbereth Ward Check
        if self.elbereth_turns > 0 and cur_pos == self.elbereth_pos:
            # If hero took damage while standing on Elbereth, the ward has eroded or failed!
            if self._prev_hp > 0 and blstats.hp < self._prev_hp:
                self.elbereth_pos = None
                self.elbereth_turns = 0
                self.elbereth_waited_turns = 0
                self.elbereth_cooldown = 10
            elif not adjacent_monsters:
                # No adjacent monsters: monsters fled! Safe to rest/wait on Elbereth
                self.elbereth_turns -= 1
                if blstats.hp < int(blstats.max_hp * 0.75):
                    self._prev_hp = blstats.hp
                    return Task("WAIT", is_primitive=True)
            elif self.elbereth_waited_turns == 0:
                # Freshly engraved Elbereth: grant 1 turn of WAIT to allow adjacent monster to flee
                self.elbereth_waited_turns = 1
                self.elbereth_turns -= 1
                if blstats.hp < int(blstats.max_hp * 0.75):
                    self._prev_hp = blstats.hp
                    return Task("WAIT", is_primitive=True)
            else:
                # Monsters are still adjacent after grace turn: they did NOT flee!
                # Do NOT passively wait while being beaten to death! Break ward and fight!
                self.elbereth_pos = None
                self.elbereth_turns = 0
                self.elbereth_waited_turns = 0
                self.elbereth_cooldown = 10

        # 1. Corridor Tactical Funneling
        # If surrounded by >= 2 adjacent active hostiles, step into an adjacent corridor ('#')
        # or doorway ('+' or '\'') to funnel them single-file.
        # Skip funneling if trapped in bear trap, web, or pit!
        is_trapped = any(w in message.lower() for w in ("bear trap", "caught in", "stuck in", "fall into a pit", "in a web"))
        if len(active_adjacent) >= 2 and not is_trapped:
            threat_pos = {m.pos for m in monsters}
            funnel_step = self._find_funnel_step(py, px, chars, threat_pos)
            if funnel_step is not None:
                self.elbereth_pos = None
                self.elbereth_turns = 0
                self.elbereth_waited_turns = 0
                self._prev_hp = blstats.hp
                return Task("STEP", is_primitive=True, args={"delta": funnel_step})

        # 1.5 Emergency Elbereth Engraving
        # Trigger when adjacent monsters exist and:
        # (a) HP is <= 60% or <= 8, OR
        # (b) Surrounded by >= 2 adjacent active monsters (in open space without immediate choke point)
        if adjacent_monsters and self.elbereth_turns <= 0 and self.elbereth_cooldown <= 0:
            is_fragile = role.lower() in ("tourist", "wizard", "healer", "priest", "rogue")
            hp_thresh = 0.75 if is_fragile else 0.60
            min_hp = 10 if is_fragile else 8
            is_critical_hp = blstats.hp <= max(min_hp, int(blstats.max_hp * hp_thresh))
            is_surrounded = len(active_adjacent) >= 2
            if is_critical_hp or is_surrounded:
                permanent_slot = self._find_athame_or_permanent_engraver(inv_tracker)
                self.elbereth_pos = cur_pos
                self.elbereth_turns = 12 if permanent_slot else 8
                self.elbereth_waited_turns = 0
                args = {"text": "Elbereth"}
                if permanent_slot:
                    args["slot"] = permanent_slot
                self._prev_hp = blstats.hp
                return Task(
                    name="ENGRAVE_DUST",
                    is_primitive=True,
                    args=args,
                )

        # 2. Tactical Door Closing
        # If hero is in a corridor ('#') or doorway ('+' or '\'') and an adjacent tile is an open door ('\'')
        # with hostiles pursuing on the other side, close the door to break line of sight and halt pursuit!
        door_task = self._evaluate_door_closing(chars, py, px, monsters)
        if door_task is not None:
            return door_task

        # 3. Adjacent monster handling
        if adjacent_monsters:
            # Separate active hostiles (capable of moving and attacking) from passive non-attacking hazards
            active_adjacent = [
                m for m in adjacent_monsters
                if "floating eye" not in m.name and "gas spore" not in m.name
            ]
            passive_adjacent = [
                m for m in adjacent_monsters
                if "floating eye" in m.name or "gas spore" in m.name
            ]

            if active_adjacent:
                # Prioritize the most dangerous active attacker!
                primary_target = active_adjacent[0]
            else:
                # Only passive hazards adjacent (floating eye or gas spore)
                primary_target = passive_adjacent[0]

            dr = primary_target.pos[0] - py
            dc = primary_target.pos[1] - px

            # Floating eye & Gas spore guard: Melee attack strictly forbidden for eyes
            # Floating eye causes permanent paralysis; Gas spore explodes for 4d6 fatal damage
            if ("floating eye" in primary_target.name and not is_blind) or "gas spore" in primary_target.name:
                is_spore = "gas spore" in primary_target.name
                # Ranged kills are safe for both eyes and spores (spore explodes at range)
                if inv_tracker is not None:
                    offensive_wand = self._find_offensive_wand_slot(inv_tracker)
                    if offensive_wand is not None:
                        return Task("ZAP_WAND", is_primitive=True, args={"slot": offensive_wand, "delta": (dr, dc)})
                    missile_slot, is_quivered = self._find_quivered_or_missile(inv_tracker)
                    if is_quivered:
                        return Task("FIRE", is_primitive=True, args={"delta": (dr, dc)})
                    if missile_slot is not None:
                        return Task("THROW", is_primitive=True, args={"slot": missile_slot, "delta": (dr, dc)})

                # Must not melee eyes! Find a non-monster tile to step away
                step_away_delta = self._find_escape_step(py, px, adjacent_monsters, chars, glyphs=glyphs)
                if step_away_delta is not None:
                    self.elbereth_pos = None
                    self.elbereth_turns = 0
                    return Task("STEP", is_primitive=True, args={"delta": step_away_delta})

                # If cannot step away and taking damage from an active attacker, do NOT wait!
                if not is_spore and self.turns_since_damaged < 3:
                    # Elbereth: eyes flee from it (gas spores are mindless and ignore it!)
                    if self.elbereth_turns <= 0 and self.elbereth_cooldown <= 0:
                        self.elbereth_pos = cur_pos
                        self.elbereth_turns = 8
                        self.elbereth_waited_turns = 0
                        return Task("ENGRAVE_DUST", is_primitive=True, args={"text": "Elbereth"})
                    # If cannot step away, wait rather than strike
                    return Task("WAIT", is_primitive=True)
                if is_spore:
                    # Cornered by a gas spore: waiting changes nothing (it explodes anyway).
                    # Kill it in melee — the explosion damage is unavoidable and a dead spore
                    # stops calling more spores. (Stall fix: previously WAIT-looped forever.)
                    self.elbereth_pos = None
                    self.elbereth_turns = 0
                    self._prev_hp = blstats.hp
                    return Task("MELEE_ATTACK", is_primitive=True, args={"delta": (dr, dc)})

            # Cockatrice guard: no unarmed combat (default weapons usually equipped)
            if "cockatrice" in primary_target.name or "chickatrice" in primary_target.name:
                pass # Weapons are checked in inventory manager

            # Grid bug guard: Grid bugs can ONLY attack orthogonally (cannot attack diagonally)
            if "grid bug" in primary_target.name:
                is_diagonal = (dr != 0 and dc != 0)
                if is_diagonal:
                    # Diagonal melee attack: 100% safe from grid bug retaliation!
                    self.elbereth_pos = None
                    self.elbereth_turns = 0
                    return Task("MELEE_ATTACK", is_primitive=True, args={"delta": (dr, dc)})
                else:
                    # Orthogonal: try ranged first
                    if inv_tracker is not None:
                        missile_slot, is_quivered = self._find_quivered_or_missile(inv_tracker)
                        if is_quivered:
                            self.elbereth_pos = None
                            self.elbereth_turns = 0
                            return Task("FIRE", is_primitive=True, args={"delta": (dr, dc)})
                        if missile_slot is not None:
                            self.elbereth_pos = None
                            self.elbereth_turns = 0
                            return Task("THROW", is_primitive=True, args={"slot": missile_slot, "delta": (dr, dc)})
                    # Step to adjacent tile that makes the grid bug diagonal
                    threat_pos = {m.pos for m in monsters}
                    diag_step = self._find_grid_bug_diagonal_step(py, px, primary_target.pos, chars, threat_pos)
                    if diag_step is not None:
                        self.elbereth_pos = None
                        self.elbereth_turns = 0
                        return Task("STEP", is_primitive=True, args={"delta": diag_step})

            # Lethal poison biters (Giant spider, queen bee, scorpion) without poison resistance:
            # NetHack 3.6.6: giant spider poison bites can deal instakill or fatal strength drain.
            # Avoid direct melee: engrave dust Elbereth (spiders/bees flee immediately), shoot missiles, or step away!
            is_lethal_poison_target = (
                not has_poison_res
                and any(p in primary_target.name for p in self.LETHAL_POISON_NAMES)
            )
            if is_lethal_poison_target:
                # If already standing on active Elbereth, wait for it to flee!
                if self.elbereth_turns > 0 and cur_pos == self.elbereth_pos:
                    self._prev_hp = blstats.hp
                    return Task("WAIT", is_primitive=True)
                # Engrave Elbereth if available
                if self.elbereth_turns <= 0 and self.elbereth_cooldown <= 0:
                    permanent_slot = self._find_athame_or_permanent_engraver(inv_tracker)
                    self.elbereth_pos = cur_pos
                    self.elbereth_turns = 12 if permanent_slot else 8
                    self.elbereth_waited_turns = 0
                    args = {"text": "Elbereth"}
                    if permanent_slot:
                        args["slot"] = permanent_slot
                    self._prev_hp = blstats.hp
                    return Task("ENGRAVE_DUST", is_primitive=True, args=args)
                # Step away to safe tile first (kiting: break adjacent melee contact!)
                step_away = self._find_escape_step(py, px, adjacent_monsters, chars, glyphs=glyphs)
                if step_away is not None:
                    return Task("STEP", is_primitive=True, args={"delta": step_away})
                # If cornered (cannot step away), try ranged attack (wand, fire, throw)
                if inv_tracker is not None:
                    offensive_wand = self._find_offensive_wand_slot(inv_tracker)
                    if offensive_wand is not None:
                        return Task("ZAP_WAND", is_primitive=True, args={"slot": offensive_wand, "delta": (dr, dc)})
                    missile_slot, is_quivered = self._find_quivered_or_missile(inv_tracker)
                    if is_quivered:
                        return Task("FIRE", is_primitive=True, args={"delta": (dr, dc)})
                    if missile_slot is not None:
                        return Task("THROW", is_primitive=True, args={"slot": missile_slot, "delta": (dr, dc)})

            # Heavy Hitter Tactical Kiting: When wounded (HP <= 65% or HP <= 16),
            # do not trade blows toe-to-toe with dwarf lords, gnome lords, or ogres!
            is_heavy_hitter = any(hh in primary_target.name for hh in self.HEAVY_HITTERS)
            is_wounded_vs_heavy = is_heavy_hitter and (blstats.hp <= max(16, int(blstats.max_hp * 0.65)))
            if is_wounded_vs_heavy:
                if self.elbereth_turns > 0 and cur_pos == self.elbereth_pos:
                    self._prev_hp = blstats.hp
                    return Task("WAIT", is_primitive=True)
                if self.elbereth_turns <= 0 and self.elbereth_cooldown <= 0:
                    permanent_slot = self._find_athame_or_permanent_engraver(inv_tracker)
                    self.elbereth_pos = cur_pos
                    self.elbereth_turns = 12 if permanent_slot else 8
                    self.elbereth_waited_turns = 0
                    args = {"text": "Elbereth"}
                    if permanent_slot:
                        args["slot"] = permanent_slot
                    self._prev_hp = blstats.hp
                    return Task("ENGRAVE_DUST", is_primitive=True, args=args)
                step_away = self._find_escape_step(py, px, adjacent_monsters, chars, glyphs=glyphs)
                if step_away is not None:
                    return Task("STEP", is_primitive=True, args={"delta": step_away})
                if inv_tracker is not None:
                    missile_slot, is_quivered = self._find_quivered_or_missile(inv_tracker)
                    if is_quivered:
                        return Task("FIRE", is_primitive=True, args={"delta": (dr, dc)})
                    if missile_slot is not None:
                        return Task("THROW", is_primitive=True, args={"slot": missile_slot, "delta": (dr, dc)})

            # Phase 3: Speed-differential constraint — fast monsters (speed > 1.3x hero, e.g.
            # giant bats speed 22, soldier ants 18, killer bees 18) shred the hero with
            # attrition damage in open rooms. Engage only with ranged attacks; fall back to
            # melee only when no missiles/wands are available (cornered).
            if primary_target.speed > int(self.player_speed * 1.3) and inv_tracker is not None:
                offensive_wand = self._find_offensive_wand_slot(inv_tracker)
                if offensive_wand is not None:
                    self.elbereth_pos = None
                    self.elbereth_turns = 0
                    self._prev_hp = blstats.hp
                    return Task("ZAP_WAND", is_primitive=True, args={"slot": offensive_wand, "delta": (dr, dc)})
                missile_slot, is_quivered = self._find_quivered_or_missile(inv_tracker)
                if is_quivered:
                    self.elbereth_pos = None
                    self.elbereth_turns = 0
                    self._prev_hp = blstats.hp
                    return Task("FIRE", is_primitive=True, args={"delta": (dr, dc)})
                if missile_slot is not None:
                    self.elbereth_pos = None
                    self.elbereth_turns = 0
                    self._prev_hp = blstats.hp
                    return Task("THROW", is_primitive=True, args={"slot": missile_slot, "delta": (dr, dc)})

            # Critical-HP Universal Retreat (attrition-death fix): at <= 25% HP with no
            # usable healing resource, NEVER trade blows — heroes were observed grinding
            # to death at 3 HP against kittens and dogs. Engrave Elbereth (most monsters
            # flee), retreat maximizing separation, or fire ranged — melee is the last
            # resort only when fully cornered with zero alternatives.
            if blstats.hp <= int(blstats.max_hp * 0.25) and not self._has_usable_healing(inv_tracker):
                # Ranged first: kill from a "distance" without exposing further
                if inv_tracker is not None:
                    offensive_wand = self._find_offensive_wand_slot(inv_tracker)
                    if offensive_wand is not None:
                        self.elbereth_pos = None
                        self.elbereth_turns = 0
                        self._prev_hp = blstats.hp
                        return Task("ZAP_WAND", is_primitive=True, args={"slot": offensive_wand, "delta": (dr, dc)})
                    missile_slot, is_quivered = self._find_quivered_or_missile(inv_tracker)
                    if is_quivered:
                        self.elbereth_pos = None
                        self.elbereth_turns = 0
                        self._prev_hp = blstats.hp
                        return Task("FIRE", is_primitive=True, args={"delta": (dr, dc)})
                    if missile_slot is not None:
                        self.elbereth_pos = None
                        self.elbereth_turns = 0
                        self._prev_hp = blstats.hp
                        return Task("THROW", is_primitive=True, args={"slot": missile_slot, "delta": (dr, dc)})
                # Elbereth ward: engrave once, then wait for monsters to flee
                if self.elbereth_turns > 0 and cur_pos == self.elbereth_pos:
                    self._prev_hp = blstats.hp
                    return Task("WAIT", is_primitive=True)
                if self.elbereth_turns <= 0 and self.elbereth_cooldown <= 0:
                    permanent_slot = self._find_athame_or_permanent_engraver(inv_tracker)
                    self.elbereth_pos = cur_pos
                    self.elbereth_turns = 12 if permanent_slot else 8
                    self.elbereth_waited_turns = 0
                    args = {"text": "Elbereth"}
                    if permanent_slot:
                        args["slot"] = permanent_slot
                    self._prev_hp = blstats.hp
                    return Task("ENGRAVE_DUST", is_primitive=True, args=args)
                # Retreat maximizing separation from ALL adjacent monsters
                step_away = self._find_escape_step(py, px, adjacent_monsters, chars, glyphs=glyphs)
                if step_away is not None:
                    self._prev_hp = blstats.hp
                    return Task("STEP", is_primitive=True, args={"delta": step_away})
                # Cornered with zero alternatives: melee is better than standing idle

            # Standard melee strike: attacking erases Elbereth
            self.elbereth_pos = None
            self.elbereth_turns = 0
            self.elbereth_waited_turns = 0
            self._prev_hp = blstats.hp
            return Task(
                name="MELEE_ATTACK",
                is_primitive=True,
                args={"delta": (dr, dc)},
            )

        # 4. Ranged Combat (Chebyshev distance 2 to 6)
        if inv_tracker is not None:
            sorted_ranged = sorted(
                [m for m in monsters if 2 <= m.distance <= 6],
                key=lambda m: (-m.is_instakill, -m.threat_score, m.distance)
            )
            if sorted_ranged:
                walkable_los = {ord("."), ord("#"), ord("+"), ord("'"), ord("<"), ord(">"), ord(" "), ord("{")}
                offensive_wand = self._find_offensive_wand_slot(inv_tracker)
                missile_slot, is_quivered = self._find_quivered_or_missile(inv_tracker)

                for rm in sorted_ranged:
                    dr = rm.pos[0] - py
                    dc = rm.pos[1] - px
                    is_cardinal = (dr == 0 and dc != 0) or (dr != 0 and dc == 0)
                    is_diagonal = (abs(dr) == abs(dc) and dr != 0)
                    if is_cardinal or is_diagonal:
                        sr = 0 if dr == 0 else (1 if dr > 0 else -1)
                        sc = 0 if dc == 0 else (1 if dc > 0 else -1)
                        cur_r, cur_c = py + sr, px + sc
                        los_clear = True
                        while (cur_r, cur_c) != rm.pos:
                            if chars[cur_r, cur_c] not in walkable_los:
                                los_clear = False
                                break
                            cur_r += sr
                            cur_c += sc
                        if los_clear:
                            # If instakill or high threat and offensive wand available:
                            if (rm.is_instakill or rm.threat_score >= 15.0) and offensive_wand is not None:
                                self.elbereth_pos = None
                                self.elbereth_turns = 0
                                return Task(
                                    name="ZAP_WAND",
                                    is_primitive=True,
                                    args={"slot": offensive_wand, "delta": (sr, sc)},
                                )
                            # If quivered missile available, fire from quiver
                            if is_quivered:
                                self.elbereth_pos = None
                                self.elbereth_turns = 0
                                return Task("FIRE", is_primitive=True, args={"delta": (sr, sc)})
                            # Otherwise throw missile
                            if missile_slot is not None:
                                self.elbereth_pos = None
                                self.elbereth_turns = 0
                                return Task(
                                    name="THROW",
                                    is_primitive=True,
                                    args={"slot": missile_slot, "delta": (sr, sc)},
                                )

        # 4.5 Ranged Wand Threat Cover Evasion (when lacking Reflection)
        if not has_reflection:
            wand_threats = [
                m for m in monsters
                if 2 <= m.distance <= 6 and any(w in m.name for w in self.WAND_WIELDER_NAMES)
            ]
            for wm in wand_threats:
                dr = wm.pos[0] - py
                dc = wm.pos[1] - px
                is_aligned = (dr == 0 and dc != 0) or (dr != 0 and dc == 0) or (abs(dr) == abs(dc) and dr != 0)
                if is_aligned:
                    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
                    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
                    cur_r, cur_c = py + sr, px + sc
                    los_clear = True
                    walkable_los = {ord("."), ord("#"), ord("+"), ord("'"), ord("<"), ord(">"), ord(" "), ord("{")}
                    while (cur_r, cur_c) != wm.pos:
                        if chars[cur_r, cur_c] not in walkable_los:
                            los_clear = False
                            break
                        cur_r += sr
                        cur_c += sc

                    if los_clear:
                        # Direct line of fire! Duck into cover that breaks ray line of sight!
                        cover_step = self._find_cover_step(py, px, wm.pos, chars, {m.pos for m in monsters})
                        if cover_step is not None:
                            self.elbereth_pos = None
                            self.elbereth_turns = 0
                            self._prev_hp = blstats.hp
                            return Task("STEP", is_primitive=True, args={"delta": cover_step})

        # 5. Non-adjacent monsters (Distance >= 2) - Speed differential constraint
        closest_monster = monsters[0]
        speed_delta = self.player_speed - closest_monster.speed
        if speed_delta < 0 and closest_monster.distance == 2:
            # Bounded wait (stall fix): fast monsters that hover at distance 2 (leprechauns,
            # giant bats) never close and never leave — waiting forever deadlocks the agent.
            if getattr(self, "_fast_wait_pos", None) != closest_monster.pos:
                self._fast_wait_pos = closest_monster.pos
                self._fast_wait_streak = 0
            self._fast_wait_streak = getattr(self, "_fast_wait_streak", 0) + 1
            if self._fast_wait_streak <= 6:
                self._prev_hp = blstats.hp
                return Task("WAIT", is_primitive=True)
            self._fast_wait_streak = 0
        else:
            self._fast_wait_pos = None
            self._fast_wait_streak = 0

        self._prev_hp = blstats.hp
        return None

    def _find_escape_step(
        self,
        py: int,
        px: int,
        threats: list[MonsterTrack],
        chars: np.ndarray,
        glyphs: np.ndarray | None = None,
    ) -> tuple[int, int] | None:
        """Finds an adjacent walkable cell that does not move closer to threats."""
        threat_positions = {t.pos for t in threats}
        walkable_chars = {ord("."), ord("#"), ord("+"), ord("'"), ord("<"), ord(">")}

        scored: list[tuple[int, tuple[int, int]]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = py + dr, px + dc
                if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                    if (nr, nc) in threat_positions:
                        continue
                    if chars[nr, nc] not in walkable_chars:
                        continue
                    if chars[nr, nc] == ord("@") and (nr, nc) != (py, px):
                        continue
                    if (nr, nc) in self.peaceful_positions:
                        continue
                    if glyphs is not None:
                        g = int(glyphs[nr, nc])
                        if nethack.glyph_is_monster(g) or nethack.glyph_is_pet(g):
                            continue
                    # Doorway diagonal interlock: forbidden into/out of door tiles
                    if dr != 0 and dc != 0:
                        if chars[py, px] in (ord("+"), ord("'")) or chars[nr, nc] in (ord("+"), ord("'")):
                            continue
                        # Corner clip interlock: diagonal step through wall corner forbidden
                        if chars[py + dr, px] not in walkable_chars or chars[py, px + dc] not in walkable_chars:
                            continue
                    # Score by resulting Chebyshev distance to the nearest threat —
                    # pick the tile that MAXIMIZES separation so slow pursuers
                    # (gas spores, molds) actually fall behind instead of the hero
                    # oscillating between two tiles (escape-oscillation fix)
                    min_dist = min(
                        max(abs(nr - tr), abs(nc - tc)) for (tr, tc) in threat_positions
                    ) if threat_positions else 0
                    scored.append((min_dist, (dr, dc)))

        if scored:
            scored.sort(key=lambda s: -s[0])
            return scored[0][1]
        return None

    def _find_funnel_step(
        self,
        py: int,
        px: int,
        chars: np.ndarray,
        threat_positions: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """
        Finds an adjacent walkable corridor ('#') or doorway ('+' or '\'')
        to funnel multiple adjacent enemies into a 1-on-1 choke point.
        """
        choke_chars = {ord("#"), ord("+"), ord("'")}
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                # NetHack forbids diagonal movement through doorways
                is_diagonal = (dr != 0 and dc != 0)
                nr, nc = py + dr, px + dc
                if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                    tile = chars[nr, nc]
                    if (nr, nc) not in threat_positions and tile in choke_chars:
                        if is_diagonal and tile in (ord("+"), ord("'")):
                            continue  # Doorways cannot be entered diagonally
                        return (dr, dc)
        return None

    def _evaluate_door_closing(
        self,
        chars: np.ndarray,
        py: int,
        px: int,
        monsters: list[MonsterTrack],
    ) -> Task | None:
        """
        If hero is adjacent to an open door ('\'') with hostiles pursuing on the
        other side, close the door to break line of sight and halt pursuit!
        """
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = py + dr, px + dc
            if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                if chars[nr, nc] == ord("'"):
                    has_enemy_beyond = any(
                        abs(m.pos[0] - nr) + abs(m.pos[1] - nc) == 1
                        and m.pos != (py, px)
                        and not m.is_adjacent
                        for m in monsters
                    )
                    if has_enemy_beyond:
                        return Task("CLOSE_DOOR", is_primitive=True, args={"delta": (dr, dc)})
        return None

    def _find_grid_bug_diagonal_step(
        self,
        py: int,
        px: int,
        bug_pos: tuple[int, int],
        chars: np.ndarray,
        threat_positions: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """
        Finds an adjacent walkable step that positions the hero diagonal to the grid bug,
        where the grid bug cannot attack.
        """
        walkable_chars = {ord("."), ord("#"), ord("+"), ord("<"), ord(">")}
        by, bx = bug_pos
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = py + dr, px + dc
                if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                    if (nr, nc) not in threat_positions and chars[nr, nc] in walkable_chars:
                        # Check if from (nr, nc), the bug at (by, bx) is diagonal
                        d_bug_r = abs(by - nr)
                        d_bug_c = abs(bx - nc)
                        if d_bug_r == 1 and d_bug_c == 1:
                            return (dr, dc)
        return None

    def _find_cover_step(
        self,
        py: int,
        px: int,
        threat_pos: tuple[int, int],
        chars: np.ndarray,
        occupied_positions: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        """
        Finds an adjacent walkable step that breaks cardinal and diagonal line of sight
        to a ranged wand-wielding threat, ducking behind a corridor corner or door.
        """
        walkable_chars = {ord("."), ord("#"), ord("+"), ord("'"), ord("<"), ord(">")}
        ty, tx = threat_pos
        candidates = []

        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = py + dr, px + dc
                if not (0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]):
                    continue
                if (nr, nc) in occupied_positions or chars[nr, nc] not in walkable_chars:
                    continue
                # NetHack forbids diagonal doorway moves
                if dr != 0 and dc != 0 and chars[nr, nc] in (ord("+"), ord("'")):
                    continue

                # Check if from (nr, nc) to (ty, tx), line of sight is broken or not aligned
                delta_r = ty - nr
                delta_c = tx - nc
                is_cardinal = (delta_r == 0 and delta_c != 0) or (delta_r != 0 and delta_c == 0)
                is_diagonal = (abs(delta_r) == abs(delta_c) and delta_r != 0)

                # Not aligned cardinally or diagonally: ray wands cannot hit!
                if not (is_cardinal or is_diagonal):
                    candidates.append((dr, dc))
                    continue

                # If aligned, check if a wall blocks line of sight
                sr = 0 if delta_r == 0 else (1 if delta_r > 0 else -1)
                sc = 0 if delta_c == 0 else (1 if delta_c > 0 else -1)
                curr_r, curr_c = nr + sr, nc + sc
                wall_blocked = False
                while (curr_r, curr_c) != (ty, tx):
                    if chars[curr_r, curr_c] not in walkable_chars:
                        wall_blocked = True
                        break
                    curr_r += sr
                    curr_c += sc
                if wall_blocked:
                    candidates.append((dr, dc))

        if candidates:
            # Prefer steps into corridors or doorways (natural choke points)
            for step in candidates:
                nr, nc = py + step[0], px + step[1]
                if chars[nr, nc] in (ord("#"), ord("+"), ord("'")):
                    return step
            return candidates[0]
        return None

