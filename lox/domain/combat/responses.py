"""Extracted from lox/domain/combat_manager.py — behavior-preserving split."""

from dataclasses import dataclass
from typing import Any
import numpy as np
from nle import nethack

from lox.env.blstats import BottomLineStats
from lox.policy.config import PolicyConfig, default_config
from lox.env.blstats import ConditionFlag, HungerState
from lox.planner.htn import Task, PrimitiveTask
from lox.domain.combat.threat_scan import MonsterTrack


class CombatResponsesMixin:
    def _ranged_attack_response(self, blstats, inv_tracker, py, px, primary_target):
        """Wand → quivered fire → thrown missile toward the primary target (elbereth-
        erasing, as in the pre-R4 ranged paths). Returns None when unarmed."""
        if inv_tracker is None:
            return None
        dr = primary_target.pos[0] - py
        dc = primary_target.pos[1] - px
        offensive_wand = self._find_offensive_wand_slot(inv_tracker)
        if offensive_wand is not None:
            self.elbereth_pos = None
            self.elbereth_turns = 0
            return Task("ZAP_WAND", is_primitive=True, args={"slot": offensive_wand, "delta": (dr, dc)})
        missile_slot, is_quivered = self._find_quivered_or_missile(inv_tracker)
        if is_quivered:
            self.elbereth_pos = None
            self.elbereth_turns = 0
            return Task("FIRE", is_primitive=True, args={"delta": (dr, dc)})
        if missile_slot is not None:
            self.elbereth_pos = None
            self.elbereth_turns = 0
            return Task("THROW", is_primitive=True, args={"slot": missile_slot, "delta": (dr, dc)})
        return None

    def _defensive_kite_response(self, blstats, inv_tracker, cur_pos, py, px,
                                 adjacent_monsters, chars, glyphs, primary_target):
        """The pre-R4 lethal-poison/heavy-hitter response, verbatim: wait on an active
        Elbereth ward → engrave → escape step → cornered ranged attack."""
        if self.elbereth_turns > 0 and cur_pos == self.elbereth_pos:
            return Task("WAIT", is_primitive=True)
        if self.elbereth_turns <= 0 and self.elbereth_cooldown <= 0:
            permanent_slot = self._find_athame_or_permanent_engraver(inv_tracker)
            self.elbereth_pos = cur_pos
            self.elbereth_turns = self.cfg.combat.elbereth_permanent_turns if permanent_slot else self.cfg.combat.elbereth_dust_turns
            self.elbereth_waited_turns = 0
            args = {"text": "Elbereth"}
            if permanent_slot:
                args["slot"] = permanent_slot
            return Task("ENGRAVE_DUST", is_primitive=True, args=args)
        step_away = self._find_escape_step(py, px, adjacent_monsters, chars, glyphs=glyphs)
        if step_away is not None:
            return Task("STEP", is_primitive=True, args={"delta": step_away})
        if inv_tracker is not None:
            offensive_wand = self._find_offensive_wand_slot(inv_tracker)
            if offensive_wand is not None:
                self.elbereth_pos = None
                self.elbereth_turns = 0
                dr = primary_target.pos[0] - py
                dc = primary_target.pos[1] - px
                return Task("ZAP_WAND", is_primitive=True, args={"slot": offensive_wand, "delta": (dr, dc)})
            missile_slot, is_quivered = self._find_quivered_or_missile(inv_tracker)
            if is_quivered:
                dr = primary_target.pos[0] - py
                dc = primary_target.pos[1] - px
                return Task("FIRE", is_primitive=True, args={"delta": (dr, dc)})
            if missile_slot is not None:
                dr = primary_target.pos[0] - py
                dc = primary_target.pos[1] - px
                return Task("THROW", is_primitive=True, args={"slot": missile_slot, "delta": (dr, dc)})
        return None

    def _verb_response(self, verb, blstats, inv_tracker, cur_pos, py, px,
                       adjacent_monsters, chars, glyphs, primary_target):
        """Maps a matched tactic verb to a concrete Task. Returns RULE_FALLTHROUGH when
        the verb's condition is unmet this turn (e.g. unwounded vs retreat_when_wounded,
        unarmed ranged_then_kill) so the default combat flow resumes."""
        if verb == "elbereth_first":
            return self._defensive_kite_response(
                blstats, inv_tracker, cur_pos, py, px, adjacent_monsters, chars, glyphs, primary_target)
        if verb == "retreat_when_wounded":
            cb = self.cfg.combat
            if blstats.hp <= max(cb.heavy_wounded_min_hp, int(blstats.max_hp * cb.heavy_wounded_frac)):
                return self._defensive_kite_response(
                    blstats, inv_tracker, cur_pos, py, px, adjacent_monsters, chars, glyphs, primary_target)
            return self.RULE_FALLTHROUGH
        if verb in ("ranged_only", "never_melee"):
            # NEVER melee: ranged → escape → hold ground (interlock-safe by construction)
            t = self._ranged_attack_response(blstats, inv_tracker, py, px, primary_target)
            if t is not None:
                return t
            step_away = self._find_escape_step(py, px, adjacent_monsters, chars, glyphs=glyphs)
            if step_away is not None:
                return Task("STEP", is_primitive=True, args={"delta": step_away})
            return Task("WAIT", is_primitive=True)
        if verb == "ranged_then_kill":
            t = self._ranged_attack_response(blstats, inv_tracker, py, px, primary_target)
            return t if t is not None else self.RULE_FALLTHROUGH
        if verb == "kite":
            t = self._ranged_attack_response(blstats, inv_tracker, py, px, primary_target)
            if t is not None:
                return t
            step_away = self._find_escape_step(py, px, adjacent_monsters, chars, glyphs=glyphs)
            if step_away is not None:
                return Task("STEP", is_primitive=True, args={"delta": step_away})
            return self.RULE_FALLTHROUGH
        if verb in ("retreat", "avoid"):
            step_away = self._find_escape_step(py, px, adjacent_monsters, chars, glyphs=glyphs)
            if step_away is not None:
                return Task("STEP", is_primitive=True, args={"delta": step_away})
            # Never engage: hold ground rather than melee
            return Task("WAIT", is_primitive=True)
        return self.RULE_FALLTHROUGH

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

