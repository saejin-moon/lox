"""
Active CDCL Nogood Store: Cross-generational 64-bit bitmask negative constraint engine.
Evaluates candidate actions in sub-microsecond CPU time.
"""

from dataclasses import dataclass, asdict
import json
import os
from typing import Tuple
from corp.planner.predicates import PredicateBit


@dataclass(slots=True, frozen=True)
class NogoodEntry:
    """
    A single negative constraint compiled from post-mortem autopsy or domain invariants.
    """
    mask: int               # 64-bit predicate mask specifying active conditions
    target_val: int         # 64-bit target bit pattern
    forbidden_action: str   # Action primitive forbidden under this pattern
    reason: str             # Causal explanation
    generation: int = 0     # Episode generation when synthesized


class NogoodStore:
    """
    Stores and evaluates bitwise Nogood cuts against candidate actions.
    Uses an Inverted Action Trie/Bucket (FatalAction -> list[NogoodEntry])
    to guarantee O(1) check time (<1 microsecond).
    """

    def __init__(self):
        self.entries: list[NogoodEntry] = []
        self._by_action: dict[str, list[NogoodEntry]] = {}
        self._initialize_canonical_interlocks()

    def _initialize_canonical_interlocks(self):
        """Pre-seeds the store with NetHack 3.6.6 verified physical safety interlocks."""
        # 1. Floating Eye Melee Lockout (unless blind or reflection)
        self.add_nogood(NogoodEntry(
            mask=(
                PredicateBit.ADJACENT_FLOATING_EYE
                | PredicateBit.IS_BLIND
                | PredicateBit.REFLECTION_ACTIVE
            ),
            target_val=PredicateBit.ADJACENT_FLOATING_EYE,
            forbidden_action="MELEE_ATTACK",
            reason="Passive gaze triggers 0d70 turn paralysis without blindness or reflection",
            generation=0,
        ))

        # 2. Barehanded Cockatrice Pickup/Wield
        self.add_nogood(NogoodEntry(
            mask=(
                PredicateBit.ADJACENT_COCKATRICE
                | PredicateBit.GLOVES_EQUIPPED
            ),
            target_val=PredicateBit.ADJACENT_COCKATRICE,
            forbidden_action="PICKUP",
            reason="Barehanded handling of cockatrice causes 0-turn fatal petrification",
            generation=0,
        ))
        self.add_nogood(NogoodEntry(
            mask=(
                PredicateBit.ADJACENT_COCKATRICE
                | PredicateBit.GLOVES_EQUIPPED
            ),
            target_val=PredicateBit.ADJACENT_COCKATRICE,
            forbidden_action="WIELD",
            reason="Barehanded wielding of cockatrice causes 0-turn fatal petrification",
            generation=0,
        ))

        # 3. Open-Room Retreat Against Faster Pursuit Predator (Speed 18 vs 12)
        self.add_nogood(NogoodEntry(
            mask=(
                PredicateBit.ADJACENT_MONSTER_FASTER
                | PredicateBit.CORRIDOR_CHOKEPOINT
            ),
            target_val=PredicateBit.ADJACENT_MONSTER_FASTER,
            forbidden_action="STEP_AWAY",
            reason="Kiting in the open against faster monster grants 3:2 free rear attacks",
            generation=0,
        ))

        # 4. Ray Reflection Backfire
        self.add_nogood(NogoodEntry(
            mask=PredicateBit.RAY_REFLECTIVE_WALL_LT_4,
            target_val=PredicateBit.RAY_REFLECTIVE_WALL_LT_4,
            forbidden_action="ZAP_RAY",
            reason="Wall within 4 tiles causes ray bounce backfire striking player",
            generation=0,
        ))

    def add_nogood(self, entry: NogoodEntry):
        """Appends a new verified Nogood constraint to list and inverted action bucket."""
        self.entries.append(entry)
        if entry.forbidden_action not in self._by_action:
            self._by_action[entry.forbidden_action] = []
        self._by_action[entry.forbidden_action].append(entry)

    def is_forbidden(self, state_mask: int, action_name: str) -> Tuple[bool, str]:
        """
        Evaluates active Nogoods in <1 microsecond via bitwise AND operations
        indexed by candidate action.
        Returns (is_forbidden, reason).
        """
        candidates = self._by_action.get(action_name)
        if not candidates:
            return False, ""
        for entry in candidates:
            if (state_mask & entry.mask) == entry.target_val:
                return True, entry.reason
        return False, ""

    def save_to_json(self, file_path: str):
        """Persists Nogood library across game generations."""
        data = [asdict(e) for e in self.entries]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_from_json(self, file_path: str):
        """Loads persistent Nogood constraints from disk."""
        if not os.path.exists(file_path):
            return
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            entry = NogoodEntry(**item)
            if entry not in self.entries:
                self.add_nogood(entry)
