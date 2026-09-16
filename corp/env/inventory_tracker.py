"""
Permanent Inventory UID Normalizer: Tracks items across volatile NetHack letter shifts.
"""

from dataclasses import dataclass, field
import uuid
import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment


@dataclass
class NormalizedItem:
    """Stable representation of an inventory item decoupled from transient letters."""
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    raw_str: str = ""
    current_letter: str = ""
    glyph: int = 0
    oclass: int = 0
    quantity: int = 1
    equipped: bool = False
    first_seen_turn: int = 0
    last_updated_turn: int = 0
    is_active: bool = True

    # Beatitude state (0=Unknown, 1=Uncursed, 2=Blessed, 3=Cursed)
    buc_state: str = "UNKNOWN"  # UNKNOWN, BLESSED, UNCURSED, CURSED


class InventoryNormalizer:
    """
    Maintains a 1-to-1 persistent mapping from physical game objects to stable UIDs,
    protecting higher-order planners against NetHack letter shifting race conditions.
    """

    def __init__(self):
        self.active_items: dict[str, NormalizedItem] = {}  # Keyed by UID
        self.letter_to_uid: dict[str, str] = {}           # Maps 'a' -> UID
        self.uid_to_letter: dict[str, str] = {}           # Maps UID -> 'a'

    def synchronize(
        self,
        inv_strs: npt.NDArray[np.uint8],
        inv_letters: npt.NDArray[np.uint8],
        inv_glyphs: npt.NDArray[np.int16],
        inv_oclasses: npt.NDArray[np.uint8] | None = None,
        turn: int = 0,
    ) -> dict[str, NormalizedItem]:
        """
        Synchronizes raw NLE inventory observation tensors into stable UIDs
        via bipartite Hungarian matching.
        """
        current_slots: list[dict] = []
        for i in range(len(inv_letters)):
            letter_byte = int(inv_letters[i])
            if letter_byte == 0 or letter_byte == 32:
                continue
            letter = chr(letter_byte)
            raw_bytes = bytes(inv_strs[i])
            desc = raw_bytes.split(b"\x00")[0].decode("latin-1", errors="replace").strip()
            glyph = int(inv_glyphs[i])
            oclass = int(inv_oclasses[i]) if inv_oclasses is not None and i < len(inv_oclasses) else 0

            # Quick heuristic parse
            equipped = "(weapon in hands)" in desc or "(being worn)" in desc or "(wielded)" in desc
            buc = "UNKNOWN"
            if "blessed" in desc:
                buc = "BLESSED"
            elif "cursed" in desc:
                buc = "CURSED"
            elif "uncursed" in desc:
                buc = "UNCURSED"

            current_slots.append({
                "letter": letter,
                "desc": desc,
                "glyph": glyph,
                "oclass": oclass,
                "equipped": equipped,
                "buc": buc,
            })

        # Previous active items
        active_uids = [uid for uid, it in self.active_items.items() if it.is_active]

        # Case 1: If no previous active items, initialize all
        if not active_uids:
            for slot in current_slots:
                new_item = NormalizedItem(
                    raw_str=slot["desc"],
                    current_letter=slot["letter"],
                    glyph=slot["glyph"],
                    oclass=slot["oclass"],
                    equipped=slot["equipped"],
                    buc_state=slot["buc"],
                    first_seen_turn=turn,
                    last_updated_turn=turn,
                )
                self.active_items[new_item.uid] = new_item
            self._rebuild_lookup_tables()
            return self.active_items

        # Case 2: Bipartite Hungarian matching between active_uids and current_slots
        cost_matrix = np.zeros((len(active_uids), len(current_slots)), dtype=float)
        for r, uid in enumerate(active_uids):
            old_item = self.active_items[uid]
            for c, slot in enumerate(current_slots):
                # Matching cost: 0.0 for identical glyph and desc, lower cost for closer matches
                cost = 10.0
                if old_item.glyph == slot["glyph"]:
                    cost -= 4.0
                if old_item.raw_str == slot["desc"]:
                    cost -= 4.0
                elif old_item.raw_str and slot["desc"] and (old_item.raw_str in slot["desc"] or slot["desc"] in old_item.raw_str):
                    cost -= 2.0
                if old_item.current_letter == slot["letter"]:
                    cost -= 2.0
                cost_matrix[r, c] = max(0.0, cost)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_slots = set()
        matched_uids = set()

        for r, c in zip(row_ind, col_ind):
            # If cost <= 6.0, we accept this match (at least glyph or desc matched)
            if cost_matrix[r, c] <= 6.0:
                uid = active_uids[r]
                slot = current_slots[c]
                matched_uids.add(uid)
                matched_slots.add(c)
                # Update item state
                item = self.active_items[uid]
                item.current_letter = slot["letter"]
                item.raw_str = slot["desc"]
                item.glyph = slot["glyph"]
                item.oclass = slot["oclass"]
                item.equipped = slot["equipped"]
                if slot["buc"] != "UNKNOWN":
                    item.buc_state = slot["buc"]
                item.last_updated_turn = turn
                item.is_active = True

        # Invalidate un-matched active items (they were dropped or consumed)
        for uid in active_uids:
            if uid not in matched_uids:
                self.active_items[uid].is_active = False

        # Add newly discovered slots
        for c, slot in enumerate(current_slots):
            if c not in matched_slots:
                new_item = NormalizedItem(
                    raw_str=slot["desc"],
                    current_letter=slot["letter"],
                    glyph=slot["glyph"],
                    oclass=slot["oclass"],
                    equipped=slot["equipped"],
                    buc_state=slot["buc"],
                    first_seen_turn=turn,
                    last_updated_turn=turn,
                )
                self.active_items[new_item.uid] = new_item

        self._rebuild_lookup_tables()
        return self.active_items

    def _rebuild_lookup_tables(self):
        self.letter_to_uid.clear()
        self.uid_to_letter.clear()
        for uid, item in self.active_items.items():
            if item.is_active and item.current_letter:
                self.letter_to_uid[item.current_letter] = uid
                self.uid_to_letter[uid] = item.current_letter

    def get_letter(self, uid: str) -> str | None:
        """Returns the current dynamic slot letter for a persistent UID."""
        return self.uid_to_letter.get(uid)

    def get_by_letter(self, letter: str) -> NormalizedItem | None:
        """Returns the NormalizedItem currently residing at the given slot letter."""
        uid = self.letter_to_uid.get(letter)
        return self.active_items.get(uid) if uid else None

    def get_active_items(self) -> list[NormalizedItem]:
        """Returns all currently held items."""
        return [it for it in self.active_items.values() if it.is_active]
