"""
Domain Layer: Economy, Shopkeeper & Temple Priest Subsystem.
Implements:
1. Shop detection (messages, unpaid items, shopkeeper presence).
2. Base price inversion and Price-ID for scrolls, potions, and wands.
3. Automated shop price probing (dropping unknown items, parsing buy/sell quotes).
4. Temple priest divine protection donations (400 * XL gold for -10 AC).
"""

import re
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from corp.env.blstats import BottomLineStats
from corp.planner.htn import Task, PrimitiveTask


@dataclass
class PriceTier:
    """Base price tier and candidate item identities for NetHack 3.6.6."""
    base_price: int
    sell_range_normal: tuple[int, int]
    sell_range_greedy: tuple[int, int]
    candidates: list[str]


# Verified NetHack 3.6.6 Base Price Tables
SCROLL_PRICE_TIERS = [
    PriceTier(20, (10, 10), (7, 8), ["scroll of identify"]),
    PriceTier(50, (25, 25), (18, 19), ["scroll of light"]),
    PriceTier(60, (30, 30), (22, 23), ["scroll of blank paper"]),
    PriceTier(80, (40, 40), (30, 30), ["scroll of enchant weapon", "scroll of remove curse"]),
    PriceTier(100, (50, 50), (37, 38), [
        "scroll of destroy armor", "scroll of fire", "scroll of food detection",
        "scroll of gold detection", "scroll of magic mapping", "scroll of scare monster",
        "scroll of teleportation",
    ]),
    PriceTier(200, (100, 100), (75, 75), [
        "scroll of create monster", "scroll of amnesia", "scroll of earth", "scroll of taming",
    ]),
    PriceTier(300, (150, 150), (112, 113), ["scroll of genocide", "scroll of charging"]),
]

POTION_PRICE_TIERS = [
    PriceTier(50, (25, 25), (18, 19), [
        "potion of booze", "potion of fruit juice", "potion of see invisible", "potion of sickness",
    ]),
    PriceTier(100, (50, 50), (37, 38), [
        "potion of confusion", "potion of cure blindness", "potion of extra healing",
        "potion of hallucination", "potion of restore ability", "potion of sleeping",
    ]),
    PriceTier(150, (75, 75), (56, 57), [
        "potion of blindness", "potion of gain energy", "potion of invisibility",
        "potion of monster detection", "potion of object detection",
    ]),
    PriceTier(200, (100, 100), (75, 75), [
        "potion of enlightenment", "potion of full healing", "potion of levitation",
        "potion of polymorph", "potion of speed",
    ]),
    PriceTier(250, (125, 125), (93, 94), ["potion of acid", "potion of oil"]),
    PriceTier(300, (150, 150), (112, 113), ["potion of gain ability", "potion of gain level"]),
]

WAND_PRICE_TIERS = [
    PriceTier(100, (50, 50), (37, 38), ["wand of light", "wand of secret door detection"]),
    PriceTier(150, (75, 75), (56, 57), [
        "wand of digging", "wand of magic missile", "wand of opening",
        "wand of probing", "wand of slow monster", "wand of speed monster", "wand of undead turning",
    ]),
    PriceTier(175, (87, 88), (65, 66), [
        "wand of cold", "wand of fire", "wand of lightning", "wand of sleep",
    ]),
    PriceTier(200, (100, 100), (75, 75), [
        "wand of cancellation", "wand of create monster", "wand of locking",
        "wand of make invisible", "wand of nothing", "wand of polymorph",
        "wand of striking", "wand of teleportation",
    ]),
    PriceTier(500, (250, 250), (187, 188), ["wand of death", "wand of wishing"]),
]


class ShopManager:
    """
    Manages commercial and religious economy interactions:
    - Shop perception & boundary tracking
    - Price-ID derivation for unknown consumable stacks
    - Temple priest donations for intrinsic protection
    """

    def __init__(self):
        self.in_shop: bool = False
        self.shop_type: str = "general"
        self.shopkeeper_pos: tuple[int, int] | None = None
        self.priest_pos: tuple[int, int] | None = None
        self.priest_donations_count: int = 0
        self.last_tested_item_slot: str | None = None
        self.last_price_quote: int | None = None
        self.shop_test_cooldown: int = 0
        self.identified_by_price: dict[str, str] = {}  # item description -> resolved name

    def reset(self):
        self.in_shop = False
        self.shop_type = "general"
        self.shopkeeper_pos = None
        self.priest_pos = None
        self.priest_donations_count = 0
        self.last_tested_item_slot = None
        self.last_price_quote = None
        self.shop_test_cooldown = 0
        self.identified_by_price.clear()

    def update_perception(
        self,
        blstats: BottomLineStats,
        message: str = "",
        chars: np.ndarray | None = None,
        glyphs: np.ndarray | None = None,
    ):
        """Processes turn messages and spatial cues for shopkeeper and temple priest presence."""
        msg_lower = message.lower()
        if self.shop_test_cooldown > 0:
            self.shop_test_cooldown -= 1

        # Detect shop presence
        if "welcome to" in msg_lower and "shop" in msg_lower:
            self.in_shop = True
            if "general store" in msg_lower:
                self.shop_type = "general"
            elif "delicatessen" in msg_lower:
                self.shop_type = "deli"
            elif "book" in msg_lower:
                self.shop_type = "books"
            elif "hardware" in msg_lower:
                self.shop_type = "hardware"

        if "(unpaid" in msg_lower:
            self.in_shop = True

        # Parse sell offer quote: "The shopkeeper offers you 10 gold pieces for it."
        # or "offers 10 gold"
        offer_match = re.search(r"offers (?:you )?(\d+) gold", msg_lower)
        if offer_match:
            offer = int(offer_match.group(1))
            self.last_price_quote = offer
            if self.last_tested_item_slot:
                self._resolve_tested_slot(offer)

        # Detect temple priest presence
        if "shrine of" in msg_lower or "temple" in msg_lower or "priest of" in msg_lower:
            # Locate adjacent or visible human peaceful cleric (@)
            if chars is not None:
                py, px = blstats.y, blstats.x
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = py + dr, px + dc
                        if 0 <= nr < chars.shape[0] and 0 <= nc < chars.shape[1]:
                            if chars[nr, nc] == ord("@"):
                                self.priest_pos = (nr, nc)

    def _resolve_tested_slot(self, offer: int):
        """Maps sell offer to exact base price candidates."""
        # Check scrolls
        candidates = self.resolve_scroll_price(offer)
        if candidates and len(candidates) == 1:
            self.identified_by_price[self.last_tested_item_slot] = candidates[0]

    @staticmethod
    def resolve_scroll_price(offer: int) -> list[str]:
        """Resolves scroll identities from shopkeeper sell offer in NetHack 3.6.6."""
        for tier in SCROLL_PRICE_TIERS:
            norm_min, norm_max = tier.sell_range_normal
            gr_min, gr_max = tier.sell_range_greedy
            if (norm_min <= offer <= norm_max) or (gr_min <= offer <= gr_max):
                return list(tier.candidates)
        return []

    @staticmethod
    def resolve_potion_price(offer: int) -> list[str]:
        """Resolves potion identities from shopkeeper sell offer in NetHack 3.6.6."""
        for tier in POTION_PRICE_TIERS:
            norm_min, norm_max = tier.sell_range_normal
            gr_min, gr_max = tier.sell_range_greedy
            if (norm_min <= offer <= norm_max) or (gr_min <= offer <= gr_max):
                return list(tier.candidates)
        return []

    @staticmethod
    def resolve_wand_price(offer: int) -> list[str]:
        """Resolves wand identities from shopkeeper sell offer in NetHack 3.6.6."""
        for tier in WAND_PRICE_TIERS:
            norm_min, norm_max = tier.sell_range_normal
            gr_min, gr_max = tier.sell_range_greedy
            if (norm_min <= offer <= norm_max) or (gr_min <= offer <= gr_max):
                return list(tier.candidates)
        return []

    def evaluate_temple_donation(
        self,
        blstats: BottomLineStats,
        chars: np.ndarray,
        has_adjacent_hostiles: bool = False,
    ) -> Task | None:
        """
        Donates 400 * XL gold to an aligned temple priest to acquire permanent intrinsic AC protection.
        NetHack 3.6.6 grants 2-4 points of protection per successful donation!
        """
        if has_adjacent_hostiles:
            return None

        # Protection formula threshold: 400 * XL
        required_gold = 400 * max(1, blstats.experience_level)
        if blstats.gold < required_gold:
            return None

        if self.priest_donations_count >= 5:
            # Donating 5 times provides ~10 points of divine protection, capping early AC benefits
            return None

        py, px = blstats.y, blstats.x
        # Check if priest is adjacent
        if self.priest_pos is not None:
            pr, pc = self.priest_pos
            dr, dc = pr - py, pc - px
            if abs(dr) <= 1 and abs(dc) <= 1 and (dr != 0 or dc != 0):
                self.priest_donations_count += 1
                return Task(
                    "CHAT",
                    is_primitive=True,
                    args={"delta": (dr, dc), "amount": required_gold},
                )

        return None

    def evaluate_shop_probing(
        self,
        blstats: BottomLineStats,
        inv_tracker: Any,
        chars: np.ndarray,
        has_adjacent_hostiles: bool = False,
    ) -> Task | None:
        """
        Probes shop prices by dropping an unidentified scroll, potion, or wand to query sell price.
        """
        if not self.in_shop or has_adjacent_hostiles or self.shop_test_cooldown > 0:
            return None
        if inv_tracker is None:
            return None

        # Find unidentified scroll or potion
        for slot, item in inv_tracker.active_items.items():
            desc = item.raw_str.lower()
            if "scroll labeled" in desc or "unlabeled scroll" in desc or "potion" in desc:
                if slot not in self.identified_by_price:
                    self.last_tested_item_slot = slot
                    self.shop_test_cooldown = 15
                    # Issue DROP to trigger shopkeeper quote
                    return Task("DROP", is_primitive=True, args={"slot": slot})

        return None

    def has_unpaid_items(self, inv_tracker: Any) -> bool:
        """Returns True if hero carries any unpaid merchandise."""
        if inv_tracker is None:
            return False
        active_items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
        return any("(unpaid" in it.raw_str.lower() for it in active_items)

    def evaluate_shop_payment(
        self,
        blstats: BottomLineStats,
        inv_tracker: Any,
        message: str = "",
    ) -> Task | None:
        """
        Safely resolves unpaid items in shops:
        - If holding unpaid items and hero has gold: issues Task("PAY") to complete purchase.
        - If holding unpaid items and hero has 0 gold: issues Task("DROP", slot=item.slot) to return item safely to avoid shopkeeper wrath.
        """
        if inv_tracker is None:
            return None

        active_items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
        unpaid_items = [
            it for it in active_items
            if "(unpaid" in it.raw_str.lower()
        ]
        if not unpaid_items:
            return None

        # If hero has gold, execute PAY
        if blstats.gold > 0:
            return Task("PAY", is_primitive=True)

        # If broke, drop the first unpaid item to safely relieve debt
        first_unpaid = unpaid_items[0]
        if first_unpaid.current_letter:
            return Task("DROP", is_primitive=True, args={"slot": first_unpaid.current_letter})

        return None
