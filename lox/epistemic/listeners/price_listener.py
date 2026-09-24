"""
PriceIDListener: NetHack 3.6.6 merchant pricing model for candidate elimination.
Decouples buying (Charisma-dependent) from selling (Charisma-independent).
"""

from typing import Iterable
from lox.epistemic.belief_state import ItemBeliefState


class PriceIDListener:
    """
    Inverts shopkeeper pricing to derive candidate item base prices
    and eliminates incompatible candidate identities.
    """

    CHA_MULTIPLIERS: dict[str, tuple[int, int]] = {
        # Charisma range -> (numerator, denominator)
        "low": (2, 1),       # <= 5: 2.0x
        "poor": (3, 2),      # 6-7: 1.5x
        "below_avg": (4, 3), # 8-10: 1.33x
        "avg": (1, 1),       # 11-15: 1.0x
        "good": (3, 4),      # 16-17: 0.75x
        "high": (2, 3),      # 18: 0.67x
        "max": (1, 2),       # >= 19: 0.5x
    }

    @classmethod
    def get_cha_ratio(cls, cha: int) -> tuple[int, int]:
        if cha <= 5:
            return (2, 1)
        elif cha <= 7:
            return (3, 2)
        elif cha <= 10:
            return (4, 3)
        elif cha <= 15:
            return (1, 1)
        elif cha <= 17:
            return (3, 4)
        elif cha == 18:
            return (2, 3)
        else:
            return (1, 2)

    @classmethod
    def invert_sale_price(cls, offer_price: int, is_sucker: bool = False) -> set[int]:
        """
        Derives possible base prices from shopkeeper's purchase offer (player selling item).
        Selling does NOT depend on charisma.
        """
        possible_bases = set()
        p = offer_price

        # Standard: offer is base // 2
        for rem in range(2):
            possible_bases.add(2 * p + rem)

        # Sucker: offer is base // 3
        if is_sucker:
            for rem in range(3):
                possible_bases.add(3 * p + rem)

        # Unidentified 25% discount (offer is 3/8 base)
        base_38 = round(p * 8 / 3)
        possible_bases.add(base_38)
        possible_bases.add(base_38 - 1)
        possible_bases.add(base_38 + 1)

        return {b for b in possible_bases if b > 0}

    @classmethod
    def invert_buy_price(cls, asking_price: int, cha: int, is_sucker: bool = False) -> set[int]:
        """
        Derives possible base prices from shopkeeper's asking price (player buying item).
        """
        possible_bases = set()
        num, den = cls.get_cha_ratio(cha)

        # Base nominal calculation: asking = base * (num/den)
        nominal_base = round(asking_price * den / num)
        for delta in (-2, -1, 0, 1, 2):
            if nominal_base + delta > 0:
                possible_bases.add(nominal_base + delta)

        # Sucker (4/3 multiplier)
        if is_sucker:
            sucker_base = round(asking_price * den * 3 / (num * 4))
            for delta in (-2, -1, 0, 1, 2):
                if sucker_base + delta > 0:
                    possible_bases.add(sucker_base + delta)

        # Unidentified item surcharge (4/3 multiplier applied in 1/3 of items)
        unid_base = round(asking_price * den * 3 / (num * 4))
        for delta in (-2, -1, 0, 1, 2):
            if unid_base + delta > 0:
                possible_bases.add(unid_base + delta)

        return possible_bases

    @classmethod
    def filter_candidates_by_price(
        cls,
        item_belief: ItemBeliefState,
        candidate_price_map: dict[str, int],
        observed_price: int,
        is_selling: bool = True,
        cha: int = 10,
        is_sucker: bool = False,
    ) -> list[str]:
        """
        Prunes candidate identities whose base cost does not match the inverted price range.
        """
        if is_selling:
            valid_bases = cls.invert_sale_price(observed_price, is_sucker=is_sucker)
        else:
            valid_bases = cls.invert_buy_price(observed_price, cha=cha, is_sucker=is_sucker)

        surviving = set()
        for cand in item_belief.candidate_identities:
            base_cost = candidate_price_map.get(cand)
            if base_cost is not None and base_cost in valid_bases:
                surviving.add(cand)

        if surviving:
            item_belief.prune_candidates(surviving)

        return item_belief.candidate_identities
