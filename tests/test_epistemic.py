"""
Unit tests for Phase 2: Epistemic POMDP Engine, Bayesian Listeners,
Shannon Entropy Safe-Gates, and EpistemicManager.
"""

import numpy as np
import pytest

from corp.epistemic.belief_state import ItemBeliefState
from corp.epistemic.listeners.altar_listener import AltarListener
from corp.epistemic.listeners.pet_listener import PetListener
from corp.epistemic.listeners.price_listener import PriceIDListener
from corp.epistemic.listeners.engrave_listener import EngraveTestListener, CursedWandExplosionRiskError
from corp.epistemic.entropy_gates import ShannonSafeGate
from corp.epistemic.epistemic_manager import EpistemicManager
from corp.env.blstats import BottomLineStats, ConditionFlag
from corp.env.inventory_tracker import NormalizedItem


def create_dummy_blstats(blind: bool = False, hallucinating: bool = False) -> BottomLineStats:
    raw = np.zeros(27, dtype=np.int64)
    raw[10] = 20  # HP
    raw[11] = 20  # Max HP
    cond = 0
    if blind:
        cond |= ConditionFlag.BLIND
    if hallucinating:
        cond |= ConditionFlag.HALLU
    raw[25] = cond
    return BottomLineStats.from_blstats(raw)


def test_belief_state_entropy():
    # 4 uniform candidates: H = log2(4) = 2.0 bits
    belief = ItemBeliefState(
        uid="pot1",
        item_class="potion",
        candidate_identities=["healing", "extra healing", "sleeping", "poison"],
    )
    assert np.isclose(belief.entropy_identity(), 2.0)

    # Collapse to 1 candidate: H = 0.0 bits
    belief.prune_candidates({"extra healing"})
    assert np.isclose(belief.entropy_identity(), 0.0)
    assert belief.is_formally_identified()

    # BUC collapse
    belief.collapse_buc("BLESSED")
    assert belief.p_blessed == 1.0
    assert belief.p_cursed == 0.0
    assert np.isclose(belief.entropy_buc(), 0.0)


def test_altar_listener_mechanics():
    bl_normal = create_dummy_blstats(blind=False)
    bl_blind = create_dummy_blstats(blind=True)

    belief = ItemBeliefState(uid="item1")

    # Sensory gate test: Blindness blocks altar inference
    processed = AltarListener.process_drop(belief, bl_blind, "There is an amber flash of light.")
    assert processed is False
    assert belief.p_blessed != 1.0  # Did not update

    # Normal perception: Black light -> Cursed
    processed = AltarListener.process_drop(belief, bl_normal, "There is a flash of black light.")
    assert processed is True
    assert belief.p_cursed == 1.0

    # Amber flash -> Blessed
    belief2 = ItemBeliefState(uid="item2")
    AltarListener.process_drop(belief2, bl_normal, "There is an amber flash.")
    assert belief2.p_blessed == 1.0

    # No flash -> Uncursed
    belief3 = ItemBeliefState(uid="item3")
    AltarListener.process_drop(belief3, bl_normal, "You drop the long sword.")
    assert belief3.p_uncursed == 1.0


def test_pet_listener_and_comestible_exception():
    # Non-food item
    armor_belief = ItemBeliefState(uid="arm1", item_class="armor")
    assert armor_belief.p_cursed == 0.10

    # Pet steps willingly -> Cursed eliminated
    valid = PetListener.process_pet_interaction(armor_belief, pet_stepped_willingly=True)
    assert valid is True
    assert armor_belief.p_cursed == 0.0

    # Pet whines and refuses -> Cursed = 0.99
    weapon_belief = ItemBeliefState(uid="wep1", item_class="weapon")
    valid = PetListener.process_pet_interaction(weapon_belief, pet_stepped_willingly=False)
    assert valid is True
    assert weapon_belief.p_cursed == 0.99

    # Food item: Comestible exception (pets eat cursed food greedily)
    food_belief = ItemBeliefState(uid="food1", item_class="food")
    valid = PetListener.process_pet_interaction(food_belief, pet_stepped_willingly=True)
    assert valid is False  # Must be rejected!
    assert food_belief.p_cursed == 0.10  # Untouched


def test_price_listener_candidate_elimination():
    price_map = {
        "potion of healing": 100,
        "potion of extra healing": 100,
        "potion of full healing": 200,
        "potion of gain ability": 300,
    }

    belief = ItemBeliefState(
        uid="pot1",
        item_class="potion",
        candidate_identities=list(price_map.keys()),
    )

    # Shopkeeper offers 50 zorkmids to buy the potion from player (selling)
    # In NetHack, selling offer is base // 2 -> candidate base cost is 100
    surviving = PriceIDListener.filter_candidates_by_price(
        item_belief=belief,
        candidate_price_map=price_map,
        observed_price=50,
        is_selling=True,
    )

    assert "potion of healing" in surviving
    assert "potion of extra healing" in surviving
    assert "potion of full healing" not in surviving
    assert "potion of gain ability" not in surviving


def test_engrave_listener_and_cursed_explosion_guard():
    # Potentially cursed wand (10% cursed prior)
    wand_untested = ItemBeliefState(
        uid="wand1",
        item_class="wand",
        candidate_identities=["wand of sleep", "wand of death", "wand of digging"],
        p_buc=np.array([0.10, 0.80, 0.10]),
    )

    # MUST raise CursedWandExplosionRiskError
    with pytest.raises(CursedWandExplosionRiskError):
        EngraveTestListener.process_engrave_result(
            wand_untested,
            "The bugs on the floor stop moving!",
            enforce_safety=True,
        )

    # Once verified uncursed:
    wand_untested.collapse_buc("UNCURSED")
    surviving = EngraveTestListener.process_engrave_result(
        wand_untested,
        "The bugs on the floor stop moving!",
        enforce_safety=True,
    )

    assert "wand of sleep" in surviving
    assert "wand of death" in surviving
    assert "wand of digging" not in surviving


def test_shannon_entropy_safe_gates():
    # Cursed armor: Veto equip
    cursed_armor = ItemBeliefState(
        uid="helm1",
        item_class="armor",
        p_buc=np.array([0.0, 0.0, 1.0]),
    )
    safe, reason = ShannonSafeGate.can_safely_equip(cursed_armor)
    assert safe is False
    assert "VETO: P(Cursed)" in reason

    # Unidentified lethal potion (candidate contains potion of death)
    mystery_potion = ItemBeliefState(
        uid="pot_mystery",
        item_class="potion",
        candidate_identities=["potion of water", "potion of death", "potion of booze"],
        p_buc=np.array([0.0, 1.0, 0.0]),  # Uncursed, but identity unknown
    )
    safe, reason = ShannonSafeGate.can_safely_quaff(mystery_potion)
    assert safe is False
    assert "VETO" in reason

    # Safe known potion
    known_healing = ItemBeliefState(
        uid="pot_heal",
        item_class="potion",
        candidate_identities=["potion of extra healing"],
        p_buc=np.array([0.0, 1.0, 0.0]),
    )
    safe, reason = ShannonSafeGate.can_safely_quaff(known_healing)
    assert safe is True
    assert reason == "SAFE"


def test_epistemic_manager_coordination():
    mgr = EpistemicManager()
    mgr.register_price_map({"long sword": 15, "silver sword": 40})

    item = NormalizedItem(
        uid="sword1",
        raw_str="a sword",
        current_letter="a",
        glyph=100,
    )

    belief = mgr.get_or_create_belief(item)
    assert belief.uid == "sword1"
    assert belief.item_class == "weapon"

    # Verify action safety: equip is blocked by default cursed prior (10% > 5%)
    safe, reason = mgr.verify_action_safety("EQUIP", "sword1")
    assert safe is False
    assert "VETO" in reason

    # Drop on altar -> Uncursed
    blstats = create_dummy_blstats(blind=False)
    mgr.on_altar_drop("sword1", blstats, "You drop the sword.")

    # Now equip should be permitted!
    safe, reason = mgr.verify_action_safety("EQUIP", "sword1")
    assert safe is True
    assert reason == "SAFE"
