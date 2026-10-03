"""
Unit tests for LOX 2.0 Epistemic POMDP Engine, Shannon Entropy Safe-Gates,
and Altar/Pet/Price listeners.
"""
from __future__ import annotations

import numpy as np
import pytest
from lox.core.epistemic import (
    ItemBeliefState,
    AltarListener,
    PetListener,
    PriceIDListener,
    ShannonSafeGate,
    EpistemicEngine,
)


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
    belief = ItemBeliefState(uid="item1")

    # Blindness blocks altar perception
    processed = AltarListener.process_drop(belief, is_blind=True, message="There is an amber flash of light.")
    assert processed is False
    assert belief.p_blessed != 1.0

    # Normal perception: Black light -> Cursed
    processed = AltarListener.process_drop(belief, is_blind=False, message="There is a flash of black light.")
    assert processed is True
    assert belief.p_cursed == 1.0

    # Amber flash -> Blessed
    belief2 = ItemBeliefState(uid="item2")
    AltarListener.process_drop(belief2, is_blind=False, message="There is an amber flash.")
    assert belief2.p_blessed == 1.0

    # No flash -> Uncursed
    belief3 = ItemBeliefState(uid="item3")
    AltarListener.process_drop(belief3, is_blind=False, message="You drop the long sword.")
    assert belief3.p_uncursed == 1.0


def test_pet_listener_and_comestible_exception():
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
    assert valid is False  # Must be rejected for food!
    assert food_belief.p_cursed == 0.10


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

    # Shopkeeper offers 50 zm to buy from player (selling = base // 2 -> 100)
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


def test_shannon_entropy_safe_gates():
    # Cursed armor: Veto equip
    cursed_armor = ItemBeliefState(
        uid="helm1",
        item_class="armor",
        p_buc=np.array([0.0, 0.0, 1.0]),
    )
    safe, reason = ShannonSafeGate.can_safely_equip(cursed_armor)
    assert safe is False
    assert "VETO" in reason

    # Unidentified lethal potion in pool
    mystery_potion = ItemBeliefState(
        uid="pot_mystery",
        item_class="potion",
        candidate_identities=["potion of water", "potion of death", "potion of booze"],
        p_buc=np.array([0.0, 1.0, 0.0]),  # Uncursed, but identity unknown
    )
    safe, reason = ShannonSafeGate.can_safely_quaff(mystery_potion)
    assert safe is False
    assert "VETO: Lethal" in reason

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


def test_epistemic_engine_coordination():
    engine = EpistemicEngine()
    belief = engine.get_or_create(uid="helm1", name="iron helm", item_class="armor", slot_letter="a")
    assert belief.p_cursed == 0.10
    assert engine.count_untested_buc() == 1

    # Safe gate prevents wearing untested 10% cursed armor
    assert engine.can_safely_wear("helm1") is False

    # Drop on altar -> Amber flash (Blessed)
    engine.update_from_altar_drop("helm1", is_blind=False, message="There is an amber flash.")
    assert belief.p_blessed == 1.0
    assert belief.p_cursed == 0.0
    assert engine.count_untested_buc() == 0

    # Now safe to wear!
    assert engine.can_safely_wear("helm1") is True
