"""
Unit tests for Gen-3 sensory and epistemic predicates in predicates.py and manifest.py.
"""
import pytest
from lox.policy.predicates import evaluate, eval_condition, nethack_bindings, default_ctx
from lox.policy.manifest import build_manifest


def test_manifest_contains_new_sensory_predicates():
    manifest = build_manifest("nethack")
    new_preds = [
        "stairs_dist", "unvisited_count", "is_corridor", "is_doorway",
        "count_hostiles", "closest_threat", "closest_speed", "has_ranged_target",
        "is_monster_fleeing", "carried_food_count", "turns_since_pray",
        "can_safely_pray", "has_intrinsic", "current_ac", "weapon_enchantment",
        "has_item", "hunger_level",
    ]
    for p in new_preds:
        assert p in manifest.predicates, f"Missing predicate {p} in manifest"


def test_sensory_predicates_eval_default_ctx():
    ctx = default_ctx()
    bindings = nethack_bindings(ctx)

    assert bindings["is_corridor"]() is False
    assert bindings["is_doorway"]() is False
    assert bindings["can_safely_pray"]() is True
    assert bindings["stairs_dist"]("<=", 15) is True
    assert bindings["stairs_dist"](">", 15) is False
    assert bindings["count_hostiles"]("==", 0) is True
    assert bindings["closest_speed"]("<=", 12) is True
    assert bindings["carried_food_count"](">", 0) is True
    assert bindings["turns_since_pray"](">=", 400) is True
    assert bindings["current_ac"]("<=", 10) is True


def test_sensory_predicates_with_overrides():
    ctx = default_ctx(
        stairs_dist=3,
        unvisited_count=5,
        is_corridor=True,
        count_hostiles=2,
        closest_threat=8.5,
        closest_speed=18,
        carried_food_count=0,
        turns_since_pray=120,
        can_safely_pray=False,
        intrinsics={"poison_res", "reflection"},
        ac=-2,
        inventory_items={"scroll of teleportation", "potion of healing"},
    )
    bindings = nethack_bindings(ctx)

    assert bindings["is_corridor"]() is True
    assert bindings["stairs_dist"]("<=", 5) is True
    assert bindings["unvisited_count"]("<=", 10) is True
    assert bindings["count_hostiles"](">", 0) is True
    assert bindings["closest_threat"](">=", 8.0) is True
    assert bindings["closest_speed"](">", 12) is True  # fast monster!
    assert bindings["carried_food_count"]("==", 0) is True  # food crisis!
    assert bindings["can_safely_pray"]() is False
    assert bindings["has_intrinsic"]("poison_res") is True
    assert bindings["has_intrinsic"]("cold_res") is False
    assert bindings["current_ac"]("<=", 0) is True
    assert bindings["has_item"]("scroll", "teleport") is True
    assert bindings["has_item"]("wand", "wishing") is False
