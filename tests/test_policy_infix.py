"""
Tests for lox.policy.infix (Pythonic Infix AST macro and condition parsing).
"""
import pytest

from lox.policy.infix import (
    parse_infix_condition,
    parse_infix_macro,
    parse_infix_rule,
    InfixSyntaxError,
)


def test_parse_simple_comparison():
    res = parse_infix_condition("hp_frac <= 0.5")
    assert res == ("compare", "hp_frac", "<=", 0.5)

    res_gt = parse_infix_condition("adjacent_hostiles > 1")
    assert res_gt == ("compare", "adjacent_hostiles", ">", 1)


def test_parse_boolean_logic():
    res = parse_infix_condition("stairs_known and not blind")
    assert res == ("and", ("stairs_known",), ("not", ("blind",)))

    res_or = parse_infix_condition("low_hp or poisoned")
    assert res_or == ("or", ("low_hp",), ("poisoned",))


def test_parse_nested_condition():
    res = parse_infix_condition("low_hp and (adjacent_hostiles >= 2 or stunned)")
    assert res == (
        "and",
        ("low_hp",),
        ("or", ("compare", "adjacent_hostiles", ">=", 2), ("stunned",)),
    )


def test_depth_budget_enforcement():
    # Allowed: depth <= 3
    valid_expr = "not (a and (b or c))"
    res = parse_infix_condition(valid_expr, max_depth=3)
    assert res is not None

    # Exceeds: depth = 4
    too_deep_expr = "not (a and (b or (c and d)))"
    with pytest.raises(InfixSyntaxError, match="exceeds allowed budget"):
        parse_infix_condition(too_deep_expr, max_depth=3)


def test_sandboxing_rejection():
    # Function calls are prohibited
    with pytest.raises(InfixSyntaxError, match="Disallowed syntax element: Call"):
        parse_infix_condition("__import__('os').system('ls')")

    # Attributes are prohibited
    with pytest.raises(InfixSyntaxError, match="Disallowed syntax element: Attribute"):
        parse_infix_condition("stats.hp <= 10")

    # List comprehensions/loops prohibited
    with pytest.raises(InfixSyntaxError):
        parse_infix_condition("[x for x in range(10)]")


def test_parse_infix_macro():
    macro = parse_infix_macro("defmacro low_hp = hp_frac <= 0.5")
    assert macro.name == "low_hp"
    assert macro.body == ("compare", "hp_frac", "<=", 0.5)

    macro2 = parse_infix_macro("macro critical_threat = low_hp and adjacent_hostiles >= 2")
    assert macro2.name == "critical_threat"
    assert macro2.body == (
        "and",
        ("low_hp",),
        ("compare", "adjacent_hostiles", ">=", 2),
    )


def test_parse_infix_rule():
    rule = parse_infix_rule(
        'rule: when low_hp and adjacent_hostiles >= 2 do retreat unless blind note "emergency retreat"'
    )
    assert rule.do == "retreat"
    assert rule.when == (
        "and",
        ("low_hp",),
        ("compare", "adjacent_hostiles", ">=", 2),
    )
    assert rule.unless == ("blind",)
    assert rule.note == "emergency retreat"
