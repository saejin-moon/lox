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


def test_dsl_diff_with_pythonic_infix():
    from lox.policy.dsl import parse_diff

    diff_text = """(revision 1 (parent 0) (domain "minihack"))
(goal prioritize reach_stairs (when stairs_known and not adjacent_hostiles))
(rule add tactic_rules (do retreat) (when monster == "j" and hp_frac <= 0.35))
"""
    diff = parse_diff(diff_text)
    assert len(diff.goals) == 1
    assert diff.goals[0].when == (
        "call",
        "and",
        [("call", "stairs_known", []), ("call", "not", [("call", "adjacent_hostiles", [])])],
    )

    assert len(diff.rules) == 1
    assert diff.rules[0].rule["when"] == (
        "call",
        "and",
        [("call", "monster", ["j"]), ("call", "hp_frac", ["<=", 0.35])],
    )


def test_infix_canonical_equivalence():
    from lox.policy.dsl import parse_diff

    # S-expression style
    sexpr_diff = """(revision 1 (parent 0))
(rule add tactic_rules (do retreat) (when (and (monster "j") (hp_frac <= 0.35))))
"""
    # Pythonic infix style
    infix_diff = """(revision 1 (parent 0))
(rule add tactic_rules (do retreat) (when monster == "j" and hp_frac <= 0.35))
"""
    d_sexpr = parse_diff(sexpr_diff)
    d_infix = parse_diff(infix_diff)

    assert d_sexpr.rules[0].rule["when"] == d_infix.rules[0].rule["when"]


def test_validate_infix_diff_against_program():
    from lox.policy.dsl import parse_diff
    from lox.policy.validator import validate_diff
    from lox.policy.program import PolicyProgram
    from lox.executor.minihack_adapter import MiniHackAdapter

    adapter = MiniHackAdapter()
    manifest = adapter.manifest("v1")
    base = PolicyProgram.load("data/compiled/minihack.json")

    diff_text = f"""(revision {base.version + 1} (parent {base.version}) (domain "minihack"))
(goal prioritize reach_stairs (when stairs_known and not adjacent_hostiles))
(rule add tactic_rules (do retreat) (when monster == "j" and hp_frac <= 0.35))
"""
    res = validate_diff(diff_text, base, manifest=manifest)
    assert res.ok, f"Validation failed: {res.error_code} - {res.detail}"


def test_pure_pythonic_diff_zero_parentheses():
    from lox.policy.dsl import parse_diff

    pythonic_diff = """
    # Pure Pythonic Infix AST diff (zero S-expression parentheses)
    revision: 5, parent: 4, author: "qwen-8b", domain: "minihack", reason: "retreat when wounded"
    set policy_params.hp_retreat_ratio = 0.35
    rule add tactic_rules: when monster == "j" and hp_frac <= 0.40 do retreat unless blind note "emergency retreat"
    rule remove tactic_rules: 2
    goal prioritize reach_stairs: when stairs_known and not adjacent_hostiles
    goal deprioritize explore: after hp_frac < 0.20
    macro low_hp = hp_frac <= 0.50
    """
    diff = parse_diff(pythonic_diff)
    assert diff.header.revision == 5
    assert diff.header.parent == 4
    assert diff.header.author == "qwen-8b"
    assert diff.header.domain == "minihack"
    assert diff.header.reason == "retreat when wounded"

    assert len(diff.sets) == 1
    assert diff.sets[0].path == "policy_params.hp_retreat_ratio"
    assert diff.sets[0].value == 0.35

    assert len(diff.rules) == 2
    assert diff.rules[0].action == "add"
    assert diff.rules[0].rule["do"] == "retreat"
    assert diff.rules[0].rule["note"] == "emergency retreat"
    assert diff.rules[0].rule["when"] == (
        "call",
        "and",
        [("call", "monster", ["j"]), ("call", "hp_frac", ["<=", 0.40])],
    )
    assert diff.rules[0].rule["unless"] == ("call", "blind", [])
    assert diff.rules[1].action == "remove"
    assert diff.rules[1].index == 2

    assert len(diff.goals) == 2
    assert diff.goals[0].kind == "prioritize"
    assert diff.goals[0].goal == "reach_stairs"
    assert diff.goals[0].when == (
        "call",
        "and",
        [("call", "stairs_known", []), ("call", "not", [("call", "adjacent_hostiles", [])])],
    )
    assert diff.goals[1].kind == "deprioritize"
    assert diff.goals[1].goal == "explore"
    assert diff.goals[1].after == ("call", "hp_frac", ["<", 0.20])

    assert len(diff.defmacros) == 1
    assert diff.defmacros[0].name == "low_hp"
    assert diff.defmacros[0].body == ("call", "hp_frac", ["<=", 0.50])


def test_validate_pure_pythonic_diff():
    from lox.policy.validator import validate_diff
    from lox.policy.program import PolicyProgram
    from lox.executor.minihack_adapter import MiniHackAdapter

    adapter = MiniHackAdapter()
    manifest = adapter.manifest("v1")
    base = PolicyProgram.load("data/compiled/minihack.json")

    pythonic_text = f"""
    revision: {base.version + 1}, parent: {base.version}, author: "local-qwen", domain: "minihack", reason: "retreat discipline"
    rule add tactic_rules: when monster == "j" and hp_frac <= 0.35 do retreat
    goal prioritize reach_stairs: when stairs_known and not adjacent_hostiles
    """
    res = validate_diff(pythonic_text, base, manifest=manifest)
    assert res.ok, f"Validation failed: {res.error_code} - {res.detail}"


def test_extract_diff_text_pythonic():
    from lox.policy.dsl import extract_diff_text, parse_diff

    llm_output = """
    I reviewed the duckdb telemetry. We are dying to monster 'j'.
    Here is my proposed diff:

    ```python
    revision: 5, parent: 4, author: "qwen-8b", domain: "minihack", reason: "kite jacks"
    rule add tactic_rules: when monster == "j" and hp_frac <= 0.40 do retreat
    ```

    # SUMMARY
    This will prevent melee deaths against jackals.
    """

    extracted = extract_diff_text(llm_output)
    diff = parse_diff(extracted)
    assert diff.header.revision == 5
    assert len(diff.rules) == 1
    assert diff.rules[0].rule["do"] == "retreat"


def test_predicates_parse_and_eval_infix():
    from lox.policy.predicates import parse, eval_condition, nethack_bindings, default_ctx
    ctx = default_ctx(hp=10, max_hp=20, xl=5, lawful=True, monster_name="jackal")
    bindings = nethack_bindings(ctx)

    # 1. Pure infix without outer parentheses
    node = parse("monster == 'jackal' and hp_frac <= 0.60")
    assert node == ("call", "and", [("call", "monster", ["jackal"]), ("call", "hp_frac", ["<=", 0.60])])
    assert eval_condition("monster == 'jackal' and hp_frac <= 0.60", bindings) is True

    # 2. Infix with grouping parentheses
    assert eval_condition("(monster == 'jackal' and hp_frac <= 0.60) or not lawful", bindings) is True
    assert eval_condition("(monster == 'soldier ant' and hp_frac <= 0.60) and lawful", bindings) is False

    # 3. Simple bare-predicate infix
    assert eval_condition("lawful and not has_excalibur", bindings) is True


def test_render_infix_roundtrip():
    from lox.policy.dsl import render_infix
    from lox.policy.infix import parse_infix_to_canonical

    exprs = [
        "monster == 'jackal' and hp_frac <= 0.4",
        "stairs_known and not adjacent_hostiles",
        "hp_frac <= 0.35 or is_fighting",
        "true",
        "false",
        "lawful",
    ]
    for expr in exprs:
        node = parse_infix_to_canonical(expr)
        rendered = render_infix(node)
        re_parsed = parse_infix_to_canonical(rendered)
        assert re_parsed == node, f"Mismatch on {expr}: rendered={rendered}, re_parsed={re_parsed}"



