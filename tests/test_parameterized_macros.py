"""
Tests for Parameterized Macros and Extended Goal Syntax.
Verifies defmacro with parameters, substitution, arity checking,
cycle detection, and end-to-end policy diff compilation.
"""
import pytest

from lox.policy import dsl
from lox.policy.infix import parse_infix_macro, parse_pythonic_diff
from lox.policy.macros import (
    MacroEnv, MacroError, expand, expand_all, check_closure, node_depth,
)
from lox.policy.manifest import build_manifest
from lox.policy.program import PolicyProgram, DEFAULT_PROGRAM
from lox.policy.validator import validate_diff

MANIFEST = build_manifest(1, "nethack")
PREDS = set(MANIFEST.predicates)


class TestParameterizedMacros:
    def test_parse_parameterized_macro_infix(self):
        m = parse_infix_macro("defmacro danger(dist, hp) = hostile_dist <= dist and hp_frac <= hp")
        assert m.name == "danger"
        assert m.params == ["dist", "hp"]
        assert m.body == (
            "compare", "hostile_dist", "<=", "dist"
        ) or "hostile_dist" in str(m.body)

    def test_macro_env_substitution(self):
        env = MacroEnv()
        # Macro: low_stat(stat, thresh) = stat <= thresh
        body = ("call", "stat", ["<=", "thresh"])
        env.add("low_stat", body, params=["stat", "thresh"])

        # Call: low_stat(hp_frac, 0.35)
        call_expr = ("call", "low_stat", ["hp_frac", 0.35])
        expanded = expand(call_expr, env)
        assert expanded == ("call", "hp_frac", ["<=", 0.35])

    def test_macro_substitution_with_combinators(self):
        env = MacroEnv()
        # Macro: danger(dist, hp) = hostile_dist <= dist and hp_frac <= hp
        body = (
            "call", "and", [
                ("call", "hostile_dist", ["<=", "dist"]),
                ("call", "hp_frac", ["<=", "hp"]),
            ]
        )
        env.add("danger", body, params=["dist", "hp"])

        call_expr = ("call", "danger", [3, 0.40])
        expanded = expand(call_expr, env)
        assert expanded == (
            "call", "and", [
                ("call", "hostile_dist", ["<=", 3]),
                ("call", "hp_frac", ["<=", 0.40]),
            ]
        )

    def test_macro_arity_mismatch_raises(self):
        env = MacroEnv()
        body = ("call", "hp_frac", ["<=", "thresh"])
        env.add("low_hp", body, params=["thresh"])

        # Missing argument
        with pytest.raises(MacroError) as exc_info:
            expand(("call", "low_hp", []), env)
        assert exc_info.value.code == "ERR_SIGNATURE"

        # Too many arguments
        with pytest.raises(MacroError) as exc_info:
            expand(("call", "low_hp", [0.4, 0.5]), env)
        assert exc_info.value.code == "ERR_SIGNATURE"

    def test_parameterized_macro_cannot_be_bare_symbol(self):
        env = MacroEnv()
        body = ("call", "hp_frac", ["<=", "thresh"])
        env.add("low_hp", body, params=["thresh"])

        with pytest.raises(MacroError) as exc_info:
            expand("low_hp", env)
        assert exc_info.value.code == "ERR_SIGNATURE"

    def test_macro_closure_with_declared_parameters(self):
        # Body referencing parameter names
        body = (
            "call", "and", [
                ("call", "stairs_dist", ["<=", "dist"]),
                ("call", "hp_frac", ["<=", "hp"]),
            ]
        )
        # Should succeed because 'dist' and 'hp' are declared parameters
        check_closure(body, PREDS, set(), params={"dist", "hp"})

        # Should fail if parameters are undeclared
        with pytest.raises(MacroError) as exc_info:
            check_closure(body, PREDS, set(), params=set())
        assert exc_info.value.code == "ERR_MACRO_CLOSURE"

    def test_nested_macro_composition(self):
        env = MacroEnv()
        # low_stat(stat, thresh) = stat <= thresh
        env.add("low_stat", ("call", "stat", ["<=", "thresh"]), params=["stat", "thresh"])
        # critical(hp) = low_stat(hp_frac, hp) and in_combat
        env.add(
            "critical",
            ("call", "and", [
                ("call", "low_stat", ["hp_frac", "hp"]),
                ("call", "in_combat", []),
            ]),
            params=["hp"]
        )

        call_expr = ("call", "critical", [0.30])
        expanded = expand(call_expr, env)
        assert expanded == (
            "call", "and", [
                ("call", "hp_frac", ["<=", 0.30]),
                ("call", "in_combat", []),
            ]
        )


class TestExtendedGoalAndDiffValidation:
    def test_pythonic_diff_with_parameterized_macros(self):
        diff_text = """
revision: 2, parent: 1, author: "gen3-test"
reason: "test parameterized macros in pythonic diff"
defmacro danger(hp) = hp_frac <= hp and monster == "jackal"
rule: when danger(0.40) do retreat
"""
        prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
        result = validate_diff(diff_text, prog, MANIFEST)
        assert result.ok, f"Validation failed: {result.error_code} - {result.detail}"
        assert len(result.candidate.macros) == 1
        assert result.candidate.macros[0]["name"] == "danger"
        assert result.candidate.macros[0]["params"] == ["hp"]

    def test_extended_goal_syntax(self):
        diff_text = """
revision: 2, parent: 1, author: "gen3-test"
reason: "test extended goal clauses"
goal deprioritize explore_floor: when hunger_ge(2)
goal prioritize descend: when stairs_known until can_safely_pray
"""
        diff = parse_pythonic_diff(diff_text)
        assert len(diff.goals) == 2

        g_deprio = next(g for g in diff.goals if g.goal == "explore_floor")
        assert g_deprio.kind == "deprioritize"
        assert g_deprio.when is not None

        g_prio = next(g for g in diff.goals if g.goal == "descend")
        assert g_prio.kind == "prioritize"
        assert g_prio.when is not None
        assert g_prio.until is not None

        prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
        result = validate_diff(diff_text, prog, MANIFEST)
        assert result.ok, f"Validation failed: {result.error_code} - {result.detail}"
