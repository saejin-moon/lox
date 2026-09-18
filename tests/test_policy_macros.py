"""R3 tests: defmacro expansion + closure checker (corp/policy/macros.py, MACRO.md §5)."""
import pytest

from corp.policy import dsl
from corp.policy.macros import (
    MacroEnv, MacroError, expand, expand_all, check_closure, node_depth, free_symbols,
)
from corp.policy.manifest import build_manifest

MANIFEST = build_manifest(1, "nethack")
PREDS = set(MANIFEST.predicates)


def parse_expr(text: str):
    return dsl.parse_forms(text)[0]


class TestExpansion:
    def test_basic_substitution(self):
        env = MacroEnv({"when_endangered": parse_expr(
            "(and (adjacent_hostiles) (hp_frac <= 0.40) (not has_healing))")})
        out = expand(parse_expr("(and (in_mines) (when_endangered))"), env)
        # bare (not has_healing) sugar: bare string preserved (resolves at eval time)
        assert out == ("call", "and", [
            ("call", "in_mines", []),
            ("call", "and", [("call", "adjacent_hostiles", []),
                             ("call", "hp_frac", ["<=", 0.4]),
                             ("call", "not", ["has_healing"])]),
        ])

    def test_bare_symbol_sugar(self):
        env = MacroEnv({"when_hurt": parse_expr("(hp_frac <= 0.30)")})
        out = expand(parse_expr("(not when_hurt)"), env)
        assert out == ("call", "not", [("call", "hp_frac", ["<=", 0.3])])

    def test_macro_with_args_rejected(self):
        env = MacroEnv({"m": parse_expr("(lawful)")})
        with pytest.raises(MacroError) as e:
            expand(parse_expr("(m 1)"), env)
        assert e.value.code == "ERR_SIGNATURE"

    def test_cycle_detected(self):
        env = MacroEnv({"a": parse_expr("(and (b) (true))"),
                        "b": parse_expr("(and (a) (true))")})
        with pytest.raises(MacroError) as e:
            expand(parse_expr("(a)"), env)
        assert e.value.code == "ERR_MACRO_CYCLE"

    def test_expansion_budget(self):
        # macro-in-macro composition consumes the caller's budget (MACRO.md §5.3):
        # m1 = depth-2 body; (and (m1) (m1)) = depth 3 (ok); (and (m2) (m2)) = depth 4 → reject
        env = MacroEnv({"m1": parse_expr("(and (and (lawful)))"),
                        "m2": parse_expr("(and (m1) (m1))"),
                        "m3": parse_expr("(and (m2) (m2))")})
        ok = expand_all(parse_expr("(m2)"), env, PREDS)
        assert node_depth(ok) == 3
        with pytest.raises(MacroError) as e:
            expand_all(parse_expr("(m3)"), env, PREDS)
        assert e.value.code == "ERR_DEPTH_BUDGET"

    def test_deterministic(self):
        env = MacroEnv({"m": parse_expr("(and (lawful) (xl_ge 5))")})
        a = expand_all(parse_expr("(and (m) (m))"), env, PREDS)
        b = expand_all(parse_expr("(and (m) (m))"), env, PREDS)
        assert a == b


class TestClosure:
    def test_unknown_predicate_rejected(self):
        with pytest.raises(MacroError) as e:
            check_closure(parse_expr("(and (lawful) (uses_wand))"), PREDS, set())
        assert e.value.code == "ERR_MACRO_CLOSURE"

    def test_live_macro_reference_ok(self):
        check_closure(parse_expr("(and (lawful) (when_hurt))"), PREDS, {"when_hurt"})

    def test_shadowed_primitive_name_rejected_by_validator(self):
        # 'retreat' is a verb — check_closure passes it as a "known macro" name, but
        # the validator's shadow gate (MACRO.md §5.3 no-shadowing) must reject it.
        from corp.policy.validator import validate_diff, ValidatorHooks
        from corp.policy.program import PolicyProgram, DEFAULT_PROGRAM
        prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
        text = ('(revision 2 (parent 1) (author "t") (domain nethack) (reason "t"))\n'
                '(defmacro retreat (and (lawful)))')
        r = validate_diff(text, prog)
        assert not r.ok and r.error_code == "ERR_MACRO_SHADOW"


class TestNodeDepth:
    def test_depths(self):
        assert node_depth(parse_expr("(lawful)")) == 0
        assert node_depth(parse_expr("(and (lawful) (true))")) == 1
        assert node_depth(parse_expr("(and (or (lawful)) (true))")) == 2
        assert node_depth("lawful") == 0
        assert node_depth(0.4) == 0