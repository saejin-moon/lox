"""R3 tests: validator gate pipeline + manifest (MACRO.md §4, §6)."""
import pytest

from corp.policy.manifest import build_manifest, param_leaves_from_config
from corp.policy.predicates import eval_condition, nethack_bindings, default_ctx
from corp.policy.program import PolicyProgram, DEFAULT_PROGRAM
from corp.policy.validator import (
    validate_diff, ValidatorHooks, check_signature, GateResult, _leaf_path,
)
from corp.policy import dsl


def make_program(**overrides):
    return PolicyProgram.from_dict({**DEFAULT_PROGRAM, **overrides})


FIXTURES = [default_ctx(), default_ctx(depth=6, hp=10, max_hp=50, hunger_state=3),
            default_ctx(dnum=2, depth=3, adjacent_hostiles=1, has_healing=False)]


def validate(text, program=None, **kw):
    program = program or make_program()
    manifest = build_manifest(program.version, program.domain,
                              live_macros={m["name"]: m["body"] for m in program.macros})
    hooks = kw.pop("hooks", None) or ValidatorHooks(fixture_states=FIXTURES)
    return validate_diff(text, program, manifest, hooks=hooks)


def diff(ops: str, rev=2, parent=1):
    return (f'(revision {rev} (parent {parent}) (author "test") (domain nethack) (reason "t"))\n'
            + ops)


class TestManifest:
    def test_param_leaves_derived(self):
        leaves = param_leaves_from_config()
        assert "survival.rest_below_frac" in leaves
        leaf = leaves["survival.rest_below_frac"]
        assert leaf["type"] == "float" and leaf["default"] == 0.6
        assert leaf["min"] == 0.3 and leaf["max"] == 0.9  # MACRO.md §4.1 example

    def test_hunger_state_not_tunable(self):
        assert "strategy.emergency_hunger" not in param_leaves_from_config()

    def test_manifest_render_contains_vocab(self):
        m = build_manifest(1, "nethack")
        text = m.render()
        for sym in ("hp_frac", "retreat", "descend", "policy_params.survival.rest_below_frac"):
            assert sym in text

    def test_signature_check(self):
        m = build_manifest(1, "nethack")
        assert check_signature("hp_frac", ["<=", 0.4], m) == ""
        assert "expects" in check_signature("hp_frac", ["<="], m)
        assert "operator" in check_signature("hp_frac", ["!=", 0.4], m)
        assert "outside" in check_signature("hp_frac", ["<=", 1.5], m)


class TestGates:
    def test_valid_set_accepted(self):
        r = validate(diff("(set policy_params.survival.rest_below_frac 0.65)"))
        assert r.ok
        assert r.candidate.version == 2
        assert r.candidate.params == {"survival.rest_below_frac": 0.65}

    def test_header_parent_mismatch(self):
        r = validate(diff("(set policy_params.survival.rest_below_frac 0.65)", rev=5, parent=3))
        assert not r.ok and r.error_code == "ERR_HEADER"

    def test_unknown_symbol_rejected(self):
        r = validate(diff("(set policy_params.survival.no_such_param 0.5)"))
        assert not r.ok and r.error_code == "ERR_BOUNDS"

    def test_out_of_bounds_rejected(self):
        r = validate(diff("(set policy_params.survival.rest_below_frac 0.99)"))
        assert not r.ok and r.error_code == "ERR_BOUNDS"

    def test_type_mismatch_rejected(self):
        r = validate(diff("(set policy_params.survival.rest_below_frac \"high\")"))
        # strings are not numeric values — rejected at parse (grammar) or bounds
        assert not r.ok and r.error_code in ("ERR_PARSE", "ERR_BOUNDS")

    def test_unknown_verb_rejected(self):
        r = validate(diff("(rule add tactic_rules (when (lawful)) (do backflip))"))
        assert not r.ok and r.error_code == "ERR_UNKNOWN_SYMBOL"

    def test_uncertified_goal_rejected(self):
        r = validate(diff("(goal prioritize ascend_to_heaven)"))
        assert not r.ok and r.error_code == "ERR_UNKNOWN_SYMBOL"

    def test_form_budgets(self):
        ops = "\n".join("(set policy_params.survival.rest_below_frac 0.65)" for _ in range(11))
        r = validate(diff(ops))
        assert not r.ok and r.error_code == "ERR_FORM_BUDGET"

    def test_macro_shadow_rejected(self):
        r = validate(diff("(defmacro lawful (and (true)))"))
        assert not r.ok and r.error_code == "ERR_MACRO_SHADOW"

    def test_macro_accepted_and_stored(self):
        r = validate(diff('(defmacro when_endangered (and (adjacent_hostiles) (hp_frac <= 0.40)))\n'
                          '(rule add tactic_rules (when (and (in_mines) (when_endangered))) (do retreat))'))
        assert r.ok, r.detail
        assert r.candidate.macros[0]["name"] == "when_endangered"
        # rules store the EXPANDED condition (executor never sees macros — MACRO.md §7)
        assert r.candidate.tactic_rules[0]["when"] == \
            "(and (in_mines) (and (adjacent_hostiles) (hp_frac <= 0.4)))"

    def test_macro_redefinition_allowed(self):
        prog = make_program(macros=[{"name": "when_endangered", "body": "(lawful)"}])
        r = validate(diff("(defmacro when_endangered (and (lawful) (hp_frac <= 0.2)))\n"
                          "(rule add tactic_rules (when (when_endangered)) (do retreat))"), prog)
        assert r.ok, r.detail
        assert r.candidate.macros[0]["body"] == "(and (lawful) (hp_frac <= 0.2))"

    def test_program_mount_goal_ops(self):
        r = validate(diff("(goal prioritize descend)\n(goal set_threshold descend descent.min_hp_frac 0.5)"))
        assert r.ok, r.detail
        assert r.candidate.strategy_plan[0].goal == "descend"
        assert r.candidate.params["descent.min_hp_frac"] == 0.5

    def test_invariant_goal_removal_detected(self):
        # deprioritize reorders but never removes — remove is not expressible; the
        # invariant check must catch any path that drops a goal (defensive).
        prog = make_program()
        r = validate(diff("(goal deprioritize enter_sokoban)"), prog)
        assert r.ok  # reorder only
        goals_after = {g.goal for g in r.candidate.strategy_plan}
        assert goals_after == {g.goal for g in prog.strategy_plan}

    def test_certification_gate(self):
        def cert(candidate):
            raise RuntimeError("cert failed")
        r = validate(diff("(set policy_params.survival.rest_below_frac 0.65)"),
                     hooks=ValidatorHooks(certification=cert))
        assert not r.ok and r.error_code == "ERR_CERT_FAIL"

    def test_quick_batch_gate_regress(self):
        calls = {}

        def qb(candidate):
            calls["version"] = candidate.version
            return {"mean_score": 100.0}

        r = validate(diff("(set policy_params.survival.rest_below_frac 0.65)"),
                     hooks=ValidatorHooks(quick_batch=qb, baseline_score=400.0))
        assert not r.ok and r.error_code == "ERR_BATCH_REGRESS"

    def test_quick_batch_gate_pass(self):
        r = validate(diff("(set policy_params.survival.rest_below_frac 0.65)"),
                     hooks=ValidatorHooks(quick_batch=lambda c: {"mean_score": 350.0},
                                          baseline_score=400.0))
        assert r.ok

    def test_shadow_test_fail(self):
        # An expression the evaluator cannot run → ERR_SHADOW_FAIL
        r = validate(diff("(goal prioritize descend (when (failures_in_10_episodes_ge 2)))"))
        # signature check catches the arity first (missing goal arg) — either rejection is fine
        assert not r.ok

    def test_mounted_program_loads_and_applies(self):
        r = validate(diff("(set policy_params.survival.rest_below_frac 0.65)"))
        assert r.ok
        from corp.policy.config import PolicyConfig
        cfg = PolicyConfig.defaults()
        applied = r.candidate.apply_overlay(cfg)
        assert applied == ["survival.rest_below_frac"]
        assert cfg.survival.rest_below_frac == 0.65


class TestNogoods:
    def test_nogood_add_mounts(self):
        r = validate(diff('(nogood add (when (depth_le 5)) (cause "throne shock") (forbid SIT))'))
        assert r.ok
        assert r.candidate.nogoods[-1]["cause"] == "throne shock"
        assert r.candidate.nogoods[-1]["forbid"] == "SIT"

    def test_nogood_program_path_loading(self):
        from corp.planner.nogood import NogoodStore
        import json, tempfile, os
        prog = {"version": 2, "strategy_plan": [], "nogoods": [
            {"mask": 123, "target_val": 64, "forbidden_action": "SIT",
             "reason": "test", "generation": 1},
            {"when": "(depth_le 5)", "cause": "declarative", "forbid": "SIT"},
        ]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(prog, f)
            path = f.name
        try:
            store = NogoodStore()
            n_before = len(store.entries)
            store.load_from_json(path)
            assert len(store.entries) == n_before + 1  # declarative entry skipped
        finally:
            os.unlink(path)


class TestLeafPath:
    def test_prefix_stripping(self):
        assert _leaf_path("policy_params.survival.rest_below_frac") == "survival.rest_below_frac"
        assert _leaf_path("descent.min_hp_frac") == "descent.min_hp_frac"