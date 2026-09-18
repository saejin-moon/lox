"""R4 tests: tactic-rule engine, interlock invariants (#9b), profiles, default-program parity."""
import pytest

from corp.domain.combat_manager import TacticalCombatManager
from corp.policy import tactics as tactic_mod
from corp.policy.manifest import build_manifest
from corp.policy.predicates import eval_condition, nethack_bindings, default_ctx
from corp.policy.program import PolicyProgram, DEFAULT_PROGRAM, DEFAULT_TACTIC_RULES
from corp.policy.profiles import apply_profile_overlay, resolve_config, ProfileError
from corp.policy.validator import validate_diff, ValidatorHooks
from corp.policy.config import PolicyConfig

FIXTURES = [default_ctx(), default_ctx(depth=6, hp=10, max_hp=50, hunger_state=3)]


def combat_ctx(monster_name, hp=50, max_hp=50, poison_res=True, depth=3):
    return {
        "monster_name": monster_name, "item_name": "", "depth": depth,
        "hp": hp, "max_hp": max_hp, "hunger_state": 1, "xl": 5, "turn": 100,
        "has_poison_res": poison_res, "has_reflection": False,
        "adjacent_hostiles": 1, "has_healing": False, "is_fighting": True,
    }


class TestDefaultRuleParity:
    """Drift guard (MACRO.md §9): the default program's rules must reproduce the
    combat manager's canonical name sets exactly."""

    def test_lethal_poison_sets_match(self):
        assert set(tactic_mod.TACTIC_LETHAL_POISON_NAMES) == \
            set(TacticalCombatManager.LETHAL_POISON_NAMES.keys())

    def test_heavy_hitter_sets_match(self):
        assert set(tactic_mod.TACTIC_HEAVY_HITTER_NAMES) == set(TacticalCombatManager.HEAVY_HITTERS)

    def test_default_program_ships_interlock_rules(self):
        prog = PolicyProgram.from_dict(DEFAULT_PROGRAM)
        interlocks = [r for r in prog.tactic_rules if r.get("interlock")]
        assert len(interlocks) == len(tactic_mod.TACTIC_LETHAL_POISON_NAMES) + \
            len(tactic_mod.TACTIC_HEAVY_HITTER_NAMES)
        verbs = {r["do"] for r in interlocks}
        assert verbs == {"elbereth_first", "retreat_when_wounded"}


class TestEngine:
    def test_lethal_poison_without_res_matches(self):
        engine = tactic_mod.TacticRuleEngine(DEFAULT_TACTIC_RULES)
        m = engine.evaluate(combat_ctx("giant spider", poison_res=False))
        assert m is not None and m.verb == "elbereth_first" and m.interlock

    def test_lethal_poison_with_res_does_not_match_poison_rule(self):
        engine = tactic_mod.TacticRuleEngine(DEFAULT_TACTIC_RULES)
        m = engine.evaluate(combat_ctx("killer bee", poison_res=True))
        # killer bee contains no heavy-hitter substring → no rule fires
        assert m is None

    def test_heavy_hitter_matches(self):
        engine = tactic_mod.TacticRuleEngine(DEFAULT_TACTIC_RULES)
        m = engine.evaluate(combat_ctx("ogre", poison_res=True))
        assert m is not None and m.verb == "retreat_when_wounded" and m.interlock

    def test_harmless_monster_no_match(self):
        engine = tactic_mod.TacticRuleEngine(DEFAULT_TACTIC_RULES)
        assert engine.evaluate(combat_ctx("kitten")) is None

    def test_first_match_wins(self):
        rules = [
            {"when": '(and (monster "jackal") (hp_frac <= 0.5))', "do": "retreat"},
            {"when": '(monster "jackal")', "do": "ranged_only"},
        ]
        engine = tactic_mod.TacticRuleEngine(rules)
        m = engine.evaluate(combat_ctx("jackal", hp=20, max_hp=50))
        assert m.verb == "retreat" and m.index == 0
        m2 = engine.evaluate(combat_ctx("jackal", hp=45, max_hp=50))
        assert m2.verb == "ranged_only" and m2.index == 1

    def test_llm_rule_execution(self):
        """The v3-authored rule class (adjacent hostiles + wounded → retreat) must
        actually fire through the engine — the R4 milestone."""
        rules = DEFAULT_TACTIC_RULES + [
            {"when": "(and (adjacent_hostiles) (hp_frac <= 0.5))", "do": "retreat_when_wounded"},
        ]
        engine = tactic_mod.TacticRuleEngine(rules)
        m = engine.evaluate(combat_ctx("kitten", hp=20, max_hp=50))
        assert m is not None and m.verb == "retreat_when_wounded" and not m.interlock

    def test_unparseable_condition_never_fires(self):
        engine = tactic_mod.TacticRuleEngine([{"when": "(broken", "do": "retreat"}])
        assert engine.evaluate(combat_ctx("kitten")) is None


class TestVerbResponses:
    """Verb → Task mapping through the real combat manager."""

    def _mgr(self, rules=None):
        return TacticalCombatManager(PolicyConfig.defaults(), tactic_rules=rules)

    def test_ranged_only_never_melee_when_unarmed(self):
        import numpy as np
        mgr = self._mgr([{"when": '(monster "kitten")', "do": "ranged_only"}])
        chars = np.full((21, 80), ord("."), dtype=int)
        resp = mgr._verb_response("ranged_only", None, None, (5, 5), 5, 5, [], chars, None, None)
        assert resp.name in ("STEP", "WAIT")  # unarmed → escape or hold ground, never melee

    def test_retreat_response_steps(self):
        import numpy as np
        mgr = self._mgr()
        chars = np.full((21, 80), ord("."), dtype=int)

        class FakeTarget:
            pos = (4, 5)

        resp = mgr._verb_response("retreat", None, None, (5, 5), 5, 5,
                                  [type("M", (), {"pos": (4, 5)})()], chars, None, FakeTarget())
        # no glyphs grid → escape finder returns None → WAIT (never engage)
        assert resp.name in ("STEP", "WAIT")

    def test_avoid_never_falls_through_to_melee(self):
        import numpy as np
        mgr = self._mgr()
        chars = np.full((21, 80), ord("."), dtype=int)
        resp = mgr._verb_response("avoid", None, None, (5, 5), 5, 5, [], chars, None, None)
        assert resp.name in ("STEP", "WAIT")  # route around, never engage

    def test_ranged_then_kill_unarmed_fallthrough(self):
        mgr = self._mgr()
        assert mgr._verb_response("ranged_then_kill", None, None, (5, 5), 5, 5,
                                  [], None, None, None) is TacticalCombatManager.RULE_FALLTHROUGH


class TestInterlockInvariant9b:
    """Validator invariant #9b: interlock default rules can never be removed by a diff."""

    def _program(self):
        return PolicyProgram.from_dict(DEFAULT_PROGRAM)

    def _validate(self, ops, program=None):
        program = program or self._program()
        manifest = build_manifest(program.version, program.domain,
                                  live_macros={m["name"]: m["body"] for m in program.macros})
        return validate_diff(
            f'(revision {program.version + 1} (parent {program.version}) (author "t") '
            f'(domain nethack) (reason "t"))\n' + ops,
            program, manifest, hooks=ValidatorHooks(fixture_states=FIXTURES))

    def test_remove_interlock_by_index_rejected(self):
        r = self._validate("(rule remove tactic_rules 0)")
        assert not r.ok and r.error_code == "ERR_INVARIANT" and "interlock" in r.detail

    def test_remove_interlock_by_match_rejected(self):
        rule = DEFAULT_TACTIC_RULES[0]
        r = self._validate(f'(rule remove tactic_rules (when {rule["when"]}) (do {rule["do"]}))')
        assert not r.ok and r.error_code == "ERR_INVARIANT"

    def test_remove_noninterlock_rule_allowed(self):
        program = self._program()
        program.tactic_rules.append({"when": "(monster \"kitten\")", "do": "avoid"})
        r = self._validate(f'(rule remove tactic_rules {len(program.tactic_rules) - 1})', program)
        assert r.ok, r.detail
        assert all(x.get("when") != '(monster "kitten")' for x in r.candidate.tactic_rules)

    def test_llm_can_add_rules(self):
        r = self._validate('(rule add tactic_rules (when (and (monster "coyote") (hp_frac <= 0.6))) (do kite))')
        assert r.ok, r.detail
        assert not r.candidate.tactic_rules[-1].get("interlock")

    def test_llm_cannot_forge_interlock_flag(self):
        """A diff-authored rule claiming interlock status must not gain protection —
        and more importantly must not be treated as an interlock by the engine."""
        r = self._validate('(rule add tactic_rules (when (monster "kitten")) (do avoid) (note "x"))')
        assert r.ok


class TestProfiles:
    def test_role_profile_overlay(self):
        program = PolicyProgram.from_dict({
            **DEFAULT_PROGRAM,
            "role_profiles": {"valkyrie": {"combat.fast_ratio": 1.5, "survival.rest_below_frac": 0.7}},
        })
        cfg = resolve_config(program, role="valkyrie")
        assert cfg.combat.fast_ratio == 1.5
        assert cfg.survival.rest_below_frac == 0.7

    def test_domain_profile_under_role_profile(self):
        program = PolicyProgram.from_dict({
            **DEFAULT_PROGRAM,
            "domain_profiles": {"nethack": {"combat.fast_ratio": 1.1}},
            "role_profiles": {"valkyrie": {"combat.fast_ratio": 1.5}},
        })
        cfg = resolve_config(program, role="valkyrie", domain="nethack")
        assert cfg.combat.fast_ratio == 1.5  # role wins

    def test_params_beat_profiles(self):
        program = PolicyProgram.from_dict({
            **DEFAULT_PROGRAM,
            "role_profiles": {"valkyrie": {"combat.fast_ratio": 1.5}},
            "params": {"combat.fast_ratio": 2.0},
        })
        cfg = resolve_config(program, role="valkyrie")
        assert cfg.combat.fast_ratio == 2.0  # program.params (global) applies last

    def test_protected_paths_rejected(self):
        cfg = PolicyConfig.defaults()
        with pytest.raises(ProfileError):
            apply_profile_overlay(cfg, {"nutrition.pray_routine": 100}, "test")
        with pytest.raises(ProfileError):
            apply_profile_overlay(cfg, {"nutrition.corpse_fresh_turns": 999}, "test")

    def test_unknown_path_rejected(self):
        with pytest.raises(ProfileError):
            apply_profile_overlay(PolicyConfig.defaults(), {"combat.no_such": 1}, "test")

    def test_no_profiles_no_behavior_change(self):
        program = PolicyProgram.from_dict(DEFAULT_PROGRAM)
        cfg = resolve_config(program, role="valkyrie")
        assert cfg.combat.fast_ratio == PolicyConfig.defaults().combat.fast_ratio


class TestCombatManagerIntegration:
    def test_default_construction_uses_interlock_rules(self):
        mgr = TacticalCombatManager()
        assert len(mgr.tactic_engine) == len(DEFAULT_TACTIC_RULES)

    def test_explicit_rules_override(self):
        mgr = TacticalCombatManager(tactic_rules=[{"when": "(true)", "do": "retreat"}])
        assert len(mgr.tactic_engine) == 1

    def test_item_predicate_tile_naming(self):
        import numpy as np
        chars = np.full((21, 80), ord("."), dtype=int)
        chars[10, 11] = ord("\\")  # throne
        name = TacticalCombatManager._tile_item_name(chars, (10, 11), (10, 10))
        assert "throne" in name

    def test_engine_item_matching(self):
        engine = tactic_mod.TacticRuleEngine(
            [{"when": '(item "throne")', "do": "avoid"}])
        ctx = combat_ctx("kitten")
        ctx["item_name"] = "throne"
        assert engine.evaluate(ctx).verb == "avoid"