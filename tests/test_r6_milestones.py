"""R6 Ascension Knowledge Stack: milestone certification tests.

§8 contract: each milestone lands as goal handlers (code) + default-program
entries + certification cases, THEN the loop may tune it. These tests are the
per-milestone certification: goal gating (no milestone may be skipped-to),
completion semantics, directive wiring, inventory/message-derived milestone
flags, vocabulary-manifest closure, and owned-param tunability.
"""

import numpy as np
import pytest

from lox.env.blstats import BottomLineStats, HungerState
from lox.env.inventory_tracker import InventoryNormalizer, NormalizedItem
from lox.policy.config import PolicyConfig
from lox.policy.goal_interpreter import GoalInterpreter
from lox.policy.goal_state import AscensionPhase, MacroDirectorState


def make_bl(
    xl: int = 1, depth: int = 1, dnum: int = 0, dlevel: int = 1,
    ac: int = 10, alignment: int = 1, turn: int = 100,
    hp: int = 40, max_hp: int = 40,
) -> BottomLineStats:
    raw = np.zeros(27, dtype=np.int64)
    raw[0], raw[1] = 10, 10
    raw[10], raw[11] = hp, max_hp
    raw[12] = depth
    raw[16] = ac
    raw[18] = xl
    raw[20] = turn
    raw[23] = dnum
    raw[24] = dlevel
    raw[26] = alignment
    return BottomLineStats.from_blstats(raw)


def make_gi(**strategy_over) -> GoalInterpreter:
    gi = GoalInterpreter(PolicyConfig.defaults(), program_path="")
    for k, v in strategy_over.items():
        setattr(gi.cfg.strategy, k, v)
    return gi


def inv_with(*descriptions: str) -> InventoryNormalizer:
    inv = InventoryNormalizer()
    for i, d in enumerate(descriptions):
        it = NormalizedItem(current_letter=chr(ord("a") + i), raw_str=d, buc_state="UNCURSED")
        inv.active_items[it.uid] = it
    return inv


def goal_index(gi: GoalInterpreter, name: str) -> int:
    return [g.goal for g in gi.program.strategy_plan].index(name)


# ---------------------------------------------------------------------------
# Milestone 1: early_survival_stack
# ---------------------------------------------------------------------------

def test_m1_activates_after_explore_before_anything_deep():
    gi = make_gi()
    bl = make_bl(xl=5, depth=3)  # explore's until (xl>=4) met at first update
    gi.update_state(bl, role="valkyrie")
    assert gi.program.strategy_plan[gi._goal_index].goal == "early_survival_stack"
    assert gi.state.current_phase == AscensionPhase.SURVIVAL_STACK


def test_m1_completes_on_ac_target_param():
    gi = make_gi(survival_ac_target=2)
    bl = make_bl(xl=5, depth=3, ac=2, alignment=0)
    for _ in range(2):  # 1: explore completes → m1 activates; 2: ac target met → advance
        gi.update_state(bl, role="valkyrie")
    assert gi.program.strategy_plan[gi._goal_index].goal == "forge_excalibur"


def test_m1_completes_on_exit_depth_param():
    gi = make_gi(survival_exit_depth=4)
    bl = make_bl(xl=5, depth=4, alignment=0)
    for _ in range(2):
        gi.update_state(bl, role="valkyrie")
    assert gi.program.strategy_plan[gi._goal_index].goal == "forge_excalibur"


def test_m1_completes_on_timeout_param():
    gi = make_gi(survival_timeout_steps=10)
    bl = make_bl(xl=5, depth=3, ac=10, alignment=0)
    for _ in range(14):  # 1 explore + 12 m1 (timeout at 10) + 1 advance
        gi.update_state(bl, role="valkyrie")
    assert gi.program.strategy_plan[gi._goal_index].goal == "forge_excalibur"


def test_m1_holds_while_ac_worse_than_target():
    gi = make_gi(survival_timeout_steps=10_000)
    bl = make_bl(xl=5, depth=3, ac=6)  # 6 > 4, depth 3 < 7
    for _ in range(50):
        gi.update_state(bl, role="valkyrie")
    assert gi.program.strategy_plan[gi._goal_index].goal == "early_survival_stack"


# ---------------------------------------------------------------------------
# Milestone 2: equip_upgrade
# ---------------------------------------------------------------------------

def test_m2_activation_and_completion():
    gi = make_gi()
    # Neutral alignment skips forge; poison message completes hunt; reflection
    # completes sokoban → equip_upgrade when-gates on reflection-or-depth-6.
    refl = inv_with("an uncursed amulet of reflection")
    bl = make_bl(xl=6, depth=3, ac=0, alignment=0)
    for _ in range(20):
        gi.update_state(bl, inv_tracker=refl, role="monk", message="You feel healthy.")
    assert gi.program.strategy_plan[gi._goal_index].goal == "equip_upgrade"
    assert gi.state.current_phase == AscensionPhase.EQUIP_UPGRADE

    # AC <= -10 completes it → goto_minetown (exactly one advance: its own until
    # (ac_le 0) is already true, so a second update would complete it as well)
    gi.update_state(make_bl(xl=6, depth=3, ac=-10, alignment=0),
                    inv_tracker=refl, role="monk")
    assert gi.program.strategy_plan[gi._goal_index].goal == "goto_minetown"


# ---------------------------------------------------------------------------
# Milestone 3: survival_intrinsics (activation gate = depth param)
# ---------------------------------------------------------------------------

def test_m3_cannot_skip_to_below_min_depth():
    gi = make_gi()
    inv = inv_with("an uncursed amulet of reflection", "a wand of wishing")
    # At DL 3 with AC -10 (upstream goals complete instantly) the walk reaches
    # the intrinsics slot but its depth gate (8) fails → skipped permanently.
    bl = make_bl(xl=9, depth=3, ac=-10, alignment=0)
    for _ in range(20):
        gi.update_state(bl, inv_tracker=inv, role="monk", message="You feel healthy.")
    assert goal_index(gi, "survival_intrinsics") in gi._skipped
    assert gi.program.strategy_plan[gi._goal_index].goal == "descend"


def test_m3_activates_at_min_depth_and_completes_on_mr():
    gi = make_gi(intrinsics_min_depth=8)
    inv = inv_with("gray dragon scale mail", "an uncursed amulet of reflection")
    bl = make_bl(xl=12, depth=8, ac=-12, alignment=0)
    for _ in range(20):
        gi.update_state(bl, inv_tracker=inv, role="monk")
    # MR flag set by the same inventory scan that gates completion; intrinsics
    # completes on it → walk lands on descend (gehennom gates at 14)
    assert gi.state.mr_obtained is True
    assert gi.program.strategy_plan[gi._goal_index].goal == "descend"


# ---------------------------------------------------------------------------
# Milestones 4-6: gehennom / castle / vlad gates
# ---------------------------------------------------------------------------

def test_m4_5_6_depth_gates_ordered():
    gi = make_gi()
    assert gi.cfg.strategy.gehennom_min_depth < gi.cfg.strategy.castle_min_depth
    assert gi.cfg.strategy.castle_min_depth < gi.cfg.strategy.vlad_min_depth


def test_m4_enters_at_depth_and_emits_descend():
    gi = make_gi(gehennom_min_depth=14)
    inv = inv_with("an uncursed oil lamp", "gray dragon scale mail",
                   "an uncursed amulet of reflection")
    bl = make_bl(xl=14, depth=14, ac=-12, alignment=0)
    for _ in range(20):
        gi.update_state(bl, inv_tracker=inv, role="monk")
    d = gi.get_navigation_directive(bl, None)
    assert d == "DESCEND"
    assert gi.state.current_phase == AscensionPhase.GEHENNOM


def test_m5_castle_completes_on_prize():
    gi = make_gi()
    inv = inv_with("a wand of wishing", "a bag of holding",
                   "gray dragon scale mail", "an uncursed oil lamp")
    bl = make_bl(xl=20, depth=24, ac=-12)
    for _ in range(3):
        gi.update_state(bl, inv_tracker=inv, role="valkyrie")
    assert gi.state.castle_wishing_done is True


def test_m6_vlad_defeat_implies_candelabrum():
    gi = make_gi()
    gi.update_state(make_bl(xl=25, depth=33, ac=-12),
                    inv_tracker=inv_with("the Candelabrum of Invocation"), role="valkyrie")
    st = gi.state
    assert st.candelabrum_obtained and st.vlad_defeated


# ---------------------------------------------------------------------------
# Milestone 7: ascension_run
# ---------------------------------------------------------------------------

def test_m7_requires_amulet_and_ascends_with_it():
    gi = make_gi()
    # Without the amulet the walk can never reach ascension_run at low depth:
    gi.update_state(make_bl(xl=1, depth=1), role="valkyrie")
    assert goal_index(gi, "ascension_run") != gi._goal_index

    # With the amulet at depth: directive ASCEND (climb the Planes)
    gi2 = make_gi()
    bl = make_bl(xl=30, depth=40, ac=-20)
    amulet = inv_with("the Amulet of Yendor")
    for _ in range(3):
        gi2.update_state(bl, inv_tracker=amulet, role="valkyrie")
    assert gi2.state.amulet_obtained
    # ascension_run's when (has_amulet) passes once the walk reaches it —
    # force-activate to certify the directive contract directly
    assert gi2.activate_goal("ascension_run")
    assert gi2.get_navigation_directive(bl, None) == "ASCEND"


# ---------------------------------------------------------------------------
# Vocabulary closure + tunability (loop may re-prioritize only via owned params)
# ---------------------------------------------------------------------------

def test_all_r6_goals_certified_in_manifest_and_adapter():
    from lox.policy.manifest import CERTIFIED_GOALS
    from lox.executor.nethack_adapter import NethackAdapter

    r6 = {"early_survival_stack", "equip_upgrade", "survival_intrinsics",
          "enter_gehennom", "castle_wishing", "vlad_invocation", "ascension_run"}
    assert r6 <= set(CERTIFIED_GOALS)
    adapter_goals = NethackAdapter().goal_handlers()
    assert r6 <= set(adapter_goals)
    # owned params agree between manifest and adapter
    for name in r6:
        assert sorted(CERTIFIED_GOALS[name]["params"]) == sorted(adapter_goals[name].owned_params)


def test_all_plan_goals_evaluable_on_fixture_ctx():
    """Every plan goal's when/until must parse+evaluate on the default ctx
    (R6 predicates included) — the drift guard for the 13-goal plan."""
    from lox.policy.predicates import eval_condition, nethack_bindings, default_ctx

    gi = make_gi()
    for g in gi.program.strategy_plan:
        ctx = nethack_bindings(default_ctx())
        assert isinstance(eval_condition(g.when, ctx), bool), g.goal
        assert isinstance(eval_condition(g.until, ctx), bool), g.goal


def test_set_threshold_on_owned_param_tunes_milestone():
    """set_threshold → params overlay → interpreter honors the tuned threshold."""
    from lox.policy.program import PolicyProgram

    prog = PolicyProgram.load("data/policy_program.json")
    prog.params["strategy.survival_ac_target"] = 0
    gi = GoalInterpreter.__new__(GoalInterpreter)
    from lox.policy.config import PolicyConfig as PC
    gi.cfg = PC.defaults()
    prog.apply_overlay(gi.cfg)
    assert gi.cfg.strategy.survival_ac_target == 0


def test_default_state_has_r6_flags():
    st = MacroDirectorState()
    for f in ("mr_obtained", "light_source_carried", "wand_of_wishing_carried",
              "amulet_obtained", "candelabrum_obtained", "bell_of_opening_obtained",
              "book_of_the_dead_obtained", "castle_wishing_done", "vlad_defeated",
              "invocation_done"):
        assert getattr(st, f) is False


def test_light_and_message_flags():
    gi = make_gi()
    gi.update_state(make_bl(xl=5, depth=3),
                    inv_tracker=inv_with("a brass lantern"), role="valkyrie")
    assert gi.state.light_source_carried
    gi2 = make_gi()
    gi2.update_state(make_bl(xl=5, depth=3), message="You feel the Vibrating Square!")
    assert gi2.state.invocation_done
