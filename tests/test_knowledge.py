"""
Tests for LOX Empirical Knowledge Base and Invariant Registry.
"""

from lox.author.prompts import build_user_prompt
from lox.author.tools import DuckDBToolRegistry
from lox.knowledge import REGISTRY


def test_registry_basics():
    all_invs = REGISTRY.get_all()
    assert len(all_invs) >= 30

    ids = [inv.id for inv in all_invs]
    assert len(ids) == len(set(ids)), "Invariant IDs must be unique"

    for inv in all_invs:
        assert inv.id.startswith("INV-")
        assert len(inv.title) > 0
        assert inv.category in {
            "combat",
            "navigation",
            "nutrition",
            "equipment",
            "dialog_harness",
            "architecture",
        }
        assert len(inv.tags) > 0
        assert len(inv.rule) > 20
        assert len(inv.anti_pattern) > 10
        assert len(inv.code_snippet.strip()) > 0


def test_registry_lookup_by_id():
    nav1 = REGISTRY.get_by_id("INV-NAV-001")
    assert nav1 is not None
    assert "descend" in nav1.tags or "stairs" in nav1.tags
    assert "descend()" in nav1.code_snippet

    missing = REGISTRY.get_by_id("NONEXISTENT")
    assert missing is None


def test_registry_category_filtering():
    combat_invs = REGISTRY.get_by_category("combat")
    assert len(combat_invs) >= 8
    for inv in combat_invs:
        assert inv.category == "combat"

    nav_invs = REGISTRY.get_by_category("navigation")
    assert len(nav_invs) >= 6
    for inv in nav_invs:
        assert inv.category == "navigation"

    nut_invs = REGISTRY.get_by_category("nutrition")
    assert len(nut_invs) >= 4
    for inv in nut_invs:
        assert inv.category == "nutrition"


def test_registry_search():
    results = REGISTRY.search("gas spore explosion")
    assert len(results) > 0
    assert any(inv.id == "INV-CBT-002" for inv in results)

    results_prayer = REGISTRY.search("divine favor 850 turns prayer")
    assert len(results_prayer) > 0
    assert any(inv.id == "INV-NUT-004" for inv in results_prayer)

    results_cat = REGISTRY.search("door", category="navigation")
    for inv in results_cat:
        assert inv.category == "navigation"


def test_trigger_relevance():
    spore_invs = REGISTRY.get_relevant_invariants_for_trigger(
        "cluster: Killed in combat: gas spore"
    )
    spore_ids = [inv.id for inv in spore_invs]
    assert "INV-CBT-002" in spore_ids

    starve_invs = REGISTRY.get_relevant_invariants_for_trigger(
        "cluster: Starvation / Hunger fainting"
    )
    starve_ids = [inv.id for inv in starve_invs]
    assert "INV-NUT-003" in starve_ids or "INV-NUT-001" in starve_ids

    passive_invs = REGISTRY.get_relevant_invariants_for_trigger(
        "stuck in combat with passive hazard"
    )
    passive_ids = [inv.id for inv in passive_invs]
    assert "INV-CBT-009" in passive_ids


def test_formatting_for_llm():
    inv = REGISTRY.get_by_id("INV-NAV-001")
    assert inv is not None
    block = inv.format_prompt_block()
    assert "### [INV-NAV-001]" in block
    assert "**Rule**:" in block
    assert "**Avoid**:" in block
    assert "```python" in block

    ref_text = REGISTRY.format_llm_reference([inv])
    assert "Mandatory Empirical Tactical Invariants" in ref_text
    assert "INV-NAV-001" in ref_text


def test_author_tools_query_invariants():
    tools = DuckDBToolRegistry(db_path=":memory:", wiki_db_path=":memory:")
    output = tools.query_invariants("floating eye paralysis")
    assert "Mandatory Empirical Tactical Invariants" in output
    assert "INV-CBT-004" in output

    no_match = tools.query_invariants("zyxwvutsrqp12345")
    assert "No invariants found matching" in no_match


def test_prompt_injection_with_invariants():
    prompt = build_user_prompt(
        current_policy="class Agent: pass",
        trigger_reason="cluster: Killed in combat: gas spore",
        status_report="Top killer: gas spore (12 deaths)",
    )
    assert "Mandatory Empirical Tactical Invariants" in prompt
    assert "INV-CBT-002" in prompt
    assert "class Agent: pass" in prompt


def test_new_invariants_and_system_prompt():
    from lox.author.prompts import build_system_prompt

    # Invariants exist and can be retrieved
    cbt12 = REGISTRY.get_by_id("INV-CBT-012")
    assert cbt12 is not None and "corrosive" in cbt12.tags
    assert "is_corrosive_target" in cbt12.code_snippet

    cbt13 = REGISTRY.get_by_id("INV-CBT-013")
    assert cbt13 is not None and "heavy weapon" in cbt13.tags
    assert "is_heavy_weapon_threat" in cbt13.code_snippet

    cbt14 = REGISTRY.get_by_id("INV-CBT-014")
    assert cbt14 is not None and "emergency" in cbt14.tags
    assert "quaff_emergency_potion" in cbt14.code_snippet

    nav7 = REGISTRY.get_by_id("INV-NAV-007")
    assert nav7 is not None and "sokoban" in nav7.tags
    assert "step_to_sokoban_entrance" in nav7.code_snippet

    eqp7 = REGISTRY.get_by_id("INV-EQP-007")
    assert eqp7 is not None and "armor" in eqp7.tags

    # Trigger heuristics match
    invs_corrosive = REGISTRY.get_relevant_invariants_for_trigger("cluster: Killed by corrosive acid blob")
    assert any(inv.id == "INV-CBT-012" for inv in invs_corrosive)

    invs_heavy = REGISTRY.get_relevant_invariants_for_trigger("cluster: Killed by heavy weapon orc captain")
    assert any(inv.id == "INV-CBT-013" for inv in invs_heavy)

    invs_sokoban = REGISTRY.get_relevant_invariants_for_trigger("milestone: sokoban entrance discovered")
    assert any(inv.id == "INV-NAV-007" for inv in invs_sokoban)

    # System prompt covers the new predicates and actions
    sys_prompt = build_system_prompt()
    assert "is_corrosive_target" in sys_prompt
    assert "is_heavy_weapon_threat" in sys_prompt
    assert "has_sokoban_entrance" in sys_prompt
    assert "step_to_sokoban_entrance" in sys_prompt
    assert "quaff_emergency_potion" in sys_prompt
    assert "read_emergency_scroll" in sys_prompt
    assert "phase_early_rush" in sys_prompt
    assert "phase_early_scaling" in sys_prompt
    assert "phase_mid_branches" in sys_prompt
    assert "phase_deep_dungeon" in sys_prompt

