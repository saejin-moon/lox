"""
S1 tests: the behavior-primitive manifest (AGENT_PLAN §1.2).

Guarantees:
  * every task the dispatcher accepts is covered by a primitive;
  * preconditions are either parseable over the closed predicate vocabulary or
    explicit "mechanism:*" notes (never free prose pretending to be logic);
  * interlocks reference the AGENTS.md §4 guardrails;
  * the author-facing renderer exposes the whole plan vocabulary.
"""
from __future__ import annotations

import os
import re

import pytest

from lox.policy.primitives import (
    COMPOSITE_TASKS, DISPATCHER_TASKS, PRIMITIVES, primitive_for_task,
    render_primitives,
)
from lox.policy.predicates import parse as parse_condition


def _dispatcher_create_fiber_task_names() -> set[str]:
    """Extracts the task names ActionDispatcher.create_fiber handles (drift guard
    against lox/workers/dispatcher.py)."""
    src = open(os.path.join("lox", "workers", "dispatcher.py"), encoding="utf-8").read()
    seg = src[src.index("def create_fiber"):src.index("def dispatch")]
    names = set(re.findall(r'name == "([A-Z_]+)"', seg))
    for group in re.findall(r"name in \(([^)]*)\)", seg):
        names |= set(re.findall(r'"([A-Z_]+)"', group))
    return names


class TestCoverage:
    def test_every_dispatcher_task_covered(self):
        handled = _dispatcher_create_fiber_task_names()
        assert handled, "dispatcher extraction failed"
        missing = handled - DISPATCHER_TASKS
        assert not missing, f"dispatcher tasks without a primitive: {sorted(missing)}"

    def test_every_primitive_targets_a_real_task(self):
        handled = _dispatcher_create_fiber_task_names()
        for p in PRIMITIVES.values():
            assert p.task in handled, f"primitive {p.name} targets unknown task {p.task}"

    def test_plan_vocabulary_present(self):
        """AGENT_PLAN §1.2's named plan primitives must all exist."""
        required = {"goto", "pickup", "eat", "wield", "wear", "puton", "quaff",
                    "read", "zap", "apply", "engrave", "descend", "ascend",
                    "pray", "search", "attack", "wait"}
        assert required <= set(PRIMITIVES)

    def test_no_duplicate_task_or_alias(self):
        seen: dict[str, str] = {}
        for p in PRIMITIVES.values():
            for task in (p.task,) + p.aliases:
                assert task not in seen, f"{task} assigned to both {seen[task]} and {p.name}"
                seen[task] = p.name

    def test_composite_tasks_are_not_primitives(self):
        assert not (COMPOSITE_TASKS & DISPATCHER_TASKS)


class TestPreconditions:
    @pytest.mark.parametrize("name", sorted(PRIMITIVES))
    def test_preconditions_are_machine_readable(self, name):
        p = PRIMITIVES[name]
        assert p.preconditions, f"{name} has no preconditions"
        for pre in p.preconditions:
            if pre.startswith("mechanism:"):
                assert len(pre) > len("mechanism:"), f"{name}: empty mechanism note"
                continue
            # must parse over the closed predicate vocabulary
            node = parse_condition(pre)
            assert node[0] == "call", f"{name}: precondition is not a predicate call: {pre}"

    @pytest.mark.parametrize("name", sorted(PRIMITIVES))
    def test_effects_and_desc_nonempty(self, name):
        p = PRIMITIVES[name]
        assert p.effects.strip() and p.desc.strip()

    def test_interlocks_reference_agents_section4(self):
        for p in PRIMITIVES.values():
            if p.interlock:
                assert p.interlock.startswith("§4."), \
                    f"{p.name}: interlock must cite AGENTS.md §4: {p.interlock}"


class TestRendering:
    def test_render_contains_every_primitive(self):
        text = render_primitives()
        for name in PRIMITIVES:
            assert f"| {name} |" in text

    def test_reverse_lookup(self):
        assert primitive_for_task("STEP").name == "goto"
        assert primitive_for_task("ZAP_WAND").name == "zap"
        assert primitive_for_task("MELEE_ATTACK").name == "attack"
        assert primitive_for_task("NOPE") is None

    def test_env_schema_includes_primitives(self):
        from lox.policy.author_tools import read_env_schema
        text = read_env_schema()
        assert "BEHAVIOR PRIMITIVES" in text
        assert "| goto |" in text and "| pray |" in text