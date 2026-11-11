"""
The Policy Modification Modes of LOX:
1. `CREATE`: Initial full policy synthesis (default: 250 episodes).
2. `WRITE`: Targeted function modification/addition (default: 150 triggered episodes, max 2 functions).
3. `REFACTOR`: Subsystem modularization into `data/workspace/skills/` (default: 200 episodes).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModeConfig:
    """Configuration for a policy modification mode."""

    name: str
    episodes_per_unit: int
    requires_llm: bool
    requires_trigger_filter: bool


MODES: dict[str, ModeConfig] = {
    "CREATE": ModeConfig(
        name="CREATE",
        episodes_per_unit=250,
        requires_llm=True,
        requires_trigger_filter=False,
    ),
    "WRITE": ModeConfig(
        name="WRITE",
        episodes_per_unit=150,
        requires_llm=True,
        requires_trigger_filter=True,
    ),
    "REFACTOR": ModeConfig(
        name="REFACTOR",
        episodes_per_unit=200,
        requires_llm=True,
        requires_trigger_filter=False,
    ),
    "FULL_WRITE": ModeConfig(
        name="CREATE",
        episodes_per_unit=250,
        requires_llm=True,
        requires_trigger_filter=False,
    ),
    "FUNC_MOD_ADD": ModeConfig(
        name="WRITE",
        episodes_per_unit=150,
        requires_llm=True,
        requires_trigger_filter=True,
    ),
}
