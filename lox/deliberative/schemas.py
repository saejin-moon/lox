"""
Pydantic schemas for structured LLM reasoning and structured outputs.
"""

from typing import Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class SubTaskSpec(BaseModel):
    task_name: str = Field(description="Name of primitive or compound task")
    target_slot: Optional[str] = Field(None, description="Inventory letter or UID if applicable")
    target_direction: Optional[str] = Field(None, description="Direction key: h,j,k,l,y,u,b,n")
    target_coordinate: Optional[Tuple[int, int]] = Field(None, description="(row, col) map position")


class HTNGraphPatch(BaseModel):
    deadlock_cause: str = Field(description="Causal summary of why HTN failed to decompose")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the proposed plan repair")
    abandon_current_macro: bool = Field(description="Whether to purge active compound goal")
    injected_subtasks: List[SubTaskSpec] = Field(description="Ordered sequence of tasks to execute")
    new_invariants: List[str] = Field(default_factory=list, description="Guard conditions that must hold")


class NogoodClause(BaseModel):
    trigger_predicates: dict[str, Any] = Field(
        description="Boolean state conditions characterizing the trap (e.g. adjacent_monster, hp_pct)"
    )
    fatal_action: str = Field(description="The exact primitive action that caused fatal damage")
    derived_constraint: str = Field(description="The negative rule to compile into HTN branch guards")


class AutopsyReport(BaseModel):
    death_cause: str = Field(description="Official game over reason")
    lethal_turn: int = Field(description="Game turn where fatal damage was received")
    causal_chain: List[str] = Field(description="Multi-step progression leading to death")
    counterfactual_fix: str = Field(description="What action would have preserved survival")
    nogood: NogoodClause = Field(description="Compiled negative constraint for episodic memory")
