"""
DeadlockResolver: Resolves spatial loops, oscillation cycles, and HTN decomposition deadlocks
by querying the deliberative core for an emergency HTNGraphPatch.
"""

from typing import Any, List
from corp.deliberative.schemas import HTNGraphPatch, SubTaskSpec
from corp.deliberative.providers.base import LLMProvider
from corp.planner.htn import Task, PrimitiveTask, CompoundTask
from corp.env.blstats import BottomLineStats


SYSTEM_DEADLOCK_PROMPT = """You are the Strategic Tactical Advisor for an autonomous NetHack 3.6.6 agent.
The agent's fast HTN planner has encountered a deadlock or oscillation cycle and requires a high-level strategic plan repair.

### Guidelines:
1. Reason carefully inside <think>...</think> tags.
2. Output a valid JSON object matching the HTNGraphPatch schema.
3. Recommend safe actions (e.g. retreating to a chokepoint, resting to regenerate HP, or abandoning an impossible goal).
"""


class DeadlockResolver:
    """
    Handles plan repairs during Anomaly Deadlock events.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def resolve_deadlock(
        self,
        failed_task: str,
        blstats: BottomLineStats,
        cycle_detected: bool = False,
        extra_context: str = "",
    ) -> HTNGraphPatch:
        """
        Executes plan repair request to the LLM core.
        """
        prompt = (
            f"HTN Deadlock Event:\n"
            f"- Failed Task: {failed_task}\n"
            f"- Turn: {blstats.turn} | HP: {blstats.hp}/{blstats.max_hp} | Depth: {blstats.depth}\n"
            f"- Position: ({blstats.x}, {blstats.y}) | Hunger: {blstats.hunger_state}\n"
            f"- Oscillation Cycle Detected: {cycle_detected}\n"
            f"- Context: {extra_context}\n\n"
            "Formulate an HTNGraphPatch to safely bypass this deadlock."
        )

        response = await self.llm.generate_reasoning_and_json(
            system_prompt=SYSTEM_DEADLOCK_PROMPT,
            user_prompt=prompt,
            schema=HTNGraphPatch,
        )

        return response.parsed_payload

    @staticmethod
    def patch_to_tasks(patch: HTNGraphPatch) -> List[Task]:
        """Converts SubTaskSpec entries from patch into executable Task objects."""
        tasks: list[Task] = []
        for spec in patch.injected_subtasks:
            # If task name is a known primitive like STEP, MELEE_ATTACK, SEARCH, WAIT
            is_primitive = spec.task_name in (
                "STEP", "MELEE_ATTACK", "SEARCH", "WAIT", "EAT", "QUAFF", "READ", "APPLY", "PRAY"
            )
            args = {
                "slot": spec.target_slot,
                "direction": spec.target_direction,
                "coordinate": spec.target_coordinate,
            }
            if is_primitive:
                tasks.append(PrimitiveTask(name=spec.task_name, args=args))
            else:
                tasks.append(CompoundTask(name=spec.task_name, args=args))
        return tasks
