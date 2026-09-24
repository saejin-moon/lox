"""
DeadlockResolver: Resolves spatial loops, oscillation cycles, and HTN decomposition deadlocks
by querying the deliberative core for an emergency HTNGraphPatch.
"""

from typing import Any, List
from lox.deliberative.schemas import HTNGraphPatch, SubTaskSpec
from lox.deliberative.providers.base import LLMProvider
from lox.planner.htn import Task, PrimitiveTask, CompoundTask
from lox.env.blstats import BottomLineStats


def emit_deadlock_diff(diff_text: str, program, program_path: str = None) -> bool:
    """Validates a deadlock-generated diff against the current policy program and, on
    acceptance, persists the bumped program (next-episode effect — MACRO.md §10).
    Returns True iff the diff was accepted. Rejections are recorded by the caller in
    the revision ledger. Zero rejected-diff leaks: the program file is only rewritten
    by a fully validated candidate."""
    from lox.policy.manifest import build_manifest
    from lox.policy.predicates import default_ctx
    from lox.policy.validator import validate_diff, ValidatorHooks

    manifest = build_manifest(program.version, program.domain,
                              live_macros={m["name"]: m["body"] for m in program.macros})
    result = validate_diff(diff_text, program, manifest,
                           hooks=ValidatorHooks(fixture_states=[default_ctx(depth=depth_from_diff(diff_text))]))
    if result.ok and program_path is not None:
        result.candidate.save(program_path)
    return bool(result.ok)


def depth_from_diff(diff_text: str) -> int:
    """Extracts the depth_le clause value for shadow-fixture construction (best effort)."""
    import re
    m = re.search(r"depth_le (\d+)", diff_text)
    return int(m.group(1)) if m else 1


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
        """Converts SubTaskSpec entries from patch into executable Task objects.

        R3 NOTE: this transient plan-injection path is retained ONLY for backward
        compatibility; the in-game deliberation path now emits policy diffs through
        the validator (see patch_to_diff + emit_deadlock_diff). Per MACRO.md §10,
        accepted deadlock diffs take effect NEXT EPISODE — no runtime re-compile.
        """
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

    @staticmethod
    def patch_to_diff(
        patch: HTNGraphPatch,
        *,
        parent_version: int,
        failed_task: str = "",
        depth: int | None = None,
        turn: int = 0,
        domain: str = "nethack",
        author: str = "deadlock-resolver",
    ) -> str:
        """R3 unification (AGENT_PLAN step 9): converts an HTNGraphPatch into a policy
        diff through the closed vocabulary. The LLM's deadlock diagnosis becomes a
        declarative nogood — 'this task under these conditions is fatal/stalling' —
        never a transient task injection. Mask compilation lands in R4; the declarative
        entry is stored in the program's nogoods list."""
        depth_clause = f"(depth_le {int(depth)}) " if depth is not None else ""
        cause = (patch.deadlock_cause or "deadlock")[:120].replace('"', "'")
        reason = f"deadlock in {failed_task or 'unknown task'}: {cause}"
        return (
            f'(revision {parent_version + 1} (parent {parent_version}) (author "{author}") '
            f'(domain {domain}) (reason "{reason}"))\n'
            f'(nogood add (when (and {depth_clause}(turn_ge {int(turn)}))) '
            f'(cause "{cause}") (forbid {failed_task or "WAIT"}))'
        )
