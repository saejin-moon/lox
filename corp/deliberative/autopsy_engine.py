"""
Post-Mortem Autopsy Engine: Formulates causal root-cause analysis upon player death
and extracts cross-generational 64-bit CDCL Nogood clauses.
"""

from typing import Tuple, Any
from corp.deliberative.schemas import AutopsyReport, NogoodClause
from corp.deliberative.providers.base import LLMProvider
from corp.env.flight_recorder import FlightRecorderRingBuffer
from corp.planner.nogood import NogoodStore, NogoodEntry
from corp.planner.predicates import PredicateBit


SYSTEM_AUTOPSY_PROMPT = """You are the Grandmaster Post-Mortem Autopsy Engine for an elite NetHack 3.6.6 autonomous agent.
Your mission is to perform rigorous causal root-cause analysis on the agent's death and synthesize a durable 64-bit CDCL Nogood clause to prevent this failure mode in all future generations.

### Ground-Truth NetHack 3.6.6 Mechanics to Respect:
1. Floating eyes have a 0d70 passive gaze that paralyzes anyone hitting them in melee unless the attacker has reflection or is blind.
2. Barehanded contact with cockatrice corpses causes 0-turn petrification. Falling into a pit while holding one is fatal.
3. In NetHack 3.6.x, dust Elbereth erodes 100% upon monster flee; it is an escape tool, not a stationary combat bunker.
4. Safe corpse consumption cutoff is strictly <= 25 turns (due to rottenness math: age / [10..29] >= 6 is fatal food poisoning).
5. Kiting in open rooms against faster monsters (e.g. speed 18 soldier ants vs speed 12 player) gives free rear attacks; corridor funneling is required.

### Output Protocol:
1. Output your multi-step causal reasoning inside <think>...</think> tags.
2. Output a single JSON object conforming strictly to the AutopsyReport schema.
"""


class AutopsyEngine:
    """
    Analyzes game failure trajectories and compiles learned negative constraints.
    """

    def __init__(self, llm_provider: LLMProvider, nogood_store: NogoodStore | None = None):
        self.llm = llm_provider
        self.nogood_store = nogood_store or NogoodStore()

    async def execute_autopsy(
        self,
        recorder: FlightRecorderRingBuffer,
        death_message: str = "",
        generation: int = 1,
    ) -> Tuple[AutopsyReport, NogoodEntry]:
        """
        Executes causal autopsy call and registers synthesized Nogood into memory.
        """
        history_summary = recorder.export_autopsy_summary(recent_turns=20)

        user_prompt = (
            f"Game Over Notification: '{death_message}'\n\n"
            f"{history_summary}\n\n"
            "Analyze the root cause of death, identify the fatal primitive action, "
            "and formulate the exact Nogood trigger predicates to prune this failure mode forever."
        )

        response = await self.llm.generate_reasoning_and_json(
            system_prompt=SYSTEM_AUTOPSY_PROMPT,
            user_prompt=user_prompt,
            schema=AutopsyReport,
        )

        report: AutopsyReport = response.parsed_payload
        nogood_entry = self._compile_clause_to_entry(report.nogood, generation=generation)

        # Register in episodic memory
        self.nogood_store.add_nogood(nogood_entry)

        return report, nogood_entry

    def _compile_clause_to_entry(self, clause: NogoodClause, generation: int) -> NogoodEntry:
        """Translates high-level predicate dictionary into 64-bit Nogood mask."""
        mask = 0
        target = 0

        pred_map = {
            "adjacent_floating_eye": PredicateBit.ADJACENT_FLOATING_EYE,
            "adjacent_cockatrice": PredicateBit.ADJACENT_COCKATRICE,
            "adjacent_rust_monster": PredicateBit.ADJACENT_RUST_MONSTER,
            "adjacent_monster_faster": PredicateBit.ADJACENT_MONSTER_FASTER,
            "is_blind": PredicateBit.IS_BLIND,
            "reflection_active": PredicateBit.REFLECTION_ACTIVE,
            "gloves_equipped": PredicateBit.GLOVES_EQUIPPED,
            "hp_critical": PredicateBit.HP_CRITICAL_BELOW_20_PCT,
            "corridor_chokepoint": PredicateBit.CORRIDOR_CHOKEPOINT,
            "tainted_corpse": PredicateBit.TARGET_CORPSE_TAINTED_AGE,
        }

        for key, val in clause.trigger_predicates.items():
            k_lower = key.lower()
            if k_lower in pred_map:
                bit = pred_map[k_lower]
                mask |= bit
                if bool(val):
                    target |= bit

        return NogoodEntry(
            mask=mask,
            target_val=target,
            forbidden_action=clause.fatal_action.upper(),
            reason=clause.derived_constraint,
            generation=generation,
        )
