"""
MockProvider: Deterministic offline LLM provider for unit tests and local simulation.
"""

from typing import Any, Type, TypeVar
import time
from pydantic import BaseModel

from corp.deliberative.providers.base import LLMProvider, LLMResponse
from corp.deliberative.response_parser import ResponseParser

T = TypeVar("T", bound=BaseModel)


class MockProvider(LLMProvider):
    """
    Returns scripted or auto-generated responses for offline unit testing.

    R3: `generate_text` cycles through canned policy-diff templates (formatted with the
    current program version from `context`) — one valid `set`, one valid `rule add`,
    one out-of-vocab (rejection test) — per the R3 mock-first development protocol.
    """

    CANNED_DIFF_TEMPLATES: list[str] = [
        # 1. valid set op (accepted)
        '(revision {revision} (parent {parent}) (author "mock") (domain nethack) (reason "mock: widen resting gate — stall telemetry shows HP-resting too timid"))\n'
        '(set policy_params.survival.rest_below_frac 0.65)',
        # 2. valid rule add (accepted)
        '(revision {revision} (parent {parent}) (author "mock") (domain nethack) (reason "mock: mines attrition — retreat when wounded underground"))\n'
        '(rule add tactic_rules (when (and (in_mines) (hp_frac <= 0.40))) (do retreat) (note "mines attrition fix"))',
        # 3. out-of-vocab param (rejected: ERR_BOUNDS)
        '(revision {revision} (parent {parent}) (author "mock") (domain nethack) (reason "mock: attempt an unknown param"))\n'
        '(set policy_params.survival.no_such_param 0.5)',
        # 4. valid defmacro + use (accepted)
        '(revision {revision} (parent {parent}) (author "mock") (domain nethack) (reason "mock: compose when_endangered macro"))\n'
        '(defmacro when_endangered (and (adjacent_hostiles) (hp_frac <= 0.40) (not has_healing)))\n'
        '(rule add tactic_rules (when (and (in_dungeons) (when_endangered))) (do retreat))',
    ]

    def __init__(self, canned_response: str | None = None):
        self.canned_response = canned_response
        self.call_history: list[dict[str, Any]] = []
        self._text_call_count = 0

    def set_canned_response(self, text: str):
        self.canned_response = text

    async def generate_reasoning_and_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
    ) -> LLMResponse:
        start_time = time.perf_counter()
        self.call_history.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": schema.__name__,
        })

        raw_text = self.canned_response
        if not raw_text:
            if schema.__name__ == "AutopsyReport":
                raw_text = """<think>
Root cause analysis: Character died in tactical situation.
Fatal action identified. Formulating Nogood constraint.
</think>
```json
{
  "death_cause": "Fatal combat or hazard encounter",
  "lethal_turn": 100,
  "causal_chain": ["Encountered danger", "Accumulated damage", "HP reduced to 0"],
  "counterfactual_fix": "Retreat to corridor and rest to full HP before engaging",
  "nogood": {
    "trigger_predicates": {
      "adjacent_floating_eye": false
    },
    "fatal_action": "MELEE_ATTACK",
    "derived_constraint": "Avoid reckless melee without adequate HP"
  }
}
```"""
            elif schema.__name__ == "HTNGraphPatch":
                raw_text = """<think>
Resolving deadlock: Injecting search and step sequence to break spatial oscillation.
</think>
```json
{
  "deadlock_cause": "Spatial oscillation or exploration stall",
  "confidence": 0.90,
  "abandon_current_macro": false,
  "injected_subtasks": [
    {"task_name": "SEARCH"},
    {"task_name": "SEARCH"},
    {"task_name": "STEP", "target_direction": "l"}
  ],
  "new_invariants": []
}
```"""
            else:
                raise ValueError(f"MockProvider has no canned response for schema {schema.__name__}.")

        thinking, parsed = ResponseParser.parse_dual_phase(raw_text, schema)
        latency = (time.perf_counter() - start_time) * 1000.0

        return LLMResponse(
            thinking_content=thinking,
            parsed_payload=parsed,
            raw_text=raw_text,
            tokens_consumed=len(raw_text.split()),
            latency_ms=latency,
        )

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        context: dict | None = None,
    ) -> LLMResponse:
        start_time = time.perf_counter()
        self.call_history.append({
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "context": context,
        })
        version = int((context or {}).get("version", 1))
        if self.canned_response:
            raw = self.canned_response.format(revision=version + 1, parent=version)
        else:
            template = self.CANNED_DIFF_TEMPLATES[self._text_call_count % len(self.CANNED_DIFF_TEMPLATES)]
            raw = template.format(revision=version + 1, parent=version)
        self._text_call_count += 1
        latency = (time.perf_counter() - start_time) * 1000.0
        return LLMResponse(
            thinking_content="",
            parsed_payload=raw,
            raw_text=raw,
            tokens_consumed=len(raw.split()),
            latency_ms=latency,
            tokens_in=len(system_prompt.split()) + len(user_prompt.split()),
            tokens_out=len(raw.split()),
            tokens_thought=0,
        )
