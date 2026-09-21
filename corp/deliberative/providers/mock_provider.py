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
        '(revision {revision} (parent {parent}) (author "mock") (domain {domain}) (reason "mock: widen resting gate — stall telemetry shows HP-resting too timid"))\n'
        '(set policy_params.{rest_param} {rest_value})',
        # 2. valid rule add (accepted) — combat rules are target-conditional
        # (1a invariant; macro bodies are parameterless, so the target match
        # lives in the rule's own when)
        '(revision {revision} (parent {parent}) (author "mock") (domain {domain}) (reason "mock: mines attrition — retreat when wounded underground"))\n'
        '{rule_op}',
        # 3. out-of-vocab param (rejected: ERR_BOUNDS)
        '(revision {revision} (parent {parent}) (author "mock") (domain {domain}) (reason "mock: attempt an unknown param"))\n'
        '(set policy_params.survival.no_such_param 0.5)',
        # 4. valid defmacro + use (accepted)
        '(revision {revision} (parent {parent}) (author "mock") (domain {domain}) (reason "mock: compose when_endangered macro"))\n'
        '(defmacro when_endangered (and (hp_frac <= 0.40) (not has_healing)))\n'
        '{macro_rule_op}',
    ]

    DOMAIN_TEMPLATES: dict[str, dict] = {
        "nethack": {
            "rest_param": "survival.rest_below_frac", "rest_value": "0.65",
            "rule_op": '(rule add tactic_rules (when (and (in_mines) (monster "jackal") (hp_frac <= 0.40))) (do retreat) (note "mines attrition fix"))',
            "macro_rule_op": '(rule add tactic_rules (when (and (in_dungeons) (monster "goblin") (when_endangered))) (do retreat))',
        },
        "minihack": {
            "rest_param": "explore.stuck_patience", "rest_value": "40",
            "rule_op": '(rule add tactic_rules (when (and (monster "j") (hp_frac <= 0.30))) (do retreat) (note "wounded: avoid monsters"))',
            "macro_rule_op": '(rule add tactic_rules (when (and (monster "j") (when_endangered))) (do retreat))',
        },
    }

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
        domain = (context or {}).get("domain", "nethack")
        fills = dict(self.DOMAIN_TEMPLATES.get(domain, self.DOMAIN_TEMPLATES["nethack"]))
        fills.update({"revision": version + 1, "parent": version, "domain": domain})
        if self.canned_response:
            raw = self.canned_response.format(revision=version + 1, parent=version, domain=domain)
        else:
            template = self.CANNED_DIFF_TEMPLATES[self._text_call_count % len(self.CANNED_DIFF_TEMPLATES)]
            raw = template.format(**fills)
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
