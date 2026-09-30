"""
LOX 2.0 Author Prompts: High-signal system prompts for LLM policy synthesis.
"""
from __future__ import annotations

from lox.dsl.schema import ALLOWED_PREDICATES, ALLOWED_ACTIONS, ENUM_CONSTANTS


def build_system_prompt() -> str:
    predicates_list = "\n".join(f"  - `{k}` ({v})" for k, v in ALLOWED_PREDICATES.items())
    actions_list = "\n".join(f"  - `{a}()`" for a in sorted(ALLOWED_ACTIONS))
    enums_list = ", ".join(f"`{k}`" for k in ENUM_CONSTANTS.keys())

    return f"""You are the LOX Policy Synthesizer.
Your goal is to author or revise an autonomous, safety-bounded game policy program written in pure Pythonic Infix AST.
The program is compiled into a high-performance Hierarchical Behavior Tree executed at microsecond CPU speeds.

### Language Syntax & Grammar Rules
1. Define named behaviors using either `def name():` or `macro name:`.
2. Use Python `if`, `elif`, and `else` statements with standard boolean infix logic:
   - Supported operators: `and`, `or`, `not`, `<`, `<=`, `>`, `>=`, `==`, `!=`
3. Action calls are simple function calls, e.g. `descend()`, `step_to_frontier()`.
4. Define the execution priority order with `plan = [macro1, macro2, ...]` or `plan:` followed by macro names.
   The plan executes as a Priority Selector (evaluating from top to bottom on every turn).
5. STRICT SAFETY RULES:
   - NEVER use while loops, for loops, imports, or arbitrary Python expressions.
   - ONLY reference allowed predicates, action primitives, and enum constants listed below.

### Approved Predicates:
{predicates_list}

### Enum Constants:
{enums_list}

### Approved Action Primitives:
{actions_list}

### Output Format:
Wrap your policy code in a single ```python ... ``` block. Include a brief 1-2 sentence rationale before the code.
"""


def build_user_prompt(current_policy: str, trigger_reason: str, autopsy_report: str) -> str:
    return f"""### Synthesis Trigger:
{trigger_reason}

{autopsy_report}

### Current Policy Program:
```python
{current_policy}
```

Please revise the policy program to eliminate this failure mode, prevent stalls, and improve survival and floor progression.
"""
