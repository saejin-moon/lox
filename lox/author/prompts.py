"""
LOX 2.0 Author Prompts: Ultra-compact, token-frugal prompts for LLM policy synthesis (<450 tokens total).
Maximizes signal-to-token ratio and equips the agent with DuckDB analytical tooling.
"""
from __future__ import annotations

from lox.dsl.schema import ALLOWED_PREDICATES, ALLOWED_ACTIONS, ENUM_CONSTANTS


def build_system_prompt() -> str:
    """Ultra-dense system prompt (~220 tokens) defining syntax, schema, and DuckDB analytical tools."""
    preds = ", ".join(ALLOWED_PREDICATES.keys())
    actions = ", ".join(f"{a}()" for a in sorted(ALLOWED_ACTIONS))
    enums = ", ".join(f"{k}={v}" for k, v in ENUM_CONSTANTS.items())

    return f"""You are the LOX Policy Synthesizer.
Synthesize autonomous NetHack/MiniHack policies in safe Pythonic Infix AST, compiled into microsecond CPU Behavior Trees.

### Syntax Rules
1. Define behaviors: `def name(): if cond: act() else: act()`
2. Infix operators: `and`, `or`, `not`, `<`, `<=`, `>`, `>=`, `==`, `!=`
3. Plan priority: `plan = [fn1, fn2, ...]` (evaluates top-to-bottom as a priority Selector).
4. Safety: No loops, no imports, no arbitrary expressions. Use ONLY approved identifiers.

### Approved Vocabulary
- Predicates: {preds}
- Enums: {enums}
- Actions: {actions}

### Empirical Analysis Tools (DuckDB)
Before writing code, you may call tools to analyze historical gameplay data:
- `query_duckdb(sql)`: Execute read-only SQL on `data/lox.duckdb` (tables: `episodes`, `ticks`, `events`).
- `get_duckdb_schema()`: View database tables and column types.
- `get_death_taxonomy(window)`: Top death causes, frequencies, and avg depth.
- `get_action_distribution(run_id)`: Action frequencies and search vs step ratios.

### Output Requirement
Provide 1 brief rationale sentence, then your full revised code in a single ```python ... ``` block.
"""


def build_user_prompt(current_policy: str, trigger_reason: str, status_report: str) -> str:
    """Compact user prompt (<200 tokens) presenting the incident report and current policy."""
    return f"""### Status Report:
{status_report}

### Current Policy:
```python
{current_policy.strip()}
```

Diagnose the failure mode (query DuckDB if needed) and synthesize the updated policy program to resolve it.
"""
