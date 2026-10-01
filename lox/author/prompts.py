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

### Empirical & Knowledge Tools
Before writing code, you may call tools to analyze data or research game mechanics:
- `query_duckdb(sql)`: Execute read-only SQL on `data/lox.duckdb`.
  * Table `episodes`: (run_id, episode_id, depth, score, turns, death_reason, death_category, steps, attacks, descents, eats, prayers, gold)
  * Table `ticks`: (run_id, episode_id, turn, depth, hp, max_hp, hunger, y, x, action, message, reward)
- `get_duckdb_schema()`: View database tables and column types.
- `get_death_taxonomy(window)`: Top death causes, frequencies, and avg depth.
- `get_action_distribution(run_id)`: Action frequencies and search vs step ratios.
- `query_wiki(query)`: Search offline NetHack 3.6.6 encyclopedia (monsters, intrinsics, corpses, rituals).
- `request_macro(macro_name, rationale, proposed_interface, priority)`: Queue an unimplemented macro/primitive for post-run human engineering if you need a capability that does not yet exist.

### Output Requirement
Provide 1 brief rationale sentence, then your full revised code in a single ```python ... ``` block.
"""


def build_user_prompt(current_policy: str, trigger_reason: str, status_report: str) -> str:
    """User prompt presenting the empirical autopsy, ranked mortality causes, and current policy."""
    return f"""### Empirical Incident Report:
{status_report}

### Synthesis Objective:
{trigger_reason}

### Diagnostic Notice:
The telemetry database `data/lox.duckdb` holds complete per-tick flight recordings and death messages for this evaluation batch.
You can call `query_duckdb("SELECT death_reason, depth, turns FROM episodes ORDER BY rowid DESC LIMIT 10")` or `get_death_taxonomy(20)` to inspect exact failure traces before synthesizing revised logic.

### Current Policy:
```python
{current_policy.strip()}
```

Synthesize the revised policy program that directly counters the ranked mortality bottlenecks while maintaining aggressive stair navigation and frontier exploration.
"""
