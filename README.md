# LOX 2.0: Neuro-Symbolic Policy Synthesis Engine

**LOX 2.0** is an autonomous neuro-symbolic policy synthesis architecture designed from first principles for embodied, long-horizon decision making in complex roguelike environments (**NetHack**, **MiniHack**, **Craftax**).

Instead of relying on an LLM to blindly emit action tokens step-by-step (which is slow, fragile, and financially prohibitive) or tuning arbitrary scalar parameters over brittle heuristic scripts, LOX 2.0 decouples **high-level behavioral reasoning** from **real-time spatial/tactical execution**:
1. **The LLM Author Agent** synthesizes and evolves policies written in a sandboxed, safe **Pythonic Infix AST DSL**.
2. **The Compiler** transforms this AST into an executable **Hierarchical Behavior Tree** (`Selector`, `Sequence`, `Condition`, `Action`).
3. **The Inner Engine** executes compiled Behavior Trees alongside high-performance Numba-accelerated spatial algorithms at microsecond CPU latency (**<20–50 µs/tick**, enabling 10,000+ steps/second).
4. **The Telemetry Pipeline** logs flight data via zero-lock partitioned Parquet streams, automatically consolidated into DuckDB for vector-accelerated SQL analysis, death autopsies, and token usage accounting.

---

## 1. System Architecture

LOX 2.0 operates as an asynchronous, two-tier hierarchical system: an **Event-Driven Outer Synthesis Loop** and a **High-Frequency Inner Execution Loop**.

```
                           OUTER SYNTHESIS LOOP
          (Asynchronous / Trigger-Driven / LLM Multi-Turn ReAct)
  ┌──────────────────────────────────────────────────────────────────────┐
  │                        Dynamic Trigger Engine                        │
  │   - Online: Floor Stagnation (>=80 turns) | Starvation Crisis        │
  │   - Offline: Fatality Clusters (>=3/5 deaths) | Milestone Progress   │
  └──────────────────────────────────┬───────────────────────────────────┘
                                     │ Trigger Alert + Compact Status (<100 tok)
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │                           LLM Author Agent                           │
  │              (Gemini 2.5 Flash / OpenRouter / Local LLM)             │
  │                                                                      │
  │  Interactive Analytical ReAct Tools:                                 │
  │   ├── query_duckdb()              -> SQL queries against flight data │
  │   ├── get_duckdb_schema()         -> Relational table structure      │
  │   ├── get_death_taxonomy()        -> Clustered causes of death       │
  │   ├── get_floor_pacing_stats()    -> Turns/depth, frontier clearance │
  │   ├── get_action_distribution()   -> Frequency of tactical actions   │
  │   ├── query_wiki()                -> Sub-5ms BM25 NetHack 3.6.6 wiki │
  │   └── request_macro()             -> Queue missing primitive to human│
  │                                                                      │
  │  Output: Pure Pythonic Infix AST DSL Program                         │
  └──────────────────────────────────┬───────────────────────────────────┘
                                     │ Validated AST Code
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │                 Formal AST Validator & Tree Compiler                 │
  │   - Safe AST Visitor (Blocks loops, imports, calls, attribute writes)│
  │   - Infix Boolean Parser (and / or / not / comparisons)              │
  │   - Behavioral Tree Compiler (if-elif-else -> Selectors & Sequences) │
  └──────────────────────────────────┬───────────────────────────────────┘
                                     │ Deployed Behavior Tree
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────┐
                          INNER EXECUTION LOOP
                  (Pure CPU / <50 µs Latency / 10,000+ SPS)
  │                                                                      │
  │   Environment Adapter (NetHackAdapter / MiniHackAdapter)             │
  │     - Raw Gym/NLE Step -> Structured Observation Dataclass           │
  │                                  │                                   │
  │   Standalone Spatial Engine (Numba Accelerated)                      │
  │     - 8-Connected Dijkstra / A* Pathfinding                          │
  │     - Frontier BFS Exploration Mask Calculation                      │
  │     - Raycast Line-of-Sight & Tactical Threat Distance Field         │
  │                                  │                                   │
  │   Compiled Hierarchical Behavior Tree (Priority Evaluator)           │
  │     - Blackboard Context Sharing                                     │
  │     - Fallback Selectors & Reactive Sequences                        │
  │                                  │                                   │
  │   Tactical Action Dispatcher                                         │
  │     - Navigation: step_to_frontier, step_to_stairs_down, flee        │
  │     - Combat: melee_attack_adjacent, fire_missile, engrave_elbereth  │
  │     - Survival: eat_floor_corpse, eat_inventory_food, pray           │
  │     - Manipulation: kick_closed_door, open_door, dip_excalibur       │
  │                                  │                                   │
  │   Zero-Lock Streaming Parquet Logger                                 │
  │     - Per-episode partitions (ticks.parquet, events.parquet)         │
  │                                  │                                   │
  │   Vectorized DuckDB Consolidation & Post-Run Cleanup                 │
  │     - Fast SQL ingest -> Cleaned temporary storage                   │
  └──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
lox/
├── core/
│   ├── types.py             # Standardized dataclasses (Observation, HeroState, Item, Action, HungerState)
│   ├── tree.py              # Pure Behavior Tree Engine (Selector, Sequence, Condition, Action, Blackboard)
│   └── spatial.py           # Numba-compiled A*, Dijkstra, Frontier BFS, and Distance Transforms (<150 LOC)
├── dsl/
│   ├── schema.py            # Strict whitelist of sensory predicates, action primitives, and enum constants
│   ├── parser.py            # AST NodeVisitor sandboxing (rejects loops, imports, arbitrary calls, attribute mutation)
│   └── compiler.py          # Python AST -> Behavior Tree Compiler (maps if/else to Selectors and Sequences)
├── envs/
│   ├── base.py              # Abstract EnvironmentAdapter base class
│   ├── minihack.py          # Gymnasium MiniHack adapter with dynamic action lookup and seed management
│   └── nethack.py           # NLE NetHack adapter (structured inventory, keystroke macros, safety interlocks)
├── telemetry/
│   ├── recorder.py          # In-memory circular flight recorder & compact markdown autopsy generator
│   ├── triggers.py          # 2-Tier Dynamic Trigger Engine (online stalls/starvation, offline death clusters)
│   ├── parquet.py           # Streaming PyArrow Parquet partition logger (ticks, episodes, events)
│   ├── consolidator.py      # Vectorized SQL DuckDB consolidation & automatic parquet cleanup
│   └── tokens.py            # Multi-provider LLM token usage and dollar cost tracking in DuckDB
├── author/
│   ├── prompts.py           # Ultra-compact system prompt (<250 tok) & status reporting (<100 tok)
│   ├── tools.py             # DuckDBToolRegistry: query_duckdb, get_death_taxonomy, query_wiki, request_macro
│   ├── wiki.py              # Sub-5ms SQLite FTS5 BM25 search over offline NetHack 3.6.6 knowledge base
│   ├── requests.py          # Macro request queue logging to DuckDB and data/macro_requests.md
│   └── agent.py             # ReAct Author Agent with tool calling (OpenRouter, Gemini, and Mock)
scripts/
├── run_nethack.py           # Production NetHack evaluation runner with telemetry streaming and consolidation
├── run_minihack.py          # Rapid MiniHack sandbox runner and policy debugger
├── run_synthesis.py         # Autonomous policy evolution loop (Triggers -> ReAct Agent -> Compilation -> Exec)
└── collect_telemetry.py     # High-throughput multi-role baseline telemetry gatherer
data/
├── lox.duckdb               # Master analytical DuckDB database (episodes, ticks, events, token_usage, macro_requests)
├── wiki_index.db            # Offline SQLite FTS5 NetHack 3.6.6 knowledge index (182 MB, BM25)
└── macro_requests.md        # Human-readable queue of primitives requested by the LLM
```

---

## 3. Pythonic Infix AST DSL Specification

LOX 2.0 does not use YAML, JSON, or obscure Lisp syntax for policies. Instead, policies are written in **standard, valid Python syntax** using infix boolean operators, standard control flow (`if`, `elif`, `else`), and explicit list composition (`plan = [...]`).

### Invariant & Safety Guarantees
The AST validator ([`lox/dsl/parser.py`](file:///home/bae/lox/lox/dsl/parser.py)) strictly enforces safety invariants:
- **No loops**: `for` and `while` statements are rejected with a compile error.
- **No side-effects**: `import`, `exec`, `eval`, class declarations, and function definitions outside `plan` are forbidden.
- **No arbitrary calls**: Calls may only invoke registered predicates or actions from [`lox/dsl/schema.py`](file:///home/bae/lox/lox/dsl/schema.py).
- **No mutations**: Attribute or variable assignment is disallowed.

### Compilation Semantics
The compiler ([`lox/dsl/compiler.py`](file:///home/bae/lox/lox/dsl/compiler.py)) converts Python AST structures into Behavior Tree primitives:
- `plan = [func1, func2]` becomes the **Root Priority Selector**.
- Independent statements in a function become a **Sequence**.
- `if condition: action_a else: action_b` becomes a **Priority Selector** fallback.
- Boolean expressions (`and`, `or`, `not`) compile into **Composite Condition Nodes**.

### Example Valid Policy
```python
def emergency():
    if hp < 6 and has_elbereth_engraved is False:
        engrave_elbereth()
    elif hp < 8 and has_adjacent_monster:
        flee()

def sustenance():
    if is_starving and has_edible_corpse:
        eat_floor_corpse()
    elif is_hungry and has_food:
        eat_inventory_food()

def combat():
    if has_adjacent_monster:
        melee_attack_adjacent()

def explore():
    if has_frontier:
        step_to_frontier()
    else:
        search()

plan = [
    emergency,
    sustenance,
    combat,
    explore,
]
```

### Whitelisted DSL Primitives

#### Sensory Predicates
- **Vitals & Status**: `hp`, `max_hp`, `depth`, `turn`, `is_hungry`, `is_starving`, `is_blind`, `is_confused`, `is_stunned`, `is_hallucinating`, `can_pray`
- **Spatial & Navigation**: `has_frontier`, `has_adjacent_monster`, `has_stairs_down`, `has_stairs_up`, `has_adjacent_door`, `has_adjacent_chest`
- **Tactical & Items**: `has_food`, `has_edible_corpse`, `has_ranged_weapon`, `has_elbereth_engraved`, `monster_distance`

#### Action Primitives
- **Movement & Spatial**: `step_to_frontier()`, `step_to_stairs_down()`, `step_to_stairs_up()`, `flee()`, `search()`, `rest()`
- **Combat & Tactics**: `melee_attack_adjacent()`, `fire_missile()`, `engrave_elbereth()`, `pray()`
- **Item & World Interaction**: `eat_floor_corpse()`, `eat_inventory_food()`, `open_door()`, `kick_closed_door()`, `dip_excalibur()`, `pickup()`
- **Equipment Management**: `wield_weapon()`, `wear_armor()`

---

## 4. Analytical Tooling & Offline Knowledge Engine

When the Dynamic Trigger fires (e.g. stagnation on a floor or a cluster of deaths), the Author Agent enters a **ReAct reasoning loop**, wielding specialized diagnostic tools:

| Tool | Purpose | Output Format |
| :--- | :--- | :--- |
| `query_duckdb(sql)` | Execute arbitrary read-only SQL queries against flight telemetry | Clean Markdown table |
| `get_duckdb_schema()` | Inspect table schemas (`episodes`, `ticks`, `events`, `token_usage`, `macro_requests`) | DDL column listing |
| `get_death_taxonomy(window)` | Retrieve clustered fatality reasons, hero levels, and depths over recent runs | Categorized frequency table |
| `get_floor_pacing_stats(depth)` | Measure average turns spent, frontier clearance rate, and survival rate per floor | Aggregated statistical summary |
| `get_action_distribution(run_id)`| Breakdown of action frequencies to detect loops or underutilized primitives | Action count & percentage table |
| `query_wiki(query, top_k)` | Sub-5ms BM25 search over NetHack 3.6.6 offline wiki (items, monsters, mechanics) | Clean text excerpt with titles |
| `request_macro(macro_name, rationale, proposed_interface)` | Queue an unimplemented primitive to DuckDB and `data/macro_requests.md` | Confirmation string |

---

## 5. Telemetry & DuckDB Storage Pipeline

Flight telemetry is recorded without slowing down the microsecond inner execution loop:
1. **Streaming PyArrow Logging**: Each tick, episode, and event writes to local partitioned Parquet files (`data/telemetry/<run_id>/`).
2. **Vectorized Consolidation**: At run completion, `consolidate_run()` executes a vectorized SQL load into `data/lox.duckdb` using DuckDB's native Parquet reader.
3. **Automatic Cleanup**: Parquet chunk files are safely deleted upon verification.
4. **Token Auditing**: Every LLM call records prompt/completion tokens and estimated cost in USD into the `token_usage` table.

### Primary DuckDB Tables

```sql
-- High-level episode outcomes
episodes(
    run_id VARCHAR,
    episode_id VARCHAR,
    role VARCHAR,
    turns BIGINT,
    depth BIGINT,
    score BIGINT,
    hp BIGINT,
    max_hp BIGINT,
    death_reason VARCHAR,
    wall_time_sec DOUBLE
);

-- Turn-by-turn trajectory metrics
ticks(
    run_id VARCHAR,
    episode_id VARCHAR,
    turn BIGINT,
    depth BIGINT,
    hp BIGINT,
    max_hp BIGINT,
    hunger VARCHAR,
    y BIGINT,
    x BIGINT,
    action VARCHAR
);

-- Discrete game events (milestones, engravings, starvation)
events(
    run_id VARCHAR,
    episode_id VARCHAR,
    turn BIGINT,
    event_type VARCHAR,
    data VARCHAR
);

-- Financial & token telemetry
token_usage(
    run_id VARCHAR,
    session_id VARCHAR,
    timestamp TIMESTAMP,
    provider VARCHAR,
    model VARCHAR,
    prompt_tokens BIGINT,
    completion_tokens BIGINT,
    total_tokens BIGINT,
    cost_usd DOUBLE,
    trigger_reason VARCHAR,
    tools_called VARCHAR
);

-- Human-in-the-loop feature request queue
macro_requests(
    run_id VARCHAR,
    timestamp TIMESTAMP,
    macro_name VARCHAR,
    rationale VARCHAR,
    proposed_interface VARCHAR,
    status VARCHAR
);
```

---

## 6. Getting Started & Command Reference

### Installation & Environment Setup

LOX 2.0 requires Python 3.10+ and uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic virtual environment management:

```bash
# Clone the repository
git clone https://github.com/your-username/lox.git
cd lox

# Install all dependencies with uv
uv sync
```

### Running the Test Suite

```bash
uv run pytest -s
```

### Running Autonomous Policy Synthesis

Run the dynamic synthesis loop where the Author Agent monitors live episodes, triggers autopsies on stalls or fatalities, queries DuckDB and the Wiki, and compiles evolved behavior trees:

```bash
# Offline run with Mock LLM
uv run python scripts/run_synthesis.py --provider mock --generations 5

# Live synthesis with Gemini (requires GEMINI_API_KEY)
uv run python scripts/run_synthesis.py --provider gemini --model gemini-2.5-flash --generations 10

# Live synthesis with OpenRouter (requires OPENROUTER_API_KEY)
uv run python scripts/run_synthesis.py --provider openrouter --model google/gemma-4-31b-it --generations 10
```

### NetHack Evaluation & Benchmarks

Run evaluated episodes with compiled Behavior Trees, streaming Parquet logging, and automatic DuckDB consolidation:

```bash
uv run python scripts/run_nethack.py --episodes 20 --max-turns 500 --role valkyrie
```

### High-Volume Baseline Telemetry Gathering

Populate DuckDB across multiple roles for author agent analysis:

```bash
uv run python scripts/collect_telemetry.py --roles valkyrie,barbarian,monk --episodes 15 --max-turns 400
```

### Long-Running Background Execution (nohup)

To execute a 500-generation synthesis campaign followed by an evaluation benchmark in the background:

```bash
nohup bash -c 'uv run python -u scripts/run_synthesis.py --provider openrouter --model google/gemma-4-31b-it --generations 500 && uv run python -u scripts/run_nethack.py --episodes 500' > synthesis_benchmark.log 2>&1 &
```

### Querying DuckDB Telemetry Directly

```bash
# Check episode statistics
uv run duckdb data/lox.duckdb "SELECT role, COUNT(*) as games, AVG(depth) as avg_depth, MAX(depth) as max_depth FROM episodes GROUP BY role;"

# Inspect causes of death
uv run duckdb data/lox.duckdb "SELECT death_reason, COUNT(*) as count FROM episodes GROUP BY death_reason ORDER BY count DESC LIMIT 10;"

# Audit token usage and LLM spending
uv run duckdb data/lox.duckdb "SELECT model, SUM(prompt_tokens) as prompt_tok, SUM(completion_tokens) as comp_tok, SUM(cost_usd) as total_usd FROM token_usage GROUP BY model;"

# Inspect primitives requested by the LLM
uv run duckdb data/lox.duckdb "SELECT macro_name, rationale FROM macro_requests;"
```
