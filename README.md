# LOX 2.0: Neuro-Symbolic Policy Synthesis Engine

**LOX 2.0** is an autonomous policy synthesis architecture designed from first principles for embodied, long-horizon decision making (MiniHack, NetHack, Craftax).

Instead of an LLM adjusting static scalar parameters while thousands of lines of fragile heuristic code handle navigation, LOX 2.0 enables the LLM to synthesize **Hierarchical Behavior Trees** written in a safe, sandboxed **Pythonic Infix AST DSL** that compiles directly into microsecond CPU execution (<20 µs latency).

---

## 1. System Architecture

```
  OUTER SYNTHESIS LOOP (Asynchronous, Event-Driven)
  ┌────────────────────────────────────────────────────────────────────┐
  │ Dynamic Trigger Engine (Stall >100 turns | Death Clusters | Record)│
  │                              ▼                                     │
  │ LLM Author Agent (Receives failure autopsy & domain manifest)       │
  │                              ▼                                     │
  │ Policy DSL Output (Clean Pythonic Infix AST)                       │
  │                              ▼                                     │
  │ Formal AST & Safety Invariant Validator (Strict NodeVisitor)       │
  │                              ▼                                     │
  │ Tree Compiler -> Executable Behavior Tree Object                   │
  └──────────────────────────────┬─────────────────────────────────────┘
                                 │ Deployed
  INNER EXECUTION LOOP (Pure CPU, <50 µs Latency, 10,000+ SPS)
  ┌──────────────────────────────▼─────────────────────────────────────┐
  │ Environment Adapter (MiniHack / NetHack / Craftax)                 │
  │   - Standardized Step & Observation (Grid, Hero State, Inventory)  │
  │                              ▼                                     │
  │ Standalone Spatial Engine (Numba A*, Frontier BFS, Raycasting)     │
  │                              ▼                                     │
  │ Executable Behavior Tree Engine (Selectors, Sequences, Actions)    │
  │                              ▼                                     │
  │ Action Dispatcher (Atomic Steps, In-Game Commands)                 │
  │                              ▼                                     │
  │ Streaming Parquet Logger -> Post-Run DuckDB Consolidation & Cleanup│
  └────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure

```
lox/
├── core/
│   ├── types.py             # Standardized dataclasses (Observation, HeroState, Item, Action)
│   ├── tree.py              # Pure Behavior Tree Engine (Selector, Sequence, Condition, Action)
│   └── spatial.py           # Numba-accelerated A*, Frontier BFS, Distance Transforms (<150 LOC)
├── dsl/
│   ├── schema.py            # Whitelist of approved predicates, actions, and enum constants
│   ├── parser.py            # AST NodeVisitor sandboxing (rejects loops, imports, unsafe code)
│   └── compiler.py          # AST -> BehaviorTree Compiler
├── envs/
│   ├── base.py              # Abstract EnvironmentAdapter interface
│   ├── minihack.py          # Gymnasium MiniHack wrapper with dynamic action lookup
│   └── nethack.py           # Clean NLE NetHack wrapper (structured inventory, safety interlocks)
├── telemetry/
│   ├── recorder.py          # Circular flight recorder & markdown autopsy generator
│   ├── triggers.py          # Dynamic synthesis triggers (floor pacing stalls, cluster deaths)
│   ├── parquet.py           # Streaming PyArrow Parquet partition logger
│   └── consolidator.py      # Vectorized SQL DuckDB consolidation & auto-cleanup
└── author/
    ├── prompts.py           # System and user prompts for policy authoring
    └── agent.py             # Author agent supporting Gemini, OpenRouter, local vLLM, and Mock
```

---

## 3. Telemetry Pipeline: Streaming Parquet to Consolidated DuckDB

1. **During Run Execution**:
   - Each tick and episode record is streamed to partitioned Parquet files under `data/telemetry/<run_id>/ticks_part_*.parquet` with zero database lock contention.
2. **Post-Run Consolidation**:
   - `consolidate_run(run_id, db_path="data/lox.duckdb", cleanup=True)` executes a vectorized SQL merge using DuckDB's `read_parquet()`.
   - Raw parquet partition directories are automatically deleted upon successful consolidation, leaving a clean, single `data/lox.duckdb` file for SQL queries.

---

## 4. Quickstart & CLI Commands

### Run the full test suite:
```bash
uv run pytest -s
```

### Run MiniHack evaluation:
```bash
uv run python scripts/run_minihack.py --episodes 10
```

### Run an autonomous policy synthesis loop:
```bash
# Offline verification with mock author
uv run python scripts/run_synthesis.py --provider mock

# Live synthesis with Gemini (requires GEMINI_API_KEY in .env)
uv run python scripts/run_synthesis.py --provider gemini --model gemini-2.5-flash
```
