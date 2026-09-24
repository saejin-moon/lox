# LOX-ψ: LLM-Oriented Synthesis of Symbolic Policies

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Test Suite](https://img.shields.io/badge/tests-578%20passed%20(100%25)-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Execution: Pure CPU](https://img.shields.io/badge/execution-Pure%20CPU%20(0%20GPU)-orange.svg)]()

**LOX-ψ** is an autonomous policy synthesis engine designed to systematically beat expert-engineered systems (e.g., AutoAscend on NetHack 3.6.6 / `NetHackChallenge-v0`) and established transfer baselines on MiniHack and Craftax.

Instead of hand-crafting thousands of heuristic rules, **the human role is deliberately confined to mechanisms**: pathfinding, tactical mechanics, guardrails (interlocks), and certification gates. Everything strategic—priorities, survival thresholds, tactical kiting rules, compositions, and declarative goal handlers—is **authored by an LLM** through a grammar-constrained diff language, verified by machine-checkable gates, and evolved against millions of environment ticks.

---

## 1. System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│ AUTHOR AGENT (Agentic LLM Tool-Calling Loop)                               │
│   tools: query_duckdb(sql) · read_trajectory(ep, turns)                   │
│          read_env_schema() · read_manifest() · read_program_tree()         │
│   input: full run telemetry (7.2M+ ticks, episodes, deaths, stalls)        │
│          + domain-scoped vocabulary manifest & compact summaries           │
│   output: S-EXPRESSION POLICY DIFF (Pythonic Infix AST, depth ≤ 3)         │
└──────────────┬─────────────────────────────────────────────────────────────┘
               ▼ VALIDATOR GATES (Machine-checkable, no human in the loop)
   grammar → schema → bounds → target-conditionality → macro expansion
   → interlock invariants → handler schema → certification → tiered batch
┌────────────────────────────────────────────────────────────────────────────┐
│ POLICY PROGRAM (Per-File Authoring Tree & Compiled Artifact)               │
│   data/program/<domain>/                                                   │
│     params.json            # typed, bounded numeric tunables               │
│     macros/*.sexpr         # named condition sugar                         │
│     rules/*.sexpr          # tactic rules (target-conditional)             │
│     goals/*.sexpr          # strategy_plan entries                         │
│     handlers/*.sexpr       # declarative goal-handler sub-programs         │
│   compiled to: data/compiled/<domain>.json                                 │
└──────────────┬─────────────────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ EVOLUTION & EVALUATION ENGINE (scripts/run_evolution.py)                   │
│   Multi-candidate populations → 30 parallel CPU workers                    │
│   Pareto selection (score, depth, step efficiency) + DuckDB lineage       │
└──────────────┬─────────────────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ SYMBOLIC EXECUTOR CORE (DomainAdapter Boundary)                            │
│   Numba JIT Accelerated A*, BFS, & Line-of-Sight (<15 µs latency)          │
│   Adapters: NetHack 3.6.6 (NLE) · MiniHack · Craftax                       │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Core Innovations & Design Principles

### 2.1 The Mechanism vs. Strategy Boundary
- **Humans own the mechanisms**: A* pathfinding, inventory slots, menu wrappers, and hardcoded safety interlocks (corpse freshness, floating eye melee lockouts, Sokoban push solver).
- **The LLM owns the strategy**: Goals, priorities, tactics, and threshold trade-offs. The LLM expresses *what to value and when*; it can **never** execute raw commands, disable guardrails, or emit arbitrary Python code.

### 2.2 Pure CPU Symbolic Execution & Numba Acceleration
Profiled decision latency is dropping toward **<150 µs** per step without touching a GPU:
- **Grid A\* (`astar.py`), Frontier BFS (`frontier.py`), & Distance Grid (`exploration.py`)**: JIT-compiled with `@njit(fastmath=True, nogil=True)`. Slashes pathfinding latency from ~350 µs to **2–15 µs** (30x speedup).
- **Line-of-Sight Raycasting (`stepping.py`)**: Bresenham raycasting compiled via `@njit` (<0.5 µs).
- **Static Monster Threat Scanning (`threat_scan.py`)**: Precomputed 512-entry NumPy LUT computed at import.
- **Zero CUDA Contexts**: Pure single-threaded NLE environments scale linearly across CPU cores (**19,000+ eps/hr** on 30 workers for MiniHack; **5,000–6,000 eps/hr** for NetHack).

### 2.3 Token-Optimized Agentic Authoring
The LLM author operates via an agentic tool loop over live telemetry:
- **Domain-Scoped Manifests**: Primitives, verbs, and params are queried dynamically from the target domain adapter. MiniHack author sessions consume **~500 chars** instead of NetHack's 6,000-character vocabulary.
- **Concise Markdown Telemetry Summaries**: Raw 14,000-character JSON dumps are replaced with high-signal markdown tables (~500 chars), saving **>75% of Turn 1 tokens**.
- **On-Demand Inspection**: The author issues read-only SQL queries (`(tool query_duckdb "SELECT ...")`) and flight-recorder autopsies (`(tool read_trajectory ...)`) only when it needs causal context.

### 2.4 Pythonic Infix AST for Conditions
Replaces error-prone nested Lisp parenthesis trees with human-readable Pythonic infix syntax:
```lisp
;; Author diff:
(rule add tactic_rules (when (hp_frac <= 0.40 and not has_healing and monster "jackal")) (do retreat))
```
- Fully sandboxed AST parser with strict depth bounds ($\le 3$).
- Zero prompt parenthesis hallucinations, full machine checkability.

---

## 3. Supported Domains

| Domain | Environment | Observation / State | Key Primitives & Goals |
| :--- | :--- | :--- | :--- |
| **NetHack 3.6.6** | `NetHackChallenge-v0` (NLE) | 21×79 grid (glyphs, chars, blstats, inventory) | `descend`, `explore_floor`, `goto_minetown`, `enter_sokoban`, `forge_excalibur`, `castle_wishing` |
| **MiniHack** | `MiniHack-ExploreMaze-Hard-v0` | 21×79 grid (chars, blstats, message) | `explore_floor` (frontier BFS), `reach_stairs` (A* navigation), tactical retreats |
| **Craftax** | Classic & Full (JAX) | 8268-float observation vector + crafting DAG | `gather_wood`, `craft_pickaxe`, `mine_stone`, `survive_night` |

---

## 4. Hardcoded Interlocks (Human-Owned Guardrails)

These guardrails correspond to measured NetHack fatality modes and can **never** be relaxed by an LLM diff:
1. **Corpse Freshness**: Mortal corpses safe $\le 25$ turns from witnessed kill; non-corpse food never rots; lichen never rots. Eating forbidden with adjacent hostiles.
2. **Peaceful & Monster Walkability**: Monster tiles unwalkable for standard pathing (+1000 penalty). No bump attacks unless deliberate combat issued.
3. **Shop Doors (DL $\ge 2$)**: Never kicked; unlock tools prioritized, otherwise routed around.
4. **Emergency Health Triage**: Healing quaff at $\le 55\%$ HP ($60\%$ with hostiles); emergency prayer at $\le 25\%$ when prayer is safe.
5. **Floating Eye Lockout**: Direct melee attacks strictly blocked unless blind or reflection active.
6. **Starvation & Food Security**: Descent never blocked for food; prayer resets nutrition at WEAK+. Food chasing at any radius is mathematically locked out after A/B testing proved it increased faints.
7. **Search Stall Elimination**: No secret door searches on corridor tiles; dead ends capped $\le 6$ searches; perimeter walls capped $\le 5$ searches.

---

## 5. Quickstart

### Prerequisites
- Python 3.12+
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- OS: Linux (Ubuntu 22.04 / Debian 12 recommended)

### Installation
```bash
git clone git@github.com:saejin-moon/lox.git
cd lox
uv sync
```

### Running Tests
The test suite enforces 100% regression-free verification:
```bash
uv run pytest -q
# 578 passed in ~18s
```

---

## 6. Execution & Authoring CLI Commands

### 6.1 Parallel Evaluation Batches (30 Workers)
Run high-throughput parallel evaluation across CPU cores:
```bash
# MiniHack parallel batch (19,000+ eps/hr)
uv run python scripts/run_minihack_batch.py \
    --task MiniHack-ExploreMaze-Easy-Mapped-v0 \
    --episodes 100 \
    --jobs 30

# NetHack parallel batch (100 episodes, 20k step cap)
uv run python scripts/run_parallel_batch.py \
    --role valkyrie \
    --episodes 100 \
    --max-steps 20000 \
    --jobs 30 \
    --output data/batch.json
```

### 6.2 Agentic Authoring Sessions
Launch an agentic LLM author session that gathers DuckDB evidence, proposes diffs, and validates gates:
```bash
# Dry run with deterministic mock author
uv run python scripts/run_author_session.py \
    --provider mock \
    --domain minihack \
    --no-commit

# Live author session via OpenRouter (e.g. GLM, QwQ, Claude, Gemma)
uv run python scripts/run_author_session.py \
    --provider openrouter \
    --model z-ai/glm-5.3-flash \
    --domain minihack
```

### 6.3 Evolutionary Synthesis Campaigns
Evolve candidate policy programs across multiple generations:
```bash
uv run python scripts/run_evolution.py \
    --population 4 \
    --domain minihack \
    --episodes 50 \
    --hours 4
```

### 6.4 Telemetry & Database Management
Consolidate parquet partitions into DuckDB and query telemetry:
```bash
# Consolidate raw parquet logs into data/lox_telemetry.duckdb
uv run python scripts/clean_telemetry.py

# Query DuckDB telemetry
uv run python -c "import duckdb; con = duckdb.connect('data/lox_telemetry.duckdb', read_only=True); print(con.execute('SELECT COUNT(*) FROM episodes').fetchone())"
```

---

## 7. Scientific Rigor & Audit Standard

To maintain publication integrity (NeurIPS / ICML / Nature MI):
- **Clean-Room Guarantee**: Zero lines of AutoAscend code, lookup tables, or handcrafted heuristic values are used.
- **Mandatory 3-Arm Ablation**: Every major claim requires `LLM-Grown Policy > Frozen Initial Policy > Random Mutation Policy`.
- **Complete Lineage Transparency**: Every prompt, token count, cost, rejected diff, and accepted version is recorded in `data/revision_ledger.jsonl`.
- **Pure CPU Reproducibility**: Zero GPU dependencies guarantee exact reproducibility across standard server hardware.

---

## License
MIT License. Copyright (c) 2026 LOX-ψ Contributors.
