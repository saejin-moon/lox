# LOX — Autonomous Empirical Policy Synthesis Engine

> **Architectural Motto:** *"Hard Code Nothing."*

**LOX** is an autonomous, high-throughput neuro-symbolic program synthesis engine that discovers, tests, attributes, and refines executable Python control policies and reusable macro libraries for complex, long-horizon environments from first principles—supporting both the **NetHack Learning Environment (NLE)** and the **Crafter / Craftax** survival benchmark.

Unlike traditional Deep Reinforcement Learning (which trains opaque neural weights over billions of frames) or online LLM agents (which query a slow, expensive language model at every environment step), **LOX puts the LLM in the outer scientific loop and compiled Python in the inner execution loop**:
- **Inner Loop (CPU)**: Compiled Python 3.11+ generator policies execute directly against the C/Python environment harness at **50,000–121,000+ steps/sec** across worker cores (`0` LLM tokens per step), featuring automatic C prompt settlement (`flags.xwaitforspace` / `--More--` dismissal) and microsecond byte replay.
- **Outer Loop (Stateless Scientist LLM)**: Between batches, a stateless Author Agent queries a multi-table **DuckDB** analytical warehouse (`runs`, `ticks`, `function_stats`, `expert_runs`, `expert_ticks`) and an offline **NetHack 3.6.6 Wiki / Knowledge Base**, formulates a falsifiable causal hypothesis, tests candidate skills in a fast **5 × 30-episode self-iterative pilot revision loop** with behavioral critics and deterministic prefix-replay shards, and triggers targeted evolutionary rollouts.

---

## Table of Contents

1. [Core Architectural Philosophy ("Hard Code Nothing")](#1-core-architectural-philosophy-hard-code-nothing)
2. [System Architecture & Dataflow](#2-system-architecture--dataflow)
3. [Multi-Benchmark Architecture: NetHack & Craftax](#3-multi-benchmark-architecture-nethack--craftax)
4. [Prefix-Replay Evaluation Shards & Spatial WorldMemory](#4-prefix-replay-evaluation-shards--spatial-worldmemory)
5. [The Four Modification Modes & Library Learning](#5-the-four-modification-modes--library-learning)
6. [Multi-Table DuckDB Schema & Outlier Attribution](#6-multi-table-duckdb-schema--outlier-attribution)
7. [Parameter-Free Sequential Hypothesis Testing](#7-parameter-free-sequential-hypothesis-testing)
8. [Repository Structure](#8-repository-structure)
9. [Quickstart & CLI Usage](#9-quickstart--cli-usage)
10. [References & Citations](#10-references--citations)

---

## 1. Core Architectural Philosophy ("Hard Code Nothing")

Earlier generations of LLM-guided game agents rely heavily on human-engineered wrappers: hardcoded monster threat tiers, pre-written corpse safety tables, hidden auto-explore loops, or fixed finite-state machines. While those crutches produce quick early-game demos, they impose a strict glass ceiling when the agent encounters mechanics the human architect forgot to hardcode.

**LOX enforces a strict separation between the Architect (the harness) and the Author (the policy agent):**

1. **Zero Hardcoded Strategy in the Harness ([`lox/env/`](lox/env/))**:
   - The harness contains **zero** game heuristics, **zero** monster/corpse/item lists, and **zero** hidden macro adapters.
   - It provides a **pure 1-to-1 connection** between policy yields and native environment actions ([`Action`](lox/env/actions.py) / [`CraftaxAction`](lox/env/craftax/actions.py), [`key(ch)`](lox/env/actions.py) raw ASCII bytes, and [`seq(...)`](lox/env/actions.py) multi-keystroke sequences).
   - It exposes the game state lazily via [`GameState`](lox/env/state.py): all 27 `blstats` fields, `BL_MASK_*` status bitmasks, 21×79 or 48×48 screen grids (`glyphs`, `chars`, `colors`, `specials`), inventory items, and pre-indexed C symbol table lookups over `nle.nethack.permonst` and `nle.nethack.objclass`, alongside generic Numba-JIT grid math ([`bfs_distance_grid`](lox/env/state.py), [`astar_next_step`](lox/env/state.py)).
   - Transparently handles NetHack C engine pagination (`flags.xwaitforspace == 1`) by automatically clearing `--More--` pauses while logging every keystroke byte for bit-identical replay.
2. **Near-Full Python 3.11+ Policy Language & Macro Library ([`lox/lang/`](lox/lang/))**:
   - The Author Agent writes full Python 3.11+ in `data/workspace/policy.py`, `data/workspace/macros.py`, and modular skills in `data/workspace/skills/`: classes, stateful generator coroutines (`yield` / `yield from`), `match/case` structural pattern matching, comprehensions, `@dataclass`, closures, and safe standard library modules (`math`, `collections`, `heapq`, `itertools`, `functools`, `re`, `numpy`, `random`, `bisect`, `copy`).
   - Compile-time AST instrumentation ([`PolicyInstrumentTransformer`](lox/lang/instrument.py) & [`ExecutionTracer`](lox/lang/instrument.py)) injects `<1 µs` per-function call counting, per-function turn attribution, zero-yield infinite-loop protection ([`PolicyLoopError`](lox/lang/instrument.py)), and tunable numerical parameter hooks (`self.param(name, default, min_val, max_val)`).

---

## 2. System Architecture & Dataflow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     STATELESS AUTHOR AGENT (Per-Generation)                      │
│  • 100% Static Cached System Prompt (Policy Language Specs + DuckDB Schema)      │
│  • Dynamic User Prompt: Mode Instructions + Active Code + Contrastive Digest     │
│  • Tools: query_db(sql) | query_wiki(q) | query_policy_docs(t) | dry_run_policy  │
│  • Self-Iterative Pilot Revisions: Up to 5 runs x 30 paired episodes (~0.8s)     │
│    with Auto-Pilot Gate & Behavioral Critic (real-time physical stall telemetry)  │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Emits Single-Skill YAML Hypothesis + Python Code
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        HYBRID CADENCE / BANDIT SCHEDULER                         │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ 1. FULL_WRITE  │  │ 2. MACRO_WRITE │  │ 3. FUNC_MOD_ADD  │  │ 4. PARAM_TUNE│  │
│  │    250 eps     │  │  0 eps (skip)  │  │ 150 trig eps / fn│  │  0 LLM tokens│  │
│  │ (Gen 0 init)   │  │ -> macros.py   │  │ (single skill)   │  │  (CRN-SPSA)  │  │
│  └───────┬────────┘  └───────┬────────┘  └────────┬─────────┘  └──────┬───────┘  │
└──────────┼───────────────────┼────────────────────┼───────────────────┼──────────┘
           │                   │ (Instant Next Gen) │                   │
           ▼                   ▼                    ▼                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│             PERSISTENT MULTI-CORE WORKER POOL (NLE & Craftax Adapters)           │
│  • Direct C / Numba Step Execution (50k–121k+ steps/sec across workers)          │
│  • Checkpoint Prefix Shards: 200-step replay for pre-death & floor entry windows  │
│  • Conditional Trigger Filter: discards episodes where target_fn call_count == 0 │
│  • Unreachable Function Circuit Breaker + Sequential Futility Early Stopping     │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Recorded Triggered Episodes + Replay Bytes
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│           MULTI-TABLE DUCKDB WAREHOUSE & OUTLIER SUMMARIZER (lox.duckdb)         │
│  • RLE Phase Timeline Compressor + Critical Inflection & Stall Detector          │
│  • Go-Explore Outlier Good-Run Marker (is_outlier_good = TRUE via Z-score / IQR) │
│  • Spectrum-Based Function Attribution (Ochiai Good-Run & Liblit Failure Scores) │
│  • Consolidated Human Ascensions & Expert Traces (expert_runs & expert_ticks)    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Multi-Benchmark Architecture: NetHack & Craftax

LOX uses a unified [`BenchmarkAdapter`](lox/env/base.py) protocol that decouples environments from policy execution and statistical hypothesis testing:

### 3.1 NetHack Learning Environment (`--benchmark nethack`)
- **Engine**: Direct C++ `nle.nethack.Nethack` engine with `--More--` pause dismissal.
- **State Representation**: 21×79 character grids, 27 status indicators, C permonst/objclass symbols.
- **Progression Metric**: Composite progression index based on dungeon depth, character level, armor class, and gold.

### 3.2 Crafter / Craftax Benchmark (`--benchmark craftax`)
- **Engine**: Pure deterministic Crafter/Craftax survival engine with 17 discrete actions.
- **Facing Direction Mechanics**: `LEFT`, `RIGHT`, `UP`, and `DOWN` orient the player; `Action.DO` and `PLACE_*` act on the targeted coordinate `(y + dy, x + dx)`.
- **Full 22-Achievement Survival Tech Tree**: Wood gathering, crafting table placement, wooden pickaxe crafting, stone mining, furnace placement, stone pickaxe, iron mining, iron smelting, weapon crafting, hostile mob combat, and thirst/hunger maintenance.
- **Progression Metric**: Total survival achievements unlocked plus health and step survival bonuses.

---

## 4. Prefix-Replay Evaluation Shards & Spatial WorldMemory

### 4.1 Checkpoint Prefix Shards ([`lox/db/shards.py`](lox/db/shards.py))
To evaluate late-game survival skills without waiting thousands of exploration steps, LOX automatically extracts **200-step Checkpoint Prefix Shards** from recorded baseline runs:
1. **Pre-Death Windows**: Checkpoint sliced 30 turns prior to a fatal encounter.
2. **Pre-Stall Windows**: Checkpoint sliced 5 turns prior to detected oscillation/stagnation.
3. **Floor Entry Checkpoints**: Arrival checkpoint on new dungeon floors (`depth >= 2`).

In `FUNC_MOD_ADD` and pilot testing, shards are replayed at microsecond speeds up to the cut point, after which the candidate policy takes control. Shards are paired **strictly against baseline shards** on identical prefixes, preventing false regressions against full 25,000-turn runs.

### 4.2 Dynamic Spatial WorldMemory ([`lox/env/state.py`](lox/env/state.py))
Policies have access to `state.mem` (`WorldMemory`), which dynamically adapts to map dimensions ($21 \times 79$ for NetHack, $48 \times 48$ for Craftax):
- `visited`: Boolean grid tracking all observed/explored tiles.
- `visit_count`: Integer matrix tracking exploration density and detecting loops.
- `searched_counts`: Spatial grid incremented when `Action.SEARCH` is taken, aiding secret door discovery.
- `stair_graph`: Bidirectional inter-floor transition graph tracking stair connections.
- `pos_history`: Bounded coordinate history for cycle breaking and frontier navigation.

---

## 5. The Four Modification Modes & Library Learning

| Mode | Episode Budget | Verification Gate | Purpose |
| :--- | :--- | :--- | :--- |
| **`FULL_WRITE`** | `250` episodes | 30-episode live smoke test | Synthesizes `class Policy` from scratch on Generation 0 (no starter policy fed). |
| **`MACRO_WRITE`** | `0` episodes *(instant)* | Real-state replay test across 3 diverse fixtures | Synthesizes verified generator/helper macros into `data/workspace/macros.py` with static contract linting. |
| **`FUNC_MOD_ADD`** | `150` triggered eps / fn | 5 × 30-episode pilot revision loop | Surgically adds or modifies a single targeted method on `Policy` ([`splice_functions_into_policy`](lox/lang/validator.py)). |
| **`PARAM_TUNE`** | `0` LLM tokens (`CRN-SPSA`) | Paired CRN validation batch | Optimizes all `self.param(name, default, min, max)` thresholds via Common Random Numbers SPSA and bakes winning constants into the AST. |

### Bottom-Up Macro Harvesting & Skill Packaging
In addition to top-down LLM generation, LOX implements **Wake-Sleep Library Learning**:
- **Macro Harvester** ([`lox/lang/macros.py`](lox/lang/macros.py)): After successful rollouts, helper methods with high Ochiai Good-Run correlation are automatically extracted into `data/workspace/macros.py`.
- **Hierarchical Skills** ([`data/workspace/skills/`](data/workspace/skills/)): Author Agent can create and import modular sub-generators (e.g. `combat_tactics.py`, `trap_navigation.py`), which are dynamically loaded and scoped into the policy outline.

---

## 6. Multi-Table DuckDB Schema & Outlier Attribution

All experimental telemetry and reference human ascensions live in a consolidated DuckDB warehouse (`data/lox.duckdb`):

1. **`batches`**: Metadata per generation batch (`batch_id`, `generation`, `mode`, `target_functions`, `avg_depth`, `hypothesis_passed`, `wilcoxon_p`, `ci_lower`, `ci_upper`).
2. **`runs`**: Episode outcomes (`run_id`, `seed`, `depth`, `score`, `turns`, `end_reason`, `summary_digest`, `game_log`, `milestones_json`, `is_outlier_good`, `action_bytes`).
3. **`ticks`**: Tail-sampled step traces (`run_id`, `turn`, `hp`, `ac`, `y`, `x`, `action_name`, `active_function`, `message`).
4. **`function_stats`**: Spectrum-based attribution (`episodes_triggered`, `ochiai_good_run_score`, `liblit_failure_score`).
5. **`policies`**: Complete snapshots of promoted and candidate policies and macros.
6. **`experiments`**: Qualitative causal hypotheses, pilot outcomes, and statistical verdicts.
7. **`expert_runs`** & **`expert_ticks`**: Verified human NetHack ascensions (`archive.alt.org` xlogfile + dumplogs) and expert reference trajectories.

---

## 7. Parameter-Free Sequential Hypothesis Testing

The Author Agent never guesses numerical effect sizes. Instead, it emits a structured qualitative causal hypothesis:

```yaml
mode: FUNC_MOD_ADD
target_functions:
  - explore_step
skill_name: "Frontier Exploration with Multi-Tile Cycle Breaking"
precondition: "No adjacent hostile monsters; unvisited walkable tiles exist."
state_persistence: "self.target_tile (held up to 25 steps) and self.pos_history (deque(maxlen=8))"
postcondition: "Descends downstairs '>' or reaches an unvisited room."
observation_from_db: "In stalled runs on Dlvl 1, 85% of episodes oscillated between two corridor tiles."
mechanism: "Commit to target_tile and take random escape steps if (y, x) is in pos_history[:-1]."
```

Candidate policies are evaluated on paired seeds against the active baseline using three objective statistical gates ([`lox/engine/hypothesis.py`](lox/engine/hypothesis.py)):
1. **Sequential Futility Early Stopping**: If after $n \ge 60$ paired triggered seeds the 95% Empirical Bernstein Confidence Sequence upper bound satisfies $U_n \le 0.0$, rollout terminates early to conserve compute.
2. **Positive Mean Progression Delta**: $\bar{d} > 0.0$ and strictly more improved seeds than regressed seeds ($\text{wins} > \text{losses}$).
3. **Statistical Significance**: Either 95% Confidence Sequence lower bound $L_N > 0.0$ or one-sided Paired Wilcoxon Signed-Rank $p < 0.05$.

---

## 8. Repository Structure

```text
lox/
├── env/
│   ├── base.py             # BenchmarkAdapter protocol & registry
│   ├── actions.py          # 1-to-1 Action IntEnum, key(), seq(), DIR_DELTAS
│   ├── state.py            # Lazy zero-copy GameState, WorldMemory, Numba BFS & A*
│   ├── harness.py          # Direct C++ NLE harness & microsecond byte replay
│   ├── nethack/            # NetHackAdapter & knowledge base integration
│   └── craftax/            # CraftaxAdapter, 17 actions, 22-achievement survival ruleset
├── lang/
│   ├── validator.py        # Python 3.11+ AST sandbox validator & macro contract linter
│   ├── instrument.py       # ExecutionTracer, TunableParam, PolicyInstrumentTransformer
│   ├── macros.py           # MacroLibrary manager & MacroHarvester bottom-up library learning
│   └── compiler.py         # Compiles macros.py + policy.py + skills into PolicyRunner instances
├── db/
│   ├── schema.py           # 8-table DuckDB schema initialization & connections
│   ├── shards.py           # PrefixShard extractor (pre-death, pre-stall, floor-entry)
│   ├── summarizer.py       # RLE timeline compressor, stall analyzer, Ochiai/Liblit attribution
│   └── logger.py           # High-throughput DuckDB batch logger with tail-based tick sampling
├── datasets/
│   ├── wiki.py             # Offline NetHack 3.6.6 Wiki SQLite FTS5 BM25 search engine
│   └── download_experts.py # Compiles NAO human ascensions & NLE traces into expert_runs / expert_ticks
├── runner/
│   └── pool.py             # Persistent multi-core worker pool & conditional trigger filter
├── author/
│   ├── prompts.py          # Static cacheable System Prompt & dynamic User Prompt builder
│   ├── tools.py            # AuthorToolRegistry: query_db, query_wiki, query_policy_docs, dry_run_policy
│   └── agent.py            # Stateless ReAct AuthorAgent with compiler-in-the-loop self-repair
├── engine/
│   ├── hypothesis.py       # Confidence Sequences, Wilcoxon Signed-Rank, Futility Early Stopping
│   ├── modes.py            # CRN-SPSA parameter optimizer & AST constant baker
│   ├── scheduler.py        # Hybrid cadence + Extreme-Value UCB mode scheduler
│   └── orchestrator.py     # End-to-end CampaignOrchestrator
└── ui/
    └── console.py          # Compact ANSI console UI with single-line progress & summary tables
scripts/
├── run_lox.py              # Main CLI entrypoint for running synthesis campaigns
├── reset_clean.py          # Resets campaign to Gen 0 while preserving expert corpus & wiki
├── download_experts.py     # Populates expert_runs and expert_ticks in data/lox.duckdb
└── build_wiki_db.py        # Builds data/wiki_index.db from data/nethackwikidata.json
tests/
├── test_overhaul_verification.py       # Full regression test suite for adapters, shards, and logging
├── test_skill_graph_and_adapters.py    # Multi-benchmark adapter and skill compilation tests
├── test_todo_architectural_overhaul.py  # Spatial WorldMemory and prefix shard extraction tests
├── test_harness_speed.py               # 1-to-1 actions, GameState queries, replay bit-identity
├── test_lang_instrument.py             # Python 3.11+ features, tracer, loop guard, splicing
├── test_db_summarizer.py               # RLE summarizer, outlier marker, Ochiai/Liblit, DuckDB logger
├── test_modes_and_eval.py              # Trigger filtering, hypothesis testing, SPSA tuning
└── test_author_and_ui.py               # Author YAML/code parser, macro verification, pilot revisions
```

---

## 9. Quickstart & CLI Usage

### 1. Verify the Test Suite
```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

### 2. Reset Workspace to Clean State (Optional)
To start a brand-new campaign while preserving the offline Wiki and expert ascensions:
```bash
uv run python scripts/reset_clean.py
```

### 3. Ingest Expert Human NetHack Ascensions (Optional)
```bash
uv run python scripts/download_experts.py --max-ascensions 50
```

### 4. Launch a Synthesis Campaign
Ensure a valid `OPENROUTER_API_KEY` is set in `.env` (or exported in your shell):

#### NetHack Learning Environment
```bash
uv run python scripts/run_lox.py \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --benchmark nethack \
  --workers 16 \
  --generations 50 \
  --full-write-episodes 250 \
  --func-mod-episodes 150 \
  --pilot-revisions 5 \
  --pilot-episodes 30
```

#### Crafter / Craftax Survival Benchmark
```bash
uv run python scripts/run_lox.py \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --benchmark craftax \
  --workers 16 \
  --generations 50 \
  --full-write-episodes 100 \
  --func-mod-episodes 80 \
  --pilot-revisions 5 \
  --pilot-episodes 20
```

---

## 10. References & Citations

1. **Küttler, H., et al.** (2020). *The NetHack Learning Environment.* NeurIPS 33.
2. **Hambro, E., et al.** (2022). *Insights from the NeurIPS 2021 NetHack Challenge.* PMLR 176.
3. **Ecoffet, A., et al.** (2021). *First return, then explore (Go-Explore).* Nature 590.
4. **Romera-Paredes, B., et al.** (2024). *Mathematical discoveries from program search with large language models (FunSearch).* Nature 625.
5. **Ellis, K., et al.** (2021). *DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning.* ACM PLDI.
6. **Wang, G., et al.** (2023). *Voyager: An Open-Ended Embodied Agent with Large Language Models.* TMLR.
7. **Böhme, M., et al.** (2017). *Coverage-Based Greybox Fuzzing as Markov Chain (AFLFast).* IEEE TSE / ACM CCS.
8. **Abreu, R., et al.** (2007). *On the Accuracy of Spectrum-based Fault Localization (Ochiai).* IEEE TAICPART.
9. **Liblit, B., et al.** (2005). *Scalable statistical bug isolation.* ACM PLDI.
10. **Spall, J. C.** (1992). *Multivariate stochastic approximation using a simultaneous perturbation gradient approximation (SPSA).* IEEE TAC 37(3).
11. **Waudby-Smith, I., & Ramdas, A.** (2024). *Estimating means of bounded random variables by betting.* JRSS-B 86(1).
12. **Howard, S. R., et al.** (2021). *Time-uniform, nonparametric, nonasymptotic confidence sequences.* Annals of Statistics 49(2).
