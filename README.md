# LOX: Autonomous Empirical Policy Synthesis Engine

> **Architectural Motto:** *"Hard Code Nothing."*

`LOX` is an autonomous neuro-symbolic program synthesis engine that designs, verifies, attributes, and evolves executable Python control policies and modular skill packages from first principles. It is engineered for long-horizon, high-dimensional, stochastic environments with permadeath—primarily the **NetHack Learning Environment (NLE)**, with native support for the **Crafter / Craftax** survival benchmark.

Standard reinforcement learning requires billions of environment frames to train opaque neural weights, frequently stalling on NetHack's combinatorial item interactions, multi-step tactical puzzles, and sparse delayed rewards. Turn-by-turn LLM game agents pay prohibitive latency (1–3 seconds per action) and API costs ($50–$200 per game) that make statistical evaluation impossible.

`LOX` decouples reasoning speed from execution speed through a **dual-loop architecture**:

- **Inner Execution Loop (CPU / Numba)**: Compiled Python 3.11+ generator policies execute directly against native C environment harnesses at **120,000 to 285,000+ steps/second** across multi-core worker pools (`0` LLM tokens per action). The harness utilizes lean observation channels (eliminating redundant C-level string descriptions), handles pagination (`--More--`) transparently, passes interactive decision prompts (`[yn]`, `[ynq]`, getlin) directly to policies with watchdog protection, and exposes zero-turn action stall telemetry (`state.turn_advanced`, `state.consecutive_stalls`).
- **Outer Scientific Loop (Stateless LLM Author Agent)**: Between evaluation batches, a stateless Author Agent queries an 8-table **DuckDB** analytical warehouse (`runs`, `ticks`, `function_stats`, `discoveries`, `expert_runs`, `expert_ticks`) and an offline **NetHack 3.6.6 Wiki / Strategy Knowledge Base** (SQLite FTS5 BM25). It diagnoses empirical bottlenecks, checks game mechanics against C symbol tables (`permonst`, `objclass`), tests candidate revisions in a **4-attempt pilot loop** (`dry_run_policy`) against 30 paired baseline episodes with prefix-replay shards, and promotes candidates to full rollouts governed by parameter-free sequential hypothesis testing.

---

## Table of Contents

1. [Core Architectural Philosophy ("Hard Code Nothing")](#1-core-architectural-philosophy-hard-code-nothing)
2. [System Architecture & Lifecycle](#2-system-architecture--lifecycle)
3. [The Canonical Modification Modes](#3-the-canonical-modification-modes)
4. [Pilot Revision Gate & Negative-Delta Abort](#4-pilot-revision-gate--negative-delta-abort)
5. [Harness Architecture & High-Throughput Telemetry](#5-harness-architecture--high-throughput-telemetry)
6. [Prefix-Replay Checkpoint Shards & Spatial WorldMemory](#6-prefix-replay-checkpoint-shards--spatial-worldmemory)
7. [Parameter-Free Sequential Hypothesis Testing](#7-parameter-free-sequential-hypothesis-testing)
8. [DuckDB Analytical Schema & Fault Attribution](#8-duckdb-analytical-schema--fault-attribution)
9. [Long-Horizon Scaling & Unattended Operations](#9-long-horizon-scaling--unattended-operations)
10. [Repository Structure](#10-repository-structure)
11. [Quickstart & CLI Usage](#11-quickstart--cli-usage)
12. [References & Citations](#12-references--citations)

---

## 1. Core Architectural Philosophy ("Hard Code Nothing")

Most game-playing agents depend on human-engineered domain scaffolding: hardcoded monster threat tiers, pre-scripted nutrition tables, built-in pathfinding macros, or hand-tuned phase guides. While these heuristics ease progress through early levels, they create an impenetrable barrier once the agent encounters un-scripted mechanics.

`LOX` strictly enforces a separation between the **Architect** (the domain-agnostic execution and evaluation engine) and the **Author** (the self-synthesizing policy and modular skill codebase):

1. **Zero Domain Heuristics in the Harness ([`lox/env/`](lox/env/))**:
   - The harness contains **zero** monster, corpse, or item lists, and **zero** hidden macro adapters.
   - It provides a **1-to-1 connection** between policy yields and native environment actions ([`Action`](lox/env/actions.py) / [`CraftaxAction`](lox/env/craftax/actions.py), [`key(ch)`](lox/env/actions.py) raw ASCII characters, and [`seq(...)`](lox/env/actions.py) multi-keystroke action sequences).
   - Game state is exposed lazily via [`GameState`](lox/env/state.py): 27 `blstats` fields, `BL_MASK_*` status bitmasks, 21×79 screen grids (`glyphs`, `chars`, `colors`, `specials`), inventory items, and pre-indexed C symbol table queries over `nle.nethack.permonst` (`state.mon_info(y, x)`) and `nle.nethack.objclass` (`state.obj_info(glyph)`), paired with Numba-JIT spatial topology primitives in [`lox/env/grid_utils.py`](lox/env/grid_utils.py).
   - **Lean Observation Channels**: Bypasses redundant C-level rasterization buffers (`screen_descriptions`, `misc`) by default to reach **285,000+ steps/second** while retaining complete `tty_*` terminal line access.
   - **Interactive Prompt Passthrough**: Decouples pagination (`--More--`, auto-cleared with `SPACE`) from true decision prompts (`in_yn_function`, `in_getlin`), allowing policies to respond interactively across steps with a 3-step watchdog.
   - **Zero-Turn Action Stall Guard**: Automatically tracks `state.turn_advanced` and `state.consecutive_stalls`. Runaway action loops (e.g. repeatedly bumping into peaceful monsters or overweight containers) are truncated after 100 consecutive zero-turn steps with reason `stalled_zero_turn`.

2. **Near-Full Python 3.11+ Policy Language & Modular Skills ([`lox/lang/`](lox/lang/))**:
   - The Author Agent writes standard Python 3.11+ in `data/workspace/policy.py` and modular skill packages in `data/workspace/skills/`: classes, stateful generator coroutines (`yield` / `yield from`), `match/case` pattern matching, comprehensions, `@dataclass`, closures, and safe standard library modules (`math`, `collections`, `heapq`, `itertools`, `functools`, `re`, `numpy`, `random`, `bisect`, `copy`).
   - Compile-time AST instrumentation ([`PolicyInstrumentTransformer`](lox/lang/instrument.py) and [`ExecutionTracer`](lox/lang/instrument.py)) injects `<1 µs` per-function call tracking, per-function turn attribution, zero-yield loop protection ([`PolicyLoopError`](lox/lang/instrument.py)), and tunable parameter hooks (`self.param(name, default, min_val, max_val)`).
   - Automatic naming prompt fallback: exposes `state.in_text_prompt` / `state.in_naming_prompt` directly to policy state and features a safety fallback auto-`ESC` in the execution runner to prevent unhandled text prompts (e.g. `Call a scroll: `) from stalling policies.

---

## 2. System Architecture & Lifecycle

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     STATELESS AUTHOR AGENT (Per-Generation)                      │
│  • 100% Static Cached System Prompt (Epistemic Discovery Loop + DuckDB Schema)   │
│  • Dynamic User Prompt: Mode Instructions + Active Code + Contrastive Digest     │
│  • Epistemic Memory: Verified Breakthroughs (discoveries) & Rejected Hypotheses  │
│  • Tools: query_db(sql) | query_wiki(q) | query_policy_docs(t) | dry_run_policy  │
│  • 4-Revision Pilot Loop: Up to 4 revisions × 30 paired episodes (~0.8s)         │
│    with Negative-Delta Abort (never rolls out a known regression)                │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Emits Causal YAML Hypothesis + Python Code
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│             HYBRID MODE SCHEDULER & QUALITY-DIVERSITY BEHAVIOR ARCHIVE           │
│  ┌────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌─────────────┐ │
│  │   CREATE   │ │    WRITE     │ │   REFACTOR   │ │ PARAM_TUNE │ │ MACRO_WRITE │ │
│  │  250 eps   │ │150 trig eps  │ │   200 eps    │ │ (CRN-SPSA) │ │  (Optional  │ │
│  │  (Gen 0)   │ │  (Max 2 fns) │ │ (>=450 lines)│ │(every 100g)│ │  Library)   │ │
│  └─────┬──────┘ └──────┬───────┘ └──────┬───────┘ └─────┬──────┘ └──────┬──────┘ │
└────────┼───────────────┼────────────────┼───────────────┼───────────────┼────────┘
         │               │ (Instant Gen) │               │               │
         ▼               ▼               ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│             PERSISTENT MULTI-CORE WORKER POOL (NLE & Craftax Adapters)           │
│  • Direct C / Numba Step Execution (120k–285k+ steps/sec across worker cores)    │
│  • Adaptive Horizon: Dynamically scales 5,000 -> 50,000 steps matching depth     │
│  • 3-Tier Stratified Shards: Frontier (>=P90), Mid-Band, Foundation (up to 400s) │
│  • Conditional Trigger Filter: discards episodes where target_fn call_count == 0 │
│  • Preemptive Generator Interrupt: aborts macro on HP drop, hostiles, or prompts │
│  • Unreachable Function Circuit Breaker + Sequential Futility Early Stopping     │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Recorded Triggered Episodes + Replay Bytes
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│           MULTI-TABLE DUCKDB WAREHOUSE & OUTLIER SUMMARIZER (lox.duckdb)         │
│  • RLE Phase Timeline Compressor + Critical Inflection & Stall Detector          │
│  • Go-Explore Outlier Good-Run Marker (is_outlier_good = TRUE via Z-score / IQR) │
│  • Spectrum-Based Function Attribution (Ochiai Good-Run & Liblit Failure Scores) │
│  • Epistemic Breakthroughs (discoveries) & Milestone Checkpoint Archiver         │
│  • Automated Telemetry Pruning (keeps DB nimble during 5,000+ generation runs)   │
│  • Consolidated Human Ascensions & Expert Traces (expert_runs & expert_ticks)    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Canonical Modification Modes

Mode scheduling is governed by [`ModeScheduler`](lox/engine/scheduler.py):

| Canonical Mode | Episode Budget | Verification Gate | Purpose & Constraints |
|:---|:---|:---|:---|
| **`CREATE`** | `250` episodes | 30-episode live smoke test | Synthesizes `class Policy` from scratch on Generation 0 (or when starting fresh). Writes full class definition with core survival/navigation loops. |
| **`WRITE`** | `150` triggered episodes + `40` shards | 4 × 30-episode pilot revision loop | The iterative workhorse mode. Surgically adds or modifies targeted methods on `Policy`. **Mutation Scope Guard:** Strictly limited to at most 2 functions (1 primary method + 1 caller/helper) to isolate causality and prevent cross-subsystem regressions. Untriggered episodes (`triggered == False`) are discarded. |
| **`REFACTOR`** | `200` episodes | Behavioral preservation gate ($\bar{d} \ge -0.015$, $\text{losses} \le \text{wins} + 5$) | Triggered automatically when `policy.py` reaches $\ge 450$ lines (every $\ge 4$ generations). Modularizes monolithic methods into external skill packages under `data/workspace/skills/`. Multi-function restructuring is permitted. |
| **`PARAM_TUNE`** | `100` paired episodes (`0` LLM tokens) | Paired CRN validation batch | Zero-LLM parameter optimizer using Common Random Numbers Simultaneous Perturbation Stochastic Approximation (CRN-SPSA). Optimizes numeric arguments declared via `self.param(name, default, min, max)` and bakes winning constants into the AST. Runs every 100 generations. |
| **`MACRO_WRITE`** | `0` episodes (instant advance) | Static syntax & AST validation | Optional mode for synthesizing domain-agnostic macros in `macros.py` (disabled by default, `--macro-every 0`). |

---

## 4. Pilot Revision Gate & Negative-Delta Abort

Before any candidate policy is allowed to run a full 150-seed rollout, it must pass through the **Execution-Guided Pilot Gate** inside `dry_run_policy`:

1. **Paired Seed Pilot Evaluation**:
   - Evaluates the candidate against the active champion across 30 paired baseline episodes in ~0.8 seconds.
   - Measures exact paired wins, losses, ties, progression delta ($\Delta$), and behavioral stall telemetry (`avg_unique_tiles`, `pct_wait`, `pct_search`).
2. **Empirical Revision Budget (`--pilot-revisions 4`)**:
   - If a pilot revision fails, the author receives concrete paired regression traces (`Candidate < Baseline`) and failure diagnostics.
   - The revision budget is capped at 4. Empirical calibration shows that attempts $\ge 5$ have a 0.0% promotion transfer rate; capping at 4 eliminates wasted LLM calls.
3. **Negative-Delta Rollout Abort**:
   - When `--pilot-fallback best` is enabled, candidate promotion to full rollout is permitted **if and only if its pilot delta is non-negative ($\Delta \ge 0.0$)**.
   - If all pilot revisions show net negative progression ($\Delta < 0.0$), the generation is **immediately aborted** (`PILOT_GATE_ABORT`), saving 50–150 rollout episodes from executing a known regression.
4. **Mutation Scope Intercept**:
   - In `WRITE` mode, modifying $>2$ functions returns `MUTATION_SCOPE_ERROR` in 0ms, instructing the agent to narrow its patch before burning pilot runs.

---

## 5. Harness Architecture & High-Throughput Telemetry

LOX achieves **285,000+ steps/second** on multi-core CPU instances through clean architectural separation:

- **Lean Observations**: By default, `LEAN_OBSERVATION_KEYS` requests only the 15 necessary observation buffers from NLE (omitting `screen_descriptions` and `misc`). This eliminates expensive string formatting in C while preserving full terminal grids (`tty_chars`, `tty_colors`, `tty_cursor`).
- **Zero-Turn Stall Detection**: In NetHack, invalid actions (e.g., walking into a solid wall, trying to eat an un-edible item) return a valid observation but consume 0 turns (`blstats[21]` does not advance). LOX tracks `state.turn_advanced`, `state.consecutive_stalls`, and `state.time_delta`, terminating runaway loops at 100 consecutive zero-turn steps with reason `stalled_zero_turn`.
- **Interactive Prompt Passthrough**: Pure pagination prompts (`--More--`) are dismissed instantly via `SPACE`. Real decision prompts (`in_yn_function`, `in_getlin`) are passed through to the policy along with the prompt query string (`state.yn_prompt_query`), protected by a 3-step safety watchdog.

---

## 6. Prefix-Replay Checkpoint Shards & Spatial WorldMemory

### 6.1 3-Tier Stratified Checkpoint Prefix Shards ([`lox/db/shards.py`](lox/db/shards.py))
Evaluating mid-game combat from turn 0 wastes thousands of steps on early floor exploration. LOX extracts up to 40 stratified **Checkpoint Prefix Shards** directly from baseline action bytes:
1. **Pre-Death Windows**: Sliced 30 turns before a fatal encounter (`end_reason == 'died'`).
2. **Pre-Stall Windows**: Sliced 5 turns before detected oscillation or timeout stagnation.
3. **Floor-Entry Checkpoints**: Sliced at the exact turn the hero arrives on a deeper dungeon floor (`depth >= 2`).

Shards are dynamically stratified across three tiers:
- **Frontier Tier (≥ P90 Depth, up to 55%)**: Deepest floor entries and critical encounters. Deep floor entries (`depth >= 5`) receive an extended **400-step evaluation horizon**.
- **Mid-Band Tier (Mean Depth to P90 Depth, up to 30%)**: Intermediate progression checkpoints.
- **Foundation Tier (< Mean Depth, remaining 15%)**: Shallow robustness checkpoints.

Workers replay prefix action bytes in microseconds (`harness.replay_bytes`), reset execution counters, and hand control to the candidate policy. Surviving a fatal pre-death shard earns a direct survival bonus in the composite progression metric.

### 6.2 Dynamic Spatial WorldMemory & Numba Topology ([`lox/env/state.py`](lox/env/state.py), [`lox/env/grid_utils.py`](lox/env/grid_utils.py))
Every [`GameState`](lox/env/state.py) maintains a persistent per-episode [`WorldMemory`](lox/env/state.py) (`state.mem`) that tracks spatial features across levels even when occluded:
- `visited` & `visit_count`: 2D grids tracking explored tiles and per-tile visit frequency.
- `passable_mask` & `door_mask`: Dynamically updated walkability and closed-door masks that remember walkable floor tiles when monsters step onto them and track locked doors.
- `remembered_chars`: Preserves underlying terrain (`'>'`, `'<'`, `'_'`, `'{'`, `'+'`, `'#'`, `'.'`) when occluded by the hero `@` or monsters, preventing stair/altar occlusion loops.
- `kv`: Persistent dictionary on `state.mem.kv` (and `state.kv`) synchronized across floor transitions and replay shards for tracking items, altars, and shop locations.
- `recent_messages`: Ring buffer of recent non-empty message tuples `(turn, dlvl, message)` exposed via `state.recent_messages`.
- `searched_counts`: 2D counter incremented around `(y, x)` on `Action.SEARCH` to guide secret door searches.
- `stair_graph`: Bidirectional inter-floor dictionary mapping `(dlvl, y, x)` stair coordinates across levels.
- `pos_history` & `is_oscillating`: Bounded coordinate history (`deque(maxlen=32)`) and automatic cycle detector.
- **Numba-JIT Spatial Helpers** ([`lox/env/grid_utils.py`](lox/env/grid_utils.py)):
  - `state.bfs_distance_map(sources, passable)` -> 2D int32 distance grid in `<50 µs`.
  - `state.pathfind_step_to(target_y, target_x, passable, cost_grid)` -> 8-way A* first-step `(dy, dx)` with cardinal preference at doorways.
  - `flood_fill_rooms(passable_mask)` -> Connected room component labeling.
  - `find_chokepoints(passable_mask)` -> Boolean mask of 1-tile-wide corridor and doorway bottlenecks for tactical positioning.
  - `line_of_sight_mask(y, x, opaque_mask, max_range)` -> 8-direction raycast visibility mask for ranged wand/missile alignment.

---

## 7. Parameter-Free Sequential Hypothesis Testing

The Author Agent never guesses numerical effect sizes. It emits a structured qualitative YAML hypothesis before its Python code:

```yaml
mode: WRITE
target_functions:
  - descend_or_explore
binding_constraint: "68% of Dlvl 1 timeouts occur when the hero steps onto '>' (stairs down) and immediately steps off because state.chars[y, x] is '@' rather than '>'."
verified_c_rule: "Descending stairs requires yielding Action.DOWN ('>') while standing on the staircase down tile (state.is_on_stairs_down or state.stairs_down_pos == state.pos)."
skill_name: "Immediate Stair Descent & Frontier Navigation"
precondition: "No adjacent high-threat monsters; stairs down '>' known or visible."
state_persistence: "state.mem.remembered_chars and self.explore_target"
postcondition: "Increments state.depth by descending '>' upon reaching stairs_down_pos."
observation_from_db: "Outlier runs reaching Dlvl 4+ execute Action.DOWN within 1 turn of reaching stairs_down_pos."
mechanism: "Check state.is_on_stairs_down before pathfinding so the policy yields Action.DOWN immediately when standing on '>'."
```

Candidate policies are evaluated on paired seeds against the active baseline using objective statistical criteria in [`lox/engine/hypothesis.py`](lox/engine/hypothesis.py):
1. **Sequential Futility Early Stopping**: If after $n \ge 60$ paired triggered seeds the 95% Empirical Bernstein Confidence Sequence upper bound satisfies $U_n \le 0.0$, rollout halts immediately to preserve compute.
2. **Positive Mean Progression Delta**: $\bar{d} \ge -0.005$ and strictly more improved seeds than regressed seeds ($\text{wins} > \text{losses}$).
3. **Decisive-Pair Statistical Significance**: Either the 95% Confidence Sequence lower bound $L_N > 0.0$, the one-sided Paired Wilcoxon Signed-Rank test satisfies $p < 0.05$, or the decisive subset ($|\Delta| > 10^{-6}$) achieves positive significance ($L_{dec} > 0.0$ or $\text{wins} \ge 2 \times \text{losses}$ with $\bar{d}_{dec} \ge 0.15$). This eliminates the *tie-dilution trap* where 60–80% trajectory ties on mature baselines crush statistical power for modular patches.
4. **Tie-Aware Directional Pilot Filtering**: Pilot validation verifies directional signal ($\text{wins} > \text{losses}$, $\bar{d} \ge 0$, and $p_{sign} \le 0.20$ or $\Delta_{depth} \ge 0.04$ or $\text{wins} \ge \text{losses} + 2$).
5. **Decoupled Refactor Gate**: Evaluates `REFACTOR` on architectural modularization and behavioral preservation ($\bar{d} \ge -0.015$, $\text{losses} \le \text{wins} + 5$) rather than demanding strict statistical superiority ($p < 0.05$), enabling modular skill extraction into `skills/` without rejection.

---

## 8. DuckDB Analytical Schema & Fault Attribution

All experimental telemetry and reference human ascensions are stored in a single DuckDB database (`data/lox.duckdb`, defined in [`lox/db/schema.py`](lox/db/schema.py)):

1. **`batches`**: Per-generation summary (`batch_id`, `generation`, `mode`, `target_functions`, `avg_depth`, `p90_depth`, `max_depth`, `hypothesis_passed`, `wilcoxon_p`, `ci_lower`, `ci_upper`, `seeds_evaluated`, `is_early_stopped`, `pilot_best_delta`).
2. **`runs`**: Episode telemetry (`run_id`, `seed`, `max_depth`, `final_hp`, `final_ac`, `xp_level`, `score`, `turns`, `end_reason`, `death_message`, `summary_digest`, `game_log`, `milestones_json`, `is_outlier_good`, `outlier_z_score`, `auto_esc_count`, `subgen_preemptions`, `policy_exceptions`, `unique_tiles`, `pct_wait`, `pct_search`, `starved_with_food`, `died_unarmored`, `action_bytes`).
3. **`ticks`**: Tail-sampled turn traces (`run_id`, `turn`, `dlvl`, `hp`, `max_hp`, `ac`, `xp_level`, `hunger_state`, `y`, `x`, `action_name`, `active_function`, `message`, `nearby_monsters`).
4. **`function_stats`**: Spectrum-based fault attribution per method (`episodes_triggered`, `total_calls`, `total_turns_active`, `ochiai_good_run_score`, `liblit_failure_score`, `stagnation_turns`, `stagnation_share`, `livelock_score`).
5. **`policies`**: Versioned snapshots of every candidate and promoted policy and skill file.
6. **`experiments`**: Causal hypotheses, binding constraints, verified C rules, pilot diagnostics, and statistical verdicts.
7. **`discoveries`**: Validated campaign breakthroughs (`discovery_id`, `generation`, `batch_id`, `mode`, `target_functions`, `binding_constraint`, `verified_c_rule`, `mechanism`, `mean_delta`, `avg_depth`, `p90_depth`, `max_depth`, `created_at`).
8. **`expert_runs`** & **`expert_ticks`**: Real human NetHack ascensions (`archive.alt.org` xlogfile + dumplogs) and NLE reference trajectories stored in the exact same schema as `runs` and `ticks`.

---

## 9. Long-Horizon Scaling & Unattended Operations

Scaling an autonomous campaign to thousands of generations (and depths > 20 towards ascension) requires robust safeguards:

### 9.1 Adaptive Step Horizon Scaling
Early policies need short horizons to fail fast; deep policies need tens of thousands of steps to clear the Quest, Castle, and Gehennom. [`compute_adaptive_max_steps`](lox/engine/orchestrator.py) dynamically scales the episode horizon based on baseline depth milestones:
- $D_{p90} < 4$: **5,000 steps**
- $D_{p90} \in [4, 7]$: **10,000 steps**
- $D_{p90} \in [8, 12]$: **18,000 steps**
- $D_{p90} \in [13, 20]$: **30,000 steps**
- $D_{p90} > 20$: **50,000 steps**

When the horizon steps up, baseline timeout seeds are automatically refreshed at the new limit so paired hypothesis tests remain apples-to-apples.

### 9.2 Preemptive Generator Interrupt
Multi-step generator skills (such as tactical pathfinding, lock picking, or food searching) run through `_active_subgen`. If hero HP drops below critical ratio (`flee_hp_ratio`), an interactive prompt appears (`in_yn_prompt`, `in_text_prompt`), or hostile monsters appear adjacent, [`PolicyRunner.next_action`](lox/lang/compiler.py) preemptively clears `_active_subgen = None`, immediately returning control to the policy's primary dispatcher.

### 9.3 Telemetry Pruning & Milestone Checkpointing
- **Pruning**: Every 20 generations, [`prune_old_telemetry`](lox/db/logger.py) removes detailed turn-level `ticks` and failed `runs` older than 20 generations, executing a DuckDB `CHECKPOINT` to keep file size compact and query speeds under 100ms.
- **Checkpoints**: Whenever a candidate achieves a new record average depth, record max depth, or an ascension, `data/workspace/checkpoints/gen_XXXX_dYY.YY/` archives `policy.py`, `skills/`, and `meta.json`.

### 9.4 Fault-Tolerant Unattended Execution
- **Transient Exception Recovery**: The generation loop catches unexpected exceptions, logs full stack traces to `data/workspace/campaign_errors.log`, sleeps briefly, and advances to the next generation without terminating the campaign.
- **Live Markdown Dashboard**: Atomically updates [`data/workspace/STATUS.md`](data/workspace/STATUS.md) on every generation with campaign uptime, generation progress, progression overview tables, promoted discoveries, and diversity archive stats.
- **Read-Only Check-In CLI**: [`scripts/status_lox.py`](scripts/status_lox.py) queries DuckDB with `read_only=True` to inspect depth velocity, recent promotions, and discoveries while `run_lox.py` is actively evaluating.

---

## 10. Repository Structure

```text
lox/
├── core/
│   └── contracts.py        # Typed EpisodeRun, BatchEvaluation, CandidatePackage contracts
├── env/
│   ├── actions.py          # 1-to-1 Action IntEnum, key(), seq(), DIR_DELTAS
│   ├── base.py             # BenchmarkAdapter protocol & registry
│   ├── craftax/            # CraftaxAdapter, 17 actions, 22-achievement ruleset
│   ├── grid_utils.py       # Numba-JIT BFS, A*, room flood-fill, chokepoints, LOS raycasting
│   ├── harness.py          # Direct C++ NLE harness, lean observations, byte replay & watchdog
│   ├── nethack/            # NetHackAdapter & composite progression index
│   └── state.py            # Lazy zero-copy GameState, WorldMemory, C struct LUTs, stall tracking
├── lang/
│   ├── compiler.py         # Compiles policy.py + skills into PolicyRunner instances
│   ├── instrument.py       # ExecutionTracer, TunableParam, PolicyInstrumentTransformer
│   ├── macros.py           # Persistent MacroLibrary manager
│   └── validator.py        # Python 3.11+ AST sandbox validator, function splicer & contract linter
├── db/
│   ├── logger.py           # High-throughput DuckDB batch logger with tail-based tick sampling
│   ├── schema.py           # 8-table DuckDB analytical schema initialization & connections
│   ├── shards.py           # 3-tier PrefixShard extractor (Frontier, Mid-Band, Foundation)
│   └── summarizer.py       # RLE timeline compressor, stall analyzer, Ochiai/Liblit attribution
├── datasets/
│   ├── download_experts.py # Compiles NAO human ascensions & NLE traces into expert_runs / expert_ticks
│   └── wiki.py             # Offline NetHack 3.6.6 Wiki SQLite FTS5 BM25 search engine
├── runner/
│   └── pool.py             # Persistent multi-core worker pool & conditional trigger filter
├── author/
│   ├── agent.py            # Stateless ReAct AuthorAgent with compiler-in-the-loop self-repair
│   ├── prompts.py          # Epistemic System Prompt, dynamic User Prompt & Contrastive Digest
│   └── tools.py            # AuthorToolRegistry: query_db, query_wiki, query_policy_docs, dry_run_policy
├── engine/
│   ├── hypothesis.py       # Empirical Bernstein Confidence Sequences, Wilcoxon Signed-Rank, Futility Early Stopping
│   ├── modes.py            # CRN-SPSA parameter optimizer & AST constant baker
│   ├── orchestrator.py     # End-to-end CampaignOrchestrator, Adaptive Horizon & Status Reporter
│   └── scheduler.py        # Hybrid ModeScheduler (CREATE, WRITE, REFACTOR, PARAM_TUNE)
└── ui/
    └── console.py          # Compact ANSI console UI with single-line progress & summary tables
scripts/
├── build_strategy_wiki.py  # Indexes high-level NetHack strategy & mechanics articles into wiki_index.db
├── build_wiki_db.py        # Builds data/wiki_index.db from data/nethackwikidata.json
├── calibrate_pilot.py      # Calibrates pilot revision budget and evaluates transfer rates from DuckDB
├── download_experts.py     # Downloads and populates expert_runs and expert_ticks in data/lox.duckdb
├── reset_clean.py          # Resets campaign to Gen 0 while preserving expert corpus & wiki
├── run_lox.py              # Main CLI entrypoint for running synthesis campaigns
└── status_lox.py           # Morning/evening read-only check-in dashboard CLI
tests/
├── test_author_and_ui.py   # Author YAML/code parser, macro verification, pilot revisions
├── test_codebase_hardening.py # Typed contracts, alias synchronization, zero shard leakage
├── test_db_summarizer.py   # RLE summarizer, outlier marker, Ochiai/Liblit, DuckDB logger
├── test_harness_speed.py   # 1-to-1 actions, GameState queries, replay bit-identity
├── test_lang_instrument.py # Python 3.11+ features, tracer, loop guard, splicing
├── test_modes_and_eval.py  # Trigger filtering, hypothesis testing, SPSA tuning
├── test_overhaul_verification.py       # End-to-end regression suite
├── test_pilot_hardening_and_mutation_scope.py # Pilot revision capping, negative abort, mutation scope
├── test_skill_graph_and_adapters.py    # Multi-benchmark adapter and skill compilation tests
├── test_stall_and_prompt_passthrough.py# Stall telemetry, truncation, and prompt passthrough tests
└── test_todo_architectural_overhaul.py # Spatial WorldMemory and prefix shard extraction tests
```

---

## 11. Quickstart & CLI Usage

### 1. Installation, Linting & Automated Tests
LOX uses [`uv`](https://docs.astral.sh/uv/) for dependency management:
```bash
# Clone the repository
git clone git@github.com:saejin-moon/lox.git
cd lox

# Install dependencies into .venv
uv sync

# Run linter, formatter checks, and full test suite
uv run ruff check --fix .
uv run ruff format .
uv run pytest -v
```

### 2. Build Offline Wiki & Ingest Human Ascensions (Optional)
*Note: The orchestrator automatically performs this if databases are missing.*
```bash
uv run python scripts/build_wiki_db.py
uv run python scripts/build_strategy_wiki.py
uv run python scripts/download_experts.py --max-ascensions 50
```

### 3. Launch an Autonomous Campaign
Export your preferred LLM provider API key:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
# or export OPENAI_API_KEY="..."

# Run interactive synthesis campaign
uv run python scripts/run_lox.py \
  --provider openrouter \
  --model z-ai/glm-5.3-flash:nitro \
  --benchmark nethack \
  --workers 16 \
  --generations 50
```

### 4. Resume an Existing Campaign
To resume directly from a checkpoint or specific directory without restarting from Gen 0:
```bash
uv run python scripts/run_lox.py \
  --provider openrouter \
  --model z-ai/glm-5.3-flash:nitro \
  --db-path forest/lox.duckdb \
  --policy-path forest/workspace/policy.py \
  --workers 16 \
  --generations 50
```

### 5. Launch Long-Horizon Unattended Operations
For multi-day runs, launch inside a detached `tmux` session:
```bash
tmux new -s lox

uv run python scripts/run_lox.py \
  --provider openrouter \
  --model z-ai/glm-5.3-flash:nitro \
  --generations 5000 \
  --target-depth 50.0 \
  --workers 16

# Detach: Ctrl+B then D
```

### 6. Monitor Active Campaigns
Inspect campaign status without disturbing workers:
```bash
# Read-only check-in CLI
uv run python scripts/status_lox.py --since-hours 12

# Calibrate pilot revision budget from empirical DuckDB history
uv run python scripts/calibrate_pilot.py --db-path data/lox.duckdb

# Live status dashboard
cat data/workspace/STATUS.md

# Monitor errors (if any)
tail -f data/workspace/campaign_errors.log
```

---

## 12. References & Citations

### Environments & Long-Horizon Benchmarks
1. **Küttler, H., Nardelli, N., Miller, A. H., Raileanu, R., Selvatici, M., Grefenstette, E., & Rocktäschel, T.** (2020). *The NetHack Learning Environment.* Advances in Neural Information Processing Systems (NeurIPS), 33, 7671-7684.
2. **Hambro, E., Mohanty, S., Babaev, D., Byeon, M., Chakraborty, D., Grefenstette, E., Jiang, M., Jo, J., Kanervisto, A., Kim, J., et al.** (2022). *Insights from the NeurIPS 2021 NetHack Challenge.* Proceedings of Machine Learning Research (PMLR), 176, 41-52.
3. **Hafner, D.** (2022). *Benchmarking the Spectrum of Agent Capabilities (Crafter).* International Conference on Learning Representations (ICLR).
4. **Matthews, M., Beukman, M., Ellis, B., Samvelyan, M., Jackson, M., Coward, S., & Foerster, J.** (2024). *Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning.* International Conference on Machine Learning (ICML).

### Evolutionary Program Synthesis, Quality-Diversity & Library Learning
5. **Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J. R., Ellenberg, J. S., Wang, P., Fawzi, O., Kohli, P., & Fawzi, A.** (2024). *Mathematical discoveries from program search with large language models (FunSearch).* Nature, 625, 468-475.
6. **Novikov, A., Vu, N., Eisenberger, M., Dupont, E., Huang, P.-S., Wagner, A. Z., Shirobokov, S., Kozlovskii, B., Ruiz, F. J. R., Mehrabian, A., Kumar, M. P., See, A., Chaudhuri, S., Holland, G., Davies, A., Nowozin, S., Kohli, P., & Balog, M.** (2025). *AlphaEvolve: A coding agent for scientific and algorithmic discovery.* Google DeepMind Technical Report (and open-source reference architecture `rom1nux/alpha-evolve`).
7. **Ellis, K., Wong, C., Nye, M., Sablé-Meyer, M., Morales, L., Hewitt, L., Cary, L., Solar-Lezama, A., & Tenenbaum, J. B.** (2021). *DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning.* ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), 835-850.
8. **Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., & Anandkumar, A.** (2023). *Voyager: An Open-Ended Embodied Agent with Large Language Models.* Transactions on Machine Learning Research (TMLR).
9. **Mouret, J.-B., & Clune, J.** (2015). *Illuminating search spaces by mapping elites (MAP-Elites).* arXiv:1504.04909.

### Epistemic World Modeling, Contrastive Reflection & Execution-Guided Synthesis
10. **Tang, H., Key, D., & Ellis, K.** (2024). *WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment.* Advances in Neural Information Processing Systems (NeurIPS), 37.
11. **Wu, Y., Prabhumoye, S., Min, S. Y., Bisk, Y., Salakhutdinov, R., Azaria, A., Mitchell, T., & Li, Y.** (2023). *SPRING: Studying the Paper and Reasoning to Play Games.* Advances in Neural Information Processing Systems (NeurIPS), 36.
12. **Bundy, A., et al.** (2026). *Automated Epistemic Theory Revision and Failure-Driven World Model Repair in Symbolic Agents.* Artificial Intelligence.
13. **Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S.** (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning.* Advances in Neural Information Processing Systems (NeurIPS), 36.
14. **Zhao, A., Huang, D., Xu, Q., Lin, M., Liu, Y.-J., & Huang, G.** (2024). *ExpeL: LLM Agents Are Experiential Learners.* Proceedings of the AAAI Conference on Artificial Intelligence, 38(17), 19632-19642.
15. **Ridnik, T., Kredo, D., & Friedman, I.** (2024). *Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering.* arXiv:2401.08500.
16. **Chen, X., Lin, M., Schärli, N., & Zhou, D.** (2023). *Teaching Large Language Models to Self-Debug.* International Conference on Learning Representations (ICLR).
17. **Loi, M., et al.** (2026). *Contrastive Evaluation-Driven Program Synthesis for Long-Horizon Decision Making.* International Conference on Learning Representations (ICLR).

### Exploration, Subsumption Control & Temporal Abstractions
18. **Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., & Clune, J.** (2021). *First return, then explore (Go-Explore).* Nature, 590, 580-586.
19. **Brooks, R. A.** (1986). *A robust layered control system for a mobile robot (Subsumption Architecture).* IEEE Journal on Robotics and Automation, 2(1), 14-23.
20. **Sutton, R. S., Precup, D., & Singh, S.** (1999). *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning (Options Framework).* Artificial Intelligence, 112(1-2), 181-211.
21. **Böhme, M., Pham, V.-T., & Roychoudhury, A.** (2017). *Coverage-Based Greybox Fuzzing as Markov Chain (AFLFast).* IEEE Transactions on Software Engineering, 45(5), 489-506.

### Spectrum-Based Attribution, Stochastic Optimization & Sequential Hypothesis Testing
22. **Abreu, R., Zoeteweij, P., & van Gemund, A. J. C.** (2007). *On the Accuracy of Spectrum-based Fault Localization (Ochiai).* Testing: Academic and Industrial Conference Practice and Research Techniques (TAICPART-MUTATION), 89-98.
23. **Liblit, B., Naik, M., Zheng, A. X., Aiken, A., & Jordan, M I.** (2005). *Scalable statistical bug isolation.* ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), 15-26.
24. **Spall, J. C.** (1992). *Multivariate stochastic approximation using a simultaneous perturbation gradient approximation (SPSA).* IEEE Transactions on Automatic Control, 37(3), 332-341.
25. **Kleijnen, J. P. C.** (2009). *Design and Analysis of Simulation Experiments (Common Random Numbers).* Springer Series in Statistics.
26. **Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J.** (2021). *Time-uniform, nonparametric, nonasymptotic confidence sequences.* The Annals of Statistics, 49(2), 1055-1080.
27. **Waudby-Smith, I., & Ramdas, A.** (2024). *Estimating means of bounded random variables by betting.* Journal of the Royal Statistical Society Series B: Statistical Methodology, 86(1), 1-27.
