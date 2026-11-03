# LOX: Autonomous Empirical Policy Synthesis Engine

> **Architectural Motto:** *"Hard Code Nothing."*

`LOX` is an autonomous neuro-symbolic program synthesis engine that writes, tests, attributes, and evolves executable Python control policies and reusable macro libraries for long-horizon environments from first principles. It supports both the **NetHack Learning Environment (NLE)** and the **Crafter / Craftax** survival benchmark.

Standard deep reinforcement learning trains opaque neural weights over billions of environment frames, and online LLM agents pay a high latency and token cost on every single turn. `LOX` places the language model in the **outer scientific loop** and compiled Python 3.11+ in the **inner execution loop**:

- **Inner Loop (CPU / Numba)**: Compiled Python 3.11+ generator policies execute directly against the C/Python environment harness at **50,000 to 121,000+ steps/sec** across worker cores (`0` LLM tokens per step). The harness handles C prompt pagination (`flags.xwaitforspace` / `--More--` dismissal) automatically and supports microsecond action-byte prefix replay.
- **Outer Loop (Stateless Scientist LLM)**: Between evaluation batches, a stateless Author Agent queries a multi-table **DuckDB** warehouse (`runs`, `ticks`, `function_stats`, `discoveries`, `expert_runs`, `expert_ticks`) and an offline **NetHack 3.6.6 Wiki / Strategy Knowledge Base**. It isolates the active policy's top binding constraint, verifies environment rules against C struct tables (`permonst`, `objclass`) and Wiki articles, tests candidate code in a **5 x 30-episode execution-guided pilot loop** (`dry_run_policy`) with behavioral stall telemetry and prefix-replay shards, and submits verified candidates to sequential statistical hypothesis testing.

---

## Table of Contents

1. [Core Architectural Philosophy ("Hard Code Nothing")](#1-core-architectural-philosophy-hard-code-nothing)
2. [System Architecture & Dataflow](#2-system-architecture--dataflow)
3. [Autonomous Epistemic Discovery & Scientific System Prompt](#3-autonomous-epistemic-discovery--scientific-system-prompt)
4. [Multi-Benchmark Architecture: NetHack & Craftax](#4-multi-benchmark-architecture-nethack--craftax)
5. [Prefix-Replay Evaluation Shards & Spatial WorldMemory](#5-prefix-replay-evaluation-shards--spatial-worldmemory)
6. [The Five Modification Modes, MAP-Elites Archive & Library Learning](#6-the-five-modification-modes-map-elites-archive--library-learning)
7. [Multi-Table DuckDB Schema & Epistemic Persistence](#7-multi-table-duckdb-schema--epistemic-persistence)
8. [Parameter-Free Sequential Hypothesis Testing](#8-parameter-free-sequential-hypothesis-testing)
9. [Long-Horizon Scaling & Unattended Multi-Week Operations](#9-long-horizon-scaling--unattended-multi-week-operations)
10. [Repository Structure](#10-repository-structure)
11. [Quickstart & CLI Usage](#11-quickstart--cli-usage)
12. [References & Citations](#12-references--citations)

---

## 1. Core Architectural Philosophy ("Hard Code Nothing")

Most game-playing LLM agents depend on human-engineered domain wrappers: hardcoded monster threat tiers, pre-written corpse nutrition tables, built-in auto-explore routines, or fixed phase guides injected into the prompt. Those shortcuts help on Dungeon Level 1 but create a hard ceiling whenever the agent hits mechanics the human author did not pre-script.

`LOX` enforces a strict boundary between the **Architect** (the domain-agnostic execution and evaluation harness) and the **Author** (the evolving policy and macro codebase):

1. **Zero Hardcoded Strategy in the Harness ([`lox/env/`](lox/env/))**:
   - The harness contains **zero** game heuristics, **zero** monster/corpse/item lists, and **zero** hidden macro adapters.
   - It provides a **1-to-1 connection** between policy yields and native environment actions ([`Action`](lox/env/actions.py) / [`CraftaxAction`](lox/env/craftax/actions.py), [`key(ch)`](lox/env/actions.py) raw ASCII bytes, and [`seq(...)`](lox/env/actions.py) multi-keystroke sequences).
   - It exposes the game state lazily via [`GameState`](lox/env/state.py): all 27 `blstats` fields, `BL_MASK_*` status bitmasks, 21x79 or 48x48 grids (`glyphs`, `chars`, `colors`, `specials`), inventory slots, and pre-indexed C symbol table lookups over `nle.nethack.permonst` (`state.mon_info(y, x)`) and `nle.nethack.objclass` (`state.obj_info(glyph)`), paired with Numba-JIT grid topology primitives in [`lox/env/grid_utils.py`](lox/env/grid_utils.py).
   - It clears NetHack C engine pagination (`flags.xwaitforspace == 1` / `--More--` prompts) transparently while recording every raw byte for bit-identical replay.
2. **Near-Full Python 3.11+ Policy Language & Macro Library ([`lox/lang/`](lox/lang/))**:
   - The Author Agent writes standard Python 3.11+ in `data/workspace/policy.py`, `data/workspace/macros.py`, and modular skills in `data/workspace/skills/`: classes, stateful generator coroutines (`yield` / `yield from`), `match/case` pattern matching, comprehensions, `@dataclass`, closures, and safe standard library modules (`math`, `collections`, `heapq`, `itertools`, `functools`, `re`, `numpy`, `random`, `bisect`, `copy`).
   - Compile-time AST instrumentation ([`PolicyInstrumentTransformer`](lox/lang/instrument.py) and [`ExecutionTracer`](lox/lang/instrument.py)) adds `<1 us` per-function call tracking, per-function turn attribution, zero-yield loop protection ([`PolicyLoopError`](lox/lang/instrument.py)), and tunable numerical parameter hooks (`self.param(name, default, min_val, max_val)`).

---

## 2. System Architecture & Dataflow

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     STATELESS AUTHOR AGENT (Per-Generation)                      │
│  • 100% Static Cached System Prompt (Epistemic Discovery Loop + DuckDB Schema)   │
│  • Dynamic User Prompt: Mode Instructions + Active Code + Contrastive Digest     │
│  • Epistemic Memory: Verified Breakthroughs (discoveries) & Rejected Hypotheses  │
│  • Tools: query_db(sql) | query_wiki(q) | query_policy_docs(t) | dry_run_policy  │
│  • Execution-Guided Pilot Loop: Up to 5 revisions x 30 paired episodes (~0.8s)   │
│    with Behavioral Critic (stall telemetry) + Anti-Complexity Ratchet            │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Emits Causal YAML Hypothesis + Python Code
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│            HYBRID CADENCE / EXTREME-VALUE UCB SCHEDULER & MAP-ELITES             │
│  ┌────────────┐ ┌─────────────┐ ┌──────────────┐ ┌────────────┐ ┌──────────────┐ │
│  │ FULL_WRITE │ │ MACRO_WRITE │ │ FUNC_MOD_ADD │ │ PARAM_TUNE │ │ STRUCTURAL_  │ │
│  │  250 eps   │ │ 0 eps (inst)│ │ 150 trig eps │ │ 0 LLM toks │ │   REFACTOR   │ │
│  │ (Gen 0)    │ │ ->macros.py │ │ + 40 shards  │ │ (CRN-SPSA) │ │   200 eps    │ │
│  └─────┬──────┘ └──────┬──────┘ └──────┬───────┘ └─────┬──────┘ └──────┬───────┘ │
└────────┼───────────────┼───────────────┼───────────────┼───────────────┼─────────┘
         │               │ (Instant Gen) │               │               │
         ▼               ▼               ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│             PERSISTENT MULTI-CORE WORKER POOL (NLE & Craftax Adapters)           │
│  • Direct C / Numba Step Execution (50k-121k+ steps/sec across workers)          │
│  • Adaptive Horizon: Dynamically scales 5,000 -> 50,000 steps matching depth     │
│  • 3-Tier Stratified Shards: Frontier (>=P90), Mid-Band, Foundation (up to 400s) │
│  • Conditional Trigger Filter: discards episodes where target_fn call_count == 0 │
│  • Preemptive Generator Interrupt: aborts macro on HP drop, hostiles, or prompts  │
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
│  • Automated Telemetry Pruning (keeps DB nimble during 5k+ generation campaigns) │
│  • Consolidated Human Ascensions & Expert Traces (expert_runs & expert_ticks)    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Autonomous Epistemic Discovery & Scientific System Prompt

Instead of feeding the LLM human strategy guides for different dungeon depths, `LOX` structures the Author Agent's system prompt ([`lox/author/prompts.py`](lox/author/prompts.py)) as a domain-general **epistemic discovery loop** grounded in recent program synthesis and world-modeling research:

1. **Binding Constraint Diagnosis (Theory of Bottlenecks)**:
   - Before proposing code, the agent queries `runs`, `ticks`, and `function_stats` to isolate the single highest-frequency cause of death or stagnation holding back the current policy distribution (`avg_depth`, `p90_depth`, `max_depth`). It must record this empirical bottleneck in the YAML `binding_constraint` field.
2. **Contrastive Trajectory Attribution (Outliers & Experts vs. Failures)**:
   - Every generation receives an automated contrastive telemetry digest ([`build_batch_contrast_digest`](lox/db/summarizer.py)) comparing the top 10% outlier trajectories (`is_outlier_good = TRUE`) and verified human ascensions (`expert_runs`, `expert_ticks`) against bottom-decile stalled or dead runs (`end_reason`, `killer_name`, `active_function` at death).
3. **Epistemic Grounding ("Wiki-Before-Write" & C-Struct Inspection)**:
   - The agent is prohibited from guessing hidden game mechanics (e.g., monster speed, MR, corpse intrinsics, diagonal doorway constraints, or multi-keystroke command syntax). It queries the offline NetHack 3.6.6 Wiki (`query_wiki`) and runtime C symbol tables (`state.mon_info(y, x)`, `state.obj_info(glyph)`), and states the verified mechanic in the YAML `verified_c_rule` field.
4. **Epistemic Persistence (`discoveries` & Failure Memory)**:
   - Promoted breakthroughs with verified C rules are saved into the `discoveries` table and rendered into every subsequent generation prompt under `### Verified Campaign Discoveries (Do Not Regress)`.
   - Past rejected hypotheses are rendered under `### Recent Rejected Hypotheses (Do NOT Repeat)` to prevent cyclic re-testing of failed ideas.
5. **Subsumption & Preemption Priority Architecture**:
   - Policies structure `run(self, state)` as an ordered priority dispatcher (multi-turn prompt safety -> immediate survival -> active engagement -> progression & descent -> frontier exploration -> fallback). Multi-step generator skills are checked for preemption conditions (`state.adjacent_hostiles()`, `state.hp_ratio`, `state.in_yn_prompt`) so exploration or item management yields control the instant a higher-priority threat appears.
6. **Execution-Guided Pilot Loop with Behavioral Critic & Complexity Ratchet**:
   - During `dry_run_policy`, the candidate runs across 30 paired baseline episodes (including failure seeds and prefix shards) in ~0.8 seconds. The tool returns per-seed progression deltas, function call counts, and behavioral stall metrics (`avg_unique_tiles`, `stall_rate`, `pct_WAIT`, `pct_SEARCH`). An anti-complexity ratchet rejects speculative bloat that adds lines of code without improving triggered progression.

---

## 4. Multi-Benchmark Architecture: NetHack & Craftax

`LOX` uses a unified [`BenchmarkAdapter`](lox/env/base.py) protocol that decouples environment mechanics from policy compilation and statistical hypothesis testing:

### 4.1 NetHack Learning Environment (`--benchmark nethack`)
- **Engine**: Direct C++ `nle.nethack.Nethack` instance with automatic `--More--` pause dismissal.
- **State Representation**: 21x79 grids, 27 `blstats` fields, status masks (`BL_MASK_BLIND`, `BL_MASK_CONF`, `BL_MASK_FOODPOIS`, `BL_MASK_GRAB`, etc.), inventory vectors, and C `permonst`/`objclass` metadata.
- **Progression Metric**: Continuous composite index over max dungeon depth (`max_depth`), experience level (`xp_level`), armor class improvement (`10 - final_ac`), log-scaled score, and pre-death shard survival.

### 4.2 Crafter / Craftax Survival Benchmark (`--benchmark craftax`)
- **Engine**: Pure deterministic Crafter/Craftax survival engine with 17 discrete actions.
- **Facing Direction Mechanics**: `LEFT`, `RIGHT`, `UP`, and `DOWN` orient the player; `Action.DO` and `PLACE_*` act on the faced coordinate `(y + dy, x + dx)`.
- **Full 22-Achievement Survival Tech Tree**: Wood gathering, crafting table placement, wooden pickaxe crafting, stone mining, furnace placement, stone pickaxe, iron mining, iron smelting, weapon crafting, hostile mob combat, and thirst/hunger/energy maintenance.
- **Progression Metric**: Total survival achievements unlocked plus health and survival duration bonuses.

---

## 5. Prefix-Replay Evaluation Shards & Spatial WorldMemory

### 5.1 3-Tier Stratified Checkpoint Prefix Shards ([`lox/db/shards.py`](lox/db/shards.py))
Evaluating mid-game combat or deeper dungeon tactics from turn 0 wastes thousands of steps on early exploration. `LOX` extracts up to 40 stratified **Checkpoint Prefix Shards** directly from recorded baseline `action_bytes`:
1. **Pre-Death Windows**: Sliced 30 turns before a fatal encounter (`end_reason == 'died'`).
2. **Pre-Stall Windows**: Sliced 5 turns before detected oscillation or timeout stagnation.
3. **Floor-Entry Checkpoints**: Sliced at the exact turn the hero arrives on a deeper dungeon floor (`depth >= 2`).

Shards are dynamically stratified to focus compute where discovery happens:
- **Frontier Tier (≥ P90 Depth, up to 55%)**: Deepest floor entries and pre-death encounters. Deep floor entries (`depth >= 5`) receive an extended **400-step evaluation horizon**.
- **Mid-Band Tier (Mean Depth to P90 Depth, up to 30%)**: Intermediate progression checkpoints.
- **Foundation Tier (< Mean Depth, remaining 15%)**: Shallow robustness checkpoints.

During `FUNC_MOD_ADD` evaluation and `dry_run_policy` pilots, workers replay prefix bytes in microseconds (`harness.replay_bytes`), reset the post-replay step counter and execution tracer, and hand control to the candidate policy. Candidate shards are paired strictly against baseline shards on the exact same prefix (`seed >= 1,000,000`), and surviving a fatal pre-death shard awards a direct survival bonus in [`progression_metric`](lox/engine/hypothesis.py).

### 5.2 Dynamic Spatial WorldMemory & Numba Topology ([`lox/env/state.py`](lox/env/state.py), [`lox/env/grid_utils.py`](lox/env/grid_utils.py))
Every [`GameState`](lox/env/state.py) maintains a persistent per-episode [`WorldMemory`](lox/env/state.py) (`state.mem`) that tracks per-floor spatial state and preserves terrain features even when occluded by the hero `@`, monsters, or objects:
- `visited` & `visit_count`: 2D grids tracking explored tiles and per-tile visit frequency.
- `passable_mask` & `door_mask`: Dynamically updated walkability and closed-door masks that remember walkable floor tiles when monsters step onto them and block locked doors until unlocked or kicked open.
- `remembered_chars`: Persistent map memory that preserves underlying terrain (`'>'`, `'<'`, `'_'`, `'{'`, `'+'`, `'#'`, `'.'`) when the hero `'@'` or a monster occupies the tile, preventing stair/altar occlusion loops. Trees and iron bars are normalized to `'|'`, while ice and lowered drawbridges are normalized to `'.'`.
- `kv`: Persistent dictionary on `state.mem.kv` (and exposed via `state.kv`) synchronized across floor transitions and replay shards for tracking items, altars, and shop locations.
- `recent_messages`: Ring buffer of recent non-empty message tuples `(turn, dlvl, message)` exposed via `state.recent_messages`.
- `searched_counts`: 2D counter incremented around `(y, x)` on `Action.SEARCH` to guide dead-end secret door searches.
- `stair_graph`: Bidirectional inter-floor dictionary mapping `(dlvl, y, x)` stair coordinates across levels.
- `pos_history` & `is_oscillating`: Bounded coordinate history (`deque(maxlen=32)`) and automatic 2-step/3-step cycle detector.
- **Numba-JIT Spatial Helpers** ([`lox/env/grid_utils.py`](lox/env/grid_utils.py)):
  - `state.bfs_distance_map(sources, passable)` -> 2D int32 distance grid in `<50 us`.
  - `state.pathfind_step_to(target_y, target_x, passable, cost_grid)` -> 8-way A* first-step `(dy, dx)` with cardinal preference at doorway transitions.
  - `flood_fill_rooms(passable_mask)` -> Connected room component labeling.
  - `find_chokepoints(passable_mask)` -> Boolean mask of 1-tile-wide corridor and doorway bottlenecks for tactical positioning.
  - `line_of_sight_mask(y, x, opaque_mask, max_range)` -> 8-direction raycast visibility mask for ranged wand/missile alignment.

---

## 6. The Five Modification Modes, MAP-Elites Archive & Library Learning

| Mode | Episode Budget | Verification Gate | Purpose |
| :--- | :--- | :--- | :--- |
| **`FULL_WRITE`** | `250` episodes | 30-episode live smoke test | Synthesizes `class Policy` from scratch on Generation 0 without a pre-fed starter policy. |
| **`MACRO_WRITE`** | `0` episodes *(instant)* | Real-state execution across 3 fixtures | Synthesizes reusable generator/helper macros into `data/workspace/macros.py` with static AST contract linting. |
| **`FUNC_MOD_ADD`** | `150` triggered eps / fn + `40` shards | 5 x 30-episode pilot revision loop | Surgically adds or modifies targeted methods on `Policy` ([`splice_functions_into_policy`](lox/lang/validator.py)); untriggered episodes are discarded (`triggered == False`). |
| **`PARAM_TUNE`** | `0` LLM tokens (`CRN-SPSA`) | Paired CRN validation batch | Optimizes all `self.param(name, default, min, max)` thresholds via Common Random Numbers SPSA and bakes winning constants back into the AST. |
| **`STRUCTURAL_REFACTOR`** | `200` episodes | 5 x 30-episode pilot revision loop | Triggered when `policy.py` exceeds 550 lines or during stagnation to break large monolithic logic into modular files under `data/workspace/skills/`. |

### MAP-Elites Behavioral Archive ([`lox/engine/orchestrator.py`](lox/engine/orchestrator.py))
To preserve stepping-stone adaptations without reverting promoted gains, `CampaignOrchestrator` maintains a quality-diversity archive across three behavioral niches:
- `descent`: Champion policy maximizing mean and 90th-percentile dungeon depth (`avg_depth`, `p90_depth`).
- `combat_ac`: Elite policy minimizing armor class (`avg_ac`) while maintaining non-regressing depth.
- `progression_xl`: Elite policy maximizing experience level (`avg_xl`) while maintaining non-regressing depth.

### Wake-Sleep Library Learning ([`lox/lang/macros.py`](lox/lang/macros.py))
- **Top-Down Synthesis (`MACRO_WRITE`)**: Verified against three real game-state fixtures (open room with items/stairs, adjacent hostile combat, and locked-door dead-end corridor) before committing to `data/workspace/macros.py`.
- **Bottom-Up Extraction (`MacroHarvester`)**: Automatically harvests pure/stateless helper methods with high Ochiai Good-Run scores (`ochiai_good_run_score >= 0.25`) from promoted policies, rewriting `self.<mem_attr>` references to `state.mem.<mem_attr>` and promoting them into `data/workspace/macros.py`.

---

## 7. Multi-Table DuckDB Schema & Epistemic Persistence

All experimental telemetry and reference human ascensions are stored in a single DuckDB analytical database (`data/lox.duckdb`, defined in [`lox/db/schema.py`](lox/db/schema.py)):

1. **`batches`**: Per-generation summary (`batch_id`, `generation`, `mode`, `target_functions`, `avg_depth`, `p90_depth`, `max_depth`, `hypothesis_passed`, `wilcoxon_p`, `ci_lower`, `ci_upper`).
2. **`runs`**: Episode-level telemetry (`run_id`, `seed`, `max_depth`, `final_hp`, `final_ac`, `xp_level`, `score`, `turns`, `end_reason`, `killer_name`, `summary_digest`, `game_log`, `milestones_json`, `is_outlier_good`, `outlier_z_score`, `action_bytes`).
3. **`ticks`**: Tail-sampled turn traces (`run_id`, `turn`, `dlvl`, `hp`, `max_hp`, `ac`, `xp_level`, `hunger_state`, `y`, `x`, `action_name`, `active_function`, `message`, `nearby_monsters`).
4. **`function_stats`**: Spectrum-based fault and success attribution per policy method (`episodes_triggered`, `total_calls`, `total_turns_active`, `ochiai_good_run_score`, `liblit_failure_score`).
5. **`policies`**: Versioned snapshots of every candidate and promoted `policy_code` and `macros_code`.
6. **`experiments`**: Causal hypotheses, binding constraints, verified C rules, pilot diagnostics, and statistical verdicts.
7. **`discoveries`**: Validated campaign breakthroughs (`discovery_id`, `generation`, `batch_id`, `mode`, `target_functions`, `binding_constraint`, `verified_c_rule`, `mechanism`, `mean_delta`, `avg_depth`, `p90_depth`, `max_depth`, `created_at`).
8. **`expert_runs`** & **`expert_ticks`**: Real human NetHack ascensions (`archive.alt.org` xlogfile + dumplogs) and NLE reference trajectories stored in the exact same schema as `runs` and `ticks`.

---

## 8. Parameter-Free Sequential Hypothesis Testing

The Author Agent never guesses numerical effect sizes. It emits a structured qualitative YAML hypothesis before its Python code block:

```yaml
mode: FUNC_MOD_ADD
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

Candidate policies are evaluated on paired seeds against the active baseline using three objective statistical gates in [`lox/engine/hypothesis.py`](lox/engine/hypothesis.py):
1. **Sequential Futility Early Stopping**: If after $n \ge 60$ paired triggered seeds the 95% Empirical Bernstein Confidence Sequence upper bound satisfies $U_n \le 0.0$, evaluation terminates early to save compute.
2. **Positive Mean Progression Delta**: $\bar{d} > 0.0$ and strictly more improved seeds than regressed seeds ($\text{wins} > \text{losses}$).
3. **Statistical Significance**: Either the 95% Confidence Sequence lower bound $L_N > 0.0$ or the one-sided Paired Wilcoxon Signed-Rank p-value satisfies $p < 0.05$.

---

## 9. Long-Horizon Scaling & Unattended Multi-Week Operations

Scaling an autonomous campaign to thousands of generations (and depths > 20 towards ascension) introduces unique challenges: false timeouts on deep levels, memory bloat, monolithic policy code, and network/process hiccups. `LOX` implements automated safeguards:

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
- **Checkpoints**: Whenever a candidate achieves a new record average depth, record max depth, or an ascension, `data/workspace/checkpoints/gen_XXXX_dYY.YY/` archives `policy.py`, `macros.py`, and `meta.json`.

### 9.4 Fault-Tolerant Unattended Operations
- **Transient Logging**: The generation loop catches unexpected transient exceptions, logs full stack traces to `data/workspace/campaign_errors.log`, sleeps briefly, and advances to the next generation without crashing the campaign.
- **Live Markdown Dashboard**: Atomically updates [`data/workspace/STATUS.md`](data/workspace/STATUS.md) on every generation with campaign uptime, generation progress, progression overview tables, promoted discoveries, and diversity archive stats.
- **Read-Only Check-In CLI**: [`scripts/status_lox.py`](scripts/status_lox.py) queries DuckDB with `read_only=True` to inspect depth velocity, recent promotions, and recent discoveries while `run_lox.py` is actively evaluating.

---

## 10. Repository Structure

```text
lox/
├── env/
│   ├── base.py             # BenchmarkAdapter protocol & registry
│   ├── actions.py          # 1-to-1 Action IntEnum, key(), seq(), DIR_DELTAS
│   ├── state.py            # Lazy zero-copy GameState, WorldMemory, NLE C LUTs
│   ├── grid_utils.py       # Numba-JIT BFS, A*, room flood-fill, chokepoints, LOS raycasting
│   ├── harness.py          # Direct C++ NLE harness & microsecond byte replay
│   ├── nethack/            # NetHackAdapter & knowledge base integration
│   └── craftax/            # CraftaxAdapter, 17 actions, 22-achievement survival ruleset
├── lang/
│   ├── validator.py        # Python 3.11+ AST sandbox validator, function splicer & contract linter
│   ├── instrument.py       # ExecutionTracer, TunableParam, PolicyInstrumentTransformer
│   ├── macros.py           # MacroLibrary manager & MacroHarvester bottom-up library learning
│   └── compiler.py         # Compiles macros.py + policy.py + skills into PolicyRunner instances
├── db/
│   ├── schema.py           # 8-table DuckDB schema initialization & read/write connections
│   ├── shards.py           # 3-tier PrefixShard extractor (Frontier, Mid-Band, Foundation)
│   ├── summarizer.py       # RLE timeline compressor, stall analyzer, Ochiai/Liblit attribution
│   └── logger.py           # High-throughput DuckDB batch logger with tail-based tick sampling
├── datasets/
│   ├── wiki.py             # Offline NetHack 3.6.6 Wiki SQLite FTS5 BM25 search engine
│   └── download_experts.py # Compiles NAO human ascensions & NLE traces into expert_runs / expert_ticks
├── runner/
│   └── pool.py             # Persistent multi-core worker pool & conditional trigger filter
├── author/
│   ├── prompts.py          # Epistemic System Prompt, dynamic User Prompt & Contrastive Digest
│   ├── tools.py            # AuthorToolRegistry: query_db, query_wiki, query_policy_docs, dry_run_policy
│   └── agent.py            # Stateless ReAct AuthorAgent with compiler-in-the-loop self-repair
├── engine/
│   ├── hypothesis.py       # Confidence Sequences, Wilcoxon Signed-Rank, Futility Early Stopping
│   ├── modes.py            # CRN-SPSA parameter optimizer & AST constant baker
│   ├── scheduler.py        # Hybrid cadence + Extreme-Value UCB mode scheduler
│   └── orchestrator.py     # End-to-end CampaignOrchestrator, Adaptive Horizon & Status Reporter
└── ui/
    └── console.py          # Compact ANSI console UI with single-line progress & summary tables
scripts/
├── run_lox.py              # Main CLI entrypoint for running synthesis campaigns
├── status_lox.py           # Morning/evening read-only check-in dashboard CLI
├── reset_clean.py          # Resets campaign to Gen 0 while preserving expert corpus & wiki
├── download_experts.py     # Populates expert_runs and expert_ticks in data/lox.duckdb
├── build_wiki_db.py        # Builds data/wiki_index.db from data/nethackwikidata.json
└── build_strategy_wiki.py  # Indexes high-level NetHack strategy & mechanics articles into wiki_index.db
tests/
├── test_overhaul_verification.py       # End-to-end regression suite (Passes 1-8 checks)
├── test_skill_graph_and_adapters.py    # Multi-benchmark adapter and skill compilation tests
├── test_todo_architectural_overhaul.py # Spatial WorldMemory and prefix shard extraction tests
├── test_harness_speed.py               # 1-to-1 actions, GameState queries, replay bit-identity
├── test_lang_instrument.py             # Python 3.11+ features, tracer, loop guard, splicing
├── test_db_summarizer.py               # RLE summarizer, outlier marker, Ochiai/Liblit, DuckDB logger
├── test_modes_and_eval.py              # Trigger filtering, hypothesis testing, SPSA tuning
└── test_author_and_ui.py               # Author YAML/code parser, macro verification, pilot revisions
```

---

## 11. Quickstart & CLI Usage

### 1. Run Linting & Automated Tests
Always use `uv` and `ruff`:
```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest -v
```

### 2. Build the Offline Wiki & Ingest Expert Ascensions (Optional)
```bash
uv run python scripts/build_wiki_db.py
uv run python scripts/build_strategy_wiki.py
uv run python scripts/download_experts.py --max-ascensions 50
```

### 3. Reset Workspace to Clean Gen 0 State (Optional)
Resets active policies and run telemetry while keeping `expert_runs`, `expert_ticks`, and `data/wiki_index.db` intact:
```bash
uv run python scripts/reset_clean.py
```

### 4. Launch an Autonomous Campaign (Standard / Interactive)
Ensure `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) is configured in `.env` or exported in your shell:

```bash
uv run python scripts/run_lox.py \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --benchmark nethack \
  --workers 16 \
  --generations 50
```

### 5. Launch a Multi-Day / 5,000-Generation Unattended Campaign
For multi-week unattended progression, run inside a detached `tmux` session:

```bash
# 1. Create a tmux session
tmux new -s lox

# 2. Launch 5,000 generations targeting full dungeon depth
uv run python scripts/run_lox.py \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --generations 5000 \
  --target-depth 50.0 \
  --workers 16

# 3. Detach with Ctrl+B then D
```

### 6. Monitor Active Campaigns While Running
Safely inspect progress without interrupting workers:

```bash
# CLI summary of recent activity and depth velocity
uv run python scripts/status_lox.py --since-hours 12

# View the latest generated markdown status report
cat data/workspace/STATUS.md

# Monitor transient error logs (if any)
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
23. **Liblit, B., Naik, M., Zheng, A. X., Aiken, A., & Jordan, M. I.** (2005). *Scalable statistical bug isolation.* ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), 15-26.
24. **Spall, J. C.** (1992). *Multivariate stochastic approximation using a simultaneous perturbation gradient approximation (SPSA).* IEEE Transactions on Automatic Control, 37(3), 332-341.
25. **Kleijnen, J. P. C.** (2009). *Design and Analysis of Simulation Experiments (Common Random Numbers).* Springer Series in Statistics.
26. **Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J.** (2021). *Time-uniform, nonparametric, nonasymptotic confidence sequences.* The Annals of Statistics, 49(2), 1055-1080.
27. **Waudby-Smith, I., & Ramdas, A.** (2024). *Estimating means of bounded random variables by betting.* Journal of the Royal Statistical Society Series B: Statistical Methodology, 86(1), 1-27.
