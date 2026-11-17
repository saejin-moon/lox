# LOX Future Architectural Roadmap — Path B (Constructed Mid/Late-Game Scenarios & Hybrid Scaffold Pivot)

> **Status:** Deferred until **Path A + Organic Auto-Mined Scenario Bank** (see `implementation_plan.md`) is implemented, verified, and benchmarked across 10+ live generations.
> **Constraint ("Hard Code Nothing"):** Per `AGENTS.md`, `lox/env/` and `lox/author/prompts.py` must NEVER encode NetHack strategy heuristics, monster/item name lists, or hardcoded solutions. Everything below adds **evaluation environments** (benchmarks) or **domain-agnostic framework structure**, not game-specific answers.

---

## Why Path B Exists & When to Trigger It

Path A fixes LOX's 9 structural bottlenecks and adds an **Organic Auto-Mined Scenario Bank** extracted from LOX's own `(seed, action_bytes)` trajectories in DuckDB. However, first-principles analysis of NetHack (`AutoAscend`, `NLD-AA`, `BALROG`, `AlphaEvolve`) identifies two remaining scaling cliffs that Path A alone may hit once LOX reliably reaches `Dlvl 10–15+`:

1. **Late-Game Sample Starvation (Zero-Gradient Reachability Wall):**
   - Rare late-game mechanics—Sokoban boulder puzzles, Medusa gaze reflection, Castle drawbridges, Demon Princes, Wizard of Yendor, and the Astral Plane—occur in `< 0.1%–5%` of turn-0 runs. Even with Go-Explore `FrontierArchive` checkpoints, LOX cannot optimize a branch it has never reached once from turn 0.
2. **Global Code Coupling Ceiling (~4K–5K LOC Wall):**
   - Literature (`AlphaEvolve`, `FunSearch`, `Voyager`) shows freeform multi-file LLM co-evolution degrades beyond ~4,000–5,000 LOC of coupled stateful control without frozen architectural contracts and per-module test suites.

### Graduation Criteria (When to Start Path B)
Activate Path B only after Path A meets **both**:
- Sustained depth/vital progression (`avg_depth >= 8.0`, `avg_ac <= 2.0`, working armor + healing/consumables + multi-floor navigation) across held-out seeds; AND
- Clear plateau where > 50% of runs die in unreached/rare structural stages (e.g., Sokoban, Mines' End, Quest, Medusa, Castle) or skill-library LOC exceeds ~3,500 LOC with cross-module interference.

---

## Pillar B1: Constructed Deterministic Late-Game Scenario Suite (MiniHack / `.des` Benchmarks)

Build a curated curriculum of isolated, deterministic NLE/`minihack` scenario adapters under `lox/env/benchmarks/` (extending the existing `BenchmarkAdapter` protocol alongside `nethack` and `craftax`):

1. **Scenario Benchmarks (Pure Evaluation Environments — Zero Strategy in Harness):**
   - **Shop Commerce & Debt Scenario:** Small shop room (`gold`, unpaid armor/scrolls, shopkeeper) scoring safe exit/purchase without shopkeeper anger.
   - **Altar / BUC & Sacrifice Scenario:** Co-aligned/cross-aligned altar + mixed cursed/uncursed/blessed items + corpses scoring safe BUC detection/blessing/sacrifice.
   - **Sokoban Puzzle Scenario:** Standard NLE/MiniHack Sokoban boulder/pit room scoring boulder pushes without deadlocking paths.
   - **Medusa / Gaze / Reflection Scenario:** Reflection gear (`"`) + Medusa room scoring safe survival/stairs traversal.
   - **Castle / Moat / Drawbridge Scenario:** Moat, drawbridge, trap doors, and wand/instrument entry puzzle.
   - **Disease / Stoning / Inventory Recovery Scenario:** Food poisoning / delayed stoning / rust / engulfed states starting 15 turns before fatal timer.
2. **Curriculum Gating (`SwarmReferee` Tiered Evaluation):**
   - Candidates targeting a specific capability must pass both its **Constructed Scenario Unit Suite** (100% trigger rate, fast ms-scale checks) AND the **Rotating Full-Game CRN + Held-Out Validation Suite** (`N >= 48` fresh seeds) before promotion.
   - Keep scenario suite definitions strictly separate from the policy's code (`"Hard Code Nothing"` in `lox/env/` and policy): the harness provides the *problem room*, never hints on *how* to solve it.

---

## Pillar B2: Frozen World-Model Scaffold + `#EVOLVE-BLOCK` Subsystems (`AlphaEvolve` Pattern)

Instead of allowing peers to mutate arbitrary global blackboard keys and cross-module signatures across 4,000+ LOC:

1. **Frozen Domain-Agnostic Architectural Interfaces (`scaffold/`):**
   - Define narrow typed interfaces (still 100% domain-agnostic, exposing only pure C/NLE concepts):
     - `WorldTracker`: multi-floor `FloorMemory`, `level_exits`, known object/monster memory, shop/danger annotations derived by policy utilities.
     - `CapabilityScorer` / `ThreatEvaluator`: pure scoring functions over `VisibleMonster` (`permonst` stats) and `InventoryItem` (`objclass` stats, `oc_oprop`, `enchantment`, BUC flags).
     - `GoalPlanner` / `StrategyStack`: explicit preemptible task stack replacing monolithic reactive fallbacks.
2. **Bounded `#EVOLVE-BLOCK` Peer Mutation:**
   - Each LLM peer evolves a single **20–200 LOC pure function or strategy class** behind frozen typed signatures verified by strict unit-contract tests + scenario suites, preventing cross-subsystem regressions as total codebase size grows past 5,000+ LOC.

---

## Pillar B3: Preemptive Imperative Strategy-Stack Engine (`AutoAscend`-Style)

For tasks that exceed even `MemorySequence` + coroutine `ActionLeaf` complexity (multi-level questing, stash consolidation, Sokoban search, wish selection):

1. **Composable Strategy Combinators:**
   - Replace top-level BT fallback loops with explicit preemptible imperative `Strategy` generators supporting structured `preempt_if(predicate)`, `until(predicate)`, `subgoal_stack`, and `rollback/cleanup` frames across floor transitions.
2. **Domain-Agnostic Multi-Objective Promotion:**
   - Evolve policies against **Pareto milestone vectors** across full games (`depth`, `xp_level`, `-ac`, `dungeons_visited`, `items_identified`, `resistances_gained` via `worn_oprop_mask`, `scenario_pass_rate`) rather than single-axis `avg_depth` hill-climbing.
3. **Optional Offline Imitation / Distillation Warm-Start Benchmark (`NLD-AA` / `expert_runs`):**
   - Use `expert_runs` / `expert_ticks` not just as SQL contrast tables, but as state-action calibration benchmarks (measuring action-agreement / vital-trajectory parity on early-to-mid dungeon floors) to bootstrap missing mid-game reflexes faster.
