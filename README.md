# LOX: Autonomous Empirical Policy Synthesis Engine

LOX is an autonomous, empirical neuro-symbolic policy synthesis engine for NetHack, Craftax, and open-ended domains. It enables open, efficient large language models (such as `qwen/qwen3.5-9b` and `google/gemma-4-31b-it`) to iteratively author, evaluate, diagnose, and evolve pure Python generator policies based on empirical execution telemetry, DuckDB incident autopsies, and deep domain encyclopedias.

LOX achieves **826+ steps/second** (>1,000 game turns/second across 20 parallel workers) by executing sandboxed Python generator policies alongside precomputed vectorized glyph lookup tables and a disk-cached Numba JIT A* spatial navigation engine. Across 68 continuous evolutionary campaigns and over 12,800 real episodes evaluated (38.0M+ game ticks), LOX has systematically eliminated zero-progress stalls, locked-door loops, premature prayer smiting, and starvation blackouts, autonomously forged **the blessed rustproof +1 Excalibur**, achieved **Record Average Dungeon Depth 5.10**, reached **Dungeon Depth 15**, and achieved **Peak Score 4,612** with average lifespans exceeding **8,096 turns**—all for **~$0.03 per 100-episode batch**.

---

## 1. System Architecture: The Three-Tier Meta-Optimization Loop

LOX is structured as a **Three-Tier Hierarchical Meta-Optimization Architecture**, separating high-speed environment execution from inner-loop policy mutation and outer-loop platform hardening:

```mermaid
flowchart TD
    subgraph Level2["Level 2: The Meta-Architect (Outer Loop / Developer & Meta-Agent)"]
        direction TB
        L2_A["Harness & Engine Optimization (C-level NLE, Numba JIT, Vector LUTs)"]
        L2_B["System Architecture & 47 Invariants (AGENTS.md, invariants.py)"]
        L2_C["Post-Campaign Empirical Failure Diagnosis & Autopsy"]
    end

    subgraph Level1["Level 1: The Inner-Loop Policy Synthesizer (AuthorAgent / Qwen / Gemma)"]
        direction TB
        L1_A["DuckDB Telemetry Ingestion (Trustworthy Triad Dossier on Turn 0)"]
        L1_B["AST Modular Skill Splicing & Hypothesis Formulation (YAML)"]
        L1_C["Multi-Scenario Dry-Run Validation (SIGALRM Timeout Ceiling)"]
    end

    subgraph Level0["Level 0: The Embodied Execution Policy (data/latest_policy.py)"]
        direction TB
        L0_A["Numba JIT Pathfinding & Persistent Topological Memory"]
        L0_B["826+ steps/sec Parallel Multiprocessing (20 Workers)"]
        L0_C["Real NetHack Dungeon Interaction (Depth 1 to 50 / Ascension)"]
    end

    Level2 -->|"Hardens Environment & Scaffolds Invariants"| Level1
    Level1 -->|"Compiles & Promotes latest_policy.py"| Level0
    Level0 -->|"Streams Ticks & Episode Telemetry into DuckDB"| Level1
    Level1 -->|"Reports Batch Progression & Fatalities"| Level2
```

### The Three Operational Tiers
1. **Level 0: The Embodied Execution Policy ([`data/latest_policy.py`](file:///home/moose/git/lox/data/latest_policy.py))**:
   - A standalone, deterministic Python generator class (`Agent.run(obs)`) yielding action primitives (`obs = yield action`).
   - Runs directly against the NetHack C-library via Gym/NLE at **826+ steps/second** with zero network latency and zero API calls at runtime.
   - Pacing is fully autonomous: exploration budgets and descent timing are governed strictly within `Agent.run(obs)` and `skill_explore_and_dive`, keeping the environment harness purely objective.
   - Maintains full-floor topological memory across FOV boundaries and routes around passive hazards using Numba-accelerated A* pathfinding.
2. **Level 1: The Inner-Loop Policy Synthesizer ([`lox/author/agent.py`](file:///home/moose/git/lox/lox/author/agent.py))**:
   - Driven by `qwen/qwen3.5-9b` or `google/gemma-4-31b-it` via OpenRouter.
   - **Turn 0 Pre-Compiled Pareto Dossier & 5-Turn Hard Budget Ceiling**: Receives the pre-compiled **Trustworthy Triad** (Macro Archetype Table, Orthogonal Systemic Drivers, Ground-Truth Telemetry) alongside targeted offline Wiki mechanics directly on Turn 0. Hard ceiling of 5 tool turns (`max_tool_turns = 5`, `tool_choice = "auto"`) eliminates context snowballing and slashes synthesis costs.
   - Formulates a structured, falsifiable scientific hypothesis in YAML (`causal_finding`, `targeted_skill`, `mechanism`, `predicted_outcome`).
   - Surgically modifies individual modular skills via AST splicing, validating candidate policies against sandbox rules and a hardware-timed multi-scenario dry-run.
   - Evaluates candidate policies against identical seed twins via counterfactual replay (`--twin-test`), validating or falsifying hypotheses with paired bootstrap confidence intervals in DuckDB.
3. **Level 2: The Meta-Architect (Outer Loop / Developer & Meta-Agent)**:
   - Evaluates cross-campaign trends, profiles execution bottlenecks, and eliminates C-level edge cases (e.g. prompt interception, keystroke preservation, doorway chokepoint detection).
   - Maintains and indexes the 47 canonical empirical invariants in [`lox/knowledge/invariants.py`](file:///home/moose/git/lox/lox/knowledge/invariants.py) and [`AGENTS.md`](file:///home/moose/git/lox/AGENTS.md).
   - Enforces 100% test suite verification across all unit and tactical integration suites.

---

## 2. How LOX Works: The Scientific Method Generational Lifecycle

Each evolutionary generation in LOX executes a closed-loop 6-stage scientific method lifecycle:

```
[1. Parallel Batch Eval] ──► [2. Trustworthy Triad Autopsy] ──► [3. Hypothesis Formulation]
        ▲                                                                     │
        │                                                                     ▼
[6. Counterfactual Twin Replay] ◄── [5. Multi-Scenario Dry-Run] ◄── [4. Surgical Skill Splicing]
```

### Phase 1: Parallel Batch Evaluation
- 100 real NetHack episodes are dispatched concurrently across 20 CPU workers via `multiprocessing.Pool`.
- Each worker runs up to 25,000 game turns per episode using the active policy checkpoint ([`data/latest_policy.py`](file:///home/moose/git/lox/data/latest_policy.py)).
- Zero-lock buffered Parquet streams log every game tick (hero stats, inventory, spatial coordinates, messages) without disk I/O contention.

### Phase 2: Telemetry Ingestion & The Trustworthy Triad Autopsy
- Worker results are merged into `data/lox.duckdb` via name-matched vectorized SQL.
- Ground-truth scores are extracted directly from NetHack C-level `blstats[nh.NLE_BL_SCORE]`.
- The **Causal Timeline Telemetry Engine** ([`lox/telemetry/diagnostics.py`](file:///home/moose/git/lox/lox/telemetry/diagnostics.py)) synthesizes the **Trustworthy Triad**, eliminating single-label misattribution:
  * **Factor 1: Macro Archetype Distribution**: Full 100-episode batch percentage breakdown across 18 canonical failure archetypes (`EQUIPMENT_NEGLECT`, `PACING_STALL`, `NUTRITION_BLINDNESS`, `TACTICAL_HOARDING`, `MISSED_ARTIFACT`, `COCKATRICE_BLINDNESS`, `PROVOKED_NEUTRAL`, `TRAP_FATALITY`, `DROWNING_OR_LAVA`, etc.).
  * **Factor 2: Orthogonal Systemic Drivers**: Multi-label co-occurrence tracking independent of the fatal blow (e.g. unarmored AC 9 on DL 4, 800+ turns wandering DL 1, unused wands/potions).
  * **Factor 3: Ground-Truth Empirical Telemetry**: Concrete killer names, mortality depth distribution, average AC at death, and 15-tick pre-mortem flight traces.

### Phase 3: Falsifiable Scientific Hypothesis Formulation
- The `AuthorAgent` analyzes the Turn 0 dossier, consults invariants and the offline NetHack encyclopedia (`lox.author.wiki`), and formulates a structured scientific hypothesis in YAML:
  ```yaml
  hypothesis:
    causal_finding: "EQUIPMENT_NEGLECT: 82% of heroes died with unarmored AC >= 6 on DL 3+"
    targeted_skill: "skill_scavenge_armor"
    mechanism: "Prioritize step_to_loot() and wear_armor() over deep descent on DL 1-4"
    predicted_outcome:
      target_metric: "avg_depth"
      expected_direction: "increase"
      min_improvement: 0.50
  ```

### Phase 4: Surgical Modular Skill Splicing
- Rather than risking monolithic whole-program mutations, the LLM outputs only the modified method (e.g. `def skill_scavenge_armor(self, obs): ...`).
- `AuthorAgent.splice_policy_methods()` splices the replacement method into `class Agent` via AST, preserving all other verified modular skills 100% intact.
- The policy AST is strictly validated against `ALLOWED_PREDICATES` and `ALLOWED_ACTIONS` in [`lox/dsl/schema.py`](file:///home/moose/git/lox/lox/dsl/schema.py).

### Phase 5: Multi-Scenario Dry-Run & Loop-Guard Transformation
- The candidate code is passed through `LoopGuardTransformer`, injecting `_guard.tick()` into all loops.
- A multi-scenario dry-run validator tests the policy against simulated states (`normal`, `hungry_unsafe_corpse`, `combat`, `floating_eye_combat`) under a hardware `SIGALRM` 1.0-second ceiling to guarantee that every branch yields an action without infinite loops.

### Phase 6: Counterfactual Twin Replay & Empirical Falsification
- **Counterfactual Seed-Pinned Twin Evaluation (`--twin-test`)**: The candidate policy is re-evaluated against the *exact identical NetHack seeds* that failed previously, measuring paired delta ($\Delta_{\text{paired}} = \text{depth}_{\text{cand}} - \text{depth}_{\text{base}}$), win/loss/draw rates, 95% bootstrap confidence intervals, and incident resolution rate with 0 environment variance.
- If validated ($\Delta_{\text{paired}} \ge +0.40$, improved seeds $\ge 20$, positive CI lower bound), the policy is promoted to [`data/latest_policy.py`](file:///home/moose/git/lox/data/latest_policy.py) and archived to `data/policies/gen_XXXX_validated.py`.
- If falsified, the candidate is discarded, the baseline is preserved, and the falsification drivers are passed to the next generation.
- Complete scientific hypothesis dossiers and counterfactual outcomes are logged to the `meta_experiments` table in DuckDB for 100% research reproducibility.

---

## 3. Core Technical Invariants & Defensive Shields

LOX's reliability is anchored on 47 typed empirical invariants across 6 operational domains (documented in full in [`AGENTS.md`](file:///home/moose/git/lox/AGENTS.md) and [`lox/knowledge/invariants.py`](file:///home/moose/git/lox/lox/knowledge/invariants.py)), alongside 18 canonical failure archetypes in [`lox/telemetry/diagnostics.py`](file:///home/moose/git/lox/lox/telemetry/diagnostics.py):

1. **Python Generator Protocol & Subroutine Yields (`INV-ARC-001`)**:
   Policies yield actions (`obs = yield action`). Subroutines are invoked via `obs = yield from self.subroutine(obs)` and conclude with `return obs`. Every code branch MUST yield an action to prevent zero-yield CPU spins.
2. **Four-Layer Anti-Loop Defense Architecture**:
   - **Layer 1 (AST LoopGuard)**: Raises `RuntimeError` if any loop exceeds 2,000 iterations without yielding.
   - **Layer 2 (PolicyRunner Fallback)**: Catches runtime errors and yields `wait()` (`.`) to advance the game clock safely.
   - **Layer 3 (Episode Wall-Clock Ceiling)**: 90-second worker ceiling breaks out of C-level NLE hangs.
   - **Layer 4 (Batch Ceiling)**: 180-second pool timeout terminates and restarts hung worker processes.
3. **Prompt Auto-Dismissal & Keystroke Preservation**:
   - Auto-confirms safe prompts (`"eat it? [ynq]"` $\to$ `'y'`, `#dip` $\to$ `'y'`) and dismisses text prompts with ESC (`\x1b`).
   - Intermediate multi-key commands (`eat_carried_food`, `wear_armor`, `zap_offensive_wand`, `wish`, `chat_with_leader`) preserve sub-prompts without premature escape.
   - A consecutive zero-turn circuit breaker forces `wait()` after 4 stalled turns, eliminating zero-progress aborts.
4. **Vectorized Glyph Lookup Tables (11x Speedup)**:
   Precomputed 6KB boolean lookup tables classify all 5,976 NetHack glyphs in $< 1\text{µs}$, driving turn throughput to **826+ steps/second**.
5. **Full-Floor Persistent Topological Memory**:
   Preserves discovered walkable terrain and fixed fixtures (stairs up/down, fountains, altars, quest portals) across line-of-sight boundaries.
6. **Dust Elbereth Sanctuary & Species Discrimination (`INV-CMB-001`)**:
   Writing `"Elbereth"` in dust provides a 100% ward against non-humanoids (ants, bees, spiders, wolves). Discriminated immune species (orcs, elves, humans, trolls) trigger corridor retreats or melee.
7. **Passive Hazard Discrimination & Buffer Navigation (`INV-CMB-002`)**:
   Floating eyes and gas spores are masked out of navigation pathfinding. Pathfinding routes around them at distance $\ge 2$. Distance-1 projectile throwing eliminates floating eyes safely without passive paralysis.
8. **Proactive Nutrition & Rotten Corpse Shield (`INV-NUT-001`)**:
   Carried food rations are eaten proactively at `hunger_state >= 2` ("Weak") when away from hostiles or on Elbereth, preventing fainting blackouts. Rotten corpses are strictly filtered out of food slots.
9. **Excalibur Artifact Scaling (`INV-EQP-004`)**:
   Lawful Valkyries navigate to fountains at XL $\ge 5$ to dip long swords, forging Excalibur (+1d10 damage, secret door detection, level drain immunity).
10. **Safe Divine Favor Threshold (`INV-NUT-003`)**:
    Enforces `turn - last_prayer_turn >= 850` turns and gates hunger prayer strictly to `hunger_state >= 3` ("Weak"), guaranteeing safe divine feeding or full HP recovery without angering the deity.
11. **Dead-End Search Persistence (`INV-NAV-005`)**:
    Searches 12–15 times at dead ends before moving to the next candidate tile, raising secret door discovery probability to $>91\%$ and eliminating exploration deadlocks.
12. **Endgame Ascension Primitives (`INV-END-001` through `INV-END-007`)**:
    Full action vocabulary and solver support for beating NetHack:
    - `wish`: Safe wishing for speed boots, gray dragon scale mail, or wand of death.
    - `chat_with_leader`: Quest leader dialogue to unlock the Quest nemesis branch.
    - `step_to_quest_portal` & `step_to_plane_portal`: Inter-branch navigation.
    - `stash_in_bag`: Cursed item and weight management via Bag of Holding.
    - `offer_amulet_on_altar`: Offering the real Amulet of Yendor on the matched Astral high altar to complete Ascension.

---

## 4. Empirical Milestones & Benchmark History

LOX has evaluated **12,800+ episodes** across **38.0M+ game turns** in DuckDB:

| Metric | Campaign 1 Baseline | Mid-Campaigns (C10–C20) | Current Highs (C28–C32) | Target (Ascension) |
| :--- | :---: | :---: | :---: | :---: |
| **Batch Avg Depth** | 1.91 | 3.51 – 3.85 | **5.10** (C27) / **4.80** (C30) / **4.65** (C28, C29) | **$\ge 20.0$** |
| **Peak Single Depth** | 5 | 9 – 11 | **15** (C32) / **14** (C25) / **12** (C28, C30) | **45 – 50** (Astral Plane) |
| **Batch Avg Turns** | ~1,200 | ~3,500 | **8,096.4** (C31 Gen 9) / **7,374.0** (C29 Gen 9) | **25,000** |
| **Peak Single Score** | 922 | 2,645 | **4,612** (C24) / **4,551** (C28) / **4,368** (C29) | **$\ge 50,000$** |
| **Artifact Scaling** | 0 Excalibur | 0 Excalibur | **Autonomous Excalibur Forged** (`g005_e006`, `g008_e013`) | **Excalibur + Bell + Candelabrum + Book** |
| **Turn Step Speed** | ~75 steps/s | ~75 steps/s | **826+ steps/s** (11x Vectorized LUT Speedup) | **>1,000 steps/s** |
| **Batch Wall-Clock** | ~270s | ~180s | **~35s** (20 workers concurrent) | **~30s** |
| **Batch Spend** | N/A | ~$0.04 | **~$0.027** per 100-episode batch | **~$0.03** |
| **Starvation Mortality**| ~28% | ~12% | **0.0%** (Completely Eliminated) | **0.0%** |
| **Zero-Progress Aborts**| ~18% | ~4% | **0.0%** (Completely Eliminated) | **0.0%** |

---

## 5. Quickstart: Run in 2 Minutes

### Prerequisites
- **OS**: Linux (Ubuntu 20.04+, Debian, Arch, Fedora) or macOS.
- **Python**: `>= 3.11`.
- **System Packages** (for NetHack C bindings):
  - Ubuntu/Debian: `sudo apt update && sudo apt install -y build-essential cmake bison flex libz-dev`
  - macOS: `brew install cmake bison flex`
- **uv**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.cargo/env
  ```

### Step 1: Clone and Install
```bash
git clone https://github.com/saejin-moon/lox.git
cd lox
uv sync
```

### Step 2: Verify Installation (Run Unit Tests)
```bash
uv run pytest -v
```
All **159 test items** across 25 test suites pass cleanly in ~4.4 seconds.

### Step 3: Run Synthesis

#### Overnight Automated Run (Recommended)
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

# Launch detached tmux session running the full scientific synthesis loop
./scripts/run_overnight.sh --fresh
```

#### Manual OpenRouter Campaign (Targeting Depth 20.0 / Ascension)
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

uv run python -u -m scripts.run_synthesis \
  --provider openrouter \
  --model qwen/qwen3.5-9b \
  --generations 20 \
  --eval-episodes 100 \
  --max-turns 25000 \
  --target-depth 50.0 \
  --min-delta 0.40 \
  --twin-test \
  --workers 20 \
  --starter-policy data/modular_starter_policy.py \
  --policy-path data/latest_policy.py
```

#### Offline Mock Mode (Zero Setup, No API Keys)
```bash
uv run python -u -m scripts.run_synthesis \
  --provider mock \
  --generations 3 \
  --eval-episodes 5 \
  --max-turns 500
```

---

## 6. Inspecting Campaign Telemetry with DuckDB & YAML

Every tick, episode, scientific hypothesis, and LLM token usage is recorded in `data/lox.duckdb`, with pure YAML summaries automatically exported to `data/latest_diagnostics.yaml` and `data/latest_report.yaml`:

### View Latest Generation Failure Breakdown (YAML)
```bash
cat data/latest_diagnostics.yaml
```

### Inspect Scientific Hypotheses & Validation Outcomes
```bash
uv run python -c "
from lox.telemetry.consolidator import safe_duckdb_connect
con = safe_duckdb_connect('data/lox.duckdb', read_only=True)
rows = con.execute('''
    SELECT generation, targeted_skill, causal_finding, baseline_avg_depth, actual_avg_depth, outcome_validated
    FROM meta_experiments
    ORDER BY generation DESC
    LIMIT 10
''').fetchall()
for r in rows:
    print(r)
con.close()
"
```

### Check Generation Performance Summary
```bash
uv run python -c "
from lox.telemetry.consolidator import safe_duckdb_connect
con = safe_duckdb_connect('data/lox.duckdb', read_only=True)
rows = con.execute('''
    SELECT run_id, count(*) as eps, round(avg(max_depth), 2) as avg_depth, max(max_depth) as max_depth, round(avg(score), 1) as avg_score, max(score) as peak_score
    FROM episodes
    GROUP BY run_id
    ORDER BY min(episode_id) DESC
    LIMIT 10
''').fetchall()
for r in rows:
    print(r)
con.close()
"
```

### Inspect Recent Episode Fatalities & Root Causes
```bash
uv run python -c "
from lox.telemetry.consolidator import safe_duckdb_connect
con = safe_duckdb_connect('data/lox.duckdb', read_only=True)
rows = con.execute('''
    SELECT episode_id, depth, max_depth, turns, root_cause, death_reason, ac_at_death
    FROM episodes
    ORDER BY episode_id DESC
    LIMIT 10
''').fetchall()
for r in rows:
    print(r)
con.close()
"
```

### Audit LLM Token Usage & Campaign Costs
```bash
uv run python -c "
from lox.telemetry.consolidator import safe_duckdb_connect
con = safe_duckdb_connect('data/lox.duckdb', read_only=True)
print(con.execute('''
    SELECT model, count(*) as calls, sum(total_tokens) as total_toks, round(sum(estimated_cost_usd), 4) as cost_usd
    FROM token_usage
    GROUP BY model
''').fetchall())
con.close()
"
```

---

## 7. Modular Test Suite & Verification

The test suite is structured into typed subpackages matching `lox/`:

```
tests/
├── conftest.py             # Shared fixtures (dummy_obs, dummy_hero_state, auto-markers)
├── unit/                   # Fast in-memory tests (<1ms each, no Gym or external dependencies)
│   ├── core/               # Spatial A*, BehaviorTree nodes, Epistemic belief, Agenda
│   ├── dsl/                # AST compiler, parser whitelist, infinite loop protection
│   ├── telemetry/          # Causal diagnostics, DuckDB consolidation, token accounting
│   ├── knowledge/          # 47-invariant registry, offline wiki engine
│   └── author/             # LLM ReAct loop, AST method splicing, tools
└── integration/            # Multi-component and environment tests
    ├── envs/               # NetHackAdapter, tactics, solvers, Sokoban push solver
    └── synthesis/          # Counterfactual seed twin replay, pure synthesis contract
```

### Pytest Execution Modes

Run fast in-memory unit tests in **<0.6s**:
```bash
uv run pytest -m unit
```

Run non-gym integration and unit tests:
```bash
uv run pytest -m "not gym"
```

Run the entire 166-test suite:
```bash
uv run pytest -v
```

---

## 8. Key Operational Documents

- **[`HANDOFF.md`](file:///home/moose/git/lox/HANDOFF.md)**: Master onboarding manual for LLM agents taking over the codebase (mission targets, policy generator rules, diagnostic engine, and ascension roadmap).
- **[`AGENTS.md`](file:///home/moose/git/lox/AGENTS.md)**: Authoritative operational guide, architectural foundations, and 47 consolidated technical invariants across 6 domains.
- **[`TODO.md`](file:///home/moose/git/lox/TODO.md)**: Real-time campaign tracking, empirical benchmark progression, autopsy findings, and active roadmap.
