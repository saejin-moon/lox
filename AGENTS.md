# AGENTS.md: Autonomous Agent Onboarding & Master Knowledge Base (Gen-2)

> **Welcome, Agent.** This file is the single source of truth for **LOX-ψ (LLM-Oriented Creation of Symbolic policies), instantiated as LOX-ψ**.
> It contains the architectural contracts, NetHack 3.6.6 ground-truth domain rules, verified test suites, and
> the phased roadmap. Read this file completely before making changes.

---

## 1. Mission & Non-Negotiable Constraints

### The Core Objective

Build a **policy synthesis engine**: a safety-gated loop in which an LLM author **grows a declarative policy
program** (macros, rules, goals, handlers) from trial data, environment telemetry, and RAG corpora — until the
resulting agent **decisively beats AutoAscend on NetHack (NLE 3.6.6 / `NetHackChallenge-v0`)** and the
strongest published equivalents on transfer domains (MiniHack curricula, Craftax).

The human role is deliberately narrow: **primitives, interlocks (guardrails), certification gates.**
Everything strategic — priorities, thresholds, tactics, compositions, and eventually whole goal handlers —
is authored by the LLM through a grammar-constrained diff language, validated by machine-checkable gates,
and measured by parallel episode evaluation.

**We do not copy expert HTNs (e.g., AutoAscend).** The contribution is that grown knowledge beats engineered
knowledge. Porting an expert HTN would falsify the claim.

### STRICT Operational Constraints

1. **PURE CPU SYMBOLIC EXECUTION ONLY**:
   - ZERO GPU / zero CUDA contexts. The user trains a MARL environment on the local GPU. All execution
     (A\*, HTN, epistemics, telemetry, evolution) is CPU-only. The LLM author runs via **API** (primary) or
     llama.cpp (repro path, run on demand — it is the only permitted CPU-heavy non-executor process).
2. **ENVIRONMENT INVOCATION**: always `uv run` for python/pytest.
3. **100% REGRESSION-FREE TEST SUITE**: `uv run pytest` — currently **572 passing tests (~17s)**. Every
   commit maintains the full count. Never disable or skip tests to mask errors.
4. **CLEAN TELEMETRY**: parquet (`logs/parquet/`) must be consolidated into
   `data/lox_telemetry.duckdb` (7.2M+ ticks) via `scripts/clean_telemetry.py` — never let raw parquet
   accumulate. During evolution campaigns, consolidate at campaign boundaries, not per batch.
5. **EVERY PROGRAM VERSION IS HOLY**: the LLM may only change the world through validated diffs; the
   validator (`lox/policy/validator.py`) is the only writer of accepted programs. A rejected diff must
   never leak into a live program. All accept/reject events go to `data/revision_ledger.jsonl`.

---

## 2. Architecture (Gen-2)

```
┌────────────────────────────────────────────────────────────────────────────┐
│ AUTHOR AGENT (LLM, agentic — tool-calling loop)                            │
│   tools: query_duckdb(sql) · wiki_search(q) · read_trajectory(ep, turns)   │
│          read_env_schema() · read_manifest() · read_program_tree()         │
│   input: full run telemetry (6.6M ticks, episodes, goal_events, deaths,    │
│          stalls, failure counters) + wiki FTS + trajectory autopsies       │
│   output: S-EXPRESSION POLICY DIFF (grammar-constrained, depth ≤ 3)        │
└──────────────┬─────────────────────────────────────────────────────────────┘
               ▼ VALIDATOR GATES (machine-checkable, no human in the loop)
   grammar → schema → bounds → target-conditionality → macro closure/expansion
   → interlock invariants → handler-plan schema → certification → tiered batch
┌────────────────────────────────────────────────────────────────────────────┐
│ POLICY PROGRAM — per-file authoring tree (the LLM's workspace)             │
│   data/program/<domain>/                                                   │
│     params.json            # typed bounded tunables                        │
│     macros/*.sexpr         # named, parameterized condition sugar          │
│     rules/*.sexpr          # tactic rules (target-conditional)             │
│     goals/*.sexpr          # strategy_plan entries (ordered)               │
│     handlers/*.sexpr       # declarative goal-handler sub-programs         │
│     nogoods.jsonl          # cross-episode CDCL negatives                  │
│   compiled by lox/policy/compiler.py → data/compiled/<domain>.json        │
│   (the executor consumes ONLY the compiled artifact — contract unchanged)  │
└──────────────┬─────────────────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ EVOLUTION ENGINE (scripts/run_evolution.py)                                │
│   population of K candidate programs → parallel gated evaluation           │
│   (scripts/run_parallel_batch.py, 30+ workers) → gated selection, lineage  │
│   tracked in the ledger; losers archived, winners become the new base      │
└──────────────┬─────────────────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ SYMBOLIC EXECUTOR CORE (CPU-only, stable, env-agnostic via DomainAdapter)  │
│   GoalInterpreter → HTN planner ← Domain Managers ← A*/epistemics          │
│   adapters: nethack · minihack · craftax                                   │
└────────────────────────────────────────────────────────────────────────────┘
```

### Division of authority (the safety + novelty claim)

- The LLM expresses **what to value and when**: goals, priorities, thresholds, tactics, compositions, and
  (via the handler ladder) declarative sub-programs over certified primitives. It can **never** express *how
  to act at the mechanism level*: no code emission, no new mechanisms, no interlock removal. Every extension
  is definitional over verified primitives — a property the validator *proves*, enabling unattended operation.
- Humans (and human-written executor code) own the mechanisms: pathfinding, combat mechanics, prompt/menu
  handling, and the interlock set (§4).
- The LLM sees **everything**: full environment schema (every action, every observation channel), full run
  telemetry via SQL, trajectories, and the wiki. Access is read-only; the diff channel is the only write path.

### Component map

```
lox/
├── policy/                      # the LLM-facing surface
│   ├── compiler.py              # NEW(S1): authoring tree → compiled program (the only writer)
│   ├── author_agent.py          # NEW(S0): agentic tool-calling author (SQL/wiki/trajectory tools)
│   ├── prompts/                 # NEW(S0): versioned system-prompt templates (data/prompts/)
│   ├── config.py                # typed bounded tunables (the ISA's numeric surface)
│   ├── program.py               # compiled-program container + overlay logic
│   ├── dsl.py / macros.py       # S-expr diff reader · defmacro (parameterized in S3)
│   ├── manifest.py              # machine-generated vocabulary: predicates/verbs/goals/actions
│   ├── validator.py             # full gate pipeline (the only program writer)
│   ├── tactics.py / predicates.py / profiles.py / goal_interpreter.py / goal_state.py
│   ├── corpus.py                # per-domain FTS5 RAG corpora (data/corpus/<domain>/)
│   ├── ledger.py / report.py    # provenance ledger · run-report bundles
│   └── grammar/nethack.sexpr.gbnf
├── evolution/                   # NEW(S2): population manager, selection, lineage
├── executor/                    # DomainAdapter ABC + nethack/minihack/craftax adapters
├── agent/                       # lox_agent core + htn_methods + episode_runner + episode
├── domain/                      # managers: combat(±mixins) · navigation(±mixins) · inventory · shop …
│   ├── combat/{threat_scan,responses}.py · navigation/{level_map,stepping,exploration}.py
├── deliberative/                # providers (gemini/openrouter/llama_cpp/mock) · deadlock resolver
├── env/ · navigation/ · planner/ · epistemic/ · telemetry/ · workers/
scripts/
├── run_parallel_batch.py        # multiprocessing batch runner (30 workers; 100 eps ≈ 3 min)
├── run_evolution.py             # NEW(S2): the population/selection campaign driver
├── run_curriculum.py            # NEW(S4): per-skill training-gauntlet driver
├── run_prompt_ab.py             # NEW(S0): system-prompt A/B harness
├── run_author_session.py        # NEW(S0): one agentic author session (tools → diff → gates)
├── run_benchmark.py · run_ablation.py · run_revision_loop.py · run_skill_certifications.py …
data/
├── program/<domain>/            # authoring tree (above) — the LLM's workspace
├── compiled/<domain>.json       # canonical compiled artifact (executor contract)
├── programs/archive/            # every version, forever (provenance)
├── prompts/                     # versioned prompt templates + A/B results
├── corpus/<domain>/             # FTS5 RAG corpora (nethack wiki · minihack · craftax self-built)
├── lox_telemetry.duckdb        # 6.6M+ ticks, episodes, goal_events (read-only to the author)
└── revision_ledger.jsonl        # append-only accept/reject/cost/lineage record
```

---

## 3. Scaling: Episode Throughput (the engine's fuel)

- Hardware: **36 cores / 31 GB RAM**. NLE is process-safe, single-threaded per env → workers scale linearly.
- Campaigns pass `--jobs 30` explicitly (the script default is a conservative 8): **NetHack ≈ 5–6k eps/hr**
  at 20k-step caps; MiniHack/Craftax curricula (short episodes) **10–50k eps/hr**.
- Campaign discipline: consolidate DuckDB at campaign boundaries; keep per-worker parquet partitions.
- Leave 4–6 cores headroom (OS, consolidation, on-demand llama.cpp).
- Curriculum episodes use per-phase step budgets (most deaths occur < 6k turns; survivors of the budget
  continue in a follow-up batch — do not burn 20k steps on every curriculum episode).

---

## 4. NetHack 3.6.6 Ground-Truth Domain Rules & Hardcoded Interlocks (GUARDRAILS — keep intact)

These are the **human-owned guardrails** the LLM can never disable. They exist because each one corresponds
to a measured fatality mode. When adding behavior, obey them; when the data proves one wrong, change it via
a human-reviewed commit — never via an LLM diff.

1. **Corpse freshness**: mortal corpses safe ≤ 25 turns from witnessed kill; pre-existing level corpses are
   rotten (instant death); lichen corpses never rot (200 nutrition forever); non-corpse food never rots.
   Never eat floor corpses with adjacent hostiles (eating takes 5–15 turns = free hits). `AutoMoreWrapper`
   intercepts rotten/stoning eat prompts → `N`.
2. **Non-pet monster tiles are unwalkable** (bump = melee; domestic/peaceful retaliation). A* penalizes
   monster tiles (+1000). No intentional attack unless the combat manager issued it.
3. **Shop doors on Depth ≥ 2 are NEVER kicked** (shopkeeper wand death). Unlock tools first, then route
   around; the door is marked unwalkable.
4. **Multi-stairs steering**: DL 2–4 has two downstairs (Mines vs Dungeons). Under-leveled (XL < 5/6) agents
   route to the Main Dungeons staircase; trapped in the Mines under-leveled → ASCEND out.
5. **Corridor search stall elimination**: no secret-door search on corridor tiles; dead ends ≤ 6 searches,
   perimeter walls ≤ 5; message-based stair detection for obscured stairs.
6. **Emergency health triage (P0.5)**: quaff full/extra/healing (uncursed+) at HP ≤ 55% (60% with hostiles);
   teleport scroll ≤ 30% if cornered; `#pray` ≤ 25% when `can_safely_pray`.
7. **Floating eye melee lockout** (paralysis 0–70 turns) unless blind/reflection; missiles/wands only.
8. **Grid bug diagonal exploit**: free hits from diagonals; orthogonal → step to diagonal alignment.
9. **Mid-game stack**: Excalibur dipping (lawful, XL ≥ 5, uncursed long sword + fountain); unpaid-item
   debt relief (PAY or drop before leaving); Sokoban progression (boulder pushes, reflection prize, ascend
   floors 1–3 → floor 4 prize); Castle drawbridge via wand of striking, wand-of-wishing protocol.
10. **Menu/pager dismissal**: `(X of Y)`/`(end)` menus → SPACE/ESC via `AutoMoreWrapper`; `#enhance` fibers
    terminate with ESC. Guards against `step on finished NetHack`.
11. **XP index alignment**: `blstats[18]` = XL, `blstats[19]` = EXP. Never read 19 as level.
12. **Locked doors DL ≥ 2**: unwalkable + never kicked when alternatives exist; `stairs_up` anchored to
    arrival tile.
13. **Heavy hitters** (ogre/soldier ant/rothe/…): no melee trading when wounded (kite/retreat/elbereth).
    Failed steps (2×) prune `walkable` to kill motion loops.
14. **Starvation**: `#pray` at WEAK+ (no food) when safe — resets nutrition to 900. Descent is never blocked
    by hunger. **A/B-proven negative result (2026-09-19)**: food-chasing at ANY radius is a net negative
    (none=396 > tight=275 > wide=250 mean score; faints RISE with chase capability) — the chase stays
    disabled; hunger is solved by descent-to-fresh-kills + carried food + prayer. The loop may never
    re-enable it (locked policy, `tests/test_food_security.py`).
15. **Diagonal door alignment** excludes blocked/shop-door tiles (kills 3-tile oscillation loops).
16. **Search-burst damage abort**: HP drop or attack keywords abort the burst; hallucination (condition 512)
    freezes vectorized map updates.

---

## 5. Priority Hierarchy (executor runtime)

```
P0.0 Emergency nutrition (weak/fainting) → P0.5 Health triage → P1.0 Corridor funnel/Elbereth
→ P2.0 Tactical combat → P2.5 Skill enhance → P3.0 Shop/temple → P3.5 Epistemic (BUC/price-ID)
→ P4.0 Inventory/equipment → P5.0 Exploration & descent   [goal-interpreter directives sit ABOVE]
```

The Gen-2 program (S1) re-expresses the *strategy* content of this cascade as the default policy program;
the mechanical guardrails (§4) stay in code.

---

## 6. Telemetry & Verification

- `data/lox_telemetry.duckdb`: `episodes`, `ticks` (7.2M+), `goal_events`, views `v_eval_summary`,
  `v_lethal_taxonomy`. The author's `query_duckdb` tool reads this read-only.
- Test suite: `uv run pytest` → **572 passed (~17s)**. Flake protocol: one transient failure → rerun once
  before diagnosing (combat-manager suites are timing-sensitive); never skip. Includes: R6 milestone
  certifications (18), target-conditionality invariant tests, food-security policy locks (6), validator
  gates, combat/navigation mixins, loop tests, epistemic, telemetry, adapters, Numba equivalents, infix parser,
  and token optimization domain isolation tests (4).
- Certifications: `scripts/run_skill_certifications.py` (4 suites, incl. the R6 Early Survival Gauntlet) —
  all CERTIFIED. Per-domain adapter certification suites gate handler promotion.
- Every campaign appends to `data/revision_ledger.jsonl` (author, tokens, cost, latency, accept/reject,
  measured deltas, lineage parent). Losers are archived, never deleted.

---

## 7. Developer Command Cheat Sheet

| Task | Command |
| :--- | :--- |
| **Unit tests** | `uv run pytest` |
| **Parallel batch (30 workers)** | `uv run python scripts/run_parallel_batch.py --role valkyrie --episodes 100 --max-steps 20000 --jobs 30 --output data/batch.json` |
| **MiniHack parallel batch** | `uv run python scripts/run_minihack_batch.py --task MiniHack-ExploreMaze-Easy-Mapped-v0 --episodes 30 --jobs 10` |
| **One agentic author session** | `uv run python scripts/run_author_session.py --domain nethack --author gemini` |
| **Evolution campaign** | `uv run python scripts/run_evolution.py --population 4 --episodes 100 --hours 8` |
| **Prompt A/B** | `uv run python scripts/run_prompt_ab.py --candidates data/prompts/v3.md,v4.md --batches 3` |
| **Curriculum eval** | `uv run python scripts/run_curriculum.py --skill retreat_discipline --episodes 200` |
| **Benchmark** | `uv run python scripts/run_benchmark.py --role valkyrie --episodes 10 --max-steps 20000 --clean-parquet` |
| **Query telemetry** | `uv run python scripts/query_duckdb.py "SELECT …"` |
| **Wiki search** | `uv run python scripts/wiki_search.py "<query>"` |
| **Clean parquet** | `uv run python scripts/clean_telemetry.py` |
| **Ablation (3-arm)** | `uv run python scripts/run_ablation.py --arms llm,frozen,random --episodes 100 --domain all` |
| **Baseline suite** | `uv run python scripts/run_baseline_suite.py --episodes 100 --step-limit 50000 --role val` |

---

## 8. The Dual-Domain Battleground, Acceleration & Scientific Rigor

### 8.1 Beating AutoAscend: The Mechanism vs. Strategy Contract
AutoAscend (median depth 10, mean score 10.7k, 4.8% ascension) is an expert system with ~2 person-years of handcrafted rules.
- **The Boundary Principle**: The LLM author synthesizes **when, what, and how much to value**. It does *not* invent algorithms out of nothing.
- To beat AutoAscend, the underlying symbolic engine must provide the complete **mechanical primitive set**:
  1. *Deterministic Sokoban Push Solver*: Guarantees Bag of Holding or Reflection prize (eliminating Sokoban attrition).
  2. *Shop Price-ID State Machine*: Tests item buy/sell rates to identify scrolls, potions, wands, and rings safely.
  3. *Safe Fountain Dipping*: Controls uncursed long sword dipping for Excalibur at XL ≥ 5 without drowning/nymph death.
  4. *Tactical Corridor Funneling*: Strict retreat-to-doorway behavior against fast/poison biters (soldier ants, bees, spiders).
  5. *Endgame Progression*: Medusa blindfold/mirror, Castle drawbridge striking, wand of wishing, Vlad, Rodney, Invocation.
- Once these primitives exist, LOX-ψ's **evolutionary synthesis across millions of ticks systematically out-optimizes AutoAscend's static, brittle human thresholds**.

### 8.2 Dominating Craftax: Solving JAX CPU Overhead
Craftax (ICML 2024) is a fast open-world roguelike benchmark in JAX with 67 achievements.
- **JAX CPU Dispatch Law**: Single-step Python-JAX interaction on CPU incurs ~17 ms/step dispatch latency.
  To sustain 10k–50k eps/hr, Craftax evaluation MUST step in vectorized batches (`CraftaxBatchWrapper` using `jax.vmap`) or compile the episode loop with `jax.lax.scan`.
- **Domain Scaffolding**: `CraftaxAdapter` decodes the 8268-float observation into player stats, inventory DAG, and local 2D grid. The LLM then authors over clean Craftax primitives (`mine`, `craft`, `eat`, `drink`, `attack`, `sleep`) and goals (`gather_wood`, `craft_pickaxe`, `mine_stone`, `survive_night`).
- **Baseline Targets**: Beat published RL baselines (PPO: ~15% on Classic, <5% on Full; PPO+EARS: ~20%) and LLM baselines (CALM: ~35% on Classic). Symbolic HTN planning over deterministic crafting DAGs is structurally favored to crush these baselines.

### 8.3 High-Performance Acceleration (Numba / Rust)
Profiled mean decision latency is dropping toward **<150 µs**:
- **Grid A\* (`astar.py`), Frontier BFS (`frontier.py`), & Distance Grid (`exploration.py`)**: Accelerated with `@njit(fastmath=True, nogil=True)` (Numba). Slashes pathfinding and exploration BFS latency from ~350 µs to ~2–15 µs (30x speedup).
- **Line-of-Sight Raycasting (`stepping.py`)**: Accelerated with `@njit` Bresenham raycasting (<0.5 µs).
- **Monster Threat Scanning (`threat_scan.py`)**: Precomputed static 512-entry NumPy LUT computed at import. Slashes scan latency from ~250 µs to 15 µs.
- **Pythonic Infix AST (`infix.py`)**: Sandboxed, depth-bounded condition parser with zero LLM parenthesis hallucination.
- **Navigation Monolith Modularization**: Split `navigation_manager.py` into `stair_routing.py`, `search_policy.py`, and `feature_navigation.py` to eliminate perimeter wall search loops causing DL1 stalls.

### 8.4 Solo-Author Scientific Integrity & Audit Standard
To make this paper airtight for top-tier venues (NeurIPS / ICML / Nature MI):
- **Clean-Room Guarantee**: Zero lines of AutoAscend code, tables, or heuristic values are read or ported. Provenance is public wiki documentation + DuckDB telemetry.
- **The Mandatory 3-Arm Ablation**: Every major claim requires `LLM-Grown Policy > Frozen Initial Policy > Random Mutation Policy`.
- **Statistical Discipline**: NetHack reports require 100-episode evaluation batches with bootstrap confidence intervals; transfer domains use fixed seeds.
- **Ledger Transparency**: Every single prompt, token count, cost, rejected diff, and accepted version is committed to `data/revision_ledger.jsonl`.

