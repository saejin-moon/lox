# LOX Agent Handoff Guide (`HANDOFF.md`)

Welcome to LOX. This document is the comprehensive onboarding guide for any LLM agent or engineer taking over the LOX codebase. It provides full context on the mission, system architecture, policy generator paradigm, diagnostic engine, operational workflows, and the empirical roadmap to Dungeon Depth 20.

---

## 1. Mission & Campaign Targets

- **Primary Goal**: Synthesize an empirical policy that achieves **Average Dungeon Depth $\ge 10.0$** (intermediate milestone) and **$\ge 20.0$** (campaign completion) over a batch of 20 real NetHack episodes.
- **Current Milestone**: Reached **Dungeon Level 13** in Campaign 51. Average batch depth hovers between 3.5 and 4.1.
- **LLM Provider**: OpenRouter (`--provider openrouter`).
- **Target Model**: `google/gemma-4-31b-it`.
- **Episode Limits**: 25,000 game turns per episode (`--max-turns 25000`).
- **Batch Size**: 20 evaluation episodes per generation (`--eval-episodes 20`).
- **Parallel Workers**: 20 concurrent worker processes (`--workers 20`).
- **Cost Efficiency**: Entire 10-generation campaigns (200 episodes, 400k+ LLM tokens) execute for **~$0.05–$0.06 USD**.
- **Stop Condition**: When a generation batch achieves `avg_depth >= 10.0` (or `20.0`), the loop logs `[CAMPAIGN GOAL ACHIEVED]` and finishes.

---

## 2. Codebase Architecture & File Map

```
lox/
├── author/
│   ├── agent.py          # AuthorAgent: LLM ReAct loop, AST validation, multi-scenario dry-run validator
│   ├── prompts.py        # System and iteration prompts with vocabulary and Invariant registry injection
│   ├── tools.py          # DuckDBToolRegistry: safe SQL query tools, wiki search, and invariant lookup (YAML output)
│   └── requests.py       # Macro request queue (data/macro_requests.md)
├── core/
│   ├── types.py          # Observation, HeroState, CombatView, SpatialView, DungeonView, InventoryView, Action
│   ├── spatial.py        # SpatialEngine: Numba JIT A* pathfinding, frontier discovery, distance grids
│   ├── digging.py        # Straight-line Gehennom maze digging router
│   ├── sokoban.py        # Sokoban boulder push state graph solver
│   └── tree.py           # AST-to-BT nodes (Selector, Sequence, Condition, ActionNode, Blackboard)
├── dsl/
│   ├── schema.py         # ALLOWED_PREDICATES, ALLOWED_ACTIONS, ENUM_CONSTANTS vocabulary whitelist
│   ├── parser.py         # AST validator (no imports, no exec/eval, whitelisted nodes)
│   └── compiler.py       # Compiles class-based generator policies (Agent.run(obs)) and AST trees
├── envs/
│   ├── base.py           # EnvironmentAdapter base class
│   ├── nethack.py        # NetHackAdapter: NLE gym wrapper, menu dismissal, navigation & tactical primitives
│   └── solvers/          # High-level procedural solvers (Castle, Invocation ritual)
├── eval/
│   └── runner.py         # Batch evaluation engine, DuckDB episode and tick telemetry logger
├── knowledge/
│   └── invariants.py     # Canonical Empirical Knowledge Base: 32 typed, indexed invariants (REGISTRY)
├── telemetry/
│   ├── consolidator.py   # DuckDB consolidation from Parquet buffers & schema evolution
│   ├── diagnostics.py    # Empirical Root Cause Diagnostic Engine (9 canonical archetypes, YAML export)
│   ├── parquet.py        # High-throughput pyarrow Parquet telemetry logger
│   ├── recorder.py       # Circular 100-turn in-memory flight recorder (YAML format)
│   ├── tokens.py         # LLM token usage tracking & cost accounting
│   └── triggers.py       # Dynamic incident triggers (stalls, cluster mortalities)
├── wiki/
│   └── engine.py         # Offline NetHack 3.6.6 encyclopedia retrieval engine (BM25 search)
scripts/
└── run_synthesis.py       # Main outer loop: evaluates batches, classifies root causes, invokes AuthorAgent
data/
├── latest_policy.py       # Current production policy checkpoint
├── latest_report.yaml     # Pure YAML report of the latest generation batch
├── latest_diagnostics.yaml# Pure YAML summary of root cause failure archetypes
└── policies/              # Historical archive: gen_0001.py through gen_XXXX.py
```

---

## 3. The Policy Generator Paradigm (`class Agent`)

Policies are synthesized as standard Python generator classes. Every turn, `run(self, obs)` receives an `obs` object and yields an action (`obs = yield action`).

```python
class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0

    def run(self, obs):
        while True:
            # 1. Absolute Emergency Survival (Major Trouble: Weak without food or Critical HP)
            if obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 3 and not obs.inventory.has_food):
                if obs.hero.turn - self.last_prayer_turn >= 850:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue

            # 2. Immediate Staircase Descent (1-turn instant escape from danger!)
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = yield descend()
                continue

            # 3. Combat Logic
            if obs.combat.adjacent_hostile or obs.combat.has_active_hostile:
                obs = yield from self.handle_combat(obs)
                continue

            # 4. Proactive Nutrition
            if any(c.is_safe for c in obs.corpses) and obs.hero.hunger_state >= 1:
                obs = yield from self.handle_corpse_consumption(obs)
                continue
            if obs.hero.hunger_state >= 1 and obs.inventory.has_food:
                obs = yield eat_carried_food()
                continue

            # 5. Equipment Optimization (Body armor when unarmored, or auxiliary armor)
            if (obs.inventory.has_unworn_body_armor and not obs.inventory.has_worn_body_armor) or (obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor):
                obs = yield wear_armor()
                continue

            # 6. Navigation & Exploration
            if obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            else:
                obs = yield from self.handle_dead_end(obs)
```

### Critical Generator Subroutine Rules (`INV-ARC-001`)
1. **Subroutine Protocol**: Helper methods must be invoked via `obs = yield from self.subroutine(obs)` and MUST conclude with `return obs`. Never return a boolean (`return True`/`return False`) from a generator subroutine! Returning a boolean overwrites `obs = True`, crashing the policy on the next turn.
2. **The Zero-Yield Busy Loop Trap**: Every subroutine called via `yield from` MUST yield an action before returning, OR the caller must avoid `continue` without yielding. Returning without yielding followed by `continue` spins the CPU at 100% without advancing the game turn.
3. **Four-Layer Anti-Loop Defense Architecture**:
   - **Layer 1 (AST `LoopGuardTransformer`)**: Instruments loops with `_guard.tick()`. Exceeding 2,000 iterations without yielding raises `RuntimeError("Infinite loop detected")`.
   - **Layer 2 (`PolicyRunner` Fallback)**: Catches `RuntimeError` and yields `Action(name="wait")` (`.`) to advance the clock.
   - **Layer 3 (Episode Wall Ceiling)**: `MAX_EPISODE_WALL_SEC = 90.0` prevents C-level hangs.
   - **Layer 4 (Pool Ceiling)**: Worker pool batch ceiling terminates frozen processes cleanly.
4. **Targeted Method Splicing**: The Author Agent can output either the full class or ONLY specific modified methods (e.g. `def handle_combat(self, obs): ...`). The synthesis engine splices updated methods into the existing AST while keeping unchanged subroutines intact.

---

## 4. Empirical Root Cause Diagnostic Engine (`lox/telemetry/diagnostics.py`)

Prior to Campaign 51, DuckDB telemetry reported that **100% of fatalities were due to "Killed in combat"**, because any death at HP $\le 0$ was attributed to the last attacking monster. This misled the LLM into endlessly tweaking combat thresholds while ignoring systemic causes like exploration stalls and unarmored runs.

The Diagnostic Engine categorizes every run into **9 mutually exclusive failure archetypes** prioritized by causal precedence:

| Archetype | Ground-Truth Definition | Causal Precedence |
| :--- | :--- | :---: |
| `STARVATION_FAINTING` | Experienced $\ge 1$ turn in `FAINTING` state prior to death; killed while asleep. | **1** |
| `STALL_SECRET_DOOR` | Spent $> 500$ turns on DL 1–2 searching dead ends without finding stairs. | **2** |
| `PING_PONG_OSCILLATION` | Oscillated between $\le 2$ adjacent tiles for $\ge 30$ consecutive turns. | **3** |
| `ARMOR_DEFICIT` | Died at Depth $\ge 4$ with $\text{AC} \ge 5$ while carrying unworn armor or starting unarmored. | **4** |
| `PREMATURE_PRAYER` | Hero prayed on `HUNGRY` or HP $> 50\%$, putting prayer on cooldown during fatal emergency. | **5** |
| `PASSIVE_HAZARD_PARALYSIS` | Hit floating eye in melee (70t paralysis) or killed by gas spore area blast. | **6** |
| `COMBAT_TACTICAL_SWARM` | Surrounded by $\ge 2$ hostiles in an open room without corridor retreat. | **7** |
| `COMBAT_FAST_PREDATOR` | Outpaced and killed by predators with speed $> 12$ (ants, bees, foxes). | **8** |
| `COMBAT_GENERAL` | Legitimate deep-dungeon combat mortality against high-tier foes. | **9** |

---

## 5. YAML-Only Telemetry & Reporting Architecture

All telemetry summaries, prompt incident reports, and DuckDB analytical query outputs use **pure YAML**:

- **No JSON, No Markdown Tables**: Tables consumed excessive tokens and suffered from column-wrapping issues. Pure YAML reduced token usage by **60–80%**.
- **`data/latest_diagnostics.yaml`**: Contains batch metrics and root cause breakdown:
  ```yaml
  batch_metrics:
    episodes: 20
    avg_depth: 4.0
    max_depth: 13
    avg_turns: 1501.7
  root_causes:
    COMBAT_GENERAL: {count: 8, pct: 40.0}
    ARMOR_DEFICIT: {count: 5, pct: 25.0}
    STALL_SECRET_DOOR: {count: 4, pct: 20.0}
    COMBAT_FAST_PREDATOR: {count: 2, pct: 10.0}
    PASSIVE_HAZARD_PARALYSIS: {count: 1, pct: 5.0}
  ```
- **`data/latest_report.yaml`**: Complete incident report including ranked fatalities, recent fatal incidents, and pre-death trajectory traces.
- **DuckDB Analytical Tools (`lox/author/tools.py`)**: All query results (`query_duckdb`, `get_death_taxonomy`, `query_root_causes`, `get_floor_pacing_stats`) return structured YAML records.

---

## 6. Canonical Empirical Invariants Knowledge Base (`lox/knowledge/invariants.py`)

32 empirical rules synthesized across 51 campaigns are indexed in [`lox/knowledge/invariants.py`](file:///home/moose/git/lox/lox/knowledge/invariants.py) via `REGISTRY`. Invariants contain canonical IDs, categories, verified rules, anti-patterns, and correct code patterns.

### Key Tactical Invariants
1. **`INV-NAV-005` (Dead-End Search Persistence)**: Search $\ge 12–15$ times in dead ends before moving to the next tile. 5 searches has a failure rate of $(1-0.15)^5 \approx 44\%$; 15 searches drops failure to $< 9\%$.
2. **`INV-NUT-003` (Conscious Divine Trouble Prayer Gate)**: Successful prayer sets divine timeout to $300 + \text{rn2}(500) \le 799$ turns. Gate hunger prayer strictly to `hunger_state >= 3` ("Weak"). Praying on `hunger_state == 2` ("Hungry") wastes favor and triggers an 850-turn cooldown during actual fainting.
3. **`INV-EQP-007` (Unarmored Body Armor Equipping)**: Valkyries start with base AC 6 from a shield and **no body armor**. When unarmored (`not obs.inventory.has_worn_body_armor`), equip dropped suits of armor immediately without waiting for BUC identification (random armor is 90% uncursed/blessed).
4. **`INV-CMB-001` (Dust Elbereth Sanctuary vs Immune Species)**: Writing `"Elbereth"` with bare fingers wards against 95% of non-humanoids. However, orcs, elves, humans, and zombies ignore Elbereth (`obs.combat.hostile_ignores_elbereth`). Engage immune hostiles in melee or retreat to 1-tile corridor chokepoints (`step_to_chokepoint()`).
5. **`INV-CMB-002` (Passive Hazard Discrimination)**: NEVER strike a floating eye in melee (70-turn paralysis). Throwing missiles at distance 1 (`obs.combat.adjacent_floating_eye`) is 100% safe. If distant immobile hazards remain and no ranged ammo exists, bypass combat and continue exploration (`walkable_nav` routes around them).
6. **`INV-NAV-002` (Gnomish Mines Avoidance)**: When descending into the dark Mines (`obs.hero.dungeon_branch == "mines"` or `dnum == 2`), immediately ascend back to the Dungeons of Doom (`yield ascend()`). The adapter automatically prunes the Mines stair from `known_stairs_down`.

---

## 7. Operational Commands & Verification Workflow

### Running a Synthesis Campaign
```bash
uv run python -u -m scripts.run_synthesis \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --generations 10 \
  --eval-episodes 20 \
  --max-turns 25000 \
  --target-depth 20.0 \
  --workers 20 \
  --policy-path data/latest_policy.py
```

### Running Unit Tests & Linter
```bash
# Full test suite (107 tests)
uv run pytest

# Fast lint and formatting checks
uvx ruff check .
uvx ruff format .
```

### Inspecting Telemetry with DuckDB
```python
import duckdb

con = duckdb.connect("data/lox.duckdb", read_only=True)

# 1. Root cause distribution across recent runs
print(con.execute("""
    SELECT root_cause, COUNT(*) as count, ROUND(AVG(depth), 2) as avg_depth, ROUND(AVG(turns), 1) as avg_turns
    FROM episodes
    GROUP BY root_cause
    ORDER BY count DESC
""").fetchall())

# 2. Deepest runs and causes of death
print(con.execute("""
    SELECT episode_id, depth, turns, death_reason, root_cause, ac_at_death
    FROM episodes
    WHERE depth >= 8
    ORDER BY depth DESC
""").fetchall())
con.close()
```

---

## 8. Next Tactical Opportunities (Advancing from DL 6 $\to$ 20)

With exploration stalls and starvation fainting eliminated, the primary blockers to reaching average depth $\ge 10.0$ are:

1. **Superior Body Armor Upgrading (`replace_body_armor`)**:
   - Dropped dwarvish mithril coats (AC 5) or iron cuirasses give vast protection over starting leather.
   - Valkyries who swap into mithril coats drop AC toward negative values, reducing monster damage by 70%.
2. **Excalibur Forging (`can_forge_excalibur` & `dip_excalibur`)**:
   - Dipping a long sword into a fountain at level $\ge 5$ has a 1/6 chance of forging Excalibur.
   - Excalibur grants +1d10 slashing damage, automatic secret door searching, and life drain resistance.
3. **Poison Resistance Harvesting (`harvest_poison_res`)**:
   - Poisonous bites from soldier ants and killer bees on DL 6–10 deal fatal bursts.
   - Seeking out and consuming killer bee/soldier ant corpses grants permanent poison resistance.
4. **Sokoban Branch Solving (`solve_sokoban`)**:
   - Completing Sokoban yields an Amulet of Reflection (reflects wand zaps and dragon breath) or Bag of Holding.
   - The graph-based solver in `lox/core/sokoban.py` calculates optimal boulder pushes without deadlocks.
5. **Castle Drawbridge & Gehennom Mazes**:
   - Zapping striking wands to breach the Castle drawbridge (`breach_drawbridge`).
   - Using digging wands/pickaxes to tunnel straight lines through Gehennom mazes (`dig_tunnel`).
