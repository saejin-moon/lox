# LOX 2.0 Agent Operational Guide (`AGENTS.md`)

Welcome to LOX 2.0, an autonomous, empirical policy synthesis engine for NetHack.
This document provides complete operational context, architectural foundations, critical invariants, and debugging toolchains for any agent working in this repository.

---

## 1. Mission & Campaign Target

- **Primary Goal**: Synthesize an empirical policy that achieves **Average Dungeon Depth $\ge 10.0$** over a batch of 10 real NetHack episodes.
- **Provider**: OpenRouter (`--provider openrouter`).
- **Target Model**: `google/gemma-4-31b-it`.
- **Max Turns**: 25,000 turns per episode (`--max-turns 25000`).
- **Batch Size**: 10 evaluation episodes per generation (`--eval-episodes 10`).
- **Checkpoint Resumption**: Policies resume seamlessly from [`data/latest_policy.py`](file:///home/bae/lox/data/latest_policy.py).
- **Completion Condition**: When a generation batch achieves `avg_depth >= 10.0`, the loop logs `[CAMPAIGN GOAL ACHIEVED]` and finishes.

---

## 2. Core Architecture

```
lox/
├── author/
│   └── agent.py          # AuthorAgent: LLM tool-calling loop, AST validation, multi-scenario dry-run validator
├── core/
│   ├── types.py          # Observation, HeroState, CombatView, SpatialView, DungeonView, Action, FloorCorpse
│   ├── spatial.py        # SpatialEngine: Numba JIT A* pathfinding, frontier discovery, distance grids
│   └── tree.py           # AST-to-BT nodes (Selector, Sequence, Condition, ActionNode, Blackboard)
├── dsl/
│   ├── schema.py         # ALLOWED_PREDICATES, ALLOWED_ACTIONS, ENUM_CONSTANTS vocabulary whitelist
│   ├── parser.py         # AST validator (no imports, no exec/eval, whitelisted nodes)
│   └── compiler.py       # Compiles class-based generator policies (Agent.run(obs)) and AST trees
├── envs/
│   ├── base.py           # EnvironmentAdapter base class
│   └── nethack.py        # NetHackAdapter: NLE gym wrapper, menu dismissal, navigation & tactical primitives
├── eval/
│   └── runner.py         # Batch evaluation engine, DuckDB episode and tick telemetry logger
├── telemetry/
│   └── database.py       # DuckDB schema: episodes, ticks, token_usage, evolved_policies tables
└── wiki/
    └── engine.py         # Offline NetHack wiki retrieval engine for LLM queries
```

---

## 3. Critical Invariants & Hard-Won Lessons

1. **Python Generator Policy Paradigm**:
   - LOX 2.0 policies are written as Python classes with generator execution:
     ```python
     class Agent:
         def __init__(self):
             self.last_prayer_turn = -1000
         def run(self, obs):
             while True:
                 # Decisions yield actions and receive fresh observations
                 obs = yield wait()
     ```
   - Subroutines are invoked with `obs = yield from self.handle_combat(obs)`.

2. **The Zero-Yield Busy Loop Trap**:
   - **Crucial Rule**: Every subroutine called via `yield from` MUST yield at least one action on EVERY code branch before returning, OR the caller must not `continue` in a tight loop.
   - If a subroutine returns without yielding (e.g. `return obs`) and the caller loop does `continue`, the Python generator spins at 100% CPU without stepping the game environment, hanging execution.
   - **Safety Shield**: [`AuthorAgent.synthesize_policy`](file:///home/bae/lox/lox/author/agent.py) features a timeout-shielded multi-scenario dry-run validator (`normal`, `hungry`, `combat` states with a 1.0s hard timeout) that catches and auto-repairs any non-yielding policy before evaluation.

3. **Peaceful Creature Prompt Immunity**:
   - Attacking peaceful or passive creatures (hobbits, gnomes, acid blobs, peaceful humans) triggers NetHack confirmation prompts: `"Really attack the ...? [yn] (n)"` or `"Hello stranger..."`.
   - In NetHack, directional keys answer `n` (No) to prompts and consume 0 turns. If the policy attempts `melee_attack_hostile` again, it enters an infinite 0-turn prompt loop.
   - [`NetHackAdapter`](file:///home/bae/lox/lox/envs/nethack.py) automatically:
     - Tracks `self.peaceful_positions` per dungeon level.
     - Auto-dismisses `[yn]` prompts with `'n'`.
     - Excludes peaceful positions from `hostile_count`, `adjacent_hostiles_count`, and `melee_attack_hostile`.
     - Exposes `obs.combat.adjacent_peaceful`.

4. **Dynamic Obstacle Learning (`self.blocked_tiles`)**:
   - NetHack has impassable obstacles (iron bars, locked iron doors, boulders, solid stone) that share ASCII characters with corridors (`#`).
   - If a movement action yields `"cannot pass through the bars"`, `"It's a wall"`, or `"It's solid stone"`, `NetHackAdapter` dynamically adds the coordinate to `self.blocked_tiles` for that dungeon depth. All navigation algorithms (`step_to_frontier`, `step_to_dead_end`, `step_to`, `step_to_stairs_down`) immediately prune these coordinates from their navigation graphs.

5. **Coordinate Navigation (`step_to(y, x)`)**:
   - `step_to(y, x)` accepts coordinate integers or tuples and executes A* pathfinding via `SpatialEngine.find_path`.

6. **Door Navigation**:
   - NetHack strictly forbids diagonal door opening or kicking (`"You see no door there"`). `NetHackAdapter` restricts door actions to cardinal directions `((-1, 0), (1, 0), (0, -1), (0, 1))`.

7. **Corpse Safety & Nutrition**:
   - Eating floor corpses without standing on them causes invalid prompts. `NetHackAdapter.step(Action(name="eat_floor_corpse"))` automatically steps towards the nearest floor corpse if not currently standing on one.

---

## 4. Key CLI Commands

### Run Synthesis Campaign
```bash
uv run python -u -m scripts.run_synthesis \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --generations 100 \
  --eval-episodes 10 \
  --max-turns 25000 \
  --target-depth 10.0 \
  --policy-path data/latest_policy.py
```

### Run Unit Tests
```bash
uv run pytest -v
```

### Query Campaign Progress via DuckDB
```bash
# Check recent episode performance
uv run python -c "
import duckdb
conn = duckdb.connect('data/lox.duckdb', read_only=True)
rows = conn.execute('''
    SELECT run_id, episode_id, depth, max_depth, turns, death_reason 
    FROM episodes 
    ORDER BY episode_id DESC 
    LIMIT 10
''').fetchall()
for r in rows:
    print(r)
"

# Check generation summary metrics
uv run python -c "
import duckdb
conn = duckdb.connect('data/lox.duckdb', read_only=True)
rows = conn.execute('''
    SELECT run_id, count(*), avg(max_depth), max(max_depth), avg(turns)
    FROM episodes 
    GROUP BY run_id
    ORDER BY run_id DESC
    LIMIT 10
''').fetchall()
for r in rows:
    print(r)
"
```
