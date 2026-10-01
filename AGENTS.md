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

6. **Door Navigation & Glyph Discrimination**:
   - NetHack strictly forbids diagonal door opening or kicking (`"You see no door there"`). `NetHackAdapter` restricts door actions to cardinal directions `((-1, 0), (1, 0), (0, -1), (0, 1))`.
   - In NetHack ASCII, spellbooks on the floor also display as `+`. Using raw character matching `chars == ord("+")` mistakenly treats spellbooks as doors, causing 2,500 consecutive invalid `open_door` commands until NLE aborts the episode (`StepStatus.ABORTED`).
   - `NetHackAdapter` strictly discriminates real closed doors using NLE CMAP glyphs (`nh.GLYPH_CMAP_OFF + 15` and `+ 16`), dynamically marks phantom door coordinates in `self.non_door_tiles` upon receiving `"You see no door there"`, and safely falls back to `wait()` if no valid door is adjacent.

7. **Corpse Safety & Nutrition**:
   - Eating floor corpses without standing on them causes invalid prompts. `NetHackAdapter.step(Action(name="eat_floor_corpse"))` automatically steps towards the nearest floor corpse if not currently standing on one.

8. **Minetown & Text-Entry Dialog Auto-Dismissal (`ESC`)**:
   - In Minetown (Depths 5–9), guards and temple priests greet the hero with `"Hello stranger, who are you?"` or `"what is your name"`.
   - Directional keys enter characters into the text-entry buffer instead of moving, consuming 0 turns and leading to NLE aborting at 2,500 consecutive 0-turn steps.
   - `NetHackAdapter._dismiss_more()` automatically intercepts text-entry dialogs (`"who are you"`, `"what is your name"`, `"call this"`, `"hello stranger"`) and dismisses them with ESC (`\x1b`, action index 38), and registers the position in `self.peaceful_positions`. This leaves guards peaceful without provoking them.

9. **Autopickup & Equipment Acquisition (`wear_armor`)**:
   - By default, characters remained naked (AC 7–10) throughout the dungeon.
   - `NetHackAdapter` initializes NLE with `options=("autopickup", "pickup_types:?!/%=[$")`, automatically collecting armor, scrolls, potions, wands, and food as the hero steps over tiles.
   - `obs.inventory.has_unworn_armor` and `wear_armor()` allow the policy to equip dropped helmets, mail, cloaks, and boots during peaceful exploration, driving Armor Class down towards negative values (vastly reducing monster hit chances).

10. **Tactical In-Combat Healing & High-Speed Attackers (`is_fast_dangerous`)**:
    - Soldier ants and killer bees move at speed 18 (nearly 2x hero speed) and deliver lethal poison stings. Engaging them in open rooms is fatal.
    - `obs.combat.is_fast_dangerous` flags lethal speedsters, triggering immediate retreat into 1-tile corridor chokepoints (`step_to_chokepoint()`) to force 1v1 fights.
    - In-combat emergency healing: Quaffing healing potions takes 1 turn and restores 10–20 HP. `handle_combat` checks `if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing: yield quaff_healing()` immediately.

11. **Unified Stride-2 Checkerboard Secret Door Navigation**:
    - Procedural generation often conceals doors along room perimeter walls (`-`, `|`) or at blind corridor tips (`#`).
    - Rather than prioritizing distant corridor dead ends over adjacent room walls, `NetHackAdapter` unifies corridor dead ends and perimeter candidates into `_compute_dead_ends_mask`.
    - Room perimeters utilize a mathematical stride-2 checkerboard pattern `(cy + cx) % 2 == 0` plus all corner/alcove tiles (`adj_wall >= 2`). Because NetHack's `search` examines a $3 \times 3$ bounding box, this guarantees 100% geometric coverage of all room walls while cutting search stops and turns by $> 50\%$.
    - `step_to_dead_end()` sorts reachable candidates by search count and BFS walking distance, ensuring heroes clear local room perimeter walls before undertaking cross-dungeon treks.

12. **Exploration Pacing & Stagnation Auto-Recovery**:
    - When visible frontiers and dead ends are exhausted without discovering stairs, idling with `wait()` causes the hero to burn thousands of turns until reaching `MaxTurnsReached`.
    - `NetHackAdapter` automatically decays search counters by 10 (`self.searched_count = np.maximum(0, self.searched_count - 10)`) when frontiers stall, triggering an active second search sweep across candidate perimeter walls.
    - Exploration policies must strictly avoid yielding `wait()`; they should continually call `handle_dead_end(obs)` or `search()` to investigate room perimeter walls.

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
