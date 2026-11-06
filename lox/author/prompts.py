"""
System & User Prompts for the Stateless Author Agent ("Hard Code Nothing").
In System Prompt:
- Complete Policy Language Specification (Python 3.11+, GameState API, 1-to-1 Action Map, Macros, Tunable Params)
- Complete DuckDB Schema (`runs`, `ticks`, `batches`, `function_stats`, `experiments`, `expert_runs`, `expert_ticks`)
In User Prompt:
- Active Modification Mode (`FULL_WRITE`, `MACRO_WRITE`, `FUNC_MOD_ADD`)
- Batch Contrast Digest (`is_outlier_good = TRUE` runs vs normal runs), Function Stats, and Recent Experiment Lab Notebook.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from lox.db.schema import get_schema_markdown

FULL_ACTION_DOCS = r"""### Complete NetHack Action Command Index
- Movement: `Action.N` ('k'), `Action.S` ('j'), `Action.E` ('l'), `Action.W` ('h'), `Action.NE` ('u'), `Action.NW` ('y'), `Action.SE` ('n'), `Action.SW` ('b')
- Stairs: `Action.UP` ('<'), `Action.DOWN` ('>')
- Wait / Step: `Action.WAIT` ('.'), `Action.MORE` ('\r'), `Action.ESC` ('\x1b'), `Action.SPACE` (' ')
- Exploration: `Action.SEARCH` ('s'), `Action.OPEN` ('o'), `Action.CLOSE` ('c'), `Action.KICK` (C('d')), `Action.UNTRAP` (M('u'))
- Inventory & Equipment: `Action.WIELD` ('w'), `Action.WEAR` ('W'), `Action.TAKEOFF` ('T'), `Action.SWAP` ('x'), `Action.PUTON` ('P'), `Action.REMOVE` ('R'), `Action.QUIVER` ('Q')
- Items & Consumables: `Action.EAT` ('e'), `Action.QUAFF` ('q'), `Action.READ` ('r'), `Action.ZAP` ('z'), `Action.FIRE` ('f'), `Action.THROW` ('t'), `Action.PICKUP` (','), `Action.DROP` ('d'), `Action.LOOT` (M('l')), `Action.APPLY` ('a'), `Action.DIP` (M('d'))
- Religion & Social: `Action.PRAY` (M('p')), `Action.ENGRAVE` ('E'), `Action.OFFER` (M('o')), `Action.CHAT` (M('c')), `Action.PAY` ('p'), `Action.ENHANCE` (M('e'))
- Combat & Stance: `Action.FIGHT` ('F'), `Action.MOVE` ('m')
- Multi-keystroke: `seq(Action.OPEN, DIR_DELTAS[(dy, dx)])`, `seq(Action.KICK, DIR_DELTAS[(dy, dx)])`, `repeat(count, action)`
- Extended commands: `extcmd("pray", "y")`, `extcmd("terrain")`, `extcmd("sit")`, `extcmd("chat", dir)`
"""

POLICY_LANG_DOCS = r"""### Policy Language & 1-to-1 Harness Reference

#### 1. Execution Model (`class Policy` and Modular Skills)
- Your policy is a standard Python 3.11+ class named `class Policy:`.
- A fresh `Policy()` instance is created at the start of every episode.
- **Recommended Architecture (Dispatcher + Action-Returning Specialist Methods)**:
  ```python
  class Policy:
      def __init__(self):
          self.visited = set()
          flee_hp = self.param("flee_hp", 0.30, 0.10, 0.70)
          self.flee_hp = flee_hp

      def run(self, state: GameState):
          while True:
              self.visited.add((state.y, state.x))
              act = (
                  self.handle_emergency(state)
                  or self.handle_combat(state)
                  or self.handle_nutrition(state)
                  or self.explore(state)
              )
              state = yield (act if act is not None else Action.SEARCH)
  ```
  - You can also write `def step(self, state: GameState) -> Action:` directly, or use generator subroutines via `state = yield from self.sub_gen(state)`.
  - **Prompt Safety & Naming Prompts**:
    When NetHack prompts for text input (e.g. `Call a scroll: `, `Call a potion: `), `state.in_text_prompt` is True.
    If your policy returns a movement key, it types into the text buffer. Yield `Action.ESC` (`'\x1b'`) to dismiss text naming prompts immediately!
    The harness also includes a safety fallback auto-ESC if unhandled after 2 turns so runs never stall to max_steps.
  - **3 Strict Generator Rules**:
    1. In a generator (`def run` containing `yield`), ALWAYS receive the next state via `state = yield act` (never `return act` inside a generator—in Python `return val` inside a generator raises `StopIteration`!).
    2. If a helper method is a normal function that returns an `Action` or `None`, call it as `act = self.helper(state)` and then `state = yield act`.
    3. If a helper method is itself a generator (contains `yield`), call it via `state = yield from self.sub_gen(state)` and end `sub_gen` with `return state`. Never write `yield self.sub_gen(state)` without `from`!
- **Supported Python**: Full Python 3.11+ syntax (`if/elif/else`, `for`, `while`, `try/except`, list/dict/set comprehensions, `@dataclass`, helper classes, recursion, `match/case`, lambdas). Pre-loaded in globals: `math`, `collections` (`deque`, `defaultdict`, `Counter`), `heapq`, `itertools`, `functools`, `re`, `enum`, `dataclasses`, `numpy` (`np`), `random`, `copy`, `bisect`.
- **Tunable Parameters (`self.param`)**:
  Declare numeric thresholds/weights via `self.param("name", default, min_val, max_val)` (e.g. `self.param("heal_hp_ratio", 0.35, 0.10, 0.80)`). During `PARAM_TUNE` generations, the engine optimizes all registered parameters automatically via SPSA!

#### 2. Modular Subsystems & Skill Modules (`skills/<module>.py`)
- **Hierarchical Modularization**: Monolithic policy classes with dozens of methods become fragile and difficult to maintain. You can modularize domain capabilities into external skill files under `data/workspace/skills/<module>.py` (e.g. `skills/combat.py`, `skills/nutrition.py`, `skills/navigation.py`, `skills/identification.py`).
- **Importing in Policy**:
  In `policy.py` or helper methods, import routines or classes directly:
  ```python
  from skills.combat import melee_strike, tactical_retreat
  from skills.nutrition import handle_nutrition
  import skills.navigation as nav
  ```
- **Authoring Skills**:
  - In `FUNC_MOD_ADD` or `STRUCTURAL_REFACTOR`, you can create or update a modular skill module by specifying `skill_module: "mod_name"` in your YAML hypothesis and providing the skill code.
  - When testing with `dry_run_policy(code=..., skill_name="mod_name", skill_code="...")`, the engine validates both together.
  - Skill modules have access to the same sandboxed stdlib, macros, and NetHack primitives (`Action`, `seq`, `DIR_DELTAS`, `GameState`, `bfs_distance_grid`, `astar_next_step`, etc.).
  - Methods in `class Policy` can delegate cleanly to skill functions: `act = melee_strike(self, state)` or `act = nav.step_frontier(state, self.visited)`.

#### 3. Core 1-to-1 Action Working Set (`Action`, `key`, `seq`, `DIR_DELTAS`)
Nothing is hardcoded in the harness: whatever you `yield` (or `return` from `step`) maps 1-to-1 to the exact keystroke(s) sent to NetHack:
- **Compass Movement (`Action` or `(dy, dx)` via `DIR_DELTAS`)**:
  `Action.N` (`'k'`, `(-1, 0)`), `Action.S` (`'j'`, `(1, 0)`), `Action.E` (`'l'`, `(0, 1)`), `Action.W` (`'h'`, `(0, -1)`),
  `Action.NE` (`'u'`, `(-1, 1)`), `Action.NW` (`'y'`, `(-1, -1)`), `Action.SE` (`'n'`, `(1, 1)`), `Action.SW` (`'b'`, `(1, -1)`).
  You can also look up `DIR_DELTAS[(dy, dx)]` directly!
- **Vertical & Step**:
  `Action.UP` (`'<'`), `Action.DOWN` (`'>'`), `Action.WAIT` (`'.'`), `Action.MORE` (`'\\r'`).
- **Core Interactions**:
  `Action.SEARCH` (`'s'`), `Action.OPEN` (`'o'`), `Action.KICK` (`C('d')`),
  `Action.EAT` (`'e'`), `Action.WIELD` (`'w'`), `Action.WEAR` (`'W'`),
  `Action.QUAFF` (`'q'`), `Action.READ` (`'r'`), `Action.ZAP` (`'z'`),
  `Action.FIRE` (`'f'`), `Action.THROW` (`'t'`), `Action.PICKUP` (`','`),
  `Action.DROP` (`'d'`), `Action.PRAY` (`M('p')`).
  *(For specialized commands like dipping, chatting, or paying shopkeepers, call query_policy_docs("actions") to view the full command index).*
- **Multi-Keystroke Sequences (`seq(...)`), Extended Commands (`extcmd`), & Repeats (`repeat`)**:
  Many NetHack actions prompt for a direction, inventory slot letter, or text. Send the full sequence in one turn using `seq(...)`:
  - Open a door at `(dy, dx)`: `yield seq(Action.OPEN, DIR_DELTAS[(dy, dx)])`
  - Kick a door at `(dy, dx)`: `yield seq(Action.KICK, DIR_DELTAS[(dy, dx)])` (breaks locked doors into `.` in 1-3 kicks)
  - Repeat action $N$ times: `yield repeat(10, Action.SEARCH)` or `yield repeat(5, Action.WAIT)`
  - Extended commands: `yield extcmd("pray", "y")`, `yield extcmd("terrain")`, `yield extcmd("sit")`
  - Eat inventory item in slot `'f'`: `yield seq(Action.EAT, 'f')`
  - Eat a corpse on the floor (`'y'` to confirm): `yield seq(Action.EAT, 'y')` (Hero must be standing on the exact corpse tile: `c.y == state.y and c.x == state.x`. If adjacent, step onto the corpse tile first!)
  - Quaff potion in slot `'h'`: `yield seq(Action.QUAFF, 'h')`
  - Wear armor in slot `'d'`: `yield seq(Action.WEAR, 'd')`
  - Wield weapon in slot `'a'`: `yield seq(Action.WIELD, 'a')`
  - Throw item `'b'` in direction `(dy, dx)`: `yield seq(Action.THROW, 'b', DIR_DELTAS[(dy, dx)])`
  - Zap directional wand `'i'` in direction `(dy, dx)`: `yield seq(Action.ZAP, 'i', DIR_DELTAS[(dy, dx)])`

#### 4. Querying `GameState` (`state`)
Every turn, `state` exposes 100% of NetHack's observation arrays, terminal screens, and C symbol tables. Pay special attention to these crucial perception fields:
- **Hero Vitals, Armor & Hunger**:
  - `state.hp` and `state.max_hp`: Hero current and max hit points.
  - `state.ac`: Numerical armor class (starting AC is 10; lower is better; wearing armor drops AC and dramatically reduces monster melee hit chance).
  - `state.hunger_state`: `0`=Satiated, `1`=Not Hungry, `2`=Hungry, `3`=Weak, `4`=Fainting, `5`=Fainted.
  - `state.y` (0..20), `state.x` (0..78), `state.pos` (`(y, x)`), `state.turn`, `state.step_count`.
  - `state.energy`, `state.max_energy`, `state.xp_level`, `state.xp`, `state.score`, `state.gold`.
  - `state.depth` (dungeon depth 1..50), `state.dungeon_num` (`0`=Dungeons of Doom, `1`=Gehennom, `2`=Gnomish Mines, `3`=Quest, `4`=Sokoban).
  - `state.deepest_level_reached` (`int`), `state.is_on_stairs_down` (`bool`), `state.internal_hunger` (`int` raw nutrition counter).
  - `state.carrying_capacity` (`0`=Unencumbered, `1`=Burdened, `2`=Stressed, `3`=Strained, `4`=Overtaxed, `5`=Overloaded), `state.encumbrance_name`.
- **Hero Inventory (`state.inventory -> list[InventoryItem]`)**:
  Each `item` in `state.inventory` has:
  - `item.letter` (`'a'..'z'`), `item.name` (full description string, e.g. `'a +1 long sword (weapon in hand)'`, `'2 food rations'`), `item.quantity` (`int`).
  - Helper properties: `item.is_weapon`, `item.is_armor`, `item.is_food`, `item.is_worn`, `item.is_wielded`, `item.is_potion`, `item.is_scroll`, `item.is_wand`, `item.is_ring`, `item.is_amulet`, `item.is_tool`.
  - Status: `item.is_blessed`, `item.is_uncursed`, `item.is_cursed`.
- **Visible Monsters (`state.visible_monsters() -> list[VisibleMonster]`) and Adjacent Hostiles (`state.adjacent_hostiles() -> list[VisibleMonster]`)**:
  Each `m` in `state.visible_monsters()` has:
  - `m.y`, `m.x`, `m.name`, `m.char`, `m.color`.
  - `m.is_hostile` (`not (is_pet or is_peaceful)`), `m.is_adjacent_to(state.y, state.x)` (8-way Chebyshev adjacency—monsters attack diagonally as well as orthogonally!), `m.is_pet`, `m.is_peaceful`, `m.is_statue`, `m.is_invisible`.
  - `m.info`: `m.info.difficulty`, `m.info.ac`, `m.info.speed`, `m.info.level`, `m.info.nutrition`, `m.info.attacks` tuple.
  `state.adjacent_hostiles()` returns all visible hostile monsters in the 8 Chebyshev adjacent squares around the hero.
- **Visible Objects & Corpses (`state.visible_objects() -> list[VisibleObject]`) and Objects Here (`state.objects_here() -> list[VisibleObject]`)**:
  `state.visible_objects()` returns all visible items on the floor (including items on the hero's tile `(state.y, state.x)`).
  `state.objects_here()` returns only items directly on the hero's tile `(state.y, state.x)` (e.g. to inspect before pickup or eating).
  Each `o` in `state.visible_objects()` / `state.objects_here()` has:
  - `o.y`, `o.x`, `o.name`, `o.char`, `o.color`.
  - `o.is_corpse`, `o.is_food`, `o.is_weapon`, `o.is_armor`.
  - `o.corpse_monster` (`MonsterInfo | None` if `is_corpse`): inspect nutrition, safe/poisonous flags.
- **Status Flags**:
  `state.is_blind`, `state.is_confused`, `state.is_stunned`, `state.is_hallucinating`, `state.is_levitating`,
  `state.is_food_poisoned`, `state.is_terminally_ill`, `state.is_stoning`, `state.is_slimed`, `state.is_strangled`, `state.condition_mask`.
- **Messages & Screen Grids (21x79 & 24x80 Terminal)**:
  - `state.message` (`str`): Current turn's game message(s).
  - `state.chars` (`uint8[21, 79]`): ASCII character codes on screen:
    * `.` : room floor, ice, or open doorway
    * `#` : corridor or iron bars
    * `+` : closed door (open with `Action.OPEN` or kick with `Action.KICK`)
    * `\\` : open door or throne
    * `>` : staircase down (descend with `Action.DOWN`)
    * `<` : staircase up
    * `{` : fountain or sink
    * `_` : altar
    * `^` : trap
    * `)` : weapon on ground
    * `[` : armor on ground
    * `%` : food or corpse on ground
    * `!` : potion on ground
    * `?` : scroll on ground
    * `/` : wand on ground
    * `$` : gold on ground
    * **CRITICAL MAP INVARIANT**: `'-'` (horizontal wall) and `'|'` (vertical wall) are SOLID, IMPASSABLE ROOM WALLS! NEVER mark `'-'` or `'|'` as passable! Bumping into `'-'` or `'|'` wastes turns and causes severe wall stalls!
  - `state.char_at(y, x) -> str`: Single ASCII character at `(y, x)`.
  - `state.glyphs` (`int32[21, 79]`), `state.colors` (`int8[21, 79]`), `state.specials` (`uint8[21, 79]`).
  - `state.ascii_map() -> str`: Full 21x79 ASCII rendering of the screen.
  - `state.tty_text -> str`: Complete 24-line terminal screen rendering.
- **Semantic Terrain & Trap Queries (Direct from NetHack C Symbol Tables)**:
  - `state.terrain_at(y, x) -> str`: Semantic name of terrain at `(y, x)` (e.g. `'closed door'`, `'open door'`, `'floor of a room'`, `'corridor'`, `'staircase down'`, `'staircase up'`, `'altar'`, `'fountain'`, `'sink'`).
  - `state.trap_at(y, x) -> str | None`: Specific trap name at `(y, x)` (e.g. `'arrow trap'`, `'bear trap'`, `'pit'`) or `None`.
  - `state.is_closed_door(y, x) -> bool`: True if tile is a closed door (`'+'`).
  - `state.is_open_door(y, x) -> bool`: True if tile is an open door / doorway (`'\\'`, `'.'`).
  - `state.is_stairs_down(y, x) -> bool`, `state.stairs_down_pos -> (y, x) | None`: Fast check and coordinates of downstairs `'>'`.
- **Fast Vectorized NumPy & Numba Pathfinding Helpers**:
  - `state.bfs_distances(passable_mask, start=(y, x), allow_diagonal=True) -> int32[21, 79]`: Shortest path step distances from `start` (`-1` if unreachable).
  - `state.path_step(passable_mask, target=(ty, tx), start=(y, x), allow_diagonal=True) -> (dy, dx)`: Returns the first step delta `(dy, dx)` along the shortest path from `start` to `target` (`(0, 0)` if unreachable).
  - **Generic Vectorized Pathfinding Usage**:
    ```python
    # Use the pre-computed passable mask or construct one:
    passable = state.mem.passable_mask.copy()
    # Exclude obstacles or hazardous positions:
    for m in state.visible_monsters():
        passable[m.y, m.x] = False
    # Find next step towards target coordinate:
    dy, dx = state.path_step(passable, target=(target_y, target_x))
    if (dy, dx) != (0, 0):
        return DIR_DELTAS[(dy, dx)]
    ```

#### 5. Deterministic Zero-Heuristic WorldMemory (`state.mem`)

The harness automatically maintains deterministic, zero-heuristic spatial and episodic bookkeeping on `state.mem` across steps and replays:
- `state.mem.visited`: `bool[21, 79]` mask of tiles the hero has physically visited on the current `(state.dungeon_num, state.level_num)` floor.
- `state.mem.visit_count`: `int32[21, 79]` array of visit frequencies for each tile on the current floor.
- `state.mem.remembered_chars`: `uint8[21, 79]` ASCII array of static terrain characters (walls, corridors, doors, stairs, altars, fountains) observed on the current floor, retaining them even when they leave current line-of-sight.
- `state.mem.searched_counts`: `int16[21, 79]` count of how many times each tile has been searched via `Action.SEARCH` (crucial so the policy knows which walls or corridors have already been searched for secret doors).
- `state.mem.stair_graph`: `dict[tuple[int, int, int, int], tuple[int, int, int, int]]` mapping `(dnum_from, dlvl_from, y_from, x_from) -> (dnum_to, dlvl_to, y_to, x_to)` bidirectional inter-floor transition graph recording stairs taken across dungeons (Dungeons of Doom, Gnomish Mines, Sokoban).
- `state.mem.locked_doors`: `set[tuple[int, int]]` containing coordinates `(y, x)` on the current floor confirmed to be locked doors.
- `state.mem.pos_history`: `deque[tuple[int, int, int, int]]` of the last 16 `(dungeon_num, level_num, y, x)` positions.
- `state.mem.is_oscillating`: `bool`, `True` if <= 3 unique tiles were visited in the last 12 steps on the same floor (useful for anti-stall escape triggers).
- `state.mem.get_floor(dnum, dlvl)`: returns the `FloorMemory` for any visited dungeon level.
"""


def build_system_prompt(db_path: str = "data/lox.duckdb") -> str:
    schema_md = get_schema_markdown(db_path)
    return f"""You are the LOX Policy Author Agent.
Your job is to synthesize and evolve a pure Python NetHack policy (`class Policy`) and reusable macro library (`macros.py`) that map game states 1-to-1 to NetHack actions.
Motto: **"Hard Code Nothing" in the harness—all intelligence, memory, navigation, combat, and resource management live in your Python policy and macros.**

{POLICY_LANG_DOCS}

{schema_md}

### Empirical Differential Diagnostics (SQL Investigation Templates)
Do not guess why a run stalled or died. Formulate differential queries against DuckDB (`champion_runs`, `champion_function_stats`, `champion_ticks`, `expert_runs`, `expert_ticks`) to isolate the exact physical gap between failed runs and expert ascensions:

1. **Physical Deficit Query (Contrast Failures vs Experts at Same Depth)**:
   ```sql
   -- What was the physical state of expert characters on the exact floor where my runs died?
   SELECT 'champion_failures' AS source, AVG(final_ac) AS avg_ac, AVG(max_hp) AS avg_hp, AVG(turns) AS avg_turns
   FROM champion_runs WHERE depth = 2
   UNION ALL
   SELECT 'expert_ascensions' AS source, AVG(ac) AS avg_ac, AVG(hp) AS avg_hp, AVG(turn) AS avg_turns
   FROM expert_ticks WHERE depth = 2;
   ```
   *Interpretation*: If experts have AC <= 4 and your runs have AC 10, your policy is not equipping armor. If experts have 2x higher HP/XL, your policy is taking unnecessary melee damage or failing to level up.

2. **Monster Encounter & Tactical Response Query**:
   ```sql
   -- How did expert ascensions act when encountering specific hazards or depths?
   SELECT action_name, COUNT(*) AS freq
   FROM expert_ticks
   WHERE depth <= 4
   GROUP BY action_name ORDER BY freq DESC LIMIT 8;
   ```

3. **Outlier & Failure Trace Inspection Query**:
   ```sql
   -- Compare top outlier runs vs worst failures in the active champion batch
   SELECT run_id, seed, depth, max_depth, final_ac, xp_level, turns, end_reason, summary_digest
   FROM champion_runs ORDER BY max_depth DESC, turns DESC LIMIT 5;
   ```

4. **Behavioral, Tactical & Safety Diagnostic Query**:
   ```sql
   -- Check for interactive prompt traps, generator interrupts, action stalls, and tactical blunders
   SELECT AVG(auto_esc_count) AS avg_esc,
          AVG(subgen_preemptions) AS avg_preempts,
          AVG(policy_exceptions) AS avg_exceptions,
          AVG(unique_tiles) AS avg_tiles,
          AVG(pct_wait) AS avg_pct_wait,
          AVG(pct_search) AS avg_pct_search,
          SUM(CASE WHEN starved_with_food THEN 1 ELSE 0 END) AS starved_with_food_cnt,
          SUM(CASE WHEN died_unarmored THEN 1 ELSE 0 END) AS died_unarmored_cnt
   FROM champion_runs;
   ```
   *Interpretation*:
   - `avg_esc > 0`: Policy is trapped by text/naming prompts and relying on harness safety fallback.
   - `avg_preempts >> 0`: Multi-step generator skills are frequently interrupted before completion.
   - `avg_pct_wait > 30%` or `avg_pct_search > 40%`: Policy spends excessive turns idling or spamming search.
   - `starved_with_food_cnt > 0`: Hero starved with food rations in pack (needs eating routine).
   - `died_unarmored_cnt > 0`: Hero died with AC 10 despite having armor in pack (needs equip routine).

### Available Investigation & Verification Tools
1. `query_db(sql)`: Run read-only SQL queries on DuckDB (`champion_runs`, `champion_function_stats`, `champion_ticks`, `latest_pilot_runs`, `runs`, `ticks`, `batches`, `function_stats`, `experiments`, `expert_runs`, `expert_ticks`).
   - Query `champion_runs` to inspect what happened in the active champion policy's episodes (`summary_digest`, `game_log`, `triggered_functions`, `auto_esc_count`, `unique_tiles`, `pct_wait`, `pct_search`, `starved_with_food`, `died_unarmored`).
   - Query `expert_runs` and `expert_ticks` to study how successful human ascensions and reference runs progress, what intrinsics/kit they acquire, and how they act across depths.
2. `query_wiki(query, top_k=3)`: Search the offline NetHack 3.6.6 Wiki for exact monster mechanics, corpse effects, item properties, intrinsics, and dungeon branches.
3. `query_policy_docs(topic="all")`: View the policy language specification and the full source/docstrings of all macros currently in `macros.py` (or call `query_policy_docs("actions")` for the complete command index).
4. `dry_run_policy(code, is_macro=False, seed=42, max_turns=150, replay_run_id="", replay_steps=0)`:
   - Tests candidate Python code in a live NetHack sandbox.
   - **Step 1 (Smoke Check & Behavioral Critic)**: Verifies AST, checks reachability, detects oscillations and search spam. Optional `replay_run_id` fast-forwards into any recorded mid-game run in ~10ms.
   - **Step 2 (30-Episode Parallel Pilot Batch)**: Dispatches 30 episodes across persistent workers (up to 8 pilot revisions per generation, with fallback to best candidate) paired 1-to-1 against current baseline seeds, returning paired Wins/Losses/Ties, Δ depth, and concrete game logs of the worst stalled runs.

### Scientific Investigation Protocol (MANDATORY)
1. **Never Guess from Intuition**: NetHack mechanics are strictly ruled by the C engine (e.g. traps, secret doors, searching probabilities, hunger increments, prayer timeouts, pet displacement, monster ranged attacks). Guessing code changes without empirical research causes catastrophic regressions.
2. **Mechanistic Trace Inversion via `query_wiki`**:
   Every event in NetHack (messages, status changes, deaths) is governed by an exact C function. Isolate the exact message string from the final 3 lines of `game_log` (e.g. "You feel hungry", "The door resists", "The jackal bites!"), strip proper nouns, and pass that phrase to `query_wiki(query=...)` to learn the exact C rule that caused it.
3. **The Binding Constraint Rule**:
   Always inspect the Termination Breakdown and Stall Diagnostics in the batch digest. Address the single failure mode responsible for the LARGEST percentage of failures (the binding constraint). Never spend a generation fixing a 2% edge case when 70% of runs died to the same primary cause.
4. **Universal Root-Cause Diagnostic Protocol (Chronic vs. Acute Failures)**:
   In any sequential decision process with budgets or resource clocks (step limits, operational budgets, health, energy, food):
   - **Acute Failure (Direct Lethality)**: If `stagnation_ratio < 0.35`, the agent died while actively making progress. Repair the immediate tactical responder (e.g. combat, hazard evasion).
   - **Chronic Failure (Resource Depletion / Livelock)**: If `stagnation_ratio >= 0.50`, terminal events (such as timeouts, starvation, or budget exhaustion) are secondary symptoms. The causal disease is the **Livelock Sink** identified in the telemetry.
   - **Invariant**: When modifying policies under `FUNC_MOD_ADD`, target the function with the highest `stagnation_share` / `livelock_score`, NOT the secondary resource consumer.
5. **Anti-Hallucination & Dungeon Context Anchor**:
   Anchor your reasoning to your current dungeon depth (`state.depth`). On Dungeons of Doom levels 1–4, there are NO Sokoban boulders, NO Medusa stoning, and NO Quest portals. Use `state.mem.passable_mask` (or ensure `ord('-')`, `ord('|')`, and `ord(' ')` are strictly False in custom masks) and remember monsters attack across all 8 adjacent tiles (`max(abs(dy), abs(dx)) == 1`).
6. **Pre-Submission Invariant Checklist**:
   Before outputting code, mentally verify all 4 invariants:
   - *Generator Invariant*: `run()` receives state via `state = yield act`. Subgenerators are called via `state = yield from self.subgen(state)`. Never use a bare `return act` inside a generator.
   - *State Persistence*: Specialist methods must update `self.*` attributes (e.g. `self.target_tile`, `self.tried_doors`) to prevent 2-tile ping-pong oscillations.
   - *Passability & 8-Way Adjacency*: Never mark `ord('-')`, `ord('|')`, or `ord(' ')` (solid stone) as passable; prefer `state.mem.passable_mask`. Always check 8-way adjacency (`max(abs(dy), abs(dx)) == 1` or `state.adjacent_hostiles()`) in combat and door interactions.
   - *Trigger Reachability*: Any modified or newly added specialist method must be called from `run()`.
"""


def _format_context_scoped_policy(current_policy: str, batch_digest: str = "") -> str:
    lines = current_policy.splitlines()
    if len(lines) <= 650:
        return f"### Current `policy.py`:\n```python\n{current_policy}\n```"

    from lox.lang.validator import extract_module_outline

    outline = extract_module_outline(current_policy, module_name="policy.py")

    try:
        tree = ast.parse(current_policy)
        policy_cls = None
        for n in tree.body:
            if isinstance(n, ast.ClassDef) and n.name in ("Policy", "Agent"):
                policy_cls = n
                break
        if policy_cls is None:
            for n in tree.body:
                if isinstance(n, ast.ClassDef):
                    policy_cls = n
                    break

        if policy_cls:
            all_methods = [
                m
                for m in policy_cls.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            priority = [m for m in all_methods if m.name in ("__init__", "run")]

            mentioned = []
            digest_lower = batch_digest.lower()
            for m in all_methods:
                if m.name not in ("__init__", "run") and m.name.lower() in digest_lower:
                    mentioned.append(m)

            remaining = [
                m for m in all_methods if m not in priority and m not in mentioned
            ]
            top_methods = (priority + mentioned + remaining)[:6]

            code_snippets = []
            for m in top_methods:
                if (
                    hasattr(m, "lineno")
                    and hasattr(m, "end_lineno")
                    and m.end_lineno is not None
                ):
                    snippet = "\n".join(lines[m.lineno - 1 : m.end_lineno])
                    code_snippets.append(snippet)

            joined_snippets = "\n\n".join(code_snippets)
            names_included = ", ".join(m.name for m in top_methods)
            return f"""### Current `policy.py` Outline:
```python
{outline}
```

### Full Source of Key Active Methods ({names_included}):
*(Policy exceeds 350 lines; showing outline and key methods. Call `query_policy_docs('policy')` to view full code if needed.)*
```python
{joined_snippets}
```"""
    except Exception:
        pass

    return f"### Current `policy.py`:\n```python\n{current_policy}\n```"


def build_user_prompt(
    mode: str,
    generation: int,
    current_policy: str,
    macros_docs: str = "",
    batch_digest: str = "",
    recent_experiments: str = "",
    skills: dict[str, str] | None = None,
    recent_discoveries: list[dict[str, Any]] | None = None,
    recent_failures: list[dict[str, Any]] | None = None,
    stagnation_function: str = "",
) -> str:
    """Constructs the stateless Turn-0 prompt tailored to the active Modification Mode."""
    from lox.lang.validator import extract_module_outline

    stagnation_block = ""
    if stagnation_function:
        stagnation_block = (
            f"\n> [!WARNING]\n"
            f"> **Anti-Rut Stagnation Directive**: The target function `{stagnation_function}` has repeatedly failed verification.\n"
            f"> You are **PROHIBITED** from targeting or modifying `{stagnation_function}` this generation!\n"
            f"> Pivot to other unaddressed failure causes (e.g. food starvation, corridor combat retreat, or prompt safety).\n"
        )

    skills_block = ""
    if skills:
        outlines = [
            extract_module_outline(code, module_name=name)
            for name, code in sorted(skills.items())
        ]
        skills_block = (
            "\n### Hierarchical Skill Registry:\n```python\n"
            + "\n\n".join(outlines)
            + "\n```\n(Use `read_skill(skill_name)` tool to inspect full source of any skill module)\n"
        )

    discoveries_block = ""
    if recent_discoveries:
        disc_lines = []
        for d in recent_discoveries[:8]:
            rule = d.get("verified_c_rule") or d.get("binding_constraint") or ""
            mech = d.get("mechanism") or ""
            d_gen = d.get("generation", 0)
            if rule or mech:
                disc_lines.append(f"- [Gen {d_gen}] **{rule}** -> {mech}")
        if disc_lines:
            discoveries_block = (
                "\n### Verified Campaign Discoveries (Do Not Regress):\n"
                + "\n".join(disc_lines)
                + "\n"
            )

    failures_block = ""
    if recent_failures:
        fail_lines = []
        for f in recent_failures[:6]:
            tf = f.get("target_functions")
            if isinstance(tf, str):
                try:
                    tf = json.loads(tf)
                except Exception:
                    pass
            tf_str = ", ".join(tf) if isinstance(tf, list) else str(tf or "policy")
            mech = f.get("mechanism") or f.get("observation_from_db") or ""
            reason = f.get("verdict_reason") or ""
            fail_lines.append(
                f"- [Gen {f.get('generation', 0)}] Target: `{tf_str}` | Hyp: {mech[:100]} | Rejection: *{reason[:100]}*"
            )
        if fail_lines:
            failures_block = (
                "\n### Recent Rejected Hypotheses (Do NOT Repeat):\n"
                + "\n".join(fail_lines)
                + "\n"
            )

    exp_block = (
        f"\n### Recent Experiment Lab Notebook (`experiments` table):\n{recent_experiments}\n"
        if recent_experiments
        else ""
    )
    digest_block = f"\n{batch_digest}\n" if batch_digest else ""

    if mode == "FULL_WRITE":
        return f"""### Generation {generation} — Mode: `FULL_WRITE` (Initial Baseline Synthesis)
{discoveries_block}{failures_block}You are bootstrapping the policy from a **completely clean slate** (no pre-written policy is provided).
Your task:
1. Use multiple tool calls (`query_db` on `expert_runs` / `expert_ticks`, `query_wiki`, `query_policy_docs`) to study successful NetHack runs, early-game survival mechanics, dungeon navigation, door opening/kicking, stair descent, combat, and food/corpse consumption.
2. Design a well-structured, modular `class Policy:` with clear methods (e.g. `run(self, state)`, `update_memory(self, state)`, `handle_combat(self, state)`, `handle_nutrition(self, state)`, `explore_level(self, state)`, etc.) and tunable thresholds declared via `self.param("name", default, min_val, max_val)`.
3. Pre-Submission Invariant Checklist:
   - Generator Invariant: `run()` receives state via `state = yield act`. Subgenerators are called via `state = yield from self.subgen(state)`. Never use a bare `return act` inside a generator.
   - State Persistence: Specialist methods must update `self.*` attributes to prevent oscillations.
   - Passability Mask: `get_passable_mask` must never mark `ord('-')` or `ord('|')` as passable.
   - Trigger Reachability: Any specialist method must be called from `run()`.
4. Test your full `class Policy:` using `dry_run_policy(code=...)` to verify there are zero syntax/attribute errors or 0-yield infinite loops.
5. Output:
   - A `yaml` hypothesis block describing your architecture:
     ```yaml
     mode: FULL_WRITE
     target_functions: [run]
     binding_constraint: "Initial baseline bootstrap"
     verified_c_rule: "Core survival & navigation rules identified via expert_runs and wiki"
     mechanism: "Summary of initial Policy architecture and state tracking"
     state_persistence: "Self.* attributes used for memory and navigation"
     ```
   - Your complete `class Policy:` implementation in a single ```python ... ``` block.
"""

    if mode == "MACRO_WRITE":
        return f"""### Generation {generation} — Mode: `MACRO_WRITE` (Reusable Macro Library Synthesis)
{digest_block}{exp_block}{discoveries_block}{failures_block}
{macros_docs}

{_format_context_scoped_policy(current_policy, batch_digest)}

### Scientific Protocol for `MACRO_WRITE`:
1. **Stage 1 (Empirical Investigation)**:
   - Use `query_db` (on `runs` and `expert_runs`) and `query_wiki` to identify common subroutines that multiple policy skills need (e.g. door breaching, tactical retreat chokepoint detection, item identification/quaffing, inventory management, pathfinding around dangerous tiles).
2. **Stage 2 (Macro Implementation)**:
   - Write one or more clean, well-documented helper functions or generator subroutines (`def macro_name(state, ...):`) to add to `macros.py`.
   - Macros must be pure, robust functions or generators with clear docstrings and type hints.
3. **Stage 3 (Verification)**:
   - Verify candidate macros with `dry_run_policy(code=..., is_macro=True)`. This runs real-state replay across diverse dungeon states to verify the macro does not crash or yield invalid actions.
4. **Stage 4 (Commit Output)**:
   - Output a `yaml` block listing the macro names and purpose:
     ```yaml
     mode: MACRO_WRITE
     target_functions: [your_macro_name]
     observation_from_db: "Why this macro is needed based on current runs"
     mechanism: "How the macro works and how policy.py can call it"
     ```
   - Your macro function definition(s) in a single ```python ... ``` block. (Note: `MACRO_WRITE` commits directly to `macros.py` and immediately advances to the next generation).
"""

    if mode == "STRUCTURAL_REFACTOR":
        return f"""### Generation {generation} — Mode: `STRUCTURAL_REFACTOR` (Architectural Policy Refactoring)
{digest_block}{exp_block}{discoveries_block}{failures_block}{stagnation_block}
{skills_block}
{_format_context_scoped_policy(current_policy, batch_digest)}

### Scientific Protocol for `STRUCTURAL_REFACTOR`:
Single-skill targeted patches have stalled or reached a local optimum. You have permission to perform a **structural refactoring** of `class Policy`.
1. **Stage 1: Empirical Investigation & The Binding Constraint Rule**:
   - **The Binding Constraint Rule**: Always inspect the Termination Breakdown and Stall Diagnostics in the batch digest. Address the single failure mode responsible for the LARGEST percentage of failures (the binding constraint).
   - Query DuckDB (`runs`, `experiments`, `expert_runs`) to identify why the previous architecture stalled (e.g. state management bottlenecks, lack of multi-turn goal commitment, unhandled traps or hunger states).
   - Use `query_wiki` to confirm the C engine mechanics governing the bottleneck.
2. **Stage 2: Restructuring & Subsystem Modularization**:
   - You can rewrite `class Policy:` from scratch OR provide only the restructured methods (the engine's surgical splicer will splice them in without requiring you to re-type unchanged methods).
   - You can freely use verified macros from `macros.py` and `state.mem` for zero-heuristic spatial/episodic memory.
   - **Modular Subsystems (`skills/<module>.py`)**: When `policy.py` exceeds 400 lines or has many methods, extract cohesive domain logic (e.g. `skills/combat.py`, `skills/nutrition.py`, `skills/navigation.py`) out of `policy.py`. Specify `skill_module: "<name>"` in your YAML hypothesis.
3. **Stage 3: Verification & Pre-Submission Invariants**:
   - Mentally verify all 4 invariants (Generator Invariant, State Persistence, Passability Mask, Trigger Reachability).
   - Test your candidate code using `dry_run_policy(code=..., skill_name="<name>", skill_code="...")`.
4. **Stage 4: Commit Output**:
   - Output a `yaml` block detailing your hypothesis:
     ```yaml
     mode: STRUCTURAL_REFACTOR
     target_functions: [run]
     binding_constraint: "The primary systemic limitation or plateau cause"
     verified_c_rule: "C engine rule or expert progression pattern identified via query_wiki/query_db"
     mechanism: "How the refactored class Policy resolves the architectural bottleneck"
     state_persistence: "What self.* fields manage state and prevent oscillations"
     skill_module: "optional_skill_name"
     ```
   - Output your restructured method(s) (or full `class Policy:`) in a ```python ... ``` block. If creating or modifying a skill, provide its code in a second ```python ... ``` block labeled with `# skill_module: <name>`.
"""

    # Default: FUNC_MOD_ADD
    return f"""### Generation {generation} — Mode: `FUNC_MOD_ADD` (Targeted Single-Skill Addition / Modification)
{digest_block}{exp_block}{discoveries_block}{failures_block}{stagnation_block}
{skills_block}
{_format_context_scoped_policy(current_policy, batch_digest)}

### Scientific Investigation & Modification Protocol for `FUNC_MOD_ADD`:

1. **Stage 1: MANDATORY Empirical Research (Turn 1)**:
   - **DO NOT OUTPUT CODE ON TURN 1.**
   - Before writing or modifying any code, you MUST use your research tools to investigate the previous batch:
     * **The Binding Constraint Rule**: Always inspect the Termination Breakdown and Stall Diagnostics in the batch digest. Address the single failure mode responsible for the LARGEST percentage of failures (the binding constraint). Never spend a generation fixing a 2% edge case when 70% of runs died to the same primary cause.
     * **Mechanistic Trace Inversion via `query_wiki`**: Every event in NetHack (messages, status changes, deaths) is governed by an exact C function. Isolate the exact message string from the final 3 lines of `game_log` (e.g. "You feel hungry", "The door resists", "The jackal bites!"), strip proper nouns, and pass that phrase to `query_wiki(query=...)` to learn the exact C rule that caused it.
     * Call `query_db(sql=...)` to inspect the failed runs, stall causes, or death reasons from the previous batch (`runs` table), or compare with how experts handled this stage in `expert_runs` / `expert_ticks`.
   - Ground your hypothesis in empirical game facts, not blind intuition.

2. **Stage 2: Focus on 1 Primary Skill or Modular Subsystem**:
   - Target **1 primary skill** (1 specialist method in `Policy`, OR an external module under `skills/<skill_module>.py`).
   - If adding or updating a modular skill module, specify `skill_module: "<name>"` in your YAML hypothesis and import it in `policy.py` (e.g. `from skills.combat import melee_strike`).
   - Modifying 1 method costs only 150 triggered episodes and isolates causal attribution; modifying multiple unrelated methods inflates rollout budgets and couples unrelated bugs.
   - **Anti-Hallucination & Dungeon Context Anchor**: Anchor your reasoning to your current dungeon depth (`state.depth`). On Dungeons of Doom levels 1–4, there are NO Sokoban boulders, NO Medusa stoning, and NO Quest portals. If the hero is bumping into walls, inspect `get_passable_mask` and verify `ord('-')` and `ord('|')` are strictly marked False.
   - NetHack is an interrupt-driven survival loop: specialist methods (`explore_step`, `handle_combat`, `handle_nutrition`, etc.) must return an action or `None` on each tick and rely on persistent attributes on `self` (`self.target_tile`, `self.pos_history = collections.deque(maxlen=8)`) to avoid 2-tile ping-pong oscillations.

3. **Stage 3: 30-Episode Pilot Verification (`dry_run_policy`)**:
   - Test your candidate code using `dry_run_policy(code=..., skill_name="<name>", skill_code="...")`.
   - It runs a **30-episode parallel pilot test** against the current baseline on 16 workers in ~1s, reporting paired Wins/Losses/Ties and the exact logs of the worst stalled runs.
   - Any episode that does NOT call your modified/added method or skill is immediately discarded (`triggered == False`). Ensure your method is called by `run()`.

4. **Stage 4: Pilot Failure Protocol (If Pilot Fails)**:
   - If `dry_run_policy` returns `PILOT_FAILED`, **STOP! Do NOT blindly guess another code variation.**
   - Inspect the failed pilot logs and behavioral critic warnings.
   - Call `query_wiki` or `query_db` to look up the exact mechanic that caused the failure before synthesizing a revised method. You have up to 5 pilot revisions per generation.

5. **Stage 5: Pre-Submission Invariant Checklist**:
   - Before outputting code, mentally verify all 4 invariants:
     * *Generator Invariant*: `run()` receives state via `state = yield act`. Subgenerators are called via `state = yield from self.subgen(state)`. Never use a bare `return act` inside a generator.
     * *State Persistence*: Specialist methods must update `self.*` attributes (e.g. `self.target_tile`, `self.tried_doors`) to prevent 2-tile ping-pong oscillations.
     * *Passability Mask*: `get_passable_mask` must never mark `ord('-')` (horizontal wall) or `ord('|')` (vertical wall) as passable.
     * *Trigger Reachability*: Any modified or newly added specialist method must be called from `run()`.

6. **Output Format (Once Ready to Test or Submit)**:
   - Output a `yaml` hypothesis block detailing your qualitative causal hypothesis and skill spec:
     ```yaml
     mode: FUNC_MOD_ADD
     target_functions: [method_you_modified_or_added]
     binding_constraint: "Largest single failure cause from batch digest (e.g. 74% starved on Dlvl 1)"
     verified_c_rule: "Exact C engine rule discovered via query_wiki/query_db"
     mechanism: "Exact behavioral change introduced by your skill modification"
     state_persistence: "What self.* attributes prevent oscillation and track targets"
     skill_module: "optional_skill_name"
     ```
   - Output your modified and/or newly added method(s) (or `class Policy:`) in a ```python ... ``` block. If providing an external skill module, provide its code in a second ```python ... ``` block labeled with `# skill_module: <name>`.
"""
