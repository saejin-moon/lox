"""
System & User Prompts for the Stateless Author Agent ("Hard Code Nothing").
In System Prompt:
- Complete Policy Language Specification (Python 3.11+, GameState API, 1-to-1 Action Map, Macros, Tunable Params)
- Complete DuckDB Schema (`runs`, `ticks`, `batches`, `function_stats`, `experiments`, `expert_runs`, `expert_ticks`)
In User Prompt:
- Active Modification Mode (`CREATE`, `WRITE`, `REFACTOR`)
- Batch Contrast Digest (`is_outlier_good = TRUE` runs vs normal runs), Function Stats, and Recent Experiment Lab Notebook.
"""

from __future__ import annotations

import ast
import json
from typing import Any

from lox.db.schema import get_schema_markdown
from lox.lang.validator import (
    audit_policy_ast_hygiene,
    extract_module_outline,
    normalize_mode,
)

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
  - **Zero-Turn Action & Stall Telemetry (`state.turn_advanced`, `state.consecutive_stalls`)**:
    In NetHack, actions that fail (bumping into a peaceful creature without confirmation, attempting to lift an overweight corpse, kicking an unkickable door) consume **0 game turns** (`state.turn` does not advance).
    - `state.turn_advanced: bool`: Returns `True` if the previous action advanced game time, `False` if stalled.
    - `state.consecutive_stalls: int`: Returns the count of consecutive steps that failed to advance game time. If your action failed (`state.consecutive_stalls > 0`), **do not blindly repeat the same action**! Runs exceeding 100 consecutive zero-turn steps are truncated as `stalled_zero_turn`.
  - **Interactive Prompts (`state.in_yn_prompt`, `state.in_text_prompt`)**:
    - When NetHack prompts for confirmation (e.g. `Really attack <name>? [yn]`, `Continue lifting? [ynq]`, `Are you sure you want to pray? [yn]`), `state.in_yn_prompt` is `True`. You can respond on the next step by yielding `'y'` or `'n'`.
    - When NetHack prompts for text input (e.g. `Call a scroll: `, `Call a potion: `), `state.in_text_prompt` is `True`.
      Yield `Action.ESC` (`'\x1b'`) to dismiss text naming prompts immediately!
    - The harness includes a safety fallback auto-ESC/auto-N after 3 unhandled turns to prevent permanent simulation lockups.
  - **3 Strict Generator Rules**:
    1. In a generator (`def run` containing `yield`), ALWAYS receive the next state via `state = yield act` (never `return act` inside a generator—in Python `return val` inside a generator raises `StopIteration`!).
    2. If a helper method is a normal function that returns an `Action` or `None`, call it as `act = self.helper(state)` and then `state = yield act`.
    3. If a helper method is itself a generator (contains `yield`), call it via `state = yield from self.sub_gen(state)` and end `sub_gen` with `return state`. Never write `yield self.sub_gen(state)` without `from`!
- **Supported Python**: Full Python 3.11+ syntax (`if/elif/else`, `for`, `while`, `try/except`, list/dict/set comprehensions, `@dataclass`, helper classes, recursion, `match/case`, lambdas). Pre-loaded in globals: `math`, `collections` (`deque`, `defaultdict`, `Counter`), `heapq`, `itertools`, `functools`, `re`, `enum`, `dataclasses`, `numpy` (`np`), `random`, `copy`, `bisect`.
- **Configurable Parameters (`self.param`)**:
  You can declare numeric thresholds/weights via `self.param("name", default, min_val, max_val)` (e.g. `self.param("heal_hp_ratio", 0.35, 0.10, 0.80)`).

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
  - In `WRITE` or `REFACTOR`, you can create or update a modular skill module by specifying `skill_module: "mod_name"` in your YAML hypothesis and providing the skill code.
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
  - Helper properties: `item.is_weapon`, `item.is_armor`, `item.is_food`, `item.is_corpse`, `item.is_worn`, `item.is_wielded`, `item.is_potion`, `item.is_scroll`, `item.is_wand`, `item.is_ring`, `item.is_amulet`, `item.is_tool`.
  - Status: `item.is_blessed`, `item.is_uncursed`, `item.is_cursed`.
  - C struct metadata: `item.info` (`ObjectInfo | None`: `name`, `descr`, `weight`, `cost`, `delay`, `oprop`) and `item.corpse_monster` (`MonsterInfo | None` if `is_corpse`).
- **Visible Monsters (`state.visible_monsters() -> list[VisibleMonster]`) and Adjacent Monsters (`state.adjacent_monsters(exclude_pets: bool = True) -> list[VisibleMonster]`)**:
  Each `m` in `state.visible_monsters()` has:
  - `m.y`, `m.x`, `m.name`, `m.char`, `m.color`.
  - `m.is_adjacent_to(state.y, state.x)` (8-way Chebyshev adjacency—monsters attack diagonally as well as orthogonally!), `m.is_pet`, `m.is_statue`, `m.is_invisible`.
  - `m.info` (`MonsterInfo` 1-to-1 C `permonst` struct): `m.info.difficulty`, `m.info.ac`, `m.info.speed`, `m.info.level`, `m.info.mr`, `m.info.nutrition`, `m.info.weight`, `m.info.size`, `m.info.resistances_mask`, `m.info.conveys_mask`, `m.info.flags1`, `m.info.flags2`, and C bitmask properties: `is_acidic`, `is_poisonous`, `is_undead`, `is_were`, `is_human`, `is_elf`, `is_dwarf`, `is_gnome`, `is_orc`, `is_demon`, `is_giant`, `is_shapeshifter`, `is_domestic`, `conveys_poison_res`, `conveys_fire_res`, `conveys_cold_res`, `conveys_sleep_res`, `conveys_shock_res`, `conveys_disint_res`, `conveys_stone_flag`, `conveys_see_invis`, `conveys_teleport`, `conveys_teleport_control`.
  `state.adjacent_monsters(exclude_pets=True)` returns all visible non-pet monsters in the 8 Chebyshev adjacent squares around the hero.
  *Note on Peaceful vs. Hostile Monsters*: In NetHack, the C screen glyphs do NOT expose whether a monster is currently hostile or peaceful. Peaceful creatures (e.g. domestic animals, shopkeepers, Minetown watchmen, co-aligned priests) spawn peaceful; bumping into them triggers a `Really attack <name>? [yn]` prompt. Your policy must handle peaceful vs hostile distinctions dynamically (handling prompts, tracking who attacked in messages, or using wiki monster data).
- **Visible Objects & Corpses (`state.visible_objects() -> list[VisibleObject]`) and Objects Here (`state.objects_here() -> list[VisibleObject]`)**:
  `state.visible_objects()` returns all visible items on the floor (including items on the hero's tile `(state.y, state.x)`).
  `state.objects_here()` returns only items directly on the hero's tile `(state.y, state.x)` (e.g. to inspect before pickup or eating).
  Each `o` in `state.visible_objects()` / `state.objects_here()` has:
  - `o.y`, `o.x`, `o.name`, `o.char`, `o.color`.
  - `o.is_corpse`, `o.is_food`, `o.is_weapon`, `o.is_armor`, `o.is_potion`, `o.is_scroll`, `o.is_wand`.
  - `o.info` (`ObjectInfo | None`: `name`, `descr`, `weight`, `cost`, `delay`, `oprop`) and `o.corpse_monster` (`MonsterInfo | None` if `is_corpse`: inspect `nutrition`, `is_poisonous`, `is_acidic`, `is_undead`, `is_were`, `conveys_poison_res`, etc. directly from C `permonst` without hardcoding monster name lists!).
  - You can also look up any monster or object by name in O(1) via `state.monster_info_by_name(name)` and `state.object_info_by_name(name)`.
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

1. **Floor Residence & Physical State Contrast Query (Compare Turns per Depth vs Experts)**:
   ```sql
   -- Contrast turns spent per depth between expert ascensions and champion runs
   -- 1A. How many turns do expert ascensions spend exploring Dlvl 1, 2, 3?
   SELECT depth, COUNT(DISTINCT run_id) AS runs, AVG(max_t - min_t) AS avg_turns_on_floor
   FROM (
       SELECT run_id, depth, MIN(turn) AS min_t, MAX(turn) AS max_t
       FROM expert_ticks
       WHERE depth <= 4
       GROUP BY run_id, depth
   ) GROUP BY depth ORDER BY depth;

   -- 1B. Physical deficit and total turns at failure depth:
   SELECT 'champion_failures' AS source, AVG(final_ac) AS avg_ac, AVG(max_hp) AS avg_hp, AVG(turns) AS avg_turns
   FROM champion_runs WHERE depth = 2
   UNION ALL
   SELECT 'expert_ascensions' AS source, AVG(ac) AS avg_ac, AVG(hp) AS avg_hp, AVG(turn) AS avg_turns
   FROM expert_ticks WHERE depth = 2;
   ```
   *Interpretation*: If expert runs spend ~40 turns on Dlvl 1 and ~85 turns on Dlvl 2 while your policy spends thousands of turns on Dlvl 1, your policy is chronically lingering instead of taking stairs down when found. If experts have AC <= 4 and your runs have AC 10, your policy is not equipping armor. If experts have 2x higher HP/XL, your policy is taking unnecessary melee damage or failing to level up.

2. **Monster Encounter & Tactical Response Query**:
   ```sql
   -- How did expert ascensions act when encountering specific hazards or depths?
   SELECT action_name, COUNT(*) AS freq
   FROM expert_ticks
   WHERE depth <= 4
   GROUP BY action_name ORDER BY freq DESC LIMIT 8;
   ```

3. **Outlier & Failure Trace Inspection Query (Full Lossless Ticks & Specialist Attribution)**:
   ```sql
   -- 3A. Compare top outlier runs vs worst failures in the active champion batch
   SELECT run_id, seed, depth, max_depth, final_ac, final_xp_level, turns, last_active_function, end_reason, summary_digest
   FROM champion_runs ORDER BY max_depth DESC, turns DESC LIMIT 5;

   -- 3B. Inspect exact step-by-step pre-mortem ticks (100% full lossless ticks with inventory & monster alignment)
   SELECT turn, depth, hp, max_hp, ac, active_function, action_name, message, visible_monsters, inventory_summary
   FROM champion_ticks
   WHERE run_id = '<run_id_from_champion_runs>'
   ORDER BY turn DESC LIMIT 20;

   -- 3C. Inspect ticks from your most recent dry_run_policy pilot batch
   SELECT r.seed, t.turn, t.depth, t.hp, t.ac, t.active_function, t.action_name, t.message, t.visible_monsters, t.inventory_summary
   FROM latest_pilot_ticks t
   JOIN latest_pilot_runs r ON t.run_id = r.run_id
   WHERE r.seed = 42
   ORDER BY t.turn DESC LIMIT 20;
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
3. `query_monster(name)`: Query the exact C-level engine `permonst` struct for any monster (e.g. `jackal`, `floating eye`, `lichen`, `orc`). Returns exact level, AC, movement speed, magic resistance, corpse weight, corpse nutrition, conveyed intrinsics (e.g. poison resistance), and monster innate resistances.
4. `query_item(name)`: Query the exact C-level engine `objclass` struct for any item (e.g. `food ration`, `potion of extra healing`, `long sword`). Returns exact object class, cost, weight, and nutrition value.
5. `query_policy_docs(topic="all")`: View the policy language specification and the full source/docstrings of all macros currently in `macros.py` (or call `query_policy_docs("actions")` for the complete command index).
6. `read_skill(skill_name)`: Retrieve the complete Python source code of any modular skill package currently residing in `skills/` (e.g. `skills/survival.py`, `skills/navigation.py`).
7. `dry_run_policy(code, is_macro=False, seed=42, max_turns=150, replay_run_id="", replay_steps=0)`:
   - Tests candidate Python code in a live NetHack sandbox.
   - **Step 1 (Smoke Check & Behavioral Critic)**: Verifies AST, checks reachability, detects oscillations and search spam. Optional `replay_run_id` fast-forwards into any recorded mid-game run in ~10ms.
   - **Step 2 (30-Episode Parallel Pilot Batch)**: Dispatches 30 episodes across persistent workers (up to 8 pilot revisions per generation, with fallback to best candidate) paired 1-to-1 against current baseline seeds, returning paired Wins/Losses/Ties, Δ depth, and concrete game logs of the worst stalled runs.

### Scientific Investigation Protocol (MANDATORY)
1. **Never Guess from Intuition**: NetHack mechanics are strictly ruled by the C engine (e.g. traps, secret doors, searching probabilities, hunger increments, prayer timeouts, pet displacement, monster ranged attacks). Guessing code changes without empirical research causes catastrophic regressions.
2. **Mechanistic Trace Inversion via `query_wiki`**:
   Every event in NetHack (messages, status changes, deaths) is governed by an exact C function. Isolate the exact message string from the final 3 lines of `game_log` (e.g. "You feel hungry", "The door resists", "The jackal bites!"), strip proper nouns, and pass that phrase to `query_wiki(query=...)` to learn the exact C rule that caused it.
3. **The Binding Constraint Rule**:
   Always inspect the Termination Breakdown and Stall Diagnostics in the batch digest. Address the single failure mode responsible for the LARGEST percentage of failures (the binding constraint). Never spend a generation fixing a 2% edge case when 70% of runs died to the same primary cause.
4. **Empirical Root-Cause Discovery**:
   Inspect the raw termination breakdown, death counts, and RLE phase timelines in the batch digest and DuckDB.
   Do not guess which subsystem failed based on preconceived notions or simple turn counts. Formulate targeted SQL queries (`query_db`) over `runs` and `ticks` to inspect the exact state, position, messages, and action history in the turns leading up to death or timeout before writing code.
5. **Anti-Hallucination & Dungeon Context Anchor**:
   Anchor your reasoning to your current dungeon depth (`state.depth`). On Dungeons of Doom levels 1–4, there are NO Sokoban boulders, NO Medusa stoning, and NO Quest portals. Use `state.mem.passable_mask` (or ensure `ord('-')`, `ord('|')`, and `ord(' ')` are strictly False in custom masks) and remember monsters attack across all 8 adjacent tiles (`max(abs(dy), abs(dx)) == 1`).
6. **Pre-Submission Invariant Checklist**:
   Before outputting code, mentally verify all 4 invariants:
   - *Generator Invariant*: `run()` receives state via `state = yield act`. Subgenerators are called via `state = yield from self.subgen(state)`. Never use a bare `return act` inside a generator.
   - *State Persistence*: Specialist methods must update `self.*` attributes (e.g. `self.target_tile`, `self.tried_doors`) to prevent 2-tile ping-pong oscillations.
   - *Passability & 8-Way Adjacency*: Never mark `ord('-')`, `ord('|')`, or `ord(' ')` (solid stone) as passable; prefer `state.mem.passable_mask`. Always check 8-way adjacency (`max(abs(dy), abs(dx)) == 1` or `state.adjacent_monsters()`) in combat and door interactions.
   - *Trigger Reachability*: Any modified or newly added specialist method must be called from `run()`.
"""


def _format_context_scoped_policy(current_policy: str, batch_digest: str = "") -> str:
    hygiene_warns = audit_policy_ast_hygiene(current_policy)
    hygiene_block = ""
    if hygiene_warns:
        hygiene_block = (
            "### ⚠️ Policy AST Hygiene Diagnostics:\n"
            + "\n".join(f"- {w}" for w in hygiene_warns)
            + "\n\n"
        )
    lines = current_policy.splitlines()
    if len(lines) <= 1200:
        return (
            f"{hygiene_block}### Current `policy.py`:\n```python\n{current_policy}\n```"
        )

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

    stagnation_block = ""
    if stagnation_function:
        stagnation_block = (
            f"\n### 🛑 ACTIVE FUNCTION COOLDOWN BAN:\n"
            f"The function `{stagnation_function}` has been targeted repeatedly in recent rejected generations without progress.\n"
            f"**DO NOT target `{stagnation_function}` this generation.** You MUST investigate and target a completely different subsystem or capability!\n"
        )

    exp_block = (
        f"\n### Recent Experiment Lab Notebook (`experiments` table):\n{recent_experiments}\n"
        if recent_experiments
        else ""
    )
    digest_block = f"\n{batch_digest}\n" if batch_digest else ""
    canon_mode = normalize_mode(mode)

    if canon_mode == "CREATE":
        return f"""### Generation {generation} — Mode: `CREATE` (Initial Baseline Synthesis)
{stagnation_block}{discoveries_block}{failures_block}You are bootstrapping the policy from a **completely clean slate** (no pre-written policy is provided).
Your task:
1. Use multiple tool calls (`query_db` on `expert_runs` / `expert_ticks`, `query_wiki`, `query_policy_docs`) to study successful NetHack runs, early-game survival mechanics, dungeon navigation, door opening/kicking, stair descent, combat, and food/corpse consumption.
2. Design a well-structured, modular `class Policy:` with clear methods (e.g. `run(self, state)`, `update_memory(self, state)`, `handle_combat(self, state)`, `handle_nutrition(self, state)`, `explore_level(self, state)`, etc.).
3. Pre-Submission Invariant Checklist:
   - Generator Invariant: `run()` receives state via `state = yield act`. Subgenerators are called via `state = yield from self.subgen(state)`. Never use a bare `return act` inside a generator.
   - State Persistence: Specialist methods must update `self.*` attributes to prevent oscillations.
   - Passability Mask: `get_passable_mask` must never mark `ord('-')` or `ord('|')` as passable.
   - Trigger Reachability: Any specialist method must be called from `run()`.
4. Test your full `class Policy:` using `dry_run_policy(code=...)` to verify there are zero syntax/attribute errors or 0-yield infinite loops.
5. Output Format:
   - Output your YAML hypothesis inside `<hypothesis> ... </hypothesis>` tags:
     <hypothesis>
     mode: CREATE
     target_functions: [run]
     binding_constraint: "Initial baseline bootstrap"
     verified_c_rule: "Core survival & navigation rules identified via expert_runs and wiki"
     mechanism: "Summary of initial Policy architecture and state tracking"
     state_persistence: "Self.* attributes used for memory and navigation"
     </hypothesis>
   - Output your complete `class Policy:` implementation inside `<policy_code> ... </policy_code>` tags (or standard ```python ... ``` block).
"""

    if canon_mode == "REFACTOR":
        return f"""### Generation {generation} — Mode: `REFACTOR` (Subsystem Modularization into Skills)
{stagnation_block}{digest_block}{exp_block}{discoveries_block}{failures_block}
{skills_block}
{_format_context_scoped_policy(current_policy, batch_digest)}

### Scientific Protocol for `REFACTOR`:
The policy has grown into a large monolith (or reached a structural threshold). Your **primary objective** in this generation is **mandatory modular decomposition into `skills/<module>.py`**.

1. **Stage 1: Mandatory Subsystem Extraction**:
   - You **MUST** extract at least one cohesive subsystem out of `policy.py` into a separate module under `skills/<module>.py` (e.g. `skills/combat.py`, `skills/navigation.py`, `skills/nutrition.py`, or `skills/equipment.py`).
   - In `policy.py`, import the extracted methods or functions (e.g. `from skills.combat import handle_combat`) and remove the extracted method bodies from `class Policy` (or delegate cleanly so `policy.py` shrinks).
   - You must specify `skill_module: "<name>"` in your YAML hypothesis block.

2. **Stage 2: CRITICAL INVARIANT: Zero Behavioral Regression (Behavioral Parity Gate)**:
   - Refactoring is for **code modularization and architectural hygiene, NOT unverified behavioral experimentation**.
   - Do **NOT** rewrite navigation rules, change descent gates, or alter survival thresholds while refactoring!
   - Keep the extracted skill's logic **behaviorally identical** to the existing working implementation in `policy.py`.
   - `REFACTOR` candidates are evaluated against a **Behavioral Parity Gate** (0 crashes and non-inferior progression vs baseline). Breaking working logic will result in immediate rejection.

3. **Stage 3: Verification with `dry_run_policy`**:
   - Test your candidate policy and extracted skill module using:
     `dry_run_policy(code=policy_code, skill_name="<name>", skill_code=skill_module_code)`
   - Verify that imports resolve cleanly, there are zero syntax or attribute errors, and the policy passes the Behavioral Parity Gate.

4. **Stage 4: Commit Output**:
   - Output your YAML hypothesis inside `<hypothesis> ... </hypothesis>` tags:
     <hypothesis>
     mode: REFACTOR
     target_functions: [run]
     skill_module: "extracted_skill_name"
     binding_constraint: "Monolithic code modularization"
     verified_c_rule: "Behavioral parity: exact preservation of existing logic"
     mechanism: "Extracted cohesive subsystem from policy.py into skills/<name>.py and imported it"
     state_persistence: "Attributes on self used by the extracted skill"
     </hypothesis>
   - Block 1: Output your refactored `class Policy:` (importing the skill) inside `<policy_code> ... </policy_code>` tags (or ```python ... ``` block).
   - Block 2: Output your new/updated skill module inside `<skill_code module="<name>"> ... </skill_code>` tags (or labeled ```python ... ``` block).
"""

    # Default: WRITE
    return f"""### Generation {generation} — Mode: `WRITE` (Targeted Single-Skill Addition / Modification)
{stagnation_block}{digest_block}{exp_block}{discoveries_block}{failures_block}
{skills_block}
{_format_context_scoped_policy(current_policy, batch_digest)}

### Scientific Investigation & 4-Step Deductive Protocol for `WRITE`:

1. **Step 1: MANDATORY Empirical Research (Turn 1) — Autonomous Database Exploration**:
   - **DO NOT OUTPUT CODE ON TURN 1.**
   - Do not rely uncritically on heuristic summaries ("Hard Code Nothing"). Run `query_db(sql=...)` on `champion_runs`, `champion_ticks`, and `champion_function_stats` to inspect what actually happened in the baseline episodes.
   - Query what caused deaths and stalls (e.g. `SELECT end_reason, COUNT(*), AVG(depth), AVG(turns), AVG(pct_wait), AVG(pct_search) FROM champion_runs GROUP BY end_reason ORDER BY COUNT(*) DESC`).
   - Query `champion_function_stats` to see Ochiai Good-Run Similarity vs. Liblit Failure Importance for each method.
   - Check where the hero spent turns (e.g. `SELECT depth, AVG(turns) FROM champion_runs GROUP BY depth`), whether food was present in inventory (`starved_with_food`), and which functions consumed turns.

2. **Step 2: Pre-Mortem & Full-Tick Trace to Exact Code Line**:
   - Run `query_db` on `champion_ticks` (which stores **100% full, lossless step-by-step ticks** with `inventory_summary` and `visible_monsters`) to trace the exact 10–20 turns before death or stall.
   - Cross-reference those ticks with the exact lines of `Current policy.py` (or `read_skill(...)`) to pinpoint the exact conditional branch, missing guard, or threshold that caused the specialist function to make the wrong decision.

3. **Step 3: C-Mechanic & Expert Verification (`query_wiki` / `expert_runs`)**:
   - Isolate the exact message or monster/item name from `game_log` and call `query_wiki(query=...)` or `query_db` on `expert_runs` / `expert_ticks` to verify the exact NetHack 3.6.6 C mechanic before writing code.
   - Target **1 primary skill** (1 specialist method in `Policy`, OR an external module under `skills/<skill_module>.py`). In WRITE mode, modify at most 2 functions (e.g. 1 primary method + 1 caller/helper). Modifying 3+ functions is strictly rejected as a mutation scope violation to isolate causality and prevent cross-subsystem regressions. Broad rewrites belong in STRUCTURAL_REFACTOR mode.

4. **Step 4: Paired Seed Counterfactual Verification (`dry_run_policy`)**:
   - Test your candidate code using `dry_run_policy(code=..., skill_name="<name>", skill_code="...")`.
   - If the pilot reports any **Paired Seed Regressions (`Candidate < Baseline`)**, inspect the side-by-side seed comparison (`Baseline [Dlvl, XL, AC, Steps, End] -> Candidate [Dlvl, XL, AC, Steps, ActiveFn, End]`) and query `latest_pilot_ticks` for that seed (`WHERE r.seed = <seed>`) to see why your change hurt that seed before revising!

5. **Pre-Submission Invariant Checklist**:
   - *Generator Invariant*: `run()` receives state via `state = yield act`. Subgenerators are called via `state = yield from self.subgen(state)`. Never use a bare `return act` inside a generator.
   - *State Persistence*: Specialist methods must update `self.*` attributes (e.g. `self.target_tile`, `self.tried_doors`) to prevent 2-tile ping-pong oscillations.
   - *Passability & 8-Way Adjacency*: Never mark `ord('-')`, `ord('|')`, or `ord(' ')` as passable; prefer `state.mem.passable_mask`. Always check 8-way adjacency (`max(abs(dy), abs(dx)) == 1` or `state.adjacent_monsters()`).
   - *Trigger Reachability*: Any modified or newly added specialist method must be called from `run()` / `dispatch()`.

6. **Output Format (Once Ready to Test or Submit)**:
   - Output your YAML hypothesis inside `<hypothesis> ... </hypothesis>` tags:
     <hypothesis>
     mode: WRITE
     target_functions: [method_you_modified_or_added]
     binding_constraint: "Empirical bottleneck identified via query_db on champion_runs"
     verified_c_rule: "Exact C engine rule discovered via query_wiki/query_db"
     mechanism: "Exact behavioral change introduced by your skill modification"
     state_persistence: "What self.* attributes prevent oscillation and track targets"
     skill_module: "optional_skill_name"
     </hypothesis>
   - Output your modified and/or newly added method(s) (or `class Policy:`) inside `<policy_code> ... </policy_code>` tags (or standard ```python ... ``` block).
   - If providing an external skill module, provide its code inside `<skill_code module="<name>"> ... </skill_code>`.
"""
