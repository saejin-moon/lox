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

from lox.db.schema import get_schema_markdown

POLICY_LANG_DOCS = r"""### Policy Language & 1-to-1 Harness Reference

#### 1. Execution Model (`class Policy` and `macros.py`)
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
  - **3 Strict Generator Rules**:
    1. In a generator (`def run` containing `yield`), ALWAYS receive the next state via `state = yield act` (never `return act` inside a generator—in Python `return val` inside a generator raises `StopIteration`!).
    2. If a helper method is a normal function that returns an `Action` or `None`, call it as `act = self.helper(state)` and then `state = yield act`.
    3. If a helper method is itself a generator (contains `yield`), call it via `state = yield from self.sub_gen(state)` and end `sub_gen` with `return state`. Never write `yield self.sub_gen(state)` without `from`!
- **Supported Python**: Full Python 3.11+ syntax (`if/elif/else`, `for`, `while`, `try/except`, list/dict/set comprehensions, `@dataclass`, helper classes, recursion, `match/case`, lambdas). Pre-loaded in globals: `math`, `collections` (`deque`, `defaultdict`, `Counter`), `heapq`, `itertools`, `functools`, `re`, `enum`, `dataclasses`, `numpy` (`np`), `random`, `copy`, `bisect`.
- **Tunable Parameters (`self.param`)**:
  Declare numeric thresholds/weights via `self.param("name", default, min_val, max_val)` (e.g. `self.param("heal_hp_ratio", 0.35, 0.10, 0.80)`). During `PARAM_TUNE` generations, the engine optimizes all registered parameters automatically via SPSA!

#### 2. 1-to-1 Actions (`Action`, `key`, `seq`, `DIR_DELTAS`)
Nothing is hardcoded in the harness: whatever you `yield` (or `return` from `step`) maps 1-to-1 to the exact keystroke(s) sent to NetHack:
- **Compass Movement (`Action` or `(dy, dx)` via `DIR_DELTAS`)**:
  `Action.N` (`'k'`, `(-1, 0)`), `Action.S` (`'j'`, `(1, 0)`), `Action.E` (`'l'`, `(0, 1)`), `Action.W` (`'h'`, `(0, -1)`),
  `Action.NE` (`'u'`, `(-1, 1)`), `Action.NW` (`'y'`, `(-1, -1)`), `Action.SE` (`'n'`, `(1, 1)`), `Action.SW` (`'b'`, `(1, -1)`).
  You can also look up `DIR_DELTAS[(dy, dx)]` directly!
- **Vertical & Wait**:
  `Action.UP` / `Action.ASCEND` (`'<'`), `Action.DOWN` / `Action.DESCEND` (`'>'`), `Action.WAIT` (`'.'`), `Action.MORE` (`'\\r'`).
- **All Native NetHack Commands**:
  `Action.SEARCH` (`'s'`), `Action.OPEN` (`'o'`), `Action.CLOSE` (`'c'`), `Action.KICK` (`C('d')`),
  `Action.EAT` (`'e'`), `Action.QUAFF` (`'q'`), `Action.READ` (`'r'`), `Action.ZAP` (`'z'`),
  `Action.WEAR` (`'W'`), `Action.TAKEOFF` (`'T'`), `Action.WIELD` (`'w'`), `Action.SWAP` (`'x'`),
  `Action.PUTON` (`'P'`), `Action.REMOVE` (`'R'`), `Action.QUIVER` (`'Q'`), `Action.FIRE` (`'f'`), `Action.THROW` (`'t'`),
  `Action.PICKUP` (`','`), `Action.DROP` (`'d'`), `Action.LOOT` (`M('l')`), `Action.APPLY` (`'a'`),
  `Action.PRAY` (`M('p')`), `Action.ENGRAVE` (`'E'`), `Action.DIP` (`M('d')`), `Action.OFFER` (`M('o')`),
  `Action.CHAT` (`M('c')`), `Action.PAY` (`'p'`), `Action.ENHANCE` (`M('e')`), `Action.UNTRAP` (`M('u')`),
  `Action.FIGHT` (`'F'`), `Action.MOVE` (`'m'`), `Action.ESC` (`'\\x1b'`), `Action.SPACE` (`' '`).
- **Multi-Keystroke Sequences (`seq(...)`)**:
  Many NetHack actions prompt for a direction, inventory slot letter, or text. Send the full sequence in one turn using `seq(...)`:
  - Open a door at `(dy, dx)`: `yield seq(Action.OPEN, DIR_DELTAS[(dy, dx)])`
  - Kick a door at `(dy, dx)`: `yield seq(Action.KICK, DIR_DELTAS[(dy, dx)])`
  - Eat inventory item in slot `'f'`: `yield seq(Action.EAT, 'f')`
  - Eat a corpse on the floor (`'y'` to confirm): `yield seq(Action.EAT, 'y')`
  - Quaff potion in slot `'h'`: `yield seq(Action.QUAFF, 'h')`
  - Wear armor in slot `'d'`: `yield seq(Action.WEAR, 'd')`
  - Wield weapon in slot `'a'`: `yield seq(Action.WIELD, 'a')`
  - Throw item `'b'` in direction `(dy, dx)`: `yield seq(Action.THROW, 'b', DIR_DELTAS[(dy, dx)])`
  - Zap directional wand `'i'` in direction `(dy, dx)`: `yield seq(Action.ZAP, 'i', DIR_DELTAS[(dy, dx)])`
  - Engrave Elbereth in dust with fingers (`'-'`): `yield seq(Action.ENGRAVE, '-', 'Elbereth\\r')`
  - Dip item `'a'` into fountain (`'y'`): `yield seq(Action.DIP, 'a', 'y')`

#### 3. Querying `GameState` (`state`)
Every turn, `state` exposes 100% of NetHack's observation arrays and C symbol tables:
- **Hero Vitals & Coordinates**:
  `state.y` (0..20), `state.x` (0..78), `state.pos` (`(y, x)`),
  `state.hp`, `state.max_hp`, `state.energy`, `state.max_energy`, `state.ac`,
  `state.xp_level`, `state.xp`, `state.score`, `state.gold`, `state.turn`, `state.step_count`,
  `state.depth` (dungeon depth 1..50), `state.dungeon_num` (`0`=Dungeons of Doom, `1`=Gehennom, `2`=Gnomish Mines, `3`=Quest, `4`=Sokoban),
  `state.strength`, `state.dex`, `state.con`, `state.intel`, `state.wis`, `state.cha`, `state.align` (`1`=Lawful, `0`=Neutral, `-1`=Chaotic),
  `state.hunger_state` (`0`=Satiated, `1`=Not Hungry, `2`=Hungry, `3`=Weak, `4`=Fainting, `5`=Fainted),
  `state.carrying_capacity` (`0`=Unencumbered, `1`=Burdened, `2`=Stressed, `3`=Strained, `4`=Overtaxed, `5`=Overloaded).
- **Status Flags**:
  `state.is_blind`, `state.is_confused`, `state.is_stunned`, `state.is_hallucinating`, `state.is_levitating`,
  `state.is_food_poisoned`, `state.is_terminally_ill`, `state.is_stoning`, `state.is_slimed`, `state.is_strangled`, `state.condition_mask`.
- **Messages & Screen Grids (21x79 NumPy Arrays)**:
  - `state.message` (`str`): Current turn's game message(s).
  - `state.chars` (`uint8[21, 79]`): ASCII character codes on screen (e.g. `ord('.')` floor, `ord('#')` corridor, `ord('+')` closed door, `ord('>')` stairs down, `ord('<')` stairs up, `ord('{')` fountain, `ord('_')` altar, `ord('^')` trap).
  - `state.char_at(y, x) -> str`: Single ASCII character at `(y, x)`.
  - `state.glyphs` (`int32[21, 79]`): Raw NLE glyph IDs.
  - `state.colors` (`int8[21, 79]`), `state.specials` (`uint8[21, 79]`).
  - `state.ascii_map() -> str`: Full 21x79 ASCII rendering of the screen.
- **Inventory (`state.inventory -> list[InventoryItem]`)**:
  Each `item` has: `item.letter` (`'a'..'z'`), `item.name` (full description string, e.g. `'a +1 long sword (weapon in hand)'`, `'2 food rations'`), `item.glyph` (`int`), `item.oclass` (`int`), `item.char` (`str`: `')'` weapon, `'['` armor, `'%'` food, `'!'` potion, `'?'` scroll, `'/'` wand, `'='` ring, `'\"'` amulet, `'('` tool, `'$'` gold), `item.quantity` (`int`).
- **Visible Monsters (`state.visible_monsters() -> list[VisibleMonster]`)**:
  Each `m` has: `m.y`, `m.x`, `m.glyph`, `m.char`, `m.color`, `m.is_pet` (`bool`), `m.is_peaceful` (`bool`), and `m.info` (`MonsterInfo` directly from NLE's C `permonst` table: `m.info.name`, `m.info.symbol`, `m.info.level`, `m.info.speed`, `m.info.ac`, `m.info.mr`, `m.info.difficulty`, `m.info.nutrition`, `m.info.weight`, `m.info.resistances_mask`, `m.info.conveys_mask`, `m.info.attacks` tuple of `(aatyp, adtyp, damn, damd)` where `aatyp == 0` indicates passive gaze/paralysis/reflection attacks like floating eyes!).
- **Visible Objects & Corpses (`state.visible_objects() -> list[VisibleObject]`)**:
  Each `o` has: `o.y`, `o.x`, `o.glyph`, `o.char`, `o.color`, `o.is_corpse` (`bool`), `o.corpse_monster` (`MonsterInfo | None` if `is_corpse`, with `o.corpse_monster.name`, `o.corpse_monster.nutrition`, etc.), `o.info` (`ObjectInfo | None`). *(Note: `VisibleObject` on the ground does NOT have `.name`; use `o.char`, `o.is_corpse`, and `o.corpse_monster`. Once picked up into `state.inventory`, `item.name` gives the full string.)*
- **Fast Vectorized NumPy & Numba Pathfinding Helpers (Always prefer vectorized NumPy over nested 21x79 Python loops)**:
  - `state.bfs_distances(passable_mask, start=(y, x), allow_diagonal=True) -> int32[21, 79]`: Shortest path step distances from `start` (`-1` if unreachable).
  - `state.path_step(passable_mask, target=(ty, tx), start=(y, x), allow_diagonal=True) -> (dy, dx)`: Returns the first step delta `(dy, dx)` along the shortest path from `start` to `target` (`(0, 0)` if unreachable).
  - **Example Vectorized Navigation Idiom**:
    ```python
    passable = np.isin(state.chars, [ord('.'), ord('#'), ord('>'), ord('<'), ord('+'), ord('_'), ord('{'), ord('$'), ord('%'), ord(')'), ord('['), ord('!'), ord('?'), ord('/')])
    for m in state.visible_monsters():
        if m.is_pet or m.is_peaceful:
            passable[m.y, m.x] = False
    # Step toward stairs '>' if visible:
    stairs = np.argwhere(state.chars == ord('>'))
    if len(stairs) > 0:
        dy, dx = state.path_step(passable, (int(stairs[0, 0]), int(stairs[0, 1])))
        if (dy, dx) != (0, 0):
            return seq(Action.OPEN, DIR_DELTAS[(dy, dx)]) if state.char_at(state.y + dy, state.x + dx) == '+' else DIR_DELTAS[(dy, dx)]
    ```

#### 4. Canonical NetHack Early-Game Tactical Survival & Navigation Guide

1. **Dungeon Level 1 Escape & Stair Priority**:
   - The primary objective on Dlvl 1 is locating and descending the downstairs (`>`) via `Action.DOWN`. Staying on Dlvl 1 past 300 turns leads to starvation and monster escalation.
   - Rooms are enclosed by walls (`|`, `-`) and closed doors (`+`). You cannot reach the rest of the dungeon without opening doors!
   - Step adjacent to closed doors (`+`) and yield `seq(Action.OPEN, dir)`. If locked, kick it open with `seq(Action.KICK, dir)`.
   - Once stairs (`>`) are visible, navigate directly to them. When standing on `>`: yield `Action.DOWN` (`'>'`) to descend to Dlvl 2.

2. **Robust Frontier Exploration & Target Persistence**:
   - **Target Persistence Rule**: Store a persistent target: `self.target_tile`. Never recompute `min(frontier, ...)` from scratch every turn! When the hero steps 1 tile toward a target, Manhattan distances to opposing frontier tiles invert, creating an infinite 2-tile ping-pong oscillation (`A <-> B`). Commit to `self.target_tile` until reached, impassable, or after 25 turns!
   - **Multi-Tile Cycle Detection**: `(state.y, state.x) == self.last_pos` is blind to 2-tile ping-pong loops (resets to 0 every turn). Maintain a position history ring buffer (`self.pos_history = collections.deque(maxlen=8)`). If `(state.y, state.x) in self.pos_history[:-1]`, you are trapped in an oscillation—immediately clear `self.target_tile` and take a random cardinal step to break the attractor.
   - **Frontier Tiles**: Identify unvisited walkable tiles adjacent to visited tiles (`macros.get_frontier_tiles`).

3. **Early Combat & Tactical Retreat**:
   - **Never fight while weakened**: Flee to a corridor (`#`) or doorway if HP $< 50\%$ so only 1 monster can attack at a time.
   - **Floating Eyes (`e`)**: NEVER melee blue `e` (floating eye) without blindfold/reflection—it paralyzes you for 50 turns. Use ranged weapons or throw daggers (`seq(Action.THROW, letter, dir)`).

4. **Nutrition & Inventory**:
   - Valkyries start with food rations (`%`) and +1 long sword + small shield.
   - Equip armor on turn 1 (`seq(Action.WEAR, item.letter)`) and wield primary weapon (`seq(Action.WIELD, item.letter)`).
   - Eat rations when `state.hunger_state >= 3` (Weak).
   - Eat fresh safe corpses (`macros.get_safe_corpses`) on the ground (`seq(Action.EAT, 'y')`). Never eat kobolds (`k`, poisonous) or chickatrices (`c`, stoning).
"""


def build_system_prompt(db_path: str = "data/lox.duckdb") -> str:
    schema_md = get_schema_markdown(db_path)
    return f"""You are the LOX Policy Author Agent.
Your job is to synthesize and evolve a pure Python NetHack policy (`class Policy`) and reusable macro library (`macros.py`) that map game states 1-to-1 to NetHack actions.
Motto: **"Hard Code Nothing" in the harness—all intelligence, memory, navigation, combat, and resource management live in your Python policy and macros.**

{POLICY_LANG_DOCS}

{schema_md}

### Available Investigation & Verification Tools
1. `query_db(sql)`: Run read-only SQL queries on DuckDB (`runs`, `ticks`, `batches`, `function_stats`, `experiments`, `expert_runs`, `expert_ticks`).
   - Query `runs WHERE is_outlier_good = TRUE` to inspect what happened in your best statistical outlier runs (`summary_digest`, `game_log`, `triggered_functions`).
   - Query `expert_runs` and `expert_ticks` to study how successful human ascensions and reference runs progress, what intrinsics/kit they acquire, and how they act.
2. `query_wiki(query, top_k=3)`: Search the offline NetHack 3.6.6 Wiki for exact monster mechanics, corpse effects, item properties, intrinsics, and dungeon branches.
3. `query_policy_docs(topic="all")`: View the policy language specification and the full source/docstrings of all macros currently in `macros.py`.
4. `dry_run_policy(code, is_macro=False, seed=42, max_turns=150, replay_run_id="", replay_steps=0)`:
   - Tests candidate Python code in a live NetHack sandbox.
   - **Step 1 (Smoke Check & Behavioral Critic)**: Verifies AST, checks reachability, detects 2-tile ping-pong oscillations and search spam. Optional `replay_run_id` fast-forwards into any recorded mid-game run in ~10ms.
   - **Step 2 (20-Episode Parallel Pilot Batch)**: Dispatches 20 episodes across persistent workers (up to 5 pilot revisions per generation) paired 1-to-1 against current baseline seeds, returning paired Wins/Losses/Ties, Δ depth, and concrete game logs of the worst stalled runs.
"""


def build_user_prompt(
    mode: str,
    generation: int,
    current_policy: str,
    macros_docs: str,
    batch_digest: str = "",
    recent_experiments: str = "",
) -> str:
    """Constructs the stateless Turn-0 prompt tailored to the active Modification Mode."""
    exp_block = (
        f"\n### Recent Experiment Lab Notebook (`experiments` table):\n{recent_experiments}\n"
        if recent_experiments
        else ""
    )
    digest_block = f"\n{batch_digest}\n" if batch_digest else ""

    if mode == "FULL_WRITE":
        return f"""### Generation {generation} — Mode: `FULL_WRITE` (Initial Baseline Synthesis)
You are bootstrapping the policy from a **completely clean slate** (no pre-written policy is provided).
Your task:
1. Use multiple tool calls (`query_db` on `expert_runs` / `expert_ticks`, `query_wiki`, `query_policy_docs`) to study successful NetHack runs, early-game survival mechanics, dungeon navigation, door opening/kicking, stair descent, combat, and food/corpse consumption.
2. Design a well-structured, modular `class Policy:` with clear methods (e.g. `run(self, state)`, `update_memory(self, state)`, `handle_combat(self, state)`, `handle_nutrition(self, state)`, `explore_level(self, state)`, etc.) and tunable thresholds declared via `self.param("name", default, min_val, max_val)`.
3. Test your full `class Policy:` using `dry_run_policy(code=...)` to verify there are zero syntax/attribute errors or 0-yield infinite loops.
4. Output:
   - A `yaml` hypothesis block describing your architecture:
     ```yaml
     mode: FULL_WRITE
     target_functions: [run]
     observation_from_db: "Insights gathered from expert_runs and wiki"
     mechanism: "Summary of initial Policy architecture and state tracking"
     ```
   - Your complete `class Policy:` implementation in a single ```python ... ``` block.
"""

    if mode == "MACRO_WRITE":
        return f"""### Generation {generation} — Mode: `MACRO_WRITE` (Reusable Macro Library Synthesis)
{digest_block}{exp_block}
{macros_docs}

### Current `policy.py`:
```python
{current_policy}
```

Your task in `MACRO_WRITE` mode:
1. Inspect `runs` (especially `is_outlier_good = TRUE` vs failed runs), `function_stats`, or `query_wiki` to identify reusable building blocks that `policy.py` needs (e.g., persistent level map/frontier helpers, multi-key item usage generators, safe corpse filtering, corridor/chokepoint detection, door breaching, etc.).
2. Write one or more clean, well-documented helper functions or generator subroutines (`def macro_name(state, ...):`) to add to `macros.py`.
3. Verify them with `dry_run_policy(code=..., is_macro=True)`.
4. Output:
   - A `yaml` block listing the macro names and their purpose:
     ```yaml
     mode: MACRO_WRITE
     target_functions: [your_macro_name]
     observation_from_db: "Why this macro is needed based on current runs"
     mechanism: "How the macro works and how policy.py can call it"
     ```
   - Your macro function definition(s) in a single ```python ... ``` block. (Note: `MACRO_WRITE` commits directly to `macros.py` and immediately starts the next generation).
"""

    # Default: FUNC_MOD_ADD
    return f"""### Generation {generation} — Mode: `FUNC_MOD_ADD` (Targeted Single-Skill Addition / Modification)
{digest_block}{exp_block}
{macros_docs}

### Current `policy.py`:
```python
{current_policy}
```

### Instructions for `FUNC_MOD_ADD`:
1. **Focus on 1 Primary Skill**:
   - Target **1 primary skill** (1 specialist method, plus the caller e.g. `run` if wiring a new method) per generation.
   - Modifying 1 method costs only 150 triggered episodes and isolates causal attribution; modifying multiple unrelated methods inflates rollout budgets and couples unrelated bugs.
2. **5 × 20-Episode Self-Iterative Pilot Revision Loop**:
   - Before submitting your final hypothesis, call `dry_run_policy(code=...)`.
   - It runs a **20-episode parallel pilot test** against the current baseline on 16 workers in ~0.5s, reporting paired Wins/Losses/Ties and the exact logs of the worst stalled runs.
   - You have **up to 5 pilot revisions per generation**: inspect the stalled run logs and behavioral critic warnings, revise your code to fix the failure mode, and test again with `dry_run_policy` until you beat the baseline!
3. **Interruptible State-Machine Skills & Anti-Oscillation Invariants**:
   - NetHack is an interrupt-driven survival loop: specialist methods (`explore_step`, `handle_combat`, `handle_nutrition`) must return an action or `None` on each tick and rely on persistent attributes on `self` (`self.target_tile`, `self.pos_history = collections.deque(maxlen=8)`) to avoid 2-tile ping-pong oscillations.
4. **Conditional Triggering & Method Wiring**:
   - Any episode that does NOT call your modified/added method is immediately discarded.
   - If adding a new method, also include the caller method (e.g. `run` or dispatcher) in your code block so it is wired in and called.
5. **Output Format**:
   - Output a `yaml` hypothesis block detailing your qualitative causal hypothesis and skill spec:
     ```yaml
     mode: FUNC_MOD_ADD
     target_functions: [method_you_modified_or_added]
     skill_name: "Name of the Targeted Skill"
     precondition: "When this skill activates"
     state_persistence: "What self.* attributes prevent oscillation and track targets"
     postcondition: "What marks completion of this skill"
     observation_from_db: "What you observed in outlier good runs or pilot runs"
     mechanism: "Exact behavioral change introduced by your skill modification"
     ```
   - Output ONLY the modified and/or newly added method(s) (or `class Policy:` containing them) in a single ```python ... ``` block.
"""
