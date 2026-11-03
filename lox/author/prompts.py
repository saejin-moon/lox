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
- **Multi-Keystroke Sequences (`seq(...)`), Extended Commands (`extcmd`), & Repeats (`repeat`)**:
  Many NetHack actions prompt for a direction, inventory slot letter, or text. Send the full sequence in one turn using `seq(...)`:
  - Open a door at `(dy, dx)`: `yield seq(Action.OPEN, DIR_DELTAS[(dy, dx)])`
  - Kick a door at `(dy, dx)`: `yield seq(Action.KICK, DIR_DELTAS[(dy, dx)])` (breaks locked doors into `.` in 1-3 kicks)
  - Repeat action $N$ times: `yield repeat(10, Action.SEARCH)` or `yield repeat(5, Action.WAIT)`
  - Extended commands: `yield extcmd("pray", "y")`, `yield extcmd("terrain")`, `yield extcmd("sit")`, `yield extcmd("chat", DIR_DELTAS[(dy, dx)])`
  - Eat inventory item in slot `'f'`: `yield seq(Action.EAT, 'f')`
  - Eat a corpse on the floor (`'y'` to confirm): `yield seq(Action.EAT, 'y')` (Hero must be standing on the exact corpse tile: `c.y == state.y and c.x == state.x`. If adjacent, step onto the corpse tile first!)
  - Quaff potion in slot `'h'`: `yield seq(Action.QUAFF, 'h')`
  - Wear armor in slot `'d'`: `yield seq(Action.WEAR, 'd')`
  - Wield weapon in slot `'a'`: `yield seq(Action.WIELD, 'a')`
  - Throw item `'b'` in direction `(dy, dx)`: `yield seq(Action.THROW, 'b', DIR_DELTAS[(dy, dx)])`
  - Zap directional wand `'i'` in direction `(dy, dx)`: `yield seq(Action.ZAP, 'i', DIR_DELTAS[(dy, dx)])`
  - Engrave Elbereth in dust with fingers (`'-'`): `yield seq(Action.ENGRAVE, '-', 'Elbereth\\r')`
  - Dip item `'a'` into fountain (`'y'`): `yield seq(Action.DIP, 'a', 'y')`

#### 3. Querying `GameState` (`state`)
Every turn, `state` exposes 100% of NetHack's observation arrays, terminal screens, and C symbol tables:
- **Hero Vitals & Coordinates**:
  `state.y` (0..20), `state.x` (0..78), `state.pos` (`(y, x)`),
  `state.hp`, `state.max_hp`, `state.energy`, `state.max_energy`, `state.ac`,
  `state.xp_level`, `state.xp`, `state.score`, `state.gold`, `state.turn`, `state.step_count`,
  `state.depth` (dungeon depth 1..50), `state.dungeon_num` (`0`=Dungeons of Doom, `1`=Gehennom, `2`=Gnomish Mines, `3`=Quest, `4`=Sokoban),
  `state.deepest_level_reached` (`int`), `state.is_on_stairs_down` (`bool`), `state.internal_hunger` (`int` raw nutrition counter),
  `state.strength`, `state.dex`, `state.con`, `state.intel`, `state.wis`, `state.cha`, `state.align` (`1`=Lawful, `0`=Neutral, `-1`=Chaotic),
  `state.hunger_state` (`0`=Satiated, `1`=Not Hungry, `2`=Hungry, `3`=Weak, `4`=Fainting, `5`=Fainted), `state.hunger_name`,
  `state.carrying_capacity` (`0`=Unencumbered, `1`=Burdened, `2`=Stressed, `3`=Strained, `4`=Overtaxed, `5`=Overloaded), `state.encumbrance_name`.
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
    * `=` : ring on ground
    * `"` : amulet on ground
    * `(` : tool on ground
    * `$` : gold on ground
    * **CRITICAL MAP INVARIANT**: `'-'` (horizontal wall) and `'|'` (vertical wall) are SOLID, IMPASSABLE ROOM WALLS! NEVER mark `'-'` or `'|'` as passable! Bumping into `'-'` or `'|'` wastes turns and causes severe wall stalls!
  - `state.char_at(y, x) -> str`: Single ASCII character at `(y, x)`.
  - `state.glyphs` (`int32[21, 79]`): Raw NLE glyph IDs.
  - `state.colors` (`int8[21, 79]`), `state.specials` (`uint8[21, 79]`).
  - `state.ascii_map() -> str`: Full 21x79 ASCII rendering of the screen.
  - `state.tty_text -> str`: Complete 24-line terminal screen rendering (includes message line 0, map lines 1-21, status lines 22-23, and all pop-up menus).
- **Semantic Terrain & Trap Queries (Direct from NetHack C Symbol Tables)**:
  - `state.terrain_at(y, x) -> str`: Semantic name of terrain at `(y, x)` (e.g. `'closed door'`, `'open door'`, `'floor of a room'`, `'corridor'`, `'staircase down'`, `'staircase up'`, `'altar'`, `'fountain'`, `'sink'`, `'grave'`, `'tree'`, `'iron bars'`, `'water'`, `'molten lava'`).
  - `state.trap_at(y, x) -> str | None`: Specific trap name at `(y, x)` (e.g. `'arrow trap'`, `'bear trap'`, `'pit'`, `'teleportation trap'`) or `None`.
  - `state.description_at(y, x) -> str`: Exact English description string from NLE's C engine for tile `(y, x)` (e.g. `'closed door'`, `'floor of a room'`, `'human valkyrie called Agent'`).
  - `state.is_closed_door(y, x) -> bool`: True if tile is a closed door (`'+'`).
  - `state.is_open_door(y, x) -> bool`: True if tile is an open door / doorway (`'\\'`, `'.'`).
  - `state.is_stairs_down(y, x) -> bool`, `state.stairs_down_pos -> (y, x) | None`: Fast check and coordinates of downstairs `'>'`.
- **Inventory (`state.inventory -> list[InventoryItem]`)**:
  Each `item` has:
  - `item.letter` (`'a'..'z'`), `item.name` (full description string, e.g. `'a +1 long sword (weapon in hand)'`, `'2 food rations'`), `item.quantity` (`int`), `item.char` (`str`: `')'` weapon, `'['` armor, `'%'` food, `'!'` potion, `'?'` scroll, `'/'` wand, `'='` ring, `'\"'` amulet, `'('` tool, `'$'` gold).
  - Typed helper properties: `item.is_weapon`, `item.is_armor`, `item.is_food`, `item.is_potion`, `item.is_scroll`, `item.is_wand`, `item.is_ring`, `item.is_amulet`, `item.is_tool`.
  - Status helper properties: `item.is_worn`, `item.is_wielded`, `item.is_quivered`, `item.is_blessed`, `item.is_uncursed`, `item.is_cursed`.
- **Visible Monsters (`state.visible_monsters() -> list[VisibleMonster]`)**:
  Each `m` has: `m.y`, `m.x`, `m.name`, `m.char`, `m.color`, `m.is_pet`, `m.is_peaceful`, `m.is_hostile` (`not (is_pet or is_peaceful)`), `m.is_statue`, `m.is_ridden`, `m.is_invisible`, and `m.info` (`MonsterInfo`: `m.info.name`, `m.info.level`, `m.info.speed`, `m.info.ac`, `m.info.difficulty`, `m.info.nutrition`, `m.info.weight`, `m.info.attacks` tuple).
- **Visible Objects & Corpses (`state.visible_objects() -> list[VisibleObject]`)**:
  Each `o` has: `o.y`, `o.x`, `o.name`, `o.char`, `o.color`, `o.is_corpse`, `o.is_food`, `o.is_weapon`, `o.is_armor`, `o.corpse_monster` (`MonsterInfo | None` if `is_corpse`), `o.info` (`ObjectInfo | None`: `o.info.name`, `o.info.cost`, `o.info.weight`).
- **Fast Vectorized NumPy & Numba Pathfinding Helpers**:
  - `state.bfs_distances(passable_mask, start=(y, x), allow_diagonal=True) -> int32[21, 79]`: Shortest path step distances from `start` (`-1` if unreachable).
  - `state.path_step(passable_mask, target=(ty, tx), start=(y, x), allow_diagonal=True) -> (dy, dx)`: Returns the first step delta `(dy, dx)` along the shortest path from `start` to `target` (`(0, 0)` if unreachable).
  - **Generic Vectorized Pathfinding Usage**:
    ```python
    # Construct a boolean passable mask (e.g. from state.chars or state.mem.remembered_chars):
    passable = np.isin(state.chars, [ord('.'), ord('#'), ord('>')])
    # Exclude obstacles or hazardous positions:
    for m in state.visible_monsters():
        passable[m.y, m.x] = False
    # Find next step towards target coordinate:
    dy, dx = state.path_step(passable, target=(target_y, target_x))
    if (dy, dx) != (0, 0):
        return DIR_DELTAS[(dy, dx)]
    ```

#### 4. Deterministic Zero-Heuristic WorldMemory (`state.mem`)

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

### Available Investigation & Verification Tools
1. `query_db(sql)`: Run read-only SQL queries on DuckDB (`runs`, `ticks`, `batches`, `function_stats`, `experiments`, `expert_runs`, `expert_ticks`).
   - Query `runs WHERE is_outlier_good = TRUE` to inspect what happened in your best statistical outlier runs (`summary_digest`, `game_log`, `triggered_functions`).
   - Query `expert_runs` and `expert_ticks` to study how successful human ascensions and reference runs progress, what intrinsics/kit they acquire, and how they act.
2. `query_wiki(query, top_k=3)`: Search the offline NetHack 3.6.6 Wiki for exact monster mechanics, corpse effects, item properties, intrinsics, and dungeon branches.
3. `query_policy_docs(topic="all")`: View the policy language specification and the full source/docstrings of all macros currently in `macros.py`.
4. `dry_run_policy(code, is_macro=False, seed=42, max_turns=150, replay_run_id="", replay_steps=0)`:
   - Tests candidate Python code in a live NetHack sandbox.
   - **Step 1 (Smoke Check & Behavioral Critic)**: Verifies AST, checks reachability, detects oscillations and search spam. Optional `replay_run_id` fast-forwards into any recorded mid-game run in ~10ms.
   - **Step 2 (30-Episode Parallel Pilot Batch)**: Dispatches 30 episodes across persistent workers (up to 5 pilot revisions per generation) paired 1-to-1 against current baseline seeds, returning paired Wins/Losses/Ties, Δ depth, and concrete game logs of the worst stalled runs.

### Scientific Investigation Protocol (MANDATORY)
1. **Never Guess from Intuition**: NetHack mechanics are strictly ruled by the C engine (e.g. traps, secret doors, searching probabilities, hunger increments, prayer timeouts, pet displacement, monster ranged attacks). Guessing code changes without empirical research causes catastrophic regressions.
2. **Four-Stage Scientific Workflow**:
   - **Stage 1 (Diagnose)**: Inspect the Batch Overview, termination causes, and stall diagnostics. Query `runs` and `expert_runs` via `query_db` to see what separated ascensions from failures.
   - **Stage 2 (Research Mechanics)**: Call `query_wiki` on the specific obstacles, monsters, or death reasons identified in Stage 1 (e.g. `query_wiki("bear trap")`, `query_wiki("praying")`, `query_wiki("secret door")`).
   - **Stage 3 (Synthesize & Pilot Test)**: Formulate your causal hypothesis and test your candidate code using `dry_run_policy(code=...)`.
   - **Stage 4 (Iterative Repair)**: If a pilot fails or regresses, STOP! Inspect the failing seed logs, call `query_wiki` or `query_db` to find the exact C engine rule you violated, and revise your skill before re-testing.
"""


def _format_context_scoped_policy(current_policy: str, batch_digest: str = "") -> str:
    lines = current_policy.splitlines()
    if len(lines) <= 350:
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
    macros_docs: str,
    batch_digest: str = "",
    recent_experiments: str = "",
    skills: dict[str, str] | None = None,
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
{digest_block}{exp_block}
{macros_docs}
{skills_block}
{_format_context_scoped_policy(current_policy, batch_digest)}

### Scientific Protocol for `STRUCTURAL_REFACTOR`:
Single-skill targeted patches have stalled or reached a local optimum. You have permission to perform a **structural refactoring** of `class Policy`.
1. **Stage 1 (Empirical Investigation)**:
   - Query DuckDB (`runs`, `experiments`, `expert_runs`) to identify why the previous architecture stalled (e.g. state management bottlenecks, lack of multi-turn goal commitment, unhandled traps or hunger states).
2. **Stage 2 (Full Class Synthesis)**:
   - Rewrite `class Policy:` from scratch or restructure its methods, attributes on `self`, and dispatch loop to break the plateau.
   - You can freely use verified macros from `macros.py` (see above) and `state.mem` for zero-heuristic spatial/episodic memory.
3. **Stage 3 (Verification)**:
   - Test your candidate class using `dry_run_policy(code=...)`.
4. **Stage 4 (Commit Output)**:
   - Output a `yaml` block with `mode: STRUCTURAL_REFACTOR`, `target_functions: [run]`, observation and mechanism.
   - Output your full `class Policy:` in a single ```python ... ``` block.
"""

    # Default: FUNC_MOD_ADD
    return f"""### Generation {generation} — Mode: `FUNC_MOD_ADD` (Targeted Single-Skill Addition / Modification)
{digest_block}{exp_block}
{macros_docs}
{skills_block}
{_format_context_scoped_policy(current_policy, batch_digest)}

### Scientific Investigation & Modification Protocol for `FUNC_MOD_ADD`:

1. **Stage 1: MANDATORY Empirical Research (Turn 1)**:
   - **DO NOT OUTPUT CODE ON TURN 1.**
   - Before writing or modifying any code, you MUST use your research tools to investigate the previous batch:
     * Call `query_db(sql=...)` to inspect the failed runs, stall causes, or death reasons from the previous batch (`runs` table), or compare with how experts handled this stage in `expert_runs`.
     * Call `query_wiki(query=...)` to look up the exact NetHack C engine mechanics for the specific monsters, items, traps, locked doors, or terrain causing stalls or deaths.
   - Ground your hypothesis in empirical game facts, not blind intuition.

2. **Stage 2: Focus on 1 Primary Skill**:
   - Target **1 primary skill** (1 specialist method, plus the caller e.g. `run` if wiring a new method) per generation.
   - Modifying 1 method costs only 150 triggered episodes and isolates causal attribution; modifying multiple unrelated methods inflates rollout budgets and couples unrelated bugs.
   - NetHack is an interrupt-driven survival loop: specialist methods (`explore_step`, `handle_combat`, `handle_nutrition`, etc.) must return an action or `None` on each tick and rely on persistent attributes on `self` (`self.target_tile`, `self.pos_history = collections.deque(maxlen=8)`) to avoid 2-tile ping-pong oscillations.

3. **Stage 3: 30-Episode Pilot Verification (`dry_run_policy`)**:
   - Test your candidate code using `dry_run_policy(code=...)`.
   - It runs a **30-episode parallel pilot test** against the current baseline on 16 workers in ~1s, reporting paired Wins/Losses/Ties and the exact logs of the worst stalled runs.
   - Any episode that does NOT call your modified/added method is immediately discarded (`triggered == False`). Ensure your method is called by `run()`.

4. **Stage 4: Pilot Failure Protocol (If Pilot Fails)**:
   - If `dry_run_policy` returns `PILOT_FAILED`, **STOP! Do NOT blindly guess another code variation.**
   - Inspect the failed pilot logs and behavioral critic warnings.
   - Call `query_wiki` or `query_db` to look up the exact mechanic that caused the failure before synthesizing a revised method. You have up to 5 pilot revisions per generation.

5. **Output Format (Once Ready to Test or Submit)**:
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
