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

GENERATOR_POLICY_DOCS = r"""### Policy Language & 1-to-1 Harness Reference

#### 1. Execution Model (`class Policy` and Modular Skills)
- Your policy is a standard Python 3.11+ class named `class Policy:`.
- A fresh `Policy()` instance is created at the start of every episode.
- **Recommended Architecture (Dispatcher + Action-Returning Specialist Methods)**:
  ```python
  class Policy:
      def __init__(self):
          self.visited = set()
          self.threshold = self.param("threshold", 0.50, 0.10, 0.90)

      def run(self, state: GameState):
          while True:
              self.visited.add((state.y, state.x))
              act = self.select_action(state)
              state = yield (act if act is not None else Action.WAIT)
  ```
  - You can also write `def step(self, state: GameState) -> Action:` directly, or use generator subroutines via `state = yield from self.sub_gen(state)`.
  - **Zero-Turn Action & Stall Telemetry (`state.turn_advanced`, `state.consecutive_stalls`)**:
    In NetHack, actions that fail (bumping into a peaceful creature without confirmation, attempting to lift an overweight corpse, kicking an unkickable door) consume **0 game turns** (`state.turn` does not advance). If `not state.turn_advanced`, do not repeat the same action blindly (`>= 100` consecutive stalls truncates as `stalled_zero_turn`).
  - **Interactive Prompts & Menus (`state.in_yn_prompt`, `state.in_text_prompt`, `state.in_menu`)**:
    - `state.in_yn_prompt`: `[yn]`/`[ynq]` prompts — respond `'y'`/`'n'`/`'q'`.
    - `state.in_text_prompt`: text naming prompts — yield `Action.ESC` (`'\x1b'`).
    - `state.in_menu`: interactive selection/pager menus (`Pick up what?`, `(end)`) — inspect `state.tty_lines` and send item letters, `Action.SPACE`, or `Action.ESC`.
  - **3 Strict Generator Rules**:
    1. In a generator (`def run` containing `yield`), ALWAYS receive the next state via `state = yield act` (never `return act` inside a generator—in Python `return val` inside a generator raises `StopIteration`!).
    2. If a helper method is a normal function that returns an `Action` or `None`, call it as `act = self.helper(state)` and then `state = yield act`.
    3. If a helper method is itself a generator (contains `yield`), call it via `state = yield from self.sub_gen(state)` and end `sub_gen` with `return state`. Never write `yield self.sub_gen(state)` without `from`!
- **Supported Python**: Full Python 3.11+ syntax (`if/elif/else`, `for`, `while`, `try/except`, comprehensions, `@dataclass`, `match/case`). Pre-loaded: `math`, `collections`, `heapq`, `itertools`, `functools`, `re`, `enum`, `dataclasses`, `numpy` (`np`), `random`, `copy`, `bisect`.
- **Configurable Parameters (`self.param`)**: `self.param("name", default, min_val, max_val)`.

#### 2. Modular Subsystems & Skill Modules (`skills/<module>.py`)
- Import routines from `skills.*`: `from skills.combat import melee_strike`.
- In `WRITE`/`REFACTOR`, specify `skill_module: "mod_name"` or test via `dry_run_policy(code="<skill_code module='mod_name'>...</skill_code>...")`.
"""

CORE_HARNESS_DOCS = r"""### 1-to-1 NetHack API & `GameState`

#### 1. Execution & Prompts
- Zero-turn stalls keep `state.turn_advanced == False`; avoid repeating stalled steps (`>=100` consecutive aborts as `stalled_zero_turn`).
- `state.in_yn_prompt`: `[yn]`/`[ynq]` -> `key('y')`/`key('n')`/`key('q')`.
- `state.in_text_prompt`: text/naming prompt -> `Action.ESC`.
- `state.in_menu`: selection/pager (`tty_lines`) -> item `key('a')`, `Action.SPACE`, or `Action.ESC`.
- Allowed modules: `lox.*`, `skills.*`, `numpy as np`, stdlib (`math`, `collections`, `heapq`, `itertools`, `functools`, `re`, `enum`, `dataclasses`, `random`, `copy`, `bisect`).

#### 2. 1-to-1 Actions (`Action`, `key`, `seq`, `repeat`, `extcmd`, `DIR_DELTAS`)
- Compass (`DIR_DELTAS[(dy,dx)]`): `Action.N` `(-1,0)`, `Action.S` `(1,0)`, `Action.E` `(0,1)`, `Action.W` `(0,-1)`, `Action.NE` `(-1,1)`, `Action.NW` `(-1,-1)`, `Action.SE` `(1,1)`, `Action.SW` `(1,-1)`.
- Vertical/Wait/Prompt: `Action.UP` `'<'`, `Action.DOWN` `'>'`, `Action.WAIT` `'.'`, `Action.MORE` `'\r'`, `Action.ESC` `'\x1b'`, `Action.SPACE` `' '`.
- Commands: `Action.SEARCH` `'s'`, `Action.OPEN` `'o'`, `Action.CLOSE` `'c'`, `Action.KICK` `C('d')`, `Action.UNTRAP` `M('u')`, `Action.EAT` `'e'`, `Action.WIELD` `'w'`, `Action.WEAR` `'W'`, `Action.TAKEOFF` `'T'`, `Action.PUTON` `'P'`, `Action.REMOVE` `'R'`, `Action.QUAFF` `'q'`, `Action.READ` `'r'`, `Action.ZAP` `'z'`, `Action.FIRE` `'f'`, `Action.THROW` `'t'`, `Action.PICKUP` `','`, `Action.DROP` `'d'`, `Action.APPLY` `'a'`, `Action.LOOT` `M('l')`, `Action.PRAY` `M('p')`, `Action.ENGRAVE` `'E'`, `Action.OFFER` `M('o')`, `Action.PAY` `'p'`, `Action.NAME` `'C'`, `Action.DIP` `M('d')`, `Action.ENHANCE` `M('e')`, `Action.TWOWEAPON` `'X'`, `Action.SWAP` `'x'`, `Action.CAST` `'Z'`, `Action.LOOK` `';'`.
- Sequences: `seq(Action.OPEN, DIR_DELTAS[(dy,dx)])`, `seq(Action.KICK, DIR_DELTAS[(dy,dx)])`, `seq(Action.EAT, 'f')`, `seq(Action.ZAP, 'i', dir_act)`, `extcmd("pray", "y")`, `repeat(5, Action.SEARCH)`.

#### 3. `GameState` (`state`)
- Vitals/Dungeon: `hp`, `max_hp`, `ac`, `energy`, `max_energy`, `xp_level`, `xp`, `score`, `gold`, `y` (0..20), `x` (0..78), `pos`, `turn`, `step_count`, `hunger_state` (0=Satiated..5=Fainted), `internal_hunger`, `depth`, `dungeon_num` (0=Doom, 1=Gehennom, 2=Mines, 3=Quest, 4=Sokoban, 5=Ludios, 6=Vlad, 7=Planes), `level_num`, `deepest_level_reached`, `is_on_stairs_down`, `carrying_capacity` (0..5), `encumbrance_name`, `worn_oprop_mask`.
- Inventory (`state.inventory -> list[InventoryItem]`): `letter`, `name`, `quantity`, `oclass`, `oclass_name`, `enchantment`, `is_weapon`/`is_armor`/`is_food`/`is_corpse`/`is_potion`/`is_scroll`/`is_wand`/`is_ring`/`is_amulet`/`is_tool`/`is_spellbook`/`is_coin`/`is_gem`, `is_worn`, `is_wielded`, `is_quivered`, `is_blessed`, `is_uncursed`, `is_cursed`, `is_for_sale`, `info` (`ObjectInfo`), `corpse_monster` (`MonsterInfo`).
- Monsters (`visible_monsters()`, `adjacent_monsters(exclude_pets=True)`): `y`, `x`, `name`, `char`, `color`, `is_adjacent_to(y,x)`, `is_pet`, `is_statue`, `is_invisible`, `info` (`difficulty`, `ac`, `speed`, `level`, `mr`, `nutrition`, `weight`, `size`, `resistances_mask`, `conveys_mask`, `is_acidic`/`is_poisonous`/`is_undead`/`is_were`/`is_human`/`is_elf`/`is_dwarf`/`is_gnome`/`is_orc`/`is_demon`/`is_giant`/`is_domestic`, `conveys_poison_res`/`conveys_fire_res`/`conveys_cold_res`/`conveys_sleep_res`/`conveys_shock_res`).
- Floor (`visible_objects()`, `objects_here()`): same item flags + `y`, `x`, `is_boulder`; C lookups `monster_info_by_name(name)`, `object_info_by_name(name)`.
- Conditions: `is_blind`, `is_confused`, `is_stunned`, `is_hallucinating`, `is_levitating`, `is_food_poisoned`, `is_terminally_ill`, `is_stoning`, `is_slimed`, `is_strangled`, `condition_mask`.
- Map/Terrain/Features: `message`, `chars` (`uint8[21,79]`), `glyphs` (`int16[21,79]`), `char_at(y,x)`, `ascii_map()`, `tty_text`, `tty_lines`, `terrain_at(y,x)`, `trap_at(y,x)`, `is_trap(y,x)`, `trap_mask`, `is_closed_door(y,x)`, `is_open_door(y,x)`, `adjacent_closed_doors()`, `unexplored_frontier_mask(passable_mask=None, include_doors=True)` *(passable tiles bordering unvisited walkable/door tiles—`state.pos` has `dist==0`, so filter `dist > 0`)*, `is_water(y,x)`, `is_lava(y,x)`, `is_ice(y,x)`, `is_iron_bars(y,x)`, `is_boulder(y,x)`, `boulder_positions`, `stairs_down_pos`/`positions`, `stairs_up_pos`/`positions`, `altar_positions`, `fountain_positions`, `portal_positions`.
- Pathfinding (8-way adjacency `max(abs(dy),abs(dx))==1`; `'-'`, `'|'`, `' '` walls IMPASSABLE): `bfs_distances(passable_mask, start=(y,x), allow_diagonal=True, no_diag=state.mem.door_mask)` & `path_step(passable_mask, target=(ty,tx), start=(y,x), allow_diagonal=True, no_diag=state.mem.door_mask) -> (dy,dx)`.

#### 4. Zero-Heuristic `WorldMemory` (`state.mem` & `state.kv`)
- Floor (`state.mem`): `visited`, `visit_count`, `remembered_chars`, `passable_mask`, `copy_passable_mask()`, `door_mask`, `searched_counts`, `remembered_obj_glyphs`, `remembered_trap_glyphs`, `locked_doors`.
- Inter-Branch Graph & Persistence: `stair_graph` (`dict[(dnum, dlvl, y, x), (dnum2, dlvl2, y2, x2)]`), `level_exits` (`dict[(dnum, dlvl), dict[(y, x), glyph]]`), `pos_history`, `is_oscillating`, `recent_messages`, `kv` (persisted across floors & replays).
"""

POLICY_LANG_DOCS = f"{GENERATOR_POLICY_DOCS}\n{CORE_HARNESS_DOCS[len('### 1-to-1 NetHack Harness & `GameState` Reference') :]}"


def build_system_prompt(
    db_path: str = "data/lox.duckdb", is_swarm: bool = False
) -> str:
    schema_md = get_schema_markdown(db_path, is_swarm=is_swarm)
    if is_swarm:
        return f"""LOX Autonomous Synthesis Agent (Hierarchical Behavior Tree Swarm).
Motto: **"Hard Code Nothing" in harness—all strategy, memory, navigation, combat live in `policy.py` + `skills/*.py`.**

{BEHAVIOR_TREE_DOCS}

{CORE_HARNESS_DOCS}

{schema_md}

### SQL Templates (`query_db`)
1. `SELECT generation, candidate_policy_id, avg_depth, p90_depth, max_depth, avg_score, avg_turns, hypothesis_passed FROM batches ORDER BY generation DESC LIMIT 10;`
2. `SELECT function_name, episodes_triggered, total_calls, total_turns_active, stagnation_share, livelock_score, liblit_failure_score, ochiai_good_run_score FROM champion_function_stats ORDER BY total_turns_active DESC LIMIT 20;`
3. `SELECT run_id, seed, depth, max_depth, final_ac, xp_level, turns, unique_tiles, policy_exceptions, last_active_function, end_reason, summary_digest FROM champion_runs ORDER BY max_depth ASC, turns DESC LIMIT 8;`
4. `SELECT COUNT(*) AS n, ROUND(AVG(final_ac),2) AS ac, ROUND(AVG(xp_level),2) AS xl, ROUND(AVG(max_depth),1) AS depth FROM expert_runs;`

### Swarm Peer Rules & Investigation Protocol
- **Never Guess**: Check C rules via `query_wiki` / `read_wiki_article` / `query_monster` / `query_item` and inspect code/data via `query_db` / `query_theories` / `diff_policies` / `read_skill(name, start_line, end_line)` / `read_symbol` / `find_references` / `grep_skills` / `architecture_map` / `benchmark_expert_parity`.
- **Anti-Recycling & Grounding**: Before first `dry_run_policy`, check `query_theories` + `query_wiki` (never tweak `REJECTED` mechanisms; protect high-Ochiai `D_succ` goals/functions).
- **Option-A Goal & Domain Architecture**:
  - `Policy.step(state)` observes `self.ws = WorldState()`, checks prompts/survival/combat interrupts, and ticks/preempts committed goals in `GoalRunner` (`skills/goals/core.py`). Never fall through silently to `Action.WAIT`.
  - Keep state typed via `ws.domain("name", MyDataclass)` inside the owning domain module (zero untyped `Blackboard` string keys).
  - Organize code cleanly under **2-level packages** `skills/<domain>/<module>.py` (e.g. `skills/survival/prayer.py`, `skills/goals/explore.py`; max depth 2; zero 5-LOC re-export facades). Add new goals via `register_goal(...)` in a goal module rather than bloating `policy.py`.
- **Surgical XML Patches**: Emit ONLY modified/new `def`/`class`/goal blocks in `<skill_code module="domain/submodule">...</skill_code>` and `<policy_code>...</policy_code>` (in BOTH `dry_run_policy` and final answer). Never re-paste unchanged functions or put `<hypothesis>` inside `dry_run_policy(code=...)`.
- **In-Turn Trigger Check**: `dry_run_policy` validates `pyright` types, cross-module signatures, claimed failure fixtures, and `prompt_torture`—if `TRIGGER_MISS` warns your edited function fired 0 times, fix preconditions before submitting.
"""

    return f"""You are the LOX Policy Author Agent.
Your job is to synthesize and evolve a pure Python NetHack policy (`class Policy`) and reusable macro library (`macros.py`) that map game states 1-to-1 to NetHack actions.
Motto: **"Hard Code Nothing" in the harness—all intelligence, memory, navigation, combat, and resource management live in your Python policy and macros.**

{POLICY_LANG_DOCS}

{schema_md}

### Empirical Differential Diagnostics (SQL Investigation Templates)
Do not guess why a run stalled or died. Formulate differential queries against DuckDB (`champion_runs`, `champion_function_stats`, `champion_ticks`, `expert_runs`, `expert_ticks`) to isolate the exact physical gap between failed runs and expert ascensions:

1. **Floor Residence & Physical State Contrast Query (Compare Turns per Depth vs Experts)**:
   ```sql
   -- 1A. How many turns do expert ascensions spend exploring each depth?
   SELECT depth, COUNT(DISTINCT run_id) AS runs, AVG(max_t - min_t) AS avg_turns_on_floor
   FROM (
       SELECT run_id, depth, MIN(turn) AS min_t, MAX(turn) AS max_t
       FROM expert_ticks
       GROUP BY run_id, depth
   ) GROUP BY depth ORDER BY depth LIMIT 10;

   -- 1B. Compare champion runs vs expert ascensions:
   SELECT 'champion_failures' AS source, AVG(final_ac) AS avg_ac, AVG(max_hp) AS avg_hp, AVG(turns) AS avg_turns
   FROM champion_runs
   UNION ALL
   SELECT 'expert_ascensions' AS source, AVG(ac) AS avg_ac, AVG(hp) AS avg_hp, AVG(turn) AS avg_turns
   FROM expert_ticks;
   ```

2. **Monster Encounter & Tactical Response Query**:
   ```sql
   -- How did expert ascensions act across depths?
   SELECT depth, action_name, COUNT(*) AS freq
   FROM expert_ticks
   GROUP BY depth, action_name ORDER BY depth, freq DESC LIMIT 15;
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

### Available Investigation & Verification Tools
1. `query_db(sql)`: Run read-only SQL queries on DuckDB (`champion_runs`, `champion_function_stats`, `champion_ticks`, `latest_pilot_runs`, `runs`, `ticks`, `batches`, `function_stats`, `experiments`, `expert_runs`, `expert_ticks`).
   - Query `champion_runs` to inspect what happened in the active champion policy's episodes (`summary_digest`, `game_log`, `triggered_functions`, `auto_esc_count`, `unique_tiles`, `pct_wait`, `pct_search`, `starved_with_food`, `died_unarmored`).
   - Query `expert_runs` and `expert_ticks` to study how successful human ascensions and reference runs progress, what intrinsics/kit they acquire, and how they act across depths.
2. `query_wiki(query, top_k=3)`: Search the offline NetHack 3.6.6 Wiki for exact monster mechanics, corpse effects, item properties, intrinsics, and dungeon branches.
3. `query_monster(name)`: Query the exact C-level engine `permonst` struct for any monster (e.g. `jackal`, `floating eye`, `lichen`, `orc`). Returns exact level, AC, movement speed, magic resistance, corpse weight, corpse nutrition, conveyed intrinsics (e.g. poison resistance), and monster innate resistances.
4. `query_item(name)`: Query the exact C-level engine `objclass` struct for any item (e.g. `food ration`, `potion of extra healing`, `long sword`). Returns exact object class, cost, weight, and nutrition value.
5. `query_policy_docs(topic="all")`: View the policy language specification and the full source/docstrings of all macros currently in `macros.py` (or call `query_policy_docs("actions")` for the complete command index).
6. `read_skill(skill_name, policy_id="")`: Retrieve the complete Python source code of any modular skill package currently residing in `skills/` (or from a historical `policy_id` snapshot).
7. `dry_run_policy(code, is_macro=False, seed=42, max_turns=150, replay_run_id="", replay_steps=0)`:
   - Tests candidate Python code in a live NetHack sandbox.
   - **Step 1 (Smoke Check & Behavioral Critic)**: Verifies AST, checks reachability, detects oscillations and search spam. Optional `replay_run_id` fast-forwards into any recorded mid-game run in ~10ms.
   - **Step 2 (30-Episode Parallel Pilot Batch)**: Dispatches 30 episodes across persistent workers (up to 8 pilot revisions per generation, with fallback to best candidate) paired 1-to-1 against current baseline seeds, returning paired Wins/Losses/Ties, Δ depth, and concrete game logs of the worst stalled runs.

### Scientific Investigation Protocol (MANDATORY)
1. **Never Guess from Intuition**: NetHack mechanics are strictly ruled by the C engine (e.g. traps, secret doors, searching probabilities, hunger increments, prayer timeouts, pet displacement, monster ranged attacks). Guessing code changes without empirical research causes catastrophic regressions.
2. **Mechanistic Trace Inversion via `query_wiki`**:
   Every event in NetHack (messages, status changes, deaths) is governed by an exact C function. Isolate the exact message string from the final 3 lines of `game_log`, strip proper nouns, and pass that phrase to `query_wiki(query=...)` to learn the exact C rule that caused it.
3. **The Binding Constraint Rule**:
   Always inspect the Termination Breakdown and Stall Diagnostics in the batch digest. Address the single failure mode responsible for the LARGEST percentage of failures (the binding constraint). Never spend a generation fixing a 2% edge case when 70% of runs died to the same primary cause.
4. **Empirical Root-Cause Discovery**:
   Inspect the raw termination breakdown, death counts, and RLE phase timelines in the batch digest and DuckDB.
   Do not guess which subsystem failed based on preconceived notions or simple turn counts. Formulate targeted SQL queries (`query_db`) over `runs` and `ticks` to inspect the exact state, position, messages, and action history in the turns leading up to death or timeout before writing code.
5. **Ground Reasoning in Observed State**:
   Anchor your reasoning to observed `state` fields (`state.depth`, `state.dungeon_num`, `state.message`, `state.mem.passable_mask`) and C-engine rules verified via `query_wiki` / `query_monster` / `query_item` rather than unverified assumptions. Use `state.mem.passable_mask` (or ensure `ord('-')`, `ord('|')`, and `ord(' ')` are strictly False in custom masks) and remember monsters attack across all 8 adjacent tiles (`max(abs(dy), abs(dx)) == 1`).
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
*(Policy exceeds 1200 lines; showing outline and key methods. Call `query_policy_docs('policy')` to view full code if needed.)*
```python
{joined_snippets}
```"""
    except Exception:
        pass

    return f"### Current `policy.py`:\n```python\n{current_policy}\n```"


BEHAVIOR_TREE_DOCS = """### Behavior Tree Primitives (`lox.lang.bt`)
Pre-loaded / importable via `from lox.lang.bt import ...`, `from lox.env.actions import Action, DIR_DELTAS, seq, key, repeat`:
- Imports allowed: `lox.*`, `skills.*`, `numpy as np`, stdlib (`math`, `collections`, `heapq`, `itertools`, `functools`, `re`, `random`, `bisect`, `copy`). NEVER import `macros` or `skills_common`.
- `Blackboard`: `bb.get_var(k, default=None)`, `bb.set_var(k, v)`, `bb.floor_get(state, k, default=None)`, `bb.floor_set(state, k, v)`, `bb.floor_blacklist_add(state, pos, ttl=50)`, `bb.floor_is_blacklisted(state, pos)`. No `self.*` state in shared leaves.
- `ActionLeaf(fn, name="...", min_commit=0)` (+ optional `preempt_if=fn`): `fn(state, bb)` -> `Action`/`seq(...)` (`SUCCESS`) or `None` (`FAILURE`, fall-through). `min_commit=N` commits N turns unless `preempt_if(state, bb)` returns `True` or `fn` returns `None`.
- `ConditionLeaf(cond_fn, name="...")` / `Sequence([...], name="...")` / `PrioritySelector([...], name="...", hysteresis=0)` / `Inverter(child)` / `BehaviorTree(root=..., blackboard=bb)`.
"""


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
    num_hypotheses: int = 3,
) -> str:
    """Constructs the stateless Turn-0 prompt tailored to the active Modification Mode."""
    from lox.lang.validator import extract_module_outline

    canon_mode = normalize_mode(mode)
    skills_block = ""
    if skills:
        if canon_mode == "SWARM_PEER":
            by_domain: dict[str, list[str]] = {}
            for s_name, s_src in sorted(skills.items()):
                loc_cnt = len(s_src.splitlines())
                exports: list[str] = []
                doc_line = ""
                try:
                    st_tree = ast.parse(s_src)
                    doc_raw = ast.get_docstring(st_tree) or ""
                    doc_line = doc_raw.splitlines()[0].strip()[:90] if doc_raw else ""
                    for nd in st_tree.body:
                        if isinstance(
                            nd, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                        ) and not nd.name.startswith("_"):
                            exports.append(nd.name)
                except Exception:
                    pass
                dom = s_name.split("/", 1)[0]
                exp_str = ", ".join(exports[:6]) + (
                    f" (+{len(exports) - 6})" if len(exports) > 6 else ""
                )
                summary_part = f" — {doc_line}" if doc_line else ""
                by_domain.setdefault(dom, []).append(
                    f"  - `skills/{s_name}.py` ({loc_cnt} LOC){summary_part}"
                    + (f" | exports: {exp_str}" if exp_str else "")
                )
            manifest_lines: list[str] = []
            for dom in sorted(by_domain.keys()):
                mods = by_domain[dom]
                manifest_lines.append(
                    f"- **{dom}/** ({len(mods)} file{'s' if len(mods) != 1 else ''}):"
                )
                manifest_lines.extend(mods)
            skills_block = (
                "\n### Domain Skill Tree (Lazy Read):\n"
                + "\n".join(manifest_lines)
                + "\n*(Inspect via `read_skill(name, start_line, end_line)`, `read_symbol(qualified_name)`, or `architecture_map()`.)*\n"
            )
        else:
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
                f"- [Gen {f.get('generation', 0)}] Target: `{tf_str}` | Hyp: {str(mech)[:240]} | Rejection: *{str(reason)[:240]}*"
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

    if canon_mode == "SWARM_WORKER":
        k_hyp = max(1, int(num_hypotheses))
        prev_subtree_block = (
            f"\n### Current Live `policy.py` Reference:\n```python\n{current_policy}\n```\n"
            if current_policy.strip()
            else ""
        )
        return f"""### Generation {generation} — Mode: `SWARM_WORKER` (Warm Multi-Hypothesis Behavior Tree Synthesis)
{digest_block}{prev_subtree_block}{skills_block}

### Warm Scientific Method Protocol ({k_hyp} Competing Hypotheses, Hard Code Nothing):
Follow these stages inside THIS conversation (your context, DB queries, and experiment results stay warm across all turns, and all tools remain available whenever you need to investigate further):

1. **Investigate What Happened Yourself (do NOT guess)**:
   - Use `query_db(sql)` (ONE `SELECT` statement per call — never `;`-separated multi-queries!) on `batches`, `runs` (`summary_digest`, `end_reason`), `ticks`, `function_stats`, `policies`, `expert_runs`, `expert_ticks`, along with `query_wiki`, `query_monster`, `query_item` (authoritative C/game mechanics), `read_skill`, and `query_policy_docs` BEFORE checking `query_theories` (past worker notes).
   - Note: when campaign tables (`runs`, `batches`, `ticks`, `function_stats`, `policies`) are empty (initial Generation 1 bootstrap), you MUST call BOTH `query_wiki` and `query_db` on `expert_runs` / `expert_ticks`.
   - Check whether an existing branch is failing, stalling, or oscillating with another leaf, or whether a capability is missing.

2. **Brainstorm {k_hyp} Distinct Competing Hypotheses BEFORE Coding**:
   - Emit a `<hypotheses>` YAML block listing {k_hyp} genuinely different mechanisms (`H1` .. `H{k_hyp}`) addressing your findings:
     ```xml
     <hypotheses>
     - id: H1
       mechanism: "..."
       trigger_condition: "..."
       expected_effect: "..."
     - id: H2
       mechanism: "..."
       trigger_condition: "..."
       expected_effect: "..."
     </hypotheses>
     ```
   - Each hypothesis must test a meaningfully distinct mechanism, condition threshold, or action sequence (not trivial constant tweaks).

3. **Sequential Translate, Test & Repair (`H1` .. `H{k_hyp}` — never stop at 1 candidate)**:
   - For each hypothesis `Hi`:
     - Translate `Hi` into a standalone Behavior Tree subtree (multiple helper functions are welcome—there is **no 2-function limit**) assigning `subtree = PrioritySelector([...], name="...")` or `subtree = ActionLeaf(...)` / `Sequence(...)`.
     - Do **NOT** wrap your subtree in `class Policy:` with `def run(self, state): yield ...`. Return `None` from `ActionLeaf` when the branch does not apply so `PrioritySelector` falls through to the rest of the live policy.
     - Test it using `dry_run_policy(code=..., hypothesis="H1: <short mechanism label>", max_turns=150)` (you may pass higher `max_turns` if needed).
     - `dry_run_policy` evaluates your subtree **inside the live baseline `policy.py` + `skills/`** across both the Frontier checkpoint and failure replays, returning `Triggered: k/N` and an **RLE execution trace**.
     - Focus on whether your `subtree:` actions fired cleanly without stalling/waiting in place; do NOT burn turns trying to fix unrelated `baseline:` segments. Do at most 1–2 quick repairs per hypothesis if it errors or never triggers, then move on and emit `<reflection>` + `<policy_code>` once all {k_hyp} are tested.
   - You MUST empirically test all {k_hyp} distinct hypotheses with `dry_run_policy(..., hypothesis=...)` before finishing.

4. **Comparative Reflection & Final Winning Subtree**:
   - After testing all {k_hyp} hypotheses, emit a `<reflection>` block summarizing what worked, what failed or was ruled out, and why the winner beat the alternatives:
     ```xml
     <reflection>
     winner: H...
     comparison:
       - H1: <observed triggered count, trace outcome, verdict>
       - H2: ...
     ruled_out: <key pitfall / edge case future generations must avoid>
     </reflection>
     ```
   - Finally, emit ONLY the winning, verified Python subtree inside `<policy_code> ... </policy_code>`.
"""

    if canon_mode == "SWARM_PEER":
        existing_policy_block = ""
        if current_policy.strip():
            pol_lines = current_policy.splitlines()
            if len(pol_lines) > 65:
                existing_policy_block = (
                    f"\n### Current `policy.py` ({len(pol_lines)} LOC)\n"
                    "*(Call `read_policy()` or `architecture_map()` to inspect coordinator & registered Goals.)*\n"
                )
            else:
                existing_policy_block = f"\n### Current `policy.py` ({len(pol_lines)} LOC):\n```python\n{current_policy}\n```\n"
        return f"""### Gen {generation} — Mode: `SWARM_PEER`
{digest_block}{existing_policy_block}{skills_block}
### Task (`SWARM_PEER`)
- Research first (`query_theories`, `query_wiki`, `read_wiki_article`, `query_db`, `read_skill`, `read_symbol`, `architecture_map`); never repeat rejected ideas from `Recent Experiment & Rejection Memory`.
- Test surgical `<skill_code module="domain/submodule">...</skill_code>` / `<policy_code>...</policy_code>` via `dry_run_policy` (omit unchanged functions).
- Output brief ```yaml``` hypothesis (plus optional `skills_to_prune: [...]`) and final `<skill_code module="...">` / `<policy_code>` XML blocks.
"""

    if canon_mode == "SWARM_META":
        existing_policy_block = (
            f"\n{_format_context_scoped_policy(current_policy, batch_digest)}\n"
            if current_policy.strip()
            else ""
        )
        return f"""### Generation {generation} — Mode: `SWARM_META` (Unified Behavior Tree Master Assembly)
{digest_block}{existing_policy_block}{skills_block}

### Meta-Agent Diagnosis & Assembly Protocol:
1. **Investigate the Full Trajectory & Root Causes Yourself**:
   - Use `query_db` (ONE `SELECT` statement per call — never `;`-separated multi-queries!), `read_policy`, `diff_policies`, `read_skill(skill_name, policy_id=...)`, `query_wiki`, `query_monster`, `query_item`, and `query_policy_docs` (over `query_theories`) to inspect `batches` across generations, `runs` (`summary_digest`, `end_reason`, `unique_tiles`, `pct_wait`, `pct_search`), `function_stats` (`stagnation_share`, `livelock_score`, `liblit_failure_score`, `ochiai_good_run_score`), `expert_runs`, and prior `policies` (when campaign tables are empty on initial Generation 1 bootstrap, both `query_wiki` and `query_db` on `expert_runs`/`expert_ticks` are mandatory).
   - If performance regressed compared to an earlier generation, inspect what changed in `policies` (`read_policy`, `read_skill(..., policy_id=...)`, `diff_policies`) and `runs.summary_digest` and restore or repair the broken capability rather than blindly stacking new patches.
2. **Modular Skill Imports (Keep `policy.py` Lean)**:
   - Specialist worker subtrees are saved as separate skill modules under `skills/<role>.py` (listed in `Hierarchical Skill Registry` above; inspect any full module with `read_skill(skill_name)`).
   - Import and compose them cleanly inside `class Policy.__init__` rather than pasting worker helper code into `policy.py`. Treat worker outputs as proposals—only wire branches that genuinely help and do not conflict or oscillate with other branches.
3. **Complete `class Policy:` Wrapping `BehaviorTree`**:
   - In `def __init__(self):`, initialize `self.blackboard = Blackboard()`, wire the priority branches (`PrioritySelector([...])`), and construct `self.tree = BehaviorTree(root=..., blackboard=self.blackboard)`.
   - Define `def step(self, state):` returning `self.tree.tick(state) or Action.WAIT` (or include a safe final fallback `ActionLeaf` so `step(state)` never returns `None`).
   - Every node function invoked during `self.tree.tick(state)` must be cleanly defined in `policy.py` or imported from `skills.*`.
4. **Prune & Merge True Hijacking or Dead Branches (Preserve Core Capabilities)**:
   - Remember: **high turn-share alone is NOT hijacking** — whichever branch handles a frequent default capability will naturally carry most of the turn share.
   - Identify true hijacking, thrashing, or broken branches from `stagnation_share`, `livelock_score`, `liblit_failure_score`, `summary_digest`, or `dry_run_policy` RLE traces showing repeated no-op/oscillating steps without progress.
   - **Never prune a working branch that provides a core capability unless you have verified in `dry_run_policy` that its replacement actually fires and makes progress**.
5. **Verify Before Finishing**:
   - Call `dry_run_policy(code=...)` on your complete `class Policy:` source to confirm clean AST compilation and healthy execution in `NetHackHarness`.
6. **Output Format**:
   - Output your complete, self-contained Python `class Policy:` coordinator inside `<policy_code> ... </policy_code>` tags or a standard ```python ... ``` code block.
"""

    if canon_mode == "CREATE":
        return f"""### Generation {generation} — Mode: `CREATE` (Initial Baseline Synthesis)
{stagnation_block}{discoveries_block}{failures_block}You are bootstrapping the policy from a **completely clean slate** (no pre-written policy is provided).
Your task:
1. Use multiple tool calls (`query_db` on `expert_runs` / `expert_ticks`, `query_wiki`, `query_policy_docs`) to study successful NetHack runs and C-engine mechanics.
2. Design a well-structured, modular `class Policy:` with clear methods.
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
