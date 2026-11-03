# LOX System Prompt, Cognitive Architecture & Codebase Bottleneck Overhaul (`TODO.md` — Pass 4)

## Objective
1. **System Prompt & Cognitive Architecture Overhaul (`lox/author/prompts.py`, `lox/author/agent.py`)**: Restructure the Author Agent's system and user prompts using 2024–2026 **Context Engineering** and **Phase-Isolated Operating Manual** principles (Anthropic, OpenAI, SWE-agent, Agentless) so the agent autonomously discovers the progression path from `Dlvl 1` to ascension via `query_db` and `query_wiki`, strictly maintaining **"Hard Code Nothing"** (zero hardcoded NetHack strategy guides).
2. **Eliminate the 6 Codebase Bottlenecks Holding LOX Back**:
   - Fix the `SELECT MAX(batch_id)` poison pill (`pilot_4_*` > `c2026_*`) via DuckDB champion views and `query_db` resolution (`lox/db/schema.py`, `lox/author/tools.py`).
   - Populate `expert_ticks` across all dungeon depths (`Dlvl 1..48`) from human ascension dumplogs and multi-depth NLE reference rollouts so `expert_ticks` queries never return `0 rows` (`lox/datasets/download_experts.py`).
   - Fix `WorldMemory` open-door ASCII collision (`'-'`/`'|'` shared between walls and open doors), object/hero underlying floor initialization, and orthogonal doorway transitions in `bfs_distances` / `path_step` (`lox/env/state.py`, `lox/env/grid_utils.py`).
   - Prevent `PolicyArchiveEntry` from overwriting `baseline_runs` with 200-step shards and include `action_bytes, milestones_json` when resuming from DuckDB (`lox/engine/orchestrator.py`).
   - Gate `PARAM_TUNE` in `ModeScheduler` so it only triggers after promoted `FUNC_MOD_ADD` generations, and forward `skills`/`benchmark` in `run_spsa_param_tuning` (`lox/engine/scheduler.py`, `lox/engine/modes.py`).
   - Fix `MacroHarvester` method name collisions and `check_macro_contracts` false positives on `self.<method>()` calls (`lox/lang/macros.py`, `lox/lang/validator.py`).

---

## Part A: Codebase Bottleneck Fixes

### 1. Fix the `SELECT MAX(batch_id)` Poison Pill (`lox/db/schema.py` & `lox/author/tools.py`)

#### Root Cause
In `data/lox.duckdb`, campaign batches are named `c2026..._g001` while `dry_run_policy` pilot batches are saved into the same tables as `pilot_1_...`, `pilot_4_...`. Because ASCII `'p' > 'c'` and `'pilot_4_' > 'pilot_1_'`, any SQL query using `WHERE batch_id = (SELECT MAX(batch_id) FROM runs)` permanently returns `pilot_4_...` (a 30-episode rejected pilot from an old generation) or a rejected/backfill batch instead of the active promoted champion batch.

#### Required Changes
1. **`lox/db/schema.py` (`init_db`)**:
   - Create four canonical SQL views inside `init_db(db_path)` so they always exist on every connection:
     ```sql
     CREATE OR REPLACE VIEW champion_runs AS
     SELECT * FROM runs
     WHERE batch_id = (
         SELECT batch_id FROM batches
         WHERE hypothesis_passed = TRUE
           AND mode NOT IN ('PILOT', 'BASELINE_BACKFILL', 'MACRO_WRITE')
         ORDER BY generation DESC, created_at DESC
         LIMIT 1
     );

     CREATE OR REPLACE VIEW champion_function_stats AS
     SELECT * FROM function_stats
     WHERE batch_id = (
         SELECT batch_id FROM batches
         WHERE hypothesis_passed = TRUE
           AND mode NOT IN ('PILOT', 'BASELINE_BACKFILL', 'MACRO_WRITE')
         ORDER BY generation DESC, created_at DESC
         LIMIT 1
     );

     CREATE OR REPLACE VIEW champion_ticks AS
     SELECT * FROM ticks
     WHERE run_id IN (SELECT run_id FROM champion_runs);

     CREATE OR REPLACE VIEW latest_pilot_runs AS
     SELECT * FROM runs
     WHERE batch_id = (
         SELECT batch_id FROM batches
         WHERE mode = 'PILOT'
         ORDER BY created_at DESC
         LIMIT 1
     );
     ```
2. **`lox/author/tools.py` (`AuthorToolRegistry.query_db` & `dry_run_policy`)**:
   - In `dry_run_policy`, format pilot batch IDs with zero-padded timestamp first (`pilot_{int(time.time() * 1000)}_{self.pilot_runs_used:02d}`) so chronological order matches lexicographical order.
   - In `query_db(self, sql: str)`:
     - Before executing `cleaned`, dynamically rewrite any `(SELECT MAX(batch_id) FROM runs)`, `(SELECT MAX(batch_id) FROM function_stats)`, or `(SELECT MAX(batch_id) FROM batches)` (case-insensitive regex `r"\(\s*SELECT\s+MAX\s*\(\s*batch_id\s*\)\s+FROM\s+(?:runs|function_stats|batches)\s*\)"`) to:
       `"(SELECT batch_id FROM batches WHERE hypothesis_passed = TRUE AND mode NOT IN ('PILOT', 'BASELINE_BACKFILL', 'MACRO_WRITE') ORDER BY generation DESC, created_at DESC LIMIT 1)"`
       (with fallback to `"(SELECT batch_id FROM batches ORDER BY created_at DESC LIMIT 1)"` if no promoted non-pilot batch exists yet).

---

### 2. Fix the `expert_ticks` "0 Rows" Trap (`lox/datasets/download_experts.py`)

#### Root Cause
`ingest_expert_runs_into_duckdb` only populates `expert_ticks` from 5 short 150-step `nle_ref_trace` runs on `depth = 1`, while the 35 human ascensions (`VERIFIED_HUMAN_ASCENSIONS` and Hardfought dumplogs) only insert into `expert_runs`. As a result, `SELECT ... FROM expert_ticks WHERE depth >= 2` returns `0 rows`.

#### Required Changes in `lox/datasets/download_experts.py`
1. **Extract Structured Milestone Ticks From Every Dumplog (`extract_milestone_ticks_from_dumplog`)**:
   - Add helper `extract_milestone_ticks_from_dumplog(run_id: str, meta: dict[str, Any], dumplog_text: str) -> list[dict[str, Any]]`:
     - For each human ascension (`max_depth` typically 25–49), generate structured progression ticks across depths `1` through `min(max_depth, 30)` plus any explicit `Dlvl` / `T:` lines parsed from `Latest messages:` in `dumplog_text`.
     - Interpolate realistic progression trajectory stats from start (`depth=1, turn=1, hp=16, max_hp=16, ac=6, xp_level=1, hunger_state=1`) to the dumplog's final ascension stats (`final_hp`, `max_hp`, `final_ac`, `xp_level`, `turns`), assigning concrete `action_name` (`WEAR`, `WIELD`, `EAT`, `OPEN`, `KICK`, `DOWN`, `SEARCH`, `ENGRAVE`, `QUAFF`, `N`, `S`, `E`, `W`) and informative `message` strings from the dumplog's `Latest messages`, `Intrinsics`, and `Your inventory` sections (including early monster encounters like jackals, grid bugs, goblins, floating eyes, ants, and food/armor actions).
2. **Append Dumplog Milestone Ticks in `ingest_expert_runs_into_duckdb`**:
   - Whenever `parse_dumplog_to_expert_run` is called (for `VERIFIED_HUMAN_ASCENSIONS`, custom dumplogs, URLs, and online Hardfought ascensions), also call `extract_milestone_ticks_from_dumplog(...)` and append the resulting rows to `tick_rows`.
   - In `ingest_expert_runs_into_duckdb`, check both `existing = SELECT COUNT(*) FROM expert_runs` AND `deep_ticks = SELECT COUNT(*) FROM expert_ticks WHERE depth >= 2`. Only early-return if `existing >= max_ascensions and deep_ticks > 0 and not force and not has_custom`.
   - When inserting `tick_rows`, delete matching `run_id`s from `expert_ticks` (`DELETE FROM expert_ticks WHERE run_id IN (SELECT DISTINCT run_id FROM _temp_expert_ticks)`) before `INSERT INTO expert_ticks SELECT * FROM _temp_expert_ticks`.

---

### 3. Fix `WorldMemory` Open-Door ASCII Collision & Doorway Diagonal Pathfinding (`lox/env/state.py` & `lox/env/grid_utils.py`)

#### Root Cause
1. **Open-Door vs. Wall ASCII Collision**: In NetHack's C engine (`state.chars`), an **open door** (`glyph 2372` / `2373`, `_CMAP_NAME_BY_GLYPH[g] == "open door"`) has ASCII character `'-'` or `'|'`—the **exact same ASCII character** as a horizontal or vertical solid wall (`glyph 2360..2370`, `_CMAP_NAME_BY_GLYPH[g] == "wall"`). When `WorldMemory.update` copies `chars[mask]` into `floor.remembered_chars`, opening a door changes its remembered char from `'+'` to `'-'` or `'|'`. Any policy that excludes walls (`remembered_chars != ord('-')` and `!= ord('|')`) immediately treats every open door as a solid wall and traps itself inside the room!
2. **Underlying Floor Initialization**: At `reset()` or when an object (`[`, `)`, `%`, `$`) or monster sits on a tile, `_IS_CMAP_LUT[glyph]` is `False`, leaving `floor.remembered_chars[y, x] == 0` so policies cannot pathfind onto tiles with items/corpses/monsters.
3. **Doorway Diagonal Collision**: In NetHack (`hack.c:test_move`), diagonal movement (`dy != 0 and dx != 0`) into or out of an intact doorway (`open door` or `closed door`) is forbidden by the C engine (`"You can't move diagonally into/out of an intact doorway."`). When `bfs_distances` or `path_step` uses `allow_diagonal=True`, it routes diagonally through open doors, causing infinite 0-turn collision loops.

#### Required Changes
1. **`lox/env/state.py` (`FloorMemory`, `WorldMemory`, `GameState`, `VisibleMonster`)**:
   - Add precomputed boolean LUTs at module level after `_CMAP_NAME_BY_GLYPH` is populated:
     - `_IS_WALL_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)` (True where `_CMAP_NAME_BY_GLYPH[g] == "wall"`)
     - `_IS_OPEN_DOOR_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)` (True where `_CMAP_NAME_BY_GLYPH[g] == "open door"`)
     - `_IS_DOORWAY_GLYPH_LUT = np.zeros(_MAX_GLYPH, dtype=bool)` (True where `_CMAP_NAME_BY_GLYPH[g] in ("open door", "closed door")`)
   - In `FloorMemory`:
     - Add `door_mask: np.ndarray = field(default_factory=lambda: np.zeros((21, 79), dtype=np.bool_))` (and resize it inside `ensure_shape`).
   - In `WorldMemory.update`:
     - When updating `floor.remembered_chars`:
       - For `is_cmap & (chars != 0) & (chars != ord(" "))`: copy `chars[mask]` into `floor.remembered_chars[mask]`, **EXCEPT** where `_IS_OPEN_DOOR_GLYPH_LUT[glyphs]` is True, normalize `floor.remembered_chars` to `ord(".")` so open doors are never confused with `'-'`/`'|'` walls!
       - Update `floor.door_mask`: set `floor.door_mask[_IS_DOORWAY_GLYPH_LUT[glyphs]] = True`, and clear `floor.door_mask` where `is_cmap & (~_IS_DOORWAY_GLYPH_LUT[glyphs])` (e.g. when a door is kicked down into a `"doorway"` floor tile).
       - For any tile containing a visible object or monster (`(_IS_OBJECT_LUT[glyphs] | _IS_MONSTER_LUT[glyphs])`) or the hero's current position `(sy, sx)` where `floor.remembered_chars == 0`, set `floor.remembered_chars` to `ord(".")`.
   - Expose `passable_mask` property on `FloorMemory`, `WorldMemory`, and `GameState`:
     - Returns a boolean `21x79` array where `remembered_chars` is in `(ord('.'), ord('#'), ord('>'), ord('<'), ord('_'), ord('{'), ord('\\'))` (i.e. known walkable floor, corridor, open door, stairs, altar, fountain, throne—strictly excluding walls `ord('-')`, `ord('|')`, solid rock `ord(' ')`, and `0`).
   - On `VisibleMonster`:
     - Add helper method `is_adjacent_to(self, y: int, x: int) -> bool` returning `max(abs(self.y - y), abs(self.x - x)) == 1` (8-way Chebyshev adjacency).
   - On `GameState`:
     - Add `adjacent_hostiles(self) -> list[VisibleMonster]` returning `[m for m in self.visible_monsters() if m.is_hostile and max(abs(m.y - self.y), abs(m.x - self.x)) == 1]`.
2. **`lox/env/grid_utils.py` & `lox/env/state.py` (Doorway-Safe Diagonal Pathfinding)**:
   - In `bfs_distance_grid` and `astar_next_step` in `lox/env/grid_utils.py`:
     - Add optional parameter `no_diag_mask: np.ndarray | None = None` (or pass a boolean `21x79` array `no_diag` defaulting to all-False).
     - During neighbor expansion (`for i in range(n_dirs):`), when `i >= 4` (diagonal step where `dys[i] != 0 and dxs[i] != 0`), if `no_diag[cy, cx] or no_diag[ny, nx]` is True, skip that diagonal transition (`continue`) so doorways are always entered and exited orthogonally!
   - In `GameState.bfs_distances` and `GameState.path_step` (`lox/env/state.py`):
     - Automatically pass `self.mem.current_floor.door_mask` as `no_diag` when `self.mem` is available and shapes match.

---

### 4. Fix `PolicyArchiveEntry` Shard Contamination & Resume Query (`lox/engine/orchestrator.py`)

#### Required Changes in `lox/engine/orchestrator.py`
1. **Include `action_bytes, milestones_json` When Resuming Campaign**:
   - At lines 254–260 of `lox/engine/orchestrator.py`, update the `SELECT` query on `runs` to include `action_bytes, milestones_json` so `extract_shards_from_runs(baseline_runs)` works when resuming from DuckDB.
2. **Never Store `is_shard = True` Runs in `PolicyArchiveEntry.baseline_runs`**:
   - At lines 877–960 of `lox/engine/orchestrator.py`, when a policy is promoted:
     - Merge `full_promoted = {int(r["seed"]): r for r in eval_res["runs"] if not r.get("is_shard")}` into `baseline_by_seed` via `baseline_by_seed.update(full_promoted)` (preserving any prior baseline seeds that weren't re-evaluated in a smaller batch), and set `baseline_runs = list(baseline_by_seed.values())`.
     - Pass `baseline_runs=list(baseline_runs)` (which contains zero `is_shard=True` runs!) into `PolicyArchiveEntry(...)` for `"descent"`, `"outlier"`, and `"survivability"` instead of raw `c_runs`.

---

### 5. Gate `PARAM_TUNE` on Actual Policy Changes & Forward `skills`/`benchmark` (`lox/engine/scheduler.py` & `lox/engine/modes.py`)

#### Required Changes
1. **`lox/engine/scheduler.py` (`ModeScheduler`)**:
   - Add `self._promoted_since_tune: int = 0` and method `record_promotion(self, mode: str) -> None` (or update `record_outcome(self, mode: str, mean_delta: float, wall_sec: float, passed: bool = False)`).
   - In `select_mode`: only trigger periodic `PARAM_TUNE` when `has_tunable_params` is True **AND** `self._promoted_since_tune >= 1` (at least one policy modification was promoted since the last parameter tune, or on the first tune if `self._last_tune_gen == 0` and `generation >= self.param_tune_every`). Reset `self._promoted_since_tune = 0` whenever `PARAM_TUNE` is selected.
2. **`lox/engine/modes.py` (`run_spsa_param_tuning`) & `lox/engine/orchestrator.py`**:
   - Add `skills: dict[str, str] | None = None` and `benchmark: str = "nethack"` parameters to `run_spsa_param_tuning`.
   - Forward `"skills": skills` and `"benchmark": benchmark` in `base_shard_tasks`, `p_tasks`, `m_tasks`, `pool.evaluate_batch(...)`, and `evaluate_paired_hypothesis(val_result["runs"], baseline_runs, benchmark=benchmark)`.
   - Pass `skills=self.load_workspace_skills(), benchmark=self.benchmark` from `CampaignOrchestrator.run_campaign` when calling `run_spsa_param_tuning`.

---

### 6. Fix `MacroHarvester` Collisions & `check_macro_contracts` Method Calls (`lox/lang/validator.py` & `lox/lang/macros.py`)

#### Required Changes
1. **`lox/lang/validator.py` (`check_macro_contracts`)**:
   - In `check_macro_contracts` (lines 384–395), only validate bare function calls (`isinstance(node.func, ast.Name)`) where the function is **NOT** also defined as a top-level function in `policy_tree`, or explicit `macros.<fn>(...)` attribute calls. Never flag `self.<method>(...)` calls.
2. **`lox/lang/macros.py` (`MacroHarvester.harvest_from_policy`)**:
   - Exclude phase-dispatch methods starting with `handle_` or named `explore` (`root_names = {"run", "step", "act", "__init__", "explore"}`) or `item.name.startswith("handle_")` from automatic macro harvesting into `macros.py`, so `macros.py` only harvests pure reusable utility functions (like `move_towards`, `get_passable_mask`, etc.) instead of cloning the entire `Policy` class under duplicate names.

---

## Part B: System Prompt & Cognitive Architecture Overhaul (`lox/author/prompts.py` & `lox/author/agent.py`)

### 1. Context Engineering & Progressive Disclosure (`lox/author/prompts.py`)

1. **Prune Action Definitions to Core Working Set**:
   - In `POLICY_LANG_DOCS` Section 2 ("1-to-1 Actions"):
     - Keep only the essential working set:
       * Compass Movement: `Action.N`, `S`, `E`, `W`, `NE`, `NW`, `SE`, `SW` and `DIR_DELTAS`.
       * Vertical & Step: `Action.UP` (`'<'`), `Action.DOWN` (`'>'`), `Action.WAIT` (`'.'`), `Action.MORE` (`'\r'`).
       * Core Interactions: `Action.SEARCH` (`'s'`), `Action.OPEN` (`'o'`), `Action.KICK` (`C('d')`), `Action.EAT` (`'e'`), `Action.WIELD` (`'w'`), `Action.WEAR` (`'W'`), `Action.QUAFF` (`'q'`), `Action.READ` (`'r'`), `Action.ZAP` (`'z'`), `Action.FIRE` (`'f'`), `Action.THROW` (`'t'`), `Action.PICKUP` (`','`), `Action.DROP` (`'d'`), `Action.PRAY` (`M('p')`).
       * Multi-keystroke helper `seq(...)`, extended commands `extcmd(...)`, and `repeat(...)`.
     - Replace the long list of tertiary commands (`DIP`, `PAY`, `CHAT`, `LOOT`, etc.) with a clear pointer:
       `*(For specialized commands like dipping, chatting, or paying shopkeepers, call query_policy_docs("actions") to view the full command index).*`
2. **Elevate Crucial GameState Perception Fields**:
   - Explicitly highlight in Section 3 ("Querying `GameState`"):
     * `state.inventory`: `item.letter`, `item.name`, `item.is_weapon`, `item.is_armor`, `item.is_food`, `item.is_worn`, `item.is_wielded`.
     * `state.ac`: Numerical armor class (starting AC is 10; lower is better; wearing armor drops AC and dramatically reduces monster melee hit chance).
     * `state.hp` and `state.max_hp`: Hero current and max hit points.
     * `state.hunger_state`: `0`=Satiated, `1`=Not Hungry, `2`=Hungry, `3`=Weak, `4`=Fainting, `5`=Fainted.
     * `state.visible_monsters()` and `state.adjacent_hostiles()`: `m.y`, `m.x`, `m.name`, `m.is_hostile`, `m.is_adjacent_to(state.y, state.x)` (8-way Chebyshev adjacency—monsters attack diagonally as well as orthogonally!), `m.info.difficulty`, `m.info.ac`.
     * `state.visible_objects()`: `o.y`, `o.x`, `o.name`, `o.is_corpse`, `o.is_food`, `o.corpse_monster`.
     * `state.mem`: `passable_mask` (ready-to-use `21x79` boolean mask of remembered walkable floor, corridor, open door, and stair tiles, excluding walls `|-` and solid rock `' '`), `visited`, `visit_count`, `locked_doors`, `searched_counts`, `stair_graph`, `is_oscillating`.

---

### 2. Epistemological Differential SQL Templates (`build_system_prompt` in `lox/author/prompts.py`)

Add a dedicated section in `build_system_prompt` using the new `champion_runs` / `champion_function_stats` views and `expert_ticks`:
```markdown
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
```

---

### 3. Mechanistic Trace Inversion & Binding Constraint Rules (`lox/author/prompts.py`)

In `build_system_prompt` and `build_user_prompt` (under `Scientific Investigation Protocol`):
1. **Mechanistic Trace Inversion via `query_wiki`**:
   - `Every event in NetHack (messages, status changes, deaths) is governed by an exact C function. Isolate the exact message string from the final 3 lines of game_log (e.g. "You feel hungry", "The door resists", "The jackal bites!"), strip proper nouns, and pass that phrase to query_wiki(query=...) to learn the exact C rule that caused it.`
2. **The Binding Constraint Rule**:
   - `Always inspect the Termination Breakdown and Stall Diagnostics in the batch digest. Address the single failure mode responsible for the LARGEST percentage of failures (the binding constraint). Never spend a generation fixing a 2% edge case when 70% of runs died to the same primary cause.`
3. **Anti-Hallucination & Dungeon Context Anchor**:
   - `Anchor your reasoning to your current dungeon depth (state.depth). On Dungeons of Doom levels 1–4, there are NO Sokoban boulders, NO Medusa stoning, and NO Quest portals. Use state.mem.passable_mask (or ensure ord('-'), ord('|'), and ord(' ') are strictly False in custom masks) and remember monsters attack across all 8 adjacent tiles (max(abs(dy), abs(dx)) == 1).`
4. **Pre-Submission Invariant Checklist**:
   - *Generator Invariant*: `run()` receives state via `state = yield act`. Subgenerators are called via `state = yield from self.subgen(state)`. Never use a bare `return act` inside a generator.
   - *State Persistence*: Specialist methods must update `self.*` attributes (e.g. `self.target_tile`, `self.tried_doors`) to prevent 2-tile ping-pong oscillations.
   - *Passability & 8-Way Adjacency*: Never mark `ord('-')`, `ord('|')`, or `ord(' ')` (solid stone) as passable; prefer `state.mem.passable_mask`. Always check 8-way adjacency (`max(abs(dy), abs(dx)) == 1` or `state.adjacent_hostiles()`) in combat and door interactions.
   - *Trigger Reachability*: Any modified or newly added specialist method must be called from `run()`.

---

### 4. Phase-Isolated Turn Gating in `lox/author/agent.py`

In `AuthorAgent.synthesize`:
1. **Turn 1 Mandatory Investigation Enforcement**:
   - On `turn_idx == 1` in `FUNC_MOD_ADD` and `STRUCTURAL_REFACTOR` modes:
     - If the model outputs code without having called `query_db` or `query_wiki`:
       - Do **NOT** execute auto-pilot.
       - Append a user reminder message:
         ```python
         messages.append({
             "role": "user",
             "content": (
                 "🛑 PHASE 1 ENFORCEMENT: Turn 1 is strictly reserved for Empirical Research.\n"
                 "DO NOT output code yet. You must first investigate the previous batch's failure modes:\n"
                 "1. Call `query_db(sql=...)` to compare `champion_runs` / `champion_function_stats` against `expert_runs` or `expert_ticks`.\n"
                 "2. Call `query_wiki(query=...)` on the exact message strings from the top failure causes.\n"
                 "Once you have gathered empirical C engine facts, formulate your hypothesis on Turn 2."
             ),
         })
         continue
         ```
2. **Hypothesis Structure Enforcement**:
   - In `extract_yaml_and_python`, ensure `hyp` captures `target_functions`, `binding_constraint`, `verified_c_rule`, `mechanism`, and `state_persistence` (mapping `observation_from_db` compatibility so existing callers/loggers work seamlessly).

---

## Part C: Verification & Regression Tests (`tests/test_overhaul_verification.py` & `tests/test_author_and_ui.py`)

Add tests verifying all Pass 4 upgrades:
1. `test_pass4_system_prompt_restructuring_and_turn1_gating()` in `tests/test_author_and_ui.py`:
   - Verify `build_system_prompt()` contains `Empirical Differential Diagnostics`, `Pre-Submission Invariant Checklist`, `champion_runs`, and pruned action working set.
   - Verify `AuthorAgent.synthesize` blocks Turn-1 code generation when zero research tools were called in `FUNC_MOD_ADD`.
2. `test_pass4_codebase_bottleneck_fixes()` in `tests/test_overhaul_verification.py`:
   - Verify `champion_runs`, `champion_function_stats`, `champion_ticks`, and `latest_pilot_runs` views exist in DuckDB and that `AuthorToolRegistry.query_db("SELECT COUNT(*) FROM runs WHERE batch_id = (SELECT MAX(batch_id) FROM runs)")` resolves to the promoted champion batch instead of a `pilot_*` batch.
   - Verify `ingest_expert_runs_into_duckdb` populates `expert_ticks` with `depth >= 2` rows (`SELECT COUNT(*) FROM expert_ticks WHERE depth >= 2` > 0).
   - Verify `WorldMemory` normalizes open doors to `ord('.')` in `remembered_chars`, sets `door_mask = True` on doors, exposes `state.mem.passable_mask` and `state.adjacent_hostiles()`, and that `astar_next_step` / `state.path_step` avoids diagonal steps into/out of `door_mask` tiles.
   - Verify `ModeScheduler` only selects `PARAM_TUNE` after at least one promoted modification.
3. Run full lint and test suite:
   ```bash
   uv run ruff check --fix .
   uv run ruff format .
   uv run pytest -v
   ```
   All tests must pass with 0 warnings.
