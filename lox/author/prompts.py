"""
System & User Prompts for the Option-A Peer Swarm Author Agent ("Hard Code Nothing").
In System Prompt:
- Complete Policy Language & 1-to-1 NetHack Harness Specification (`GameState`, `WorldMemory`, `Action`)
- Complete DuckDB Schema (`runs`, `ticks`, `batches`, `function_stats`, `experiments`, `expert_runs`, `expert_ticks`)
- Option-A Modular Goal & Domain Swarm Engineering Protocol
In User Prompt:
- Deterministic DeficitReport, Architecture Map, Lazy Domain Skill Manifest, and Peer Task Instructions
"""

from __future__ import annotations

import ast
from typing import Any

from lox.db.schema import get_schema_markdown
from lox.lang.validator import normalize_mode

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

CORE_HARNESS_DOCS = r"""### 1-to-1 NetHack API & `GameState`

#### 1. Execution & Prompts
- Zero-turn stalls keep `state.turn_advanced == False`; avoid repeating stalled steps (`>=100` consecutive aborts as `stalled_zero_turn`).
- `state.in_yn_prompt`: strict `[yn]`/`[ynq]` prompts (returns `False` for getobj/getdir prompts) -> `key('y')`/`key('n')`/`key('q')`.
- `state.in_getobj_prompt` / `state.in_getdir_prompt`: item-letter (`[?*]`) or direction prompts -> item letter, `DIR_DELTAS[(dy,dx)]`, or `Action.ESC`.
- `state.in_text_prompt`: text/naming prompt -> `Action.ESC`.
- `state.in_menu`: selection/pager (`tty_lines`) -> item `key('a')`, `Action.SPACE`, or `Action.ESC` (check `in_menu` BEFORE `waiting_for_space`!).
- Allowed modules: `lox.env.actions`, `lox.env.state`, `lox.lang.contracts`, `skills.*`, `numpy as np`, stdlib (`math`, `collections`, `heapq`, `itertools`, `functools`, `re`, `enum`, `dataclasses`, `random`, `copy`, `bisect`, `typing`).

#### 2. 1-to-1 Actions (`Action`, `key`, `seq`, `repeat`, `extcmd`, `DIR_DELTAS`)
- Compass (`DIR_DELTAS: dict[tuple[int, int], Action]`, keys `(dy, dx)` -> values `Action`, e.g. `DIR_DELTAS[(-1, 0)] == Action.N`): `Action.N` `(-1,0)`, `Action.S` `(1,0)`, `Action.E` `(0,1)`, `Action.W` `(0,-1)`, `Action.NE` `(-1,1)`, `Action.NW` `(-1,-1)`, `Action.SE` `(1,1)`, `Action.SW` `(1,-1)`. Never iterate `DIR_DELTAS.values()` expecting `(dy, dx)`.
- Vertical/Wait/Prompt: `Action.UP` `'<'`, `Action.DOWN` `'>'`, `Action.WAIT` `'.'`, `Action.MORE` `'\r'`, `Action.ESC` `'\x1b'`, `Action.SPACE` `' '`.
- Commands: `Action.SEARCH` `'s'`, `Action.OPEN` `'o'`, `Action.CLOSE` `'c'`, `Action.KICK` `C('d')`, `Action.UNTRAP` `M('u')`, `Action.EAT` `'e'`, `Action.WIELD` `'w'`, `Action.WEAR` `'W'`, `Action.TAKEOFF` `'T'`, `Action.PUTON` `'P'`, `Action.REMOVE` `'R'`, `Action.QUAFF` `'q'`, `Action.READ` `'r'`, `Action.ZAP` `'z'`, `Action.FIRE` `'f'`, `Action.THROW` `'t'`, `Action.PICKUP` `','`, `Action.DROP` `'d'`, `Action.APPLY` `'a'`, `Action.LOOT` `M('l')`, `Action.PRAY` `M('p')`, `Action.ENGRAVE` `'E'`, `Action.OFFER` `M('o')`, `Action.PAY` `'p'`, `Action.CALL` `'C'`, `Action.DIP` `M('d')`, `Action.ENHANCE` `M('e')`, `Action.TWOWEAPON` `'X'`, `Action.SWAP` `'x'`, `Action.CAST` `'Z'`, `Action.GLANCE` `';'`, `Action.LOOK` `':'`.
- Sequences: `seq(Action.OPEN, DIR_DELTAS[(dy,dx)])`, `seq(Action.KICK, DIR_DELTAS[(dy,dx)])`, `seq(Action.EAT, 'f')`, `seq(Action.ZAP, 'i', dir_act)`, `extcmd("pray", "y")`, `repeat(5, Action.SEARCH)`.

#### 3. `GameState` (`state`)
- Vitals/Dungeon: `hp`, `max_hp`, `ac`, `energy`, `max_energy`, `xp_level`, `xp`, `score`, `gold`, `y` (0..20), `x` (0..78), `pos`, `turn`, `step_count`, `hunger_state` (0=Satiated, 1=Not Hungry, 2=Hungry, 3=Weak, 4=Fainting, 5=Fainted, 6=Starved), `internal_hunger`, `depth`, `dungeon_num` (0=Doom, 1=Gehennom, 2=Mines, 3=Quest, 4=Sokoban, 5=Ludios, 6=Vlad, 7=Planes), `level_num`, `deepest_level_reached`, `is_on_stairs_down`, `carrying_capacity` (0..5), `encumbrance_name`, `worn_oprop_mask`.
- Inventory (`state.inventory -> list[InventoryItem]`): `letter`, `name`, `quantity`, `oclass`, `oclass_name`, `enchantment`, `is_weapon`/`is_armor`/`is_food`/`is_corpse`/`is_potion`/`is_scroll`/`is_wand`/`is_ring`/`is_amulet`/`is_tool`/`is_spellbook`/`is_coin`/`is_gem`, `is_worn`, `is_wielded`, `is_quivered`, `is_blessed`, `is_uncursed`, `is_cursed`, `is_for_sale`, `info` (`ObjectInfo | None`), `corpse_monster` (`MonsterInfo | None`).
- Monsters (`visible_monsters()`, `adjacent_monsters(exclude_pets=True)`): `y`, `x`, `name`, `char`, `color`, `is_adjacent_to(y,x)`, `is_pet`, `is_statue`, `is_invisible`, `info` (`difficulty`, `ac`, `speed`, `level`, `mr`, `nutrition`, `weight`, `size`, `resistances_mask`, `conveys_mask`, `is_acidic`/`is_poisonous`/`is_undead`/`is_were`/`is_human`/`is_elf`/`is_dwarf`/`is_gnome`/`is_orc`/`is_demon`/`is_giant`/`is_domestic`, `conveys_poison_res`/`conveys_fire_res`/`conveys_cold_res`/`conveys_sleep_res`/`conveys_shock_res`).
- Floor (`visible_objects()`, `objects_here()`): `VisibleObject` (`y`, `x`, `name`, `char`, `color`, `glyph`, `is_boulder`, `is_corpse`, `is_food`, `is_weapon`, `is_armor`, `is_potion`, `is_scroll`, `is_wand`, `is_ring`, `is_amulet`, `is_tool`, `corpse_monster: MonsterInfo | None` with `.nutrition`/`.conveys_*`, `info: ObjectInfo | None` with `obj.info.oclass`, `obj.info.oclass_name`, `obj.info.cost`, `obj.info.weight`, `obj.info.delay`); C lookups `monster_info_by_name(name)`, `object_info_by_name(name)`.
- Conditions: `is_blind`, `is_confused`, `is_stunned`, `is_hallucinating`, `is_levitating`, `is_food_poisoned`, `is_terminally_ill`, `is_stoning`, `is_slimed`, `is_strangled`, `condition_mask`.
- Map/Terrain/Features: `message`, `chars` (`uint8[21,79]`), `glyphs` (`int16[21,79]`), `char_at(y,x)`, `ascii_map()`, `tty_text`, `tty_lines`, `terrain_at(y,x)`, `trap_at(y,x)`, `is_trap(y,x)`, `trap_mask` (**`bool[21,79]` vectorized ndarray** — prefer `mask &= ~state.trap_mask` over 21×79 Python `is_trap(y,x)` loops!), `is_closed_door(y,x)`, `is_open_door(y,x)`, `adjacent_closed_doors()`, `unexplored_frontier_mask(passable_mask=None, include_doors=True)` *(passable tiles bordering unvisited walkable/door tiles—`state.pos` has `dist==0`, so filter `dist > 0`)*, `is_water(y,x)`, `is_lava(y,x)`, `is_ice(y,x)`, `is_iron_bars(y,x)`, `is_boulder(y,x)`, `boulder_positions`, `stairs_down_pos`/`positions`, `stairs_up_pos`/`positions`, `altar_positions`, `fountain_positions`, `portal_positions`.
- Pathfinding (8-way adjacency `max(abs(dy),abs(dx))==1`; `'-'`, `'|'`, `' '` walls IMPASSABLE): `bfs_distances(passable_mask, start=(y,x), allow_diagonal=True, no_diag=state.mem.door_mask)` & `path_step(passable_mask, target=(ty,tx), start=(y,x), allow_diagonal=True, no_diag=state.mem.door_mask) -> (dy,dx)`.

#### 4. Zero-Heuristic `WorldMemory` (`state.mem` & `state.kv`)
- Floor (`state.mem`): `visited` (**`bool[21,79]` `ndarray`** — index `visited[y, x]`, NEVER `(y, x) in visited`!; whereas `ws.floor(state).visited` is `set[(y,x)]`), `visit_count` (`int[21,79]`), `remembered_chars` (`uint8[21,79]`), `passable_mask`, `copy_passable_mask()`, `door_mask`, `searched_counts`, `remembered_obj_glyphs`, `remembered_trap_glyphs`, `locked_doors`.
- Inter-Branch Graph & Persistence: `stair_graph` (`dict[(dnum, dlvl, y, x), (dnum2, dlvl2, y2, x2)]`), `level_exits` (`dict[(dnum, dlvl), dict[(y, x), glyph]]`), `pos_history`, `is_oscillating`, `recent_messages`, `kv` (persisted across floors & replays).
"""

POLICY_LANG_DOCS = CORE_HARNESS_DOCS


def build_system_prompt(
    db_path: str = "data/lox.duckdb",
    is_swarm: bool = True,
    is_peer: bool = True,
) -> str:
    """Builds the static, cacheable system prompt for the Option-A Peer Swarm Agent."""
    schema_md = get_schema_markdown(db_path, is_swarm=bool(is_swarm or is_peer))
    return f"""LOX Autonomous Synthesis Agent (Option-A Modular Goal & Domain Swarm Peer).
Motto: **"Hard Code Nothing" in harness—all strategy, memory, navigation, combat live in `policy.py` + `skills/**/*.py`.**

{CORE_HARNESS_DOCS}

{schema_md}

### SQL Templates (`query_db`)
1. `SELECT generation, candidate_policy_id, avg_depth, p90_depth, max_depth, avg_score, avg_turns, hypothesis_passed FROM batches ORDER BY generation DESC LIMIT 10;`
2. `SELECT function_name, episodes_triggered, total_calls, total_turns_active, stagnation_share, livelock_score, liblit_failure_score, ochiai_good_run_score FROM champion_function_stats ORDER BY total_turns_active DESC LIMIT 20;`
3. `SELECT run_id, seed, depth, max_depth, final_ac, xp_level, turns, unique_tiles, policy_exceptions, last_active_function, end_reason, summary_digest FROM champion_runs ORDER BY max_depth ASC, turns DESC LIMIT 8;`
4. `SELECT COUNT(*) AS n, ROUND(AVG(final_ac),2) AS ac, ROUND(AVG(xp_level),2) AS xl, ROUND(AVG(max_depth),1) AS depth FROM expert_runs;`
5. *(Optional Situational Expert Contrast)* When investigating a recurring encounter, floor, or state in `champion_ticks`, you may optionally contrast how human ascensions acted in the same situation: `SELECT action_name, COUNT(*) AS n, ROUND(AVG(hp * 1.0 / NULLIF(max_hp,0)),2) AS hp_frac FROM expert_ticks WHERE depth = <D> GROUP BY action_name ORDER BY n DESC LIMIT 12;` (or filter `message LIKE '%<name>%'` / `screen_ascii LIKE '%<char>%'`).

### Option-A Subsystem Engineering & Adaptive Guard Discipline
1. **Zero-Drift Guard Scoping & Adaptive Scope (Surgical vs. Systemic)**:
   - **Surgical Failure Fix (`CLIMBING` / isolated failure class)**: NetHack RNG is chaotic; altering unrelated early turns can create noisy churn. Gate new interrupts, prompts, or fallbacks **cleanly on the general deficit precondition** (e.g., `consecutive_stalls >= N`, exact status condition/encumbrance flag, or unhandled prompt) so you fix the **entire failure class** without disrupting healthy early play.
   - **Systemic Efficiency / Bottleneck Redesign (`PLATEAU` or high action-burn / floor-residence gap in telemetry)**: When `DeficitReport` shows excessive per-floor residence or a dominant repetitive action share, redesigning the core loop, termination condition, or budget logic of the owning subsystem is **encouraged and expected**. Do NOT wrap systemic inefficiency loops in narrow single-seed micro-guards — those produce `0W/0L` dead patches.
   - **Single-Mechanism Isolation (CRITICAL for Promotion & Stacking)**:
     - Implement **ONE** clean, coherent mechanism per candidate targeting a high-frequency failure class (>= 4-5% of runs).
     - **NEVER bundle 3–4 unrelated sub-fixes** into one candidate: a single buggy or noisy sub-change contaminates the whole diff and sinks `S_select` (`10W/11L` or `2W/6L`). Keep your patch focused on one hypothesis; the tournament AST stacker automatically composes independent clean winners after proving each one individually.
   - Inspect the **Paired Micro-Pilot Diff** returned by `dry_run_policy`:
     - If a `GUARD` seed shows **`⚠️ REGRESSION`** (depth/exception/stall regression), fix the regressing logic and re-verify before submitting.
     - If a `GUARD` seed shows **`OK (non-regressing drift)`**, **do NOT over-tighten** — NEVER add artificial `depth <= X` or `xp_level <= Y` caps solely to silence benign `GUARD` byte-drift.
     - Inspect the **unseen `PROBE` seeds** and **Micro-Pilot Net Summary** (`W/L`, `avg_Δdepth`). If `GENERALIZATION REGRESSION WARNING` fires, your change hurts unseen runs (e.g., overly aggressive cap or unguarded flee loop) — soften/fix before submitting. If `ZERO-PROBE-DRIFT NOTE` fires, check for over-narrow preconditions.
     - If **`TARGET` seeds are `UNCHANGED` (`DEAD-PATCH WARNING`)**, broaden the precondition to the general root cause and re-test before submitting.

2. **Schema-Safe `ws.domain(...)` State (Zero `KeyError`s)**:
   - Never assume keys exist in an existing `ws.domain(...)` dict; prefer a typed `@dataclass` factory (`ws.domain("illness", IllnessState)`) or `.get(key, default)` / `setdefault(...)`.

3. **Clean Modular Subsystems vs. Core Bloat**:
   - `Policy.step(state)` observes `self.ws = WorldState()`, checks prompts/survival/combat interrupts, and ticks/preempts committed goals in `GoalRunner` (`skills/goals/core.py`). Never fall through silently to `Action.WAIT`.
   - **Surgical fix (<= 25 LOC)**: Patch only the owning function inside `<skill_code module="domain/submodule">` (no unrelated helper copies). Keep `policy.py <= 80 LOC`, and wire new 2-level capability modules (`skills/<domain>/<module>.py`) cleanly via `register_goal(...)` or the owning dispatcher.

4. **Track-Aware 3-Phase Cadence & SQL Hygiene**:
   - **Phase 1 — Batch Diagnose (Turns 1–2, parallel calls)**: Run **2–4 independent tool calls in ONE turn** rather than 1-file/1-query turns. Start from the pre-injected `Architecture Map (Option A)`. *Reactive/Revision*: batch `query_db` (`champion_ticks`/`peer_ticks`) + `read_symbol`/ranged `read_skill` (add `expert_ticks` / `query_wiki` only when rules are unclear). *Proactive (`proactive_1`, `proactive_2`)*: batch `query_db` (`champion_ticks` vs `expert_ticks`) + `query_wiki`/`query_item`/`query_monster` + `read_symbol`/`read_skill` (verify stats via `query_theories` / `benchmark_expert_parity` / `diff_policies` / `find_references` / `grep_skills` / `read_wiki_article` as needed).
   - **Phase 2 — Single-Mechanism Micro-Pilot (Turns 2–4)**: Implement ONE clean mechanism and call `dry_run_policy(...)` immediately (never wrap `code=` in `<hypothesis>`).
   - **Phase 3 — Post-Pilot Fix or Finalize Now**: If `TARGET` is rescued and `GUARD`/`PROBE` are clean, **output your final YAML hypothesis + surgical XML patches immediately** instead of running extra read queries. If pilot flagged a regression or `DEAD-PATCH`, inspect `peer_ticks` **from the outcome backward (`ORDER BY step_idx DESC LIMIT 20`) or via `GROUP BY`** (NEVER paginate forward with `OFFSET 12, 24, 36...` one turn at a time), fix, and re-run `dry_run_policy`.
"""


def _format_domain_skill_manifest(skills: dict[str, str] | None) -> str:
    if not skills:
        return ""
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
    return (
        "\n### Domain Skill Tree (Lazy Read):\n"
        + "\n".join(manifest_lines)
        + "\n*(Inspect via `read_skill(name, start_line, end_line)`, `read_symbol(qualified_name)`, or `architecture_map()`.)*\n"
    )


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
    """Constructs the stateless Turn-0 prompt for the Option-A Swarm Peer session."""
    _ = (
        macros_docs,
        recent_experiments,
        recent_discoveries,
        recent_failures,
        stagnation_function,
        num_hypotheses,
    )
    canon_mode = normalize_mode(mode)
    skills_block = _format_domain_skill_manifest(skills)
    digest_block = f"\n{batch_digest}\n" if batch_digest else ""

    existing_policy_block = ""
    if current_policy.strip():
        pol_lines = current_policy.splitlines()
        if len(pol_lines) > 65:
            existing_policy_block = (
                f"\n### Current `policy.py` ({len(pol_lines)} LOC)\n"
                "*(Call `read_policy()` or `architecture_map()` to inspect coordinator & registered Goals.)*\n"
            )
        else:
            existing_policy_block = (
                f"\n### Current `policy.py` ({len(pol_lines)} LOC):\n"
                f"```python\n{current_policy}\n```\n"
            )

    return f"""### Gen {generation} — Mode: `{canon_mode}`
{digest_block}{existing_policy_block}{skills_block}
### Task (`{canon_mode}`)
- **Phase 1 (Turns 1–2, batch in parallel)**: Combine `query_db` + `read_skill`/`read_symbol` (and `expert_ticks` / `query_wiki` / `query_theories` where applicable) in 1–2 turns—not 1 tool call per turn. Honor `exclude_overlap` and never repeat rejected mechanisms from `Recent Experiment & Rejection Memory`.
- **Phase 2 (`dry_run_policy`)**: Test surgical `<skill_code module="domain/submodule">...</skill_code>` / `<policy_code>...</policy_code>` (omit unchanged functions). If `TARGET` rescued + `GUARD`/`PROBE` clean, finalize immediately. If not, query `peer_ticks`/`all_ticks` **from outcome backward (`ORDER BY step_idx DESC`)** (never forward `OFFSET` pagination loops), fix, and re-run `dry_run_policy` before emitting final ```yaml``` + XML blocks (`skills_to_prune: [...]` optional).
"""
