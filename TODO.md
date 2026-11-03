# LOX System Prompt & Runtime Bottleneck Overhaul (`TODO.md` — Pass 6)

> **Architectural Mandate ("Hard Code Nothing")**:
> Preserve strict zero-heuristic generality in `lox/env/` and `lox/author/prompts.py`.
> Do NOT hardcode NetHack strategy guides or domain heuristics.
> After completing all 6 tasks below, run `uv run ruff check --fix .`, `uv run ruff format .`, and `uv run pytest -v` until 100% of tests pass with zero warnings or errors.

---

## Task 1: Fix `ExecutionTracer` Call-Stack Attribution, Boolean Helper Calls, and Missing `Any` Import (`lox/lang/instrument.py`)

**Problem**:
1. `lox/lang/instrument.py:91` uses `Any` without importing `from typing import Any`, causing `ruff check` to fail with `F821 Undefined name Any`.
2. `tests/test_author_and_ui.py::test_macros_called_from_policy_and_traced` fails because `ExecutionTracer.on_return` treats `val is False` as `is_noop = True`, preventing boolean predicate functions/macros (such as `macro_is_low_hp(state, 0.5)` returning `False`) from incrementing `self.fn_calls[fn_name]`.
3. `tests/test_lang_instrument.py::test_call_stack_attribution_and_caller_splice_filtering` fails because `ExecutionTracer.on_return` adds leaf helpers (such as `get_passable_mask` called inside `handle_combat`) to `self._turn_contributors` and `record_turn_yield()` increments `self.fn_turns` for all contributors, AND `exit_fn` only restores `self.active_fn = caller_fn` when `yielded_inside` is True.

**Exact Changes in `lox/lang/instrument.py`**:
1. At the top of `lox/lang/instrument.py` (around line 12), add:
   ```python
   from typing import Any
   ```
2. In `ExecutionTracer` (lines 43–141):
   - Remove `"_turn_contributors"` from `__slots__`, `__init__`, `reset_for_episode`, `on_return`, and `record_turn_yield`.
   - Use 3-element list frames `[fn_name, self.total_turns, False]` in `self.call_stack` (where index `2` tracks `returned_non_noop: bool`).
   - Replace `enter_fn`, `on_return`, `exit_fn`, and `record_turn_yield` with:
     ```python
     def enter_fn(self, fn_name: str, has_value_return: bool = False) -> None:
         self.call_stack.append([fn_name, self.total_turns, False])
         if not has_value_return:
             self.fn_calls[fn_name] = self.fn_calls.get(fn_name, 0) + 1
             self.active_fn = fn_name


     def on_return(self, fn_name: str, val: Any) -> Any:
         is_noop = val is None or (
             isinstance(val, tuple) and len(val) == 2 and val == (0, 0)
         )
         if not is_noop:
             self.fn_calls[fn_name] = self.fn_calls.get(fn_name, 0) + 1
             if self.call_stack:
                 self.call_stack[-1][2] = True
             if val is not False:
                 self.active_fn = fn_name
         return val


     def exit_fn(self, fn_name: str) -> None:
         turns_at_enter = self.total_turns
         returned_non_noop = False
         if self.call_stack:
             top_frame = self.call_stack.pop()
             top_fn, turns_at_enter, returned_non_noop = (
                 top_frame[0],
                 top_frame[1],
                 top_frame[2],
             )
             if top_fn != fn_name:
                 while self.call_stack and self.call_stack[-1][0] != fn_name:
                     f = self.call_stack.pop()
                     turns_at_enter, returned_non_noop = f[1], f[2]
                 if self.call_stack and self.call_stack[-1][0] == fn_name:
                     f = self.call_stack.pop()
                     turns_at_enter, returned_non_noop = f[1], f[2]

         if not self.call_stack:
             return

         caller_fn = self.call_stack[-1][0]
         yielded_inside = self.total_turns > turns_at_enter
         if caller_fn not in self._ROOT_FNS:
             self.active_fn = caller_fn
         elif not yielded_inside and not returned_non_noop:
             self.active_fn = caller_fn


     def record_turn_yield(self) -> str:
         """Called by the runner each time the policy yields an action to the harness."""
         self.loop_ticks = 0
         self.total_turns += 1
         fn = self.active_fn
         self.fn_turns[fn] = self.fn_turns.get(fn, 0) + 1
         if self.call_stack:
             self.active_fn = self.call_stack[-1][0]
         else:
             self.active_fn = "root"
         return fn
     ```

---

## Task 2: Fix Campaign-Destroying MAP-Elites Archive Silent Regression Bug (`lox/engine/orchestrator.py`)

**Problem**:
In `CampaignOrchestrator` (`lox/engine/orchestrator.py:40-51` and `373-393`):
1. `PolicyArchiveEntry.total_reward` is initialized to `0.0` and is **never updated anywhere**, so `q = e.total_reward / max(1, e.pull_count)` in line 378 is always `0.0`.
2. When Gen 1 promotes a deeper policy into `"descent"` and `"outlier"`, `"survivability"` can retain the obsolete Gen 0 policy (`pull_count = 0` or `1`). Because `q` is `0.0`, UCB round-robins to `"survivability"` on Gen 3 or Gen 4, **silently reverting `current_policy` and `baseline_runs` back to Gen 0** while leaving `last_batch_digest` from the deeper policy! Worse still, if a child of that Gen 0 policy beats Gen 0 (`1.25 > 1.20`), `self.save_active_policy(current_policy)` overwrites `data/workspace/policy.py` with the regressed policy!

**Exact Changes in `lox/engine/orchestrator.py`**:
1. In `PolicyArchiveEntry` (around line 51), add `batch_digest: str = ""` field:
   ```python
   @dataclass
   class PolicyArchiveEntry:
       policy_id: str
       policy_code: str
       niche: str
       avg_depth: float
       p90_depth: float
       max_depth: int
       avg_ac: float
       avg_xl: float
       pull_count: int = 0
       total_reward: float = 0.0
       baseline_runs: list[dict[str, Any]] = field(default_factory=list)
       batch_digest: str = ""
   ```
2. In `CampaignOrchestrator.run_campaign` (lines 372–393), update MAP-Elites UCB parent selection so stale niches lagging more than `0.10` depth behind `"descent"` are upgraded to `"descent"`, `q` includes `e.avg_depth + (e.total_reward / max(1, e.pull_count))`, and `last_batch_digest` is synchronized with `chosen_entry.batch_digest`:
   ```python
   # MAP-Elites UCB parent selection
   chosen_niche = "descent"
   if self.policy_archive:
       if "descent" in self.policy_archive:
           champ_entry = self.policy_archive["descent"]
           for n_key, n_entry in list(self.policy_archive.items()):
               if n_entry.avg_depth < champ_entry.avg_depth - 0.10:
                   self.policy_archive[n_key] = PolicyArchiveEntry(
                       policy_id=champ_entry.policy_id,
                       policy_code=champ_entry.policy_code,
                       niche=n_key,
                       avg_depth=champ_entry.avg_depth,
                       p90_depth=champ_entry.p90_depth,
                       max_depth=champ_entry.max_depth,
                       avg_ac=champ_entry.avg_ac,
                       avg_xl=champ_entry.avg_xl,
                       baseline_runs=list(champ_entry.baseline_runs),
                       batch_digest=champ_entry.batch_digest,
                   )
       best_niche = "descent"
       best_score = -1e9
       total_p = sum(e.pull_count for e in self.policy_archive.values())
       for n_name, e in self.policy_archive.items():
           q = e.avg_depth + (e.total_reward / max(1, e.pull_count))
           u = q + 0.15 * math.sqrt(2.0 * math.log(total_p + 1.0) / max(1, e.pull_count))
           if u > best_score:
               best_score = u
               best_niche = n_name
       chosen_niche = best_niche
       chosen_entry = self.policy_archive[best_niche]
       chosen_entry.pull_count += 1
       current_policy = chosen_entry.policy_code
       current_policy_id = chosen_entry.policy_id
       baseline_runs = chosen_entry.baseline_runs
       baseline_by_seed = {int(r["seed"]): r for r in baseline_runs if "seed" in r}
       if chosen_entry.batch_digest:
           last_batch_digest = chosen_entry.batch_digest
   ```
3. After `verdict` is computed (around line 840, right before `self.scheduler.record_outcome`), update `total_reward` on the pulled niche entry:
   ```python
                if self.policy_archive and chosen_niche in self.policy_archive:
                    self.policy_archive[chosen_niche].total_reward += max(
                        0.0, float(verdict.get("mean_delta", 0.0))
                    )
   ```
4. In the `if verdict["passed"]:` block (lines 868–970):
   - Compute `last_batch_digest = build_batch_contrast_digest(eval_res, eval_res["runs"], fn_stats)` BEFORE updating `self.policy_archive` so `batch_digest=last_batch_digest` can be stored on each `PolicyArchiveEntry`.
   - Only call `self.save_active_policy(current_policy)` when `"descent" not in self.policy_archive or float(eval_res["avg_depth"]) >= self.policy_archive["descent"].avg_depth`.
   - When updating `"survivability"`, require `c_depth >= self.policy_archive["survivability"].avg_depth - 0.05` so a much shallower policy never overwrites `"survivability"`.

---

## Task 3: Fix Conditional Episode Trigger Filter Bypass on `priority_seeds` (`lox/runner/pool.py`, `lox/engine/orchestrator.py`, `lox/author/tools.py`)

**Problem**:
In `lox/runner/pool.py` (lines 536–550), every seed in `priority_seeds` gets `is_pri = True` and `"enforce_trigger": not is_pri and bool(target_fns)` (`False`).
Because `orchestrator.py` (line 676) and `tools.py` (line 678) pass up to 250 `baseline_runs` seeds in `priority_seeds` (and `250 >= 150`), **`enforce_trigger` is `False` on 100% of episodes in `FUNC_MOD_ADD`**!
When the Author Agent adds a new specialist method (or modifies one that only triggers on 15% of seeds), `0` untriggered episodes are discarded (`episodes_discarded = 0`), `unreachable_abort` never fires even if the method triggers 0 times, and 85%–100% of the batch consists of untriggered ties (`0W / 0L / 130T`).
Mathematically, to eliminate survivorship bias *without* keeping untriggered ties, an episode on seed $s$ must be kept iff $T_{\text{base}}(s) = 1 \lor T_{\text{cand}}(s) = 1$: `enforce_trigger` should ONLY be `False` on seeds where the baseline *already* triggered `target_functions` (`unconditional_seeds`), while still allowing `priority_seeds` to order existing baseline seeds first in `seed_queue`!

**Exact Changes**:
1. In `lox/runner/pool.py` (`PersistentEvalPool.evaluate_batch`, lines 395–465 and 536–550):
   - Add parameter `unconditional_seeds: list[int] | set[int] | None = None` to `evaluate_batch`.
   - Set:
     ```python
     priority_seeds_set = {int(s) for s in (priority_seeds or [])}
     unconditional_seeds_set = (
         {int(s) for s in unconditional_seeds}
         if unconditional_seeds is not None
         else priority_seeds_set
     )
     ```
   - At line 536, change `is_pri = s in priority_seeds_set` to:
     ```python
     is_pri = s in unconditional_seeds_set
     ```
     (So direct callers/tests that pass `priority_seeds` without `unconditional_seeds` behave identically, while `orchestrator.py` and `tools.py` can pass `unconditional_seeds=base_triggered_seeds`!).
2. In `lox/engine/orchestrator.py` (lines 646–677 and 731–746):
   - Separate `base_triggered_seeds` (ONLY seeds where `is_trig = bool(target_set and (target_set & fn_keys))` is True) from `base_fail_seeds` and `base_remaining_seeds`:
     ```python
     priority_seeds = None
     unconditional_seeds: list[int] = []
     eval_shards = None
     if baseline_runs:
         base_triggered_seeds: list[int] = []
         base_fail_seeds: list[int] = []
         base_remaining_seeds: list[int] = []
         target_set = set(target_fns or [])
         for r in baseline_runs:
             fn_map = r.get("triggered_functions") or r.get("functions_triggered")
             if isinstance(fn_map, str):
                 try:
                     fn_map = json.loads(fn_map)
                 except Exception:
                     fn_map = {}
             fn_keys = set(fn_map.keys()) if isinstance(fn_map, dict) else set()
             is_trig = bool(target_set and (target_set & fn_keys))
             is_fail = r.get("end_reason") not in (
                 "ascended",
                 "max_steps",
                 "ended",
             )
             s = int(r["seed"])
             if is_trig:
                 base_triggered_seeds.append(s)
             elif is_fail:
                 base_fail_seeds.append(s)
             else:
                 base_remaining_seeds.append(s)
         unconditional_seeds = list(base_triggered_seeds)
         priority_seeds = base_triggered_seeds + base_fail_seeds + base_remaining_seeds
     ```
   - Pass `unconditional_seeds=unconditional_seeds if proposal.mode == "FUNC_MOD_ADD" else None` into `self.pool.evaluate_batch(...)` at line 731!
3. In `lox/author/tools.py` (`dry_run_policy` pilot evaluation, lines 639–730):
   - Compute `base_trig_seeds: list[int] = []` for baseline runs that already triggered `changed_fns`:
     ```python
     target_set = set(changed_fns or [])
     base_trig_seeds: list[int] = []
     for r in self.baseline_runs:
         fn_map = r.get("triggered_functions") or r.get("functions_triggered")
         if isinstance(fn_map, str):
             try:
                 import json

                 fn_map = json.loads(fn_map)
             except Exception:
                 fn_map = {}
         fn_keys = set(fn_map.keys()) if isinstance(fn_map, dict) else set()
         if target_set and (target_set & fn_keys) and "seed" in r:
             base_trig_seeds.append(int(r["seed"]))
     ```
   - Also append remaining `self.baseline_runs` seeds into `pilot_seeds` after `strat_runs` so the pool has all baseline seeds available in `seed_queue` if the first 20 don't trigger `changed_fns`, and pass `unconditional_seeds=base_trig_seeds` to `self.pool.evaluate_batch(...)` at line 718, AND backfill any newly triggered pilot seeds not in `base_by_seed` by evaluating `self.current_policy_code` on those missing seeds (just like `orchestrator.py` does) so 100% of triggered pilot runs are paired 1-to-1 against baseline!

---

## Task 4: Fix `--max-steps` Premature Truncation Default (`scripts/run_lox.py`, `lox/runner/pool.py`) and Configurable Paths in `scripts/reset_clean.py`

**Problem**:
1. `scripts/run_lox.py` line 80 sets `--max-steps` default to `2500` (and `lox/runner/pool.py:31` sets `_WORKER_MAX_STEPS = 2500`). In live campaigns, 35%–66% of runs are artificially truncated by `max_steps` at 2,500 steps while thoroughly exploring Dlvl 1–4.
2. `scripts/reset_clean.py` hardcodes `DB_PATH`, `POLICY_PATH`, and `MACROS_PATH`, ignoring `--db-path`, `--policy-path`, and `--macros-path` passed to `scripts/run_lox.py --clean`, and does not clean `data/workspace/skills/*.py`.

**Exact Changes**:
1. In `scripts/run_lox.py` (line 80) and `lox/runner/pool.py` (line 31):
   - Change the default `--max-steps` in `scripts/run_lox.py` from `2500` to `5000` (`default=5000, help="Max steps per episode (default: 5000)"`), and `_WORKER_MAX_STEPS: int = 5000` in `lox/runner/pool.py`.
2. In `scripts/reset_clean.py`:
   - Change signature to:
     ```python
     def reset_clean(
         db_path: str = DB_PATH,
         policy_path: str = POLICY_PATH,
         macros_path: str = MACROS_PATH,
     ) -> None:
     ```
   - Use `db_path`, `policy_path`, and `macros_path` throughout `reset_clean`, and also remove any `.py` files inside `Path(policy_path).parent / "skills"` if that directory exists.
3. In `scripts/run_lox.py` (lines 161–165):
   - Pass CLI paths to `reset_clean`:
     ```python
     if args.clean:
         from scripts.reset_clean import reset_clean

         reset_clean(
             db_path=args.db_path,
             policy_path=args.policy_path,
             macros_path=args.macros_path,
         )
     ```

---

## Task 5: Deduplicate `POLICY_LANG_DOCS` in `lox/author/prompts.py` & Document `state.objects_here()`

**Problem**:
1. In `lox/author/prompts.py`, `POLICY_LANG_DOCS` duplicates the entire `state.mem` (`WorldMemory` & `FloorMemory`) bullet list twice (once in Section 3 lines 119–129 and again in Section 4 lines 179–191).
2. Section 3 (`Visible Objects & Corpses`, lines 114–118) does not mention `state.objects_here() -> list[VisibleObject]` or that `state.visible_objects()` includes objects on the hero's tile `(state.y, state.x)`.
3. Section 4 (`Generic Vectorized Pathfinding Usage`, lines 167–177) uses `np.isin(state.chars, [ord('.'), ord('#'), ord('>')])` instead of `state.mem.passable_mask.copy()`.

**Exact Changes in `lox/author/prompts.py`**:
1. In Section 3 (`Visible Objects & Corpses`, lines 114–118), document:
   - `state.visible_objects() -> list[VisibleObject]` (includes objects at `(state.y, state.x)` when standing on an item/corpse)
   - `state.objects_here() -> list[VisibleObject]` (returns objects specifically at the hero's current position `(state.y, state.x)`)
2. Remove the duplicate `WorldMemory & FloorMemory` block from Section 3 (lines 119–129), keeping the canonical Section 4 (`4. Deterministic Zero-Heuristic WorldMemory (state.mem)`).
3. In Section 4 (`Generic Vectorized Pathfinding Usage`, lines 167–177), replace `passable = np.isin(state.chars, [ord('.'), ord('#'), ord('>')])` with `passable = state.mem.passable_mask.copy()`.

---

## Task 6: Align YAML Hypothesis Keys in `lox/ui/console.py` & `lox/db/logger.py` and Turn-Ratio Calculation in `lox/db/shards.py`

**Problem**:
1. `lox/author/prompts.py` instructs the Author Agent to emit `binding_constraint` and `verified_c_rule` in its YAML hypothesis block for `FUNC_MOD_ADD` and `STRUCTURAL_REFACTOR`, but `lox/ui/console.py` (`hypothesis_box`, line 122) and `lox/db/logger.py` (`save_experiment`, line 341) only look for `observation_from_db` or `causal_finding`, causing the console UI to print `Finding: N/A` and DuckDB `experiments.observation_from_db` to be saved as `""`.
2. In `lox/db/shards.py` (`extract_shards_from_runs`, lines 136 and 186), `stall_start_turn` and `d_turn_val` are NetHack game **turns**, but the fallback byte-cut ratio divides them by `steps` instead of `turns = int(r.get("turns") or steps)`.

**Exact Changes**:
1. In `lox/ui/console.py` (`hypothesis_box`, lines 122–126) and `lox/db/logger.py` (`save_experiment`, lines 341–345):
   - Extract the finding string using:
     ```python
     obs = str(
         hypothesis.get("observation_from_db")
         or (
             f"{hypothesis.get('binding_constraint')} | Rule: {hypothesis.get('verified_c_rule')}"
             if hypothesis.get("binding_constraint") and hypothesis.get("verified_c_rule")
             else (
                 hypothesis.get("binding_constraint")
                 or hypothesis.get("verified_c_rule")
                 or hypothesis.get("causal_finding")
             )
         )
         or "N/A"  # (or "" in logger.py)
     )
     ```
2. In `lox/db/shards.py` (`extract_shards_from_runs`, around line 98):
   - Define `turns = int(r.get("turns") or steps)` and use `ratio = cut_step / float(max(1, turns))` at lines 136 and 186 (where `cut_step` is derived from `stall_start_turn` and `d_turn_val`).
3. Add a unit test `test_pass6_audit_fixes` in `tests/test_overhaul_verification.py` verifying:
   - `ExecutionTracer` attributes turns to `handle_combat` (not leaf helper `get_passable_mask`) and counts boolean macro calls returning `False` (`macro_is_low_hp`).
   - `PersistentEvalPool.evaluate_batch` with `priority_seeds` and `unconditional_seeds=[]` discards untriggered episodes until collecting `target_episodes` triggered runs.
   - `ConsoleUI.hypothesis_box` and `BatchDatabaseLogger.save_experiment` properly record `binding_constraint` and `verified_c_rule`.
