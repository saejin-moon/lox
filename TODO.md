# LOX Overnight Campaign & Runtime Audit Overhaul (`TODO.md` — Pass 6)

> **Architectural Mandate ("Hard Code Nothing")**:
> Preserve strict zero-heuristic generality in `lox/env/` and `lox/author/prompts.py`.
> Do NOT hardcode NetHack strategy guides or domain heuristics.
> After completing all 6 tasks below, run `uv run ruff check --fix .`, `uv run ruff format .`, and `uv run pytest -v` until 100% of tests pass with zero warnings or errors.

---

## Task 1: Simplify & Harden `ExecutionTracer` Call-Stack Attribution (`lox/lang/instrument.py`)

**Problem**:
In `lox/lang/instrument.py`, `ExecutionTracer` currently uses `_turn_contributors` and `_recent_returns` value-equality heuristics. A cleaner, exact 3-element stack frame `[fn_name, self.total_turns, returned_non_noop]` handles all synchronous helpers, `None`-returning guards (both explicit `return None` and implicit fallthrough `None`), boolean predicates (`macro_is_low_hp`), and generators in `<1 µs`.

**Exact Changes in `lox/lang/instrument.py`**:
1. Ensure `from typing import Any` is imported at the top of `lox/lang/instrument.py`.
2. In `ExecutionTracer`:
   - Remove `"_turn_contributors"` and `"_recent_returns"` from `__slots__`, `__init__`, and `reset_for_episode`.
   - Use 3-element list frames `[fn_name, self.total_turns, False]` in `self.call_stack`:
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

## Task 2: Fix Campaign-Destroying MAP-Elites Archive Silent Regression, `PARAM_TUNE` Archive Sync, and Resume SQL Bug (`lox/engine/orchestrator.py`)

**Problem**:
1. In `PolicyArchiveEntry` (`lox/engine/orchestrator.py:40-51`), `total_reward` is initialized to `0.0` and never updated anywhere, making `q = e.total_reward / max(1, e.pull_count)` in line 378 always `0.0`. When Gen 1 promotes a deeper policy into `"descent"` and `"outlier"`, `"survivability"` can keep the obsolete Gen 0 policy. A few generations later, UCB round-robins to `"survivability"`, silently reverting `current_policy` and `baseline_runs` back to Gen 0 while keeping `last_batch_digest` from the deeper policy, and overwriting `data/workspace/policy.py` if a child beats Gen 0!
2. In `PARAM_TUNE` promotion (`lines 485–508`), `baseline_by_seed = full_promoted` replaces all 250 baseline seeds with only 100 seeds (instead of `baseline_by_seed.update(full_promoted)`), and `self.policy_archive` is NOT updated, so on the very next generation the MAP-Elites selector at line 387 overwrites `current_policy` back to the pre-`PARAM_TUNE` policy!
3. In campaign resume (`lines 236–244`), the SQL query selects `p.candidate_policy_id` from `policies p` (the column is actually `p.policy_id`!), throwing a `BinderException` that is silently swallowed by `except Exception: pass`.

**Exact Changes in `lox/engine/orchestrator.py`**:
1. In `PolicyArchiveEntry` (around line 51), add `batch_digest: str = ""` field.
2. In campaign resume (`lines 236–244`), fix the SQL query to use `p.policy_id` and exclude `PILOT` / `BASELINE_BACKFILL` / `MACRO_WRITE` batches:
   ```python
                    prom_row = con.execute(
                        """
                        SELECT p.policy_id, p.params_json, b.batch_id
                        FROM policies p
                        JOIN batches b ON b.candidate_policy_id = p.policy_id
                        WHERE p.is_promoted = TRUE
                          AND b.mode NOT IN ('PILOT', 'BASELINE_BACKFILL', 'MACRO_WRITE')
                        ORDER BY b.generation DESC, b.created_at DESC LIMIT 1
                        """
                    ).fetchone()
   ```
3. In `CampaignOrchestrator.run_campaign` (lines 372–393), update MAP-Elites UCB parent selection so stale niches lagging more than `0.10` depth behind `"descent"` are upgraded to `"descent"`, `q` includes `e.avg_depth + (e.total_reward / max(1, e.pull_count))`, and `last_batch_digest` is synchronized with `chosen_entry.batch_digest`:
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
4. In `PARAM_TUNE` promotion (`lines 485–508`):
   - Use `baseline_by_seed.update(full_promoted)` instead of `baseline_by_seed = full_promoted`.
   - Update `self.policy_archive[chosen_niche]` (and `"descent"` if `val_batch["avg_depth"] >= self.policy_archive["descent"].avg_depth`) with the tuned `current_policy`, `cand_id`, `baseline_runs`, and `last_batch_digest` so the next generation does not revert the tuned parameters!
5. After `verdict` is computed in the main evaluation block (around line 840):
   - Update `if self.policy_archive and chosen_niche in self.policy_archive: self.policy_archive[chosen_niche].total_reward += max(0.0, float(verdict.get("mean_delta", 0.0)))`.
   - In `if verdict["passed"]:` (lines 868–970), compute `last_batch_digest` BEFORE updating `self.policy_archive` so `batch_digest=last_batch_digest` is saved on each `PolicyArchiveEntry`, only call `self.save_active_policy(current_policy)` when `"descent" not in self.policy_archive or float(eval_res["avg_depth"]) >= self.policy_archive["descent"].avg_depth`, and require `c_depth >= self.policy_archive["survivability"].avg_depth - 0.05` when updating `"survivability"`.

---

## Task 3: Fix Conditional Episode Trigger Filter Bypass on `priority_seeds` (`lox/runner/pool.py`, `lox/engine/orchestrator.py`, `lox/author/tools.py`)

**Problem**:
In `lox/runner/pool.py` (lines 536–550), every seed in `priority_seeds` gets `is_pri = True` and `"enforce_trigger": not is_pri and bool(target_fns)` (`False`).
Because `orchestrator.py` and `tools.py` pass all 250 `baseline_runs` seeds in `priority_seeds`, **`enforce_trigger` is `False` on 100% of episodes in `FUNC_MOD_ADD`**! Newly added functions that trigger on only 15% of seeds (or 0% due to a bug) have `0` untriggered episodes discarded, never trigger `unreachable_abort`, and dilute the batch with 85%–100% untriggered ties (`0W / 0L / 130T`).
To eliminate survivorship bias *without* keeping untriggered ties, an episode on seed $s$ must be kept iff $T_{\text{base}}(s) = 1 \lor T_{\text{cand}}(s) = 1$: `enforce_trigger` must ONLY be `False` on seeds where the baseline *already* triggered `target_functions` (`unconditional_seeds`), while still using `priority_seeds` to order existing baseline seeds first in `seed_queue`.

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
   - At line 536, change `is_pri = s in priority_seeds_set` to `is_pri = s in unconditional_seeds_set`.
2. In `lox/engine/orchestrator.py` (lines 646–677 and 731–746):
   - Separate `base_triggered_seeds` (ONLY seeds where `is_trig = bool(target_set and (target_set & fn_keys))` is True) from `base_fail_seeds` and `base_remaining_seeds`.
   - Set `unconditional_seeds = list(base_triggered_seeds)` and `priority_seeds = base_triggered_seeds + base_fail_seeds + base_remaining_seeds`, and pass `unconditional_seeds=unconditional_seeds if proposal.mode == "FUNC_MOD_ADD" else None` to `self.pool.evaluate_batch(...)`.
3. In `lox/author/tools.py` (`dry_run_policy` pilot evaluation, lines 639–790):
   - Compute `base_trig_seeds` (seeds in `self.baseline_runs` whose `triggered_functions` / `functions_triggered` intersect `changed_fns`).
   - Append remaining `self.baseline_runs` seeds into `pilot_seeds` after `strat_runs`, pass `unconditional_seeds=base_trig_seeds` to `self.pool.evaluate_batch(...)`, and backfill any newly triggered pilot seeds not in `base_by_seed` by evaluating `self.current_policy_code` on those missing seeds so 100% of triggered pilot runs are paired 1-to-1 against baseline!

---

## Task 4: Fix `STRUCTURAL_REFACTOR` Splicing Crash, Infinite Refactor Lock, and `__init__` Attribute Preservation (`lox/author/agent.py`, `lox/engine/scheduler.py`, `lox/lang/validator.py`)

**Problem**:
1. In `lox/author/agent.py` (lines 735 and 857), `if mode in ("FULL_WRITE", "STRUCTURAL_REFACTOR")` sets `spliced_policy = code_patch` WITHOUT calling `splice_functions_into_policy(current_policy, code_patch)`. Yet `lox/author/prompts.py:429` tells the Author Agent in `STRUCTURAL_REFACTOR` to `"Output the refactored method(s) (or full class Policy:)"`, and `dry_run_policy` splices them into `current_policy`! So when the Author Agent outputs only the refactored methods in `STRUCTURAL_REFACTOR`, `dry_run_policy` passes, and then `agent.py` overwrites `policy.py` with bare `def` methods (deleting `class Policy:` and all other methods)!
2. In `lox/engine/scheduler.py` (line 61), `if consecutive_failures >= 3: return "STRUCTURAL_REFACTOR"` triggers on `3, 4, 5, 6, ...`, locking the campaign into `STRUCTURAL_REFACTOR` on **every single subsequent generation** if the refactor at `consecutive_failures == 3` is rejected!
3. In `lox/lang/validator.py` (`splice_functions_into_policy`), when a `FUNC_MOD_ADD` patch includes a short `def __init__(self): self.new_attr = ...`, replacing the existing `__init__` deletes `self.<attr>` initializations required by other elided methods on `class Policy`, causing `AttributeError` at runtime.

**Exact Changes**:
1. In `lox/author/agent.py` (lines 735 and 857):
   - Change `if mode in ("FULL_WRITE", "STRUCTURAL_REFACTOR") or not current_policy.strip():` to:
     ```python
     if mode == "FULL_WRITE" or not current_policy.strip():
         spliced_policy = code_patch
         changed_fns = hyp.get("target_functions") or ["run"]
     else:
         spliced_policy, changed_fns = splice_functions_into_policy(
             current_policy, code_patch
         )
     ```
2. In `lox/engine/scheduler.py` (line 61):
   - Change `if consecutive_failures >= 3:` to:
     ```python
     if consecutive_failures > 0 and consecutive_failures % 3 == 0:
         self.counts["STRUCTURAL_REFACTOR"] += 1
         return "STRUCTURAL_REFACTOR"
     ```
3. In `lox/lang/validator.py` (`splice_functions_into_policy`, around line 281):
   - If `"__init__"` is in both `existing_methods` and `patch_methods`:
     Find all `self.<attr>` target names assigned via `ast.Assign` / `ast.AnnAssign` inside `patch_methods["__init__"]`. For any `ast.Assign` / `ast.AnnAssign` statement in `existing_methods["__init__"].body` that assigns to `self.<old_attr>` where `<old_attr>` is NOT assigned in `patch_methods["__init__"]`, prepend a copy of that statement to `patch_methods["__init__"].body` (after any docstring) so previously initialized attributes are never lost!

---

## Task 5: Overnight OpenRouter API Resilience, `:floor` Failover, Reasoning Token Stripping, `query_policy_docs("policy")`, & DuckDB Pilot Bloat Prevention (`lox/author/agent.py`, `lox/author/tools.py`, `lox/db/logger.py`)

**Problem**:
1. In `_call_llm_api` (`lox/author/agent.py:271-301`), OpenRouter frequently returns HTTP 200 JSON containing `{"error": ...}` or `{"choices": []}` when an upstream `:floor` provider blips. `data["choices"][0]["message"]` raises `KeyError` or `IndexError`, which is NOT caught by `except (httpx.RequestError, httpx.HTTPStatusError):`, immediately crashing the generation without retrying!
2. In `AuthorAgent.synthesize` (`lox/author/agent.py:505`), `messages.append(msg)` appends raw `reasoning` / `reasoning_details` fields and `<think>...</think>` blocks into the conversation history sent on every subsequent turn, inflating multi-turn API token spend by 2x–3x.
3. In `AuthorAgent.synthesize` (`lox/author/agent.py:703`), if an earlier pilot passed (`self.tools.last_passed_code` is non-empty) and a later turn emitted code that failed its pilot (`not self.tools.last_pilot_improved`), `code_patch` uses the failing last-turn code instead of `self.tools.last_passed_code`.
4. `_format_context_scoped_policy` (`lox/author/prompts.py:321`) instructs the agent to call `query_policy_docs('policy')` when `policy.py` exceeds 350 lines, but `AuthorToolRegistry.query_policy_docs` (`lox/author/tools.py:196`) does not handle `topic='policy'`.
5. Saving `mode='PILOT'` batches in `lox/author/tools.py:931` writes full `ticks` rows for every pilot batch into DuckDB (~3.1M tick rows over 50 generations), and `BatchDatabaseLogger` calls `init_db` (re-executing 12 DDL statements) on every `save_batch`/`save_policy`/`save_experiment` call.

**Exact Changes**:
1. In `lox/author/agent.py` (`_call_llm_api`, lines 271–301):
   - Increase retries from `range(3)` to `range(5)` with backoff `time.sleep(min(16.0, 2.0 * (2 ** attempt)))`.
   - On `attempt >= 2`, if `payload.get("model", "").endswith(":floor")`, strip `":floor"` (`payload["model"] = payload["model"][:-6]`) so OpenRouter fails over to healthy providers if the cheapest `:floor` provider is down.
   - Inside the `try:` block after `data = resp.json()`:
     ```python
     if isinstance(data, dict) and "error" in data:
         err_obj = data["error"]
         err_code = err_obj.get("code") if isinstance(err_obj, dict) else None
         err_msg = (
             err_obj.get("message", str(err_obj))
             if isinstance(err_obj, dict)
             else str(err_obj)
         )
         if err_code in (401, 402, 403):
             raise PermissionError(f"LLM API Error {err_code}: {err_msg}")
         raise ValueError(f"Upstream provider error in JSON: {err_msg}")
     choices = data.get("choices") if isinstance(data, dict) else None
     if not choices:
         raise ValueError(f"Empty choices in LLM response: {str(data)[:200]}")
     msg_obj = choices[0].get("message") or {}
     if not msg_obj.get("content") and not msg_obj.get("tool_calls"):
         raise ValueError("LLM returned empty message content and empty tool_calls")
     return msg_obj
     ```
   - Catch `(httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, IndexError)` in the retry loop.
2. In `lox/author/agent.py` (`synthesize`, around line 505):
   - Extract `msg_text = msg.get("content") or ""` first, then append a sanitized assistant message to `messages` that strips `<think>...</think>` and omits `reasoning` / `reasoning_details`:
     ```python
     clean_text = re.sub(r"<think>.*?</think>", "", msg_text, flags=re.DOTALL).strip()
     clean_msg: dict[str, Any] = {"role": "assistant", "content": clean_text}
     if msg.get("tool_calls"):
         clean_msg["tool_calls"] = msg["tool_calls"]
     messages.append(clean_msg)
     ```
   - At line 703 (in the post-loop compiler gate):
     ```python
     if not code_patch or (
         not self.tools.last_pilot_improved and self.tools.last_passed_code
     ):
         code_patch = self.tools.last_passed_code or self.tools.best_pilot_code or code_patch
     ```
3. In `lox/author/tools.py`:
   - In `query_policy_docs` (around line 205), add:
     ```python
     if "policy" in t:
         pol_src = self.current_policy_code or "(Empty policy)"
         return (
             f"### Current `policy.py` Full Source:\n```python\n{pol_src}\n```",
             "policy source",
         )
     ```
   - At line 938 (inside `db_logger.save_batch` for `mode="PILOT"`), pass `runs=[{k: v for k, v in r.items() if k != "ticks"} for r in p_runs]` so pilot runs are logged without bloating the `ticks` table with millions of pilot ticks.
4. In `lox/db/logger.py`:
   - Import `connect_db` from `lox.db.schema` and use `con = connect_db(self.db_path, read_only=False)` in `save_batch`, `save_policy`, and `save_experiment` (keeping `init_db(self.db_path)` in `BatchDatabaseLogger.__init__`).

---

## Task 6: Fix `--max-steps` Default (`scripts/run_lox.py`, `lox/runner/pool.py`), `reset_clean.py` Paths, Prompt Deduplication (`lox/author/prompts.py`), Shard Turn-Ratio & Starvation Filter (`lox/db/shards.py`), SPSA Full-Episode Alignment (`lox/engine/modes.py`), and YAML Hypothesis Keys (`lox/ui/console.py`, `lox/db/logger.py`)

**Exact Changes**:
1. In `scripts/run_lox.py` (line 80) and `lox/runner/pool.py` (line 31):
   - Change `--max-steps` default from `2500` to `5000` (`default=5000`), and `_WORKER_MAX_STEPS: int = 5000` in `lox/runner/pool.py`.
2. In `scripts/reset_clean.py` and `scripts/run_lox.py` (line 164):
   - Update `reset_clean(db_path: str = DB_PATH, policy_path: str = POLICY_PATH, macros_path: str = MACROS_PATH)` to accept custom paths and clean any `.py` files in `Path(policy_path).parent / "skills"`, and pass `db_path=args.db_path, policy_path=args.policy_path, macros_path=args.macros_path` from `scripts/run_lox.py`.
3. In `lox/author/prompts.py`:
   - In Section 3 (`Visible Objects & Corpses`, lines 114–118), document `state.objects_here() -> list[VisibleObject]` and note that `state.visible_objects()` includes objects on the hero's tile `(state.y, state.x)`.
   - Remove the duplicate `WorldMemory & FloorMemory` block from Section 3 (lines 119–129), keeping Section 4 (`4. Deterministic Zero-Heuristic WorldMemory (state.mem)`).
   - In Section 4 (`Generic Vectorized Pathfinding Usage`, lines 167–177), replace `passable = np.isin(state.chars, [ord('.'), ord('#'), ord('>')])` with `passable = state.mem.passable_mask.copy()`.
4. In `lox/db/shards.py` (`extract_shards_from_runs`):
   - Define `turns = int(r.get("turns") or steps)` and use `ratio = cut_step / float(max(1, turns))` at lines 136 and 186 (where `cut_step` comes from `stall_start_turn` and `d_turn_val`).
   - In `1. Pre-Death Windows` (line 105), skip runs where `"starv" in end_reason.lower()` unless `len(runs) <= 5` (since starving to death with 0 food cannot be reversed 30 steps before death; nutrition skills must be evaluated from Turn 1 where corpses are encountered).
   - If `target_functions` is provided, sort `runs` so runs whose `triggered_functions` / `functions_triggered` intersect `target_functions` are prioritized first.
5. In `lox/engine/modes.py` (`run_spsa_param_tuning`, lines 184–300):
   - Do not evaluate SPSA perturbations exclusively on 150-step shards (`shards`); evaluate on full-episode `mb_seeds` (`shards = []`) so SPSA optimizes the exact same full-episode objective `progression_metric(r)` that `val_result` validates on at line 365.
6. In `lox/ui/console.py` (`hypothesis_box`, lines 122–126) and `lox/db/logger.py` (`save_experiment`, lines 341–345):
   - Support `binding_constraint` and `verified_c_rule` in addition to `observation_from_db` and `causal_finding` when extracting the observation/finding string.
7. Add a unit test `test_pass6_overnight_audit_fixes` in `tests/test_overhaul_verification.py` verifying:
   - `ExecutionTracer` attributes turns to `handle_combat` (not leaf helper `get_passable_mask`) and counts boolean macro calls returning `False`.
   - `PersistentEvalPool.evaluate_batch` with `priority_seeds` and `unconditional_seeds=[]` discards untriggered episodes until collecting `target_episodes` triggered runs.
   - `splice_functions_into_policy` preserves existing `self.<attr>` initializations in `__init__` when a patch defines a partial `__init__`.
   - `ModeScheduler` does not lock into `STRUCTURAL_REFACTOR` on `consecutive_failures == 4`.
   - `AuthorToolRegistry.query_policy_docs("policy")` returns `current_policy_code`.
   - `ConsoleUI.hypothesis_box` and `BatchDatabaseLogger.save_experiment` record `binding_constraint` and `verified_c_rule`.

---

# Pass 7: Final Pre-Flight Deep Sweep (5 Critical Runtime & Evolution Fixes)

All items below MUST be implemented and verified via `uv run ruff check .` and `uv run pytest -v`:

## Task 1: Fix Under-Hero Staircase & Feature Occlusion When `@` Covers `>` (`lox/env/state.py`)

**Problem**:
When the hero stands on a staircase down `'>'` (or staircase up `'<'`, altar `'_'`, fountain `'{'`), NLE's `self.chars[self.y, self.x]` is `ord('@')` (`64`), and `self.glyphs[self.y, self.x]` is the hero's monster glyph. As a result, at the exact moment the hero steps onto the stairs:
- `state.stairs_down_pos` (`np.argwhere(self.chars == ord(">"))`) returns `None`!
- `state.is_stairs_down(state.y, state.x)` returns `False`!
- `state.char_at(state.y, state.x)` returns `'@'` instead of `'>'` (causing policies that check `state.char_at(state.y, state.x) == '>'` or `state.stairs_down_pos == state.pos` to step onto the stairs and then immediately step off them in an infinite ping-pong loop!).
- Similarly, if a monster or corpse/object covers `'>'` on `self.chars`, `state.stairs_down_pos` returns `None` even when `state.mem.remembered_chars` already recorded `ord('>')` at that coordinate.

**Exact Changes in `lox/env/state.py`**:
1. In `WorldMemory.update(self, state: GameState, action_name: str = "")` (around lines 630–670):
   - After setting `floor.remembered_chars[uninit_occupied] = ord(".")`, check if the hero is standing on a terrain feature via `state.is_on_stairs_down` or `state.message`:
     ```python
     msg_low = str(state.message).lower() if hasattr(state, "message") else ""
     if (
         0 <= sy < floor.remembered_chars.shape[0]
         and 0 <= sx < floor.remembered_chars.shape[1]
     ):
         if getattr(state, "is_on_stairs_down", False) or "staircase down" in msg_low:
             floor.remembered_chars[sy, sx] = ord(">")
         elif "staircase up" in msg_low:
             floor.remembered_chars[sy, sx] = ord("<")
         elif "there is an altar" in msg_low or "altar of" in msg_low:
             floor.remembered_chars[sy, sx] = ord("_")
         elif "there is a fountain" in msg_low:
             floor.remembered_chars[sy, sx] = ord("{")
     ```
2. In `GameState.char_at(self, y: int, x: int) -> str` (lines 1088–1092):
   - When `(y, x) == (self.y, self.x)` and `int(self.chars[y, x]) == ord("@")`:
     - If `self.is_on_stairs_down`: return `">"`
     - If `self.mem is not None` and `0 <= y < self.mem.remembered_chars.shape[0]` and `0 <= x < self.mem.remembered_chars.shape[1]`:
       `rem_c = int(self.mem.remembered_chars[y, x])`
       if `rem_c != 0 and rem_c != ord(" ")`: return `_CHAR_LUT[rem_c] if rem_c < 256 else "?"`
3. In `GameState.terrain_at(self, y: int, x: int) -> str` (lines 1155–1172):
   - If `y == self.y and x == self.x and self.is_on_stairs_down`: return `"staircase down"`.
   - If `cmap is None` and `self.mem is not None` and `0 <= y < self.mem.remembered_chars.shape[0]` and `0 <= x < self.mem.remembered_chars.shape[1]`:
     check `rem_c = int(self.mem.remembered_chars[y, x])`:
     - `ord(">")` -> `"staircase down"`
     - `ord("<")` -> `"staircase up"`
     - `ord("_")` -> `"altar"`
     - `ord("{")` -> `"fountain"`
     - `ord("+")` -> `"closed door"`
     - `ord("#")` -> `"corridor"`
     - `ord(".")` -> `"floor of a room"`
4. In `GameState.stairs_down_pos` and `GameState.stairs_up_pos` (lines 1221–1236):
   - In `stairs_down_pos`:
     1. Check `stairs = np.argwhere(self.chars == ord(">"))`; if `len(stairs) > 0`: return `(int(stairs[0, 0]), int(stairs[0, 1]))`.
     2. If `self.is_on_stairs_down`: return `(self.y, self.x)`.
     3. If `self.mem is not None`: check `rem = np.argwhere(self.mem.remembered_chars == ord(">"))`; if `len(rem) > 0`: return `(int(rem[0, 0]), int(rem[0, 1]))`.
     4. Return `None`.
   - In `stairs_up_pos`:
     1. Check `stairs = np.argwhere(self.chars == ord("<"))`; if `len(stairs) > 0`: return `(int(stairs[0, 0]), int(stairs[0, 1]))`.
     2. If `self.mem is not None`: check `rem = np.argwhere(self.mem.remembered_chars == ord("<"))`; if `len(rem) > 0`: return `(int(rem[0, 0]), int(rem[0, 1]))`.
     3. Return `None`.
5. In `GameState.is_stairs_down`, `is_stairs_up`, `is_altar`, `is_fountain` (lines 1201–1212):
   - Because `char_at` and `terrain_at` now consult `is_on_stairs_down` and `self.mem.remembered_chars`, also allow `self.mem is not None and 0 <= y < self.mem.remembered_chars.shape[0] and 0 <= x < self.mem.remembered_chars.shape[1] and self.mem.remembered_chars[y, x] == ord(...)` so they remain `True` even when a monster or object stands on that tile.

---

## Task 2: Fix Promoted Policy Reversion When Conditional `eval_res["avg_depth"]` Differs From Unconditional `baseline_runs` (`lox/engine/orchestrator.py`)

**Problem**:
In `FUNC_MOD_ADD`, `eval_res` evaluates on a conditional, failure-prioritized subset of seeds (`base_triggered_seeds + base_fail_seeds` + 20 failure shards). Even when `verdict["passed"] == True` (statistically significant positive paired delta on those seeds), `eval_res["avg_depth"]` on the failure-heavy subset is often numerically lower than Gen 0's unconditional `self.policy_archive["descent"].avg_depth` (or artificially higher if the target function only triggers on Dlvl 2+).
When `eval_res["avg_depth"] < self.policy_archive["descent"].avg_depth`:
1. Line 973 skipped `self.save_active_policy(current_policy)`.
2. Line 1018 skipped updating `self.policy_archive["descent"]`.
3. At the top of the next generation (line 407), `current_policy = chosen_entry.policy_code` overwrote `current_policy` back to the old archive entry, silently discarding the promoted generation!
4. At line 380, `if n_entry.avg_depth < champ_entry.avg_depth - 0.10:` allowed UCB to select a stale niche up to `0.10` depth worse than `"descent"`, reverting intermediate promoted methods.

**Exact Changes in `lox/engine/orchestrator.py`**:
1. At line 380 (in the MAP-Elites UCB parent selection block):
   - Change `if n_entry.avg_depth < champ_entry.avg_depth - 0.10:` to `if n_entry.avg_depth < champ_entry.avg_depth or n_entry.policy_id != champ_entry.policy_id and n_entry.avg_depth <= champ_entry.avg_depth:` so no archive niche ever reverts `current_policy` to an older policy with lower or equal `avg_depth` than the latest `"descent"` champion.
2. In `PARAM_TUNE` promotion (lines 507–585):
   - Whenever `verdict["passed"]` is `True`, unconditionally call `self.save_active_policy(current_policy)`.
   - Compute `u_depth`, `u_p90`, `u_max`, `u_ac`, `u_xl` from `baseline_runs` (after `baseline_by_seed.update(full_promoted)`) rather than `val_batch["runs"]`, and unconditionally update `self.policy_archive["descent"]` and `self.policy_archive[chosen_niche]` with `avg_depth = max(u_depth, self.policy_archive["descent"].avg_depth if "descent" in self.policy_archive else 0.0)`.
3. In `FULL_WRITE` / `FUNC_MOD_ADD` / `STRUCTURAL_REFACTOR` promotion (lines 967–1069):
   - Whenever `verdict["passed"]` is `True`, unconditionally call `self.save_active_policy(current_policy)`.
   - After `baseline_by_seed.update(full_promoted)` and `baseline_runs = list(baseline_by_seed.values())`, compute `clean_archive_runs = list(baseline_runs)` and compute `c_depth`, `c_p90`, `c_max`, `c_ac`, `c_xl` from `clean_archive_runs` (falling back to `eval_res["runs"]` only if `clean_archive_runs` is empty):
     ```python
     stat_pool = clean_archive_runs if clean_archive_runs else eval_res["runs"]
     c_depth = (
         float(np.mean([r.get("max_depth", 1) for r in stat_pool]))
         if stat_pool
         else float(eval_res["avg_depth"])
     )
     if "descent" in self.policy_archive:
         c_depth = max(c_depth, self.policy_archive["descent"].avg_depth)
     c_p90 = (
         float(np.percentile([r.get("max_depth", 1) for r in stat_pool], 90))
         if stat_pool
         else float(eval_res["p90_depth"])
     )
     c_max = (
         int(max([r.get("max_depth", 1) for r in stat_pool]))
         if stat_pool
         else int(eval_res["max_depth"])
     )
     c_ac = float(np.mean([r.get("final_ac", 10) for r in stat_pool])) if stat_pool else 10.0
     c_xl = float(np.mean([r.get("xp_level", 1) for r in stat_pool])) if stat_pool else 1.0
     ```
   - Unconditionally update `self.policy_archive["descent"]` (and `self.policy_archive[chosen_niche]`) with the newly promoted `current_policy` so a statistically verified promotion is never lost on the next generation.
4. On campaign resume (around line 297) and warm-start (around line 370), if `baseline_runs` is non-empty and `"descent" not in self.policy_archive`, initialize `self.policy_archive["descent"]` with `current_policy`, `current_policy_id`, `baseline_runs`, and `last_batch_digest`.

---

## Task 3: Fix Prefix Shards Truncating After 1 Step in `run_single_episode` & Avoid Worker Harness Re-creation (`lox/runner/pool.py`, `lox/engine/hypothesis.py`)

**Problem**:
1. In `lox/runner/pool.py` (`run_single_episode`, lines 112–116):
   - When `is_shard` is `True`, `harness.replay_bytes(seed=seed, recorded_bytes=replay_bytes)` sets `harness.step_count = step_ctr` (e.g. `300..4000` steps from the prefix), while the shard task's `max_steps` is `200` (`post_replay_max_steps`).
   - Because `run_single_episode` did not reset `harness.step_count = 0` after `replay_bytes`, on the very first step (`step_idx = 1`) of `harness.step(raw_act)`, `self.step_count >= self.max_steps` (`301 >= 200`) was immediately `True`, truncating every shard with `step_ctr >= 200` after 1 single step!
2. In `_get_worker_harness` (`lox/runner/pool.py`, lines 48–66):
   - Checking `or _WORKER_MAX_STEPS != max_steps` closed and recreated the C++ `nh.Nethack` instance every time a worker switched between a 200-step shard and a 5000-step full episode.
3. In `lox/engine/hypothesis.py` (`progression_metric`, lines 18–53):
   - On a 200-step pre-death shard (`run.get("is_shard") == True`), if the candidate policy survived the fatal encounter (`final_hp > 0` and `end_reason in ("max_steps", "ascended")`) while the baseline died (`final_hp <= 0`), `progression_metric` gave `0.0` difference when `max_depth`, `xp_level`, `ac`, and `score` were unchanged.

**Exact Changes**:
1. In `lox/runner/pool.py`:
   - In `_get_worker_harness` (lines 48–66), only recreate `_WORKER_HARNESS` when `_WORKER_HARNESS is None or _WORKER_ROLE != role or _WORKER_BENCHMARK != benchmark`; otherwise simply set `_WORKER_MAX_STEPS = max_steps` and `_WORKER_HARNESS.max_steps = max_steps`.
   - In `run_single_episode` (lines 112–116), right after `state = harness.replay_bytes(seed=seed, recorded_bytes=replay_bytes)`:
     ```python
     runner.tracer.reset_for_episode()
     max_steps = int(post_replay_max_steps or 200)
     harness.step_count = 0
     harness.max_steps = max_steps
     harness.done = False
     ```
2. In `lox/engine/hypothesis.py` (`progression_metric`, lines 18–53):
   - Add a shard survival bonus when `run.get("is_shard")` is `True`:
     ```python
     shard_survival_bonus = (
         0.08
         if (
             run.get("is_shard")
             and int(run.get("final_hp", 0)) > 0
             and str(run.get("end_reason", "")) in ("max_steps", "ascended")
         )
         else 0.0
     )
     ```
     and include `+ shard_survival_bonus` in the returned sum.

---

## Task 4: Fix Void Procedures With Early `return` & Data-Helper Return Attribution in `ExecutionTracer` (`lox/lang/instrument.py`)

**Problem**:
1. In `PolicyInstrumentTransformer.visit_FunctionDef` (`lox/lang/instrument.py`, lines 279–281):
   - `has_val_ret = (not is_gen) and any(isinstance(sub, ast.Return) for sub in ast.walk(node))` treated void procedures that contain a bare `return` or `return None` (e.g. `def update_memory(self, state): if state.turn == 0: return; self.visited.add(...)`) as value-returning functions (`has_val_ret = True`).
   - Consequently, `enter_fn` did not increment `fn_calls["update_memory"]`, and `on_return` saw `val is None` (`is_noop = True`) and also did not increment `fn_calls["update_memory"]`, resulting in `0` recorded calls and `Unreachable Function Abort` whenever a void method with an early `return` was modified!
2. In `ExecutionTracer.on_return` and `ExecutionTracer.exit_fn` (`lox/lang/instrument.py`, lines 83–128):
   - If a helper returned a numpy mask (`np.ndarray`), `set`, or `dict` directly to `run()`, `on_return` set `self.active_fn = fn_name`.
   - And if `run()` called `act = self.explore(state)` (setting `self.active_fn = "explore"`) and then called another helper before `yield act` that returned `None`, `exit_fn` ran `elif not yielded_inside and not returned_non_noop: self.active_fn = caller_fn` and overwrote `self.active_fn` back to `"run"`.

**Exact Changes in `lox/lang/instrument.py`**:
1. In `PolicyInstrumentTransformer.visit_FunctionDef` (lines 279–288):
   - Compute `has_val_ret` by checking only non-`None` return expressions:
     ```python
     has_val_ret = (not is_gen) and any(
         isinstance(sub, ast.Return)
         and sub.value is not None
         and not (isinstance(sub.value, ast.Constant) and sub.value.value is None)
         for sub in ast.walk(node)
     )
     ```
   - Pass `is_gen` as a third argument to `_make_enter_stmt(fn_name, has_val_ret, is_gen)`.
2. In `ExecutionTracer`:
   - `enter_fn(self, fn_name: str, has_value_return: bool = False, is_gen: bool = False) -> None`:
     ```python
     self.call_stack.append([fn_name, self.total_turns, False])
     if not has_value_return:
         self.fn_calls[fn_name] = self.fn_calls.get(fn_name, 0) + 1
         if is_gen or fn_name in self._ROOT_FNS:
             self.active_fn = fn_name
     ```
   - `on_return(self, fn_name: str, val: Any) -> Any`:
     ```python
     is_noop = val is None or (isinstance(val, tuple) and len(val) == 2 and val == (0, 0))
     if not is_noop:
         self.fn_calls[fn_name] = self.fn_calls.get(fn_name, 0) + 1
         if self.call_stack:
             self.call_stack[-1][2] = True
         if not isinstance(val, (bool, set, dict)) and type(val).__name__ != "ndarray":
             self.active_fn = fn_name
     return val
     ```
   - `exit_fn(self, fn_name: str) -> None`:
     After popping `fn_name` from `self.call_stack`:
     ```python
     if not self.call_stack:
         return
     caller_fn = self.call_stack[-1][0]
     if caller_fn not in self._ROOT_FNS:
         self.active_fn = caller_fn
     ```
     (Do NOT overwrite `self.active_fn = "run"` when `caller_fn in self._ROOT_FNS`, because `record_turn_yield()` already resets `self.active_fn = "run"` after each yielded turn!).

---

## Task 5: Fix 0-Trigger Pilot Diagnostic (`lox/author/tools.py`) & `MacroHarvester` Attribute Rewriting (`lox/lang/macros.py`)

**Problem**:
1. In `lox/author/tools.py` (lines 925–950):
   - On Gen 1+ (`self.baseline_runs` is non-empty), if a pilot batch records `0` triggered episodes (`len(p_runs) == 0`), `base_depths` is empty (`[]`), so `if base_depths:` is `False` and execution falls into the Gen 0 `else:` branch—falsely telling the LLM `Initial Gen 0 Baseline Validation: 0 episodes ... INITIAL POLICY CRASHED. Please fix the runtime error.` instead of reporting that `target_fns` triggered 0 times!
2. In `lox/lang/macros.py` (`MacroHarvester.harvest_from_policy`, lines 266–295):
   - `SelfRewriter` rewrote `self.<method>` to `<method>` if `<method>` was any method in `target_cls.body`, even if `<method>` was NOT in `macros.py` and NOT harvested (producing broken macros that raise `NameError` when called).
   - It also omitted `"passable_mask"`, `"door_mask"`, `"searched_counts"`, and `"stair_graph"` from the `state.mem` attribute rewrite list.

**Exact Changes**:
1. In `lox/author/tools.py` (lines 925–950):
   - Add an explicit branch when `self.baseline_runs and not p_runs`:
     ```python
     if self.baseline_runs and not p_runs:
         improved = False
         raw_run_cnt = pilot_res.get("raw_episodes_run", 0)
         summary_short = f"0/{raw_run_cnt} eps triggered {target_fns}"
         paired_line = f"• Trigger Check Failed: Target function(s) {target_fns} triggered in 0/{raw_run_cnt} episodes!\n"
         verdict_line = (
             f">>> PILOT VERDICT: TARGET FUNCTION NEVER TRIGGERED AT RUNTIME ({target_fns}). "
             "Loosen its entry condition in `run()` so it triggers on current dungeon levels!"
         )
     elif base_depths:
         ...
     ```
2. In `lox/lang/macros.py` (`MacroHarvester.harvest_from_policy`, lines 237–300):
   - Track `existing_macro_names = {s.name for s in macro_lib.list_specs()}`.
   - In `SelfRewriter.visit_Attribute`:
     - Expand the `state.mem` attribute tuple to `("visited", "visit_count", "pos_history", "locked_doors", "remembered_chars", "is_oscillating", "passable_mask", "door_mask", "searched_counts", "stair_graph")`.
     - Only rewrite `self.<attr>` to `ast.Name(id=node.attr, ctx=node.ctx)` if `node.attr in existing_macro_names or node.attr in harvested_names` (otherwise set `self.invalid = True`).
3. Add a unit test `test_pass7_final_preflight_audit_fixes` in `tests/test_overhaul_verification.py` verifying:
   - Standing on a downstairs tile `'>'` (where `state.chars[state.y, state.x] == ord('@')`) still returns `'>'` for `state.char_at(state.y, state.x)`, `(state.y, state.x)` for `state.stairs_down_pos`, `True` for `state.is_stairs_down(state.y, state.x)`, and `"staircase down"` for `state.terrain_at(state.y, state.x)`.
   - A void method `def update_memory(self, state):` containing an early bare `return` increments `tracer.fn_calls["update_memory"]` and does not clobber `active_fn`.
   - A prefix shard in `run_single_episode` with a 300-byte replay prefix and `post_replay_max_steps=20` executes up to 20 post-replay steps instead of truncating at step 1.
   - `MacroHarvester.harvest_from_policy` does not harvest a method that calls an unharvested stateful `self.<method>`, while harvesting a method that accesses `self.passable_mask`.

