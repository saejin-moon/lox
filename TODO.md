# LOX Post-Run Bugfix Specification (`TODO.md` — Pass 3)

In a live 5-generation NetHack campaign (`google/gemma-4-31b-it`), Gen 000 (`FULL_WRITE`, depth `1.33`) and Gen 001 (`FUNC_MOD_ADD`, depth `1.82`) promoted cleanly, but Gen 002–004 stalled out because **`dry_run_policy` and `pool.evaluate_batch` rejected winning candidates** (in Gen 002 `pilot_5`, the candidate achieved `2.55` full-run depth vs `2.25` baseline, `8W/5L/12T`, `mean_delta = +0.1035`, yet `dry_run_policy` reported `PILOT_FAILED (8W/5L/12T, Δ=-0.20)`).

Execute **all 4 items** below completely and verify with `uv run pytest -v` and `uv run ruff check .`.

---

## 1. Fix Apples-to-Oranges `delta_depth` & Metric Mismatch in `dry_run_policy` (`lox/author/tools.py`)

### Root Cause
In `lox/author/tools.py` (lines 666–776):
1. Line 666 computes `cand_depths = [float(r["max_depth"]) for r in p_runs]` over **all 30 runs** (20 full episodes + 10 200-step `Dlvl 1` shards), dragging `c_avg_depth` down from `2.55` to `2.20`.
2. Meanwhile, line 693 (`base_depths.append(b_d)`) only appends `b_d` inside the `else:` (`if not r.get("is_shard"):`) branch, excluding the 10 shards and computing `b_avg_depth = 2.40` on full episodes only.
3. Line 717 then subtracts `delta_depth = c_avg_depth - b_avg_depth` (`2.20 - 2.40 = -0.20`), and lines 772–776 require `delta_depth >= 0.0` to set `improved = True`. Thus, adding 200-step `Dlvl 1` shards imposes an artificial `-0.20` to `-0.40` penalty on `c_avg_depth` that causes winning candidates (`8W/5L/12T`, `mean_delta = +0.1035`) to fail the pilot!
4. Additionally, lines 609–616 build `pilot_seeds` from `fail_runs[:num_fail] + normal_runs[:num_norm]` without excluding `shard_seeds = {int(sh.seed) for sh in pilot_shards}`, so `pool.evaluate_batch` deduplicates those overlapping seeds out of `seed_queue`, leaving fewer priority seeds than `target_eps - len(pilot_shards)`.

### Required Changes in `lox/author/tools.py`
1. **Exclude `shard_seeds` when building `pilot_seeds` (lines 609–616)**:
   - Compute `shard_seeds = {int(sh.seed) for sh in pilot_shards}`.
   - Build `avail_fail = [r for r in fail_runs if int(r["seed"]) not in shard_seeds]` and `avail_norm = [r for r in normal_runs if int(r["seed"]) not in shard_seeds]`.
   - Select `needed_full = max(1, target_eps - len(pilot_shards))` full seeds from `avail_fail` and `avail_norm` (falling back to any remaining `self.baseline_runs` not in `shard_seeds` if needed) so `pilot_seeds` has `needed_full` non-overlapping seeds that all exist in `base_by_seed`.
2. **Pair `cand_depths` and `base_depths` Strictly 1-to-1 & Use `progression_metric` (lines 666–718)**:
   - Import `progression_metric` from `lox.engine.hypothesis` (and use `get_adapter(self.benchmark).progression_metric` when `self.benchmark != "nethack"`).
   - Initialize `cand_depths: list[float] = []`, `base_depths: list[float] = []`, and `deltas: list[float] = []`.
   - Inside `for r in p_runs:`:
     - When `r.get("is_shard")` and `sh_key in base_shard_by_seed`:
       - Compute `d_val = float(metric_fn(r) - metric_fn(br))` and append to `deltas`.
       - Append `c_d` to `cand_depths` AND `b_d` to `base_depths` (or track `cand_full_depths` / `base_full_depths` strictly on paired full runs, and if both are empty, on paired shard runs). **Crucial rule**: `cand_depths` and `base_depths` must be appended **together in lockstep** on the exact same paired runs so `len(cand_depths) == len(base_depths)` is always true!
       - Count a win if `d_val > 1e-6`, a loss if `d_val < -1e-6`, else a tie.
     - When `not r.get("is_shard")` and `s in base_by_seed`:
       - Compute `d_val = float(metric_fn(r) - metric_fn(br))` and append to `deltas`.
       - Append `c_d` to `cand_depths` AND `b_d` to `base_depths` in lockstep.
       - Count a win if `d_val > 1e-6`, a loss if `d_val < -1e-6`, else a tie.
   - Only when `not cand_depths` (e.g. Gen 0 with no baseline) should `c_avg_depth` fall back to `float(np.mean([float(r["max_depth"]) for r in p_runs]))`.
   - Compute `c_max_depth = int(max((r.get("max_depth", 1) for r in p_runs), default=0))`.
3. **Align Pilot Pass Criterion with `evaluate_paired_hypothesis` (lines 771–776)**:
   - Set:
     ```python
     improved = (wins > losses and mean_delta > 0.0) or (
         delta_depth >= 0.05 and mean_delta >= 0.0
     )
     ```
   - Never reject a candidate that achieves both `wins > losses` and `mean_delta > 0.0`!

---

## 2. Fix Over-Submission & Fast-Death Race Condition in `PersistentEvalPool.evaluate_batch` (`lox/runner/pool.py`)

### Root Cause
In `lox/runner/pool.py` (lines 480–555):
- `batch_capacity = min(max(needed * 2 if target_fns else needed, self.workers * 8), max_raw_episodes - raw_done)` submits `max(60, 128) = 128` tasks for a 30-episode pilot (or `300` tasks for a 150-episode batch) into `pool.imap_unordered` and `break`s as soon as the first `30` finish (`if len(triggered_runs) >= target_episodes: break`).
- Because every task in `shard_queue` and `seed_queue` (`priority_seeds`) has `enforce_trigger = False` (100% guaranteed to record), queuing `128` tasks adds `96` fresh unpaired seeds, races 2ms `Dlvl 1` deaths against 300ms `Dlvl 6` runs in `imap_unordered` (dropping deep runs that finish later), and leaves `98` orphaned tasks running on `self._pool`.
- Also, in `lox/runner/pool.py` (lines 628–630), `avg_depth`, `p90_depth`, `avg_score`, and `avg_turns` are averaged across both full episodes and 200-step shards, artificially depressing batch summary metrics whenever shards are present.

### Required Changes in `lox/runner/pool.py`
1. **Do Not Over-Submit When Guaranteed Tasks (`shard_queue` + `seed_queue`) Are Available (lines 480–490)**:
   - Compute `guaranteed_ready = len(shard_queue) + len(seed_queue)`.
   - If `guaranteed_ready > 0` or `not target_fns`:
     - Set `batch_capacity = min(needed, guaranteed_ready if guaranteed_ready > 0 else needed, max_raw_episodes - raw_done)`.
     - That way, when `shard_queue` + `seed_queue` has `>= needed` episodes (which is true for all pilots and all `FUNC_MOD_ADD` batches with `priority_seeds`), `evaluate_batch` dispatches **at most `needed` tasks**, waits for all `needed` tasks to finish without racing them against fresh seeds, pairs 100% of them against the baseline, and leaves zero orphaned tasks on `self._pool`.
   - Only when `guaranteed_ready == 0` and `target_fns` is non-empty (searching fresh seeds with `enforce_trigger = True`) set `batch_capacity = min(max(needed * 2, self.workers * 2), max_raw_episodes - raw_done)`.
2. **Compute Batch Summary Averages (`avg_depth`, `p90_depth`, `avg_score`, `avg_turns`) Over Full Runs When Present (lines 628–643)**:
   - Let `full_runs = [r for r in triggered_runs if not r.get("is_shard")]`.
   - Let `stat_runs = full_runs if full_runs else triggered_runs`.
   - Compute `depths = [r["max_depth"] for r in stat_runs]`, `scores = [r["score"] for r in stat_runs]`, and `turns = [r["turns"] for r in stat_runs]` (while keeping `max_depth = int(np.max([r["max_depth"] for r in triggered_runs])) if triggered_runs else 0`).

---

## 3. Eliminate Frankenstein Baselines on Promotion (`lox/engine/orchestrator.py`)

### Root Cause
In `lox/engine/orchestrator.py` (lines 933–943):
- When Gen 000 (`FULL_WRITE`, 250 episodes) promotes, `baseline_by_seed` has 250 Gen 000 runs.
- When Gen 001 (`FUNC_MOD_ADD`, 150 episodes: 20 shards + 130 full runs) promotes, lines 940–943 overwrite only 130 seeds in `baseline_by_seed`, leaving 120 stale Gen 000 runs (with Gen 000's old `action_bytes` and depths) inside `baseline_runs`!
- Also, in `lox/engine/orchestrator.py` (lines 676–680), `eval_shards` takes up `len(eval_shards)` of `target_eps`, and `priority_seeds` contains overlapping `shard_seeds`.

### Required Changes in `lox/engine/orchestrator.py`
1. **Keep `baseline_by_seed` Purely From the Current Promoted Policy (lines 493–495 and 933–943)**:
   - On promotion (`if verdict["passed"]:` at lines 933–943, and in `PARAM_TUNE` at lines 493–495), replace `baseline_by_seed` with the newly promoted policy's full runs:
     ```python
     full_promoted = {
         int(r["seed"]): r
         for r in eval_res["runs"]
         if not r.get("is_shard")
     }
     if full_promoted:
         baseline_by_seed = full_promoted
         baseline_runs = list(baseline_by_seed.values())
     ```
     (And similarly for `val_batch["runs"]` in `PARAM_TUNE`). This guarantees `100%` of `baseline_runs` and `100%` of extracted shard `action_bytes` always come from the active champion policy `current_policy`, never from a stale ancestor.

---

## 4. Mandatory Verification Test (`tests/test_overhaul_verification.py`)

Add a new test `test_pass3_pilot_delta_depth_and_exact_batch_capacity()` in `tests/test_overhaul_verification.py` that verifies:
1. **No False Negative in `dry_run_policy` When Shards Have Lower Depth Than Full Runs**:
   - Configure `AuthorToolRegistry` with a `PersistentEvalPool(workers=2, max_steps=15)` and `baseline_runs` where 10 runs are `Dlvl 1` failures (with `action_bytes = b"k" * 20`, `max_depth = 1`, `score = 10`) and 20 runs have `max_depth = 2`, `score = 20`.
   - Call `dry_run_policy` with a policy that achieves higher score/progression than baseline on the pilot runs while the 200-step shards stay on `Dlvl 1`.
   - Assert `tools.last_pilot_improved is True`, `Δ=` in `tools.last_pilot_summary` is `>= 0.0` (not negative!), and `cand_depths` / `base_depths` were paired 1-to-1.
2. **Exact Task Dispatch in `evaluate_batch` When Priority Seeds / Shards Cover `target_episodes`**:
   - Call `pool.evaluate_batch` with `target_episodes=10`, `target_functions=["run"]`, and `priority_seeds=list(range(100, 115))`.
   - Assert `res["raw_episodes_run"] == 10` (it must NOT over-submit `workers * 8` or `needed * 2` tasks when `len(priority_seeds) >= target_episodes`), and assert `[r["seed"] for r in res["runs"]]` is a subset of `range(100, 115)` (zero fresh unpaired seeds).

Run:
```bash
uv run ruff check --fix .
uv run ruff format .
uv run pytest -v
```
All tests must pass with zero warnings or errors.
