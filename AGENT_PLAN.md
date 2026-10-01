# LOX 2.0 Campaign Action Plan (`AGENT_PLAN.md`)

This plan guides the incoming agent step-by-step from the current checkpoint to achieving the campaign target of **Average Depth $\ge 10.0$**.

---

## Current State & Baseline

- **Current Checkpoint**: [`data/latest_policy.py`](file:///home/bae/lox/data/latest_policy.py).
- **Recent Performance**: Reached **Depth 8** (7 consecutive staircases descended, 6,372 turns survived). 100% of episodes now clear Depth 1.
- **Recent Platform Hardening**:
  1. *Peaceful Creature Trapping Resolved*: Adapter tracks `peaceful_positions`, auto-dismisses `[yn]` prompts with `'n'`, and ignores peaceful monsters in combat selection.
  2. *Coordinate Navigation (`step_to(y, x)`)*: Fully supported in compiler and adapter with A* pathfinding.
  3. *Dynamic Obstacle Learning*: Blocked tiles (iron bars, walls) are dynamically detected from in-game messages and excluded from navigation masks.
  4. *Corpse Eating Safety*: `eat_floor_corpse` automatically steps toward floor corpses if the hero is not standing on one.
  5. *Dead-End & Room Perimeter Secret Door Invariants*: Guided LLM to use `step_to_dead_end()` and `search()` on corridor dead ends and room perimeter walls when frontiers are exhausted.
  6. *Granular Telemetry & Timeout Reporting*: `scripts/run_synthesis.py` extracts real causes of death (starvation, poison, combat strikes) and identifies `MaxTurnsReached`, while tracking `descents`, `attacks`, `searches`, `eats`, and `prayers`.
  7. *Minetown Dialog Auto-Dismissal (`ESC`)*: Automatically cancels text-entry prompts (`"who are you"`, `"what is your name"`, `"call this"`, `"hello stranger"`) with ESC (`\x1b`) and registers guards as peaceful, avoiding 2,500-keystroke timeouts at Depths 5–8.
  8. *Autopickup & Armor Equipping (`wear_armor`)*: NLE configured with `options=("autopickup", "pickup_types:?!/%=[$")`. Added `has_unworn_armor` and `wear_armor()` with Redundant Armor Slot Shield (automatically tracks failed armor slots and prunes them from `has_unworn_armor`, preventing 17,000-turn loops).
  9. *In-Combat Emergency Healing & Fast Monster Chokepoint Retreat*: In-combat healing triggered at `< 50% HP` inside `handle_combat()`. Soldier ants/killer bees (`is_fast_dangerous`) trigger immediate retreat to corridor chokepoints.
  10. *Test Suite*: All 36 unit tests pass (`uv run pytest -v`).
  11. *Unified Stride-2 Checkerboard Secret Door Search*: Unified corridor dead ends and perimeter candidates into `_compute_dead_ends_mask`. Uses checkerboard stride-2 pattern `(cy + cx) % 2 == 0` plus all corner tiles (`adj_wall >= 2`) to ensure 100% geometric coverage of all room walls while cutting search stops and turns by $> 50\%$. Automatically decays search counts by 10 on stagnation and sorts reachable candidates by search count and BFS walking distance.
  12. *Prompt Auto-Dismissal (`[ynq]` Post-Death / Identification Shield)*: Auto-dismisses `[ynq]` prompts (`"Do you want your possessions identified?"`, creatures slaughtered, etc.) with `'n'`, preventing NLE smooth-quit aborts and 5,000-turn dead stalls.

---

## Action Plan: Steps to Reach Depth 10+

### Step 1: Launch Campaign in Background
Run the synthesis loop resuming from `data/latest_policy.py`:
```bash
uv run python -u -m scripts.run_synthesis \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --generations 100 \
  --eval-episodes 10 \
  --max-turns 25000 \
  --target-depth 10.0 \
  --policy-path data/latest_policy.py
```
*Note*: Set `WaitMsBeforeAsync: 5000` so it transitions to a background task smoothly.

### Step 2: Reactive Monitoring Loop
- Do **not** poll or loop on `status`. Use reactive wakeups or standard timers with the `schedule` tool if checking periodically.
- Tail the log file to observe generation transitions:
  `tail -n 25 <task_log_path>`
- Watch for:
  - `[Checkpoint Saved] -> data/latest_policy.py`
  - `[Gen X Batch Metrics (10 eps)] Avg Depth: Y.YY | Max Depth: Z | Avg Turns: W.W`

### Step 3: Diagnostic Triage via DuckDB
If average depth plateaus between generations, run a diagnostic query:
```python
import duckdb
conn = duckdb.connect('data/lox.duckdb', read_only=True)
# Inspect death reasons in recent batch
print(conn.execute('''
    SELECT death_reason, count(*), avg(depth), max(depth), avg(turns)
    FROM episodes
    WHERE run_id = (SELECT run_id FROM episodes ORDER BY episode_id DESC LIMIT 1)
    GROUP BY death_reason
''').fetchall())
```
Common NetHack progression milestones to track:
- **Depths 1–3**: Sewer rats, jackals, kobolds, goblins, floating eyes (paralysis), hunger management.
- **Depths 4–7**: Gnomes with wands, soldier ants (fast poison), leocrottas, orcs with bows, traps.
- **Depths 8–10+**: Oracle level (depth 5–9), Mines entrance, Sokoban portal, stronger pack monsters. Ensure prayer is utilized at `< 25% HP` and poison resistance is prioritized when eating corpses.

### Step 4: Campaign Goal Verification & Completion
- When a batch achieves `avg_depth >= 10.0`, `scripts.run_synthesis` will log:
  ```
  =======================================================
  [CAMPAIGN GOAL ACHIEVED] Target average depth 10.0+ reached!
  Final Batch Average Depth: X.XX (Max Depth: Y)
  =======================================================
  ```
- Confirm the final metrics in `data/lox.duckdb`:
  ```bash
  uv run python -c "
  import duckdb
  conn = duckdb.connect('data/lox.duckdb', read_only=True)
  print(conn.execute('SELECT run_id, avg(depth), max(depth) FROM episodes GROUP BY run_id ORDER BY run_id DESC LIMIT 1').fetchall())
  "
  ```
- Emit the goal completion signal: `<!-- GOAL_COMPLETE -->`.
