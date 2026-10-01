# LOX 2.0: Autonomous Empirical Policy Synthesis for NetHack

**LOX 2.0** is an autonomous, empirical neuro-symbolic policy synthesis engine for NetHack. It enables large language models (such as `google/gemma-4-31b-it`) to iteratively author, evaluate, diagnose, and evolve pure Python generator policies based on flight telemetry, DuckDB incident autopsies, and an offline NetHack encyclopedia.

LOX 2.0 achieves sub-50 microsecond execution latency (>1,000 game turns/second) by compiling generator policies alongside a standalone Numba-accelerated A* spatial navigation engine.

---

## 1. Quickstart: Run on Any Machine in 2 Minutes

### Prerequisites
- **OS**: Linux (Ubuntu 20.04+, Debian, Arch, Fedora) or macOS.
- **Python**: `>= 3.11`.
- **System Packages** (standard build tools for NLE / NetHack C bindings):
  - On Ubuntu/Debian:
    ```bash
    sudo apt update && sudo apt install -y build-essential cmake bison flex libz-dev
    ```
  - On macOS (Homebrew):
    ```bash
    brew install cmake bison flex
    ```
- **uv** (recommended for instant, reproducible virtual environment setup):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source $HOME/.cargo/env  # or restart terminal
  ```

---

### Step 1: Clone and Install
```bash
git clone https://github.com/saejin-moon/lox.git
cd lox

# Install all dependencies with uv
uv sync
```

---

### Step 2: Verify Installation (Run Unit Tests)
```bash
uv run pytest -v
```
All 34 test suites should pass cleanly in ~4 seconds.

---

### Step 3: Run Synthesis

#### Option A: Offline Mock Mode (Zero Setup, No API Keys)
Runs policy evaluation and synthesis locally without making external network calls:
```bash
uv run python -u -m scripts.run_synthesis \
  --provider mock \
  --generations 3 \
  --eval-episodes 5 \
  --max-turns 500
```

#### Option B: Live OpenRouter Campaign (Targeting Depth 10+)
Set your OpenRouter API key and launch the autonomous evolution loop:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

uv run python -u -m scripts.run_synthesis \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --generations 100 \
  --eval-episodes 10 \
  --max-turns 25000 \
  --target-depth 10.0 \
  --policy-path data/latest_policy.py
```

#### Option C: Long-Running Background Execution (nohup)
To let the campaign run continuously in the background overnight:
```bash
nohup uv run python -u -m scripts.run_synthesis \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --generations 100 \
  --eval-episodes 10 \
  --max-turns 25000 \
  --target-depth 10.0 \
  --policy-path data/latest_policy.py > campaign.log 2>&1 &

# Monitor live generation progress
tail -f campaign.log
```

---

### Step 4: Transferring Historical DuckDB Data Across Machines (Optional)

Git tracks all source code and the latest evolved policy checkpoint ([`data/latest_policy.py`](file:///home/bae/lox/data/latest_policy.py)).
The DuckDB database (`data/lox.duckdb`) and archived generation snapshots (`data/policies/`) are excluded by `.gitignore` to keep git operations fast and prevent repository bloat.

To carry over historical episode telemetry, incident logs, token accounting, and generation archives when moving to another machine:

```bash
# 1. From the source machine (push to target):
scp data/lox.duckdb user@remote-machine:/path/to/lox/data/lox.duckdb
scp -r data/policies user@remote-machine:/path/to/lox/data/

# 2. Or, from the new machine (pull from source):
mkdir -p data
scp user@source-machine:/path/to/lox/data/lox.duckdb ./data/lox.duckdb
scp -r user@source-machine:/path/to/lox/data/policies ./data/
```

When you run `scripts.run_synthesis` on the new machine, it will automatically connect to `data/lox.duckdb`, inspect past episodes and death traces for synthesis, and continue incremental logging without data loss.

---

## 2. Inspecting Campaign Telemetry with DuckDB

Every tick, episode, and LLM token usage is recorded in `data/lox.duckdb`. You can query it while the campaign is running:

### Check Generation Performance Summary
```bash
uv run python -c "
import duckdb
conn = duckdb.connect('data/lox.duckdb', read_only=True)
rows = conn.execute('''
    SELECT run_id, count(*) as eps, round(avg(depth), 2) as avg_depth, max(max_depth) as max_depth, round(avg(turns), 1) as avg_turns
    FROM episodes
    GROUP BY run_id
    ORDER BY run_id DESC
    LIMIT 10
''').fetchall()
for r in rows:
    print(r)
"
```

### Inspect Recent Episode Fatalities & Depths
```bash
uv run python -c "
import duckdb
conn = duckdb.connect('data/lox.duckdb', read_only=True)
rows = conn.execute('''
    SELECT episode_id, depth, max_depth, turns, death_reason
    FROM episodes
    ORDER BY episode_id DESC
    LIMIT 10
''').fetchall()
for r in rows:
    print(r)
"
```

### Audit LLM Token Usage & Costs
```bash
uv run python -c "
import duckdb
conn = duckdb.connect('data/lox.duckdb', read_only=True)
print(conn.execute('''
    SELECT model, count(*) as calls, sum(total_tokens) as total_toks, round(sum(cost_usd), 4) as cost_usd
    FROM token_usage
    GROUP BY model
''').fetchall())
"
```

---

## 3. Architecture & Key Modules

```
lox/
├── author/
│   ├── agent.py          # AuthorAgent: ReAct LLM tool loop, AST validator, timeout dry-run validator
│   ├── prompts.py        # System prompt with NetHack 3.6.6 domain rules and action vocabulary
│   └── tools.py          # DuckDBToolRegistry: query_duckdb, get_death_taxonomy, query_wiki
├── core/
│   ├── types.py          # Dataclasses: Observation, HeroState, CombatView, SpatialView, DungeonView, Action
│   └── spatial.py        # SpatialEngine: Numba-accelerated A*, Dijkstra, Frontier BFS (<150 LOC)
├── dsl/
│   ├── schema.py         # Whitelist of sensory predicates, action primitives, and constants
│   ├── parser.py         # Sandboxed AST validator (blocks imports, eval, exec, unauthorized nodes)
│   └── compiler.py       # Generator policy compiler and AST-to-BehaviorTree compiler
├── envs/
│   ├── base.py           # EnvironmentAdapter base class
│   └── nethack.py        # NetHackAdapter: NLE gym wrapper, menu handling, navigation/tactical macros
└── telemetry/
    ├── recorder.py       # In-memory flight recorder & autopsy generator
    ├── parquet.py        # Streaming zero-lock Parquet logger
    ├── consolidator.py   # Vectorized SQL consolidation into DuckDB
    └── tokens.py         # Multi-provider token and cost accounting
```

---

## 4. Policy Execution Paradigm & Critical Invariants

1. **Python Generator Execution**:
   Policies are written as clean Python classes:
   ```python
   class Agent:
       def __init__(self):
           self.last_prayer_turn = -1000

       def run(self, obs):
           while True:
               if obs.spatial.standing_on_stairs_down:
                   obs = yield descend()
                   continue
               # Decisions yield actions and receive fresh observations
               obs = yield wait()
   ```

2. **Zero-Yield Subroutine Invariant**:
   Subroutines called via `obs = yield from self.my_subroutine(obs)` **must yield at least one action on every code path**, or the caller loop must not `continue` in a tight loop. Non-yielding paths cause a 100% CPU generator spin that freezes the environment. The `AuthorAgent` includes a hard 1.0s timeout-shielded dry-run validator that verifies this before deploying any policy.

3. **Peaceful Creature Prompt Immunity**:
   Attacking peaceful creatures (hobbits, gnomes, acid blobs, peaceful humans) triggers confirmation prompts (`"Really attack the ...? [yn] (n)"`). NetHack directional movement answers "No" to prompts, advancing 0 turns. `NetHackAdapter` tracks `self.peaceful_positions`, auto-dismisses confirmation prompts, and excludes peaceful monsters from combat targeting.

4. **Dynamic Blocked Tile Learning**:
   Obstacles such as iron bars, locked iron doors, boulders, and solid walls that reject movement are dynamically recorded in `self.blocked_tiles` per floor depth and immediately pruned from all future navigation graphs (`step_to_frontier`, `step_to_dead_end`, `step_to`, `step_to_stairs_down`).

5. **Floor-Wide Dead-End Secret Door Navigation**:
   NetHack procedural generation blocks access to deeper dungeon levels behind hidden secret doors located at dead-end corridors (`#`) and room perimeter walls (`-`, `|`). When visible frontiers are exhausted (`obs.spatial.has_unvisited_frontier == False`), `obs.spatial.has_unsearched_dead_end` evaluates floor-wide, allowing the hero to call `step_to_dead_end()` to pathfind to corridor dead-ends or room walls and `search()` repeatedly until the secret door is exposed.

6. **Minetown & Dialog Auto-Dismissal (`ESC`)**:
   Watchmen and temple priests in Minetown (Depths 5–9) greet the hero with text-entry dialogs (`"who are you"`, `"what is your name"`). `NetHackAdapter` intercepts these prompts and auto-dismisses them with ESC (`\x1b`), registering guards as peaceful and avoiding 2,500-keystroke timeout abortions.

7. **Autopickup & Equipment Optimization (`wear_armor`)**:
   NLE is configured with `options=("autopickup", "pickup_types:?!/%=[$")`, automatically collecting armor, potions, scrolls, and food as the hero steps over tiles. The policy executes `wear_armor()` during peaceful exploration to equip helmets, body armor, cloaks, and boots, driving Armor Class (AC) down and deflecting monster attacks.

8. **In-Combat Emergency Healing & High-Speed Attackers (`is_fast_dangerous`)**:
   Soldier ants and killer bees move at speed 18 with lethal poison stings. `obs.combat.is_fast_dangerous` flags lethal speedsters, triggering immediate retreat into 1-tile corridor chokepoints (`step_to_chokepoint()`). In-combat emergency healing (`quaff_healing()`) triggers at `< 50% HP` before trading further hits.

9. **Exploration Pacing & Stagnation Auto-Recovery**:
   To prevent heroes from burning thousands of turns in idle `wait()` loops (`MaxTurnsReached`), `NetHackAdapter` automatically decays search counters by 10 when frontiers stall, triggering an active second search sweep across candidate perimeter walls. Exploration policies banish idle `wait()`, continuously patrolling and searching candidate walls until stairs down are found.

---

## 5. Operational Guides for Agents & Developers

- **[`AGENTS.md`](file:///home/bae/lox/AGENTS.md)**: Complete operational context, system foundations, hard-learned lessons, and architectural invariants.
- **[`AGENT_PLAN.md`](file:///home/bae/lox/AGENT_PLAN.md)**: Step-by-step campaign execution plan from current checkpoint to target average depth $\ge 10.0$.
