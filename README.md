# LOX: Autonomous Empirical Policy Synthesis for NetHack

**LOX** is an autonomous, empirical neuro-symbolic policy synthesis engine for NetHack. It enables large language models (such as `google/gemma-4-31b-it`) to iteratively author, evaluate, diagnose, and evolve pure Python generator policies based on flight telemetry, DuckDB incident autopsies, and an offline NetHack encyclopedia.

LOX achieves sub-50 microsecond execution latency (>1,000 game turns/second) by compiling generator policies alongside a standalone Numba-accelerated A* spatial navigation engine. Across 27 continuous evolutionary campaigns and over 5,700 real episodes evaluated (18.2M game ticks), LOX has systematically eradicated zero-progress stalls, locked-door loops, and starvation blackouts, driving peak batch performance to **5.10 Average Depth** and peak exploration depth to **Dungeon Level 14**.

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
All 63 test items pass cleanly in ~1.8 seconds.

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

#### Option B: Live OpenRouter Campaign (Targeting Depth 20.0 / Endgame)
Set your OpenRouter API key and launch the 20-worker parallel synthesis loop:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."

uv run python -u -m scripts.run_synthesis \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --generations 10 \
  --eval-episodes 20 \
  --max-turns 25000 \
  --target-depth 20.0 \
  --workers 20 \
  --policy-path data/latest_policy.py
```

#### Option C: Long-Running Background Execution (nohup)
To let the campaign run continuously in the background overnight:
```bash
nohup uv run python -u -m scripts.run_synthesis \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --generations 100 \
  --eval-episodes 20 \
  --max-turns 25000 \
  --target-depth 20.0 \
  --workers 20 \
  --policy-path data/latest_policy.py > campaign.log 2>&1 &

# Monitor live generation progress
tail -f campaign.log
```

---

### Step 4: Transferring Historical DuckDB Data Across Machines (Optional)

Git tracks all source code and the latest evolved policy checkpoint ([`data/latest_policy.py`](file:///home/moose/git/lox/data/latest_policy.py)).
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
    SELECT run_id, count(*) as eps, round(avg(max_depth), 2) as avg_depth, max(max_depth) as max_depth, round(avg(score), 1) as avg_score, max(score) as peak_score
    FROM episodes
    GROUP BY run_id
    ORDER BY min(episode_id) DESC
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
    SELECT episode_id, depth, max_depth, score, turns, death_reason
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

## 3. Empirical Milestones & Benchmark History

LOX has evaluated **5,725 episodes** across **18,202,078 game turns** in DuckDB:

| Metric | Campaign 1 Baseline | Mid-Campaigns (C10–C15) | Current Highs (C24–C27) |
| :--- | :---: | :---: | :---: |
| **Batch Avg Depth** | 1.91 | 3.51 – 3.66 | **5.10** (C27 Gen 5) / **4.07** (C27 avg) |
| **Peak Max Depth** | 5 | 9 – 11 | **14** (C25 Gen 2) |
| **Batch Avg Score** | 204.3 | 440 – 530 | **1,014.5** (C27 Gen 9) |
| **Peak Single Score** | 922 | 2,645 | **4,612** (C24 Gen 6) |
| **Batch Wall-Clock** | ~270s | ~180s | **~35s** (20 workers concurrent) |
| **Starvation Mortality** | ~28% | ~12% | **0.0%** (Completely Eliminated) |
| **Zero-Progress Aborts** | ~18% | ~4% | **0.0%** (Completely Eliminated) |

---

## 4. Architecture & Key Modules

```
lox/
├── author/
│   ├── agent.py          # AuthorAgent: ReAct LLM tool loop, AST validator, timeout dry-run validator
│   ├── prompts.py        # System prompt with NetHack 3.6.6 domain rules and action vocabulary
│   └── tools.py          # DuckDBToolRegistry: query_duckdb, get_death_taxonomy, query_wiki
├── core/
│   ├── types.py          # Dataclasses: Observation, HeroState, CombatView, SpatialView, DungeonView, Action
│   └── spatial.py        # SpatialEngine: Numba JIT A*, Dijkstra, Frontier BFS, disk-cached kernels (<58 µs)
├── dsl/
│   ├── schema.py         # Whitelist of sensory predicates, action primitives, and constants
│   ├── parser.py         # Sandboxed AST validator (blocks imports, eval, exec, unauthorized nodes)
│   └── compiler.py       # Generator policy compiler and AST-to-BehaviorTree compiler
├── envs/
│   ├── base.py           # EnvironmentAdapter base class
│   └── nethack.py        # NetHackAdapter: NLE gym wrapper, menu handling, navigation/tactical macros
├── eval/
│   └── runner.py         # 20-worker parallel batch evaluation engine, Parquet telemetry streaming
└── telemetry/
    ├── recorder.py       # In-memory flight recorder & autopsy generator
    ├── parquet.py        # Streaming zero-lock Parquet logger
    ├── consolidator.py   # Vectorized SQL consolidation into DuckDB
    └── tokens.py         # Multi-provider token and cost accounting
```

---

## 5. Policy Execution Paradigm & Core Invariants

1. **Python Generator Protocol**:
   Policies are written as clean Python classes yielding action primitives:
   ```python
   class Agent:
       def __init__(self):
           self.last_prayer_turn = -1000

       def run(self, obs):
           while True:
               if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                   obs = yield descend()
                   continue
               obs = yield wait()
   ```

2. **Four-Layer Anti-Loop Shield Architecture**:
   - **Layer 1 (AST `LoopGuardTransformer`)**: Automatically instruments all `while` and `for` loops with `_guard.tick()`. Triggers a `RuntimeError` if >2,000 iterations elapse without yielding.
   - **Layer 2 (`PolicyRunner` Fallback)**: Catches infinite loop runtime errors and safely yields `Action(name="wait")` (`.`), advancing the NetHack turn clock.
   - **Layer 3 (Episode Wall-Clock Ceiling)**: Enforces a 90-second ceiling per episode worker to break out of internal C-level NLE hangs.
   - **Layer 4 (Batch Timeout Recovery)**: `map_async` with a 180-second batch ceiling terminates and cleans up stalled worker pool processes.

3. **Dust Elbereth Sanctuary Mechanics & Immune Discrimination**:
   Finger-engraving `"Elbereth"` in the dust (`E` $\to$ `-` $\to$ `Elbereth\r`) creates an absolute ward against non-humanoids (spiders, dogs, ants, bees). `IGNORES_ELBERETH_SPECIES` discriminates orcs, elves, humans, and trolls who ignore the ward, forcing melee engagements or corridor retreats.

4. **Passive Hazard Distance-1 Projectile Elimination**:
   Throwing projectiles (daggers, darts, arrows, rocks) at adjacent floating eyes (`obs.combat.adjacent_floating_eye`) is 100% safe at distance 1 without triggering passive paralysis. Cornered dead-end melee breaks eliminate floating eyes rather than starving in place.

5. **Proactive In-Combat Nutrition & Rotten Corpse Shield**:
   Rotten carried corpses are strictly filtered out of food slots. Rations are proactively consumed at `hunger_state >= 2` ("Weak") when on Elbereth or out of melee reach, preventing 30-turn unconscious fainting blackouts. Major trouble prayer (`hunger_state >= 3`) provides divine feeding when packaged food is exhausted.

6. **350-Turn Safe Divine Favor Threshold**:
   Enforces `turn - last_prayer_turn >= 350` before emergency or hunger prayer, guaranteeing safe divine intervention and avoiding deity anger ("Tyr is displeased").

7. **Full-Floor Persistent Topological Memory**:
   Terrain discovered across line-of-sight FOV boundaries (`self.visited`, `self.known_chars` for `#`, `.`, `<`, `>`, `_`, `{`, `+`) is permanently retained, providing unbroken pathfinding connectivity across entire levels.

8. **Dynamic Obstacle Learning**:
   Impassable tiles (iron bars, solid walls, unbreachable doors) are dynamically added to `self.blocked_tiles` upon physical bump messages and pruned from all pathfinding kernels.

9. **Fast Speedster Chokepoints & Swarm Defense**:
   Monsters with movement speed $> 12$ (killer bees, soldier ants, foxes, giant bats) trigger immediate retreat into 1-tile corridor chokepoints (`step_to_chokepoint()`). Facing multiple speed-18 predators triggers dust Elbereth engraving to panic and disperse the swarm.

---

## 6. Operational Guides for Agents & Developers

- **[`AGENTS.md`](file:///home/moose/git/lox/AGENTS.md)**: Complete operational guide, architecture, 48 technical invariants, and synthesis protocols.
- **[`TODO.md`](file:///home/moose/git/lox/TODO.md)**: Real-time campaign tracking, empirical benchmark progression, autopsy findings, and active roadmap.
