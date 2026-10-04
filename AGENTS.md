# LOX 2.0 Agent Operational Guide (`AGENTS.md`)

Welcome to LOX 2.0, an autonomous, empirical policy synthesis engine for NetHack.
This document provides complete operational context, architectural foundations, critical invariants, and debugging toolchains for any agent working in this repository.

---

## 1. Mission & Campaign Target

- **Primary Goal**: Synthesize an empirical policy that achieves **Average Dungeon Depth $\ge 20.0$** over a batch of 20 real NetHack episodes.
- **Provider**: OpenRouter (`--provider openrouter`).
- **Target Model**: `google/gemma-4-31b-it`.
- **Max Turns**: 25,000 turns per episode (`--max-turns 25000`).
- **Batch Size**: 20 evaluation episodes per generation (`--eval-episodes 20`).
- **Checkpoint Resumption**: Policies resume seamlessly from [`data/latest_policy.py`](file:///home/moose/git/lox/data/latest_policy.py).
- **Completion Condition**: When a generation batch achieves `avg_depth >= 20.0`, the loop logs `[CAMPAIGN GOAL ACHIEVED]` and finishes.

---

## 2. Core Architecture

```
lox/
├── author/
│   └── agent.py          # AuthorAgent: LLM tool-calling loop, AST validation, multi-scenario dry-run validator
├── core/
│   ├── types.py          # Observation, HeroState, CombatView, SpatialView, DungeonView, Action, FloorCorpse
│   ├── spatial.py        # SpatialEngine: Numba JIT A* pathfinding, frontier discovery, distance grids
│   └── tree.py           # AST-to-BT nodes (Selector, Sequence, Condition, ActionNode, Blackboard)
├── dsl/
│   ├── schema.py         # ALLOWED_PREDICATES, ALLOWED_ACTIONS, ENUM_CONSTANTS vocabulary whitelist
│   ├── parser.py         # AST validator (no imports, no exec/eval, whitelisted nodes)
│   └── compiler.py       # Compiles class-based generator policies (Agent.run(obs)) and AST trees
├── envs/
│   ├── base.py           # EnvironmentAdapter base class
│   └── nethack.py        # NetHackAdapter: NLE gym wrapper, menu dismissal, navigation & tactical primitives
├── eval/
│   └── runner.py         # Batch evaluation engine, DuckDB episode and tick telemetry logger
├── telemetry/
│   └── database.py       # DuckDB schema: episodes, ticks, token_usage, evolved_policies tables
└── wiki/
    └── engine.py         # Offline NetHack wiki retrieval engine for LLM queries
```

---

## 3. Consolidated Technical Invariants & Operational Rules

### 3.1 Policy Architecture & Anti-Infinite-Loop Execution Shields
1. **Python Generator Paradigm & Subroutine Protocol**: Policies are Python classes yielding actions: `obs = yield action`. Subroutines MUST be invoked with `obs = yield from self.subroutine(obs)`.
2. **The Zero-Yield Busy Loop Trap**: Every subroutine called via `yield from` MUST yield an action on every code branch before returning, OR the caller must avoid `continue` without yielding. If a subroutine returns without yielding (e.g. `return obs`) and the caller loops with `continue`, the Python generator spins at 100% CPU without stepping the environment.
3. **Four-Layer Anti-Loop Defense Architecture**:
   - **Layer 1 (AST `LoopGuardTransformer`)**: Instruments every `while` and `for` loop with `_guard.tick()`. If any loop exceeds 2,000 iterations without yielding, raises `RuntimeError("Infinite loop detected")` in $< 1\text{ms}$.
   - **Layer 2 (`PolicyRunner` Fallback)**: Catches `Infinite loop detected` `RuntimeError`, safely falling back to `Action(name="wait")` to advance the game turn.
   - **Layer 3 (Episode Wall-Clock Ceiling)**: `MAX_EPISODE_WALL_SEC = 90.0` in `_run_single_episode_worker` breaks out with `EpisodeWallTimeout` on C-level NLE hangs.
   - **Layer 4 (Worker Pool Batch Ceiling)**: Dispatches `mp.Pool` via `map_async` with a 180s batch ceiling; executes `pool.terminate()` and `pool.join()` to eliminate zombie processes if a batch hangs.
4. **`signal`-Based Policy Dry-Run Timeout**: `AuthorAgent` runs candidate policies across multi-scenario test states (`normal`, `hungry`, `combat`, `floating_eye_combat`) with `signal.setitimer(signal.ITIMER_REAL, 1.0)`. This interrupts zero-yield loops via `SIGALRM` / `TimeoutError` without `ThreadPoolExecutor.shutdown(wait=True)` deadlocks.

### 3.2 Dialog Interception, Prompt Auto-Dismissal & Multi-Key Sequences
5. **Prompt Auto-Dismissal Shields (`[yn]`, `[ynq]`)**: Directional keys answer `n` (No) to prompts and consume 0 turns. `NetHackAdapter._dismiss_more()` automatically intercepts confirmation prompts (`"Really attack?"`, `"Really quit?"`, `"eat it? [ynq]"`) and auto-answers `'n'` (or `'y'` for floor corpses), tracking peaceful positions per level. Exposes `obs.combat.adjacent_peaceful` and skips peaceful entities in hostile queries.
6. **Text-Entry & Item Naming Dialog Interception (ESC `\x1b`)**: In Minetown (Depths 5–9) or when picking up items, prompts like `"who are you"`, `"what is your name"`, `"call a "`, `"call the "`, or `"what do you want to call"` intercept directional keys as text input (0 turns). `_dismiss_more` immediately dismisses these with ESC (`\x1b`), preventing 2,500-step zero-turn aborts.
7. **Intermediate Multi-Key Keystroke Preservation (`is_intermediate`)**: Multi-key commands (`eat_carried_food` `[e, slot]`, `quaff_healing` `[q, slot]`, `wear_armor` `[W, slot]`, `zap_offensive_wand` `[z, slot, dir]`, `throw_dagger` `[t, slot, dir]`) open intermediate item-selection prompts (`"What do you want to eat?"`). `_step_sequence` flags `is_intermediate = (idx < len(seq) - 1)` so `_dismiss_more` does NOT press ESC during intermediate prompts, allowing the subsequent key selection to register cleanly.
8. **Universal Consecutive Zero-Turn Circuit Breaker**: If any sequence of commands generates `consecutive_zero_turns >= 4`, `NetHackAdapter.step()` unconditionally forces `Action(name="wait")` (`.`), advancing the NetHack turn clock and completely eliminating zero-progress aborts (`StepStatus.ABORTED`).

### 3.3 Spatial Navigation, Topological Memory & Level Topology
9. **Full-Floor Persistent Topological Memory**: In NetHack, tiles outside active line-of-sight FOV are zeroed in `raw_obs["glyphs"]` and closed doors (`+`) are non-walkable. `NetHackAdapter` augments `walkable` and `walkable_nav` with persistent memory of all visited tiles (`self.visited`) and discovered clean terrain (`self.known_chars` for `#`, `.`, `<`, `>`, `_`, `{`, `+`). This guarantees 100% unbroken pathfinding connectivity across the floor, preventing heroes from getting trapped in single rooms.
10. **Persistent Level Features vs Level Transitions**: Static level fixtures (`known_stairs_down`, `known_stairs_up`, `known_fountain_pos`, `known_altar_pos`) must NEVER be cleared when leaving FOV; they are strictly preserved until true dungeon level transitions or explicit branch staircase pruning.
11. **Gnomish Mines Avoidance & Staircase Discrimination (`dnum == 2`)**: In NetHack `dungeon.def`, `dnum = 0` (Dungeons of Doom), `dnum = 1` (Gehennom), `dnum = 2` (Gnomish Mines), `dnum = 3` (The Quest), `dnum = 4` (Sokoban). When descending into the Mines (`dnum == 2`), the adapter flags the stair in `mines_stairs_positions`, removes it from `known_stairs_down`, and the policy immediately yields `ascend()` on `<` back to Dungeons of Doom to continue downward toward Depth 20.
12. **Trap-Door Fall vs Staircase Arrival**: Falling through trap doors or holes (`^`) lands the hero on open floor (`.`), NOT stairs up (`<`). `NetHackAdapter` tracks `had_explicit_descent`, invalidates `known_stairs_up` upon trap door messages or `"You can't go up here."`, and shields `ascend()` from 0-turn loops by falling back to `step_to_frontier()`.
13. **Doorway Discrimination & Breaching**: Cardinal directions only `((-1, 0), (1, 0), (0, -1), (0, 1))`. Differentiates closed doors from floor spellbooks (`+`) using NLE CMAP glyphs (`nh.GLYPH_CMAP_OFF + 15` and `+ 16`). Breaches locked doors with kicking (max 6 kicks); unbreachable doors (or locked doors inside shops) are added to `blocked_tiles` and masked out of chokepoints.
14. **Dynamic Obstacle Learning (`self.blocked_tiles`)**: Impassable terrain messages (`"cannot pass through the bars"`, `"It's a wall"`, `"It's solid stone"`) add coordinates to `self.blocked_tiles` ONLY when triggered by genuine directional movement (`is_step_direction`).
15. **Stride-2 Checkerboard Secret Door Search & In-Place Prevention**: Room perimeters are searched using a stride-2 checkerboard `(cy + cx) % 2 == 0` plus corners (`adj_wall >= 2`), providing 100% geometric wall coverage while halving search turns. `step_to_dead_end` excludes the current position (`step_target_mask[hero.y, hero.x] = False`), and policies enforce `self.last_searched_pos` to prohibit consecutive searches on the same tile without movement. Stagnation decays search counts by 10 when frontiers stall.
16. **Numba JIT Disk-Cached Spatial Kernels**: `_compute_dead_ends_mask_kernel` compiles with `cache=True` and is pre-warmed via `SpatialEngine.warmup()`, executing in ~58 µs/call (7x speedup).
17. **Tactical Retreat Door Breaching & Open Floor Preference**: `step_away_from_hostile()` strictly prefers open walkable tiles over closed/locked doors. Any retreat path requiring a closed door routes through `_step_or_breach()` to kick it open rather than bumping into locked doors.
18. **No-Hostile Tactical Fallback**: If `step_away_from_hostile()` is invoked without hostiles in FOV, it immediately falls back to `step_to_stairs_down`, `step_to_frontier`, or `step_to_dead_end`, never engraving or waiting in place.
19. **Solver Fallback Closed Door Navigation**: Solvers (`harvest_poison_res`, `test_altar_buc`) must check `obs_prev.dungeon.adjacent_closed_door` to open/breach doors before falling back to distant frontiers.

### 3.4 Combat Tactics, Hazard Discrimination & Elbereth Sanctuary
20. **Dust Elbereth Sanctuary Mechanics**: Writing `"Elbereth"` in the dust with bare fingers (`E` $\to$ `-` $\to$ `Elbereth\r`) creates a 100% ward against all non-humanoid monsters. Coordinates are pre-added to `self.elbereth_positions` before sequence execution so `obs.combat.standing_on_elbereth` is immediately True (preventing re-engraving wipes). Monster attacks do NOT erase dust Elbereth; positions are discarded ONLY upon genuine smudge/wipe messages (`"wipe out"`, `"scuffed"`, `"erased"`).
21. **Elbereth-Immune Species Discrimination (`IGNORES_ELBERETH_SPECIES`)**: Orcs, elves, humans, goblins, hobgoblins, gnomes, dwarves, ogres, trolls, giants, zombies, mummies, and vampires ignore Elbereth or attack with ranged weapons. When facing immune hostiles (`hostile_ignores_elbereth == True`), policies MUST engage in melee or retreat to 1-tile corridor chokepoints rather than waiting passively on the ward.
22. **Passive & Exploding Hazard Pathfinding Masking & Cornered Wait Protocol**: Floating eyes inflict 50-turn passive paralysis when hit in melee; gas spores inflict lethal 4d6 area blast damage; molds/jellies inflict passive retaliatory damage. In addition to melee refusal in `melee_attack_hostile`, `_build_walkable_nav` strictly masks out all passive hazards (`floating eye`, `gas spore`, molds, jellies, exploding spheres) so pathfinding navigation (`step_to_frontier`, `step_to_dead_end`, `step_to_stairs_down`, `step_to_loot`) never routes through them. When cornered or unable to increase distance in `step_away_from_hostile()`, the adapter yields `wait()` rather than stepping into them. When an active attacker is present alongside a passive hazard, `obs.combat.has_safe_melee_target` allows the hero to engage the active attacker in melee while the adapter skips the passive hazard, preventing hero idling while being bitten. Passive hazards are eliminated safely at range $\ge 2$ using thrown daggers or offensive wands.
23. **High-Speed Attackers & Corridor Chokepoints (`is_fast_dangerous`)**: Dynamic NLE `pm.mmove > 12` check plus foxes (speed 15), coyotes, jaguars, soldier ants / killer bees (speed 18), and giant bats (speed 22) outpace hero speed (12). When in open rooms, the policy immediately flees to 1-tile corridor chokepoints (`step_to_chokepoint()`) to force 1v1 engagements.
24. **In-Combat Emergency Healing & Panic Escape**:
   - Quaffing healing potions takes 1 turn and restores 10–20 HP; triggered at `hp_frac < 0.50`.
   - Emergency panic escape: reading scrolls of teleportation or zapping wands of teleportation (`has_panic_escape`) triggers when surrounded or at `hp_frac < 0.25` with prayer on timeout.
25. **Ranged Missile Harassment (`throw_dagger`)**: Starting daggers thrown at distance $\ge 2$ soften high-speed pests before melee contact. NetHack drops thrown daggers onto the floor, allowing post-combat recovery.
26. **Staircase Combat Interception**: If `obs.spatial.standing_on_stairs_down and not obs.status.is_levitating`, policies prioritize `descend()` immediately (even during combat), taking the 1-turn descent to escape to the next depth.

### 3.5 Inventory, Autopickup, Equipment & Nutrition
27. **Autopickup & Equipment Acquisition (`wear_armor`)**: NetHack initialized with `options=("autopickup", "pickup_types:?!/%=[$")`. Unworn dropped armor is equipped during peaceful exploration (`wear_armor()`), driving AC toward negative values. Failed armor slots are tracked in `failed_wear_slots` and excluded from `has_unworn_armor` to prevent redundant wear loops.
28. **Floor Loot Scooping & Ping-Pong Protection (`step_to_loot`)**: Dropped monster equipment within distance $\le 4$ is scooped via `step_to_loot`. Reached tiles are added to `self.looted_tiles` (cleared on floor change) and `loot_chars` is restricted strictly to autopicked types (`[`, `!`, `?`, `/`, `=`, `$`, `%`), eliminating 2-tile oscillation loops over dropped weapons/ammo.
29. **Corpse Safety & Nutrition Harvesting**: NLE body glyph queries (`nethack.glyph_is_body` and `permonst`) verify species and poison/deadly status. `_dismiss_more` auto-confirms `"eat it? [ynq]"` with `'y'`. Fresh corpses are consumed opportunistically right after combat at `hunger_state < 3`, preserving carried food rations for deep-level hunger emergencies. Carried food is consumed in 1 turn (`eat_carried_food`) during peaceful exploration or inside `handle_combat` when hungry (`hunger_state >= 2`) while on Elbereth, adjacent to passive hazards, or away from enemies to prevent starvation fainting. Major trouble prayer (`hunger_state >= 3` or `hp_frac < 0.15`) grants divine feeding.
30. **Artifact Weapon Scaling (`dip_excalibur`)**: Lawful Valkyries XL $\ge 5$ dipping a long sword into a fountain have a 1/6 chance per dip of forging Excalibur (+1d10 damage, secret door searching, life drain immunity). Dipping strictly requires standing directly on the fountain tile (`obs.dungeon.standing_on_fountain`); dipping from an adjacent square is rejected. `known_fountain_pos` is cleared upon `"The fountain disappears!"`.
31. **Weapon Grammar Recognition**: Tracks one-handed `(weapon in hand)` (singular) and two-handed `(weapon in hands)` (plural); checking only plural previously caused one-handed weapons to falsely report as unequipped bare hands.
32. **Encumbrance-Locked Nutrition**: Strained encumbrance blocks eating. `eat_carried_food` drops unworn heavy armor before consuming rations if encumbered.
33. **Offensive Wand Zapping (`zap_offensive_wand`)**: Directs offensive wands (striking, magic missile, fire, cold, sleep, lightning) at hostiles at distance $\ge 2$.

### 3.6 Telemetry, Parallel Evaluation & System Resilience
34. **Parallel Multiprocessing Worker Scaling (`--workers 20`)**: Host has 22 CPU cores. Running with `--workers 20` executes all 20 episodes concurrently in a single wave, cutting batch evaluation time from ~4.5 minutes down to ~35 seconds (8x speedup).
35. **I/O Telemetry Buffering**: Parquet telemetry writes are buffered with `flush_interval=5000` to prevent SSD write contention across concurrent worker processes.
36. **Ground-Truth NetHack Scoring & Granular Telemetry**: Official game score is extracted from `blstats[nh.NLE_BL_SCORE]` ($2\text{XP} + \text{Gold} + 50(\text{DL}-1)$). Ephemeral death telemetry tracks `killer`, `ac_at_death`, `hp_at_death`, `max_hp_at_death`, and `excalibur_forged` into DuckDB.
37. **Truthful Mortality Classification**: Real fatalities (`hp <= 0`) are classified by cause (combat, starvation, poison); zero-progress aborts are classified as `Aborted / ZeroProgress`; `MaxTurnsReached` is strictly reserved for runs alive at the 25k turn cap.
38. **OpenRouter Network Resilience**: `AuthorAgent._call_openai_compatible` implements 6-attempt exponential backoff retry with randomized jitter for all network/timeout errors, rate limits (429), and 5xx upstream gateway errors.
39. **Vault Guard, Priest & Peaceful Non-Aggression**: In NetHack, Vault Guards (`guard`), priests (`priest`, `priestess`), and town watchmen are neutral/peaceful until provoked. Attacking or throwing missiles at a Level 12 Vault Guard turns them hostile and causes instant hero death. `is_peaceful_species` explicitly includes `"guard"`, `"priest"`, `"priestess"`, `"shopkeeper"`, `"watchman"`, `"oracle"`. Missiles (`throw_dagger`) and wands (`zap_offensive_wand`) are hard-shielded against targeting peaceful positions (`target_pos in self.peaceful_positions`), and policies yield retreat rather than attacking.
40. **Active Hostile Gating & Passive Hazard Bypass (`has_active_hostile`)**: Immobile passive hazards (`floating eye`, `gas spore`) at distance $\ge 2$ do not hunt or attack the hero. If the hero lacks ranged weapons (daggers or offensive wands) to snipe them safely, `has_active_hostile == False` allows `run()` to bypass `handle_combat` completely. Exploration continues uninterrupted as `walkable_nav` naturally routes around passive hazards, eliminating infinite combat standoffs, melee paralysis, and starvation locks.

---

## 4. Key CLI Commands

### Run Synthesis Campaign
```bash
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

### Run Unit Tests
```bash
uv run pytest -v
```

### Query Campaign Progress via DuckDB
```bash
# Check recent episode performance
uv run python -c "
import duckdb
conn = duckdb.connect('data/lox.duckdb', read_only=True)
rows = conn.execute('''
    SELECT run_id, episode_id, depth, max_depth, turns, death_reason 
    FROM episodes 
    ORDER BY episode_id DESC 
    LIMIT 10
''').fetchall()
for r in rows:
    print(r)
"

# Check generation summary metrics
uv run python -c "
import duckdb
conn = duckdb.connect('data/lox.duckdb', read_only=True)
rows = conn.execute('''
    SELECT run_id, count(*), avg(max_depth), max(max_depth), avg(turns)
    FROM episodes 
    GROUP BY run_id
    ORDER BY run_id DESC
    LIMIT 10
''').fetchall()
for r in rows:
    print(r)
"
```
