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

## 3. Critical Invariants & Hard-Won Lessons

1. **Python Generator Policy Paradigm**:
   - LOX 2.0 policies are written as Python classes with generator execution:
     ```python
     class Agent:
         def __init__(self):
             self.last_prayer_turn = -1000
         def run(self, obs):
             while True:
                 # Decisions yield actions and receive fresh observations
                 obs = yield wait()
     ```
   - Subroutines are invoked with `obs = yield from self.handle_combat(obs)`.

2. **The Zero-Yield Busy Loop Trap**:
   - **Crucial Rule**: Every subroutine called via `yield from` MUST yield at least one action on EVERY code branch before returning, OR the caller must not `continue` in a tight loop.
   - If a subroutine returns without yielding (e.g. `return obs`) and the caller loop does `continue`, the Python generator spins at 100% CPU without stepping the game environment, hanging execution.
   - **Safety Shield**: [`AuthorAgent.synthesize_policy`](file:///home/bae/lox/lox/author/agent.py) features a timeout-shielded multi-scenario dry-run validator (`normal`, `hungry`, `combat` states with a 1.0s hard timeout) that catches and auto-repairs any non-yielding policy before evaluation.

3. **Prompt Auto-Dismissal (`[yn]` and `[ynq]` Confirmation Shields)**:
   - NetHack frequently triggers confirmation prompts: `"Really attack the ...? [yn] (n)"`, `"Really quit?"`, or post-game identification prompts `"Do you want your possessions identified? [ynq] (n)"`.
   - In NetHack, directional keys answer `n` (No) to prompts and consume 0 turns. If prompts are not answered, NLE enters an infinite loop, aborts with `"Warning: smooth quitting of game failed"`, or exhausts the turn cap.
   - [`NetHackAdapter`](file:///home/bae/lox/lox/envs/nethack.py) automatically:
     - Auto-dismisses all `(y/n)`, `[yn]`, and `[ynq]` prompts with `'n'`.
     - Tracks `self.peaceful_positions` per dungeon level upon peaceful encounter prompts.
     - Excludes peaceful positions from `hostile_count`, `adjacent_hostiles_count`, and `melee_attack_hostile`.
     - Exposes `obs.combat.adjacent_peaceful`.

4. **Dynamic Obstacle Learning (`self.blocked_tiles`)**:
   - NetHack has impassable obstacles (iron bars, locked iron doors, boulders, solid stone) that share ASCII characters with corridors (`#`).
   - If a movement action yields `"cannot pass through the bars"`, `"It's a wall"`, or `"It's solid stone"`, `NetHackAdapter` dynamically adds the coordinate to `self.blocked_tiles` for that dungeon depth. All navigation algorithms (`step_to_frontier`, `step_to_dead_end`, `step_to`, `step_to_stairs_down`) immediately prune these coordinates from their navigation graphs.

5. **Coordinate Navigation (`step_to(y, x)`)**:
   - `step_to(y, x)` accepts coordinate integers or tuples and executes A* pathfinding via `SpatialEngine.find_path`.

6. **Door Navigation & Glyph Discrimination**:
   - NetHack strictly forbids diagonal door opening or kicking (`"You see no door there"`). `NetHackAdapter` restricts door actions to cardinal directions `((-1, 0), (1, 0), (0, -1), (0, 1))`.
   - In NetHack ASCII, spellbooks on the floor also display as `+`. Using raw character matching `chars == ord("+")` mistakenly treats spellbooks as doors, causing 2,500 consecutive invalid `open_door` commands until NLE aborts the episode (`StepStatus.ABORTED`).
   - `NetHackAdapter` strictly discriminates real closed doors using NLE CMAP glyphs (`nh.GLYPH_CMAP_OFF + 15` and `+ 16`), dynamically marks phantom door coordinates in `self.non_door_tiles` upon receiving `"You see no door there"`, and safely falls back to `wait()` if no valid door is adjacent.

7. **Corpse Safety, Glyph Ground-Truth & Nutrition**:
   - `NetHackAdapter` uses NLE body glyph queries (`nethack.glyph_is_body` and `permonst(glyph - GLYPH_BODY_OFF)`) to pinpoint the exact tile, monster species, and poison/deadly status of every corpse on the floor.
   - `_dismiss_more()` explicitly auto-answers `'y'` to NetHack's `"eat it? [ynq]"` confirmation prompt, preventing the prompt auto-dismissal shield from rejecting floor corpses.
   - `NetHackAdapter.step(Action(name="eat_floor_corpse"))` automatically steps towards the nearest safe floor corpse if not currently standing on one.

8. **Minetown & Text-Entry Dialog Auto-Dismissal (`ESC`)**:
   - In Minetown (Depths 5–9), guards and temple priests greet the hero with `"Hello stranger, who are you?"` or `"what is your name"`.
   - Directional keys enter characters into the text-entry buffer instead of moving, consuming 0 turns and leading to NLE aborting at 2,500 consecutive 0-turn steps.
   - `NetHackAdapter._dismiss_more()` automatically intercepts text-entry dialogs (`"who are you"`, `"what is your name"`, `"call this"`, `"hello stranger"`) and dismisses them with ESC (`\x1b`, action index 38), and registers the position in `self.peaceful_positions`. This leaves guards peaceful without provoking them.

9. **Autopickup & Equipment Acquisition (`wear_armor`)**:
   - By default, characters remained naked (AC 7–10) throughout the dungeon.
   - `NetHackAdapter` initializes NLE with `options=("autopickup", "pickup_types:?!/%=[$")`, automatically collecting armor, scrolls, potions, wands, and food as the hero steps over tiles.
   - `obs.inventory.has_unworn_armor` and `wear_armor()` allow the policy to equip dropped helmets, mail, cloaks, and boots during peaceful exploration, driving Armor Class down towards negative values (vastly reducing monster hit chances).
   - **Redundant Armor Slot Shield**: When autopickup grabs extra armor of an already-occupied slot (e.g. ring mail when wearing scale mail), NetHack rejects wearing it. `NetHackAdapter` tracks failed armor slots and automatically excludes them from `obs.inventory.has_unworn_armor` and `get_unworn_armor_slot()`, completely preventing the 17,000-turn `wear_armor` infinite loop.

10. **Tactical In-Combat Healing & High-Speed Attackers (`is_fast_dangerous`)**:
    - Soldier ants and killer bees move at speed 18 (nearly 2x hero speed) and deliver lethal poison stings. Engaging them in open rooms is fatal.
    - `obs.combat.is_fast_dangerous` flags lethal speedsters, triggering immediate retreat into 1-tile corridor chokepoints (`step_to_chokepoint()`) to force 1v1 fights.
    - In-combat emergency healing: Quaffing healing potions takes 1 turn and restores 10–20 HP. `handle_combat` checks `if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing: yield quaff_healing()` immediately.

11. **Unified Stride-2 Checkerboard Secret Door Navigation**:
    - Procedural generation often conceals doors along room perimeter walls (`-`, `|`) or at blind corridor tips (`#`).
    - Rather than prioritizing distant corridor dead ends over adjacent room walls, `NetHackAdapter` unifies corridor dead ends and perimeter candidates into `_compute_dead_ends_mask`.
    - Room perimeters utilize a mathematical stride-2 checkerboard pattern `(cy + cx) % 2 == 0` plus all corner/alcove tiles (`adj_wall >= 2`). Because NetHack's `search` examines a $3 \times 3$ bounding box, this guarantees 100% geometric coverage of all room walls while cutting search stops and turns by $> 50\%$.
    - `step_to_dead_end()` sorts reachable candidates by search count and BFS walking distance, ensuring heroes clear local room perimeter walls before undertaking cross-dungeon treks.

12. **Exploration Pacing & Stagnation Auto-Recovery**:
    - When visible frontiers and dead ends are exhausted without discovering stairs, idling with `wait()` causes the hero to burn thousands of turns until reaching `MaxTurnsReached`.
    - `NetHackAdapter` automatically decays search counters by 10 (`self.searched_count = np.maximum(0, self.searched_count - 10)`) when frontiers stall, triggering an active second search sweep across candidate perimeter walls.
    - Exploration policies must strictly avoid yielding `wait()`; they should continually call `handle_dead_end(obs)` or `search()` to investigate room perimeter walls.

13. **Dust Elbereth Sanctuary (`engrave_dust_elbereth`)**:
    - Writing `"Elbereth"` in the dust using bare fingers (`E` $\to$ `-` $\to$ `Elbereth\r`) creates a 100% reliable ward against all non-humanoid monsters (ants, bees, bats, mumakil, leocrottas, canines, jellies).
    - Prevents fatal parting strikes when low on HP (`< 35%`) or cornered in open rooms.
    - While standing on an active Elbereth ward, the policy must avoid melee attacks (which can scuff the dust) and instead heal, pray, or wait for regeneration.

14. **Ranged Missile Harassment (`throw_dagger`)**:
    - Valkyries start with daggers. Thrown at hostiles at distance $\ge 2$ (`t` $\to$ `slot` $\to$ `dir`).
    - Neutralizes high-speed lethal pests (bats, killer bees, soldier ants) before melee contact. Thrown daggers can safely be consumed down to 0 because NetHack drops them on the floor for retrieval during or after combat.

15. **Artifact Weapon Scaling (`dip_excalibur`)**:
    - Lawful Valkyries at Experience Level $\ge 5$ dipping a long sword into a fountain have a 1/6 chance per dip of forging Excalibur (+1d10 slashing damage, auto-searching for secret doors, and life drain immunity).
    - Safe dipping requires `obs.hero.hp_frac >= 0.85` to survive rare water demon spawns.
    - **Fountain Dipping Mechanism**: Dipping a weapon into a fountain strictly requires **standing directly on the fountain tile** (`obs.dungeon.standing_on_fountain`). In NetHack, executing `#dip` from an adjacent tile prompts for an inventory container to dip *into*, causing the game to reject the command with `"That is a silly thing to dip..."`. `step_to_fountain` steps directly onto `closest_fountain_pos`, and `dip_excalibur` issues `[DIP, weapon_slot]`, with `_dismiss_more` auto-confirming `"dip into the fountain? [yn]"`.

16. **Wielded Weapon Recognition Grammar (`(weapon in hand)`)**:
    - In NetHack, one-handed wielded weapons use singular `(weapon in hand)`. Only two-handed weapons use `(weapon in hands)`.
    - Inventory parsing must check for `"weapon in hand"` (covering both singular and plural), otherwise one-handed weapons are marked unequipped, causing characters to falsely report `bare hands`.

17. **In-Place Search Stall Prevention & Navigation Exclusion**:
    - When visible frontiers and dead ends are exhausted without discovering stairs, `NetHackAdapter` decays search counters by 10. If `step_to_dead_end` or stagnation decay permits the hero to target `(hero.y, hero.x)` (distance 0), the hero repeatedly searches in place for tens of thousands of turns without exploring other room walls.
    - `step_to_dead_end` must strictly exclude the current coordinate (`step_target_mask[hero.y, hero.x] = False`).
    - The policy must enforce `self.last_searched_pos`: a candidate tile cannot be searched twice consecutively without navigating to a different coordinate first.

18. **Floor Transition State Resets**:
    - Depth transitions must explicitly reset `self.known_fountain_pos = None`, `self.known_altar_pos = None`, and clear `self.elbereth_positions` to prevent carrying stale coordinates across dungeon levels.

19. **Item Naming Dialog Auto-Dismissal (`Call a ...` / ESC)**:
    - When picking up or identifying items, NetHack can prompt `Call a white potion: ` or `What do you want to call...`.
    - Directional navigation keys enter characters into the text-entry buffer instead of moving, consuming 0 turns and causing NLE to abort with `StepStatus.ABORTED` after 2,500 consecutive 0-turn steps.
    - `NetHackAdapter._dismiss_more` intercepts `call a `, `call the `, and `what do you want to call` and automatically dismisses them with ESC (`\x1b`).

20. **Solver Fallback Closed Door Navigation & Poison Gating**:
    - High-level tactical solvers (`harvest_poison_res`, `test_altar_buc`) execute fallback navigation when no active target corpse or altar exists on the floor.
    - If the hero encounters a closed or locked door during solver fallback, navigating directly to distant frontiers or dead ends causes pathfinding stalls or door bumping.
    - Solver fallbacks must explicitly check `elif obs_prev.dungeon.adjacent_closed_door: return self.step(Action(name="open_door"))` before dispatching to `step_to_frontier` or `step_to_dead_end`.
    - Policies must strictly gate calling `harvest_poison_res()` with `obs.dungeon.can_harvest_poison` so the solver is never invoked when no eligible poison corpses are present.

21. **Passive & Exploding Hazard Discrimination (`adjacent_floating_eye`, `adjacent_gas_spore`)**:
    - Gas spores explode in melee, inflicting lethal 4d6 area blast damage. Floating eyes inflict 50-turn passive paralysis when struck in melee, allowing nearby monsters to beat the hero to death.
    - `NetHackAdapter.step(Action(name="melee_attack_hostile"))` strictly inspects per-monster glyph species and refuses to target floating eyes or gas spores in melee. If no safe target is adjacent, it steps away or utilizes ranged missiles.

22. **Elbereth Immunity & Humanoid Combat Tactics (`hostile_ignores_elbereth`)**:
    - Orcs (Uruk-hai, hill orcs, mordor orcs), elves, humans, guards, and shopkeepers ignore the Elbereth ward entirely.
    - When standing on Elbereth against enemies with `obs.combat.hostile_ignores_elbereth == True`, yielding `wait()` causes the hero to be executed without resisting.
    - Policies must detect `obs.combat.hostile_ignores_elbereth` and actively engage in melee (or retreat to 1-tile corridor chokepoints) rather than waiting on the ward.

23. **Floor Loot Scooping & AC Scaling (`step_to_loot`, `has_nearby_loot`)**:
    - Slain monsters drop armor, helmets, boots, cloaks, and wands on their death tile. Because melee combat happens at distance 1 and exploration routes to distant frontiers, heroes previously never stepped on dropped monster loot, leaving AC stuck at 5–6 for whole episodes.
    - `obs.spatial.has_nearby_loot` detects dropped items within distance $\le 4$ outside shops, and `step_to_loot` steps directly onto them so NLE's autopickup collects them. This immediately provides armor for `wear_armor()` to drive AC down towards negative values.

24. **Offensive Wand Zapping (`zap_offensive_wand`, `has_offensive_wand`)**:
    - Heroes collect offensive wands (striking, fire, cold, sleep, lightning, magic missile).
    - `zap_offensive_wand()` targets the closest hostile at distance $\ge 2$, safely eliminating high-speed pests and passive hazards (floating eyes, gas spores) without taking melee damage or paralysis.

25. **Emergency Teleportation Panic Escape (`read_scroll_teleport`, `zap_wand_teleport`, `has_panic_escape`)**:
    - When cornered, surrounded by multiple attackers, or critically injured ($< 25\%$ HP) with prayer on timeout, normal retreats fail.
26. **Ground-Truth NetHack Scoring (`blstats[nh.NLE_BL_SCORE]`)**:
    - NetHack tracks official game score in `blstats[nh.NLE_BL_SCORE]` (index 9), computing $2 \times \text{XP} + \text{Gold} + 50 \times (\text{Deepest DL} - 1)$.
    - `NetHackAdapter` extracts this into `obs.hero.score` and logs it directly to DuckDB episodes, accurately capturing combat experience points.

27. **Parallel Multiprocessing Episode Evaluation (`--workers`)**:
    - High-turn episodes (25k max turns) previously took 4–5 minutes per generation on a single core.
    - `scripts.run_synthesis` supports `--workers N` (default 10), utilizing `multiprocessing.Pool` to run episodes concurrently across multi-core CPUs.
    - Cuts batch evaluation time from ~4.5 minutes down to ~35 seconds (8x speedup) with partitioned telemetry and atomic DuckDB consolidation.

28. **Loot Scooping Ping-Pong Loop Protection (`self.looted_tiles`)**:
    - Dropped monster weapons and ammunition (daggers, arrows, darts) are not picked up by NLE's autopickup. Stepping off the tile immediately re-triggered `has_nearby_loot == True`, trapping heroes in an endless 2-tile oscillation for up to 16,900 turns until death on DL1.
    - `NetHackAdapter` tracks `self.looted_tiles` (cleared on reset and floor change), adds tiles to `self.looted_tiles` when reached, and restricts `loot_chars` strictly to autopicked items (`[`, `!`, `?`, `/`, `=`, `$`, `%`).

29. **Fountain Vanishing Dipping Shield (`fountain_vanished`)**:
    - When a fountain disappears or dries up (`"The fountain disappears!"`), hero's position was not cleared from `self.known_fountain_pos`, causing consecutive `#dip` commands that prompted `"What do you want to dip ... into?"`.
    - `_extract_obs` checks for fountain disappearance and clears `self.known_fountain_pos = None`, and `_dismiss_more` intercepts `"what do you want to dip"` with ESC.

30. **Interactive Text & Item Dialog ESC Auto-Dismissal**:
    - Missing prompt checks in `_dismiss_more` previously hung NLE for 2,500 consecutive 0-turn steps (`"Call an emerald potion"`, `"What do you want to eat"`, `"What do you want to throw"`). Expanded `_dismiss_more` with comprehensive ESC dismissals.

31. **Numba JIT Disk Caching & Vector Dead End Discovery**:
    - JIT-compiled `_compute_dead_ends_mask_kernel` with `cache=True` in `SpatialEngine`, cutting dead end mask computation from ~0.55s down to 0.08s (7x speedup, ~58 µs/call). Pre-warmed via `SpatialEngine.warmup()`.

32. **Granular Mortality & Progression Telemetry**:
    - Added `killer`, `ac_at_death`, `hp_at_death`, `max_hp_at_death`, and `excalibur_forged` to episode schema and DuckDB migrations.

33. **Intermediate Multi-Key Keystroke Prompt Preservation (`is_intermediate`)**:
    - When executing multi-key actions like `eat_carried_food` (`[e, slot]`), `quaff_healing` (`[q, slot]`), `wear_armor` (`[W, slot]`), `zap_offensive_wand` (`[z, slot, dir]`), or `throw_dagger` (`[t, slot, dir]`), NetHack displays interactive item selection prompts (`"What do you want to eat?"`, `"What do you want to wear?"`).
    - If `_dismiss_more` dismisses these prompts with ESC between keystrokes, the action is cancelled with `"Never mind."`, consuming 0 turns and causing the hero to starve or loop endlessly.
    - `_step_sequence` flags `is_intermediate = (idx < len(action_indices) - 1)`, preserving item selection prompts so the subsequent key selection is delivered.

34. **Directional Step Verification for Impassable Obstacle Learning**:
    - When consecutive 0-turn steps occur, only genuine directional movement actions (`is_step_direction`) should register the candidate tile in `self.blocked_tiles`. Non-directional actions (inventory queries, failed eating, search) must not block adjacent walkable tiles.

35. **Episode Mortality Discrimination (HP <= 0 vs Zero-Progress Abort vs MaxTurnsReached)**:
    - Real fatalities (HP <= 0) must always be classified by their lethal cause (combat, starvation, poison), even if NLE triggers termination. Zero-progress aborts (`StepStatus.ABORTED`) must be logged as `Aborted / ZeroProgress` rather than `MaxTurnsReached`, reserving `MaxTurnsReached` exclusively for runs reaching `max_turns` while alive.

36. **No-Hostile Tactical Fallback Protection (`step_away_from_hostile`)**:
    - When `step_away_from_hostile()` is invoked without hostiles in field of view (`obs.combat.closest_hostile_pos is None`), falling through to Elbereth engraving and waiting traps heroes in 4,000-turn dead-end stagnation loops.
    - `NetHackAdapter` must immediately fallback to `step_to_stairs_down`, `step_to_frontier`, `search` (if standing on a dead end/perimeter candidate), or `step_to_dead_end`. It must strictly avoid engraving or waiting when no hostiles exist.

37. **Full-Core Worker Parallelism & I/O Buffering**:
    - Host CPU capacity is 22 cores. Episode evaluation runs with `--workers 20` to execute the full 20-episode batch simultaneously in a single wave.
    - Parquet telemetry logging is buffered to `flush_interval=5000` to eliminate SSD write-lock contention across concurrent processes.

38. **Tactical Retreat Door Breaching & Open Floor Preference (`step_away_from_hostile`)**:
    - When stepping away or retreating from hostiles, directional evaluation must strictly prefer open walkable floor tiles over closed or locked doors.
    - If a closed or locked door must be breached during retreat, `NetHackAdapter` must dispatch via `self._step_or_breach(obs_prev, dy, dx)` to open or kick the door open rather than issuing raw `step_direction` (which bumps into locked doors and aborts after 2,500 consecutive 0-turn steps).

39. **`standing_on_elbereth` Observation Timing Invariant**:
    - Pre-populate `self.elbereth_positions.add((obs_prev.hero.y, obs_prev.hero.x))` before executing the keystroke sequence in `engrave_dust_elbereth` so `_extract_obs` marks `obs.combat.standing_on_elbereth = True` immediately on the returned observation, preventing double-engraving wipes.

40. **Truthful Elbereth Ward Durability & Monster Hit Discrimination**:
    - NetHack monster melee and ranged strikes do not erase dust Elbereth. `self.elbereth_positions` must only be discarded upon actual smudge/wipe/erasure messages (`"wipe out the message"`, `"wiped out"`, `"rubbed out"`, `"scuffed"`, `"erased"`, `"fades"`), preventing flip-flop re-engraving loops during battle.

41. **Comprehensive Elbereth-Immune Species Discrimination (`IGNORES_ELBERETH_SPECIES`)**:
    - Goblins, hobgoblins, gnomes, dwarves, ogres, trolls, giants, zombies, mummies, and vampires ignore Elbereth or attack with ranged weapons. Checking both adjacent hostiles and closest hostile against `IGNORES_ELBERETH_SPECIES` prevents wasting turns attempting to engrave dust when pursued by immune hostiles.

42. **`signal`-Based Policy Dry-Run Timeout vs `ThreadPoolExecutor` Deadlock**:
    - In Python, `ThreadPoolExecutor.__exit__` calls `self.shutdown(wait=True)`. If a candidate policy contains a zero-yield infinite loop, the submitted worker thread runs forever, deadlocking `executor.__exit__` and freezing the synthesis process indefinitely while consuming 100% CPU.
    - `AuthorAgent` uses `signal.setitimer(signal.ITIMER_REAL, timeout)` with SIGALRM in the main thread (and process termination fallback), which interrupts the infinite loop bytecode immediately by raising `TimeoutError` without lingering threads or shutdown deadlocks.
    - `test_scenarios` explicitly flags `adjacent_floating_eye=True` in `floating_eye_combat` to prevent false invariant rejections.

43. **Four-Layer Anti-Infinite-Loop Execution & Worker Protection Architecture**:
    - **Layer 1 (AST LoopGuardTransformer & ExecutionGuard)**: Automatically instruments every `while` and `for` loop in compiled policies with `_guard.tick()`. If any loop executes >2,000 iterations without yielding an action, `RuntimeError` is raised in $< 1\text{ms}$.
    - **Layer 2 (PolicyRunner Graceful Fallback)**: Catches `Infinite loop detected` `RuntimeError` in `send(obs)` and `create_runner()`, safely falling back to `Action(name="wait")` rather than crashing or hanging the environment.
    - **Layer 3 (Episode Wall-Clock Ceiling)**: Sets `MAX_EPISODE_WALL_SEC = 90.0` in `_run_single_episode_worker`, breaking out with `EpisodeWallTimeout` if unexpected C-level NLE lockups occur.
    - **Layer 4 (Worker Pool Batch Ceiling)**: Dispatches `mp.Pool` via `map_async` with a 180s ceiling, automatically executing `pool.terminate()` and `pool.join()` to eliminate zombie worker processes if a batch stalls.

44. **Gnomish Mines Avoidance & Main Dungeon Staircase Discrimination (`dnum == 2`, `BRANCH_NAMES`)**:
    - NetHack internal branch IDs (`dungeon.def`) assign `dnum = 0` (Dungeons of Doom), `dnum = 1` (Gehennom), `dnum = 2` (Gnomish Mines), `dnum = 3` (The Quest), `dnum = 4` (Sokoban). Fixed `BRANCH_NAMES` so `dnum == 2` correctly maps to `"mines"` (previously mismapped to `"quest"`).
    - When descending from DL 2–4 of Dungeons of Doom, if the destination floor is the Gnomish Mines (`dnum == 2`), `NetHackAdapter` dynamically marks that staircase in `self.mines_stairs_positions` and filters it out of `self.known_stairs_down`.
    - The policy immediately yields `ascend()` on `<` back to the Dungeons of Doom, allowing the hero to locate and descend the true main dungeon staircase to DL 5–20 without entering the lethal dark Mines.

45. **Persistent Level Feature State & FOV Caching Invariant**:
    - Staircases, fountains, and altars are static level fixtures. In NetHack, when the hero walks into an adjacent corridor or room, fixtures leave the active FOV `glyphs` and `chars`.
    - `NetHackAdapter` must strictly preserve `self.known_stairs_down` across turns once discovered on a level. Setting `self.known_stairs_down = None` when not currently visible in FOV causes the hero to forget the exit the moment they walk away, resulting in endless search loops and stalling on DL1 for 10,000+ turns.
    - Only level transitions or explicit branch staircase pruning (e.g. entering the Mines) may invalidate or clear `self.known_stairs_down`.

46. **Trap-Door Arrival vs Genuine Staircase Discrimination & 0-Turn Ascend Protection**:
    - Characters can descend floors involuntarily by stepping on trap doors (`^`) or holes, arriving on arbitrary floor tiles (`.`), NOT stairs up (`<`).
    - `NetHackAdapter` previously assumed all floor transitions arrive on stairs up, setting `self.known_stairs_up = (y, x)` on turn 0. When in the Mines after a trap door fall, the policy yielded `ascend()` on open floor, which output `"You can't go up here."` (0 turns) and triggered NLE abort after 2,500 consecutive 0-turn steps.
    - `NetHackAdapter` now records `had_explicit_descent` (only true if `descend` was issued), invalidates `known_stairs_up` upon trap door messages or `"You can't go up here."`, and shields `ascend` by falling back to `step_to_frontier` if consecutive 0-turn steps occur.

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
