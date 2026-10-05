# LOX Agent Operational Guide (`AGENTS.md`)

Welcome to LOX, an autonomous, empirical policy synthesis engine for NetHack.
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
- **Mandatory End-of-Campaign Post-Mortem & Evolution Protocol**: At the conclusion of **EVERY** campaign run, the agent MUST ALWAYS perform a rigorous empirical post-mortem analysis:
  1. **DuckDB Autopsy**: Query `episodes` and `ticks` to categorize all mortality causes, killer species, turn distributions, AC at death, and action frequencies across all generations.
  2. **Unwanted Behavior Elimination**: Pinpoint pathological behaviors, loops, oscillations, or tactical stalls (e.g. 2-tile ping-pongs, unequipped armor, premature prayer, floating eye melee attacks, starvation deadlocks) and implement concrete code fixes.
  3. **Progression Engineering**: Introduce architectural, harness, and policy enhancements targeted at advancing deeper into the 7 canonical NetHack ascension phases.
  4. **Documentation & Test Verification**: Verify all unit tests pass with zero regressions and synchronize living documentation (`AGENTS.md`, `TODO.md`, `README.md`) before launching the next campaign.

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

### 3.1 Policy Architecture & Anti-Loop Defense
1. **Python Generator Paradigm & Subroutine Protocol**: Policies are Python classes yielding actions (`obs = yield action`). Subroutines MUST be called via `obs = yield from self.subroutine(obs)`.
2. **The Zero-Yield Busy Loop Trap**: Every subroutine called via `yield from` MUST yield an action on every code branch before returning, OR the caller must avoid `continue` without yielding. Returning without yielding (e.g. `return obs`) followed by caller `continue` spins CPU at 100% without advancing the environment clock.
3. **Four-Layer Anti-Loop Defense Architecture**:
   - **Layer 1 (AST `LoopGuardTransformer`)**: Instruments every `while` and `for` loop with `_guard.tick()`. If any loop exceeds 2,000 iterations without yielding, raises `RuntimeError("Infinite loop detected")` in $< 1\text{ms}$.
   - **Layer 2 (`PolicyRunner` Fallback)**: Catches `RuntimeError`, safely falling back to `Action(name="wait")` (`.`) to advance the game turn.
   - **Layer 3 (Episode Wall-Clock Ceiling)**: `MAX_EPISODE_WALL_SEC = 90.0` in `_run_single_episode_worker` breaks out with `EpisodeWallTimeout` on C-level NLE hangs.
   - **Layer 4 (Worker Pool Batch Ceiling)**: Dispatches `mp.Pool` via `map_async` with a 180s batch ceiling; executes `pool.terminate()` and `pool.join()` to eliminate zombie processes if a batch hangs.
4. **`signal`-Based Policy Dry-Run Timeout**: `AuthorAgent` runs candidate policies across multi-scenario test states (`normal`, `hungry`, `combat`, `floating_eye_combat`) with `signal.setitimer(signal.ITIMER_REAL, 1.0)`. This interrupts zero-yield loops via `SIGALRM` / `TimeoutError` without `ThreadPoolExecutor.shutdown(wait=True)` deadlocks.

### 3.2 Dialog Interception, Prompt Auto-Dismissal & Keystroke Protocols
5. **Prompt Auto-Dismissal Shields (`[yn]`, `[ynq]`)**: Directional keys answer `n` (No) to prompts and consume 0 turns. `NetHackAdapter._dismiss_more()` automatically intercepts confirmation prompts (`"Really attack?"`, `"Really quit?"`, `"eat it? [ynq]"`) and auto-answers `'n'` (or `'y'` for floor corpses/fountain dipping), tracking peaceful positions per level. Exposes `obs.combat.adjacent_peaceful` and skips peaceful entities in hostile queries.
6. **Text-Entry & Item Naming Dialog Interception (ESC `\x1b`)**: In Minetown (Depths 5–9) or when picking up items, prompts like `"who are you"`, `"what is your name"`, `"call a "`, `"call the "`, or `"what do you want to call"` intercept directional keys as text input (0 turns). `_dismiss_more` immediately dismisses these with ESC (`\x1b`), preventing 2,500-step zero-turn aborts.
7. **Intermediate Multi-Key Keystroke Preservation (`is_intermediate`)**: Multi-key commands (`eat_carried_food` `[e, slot]`, `quaff_healing` `[q, slot]`, `wear_armor` `[W, slot]`, `zap_offensive_wand` `[z, slot, dir]`, `throw_dagger` `[t, slot, dir]`) open intermediate item-selection prompts (`"What do you want to eat?"`). `_step_sequence` flags `is_intermediate = (idx < len(seq) - 1)` so `_dismiss_more` does NOT press ESC during intermediate prompts, allowing subsequent keys to register cleanly.
8. **Universal Consecutive Zero-Turn Circuit Breaker**: If any sequence of commands generates `consecutive_zero_turns >= 4`, `NetHackAdapter.step()` unconditionally forces `Action(name="wait")` (`.`), advancing the NetHack turn clock and completely eliminating zero-progress aborts (`StepStatus.ABORTED`).

### 3.3 Spatial Navigation, Topological Memory & Level Topology
9. **Full-Floor Persistent Topological Memory & Level Fixtures**: Outside active line-of-sight FOV, NetHack zeroes glyphs in `raw_obs`. `NetHackAdapter` preserves unbroken connectivity across the entire floor by maintaining `self.visited` and discovered clean terrain (`self.known_chars` for `#`, `.`, `<`, `>`, `_`, `{`, `+`) in `walkable` and `walkable_nav`. Static level fixtures (`known_stairs_down`, `known_stairs_up`, `known_fountain_pos`, `known_altar_pos`) are strictly preserved when leaving FOV until genuine level transitions or branch staircase pruning.
10. **Branch Discrimination & Vertical Transit**:
   - **Gnomish Mines Avoidance (`dnum == 2`)**: In NetHack `dungeon.def`, `dnum = 0` (Dungeons of Doom), `dnum = 1` (Gehennom), `dnum = 2` (Gnomish Mines), `dnum = 3` (The Quest), `dnum = 4` (Sokoban). When descending into the dark Mines (`dnum == 2`), the adapter flags the stair in `mines_stairs_positions`, removes it from `known_stairs_down`, and policies immediately yield `ascend()` on `<` back to Dungeons of Doom to continue downward toward Depth 20.
   - **Trap-Door Fall vs Stair Arrival**: Falling through trap doors or holes (`^`) lands the hero on open floor (`.`), NOT stairs up (`<`). `NetHackAdapter` tracks `had_explicit_descent`, invalidates `known_stairs_up` upon trap door messages or `"You can't go up here."`, and shields `ascend()` from 0-turn loops by falling back to `step_to_frontier()`.
   - **Staircase Combat Interception**: If `obs.spatial.standing_on_stairs_down and not obs.status.is_levitating`, policies prioritize `descend()` immediately (even during combat), taking the 1-turn descent to escape to the next depth.
11. **Doorway Breaching, Cardinal Movement & Dynamic Obstacle Learning**:
   - Cardinal movement only `((-1, 0), (1, 0), (0, -1), (0, 1))`. Distinguishes closed doors from floor spellbooks (`+`) using CMAP glyphs (`nh.GLYPH_CMAP_OFF + 15` and `+ 16`). Breaches locked doors with kicking (max 6 kicks); unbreachable doors (or locked doors inside shops) are added to `blocked_tiles` and masked out of chokepoints.
   - Impassable terrain messages (`"cannot pass through the bars"`, `"It's a wall"`, `"It's solid stone"`) add coordinates to `self.blocked_tiles` ONLY when triggered by genuine directional movement (`is_step_direction`).
   - Tactical Retreat Door Breaching: `step_away_from_hostile()` strictly prefers open walkable tiles over closed/locked doors. Any retreat path requiring a closed door routes through `_step_or_breach()` to kick it open rather than bumping into locked doors.
   - Solver Door Navigation: Solvers (`harvest_poison_res`, `test_altar_buc`) must check `obs_prev.dungeon.adjacent_closed_door` to open/breach doors before falling back to distant frontiers.
12. **Stride-2 Secret Door Search & Numba Acceleration**:
   - Room perimeters are searched using a stride-2 checkerboard `(cy + cx) % 2 == 0` plus corners (`adj_wall >= 2`), providing 100% geometric wall coverage while halving search turns. `step_to_dead_end` excludes the current position (`step_target_mask[hero.y, hero.x] = False`), and policies enforce `self.last_searched_pos` to prohibit consecutive searches on the same tile without movement. Stagnation decays search counts by 10 when frontiers stall.
   - Numba JIT Disk-Cached Spatial Kernels: `_compute_dead_ends_mask_kernel` compiles with `cache=True` and is pre-warmed via `SpatialEngine.warmup()`, executing in ~58 µs/call (7x speedup).
   - No-Hostile Fallback: If `step_away_from_hostile()` is invoked without hostiles in FOV, it immediately falls back to `step_to_stairs_down`, `step_to_frontier`, or `step_to_dead_end`, never engraving or waiting in place.

### 3.4 Combat Tactics, Hazard Discrimination & Elbereth Sanctuary
13. **Dust Elbereth Sanctuary & Species Discrimination**:
   - **Warding Mechanics**: Writing `"Elbereth"` in dust with bare fingers (`E` $\to$ `-` $\to$ `Elbereth\r`) creates a 100% ward against non-humanoids (ants, bees, spiders, wolves, animals). Coordinates are pre-added to `self.elbereth_positions` before sequence execution so `obs.combat.standing_on_elbereth` is immediately True (preventing re-engraving wipes). Monster attacks do NOT erase dust Elbereth; positions are discarded ONLY upon genuine smudge/wipe messages (`"wipe out"`, `"scuffed"`, `"erased"`).
   - **Elbereth-Immune Species Discrimination (`IGNORES_ELBERETH_SPECIES`)**: Orcs, elves, humans, goblins, hobgoblins, gnomes, dwarves, ogres, trolls, giants, zombies, mummies, and vampires ignore Elbereth or attack with ranged weapons. When facing immune hostiles (`hostile_ignores_elbereth == True`), policies MUST engage in melee or retreat to 1-tile corridor chokepoints rather than waiting passively on the ward.
14. **Passive & Exploding Hazard Discrimination & Deadlock Circuit Breaker**:
   - **Hazard Avoidance**: Floating eyes inflict 50-turn passive paralysis when hit in melee; gas spores inflict lethal 4d6 area blast damage; molds/jellies inflict passive retaliatory damage. `_build_walkable_nav` strictly masks out all passive hazards (`floating eye`, `gas spore`, molds, jellies, exploding spheres) so pathfinding navigation never routes through them.
   - **Active Hostile Gating & Bypass**: Immobile passive hazards at distance $\ge 2$ do not hunt or attack the hero. If lacking ranged weapons, `has_active_hostile == False` allows `run()` to bypass `handle_combat` completely, continuing exploration as `walkable_nav` routes around them safely without melee paralysis or starvation stalemates.
   - **Cornered Passive Hazard Deadlock Circuit Breaker**: When cornered in a dead end next to an immobile passive hazard with no retreat route (`best_tile == None`), yielding `wait()` turn after turn results in 20,000-turn starvation death locks. `NetHackAdapter.step_away_from_hostile()` tracks consecutive passive waits (`self.consecutive_passive_waits >= 2`). After 2 waits, the adapter autonomously breaks the deadlock by: (1) stepping to any non-hazard walkable adjacent tile, (2) throwing missiles or zapping offensive wands, (3) searching for secret dead-end exits, or (4) executing an emergency strike (better to risk explosion/paralysis than 100% certain starvation).
   - **Safe Ranged & Melee Elimination**: Throwing missiles (daggers, darts, arrows, rocks) at adjacent floating eyes (`obs.combat.adjacent_floating_eye`) is 100% safe at distance 1 without triggering paralysis. When an active attacker is present alongside a passive hazard, `has_safe_melee_target` engages the active attacker in melee while skipping the passive hazard.
15. **High-Speed Predators & Swarm Corridor Defense**:
   - **Fast Predator Discrimination (`is_fast_dangerous`)**: Dynamic NLE `pm.mmove > 12` check plus foxes (speed 15), coyotes, jaguars, soldier ants / killer bees (speed 18), and giant bats (speed 22) outpace hero speed (12).
   - **Open-Room Non-Retreat Rule**: Stepping away from an adjacent fast predator in open rooms gives them free hits while dealing 0 damage. When adjacent, policies MUST engage in melee (`melee_attack_hostile`) if HP $> 40\%$ (kills them in 1–2 hits) or engrave Elbereth if low HP. Tactical retreat to chokepoints (`step_to_chokepoint`) is reserved strictly for when the predator is at distance $\ge 2$ in an open room.
   - **Swarm Defense**: When facing multiple fast attackers or when surrounded (`obs.combat.is_fast_dangerous and (obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2)`), policies engrave Elbereth immediately. Insects flee in terror, allowing the hero to quaff healing or retreat to 1-tile corridor chokepoints (`step_to_chokepoint`) to fight 1v1.
16. **Tactical Ranged Weapons, Emergency Healing & Panic Escapes**:
   - **Ranged Harassment**: Starting daggers or dropped missiles thrown at distance $\ge 2$ soften high-speed pests before melee contact. Offensive wands (striking, magic missile, fire, cold, sleep, lightning) are zapped at distance $\ge 2$ via `zap_offensive_wand`.
   - **In-Combat Emergency Healing**: Quaffing healing potions takes 1 turn and restores 10–20 HP; triggered at `hp_frac < 0.50`.
   - **Panic Escapes**: Reading scrolls of teleportation or zapping wands of teleportation (`has_panic_escape`) triggers when surrounded or at `hp_frac < 0.25` with prayer on timeout.
17. **Peaceful NPC Non-Aggression**:
   - Vault Guards (`guard`), priests (`priest`, `priestess`), shopkeepers, and town watchmen are neutral/peaceful until provoked. Attacking or throwing missiles at a Level 12 Vault Guard turns them hostile and causes instant hero death.
   - `is_peaceful_species` explicitly includes `"guard"`, `"priest"`, `"priestess"`, `"shopkeeper"`, `"watchman"`, `"oracle"`. Missiles (`throw_dagger`) and wands (`zap_offensive_wand`) are hard-shielded against targeting peaceful positions (`target_pos in self.peaceful_positions`), and policies yield retreat rather than attacking.

### 3.5 Inventory, Equipment, Nutrition & Artifact Scaling
18. **Autopickup, Armor Equipping & Floor Loot Scooping**:
   - **Autopickup & Equipping (`wear_armor`)**: NetHack initialized with `options=("autopickup", "pickup_types:?!/%=[$")`. Unworn dropped armor is equipped during peaceful exploration (`wear_armor()`), driving AC toward negative values. Failed armor slots are tracked in `failed_wear_slots` and excluded from `has_unworn_armor` to prevent redundant wear loops.
   - **Floor Loot Scooping (`step_to_loot`)**: Dropped monster equipment within distance $\le 4$ is scooped via `step_to_loot`. Reached tiles are added to `self.looted_tiles` (cleared on floor change) and `loot_chars` is restricted strictly to autopicked types (`[`, `!`, `?`, `/`, `=`, `$`, `%`), eliminating 2-tile oscillation loops over dropped weapons/ammo.
   - **Weapon Grammar Recognition**: Tracks both one-handed `(weapon in hand)` (singular) and two-handed `(weapon in hands)` (plural); checking only plural previously caused one-handed weapons to falsely report as bare hands.
19. **Nutrition Harvesting, Rotten Corpse Shield & Proactive Eating**:
   - **Corpse Safety & Harvesting**: NLE body glyph queries (`glyph_is_body`, `permonst`) verify species and poison/deadly status. `_dismiss_more` auto-confirms `"eat it? [ynq]"` with `'y'`. Fresh corpses are consumed right after combat at `hunger_state < 3`, preserving carried food rations for deep-level hunger emergencies.
   - **Rotten Corpse Shield**: In NetHack, corpses rot within 50 turns in the backpack. `has_food`, `get_food_slot()`, and `food_count` strictly exclude corpses (`"corpse" not in it.name.lower()`), reserving consumption strictly for non-perishable packaged food (rations, wafers, fruit) to prevent lethal food poisoning.
   - **Proactive & Weakness Eating**: Carried food is consumed in 1 turn (`eat_carried_food`) during peaceful exploration or inside `handle_combat` when hungry (`hunger_state >= 2`, "Weak") while on Elbereth, adjacent to passive hazards, or away from enemies to prevent 30-turn unconscious fainting blackouts. Strained encumbrance blocks eating; `eat_carried_food` drops unworn heavy armor before consuming rations if encumbered.
   - **Major Trouble Divine Feeding**: When packaged food is exhausted and `hunger_state >= 3` (or `hp_frac < 0.15`), prayer grants safe divine feeding or full HP restoration.
20. **Universal Throwable Missile Ammunition**:
   - Goblins and orcs drop dozens of darts, arrows, and rocks across early dungeon floors. `has_daggers`, `get_dagger_slot()`, and `dagger_count` recognize all throwable missiles (daggers, darts, arrows, rocks, shuriken, spears, javelins). This ensures the hero never runs out of ranged ammo, allowing continuous ranged elimination of passive hazards (floating eyes, gas spores) and fast speedsters from distance $\ge 2$ without getting stuck in idle wait stalemates.
21. **Excalibur Artifact Scaling & Fountain Navigation**:
   - **Artifact Forging**: Lawful Valkyries XL $\ge 5$ dipping a long sword into a fountain have a 1/6 chance per dip of forging Excalibur (+1d10 damage, secret door searching, life drain immunity).
   - **Room-Wide Navigation**: Gating `step_to_fountain()` by `obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain` allows heroes to navigate across open rooms to forge Excalibur.
   - **Fountain Hero Occlusion**: NetHack displays `@` at `chars[y, x]` when standing on a fountain. `_extract_obs` must NOT wipe `known_fountain_pos` due to `@` character occlusion, preserving `standing_on_fountain = True`.
   - **Confirmation Prompt & Clearance**: Dipping prompts `"Dip a ... into the fountain? [yn] (n)"`; `_dismiss_more` intercepts `"dip" in msg and "fountain" in msg` and auto-answers `'y'` so long swords are safely dipped for Excalibur without ping-pong cancellation. `known_fountain_pos` is cleared upon `"The fountain disappears!"`.

### 3.6 Telemetry, Parallel Evaluation & System Resilience
22. **Parallel Multiprocessing Worker Scaling & Vectorized Glyph LUTs**:
   - **Single-Wave Worker Scaling**: Host has 22 CPU cores. Running with `--workers 20` executes all 20 episodes concurrently in a single wave, cutting batch evaluation time from ~4.5 minutes down to ~35 seconds (8x speedup). Parquet telemetry writes are buffered with `flush_interval=5000` to prevent SSD write contention.
   - **Vectorized Glyph Lookup Tables (11x Speedup)**: Precomputed 6KB module-level boolean lookup tables (`GLYPH_IS_MON_HOSTILE_LUT`, `GLYPH_IS_BODY_LUT`, `GLYPH_IS_PASSIVE_HAZARD_LUT`, `GLYPH_IS_FAST_LUT`, `GLYPH_IS_PEACEFUL_SPECIES_LUT`) for all 5,976 glyphs and C-level byte message decoding accelerate turn execution to **826+ steps/s (11x speedup)**, completing 6,000+ turn episodes in ~7 seconds.
23. **Ground-Truth Scoring, Truthful Mortality & Rolling Attacker Attribution**:
   - **Ground-Truth Scoring**: Official score is extracted from `blstats[nh.NLE_BL_SCORE]` ($2\text{XP} + \text{Gold} + 50(\text{DL}-1)$). Ephemeral telemetry tracks `killer`, `ac_at_death`, `hp_at_death`, `max_hp_at_death`, and `excalibur_forged` into DuckDB.
   - **Truthful Mortality Classification**: Real fatalities (`hp <= 0`) are classified by cause (combat, starvation, poison); zero-progress aborts are classified as `Aborted / ZeroProgress`; `MaxTurnsReached` is strictly reserved for runs alive at the 25k turn cap.
   - **Rolling Message History for Fatal Attacker Attribution**: When a hero dies, `obs.message` is frequently `"You die...  --More--"` or blank. Scanning `recorder.turns[-6:]` in reverse for hit verbs (`" bites!"`, `" hits!"`, `" stings!"`) and prioritizing adjacent monsters resolves true fatal attribution, eliminating false attribution to distant passive hazards (floating eyes, gas spores).
24. **Divine Favor Prayer Cooldown & Prompt Shielding**:
   - **Safe Divine Favor Prayer Threshold (850 Turns)**: In NetHack, successful prayer sets divine timeout to $300 + \text{rn2}(500) \le 799$ turns. Praying at $\le 800$ turns angers the deity ("Tyr is displeased"), resulting in divine smiting, paralysis, or lightning. In-combat and major trouble prayer must strictly enforce `turn - last_prayer_turn >= 850` to guarantee 100% safe divine feeding or full HP restoration.
   - **Premature Prayer Confirmation Interception**: When prayer timeout has not elapsed, NetHack prompts `"Are you sure you want to pray? [yn] (n)"`. `_dismiss_more` intercepts this confirmation and auto-answers `'n'`, safely aborting the premature prayer and completely eliminating divine wrath. If displeased messages occur, `last_prayer_turn` is penalized with $+500$ turns.
25. **Non-Rotting Corpse Exemption & Proactive Nutrition**:
   - In NetHack, lichen corpses and lizard corpses **never rot**. `has_food`, `get_food_slot()`, and `food_count` explicitly treat `"lichen corpse"` and `"lizard corpse"` as non-perishable safe food while maintaining the 50-turn rot shield against all other corpses. Carried food is consumed proactively as soon as `hunger_state >= 1` ("Hungry") outside combat, rather than delaying until "Weak".
26. **AST Method-Level Replacement & LLM Synthesis Acceleration**:
   - `AuthorAgent.splice_policy_methods()` parses candidate code via Python AST, allowing the model to output only the specific modified method(s) (e.g. `def handle_combat(...)`). Splicing merges the updated method into `class Agent` while keeping working subroutines intact, reducing LLM completion tokens by 4x–5x and synthesis latency from ~80s to ~15s without text diff brittleness.
27. **Exploding Yellow & Black Light Hazard Recognition**:
   - Yellow and black lights explode on contact, causing fatal blast damage or 100-turn blinding. Added to `GLYPH_IS_PASSIVE_HAZARD_LUT` and `GLYPH_IS_GAS_SPORE` so pathfinding steers around them and combat eliminates them from range.
28. **Absolute Floating Eye Melee Prohibition & Stalemate Breaker**:
   - Attacking a floating eye in melee triggers 70 turns of passive paralysis (`0d70`). Policies are strictly prohibited from yielding `melee_attack_hostile()` against floating eyes under any condition. Cornered heroes with 0 missiles yield `step_away_from_hostile()`, which triggers the adapter's stalemate breaker (lateral stepping, dead-end secret door search, or carried missile throw). Emergency strikes in `step_away_from_hostile` strictly veto targeting floating eyes.
29. **Stair Descent Prioritization & Altar Loop Elimination**:
   - In `run()`, stair descent (`descend()`, `step_to_stairs_down()`) takes priority immediately following essential health/nutrition/equipment upkeep. Altar BUC testing and floor loot scooping are prohibited from delaying stair descent when stairs down are already discovered, eliminating hundreds of wasted turns wandering back and forth on shallow dungeon levels.
30. **Universal Missile Collection & Thrown Ammo Recovery**:
   - NetHack options configured with `pickup_thrown` and `pickup_types:?!/%=[$*` (including `*` for rocks and gems). Rocks (weight 3, unlimited supply) and dropped gems provide endless throwable ammo, eliminating missile exhaustion against passive hazards and fast predators.
31. **Perimeter Wall Searching vs Oscillation Elimination**:
   - In `step_to_dead_end`, if the hero is already adjacent to an unexhausted wall candidate, `search()` is executed on the current tile before stepping away. Candidates in `wall_adj_mask` require `searched_count < 10`, eliminating 2-tile ping-pong oscillations when frontiers are cleared. In `handle_dead_end`, searching runs for 12 turns to ensure the tile exceeds the candidate search threshold and clears from the spatial candidate mask.
32. **Mandatory Post-Mortem Analysis & Iterative Evolution Protocol**:
   - At the conclusion of **EVERY** campaign run, agents are strictly required to perform a comprehensive DuckDB autopsy across all evaluated episodes and ticks. Never launch a new campaign without:
     1. Classifying top fatality causes, killers, turns, and AC distributions.
     2. Diagnosing and correcting unwanted behaviors, stalls, or pathological loops in both the environment adapter/core harness and the policy templates.
     3. Formulating and deploying targeted improvements to push further into the 7 canonical NetHack ascension milestones (early-game AC/poison res $\to$ Sokoban $\to$ Quest $\to$ Castle $\to$ Gehennom $\to$ Sanctum $\to$ Astral Plane).
     4. Verifying zero regressions against unit tests and recording findings in `AGENTS.md` and `TODO.md`.
33. **Mines Evacuation & Staircase Vertical Orientation**:
   - In NetHack, when entering the Gnomish Mines (`dungeon_branch == "mines"` or `dnum == 2`), the hero enters via the up-staircase (`<`). To evacuate back to the Dungeons of Doom, the hero must navigate to `stairs_up` and yield `ascend()`. Inverting this to check `stairs_down` or calling `descend()` drives the hero deeper into the lethal early-game Mines. The harness unconditionally intercepts `descend()` and `step_to_stairs_down()` while in the Mines, redirecting navigation to `stairs_up` and `ascend()`.
34. **Unconditional In-Combat Healing Under 50% HP**:
   - Drinking a potion of (extra/full) healing (`quaff_healing()`) consumes 1 turn and restores 10–20 HP. It must be executed unconditionally whenever `hp_frac < 0.50` and healing potions are available. Gating healing on `not adjacent_hostile` is lethal against fast predators in open rooms where retreat grants enemies free hits.
35. **Mindless Gas Spore Tactical Elimination**:
   - Gas spores are mindless (`M1_MINDLESS`) and explode on contact for 4d6 damage. They completely ignore dust Elbereth. Engraving Elbereth or waiting on Elbereth adjacent to a gas spore guarantees fatal blast damage. Policies must retreat from adjacent gas spores or eliminate them from distance $\ge 2$ using thrown missiles (daggers, rocks) or offensive wands with 0 damage.
36. **Dead-End Navigation & Stagnation Decay Rate Limiting**:
   - `step_to_dead_end()` is strictly a movement primitive and must NEVER return `Action(name="search")`. Searching is executed explicitly by `handle_dead_end()` or when `standing_on_dead_end` is true. `step_to_dead_end()` navigates to unsearched corridor dead ends or wall perimeters (`wall_adj_mask`). Search count decay (`searched_count - 10`) is strictly rate-limited to at most once per 50 turns to prevent resetting search counts every turn, which previously trapped heroes in 2,500-turn in-place search loops on Depth 1–2.
37. **Proactive Divine Feeding at Weakness (`hunger_state >= 3`)**:
   - In NetHack, `HungerState` transitions from `HUNGRY` (2) to `WEAK` (3) before `FAINTING` (4). At `FAINTING`, the hero randomly falls unconscious for 30 turns and cannot act or pray. Emergency prayer for food must trigger at `hunger_state >= 3` when packaged food is exhausted, ensuring divine feeding before the hero falls unconscious.
38. **Floor Corpse Aging Pruning & Safe Corpse Consumption Gating**:
   - In NetHack, non-lichen/non-lizard corpses rot within 50 turns. The adapter strictly prunes floor corpses with `age >= 50` from `self.floor_corpses`. Policies gate floor corpse consumption on `any(c.is_safe for c in obs.corpses)` rather than truthiness of `obs.corpses`, and `handle_corpse_consumption` falls back to `step_to_frontier()` or `step_to_dead_end()` rather than `wait()`. This completely eliminates the 1,000-turn stationary `wait` stall that previously affected 19% of episodes when inedible rotten corpses lingered on the floor.
39. **Offline Wiki Knowledge Retrieval (`data/wiki_index.db`)**:
   - Sub-5ms BM25 SQLite FTS5 index over 3,113 NetHack encyclopedia articles and redirects. Unblocks the LLM AuthorAgent's `query_wiki` tool during generational policy synthesis, supplying accurate domain knowledge (resistances, monster statistics, artifact dipping, Sokoban solutions, prayer mechanics).
40. **Hierarchical True Dead End Prioritization & Hero Occlusion Attribution**:
   - `_compute_true_dead_ends_mask` recognizes any walkable tile with $\le 1$ walkable cardinal neighbors (for both `#` and lit `.`) as a true dead end. In `step_to_dead_end`, reachable true dead ends take absolute Priority 1 over generic room perimeter wall candidates, forcing the hero to traverse corridors and search dead ends across the entire floor. Eliminates the local room search trap that previously killed 17.5% of episodes on DL 1. In `_extract_obs`, `tile_type` checks static level fixtures (`known_fountain_pos`, `known_altar_pos`, `known_stairs_down`, `known_stairs_up`), preventing `@` hero occlusion from wiping feature identities.
41. **Closed-Door Inter-Room Transitioning (`obs.dungeon.has_closed_door`)**:
   - When all open tiles in the current room are visited (`has_unvisited_frontier == False`), policies MUST yield `step_to_closed_door()` before falling back to wall perimeter search (`handle_dead_end`). Omitting closed-door navigation trapped heroes in 1,000+ turn perimeter search loops while unexplored rooms lay directly behind closed doors.
42. **15-Kick Door Breaching Threshold & Shatter/Hurt Message Discrimination**:
   - In NetHack, wooden locked doors can take 7–12 kicks to breach. The adapter previously gave up after 6 kicks and permanently added the door to `blocked_tiles`, cutting off 31% of runs from the remainder of the dungeon. The kick limit is expanded to 15 kicks for wooden doors (`"WHAMMM!!!"`), immediately clearing locked doors from tracking upon breach (`"shatter"`, `"crash"`), and reserving `blocked_tiles` strictly for iron doors/shop doors that injure legs (`"ouch"`, `"hurt"`).
43. **Autopickup Loot Ammunition Consistency**:
   - `loot_chars` is strictly restricted to types matching `pickup_types` (`[`, `!`, `?`, `/`, `=`, `$`, `%`). Non-autopicked items like rocks/gems (`*`) are excluded, eliminating the 1,500-turn loop where the hero continuously stepped to rocks repeatedly dropped by the pet dog.
44. **Gas Spore 4d6 Explosion Shield & Decisive Adjacent Melee**:
   - Gas spores explode in a 3x3 radius upon destruction dealing 4d6 blast damage (killing 5.5% of heroes with up to 37 HP). Policies must NEVER attack or throw missiles at adjacent gas spores (`adjacent_gas_spore`), yielding `step_away_from_hostile()` to reach safe distance $\ge 2$ before ranged elimination. Against adjacent regular hostiles, policies strike decisively with melee (`melee_attack_hostile`) at `hp_frac > 0.35` rather than stepping away dealing 0 damage while taking repeated free unretaliated attacks.

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
