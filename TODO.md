# LOX Campaign Action Plan & Roadmap (`TODO.md`)

This living document tracks empirical progress, critical discoveries, immediate next actions, and long-term milestones toward the campaign goal: **Average Dungeon Depth $\ge 20.0$** over 20 consecutive NetHack episodes.

---

## 1. Current State & Empirical Trajectory

- **Target Goal**: Average Depth $\ge 20.0$ (20 episodes, up to 25,000 max turns each).
- **Current Historical Highs**: Peak single-run record of **Depth 11** (`g007_e007`), **Depth 10** (`g009_e017`, `g002_e001`, `g008_e001`), and sustained batch averages reaching **4.10–4.60**.

### Progression Timeline
| Benchmark | Avg Depth | Max Depth | Key Diagnoses & Breakthroughs |
| :--- | :---: | :---: | :--- |
| `bench_fixes_1` | 2.00 | 4 | Initial baseline evaluation before retreat & Elbereth optimizations; 40% died on DL1. |
| `bench_fixes_2` | 2.90 | 4 | Added early corpse nutrition eating and basic Elbereth sanctuary logic; pushed to DL4. |
| `bench_fixes_3` | 1.90 | 3 | Discovered floating eye passive paralysis deaths; added absolute melee prohibition. |
| `bench_fixes_4` | 2.40 | 4 | Added ranged dagger throwing against fast pests (bats, insects); reduced DL1 deaths. |
| `bench_fixes_5` | 2.00 | 4 | Identified peaceful creature confirmation loops; added peaceful dialog dismissal. |
| `bench_fixes_6` | 2.90 | 5 | First run reaching Depth 5; introduced cardinal door kicking for locked doors. |
| `bench_fixes_8` | 2.70 | 5 | 77.7% of turns lost to phantom doors (`+` spellbooks vs closed doors); added CMAP glyph check. |
| `bench_fixes_9` | 2.10 | 5 | Discovered 60.9% turns lost to corridor retreat loops with fast monsters; tuned chokepoints. |
| `bench_fixes_10` | 3.60 | 5 | Solved DL1 door stalling; improved corridor retreat stability and healing triggers. |
| `bench_fixes_12` | 3.20 | 6 | Fixed generator zero-yield hanging in HTN subroutines (`yield from` without action). |
| `bench_fixes_13` | 3.60 | **9** | **Depth 9 reached!** Discovered hero self-block navigation bug and mutual recursion. |
| `bench_fixes_14` | 3.20 | 6 | Saved 17,000 turns by fixing altar ASCII `@` hero occlusion detection. |
| `bench_fixes_15` | 3.50 | 7 | Eliminated 0-turn locked door kick loop with kick count caps (6 max kicks). |
| `bench_fixes_16` | **4.00** | 6 | 100% DL1 clearance; 3 episodes reached DL6. Diagnosed Excalibur dipping prompt failure. |
| `bench_fixes_17` | 3.60 | 7 | Two episodes reached DL7. Uncovered 23.5k in-place search decay loop at `(5, 46)`. |
| `bench_fixes_18` | 3.30 | **8** | Reached Depth 8! 0 timeouts at 25k turns; validated `(weapon in hand)` tracking and eliminated in-place search stall. Diagnosed item naming dialog trap (`Call a ...`). |
| `bench_fixes_19` | 2.50 | 6 | Clean execution across all 10 episodes in 66.9s. 0 timeouts; validated `Call a ...` item naming dialog auto-dismissal. |

---

## 2. Core Architectural Invariants & Discoveries
All tactical, spatial, and execution rules are systematically documented and maintained in [`AGENTS.md`](file:///home/moose/git/lox/AGENTS.md#3-consolidated-technical-invariants--operational-rules) across six core modules:
1. **Policy Architecture & Anti-Loop Defense**: Python generator protocol (`yield action`, `obs = yield from subroutine`), 4-layer loop guards (AST `LoopGuardTransformer`, `PolicyRunner` fallback to `wait()`, 90s episode ceiling, 180s batch pool timeout), `signal.setitimer` dry-run validator.
2. **Dialog & Keystroke Protocols**: Prompt shields for `[yn]` / `[ynq]` confirmation queries, text dialog auto-dismissal (ESC `\x1b` for Minetown greetings and item naming prompts), intermediate multi-key preservation (`is_intermediate`), universal $\ge 4$ zero-turn circuit breaker.
3. **Spatial Navigation & Topological Memory**: Full-floor persistent topological memory (`self.visited` + `self.known_chars` for `#`, `.`, `<`, `>`, `_`, `{`, `+`), static level fixture persistence, Gnomish Mines branch avoidance (`dnum == 2`), trap-door fall vs stair arrival handling, stride-2 checkerboard secret door search, Numba JIT disk caching.
4. **Combat Tactics & Sanctuary**: Dust Elbereth warding (`E` $\to$ `-` $\to$ `Elbereth\r`), Elbereth-immune species discrimination (`IGNORES_ELBERETH_SPECIES`), floating eye / gas spore melee avoidance, corridor chokepoints (`is_fast_dangerous`), in-combat healing potions, ranged dagger harassment, immediate staircase combat escape (`descend()`).
5. **Inventory & Equipment Acquisition**: NLE autopickup initialization, armor equipping (`wear_armor`) with redundant slot tracking, floor loot scooping (`step_to_loot`) with ping-pong protection (`self.looted_tiles`), safe fresh corpse nutrition, Excalibur fountain dipping (`standing_on_fountain`), weapon hand grammar recognition (`(weapon in hand)`).
6. **Telemetry & Parallel Evaluation**: 20-worker single-wave multiprocessing, buffered Parquet writes (`flush_interval=5000`), DuckDB mortality schemas (`killer`, `ac_at_death`, `excalibur_forged`), 6-attempt exponential backoff retry for OpenRouter API resilience.

---

## 3. Autonomous LLM Synthesis Campaign History

### Campaign Progression Summary
| Campaign | Gens x Eps | Peak DL | Peak Batch Avg | Speed / Cost | Key Diagnoses & Breakthroughs |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **C1** | 10 x 20 | **DL 11** | 4.05 | ~20m / $0.02 | Gemma synthesized Gen 7 reaching DL11. Solved solver fallback closed doors & poison gating. |
| **C2** | 10 x 20 | **DL 10** | 3.80 | ~18m / $0.02 | Zero batch collapses (avg 3.0–3.8). Excalibur forged. Shielded floating eyes & gas spores. |
| **C3** | 10 x 20 | **DL 10** | 3.85 | ~16m / $0.02 | Fixed inverted room-to-corridor retreat logic; added mixed hazard swarm Elbereth engagement. |
| **C4** | 4 x 20 | DL 8 | 3.85 | ~8m / $0.01 | Stably held ~3.8 avg depth. Unlocked floor loot scooping, offensive wands, panic teleport. |
| **C5** | 10 x 20 | DL 8 | 3.05 | ~10m / $0.02 | Parallel multiprocessing baseline. Fixed loot ping-pong loop (`self.looted_tiles`) & fountain vanished. |
| **C6** | 7 x 20 | DL 6 | 2.55 | ~8m / $0.01 | Uncovered intermediate keystroke dialog cancellation trap; added `is_intermediate` preservation. |
| **C7** | 4 x 20 | DL 8 | 3.00 | ~6m / $0.01 | Zero aborts milestone. Fixed no-hostile Elbereth/wait stagnation; scaled workers to 20. |
| **C8** | 10 x 20 | **DL 9** | **4.60** | 14m / $0.02 | Peak generation average 4.60 (Gen 5). Diagnosed staircase combat interception trap. |
| **C9** | 10 x 20 | **DL 10** | 4.35 | 18m / $0.02 | DL10 reached at turn 8,612 (AC -2, 51 HP). Upgraded retreat door breaching (`_step_or_breach`). |
| **C10** | 10 x 20 | **DL 9** | 4.40 | 12m / $0.02 | Multiple DL9 runs. Diagnosed starvation fainting; prioritized post-combat fresh corpse eating. |
| **C12** | 10 x 20 | DL 8 | 4.10 | 14m / $0.02 | Fixed `standing_on_elbereth` timing, monster attack Elbereth discard, `IGNORES_ELBERETH_SPECIES`, AST loop guards. |
| **C13** | 10 x 20 | **DL 9** | 4.35 | 12m / $0.02 | Discovered Gnomish Mines branch ID (`dnum == 2`). Added immediate Mines ascend and stair persistence. |
| **C14** | 10 x 20 | **DL 11** | 4.00 | 18m / $0.02 | `e018` reached DL11 in 1,702 turns. Added trap-door fall discrimination vs genuine stair arrival. |
| **C15** | 10 x 20 | **DL 11** | 4.50 | 16m / $0.02 | 100% genuine combat deaths (0 aborts/timeouts). Added full-floor persistent topological memory (Inv 49). |
| **C16** | 10 x 20 | DL 8 | 3.35 | 14m / $0.023 | 2 episodes survived all 25k turns; 0 zero-progress aborts. Diagnosed passive hazard pathfinding bumps & fast predator speeds. |
| **C17** | 10 x 20 | **DL 11** | 3.75 | 14m / $0.023 | Gen 9 reached **Depth 11**! Gen 10 avg score 507.6; 99.5% combat deaths. Diagnosed hero idling next to passive hazards during swarms; added `has_safe_melee_target`. |
| **C18** | 10 x 20 | **DL 10** | 3.50 | 14m / $0.022 | Gen 7 reached **Depth 10**! Gen 8 reached 3.50 avg depth; 99.0% combat deaths. Diagnosed author system prompt template reverting `has_safe_melee_target`; updated `prompts.py`. |
| **C19** | 10 x 20 | **DL 11** | **4.05** | 15m / $0.022 | Gen 8 reached **4.05 avg depth** (508 score); Gen 10 reached **DL 11**! Diagnosed in-combat hunger starvation lock & floating eye stalemate; added in-combat food/prayer & solo eye resolution. |
| **C20** | 10 x 20 | **DL 10** | 3.85 | 16m / $0.026 | Gen 3 reached **DL 10**; Gen 6 achieved **650.0 avg score**; 3,666 avg turns across 200 episodes. Diagnosed Vault Guard accidental missile provocation & distant passive hazard combat locks; added Invariants 39 & 40. |
| **C21** | 10 x 20 | DL 8 | 3.75 | 15m / $0.024 | 0 Vault Guard deaths (100% peaceful protection verified). Diagnosed lethal food poisoning from rotten carried corpses & ammunition starvation stalemates; added Invariants 41 & 42. |
| **C22** | 10 x 20 | **DL 9** | 3.80 | 16m / $0.024 | 0 food poisoning deaths (0 lichen deaths). Turns exceeded 4,250. Diagnosed LLM combat trigger mutation; aligned `hostile_count_fov` in adapter to naturally exclude distant passive hazards. |
| **C23** | 10 x 20 | **DL 10** | 4.35 | 16m / $0.024 | All-time project high score: Gen 1 avg score **774.0**, Gen 7 avg depth **4.35 (755.0 score, DL 10)**, sustained 3.68 depth / 617.0 score across 200 eps. Diagnosed fast predator running trap; added Invariant 43. |
| **C24** | 10 x 20 | **DL 11** | **5.00** | 16m / $0.025 | **Historic Milestone**: Gen 4 achieved **5.00 avg depth** and **1,148.0 avg score** (max 3,230); Gen 5 reached **DL 11**; Gen 6 reached max score **4,612**; overall 200-ep avg depth **4.17**! Diagnosed 2,700-turn dead-end floating eye wait locks & hunger state 3 fainting; added Invariants 44 & 45. |
| **C25** | 10 x 20 | **DL 14** | 4.35 | 16m / $0.026 | **All-Time Max Depth Record**: Gen 2 reached **Dungeon Depth 14**! Gen 7 achieved max score **3,998**; sustained 3.86 avg depth & 4,029 avg turns across 200 eps. Diagnosed premature prayer divine anger (< 300 turns) & room-wide fountain approach; added Invariants 46 & 47. |
| **C26** | 10 x 20 | **DL 9** | **4.85** | 16m / $0.026 | Gen 6 reached **4.85 avg depth** (759.1 score); Gen 5 achieved **3,966 max score**; sustained 3.84 avg depth across 200 eps. Diagnosed open-room multi-monster swarm surround deaths from speed-18 ants/bees; added Invariant 48. |
| **C27** | 10 x 20 | **DL 10** | **5.10** | 16m / $0.026 | **Batch Average Record**: Gen 5 broke records with **5.10 avg depth** (max depth 10); Gen 9 achieved **1,014.5 avg score** (max 2,973); sustained 4.07 avg depth & 742 avg score across 200 episodes. |
| **C28** | 10 x 20 | **DL 12** | 4.65 | 18m / $0.027 | Gen 7 reached **Dungeon Depth 12** in 1,623 turns (`g007_e003`); Gen 4 achieved **1,124.8 avg score** (max 4,551); multiple DL9/DL10 runs. Diagnosed slow Python loop profiling (75 steps/s) and timeout deaths; deployed Invariant 49. |

### Recent Breakthroughs & Engine Hardening
- **Vectorized Glyph Lookup Tables & 11x Turn Execution Speedup (Invariant 49)**: Repeatedly calling C-extension helpers (`nh.glyph_is_monster`, `nh.glyph_is_body`, `permonst`) in 21x79 Python loops during `_extract_obs` and `_build_walkable_nav` bottlenecked stepping to ~75 steps/s, causing deep multi-thousand-turn runs to falsely trip the 90s episode wall-clock ceiling. Precomputing 6KB module-level boolean lookup tables (`GLYPH_IS_MON_HOSTILE_LUT`, `GLYPH_IS_BODY_LUT`, `GLYPH_IS_PASSIVE_HAZARD_LUT`, `GLYPH_IS_FAST_LUT`, `GLYPH_IS_PEACEFUL_SPECIES_LUT`) for all 5,976 glyphs and using C-level byte message decoding accelerates turn execution to **826+ steps/s (11x speedup)**, completing 6,000+ turn episodes in ~7 seconds.
- **Fast Dangerous Swarm Elbereth Defense & Corridor Chokepoint Steering (Invariant 48)**: Killer bees (speed 18) and soldier/giant ants (speed 18) spawn in swarms of 3–8 members at Depths 4–10. In open rooms, engaging a swarm in melee results in multiple surrounding attacks per turn and rapid fatal poison stings. When facing multiple fast attackers or when surrounded (`obs.combat.is_fast_dangerous and (obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2)`), policies MUST engrave Elbereth (`engrave_dust_elbereth`) immediately. As insects, ants and bees do not ignore Elbereth and flee in terror, allowing the hero to safely quaff healing or retreat to 1-tile corridor chokepoints (`step_to_chokepoint`) to fight them one at a time.
- **Safe Divine Favor Prayer Threshold & Death Attribution (Invariant 46)**: In NetHack, successful prayer sets divine timeout to $300 + \text{rn2}(500)$ turns. Praying at $\le 150$ turns angers the deity ("Tyr is displeased"), resulting in divine smiting or paralysis. In-combat and major trouble prayer must strictly enforce `turn - last_prayer_turn >= 350` to guarantee safe divine feeding or full HP restoration. Ephemeral death telemetry parses the fatal combat message (`" bites!"`, `" hits!"`, `" stings!"`) to accurately attribute fatalities to the attacking monster rather than lingering passive hazards across the room.
- **Room-Wide Fountain Navigation & Excalibur Scaling (Invariant 47)**: Dipping for Excalibur (`dip_excalibur()`) requires standing directly on the fountain tile (`obs.dungeon.standing_on_fountain`). Checking only `adjacent_fountain` previously prevented heroes from approaching fountains located $\ge 2$ tiles away in the same room. Gating `step_to_fountain()` by `obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain` allows heroes to navigate across open rooms to forge Excalibur upon reaching XL $\ge 5$.
- **Adjacent Passive Hazard Ranged Elimination & Dead-End Melee Break (Invariant 44)**: Throwing daggers, darts, arrows, or rocks (`throw_dagger()`) at an adjacent floating eye (`obs.combat.adjacent_floating_eye`) is 100% safe at distance 1 and does NOT trigger the passive melee paralysis attack. Policies must never restrict missile attacks against floating eyes to `dist >= 2`. If cornered in a dead end with an adjacent floating eye and 0 missiles, waiting will cause certain starvation (as seen in 2,700-turn wait locks); striking the floating eye in melee instantly kills its 1-HP body and clears the path. The resulting 50-turn paralysis is safe in an isolated corridor and wears off cleanly. Gas spores explode for 4d6 damage at distance 1, so heroes retreat away or attack active hostiles (`obs.combat.has_safe_melee_target`).
- **Proactive In-Combat Nutrition & Weakness Eating (Invariant 45)**: NetHack fainting occurs randomly at `hunger_state >= 3` (30-turn unconscious paralysis leading to defenseless death). Carried rations and wafers (`eat_carried_food()`) must be consumed proactively at `hunger_state >= 2` ("Weak") whenever standing on Elbereth, out of melee reach, or adjacent only to passive hazards (`adjacent_floating_eye` or `adjacent_gas_spore`). Policies must never wait until `hunger_state >= 3` to eat while monsters are in FOV.
- **High-Speed Predator Engagement & Non-Retreat Rule (Invariant 43)**: Against predators faster than hero speed (giant bats speed 22, soldier ants speed 18, foxes speed 15), stepping away in open rooms gives the predator free hits while dealing 0 damage. When adjacent, policies MUST engage in melee (`melee_attack_hostile`) if HP $> 40\%$ (a Valkyrie sword strike kills them in 1–2 hits) or engrave Elbereth (`engrave_dust_elbereth`) to repel them if low HP. Tactical retreat to chokepoints (`step_to_chokepoint`) is reserved strictly for when the predator is at distance $\ge 2$ in an open room. Giant bat fatalities dropped from 15+ down to 6.
- **Natural Passive Hazard Exclusion in `hostile_count_fov`**: `hostile_count` and `closest_hostile` tracking in `NetHackAdapter` now naturally ignore passive hazards (`floating eye`, `gas spore`, molds, jellies, spheres) unless they are immediately adjacent (`is_adjacent`). This guarantees that any generated policy checking `hostile_count_fov > 0` continues normal exploration and staircase descent without getting stalled by passive hazards across the room.
- **Rotten Corpse Food Poisoning Shield & Nutrition Discrimination (Invariant 41)**: In NetHack, corpses rot within 50 turns in the hero's backpack. When hungry, `get_food_slot()` previously picked rotten carried corpses (`a lichen corpse`, `a newt corpse`), resulting in lethal food poisoning. `has_food`, `get_food_slot()`, and `food_count` strictly exclude corpses (`"corpse" not in it.name.lower()`), reserving consumption strictly for non-perishable packaged food (rations, wafers, fruit). When packaged food is exhausted, `has_food == False` triggers major trouble prayer (`pray()`) at `hunger_state >= 3`, providing safe divine feeding without tainted meat poisoning.
- **Universal Ranged Missile Ammunition (Invariant 42)**: Goblins and orcs drop dozens of darts, arrows, and rocks across early dungeon floors. `has_daggers`, `get_dagger_slot()`, and `dagger_count` recognize all throwable missiles (daggers, darts, arrows, rocks, shuriken, spears, javelins). This ensures the hero never runs out of ranged ammo, allowing continuous ranged elimination of passive hazards (floating eyes, gas spores) and fast speedsters from distance $\ge 2$ without getting stuck in idle wait stalemates.
- **Vault Guard, Priest & Peaceful Non-Aggression (Invariant 39)**: Level 12 Vault Guards (`guard`) and temple priests (`priest`, `priestess`) are neutral until attacked. When a vault guard asks the hero to follow him out, policies previously threw daggers or zapped wands, provoking instant 1-hit deaths. Expanded `is_peaceful_species` to include `"guard"`, `"priest"`, `"priestess"`, shielded missiles (`throw_dagger`) and wands (`zap_offensive_wand`) against firing at peaceful targets, and enforced retreat.
- **Active Hostile Gating & Passive Hazard Bypass (Invariant 40)**: Immobile passive hazards (`floating eye`, `gas spore`) at distance $\ge 2$ do not attack or hunt the hero. Added `has_active_hostile` and `active_hostile_count` to `CombatView` and `ALLOWED_PREDICATES`. When no ranged weapons are in inventory and all hostiles in FOV are passive, policies bypass `handle_combat` and proceed with exploration, as `SpatialEngine` (`walkable_nav`) already steers around them safely without melee paralysis or starvation stalemates.
- **In-Combat Hunger & Solo Floating Eye Stalemate Resolution**: Added in-combat food eating (`obs.hero.hunger_state >= 2`) and major trouble prayer (`hunger_state >= 3`) inside `handle_combat` so heroes never starve while blocked by harmless monsters.
- **Author Prompt Alignment for Mixed Combat (`prompts.py`)**: Updated `build_system_prompt()` template and Invariant 1 so LLM authoring explicitly integrates `has_safe_melee_target`, `has_active_hostile`, and peaceful NPC non-aggression across all synthesis sessions.
- **Safe Melee Engagement in Mixed Hazard Swarms (`has_safe_melee_target`)**: When facing a passive hazard (gas spore or floating eye) alongside active attackers (newt, jackal, orc), `obs.combat.has_safe_melee_target` allows the hero to strike the active attacker in melee while the adapter skips the passive hazard, preventing the hero from idling and dying to minor pests.
- **Passive Hazard Pathfinding Masking & Safe Cornered Wait (Invariant 22)**: Fixed pathfinding routing through passive/exploding hazards (floating eyes, gas spores, molds, jellies) by masking them out of `walkable_nav`. When cornered or unable to retreat, `step_away_from_hostile()` yields `wait()` rather than bumping into them.
- **Dynamic Fast Monster Discrimination (Invariant 23)**: Ground-truth `permonst.mmove > 12` dynamic check plus foxes (speed 15), coyotes, and jaguars flags fast predators for immediate corridor chokepoints (`step_to_chokepoint()`).
- **Full-Floor Persistent Topological Memory (Invariant 49)**: Fixed line-of-sight amnesia where `walkable_nav` and closed doors were lost when moving out of FOV. Augmenting pathfinding with `self.visited` and `self.known_chars` preserves 100% floor connectivity.
- **Locked Door Circuit Breaker & Chokepoint Blocking (Invariant 48)**: Locked doors in shops or after 6 failed kicks are added to `self.blocked_tiles`. Universal $\ge 4$ zero-turn circuit breaker forces `wait()` (`.`), eliminating all `StepStatus.ABORTED` occurrences.
- **OpenRouter Exponential Backoff Resilience (Invariant 47)**: 6-attempt exponential backoff retry with jitter ensures uninterrupted overnight synthesis.

### Current Campaign Status
- **Campaign Synthesis Loop: Active (Campaign 28+ Resumed)**.
  - **Overarching Goal**: Continuous evolutionary synthesis until reaching average ascending (and Average Depth $\ge 20.0$).
  - **Completed Campaigns**: 27 campaigns completed (5,725 episodes evaluated).
  - **Peak Depth Achieved**: **Dungeon Depth 14** (Campaign 25).
  - **Peak Batch Average Depth**: **5.10** (Campaign 27 Gen 5).
  - **Peak Score**: **4,612** (Campaign 24).
  - **Checkpoint Status**: Resuming from `data/latest_policy.py`.

## 4. Longer-Term Goals (Roadmap to Depth 20+)

1. **Autonomous LLM Synthesis Loop Execution**:
   - Continue running iterative generations until batch average depth $\ge 20.0$.
2. **Main Dungeon vs Gnomish Mines Branch Steering**:
   - Prioritize descending the main dungeon staircase down to Depth 20 without entering the lethal dark Mines.
3. **Container & Bag Stash Management**:
   - Looting sacks and chests for additional scrolls and potions.
4. **Mid-Game Ascension Prep & Medusa / Castle Breaching**:
   - Gearing AC < -5, Excalibur forging, reflection, and poison/cold/fire resistance intrinsic stacking.
5. **Campaign Completion Verification**:
   - Batch average depth $\ge 20.0$ triggers `[CAMPAIGN GOAL ACHIEVED]`.
   - Record final DuckDB telemetry and emit `<!-- GOAL_COMPLETE -->`.

