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
| **C29** | 10 x 20 | **DL 10** | 4.65 | **13.5m** / $0.026 | **11x Speedup Production Milestone**: 200 episodes completed in 13.5m (vs 18m); Gen 3 achieved **1,018.7 avg score** (max 4,368); Gen 10 reached **4.65 avg depth**; Gen 9 reached **7,374 avg turns** (zero timeouts); 14 runs reached DL $\ge$ 8. |
| **C30** | 10 x 20 | **DL 12** | **4.80** | 14m / $0.030 | **Triple Depth 12 Breakthrough**: Depth 12 reached 3 times (`g003_e009`, `g010_e012`, `g004_e017`); Gen 10 reached **4.80 avg depth** (929.7 score); Gen 7 reached **6,630.9 avg turns**; 14 runs reached DL $\ge$ 8. Diagnosed fountain occlusion memory wipe (0 Excalibur in 6,325 eps) and fatal attacker message lag. |
| **C31** | 10 x 20 | **DL 10** | 4.35 | 14m / $0.027 | **First Excalibur Forging in History**: Excalibur successfully forged in autonomous play! `g005_e006` reached **DL 9, 5,510 turns, 2,883 score** wielding Excalibur; `g008_e013` reached **DL 6, 7,093 turns** wielding Excalibur. Gen 9 reached **8,096.4 avg turns**. Diagnosed doorway `in_corridor` masking and non-combat starvation misattribution. |
| **C32** | 10 x 20 | **DL 14** | 4.65 | 14m / $0.027 | **Depth 14 Re-Matched & Gen 6 Breakthrough**: Gen 10 reached **Dungeon Depth 14** (`g010_e017`, score 2,497); Gen 6 reached **Depth 11** (`g006_e006`, score 1,337, 4.15 avg depth); Gen 4 reached **Depth 10** (`g004_e015`); multiple generations averaged 6,000+ turns. Diagnosed cornered floating eye starvation deadlock & prayer timeout mechanics; deployed Invariant 14 adapter melee fallback. |
| **C33** | 10 x 20 | **DL 12** | 3.87 | 14m / $0.027 | Reached DL 12. Validated multi-scenario timeout shield and AST loop guards. |
| **C34** | 10 x 20 | **DL 12** | 4.30 | 14m / $0.027 | DL 12 reached (`g002_e001`); Gen 10 reached score 3,222. Diagnosed 20k-turn dead-end passive hazard wait deadlock causing 31 starvation deaths. |
| **C35** | 10 x 20 | **DL 13** | 4.65 | 14m / $0.027 | **Depth 13 Reached!** Gen 2 averaged 6,479 turns; gas spore deaths dropped by 56% confirming the cornered passive hazard circuit breaker (`consecutive_passive_waits >= 2`). |
| **C36** | 10 x 20 | **DL 10** | 4.60 | 14m / $0.027 | DL 10 reached in Gen 9 (3,581 avg turns). Autopsy of `e010` (23,962 turns on DL1) revealed `step_away_from_hostile` search fallback loop in `handle_dead_end`. Fixed fallback to `step_to_dead_end`. |
| **C37-41**| 50 x 20 | **DL 14** | 4.65 | ~60m / $0.12 | Comprehensive 2,163-episode autopsy post-C30: 0 timeouts (100% fixed), gas spores down 96%. Discovered premature prayer ("Tyr is displeased" in 402 eps) and lichen corpse rotten shield bug (330 fainting deaths). |
| **C42** | 10 x 20 | **DL 11** | 4.35 | 14m / $0.026 | **AST Method Splicing & Autopsy Remediation**: Deployed 850-turn prayer threshold, prompt 'n' abort shield, safe lichen/lizard corpse nutrition, floating eye strike elimination, and yellow light exploding hazard recognition. Gen 1 hit **DL 11 in 2,118 turns with 0 displeased prayers**! |
| **C43** | 10 x 20 | **DL 9** | 4.55 | 14m / $0.026 | **Mines Evacuation & Tactical Autopsy**: 200 episodes completed. Sustained 3.82 avg depth, zero starvation deaths, zero displeased prayers, zero zero-progress aborts. Autopsy identified Mines stair check inversion (24 gnome deaths), gas spore Elbereth waiting (18 deaths), and healing potion adjacent hostile gating. |
| **C44** | 10 x 20 | **DL 10** | 4.45 | 14m / $0.033 | **Zero Gnome Deaths & Dead-End Autopsy**: 200 episodes completed. Gnome deaths plummeted from 24 to 0 (Mines upward redirection verified). Peak DL 10, sustained 3.70 avg depth. Autopsy discovered dead-end in-place search loop affecting 63 episodes (31.5% of runs, trapped on DL 1-2 for avg 2,470 turns). Fixed `step_to_dead_end` to navigate to wall candidates rather than looping in place, rate-limited search count decay to 50 turns, restored healing threshold to 0.50, and updated hunger prayer to `hunger_state >= 3`. |
| **C45** | 10 x 20 | **DL 11** | 4.60 | 14m / $0.022 | **Peak Depth 11 & Corpse Wait Autopsy**: 200 episodes completed. Gen 9 reached **Depth 11** (avg depth 4.60, score 2,021). Floating eye deaths fell to record low 3.5%, gas spore deaths down to 4.5%, jackal deaths down by 40%. Autopsy uncovered rotten corpse wait stalls affecting 38 episodes (19% of runs). Deployed Invariant 38: adapter 50-turn floor corpse rot pruning, policy safe corpse gating `any(c.is_safe for c in obs.corpses)`, and fallback to exploration rather than stationary waiting. |
| **C47** | 10 x 20 | **DL 12** | 4.60 | 18m / $0.021 | **Depth 12 Breakthrough & Peak Average 4.60**: Reached **Depth 12** (`e020`: turn 1,676, score 1,264, AC 1) and **Depth 10** (`e007`: score 1,117). Gen 8 achieved **4.60 avg depth** with 14/20 episodes reaching DL 4+. Diagnosed `KeyError: (7, 67)` in door kicking during vertical trap-door level transition. Deployed `.get(target_door, 0)` shield. |
| **C48** | 10 x 20 | **DL 9** | **4.65** | 18m / $0.021 | **Sustained 4.18 Average Depth & Zero DL1 Deaths**: 200 episodes completed. Gen 1 hit **4.65 avg depth** (zero DL 1 deaths); Gen 6 hit **4.60 avg depth**; Gen 5 hit **4.45 avg depth** (zero DL 1 deaths). Gas spore deaths fell from 11 down to 1 (91% reduction). Autopsy revealed combat fainting when heroes were trapped in `handle_combat` without food eating or hunger prayer. Deployed Invariant 45: in-combat carried food consumption and Weak conscious prayer. |
| **C49** | 10 x 20 | **Target DL 20.0** | **Ready to Launch** | Single-Wave 20 Workers | **In-Combat Nutrition & Conscious Prayer Scaling**: Targeting Average Depth $\ge 6.0$ toward $\ge 20.0$. |

### Recent Breakthroughs & Engine Hardening
- **Post-Campaign-30 Autopsy & Divine Favor Hardening (Invariant 52)**: Querying all 2,163 post-C30 episodes revealed that 402 runs (18.6%) triggered `"Tyr is displeased"`. NetHack sets divine timeout to $300 + \text{rn2}(500) \le 799$ turns; our old 350-turn check failed ~90% of the time. Raised safe threshold to **850 turns** and updated `_dismiss_more` to auto-answer `'n'` to `"Are you sure you want to pray? [yn] (n)"`, preventing premature prayer smiting.
- **Non-Rotting Lichen & Lizard Corpse Exemption (Invariant 53)**: 500 heroes died while fainting; 330 carried food. Lichen and lizard corpses never rot in NetHack. Updated `has_food`, `get_food_slot()`, and `food_count` to treat lichen and lizard corpses as safe non-perishable food.
- **AST Method-Level Replacement & LLM Synthesis Acceleration (Invariant 54)**: Implemented `AuthorAgent.splice_policy_methods()` via Python AST. The LLM only needs to output modified methods (`def handle_combat(...)`), cutting completion tokens by 4x-5x and synthesis latency from ~80s to ~15s without diff brittleness.
- **Floating Eye Melee Fall-Through Elimination (Invariant 55)**: Removed policy fall-through on Elbereth that struck adjacent floating eyes in melee when out of missiles. Policies now strictly yield `step_away_from_hostile()` to trigger the deadlock circuit breaker, never striking floating eyes.
- **Exploding Yellow & Black Light Hazard Recognition (Invariant 56)**: Added yellow lights and black lights to `GLYPH_IS_PASSIVE_HAZARD_LUT` and `GLYPH_IS_GAS_SPORE`. Navigation paths route around them and combat handles them with ranged missiles or retreat.
- **Fountain Occlusion & Excalibur Dipping Verification (Invariant 50)**: NetHack displays `@` at `chars[y, x]` when the hero stands on a fountain tile `(fy, fx)`. Previously, `_extract_obs` checked `chr(chars[kfy, kfx]) != "{"`, saw `@`, and immediately set `known_fountain_pos = None`. Consequently, `obs.dungeon.standing_on_fountain` evaluated to `False`, aborting `dip_excalibur` and ping-ponging the hero away from fountains across 6,325 episodes. Preserving `known_fountain_pos` when `(kfy, kfx) == (y, x)` ensures `standing_on_fountain == True`, cleanly enabling automated `#dip` for Excalibur.
- **Rolling Message History for Fatal Attacker Attribution (Invariant 51)**: When a hero dies, the final observation message `obs.message` is frequently `"You die...  --More--"` or blank. Relying solely on `obs.message` previously caused fallback to `closest_hostile_name`, which attributed fatal attacks by ants, centipedes, or yetis to harmless passive floating eyes across the room (44 false attributions). Scanning `recorder.turns[-6:]` in reverse for hit verbs (`" bites!"`, `" hits!"`, `" stings!"`) and prioritizing adjacent monsters restores 100% accurate mortality attribution.
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
- **Cornered Passive Hazard Deadlock Circuit Breaker (Invariant 14 & 22)**: Fixed insidious 20,000-turn starvation wait-locks when trapped next to immobile gas spores/floating eyes in dead ends. `NetHackAdapter.step_away_from_hostile()` tracks consecutive passive waits (`self.consecutive_passive_waits >= 2`) and autonomously breaks deadlocks via lateral steps, ranged missiles, secret door dead-end searches, or emergency strikes before fainting.

### Current Campaign Status
- **Campaign Synthesis Loop: Active (Campaign 35 Concluded, Campaign 36 Running)**.
  - **Overarching Goal**: Continuous evolutionary synthesis until reaching average ascending (and Average Depth $\ge 20.0$).
  - **Completed Campaigns**: 35 campaigns completed (7,325+ episodes evaluated, 23.6M+ game turns).
  - **Campaign 35 Highlights**:
    - Generation 7 penetrated to **Depth 13** (avg depth 4.65, 4,442.9 avg turns).
    - Generation 2 achieved **Depth 10** with **6,479.0 average turns**!
    - Gas spore fatalities dropped by **56%** (from 16 down to 7) confirming the deadlock circuit breaker.
    - Zero starvation deaths, zero poison deaths, zero zero-turn aborts.
  - **All-Time Records**:
    - **Peak Batch Average Depth**: **5.15** (Campaign 33 Generation 8).
    - **Peak Turn Survival**: **6,235.3 Average Turns** (Campaign 33 Generation 10).
    - **Peak Dungeon Depth**: **14** (Campaign 25 `g002_e003` & Campaign 32 `g010_e017`).
    - **Peak Score**: **4,612** (Campaign 24).
  - **Engines Implemented**:
    - Deterministic Sokoban Solver (`lox/core/sokoban.py`) [Complete & Tested]
    - Digging & Ray Tunneling Router (`lox/core/digging.py`) [Complete & Tested]
    - Castle Drawbridge Safe Breacher (`lox/envs/solvers/castle_solver.py`) [Complete & Tested]
    - Bag of Holding Explosion Guard (`lox/core/epistemic.py`) [Complete & Tested]
    - Resistance & Artifact Properties (`HeroState`, `InventoryView`, `DungeonView`) [Complete & Tested]
    - Cornered Passive Hazard Deadlock Circuit Breaker (`lox/envs/nethack.py`) [Complete & Tested]
  - **Campaign 42 Findings & Remediation (Campaign 43 Target)**:
    - **Issue 1 (Floating Eye Melee Suicide)**: In `step_away_from_hostile()`, cornered deadlock breaker attempted melee strikes against floating eyes, causing 70-turn paralysis (`0d70`) and 100% combat fatalities. Explicitly prohibited emergency strikes against floating eyes.
    - **Issue 2 (Missile Ammo Starvation)**: NetHack options lacked `pickup_thrown` and `*` in `pickup_types`, so thrown daggers and dropped rocks were never picked up, leaving heroes with 0 missiles. Added `pickup_thrown` and `*` to options and `loot_chars`.
    - **Issue 3 (Dead-End & Wall Perimeter Oscillation)**: `step_to_dead_end` excluded current tile when falling back to `wall_adj_mask`, causing 2-tile ping-pong loops (2,600 turns in `e016` on DL 1). Fixed by searching current wall tile immediately if `searched_count < 10` before stepping away, and increasing policy dead-end searches to 12.
    - **Issue 4 (In-Combat Hunger Trap)**: Heroes next to floating eyes/gas spores refused to eat carried food even when fainting. Updated `handle_combat` to permit eating when adjacent to passive hazards or when `hunger_state >= 3`.
    - **Issue 5 (Stair Descent Prioritization)**: Altar BUC testing and floor looting were placed ahead of stair descent in `run()`, causing heroes to wander in loops on shallow depths instead of descending. Prioritized `descend()` and `step_to_stairs_down()` immediately after vital equipment and nutrition.
  - **Checkpoint Status**: Resuming from `data/latest_policy.py`.

---

## 4. Master Ascension Milestones: The 7 Phases from Depth 1 to Astral

To bridge the gap from early-game survival to average ascending, LOX must systematically master the 7 canonical NetHack progression phases documented on the [NetHack Wiki](https://nethackwiki.com/wiki/Ascension):

### Phase 1: Early-Game Consolidation (Depths 1–10) [Active / Solved Baseline]
- **Mines Evacuation (`dnum == 2`)**: Immediate ascent on `<` back to Dungeons of Doom upon hitting the Gnomish Mines staircase.
- **Tactical Sanctuaries**: Dust Elbereth finger-engraving (`E - Elbereth\r`), 1-tile corridor chokepoints, 800-turn safe divine favor cooldown.
- **Weapon & Resistance Scaling**: Dipping long sword into fountain at XL $\ge 5$ for Excalibur (+1d10 damage, automatic secret door searching, level drain immunity); harvesting poison resistance from killer bees/centipedes.
- **Nutrition**: Proactive weakness eating (`hunger_state >= 2`), rotten carried corpse filtration.

### Phase 2: Sokoban Branch & Tactical Ascension Kit (Depths 6–12) [Engine Implemented]
- **Sokoban Branch Transit (`dnum == 4`)**: Identifying the upward-leading staircase `<` located between DL 6 and 10.
- **Deterministic Boulder Solver (`SokobanSolver`)**: Navigating the 4 Sokoban levels without boulder-jumping, destroying boulders, or incurring Luck penalties.
- **Top-Floor Prize Acquisition**: Securing the guaranteed prize chest:
  - 50% chance: **Bag of Holding** (quadruples inventory weight efficiency).
  - 50% chance: **Amulet of Reflection** (immunity to death rays, lightning, and breath attacks).
- **Level Scaling**: Advancing Valkyrie experience level to **XL 14** (required for the Quest assignment).

### Phase 3: Medusa's Island & The Castle Breach (Depths 14–25) [Engine Implemented]
- **Medusa's Gaze Bypass (DL 22–24)**:
  - Traversing Medusa's Island using reflection, a blindfold/towel with telepathy, or zapping a wand of death.
  - Avoiding moat drowning using levitation (ring/boots) or water walking.
  - Slaying the statue of Perseus for potential Levitation boots or Shield of Reflection.
- **The Castle & Drawbridge Breach (DL 25–29) (`CastleDrawbridgeSolver`)**:
  - Slaying the perimeter soldiers and dragons.
  - Destroying the drawbridge portcullis safely by zapping a **Wand of Striking** or wand of opening (avoiding standing adjacent to avoid crushed-by-drawbridge instadeath).
- **Castle Armory Conquest & Wand of Wishing**:
  - Clearing the Castle barracks.
  - Looting the back storage chests to secure the guaranteed **Wand of Wishing** ($1d3+1$ charges).

### Phase 4: The Wishing & Armory Transformation
- **Priority Wishing Protocol**:
  1. `blessed +2 gray dragon scale mail` (grants permanent Magic Resistance against death rays, touch of death, and polymorph).
  2. `blessed +2 speed boots` (permanent extrinsic speed: movement speed 24 vs baseline 12).
  3. `blessed bag of holding` (if not obtained in Sokoban).
  4. `blessed magic marker` (1:99 charges to write scrolls).
- **Magic Marker Writing & Recharging**:
  - Writing 2–3 scrolls of charging (blessed charging restores 3 charges to the wishing wand).
  - Writing scrolls of genocide: genociding `L` (Liches, Demiliches, Master Liches, Arch-Liches) and `;` (Sea monsters: giant eels, krakens).
  - Writing uncursed and cursed scrolls of gold detection (for magic mapping and finding the Vibrating Square).

### Phase 5: Gehennom Traversal & The Three Invocation Relics (Depths 26–45)
- **The Valley of the Dead (DL 26–30)**: Crossing the graveyard without graveyard amnesia.
- **Gehennom Mazes & Straight-Line Digging**:
  - In NetHack 3.6, walking Gehennom mazes wastes 15,000+ turns.
  - Utilizing wands of digging and pickaxes to drill straight-line cardinal tunnels connecting stairs up and stairs down.
- **Relic 1: The Candelabrum of Abernathy (Vlad's Tower)**:
  - Branching into Vlad's Tower (DL 34–40, 3 levels).
  - Slaying Vlad the Impaler (immune to drain, easily dispatched with Excalibur).
  - Looting the *Candelabrum of Abernathy* and guaranteed water walking boots.
- **Demon Lord Bypass / Execution**:
  - Asmodeus & Baalzebub: Slaying or bribing with gold to pass peacefully.
  - Juiblex & Orcus: Slaying with wands of digging/fire/death or Excalibur melee.
- **Relic 2: The Book of the Dead (The Wizard's Tower)**:
  - Entering the real Wizard's Tower via the magic portal on the fake tower level.
  - Defeating the Wizard of Yendor in his sanctuary.
  - Looting the *Book of the Dead*.
- **Relic 3: The Bell of Opening (The Quest Nemesis)**:
  - Completed on the Valkyrie quest by defeating Lord Surtur and securing the *Orb of Fate*.

### Phase 6: The Vibrating Square, Moloch's Sanctum & The Amulet (Depths 45–53)
- **Locating the Vibrating Square (`~`)**:
  - The Vibrating Square is located on the second-to-bottom level of Gehennom (DL 45–53).
  - Reading a cursed scroll of gold detection while confused reveals the exact `~` tile instantly.
- **Performing The Invocation**:
  - Standing directly on the Vibrating Square:
    1. Attach 7 candles to the Candelabrum and `#apply` (light) it.
    2. `#apply` the Bell of Opening.
    3. `#read` the Book of the Dead.
  - The ritual opens the stairs down into **Moloch's Sanctum**.
- **Moloch's Sanctum Conquest**:
  - Navigating the sanctum graveyard and temple moat.
  - Executing the High Priest of Moloch.
  - Picking up the **genuine Amulet of Yendor**.

### Phase 7: The Ascension Run & The Astral Plane
- **The Run Up through Gehennom**:
  - Ascending back up with the Amulet: countering the *Mysterious Force* (which teleports the hero down 1–4 levels).
  - Repelling the recurring Wizard of Yendor (who resurrects and teleports to the hero).
- **The Four Elemental Planes**:
  - **Plane of Earth**: Zapping wands of digging straight to the portal.
  - **Plane of Air**: Cloud pathfinding, lightning reflection, avoiding air elementals.
  - **Plane of Fire**: Levitation over lava seas to reach the portal.
  - **Plane of Water**: Water walking boots or oilskin cloak, swimming inside air bubbles.
- **The Astral Plane & Ascension**:
  - Navigating three High Altars (Lawful, Neutral, Chaotic).
  - Equipping the Ring of Conflict to turn angel and demon hordes against each other.
  - Evading / disabling the Three Riders of the Apocalypse (**Death**, **Pestilence**, **Famine**):
    - Death: Wand of teleportation or striking (never wand of death).
    - Pestilence: Curing sickness immediately with a unicorn horn.
    - Famine: Slaying at range or eating rations.
  - Identifying the Lawful Altar.
  - Stepping onto the high altar, invoking `#offer` with the Amulet of Yendor to **ASCEND**.

---

## 5. Domain Knowledge to Embed in Engine & Prompts

1. **Resistance Stacking Matrix**:
   - Intrinsic: Poison, Cold, Shock, Fire, Sleep, Telepathy (via corpses and level advancement).
   - Extrinsic: Magic Resistance (Gray Dragon Scale Mail), Reflection (Amulet of Reflection or Silver Dragon Scale Mail).
2. **Container Explosion Guard**:
   - Never put a Wand of Cancellation, Bag of Tricks, or nested Bag of Holding inside a Bag of Holding (causes instantaneous catastrophic explosion and item destruction).
3. **Instadeath Prevention**:
   - Touch of Death: Protected 100% by Magic Resistance.
   - Disintegration: Protected by Reflection or Disintegration resistance.
   - Petrification: Never touch cockatrice/chickatrice without gloves; never stumble/blind walk over dead footrices.
   - Drowning: Protected by Levitation, Water Walking Boots, or Oilskin Cloak.
4. **Item Identification & Holy Water Chemistry**:
   - Testing BUC (Blessed/Uncursed/Cursed) on altars.
   - Crafting Holy Water: Dipping uncursed water on a co-aligned altar with positive divine favor.

---

## 6. Required Architectural Mechanisms to Construct

1. **Sokoban BFS Graph Solver (`lox.core.sokoban`) [IMPLEMENTED & VERIFIED]**:
   - Automated push-boulder pathfinding solving standard Sokoban levels without deadlocks or Luck penalties.
2. **Gehennom Digging Router (`lox.core.digging`) [IMPLEMENTED & VERIFIED]**:
   - Bresenham line tunneling using wands of digging/pickaxes between stairs up and down.
3. **The Castle Breach Macro (`lox.envs.solvers.castle_solver`) [IMPLEMENTED & VERIFIED]**:
   - Automated drawbridge detection, distance-2 safe wand of striking execution, and perimeter clearance.
4. **Epistemic Container Safety Guard (`lox.core.epistemic`) [IMPLEMENTED & VERIFIED]**:
   - Prevents Bag of Holding explosions by vetoing cancellation wands and nested bags.
5. **Invocation & Endgame State Machine (`lox.solvers.invocation`) [PLANNED]**:
   - Coordinated tracking of Candelabrum (7 candles), Bell, Book, Vibrating Square coordinates, and High Altar identification.


---

## 7. Key Operational Documents

- **[`AGENTS.md`](file:///home/moose/git/lox/AGENTS.md)**: Complete operational guide, architecture, 24 consolidated technical invariants across 6 operational domains, and synthesis protocols.
- **[`README.md`](file:///home/moose/git/lox/README.md)**: System overview, 3-tier meta-optimization architecture, end-to-end lifecycle walkthrough, quickstart guide, and empirical milestones.


