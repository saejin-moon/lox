# LOX 2.0 Campaign Action Plan & Roadmap (`TODO.md`)

This living document tracks empirical progress, critical discoveries, immediate next actions, and long-term milestones toward the campaign goal: **Average Dungeon Depth $\ge 10.0$** over 10 consecutive NetHack episodes.

---

## 1. Current State & Empirical Trajectory

- **Target Goal**: Average Depth $\ge 10.0$ (10 episodes, up to 25,000 max turns each).
- **Current Baseline (`bench_fixes_17`)**:
  - **Max Depth**: **Depth 7** (`ep_006` and `ep_007` reached DL7, `ep_001` & `ep_009` reached DL5, `ep_010` reached DL4). Half of all episodes now cluster at Depths 5–7.
  - **Historical Peak**: Single-run record of **Depth 9** (`ep_001` in `bench_fixes_13`).
  - **Dungeon Level 1 Clearance**: 100% of non-stalled episodes cleanly solve and descend DL1.

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

## 2. Hard-Won Lessons & Architectural Discoveries

1. **Fountain Dipping Protocol (`dip_excalibur`)**:
   - In NetHack, `#dip` requires **standing directly on the fountain tile** (`standing_on_fountain`). Dipping from an adjacent square prompts for an item to dip *into*, selecting arbitrary backpack gear and outputting `"That is a silly thing to dip..."`.
   - `step_to_fountain` steps directly onto `closest_fountain_pos`, and `dip_excalibur` passes `[DIP, weapon_slot]`, with `_dismiss_more` auto-answering `'y'` to `"dip into the fountain? [yn]"`.

2. **In-Place Search Stall Prevention (`self.last_searched_pos`)**:
   - Rapid search decay in `_observe_internal` and `step_to_dead_end` can repeatedly reset the hero's current tile to 0 searches. Because the current tile is at distance 0, `find_nearest_target` picks it over distant candidates, causing tens of thousands of turns of in-place searching.
   - Guarded by:
     - `step_target_mask[hero.y, hero.x] = False` in `step_to_dead_end`.
     - `self.last_searched_pos` check in policy: a coordinate cannot be searched consecutively without moving.
     - 50-turn cooldown on stagnation decay.

3. **Weapon Parsing Grammar (`(weapon in hand)`)**:
   - NetHack formats one-handed weapons (e.g. Valkyrie's long sword) as `(weapon in hand)` (singular). Two-handed weapons use `(weapon in hands)`. Checking only `(weapon in hands)` marked all one-handed weapons as unequipped (`bare hands`).

4. **Item Naming Dialog Auto-Dismissal (`Call a ...` / ESC)**:
   - When identifying or picking up unidentified items, NetHack can prompt `Call a white potion: ` or `What do you want to call...`.
   - Directional navigation keys enter characters into the text buffer (0 turns), triggering NLE's 2,500 consecutive 0-turn step abort.
   - Intercepting `call a `, `call the `, and `what do you want to call` with ESC (`\x1b`) dismisses the naming dialog instantly.

5. **Encumbrance-Locked Nutrition**:
   - Autopicking heavy chain mails and scale mails induces `STRAINED` encumbrance, blocking eating (`"You can't do that while carrying so much stuff"`). `eat_carried_food` automatically drops unworn heavy armor before consuming food.

6. **Elbereth Ward Erosion Shield**:
   - Dust Elbereth scuffs when taking damage. Adapter drops coordinates from `self.elbereth_positions` upon receiving melee damage messages, and the policy forbids calling `wait()` while hostiles are adjacent.

---

## 3. Autonomous LLM Synthesis Campaign Status

- **Status**: **Active (Task ID: `task-3358`)**
- **Configuration**: 10 generations, 20 episodes per generation, 25,000 max turns, target depth 10.0, provider OpenRouter (`google/gemma-4-31b-it`).
- **Generation 1 Results (20 episodes)**:
  - **Average Depth**: **3.90**, **Max Depth**: **Depth 9**, **Average Turns**: 1,844.5.
  - **Synthesis**: Gemma-4-31b-it evaluated mortality logs, refined Elbereth engagement and ranged harassment, archived to `data/policies/gen_0001.py`.
- **Generation 2 Results (20 episodes)**:
  - **Average Depth**: 1.50, **Max Depth**: Depth 4, **Average Turns**: 3,786.5.
- **Generation 3 Results (20 episodes)**:
  - **Average Depth**: 1.90, **Max Depth**: Depth 5, **Average Turns**: 4,687.6.
- **Generation 4 Results (20 episodes)**:
  - **Average Depth**: **3.05**, **Max Depth**: **Depth 6**, **Average Turns**: 3,933.2.
  - **Recovery**: Elevation of `stairs_down_known` immediately restored deep floor traversal across the batch.
- **Generation 5 Results (20 episodes)**:
  - **Average Depth**: **3.65**, **Max Depth**: **Depth 6**, **Average Turns**: 1,820.5.
  - **Synthesis**: Gemma integrated BUC testing with `obs.epistemic.has_untested_items` and safe armor equipping via `obs.epistemic.can_safely_wear_armor`.
- **Generation 6 Results (20 episodes)**:
  - **Average Depth**: **3.15**, **Max Depth**: **Depth 7**, **Average Turns**: 1,974.3.
  - **Milestone**: Reached Depth 7, archived to `data/policies/gen_0006.py`.
- **Generation 7 Results (20 episodes)**:
  - **Average Depth**: **3.60**, **Max Depth**: **Depth 11** (Episode 7 penetrated to Depth 11!), **Average Turns**: 3,347.4.
  - **Synthesis**: Gemma synthesized Gen 7 policy emphasizing floating eye melee avoidance and aggressive hunger prevention, archived to `data/policies/gen_0007.py`.
- **Generation 8 Results (20 episodes)**:
  - **Average Depth**: **4.05**, **Max Depth**: **Depth 7**, **Average Turns**: 2,655.9.
  - **Milestone**: Reached new peak batch average depth of 4.05! 9 out of 20 episodes penetrated to Depths 5–7, and 18/20 survived Depth 1. Archived to `data/policies/gen_0008.py`.
- **Generation 9 Results (20 episodes)**:
  - **Average Depth**: **3.20**, **Max Depth**: **Depth 6**, **Average Turns**: 1,442.1.
  - **Milestone**: Archived to `data/policies/gen_0009.py`.
- **Generation 10 Results (20 episodes)**:
  - **Average Depth**: 1.40, **Max Depth**: Depth 4, **Average Turns**: 4,701.3.
  - **Autopsy Diagnosis**: Gen 9 policy omitted `and obs.dungeon.tile_type == "corridor"` and lacked `obs.dungeon.can_harvest_poison`, calling `harvest_poison_res()` continually from turn 1. Furthermore, solver fallbacks in `NetHackAdapter` lacked `adjacent_closed_door`, stalling navigation behind closed doors.
  - **Fixes Applied**:
    1. Added `elif obs_prev.dungeon.adjacent_closed_door: return self.step(Action(name="open_door"))` to both `test_altar_buc` and `harvest_poison_res` adapter fallbacks in `NetHackAdapter`.
    2. Explicitly gated `harvest_poison_res()` with `obs.dungeon.can_harvest_poison` in `data/latest_policy.py`.
- **Campaign 2 Results (Task `task-3156`, 10 Gens x 20 Eps = 200 Episodes)**:
  - **Gen 1**: Avg Depth 3.40, Max Depth 7, Avg Turns 2,522.1
  - **Gen 2**: Avg Depth 3.75, Max Depth 7, Avg Turns 1,746.9
  - **Gen 3**: Avg Depth 2.95, Max Depth 5, Avg Turns 1,960.4
  - **Gen 4**: Avg Depth 3.25, Max Depth 6, Avg Turns 2,610.5
  - **Gen 5**: Avg Depth 3.30, Max Depth 5, Avg Turns 1,970.4
  - **Gen 6**: Avg Depth 3.80, Max Depth 8, Avg Turns 2,065.1
  - **Gen 7**: Avg Depth 3.55, Max Depth 6, Avg Turns 2,903.8
  - **Gen 8**: Avg Depth 3.65, **Max Depth 10**, Avg Turns 3,187.0 (Episode reached Depth 10!)
  - **Gen 9**: Avg Depth 3.60, Max Depth 7, Avg Turns 1,721.0
  - **Gen 10**: Avg Depth 2.90, Max Depth 7, Avg Turns 2,497.6
  - **Campaign Stability**: Zero batch collapses (averages stayed 3.0–3.8 throughout all 10 generations, with episodes penetrating to Depths 7, 8, and 10). Excalibur forging verified in telemetry (`the blessed rustproof +1 Excalibur`), alongside equipped helmets, cloaks, and boots.
  - **Autopsy Diagnoses & Fatal Traps Uncovered**:
    1. **Mindless & Exploding Hazard Targeting**: `melee_attack_hostile` blindly attacked the first adjacent glyph, including gas spores (causing instant fatal 4d6 explosions) and floating eyes (causing 50-turn paralysis).
    2. **Elbereth Resistance Waiting Trap**: Humanoids and orcs (Uruk-hai, orcs, elves, soldiers) ignore Elbereth. In `handle_combat`, when standing on Elbereth with low HP, the policy yielded `wait()`, allowing Uruk-hai to shoot poisoned arrows and melee strike the hero without resistance.
  - **Fixes Applied (Committed & Verified)**:
    1. Added `adjacent_floating_eye` and `hostile_ignores_elbereth` to `CombatView` and `ALLOWED_PREDICATES`.
    2. Shielded `NetHackAdapter.step(Action(name="melee_attack_hostile"))` to strictly skip floating eyes and gas spores, and step away rather than approaching them in melee.
- **Campaign 3 Results (Task `task-3296`, Completed - 10 Gens x 20 Eps = 200 Episodes)**:
  - **Gen 1**: Avg Depth **3.65**, Max Depth 7, Avg Turns 2,979.2
  - **Gen 2**: Avg Depth **3.85**, **Max Depth 10** (Episode reached Depth 10!), Avg Turns 2,416.8
  - **Gen 3**: Avg Depth **3.35**, Max Depth 6, Avg Turns 2,193.5
  - **Gen 4**: Avg Depth 2.70, Max Depth 5, Avg Turns 3,632.7
  - **Gen 5**: Avg Depth **3.60**, Max Depth 6, Avg Turns 2,810.9
  - **Gen 6**: Avg Depth **3.60**, **Max Depth 9** (Episode reached Depth 9!), Avg Turns 4,218.9
  - **Gen 7**: Avg Depth **3.60**, Max Depth 6, Avg Turns 2,048.2
  - **Gen 8**: Avg Depth **3.50**, Max Depth 7, Avg Turns 2,124.6
  - **Gen 9**: Avg Depth **3.30**, Max Depth 6, Avg Turns 1,862.8
  - **Gen 10**: Avg Depth **3.55**, Max Depth 8, Avg Turns 2,302.9
  - **Autopsy Diagnoses & Key Discoveries**:
    1. **Inverted Room-to-Chokepoint Logic**: Tactical retreat checked `step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()`. When in an open room fighting speed-22 giant bats, it stepped away rather than fleeing to a chokepoint doorway, allowing bats to double-move and kill the hero. Corrected to `step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()`.
    2. **Mixed Hazard Swarms**: When a floating eye or gas spore was present alongside non-passive hostiles (sewer rats, geckos, zombies), the policy previously refused to attack, repeatedly yielding `step_away_from_hostile()` while rats cornered and killed the hero. Fixed to engrave Elbereth or attack the other non-passive hostiles (which the adapter safely targets while skipping the floating eye).
  - **Fixes Applied & Verified**:
    1. Corrected chokepoint retreat direction in [`data/latest_policy.py`](file:///home/moose/git/lox/data/latest_policy.py).
    2. Added mixed hazard swarm engagement with Elbereth warding in [`data/latest_policy.py`](file:///home/moose/git/lox/data/latest_policy.py).
    3. Archived synthesized policies `gen_0009.py` and `gen_0010.py`.
- **Campaign 4 Results (Task `task-3358`, Gens 1–4 Completed)**:
  - **Gen 1**: Avg Depth **3.50**, Max Depth 8, Avg Turns 2,210.5
  - **Gen 2**: Avg Depth **3.85**, Max Depth 7, Avg Turns 1,940.2
  - **Gen 3**: Avg Depth **3.80**, Max Depth 8, Avg Turns 2,150.8
  - **Gen 4**: Avg Depth **3.85**, Max Depth 8, Avg Turns 2,340.1
  - **Strategic Pause & Upgrade**: Stably held high averages near 4.0, but identified 3 critical macro gaps needed to unlock Depths 8–15:
    1. **Floor Loot Scooping (`step_to_loot`, `has_nearby_loot`)**: Monsters dropped helmets, mail, boots, and cloaks, but heroes never stepped on death tiles. Added `has_nearby_loot` and `step_to_loot` so autopickup collects armor, enabling `wear_armor` to drop AC towards negative numbers.
    2. **Offensive Wand Zapping (`zap_offensive_wand`, `has_offensive_wand`)**: Enabled zapping wands of striking, magic missile, fire, cold, sleep, and lightning at range $\ge 2$ to safely neutralize high-threat hostiles and passive hazards.
    3. **Emergency Panic Teleportation (`read_scroll_teleport`, `zap_wand_teleport`, `has_panic_escape`)**: Teleporting out of lethal surrounds or $< 25\%$ HP situations to reset the encounter.

- **Campaign 5 (Parallel Multiprocessing Synthesis & Ground-Truth Scoring)**:
  - **Gen 1**: Avg Depth **3.05**, Max Depth 6, Avg Turns 2,776.2, Avg Score 548.4, Peak Score **1,670** (Eval time: 30.6s)
  - **Gen 2**: Avg Depth 2.95, Max Depth 7, Avg Turns 2,567.3, Avg Score 395.5, Peak Score 1,069 (Eval time: 35.6s)
  - **Gen 3**: Avg Depth 2.70, Max Depth 7, Avg Turns 4,175.9, Avg Score 482.6, Peak Score 1,088 (Eval time: 46.3s)
  - **Gen 4**: Avg Depth 2.20, Max Depth 5, Avg Turns 4,345.7, Avg Score 494.6, Peak Score **1,702** (Eval time: 61.6s)
  - **Gen 5**: Avg Depth 2.20, Max Depth 6, Avg Turns 4,968.2, Avg Score 377.2, Peak Score 1,350 (Eval time: 62.1s)
  - **Gen 7**: Avg Depth 2.10, Max Depth 4, Avg Turns 6,176.9, Avg Score 465.0 (Eval time: 70.2s)
  - **Gen 8**: Avg Depth 2.50, Max Depth 8, Avg Turns 4,301.8, Avg Score 464.1 (Eval time: 54.1s)
  - **Gen 9**: Avg Depth 2.30, Max Depth 5, Avg Turns 4,208.0, Avg Score 406.0 (Eval time: 52.8s)
  - **Gen 10**: Avg Depth 2.45, Max Depth 8, Avg Turns 5,266.1, Avg Score 502.2 (Eval time: 64.7s)
  - **Comprehensive Telemetry Autopsy & Breakthrough Fixes**:
    1. **Loot Scooping Ping-Pong Loop**: 120,000+ ticks were lost to dropped weapons/ammo that autopickup excludes. Fixed by adding `self.looted_tiles` and pruning non-autopicked item types from `loot_chars`.
    2. **Fountain Vanishing Dipping Loop**: When fountains dried up, `known_fountain_pos` was not cleared, causing endless `#dip` failures. Added `fountain_vanished` detection and reset.
    3. **Interactive Dialog Auto-Dismissals**: Uncovered unhandled prompts (`Call an emerald...`, `What do you want to eat`, `What do you want to throw`) causing 2,500-step 0-turn aborts. Added full ESC auto-dismissals in `_dismiss_more`.
    4. **Numba JIT Acceleration & Disk Caching**: JIT-compiled dead ends mask kernel with `cache=True`, cutting calculation time by 7x (~58 µs/call). Added `SpatialEngine.warmup()`.
    5. **Granular Episode Mortality Telemetry**: Added `killer`, `ac_at_death`, `hp_at_death`, `max_hp_at_death`, and `excalibur_forged` to DuckDB and Parquet pipelines.

- **Campaign 6 Results & Autopsy Breakthroughs**:
  - **Gen 1**: Avg Depth 2.55, Max Depth 6, Avg Turns 1,790.6
  - **Gen 2**: Avg Depth 1.90, Max Depth 4, Avg Turns 1,688.6
  - **Gen 3**: Avg Depth 1.95, Max Depth 4, Avg Turns 1,733.2
  - **Gen 4**: Avg Depth 2.25, Max Depth 5, Avg Turns 1,736.9
  - **Gen 5**: Avg Depth 1.70, Max Depth 4, Avg Turns 1,741.1
  - **Gen 6**: Avg Depth 2.05, Max Depth 4, Avg Turns 1,601.4
  - **Gen 7**: Avg Depth 1.75, Max Depth 3, Avg Turns 1,759.5
  - **Autopsy Diagnoses & Fatal Traps Uncovered**:
    1. **Intermediate Keystroke Dialog Cancellation Trap**:
       - When `_dismiss_more` was expanded with ESC dismissals for `"what do you want to eat"`, `_step_sequence([e, food_slot])` sent `e`, `_dismiss_more` immediately saw the item prompt and pressed ESC (printing `"Never mind."`), cancelling the eat action before the slot key was delivered.
       - At hunger $\ge 2$ (~turn 750), heroes repeatedly attempted `eat_carried_food` for over 1,200 zero-turn steps until NLE aborted the episode at ~1,700 steps.
       - **Fix**: Added `is_intermediate = (idx < len(action_indices) - 1)` to `_step_sequence` and `_dismiss_more`, preserving intermediate prompts across multi-key actions (`eat_carried_food`, `wear_armor`, `zap_offensive_wand`, `throw_dagger`).
    2. **Consecutive Zero-Turn Non-Movement Obstacle Poisoning**:
       - `self.consecutive_zero_turns >= 2` previously marked the tile in the direction of the last movement step as blocked, even when the zero-turn actions were inventory queries or failed eating.
       - **Fix**: Gated with `is_step_direction`.
    3. **False MaxTurnsReached Classification**:
       - Aborted runs (`StepStatus.ABORTED`) were misclassified as `MaxTurnsReached`, blinding the synthesis engine.
       - **Fix**: Added strict check for `obs.hero.hp <= 0` (fatality), then `StepStatus.ABORTED` (`Aborted / ZeroProgress`), and onlyalive runs reaching `max_turns` marked as `MaxTurnsReached`.
    4. **Hero Occlusion Mask Corruption**:
       - `self.known_chars` substitution in `_compute_dead_ends_mask` now strictly checks `(chars == ord("@")) & (self.known_chars > 0)`.

- **Campaign 7 (Active)**:
  - **Configuration**: 10 generations, 20 episodes/gen, 25,000 max turns, 10 workers, OpenRouter `google/gemma-4-31b-it`.
  - **Target**: Average Depth $\ge 10.0$ with working nutrition, armor equipping, and truthful mortality telemetry.

## 4. Longer-Term Goals (Roadmap to Depth 10+)

1. **Autonomous LLM Synthesis Loop Execution**:
   - Continue running iterative generations until batch average depth $\ge 10.0$.
2. **Main Dungeon vs Gnomish Mines Branch Steering**:
   - The Gnomish Mines branch (Depths 2–4) is dark and infested with gnome wand wielders. Prefer descending the main dungeon staircase down to Depth 10 before exploring deep Mines.
3. **Container & Bag Stash Management**:
   - Looting sacks and chests for additional scrolls and potions.
4. **Campaign Completion Verification**:
   - Batch average depth $\ge 10.0$ triggers `[CAMPAIGN GOAL ACHIEVED]`.
   - Record final DuckDB telemetry and emit `<!-- GOAL_COMPLETE -->`.

