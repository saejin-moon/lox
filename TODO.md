# LOX 2.0 Campaign Action Plan & Roadmap (`TODO.md`)

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

- **Campaign 7 Results & Autopsy Discoveries**:
  - **Gen 1**: Avg Depth 2.90, Max Depth 8, Avg Turns 5,386.4
  - **Gen 2**: Avg Depth 2.25, Max Depth 5, Avg Turns 2,790.2
  - **Gen 3**: Avg Depth 2.95, Max Depth 5, Avg Turns 3,629.1
  - **Gen 4**: Avg Depth 3.00, Max Depth 7, Avg Turns 3,069.2
  - **Key Breakthroughs Verified**:
    - Abort elimination: 0 zero-progress aborts! 100% of fatalities are genuine combat deaths.
    - Turns surged 3x from 1,700 to 5,300+ turns per episode.
  - **Autopsy Diagnoses & Fatal Traps Uncovered**:
    1. **Dead-End No-Hostile Elbereth/Wait Oscillation**:
       - Detailed tick inspection of episodes spending 6,000–8,000 turns on DL1/DL2 (e.g. `g003_e003` with 8,274 turns on DL2) revealed **3,945 `wait` actions** and **3,778 `step_away_from_hostile` actions** executed when `closest_hostile_dist == 999.0` (zero enemies in FOV).
       - In `handle_dead_end`, when frontiers and stairs down were not visible, the policy fell back to `step_away_from_hostile()`.
       - In `NetHackAdapter.step()`, when `step_away_from_hostile()` had no hostile target, it fell through to `engrave_dust_elbereth()`, followed by `wait()`.
       - **Fix Applied**: In `NetHackAdapter`, if `not obs_prev.combat.closest_hostile_pos`, `step_away_from_hostile` immediately falls back to `step_to_stairs_down`, `step_to_frontier`, `search` (if standing on a dead end/perimeter candidate), or `step_to_dead_end`. In the policy `handle_dead_end`, fallback to `search()` to uncover secret doors.
    2. **Host Parallelism & I/O Optimization**:
       - Host has 22 CPU cores (`os.cpu_count() == 22`). Scaled worker pool from 10 to 20 (`--workers 20`), allowing all 20 episodes to evaluate in parallel in a single wave.
       - Buffered Parquet logging to `flush_interval=5000` (cutting disk writes by 50x) and eliminated per-step `signal.alarm` syscalls.

- **Campaign 8 Results (Completed - 10 Gens x 20 Eps = 200 Episodes in 14 mins)**:
  - **Gen 1**: Avg Depth **3.05**, Max Depth 6, Avg Turns 2,868.2
  - **Gen 2**: Avg Depth **3.35**, Max Depth 8, Avg Turns 2,702.3
  - **Gen 3**: Avg Depth **3.40**, **Max Depth 9**, Avg Turns 1,914.3
  - **Gen 4**: Avg Depth **3.60**, Max Depth 8, Avg Turns 3,034.5
  - **Gen 5**: Avg Depth **4.60**, Max Depth 8, Avg Turns 2,119.9 (**Peak Generation: 14/20 episodes reached Depths 4–8!**)
  - **Gen 6**: Avg Depth **3.65**, **Max Depth 9**, Avg Turns 3,592.3
  - **Gen 7**: Avg Depth **3.55**, Max Depth 6, Avg Turns 1,738.1
  - **Gen 8**: Avg Depth **3.60**, Max Depth 8, Avg Turns 2,878.9
  - **Gen 9**: Avg Depth **3.35**, Max Depth 6, Avg Turns 3,208.8
  - **Gen 10**: Avg Depth **3.50**, Max Depth 7, Avg Turns 3,307.6
  - **Major Milestones Achieved**:
    - **Plateau Broken**: Every generation consistently held $\ge 3.0$ avg depth across 200 episodes.
    - **Gen 5 Breakthrough**: Reached **4.60 Average Depth**, with multiple episodes reaching Depths 7, 8, and 9.
    - **Ultra-Fast Parallel Throughput**: Evaluated 200 episodes of up to 25k steps each and 10 LLM synthesis cycles in just **14 minutes total** (~$0.0207 USD total cost) using 20 concurrent worker processes.
  - **Autopsy Diagnoses & Key Discoveries**:
    1. **Staircase Combat Interception Trap**:
       - Detailed tick inspection of `g006_e009` (which reached Depth 9 at turn 2,877) revealed that the hero reached the stairs down to Depth 10 at turn 2,882 (`step_to_stairs_down`). A homunculus escaped downstairs to Depth 10, but because hostiles (a wolf and lizard) were in FOV, `obs.combat.hostile_count_fov > 0` intercepted execution into `handle_combat()`.
       - Because `descend()` was only positioned at the bottom of navigation (after combat), the hero stood on the staircase fighting for 9 turns until dying on the stairs, rather than taking the 1-turn staircase descent to escape into Depth 10!
       - **Fix**: Elevate `descend()` to execute immediately whenever `obs.spatial.standing_on_stairs_down and not obs.status.is_levitating`, including within `handle_combat()` as an escape.

- **Campaign 9 Results (Completed - 10 Gens x 20 Eps = 200 Episodes in 18 mins)**:
  - **Gen 1**: Avg Depth **3.35**, Max Depth 7, Avg Turns 3,829.1
  - **Gen 2**: Avg Depth **3.20**, Max Depth 7, Avg Turns 3,547.3
  - **Gen 3**: Avg Depth **3.00**, Max Depth 6, Avg Turns 3,843.3
  - **Gen 4**: Avg Depth **4.05**, Max Depth 8, Avg Turns 2,805.5
  - **Gen 5**: Avg Depth **3.30**, Max Depth 8, Avg Turns 3,760.4
  - **Gen 6**: Avg Depth **3.70**, Max Depth 7, Avg Turns 2,708.6
  - **Gen 7**: Avg Depth **4.05**, Max Depth 7, Avg Turns 2,155.8
  - **Gen 8**: Avg Depth **4.15**, Max Depth 7, Avg Turns 4,290.4
  - **Gen 9**: Avg Depth **4.35**, **Max Depth 10** (**DEPTH 10 REACHED! Episode `g009_e017` penetrated to Depth 10 at turn 8,612 with AC -2 and 51 HP!**)
  - **Gen 10**: Avg Depth **3.45**, Max Depth 8, Avg Turns 2,549.2
  - **Major Milestones Achieved**:
    - **Campaign Depth 10 Milestone**: `g009_e017` successfully descended to **Dungeon Level 10**! Equipped full armor (AC -2) and held 51/65 HP.
    - **Sustained 4.0+ Depth Generations**: Multiple generations reached average depths > 4.0 (Gen 4: 4.05, Gen 7: 4.05, Gen 8: 4.15, Gen 9: 4.35).
  - **Autopsy Diagnoses & Key Discoveries**:
    1. **Tactical Locked Door Bumping Abort on Depth 10**:
       - Detailed tick inspection of `g009_e017` on Depth 10 revealed that at turn 2,589, `step_away_from_hostile` selected an adjacent locked door as `best_tile` because it maximized distance from the hostile.
       - However, `step_away_from_hostile` dispatched `Action(name="step_direction", direction=best_tile)` directly rather than calling `_step_or_breach(obs_prev, dy, dx)`.
       - Walking into a locked door returned `"This door is locked."` and consumed 0 turns. Because the hero position did not change and the door remained in `walkable_nav`, it repeated 2,500 consecutive zero-turn steps until NLE aborted the episode at turn 8,612 while the hero was alive with 51 HP!
       - **Fix Applied**: Upgraded `step_away_from_hostile` in `NetHackAdapter` to strictly prefer open walkable floor tiles over closed/locked doors, and route any door retreat through `self._step_or_breach(obs_prev, dy, dx)` so locked doors are breached with kicking instead of endless zero-turn walking.

- **Campaign 10 Results (Completed - 10 Gens x 20 Eps = 200 Episodes in 12 mins)**:
  - **Gen 1**: Avg Depth **3.65**, **Max Depth 9**, Avg Turns 2,588.5
  - **Gen 2**: Avg Depth **3.40**, Max Depth 6, Avg Turns 2,055.4
  - **Gen 3**: Avg Depth **3.10**, Max Depth 6, Avg Turns 3,214.8
  - **Gen 4**: Avg Depth 2.85, Max Depth 6, Avg Turns 3,302.7
  - **Gen 5**: Avg Depth **4.40**, **Max Depth 9**, Avg Turns 2,927.5
  - **Gen 6**: Avg Depth **3.75**, Max Depth 8, Avg Turns 3,352.6
  - **Gen 7**: Avg Depth **3.90**, Max Depth 8, Avg Turns 2,149.1
  - **Gen 8**: Avg Depth **3.85**, Max Depth 8, Avg Turns 2,225.8
  - **Gen 9**: Avg Depth 2.80, Max Depth 7, Avg Turns 3,505.6
  - **Gen 10**: Avg Depth **3.35**, Max Depth 7, Avg Turns 1,683.2
  - **Major Milestones Achieved**:
    - Reached **Depth 9** in both Gen 1 (`g001_e001`) and Gen 5 (`g005_e003`).
    - Ultra-fast execution: 200 episodes in 12 minutes (~1.2 min per generation) across 20 workers.
  - **Autopsy Diagnoses & Key Discoveries**:
    1. **Deep-Level Starvation Fainting (`You faint from lack of food`)**:
       - Detailed tick inspection of `g001_e001` on Depth 9 revealed that at turn 3,462, the hero began fainting from starvation and remained paralyzed for dozens of turns until death.
       - Why: The policy previously prioritized eating carried food rations over floor corpses, and only looked for corpses at `hunger_state >= 2` (hungry). By the time hunger reached 2, fresh monster corpses dropped earlier in combat had long since rotted away (decay takes only 30–50 turns). The hero consumed their carried food rations early on, leaving zero food reserves for Depths 7–10.
       - **Fix Applied**: Updated `run()` to opportunistically consume safe fresh monster corpses right after combat whenever `hunger_state < 3`, preserving carried rations exclusively for deep-level hunger emergencies.

- **Campaign 12 Results (Completed - 10 Gens x 20 Eps = 200 Episodes in 14 mins)**:
  - **Gen 1**: Avg Depth **3.45**, Max Depth 7, Avg Turns 4,389.4
  - **Gen 2**: Avg Depth **3.50**, Max Depth 6, Avg Turns 2,610.4
  - **Gen 3**: Avg Depth **3.80**, Max Depth 7, Avg Turns 3,360.1
  - **Gen 4**: Avg Depth **3.30**, Max Depth 8, Avg Turns 2,846.7
  - **Gen 5**: Avg Depth **3.70**, Max Depth 8, Avg Turns 2,020.6
  - **Gen 6**: Avg Depth **3.70**, Max Depth 7, Avg Turns 2,996.7
  - **Gen 7**: Avg Depth **4.10**, **Max Depth 8**, Avg Turns 2,266.2
  - **Gen 8**: Avg Depth **3.80**, Max Depth 7, Avg Turns 2,909.2
  - **Gen 9**: Avg Depth **3.15**, Max Depth 6, Avg Turns 1,991.2
  - **Gen 10**: Avg Depth 2.60, Max Depth 6, Avg Turns 2,238.4
  - **Autopsy Diagnoses & Critical Engine Fixes Uncovered**:
    1. **`standing_on_elbereth` Timing Trap in `engrave_dust_elbereth`**:
       - `(obs_prev.hero.y, obs_prev.hero.x)` was previously added to `self.elbereth_positions` *after* `self._step_sequence(seq)` executed.
       - Inside `_step_sequence`, `_extract_obs` checked `(y, x) in self.elbereth_positions`, which was still False.
       - The observation returned to the policy reported `standing_on_elbereth: False`.
       - Because the policy saw `not obs.combat.standing_on_elbereth`, it immediately yielded `engrave_dust_elbereth` on the very next turn, wiping the first engraving out (`"You wipe out the message that was written in the dust"`) and burning 2 full turns while being attacked.
       - **Fix Applied**: Pre-populate `self.elbereth_positions.add((obs_prev.hero.y, obs_prev.hero.x))` before executing the keystroke sequence so `obs.combat.standing_on_elbereth` is immediately `True`.
    2. **Premature Elbereth Discard on Monster Attacks**:
       - In NetHack, monsters hitting or missing the hero does not erase dust Elbereth. However, line 940 discarded `(obs.hero.y, obs.hero.x)` upon receiving `"hits!"`, `"bites!"`, `"stings!"`, etc.
       - This caused `standing_on_elbereth` to flip to False the moment an attacker struck, triggering endless engrave -> hit -> discard -> engrave loops.
       - **Fix Applied**: Discard Elbereth strictly on actual erasure/smudge messages (`"wipe out the message"`, `"wiped out"`, `"rubbed out"`, `"scuffed"`, `"erased"`, `"fades"`).
    3. **Comprehensive Elbereth-Immune Monster Species Coverage (`IGNORES_ELBERETH_SPECIES`)**:
       - Goblins, hobgoblins, gnomes, dwarves, ogres, trolls, giants, zombies, mummies, and vampires ignore Elbereth or throw ranged missiles. Previously, `hostile_ignores_elbereth` only checked for `"orc"`, `"uruk"`, `"elf"`, `"human"`.
       - Expanded `IGNORES_ELBERETH_SPECIES` and check both `closest_name` and adjacent monsters so the hero never wastes turns engraving dust Elbereth when facing immune hostiles.
    4. **Granular Killer & Stat Telemetry at Episode Death**:
       - Episodes previously logged empty killer strings because death messages were cleared on the terminal screen.
       - Added tick-level persistence for `last_known_hostile`, `last_valid_ac`, and `last_valid_max_hp`, accurately logging killer species and real AC at death into DuckDB.

    5. **`ThreadPoolExecutor` Shutdown Deadlock & Dry-Run Test Fix**:
       - In Python, `ThreadPoolExecutor.__exit__` calls `self.shutdown(wait=True)`. When candidate policies entered zero-yield while loops during multi-scenario dry runs, the worker thread ran forever and `executor.shutdown(wait=True)` deadlocked the main synthesis process, consuming 100% CPU for 40+ minutes.
       - Replaced `ThreadPoolExecutor` with a signal-based timeout runner (`signal.setitimer(signal.ITIMER_REAL, 1.0)`) in the main thread (and process termination fallback), which raises `TimeoutError` directly in the infinite loop bytecode, terminating it in milliseconds without thread lingering.
       - Fixed `floating_eye_combat` test scenario in `AuthorAgent` which previously omitted `adjacent_floating_eye=True`, causing false invariant rejections.

    6. **Four-Layer Anti-Infinite-Loop Execution & Worker Protection Architecture**:
       - Built and validated defense-in-depth protections across all execution layers:
         1. **AST `LoopGuardTransformer`**: Every `while` and `for` loop in compiled policies is automatically instrumented with `_guard.tick()`. If any loop runs >2,000 iterations without yielding an action, `RuntimeError` is raised in $< 1\text{ms}$.
         2. **`PolicyRunner.send(obs)` Safe Fallback**: Intercepts `Infinite loop detected` `RuntimeError`, safely falling back to `Action(name="wait")` to keep the episode moving.
         3. **`MAX_EPISODE_WALL_SEC = 90.0`**: Enforces a strict 90s wall-clock ceiling per episode to catch unexpected C-level NLE hangs.
         4. **`mp.Pool` Batch Ceiling (`180.0s`)**: Dispatches worker processes via `map_async` with a 180s timeout, automatically executing `pool.terminate()` and `pool.join()` to eliminate zombie worker processes.

- **Campaign 13 Results (Completed - 10 Gens x 20 Eps = 200 Episodes in 12 min)**:
  - **Gen 1**: Avg Depth **4.35**, Max Depth 8, Avg Turns 4,170.9
  - **Gen 2**: Avg Depth 3.65, Max Depth 7, Avg Turns 2,510.4
  - **Gen 3**: Avg Depth 3.45, Max Depth 7, Avg Turns 2,230.1
  - **Gen 4**: Avg Depth 3.80, Max Depth 7, Avg Turns 2,640.8
  - **Gen 5**: Avg Depth 3.55, **Max Depth 9**, Avg Turns 2,980.2
  - **Gen 6**: Avg Depth 3.50, Max Depth 8, Avg Turns 2,410.5
  - **Gen 7**: Avg Depth 3.75, Max Depth 7, Avg Turns 2,350.2
  - **Gen 8**: Avg Depth 3.30, **Max Depth 9**, Avg Turns 2,820.6
  - **Gen 9**: Avg Depth 3.40, Max Depth 7, Avg Turns 2,190.4
  - **Gen 10**: Avg Depth 3.70, Max Depth 7, Avg Turns 2,610.7
  - **Key Diagnoses**: Telemetry analysis showed 24 deaths to gnomes/gnome lords in `dnum = 2`. Discovered that NetHack's internal `dungeon.def` branch ID for the Gnomish Mines is `dnum = 2` (was mapped to `"quest"`). The Mines branch has no stairs to lower Dungeons of Doom, trapping heroes in dark levels with dangerous gnome packs.
  - **Fixes Applied**:
    1. Fixed `BRANCH_NAMES` in `NetHackAdapter` so `dnum = 2` is `"mines"`.
    2. Added `mines_stairs_positions` tracking in `NetHackAdapter`: when descending into the Mines from Dungeons of Doom, the stair position is remembered and pruned from `known_stairs_down`.
    3. Updated policy to immediately ascend back to Dungeons of Doom when `obs.hero.dungeon_branch == "mines"`.
    4. **Staircase Persistence Invariant**: Fixed bug where `self.known_stairs_down` was reset to `None` when stairs were not in immediate FOV. Preserving `known_stairs_down` allows the hero to navigate to the stairs from anywhere on the floor without forgetting their location.

- **Campaign 14 Results (Completed - 10 Gens x 20 Eps = 200 Episodes in 18 min)**:
  - **Gen 1**: Avg Depth 3.20, Max Depth 8, Avg Turns 3,300.8
  - **Gen 2**: Avg Depth 3.65, Max Depth 8, Avg Turns 3,236.2
  - **Gen 3**: Avg Depth **4.00**, Max Depth 7, Avg Turns 2,071.8 (40% of batch reached Depths 5–7)
  - **Gen 4**: Avg Depth 3.70, **Max Depth 11**! (`e018` penetrated to **Depth 11** in 1,702 turns!)
  - **Gen 5**: Avg Depth **3.95**, **Max Depth 9**, Avg Turns 2,897.3 (`e001` reached DL9, `e003` DL8 with 1,938 score)
  - **Gen 6**: Avg Depth 3.70, Max Depth 6, Avg Turns 3,268.4
  - **Gen 7**: Avg Depth **3.90**, **Max Depth 9**, Avg Turns 3,770.1 (`e004` and `e011` reached DL9)
  - **Gen 8**: Avg Depth 3.70, Max Depth 8, Avg Turns 3,687.9 (`e005` reached DL8)
  - **Gen 9**: Avg Depth **3.95**, **Max Depth 10**!, Avg Turns 3,691.3 (`e012` reached **Depth 10** with 1,942 score)
  - **Gen 10**: Avg Depth 3.30, Max Depth 7, Avg Turns 4,342.8
  - **Token & Cost Accounting**: 10 LLM synthesis sessions, 137,669 total tokens, **$0.0220 USD** total cost.
  - **Key Diagnoses & Invariant 46**:
    - Discovered that involuntary floor transitions via trap doors (`^`) or holes drop the hero on arbitrary floor tiles (`.`), NOT stairs up (`<`).
    - `NetHackAdapter` previously assumed all floor arrivals happen on stairs up (`self.known_stairs_up = (y, x)` on turn 0), which caused heroes falling into the Mines to repeatedly execute `ascend` on open floor tiles (`"You can't go up here."`), aborting after 2,500 consecutive 0-turn steps.
    - Added `had_explicit_descent` tracking (only true if `descend` was taken), invalidated `known_stairs_up` upon trap door messages or `"You can't go up here."`, and added consecutive 0-turn shields to `ascend` falling back to `step_to_frontier`.

- **Campaign 15 Results (Completed - 10 Gens x 20 Eps = 200 Episodes in 16 mins)**:
  - **Gen 1**: Avg Depth 3.75, Max Depth 8, Avg Turns 2,072.1
  - **Gen 2**: Avg Depth 3.30, Max Depth 5, Avg Turns 3,148.3
  - **Gen 3**: Avg Depth 3.65, Max Depth 7, Avg Turns 2,479.1
  - **Gen 4**: Avg Depth **4.30**, **Max Depth 11**!, Avg Turns 3,480.3, Avg Score 561.2
  - **Gen 5**: Avg Depth 2.45, Max Depth 8, Avg Turns 2,733.6
  - **Gen 6**: Avg Depth 3.90, Max Depth 8, Avg Turns 2,193.4
  - **Gen 7**: Avg Depth 3.65, Max Depth 8, Avg Turns 3,247.0
  - **Gen 8**: Avg Depth 2.50, Max Depth 5, Avg Turns 3,146.0
  - **Gen 9**: Avg Depth **4.50**, **Max Depth 9**, Avg Turns 2,330.1, Avg Score 541.5
  - **Gen 10**: Avg Depth 3.40, Max Depth 6, Avg Turns 2,682.1
  - **Batch Total**: 200 episodes, 100.0% genuine combat deaths (0 aborts, 0 timeouts!), 140,064 tokens, **$0.0227 USD**.
  - **Diagnoses & Engine Hardening (Invariant 49)**:
    1. **Full-Floor Persistent Topological Memory & Doorway Connectivity (Invariant 49)**:
       - Inspection of episodes with 12,000–17,000 turns on DL 1–3 revealed heroes trapped in rooms, searching the same room perimeter up to 511 times.
       - Discovered that because `build_walkable_mask` only examined `raw_obs["chars"]` (which zeroes out tiles outside line-of-sight FOV) and `_get_doors_mask` only checked visible CMAP glyphs, `walkable_nav` forgot the corridor and doorway path back into earlier rooms once the hero moved away.
       - Consequently, `step_to_dead_end` failed to pathfind to distant unsearched candidates, decayed search counters locally, and re-searched the current room in an endless cycle.
       - `NetHackAdapter` now augments `walkable` and `walkable_nav` with `self.visited` and `self.known_chars` (`#`, `.`, `<`, `>`, `_`, `{`), incorporates known closed doorways (`ord("+")`) into `_get_doors_mask` and `_get_all_doors_mask`, and preserves all discovered level terrain in `_compute_dead_ends_mask`.

- **Campaign 16 (Active)**:
  - **Configuration**: 10 generations, 20 episodes/gen, 25,000 max turns, 20 workers, OpenRouter `google/gemma-4-31b-it`.
  - **Target**: Average Depth $\ge 20.0$.

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

