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

- **Status**: **Active (Task ID: `task-3296`)**
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
- **Campaign 3 Results (Task `task-3296`, In Flight - Gen 9/10 Running)**:
  - **Gen 1**: Avg Depth **3.65**, Max Depth 7, Avg Turns 2,979.2
  - **Gen 2**: Avg Depth **3.85**, **Max Depth 10** (Episode reached Depth 10!), Avg Turns 2,416.8
  - **Gen 3**: Avg Depth **3.35**, Max Depth 6, Avg Turns 2,193.5
  - **Gen 4**: Avg Depth 2.70, Max Depth 5, Avg Turns 3,632.7
  - **Gen 5**: Avg Depth **3.60**, Max Depth 6, Avg Turns 2,810.9
  - **Gen 6**: Avg Depth **3.60**, **Max Depth 9** (Episode reached Depth 9!), Avg Turns 4,218.9
  - **Gen 7**: Avg Depth **3.60**, Max Depth 6, Avg Turns 2,048.2
  - **Gen 8**: Avg Depth **3.50**, Max Depth 7, Avg Turns 2,124.6
  - **Gen 9**: Currently running Episode 16 of 20.
  - **Key Progress**: Gemma adopted `obs.combat.hostile_ignores_elbereth` into `data/policies/gen_0008.py` and elevated emergency healing on Elbereth (`hp_frac < 0.70`). Zero paralysis deaths from floating eyes or gas spores recorded.

## 4. Longer-Term Goals (Roadmap to Depth 10+)

1. **Launch Autonomous LLM Synthesis Loop**:
   - Once all manual tactical and navigational primitives are empirically validated, run:
     ```bash
     uv run python -u -m scripts.run_synthesis \
       --provider openrouter \
       --model google/gemma-4-31b-it \
       --generations 15 \
       --eval-episodes 10 \
       --max-turns 25000 \
       --target-depth 10.0 \
       --policy-path data/latest_policy.py
     ```
2. **Tactical Offensive & Utility Wands**:
   - Implement `zap_wand(direction, slot)` for wands of striking, digging, sleep, fire, and lightning.
   - Zapping sleep or striking at ranged enemies (dwarves, leocrottas) dramatically reduces incoming damage at Depths 6–10.
3. **Scroll & Potion Identification & Use**:
   - Read uncursed scrolls of identify/teleportation.
   - Emergency teleportation (`read_scroll_teleport`) when surrounded by high-damage packs at Depths 7–10.
4. **Mines vs Main Dungeon Branch Selection**:
   - The Gnomish Mines branch (Depths 2–4) is dark and filled with gnome/dwarf wand wielders.
   - Add branch-awareness: prefer clearing the main dungeon staircase down to Depth 10 before exploring deep Mines.
5. **Campaign Completion Verification**:
   - Batch average depth $\ge 10.0$ triggers `[CAMPAIGN GOAL ACHIEVED]`.
   - Record final DuckDB telemetry and emit `<!-- GOAL_COMPLETE -->`.
