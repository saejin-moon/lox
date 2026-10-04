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

### Recent Breakthroughs & Engine Hardening
- **Safe Melee Engagement in Mixed Hazard Swarms (`has_safe_melee_target`)**: When facing a passive hazard (gas spore or floating eye) alongside active attackers (newt, jackal, orc), `obs.combat.has_safe_melee_target` allows the hero to strike the active attacker in melee while the adapter skips the passive hazard, preventing the hero from idling and dying to minor pests.
- **Passive Hazard Pathfinding Masking & Safe Cornered Wait (Invariant 22)**: Fixed pathfinding routing through passive/exploding hazards (floating eyes, gas spores, molds, jellies) by masking them out of `walkable_nav`. When cornered or unable to retreat, `step_away_from_hostile()` yields `wait()` rather than bumping into them.
- **Dynamic Fast Monster Discrimination (Invariant 23)**: Ground-truth `permonst.mmove > 12` dynamic check plus foxes (speed 15), coyotes, and jaguars flags fast predators for immediate corridor chokepoints (`step_to_chokepoint()`).
- **Full-Floor Persistent Topological Memory (Invariant 49)**: Fixed line-of-sight amnesia where `walkable_nav` and closed doors were lost when moving out of FOV. Augmenting pathfinding with `self.visited` and `self.known_chars` preserves 100% floor connectivity.
- **Locked Door Circuit Breaker & Chokepoint Blocking (Invariant 48)**: Locked doors in shops or after 6 failed kicks are added to `self.blocked_tiles`. Universal $\ge 4$ zero-turn circuit breaker forces `wait()` (`.`), eliminating all `StepStatus.ABORTED` occurrences.
- **OpenRouter Exponential Backoff Resilience (Invariant 47)**: 6-attempt exponential backoff retry with jitter ensures uninterrupted overnight synthesis.

### Current Campaign Status
- **Campaign 18 (Active)**:
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

