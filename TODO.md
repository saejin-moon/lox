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
| `bench_fixes_8` | 2.70 | 5 | 77.7% of turns lost to phantom doors (`+` spellbooks vs closed doors). |
| `bench_fixes_10` | 3.60 | 5 | Solved DL1 door stalling; identified 60.9% turns trapped in corridor retreat loops. |
| `bench_fixes_12` | 3.20 | 6 | Fixed generator zero-yield hanging in HTN subroutines. |
| `bench_fixes_13` | 3.60 | **9** | **Depth 9 reached!** Discovered hero self-block navigation bug and mutual recursion. |
| `bench_fixes_14` | 3.20 | 6 | Saved 17,000 turns by fixing altar ASCII `@` detection. |
| `bench_fixes_15` | 3.50 | 7 | Eliminated 0-turn locked door kick loop with kick count caps. |
| `bench_fixes_16` | **4.00** | 6 | 100% DL1 clearance; 3 episodes reached DL6. Diagnosed Excalibur dipping prompt failure. |
| `bench_fixes_17` | 3.60 | 7 | Two episodes reached DL7. Uncovered 23.5k in-place search decay loop at `(5, 46)`. |
| `bench_fixes_18` | 3.30 | **8** | Reached Depth 8! 0 timeouts at 25k turns; validated `(weapon in hand)` tracking and eliminated in-place search stall. Diagnosed item naming dialog trap (`Call a ...`). |

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

## 3. Immediate Next Steps (Priority 1)

1. **Verify `bench_fixes_18`**:
   - Confirm completion of `bench_fixes_18` via `manage_task` or DuckDB query.
   - Verify that no episodes timeout with `MaxTurnsReached` on DL1.
   - Inspect Excalibur forging occurrences and confirm `weapon_in_hand` tracks properly.
2. **Review Tactical Survival at Depths 6–8**:
   - Check death reasons in DuckDB for Depths 6–8 (e.g., dwarf aklys/spear strikes, soldier ant poison, wands).
   - Verify healing potion quaffing (`quaff_healing`) triggers reliably at `< 50% HP` during deep combat encounters.
3. **Commit & Push Current Stable Checkpoint**:
   - Keep git state clean and pushed to `master` with informative commit messages.

---

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
