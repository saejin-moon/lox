"""
LOX Author Prompts: Complete NetHack Vocabulary, Class Agent Generator Paradigm, and Deep Telemetry Tools.
"""

from __future__ import annotations

from lox.dsl.schema import ALLOWED_ACTIONS, ENUM_CONSTANTS


def build_system_prompt() -> str:
    """System prompt defining the Class Agent generator architecture and full NetHack API."""
    actions = ", ".join(f"{a}()" for a in sorted(ALLOWED_ACTIONS))
    enums = ", ".join(f"{k}={v}" for k, v in ENUM_CONSTANTS.items())

    return f"""You are the LOX Policy Synthesizer, an autonomous empirical policy synthesis engine for NetHack.
Synthesize robust NetHack policies using an unconstrained, object-oriented Python generator class: `class Agent`.
The agent is instantiated fresh at the start of each episode. Every turn, its `run(self, obs)` method receives an `obs` object and yields an `Action`.

### Policy Architecture: Decoupled Modular Skills
The policy is structured around high-level goal determination and decoupled modular skills:
```python
class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.current_goal = "explore_and_dive"

    def determine_goal(self, obs) -> str:
        if obs.hero.dungeon_branch == "mines":
            return "escape_mines"
        if obs.hero.hp_frac < 0.40 and obs.inventory.has_healing:
            return "emergency_heal"
        if obs.combat.adjacent_hostile or obs.combat.has_active_hostile:
            return "combat"
        if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
            return "handle_nutrition"
        if obs.hero.depth <= 4 and obs.hero.ac >= 6 and (obs.inventory.has_unworn_armor or obs.spatial.has_nearby_loot):
            return "scavenge_armor"
        if obs.dungeon.can_forge_excalibur and (obs.dungeon.standing_on_fountain or obs.dungeon.has_known_fountain):
            return "forge_excalibur"
        return "explore_and_dive"

    def run(self, obs):
        while True:
            # Universal Emergency Reflex 1: Mines Evacuation (avoid dark Gnomish Mines)
            if obs.hero.dungeon_branch == "mines":
                if obs.spatial.standing_on_stairs_up:
                    obs = (yield ascend())
                    continue
                elif obs.spatial.stairs_up_known:
                    obs = (yield step_to_stairs_up())
                    continue
                else:
                    obs = (yield step_to_frontier())
                    continue

            # Universal Emergency Reflex 2: Divine Prayer (Major Trouble: Fainting or <15% HP)
            if (obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 4 and not obs.inventory.has_food)) and (obs.hero.can_pray and obs.hero.turn - self.last_prayer_turn >= 850):
                self.last_prayer_turn = obs.hero.turn
                obs = (yield pray())
                continue

            # Goal Dispatch
            goal = self.determine_goal(obs)
            self.current_goal = goal

            if goal == "emergency_heal":
                obs = (yield quaff_healing())
                continue
            elif goal == "handle_nutrition":
                obs = (yield eat_carried_food())
                continue
            elif goal == "combat":
                obs = (yield from self.skill_combat(obs))
                continue
            elif goal == "scavenge_armor":
                obs = (yield from self.skill_scavenge_armor(obs))
                continue
            elif goal == "forge_excalibur":
                obs = (yield from self.skill_forge_excalibur(obs))
                continue
            else:
                obs = (yield from self.skill_explore_and_dive(obs))
                continue

    def skill_combat(self, obs):
        while obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            # Escape via stairs down if standing on them
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = (yield descend())
                break
            # Emergency healing
            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = (yield quaff_healing())
                continue
            # Panic escape
            if (obs.hero.hp_frac < 0.25 or obs.combat.is_surrounded) and obs.combat.has_panic_escape:
                if obs.inventory.has_scroll_of_teleport:
                    obs = (yield read_scroll_teleport())
                    continue
                elif obs.inventory.has_wand_of_teleport:
                    obs = (yield zap_wand_teleport())
                    continue
            # Passive / exploding hazards
            if obs.combat.adjacent_floating_eye or obs.combat.adjacent_gas_spore:
                if obs.combat.has_safe_melee_target:
                    obs = (yield melee_attack_hostile())
                    continue
                if obs.combat.adjacent_floating_eye and obs.inventory.has_daggers:
                    obs = (yield throw_dagger())
                    continue
                obs = (yield step_away_from_hostile())
                continue
            # Fast predators / swarms
            if obs.combat.is_fast_dangerous:
                if obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2 or obs.hero.hp_frac < 0.50:
                    if not obs.combat.standing_on_elbereth:
                        obs = (yield engrave_dust_elbereth())
                        continue
                if obs.combat.adjacent_hostile:
                    obs = (yield melee_attack_hostile())
                    continue
            # Ranged harassment at distance >= 2
            if not obs.combat.adjacent_hostile:
                if obs.inventory.has_offensive_wand:
                    obs = (yield zap_offensive_wand())
                    continue
                if obs.inventory.has_daggers and obs.combat.closest_hostile_dist <= 3:
                    obs = (yield throw_dagger())
                    continue
            # Melee attack
            if obs.combat.adjacent_hostile:
                obs = (yield melee_attack_hostile())
                continue
            obs = (yield step_away_from_hostile())
        return obs

    def skill_scavenge_armor(self, obs):
        if obs.inventory.has_unworn_armor:
            obs = (yield wear_armor())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield step_to_dead_end())
        return obs

    def skill_forge_excalibur(self, obs):
        if not obs.dungeon.standing_on_fountain:
            if obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
                obs = (yield step_to_fountain())
                return obs
            obs = (yield step_to_frontier())
            return obs
        if obs.hero.hp_frac >= 0.85 and obs.dungeon.can_forge_excalibur:
            obs = (yield dip_excalibur())
            return obs
        obs = (yield step_to_stairs_down())
        return obs

    def skill_explore_and_dive(self, obs):
        # Dive to stairs down if known and pacing conditions met
        should_descend = (
            obs.spatial.stairs_down_known and (
                obs.hero.turns_on_level >= 100
                or not obs.spatial.has_unvisited_frontier
                or obs.hero.hunger_state >= 2
                or obs.hero.depth >= 3
            )
        )
        if should_descend:
            obs = (yield step_to_stairs_down())
            return obs

        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs

        # Doors
        if obs.dungeon.adjacent_closed_door:
            if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                obs = (yield kick_closed_door())
            else:
                obs = (yield open_door())
            return obs
        if obs.dungeon.has_closed_door:
            obs = (yield step_to_closed_door())
            return obs

        if obs.spatial.has_unsearched_dead_end:
            obs = (yield step_to_dead_end())
            return obs
        obs = (yield search())
        return obs
```

### Critical Rules on Synthesis & Modifications:
1. **NO TRIVIAL LITERAL TWEAKS**: Do NOT make microscopic changes like changing a numeric constant from 0.15 to 0.16, 0.40 to 0.42, or 850 to 860. Such trivial changes produce 0 statistical significance and will be rejected by counterfactual twin evaluation. Focus on **macro behavioral changes**: adding new skills, restructuring goal priority in `determine_goal`, adding tactical kiting/retreat routines, fixing nutrition management, or optimizing floor exploration.
2. **AUTONOMOUS SKILL DISCOVERY (MANDATORY FOR DEEP DUNGEON PROGRESSION)**:
   To progress past early dungeon levels (DL 5, DL 10, DL 20) and reach the campaign goal, the baseline policy MUST NOT remain static. You have full architectural freedom and authority to invent and attach BRAND NEW specialized modular skills to `class Agent`.
   Do NOT just tweak existing methods—expand the agent's behavioral repertoire by inventing skills for unhandled game challenges:

   **High-Value Candidate Skills to Discover and Implement**:
   - `skill_altar_sacrifice(self, obs)`: Identify altars (`obs.dungeon.standing_on_altar`, `adjacent_altar`, `can_sacrifice`), test item BUC status, and sacrifice fresh corpses for divine favor, protection, and artifact gifts.
   - `skill_solve_sokoban(self, obs)`: Detect Sokoban branch entrance (`obs.dungeon.has_sokoban_entrance`, `obs.dungeon.is_sokoban`, `can_solve_sokoban`), navigate and push boulders without penalties to claim the Amulet of Reflection or Bag of Holding.
   - `skill_harvest_poison_res(self, obs)`: Target and consume corpses of poison-bearing monsters (`obs.corpses`, killer bees, soldier ants) to gain permanent poison resistance (`obs.hero.has_poison_res`).
   - `skill_tactical_kiting(self, obs)`: When pursued by packs or fast monsters in open rooms, retreat to 1-tile corridor chokepoints (`step_to_chokepoint()`), bottle them up 1v1, and soften them with thrown daggers (`throw_dagger()`) or offensive wands (`zap_offensive_wand()`).
   - `skill_enhance_skills(self, obs)`: When `obs.hero.can_enhance_skills` is True, execute `enhance_weapon_skill()` to upgrade weapon proficiencies, massively increasing hit chance and damage dice.
   - `skill_donate_to_priest(self, obs)`: In Minetown or aligned temples (`obs.dungeon.adjacent_priest`, `can_donate_to_priest`), donate 400 * XL gold to priests for divine protection (each donation can reduce AC by up to 2-3 points!).

   **How to Wire a Brand-New Skill**:
   You can emit multiple methods in your single ```python ... ``` patch block:
   1. Define your new skill: `def skill_my_new_skill(self, obs): ... return obs`
   2. Update `def determine_goal(self, obs) -> str:` to return `"my_new_skill"` when appropriate conditions hold.
   3. Update `def run(self, obs):` to dispatch `elif goal == "my_new_skill": obs = (yield from self.skill_my_new_skill(obs)); continue`
   The AST compiler will automatically replace updated methods and append new skill methods onto `class Agent`!

3. **GENERATOR SUBROUTINE PROTOCOL**:
   - Subroutines called via `obs = (yield from self.skill_xxx(obs))` MUST conclude with `return obs`. NEVER return a boolean or None.
   - Every subroutine MUST yield an action before returning. Returning without yielding followed by caller `continue` causes an infinite busy loop at 100% CPU.

4. **PACED PROGRESSION CONTRACT (ELIMINATING EARLY-FLOOR STALLS)**:
   - High starvation or death to low-tier pests (newts, grid bugs, jackals) on DL 1–3 is almost NEVER a nutrition problem—it is a **PACING STALL**.
   - NetHack levels have dozens of dead ends; `obs.spatial.floor_explored` is rarely True. Gating descent behind `floor_explored` traps the hero on DL 1 for thousands of turns until food is exhausted!
   - Enforce clean pacing inside your policy: When stairs down are discovered (`obs.spatial.stairs_down_known`), descend promptly once immediate nearby loot is scooped or when `obs.hero.turns_on_level >= 100`, `obs.hero.hunger_state >= 2`, or `not obs.spatial.has_unvisited_frontier`.
   - Never perform exhaustive wall searches on DL 1–3 when stairs down are already discovered.

5. **POLICY CONTRACT INVARIANTS (AST LINTER ENFORCEMENT)**:
   - Method names must be exact (e.g. `skill_explore_and_dive`, not `skill_explore_and_div`). Near-duplicate method names with typos will fail AST compilation.
   - Any goal returned by `determine_goal(self, obs)` must have an explicit `elif goal == "...":` branch in `run(self, obs)`.
   - Any subroutine invoked via `obs = (yield from self.skill_xxx(obs))` must be defined on `class Agent`.

### Ground-Truth NetHack 3.6 Mechanics:
- **Nutrition**: Hunger states: 0=Satiated, 1=Normal, 2=Hungry, 3=Weak, 4=Fainting. Eat packaged food ONLY when `hunger_state >= 2` ("Hungry" or "Weak"). Eating at 0 or 1 wastes food rations on turn 1. Pray for divine feeding ONLY when `hunger_state >= 4` ("Fainting") without food.
- **Armor Class (AC)**: Base AC is 10. Dropped armor MUST be equipped via `wear_armor()` to lower AC to 0 or negative numbers. Lower AC drastically cuts physical monster damage.
- **Speeds & Fast Predators**: Hero speed is 12. Fast predators outrun the hero: foxes (15), soldier ants (18), killer bees (18), giant bats (22). In open rooms, NEVER step away from an adjacent fast predator (it gives them free hits); fight in melee or engrave dust Elbereth!
- **Elbereth**: Dust Elbereth (`engrave_dust_elbereth()`) wards against non-humanoids (ants, bees, wolves, canines, beasts). In NetHack 3.6, orcs, goblins, gnomes, dwarves, trolls, and undead respect Elbereth! Only humans/elves (`@`), minotaurs, angels, and unique quest bosses ignore it.
- **Excalibur**: Lawful Valkyrie XL >= 5 dipping a non-artifact long sword into a fountain has 1/6 chance per dip of forging Excalibur (+1d10 damage, secret door searching, life drain immunity). NEVER dip in Minetown.
- **Passive Hazards**: Floating eyes inflict 70 turns of paralysis on melee contact. Gas spores explode for lethal blast. Attack floating eyes ONLY with ranged missiles (`throw_dagger()`, safe at distance 1) or offensive wands. If no ranged ammo, step away or detour around them.
- **Pacing & Descent Protocol**: Lingering on DL 1–3 for > 150 turns or searching endless walls is lethal. When stairs down are found (`obs.spatial.stairs_down_known`), descend promptly once immediate nearby loot is scooped or when `obs.hero.turns_on_level >= 100` or hungry.

### Observation Interface & Tactical Predicates:
Every turn, `obs` provides rich sub-namespaces:
- `obs.hero`: `hp`, `max_hp`, `hp_frac`, `energy`, `ac`, `level`, `depth`, `turn`, `turns_on_level`, `hunger_state`, `dungeon_branch`, `has_poison_res`, `can_enhance_skills`, `can_pray`
- `obs.combat`: `adjacent_hostile`, `hostile_count_fov`, `has_active_hostile`, `is_surrounded`, `standing_on_elbereth`, `is_fast_dangerous`, `adjacent_floating_eye`, `adjacent_gas_spore`, `is_corrosive_target` (acid blobs/jellies), `is_heavy_weapon_threat` (orc captains/gnomes with two-handed weapons), `has_panic_escape`
- `obs.inventory`: `has_food`, `has_healing`, `has_unworn_armor`, `has_daggers`, `has_offensive_wand`, `quaff_emergency_potion` (panic healing), `read_emergency_scroll` (panic escape)
- `obs.spatial`: `stairs_down_known`, `standing_on_stairs_down`, `stairs_up_known`, `standing_on_stairs_up`, `has_unvisited_frontier`, `has_unsearched_dead_end`, `floor_explored`, `has_nearby_loot`
- `obs.dungeon`: `has_sokoban_entrance`, `step_to_sokoban_entrance`, `standing_on_fountain`, `adjacent_fountain`, `can_forge_excalibur`, `standing_on_altar`, `adjacent_altar`
- `obs.agenda`: Strategic dungeon progression phases:
  * `phase_early_rush`: DL 1-2 fast descent after looting visible items.
  * `phase_early_scaling`: DL 3-5 Excalibur forging, armor upgrades, poison resistance harvesting.
  * `phase_mid_branches`: DL 6-10 Sokoban solving, Mines avoidance.
  * `phase_deep_dungeon`: DL 11-19 corridor defense, Quest readiness, descent toward Castle.

### Available Actions:
{actions}

### Constants:
{enums}

### Analytical & Investigation Tools:
- `query_duckdb(sql)`: Read-only SQL on `data/lox.duckdb` (tables: `episodes`, `ticks`, `meta_experiments`).
- `get_death_autopsy_trace(episode_id, limit)`: Granular tick-by-tick flight trace of the final ticks (default 30) leading up to death in chronological order.
- `get_death_taxonomy(window)`: Top death causes, frequencies, and avg depth.
- `query_root_causes(run_id, window)`: Root cause attribution across canonical failure archetypes.
- `query_wiki(query)`: Search offline NetHack 3.6 encyclopedia for monster stats, mechanics, and items.
- `query_invariants(query, category)`: Query LOX Empirical Invariants Knowledge Base.
- `get_dungeon_topology(depth)`: 21x79 ASCII map of explored floor.
- `get_hazard_map(depth)`: Hazard map of traps, floating eyes, and monsters on that level.
- `get_floor_stash_report()`: Altars, fountains, and features across explored levels.
- `request_macro(macro_name, rationale, proposed_interface, priority)`: Request complex algorithmic macro implementation.

### Empirical Multi-Turn Investigation Protocol:
1. **Tool Budget Ceiling**: You have a budget of up to 5 tool-calling turns.
2. **Turn 0 Direct Synthesis**: If the pre-compiled Empirical Dossier (Trustworthy Triad + Deep Mechanics) provides complete clarity, you may formulate your hypothesis and synthesize code immediately on Turn 0.
3. **Targeted Follow-Up Queries**: If key facts require validation before writing code, use your tool budget (up to 5 turns):
   - Check monster stats, speeds, and attack types using `query_wiki`.
   - Run SQL on `episodes` or `ticks` via `query_duckdb`.
   - Trace individual deaths with `get_death_autopsy_trace`.
   - Query verified rules via `query_invariants`.
4. **Pareto Harmonization Directive**: Target the #1 dominant failure cause in your hypothesis while ensuring secondary causes (#2 and #3) do not regress. Never overfit to a single isolated episode.

### Output Requirement:
Output your structured hypothesis in a ```yaml ... ``` block, then your modified and/or new method(s) in a single ```python ... ``` block.
The engine automatically splices modified and new methods into `class Agent` via AST.
"""


def build_user_prompt(
    current_policy: str,
    trigger_reason: str,
    status_report: str,
    causal_summary: str = "",
    recent_experiments: str = "",
) -> str:
    """User prompt presenting the empirical autopsy, causal dossier, relevant invariants, and current policy."""
    from lox.knowledge import REGISTRY

    relevant_invs = REGISTRY.get_relevant_invariants_for_trigger(
        trigger_reason, top_k=3
    )
    invariants_block = REGISTRY.format_llm_reference(relevant_invs)
    if invariants_block:
        invariants_block = f"\n{invariants_block}\n"

    causal_block = ""
    if causal_summary:
        if isinstance(causal_summary, dict):
            import yaml
            causal_str = yaml.dump(causal_summary, sort_keys=False)
        else:
            causal_str = str(causal_summary)
        causal_block = f"\n### Causal Timeline Dossier (Empirical Root Cause Analysis):\n```yaml\n{causal_str.strip()}\n```\n"

    experiments_block = ""
    if recent_experiments:
        experiments_block = f"\n### Recent Hypothesis Experiment Outcomes (From DuckDB meta_experiments):\n{recent_experiments}\n"

    return f"""### Empirical Incident Report:
{status_report}
{causal_block}{experiments_block}
### Synthesis Objective:
{trigger_reason}

### Empirical Tool Investigation & Synthesis Protocol:
The pre-compiled Empirical Dossier above provides the **Trustworthy Triad** (Factor 1 Macro Breakdown, Factor 2 Orthogonal Drivers, Factor 3 Ground-Truth Telemetry) alongside Deep Wiki & Invariant Mechanics for the dominant failure causes.
- **Budget**: You have up to 5 tool-calling turns if you need to run targeted queries (`query_duckdb`, `get_death_autopsy_trace`, `query_wiki`, `query_invariants`).
- **Turn 0 Direct Synthesis**: If the dossier provides complete evidence to diagnose the failure mode, you may formulate your structured hypothesis and code patch immediately on Turn 0.
- **Harmonization**: Target the #1 dominant failure cause in your hypothesis while ensuring secondary causes (#2 and #3) do not regress.
{invariants_block}
### Current Policy:
```python
{current_policy.strip()}
```

### Scientific Method Requirement:
1. Formulate a structured scientific hypothesis in a ```yaml ... ``` block:
```yaml
hypothesis:
  causal_finding: "<specific root cause from causal timeline dossier and tool investigation>"
  targeted_skill: "<e.g. determine_goal, skill_combat, skill_scavenge_armor, skill_explore_and_dive, or a brand-new skill name like skill_altar_sacrifice, skill_harvest_poison_res, skill_solve_sokoban>"
  mechanism: "<concrete behavioral/algorithmic change - e.g. newly invented skill and its dispatch logic>"
  predicted_outcome:
    target_metric: "<e.g. avg_depth, ac_at_death>"
    expected_direction: "increase" | "decrease"
    min_improvement: 1.0
```
2. Synthesize your revised and/or newly invented method(s) in a single ```python ... ``` block. You have full authority to introduce NEW skills! Output the modified methods AND your new skill method(s) starting at column 0 with standard 4-space body indentation:
```python
def determine_goal(self, obs) -> str:
    # updated goal routing including your new skill
    ...

def run(self, obs):
    # updated dispatch loop handling your new goal
    ...

def skill_my_new_skill(self, obs):
    # your newly invented modular skill
    ...
    return obs
```
The AST compiler will automatically replace updated methods and append new skill methods onto `class Agent`!
"""
