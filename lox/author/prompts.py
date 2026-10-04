"""
LOX Author Prompts: Complete NetHack Vocabulary, Class Agent Generator Paradigm, and Deep Telemetry Tools.
"""
from __future__ import annotations

from lox.dsl.schema import ALLOWED_ACTIONS, ENUM_CONSTANTS


def build_system_prompt() -> str:
    """System prompt defining the Class Agent generator architecture and full NetHack API."""
    actions = ", ".join(f"{a}()" for a in sorted(ALLOWED_ACTIONS))
    enums = ", ".join(f"{k}={v}" for k, v in ENUM_CONSTANTS.items())

    return f"""You are the LOX Policy Synthesizer.
Synthesize autonomous NetHack/MiniHack policies using an unconstrained, object-oriented Python generator class: `class Agent`.
The agent is instantiated fresh at the start of each episode. Every turn, its `run(self, obs)` method receives an `obs` object and yields an `Action`.

### Policy Architecture: `class Agent`
Write clean Python with local state on `self`, loops (`while`, `for`), helper methods, and sub-generators (`yield from`):
```python
class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0
        self.retreat_streak = 0

    def run(self, obs):
        while True:
            # 1. Absolute Emergency Survival (Major Trouble: Fainting or <15% HP)
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state == 4):
                if obs.hero.turn - self.last_prayer_turn >= 350:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue

            # 2. Combat Logic (Highest Priority)
            if obs.combat.adjacent_hostile or obs.combat.has_active_hostile or (obs.combat.hostile_count_fov > 0 and (obs.inventory.has_offensive_wand or obs.inventory.has_daggers)):
                obs = yield from self.handle_combat(obs)
                continue

            # 3. Hunger Prevention (Only when safe)
            if obs.hero.hunger_state >= 2:
                if obs.inventory.has_food:
                    obs = yield eat_carried_food()
                    continue
                elif obs.corpses:
                    obs = yield from self.handle_corpse_consumption(obs)
                    continue

            # 4. Equipment Optimization (Equip found armor when out of combat)
            if obs.inventory.has_unworn_armor:
                obs = yield wear_armor()
                continue

            # 5. Weapon Scaling (Forge Excalibur when lawful Valkyrie level >= 5 at a fountain)
            if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
                if obs.dungeon.standing_on_fountain:
                    obs = yield dip_excalibur()
                    continue
                elif obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
                    obs = yield step_to_fountain()
                    continue

            # 6. Nearby Floor Loot Scooping (Collect dropped armor/wands/potions to lower AC and gain tactical items)
            if obs.spatial.has_nearby_loot and not obs.combat.adjacent_hostile:
                obs = yield step_to_loot()
                continue

            # 7. Altar BUC Testing (Only when adjacent or standing on altar with untested items)
            if obs.dungeon.standing_on_altar and obs.epistemic.has_untested_items:
                obs = yield test_altar_buc()
                continue
            elif obs.dungeon.adjacent_altar and obs.epistemic.has_untested_items:
                obs = yield step_to_altar()
                continue

            # 8. Poison Resistance Harvesting (STRICTLY gate by obs.dungeon.can_harvest_poison!)
            if obs.dungeon.can_harvest_poison and not obs.hero.has_poison_res and obs.hero.hp_frac > 0.90:
                obs = yield harvest_poison_res()
                continue

            # 6. Navigation & Exploration
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = yield descend()
            elif obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
            elif obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
            elif obs.spatial.has_unvisited_frontier:
                self.search_count = 0 
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield from self.handle_dead_end(obs)
            else:
                # When visible room and frontiers are cleared, never idle!
                # Move to candidate perimeter walls and dead ends to find secret doors.
                obs = yield from self.handle_dead_end(obs)
            
            if obs.spatial.stairs_down_known and not obs.spatial.standing_on_stairs_down:
                obs = yield step_to_stairs_down()

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            # If the only monsters in FOV are distant passive hazards and we have no ranged weapons, exit combat to explore
            if not obs.combat.adjacent_hostile and not obs.combat.has_active_hostile:
                if not (obs.inventory.has_offensive_wand or obs.inventory.has_daggers):
                    break

            # 0. Emergency Panic Escape: Teleport away if critically low HP (< 25%) or surrounded
            if (obs.hero.hp_frac < 0.20 or obs.combat.is_surrounded) and obs.combat.has_panic_escape:
                if obs.inventory.has_scroll_of_teleport:
                    obs = yield read_scroll_teleport()
                    continue
                elif obs.inventory.has_wand_of_teleport:
                    obs = yield zap_wand_teleport()
                    continue

            # 0.1 In-Combat Hunger Emergency: Eat carried food to avoid fainting
            if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
                if not obs.combat.adjacent_hostile or obs.combat.standing_on_elbereth or obs.combat.adjacent_floating_eye or obs.combat.adjacent_gas_spore:
                    obs = yield eat_carried_food()
                    continue

            # 0.2 Major Trouble Divine Intervention (Fainting or Critical HP)
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state >= 3) and (obs.hero.turn - self.last_prayer_turn >= 350):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue

            # 1. Passive Hazards (Floating Eye): Safe to snipe with daggers/wands at distance 1!
            if obs.combat.adjacent_floating_eye:
                if obs.inventory.has_daggers:
                    obs = yield throw_dagger()
                    continue
                elif obs.inventory.has_offensive_wand:
                    obs = yield zap_offensive_wand()
                    continue
                elif obs.combat.can_retreat:
                    obs = yield step_away_from_hostile()
                    continue
                elif obs.combat.has_safe_melee_target:
                    obs = yield melee_attack_hostile()
                    continue
                else:
                    # Trapped in dead end with floating eye and 0 missiles: strike in melee to kill 1-HP eye and escape!
                    obs = yield melee_attack_hostile()
                    continue

            # 1.1 Exploding Hazards (Gas Spore): 4d6 explosion blast at distance 1; retreat or fight safe targets!
            if obs.combat.adjacent_gas_spore:
                if obs.combat.can_retreat:
                    obs = yield step_away_from_hostile()
                    continue
                elif obs.combat.has_safe_melee_target:
                    obs = yield melee_attack_hostile()
                    continue
                elif not obs.combat.standing_on_elbereth:
                    obs = yield engrave_dust_elbereth()
                    continue
                else:
                    obs = yield wait()
                    continue

            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain", "guard", "priest", "priestess", "oracle") or obs.dungeon.in_shop:
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue
            if obs.hero.hp_frac < 0.60 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 2. Panic Sanctuary: Engrave Elbereth immediately if low HP (< 35%) or surrounded
            if (obs.hero.hp_frac < 0.35 or obs.combat.is_surrounded) and not obs.combat.standing_on_elbereth:
                obs = yield engrave_dust_elbereth()
                continue

            # 3. While standing on Elbereth:
            if obs.combat.standing_on_elbereth:
                # Hostiles that ignore Elbereth (orcs, elves, humans) must be fought or retreated from!
                if obs.combat.hostile_ignores_elbereth and obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.40 or not obs.combat.can_retreat:
                        obs = yield melee_attack_hostile()
                        continue
                    else:
                        obs = yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()
                        continue

                if obs.hero.hp_frac < 0.30 and obs.hero.turn - self.last_prayer_turn >= 350:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue
                elif obs.hero.hp_frac < 0.60 and obs.inventory.has_healing:
                    obs = yield quaff_healing()
                    continue
                elif obs.inventory.has_offensive_wand and obs.combat.closest_hostile_dist >= 2:
                    obs = yield zap_offensive_wand()
                    continue
                elif obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                    obs = yield throw_dagger()
                    continue
                elif not obs.combat.adjacent_hostile:
                    if obs.spatial.stairs_down_known:
                        obs = yield step_to_stairs_down()
                    elif obs.spatial.has_unvisited_frontier:
                        obs = yield step_to_frontier()
                    else:
                        obs = yield step_away_from_hostile()
                    continue
                else:
                    obs = yield melee_attack_hostile()
                    continue

            # 4. Ranged Harassment: Offensive wands and daggers at distance >= 2
            if obs.inventory.has_offensive_wand and obs.combat.closest_hostile_dist >= 2:
                obs = yield zap_offensive_wand()
                continue
            if obs.combat.closest_hostile_dist >= 2 and obs.inventory.has_daggers:
                obs = yield throw_dagger()
                continue

            # 5. Fast Dangerous Attackers (Soldier ants, killer bees, giant bats, foxes)
            if obs.combat.is_fast_dangerous:
                # If surrounded or facing multiple fast attackers in open rooms: engrave Elbereth immediately! They flee in terror!
                if (obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2) and not obs.combat.standing_on_elbereth:
                    obs = yield engrave_dust_elbereth()
                    continue

                # If adjacent, NEVER run in the open against faster predators! Strike in melee or engrave Elbereth
                if obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.40 or not obs.combat.can_retreat:
                        obs = yield melee_attack_hostile()
                        continue
                    else:
                        if not obs.combat.standing_on_elbereth:
                            obs = yield engrave_dust_elbereth()
                        else:
                            obs = yield melee_attack_hostile()
                        continue
                elif not obs.combat.in_corridor and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                    continue

            # 6. Tactical Melee / Chokepoint Retreat
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.45 or not obs.combat.can_retreat:
                    obs = yield melee_attack_hostile()
                else:
                    obs = yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()
            else:
                if obs.hero.hp_frac > 0.40:
                    obs = yield melee_attack_hostile()
                else:
                    obs = yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()
            if obs.combat.hostile_count_fov == 0 and not obs.combat.adjacent_hostile:
                break
        return obs

    def handle_corpse_consumption(self, obs):
        if obs.combat.hostile_count_fov > 0:
            obs = yield wait()
            return obs
        for corpse in obs.corpses:
            if corpse.is_safe:
                if (obs.hero.y, obs.hero.x) == (corpse.y, corpse.x):
                    obs = yield eat_floor_corpse()
                else:
                    obs = yield step_to(corpse.y, corpse.x)
                return obs
        obs = yield wait()
        return obs

    def handle_dead_end(self, obs):
        current_pos = (obs.hero.y, obs.hero.x)
        if obs.spatial.standing_on_dead_end and current_pos != getattr(self, "last_searched_pos", None):
            self.last_searched_pos = current_pos
            for _ in range(8):
                if obs.combat.hostile_count_fov > 0:
                    return obs
                if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                    break
                obs = yield search()
        else:
            obs = yield step_to_dead_end()
        return obs
```

### Critical Generator Subroutine Rules:
1. Always maintain `obs` as the Observation object:
   - When calling a helper generator subroutine using `obs = yield from self.my_subroutine(obs)`, the subroutine MUST conclude with `return obs`. NEVER return a boolean (`return True` / `return False`) from a generator subroutine! Returning a boolean will overwrite `obs = True`, causing an immediate crash on the next turn.
2. Do not check generator truthiness with `if self.my_routine(obs):` because calling a generator function always returns a truthy generator object. Instead, inspect predicates on `obs` directly in `run()` (e.g. `if obs.combat.adjacent_hostile:`), and then do `obs = yield from self.handle_combat(obs)`.
3. Every execution path through a generator subroutine MUST yield at least one action (e.g. `obs = yield wait()`) before returning! If a subroutine returns without yielding and the caller continues, the generator enters an infinite busy loop at 100% CPU.

### Observation Interface (`obs`)
Every turn, `obs` provides rich sub-namespaces:
- `obs.hero`: `hp`, `max_hp`, `hp_frac`, `energy`, `energy_frac`, `ac`, `level`, `depth`, `turn`, `turns_on_level`, `gold`, `hunger_state` (SATIATED, NORMAL, HUNGRY, WEAK, FAINTING), `dungeon_branch`, `has_poison_res`
- `obs.status`: `is_blind`, `is_poisoned`, `is_confused`, `is_stunned`, `is_sick`, `is_encumbered`, `encumbrance_level`, `is_levitating`
- `obs.inventory`: `has_food`, `has_healing`, `has_unworn_armor`, `has_daggers`, `has_offensive_wand`, `has_wand_of_teleport`, `has_scroll_of_teleport`, `get_food_slot()`, `get_healing_slot()`, `get_dagger_slot()`, `get_offensive_wand_slot()`, `items`
- `obs.combat`: `adjacent_hostile`, `hostile_count_fov`, `has_active_hostile`, `active_hostile_count`, `closest_hostile_name`, `closest_hostile_dist`, `is_surrounded`, `in_corridor`, `can_retreat`, `standing_on_elbereth`, `is_fast_dangerous`, `adjacent_floating_eye`, `adjacent_gas_spore`, `hostile_ignores_elbereth`, `has_panic_escape`, `has_safe_melee_target`
- `obs.spatial`: `stairs_down_known`, `standing_on_stairs_down`, `has_unvisited_frontier`, `has_unsearched_dead_end`, `floor_explored`, `has_nearby_loot`
- `obs.dungeon`: `tile_type` (corridor, room, doorway, fountain, altar, trap), `in_shop`, `in_temple`, `is_dark_level`, `adjacent_closed_door`, `door_is_locked`, `adjacent_fountain`, `standing_on_fountain`, `adjacent_altar`, `standing_on_altar`, `can_forge_excalibur`, `can_harvest_poison`
- `obs.epistemic`: `untested_buc_count` (int), `has_untested_items` (bool), `can_safely_wear_armor` (bool), `can_safely_quaff_healing` (bool), `items_belief` (dict of ItemBeliefState)
- `obs.agenda`: `active_goal` (str), `goal_stack` (list of str), `is_active(goal_name)` (bool)
- `obs.corpses`: list of `FloorCorpse(name, y, x, age_turns, is_fresh, is_poisonous, is_deadly, is_safe)`
- `obs.message`: last raw game message

### Critical NetHack 3.6.6 Mechanics & Invariants:
1. **Passive & Exploding Hazards (`adjacent_floating_eye`, `adjacent_gas_spore`, `has_safe_melee_target`, `has_active_hostile`)**: Attacking a floating eye in melee triggers a passive 70-turn paralysis gaze (`0d70`). Striking a gas spore explodes for lethal 4d6 area blast. NEVER melee attack them! Destroy with ranged daggers or offensive wands. However, if an active attacker (newt, jackal, orc) is adjacent alongside the hazard, `obs.combat.has_safe_melee_target` is True: yield `melee_attack_hostile()` to eliminate the active attacker (the adapter automatically skips the passive hazard). If no ranged weapons exist and passive hazards are distant (distance >= 2, `has_active_hostile == False`), do not enter combat; continue exploration as `walkable_nav` steers around them!
2. **Emergency Panic Escape (`read_scroll_teleport`, `zap_wand_teleport`)**: When low on HP (< 25%) or surrounded by high-speed attackers, teleport away immediately. NetHack teleport scrolls and wands (zapped at `.`) instantly relocate the hero to a random safe tile.
3. **Nearby Floor Loot Scooping (`step_to_loot`)**: Defeated monsters drop armor, weapons, wands, and scrolls on their death tile. When out of combat and `obs.spatial.has_nearby_loot` is True, yield `step_to_loot()` to walk over the dropped items. Autopickup collects them, allowing `wear_armor()` to lower Armor Class (AC) towards negative numbers!
4. **Elbereth Immunity & Humanoid Combat Tactics (`hostile_ignores_elbereth`)**: Orcs, elves, and humans ignore Elbereth. While standing on Elbereth, if `obs.combat.hostile_ignores_elbereth` is True, do NOT wait passively; actively fight in melee or retreat to a 1-tile corridor chokepoint.
5. **Prayer & Divine Favor**: In NetHack, successful prayer resets divine timeout to 300 + rn2(500) turns. Enforce `obs.hero.turn - self.last_prayer_turn >= 350` before emergency prayer (`pray()`) to guarantee divine aid and avoid angering your deity ("Tyr is displeased"), which causes divine smiting or paralysis.
6. **Corpse Consumption Hazards**: Eating a corpse takes multiple turns (`weight / 64 + 3`), leaving the hero completely helpless and vulnerable. NEVER eat a corpse if enemies are in FOV. Corpses older than 50 turns cause food poisoning; kobolds are poisonous; cockatrices cause lethal petrification without gloves. Check `corpse.is_safe` before eating!
7. **Door Breaching**: Always try `open_door()` first on closed doors. Only use `kick_closed_door()` if `obs.dungeon.door_is_locked` is True (kicking unlocked doors can hurt your leg and immobilize you for 5-20 turns).
8. **Excalibur Dipping (`dip_excalibur`)**: Dipping a long sword into a fountain strictly requires **standing directly on the fountain tile** (`obs.dungeon.standing_on_fountain`). It has a 1/6 chance of forging Excalibur when lawful Valkyrie/Knight at level >= 5 (`obs.dungeon.can_forge_excalibur`). Wielding Excalibur gives +1d10 slashing damage, auto-searching for doors, and drain resistance. Never dip from an adjacent tile!
9. **Secret Doors & Corridor Dead Ends**: NetHack procedural generation regularly seals off deeper dungeon sections and staircases behind hidden secret doors located at dead-end corridors (`#`) and room perimeter walls. When visible frontiers are fully explored (`obs.spatial.has_unvisited_frontier == False`), call `step_to_dead_end()` to navigate to dead ends or perimeter walls and `search()` repeatedly until the secret door is exposed. Searching repeatedly inside open rooms will NOT find the stairs.
10. **Tactical In-Combat Emergency Healing**: Quaffing a healing potion takes only 1 turn and restores 10-20 HP. When `obs.hero.hp_frac < 0.50` and `obs.inventory.has_healing`, ALWAYS quaff healing immediately inside `handle_combat` before taking another attack! Never die with healing potions in your pack.
11. **Soldier Ants & High-Speed Attackers**: Soldier ants and killer bees move at speed 18 (nearly 2x the hero) and inflict lethal poison stings. If `obs.combat.is_fast_dangerous` is True or `obs.combat.closest_hostile_name in ("soldier ant", "killer bee")`, retreat immediately to a 1-tile corridor chokepoint (`step_to_chokepoint()`), quaff healing, or pray.
12. **Equipment & Armor Optimization (`wear_armor`)**: Defeated monsters drop helmets, boots, cloaks, and armor. When out of combat, `obs.inventory.has_unworn_armor` is True, and `obs.epistemic.can_safely_wear_armor` is True, yield `wear_armor()` to lower your Armor Class (AC). Lower AC drastically reduces damage from deep monsters.
13. **Shopkeeper, Vault Guard & Peaceful Non-Aggression**: Never attack shopkeepers, vault guards (`guard`), priests, or town watchmen (`obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain", "guard", "priest", "priestess", "oracle")`), and never kick doors when `obs.dungeon.in_shop` is True. Provoking vault guards or shopkeepers will instantly kill your hero.
14. **Exploration Pacing & No Idle Waiting**: Waiting (`wait()`) during active exploration when seeking stairs is strictly forbidden. Never yield `wait()` when frontiers are clear; instead call `handle_dead_end(obs)` or `search()` to investigate room perimeter walls. Sitting in place produces 0 turns of progress and leads to `MaxTurnsReached`.
15. **Dust Elbereth Sanctuary (`engrave_dust_elbereth`)**: Engraving "Elbereth" into the dust with bare fingers takes 1 turn and creates an impenetrable ward against 95% of non-humanoid monsters (ants, bees, bats, mumakil, leocrottas, canines). Monsters cannot attack on an Elbereth tile and flee in panic. When cornered or HP < 35%, yield `engrave_dust_elbereth()`.
16. **Ranged Missile Harassment (`throw_dagger`, `zap_offensive_wand`)**: Valkyries start with daggers. When enemies are at distance >= 2, yield `throw_dagger()` or `zap_offensive_wand()` to kill fast speedsters before they close to melee. Thrown daggers drop onto the ground and are automatically recovered after combat, so freely throw daggers down to 0.
17. **Altar BUC Identification (`test_altar_buc`)**: Equipping cursed armor or weapons welds them to your body. When an altar is nearby (`obs.dungeon.adjacent_altar` or `obs.dungeon.standing_on_altar`) and `obs.epistemic.has_untested_items` is True, yield `test_altar_buc()` to automatically batch-drop and identify the BUC status of untested inventory items.
18. **Poison Resistance Harvesting (`harvest_poison_res`)**: Poisonous bites from ants and bees kill heroes instantly at Depth 4+. When out of combat and `obs.dungeon.can_harvest_poison` is True and `not obs.hero.has_poison_res`, yield `harvest_poison_res()` to seek out and consume killer bee or soldier ant corpses to gain permanent poison resistance intrinsic.
19. **Hybrid HTN-BT Goal Agenda**: Structure strategic progression around milestones (`obs.agenda.active_goal` or `GoalDirective`). Reactive reflexes (Combat, Healing, Hunger, Panic Elbereth) ALWAYS execute first. When safe, evaluate your strategic goal: e.g. acquire poison resistance (`GOAL_COLLECT_POISON_RES`), forge Excalibur (`GOAL_FORGE_EXCALIBUR`), test BUC on altars (`GOAL_TEST_BUC_ALTAR`), or clear the floor and descend (`GOAL_DESCEND_STAIRS`).
20. **Gnomish Mines Avoidance & Main Dungeon Steering**: The Gnomish Mines entrance (`obs.hero.dungeon_branch == "mines"`) appears on Depths 2–4. The Mines are pitch dark and infested with deadly gnome packs with wands and crossbows. If `obs.hero.dungeon_branch == "mines"`, immediately ascend back up to the main dungeon (`yield ascend()` if on stairs up, or `yield step_to_stairs_up()`). NetHackAdapter will dynamically record and prune that Mines staircase from `known_stairs_down` upon returning to the Dungeons of Doom, allowing the hero to locate and descend the true main dungeon staircase down toward Depth 20.
21. **Adjacent Passive Hazard Ranged Elimination & Dead-End Melee Break**: Throwing daggers, darts, arrows, or rocks (`throw_dagger()`) at an adjacent floating eye (`obs.combat.adjacent_floating_eye`) is 100% safe at distance 1 and does NOT trigger the passive melee paralysis attack. Never restrict ranged attacks against floating eyes to distance >= 2. If cornered in a dead end with an adjacent floating eye and 0 missiles, waiting will cause certain starvation; striking it in melee kills its 1-HP body and clears the path. Gas spores explode for 4d6 damage at distance 1, so retreat away or fight adjacent safe targets.
22. **Proactive In-Combat Nutrition**: NetHack fainting occurs randomly at `hunger_state >= 3`, leaving the hero unconscious for 30 turns and defenseless against monsters. Consume carried food (`eat_carried_food()`) proactively at `hunger_state >= 2` ("Weak") when standing on Elbereth, out of melee reach, or adjacent only to passive hazards (floating eyes or gas spores). Never wait until `hunger_state >= 3` to eat while monsters are in FOV.

### Available Actions:
{actions}

### Constants:
{enums}

### Analytical & Empirical Investigation Tools:
- `query_duckdb(sql)`: Read-only SQL on `data/lox.duckdb` (tables: `episodes`, `ticks`, `events`).
  * `episodes`: run_id, episode_id, depth, score, turns, death_reason, inventory_at_death, last_5_actions, turns_dl1, turns_dl2, turns_mines
  * `ticks`: episode_id, turn, depth, hp, max_hp, hunger, y, x, action, closest_hostile_name, closest_hostile_dist, tile_type, dungeon_branch, message
- `get_death_autopsy_trace(episode_id)`: Granular tick-by-tick flight trace of final 15 ticks of the fatal run.
- `get_dungeon_topology(depth)`: 21x79 ASCII visual footprint map of visited rooms, corridors, and stairs.
- `get_hazard_map(depth)`: Discovered traps, floating eyes, and dangerous monster locations.
- `get_floor_stash_report()`: Altars, fountains, and features discovered across all explored dungeon levels.
- `get_death_taxonomy(window)`: Top death causes, frequencies, and avg depth.
- `query_wiki(query)`: Search offline NetHack 3.6.6 encyclopedia (monsters, intrinsics, corpses, rituals).
- `request_macro(macro_name, rationale, proposed_interface, priority)`:
  * **When to request a macro**: Call `request_macro` when you identify a complex, multi-turn algorithmic procedure that cannot be cleanly implemented in simple reactive `Agent` code (for example: Sokoban boulder-push pathfinding, shop price-identification sequences, complex container stash packing, or water-crossing solvers).
  * **Contract**:
    - `macro_name`: Concise action identifier (e.g. `solve_sokoban_boulder`, `identify_shop_prices`, `stash_items`).
    - `rationale`: Clear explanation of the empirical mortality bottleneck or tactical need observed in DuckDB telemetry.
    - `proposed_interface`: Proposed Python signature, input arguments, expected return behavior, and failure fallbacks.
    - `priority`: `"low"`, `"medium"`, `"high"`, or `"critical"`.
  * **Result**: Your request is registered into `data/macro_requests.md` for human platform engineering integration into future synthesis cycles.

### Safety Rules:
No imports, no filesystem calls, no arbitrary exec/eval. All logic must reside within `class Agent`.

### Output Requirement:
Provide 1 brief rationale sentence, then your complete revised code in a single ```python ... ``` block.
"""


def build_user_prompt(current_policy: str, trigger_reason: str, status_report: str) -> str:
    """User prompt presenting the empirical autopsy, ranked mortality causes, and current policy."""
    return f"""### Empirical Incident Report:
{status_report}

### Synthesis Objective:
{trigger_reason}

### Diagnostic Notice:
The telemetry database `data/lox.duckdb` holds complete per-tick flight recordings and death traces.
Call `get_death_autopsy_trace()` to inspect the exact final 15 ticks, `get_hazard_map(depth)` to view known traps/monsters, or `query_wiki(monster_name)` to look up mechanics before writing code.

### Current Policy:
```python
{current_policy.strip()}
```

Synthesize the complete revised `class Agent` policy that directly addresses the ranked mortality bottlenecks while maintaining aggressive stair navigation and frontier exploration.
"""
