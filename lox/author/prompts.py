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

    def run(self, obs):
        while True:
            # Universal Reflex 1: Mines Immediate Evacuation (DL 2-4 branch trap elimination)
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

            # Universal Reflex 2: Major Trouble Prayer (Fainting starvation or lethal HP emergency)
            if (obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 4 and not obs.inventory.has_food)) and (obs.hero.can_pray and obs.hero.turn - self.last_prayer_turn >= 850):
                self.last_prayer_turn = obs.hero.turn
                obs = (yield pray())
                continue

            # Universal Reflex 3: Immediate Stair Descent (even in combat)
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = (yield descend())
                continue

            # Universal Reflex 4: Active Combat Defense
            if obs.combat.adjacent_hostile or obs.combat.has_active_hostile:
                obs = (yield from self.handle_combat(obs))
                continue

            # Universal Reflex 5: Healing & Nutrition Recovery
            if obs.hero.hp_frac < 0.35 and obs.inventory.has_healing:
                obs = (yield quaff_healing())
                continue
            if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
                obs = (yield eat_carried_food())
                continue

            # HTN Strategic Phase Dispatch
            phase = getattr(obs, "dungeon_phase", "early_rush")
            if phase == "early_rush" or obs.hero.depth <= 2:
                obs = (yield from self.phase_early_rush(obs))
            elif phase == "early_scaling" or obs.hero.depth <= 5:
                obs = (yield from self.phase_early_scaling(obs))
            elif phase == "mid_branches" or obs.hero.depth <= 10:
                obs = (yield from self.phase_mid_branches(obs))
            else:
                obs = (yield from self.phase_deep_dungeon(obs))

    def phase_early_rush(self, obs):
        # Phase 0: Depths 1-2.
        # Priority: Fast exploration & descent. Strict zero-loot staircase priority.
        if obs.inventory.has_unworn_body_armor and not obs.inventory.has_worn_body_armor:
            obs = (yield wear_armor())
            return obs
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        if obs.hero.can_enhance_skills:
            obs = (yield enhance_weapon_skill())
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        if obs.dungeon.adjacent_closed_door:
            if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                obs = (yield kick_closed_door())
            else:
                obs = (yield open_door())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        if obs.dungeon.has_closed_door:
            obs = (yield step_to_closed_door())
            return obs
        obs = (yield from self.handle_dead_end(obs))
        return obs

    def phase_early_scaling(self, obs):
        # Phase 1: Depths 3-5.
        # Priority: Character power-spiking. Forge Excalibur, upgrade starting armor to mithril/iron,
        # harvest safe corpses for poison resistance, and test BUC at altars.
        if obs.inventory.get_superior_body_armor_slot() is not None:
            obs = (yield replace_body_armor())
            return obs
        if obs.inventory.has_unworn_body_armor and not obs.inventory.has_worn_body_armor:
            obs = (yield wear_armor())
            return obs
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        if obs.hero.can_enhance_skills:
            obs = (yield enhance_weapon_skill())
            return obs
        if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
            if obs.dungeon.standing_on_fountain:
                obs = (yield dip_excalibur())
                return obs
            elif obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
                obs = (yield step_to_fountain())
                return obs
        if obs.dungeon.standing_on_altar:
            if obs.dungeon.can_sacrifice:
                obs = (yield sacrifice_on_altar())
                return obs
            elif obs.epistemic.has_untested_items:
                obs = (yield test_altar_buc())
                return obs
        elif obs.dungeon.adjacent_altar and obs.epistemic.has_untested_items:
            obs = (yield step_to_altar())
            return obs
        if obs.dungeon.can_harvest_poison and not obs.hero.has_poison_res and obs.hero.hp_frac > 0.85:
            obs = (yield harvest_poison_res())
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.dungeon.adjacent_closed_door:
            if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                obs = (yield kick_closed_door())
            else:
                obs = (yield open_door())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        if obs.dungeon.has_closed_door:
            obs = (yield step_to_closed_door())
            return obs
        obs = (yield from self.handle_dead_end(obs))
        return obs

    def phase_mid_branches(self, obs):
        # Phase 2: Depths 6-10.
        # Priority: Branch discrimination & ascension tools.
        # Avoid dark Mines, detect and solve Sokoban for Reflection/Bag of Holding, donate for temple protection.
        if obs.dungeon.has_sokoban_entrance and obs.hero.hp_frac >= 0.70:
            if obs.spatial.standing_on_stairs_up and obs.hero.dungeon_branch != "mines":
                obs = (yield ascend())
                return obs
            obs = (yield step_to_sokoban_entrance())
            return obs
        if obs.dungeon.can_solve_sokoban:
            obs = (yield step_solve_sokoban())
            return obs
        if obs.inventory.get_superior_body_armor_slot() is not None:
            obs = (yield replace_body_armor())
            return obs
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        if obs.hero.can_enhance_skills:
            obs = (yield enhance_weapon_skill())
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.dungeon.adjacent_closed_door:
            if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                obs = (yield kick_closed_door())
            else:
                obs = (yield open_door())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        if obs.dungeon.has_closed_door:
            obs = (yield step_to_closed_door())
            return obs
        obs = (yield from self.handle_dead_end(obs))
        return obs

    def phase_deep_dungeon(self, obs):
        # Phase 3: Depths 11-19.
        # Priority: High caution, corridor chokepoints, Quest readiness, and descent toward the Castle.
        if obs.hero.dungeon_branch == "mines":
            if obs.spatial.standing_on_stairs_up:
                obs = (yield ascend())
                return obs
            obs = (yield step_to_stairs_up())
            return obs
        if obs.inventory.get_superior_body_armor_slot() is not None:
            obs = (yield replace_body_armor())
            return obs
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        if obs.hero.can_enhance_skills:
            obs = (yield enhance_weapon_skill())
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.dungeon.adjacent_closed_door:
            if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                obs = (yield kick_closed_door())
            else:
                obs = (yield open_door())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        if obs.dungeon.has_closed_door:
            obs = (yield step_to_closed_door())
            return obs
        obs = (yield from self.handle_dead_end(obs))
        return obs

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            # Instant descent escape if standing on stairs down during combat
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = (yield descend())
                break

            # If the only monsters in FOV are distant passive hazards and we have no ranged weapons, exit combat to explore
            if not obs.combat.adjacent_hostile and not obs.combat.has_active_hostile:
                if not (obs.inventory.has_offensive_wand or obs.inventory.has_daggers):
                    break

            # 0. Emergency Panic Escape: Teleport away if critically low HP (< 20%) or surrounded
            if (obs.hero.hp_frac < 0.20 or obs.combat.is_surrounded) and obs.combat.has_panic_escape:
                if obs.inventory.has_scroll_of_teleport:
                    obs = (yield read_scroll_teleport())
                    continue
                elif obs.inventory.has_wand_of_teleport:
                    obs = (yield zap_wand_teleport())
                    continue

            # 0.1 In-Combat Hunger Emergency: Eat carried food to avoid fainting
            if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
                if not obs.combat.adjacent_hostile or obs.combat.standing_on_elbereth:
                    obs = (yield eat_carried_food())
                    continue

            # 0.2 Major Trouble Divine Intervention (Fainting without food or Critical HP - pray while conscious!)
            if (obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 4 and not obs.inventory.has_food)) and (obs.hero.can_pray and obs.hero.turn - self.last_prayer_turn >= 850):
                if not obs.combat.adjacent_hostile or obs.combat.standing_on_elbereth:
                    self.last_prayer_turn = obs.hero.turn
                    obs = (yield pray())
                    continue

            # 0.3 In-Combat Emergency Healing
            if obs.hero.hp_frac < 0.40 and obs.inventory.has_healing:
                obs = (yield quaff_healing())
                continue

            # 0.4 Emergency Unidentified Consumables Panic Consumption
            if obs.hero.hp_frac < 0.20 and not obs.inventory.has_healing and not obs.combat.has_panic_escape:
                if obs.inventory.has_unidentified_potion:
                    obs = (yield quaff_emergency_potion())
                    continue
                elif obs.inventory.has_unidentified_scroll:
                    obs = (yield read_emergency_scroll())
                    continue

            # 1. Passive / Exploding Hazards
            if obs.combat.adjacent_floating_eye or obs.combat.adjacent_gas_spore:
                if obs.combat.has_safe_melee_target:
                    obs = (yield melee_attack_hostile())
                    continue
                if obs.combat.adjacent_floating_eye:
                    if obs.inventory.has_daggers:
                        obs = (yield throw_dagger())
                        continue
                    elif obs.inventory.has_offensive_wand:
                        obs = (yield zap_offensive_wand())
                        continue
                obs = (yield step_away_from_hostile())
                continue

            closest_name = obs.combat.closest_hostile_name.lower()
            if closest_name in ("shopkeeper", "watchman", "watch captain", "guard", "priest", "priestess", "oracle") or obs.dungeon.in_shop:
                obs = (yield retreat() if obs.combat.can_retreat else step_away_from_hostile())
                continue

            # 1.1 Priority status hazards (yellow light, homunculus)
            if "yellow light" in closest_name or "homunculus" in closest_name:
                if obs.combat.closest_hostile_dist >= 2:
                    if obs.inventory.has_offensive_wand:
                        obs = (yield zap_offensive_wand())
                        continue
                    elif obs.inventory.has_daggers:
                        obs = (yield throw_dagger())
                        continue
                if obs.combat.adjacent_hostile:
                    if not obs.combat.standing_on_elbereth:
                        obs = (yield engrave_dust_elbereth())
                        continue
                    else:
                        obs = (yield melee_attack_hostile())
                        continue

            # 1.2 Corrosive / acid hazards: strictly forbid melee attacks
            if obs.combat.is_corrosive_target:
                if obs.combat.closest_hostile_dist >= 2:
                    if obs.inventory.has_offensive_wand:
                        obs = (yield zap_offensive_wand())
                        continue
                    elif obs.inventory.has_daggers:
                        obs = (yield throw_dagger())
                        continue
                if obs.combat.adjacent_hostile:
                    obs = (yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile())
                    continue

            # 1.3 Pack threat defense: immediate dust Elbereth or chokepoint retreat on turn 1
            if obs.combat.is_pack_threat and not obs.combat.in_corridor:
                if not obs.combat.standing_on_elbereth and not obs.combat.hostile_ignores_elbereth:
                    obs = (yield engrave_dust_elbereth())
                    continue
                elif obs.combat.closest_hostile_dist >= 2 and obs.combat.can_retreat:
                    obs = (yield step_to_chokepoint())
                    continue
                elif obs.combat.adjacent_hostile:
                    obs = (yield melee_attack_hostile())
                    continue

            # 1.4 Heavy weapon threats (mattocks, battle-axes, two-handed swords, crossbows)
            if obs.combat.is_heavy_weapon_threat:
                if obs.combat.closest_hostile_dist >= 2:
                    if obs.inventory.has_offensive_wand:
                        obs = (yield zap_offensive_wand())
                        continue
                    elif obs.inventory.has_daggers:
                        obs = (yield throw_dagger())
                        continue
                if obs.combat.adjacent_hostile and obs.hero.hp_frac < 0.50 and obs.combat.can_retreat:
                    obs = (yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile())
                    continue

            # 2. Panic Sanctuary: Engrave Elbereth if low HP (< 35%) or facing multiple hostiles / surrounded (< 60% HP)
            if ((obs.hero.hp_frac < 0.35) or (obs.hero.hp_frac < 0.60 and (obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2))) and not obs.combat.standing_on_elbereth:
                obs = (yield engrave_dust_elbereth())
                continue

            # 3. While standing on Elbereth:
            if obs.combat.standing_on_elbereth:
                if obs.combat.hostile_ignores_elbereth and obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.40 or not obs.combat.can_retreat:
                        obs = (yield melee_attack_hostile())
                        continue
                    else:
                        obs = (yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile())
                        continue

                if obs.hero.hp_frac < 0.30 and obs.hero.turn - self.last_prayer_turn >= 850:
                    self.last_prayer_turn = obs.hero.turn
                    obs = (yield pray())
                    continue
                elif obs.hero.hp_frac < 0.60 and obs.inventory.has_healing:
                    obs = (yield quaff_healing())
                    continue
                elif obs.inventory.has_offensive_wand and obs.combat.closest_hostile_dist >= 2:
                    obs = (yield zap_offensive_wand())
                    continue
                elif obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                    obs = (yield throw_dagger())
                    continue
                elif not obs.combat.adjacent_hostile:
                    if obs.spatial.stairs_down_known:
                        obs = (yield step_to_stairs_down())
                    elif obs.spatial.has_unvisited_frontier:
                        obs = (yield step_to_frontier())
                    else:
                        obs = (yield step_away_from_hostile())
                    continue
                else:
                    obs = (yield melee_attack_hostile())
                    continue

            # 4. Ranged Harassment: Offensive wands and daggers at distance >= 2
            if obs.inventory.has_offensive_wand and obs.combat.closest_hostile_dist >= 2:
                obs = (yield zap_offensive_wand())
                continue
            if obs.combat.closest_hostile_dist >= 2 and obs.inventory.has_daggers:
                obs = (yield throw_dagger())
                continue

            # 5. Fast Dangerous Attackers (Soldier ants, killer bees, giant bats, foxes)
            if obs.combat.is_fast_dangerous:
                if (obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2) and not obs.combat.standing_on_elbereth:
                    obs = (yield engrave_dust_elbereth())
                    continue
                if obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.40 or not obs.combat.can_retreat:
                        obs = (yield melee_attack_hostile())
                        continue
                    else:
                        if not obs.combat.standing_on_elbereth:
                            obs = (yield engrave_dust_elbereth())
                        else:
                            obs = (yield melee_attack_hostile())
                        continue
                elif not obs.combat.in_corridor and obs.combat.can_retreat:
                    obs = (yield step_to_chokepoint())
                    continue

            # 6. Tactical Melee / Chokepoint Retreat
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.45 or not obs.combat.can_retreat:
                    obs = (yield melee_attack_hostile())
                else:
                    obs = (yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile())
            else:
                if obs.hero.hp_frac > 0.40:
                    obs = (yield melee_attack_hostile())
                else:
                    obs = (yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile())
            if obs.combat.hostile_count_fov == 0 and not obs.combat.adjacent_hostile:
                break
        return obs

    def handle_corpse_consumption(self, obs):
        if obs.combat.hostile_count_fov > 0:
            obs = (yield wait())
            return obs
        for corpse in obs.corpses:
            if corpse.is_safe:
                if (obs.hero.y, obs.hero.x) == (corpse.y, corpse.x):
                    obs = (yield eat_floor_corpse())
                else:
                    obs = (yield step_to(corpse.y, corpse.x))
                return obs
        obs = (yield step_to_frontier() if obs.spatial.has_unvisited_frontier else step_to_dead_end())
        return obs

    def handle_dead_end(self, obs):
        if obs.spatial.standing_on_dead_end:
            prev_hp = obs.hero.hp
            for _ in range(12):
                if (
                    obs.combat.hostile_count_fov > 0
                    or obs.combat.adjacent_hostile
                    or obs.hero.hp < prev_hp
                    or obs.spatial.stairs_down_known
                    or obs.spatial.has_unvisited_frontier
                ):
                    return obs
                obs = (yield search())
            if obs.spatial.stairs_down_known:
                obs = (yield step_to_stairs_down())
            elif obs.spatial.has_unvisited_frontier:
                obs = (yield step_to_frontier())
            else:
                obs = (yield step_to_dead_end())
            return obs
        else:
            obs = (yield step_to_dead_end())
            return obs
```

### Critical Generator Subroutine Rules:
1. Always maintain `obs` as the Observation object:
   - When calling a helper generator subroutine using `obs = yield from self.my_subroutine(obs)`, the subroutine MUST conclude with `return obs`. NEVER return a boolean (`return True` / `return False`) from a generator subroutine! Returning a boolean will overwrite `obs = True`, causing an immediate crash on the next turn.
2. Do not check generator truthiness with `if self.my_routine(obs):` because calling a generator function always returns a truthy generator object. Instead, inspect predicates on `obs` directly in `run()` (e.g. `if obs.combat.adjacent_hostile:`), and then do `obs = yield from self.handle_combat(obs)`.
3. Every execution path through a generator subroutine MUST yield at least one action (e.g. `obs = yield wait()`) before returning! If a subroutine returns without yielding and the caller continues, the generator enters an infinite busy loop at 100% CPU.

### Observation Interface (`obs`)
Every turn, `obs` provides rich sub-namespaces:
- `obs.hero`: `hp`, `max_hp`, `hp_frac`, `energy`, `energy_frac`, `ac`, `level`, `depth`, `turn`, `turns_on_level`, `gold`, `hunger_state` (SATIATED, NORMAL, HUNGRY, WEAK, FAINTING), `dungeon_branch`, `has_poison_res`, `can_enhance_skills`, `can_pray`
- `obs.status`: `is_blind`, `is_poisoned`, `is_confused`, `is_stunned`, `is_sick`, `is_encumbered`, `encumbrance_level`, `is_levitating`
- `obs.inventory`: `has_food`, `has_healing`, `has_unworn_armor`, `has_unworn_body_armor`, `has_worn_body_armor`, `get_superior_body_armor_slot()`, `has_unidentified_potion`, `has_unidentified_scroll`, `get_unidentified_potion_slot()`, `get_unidentified_scroll_slot()`, `has_daggers`, `has_offensive_wand`, `has_wand_of_teleport`, `has_scroll_of_teleport`, `get_food_slot()`, `get_healing_slot()`, `get_dagger_slot()`, `get_offensive_wand_slot()`, `items`
- `obs.combat`: `adjacent_hostile`, `hostile_count_fov`, `has_active_hostile`, `active_hostile_count`, `closest_hostile_name`, `closest_hostile_dist`, `is_surrounded`, `in_corridor`, `can_retreat`, `standing_on_elbereth`, `is_fast_dangerous`, `adjacent_floating_eye`, `adjacent_gas_spore`, `hostile_ignores_elbereth`, `has_panic_escape`, `has_safe_melee_target`, `is_corrosive_target`, `is_heavy_weapon_threat`, `is_pack_threat`
- `obs.spatial`: `stairs_down_known`, `standing_on_stairs_down`, `stairs_up_known`, `standing_on_stairs_up`, `has_unvisited_frontier`, `has_unsearched_dead_end`, `floor_explored`, `has_nearby_loot`
- `obs.dungeon`: `tile_type` (corridor, room, doorway, fountain, altar, trap), `in_shop`, `in_temple`, `is_dark_level`, `adjacent_closed_door`, `door_is_locked`, `adjacent_fountain`, `standing_on_fountain`, `adjacent_altar`, `standing_on_altar`, `can_forge_excalibur`, `can_harvest_poison`, `has_sokoban_entrance`, `sokoban_entrance_in_fov`, `sokoban_entrance_pos`, `can_solve_sokoban`, `can_sacrifice`
- `obs.epistemic`: `untested_buc_count` (int), `has_untested_items` (bool), `can_safely_wear_armor` (bool), `can_safely_quaff_healing` (bool), `items_belief` (dict of ItemBeliefState)
- `obs.agenda`: `active_goal` (str), `goal_stack` (list of str), `dungeon_phase` (str: "early_rush", "early_scaling", "mid_branches", "deep_dungeon", "castle"), `is_active(goal_name)` (bool)
- `obs.corpses`: list of `FloorCorpse(name, y, x, age_turns, is_fresh, is_poisonous, is_deadly, is_safe)`
- `obs.message`: last raw game message

### Critical NetHack 3.6.6 Mechanics & Invariants:
1. **Passive & Exploding Hazards (`adjacent_floating_eye`, `adjacent_gas_spore`, `has_safe_melee_target`, `has_active_hostile`)**: Attacking a floating eye in melee triggers a passive 70-turn paralysis gaze (`0d70`). Striking a gas spore, yellow light, or black light explodes for lethal blast or blinding. NEVER melee attack them! Destroy with ranged daggers or offensive wands. However, if an active attacker (newt, jackal, orc) is adjacent alongside the hazard, `obs.combat.has_safe_melee_target` is True: yield `melee_attack_hostile()` to eliminate the active attacker (the adapter automatically skips the passive hazard). If no ranged weapons exist and passive hazards are distant (distance >= 2, `has_active_hostile == False`), do not enter combat; continue exploration as `walkable_nav` steers around them!
2. **Emergency Panic Escape (`read_scroll_teleport`, `zap_wand_teleport`)**: When low on HP (< 25%) or surrounded by high-speed attackers, teleport away immediately. NetHack teleport scrolls and wands (zapped at `.`) instantly relocate the hero to a random safe tile.
3. **Nearby Floor Loot Scooping (`step_to_loot`)**: Defeated monsters drop armor, weapons, wands, and scrolls on their death tile. When out of combat and `obs.spatial.has_nearby_loot` is True, yield `step_to_loot()` to walk over the dropped items. Autopickup collects them, allowing `wear_armor()` to lower Armor Class (AC) towards negative numbers!
4. **Elbereth Immunity & Humanoid Combat Tactics (`hostile_ignores_elbereth`)**: Orcs, elves, and humans ignore Elbereth. While standing on Elbereth, if `obs.combat.hostile_ignores_elbereth` is True, do NOT wait passively; actively fight in melee or retreat to a 1-tile corridor chokepoint.
5. **Prayer & Divine Favor**: In NetHack, successful prayer resets divine timeout to 300 + rn2(500) turns (up to 799 turns). Enforce `obs.hero.turn - self.last_prayer_turn >= 850` before emergency prayer (`pray()`) to guarantee divine aid and avoid angering your deity ("Tyr is displeased"), which causes divine smiting or paralysis. If NetHack asks "Are you sure you want to pray? [yn]", the harness automatically answers 'n' to abort the premature prayer and prevent divine wrath.
6. **Corpse Consumption Hazards & Non-Rotting Food**: Eating a corpse takes multiple turns (`weight / 64 + 3`), leaving the hero completely helpless and vulnerable. NEVER eat a corpse if enemies are in FOV. Corpses older than 50 turns cause food poisoning; kobolds are poisonous; cockatrices cause lethal petrification without gloves. Check `corpse.is_safe` before eating! Lichen corpses and lizard corpses NEVER rot and are automatically safe to eat from inventory as carried food.
7. **Door Breaching**: Always try `open_door()` first on closed doors. Only use `kick_closed_door()` if `obs.dungeon.door_is_locked` is True (kicking unlocked doors can hurt your leg and immobilize you for 5-20 turns).
8. **Excalibur Dipping (`dip_excalibur`)**: Dipping a long sword into a fountain strictly requires **standing directly on the fountain tile** (`obs.dungeon.standing_on_fountain`). It has a 1/6 chance of forging Excalibur when lawful Valkyrie/Knight at level >= 5 (`obs.dungeon.can_forge_excalibur`). Wielding Excalibur gives +1d10 slashing damage, auto-searching for doors, and drain resistance. Never dip from an adjacent tile!
9. **Secret Doors & Corridor Dead Ends**: NetHack procedural generation regularly seals off deeper dungeon sections and staircases behind hidden secret doors located at dead-end corridors (`#`) and room perimeter walls. When visible frontiers are fully explored (`obs.spatial.has_unvisited_frontier == False`), call `step_to_dead_end()` to navigate to dead ends or perimeter walls and `search()` repeatedly until the secret door is exposed. Searching repeatedly inside open rooms will NOT find the stairs.
10. **Tactical In-Combat Emergency Healing**: Quaffing a healing potion takes only 1 turn and restores 10-20 HP. When `obs.hero.hp_frac < 0.50` and `obs.inventory.has_healing`, ALWAYS quaff healing immediately inside `handle_combat` before taking another attack! Never die with healing potions in your pack.
11. **Soldier Ants & High-Speed Attackers**: Soldier ants and killer bees move at speed 18 (nearly 2x the hero) and inflict lethal poison stings. If `obs.combat.is_fast_dangerous` is True or `obs.combat.closest_hostile_name in ("soldier ant", "killer bee")`, retreat immediately to a 1-tile corridor chokepoint (`step_to_chokepoint()`), quaff healing, or pray.
12. **Equipment & Armor Optimization (`wear_armor`)**: Defeated monsters drop helmets, boots, cloaks, and armor. When out of combat, `obs.inventory.has_unworn_armor` is True, and `obs.epistemic.can_safely_wear_armor` is True, yield `wear_armor()` to lower your Armor Class (AC). Lower AC drastically reduces damage from deep monsters.
13. **Shopkeeper, Vault Guard & Peaceful Non-Aggression**: Never attack shopkeepers, vault guards (`guard`), priests, or town watchmen (`obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain", "guard", "priest", "priestess", "oracle")`), and never kick doors when `obs.dungeon.in_shop` is True. Provoking vault guards or shopkeepers will instantly kill your hero.
14. **Exploration Pacing & No Idle Waiting**: Waiting (`wait()`) during active exploration when seeking stairs is strictly forbidden. Never yield `wait()` when frontiers are clear; instead call `handle_dead_end(obs)` or `search()` to investigate room perimeter walls. Sitting in place produces 0 turns of progress and leads to `MaxTurnsReached`.
15. **Dust Elbereth Sanctuary (`engrave_dust_elbereth`)**: Engraving "Elbereth" into the dust with bare fingers takes 1 turn and creates an impenetrable ward against 95% of non-humanoid monsters (ants, bees, bats, mumakil, leocrottas, canines). Monsters cannot attack on an Elbereth tile and flee in panic. When surrounded or HP < 60% facing multiple hostiles, yield `engrave_dust_elbereth()`.
16. **Ranged Missile Harassment (`throw_dagger`, `zap_offensive_wand`)**: Valkyries start with daggers. When enemies are at distance >= 2, yield `throw_dagger()` or `zap_offensive_wand()` to kill fast speedsters before they close to melee. Thrown daggers drop onto the ground and are automatically recovered after combat, so freely throw daggers down to 0.
17. **Altar BUC Identification (`test_altar_buc`)**: Equipping cursed armor or weapons welds them to your body. When an altar is nearby (`obs.dungeon.adjacent_altar` or `obs.dungeon.standing_on_altar`) and `obs.epistemic.has_untested_items` is True, yield `test_altar_buc()` to automatically batch-drop and identify the BUC status of untested inventory items.
18. **Poison Resistance Harvesting (`harvest_poison_res`)**: Poisonous bites from ants and bees kill heroes instantly at Depth 4+. When out of combat and `obs.dungeon.can_harvest_poison` is True and `not obs.hero.has_poison_res`, yield `harvest_poison_res()` to seek out and consume killer bee or soldier ant corpses to gain permanent poison resistance intrinsic.
19. **Hybrid Depth-Tiered HTN-BT Architecture**: Policies are structured into Universal Reflexes plus 4 depth-gated strategic phase subroutines via `obs.agenda.dungeon_phase`:
    - **Universal Reflexes**: Major Trouble Prayer (`pray()`), Immediate Stairs Descent (`descend()`), Emergency Combat Defense (`handle_combat()`), and Corpse/Food Nutrition (`eat_carried_food()`).
    - **Phase 0: Early Rush (`def phase_early_rush(self, obs)`) [DL 1-2]**: Fast descent, surface loot/armor scooping, zero dead-end lingering. If down stairs are known, prioritize descending immediately!
    - **Phase 1: Early Scaling (`def phase_early_scaling(self, obs)`) [DL 3-5]**: Power spiking: Excalibur forging at fountains (XL >= 5), upgrading body armor to mithril/iron (`replace_body_armor()`), harvesting poison resistance corpses (`harvest_poison_res()`), altar BUC testing.
    - **Phase 2: Mid Branches (`def phase_mid_branches(self, obs)`) [DL 6-10]**: Sokoban solving (`step_solve_sokoban()`), Mines avoidance (ascend on '<'), temple priest donations.
    - **Phase 3: Deep Dungeon (`def phase_deep_dungeon(self, obs)`) [DL 11-19]**: Corridor defense, Quest readiness, descent toward Castle.
    *When addressing depth-specific bottlenecks, output the specific phase method (`def phase_early_rush(...)`, `def phase_early_scaling(...)`, `def phase_mid_branches(...)`) rather than endlessly micro-tuning `handle_combat`.*
20. **Gnomish Mines Avoidance & Main Dungeon Steering**: The Gnomish Mines entrance (`obs.hero.dungeon_branch == "mines"`) appears on Depths 2–4. The Mines are pitch dark and infested with deadly gnome packs with wands and crossbows. If `obs.hero.dungeon_branch == "mines"`, immediately ascend back up to the main dungeon (`yield ascend()` if on stairs up, or `yield step_to_stairs_up()`). NetHackAdapter will dynamically record and prune that Mines staircase from `known_stairs_down` upon returning to the Dungeons of Doom, allowing the hero to locate and descend the true main dungeon staircase down toward Depth 20.
21. **Adjacent Passive Hazard Ranged Elimination & Dead-End Safety**: Throwing daggers, darts, arrows, or rocks (`throw_dagger()`) at an adjacent floating eye (`obs.combat.adjacent_floating_eye`) is 100% safe at distance 1 and does NOT trigger the passive melee paralysis attack. Never restrict ranged attacks against floating eyes to distance >= 2. NEVER strike a floating eye in melee under any circumstances (doing so inflicts 70 turns of paralysis and causes certain death); if trapped with 0 missiles, call `step_away_from_hostile()` to trigger the deadlock circuit breaker. Gas spores, yellow lights, and black lights explode on melee contact, so retreat away or fight adjacent safe targets.
22. **Proactive In-Combat Nutrition & Divine Feeding**: In NetHack 3.6, divine feeding by prayer (`pray()`) strictly requires `hunger_state >= 4` ("Fainting"). Praying at `hunger_state == 3` ("Weak") wastes divine favor. Consume carried food (`eat_carried_food()`) proactively as soon as `hunger_state >= 1` ("Hungry") outside combat, or at `hunger_state >= 2` ("Weak") when standing on Elbereth, out of melee reach, or adjacent only to passive hazards. Only pray for divine feeding when packaged food is exhausted and `hunger_state >= 4`. Never wait until fainting to eat carried food while monsters are in FOV.
23. **Sokoban Branch Solving (`solve_sokoban`)**: When `obs.dungeon.is_sokoban` and `obs.dungeon.has_boulders`, yield `solve_sokoban()`. The underlying deterministic graph solver calculates optimal boulder pushes into pits without Luck penalties or corner deadlocks, securing the Bag of Holding or Amulet of Reflection.
24. **Castle Drawbridge Breaching (`breach_drawbridge`)**: At the Castle (Depths 25-29), closed drawbridges kill heroes instantly if bumped or closed upon. When `obs.dungeon.drawbridge_in_fov` or `obs.dungeon.closest_drawbridge_pos` is detected, yield `breach_drawbridge()` to align at distance 2 and zap a wand of striking to safely shatter the bridge.
25. **Gehennom Mazes & Tunneling (`dig_tunnel`)**: Gehennom (Depths 30-45) consists of vast unlit stone mazes. When in Gehennom and equipped with a wand of digging or pickaxe, yield `dig_tunnel()` to blast straight-line rays through the rock directly toward target stairs or the Sanctum, saving thousands of turns.
26. **Unicorn Horn Affliction Cleansing (`apply_unicorn_horn`)**: When afflicted with sickness, hallucination, blindness, or confusion (`obs.hero.is_sick` or `obs.hero.is_blind` or `obs.hero.is_hallucinating`), yield `apply_unicorn_horn()` to immediately cleanse status debuffs and restore lost ability scores.
27. **Seven Canonical Ascension Phases**:
    - Phase 1 (Early-Game, DL 1-5): AC reduction, poison resistance harvesting, Excalibur forging.
    - Phase 2 (Sokoban & Gearing, DL 6-10): complete Sokoban with `solve_sokoban()`, secure Reflection/Bag of Holding.
    - Phase 3 (Mid-Game & Quest, DL 11-18): complete Valkyrie Quest, obtain the Bell of Opening and Orb of Fate.
    - Phase 4 (Castle & Ascension Kit, DL 20-29): breach drawbridge with `breach_drawbridge()`, wish for GDSM/SDSM and speed boots.
    - Phase 5 (Gehennom Descent, DL 30-45): tunnel mazes with `dig_tunnel()`, defeat Vlad for Candelabra, Rodney for Book of the Dead.
    - Phase 6 (Sanctum & Invocation, DL 50): perform Bell-Book-Candle ritual, defeat High Priest of Moloch, grab the Amulet of Yendor.
    - Phase 7 (The Run & Astral Plane): ascend to dungeon level 1, traverse Elemental Planes (Earth, Air, Fire, Water), use Conflict to cross the Astral Plane and offer the Amulet at the Lawful Altar to Ascend!
28. **Weapon Skill Enhancement (`enhance_weapon_skill`, `obs.hero.can_enhance_skills`)**: In NetHack, Valkyries start at Basic (+0 to-hit, +0 damage). Striking enemies trains skills: 80 hits unlocks Skilled (+2 to-hit, +1 damage), 180 hits unlocks Expert (+3 to-hit, +2 damage). When `obs.hero.can_enhance_skills` is True outside combat, yield `enhance_weapon_skill()` to dramatically increase melee accuracy and damage.
29. **Superior Body Armor Replacement (`replace_body_armor`)**: NetHack forbids wearing body armor while already wearing body armor. Dropped dwarvish mithril coats (AC 5) or iron cuirasses give vastly superior protection over the starting leather jacket (AC 1). When `obs.inventory.get_superior_body_armor_slot()` is not None, yield `replace_body_armor()` to take off the inferior jacket and wear the superior coat, dropping AC towards negative numbers.
30. **Pack & Herd Tactical Defense (`obs.combat.is_pack_threat`)**: On DL 4–6, monsters like rothes spawn in herds of 3–5 animals. Each rothe executes 3 attacks per turn, dealing up to 25+ damage per turn against unarmored heroes. When `obs.combat.is_pack_threat` is True in open rooms (`not in_corridor`), do NOT stand and trade melee blows; engrave dust Elbereth (`engrave_dust_elbereth()`) or retreat to a 1-tile corridor chokepoint (`step_to_chokepoint()`).
31. **Corrosive Hazard Melee Prohibition (`obs.combat.is_corrosive_target`)**: Striking acidic monsters (acid blobs, ochre jellies) in melee instantly corrodes weapons (-1 enchantment/damage) and deals passive acid splash damage. NEVER attack corrosive targets in melee. Eliminate them with ranged missiles (`throw_dagger()`) or offensive wands (`zap_offensive_wand()`), or retreat (`step_away_from_hostile()`).
32. **Heavy Weapon Threat Gating (`obs.combat.is_heavy_weapon_threat`)**: Orc chieftains, captains, and gnomes wielding two-handed swords, battle-axes, or crossbows deal deadly burst damage capable of one-shotting heroes. When `is_heavy_weapon_threat` is detected at distance >= 2, harass with thrown missiles (`throw_dagger()`) or offensive wands, or retreat to 1-tile corridor chokepoints (`step_to_chokepoint()`) rather than rushing into open melee.
33. **Emergency Unidentified Consumables Panic Consumption (`quaff_emergency_potion`, `read_emergency_scroll`)**: When HP is critically low (< 20%), prayer is on cooldown or unavailable, and no identified healing potions remain, sitting idle or taking another fatal attack is a catastrophic failure mode. Unidentified potions (extra healing, full healing, speed, gain energy) and unidentified scrolls (teleportation, earth, scare monster) have high probabilities of saving the hero's life. Yield `quaff_emergency_potion()` or `read_emergency_scroll()`.
34. **Sokoban Branch Entrance Navigation (`obs.dungeon.has_sokoban_entrance`, `step_to_sokoban_entrance`)**: On Depths 6–10, the Sokoban entrance generates as a distinct up-stair `<` that is not the floor's arrival staircase. When `obs.dungeon.has_sokoban_entrance` is True, yield `step_to_sokoban_entrance()` to navigate to the entrance and ascend into Sokoban to acquire non-rotting food and the Bag of Holding or Amulet of Reflection.

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
- `query_root_causes(run_id, window)`: Empirical root cause attribution breakdown table across 9 canonical failure archetypes (secret door stalls, starvation fainting, ping-pong oscillation, armor deficit, premature prayer, passive hazards, swarms, fast predators, general combat).
- `query_wiki(query)`: Search offline NetHack 3.6.6 encyclopedia (monsters, intrinsics, corpses, rituals).
- `query_invariants(query, category)`: Query the canonical LOX Empirical Invariant Knowledge Base for ground-truth tactical rules, anti-patterns, and verified code patterns (topics: 'prayer', 'hunger', 'floating eye', 'gas spore', 'door', 'dead end', 'elbereth', 'mines').
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

### Output Requirement (Targeted Method Splicing Required):
Provide 1 brief rationale sentence, then your modified method(s) in a single ```python ... ``` block.
Output ONLY the specific method(s) you are updating or adding (e.g. `def phase_early_rush(self, obs): ...`, `def phase_early_scaling(self, obs): ...`, `def phase_mid_branches(self, obs): ...`, `def handle_combat(self, obs): ...`, or `def run(self, obs): ...`).
Do NOT output unchanged methods or the full `class Agent` wrapper. The engine automatically splices your updated method into `class Agent` via AST, preserving all other verified subroutines intact and eliminating indentation errors.
"""


def build_user_prompt(
    current_policy: str, trigger_reason: str, status_report: str
) -> str:
    """User prompt presenting the empirical autopsy, ranked mortality causes, relevant invariants, and current policy."""
    from lox.knowledge import REGISTRY

    relevant_invs = REGISTRY.get_relevant_invariants_for_trigger(
        trigger_reason, top_k=3
    )
    invariants_block = REGISTRY.format_llm_reference(relevant_invs)
    if invariants_block:
        invariants_block = f"\n{invariants_block}\n"

    return f"""### Empirical Incident Report:
{status_report}

### Synthesis Objective:
{trigger_reason}

### Diagnostic Notice:
The telemetry database `data/lox.duckdb` holds complete per-tick flight recordings and death traces.
Call `get_death_autopsy_trace()` to inspect the exact final 15 ticks, `query_invariants(topic)` to check verified tactical rules, or `query_wiki(monster_name)` to look up mechanics before writing code.
{invariants_block}
### Current Policy:
```python
{current_policy.strip()}
```

Synthesize your revised method(s) to address the empirical mortality bottlenecks. Output ONLY the specific modified method(s) (e.g. `def handle_combat(self, obs): ...` or `def run(self, obs): ...`) in a single ```python ... ``` block.
"""
