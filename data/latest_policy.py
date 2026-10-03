class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0
        self.retreat_streak = 0
        self.altar_tested = False
        self.last_depth = 1
        self.last_searched_pos = None

    def run(self, obs):
        while True:
            if obs.hero.depth != self.last_depth:
                self.altar_tested = False
                self.last_searched_pos = None
                self.last_depth = obs.hero.depth

            # 1. Absolute Emergency Survival (Major Trouble: Weak/Fainting or <15% HP)
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state >= 3):
                if obs.inventory.has_food and obs.hero.hunger_state >= 2:
                    obs = yield eat_carried_food(subroutine="eating_food")
                    continue
                if obs.hero.turn - self.last_prayer_turn >= 350:
                    if obs.combat.adjacent_hostile and not obs.combat.standing_on_elbereth:
                        obs = yield engrave_dust_elbereth(subroutine="emergency_elbereth")
                        continue
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray(subroutine="emergency_prayer")
                    continue

            # 2. Combat Logic (Highest Priority - Reactive Reflexes)
            if obs.combat.hostile_count_fov > 0:
                obs = yield from self.handle_combat(obs)
                continue

            # 3. Hunger Prevention (Strictly safe: No hostiles in FOV)
            if obs.hero.hunger_state >= 2:
                if obs.inventory.has_food:
                    obs = yield eat_carried_food(subroutine="eating_food")
                    continue
                elif obs.corpses:
                    if not obs.combat.in_corridor:
                        obs = yield from self.handle_corpse_consumption(obs)
                        continue

            # 4. Equipment Optimization
            if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
                obs = yield wear_armor(subroutine="wearing_armor")
                continue

            # 5. Altar BUC Testing (Strictly out of combat)
            if obs.epistemic.has_untested_items and not self.altar_tested:
                if obs.dungeon.standing_on_altar:
                    self.altar_tested = True
                    obs = yield test_altar_buc(subroutine="altar_buc")
                    continue
                elif obs.dungeon.adjacent_altar:
                    obs = yield step_to_altar(subroutine="altar_buc")
                    continue

            # 6. Weapon Scaling (Excalibur)
            if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
                if obs.dungeon.standing_on_fountain:
                    obs = yield dip_excalibur(subroutine="forge_excalibur")
                    continue
                elif obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
                    obs = yield step_to_fountain(subroutine="forge_excalibur")
                    continue

            # 7. Poison Resistance Harvesting
            if obs.dungeon.can_harvest_poison and not obs.hero.has_poison_res and obs.hero.hp_frac > 0.90:
                obs = yield harvest_poison_res(subroutine="harvest_poison")
                continue

            # 8. Navigation & Exploration (Strictly No Idling)
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = yield descend(subroutine="descending")
                continue
            elif obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                    obs = yield kick_closed_door(subroutine="door_navigation")
                else:
                    obs = yield open_door(subroutine="door_navigation")
                continue
            elif obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down(subroutine="navigating_to_stairs")
                continue
            elif obs.spatial.has_unvisited_frontier:
                self.search_count = 0 
                obs = yield step_to_frontier(subroutine="exploring_frontier")
                continue
            elif obs.dungeon.closed_door_in_fov:
                obs = yield step_to_closed_door(subroutine="navigating_to_door")
                continue
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield from self.handle_dead_end(obs)
                continue
            else:
                # Perimeter search to find secret doors/stairs. 
                # Never yield wait() here; always move or search.
                obs = yield from self.handle_dead_end(obs)
                continue

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0:
            # 1. Floating Eye: Absolute melee prohibition. It is stationary (speed 0) and harmless at distance >= 2.
            if obs.combat.closest_hostile_name == "floating eye":
                if obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                    obs = yield throw_dagger(subroutine="combat_ranged")
                    continue
                elif obs.combat.hostile_count_fov == 1:
                    # Eye cannot move or attack at range. Yield navigation action to move around it.
                    if obs.spatial.stairs_down_known:
                        obs = yield step_to_stairs_down(subroutine="navigating_to_stairs")
                    elif obs.spatial.has_unvisited_frontier:
                        obs = yield step_to_frontier(subroutine="exploring_frontier")
                    else:
                        obs = yield step_to_dead_end(subroutine="hunting_secret_door")
                    continue
                elif obs.combat.adjacent_hostile and obs.combat.can_retreat:
                    obs = yield step_away_from_hostile(subroutine="combat_retreat")
                    continue

            # 2. Gas Spore: Never attack in melee (explodes on contact).
            if obs.combat.closest_hostile_name == "gas spore" or obs.combat.gas_spore_in_fov:
                if obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                    obs = yield throw_dagger(subroutine="combat_ranged")
                    continue
                elif obs.combat.hostile_count_fov == 1:
                    # Spore moves at speed 3; hero moves at speed 12. Yield navigation action to outrun it.
                    if obs.spatial.stairs_down_known:
                        obs = yield step_to_stairs_down(subroutine="navigating_to_stairs")
                    elif obs.spatial.has_unvisited_frontier:
                        obs = yield step_to_frontier(subroutine="exploring_frontier")
                    else:
                        obs = yield step_to_dead_end(subroutine="hunting_secret_door")
                    continue
                elif obs.combat.adjacent_hostile and obs.combat.can_retreat:
                    obs = yield step_away_from_hostile(subroutine="combat_retreat")
                    continue

            # 3. Non-Aggression (Peaceful shopkeepers / town watch)
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain", "aligned priest") or obs.dungeon.in_shop:
                if obs.combat.hostile_count_fov == 1:
                    if obs.spatial.stairs_down_known:
                        obs = yield step_to_stairs_down(subroutine="navigating_to_stairs")
                    elif obs.spatial.has_unvisited_frontier:
                        obs = yield step_to_frontier(subroutine="exploring_frontier")
                    else:
                        obs = yield step_to_dead_end(subroutine="hunting_secret_door")
                    continue
                elif obs.combat.adjacent_hostile:
                    obs = yield step_away_from_hostile(subroutine="combat_retreat")
                    continue

            # 4. Tactical In-Combat Healing: Aggressive priority if HP is low
            if obs.hero.hp_frac < 0.60 and obs.inventory.has_healing:
                obs = yield quaff_healing(subroutine="combat_healing")
                continue

            # 5. Panic Sanctuary: Engrave Elbereth if low HP (<45%) or surrounded
            if (obs.hero.hp_frac < 0.45 or obs.combat.is_surrounded) and not obs.combat.standing_on_elbereth:
                obs = yield engrave_dust_elbereth(subroutine="recovering_on_elbereth")
                continue

            # 6. While standing on Elbereth: Recovery and Tactical Reset
            if obs.combat.standing_on_elbereth:
                # Emergency healing / prayer
                if obs.hero.hp_frac < 0.30 and obs.hero.turn - self.last_prayer_turn >= 350:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray(subroutine="emergency_prayer")
                    continue
                elif obs.hero.hp_frac < 0.65 and obs.inventory.has_healing:
                    obs = yield quaff_healing(subroutine="combat_healing")
                    continue

                # Check if any adjacent hostile ignores Elbereth (humanoids, orcs, elves, gnomes, dwarves, kobolds)
                fearless_adjacent = any(
                    any(k in m.lower() for k in ("orc", "goblin", "gnome", "dwarf", "elf", "human", "soldier", "werejackal", "shopkeeper", "watchman", "kobold"))
                    for m in obs.combat.adjacent_monsters
                )

                if fearless_adjacent:
                    # Adjacent monster ignores Elbereth! Fight or retreat
                    if obs.hero.hp_frac > 0.40 or not obs.combat.can_retreat:
                        obs = yield melee_attack_hostile(subroutine="combat_melee")
                    else:
                        obs = yield step_away_from_hostile(subroutine="combat_retreat")
                    continue

                # If hostile is adjacent: NEVER wait for free hits! Fight or re-engrave if ward scuffed
                if obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac < 0.35:
                        obs = yield engrave_dust_elbereth(subroutine="recovering_on_elbereth")
                    else:
                        obs = yield melee_attack_hostile(subroutine="combat_melee")
                    continue

                # NO adjacent monsters -> SAFE TO REST & REGENERATE!
                if obs.hero.hp_frac < 0.80:
                    obs = yield wait(subroutine="recovering_on_elbereth")
                    continue

                # HP >= 80%: HP recovered! Clear adjacent hostiles or missile them
                if obs.combat.adjacent_hostile:
                    obs = yield melee_attack_hostile(subroutine="combat_melee")
                    continue
                elif obs.combat.closest_hostile_dist >= 2 and obs.inventory.has_daggers:
                    obs = yield throw_dagger(subroutine="combat_ranged")
                    continue
                else:
                    if obs.spatial.stairs_down_known:
                        obs = yield step_to_stairs_down(subroutine="navigating_to_stairs")
                    elif obs.spatial.has_unvisited_frontier:
                        obs = yield step_to_frontier(subroutine="exploring_frontier")
                    else:
                        obs = yield melee_attack_hostile(subroutine="combat_melee")
                    continue

            # 7. High-Speed Attackers (soldier ants, killer bees, giant spiders)
            if obs.combat.is_fast_dangerous:
                if obs.combat.closest_hostile_dist >= 2 and obs.inventory.has_daggers:
                    obs = yield throw_dagger(subroutine="combat_ranged")
                    continue
                elif obs.combat.adjacent_hostile:
                    if not obs.hero.has_poison_res and not obs.combat.standing_on_elbereth:
                        obs = yield engrave_dust_elbereth(subroutine="recovering_on_elbereth")
                    elif obs.hero.hp_frac > 0.50:
                        obs = yield melee_attack_hostile(subroutine="combat_melee")
                    else:
                        obs = yield engrave_dust_elbereth(subroutine="recovering_on_elbereth")
                    continue
                elif not obs.combat.in_corridor and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint(subroutine="retreating_to_corridor")
                    continue

            # 8. Ranged Missile Harassment
            if obs.combat.closest_hostile_dist >= 2 and obs.inventory.has_daggers:
                obs = yield throw_dagger(subroutine="combat_ranged")
                continue

            # 9. Melee Engagement Logic
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.50 or not obs.combat.can_retreat:
                    obs = yield melee_attack_hostile(subroutine="combat_melee")
                else:
                    if obs.combat.in_corridor:
                        obs = yield step_to_chokepoint(subroutine="retreating_to_corridor")
                    else:
                        obs = yield step_away_from_hostile(subroutine="combat_retreat")
            else:
                if obs.hero.hp_frac > 0.45:
                    obs = yield melee_attack_hostile(subroutine="combat_melee")
                else:
                    if obs.combat.in_corridor:
                        # In corridor: wait for hostile to approach to gain first strike
                        obs = yield wait(subroutine="combat_melee")
                    else:
                        obs = yield step_to_chokepoint(subroutine="retreating_to_corridor")

            if obs.combat.hostile_count_fov == 0:
                break
        return obs

    def handle_corpse_consumption(self, obs):
        if obs.combat.hostile_count_fov > 0:
            obs = yield step_away_from_hostile(subroutine="combat_retreat")
            return obs
        for corpse in obs.corpses:
            if corpse.is_safe:
                if (obs.hero.y, obs.hero.x) == (corpse.y, corpse.x):
                    obs = yield eat_floor_corpse(subroutine="eating_corpse")
                else:
                    obs = yield step_to(corpse.y, corpse.x, subroutine="harvesting_corpse")
                return obs
        # Avoid wait() in exploration; move to frontier instead
        if obs.spatial.has_unvisited_frontier:
            obs = yield step_to_frontier(subroutine="exploring_frontier")
        else:
            obs = yield step_to_dead_end(subroutine="hunting_secret_door")
        return obs

    def handle_dead_end(self, obs):
        current_pos = (obs.hero.y, obs.hero.x)
        if obs.spatial.standing_on_dead_end and current_pos != self.last_searched_pos:
            self.last_searched_pos = current_pos
            for _ in range(8):
                if obs.combat.hostile_count_fov > 0 or obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                    break
                obs = yield search(subroutine="hunting_secret_door")
        else:
            obs = yield step_to_dead_end(subroutine="hunting_secret_door")
        return obs
