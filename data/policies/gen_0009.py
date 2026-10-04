class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.last_searched_pos = None
        self.search_count = 0

    def run(self, obs):
        while True:
            # 1. Absolute Emergency Survival (Major Trouble: Fainting or <15% HP)
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state == 4):
                if obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue

            # 2. Combat Logic (Highest Priority)
            if obs.combat.hostile_count_fov > 0:
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

            # 4. Equipment Optimization
            if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
                obs = yield wear_armor()
                continue

            # 5. Weapon Scaling (Excalibur)
            if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
                if obs.dungeon.standing_on_fountain:
                    obs = yield dip_excalibur()
                    continue
                elif obs.dungeon.adjacent_fountain:
                    obs = yield step_to_fountain()
                    continue

            # 6. Safe Loot Scooping (Prevent "Grid Bug" deaths by ensuring no adjacent hostiles)
            if obs.spatial.has_nearby_loot and not obs.combat.adjacent_hostile:
                # Only scoop if we are healthy enough to survive a surprise trap/encounter
                if obs.hero.hp_frac > 0.30:
                    obs = yield step_to_loot()
                    continue

            # 7. Altar BUC Testing
            if obs.epistemic.has_untested_items and obs.hero.hp_frac > 0.50:
                if obs.dungeon.standing_on_altar:
                    obs = yield test_altar_buc()
                    continue
                elif obs.dungeon.adjacent_altar:
                    obs = yield step_to_altar()
                    continue

            # 8. Poison Resistance Harvesting
            if obs.dungeon.can_harvest_poison and not obs.hero.has_poison_res and obs.hero.hp_frac > 0.90:
                obs = yield harvest_poison_res()
                continue

            # 9. Navigation & Exploration
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
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield from self.handle_dead_end(obs)
            else:
                # Search perimeter walls to find secret doors/stairs
                obs = yield from self.handle_dead_end(obs)

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0:
            # 0. Emergency Panic Escape
            if (obs.hero.hp_frac < 0.25 or obs.combat.is_surrounded) and obs.combat.has_panic_escape:
                if obs.inventory.has_scroll_of_teleport:
                    obs = yield read_scroll_teleport()
                    continue
                elif obs.inventory.has_wand_of_teleport:
                    obs = yield zap_wand_teleport()
                    continue

            # 1. Non-Aggression
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain") or obs.dungeon.in_shop:
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue

            # 2. Critical Healing (Aggressive threshold)
            if obs.hero.hp_frac < 0.60 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 3. Gaze/Touch Hazard Avoidance
            if obs.combat.adjacent_floating_eye or obs.combat.adjacent_gas_spore or \
               obs.combat.closest_hostile_name in ("floating eye", "gas spore"):
                if obs.combat.closest_hostile_dist >= 2:
                    if obs.inventory.has_offensive_wand:
                        obs = yield zap_offensive_wand()
                        continue
                    if obs.inventory.has_daggers:
                        obs = yield throw_dagger()
                        continue
                obs = yield step_away_from_hostile()
                continue

            # 4. Ranged Harassment
            if not obs.combat.adjacent_hostile and obs.combat.closest_hostile_dist <= 4:
                if obs.inventory.has_offensive_wand:
                    obs = yield zap_offensive_wand()
                    continue
                elif obs.inventory.has_daggers:
                    obs = yield throw_dagger()
                    continue

            # 5. Panic Sanctuary (Elbereth)
            if (obs.hero.hp_frac < 0.35 or obs.combat.is_surrounded) and not obs.combat.standing_on_elbereth:
                # Only engrave if we aren't being actively outranged or if we are desperate
                if obs.combat.adjacent_hostile or obs.hero.hp_frac < 0.20:
                    obs = yield engrave_dust_elbereth()
                    continue

            # 6. Elbereth Recovery Logic
            if obs.combat.standing_on_elbereth:
                if obs.combat.hostile_ignores_elbereth:
                    if obs.hero.hp_frac > 0.40:
                        obs = yield melee_attack_hostile()
                        continue
                    else:
                        obs = yield step_away_from_hostile()
                        continue
                
                if obs.hero.hp_frac < 0.60 and obs.inventory.has_healing:
                    obs = yield quaff_healing()
                    continue
                if obs.hero.hp_frac < 0.40 and obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue
                
                if not obs.combat.adjacent_hostile:
                    if obs.spatial.stairs_down_known:
                        obs = yield step_to_stairs_down()
                        break 
                    elif obs.spatial.has_unvisited_frontier:
                        obs = yield step_to_frontier()
                        break
                    obs = yield step_away_from_hostile()
                    continue
                else:
                    obs = yield melee_attack_hostile()
                    continue

            # 7. High-Speed Attackers / Tactical Retreat
            if obs.combat.is_fast_dangerous or (obs.hero.hp_frac < 0.50 and obs.combat.can_retreat):
                if obs.combat.can_retreat:
                    obs = yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()
                    continue

            # 8. Melee Engagement
            if obs.combat.adjacent_hostile:
                obs = yield melee_attack_hostile()
            else:
                if obs.hero.hp_frac > 0.50:
                    obs = yield melee_attack_hostile()
                else:
                    obs = yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()

            if obs.combat.hostile_count_fov == 0:
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
        if obs.combat.hostile_count_fov > 0:
            obs = yield step_away_from_hostile()
            return obs

        current_pos = (obs.hero.y, obs.hero.x)
        if obs.spatial.standing_on_dead_end:
            if current_pos != self.last_searched_pos:
                self.last_searched_pos = current_pos
                for _ in range(8):
                    if obs.combat.hostile_count_fov > 0 or obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                        break
                    obs = yield search()
            else:
                if obs.spatial.has_unvisited_frontier:
                    obs = yield step_to_frontier()
                elif obs.spatial.stairs_down_known:
                    obs = yield step_to_stairs_down()
                else:
                    obs = yield search()
        else:
            obs = yield step_to_dead_end()
        return obs
