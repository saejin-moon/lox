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

            # 2. Immediate Staircase Descent (Dungeon Progression & Escape)
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = yield descend()
                continue

            # 3. Combat Logic (Highest Priority)
            if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
                obs = yield from self.handle_combat(obs)
                continue

            # 4. Post-Combat Recovery & Healing
            if obs.hero.hp_frac < 0.95 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 5. Safe Natural HP Regeneration (Do not explore corridors critically wounded)
            if obs.hero.hp_frac < 0.65 and obs.hero.hunger_state < 2:
                if not obs.combat.standing_on_elbereth:
                    obs = yield engrave_dust_elbereth()
                    continue
                else:
                    obs = yield wait()
                    continue

            # 6. Opportunistic Fresh Corpse Nutrition (Eat kills before they rot)
            safe_corpses = [c for c in obs.corpses if c.is_safe]
            if safe_corpses and obs.hero.hunger_state < 3 and not obs.combat.adjacent_hostile:
                obs = yield from self.handle_corpse_consumption(obs)
                continue

            # 7. Emergency Hunger Prevention (Carried rations only when hungry and no fresh corpses)
            if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
                obs = yield eat_carried_food()
                continue

            # 5. Equipment Optimization
            if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
                obs = yield wear_armor()
                continue

            # 6. Weapon Scaling (Excalibur)
            if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
                if obs.dungeon.standing_on_fountain:
                    obs = yield dip_excalibur()
                    continue
                elif obs.dungeon.adjacent_fountain:
                    obs = yield step_to_fountain()
                    continue

            # 7. Safe Loot Scooping
            if obs.spatial.has_nearby_loot and obs.hero.hp_frac > 0.80:
                obs = yield step_to_loot()
                continue

            # 8. Altar BUC Testing
            if obs.epistemic.has_untested_items and obs.hero.hp_frac > 0.80:
                if obs.dungeon.standing_on_altar:
                    obs = yield test_altar_buc()
                    continue
                elif obs.dungeon.adjacent_altar:
                    obs = yield step_to_altar()
                    continue

            # 9. Poison Resistance Harvesting
            if obs.dungeon.can_harvest_poison and not obs.hero.has_poison_res and obs.hero.hp_frac > 0.90:
                obs = yield harvest_poison_res()
                continue

            # 10. Navigation & Exploration
            if obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
                continue

            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = yield descend()
            elif obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield from self.handle_dead_end(obs)
            else:
                # Prevent idling: search perimeter walls to find secret doors/stairs
                obs = yield from self.handle_dead_end(obs)

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            # 0. Immediate Exit if on stairs (Tactical retreat)
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = yield descend()
                return obs

            # 0.1 Emergency Panic Escape (Teleport)
            if (obs.hero.hp_frac < 0.20 or obs.combat.is_surrounded) and obs.combat.has_panic_escape:
                if obs.inventory.has_scroll_of_teleport:
                    obs = yield read_scroll_teleport()
                    continue
                elif obs.inventory.has_wand_of_teleport:
                    obs = yield zap_wand_teleport()
                    continue

            # 1. Non-Aggression (Shopkeepers/Town)
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain") or obs.dungeon.in_shop:
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue

            # 2. Proactive Healing (Critical: Quaff before taking another hit)
            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 3. Tactical Sanctuary (Elbereth)
            if (obs.hero.hp_frac < 0.35 or obs.combat.is_surrounded) and not obs.combat.standing_on_elbereth:
                obs = yield engrave_dust_elbereth()
                continue

            # 4. Elbereth Recovery & Defense
            if obs.combat.standing_on_elbereth:
                if obs.combat.hostile_ignores_elbereth:
                    if obs.hero.hp_frac > 0.60 or not obs.combat.can_retreat:
                        obs = yield melee_attack_hostile()
                        continue
                    else:
                        obs = yield step_away_from_hostile()
                        continue
                
                if obs.hero.hp_frac < 0.80 and obs.inventory.has_healing:
                    obs = yield quaff_healing()
                    continue
                if obs.hero.hp_frac < 0.50 and obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue
                
                if not obs.combat.adjacent_hostile:
                    if obs.spatial.stairs_down_known:
                        obs = yield step_to_stairs_down()
                        return obs
                    obs = yield step_away_from_hostile()
                    continue
                else:
                    obs = yield melee_attack_hostile()
                    continue

            # 5. Gaze/Touch Hazard Avoidance (Floating Eyes/Gas Spores)
            if obs.combat.adjacent_floating_eye or obs.combat.adjacent_gas_spore or \
               (obs.combat.closest_hostile_name and obs.combat.closest_hostile_name in ("floating eye", "gas spore")):
                if obs.inventory.has_offensive_wand:
                    obs = yield zap_offensive_wand()
                    continue
                if obs.inventory.has_daggers:
                    obs = yield throw_dagger()
                    continue
                obs = yield step_away_from_hostile()
                continue

            # 6. Aggressive Ranged Harassment (Keep distance from fast/dangerous enemies)
            if not obs.combat.adjacent_hostile and obs.combat.closest_hostile_dist <= 5:
                if obs.inventory.has_offensive_wand:
                    obs = yield zap_offensive_wand()
                    continue
                elif obs.inventory.has_daggers:
                    obs = yield throw_dagger()
                    continue

            # 7. High-Speed Attackers / Tactical Retreat
            if obs.combat.is_fast_dangerous or (obs.hero.hp_frac < 0.60 and obs.combat.can_retreat):
                if obs.combat.in_corridor:
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield step_away_from_hostile()
                continue

            # 8. Melee Engagement
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.70 or not obs.combat.can_retreat:
                    obs = yield melee_attack_hostile()
                else:
                    obs = yield step_away_from_hostile()
            else:
                if obs.hero.hp_frac > 0.80:
                    obs = yield melee_attack_hostile()
                else:
                    if obs.combat.in_corridor:
                        obs = yield step_to_chokepoint()
                    else:
                        obs = yield step_away_from_hostile()

            if obs.combat.hostile_count_fov == 0 and not obs.combat.adjacent_hostile:
                break
        return obs

    def handle_corpse_consumption(self, obs):
        if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
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
        # Safety check: Never search if we are wounded or enemies are nearby
        if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile or obs.hero.hp_frac < 0.90:
            if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
                obs = yield from self.handle_combat(obs)
                return obs
            obs = yield wait()
            return obs

        current_pos = (obs.hero.y, obs.hero.x)
        if obs.spatial.standing_on_dead_end:
            if current_pos != self.last_searched_pos:
                self.last_searched_pos = current_pos
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
