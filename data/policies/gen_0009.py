class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.last_searched_pos = None
        self.retreat_streak = 0

    def run(self, obs):
        while True:
            # 1. Absolute Emergency Survival (Fainting or Critical HP)
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state == 4):
                if obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue

            # 2. Gnomish Mines Avoidance (Immediate Exit)
            if obs.hero.dungeon_branch == "mines":
                if obs.spatial.standing_on_stairs_up:
                    obs = yield ascend()
                    continue
                elif obs.spatial.stairs_up_known:
                    obs = yield step_to_stairs_up()
                    continue

            # 3. Combat Logic (Highest Priority)
            if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
                obs = yield from self.handle_combat(obs)
                continue

            # 4. Post-Combat Recovery (Top off HP before exploring)
            if obs.hero.hp_frac < 0.90 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 5. Hunger Prevention
            if obs.hero.hunger_state >= 2:
                if obs.inventory.has_food:
                    obs = yield eat_carried_food()
                    continue
                elif obs.corpses:
                    obs = yield from self.handle_corpse_consumption(obs)
                    continue

            # 6. Equipment Optimization
            if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
                obs = yield wear_armor()
                continue

            # 7. Weapon Scaling (Excalibur)
            if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
                if obs.dungeon.standing_on_fountain:
                    obs = yield dip_excalibur()
                    continue
                elif obs.dungeon.adjacent_fountain:
                    obs = yield step_to_fountain()
                    continue

            # 8. Nearby Floor Loot Scooping
            if obs.spatial.has_nearby_loot and not obs.combat.adjacent_hostile:
                if not obs.status.is_encumbered or obs.status.encumbrance_level < 3:
                    obs = yield step_to_loot()
                    continue

            # 9. Altar BUC Testing
            if obs.dungeon.standing_on_altar and obs.epistemic.has_untested_items:
                obs = yield test_altar_buc()
                continue
            elif obs.dungeon.adjacent_altar and obs.epistemic.has_untested_items:
                obs = yield step_to_altar()
                continue

            # 10. Poison Resistance Harvesting
            if obs.dungeon.can_harvest_poison and not obs.hero.has_poison_res and obs.hero.hp_frac > 0.90:
                obs = yield harvest_poison_res()
                continue

            # 11. Navigation & Exploration
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = yield descend()
                continue
            elif obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
                continue
            elif obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
                continue
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
                continue
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield from self.handle_dead_end(obs)
                continue
            else:
                obs = yield from self.handle_dead_end(obs)

    def handle_combat(self, obs):
        self.retreat_streak = 0
        while obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            # 0. Emergency Panic Escape
            if (obs.hero.hp_frac < 0.25 or obs.combat.is_surrounded) and obs.combat.has_panic_escape:
                if obs.inventory.has_scroll_of_teleport:
                    obs = yield read_scroll_teleport()
                    continue
                elif obs.inventory.has_wand_of_teleport:
                    obs = yield zap_wand_teleport()
                    continue

            # 1. Hard-Stop Tactical Healing (Aggressive threshold)
            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 2. Passive & Exploding Hazards (STRICT NO-MELEE)
            if (obs.combat.adjacent_floating_eye or 
                obs.combat.adjacent_gas_spore or 
                "mold" in obs.combat.closest_hostile_name.lower()):
                if obs.inventory.has_offensive_wand:
                    obs = yield zap_offensive_wand()
                    continue
                elif obs.inventory.has_daggers:
                    obs = yield throw_dagger()
                    continue
                else:
                    # If we have no ranged, we MUST move away. 
                    # If we can't move, we wait and hope for the best, but NEVER melee.
                    obs = yield step_away_from_hostile()
                    continue

            # 3. Non-Aggression
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain") or obs.dungeon.in_shop:
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue

            # 4. Panic Sanctuary (Engrave Elbereth)
            if (obs.hero.hp_frac < 0.35 or obs.combat.is_surrounded) and not obs.combat.standing_on_elbereth:
                obs = yield engrave_dust_elbereth()
                continue

            # 5. Sanctuary Logic
            if obs.combat.standing_on_elbereth:
                if obs.combat.hostile_ignores_elbereth and obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.60 or not obs.combat.can_retreat:
                        obs = yield melee_attack_hostile()
                        continue
                    else:
                        obs = yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()
                        continue

                if obs.hero.hp_frac < 0.40 and obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue
                elif obs.hero.hp_frac < 0.70 and obs.inventory.has_healing:
                    obs = yield quaff_healing()
                    continue
                elif obs.combat.closest_hostile_dist >= 2:
                    if obs.inventory.has_offensive_wand:
                        obs = yield zap_offensive_wand()
                        continue
                    elif obs.inventory.has_daggers:
                        obs = yield throw_dagger()
                        continue
                elif not obs.combat.adjacent_hostile:
                    obs = yield wait()
                    continue
                else:
                    obs = yield melee_attack_hostile()
                    continue

            # 6. Ranged Harassment
            if obs.combat.closest_hostile_dist >= 2:
                if obs.inventory.has_offensive_wand:
                    obs = yield zap_offensive_wand()
                    continue
                elif obs.inventory.has_daggers:
                    obs = yield throw_dagger()
                    continue

            # 7. Fast Dangerous Attackers
            if obs.combat.is_fast_dangerous:
                if not obs.combat.in_corridor and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                    self.retreat_streak += 1
                    continue

            # 8. Tactical Melee / Retreat
            if obs.combat.adjacent_hostile:
                if self.retreat_streak >= 4:
                    if obs.hero.turn - self.last_prayer_turn >= 150:
                        self.last_prayer_turn = obs.hero.turn
                        obs = yield pray()
                    else:
                        obs = yield melee_attack_hostile()
                    self.retreat_streak = 0
                    continue

                if obs.hero.hp_frac > 0.70:
                    if obs.hero.hp_frac > 0.90 or not obs.combat.can_retreat:
                        obs = yield melee_attack_hostile()
                    else:
                        if not obs.combat.in_corridor:
                            obs = yield step_to_chokepoint()
                            self.retreat_streak += 1
                        else:
                            obs = yield melee_attack_hostile()
                else:
                    if obs.combat.can_retreat and not obs.combat.in_corridor:
                        obs = yield step_to_chokepoint()
                        self.retreat_streak += 1
                    elif obs.hero.turn - self.last_prayer_turn >= 150:
                        self.last_prayer_turn = obs.hero.turn
                        obs = yield pray()
                    else:
                        obs = yield melee_attack_hostile()
            else:
                if obs.hero.hp_frac > 0.70:
                    obs = yield melee_attack_hostile()
                else:
                    if not obs.combat.in_corridor:
                        obs = yield step_to_chokepoint()
                        self.retreat_streak += 1
                    else:
                        obs = yield step_away_from_hostile()
                        self.retreat_streak += 1
            
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
        if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            obs = yield from self.handle_combat(obs)
            return obs

        current_pos = (obs.hero.y, obs.hero.x)
        if obs.spatial.standing_on_dead_end and current_pos != getattr(self, "last_searched_pos", None):
            self.last_searched_pos = current_pos
            for _ in range(5):
                if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
                    obs = yield from self.handle_combat(obs)
                    return obs
                if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                    break
                obs = yield search()
        else:
            obs = yield step_to_dead_end()
        return obs
