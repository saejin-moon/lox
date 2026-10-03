class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.last_depth = 1
        self.last_searched_pos = None

    def run(self, obs):
        while True:
            if obs.hero.depth != self.last_depth:
                self.last_searched_pos = None
                self.last_depth = obs.hero.depth

            # 1. Absolute Emergency Survival (Major Trouble)
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state == 4):
                if obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue

            # 2. Combat Logic (Highest Priority - Strict Gating)
            # We must resolve all hostiles in FOV before attempting any navigation
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

            # 5. Altar BUC Testing
            if obs.epistemic.has_untested_items:
                if obs.dungeon.standing_on_altar:
                    obs = yield test_altar_buc()
                    continue
                elif obs.dungeon.adjacent_altar:
                    obs = yield step_to_altar()
                    continue

            # 6. Weapon Scaling (Excalibur)
            if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
                if obs.dungeon.standing_on_fountain:
                    obs = yield dip_excalibur()
                    continue
                elif obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
                    obs = yield step_to_fountain()
                    continue

            # 7. Poison Resistance Harvesting
            if obs.dungeon.can_harvest_poison and not obs.hero.has_poison_res and obs.hero.hp_frac > 0.90:
                obs = yield harvest_poison_res()
                continue

            # 8. Navigation & Exploration
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
                # Search perimeter walls/dead ends for secret doors when floor is "cleared"
                obs = yield from self.handle_dead_end(obs)
                continue

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0:
            # 1. Non-Aggression (Shopkeepers/Town Watch)
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain") or obs.dungeon.in_shop:
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue

            # 2. Critical Healing (Immediate Priority)
            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 3. Gaze/Touch Hazard Avoidance
            if obs.combat.closest_hostile_name in ("floating eye", "gas spore"):
                if obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                    obs = yield throw_dagger()
                    continue
                obs = yield step_away_from_hostile()
                continue

            # 4. Panic Sanctuary (Elbereth)
            # Lowered threshold to 30% to avoid over-engraving, but strictly enforce it
            if (obs.hero.hp_frac < 0.30 or obs.combat.is_surrounded) and not obs.combat.standing_on_elbereth:
                obs = yield engrave_dust_elbereth()
                continue

            # 5. Elbereth Recovery Logic
            if obs.combat.standing_on_elbereth:
                if obs.hero.hp_frac < 0.70 and obs.inventory.has_healing:
                    obs = yield quaff_healing()
                    continue
                if obs.hero.hp_frac < 0.40 and obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue
                if not obs.combat.adjacent_hostile:
                    # Only leave Elbereth if we are healthy enough to fight or the coast is clear
                    if obs.hero.hp_frac > 0.60:
                        if obs.spatial.stairs_down_known:
                            obs = yield step_to_stairs_down()
                        elif obs.spatial.has_unvisited_frontier:
                            obs = yield step_to_frontier()
                        else:
                            obs = yield step_away_from_hostile()
                        continue
                    else:
                        obs = yield wait()
                        continue
                elif obs.combat.hostile_ignores_elbereth or obs.hero.hp_frac > 0.80:
                    obs = yield melee_attack_hostile()
                    continue
                else:
                    obs = yield wait()
                    continue

            # 6. Ranged Harassment
            if obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                obs = yield throw_dagger()
                continue

            # 7. High-Speed Attackers / Tactical Retreat
            if obs.combat.is_fast_dangerous or (obs.hero.hp_frac < 0.60 and obs.combat.can_retreat):
                if obs.combat.can_retreat:
                    obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
                    continue

            # 8. Melee Engagement
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.60 or not obs.combat.can_retreat:
                    obs = yield melee_attack_hostile()
                else:
                    obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
            else:
                # If not adjacent, close in only if healthy
                if obs.hero.hp_frac > 0.70:
                    obs = yield melee_attack_hostile()
                else:
                    obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()

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
        current_pos = (obs.hero.y, obs.hero.x)
        if obs.spatial.standing_on_dead_end and current_pos != self.last_searched_pos:
            self.last_searched_pos = current_pos
            for _ in range(8):
                if obs.combat.hostile_count_fov > 0 or obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                    break
                obs = yield search()
        else:
            if obs.combat.hostile_count_fov > 0:
                obs = yield step_away_from_hostile()
            else:
                obs = yield step_to_dead_end()
        return obs
