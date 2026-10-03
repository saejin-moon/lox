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
                obs = yield from self.handle_dead_end(obs)
                continue

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0:
            # 1. Non-Aggression
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain") or obs.dungeon.in_shop:
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue

            # 2. Critical Healing (Highest Combat Priority)
            # If HP is low, quaff healing immediately regardless of position
            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 3. Gaze/Touch Hazard Avoidance (Floating Eyes, Gas Spores)
            # Never melee these. Use daggers or distance.
            if obs.combat.closest_hostile_name in ("floating eye", "gas spore"):
                if obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                    obs = yield throw_dagger()
                    continue
                # If adjacent or no daggers, get away immediately
                obs = yield step_away_from_hostile()
                continue

            # 4. Panic Sanctuary (Elbereth)
            # Engrave if critically low or surrounded to create a safe zone
            if (obs.hero.hp_frac < 0.35 or obs.combat.is_surrounded) and not obs.combat.standing_on_elbereth:
                obs = yield engrave_dust_elbereth()
                continue

            # 5. Elbereth Recovery Logic
            if obs.combat.standing_on_elbereth:
                # Priority 1: Heal to safety
                if obs.hero.hp_frac < 0.70 and obs.inventory.has_healing:
                    obs = yield quaff_healing()
                    continue
                # Priority 2: Divine intervention
                if obs.hero.hp_frac < 0.40 and obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue
                # Priority 3: Disengage if safe, otherwise fight only if healthy
                if not obs.combat.adjacent_hostile:
                    if obs.spatial.stairs_down_known:
                        obs = yield step_to_stairs_down()
                    elif obs.spatial.has_unvisited_frontier:
                        obs = yield step_to_frontier()
                    else:
                        obs = yield step_away_from_hostile()
                    continue
                elif obs.combat.hostile_ignores_elbereth or obs.hero.hp_frac > 0.80:
                    obs = yield melee_attack_hostile()
                    continue
                else:
                    # Wait for regeneration or for monster to leave, but only if not dying
                    obs = yield wait()
                    continue

            # 6. Ranged Harassment
            if obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                obs = yield throw_dagger()
                continue

            # 7. High-Speed Attackers (Soldier Ants, etc.)
            if obs.combat.is_fast_dangerous:
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
            obs = yield step_to_dead_end()
        return obs
