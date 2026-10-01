class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0
        self.combat_turn_start = 0

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

            # 3. Post-Combat Recovery (Heal before exploring further)
            if obs.hero.hp_frac < 0.90 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 4. Hunger Prevention (Only when safe)
            if obs.hero.hunger_state >= 2:
                if obs.inventory.has_food:
                    obs = yield eat_carried_food()
                    continue
                elif obs.corpses:
                    obs = yield from self.handle_corpse_consumption(obs)
                    continue

            # 5. Equipment Optimization
            if obs.inventory.has_unworn_armor:
                obs = yield wear_armor()
                continue

            # 6. Navigation & Exploration
            if obs.spatial.standing_on_stairs_down:
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
                # Prevent MaxTurnsReached: Search perimeter walls/dead ends for secret doors
                obs = yield from self.handle_dead_end(obs)
            
            # Final safety check to ensure we are moving toward the exit
            if obs.spatial.stairs_down_known and not obs.spatial.standing_on_stairs_down:
                obs = yield step_to_stairs_down()

    def handle_combat(self, obs):
        if self.combat_turn_start == 0:
            self.combat_turn_start = obs.hero.turn

        while obs.combat.hostile_count_fov > 0:
            combat_duration = obs.hero.turn - self.combat_turn_start
            
            # Stalemate/Attrition Detection: Retreat if combat drags on too long
            if combat_duration > 40:
                if obs.combat.can_retreat:
                    obs = yield retreat()
                    self.combat_turn_start = 0
                    return obs
                elif obs.hero.turn - self.last_prayer_turn >= 200:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue

            # Floating Eye: Absolute melee prohibition
            if obs.combat.closest_hostile_name == "floating eye":
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue
            
            # Non-Aggression Pact
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain") or obs.dungeon.in_shop:
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue
            
            # Tactical In-Combat Emergency Healing
            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue
            
            # High-Speed Attackers (Bats, Ants, Bees)
            if obs.combat.is_fast_dangerous or "bat" in obs.combat.closest_hostile_name:
                if obs.combat.in_corridor:
                    obs = yield step_to_chokepoint()
                    continue
                elif obs.combat.can_retreat:
                    obs = yield retreat()
                    continue
                else:
                    obs = yield step_away_from_hostile()
                    continue

            # Combat Engagement Logic
            if obs.combat.adjacent_hostile:
                # Retreat if surrounded or critically low HP
                if obs.hero.hp_frac < 0.30 or obs.combat.is_surrounded:
                    if obs.combat.in_corridor:
                        obs = yield step_to_chokepoint()
                    elif obs.combat.can_retreat:
                        obs = yield retreat()
                    else:
                        obs = yield melee_attack_hostile()
                else:
                    obs = yield melee_attack_hostile()
            else:
                # Close gap if healthy, otherwise maintain distance
                if obs.hero.hp_frac > 0.60:
                    obs = yield melee_attack_hostile()
                else:
                    if obs.combat.closest_hostile_dist > 2:
                        obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                    else:
                        obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
            
            if obs.combat.hostile_count_fov == 0:
                self.combat_turn_start = 0
                break
        return obs

    def handle_corpse_consumption(self, obs):
        if obs.combat.hostile_count_fov > 0:
            obs = yield wait()
            return obs
        for corpse in obs.corpses:
            if corpse.is_safe:
                obs = yield step_to(corpse.y, corpse.x)
                if obs.combat.hostile_count_fov == 0:
                    obs = yield eat_floor_corpse()
                return obs
        obs = yield wait()
        return obs

    def handle_dead_end(self, obs):
        obs = yield step_to_dead_end()
        for _ in range(8):
            if obs.combat.hostile_count_fov > 0:
                return obs
            if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                break
            obs = yield search()
            self.search_count += 1
        return obs
