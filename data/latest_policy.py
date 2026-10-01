class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.retreat_streak = 0
        self.search_count = 0

    def run(self, obs):
        while True:
            # 1. Absolute Emergency Survival (Major Trouble)
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state == 4):
                if obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue

            # 2. Immediate Depth Progression (Anti-MaxTurnsReached)
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
                continue
            
            if obs.spatial.stairs_down_known and not obs.combat.adjacent_hostile:
                obs = yield step_to_stairs_down()
                continue

            # 3. Combat Logic (Highest Priority)
            if obs.combat.hostile_count_fov > 0:
                obs = yield from self.handle_combat(obs)
                continue

            # 4. Hunger Prevention (Safe zones only)
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
            if obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
                continue

            if obs.spatial.has_unvisited_frontier:
                self.retreat_streak = 0
                obs = yield step_to_frontier()
                continue
            
            if obs.spatial.has_unsearched_dead_end:
                obs = yield from self.handle_dead_end(obs)
                continue
            
            # Final fallback: Search perimeter/dead-ends to find secret doors to stairs
            # This replaces any idle/wait behavior to prevent MaxTurnsReached
            obs = yield from self.handle_dead_end(obs)

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0:
            # Floating Eye Gaze Mitigation
            if obs.combat.closest_hostile_name == "floating eye":
                obs = yield step_away_from_hostile()
                continue

            # Shopkeeper/Town Protection
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain") or obs.dungeon.in_shop:
                if obs.combat.closest_hostile_dist > 3:
                    return obs
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue

            # Tactical In-Combat Emergency Healing
            if obs.hero.hp_frac < 0.40 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # Fast & Dangerous Monster Handling
            if obs.combat.is_fast_dangerous:
                if not obs.combat.in_corridor and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                    continue

            # Combat Engagement: Prevent infinite kiting loops
            if obs.combat.adjacent_hostile:
                # Force attack if HP is okay, or if we've retreated too much, or if we can't retreat
                if obs.hero.hp_frac > 0.50 or self.retreat_streak > 3 or not obs.combat.can_retreat:
                    self.retreat_streak = 0
                    obs = yield melee_attack_hostile()
                else:
                    self.retreat_streak += 1
                    obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
            else:
                # Move toward target to engage. If HP is very low, try to maintain distance.
                if obs.hero.hp_frac > 0.30:
                    self.retreat_streak = 0
                    obs = yield melee_attack_hostile()
                else:
                    self.retreat_streak += 1
                    obs = yield step_away_from_hostile()
            
            if obs.combat.hostile_count_fov == 0:
                break
        return obs

    def handle_corpse_consumption(self, obs):
        # Never eat if enemies are present
        if obs.combat.hostile_count_fov > 0:
            # Instead of wait(), we move toward the nearest frontier to keep the turn count moving
            obs = yield step_to_frontier()
            return obs
        for corpse in obs.corpses:
            if corpse.is_safe:
                obs = yield step_to(corpse.y, corpse.x)
                if obs.combat.hostile_count_fov == 0:
                    obs = yield eat_floor_corpse()
                return obs
        # Fallback to movement if no safe corpses
        obs = yield step_to_frontier()
        return obs

    def handle_dead_end(self, obs):
        obs = yield step_to_dead_end()
        # Search up to 8 times to ensure secret doors are found
        for _ in range(8):
            if obs.combat.hostile_count_fov > 0:
                return obs
            if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                break
            obs = yield search()
        return obs
