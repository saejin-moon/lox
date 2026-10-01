class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0
        self.retreat_streak = 0
        self.last_pos = (None, None)
        self.stuck_turns = 0

    def run(self, obs):
        while True:
            # 0. Anti-Stall / Stuck Detection
            current_pos = (obs.hero.y, obs.hero.x)
            if current_pos == self.last_pos:
                self.stuck_turns += 1
            else:
                self.stuck_turns = 0
            self.last_pos = current_pos

            if self.stuck_turns > 6:
                self.stuck_turns = 0
                if obs.spatial.has_unvisited_frontier:
                    obs = yield step_to_frontier()
                else:
                    obs = yield step_to_dead_end()
                continue

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

            # 3. Immediate Depth Progression (Highest non-combat priority)
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
                continue
            if obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
                continue

            # 4. Immediate Health Recovery (Only if safe)
            if obs.hero.hp_frac < 0.40 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 5. Hunger Prevention (Only when safe)
            if obs.hero.hunger_state >= 2:
                if obs.inventory.has_food:
                    obs = yield eat_carried_food()
                    continue
                elif obs.corpses:
                    obs = yield from self.handle_corpse_consumption(obs)
                    continue

            # 6. Equipment Optimization (Wear picked-up armor when safe)
            if obs.inventory.has_unworn_armor:
                obs = yield wear_armor()
                continue

            # 7. Navigation & Active Exploration (Zero Idle Waiting)
            if obs.dungeon.adjacent_closed_door:
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

    def handle_combat(self, obs):
        """Tactical combat handler with anti-pinning and gaze mitigation."""
        while obs.combat.hostile_count_fov > 0:
            # Gaze Mitigation: Floating Eyes are NEVER attacked in melee.
            if obs.combat.closest_hostile_name == "floating eye":
                if obs.combat.adjacent_hostile:
                    obs = yield step_away_from_hostile()
                else:
                    obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue

            # Shopkeeper Protection
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain") or obs.dungeon.in_shop:
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue

            # Tactical In-Combat Emergency Healing
            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # Fast & Dangerous Monster Handling
            if obs.combat.is_fast_dangerous:
                if not obs.combat.in_corridor and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                    continue

            # Survival Logic: Prevent the "Retreat Loop"
            if obs.hero.hp_frac < 0.35 or obs.status.is_blind:
                if self.retreat_streak > 5:
                    if obs.hero.turn - self.last_prayer_turn >= 150:
                        self.last_prayer_turn = obs.hero.turn
                        obs = yield pray()
                    else:
                        obs = yield melee_attack_hostile()
                    self.retreat_streak = 0
                    continue

                if obs.combat.adjacent_hostile:
                    obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
                    self.retreat_streak += 1
                elif obs.combat.can_retreat:
                    obs = yield retreat()
                    self.retreat_streak += 1
                else:
                    obs = yield melee_attack_hostile()
                    self.retreat_streak = 0
                continue

            # Standard Combat Logic
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.60 or not obs.combat.can_retreat:
                    obs = yield melee_attack_hostile()
                    self.retreat_streak = 0
                else:
                    obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
                    self.retreat_streak += 1
            else:
                if obs.hero.hp_frac > 0.50:
                    obs = yield melee_attack_hostile()
                    self.retreat_streak = 0
                else:
                    obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
                    self.retreat_streak += 1
            
            if obs.combat.hostile_count_fov == 0:
                self.retreat_streak = 0
                break
        return obs

    def handle_corpse_consumption(self, obs):
        """Safely consumes corpses only when the area is completely clear."""
        if obs.combat.hostile_count_fov > 0:
            obs = yield wait()
            return obs

        for corpse in obs.corpses:
            if corpse.is_safe:
                obs = yield step_to(corpse.y, corpse.x)
                if obs.combat.hostile_count_fov > 0:
                    return obs
                obs = yield eat_floor_corpse()
                return obs
        
        obs = yield wait()
        return obs

    def handle_dead_end(self, obs):
        """Systematically searches dead ends to find secret doors/stairs."""
        obs = yield step_to_dead_end()
        
        for _ in range(5):
            if obs.combat.hostile_count_fov > 0:
                return obs
            if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                break
            obs = yield search()
        return obs
