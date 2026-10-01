class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0

    def run(self, obs):
        while True:
            # 1. Emergency Survival (Highest Priority)
            # Major Trouble: HP < 15% allows prayer even with shorter timeouts
            if obs.hero.hp_frac < 0.15 and (obs.hero.turn - self.last_prayer_turn >= 150):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue

            # 2. Tactical Combat (Immediate Reaction)
            if obs.combat.hostile_count_fov > 0:
                obs = yield from self.handle_combat(obs)
                continue

            # 3. Post-Combat Recovery & Maintenance
            # Only heal/eat when absolutely safe (no enemies in FOV)
            if obs.hero.hp_frac < 0.60 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue
                
            if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
                obs = yield eat_carried_food()
                continue

            # 4. Navigation & Exploration (Prioritize Descent)
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
            elif obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
            elif obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield from self.handle_dead_end(obs)
            else:
                # Final attempt to find stairs via searching
                obs = yield search()
                if obs.spatial.stairs_down_known:
                    obs = yield step_to_stairs_down()
                else:
                    obs = yield wait()

    def handle_combat(self, obs):
        """Manages combat with a focus on survival, gaze mitigation, and chokepoints."""
        while obs.combat.hostile_count_fov > 0:
            # Critical HP: Prioritize distance and prayer
            if obs.hero.hp_frac < 0.30:
                if obs.combat.adjacent_hostile:
                    obs = yield step_away_from_hostile()
                elif obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                else:
                    if (obs.hero.turn - self.last_prayer_turn >= 150):
                        self.last_prayer_turn = obs.hero.turn
                        obs = yield pray()
                    else:
                        obs = yield melee_attack_hostile()
                continue

            if obs.combat.adjacent_hostile:
                # Floating Eye Gaze Mitigation: Never melee attack them
                if obs.combat.closest_hostile_name == "floating eye":
                    obs = yield step_away_from_hostile()
                # Tactical retreat if moderately wounded or outnumbered
                elif (obs.hero.hp_frac < 0.60 or obs.combat.hostile_count_fov > 2) and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield melee_attack_hostile()
            else:
                # Hostiles in FOV but not adjacent
                if obs.combat.in_corridor:
                    obs = yield melee_attack_hostile()
                elif obs.combat.hostile_count_fov > 1:
                    # Avoid being surrounded in rooms
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield melee_attack_hostile()
            
            if obs.combat.hostile_count_fov == 0:
                break
        return obs

    def handle_dead_end(self, obs):
        """Systematically searches dead ends for secret doors to find stairs."""
        obs = yield step_to_dead_end()
        
        # Search multiple times to ensure secret doors are revealed
        # We check for hostiles every single turn to avoid the 'search-death' loop
        for _ in range(15):
            if obs.combat.hostile_count_fov > 0:
                return obs
                
            obs = yield search()
            
            # If searching reveals the stairs or a new path, exit immediately
            if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                break
        return obs
