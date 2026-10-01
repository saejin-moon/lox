class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0

    def run(self, obs):
        while True:
            # 1. Absolute Emergency Survival (Major Trouble)
            # Prayer is a lifeline for HP < 15%, Fainting, or Blindness in combat.
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state == 4 or (obs.status.is_blind and obs.combat.hostile_count_fov > 0)) and (obs.hero.turn - self.last_prayer_turn >= 150):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue

            # 2. Immediate Combat Reaction
            if obs.combat.hostile_count_fov > 0:
                obs = yield from self.handle_combat(obs)
                continue

            # 3. Critical Maintenance (Hunger & Health)
            # Only perform multi-turn actions (eating/healing) when no enemies are in FOV
            if obs.hero.hunger_state >= 2:
                if obs.inventory.has_food:
                    obs = yield eat_carried_food()
                    continue
                elif obs.corpses:
                    obs = yield from self.handle_corpse_consumption(obs)
                    continue

            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 4. Navigation & Exploration
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
                # Exhaustive search for secret doors/stairs to prevent MaxTurnsReached
                if self.search_count < 20:
                    obs = yield search()
                    self.search_count += 1
                else:
                    # If we've searched everything, try to find a new frontier or wait
                    if obs.spatial.has_unvisited_frontier:
                        obs = yield step_to_frontier()
                    else:
                        obs = yield wait()
                    self.search_count = 0
                
                if obs.spatial.stairs_down_known:
                    obs = yield step_to_stairs_down()

    def handle_combat(self, obs):
        """Tactical combat handler with strict retreat and gaze mitigation."""
        while obs.combat.hostile_count_fov > 0:
            # Blindness/Panic State: Retreat immediately.
            if obs.status.is_blind or obs.hero.hp_frac < 0.30:
                if obs.combat.adjacent_hostile:
                    obs = yield step_away_from_hostile()
                elif obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                else:
                    # Trapped and desperate
                    if (obs.hero.turn - self.last_prayer_turn >= 150):
                        self.last_prayer_turn = obs.hero.turn
                        obs = yield pray()
                    else:
                        obs = yield melee_attack_hostile()
                continue

            # In-Combat Emergency Healing
            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # Shopkeeper and Shop Protection: Never fight shopkeepers
            if obs.combat.closest_hostile_name == "shopkeeper" or obs.dungeon.in_shop:
                if obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield step_away_from_hostile()
                continue

            # Gaze Mitigation: Floating Eyes are never attacked in melee
            if obs.combat.closest_hostile_name == "floating eye":
                obs = yield step_away_from_hostile()
                continue

            # Standard Combat Logic
            if obs.combat.adjacent_hostile:
                # Retreat if outnumbered or wounded to a chokepoint to force 1v1s
                if (obs.hero.hp_frac < 0.60 or obs.combat.hostile_count_fov > 1) and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield melee_attack_hostile()
            else:
                # Close gap if safe, otherwise maintain tactical position
                if obs.combat.in_corridor:
                    obs = yield melee_attack_hostile()
                elif obs.combat.hostile_count_fov > 1:
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield melee_attack_hostile()
            
            if obs.combat.hostile_count_fov == 0:
                break
        return obs

    def handle_corpse_consumption(self, obs):
        """Safely consumes corpses only when the area is clear."""
        if obs.combat.hostile_count_fov > 0:
            obs = yield wait()
            return obs

        for corpse in obs.corpses:
            if corpse.is_safe:
                obs = yield step_to(corpse.y, corpse.x)
                # Re-verify safety before the multi-turn eating action
                if obs.combat.hostile_count_fov == 0:
                    obs = yield eat_floor_corpse()
                return obs
        
        obs = yield wait()
        return obs

    def handle_dead_end(self, obs):
        """Systematically searches dead ends for secret doors to find stairs."""
        obs = yield step_to_dead_end()
        # Search up to 15 times at the dead end to find hidden doors
        for _ in range(15):
            if obs.combat.hostile_count_fov > 0:
                return obs
            if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                break
            obs = yield search()
        return obs
