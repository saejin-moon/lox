class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0

    def run(self, obs):
        while True:
            # 1. Absolute Emergency Survival (Major Trouble)
            # HP < 15% or Fainting from hunger allows prayer with shorter timeouts
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state == 4) and (obs.hero.turn - self.last_prayer_turn >= 150):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue

            # 2. Immediate Combat Reaction
            if obs.combat.hostile_count_fov > 0:
                obs = yield from self.handle_combat(obs)
                continue

            # 3. Critical Maintenance (Hunger & Health)
            # Prevent fainting: Eat if HUNGRY (2) or WEAK (3) and safe
            if obs.hero.hunger_state >= 2:
                if obs.inventory.has_food:
                    obs = yield eat_carried_food()
                    continue
                elif obs.corpses and any(c.is_safe for c in obs.corpses):
                    # Attempt to find a safe corpse to eat
                    obs = yield from self.handle_corpse_consumption(obs)
                    continue

            if obs.hero.hp_frac < 0.60 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 4. Navigation & Exploration
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
                # Search for secret doors to find stairs
                if self.search_count < 20:
                    obs = yield search()
                    self.search_count += 1
                else:
                    obs = yield wait()
                    self.search_count = 0
                
                if obs.spatial.stairs_down_known:
                    obs = yield step_to_stairs_down()

    def handle_combat(self, obs):
        """Manages combat with gaze mitigation and chokepointing."""
        while obs.combat.hostile_count_fov > 0:
            # Emergency retreat/prayer if HP is critical
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
                # Floating Eye Gaze Mitigation
                if obs.combat.closest_hostile_name == "floating eye":
                    obs = yield step_away_from_hostile()
                elif (obs.hero.hp_frac < 0.60 or obs.combat.hostile_count_fov > 2) and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield melee_attack_hostile()
            else:
                # Close gap in corridors, otherwise move cautiously
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
        """Safely consumes corpses to prevent hunger-fainting."""
        # Only eat if no enemies are in FOV (eating takes multiple turns)
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
        """Systematically searches dead ends for secret doors."""
        obs = yield step_to_dead_end()
        for _ in range(15):
            if obs.combat.hostile_count_fov > 0:
                return obs
            obs = yield search()
            if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                break
        return obs
