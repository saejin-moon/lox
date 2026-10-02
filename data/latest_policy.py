class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.last_pos = (None, None)
        self.stuck_turns = 0

    def run(self, obs):
        # --- Robust Stall Detection ---
        current_pos = (obs.hero.y, obs.hero.x)
        if current_pos == self.last_pos:
            self.stuck_turns += 1
        else:
            self.last_pos = current_pos
            self.stuck_turns = 0

        while True:
            # 0. Absolute Deadlock Breaker (Mitigate MaxTurnsReached)
            # If we haven't moved in 5 turns, we are likely attempting an invalid action
            if self.stuck_turns >= 5:
                if obs.spatial.stairs_down_known:
                    obs = yield step_to_stairs_down()
                elif obs.spatial.has_unvisited_frontier:
                    obs = yield step_to_frontier()
                elif obs.spatial.has_unsearched_dead_end:
                    obs = yield step_to_dead_end()
                else:
                    obs = yield search()
                self.stuck_turns = 0
                continue

            # 1. Absolute Emergency Survival
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
            if obs.inventory.has_unworn_armor:
                obs = yield wear_armor()
                continue

            # 5. Weapon Scaling
            if obs.dungeon.adjacent_fountain and obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
                obs = yield dip_excalibur()
                continue

            # 6. Navigation & Exploration
            # Priority 1: Descend immediately
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
                continue
            
            # Priority 2: Move to known stairs (Absolute priority to prevent MaxTurns)
            if obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
                continue

            # Priority 3: Breach Doors (Only if not in shop)
            if obs.dungeon.adjacent_closed_door:
                if obs.dungeon.in_shop:
                    # Avoid provoking shopkeepers; move around the door
                    obs = yield step_to_frontier()
                    continue
                if obs.dungeon.door_is_locked:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
                continue

            # Priority 4: Explore Frontiers
            if obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
                continue

            # Priority 5: Exhaustive Search for Secret Doors/Stairs
            # If no frontiers and no stairs, we MUST search dead ends.
            if obs.spatial.has_unsearched_dead_end or not obs.spatial.has_unvisited_frontier:
                obs = yield from self.handle_dead_end(obs)
                continue
            
            # Final fallback to prevent idle
            obs = yield step_to_frontier()

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0:
            # Floating Eye: Never melee
            if obs.combat.closest_hostile_name == "floating eye":
                if obs.combat.adjacent_hostile and not obs.combat.standing_on_elbereth:
                    obs = yield engrave_dust_elbereth()
                    continue
                if obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                    obs = yield throw_dagger()
                else:
                    obs = yield step_away_from_hostile()
                continue

            # Non-Aggression: Shopkeepers/Town Watch
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain") or obs.dungeon.in_shop:
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                if obs.combat.closest_hostile_dist > 3:
                    break
                continue

            # Tactical Healing
            if obs.hero.hp_frac < 0.45 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # Panic Sanctuary
            if (obs.hero.hp_frac < 0.35 or obs.combat.is_surrounded) and not obs.combat.standing_on_elbereth:
                obs = yield engrave_dust_elbereth()
                continue

            # Elbereth Logic
            if obs.combat.standing_on_elbereth:
                if obs.hero.hp_frac < 0.30 and obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue
                elif obs.hero.hp_frac < 0.60 and obs.inventory.has_healing:
                    obs = yield quaff_healing()
                    continue
                elif not obs.combat.adjacent_hostile:
                    if obs.spatial.stairs_down_known:
                        obs = yield step_to_stairs_down()
                    elif obs.spatial.has_unvisited_frontier:
                        obs = yield step_to_frontier()
                    else:
                        obs = yield step_away_from_hostile()
                    continue
                else:
                    obs = yield melee_attack_hostile()
                    continue

            # Ranged Harassment
            if obs.combat.closest_hostile_dist >= 2 and obs.inventory.has_daggers:
                obs = yield throw_dagger()
                continue

            # High-Speed Attackers
            if obs.combat.is_fast_dangerous:
                if not obs.combat.in_corridor and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                    continue

            # Combat Engagement
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.60 or not obs.combat.can_retreat:
                    obs = yield melee_attack_hostile()
                else:
                    obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
            else:
                if obs.hero.hp_frac > 0.50:
                    obs = yield melee_attack_hostile()
                else:
                    obs = yield step_away_from_hostile()
            
            if obs.combat.hostile_count_fov == 0:
                break
        return obs

    def handle_corpse_consumption(self, obs):
        if obs.combat.hostile_count_fov > 0:
            obs = yield step_away_from_hostile()
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
        # Navigate to a dead end or perimeter wall
        obs = yield step_to_dead_end()
        
        # Search repeatedly at the dead end to find secret doors
        for _ in range(8):
            if obs.combat.hostile_count_fov > 0:
                return obs
            if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                break
            obs = yield search()
        
        # After searching, if we still haven't found stairs, move to the next target
        if not obs.spatial.stairs_down_known:
            if obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            else:
                obs = yield step_to_dead_end()
        return obs
