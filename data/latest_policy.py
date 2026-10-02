class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0
        self.retreat_streak = 0

    def run(self, obs):
        while True:
            # 1. Absolute Emergency Survival (Major Trouble: Fainting or <15% HP)
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state == 4):
                if obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue

            # 2. Combat Logic (Highest Priority)
            if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
                obs = yield from self.handle_combat(obs)
                continue

            # 3. Hunger Prevention (Strictly only when no hostiles are in FOV)
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

            # 5. Weapon Scaling (Excalibur)
            if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
                if obs.dungeon.adjacent_fountain:
                    obs = yield dip_excalibur()
                    continue
                elif obs.dungeon.fountain_in_fov:
                    obs = yield step_to_fountain()
                    continue

            # 6. Navigation & Exploration
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
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
                # Prevent MaxTurnsReached by actively searching for secret doors/stairs
                obs = yield from self.handle_dead_end(obs)

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            # Floating Eye: Absolute melee prohibition
            if obs.combat.closest_hostile_name == "floating eye":
                if obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                    obs = yield throw_dagger()
                else:
                    obs = yield step_away_from_hostile()
                continue

            # Non-Aggression (Shopkeepers/Town)
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain") or obs.dungeon.in_shop:
                obs = yield retreat() if obs.combat.can_retreat else step_away_from_hostile()
                continue

            # Tactical Healing: Immediate priority when damaged
            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # Panic Sanctuary: Engrave Elbereth if low HP or surrounded
            if (obs.hero.hp_frac < 0.35 or obs.combat.is_surrounded) and not obs.combat.standing_on_elbereth:
                obs = yield engrave_dust_elbereth()
                continue

            # While standing on Elbereth: Heal, pray, or actively seek exit
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
                    if obs.hero.hp_frac > 0.70:
                        obs = yield melee_attack_hostile()
                    else:
                        obs = yield wait()
                    continue

            # Ranged Harassment: Kill fast pests before they close in
            if obs.combat.closest_hostile_dist >= 2 and obs.inventory.has_daggers:
                obs = yield throw_dagger()
                continue

            # High-Speed Attackers (Ants/Bees)
            if obs.combat.is_fast_dangerous:
                if not obs.combat.in_corridor and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                    continue

            # Melee Engagement Logic
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.60 or not obs.combat.can_retreat:
                    obs = yield melee_attack_hostile()
                else:
                    obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
            else:
                if obs.hero.hp_frac > 0.50:
                    obs = yield melee_attack_hostile()
                else:
                    obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
            
            if obs.combat.hostile_count_fov == 0 and not obs.combat.adjacent_hostile:
                break
        return obs

    def handle_corpse_consumption(self, obs):
        if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
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
        obs = yield step_to_dead_end()
        for _ in range(15):
            if obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
                return obs
            if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                break
            obs = yield search()
        return obs
