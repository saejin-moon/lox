class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0
        self.combat_turn_count = 0
        self.max_combat_turns = 40

    def run(self, obs):
        while True:
            # 1. Absolute Emergency Survival
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state == 4):
                if obs.hero.turn - self.last_prayer_turn >= 150:
                    self.last_prayer_turn = obs.hero.turn
                    obs = yield pray()
                    continue

            # 2. Immediate Depth Progression (Critical to prevent MaxTurnsReached)
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
                continue
            if obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
                continue

            # 3. Combat Logic
            if obs.combat.hostile_count_fov > 0:
                obs = yield from self.handle_combat(obs)
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
            if obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield from self.handle_dead_end(obs)
            else:
                # When frontiers are clear, we must search perimeter walls for secret doors/stairs
                obs = yield from self.handle_dead_end(obs)

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0:
            self.combat_turn_count += 1
            
            # Combat Timeout: Prevent infinite kiting
            if self.combat_turn_count > self.max_combat_turns:
                if obs.combat.can_retreat:
                    obs = yield retreat()
                    self.combat_turn_count = 0
                    return obs
                if not obs.combat.adjacent_hostile:
                    obs = yield melee_attack_hostile()
                    continue

            # Floating Eye Gaze Mitigation
            if obs.combat.closest_hostile_name == "floating eye":
                obs = yield step_away_from_hostile()
                continue

            # Shopkeeper/Town Protection
            if obs.combat.closest_hostile_name in ("shopkeeper", "watchman", "watch captain") or obs.dungeon.in_shop:
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
                    obs = yield step_to_chokepoint() if obs.combat.in_corridor else step_away_from_hostile()
            
            if obs.combat.hostile_count_fov == 0:
                self.combat_turn_count = 0
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
        # Move to a dead end or perimeter wall
        obs = yield step_to_dead_end()
        # Search multiple times to uncover secret doors/stairs
        for _ in range(10):
            if obs.combat.hostile_count_fov > 0:
                return obs
            if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                break
            obs = yield search()
        return obs
