class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000

    def run(self, obs):
        while True:
            # 1. Emergency Survival (Highest Priority)
            # Prayer: Use shorter timeout for critical HP (Major Trouble)
            prayer_timeout = 150 if obs.hero.hp_frac < 0.15 else 350
            if obs.hero.hp_frac < 0.20 and (obs.hero.turn - self.last_prayer_turn >= prayer_timeout):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue
            
            # Healing: Quaff immediately if wounded and safe
            if obs.hero.hp_frac < 0.40 and obs.inventory.has_healing:
                if not obs.combat.adjacent_hostile:
                    obs = yield quaff_healing()
                    continue

            # Hunger: Prevent fainting. Eat if HUNGRY and no immediate threat.
            # If FAINTING, we must eat regardless of FOV if we have food.
            if obs.hero.hunger_state >= HUNGRY and obs.inventory.has_food:
                if obs.hero.hunger_state == FAINTING or obs.combat.hostile_count_fov == 0:
                    obs = yield eat_carried_food()
                    continue

            # 2. Tactical Combat Management
            if obs.combat.adjacent_hostile:
                obs = yield from self.handle_combat(obs)
                continue
            
            if obs.combat.hostile_count_fov > 0:
                # RISK ASSESSMENT: Avoid high-threats unless we are healthy.
                high_threats = ["valkyrie", "knight", "paladin", "floating eye"]
                is_high_threat = obs.combat.closest_hostile_name in high_threats
                
                if obs.hero.hp_frac < 0.70 or is_high_threat:
                    # Try to maintain distance, but don't get pinned in a corner
                    if obs.combat.can_retreat:
                        obs = yield step_away_from_hostile()
                    else:
                        # If we can't retreat, we must engage to clear the path
                        obs = yield melee_attack_hostile()
                else:
                    obs = yield melee_attack_hostile()
                continue

            # 3. Navigation & Exploration
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
            elif obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
            elif obs.dungeon.adjacent_closed_door:
                if obs.hero.hp_frac > 0.40:
                    if obs.dungeon.door_is_locked:
                        obs = yield kick_closed_door()
                    else:
                        obs = yield open_door()
                else:
                    obs = yield step_to_frontier()
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield search()
            else:
                # Avoid idling if we can move; otherwise wait
                obs = yield wait()

    def handle_combat(self, obs):
        # 1. Floating Eye Gaze Mitigation (Absolute Priority)
        if obs.combat.closest_hostile_name == "floating eye":
            if obs.combat.can_retreat:
                obs = yield step_away_from_hostile()
            else:
                # If pinned by an eye, we have to fight or we die
                obs = yield melee_attack_hostile()
            return obs

        # 2. Tactical Retreat / Chokepointing
        # If HP is low or we are surrounded, prioritize survival.
        if obs.hero.hp_frac < 0.50 or obs.combat.is_surrounded:
            if obs.combat.can_retreat:
                # Try to find a corridor or doorway to limit attackers
                if obs.combat.in_corridor:
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield step_away_from_hostile()
            else:
                # No retreat possible: Fight for your life
                obs = yield melee_attack_hostile()
            return obs

        # 3. Aggressive Engagement
        # If we are healthy and not facing a gaze-monster, attack.
        obs = yield melee_attack_hostile()
        return obs
