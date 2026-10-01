class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000

    def run(self, obs):
        while True:
            # 1. Critical Survival (HP/Hunger/Prayer)
            # Replace IfExp with standard if/else for AST compliance
            prayer_cooldown = 350
            if obs.hero.hp_frac < 0.15:
                prayer_cooldown = 200

            if obs.hero.hp_frac < 0.20 and (obs.hero.turn - self.last_prayer_turn >= prayer_cooldown):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue

            if obs.hero.hp_frac < 0.30 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            if obs.hero.hunger_state >= 3 and obs.inventory.has_food:
                obs = yield eat_carried_food()
                continue

            # 2. Immediate Combat Response
            if obs.combat.adjacent_hostile:
                obs = yield from self.handle_combat(obs)
                continue
            
            if obs.combat.hostile_count_fov > 0:
                if obs.hero.hp_frac < 0.40:
                    obs = yield step_away_from_hostile()
                else:
                    obs = yield melee_attack_hostile()
                continue

            # 3. Navigation & Progression
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
                continue

            if obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
                continue

            if obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
                continue

            # Exploration
            if obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield search()
            else:
                obs = yield wait()

    def handle_combat(self, obs):
        while obs.combat.adjacent_hostile:
            if obs.combat.closest_hostile_name == "floating eye":
                obs = yield step_away_from_hostile()
            elif obs.hero.hp_frac < 0.35 and obs.combat.can_retreat:
                obs = yield step_to_chokepoint()
            else:
                obs = yield melee_attack_hostile()
        return obs
