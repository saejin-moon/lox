class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000
        self.turns_searching = 0

    def run(self, obs):
        while True:
            # 1. Emergency Survival & Self-Monitored Recovery
            if obs.hero.hp_frac < 0.15 and (obs.hero.turn - self.last_prayer_turn >= 350):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue
            elif obs.hero.hp_frac < 0.30 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue
            elif obs.hero.hunger_state >= HUNGRY and obs.inventory.has_food:
                obs = yield eat_carried_food()
                continue

            # 2. Tactical Hostile Combat & Retreat
            if obs.combat.adjacent_hostile:
                if obs.combat.closest_hostile_name == "floating eye":
                    obs = yield step_away_from_hostile()
                elif obs.hero.hp_frac < 0.35 and obs.combat.can_retreat:
                    obs = yield step_to_chokepoint()
                else:
                    obs = yield melee_attack_hostile()
                continue

            # 3. Environment & Maintenance
            if obs.dungeon.adjacent_closed_door:
                obs = yield open_door()
                continue
            elif obs.spatial.standing_on_stairs_down:
                obs = yield descend()
                continue

            # 4. Spatial Navigation & Exploration
            if obs.spatial.stairs_down_known and not obs.spatial.has_unvisited_frontier:
                obs = yield step_to_stairs_down()
            elif obs.spatial.has_unvisited_frontier:
                self.turns_searching = 0
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                self.turns_searching += 1
                obs = yield search()
            elif obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
            else:
                obs = yield search()
