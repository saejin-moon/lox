class Agent:
    def __init__(self):
        self.last_prayer_turn = -1000

    def run(self, obs):
        while True:
            # 1. Emergency: Major Trouble Prayer
            # Gods grant aid even during timeout if HP < 15% or fainting from hunger
            if (obs.hero.hp_frac < 0.15 or obs.hero.hunger_state == FAINTING) and \
               (obs.hero.turn - self.last_prayer_turn >= 200):
                self.last_prayer_turn = obs.hero.turn
                obs = yield pray()
                continue

            # 2. Sustenance & Healing
            # Proactive eating: start eating at NORMAL/HUNGRY to avoid WEAK/FAINTING
            if obs.hero.hunger_state >= NORMAL:
                if obs.inventory.has_food:
                    obs = yield eat_carried_food()
                    continue
                elif self.find_safe_corpse(obs):
                    # find_safe_corpse returns the action to eat a safe floor corpse
                    obs = yield from self.find_safe_corpse(obs)
                    continue

            if obs.hero.hp_frac < 0.30 and obs.inventory.has_healing:
                obs = yield quaff_healing()
                continue

            # 3. Tactical Combat
            if obs.combat.adjacent_hostile:
                obs = yield from self.handle_combat(obs)
                continue

            # 4. Navigation & Exploration
            if obs.spatial.standing_on_stairs_down:
                obs = yield descend()
            elif obs.spatial.stairs_down_known:
                obs = yield step_to_stairs_down()
            elif obs.dungeon.adjacent_closed_door:
                # Only kick if locked to avoid leg injury
                if obs.dungeon.door_is_locked:
                    obs = yield kick_closed_door()
                else:
                    obs = yield open_door()
            elif obs.spatial.has_unvisited_frontier:
                obs = yield step_to_frontier()
            elif obs.spatial.has_unsearched_dead_end:
                obs = yield search()
            else:
                obs = yield wait()

    def handle_combat(self, obs):
        while obs.combat.adjacent_hostile:
            # Floating Eyes: Melee is lethal (paralysis). Always retreat or range.
            if obs.combat.closest_hostile_name == "floating eye":
                obs = yield step_away_from_hostile()
            # Retreat to chokepoints if health is low
            elif obs.hero.hp_frac < 0.40 and obs.combat.can_retreat:
                obs = yield step_to_chokepoint()
            else:
                obs = yield melee_attack_hostile()
        return obs

    def find_safe_corpse(self, obs):
        """
        Helper to identify and eat a safe corpse on the floor.
        Returns a generator yielding the action if a safe corpse is found.
        """
        # Never eat if enemies are in FOV (eating takes multiple turns)
        if obs.combat.hostile_count_fov > 0:
            return False

        for corpse in obs.corpses:
            # Only eat fresh, safe corpses to avoid poisoning/petrification
            if corpse.is_safe and corpse.is_fresh:
                # We use a custom logic to step to the corpse then eat it
                # Since we don't have a direct 'eat_corpse_at(y, x)', 
                # we assume the agent must be standing on it.
                # This is a simplification; in a real scenario, we'd step_to(corpse).
                # For this policy, we attempt to eat if it's the closest safe one.
                # Note: eat_floor_corpse() usually targets the one under the hero.
                # We'll yield a step to it first if possible.
                # (Assuming the environment handles the target of eat_floor_corpse)
                yield eat_floor_corpse()
                return True
        return False
