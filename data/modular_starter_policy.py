class Agent:

    def __init__(self):
        self.last_prayer_turn = -1000
        self.current_goal = "explore_and_dive"

    def determine_goal(self, obs) -> str:
        if obs.combat.adjacent_hostile or obs.combat.has_active_hostile:
            return "combat"
        if obs.hero.hunger_state >= 2 or (obs.hero.hunger_state >= 1 and any(c.is_safe for c in obs.corpses)):
            return "nutrition"
        if obs.dungeon.can_forge_excalibur and (obs.dungeon.standing_on_fountain or obs.dungeon.fountain_in_fov or obs.dungeon.has_known_fountain):
            return "forge_excalibur"
        if obs.hero.ac >= 5 or obs.inventory.has_unworn_armor or obs.inventory.get_superior_body_armor_slot() is not None or obs.spatial.has_nearby_loot:
            return "scavenge_armor"
        return "explore_and_dive"

    def run(self, obs):
        while True:
            # Reflex 1: Armor Upgrades
            if obs.inventory.get_superior_body_armor_slot() is not None:
                obs = (yield replace_body_armor())
                continue
            if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
                obs = (yield wear_armor())
                continue

            # Reflex 2: Nutrition (Corpse at >=1, Carried food ONLY at >=2)
            if obs.hero.hunger_state >= 1 and any(c.is_safe for c in obs.corpses) and not obs.combat.adjacent_hostile:
                obs = (yield eat_floor_corpse())
                continue
            if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
                obs = (yield eat_carried_food())
                continue

            # Reflex 3: Major Trouble Prayer
            if (obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 4 and not obs.inventory.has_food)) and (obs.hero.can_pray and obs.hero.turn - self.last_prayer_turn >= 850):
                self.last_prayer_turn = obs.hero.turn
                obs = (yield pray())
                continue

            # Reflex 4: Combat Engagement
            if obs.combat.adjacent_hostile or obs.combat.has_active_hostile:
                obs = (yield from self.skill_combat(obs))
                continue

            # Reflex 5: Paced Vertical Transit
            should_descend = (
                obs.spatial.stairs_down_known and (
                    obs.hero.turns_on_level >= 100
                    or not obs.spatial.has_unvisited_frontier
                    or obs.hero.hunger_state >= 2
                    or obs.hero.depth >= 3
                    or obs.combat.adjacent_hostile
                    or obs.hero.hp_frac < 0.35
                )
            )
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                if should_descend:
                    obs = (yield descend())
                    continue
            if should_descend:
                obs = (yield step_to_stairs_down())
                continue

            # Strategic Goal Dispatch
            goal = self.determine_goal(obs)
            self.current_goal = goal
            if goal == "combat":
                obs = (yield from self.skill_combat(obs))
                continue
            elif goal == "nutrition":
                obs = (yield from self.skill_handle_nutrition(obs))
                continue
            elif goal == "forge_excalibur":
                obs = (yield from self.skill_forge_excalibur(obs))
                continue
            elif goal == "scavenge_armor":
                obs = (yield from self.skill_scavenge_armor(obs))
                continue
            elif goal == "explore_and_dive":
                obs = (yield from self.skill_explore_and_dive(obs))
                continue
            else:
                obs = (yield from self.skill_explore_and_dive(obs))
                continue

    def skill_combat(self, obs):
        # Emergency healing during combat
        if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing and obs.epistemic.can_safely_quaff_healing:
            obs = (yield quaff_healing())
            return obs

        if obs.combat.adjacent_hostile:
            # Passive hazard discrimination
            if obs.combat.adjacent_floating_eye or obs.combat.adjacent_gas_spore:
                if obs.combat.has_safe_melee_target:
                    obs = (yield melee_attack_hostile())
                    return obs
                if obs.inventory.has_daggers:
                    obs = (yield throw_dagger())
                    return obs
                if obs.inventory.has_offensive_wand:
                    obs = (yield zap_offensive_wand())
                    return obs
                obs = (yield step_away_from_hostile())
                return obs

            # Standard melee vs sanctuary
            if obs.hero.hp_frac > 0.35:
                obs = (yield melee_attack_hostile())
                return obs
            else:
                if not obs.combat.standing_on_elbereth and not obs.combat.hostile_ignores_elbereth:
                    obs = (yield engrave_dust_elbereth())
                    return obs
                obs = (yield melee_attack_hostile())
                return obs

        if obs.combat.has_active_hostile:
            if obs.combat.closest_hostile_dist >= 2:
                if obs.inventory.has_offensive_wand:
                    obs = (yield zap_offensive_wand())
                    return obs
                if obs.inventory.has_daggers:
                    obs = (yield throw_dagger())
                    return obs
            obs = (yield melee_attack_hostile())
            return obs

        obs = (yield wait())
        return obs

    def skill_forge_excalibur(self, obs):
        if obs.dungeon.standing_on_fountain:
            obs = (yield dip_excalibur())
            return obs
        if obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
            obs = (yield step_to_fountain())
            return obs
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_scavenge_armor(self, obs):
        if obs.inventory.get_superior_body_armor_slot() is not None:
            obs = (yield replace_body_armor())
            return obs
        if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
            obs = (yield wear_armor())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield step_to_dead_end())
        return obs

    def skill_explore_and_dive(self, obs):
        should_descend = (
            obs.spatial.stairs_down_known and (
                obs.hero.turns_on_level >= 100
                or not obs.spatial.has_unvisited_frontier
                or obs.hero.hunger_state >= 2
                or obs.hero.depth >= 3
            )
        )
        if should_descend:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield from self.handle_dead_end(obs))
        return obs

    def handle_dead_end(self, obs):
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.spatial.standing_on_dead_end:
            obs = (yield search())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield step_to_dead_end())
        return obs

    def skill_handle_nutrition(self, obs):
        if obs.combat.hostile_count_fov > 0:
            obs = (yield wait())
            return obs
        for corpse in obs.corpses:
            if corpse.is_safe:
                if (obs.hero.y, obs.hero.x) == (corpse.y, corpse.x):
                    obs = (yield eat_floor_corpse())
                    return obs
                else:
                    obs = (yield step_to(corpse.y, corpse.x))
                    return obs
        if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
            obs = (yield eat_carried_food())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
        else:
            obs = (yield step_to_dead_end())
        return obs
