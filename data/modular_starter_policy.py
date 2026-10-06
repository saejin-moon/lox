class Agent:

    def __init__(self):
        self.last_prayer_turn = -1000
        self.current_goal = "explore"

    def determine_goal(self, obs) -> str:
        """
        Determines the active strategic goal:
        - 'escape_mines': stuck in Gnomish Mines
        - 'emergency_heal': HP < 40% with healing available
        - 'combat': adjacent hostile or active threatening hostiles in FOV
        - 'scavenge_armor': missing body armor on early floors with loot nearby
        - 'forge_excalibur': Valkyrie XL >= 5 with long sword and fountain accessible
        - 'explore_and_dive': default progression
        """
        if obs.hero.dungeon_branch == "mines":
            return "escape_mines"
        if obs.hero.hp_frac < 0.4 and obs.inventory.has_healing:
            return "emergency_heal"
        if obs.combat.adjacent_hostile or obs.combat.has_active_hostile:
            return "combat"
        if (
            obs.hero.depth <= 4
            and obs.hero.ac >= 6
            and (obs.inventory.has_unworn_armor or obs.spatial.has_nearby_loot)
        ):
            return "scavenge_armor"
        if obs.dungeon.can_forge_excalibur and (
            obs.dungeon.standing_on_fountain
            or obs.dungeon.adjacent_fountain
            or obs.dungeon.fountain_in_fov
            or obs.dungeon.has_known_fountain
        ):
            return "forge_excalibur"
        return "explore_and_dive"

    def run(self, obs):
        while True:
            # Universal Emergency Reflex 1: Escape Mines back to Dungeons of Doom
            if obs.hero.dungeon_branch == "mines":
                if obs.spatial.standing_on_stairs_up:
                    obs = (yield ascend())
                    continue
                elif obs.spatial.stairs_up_known:
                    obs = (yield step_to_stairs_up())
                    continue
                else:
                    obs = (yield step_to_frontier())
                    continue

            # Universal Emergency Reflex 2: Divine Favor Prayer (fainting or HP < 15%)
            if (
                obs.hero.hp_frac < 0.15
                or (obs.hero.hunger_state >= 4 and not obs.inventory.has_food)
            ) and (obs.hero.can_pray and obs.hero.turn - self.last_prayer_turn >= 850):
                self.last_prayer_turn = obs.hero.turn
                obs = (yield pray())
                continue

            # Universal Reflex 3: Immediate Descent on stairs down
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                if not (6 <= obs.hero.depth <= 10 and not obs.dungeon.has_sokoban_entrance and obs.spatial.has_unvisited_frontier):
                    obs = (yield descend())
                    continue

            # Strategic Goal Selection
            goal = self.determine_goal(obs)
            self.current_goal = goal

            if goal == "emergency_heal":
                obs = (yield quaff_healing())
                continue
            elif goal == "combat":
                obs = (yield from self.skill_combat(obs))
                continue
            elif goal == "scavenge_armor":
                obs = (yield from self.skill_scavenge_armor(obs))
                continue
            elif goal == "forge_excalibur":
                obs = (yield from self.skill_forge_excalibur(obs))
                continue
            else:
                obs = (yield from self.skill_explore_and_dive(obs))
                continue

    def skill_combat(self, obs):
        """
        Modular Skill: Combat Tactics.
        Handles adjacent hostiles, ranged throwing, Elbereth, and chokepoint fighting.
        """
        # Nutrition emergency in combat: prevent fainting coma
        if obs.hero.hunger_state >= 3 and obs.inventory.has_food:
            obs = (yield eat_carried_food())
            return obs

        # Floating eye and gas spore safety: ranged elimination or retreat
        if obs.combat.adjacent_floating_eye or obs.combat.adjacent_gas_spore:
            if obs.combat.has_safe_melee_target:
                obs = (yield melee_attack_hostile())
                return obs
            if obs.combat.adjacent_floating_eye:
                if obs.inventory.has_daggers:
                    obs = (yield throw_dagger())
                    return obs
                elif obs.inventory.has_offensive_wand:
                    obs = (yield zap_offensive_wand())
                    return obs
            obs = (yield step_away_from_hostile())
            return obs

        # Adjacent hostiles
        if obs.combat.adjacent_hostile:
            if (
                obs.combat.is_fast_dangerous
                and obs.hero.hp_frac < 0.40
                and not obs.combat.standing_on_elbereth
                and not obs.combat.hostile_ignores_elbereth
            ):
                obs = (yield engrave_elbereth())
                return obs
            if obs.combat.has_safe_melee_target or obs.hero.hp_frac > 0.35 or not obs.combat.can_retreat:
                obs = (yield melee_attack_hostile())
                return obs
            elif not obs.combat.standing_on_elbereth and not obs.combat.hostile_ignores_elbereth:
                obs = (yield engrave_elbereth())
                return obs
            obs = (yield step_away_from_hostile())
            return obs

        # Active hostiles at distance >= 2
        if obs.combat.has_active_hostile:
            if obs.inventory.has_offensive_wand:
                obs = (yield zap_offensive_wand())
                return obs
            if obs.inventory.has_daggers:
                obs = (yield throw_dagger())
                return obs
            if obs.combat.is_fast_dangerous and not obs.combat.in_corridor:
                obs = (yield step_to_chokepoint())
                return obs
            if not obs.combat.in_corridor:
                obs = (yield step_to_chokepoint())
                return obs
            obs = (yield wait())
            return obs

        obs = (yield wait())
        return obs

    def skill_scavenge_armor(self, obs):
        """
        Modular Skill: Equipment Scavenging.
        Prioritizes wearing unworn armor and collecting dropped armor/loot on early floors.
        """
        if obs.inventory.has_unworn_armor:
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

    def skill_forge_excalibur(self, obs):
        """
        Modular Skill: Excalibur Forging.
        Dips long sword in fountains when Valkyrie XL >= 5.
        """
        if obs.dungeon.standing_on_fountain:
            obs = (yield dip_excalibur())
            return obs
        if obs.dungeon.adjacent_fountain or obs.dungeon.fountain_in_fov:
            obs = (yield step_to_fountain())
            return obs
        if obs.dungeon.has_known_fountain and obs.dungeon.closest_fountain_depth is not None:
            if obs.dungeon.closest_fountain_depth != obs.hero.depth:
                obs = (yield backtrack_to_depth(target_depth=obs.dungeon.closest_fountain_depth))
                return obs

        obs = (yield step_to_frontier())
        return obs

    def skill_explore_and_dive(self, obs):
        """
        Modular Skill: Floor Exploration & Progressive Descent.
        Balances nutrition, armor equipping, and diving deeper.
        """
        if obs.hero.hunger_state >= 1 and any(c.is_safe for c in obs.corpses):
            obs = (yield eat_floor_corpse())
            return obs

        if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
            obs = (yield eat_carried_food())
            return obs

        if obs.inventory.has_unworn_armor:
            obs = (yield wear_armor())
            return obs

        if obs.spatial.has_nearby_loot and not obs.dungeon.in_shop:
            obs = (yield step_to_loot())
            return obs

        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs

        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs

        obs = (yield step_to_dead_end())
        return obs
