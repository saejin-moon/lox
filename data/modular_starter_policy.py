class Agent:

    def __init__(self):
        self.last_prayer_turn = -1000
        self.current_goal = "explore_and_dive"
        self._dead_end_search_count = 0

    def determine_goal(self, obs) -> str:
        # Phase 1: Endgame Ascension run
        if obs.inventory.has_amulet_of_yendor or obs.dungeon.phase == "ascension_run":
            return "ascension_run"
        # Phase 2: Invocations & Sanctum
        if obs.dungeon.phase == "invocation" or (obs.hero.depth >= 45 and obs.hero.turn > 15000):
            return "perform_invocation"
        # Phase 3: Castle Drawbridge Breach
        if obs.dungeon.phase == "castle_breach" or (obs.hero.depth in (25, 26, 27) and obs.dungeon.is_castle_level):
            return "breach_castle"
        # Phase 4: Sokoban Solver
        if obs.dungeon.phase == "sokoban" or obs.dungeon.is_sokoban_level:
            return "solve_sokoban"
        # Tactical & Survival Goals
        if obs.combat.adjacent_hostile or obs.combat.has_active_hostile:
            return "combat"
        if obs.hero.hunger_state >= 2 or (obs.hero.hunger_state >= 1 and any(c.is_safe for c in obs.corpses)):
            return "nutrition"
        if obs.dungeon.can_forge_excalibur and (obs.dungeon.standing_on_fountain or obs.dungeon.fountain_in_fov or obs.dungeon.has_known_fountain):
            return "forge_excalibur"
        if obs.inventory.has_unworn_armor or obs.inventory.get_superior_body_armor_slot() is not None or obs.spatial.has_nearby_loot:
            return "scavenge_armor"
        return "explore_and_dive"

    def run(self, obs):
        while True:
            # Reflex 1: Mines Evacuation (avoid dark Gnomish Mines branch)
            if obs.hero.dungeon_branch == "mines":
                if obs.spatial.standing_on_stairs_up and not obs.status.is_levitating:
                    obs = (yield ascend())
                    continue
                elif obs.spatial.stairs_up_known:
                    obs = (yield step_to_stairs_up())
                    continue
                else:
                    obs = (yield step_to_frontier())
                    continue

            # Reflex 2: Armor Upgrades
            if obs.inventory.get_superior_body_armor_slot() is not None:
                obs = (yield replace_body_armor())
                continue
            if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
                obs = (yield wear_armor())
                continue

            # Reflex 3: Nutrition (Fresh floor corpse at >=1, Carried food ONLY at >=2)
            if obs.hero.hunger_state >= 1 and any(c.is_safe for c in obs.corpses) and not obs.combat.adjacent_hostile:
                obs = (yield eat_floor_corpse())
                continue
            if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
                obs = (yield eat_carried_food())
                continue

            # Reflex 4: Major Trouble Divine Prayer (HP < 15% or Fainting without food)
            if (obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 4 and not obs.inventory.has_food)) and (obs.hero.can_pray and obs.hero.turn - self.last_prayer_turn >= 850):
                self.last_prayer_turn = obs.hero.turn
                obs = (yield pray())
                continue

            # Reflex 5: Combat Engagement
            if obs.combat.adjacent_hostile or obs.combat.has_active_hostile:
                obs = (yield from self.skill_combat(obs))
                continue

            # Reflex 6: Fast Vertical Transit (Descend immediately when on stairs down)
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = (yield descend())
                continue
            if obs.spatial.stairs_down_known and (not obs.spatial.has_nearby_loot or obs.hero.turns_on_level >= 30 or obs.hero.depth >= 3):
                obs = (yield step_to_stairs_down())
                continue

            # Strategic Goal Dispatch
            goal = self.determine_goal(obs)
            self.current_goal = goal
            if goal == "ascension_run":
                obs = (yield from self.skill_ascension_run(obs))
                continue
            elif goal == "perform_invocation":
                obs = (yield from self.skill_perform_invocation(obs))
                continue
            elif goal == "breach_castle":
                obs = (yield from self.skill_breach_castle(obs))
                continue
            elif goal == "solve_sokoban":
                obs = (yield from self.skill_solve_sokoban(obs))
                continue
            elif goal == "combat":
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

    def skill_ascension_run(self, obs):
        if obs.dungeon.standing_on_high_altar:
            obs = (yield offer_amulet_on_altar())
            return obs
        if obs.dungeon.standing_on_plane_portal:
            obs = (yield step_direction())
            return obs
        if obs.dungeon.has_plane_portal:
            obs = (yield step_to_plane_portal())
            return obs
        if obs.spatial.standing_on_stairs_up and not obs.status.is_levitating:
            obs = (yield ascend())
            return obs
        if obs.spatial.stairs_up_known:
            obs = (yield step_to_stairs_up())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield step_to_dead_end())
        return obs

    def skill_perform_invocation(self, obs):
        if obs.dungeon.standing_on_vibrating_square or obs.dungeon.can_perform_invocation:
            obs = (yield perform_invocation_step())
            return obs
        if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
            obs = (yield descend())
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield step_to_dead_end())
        return obs

    def skill_breach_castle(self, obs):
        if obs.dungeon.can_breach_drawbridge or obs.dungeon.drawbridge_in_fov:
            obs = (yield breach_drawbridge())
            return obs
        if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
            obs = (yield descend())
            return obs
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield step_to_dead_end())
        return obs

    def skill_solve_sokoban(self, obs):
        if obs.dungeon.can_solve_sokoban or obs.dungeon.is_sokoban:
            obs = (yield solve_sokoban())
            return obs
        if obs.dungeon.has_sokoban_entrance:
            obs = (yield step_to_sokoban_entrance())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        obs = (yield step_to_dead_end())
        return obs

    def skill_combat(self, obs):
        # Emergency healing during combat
        if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing and obs.epistemic.can_safely_quaff_healing:
            obs = (yield quaff_healing())
            return obs

        # Tactical stairs escape
        if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating and obs.hero.hp_frac < 0.60:
            obs = (yield descend())
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

            # Decisive melee vs sanctuary
            if obs.hero.hp_frac > 0.30 or obs.combat.in_corridor or not obs.combat.can_retreat:
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
                if obs.inventory.has_daggers and obs.combat.closest_hostile_dist <= 3:
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
        obs = (yield from self.skill_explore_and_dive(obs))
        return obs

    def skill_explore_and_dive(self, obs):
        # Fast descent: take stairs down immediately if known and immediate loot cleared
        if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
            obs = (yield descend())
            return obs
        if obs.spatial.stairs_down_known and (not obs.spatial.has_nearby_loot or obs.hero.turns_on_level >= 30 or obs.hero.depth >= 3):
            obs = (yield step_to_stairs_down())
            return obs
        if obs.spatial.has_unvisited_frontier:
            obs = (yield step_to_frontier())
            return obs
        if obs.spatial.has_nearby_loot:
            obs = (yield step_to_loot())
            return obs
        obs = (yield from self.handle_dead_end(obs))
        return obs

    def handle_dead_end(self, obs):
        if obs.spatial.stairs_down_known:
            obs = (yield step_to_stairs_down())
            return obs
        if obs.spatial.standing_on_dead_end:
            if getattr(self, "_dead_end_search_count", 0) < 5:
                self._dead_end_search_count = getattr(self, "_dead_end_search_count", 0) + 1
                obs = (yield search())
                return obs
            self._dead_end_search_count = 0
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
