class Agent:

    def __init__(self):
        self.last_prayer_turn = -1000
        self.search_count = 0
        self.retreat_streak = 0

    def run(self, obs):
        while True:
            # 1. Emergency Prayer
            if obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 2 and not obs.inventory.has_food):
                if obs.hero.turn - self.last_prayer_turn >= 850:
                    self.last_prayer_turn = obs.hero.turn
                    obs = (yield pray())
                    continue

            # 2. Immediate Staircase Descent Priority
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = (yield descend())
                continue

            # 3. Combat Logic (Bypass distant passive hazards if lacking ranged ammo)
            if (
                obs.combat.adjacent_hostile
                or obs.combat.has_active_hostile
                or (
                    obs.combat.hostile_count_fov > 0
                    and (
                        obs.inventory.has_offensive_wand
                        or obs.inventory.has_daggers
                    )
                )
            ):
                obs = (yield from self.handle_combat(obs))
                continue

            # 4. Proactive Nutrition
            if any(c.is_safe for c in obs.corpses) and obs.hero.hunger_state >= 1:
                obs = (yield from self.handle_corpse_consumption(obs))
                continue
            if obs.hero.hunger_state >= 1 and obs.inventory.has_food:
                obs = (yield eat_carried_food())
                continue

            # 5. Weapon Skill Enhancement Mastery
            if obs.hero.can_enhance_skills:
                obs = (yield enhance_weapon_skill())
                continue

            # 6. Superior Body Armor Replacement (Dwarvish Mithril / Iron Cuirass over Leather Jacket)
            if obs.inventory.get_superior_body_armor_slot() is not None:
                obs = (yield replace_body_armor())
                continue

            # 7. Unworn Auxiliary Armor (Helmets, Boots, Cloaks, Gloves, Shields)
            if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
                obs = (yield wear_armor())
                continue

            # 8. Known Staircase Descent
            if obs.spatial.stairs_down_known:
                obs = (yield step_to_stairs_down())
                continue

            # 9. Excalibur Forging
            if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
                if obs.dungeon.standing_on_fountain:
                    obs = (yield dip_excalibur())
                    continue
                elif obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
                    obs = (yield step_to_fountain())
                    continue

            # 10. Floor Loot Scooping
            if obs.spatial.has_nearby_loot:
                obs = (yield step_to_loot())
                continue

            # 11. Altar BUC Testing & Divine Sacrificing
            if obs.dungeon.standing_on_altar:
                if obs.dungeon.can_sacrifice:
                    obs = (yield sacrifice_on_altar())
                    continue
                elif obs.epistemic.has_untested_items:
                    obs = (yield test_altar_buc())
                    continue
            elif obs.dungeon.adjacent_altar and obs.epistemic.has_untested_items:
                obs = (yield step_to_altar())
                continue

            # 12. Temple Priest Intrinsic AC Protection Donation
            if obs.dungeon.can_donate_to_priest:
                obs = (yield donate_to_priest())
                continue

            # 13. Telepathy Blindfold Periodic Scouting
            if obs.inventory.has_blindfold and not obs.hero.is_blind and not obs.combat.adjacent_hostile and obs.hero.turn % 150 == 0:
                obs = (yield apply_blindfold())
                continue

            # 14. Poison Resistance Harvesting
            if obs.dungeon.can_harvest_poison and not obs.hero.has_poison_res and obs.hero.hp_frac > 0.9:
                obs = (yield harvest_poison_res())
                continue

            # 15. Sokoban Branch Boulder Solver
            if obs.dungeon.can_solve_sokoban:
                obs = (yield step_solve_sokoban())
                continue

            # 16. Castle Drawbridge Breaching
            if obs.dungeon.can_breach_drawbridge:
                obs = (yield breach_drawbridge())
                continue

            # 17. Gehennom Straight-Line Tunneling
            if obs.dungeon.can_tunnel_gehennom and not obs.spatial.stairs_down_known:
                obs = (yield dig_tunnel())
                continue

            # 18. Endgame Invocation Ritual
            if obs.dungeon.can_perform_invocation:
                obs = (yield perform_invocation_step())
                continue

            # 19. Gnomish Mines Evacuation
            if obs.hero.dungeon_branch == "mines":
                if obs.spatial.standing_on_stairs_up:
                    obs = (yield ascend())
                    continue
                obs = (yield step_to_stairs_up())
                continue

            # 14. Closed-Door Navigation & Breaching
            if obs.dungeon.adjacent_closed_door:
                if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
                    obs = (yield kick_closed_door())
                else:
                    obs = (yield open_door())
                continue
            elif obs.spatial.has_unvisited_frontier:
                self.search_count = 0
                obs = (yield step_to_frontier())
                continue
            elif obs.dungeon.has_closed_door:
                obs = (yield step_to_closed_door())
                continue
            elif obs.spatial.has_unsearched_dead_end:
                obs = (yield from self.handle_dead_end(obs))
                continue
            else:
                obs = (yield from self.handle_dead_end(obs))
                continue

    def handle_combat(self, obs):
        while obs.combat.hostile_count_fov > 0 or obs.combat.adjacent_hostile:
            # Immediate staircase combat escape
            if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
                obs = (yield descend())
                break

            # Passive hazard exit break (return to exploration if distant immobile hazards remain)
            if not obs.combat.adjacent_hostile and not obs.combat.has_active_hostile:
                if not (obs.inventory.has_offensive_wand or obs.inventory.has_daggers):
                    break

            # Panic Escape
            if (obs.hero.hp_frac < 0.20 or obs.combat.is_surrounded) and obs.combat.has_panic_escape:
                if obs.inventory.has_scroll_of_teleport:
                    obs = (yield read_scroll_teleport())
                    continue
                elif obs.inventory.has_wand_of_teleport:
                    obs = (yield zap_wand_teleport())
                    continue

            # Emergency In-Combat Healing
            if obs.hero.hp_frac < 0.50 and obs.inventory.has_healing:
                obs = (yield quaff_healing())
                continue

            # In-Combat Hunger Eating / Weakness Conscious Prayer
            if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
                if obs.combat.standing_on_elbereth or not obs.combat.adjacent_hostile or obs.combat.adjacent_floating_eye or obs.combat.adjacent_gas_spore:
                    obs = (yield eat_carried_food())
                    continue
            if (obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 2 and not obs.inventory.has_food)) and (obs.hero.turn - self.last_prayer_turn >= 850):
                if not obs.combat.adjacent_hostile or obs.combat.standing_on_elbereth:
                    self.last_prayer_turn = obs.hero.turn
                    obs = (yield pray())
                    continue

            # Passive & Exploding Hazards
            if obs.combat.adjacent_gas_spore:
                obs = (yield step_away_from_hostile())
                continue
            if obs.combat.adjacent_floating_eye:
                if obs.combat.has_safe_melee_target:
                    obs = (yield melee_attack_hostile())
                    continue
                if obs.inventory.has_daggers:
                    obs = (yield throw_dagger())
                    continue
                elif obs.inventory.has_offensive_wand:
                    obs = (yield zap_offensive_wand())
                    continue
                else:
                    obs = (yield step_away_from_hostile())
                    continue

            # Status Hazards: Homunculi (Sleep) & Yellow Lights (Blindness)
            closest_name_low = obs.combat.closest_hostile_name.lower()
            if any(h in closest_name_low for h in ("homunculus", "yellow light")):
                if obs.combat.closest_hostile_dist >= 2:
                    if obs.inventory.has_offensive_wand:
                        obs = (yield zap_offensive_wand())
                        continue
                    if obs.inventory.has_daggers:
                        obs = (yield throw_dagger())
                        continue
                elif not obs.combat.standing_on_elbereth:
                    obs = (yield engrave_dust_elbereth())
                    continue

            # Peaceful NPC Non-Aggression
            if closest_name_low in ("shopkeeper", "watchman", "watch captain", "guard", "priest", "priestess", "oracle") or obs.dungeon.in_shop:
                obs = (yield (retreat() if obs.combat.can_retreat else step_away_from_hostile()))
                continue

            # Pack / Herd Threat Defense: Rothes, ant swarms, orc squads
            if obs.combat.is_pack_threat and not obs.combat.in_corridor and not obs.combat.standing_on_elbereth:
                if not (obs.combat.adjacent_hostile and obs.combat.hostile_ignores_elbereth):
                    obs = (yield engrave_dust_elbereth())
                    continue
                elif obs.combat.can_retreat:
                    obs = (yield step_to_chokepoint())
                    continue

            # Dust Elbereth Sanctuary Warding
            if not obs.combat.standing_on_elbereth:
                if obs.hero.hp_frac < 0.50 or obs.combat.hostile_count_fov >= 2 or obs.combat.is_fast_dangerous:
                    if not (obs.combat.adjacent_hostile and obs.combat.hostile_ignores_elbereth):
                        obs = (yield engrave_dust_elbereth())
                        continue

            # Standing on Elbereth Tactical Response
            if obs.combat.standing_on_elbereth:
                if obs.combat.hostile_ignores_elbereth and obs.combat.adjacent_hostile:
                    if obs.hero.hp_frac > 0.50 or not obs.combat.can_retreat:
                        obs = (yield melee_attack_hostile())
                        continue
                    else:
                        obs = (yield (step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()))
                        continue
                if not obs.combat.adjacent_hostile:
                    if obs.inventory.has_offensive_wand and obs.combat.closest_hostile_dist >= 2:
                        obs = (yield zap_offensive_wand())
                        continue
                    if obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
                        obs = (yield throw_dagger())
                        continue
                    if obs.spatial.stairs_down_known:
                        obs = (yield step_to_stairs_down())
                        continue
                    obs = (yield step_away_from_hostile())
                    continue
                else:
                    obs = (yield melee_attack_hostile())
                    continue

            # Distance >= 2 Ranged Elimination
            if obs.combat.closest_hostile_dist >= 2:
                if obs.inventory.has_offensive_wand:
                    obs = (yield zap_offensive_wand())
                    continue
                elif obs.inventory.has_daggers:
                    obs = (yield throw_dagger())
                    continue

            # Decisive Adjacent Melee vs Corridor Retreat
            if obs.combat.adjacent_hostile:
                if obs.hero.hp_frac > 0.40 or not obs.combat.can_retreat:
                    obs = (yield melee_attack_hostile())
                elif obs.combat.in_corridor:
                    obs = (yield step_away_from_hostile())
                else:
                    obs = (yield step_to_chokepoint())
            elif obs.hero.hp_frac > 0.50:
                obs = (yield melee_attack_hostile())
            else:
                obs = (yield (step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()))

            if obs.combat.hostile_count_fov == 0 and not obs.combat.adjacent_hostile:
                break
        return obs

    def handle_corpse_consumption(self, obs):
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
        obs = (yield (step_to_frontier() if obs.spatial.has_unvisited_frontier else step_to_dead_end()))
        return obs

    def handle_dead_end(self, obs):
        if obs.spatial.standing_on_dead_end:
            for _ in range(5):
                if obs.combat.hostile_count_fov > 0:
                    return obs
                if obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
                    return obs
                obs = (yield search())
            if obs.spatial.stairs_down_known:
                obs = (yield step_to_stairs_down())
            elif obs.spatial.has_unvisited_frontier:
                obs = (yield step_to_frontier())
            elif obs.dungeon.has_closed_door:
                obs = (yield step_to_closed_door())
            else:
                obs = (yield step_to_dead_end())
            return obs
        else:
            obs = (yield step_to_dead_end())
            return obs
