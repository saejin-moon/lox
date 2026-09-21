"""HTN method registry — extracted from CORPAgent (behavior-preserving split).

Mixed into CORPAgent; methods reference agent state via self.*"""

from typing import Any

from corp.env.blstats import HungerState
from corp.env.inventory_tracker import InventoryNormalizer
from corp.planner.htn import Task, Method, CompoundTask, HTNDeadlockException
from corp.epistemic.epistemic_manager import EpistemicManager
from corp.epistemic.entropy_gates import ShannonSafeGate
from corp.domain.medusa_handler import MedusaHandler
from corp.domain.shop_manager import ShopManager
from corp.workers.dispatcher import ActionDispatcher


class HTNMethodsMixin:
    def _setup_htn_methods(self):
        """
        Formalizes the HTN compound hierarchy and registers domain capability methods
        governed by continuous persona prioritization and CDCL Nogood branch cuts.
        """
        # Tier 1: Root Compound SURVIVE_AND_ASCEND decompositions
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_NUTRITION",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_emergency_eat,
            subtasks_fn=lambda s: [CompoundTask("EMERGENCY_NUTRITION")],
            base_utility=1000.0,
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_TRIAGE",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_emergency_triage,
            subtasks_fn=lambda s: [CompoundTask("EMERGENCY_TRIAGE")],
            base_utility=900.0,
            priority_weights=(2.0, 0.0, 0.0, 3.0, 1.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_COMBAT_ADJ",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_adjacent_combat,
            subtasks_fn=lambda s: [CompoundTask("TACTICAL_COMBAT")],
            base_utility=800.0,
            priority_weights=(4.0, -1.0, 0.0, -2.0, 0.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_RESOURCE",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=lambda s: True,
            subtasks_fn=lambda s: [CompoundTask("RESOURCE_MANAGEMENT")],
            base_utility=700.0,
            priority_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_SHOP",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_shop_action,
            subtasks_fn=lambda s: [CompoundTask("SHOP_ACTION")],
            base_utility=680.0,
            priority_weights=(0.0, 1.0, 3.0, 0.0, 3.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_EPISTEMIC",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_epistemic_action,
            subtasks_fn=lambda s: [CompoundTask("EPISTEMIC_ACTION")],
            base_utility=650.0,
            priority_weights=(0.0, 0.0, 2.0, 0.0, 4.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_MEDUSA_PREP",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_medusa_prep,
            subtasks_fn=lambda s: [CompoundTask("MEDUSA_PREP")],
            base_utility=660.0,
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_SKILL",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=lambda s: True,
            subtasks_fn=lambda s: [CompoundTask("SKILL_ENHANCEMENT")],
            base_utility=600.0,
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_COMBAT_RANGED",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=self._htn_can_ranged_combat,
            subtasks_fn=lambda s: [CompoundTask("TACTICAL_COMBAT")],
            base_utility=550.0,
            priority_weights=(-2.0, 5.0, 3.0, 2.0, 0.0),
        ))
        self.planner.register_method(Method(
            name="METHOD_SURVIVE_NAV",
            target_task="SURVIVE_AND_ASCEND",
            preconditions=lambda s: True,
            subtasks_fn=lambda s: [CompoundTask("SPATIAL_NAVIGATION")],
            base_utility=500.0,
            priority_weights=(0.0, 0.0, 0.0, 1.0, 0.0),
        ))

        # Tier 2: Domain-specific compound decompositions
        # EMERGENCY_NUTRITION
        self.planner.register_method(Method(
            name="EMERGENCY_NUTRITION_DECOMP",
            target_task="EMERGENCY_NUTRITION",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_emergency_eat,
        ))
        # EMERGENCY_TRIAGE
        self.planner.register_method(Method(
            name="EMERGENCY_TRIAGE_DECOMP",
            target_task="EMERGENCY_TRIAGE",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_emergency_triage,
        ))
        # TACTICAL_COMBAT (Melee vs Ranged methods)
        self.planner.register_method(Method(
            name="MELEE_COMBAT_DECOMP",
            target_task="TACTICAL_COMBAT",
            preconditions=self._htn_can_adjacent_combat,
            subtasks_fn=self._htn_do_adjacent_combat,
            priority_weights=(4.0, -2.0, 0.0, 0.0, 0.0),
        ))
        self.planner.register_method(Method(
            name="RANGED_COMBAT_DECOMP",
            target_task="TACTICAL_COMBAT",
            preconditions=self._htn_can_ranged_combat,
            subtasks_fn=self._htn_do_ranged_combat,
            priority_weights=(-2.0, 4.0, 2.0, 1.0, 0.0),
        ))
        # RESOURCE_MANAGEMENT
        self.planner.register_method(Method(
            name="RESOURCE_MANAGEMENT_DECOMP",
            target_task="RESOURCE_MANAGEMENT",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_resource_mgmt,
        ))
        # SHOP_ACTION
        self.planner.register_method(Method(
            name="SHOP_ACTION_DECOMP",
            target_task="SHOP_ACTION",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_shop_action,
        ))
        # EPISTEMIC_ACTION
        self.planner.register_method(Method(
            name="EPISTEMIC_ACTION_DECOMP",
            target_task="EPISTEMIC_ACTION",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_epistemic_action,
        ))
        # SKILL_ENHANCEMENT
        self.planner.register_method(Method(
            name="SKILL_ENHANCEMENT_DECOMP",
            target_task="SKILL_ENHANCEMENT",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_skill_enhance,
        ))
        # MEDUSA_PREP (Phase 4: DL 19-25 gaze protection equipping)
        self.planner.register_method(Method(
            name="MEDUSA_PREP_DECOMP",
            target_task="MEDUSA_PREP",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_medusa_prep,
        ))
        # SPATIAL_NAVIGATION
        self.planner.register_method(Method(
            name="SPATIAL_NAVIGATION_DECOMP",
            target_task="SPATIAL_NAVIGATION",
            preconditions=lambda s: True,
            subtasks_fn=self._htn_do_navigate,
        ))

    def _htn_can_emergency_eat(self, agent: Any) -> bool:
        return agent.current_blstats is not None and agent.current_blstats.hunger_state >= HungerState.WEAK

    def _htn_do_emergency_eat(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        lvl_map = agent.nav_mgr.get_or_create_level(blstats)
        msg_lower = agent.current_message.lower()
        is_on_fountain = (blstats.y, blstats.x) in lvl_map.fountains or "fountain" in msg_lower
        has_poison_res = agent.has_poison_resistance(agent.env.inventory_tracker)
        monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, blstats, has_poison_res=has_poison_res) if agent.current_glyphs is not None else []
        has_adj = any(m.is_adjacent for m in monsters)
        faint_task = agent.inv_mgr.evaluate_resource_turn(
            blstats,
            agent.env.inventory_tracker,
            message=agent.current_message,
            is_on_fountain=is_on_fountain,
            role=agent.current_role,
            race=agent.current_race,
            has_poison_res=has_poison_res,
            epistemic_mgr=agent.epistemic,
            has_adjacent_hostiles=has_adj,
            lvl_map=lvl_map,
        )
        if faint_task is not None:
            return [faint_task]
        # If no carried food and starving, emergency prayer immediately restores nutrition to 900
        if agent.inv_mgr.can_safely_pray(blstats):
            agent.inv_mgr.prayer_state.last_prayer_turn = blstats.turn
            agent.inv_mgr.prayer_state.prayer_count += 1
            return [Task("PRAY", is_primitive=True)]
        return []

    def _htn_can_emergency_triage(self, agent: Any) -> bool:
        blstats = agent.current_blstats
        if blstats is None or agent.current_glyphs is None:
            return False
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        has_adj = any(m.is_adjacent for m in agent.combat_mgr.scan_monsters(agent.current_glyphs, blstats, has_poison_res=has_pr))
        s = agent.policy_config.strategy
        return (
            blstats.hp <= max(s.triage_min_hp, int(blstats.max_hp * s.triage_hp_frac))
            or (has_adj and blstats.hp <= max(s.triage_adj_min_hp, int(blstats.max_hp * s.triage_adj_hp_frac)))
        )

    def _htn_do_emergency_triage(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        inv_tracker = agent.env.inventory_tracker
        # 1. Healing potion
        if inv_tracker is not None:
            active_items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
            for item in active_items:
                if item.buc_state == "CURSED":
                    continue
                if agent.epistemic is not None:
                    belief = agent.epistemic.get_or_create_belief(item)
                    safe, _ = ShannonSafeGate.can_safely_quaff(belief)
                    if not safe:
                        continue
                desc = item.raw_str.lower()
                if any(p in desc for p in ("potion of full healing", "potion of extra healing", "potion of healing")):
                    return [Task("QUAFF", is_primitive=True, args={"slot": item.current_letter})]

        # 2. Teleport scroll
        if inv_tracker is not None and blstats.hp <= max(5, int(blstats.max_hp * 0.30)):
            active_items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
            for item in active_items:
                if item.buc_state in ("BLESSED", "UNCURSED") and "scroll of teleportation" in item.raw_str.lower():
                    if agent.epistemic is not None:
                        belief = agent.epistemic.get_or_create_belief(item)
                        safe, _ = ShannonSafeGate.can_safely_read(belief)
                        if not safe:
                            continue
                    return [Task("READ", is_primitive=True, args={"slot": item.current_letter})]

        # 3. Divine prayer
        if blstats.hp <= max(5, int(blstats.max_hp * self.policy_config.survival.critical_retreat_frac)):
            if agent.inv_mgr.can_safely_pray(blstats):
                agent.inv_mgr.prayer_state.last_prayer_turn = blstats.turn
                agent.inv_mgr.prayer_state.prayer_count += 1
                return [Task("PRAY", is_primitive=True)]
        return []

    def _htn_can_adjacent_combat(self, agent: Any) -> bool:
        if agent.current_glyphs is None or agent.current_blstats is None:
            return False
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        return any(m.is_adjacent for m in agent.combat_mgr.scan_monsters(agent.current_glyphs, agent.current_blstats, has_poison_res=has_pr))

    def _htn_do_adjacent_combat(self, agent: Any) -> list[Task]:
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        has_refl = agent.has_reflection(agent.env.inventory_tracker)
        task = agent.combat_mgr.evaluate_combat_turn(
            agent.current_glyphs,
            agent.current_chars,
            agent.current_blstats,
            inv_tracker=agent.env.inventory_tracker,
            role=agent.current_role,
            message=agent.current_message,
            has_poison_res=has_pr,
            has_reflection=has_refl,
        )
        return [task] if task is not None else []

    def _htn_do_resource_mgmt(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        lvl_map = agent.nav_mgr.get_or_create_level(blstats)
        msg_lower = agent.current_message.lower()
        is_on_fountain = (blstats.y, blstats.x) in lvl_map.fountains or "fountain" in msg_lower
        has_poison_res = agent.has_poison_resistance(agent.env.inventory_tracker)
        monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, blstats, has_poison_res=has_poison_res) if agent.current_glyphs is not None else []
        has_adj = any(m.is_adjacent for m in monsters)
        task = agent.inv_mgr.evaluate_resource_turn(
            blstats,
            agent.env.inventory_tracker,
            message=agent.current_message,
            is_on_fountain=is_on_fountain,
            role=agent.current_role,
            race=agent.current_race,
            has_poison_res=has_poison_res,
            epistemic_mgr=agent.epistemic,
            has_adjacent_hostiles=has_adj,
            lvl_map=lvl_map,
        )
        return [task] if task is not None else []


    def _htn_can_epistemic_action(self, agent: Any) -> bool:
        if agent.current_blstats is None:
            return False
        lvl_map = agent.nav_mgr.get_or_create_level(agent.current_blstats)
        if not lvl_map.altars or agent.epistemic_worker.altar_test_count >= 10:
            return False
        inv_tracker = agent.env.inventory_tracker
        return (
            agent.epistemic_worker.has_untested_items(inv_tracker)
            or agent.epistemic_worker.get_sacrificable_corpse(inv_tracker, agent.current_blstats.turn) is not None
        )

    def _htn_do_epistemic_action(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        lvl_map = agent.nav_mgr.get_or_create_level(blstats)
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, blstats, has_poison_res=has_pr)
        has_adj = any(m.is_adjacent for m in monsters)
        task = agent.epistemic_worker.evaluate_epistemic_turn(
            blstats=blstats,
            chars=agent.current_chars,
            inv_tracker=agent.env.inventory_tracker,
            lvl_altars=lvl_map.altars,
            has_adjacent_hostiles=has_adj,
            visible_monster_count=len(monsters),
        )
        return [task] if task is not None else []

    def _htn_do_skill_enhance(self, agent: Any) -> list[Task]:
        task = agent.skill_mgr.evaluate_skill_turn(agent.current_blstats, message=agent.current_message)
        return [task] if task is not None else []

    def _htn_can_medusa_prep(self, agent: Any) -> bool:
        return (
            agent.medusa_handler.is_medusa_approach(agent.current_blstats)
            and not agent.has_reflection(agent.env.inventory_tracker)
        )

    def _htn_do_medusa_prep(self, agent: Any) -> list[Task]:
        task = agent.medusa_handler.evaluate_medusa_prep(
            agent.current_blstats,
            inv_tracker=agent.env.inventory_tracker,
            has_reflection=agent.has_reflection(agent.env.inventory_tracker),
        )
        return [task] if task is not None else []

    def _htn_can_ranged_combat(self, agent: Any) -> bool:
        if agent.current_glyphs is None or agent.current_blstats is None:
            return False
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        return len(agent.combat_mgr.scan_monsters(agent.current_glyphs, agent.current_blstats, has_poison_res=has_pr)) > 0

    def _htn_do_ranged_combat(self, agent: Any) -> list[Task]:
        has_pr = agent.has_poison_resistance(agent.env.inventory_tracker)
        has_refl = agent.has_reflection(agent.env.inventory_tracker)
        task = agent.combat_mgr.evaluate_combat_turn(
            agent.current_glyphs,
            agent.current_chars,
            agent.current_blstats,
            inv_tracker=agent.env.inventory_tracker,
            role=agent.current_role,
            message=agent.current_message,
            has_poison_res=has_pr,
            has_reflection=has_refl,
        )
        return [task] if task is not None else []

    def _htn_can_shop_action(self, agent: Any) -> bool:
        if agent.current_blstats is None or agent.current_chars is None:
            return False
        # 1. Unpaid item resolution (Top commercial priority to avoid shopkeeper rage)
        inv = agent.env.inventory_tracker if hasattr(agent.env, "inventory_tracker") else None
        if inv is not None and agent.shop_mgr.has_unpaid_items(inv):
            return True
        # 2. Temple priest donation
        if agent.shop_mgr.priest_pos is not None:
            monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, agent.current_blstats)
            has_adj = any(m.is_adjacent for m in monsters)
            if agent.shop_mgr.evaluate_temple_donation(agent.current_blstats, agent.current_chars, has_adjacent_hostiles=has_adj) is not None:
                return True
        # 3. Shop price probing
        if agent.shop_mgr.in_shop:
            monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, agent.current_blstats)
            has_adj = any(m.is_adjacent for m in monsters)
            if agent.shop_mgr.evaluate_shop_probing(agent.current_blstats, inv, agent.current_chars, has_adjacent_hostiles=has_adj) is not None:
                return True
        return False

    def _htn_do_shop_action(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        chars = agent.current_chars
        monsters = agent.combat_mgr.scan_monsters(agent.current_glyphs, blstats)
        has_adj = any(m.is_adjacent for m in monsters)
        inv = agent.env.inventory_tracker if hasattr(agent.env, "inventory_tracker") else None
        # 1. Priority: Unpaid item payment or debt drop
        if inv is not None and agent.shop_mgr.has_unpaid_items(inv):
            task = agent.shop_mgr.evaluate_shop_payment(blstats, inv, agent.current_message)
            if task is not None:
                return [task]
        # 2. Priority: Temple donation (divine protection for -10 AC)
        if agent.shop_mgr.priest_pos is not None:
            task = agent.shop_mgr.evaluate_temple_donation(blstats, chars, has_adjacent_hostiles=has_adj)
            if task is not None:
                return [task]
        # 3. Priority: Shop price probing
        if agent.shop_mgr.in_shop:
            task = agent.shop_mgr.evaluate_shop_probing(blstats, inv, chars, has_adjacent_hostiles=has_adj)
            if task is not None:
                return [task]
        return []

    def _htn_do_navigate(self, agent: Any) -> list[Task]:
        blstats = agent.current_blstats
        lvl_map = agent.nav_mgr.get_or_create_level(blstats)
        inv_tracker = agent.env.inventory_tracker
        target_fountain = False
        long_sword_slot = None
        is_lawful = (blstats.alignment == 1) or (agent.persona is not None and agent.persona.alignment >= 0.8) or (agent.current_role.lower() in ("valkyrie", "samurai", "knight"))
        if blstats.experience >= 5 and is_lawful and inv_tracker is not None:
            active_items = inv_tracker.get_active_items() if hasattr(inv_tracker, "get_active_items") else [it for it in getattr(inv_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
            has_excalibur = any("excalibur" in it.raw_str.lower() for it in active_items)
            if not has_excalibur:
                for it in active_items:
                    raw = it.raw_str.lower()
                    if "long sword" in raw and "cursed" not in raw:
                        target_fountain = True
                        long_sword_slot = it.current_letter
                        break

        target_altar = False
        if lvl_map.altars and lvl_map.turns_spent < 50 and agent.epistemic_worker.altar_test_count < 10 and (
            agent.epistemic_worker.has_untested_items(inv_tracker)
            or agent.epistemic_worker.get_sacrificable_corpse(inv_tracker, blstats.turn) is not None
        ):
            target_altar = True

        has_poison_res = agent.has_poison_resistance(inv_tracker)
        task = agent.nav_mgr.evaluate_navigation_turn(
            agent.current_chars,
            blstats,
            target_fountain=target_fountain,
            target_altar=target_altar,
            message=agent.current_message,
            inv_tracker=inv_tracker,
            glyphs=agent.current_glyphs,
            has_poison_res=has_poison_res,
            long_sword_slot=long_sword_slot,
            dungeon_graph=agent.dungeon_graph,
            macro_director=agent.goals,
        )
        if task is not None:
            if task.name == "KICK" and "delta" in task.args:
                dr, dc = task.args["delta"]
                door_pos = (blstats.y + dr, blstats.x + dc)
                lvl_map = agent.nav_mgr.get_or_create_level(blstats)
                if (
                    door_pos not in lvl_map.shop_doors
                    and not lvl_map.has_shops
                    and (agent.current_chars == ord("@")).sum() <= 1
                ):
                    agent.locked_intent = LockedIntent(
                        name="DOOR_KICK",
                        target_pos=door_pos,
                        remaining_steps=29,
                    )
            return [task]
        return []
