"""
LOX Empirical Invariant Knowledge Base.
Structured, typed, machine-readable repository of all empirical rules, tactical
mechanisms, anti-patterns, and code snippets derived from 48+ synthesis campaigns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Invariant:
    """A single empirical tactical or architectural invariant."""

    id: str
    title: str
    category: str  # "combat", "navigation", "nutrition", "equipment", "dialog_harness", "architecture"
    tags: list[str]
    rule: str
    anti_pattern: str
    code_snippet: str
    related_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "tags": self.tags,
            "rule": self.rule,
            "anti_pattern": self.anti_pattern,
            "code_snippet": self.code_snippet,
            "related_ids": self.related_ids,
        }

    def format_prompt_block(self) -> str:
        """Formats the invariant as a high-density reference block for LLM prompts."""
        return (
            f"### [{self.id}] {self.title} ({self.category.upper()})\n"
            f"**Rule**: {self.rule}\n"
            f"**Avoid**: {self.anti_pattern}\n"
            f"**Verified Code Pattern**:\n```python\n{self.code_snippet.strip()}\n```\n"
        )


# =====================================================================
# Canonical Invariant Definitions
# =====================================================================

INVARIANTS: list[Invariant] = [
    # -----------------------------------------------------------------
    # NAVIGATION & TOPOLOGY
    # -----------------------------------------------------------------
    Invariant(
        id="INV-NAV-001",
        title="Immediate Staircase Descent Priority",
        category="navigation",
        tags=["stairs", "descend", "combat", "escape", "progression"],
        rule=(
            "If standing on stairs down and not levitating, policies must prioritize descend() "
            "immediately, even during combat. Escaping takes 1 turn and guarantees 100% survival "
            "against lethal mid-dungeon threats. In skill_combat(), standing on stairs down must "
            "instantly yield descend() and break combat."
        ),
        anti_pattern=(
            "Placing skill_combat() before checking standing_on_stairs_down, or engaging in "
            "melee combat while standing directly on the down-stairs."
        ),
        code_snippet="""
# In run():
if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
    obs = (yield descend())
    continue

# In skill_combat():
if obs.spatial.standing_on_stairs_down and not obs.status.is_levitating:
    obs = (yield descend())
    return obs
""",
        related_ids=["INV-NAV-002", "INV-CBT-001"],
    ),
    Invariant(
        id="INV-NAV-002",
        title="Gnomish Mines Immediate Evacuation",
        category="navigation",
        tags=["mines", "gnomish mines", "stairs_up", "ascend", "branch"],
        rule=(
            "In NetHack, the Gnomish Mines branch (dnum == 2 or dungeon_branch == 'mines') is "
            "lethal for early-game heroes. The hero enters via up-stairs (<); policies must "
            "navigate to stairs_up and yield ascend() back to the Dungeons of Doom to continue "
            "toward Depth 20. The harness prunes mines stairs from known_stairs_down."
        ),
        anti_pattern=(
            "Checking stairs_down or yielding descend() while in the Mines, driving the hero deeper "
            "into dark, trap-filled gnome/dwarf swarms."
        ),
        code_snippet="""
if obs.hero.dungeon_branch == "mines":
    if obs.spatial.standing_on_stairs_up:
        obs = (yield ascend())
        continue
    obs = (yield step_to_stairs_up())
    continue
""",
        related_ids=["INV-NAV-001"],
    ),
    Invariant(
        id="INV-NAV-003",
        title="Closed-Door Inter-Room Transitioning",
        category="navigation",
        tags=["closed_door", "door", "frontier", "dead_end", "search"],
        rule=(
            "When all floor tiles in the current room are visited (has_unvisited_frontier == False), "
            "policies must check obs.dungeon.has_closed_door and yield step_to_closed_door() before "
            "falling back to dead-end wall search. Omitting step_to_closed_door() previously caused "
            "heroes to spend 1,000+ turns searching walls while unexplored rooms lay directly behind closed doors."
        ),
        anti_pattern=(
            "Falling directly into wall search when has_unvisited_frontier is False while "
            "unopened closed doors are visible in the room."
        ),
        code_snippet="""
# In skill_explore_and_dive():
if obs.spatial.has_unvisited_frontier:
    obs = (yield step_to_frontier())
    return obs
if obs.dungeon.has_closed_door:
    obs = (yield step_to_closed_door())
    return obs
if obs.spatial.has_unsearched_dead_end:
    obs = (yield step_to_dead_end())
    return obs
""",
        related_ids=["INV-NAV-004"],
    ),
    Invariant(
        id="INV-NAV-004",
        title="15-Kick Door Breaching & Dynamic Obstacle Learning",
        category="navigation",
        tags=["kick", "locked_door", "breach", "blocked_tiles", "door"],
        rule=(
            "In NetHack, wooden locked doors can take 7 to 12 kicks for an 18 Str Valkyrie. "
            "The kick limit must be 15 kicks for wooden doors ('WHAMMM!!!'). Breached doors ('crash', 'shatter') "
            "are cleared from locked tracking. Tiles are added to blocked_tiles ONLY after 15 failed kicks "
            "or when the door injures the hero's leg ('ouch', 'hurt'), preventing heroes from being locked out."
        ),
        anti_pattern=(
            "Giving up after 6 kicks and permanently adding the only exit door to blocked_tiles, "
            "trapping the hero on DL 1-2 for thousands of turns until starvation."
        ),
        code_snippet="""
if obs.dungeon.adjacent_closed_door:
    if obs.dungeon.door_is_locked and not obs.dungeon.in_shop:
        obs = (yield kick_closed_door())
    else:
        obs = (yield open_door())
    continue
""",
        related_ids=["INV-NAV-003"],
    ),
    Invariant(
        id="INV-NAV-005",
        title="Hierarchical True Dead End Prioritization",
        category="navigation",
        tags=["dead_end", "secret_door", "corridor", "search", "spatial"],
        rule=(
            "Corridor dead ends (walkable # or lit . with <= 1 cardinal neighbor) connect directly "
            "to unexplored rooms and must take absolute Priority 1 over room perimeter wall searching in "
            "step_to_dead_end. In skill_explore_and_dive(), execute up to 15 searches per tile (>91% discovery chance) "
            "breaking immediately if stairs or frontiers are revealed. When dead ends are exhausted, decay searched "
            "count immediately to prevent 50-turn 2-tile ping-pong oscillations."
        ),
        anti_pattern=(
            "Searching only 5 times (44% failure rate) and then bouncing between 2 tiles in a 50-turn decay stall."
        ),
        code_snippet="""
# In skill_explore_and_dive(self, obs):
if obs.spatial.standing_on_dead_end:
    for _ in range(15):
        if obs.combat.hostile_count_fov > 0 or obs.spatial.stairs_down_known or obs.spatial.has_unvisited_frontier:
            return obs
        obs = (yield search())
    return obs
if obs.spatial.stairs_down_known:
    obs = (yield step_to_stairs_down())
    return obs
if obs.spatial.has_unvisited_frontier:
    obs = (yield step_to_frontier())
    return obs
if obs.dungeon.has_closed_door:
    obs = (yield step_to_closed_door())
    return obs
obs = (yield step_to_dead_end())
return obs
""",
        related_ids=["INV-NAV-003", "INV-NAV-006"],
    ),
    Invariant(
        id="INV-NAV-006",
        title="Full-Floor Persistent Topological Memory",
        category="navigation",
        tags=["memory", "fov", "visited", "known_chars", "topological"],
        rule=(
            "NetHack zeroes glyphs outside active line-of-sight FOV. The adapter maintains self.visited "
            "and discovered terrain (self.known_chars for #, ., <, >, _, {, +) across the entire floor. "
            "Static fixtures (stairs_down, stairs_up, fountains, altars) are preserved across FOV exits "
            "until genuine level transitions or branch stair pruning."
        ),
        anti_pattern=(
            "Relying on raw_obs glyphs for pathfinding, causing amnesia where explored rooms and stairs "
            "disappear from navigation as soon as the hero walks into a dark corridor."
        ),
        code_snippet="""
# Harness invariant: SpatialEngine navigates via walkable_nav built from:
# self.visited | clean_terrain(self.known_chars) | known_stairs_down | known_doors
""",
        related_ids=["INV-NAV-005"],
    ),
    Invariant(
        id="INV-NAV-007",
        title="Sokoban Branch Entrance Detection & Ascend Transit",
        category="navigation",
        tags=["sokoban", "branch", "stairs_up", "ascend", "entrance", "bag of holding", "reflection"],
        rule=(
            "On Depths 6–10, the Sokoban entrance appears as an additional staircase up (<) that is not the floor's arrival stairs. "
            "Sokoban offers guaranteed non-rotting food and either the Amulet of Reflection or Bag of Holding without monster spawns "
            "in solved puzzle chambers. When obs.dungeon.has_sokoban_entrance is True and HP is safe (hp_frac >= 0.70), policies must "
            "yield step_to_sokoban_entrance() to navigate to and ascend into Sokoban."
        ),
        anti_pattern=(
            "Confusing the Sokoban up-staircase with the level arrival staircase, or ignoring Sokoban entirely and descending into "
            "deadly DL 10+ monster clusters unequipped."
        ),
        code_snippet="""
if obs.dungeon.has_sokoban_entrance and obs.hero.hp_frac >= 0.70:
    if obs.spatial.standing_on_stairs_up and obs.hero.dungeon_branch != "mines":
        obs = (yield ascend())
        return obs
    obs = (yield step_to_sokoban_entrance())
    return obs
""",
        related_ids=["INV-NAV-001", "INV-NAV-002"],
    ),
    Invariant(
        id="INV-NAV-008",
        title="Early-Game Fast Descent & Zero-Loot Staircase Priority",
        category="navigation",
        tags=["early rush", "stairs_down", "descend", "dl1", "dl2", "zero-loot", "pacing"],
        rule=(
            "On shallow levels (DL 1–2), monster threat is minimal and dropped monster equipment consists mostly of junk weapons. "
            "Lingering on DL 1–2 to scoop loot burns food nutrition and wastes hundreds of turns. In skill_explore_and_dive(), discovering "
            "stairs down (obs.spatial.stairs_down_known) MUST take absolute precedence over step_to_loot(). Descend immediately to DL 3+ "
            "where mithril armor and fountains generate."
        ),
        anti_pattern=(
            "Prioritizing step_to_loot() over step_to_stairs_down() on DL 1–2, burning 3,000+ turns wandering back and forth over "
            "dropped darts and daggers while starving."
        ),
        code_snippet="""
# In skill_explore_and_dive() [DL 1-2]:
if obs.spatial.stairs_down_known:
    obs = (yield step_to_stairs_down())
    return obs
if obs.spatial.has_nearby_loot and obs.hero.depth >= 3:
    obs = (yield step_to_loot())
    return obs
""",
        related_ids=["INV-NAV-001", "INV-EQP-001"],
    ),
    Invariant(
        id="INV-NAV-009",
        title="Starting Room Secret Door Escalation & 100-Turn Stagnation Decay",
        category="navigation",
        tags=["dead_end", "search", "secret_door", "stagnation", "decay", "dl1", "starting_room"],
        rule=(
            "On DL 1–2, procedural generation can seal the starting room behind secret doors with no external corridors or visible frontiers. "
            "In NetHackAdapter, when frontiers are empty and stairs down are unknown, search decay triggers after 100 turns (instead of 500), "
            "and starting room perimeter search escalates to 20 sweeps. In policies, skill_explore_and_dive() must abort searching immediately "
            "upon taking unexpected damage or detecting adjacent hostiles."
        ),
        anti_pattern=(
            "Lingering in a 500-turn stagnation lockout on DL 1, waiting in place until starving to death because secret door search "
            "counts were frozen."
        ),
        code_snippet="""
prev_hp = obs.hero.hp
for _ in range(20 if obs.hero.depth <= 2 else 12):
    if (
        obs.combat.hostile_count_fov > 0
        or obs.combat.adjacent_hostile
        or obs.hero.hp < prev_hp
        or obs.spatial.stairs_down_known
        or obs.spatial.has_unvisited_frontier
    ):
        return obs
    obs = (yield search())
""",
        related_ids=["INV-NAV-005", "INV-CBT-011"],
    ),
    # -----------------------------------------------------------------
    # COMBAT TACTICS & WARDING
    # -----------------------------------------------------------------
    Invariant(
        id="INV-CBT-001",
        title="Decisive Adjacent Melee Engagement vs Non-Retreat Rule",
        category="combat",
        tags=["melee", "retreat", "free_hits", "rats", "newts", "combat"],
        rule=(
            "Against adjacent hostiles in open rooms, stepping away deals 0 damage and grants enemies "
            "free attacks every turn. Valkyries with +1 long sword one-shot or two-shot early pests "
            "(rats, newts, grid bugs, goblins, bats). Policies must strike decisively (melee_attack_hostile) "
            "at hp_frac > 0.35. If hp_frac <= 0.35, engrave dust Elbereth. Never step away dealing 0 damage."
        ),
        anti_pattern=(
            "Yielding step_away_from_hostile() or step_to_chokepoint() from an adjacent enemy when at "
            "50-60% HP, taking repeated free hits until dead."
        ),
        code_snippet="""
if obs.combat.adjacent_hostile:
    if obs.hero.hp_frac > 0.35 or not obs.combat.can_retreat:
        obs = (yield melee_attack_hostile())
    else:
        if not obs.combat.standing_on_elbereth and not obs.combat.hostile_ignores_elbereth:
            obs = (yield engrave_dust_elbereth())
        else:
            obs = (yield melee_attack_hostile())
""",
        related_ids=["INV-CBT-002", "INV-CBT-003"],
    ),
    Invariant(
        id="INV-CBT-002",
        title="Gas Spore 4d6 Explosion Shield",
        category="combat",
        tags=["gas spore", "explosion", "blast", "mindless", "hazard"],
        rule=(
            "Gas spores detonate for 4d6 damage (4 to 24 HP) in a 3x3 radius upon destruction. "
            "They are mindless and ignore Elbereth. Policies must NEVER attack in melee or throw missiles "
            "at adjacent gas spores (adjacent_gas_spore). Strictly yield step_away_from_hostile() to reach "
            "distance >= 2 before any ranged missile or wand elimination."
        ),
        anti_pattern=(
            "Throwing daggers or striking an adjacent gas spore, triggering an instant 4d6 explosion "
            "that vaporizes heroes with up to 37 HP."
        ),
        code_snippet="""
if obs.combat.adjacent_gas_spore:
    obs = (yield step_away_from_hostile())
    continue
""",
        related_ids=["INV-CBT-001", "INV-CBT-004"],
    ),
    Invariant(
        id="INV-CBT-003",
        title="Dust Elbereth Sanctuary & Species Discrimination",
        category="combat",
        tags=["elbereth", "sanctuary", "ward", "ignores_elbereth", "insects"],
        rule=(
            "Writing 'Elbereth' in dust creates a 100% ward against animals, insects, ants, bees, and spiders. "
            "Hostiles with IGNORES_ELBERETH_SPECIES (orcs, elves, humans, goblins, dwarves, giants, zombies) "
            "ignore Elbereth. When facing immune hostiles on Elbereth, strike them in melee or retreat to a corridor. "
            "Standing on Elbereth coordinates must be tracked before command execution to prevent re-engraving wipes."
        ),
        anti_pattern=(
            "Idling passively on Elbereth while an orc, dwarf, or elf attacks with ranged or melee weapons."
        ),
        code_snippet="""
if not obs.combat.standing_on_elbereth:
    if obs.hero.hp_frac < 0.4 or obs.combat.hostile_count_fov >= 2 or obs.combat.is_fast_dangerous:
        if not (obs.combat.adjacent_hostile and obs.combat.hostile_ignores_elbereth):
            obs = (yield engrave_dust_elbereth())
            continue

if obs.combat.standing_on_elbereth:
    if obs.combat.hostile_ignores_elbereth and obs.combat.adjacent_hostile:
        if obs.hero.hp_frac > 0.5 or not obs.combat.can_retreat:
            obs = (yield melee_attack_hostile())
            continue
        else:
            obs = (yield (step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile()))
            continue
""",
        related_ids=["INV-CBT-001", "INV-CBT-005"],
    ),
    Invariant(
        id="INV-CBT-004",
        title="Floating Eye Melee Prohibition & Stalemate Breaker",
        category="combat",
        tags=["floating eye", "paralysis", "stalemate", "ranged", "hazard"],
        rule=(
            "Hitting a floating eye in melee triggers a 50-turn passive paralysis and inevitable death. "
            "Throwing missiles (daggers, darts, arrows, rocks) at adjacent floating eyes (dist 1) is 100% safe "
            "and does not trigger paralysis. When cornered in a dead end next to an immobile floating eye with 0 missiles, "
            "the adapter's stalemate breaker activates after 2 waits (lateral step, search, or intentional strike before starvation)."
        ),
        anti_pattern=(
            "Striking a floating eye in melee with a sword, or waiting 20,000 turns in a dead end until starvation."
        ),
        code_snippet="""
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
""",
        related_ids=["INV-CBT-002", "INV-CBT-006"],
    ),
    Invariant(
        id="INV-CBT-005",
        title="High-Speed Predator & Swarm Corridor Defense",
        category="combat",
        tags=["fast_dangerous", "speed", "ants", "bees", "bats", "chokepoint"],
        rule=(
            "Fast predators (giant bats speed 22, soldier ants speed 18, killer bees speed 18, foxes speed 15) "
            "outpace hero speed (12). In open rooms, engaging swarms yields fatal poison stings or multi-attack surrounds. "
            "If is_fast_dangerous and surrounded or hostile_count >= 2, engrave Elbereth immediately. Insects flee, "
            "allowing retreat to 1-tile corridor chokepoints (step_to_chokepoint) to fight 1v1."
        ),
        anti_pattern=(
            "Trying to run away across an open room from a speed-18 ant, dealing 0 damage while taking 2 hits per turn."
        ),
        code_snippet="""
if obs.combat.is_fast_dangerous:
    if (obs.combat.is_surrounded or obs.combat.hostile_count_fov >= 2) and not obs.combat.standing_on_elbereth:
        obs = (yield engrave_dust_elbereth())
        continue
    if obs.combat.adjacent_hostile:
        if obs.hero.hp_frac > 0.40 or not obs.combat.can_retreat:
            obs = (yield melee_attack_hostile())
            continue
        else:
            if not obs.combat.standing_on_elbereth:
                obs = (yield engrave_dust_elbereth())
            else:
                obs = (yield melee_attack_hostile())
            continue
""",
        related_ids=["INV-CBT-001", "INV-CBT-003"],
    ),
    Invariant(
        id="INV-CBT-006",
        title="Ranged Missile Harassment at Distance >= 2",
        category="combat",
        tags=["ranged", "throw_dagger", "zap_wand", "harassment"],
        rule=(
            "Softening approaching enemies from distance >= 2 using thrown daggers/darts/arrows or offensive wands "
            "(magic missile, striking, fire, cold, lightning, sleep) eliminates pests before they close to melee. "
            "Missiles must be shielded against targeting peaceful NPCs."
        ),
        anti_pattern=(
            "Allowing high-speed predators to close into melee range without throwing available missiles."
        ),
        code_snippet="""
if obs.combat.closest_hostile_dist >= 2:
    if obs.inventory.has_offensive_wand:
        obs = (yield zap_offensive_wand())
        continue
    elif obs.inventory.has_daggers:
        obs = (yield throw_dagger())
        continue
""",
        related_ids=["INV-CBT-005", "INV-EQP-004"],
    ),
    Invariant(
        id="INV-CBT-007",
        title="In-Combat Emergency Healing & Panic Teleport Escapes",
        category="combat",
        tags=["healing", "quaff_healing", "teleport", "panic", "emergency"],
        rule=(
            "Quaffing healing potions (quaff_healing()) takes 1 turn and restores 10-20 HP; trigger at "
            "hp_frac < 0.40–0.50 unconditionally. Panic teleport escapes (reading scroll of teleport or "
            "zapping wand of teleport) must trigger when surrounded or at hp_frac < 0.20."
        ),
        anti_pattern=(
            "Gating healing potions on 'not adjacent_hostile', dying with 3 healing potions in backpack."
        ),
        code_snippet="""
if (obs.hero.hp_frac < 0.2 or obs.combat.is_surrounded) and obs.combat.has_panic_escape:
    if obs.inventory.has_scroll_of_teleport:
        obs = (yield read_scroll_teleport())
        continue
    elif obs.inventory.has_wand_of_teleport:
        obs = (yield zap_wand_teleport())
        continue

if obs.hero.hp_frac < 0.4 and obs.inventory.has_healing:
    obs = (yield quaff_healing())
    continue
""",
        related_ids=["INV-CBT-001"],
    ),
    Invariant(
        id="INV-CBT-008",
        title="Peaceful NPC Shield & Universal Attack Retaliation",
        category="combat",
        tags=["peaceful", "guard", "priest", "shopkeeper", "oracle", "retaliation"],
        rule=(
            "Vault Guards (guard), temple priests, shopkeepers, watchmen, and oracles are peaceful until provoked. "
            "Attacking them prompts 'Really attack? [yn]'. The adapter auto-answers 'n' and maintains persistent "
            "peaceful_positions and GLYPH_IS_PEACEFUL_SPECIES_LUT. Pathfinding routes around peaceful NPCs to avoid attack prompts. "
            "CRITICAL: When ANY monster attacks the hero ('hits!', 'bites!', 'scratches!'), universal attack retaliation "
            "immediately strips peaceful status from all adjacent tiles and marks them hostile. Policies must defend themselves "
            "and eliminate attackers in melee. Wild/feral animals are NOT blanket peaceful in the LUT, ensuring the hero defends "
            "against biting dogs and kittens. NEVER check obs.dungeon.in_shop to refuse fighting hostiles."
        ),
        anti_pattern=(
            "Treating wild/feral biting animals as unattackable peaceful NPCs, or firing ranged weapons at Vault Guards "
            "or temple priests, or retreating from mimics/orcs inside shops because obs.dungeon.in_shop is True."
        ),
        code_snippet="""
closest_name = obs.combat.closest_hostile_name.lower()
if closest_name in ('shopkeeper', 'watchman', 'watch captain', 'guard', 'priest', 'priestess', 'oracle') and not obs.combat.adjacent_hostile:
    obs = (yield (retreat() if obs.combat.can_retreat else step_away_from_hostile()))
    continue
""",
        related_ids=["INV-CBT-006", "INV-CBT-011"],
    ),
    Invariant(
        id="INV-CBT-011",
        title="Disguised Mimic & Unseen Hostile Melee Counter-Strike",
        category="combat",
        tags=["mimic", "unseen", "invisible", "search", "dead_end", "damage"],
        rule=(
            "Mimics disguise themselves as items, doors, or dungeon features, and invisible monsters cannot be "
            "directly rendered on the map. When attacked by an unseen monster or disguised mimic ('hits!', 'bites!'), "
            "or when taking unexpected damage during searching (obs.hero.hp < prev_hp), policies must IMMEDIATELY "
            "break out of search loops. The adapter detects attack messages and flags obs.combat.adjacent_hostile = True. "
            "Policies must immediately engage in melee counter-strikes (melee_attack_hostile()) to force the mimic to "
            "undisguise and destroy it rather than passively taking hits."
        ),
        anti_pattern=(
            "Executing a rigid 12-turn search loop without checking obs.hero.hp or obs.combat.adjacent_hostile, "
            "standing still while a disguised mimic or invisible monster beats the hero from 67 HP to death."
        ),
        code_snippet="""
# In skill_explore_and_dive: abort search immediately if taking damage or attacked
prev_hp = obs.hero.hp
for _ in range(12):
    if (
        obs.combat.hostile_count_fov > 0
        or obs.combat.adjacent_hostile
        or obs.hero.hp < prev_hp
        or obs.spatial.stairs_down_known
        or obs.spatial.has_unvisited_frontier
    ):
        return obs
    obs = (yield search())
""",
        related_ids=["INV-CBT-001", "INV-NAV-005"],
    ),
    Invariant(
        id="INV-CBT-012",
        title="Corrosive Hazard Melee Prohibition & Ranged Neutralization",
        category="combat",
        tags=["corrosive", "acid", "acid blob", "ochre jelly", "weapon erosion", "rust", "ranged"],
        rule=(
            "Melee attacks against acidic monsters (acid blobs, ochre jellies) instantly corrode weapons and armor and "
            "inflict passive acid splash damage on the hero. When obs.combat.is_corrosive_target is True, policies must "
            "NEVER melee attack. Instead, eliminate them with thrown missiles (throw_dagger()) or offensive wands "
            "(zap_offensive_wand()), or retreat (step_to_chokepoint() / step_away_from_hostile())."
        ),
        anti_pattern=(
            "Engaging acid blobs or ochre jellies in melee, reducing weapon enchantment to -3 and dissolving iron armor."
        ),
        code_snippet="""
if obs.combat.is_corrosive_target:
    if obs.combat.closest_hostile_dist >= 2:
        if obs.inventory.has_offensive_wand:
            obs = (yield zap_offensive_wand())
            continue
        elif obs.inventory.has_daggers:
            obs = (yield throw_dagger())
            continue
    if obs.combat.adjacent_hostile:
        obs = (yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile())
        continue
""",
        related_ids=["INV-CBT-002", "INV-CBT-004", "INV-CBT-009"],
    ),
    Invariant(
        id="INV-CBT-013",
        title="Heavy Weapon Threat Gating & Chokepoint Defense",
        category="combat",
        tags=["heavy weapon", "two-handed", "battle-axe", "orc captain", "gnome", "crossbow", "ranged"],
        rule=(
            "Orc chieftains, captains, and gnomes wielding two-handed swords, battle-axes, or crossbows deal deadly burst "
            "damage capable of killing early heroes in 1-2 hits. When obs.combat.is_heavy_weapon_threat is True at distance "
            ">= 2, harass with thrown missiles or offensive wands, or retreat to 1-tile corridor chokepoints rather than rushing "
            "blindly into open-room melee."
        ),
        anti_pattern=(
            "Charging into open-room melee against an orc captain wielding a two-handed sword, taking 25+ damage in a single round."
        ),
        code_snippet="""
if obs.combat.is_heavy_weapon_threat:
    if obs.combat.closest_hostile_dist >= 2:
        if obs.inventory.has_offensive_wand:
            obs = (yield zap_offensive_wand())
            continue
        elif obs.inventory.has_daggers:
            obs = (yield throw_dagger())
            continue
    if obs.combat.adjacent_hostile and obs.hero.hp_frac < 0.50 and obs.combat.can_retreat:
        obs = (yield step_to_chokepoint() if not obs.combat.in_corridor else step_away_from_hostile())
        continue
""",
        related_ids=["INV-CBT-001", "INV-CBT-010"],
    ),
    Invariant(
        id="INV-CBT-014",
        title="Emergency Unidentified Consumables Panic Consumption",
        category="combat",
        tags=["emergency", "potion", "scroll", "unidentified", "panic", "survival"],
        rule=(
            "When HP is critically low (hp_frac < 0.20), prayer is on cooldown or unavailable, and no identified healing potions "
            "remain, dying with unidentified potions and scrolls in inventory is a fatal failure mode. Unidentified potions "
            "(extra/full healing, speed, gain energy) and unidentified scrolls (teleportation, earth, scare monster) have high "
            "probabilities of saving the hero. Yield quaff_emergency_potion() or read_emergency_scroll()."
        ),
        anti_pattern=(
            "Dying to an adjacent monster while carrying 3 unidentified potions and 2 unidentified scrolls without quaffing or reading."
        ),
        code_snippet="""
if obs.hero.hp_frac < 0.20 and not obs.inventory.has_healing and not obs.combat.has_panic_escape:
    if obs.inventory.has_unidentified_potion:
        obs = (yield quaff_emergency_potion())
        continue
    elif obs.inventory.has_unidentified_scroll:
        obs = (yield read_emergency_scroll())
        continue
""",
        related_ids=["INV-CBT-001", "INV-NUT-004"],
    ),
    Invariant(
        id="INV-CBT-015",
        title="Status Hazard Neutralization (Yellow Light & Homunculus)",
        category="combat",
        tags=["status", "yellow light", "homunculus", "sleep", "blind", "stun", "ranged", "elbereth"],
        rule=(
            "Yellow lights explode on contact causing 30+ turns of blindness and stunning; homunculi possess a sleep bite "
            "that puts heroes to sleep for 10-30 turns, allowing adjacent monsters to kill them while helpless. When facing "
            "yellow lights or homunculi at distance >= 2, eliminate them with offensive wands or thrown missiles. If adjacent, "
            "engrave dust Elbereth immediately or eliminate before taking status effects."
        ),
        anti_pattern=(
            "Trading blows in melee with yellow lights or homunculi, resulting in lethal 30-turn sleep or blindness locks."
        ),
        code_snippet="""
# Priority status hazards: yellow light and homunculus
if "yellow light" in closest_name or "homunculus" in closest_name:
    if obs.combat.closest_hostile_dist >= 2:
        if obs.inventory.has_offensive_wand:
            obs = (yield zap_offensive_wand())
            continue
        elif obs.inventory.has_daggers:
            obs = (yield throw_dagger())
            continue
    if obs.combat.adjacent_hostile:
        if not obs.combat.standing_on_elbereth:
            obs = (yield engrave_dust_elbereth())
            continue
        else:
            obs = (yield melee_attack_hostile())
            continue
""",
        related_ids=["INV-CBT-001", "INV-CBT-002", "INV-CBT-004"],
    ),
    Invariant(
        id="INV-CBT-016",
        title="Decisive Melee Engagement Over Chokepoint Retreat for Adjacent Threats",
        category="combat",
        tags=["melee", "jackal", "coyote", "pack", "chokepoint", "retreat", "free attacks", "valkyrie"],
        rule=(
            "In NetHack, attempting to retreat (via step_to_chokepoint or step_away_from_hostile) while adjacent to "
            "monsters in open rooms grants them free pursuit attacks every turn while the hero deals 0 damage, resulting "
            "in deaths from full HP against trivial pests (jackals, coyotes, giant rats, foxes). When adjacent to any "
            "non-corrosive, non-passive hostile at hp_frac > 0.35, the hero MUST strike in melee (melee_attack_hostile). "
            "Valkyrie starting weapons (+1 long sword) kill early pests in 1-2 hits. Chokepoint retreats (step_to_chokepoint) "
            "are strictly restricted to when hostiles are at closest_hostile_dist >= 2 or when HP is critically low."
        ),
        anti_pattern=(
            "Yielding step_to_chokepoint() while adjacent to lone jackals or coyotes, allowing them to bite the hero to "
            "death across an open room without ever swinging the long sword."
        ),
        code_snippet="""
# Pack threat defense: retreat to chokepoints ONLY from distance >= 2
if obs.combat.is_pack_threat and not obs.combat.in_corridor:
    if not obs.combat.standing_on_elbereth and not obs.combat.hostile_ignores_elbereth:
        obs = (yield engrave_dust_elbereth())
        continue
    elif obs.combat.closest_hostile_dist >= 2 and obs.combat.can_retreat:
        obs = (yield step_to_chokepoint())
        continue
    elif obs.combat.adjacent_hostile:
        obs = (yield melee_attack_hostile())
        continue

# Decisive melee strike for adjacent hostiles when healthy
if obs.combat.adjacent_hostile and obs.hero.hp_frac > 0.35:
    obs = (yield melee_attack_hostile())
    continue
""",
        related_ids=["INV-CBT-001", "INV-CBT-005", "INV-CBT-013"],
    ),
    # -----------------------------------------------------------------
    # NUTRITION & DIVINE FAVOR
    # -----------------------------------------------------------------
    Invariant(
        id="INV-NUT-001",
        title="Fresh Safe Floor Corpse Prioritization",
        category="nutrition",
        tags=["corpse", "food", "nutrition", "rations", "eat_floor_corpse"],
        rule=(
            "In NetHack, packaged food rations NEVER ROT, while floor corpses rot in 50 turns. Fresh safe corpses "
            "must be eaten immediately right after combat when hunger_state >= 1 (Hungry). Carried rations are "
            "consumed only when no safe corpses are available, preserving non-perishable rations for deep levels."
        ),
        anti_pattern=(
            "Eating carried food rations first upon hunger, allowing dozens of fresh monster corpses to rot to waste "
            "and exhausting rations by turn 1,500."
        ),
        code_snippet="""
# Fresh floor corpses first, carried food second
if any(c.is_safe for c in obs.corpses) and obs.hero.hunger_state >= 1 and not obs.combat.adjacent_hostile:
    obs = (yield eat_floor_corpse())
    continue
if obs.hero.hunger_state >= 1 and obs.inventory.has_food:
    obs = (yield eat_carried_food())
    continue
""",
        related_ids=["INV-NUT-002", "INV-NUT-003"],
    ),
    Invariant(
        id="INV-NUT-002",
        title="50-Turn Floor Corpse Pruning & Rotten Corpse Shield",
        category="nutrition",
        tags=["rotten", "poison", "food poisoning", "lichen", "lizard"],
        rule=(
            "The adapter strictly prunes floor corpses with age >= 50. Carried food item queries (has_food, "
            "get_food_slot, food_count) strictly exclude corpses to prevent lethal food poisoning from rotten meat. "
            "Lichen corpses and lizard corpses NEVER rot in NetHack and are treated as safe food indefinitely."
        ),
        anti_pattern=(
            "Eating an aged carried corpse from the backpack, causing fatal food poisoning or illness."
        ),
        code_snippet="""
# Safe floor corpse harvesting inside run() or skill_combat():
if obs.combat.hostile_count_fov == 0:
    for corpse in obs.corpses:
        if corpse.is_safe:
            if (obs.hero.y, obs.hero.x) == (corpse.y, corpse.x):
                obs = (yield eat_floor_corpse())
                return obs
            else:
                obs = (yield step_to(corpse.y, corpse.x))
                return obs
""",
        related_ids=["INV-NUT-001"],
    ),
    Invariant(
        id="INV-NUT-003",
        title="In-Combat Hunger Resolution & Conscious Weakness Prayer",
        category="nutrition",
        tags=["faint", "weak", "prayer", "divine feeding", "hunger_state"],
        rule=(
            "In NetHack, HungerState enum values are SATIATED=0, NORMAL=1, HUNGRY=2, WEAK=3, FAINTING=4. "
            "Carried food should be eaten proactively at hunger_state >= 1 ('Hungry'). However, in NetHack 3.6 deities "
            "only grant divine feeding for major trouble, which requires u.uhs > WEAK, meaning hunger_state >= 4 "
            "('Fainting'). Praying at hunger_state == 3 ('Weak') or hunger_state == 2 ('Hungry') fails major trouble checks, "
            "wastes divine favor, and puts prayer on an 850-turn cooldown without feeding the hero. When carrying food, "
            "heroes must consume rations. Only when food is exhausted (not obs.inventory.has_food) AND the hero reaches "
            "hunger_state >= 4 ('Fainting') should divine prayer be invoked for divine feeding."
        ),
        anti_pattern=(
            "Praying for food when merely hunger_state == 3 ('Weak') or hunger_state == 2 ('Hungry'), wasting divine favor without receiving food."
        ),
        code_snippet="""
# In skill_combat() and run():
# 1. Proactive eating of packaged food
if obs.hero.hunger_state >= 1 and obs.inventory.has_food:
    obs = (yield eat_carried_food())
    continue

# 2. Major trouble divine feeding (strictly hunger_state >= 4 'Fainting')
if (obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 4 and not obs.inventory.has_food)) and (obs.hero.can_pray and obs.hero.turn - self.last_prayer_turn >= 850):
    self.last_prayer_turn = obs.hero.turn
    obs = (yield pray())
    continue
""",
        related_ids=["INV-NUT-001", "INV-NUT-004"],
    ),
    Invariant(
        id="INV-NUT-004",
        title="850-Turn Safe Divine Favor Prayer Threshold",
        category="nutrition",
        tags=["prayer", "divine favor", "cooldown", "tyr is displeased", "smite"],
        rule=(
            "In NetHack, successful prayer sets divine timeout to 300 + rn2(500) <= 799 turns. Praying at <= 800 turns "
            "angers the deity ('Tyr is displeased'), causing divine smiting or paralysis. Prayer must enforce "
            "turn - last_prayer_turn >= 850 turns. The adapter auto-answers 'n' to premature prayer confirmation "
            "prompts ('Are you sure you want to pray?'), completely eliminating divine wrath."
        ),
        anti_pattern=(
            "Praying with a 300 or 500-turn cooldown, triggering divine smiting in 18.6% of runs."
        ),
        code_snippet="""
# In run():
if obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 2 and not obs.inventory.has_food):
    if obs.hero.turn - self.last_prayer_turn >= 850:
        self.last_prayer_turn = obs.hero.turn
        obs = (yield pray())
        continue
""",
        related_ids=["INV-NUT-003", "INV-NUT-005"],
    ),
    Invariant(
        id="INV-NUT-005",
        title="Unconditional Emergency In-Combat Eating and Conscious Fainting Prayer",
        category="nutrition",
        tags=["fainting", "starvation", "prayer", "eat", "adjacent", "coma", "in-combat"],
        rule=(
            "In NetHack, transitioning to hunger_state >= 4 ('Fainting') triggers 30-turn unconscious blackouts where any "
            "adjacent hostile deals unretaliated free damage. Eating carried food takes 1 turn; praying for divine food "
            "takes 1 turn. Gating in-combat eating or prayer on 'not adjacent_hostile' traps heroes next to slow/immobile "
            "monsters (e.g. acid blobs, molds) into repeated fainting comas until starving to death with backpacks full "
            "of rations. When hunger_state >= 3 (Weak) or Fainting, the hero MUST consume carried food unconditionally "
            "(eat_carried_food()). If food is exhausted and hunger_state >= 4, the hero MUST pray unconditionally (pray()) "
            "while conscious before falling into a coma."
        ),
        anti_pattern=(
            "Refusing to eat or pray during combat because adjacent_hostile is True, falling unconscious into a 30-turn coma "
            "and dying of starvation with food rations in inventory."
        ),
        code_snippet="""
# In skill_combat():
# 1. Emergency eating: unconditional at Weak/Fainting
if obs.hero.hunger_state >= 2 and obs.inventory.has_food:
    if not obs.combat.adjacent_hostile or obs.combat.standing_on_elbereth or obs.hero.hunger_state >= 3:
        obs = (yield eat_carried_food())
        return obs

# 2. Emergency prayer: unconditional at Fainting when food exhausted
if (obs.hero.hp_frac < 0.15 or (obs.hero.hunger_state >= 4 and not obs.inventory.has_food)) and (obs.hero.can_pray and obs.hero.turn - self.last_prayer_turn >= 850):
    self.last_prayer_turn = obs.hero.turn
    obs = (yield pray())
    return obs
""",
        related_ids=["INV-NUT-001", "INV-NUT-003", "INV-NUT-004"],
    ),
    # -----------------------------------------------------------------
    # EQUIPMENT, LOOTING & ARTIFACTS
    # -----------------------------------------------------------------
    Invariant(
        id="INV-EQP-001",
        title="Armor Equipping with Failed Slot Tracking",
        category="equipment",
        tags=["wear_armor", "ac", "armor", "defense", "equipment"],
        rule=(
            "Unworn dropped armor is equipped during peaceful exploration (wear_armor()), driving AC toward "
            "negative values. Failed armor slots (e.g. helmets while wearing cursed headgear) are tracked in "
            "failed_wear_slots and excluded from has_unworn_armor to prevent redundant wear loops."
        ),
        anti_pattern=(
            "Failing to wear dropped shields or armor, leaving heroes with baseline AC 10 on Depth 5."
        ),
        code_snippet="""
if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
    obs = (yield wear_armor())
    continue
""",
        related_ids=["INV-EQP-002"],
    ),
    Invariant(
        id="INV-EQP-002",
        title="Autopickup Loot Ammunition Consistency",
        category="equipment",
        tags=["loot", "autopickup", "step_to_loot", "rocks", "pet"],
        rule=(
            "Floor loot scooping (step_to_loot) must be strictly restricted to types matching NetHack autopickup "
            "(options pickup_types:?!/%=[$). Non-autopicked items like rocks/gems (*) must be excluded from loot_chars. "
            "Reached tiles are stored in self.looted_tiles (cleared on floor change) to eliminate 2-tile pet-rock oscillation loops."
        ),
        anti_pattern=(
            "Including '*' (rocks) in loot_chars, causing the hero to step back and forth to rocks dropped by the pet dog for 1,500 turns."
        ),
        code_snippet="""
if obs.spatial.has_nearby_loot and obs.combat.hostile_count_fov == 0:
    obs = (yield step_to_loot())
    continue
""",
        related_ids=["INV-EQP-001", "INV-EQP-003"],
    ),
    Invariant(
        id="INV-EQP-003",
        title="Excalibur Artifact Forging & Fountain Navigation",
        category="equipment",
        tags=["excalibur", "fountain", "dip", "artifact", "valkyrie"],
        rule=(
            "Lawful Valkyries XL >= 5 dipping a long sword into a fountain have a 1/6 chance per dip of forging Excalibur "
            "(+1d10 damage, secret door searching, life drain immunity). Gating step_to_fountain() by "
            "(fountain_in_fov or adjacent_fountain) enables room-wide transit. Standing on fountain (@) hero occlusion "
            "must not clear known_fountain_pos. Confirmation prompt ('Dip into fountain?') is auto-answered 'y'."
        ),
        anti_pattern=(
            "Checking only adjacent_fountain, preventing the hero from crossing a room to reach a fountain."
        ),
        code_snippet="""
if obs.dungeon.can_forge_excalibur and obs.hero.hp_frac >= 0.85:
    if obs.dungeon.standing_on_fountain:
        obs = (yield dip_excalibur())
        continue
    elif obs.dungeon.fountain_in_fov or obs.dungeon.adjacent_fountain:
        obs = (yield step_to_fountain())
        continue
""",
        related_ids=["INV-EQP-001"],
    ),
    Invariant(
        id="INV-EQP-004",
        title="Universal Throwable Missile Recognition",
        category="equipment",
        tags=["missiles", "daggers", "darts", "arrows", "rocks", "ammo"],
        rule=(
            "Goblins and orcs drop dozens of darts, arrows, and rocks across early dungeon floors. "
            "has_daggers, get_dagger_slot, and dagger_count recognize all throwable missiles (daggers, darts, arrows, "
            "rocks, shuriken, spears, javelins). This ensures heroes never run out of ranged ammo for sniping passive hazards."
        ),
        anti_pattern=(
            "Restricting throwable weapons to daggers only, running out of ranged ammunition on Depth 2."
        ),
        code_snippet="""
# In CombatView / InventoryView:
# has_daggers includes daggers, darts, arrows, rocks, shuriken, spears, javelins
if obs.inventory.has_daggers and obs.combat.closest_hostile_dist >= 2:
    obs = (yield throw_dagger())
    continue
""",
        related_ids=["INV-CBT-006"],
    ),
    Invariant(
        id="INV-EQP-005",
        title="Weapon Skill Enhancement & Superior Body Armor Replacement",
        category="equipment",
        tags=["enhance", "skills", "damage", "armor", "mithril", "replace_body_armor"],
        rule=(
            "In NetHack, Valkyries start at Basic (+0 to-hit, +0 damage). Striking enemies 80 times qualifies the hero "
            "for Skilled (+2 to-hit, +1 damage); 180 hits for Expert (+3 to-hit, +2 damage). When obs.hero.can_enhance_skills "
            "is True, yield enhance_weapon_skill() outside combat to dramatically increase damage. Additionally, NetHack "
            "forbids wearing body armor while already wearing body armor; when an unworn body armor (e.g. dwarvish mithril coat "
            "AC 5) is superior to the worn leather jacket (AC 1), yield replace_body_armor() to take off the inferior jacket "
            "and equip the superior armor, dropping AC from 6 down to 2 or 1."
        ),
        anti_pattern=(
            "Never enhancing weapon skills, remaining permanently at Basic skill with 35% miss rates, and leaving dwarvish "
            "mithril coats unequipped in inventory while wearing the starting leather jacket."
        ),
        code_snippet="""
# In run():
if obs.hero.can_enhance_skills:
    obs = (yield enhance_weapon_skill())
    continue
if obs.inventory.get_superior_body_armor_slot() is not None:
    obs = (yield replace_body_armor())
    continue
""",
        related_ids=["INV-EQP-001", "INV-CBT-001"],
    ),
    Invariant(
        id="INV-CBT-010",
        title="Multi-Monster Pack Threat & Rothe Herd Corridor Defense",
        category="combat",
        tags=["pack", "herd", "rothe", "swarm", "surrounded", "chokepoint", "elbereth"],
        rule=(
            "On DL 4-6, monsters begin spawning in herds and squads (e.g. rothes in packs of 3-5, orc squads, ant swarms). "
            "Each rothe executes 3 attacks per turn (bite, butt, kick) dealing up to 25+ damage per turn against unarmored heroes. "
            "Rothes fully respect Elbereth! When obs.combat.is_pack_threat is True in open rooms (not in_corridor), policies must "
            "NEVER stand and trade melee blows. They must immediately engrave dust Elbereth (engrave_dust_elbereth()) or retreat to "
            "a 1-tile corridor chokepoint (step_to_chokepoint()) to engage enemies 1v1."
        ),
        anti_pattern=(
            "Treating rothe herds like isolated jackals and fighting in the open, allowing 3 rothes to deal 9 unretaliated attacks per turn."
        ),
        code_snippet="""
# In skill_combat():
if obs.combat.is_pack_threat and not obs.combat.in_corridor and not obs.combat.standing_on_elbereth:
    if not (obs.combat.adjacent_hostile and obs.combat.hostile_ignores_elbereth):
        obs = (yield engrave_dust_elbereth())
        return obs
    elif obs.combat.can_retreat:
        obs = (yield step_to_chokepoint())
        return obs
""",
        related_ids=["INV-CBT-003", "INV-CBT-005"],
    ),
    # -----------------------------------------------------------------
    # HARNESS & DIALOG INTERCEPTION
    # -----------------------------------------------------------------
    Invariant(
        id="INV-HAR-001",
        title="Text-Entry Dialog Interception (ESC \\x1b)",
        category="dialog_harness",
        tags=["esc", "minetown", "dialog", "call a", "name"],
        rule=(
            "In Minetown (Depths 5–9) or when picking up items, prompts like 'who are you', 'call a ', or 'what do you want to call' "
            "intercept directional keys as text input (consuming 0 turns). NetHackAdapter._dismiss_more() immediately dismisses "
            "these with ESC (\\x1b), preventing 2,500-step zero-turn aborts."
        ),
        anti_pattern=(
            "Sending directional movement keys to a text-entry prompt, resulting in 2,500 consecutive 0-turn steps."
        ),
        code_snippet="""
# Harness invariant in _dismiss_more():
if any(p in text_low for p in ("who are you", "what is your name", "call a ", "call the ")):
    raw_obs, _, _, _ = self.env.step(self.char_to_act.get("\\x1b", 0))
""",
        related_ids=["INV-HAR-002", "INV-HAR-003"],
    ),
    Invariant(
        id="INV-HAR-002",
        title="Intermediate Multi-Key Keystroke Preservation",
        category="dialog_harness",
        tags=["intermediate", "multi-key", "wear_armor", "eat", "quaff"],
        rule=(
            "Multi-key commands (eat_carried_food [e, slot], quaff_healing [q, slot], wear_armor [W, slot]) "
            "open intermediate prompts ('What do you want to eat?'). _step_sequence flags is_intermediate = (idx < len(seq) - 1) "
            "so _dismiss_more does NOT press ESC during intermediate prompts, allowing subsequent keys to register cleanly."
        ),
        anti_pattern=(
            "Pressing ESC during an intermediate item selection prompt, aborting the command before the slot key is sent."
        ),
        code_snippet="""
# Harness invariant in _step_sequence():
is_intermediate = (idx < len(actions_seq) - 1)
obs, r, term, trunc, info = self._step_with_dismissal(act, is_intermediate=is_intermediate)
""",
        related_ids=["INV-HAR-001"],
    ),
    Invariant(
        id="INV-HAR-003",
        title="Universal Consecutive Zero-Turn Circuit Breaker",
        category="dialog_harness",
        tags=["zero-turn", "abort", "wait", "circuit breaker"],
        rule=(
            "If any sequence of commands generates consecutive_zero_turns >= 4, NetHackAdapter.step() unconditionally "
            "forces Action(name='wait') ('.'), advancing the NetHack turn clock and completely eliminating zero-progress aborts."
        ),
        anti_pattern=(
            "Allowing an unexpected dialog or prompt to spin for thousands of steps without advancing the environment clock."
        ),
        code_snippet="""
# Harness invariant in NetHackAdapter.step():
if self.consecutive_zero_turns >= 4:
    return self.step(Action(name="wait"))
""",
        related_ids=["INV-HAR-001"],
    ),
    Invariant(
        id="INV-HAR-004",
        title="Vectorized Glyph Lookup Tables (11x Acceleration)",
        category="dialog_harness",
        tags=["numba", "lut", "vectorized", "performance", "glyphs"],
        rule=(
            "Precomputed 6KB module-level boolean lookup tables (GLYPH_IS_MON_HOSTILE_LUT, GLYPH_IS_BODY_LUT, "
            "GLYPH_IS_PASSIVE_HAZARD_LUT, GLYPH_IS_FAST_LUT, GLYPH_IS_PEACEFUL_SPECIES_LUT) for all 5,976 glyphs and "
            "C-level byte message decoding accelerate turn execution to 826+ steps/s, finishing 6,000-turn episodes in 7 seconds."
        ),
        anti_pattern=(
            "Calling slow C-extension helpers (nh.glyph_is_monster, permonst) inside 21x79 Python loops during _extract_obs."
        ),
        code_snippet="""
# Precomputed module LUTs:
hostile_mask = GLYPH_IS_MON_HOSTILE_LUT[valid_sub]
body_mask = GLYPH_IS_BODY_LUT[valid_glyphs]
""",
        related_ids=["INV-HAR-003"],
    ),
    # -----------------------------------------------------------------
    # ARCHITECTURE & SYNTHESIS
    # -----------------------------------------------------------------
    Invariant(
        id="INV-ARC-001",
        title="Python Generator Paradigm & Subroutine Protocol",
        category="architecture",
        tags=["generator", "yield", "yield from", "subroutine", "policy"],
        rule=(
            "Policies are Python generator classes (Agent.run(obs)) yielding typed Action objects (obs = yield action). "
            "Subroutines MUST be called via obs = (yield from self.subroutine(obs)). Every subroutine MUST yield an "
            "action on every code branch before returning, OR caller must avoid 'continue' without yielding to prevent 100% CPU loops."
        ),
        anti_pattern=(
            "Writing 'return obs' in a subroutine without yielding an action, followed by caller 'continue', spinning CPU at 0-progress."
        ),
        code_snippet="""
class Agent:
    def run(self, obs):
        while True:
            goal = self.determine_goal(obs)
            if goal == "combat":
                obs = (yield from self.skill_combat(obs))
                continue
            obs = (yield from self.skill_explore_and_dive(obs))

    def skill_combat(self, obs):
        if obs.combat.adjacent_hostile:
            obs = (yield melee_attack_hostile())
            return obs
        obs = (yield wait())
        return obs
""",
        related_ids=["INV-ARC-002", "INV-ARC-003"],
    ),
    Invariant(
        id="INV-ARC-002",
        title="Four-Layer Anti-Loop Defense Architecture",
        category="architecture",
        tags=["anti-loop", "ast", "guard", "wall_clock", "timeout"],
        rule=(
            "Layer 1: AST LoopGuardTransformer instruments while/for loops with _guard.tick() (raises RuntimeError at >2,000 iter). "
            "Layer 2: PolicyRunner catches RuntimeError and falls back to Action(name='wait'). "
            "Layer 3: 90s episode wall-clock ceiling in worker process. "
            "Layer 4: 180s worker pool batch ceiling terminating zombie processes."
        ),
        anti_pattern=(
            "Unprotected while True loops that freeze worker processes during unhandled game states."
        ),
        code_snippet="""
# Four-Layer defense is built into PolicyRunner and run_synthesis worker pool.
""",
        related_ids=["INV-ARC-001"],
    ),
    Invariant(
        id="INV-ARC-003",
        title="AST Method-Level Replacement & LLM Synthesis",
        category="architecture",
        tags=["ast", "splicing", "synthesis", "methods", "tokens"],
        rule=(
            "AuthorAgent.splice_policy_methods() parses candidate code via Python AST, allowing LLMs to output only "
            "the specific modified method(s) (e.g. def skill_combat(...)). Splicing merges the updated method into "
            "class Agent while keeping working subroutines intact, reducing completion tokens by 4x-5x and latency to ~15s."
        ),
        anti_pattern=(
            "Generating entire 250-line policy files on every single generation, multiplying token costs and diff breakage."
        ),
        code_snippet="""
# AuthorAgent automatically applies AST splicing on partial method outputs:
new_policy_code = author.splice_policy_methods(base_code=current_policy, patch_code=candidate_code)
""",
        related_ids=["INV-ARC-001"],
    ),
    Invariant(
        id="INV-CBT-009",
        title="Passive Hazard Combat Gating & Exploration Break",
        category="combat",
        tags=[
            "passive",
            "hazard",
            "floating eye",
            "gas spore",
            "mold",
            "bypass",
            "break",
            "loop",
        ],
        rule=(
            "Immobile passive hazards (floating eyes, gas spores, brown molds) at distance >= 2 do NOT hunt or pursue "
            "the hero. If the hero is not adjacent to hostiles, has no active hostile chasing them (not obs.combat.has_active_hostile), "
            "and possesses no ranged weapons (daggers or offensive wands), skill_combat() MUST return control "
            "to run(). The spatial pathfinding engine (walkable_nav) automatically routes around passive hazards. Staying inside "
            "skill_combat() without this exit guard traps the hero in endless step_away_from_hostile() oscillations at distance 2-4, "
            "starving and dying prematurely without exploring."
        ),
        anti_pattern=(
            "Running `while obs.combat.hostile_count_fov > 0` without breaking when not adjacent and not has_active_hostile and "
            "no ranged weapons. This traps the agent in combat mode against distant immobile eyes and spores."
        ),
        code_snippet="""
# In skill_combat():
if not obs.combat.adjacent_hostile and not obs.combat.has_active_hostile:
    if not (obs.inventory.has_offensive_wand or obs.inventory.has_daggers):
        return obs

# In determine_goal():
# Immobile hazards at distance >= 2 without active hostility allow goal 'explore_and_dive'
""",
        related_ids=["INV-CBT-002", "INV-CBT-004", "INV-NAV-006"],
    ),
    Invariant(
        id="INV-HAR-005",
        title="Rolling Message History for Truthful Fatal Attacker Attribution",
        category="dialog_harness",
        tags=["telemetry", "killer", "mortality", "attribution", "autopsy"],
        rule=(
            "In NetHack, when a hero dies, the final observation message is frequently blank or 'You die...  --More--'. "
            "Naive parsing attributed deaths to whatever monster was nearest in FOV (frequently distant passive hazards like "
            "floating eyes or gas spores). Scanning recorder.turns[-6:] in reverse for genuine attack verbs ('bites!', 'hits!', "
            "'stings!') and prioritizing adjacent active attackers resolves true fatal killer attribution into DuckDB."
        ),
        anti_pattern=(
            "Assigning killer to the first monster found in raw glyphs or closest_hostile_name upon death, corrupting empirical telemetry."
        ),
        code_snippet="""
# In NetHackAdapter._resolve_killer():
for turn_record in reversed(recorder.turns[-6:]):
    msg = turn_record.message
    for verb, species in HIT_VERBS:
        if verb in msg:
            return species
""",
        related_ids=["INV-HAR-003", "INV-ARC-004"],
    ),
    Invariant(
        id="INV-ARC-004",
        title="Mandatory Post-Mortem DuckDB Autopsy & Iterative Evolution Protocol",
        category="architecture",
        tags=["autopsy", "duckdb", "campaign", "evolution", "protocol"],
        rule=(
            "At the conclusion of EVERY campaign run, agents must perform a comprehensive DuckDB autopsy across all evaluated "
            "episodes and ticks: (1) classify top fatality causes, killers, turns, and AC distributions; (2) diagnose and correct "
            "unwanted behaviors, stalls, or pathological loops in both the environment adapter/core harness and policy templates; "
            "(3) formulate and deploy targeted improvements toward the 7 canonical NetHack ascension milestones; (4) verify zero regressions "
            "against unit tests before launching the next campaign."
        ),
        anti_pattern=(
            "Launching consecutive synthesis campaigns without querying DuckDB telemetry or without fixing concrete empirical mortality causes."
        ),
        code_snippet="""
# Empirical Autopsy SQL:
SELECT death_reason, count(*), round(avg(depth), 2) AS avg_dl, max(depth) AS max_dl
FROM episodes
WHERE run_id = :run_id
GROUP BY 1 ORDER BY 2 DESC;
""",
        related_ids=["INV-ARC-003", "INV-HAR-005"],
    ),
    Invariant(
        id="INV-EQP-006",
        title="Semi-Permanent Athame & Burned Wand Engraving",
        category="equipment",
        tags=["engrave", "elbereth", "athame", "wand", "fire", "lightning", "digging"],
        rule=(
            "Writing Elbereth in dust with fingers has a small chance to smudge or wipe upon movement. Writing with "
            "an athame (gouges) or a wand of fire/lightning/digging (burns) creates a permanent or semi-permanent ward "
            "that NEVER smudges or degrades from movement, guaranteeing 100% persistent sanctuary against non-humanoids."
        ),
        anti_pattern=(
            "Engraving in dust when possessing an athame or charged wand of fire/lightning/digging in inventory."
        ),
        code_snippet="""
# Use burn wand or athame slot if available for permanent ward
burn_slot = obs.inventory.get_burn_wand_slot()
athame_slot = obs.inventory.get_athame_slot()
tool = burn_slot or athame_slot or "-"
obs = (yield engrave_dust_elbereth())
""",
        related_ids=["INV-CBT-001", "INV-EQP-003"],
    ),
    Invariant(
        id="INV-RIT-001",
        title="Co-Aligned Altar Sacrificing & Temple Protection Donation",
        category="equipment",
        tags=["altar", "sacrifice", "priest", "protection", "gold", "luck"],
        rule=(
            "Co-aligned altar sacrifices (#offer) reset prayer timeout to 0, increase Luck, and grant powerful divine artifact "
            "gifts. Donating 400 * XL gold to an aligned temple priest via #chat grants permanent intrinsic AC protection (+2 to +4 AC "
            "initially, +1 AC thereafter), driving hero AC toward negative values for deep-game survivability."
        ),
        anti_pattern=(
            "Leaving 2000+ gold unspent in inventory while AC is positive, or ignoring co-aligned altars while carrying safe fresh corpses."
        ),
        code_snippet="""
if obs.dungeon.standing_on_altar and obs.dungeon.can_sacrifice:
    obs = (yield sacrifice_on_altar())
    continue
if obs.dungeon.can_donate_to_priest:
    obs = (yield donate_to_priest())
    continue
""",
        related_ids=["INV-NUT-004", "INV-EQP-002"],
    ),
    Invariant(
        id="INV-EQP-007",
        title="Empty-Slot Priority Armor Equipping",
        category="equipment",
        tags=["armor", "equipment", "empty slot", "body", "helmet", "boots", "cloak", "gloves", "shield", "ac"],
        rule=(
            "Armor slots provide additive AC reductions. When unworn armor is picked up, prioritizing empty armor slots "
            "(Body > Helm > Boots > Cloak > Gloves > Shield) before attempting to replace existing worn armor rapidly reduces AC "
            "from 10 to negative values with zero turn waste."
        ),
        anti_pattern=(
            "Attempting to wear a cloak or helmet while already wearing one, causing failed wear attempts while the shield or "
            "boots slot sits empty."
        ),
        code_snippet="""
if obs.inventory.has_unworn_armor and obs.epistemic.can_safely_wear_armor:
    obs = (yield wear_armor())
    continue
""",
        related_ids=["INV-EQP-001", "INV-EQP-004", "INV-EQP-005"],
    ),
]


class InvariantRegistry:
    """Indexed empirical invariant registry with search, category filtering, and trigger-based retrieval."""

    def __init__(self, invariants: list[Invariant] | None = None):
        self._invariants = invariants if invariants is not None else INVARIANTS
        self._by_id = {inv.id: inv for inv in self._invariants}

    def __len__(self) -> int:
        """Returns total number of registered invariants."""
        return len(self._invariants)

    def get_all(self) -> list[Invariant]:
        """Returns all registered invariants."""
        return list(self._invariants)

    def get_by_id(self, inv_id: str) -> Invariant | None:
        """Retrieves an invariant by its canonical ID (e.g. 'INV-NAV-001')."""
        return self._by_id.get(inv_id)

    def get_by_category(self, category: str) -> list[Invariant]:
        """Filters invariants by category ('combat', 'navigation', 'nutrition', 'equipment', etc.)."""
        cat_low = category.lower().strip()
        return [inv for inv in self._invariants if inv.category.lower() == cat_low]

    def search(
        self, query: str, category: str | None = None, top_k: int = 5
    ) -> list[Invariant]:
        """
        Ranks invariants by relevance to a search query using token match scoring across
        title, tags, rule, and anti-pattern.
        """
        tokens = set(re.findall(r"\w+", query.lower()))
        if not tokens:
            candidates = (
                self.get_by_category(category) if category else self._invariants
            )
            return candidates[:top_k]

        candidates = self.get_by_category(category) if category else self._invariants
        scored: list[tuple[float, Invariant]] = []

        for inv in candidates:
            score = 0.0
            inv_text = f"{inv.title} {' '.join(inv.tags)} {inv.rule} {inv.anti_pattern}".lower()
            for token in tokens:
                if token in inv.tags:
                    score += 5.0
                if token in inv.title.lower():
                    score += 3.0
                if token in inv_text:
                    score += 1.0
            if score > 0:
                scored.append((score, inv))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [inv for _, inv in scored[:top_k]]

    def get_relevant_invariants_for_trigger(
        self, trigger_reason: str, killers: list[str] | None = None, top_k: int = 4
    ) -> list[Invariant]:
        """
        Automatically selects the most pertinent tactical invariants given an empirical
        synthesis trigger reason (e.g. 'cluster: Killed in combat: gas spore') and top killer list.
        """
        query_parts = [trigger_reason]
        if killers:
            query_parts.extend(killers)
        full_query = " ".join(query_parts)

        # Trigger-specific heuristics
        low_query = full_query.lower()
        priority_ids = []
        if "gas spore" in low_query or "explosion" in low_query:
            priority_ids.append("INV-CBT-002")
        if "floating eye" in low_query or "paralysis" in low_query:
            priority_ids.append("INV-CBT-004")
        if "starv" in low_query or "faint" in low_query or "hunger" in low_query:
            priority_ids.extend(["INV-NUT-003", "INV-NUT-001", "INV-NUT-004", "INV-NUT-005"])
        if "door" in low_query or "locked" in low_query:
            priority_ids.extend(["INV-NAV-003", "INV-NAV-004"])
        if "dead_end" in low_query or "search" in low_query:
            priority_ids.append("INV-NAV-005")
        if "rat" in low_query or "newt" in low_query or "jackal" in low_query or "coyote" in low_query or "retreat" in low_query:
            priority_ids.extend(["INV-CBT-001", "INV-CBT-016"])
        if "ant" in low_query or "bee" in low_query or "bat" in low_query:
            priority_ids.append("INV-CBT-005")
        if (
            "passive" in low_query
            or "hazard" in low_query
            or "eye" in low_query
            or "spore" in low_query
            or "stuck" in low_query
        ):
            priority_ids.append("INV-CBT-009")
        if "corrosive" in low_query or "acid" in low_query or "ochre" in low_query:
            priority_ids.append("INV-CBT-012")
        if "heavy weapon" in low_query or "battle-axe" in low_query or "two-handed" in low_query or "orc captain" in low_query:
            priority_ids.append("INV-CBT-013")
        if "consumable" in low_query or "potion" in low_query or "scroll" in low_query or "emergency" in low_query:
            priority_ids.append("INV-CBT-014")
        if "sokoban" in low_query:
            priority_ids.append("INV-NAV-007")
        if "armor" in low_query or "ac" in low_query or "equip" in low_query:
            priority_ids.append("INV-EQP-007")
        if "yellow light" in low_query or "homunculus" in low_query or "sleep" in low_query:
            priority_ids.append("INV-CBT-015")
        if "zero-loot" in low_query or "early rush" in low_query or "pacing" in low_query:
            priority_ids.append("INV-NAV-008")
        if "starting room" in low_query or "stagnation decay" in low_query:
            priority_ids.append("INV-NAV-009")

        results: list[Invariant] = []
        for pid in priority_ids:
            inv = self.get_by_id(pid)
            if inv and inv not in results:
                results.append(inv)

        if len(results) < top_k:
            ranked = self.search(full_query, top_k=top_k)
            for inv in ranked:
                if inv not in results and len(results) < top_k:
                    results.append(inv)

        return results[:top_k]

    def format_llm_reference(self, invariants: list[Invariant]) -> str:
        """Formats a list of invariants into an authoritative guidance block for LLM prompts."""
        if not invariants:
            return ""
        blocks = [inv.format_prompt_block() for inv in invariants]
        header = "## Mandatory Empirical Tactical Invariants (Ground-Truth Rules derived from 48 Campaigns):\n"
        return header + "\n".join(blocks)


# Global singleton registry instance
REGISTRY = InvariantRegistry()
