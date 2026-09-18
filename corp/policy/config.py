"""
CORP-Ω Policy Layer: PolicyConfig — the typed surface the offline LLM tunes (R1).

Every strategy constant that used to live as a literal in the domain managers now lives here.
`PolicyConfig.defaults()` reproduces the pre-R1 behavior byte-for-byte; later phases (R2+) load
this from the policy program (`data/policy_program.json`) instead of defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

from corp.env.blstats import HungerState


@dataclass
class SurvivalPolicy:
    # Tactical HP resting (nav 1.04): rest in place when wounded, no hostiles nearby
    rest_below_frac: float = 0.60          # start resting below this HP fraction
    rest_until_frac: float = 0.85          # stop resting at this HP fraction
    rest_hostile_radius: int = 2           # abort rest if a hostile is within this Chebyshev distance
    # Rest-on-stairs interlock: WAIT on stairs instead of wandering off
    rest_stairs_lower_frac: float = 0.20   # below this → critical triage owns it, not rest
    rest_stairs_upper_frac: float = 0.45   # above this → can_descend owns it
    # Critical-HP universal retreat (combat): never trade blows below this fraction w/o healing
    critical_retreat_frac: float = 0.25
    # Emergency Elbereth engraving HP thresholds (role-scaled)
    elbereth_hp_frac: float = 0.60
    elbereth_fragile_hp_frac: float = 0.75
    # Escape grace: attack-adjacent peaceful animals only when damaged within N turns
    escape_grace_turns: int = 3


@dataclass
class DescentPolicy:
    # can_descend (guards)
    min_hp_frac: float = 0.45              # forbid descent below this HP fraction
    deep_min_hp_frac: float = 0.70         # ...at depth >= deep_min_depth without poison res
    deep_min_depth: int = 8
    mines_min_xl: int = 5                  # forbid deeper Mines descent below this XL
    mines_retreat_xl: int = 6              # nav 1.01: retreat from Mines below this XL w/o poison res
    # should_descend triggers (guards)
    should_descend_unvisited: int = 25     # explored-enough threshold
    should_descend_turns: int = 100        # lingered-long-enough threshold
    should_descend_xl: int = 2             # XL-gated fast descent
    should_descend_xl_hp_frac: float = 0.70
    should_descend_xl_turns: int = 35
    should_descend_any_hp_frac: float = 0.50
    # Mines general retreat (nav 1.015): no downstairs + lingered this long → ascend
    mines_linger_turns: int = 50
    mines_end_dlevel: int = 10             # Mine's End: no downstairs below this
    minetown_donations_done: int = 3       # donations complete → ascend
    # Sokoban
    sokoban_prize_floor: int = 4
    sokoban_exit_min_turns: int = 60
    sokoban_entry_depth_lo: int = 5
    sokoban_entry_depth_hi: int = 9
    sokoban_hunt_step_cap: int = 6000
    # Minetown routing
    minetown_depth_lo: int = 2
    minetown_depth_hi: int = 4


@dataclass
class SearchPolicy:
    # Adaptive secret-door search caps (nav §7 + dead-end/wall finders)
    burst_cap_deep: int = 20               # burst cap, no stairs, depth >= 4
    burst_cap_early: int = 15              # burst cap, no stairs, depth < 4
    hard_cap_deep: int = 25                # cumulative per-tile cap, no stairs, depth >= 4
    hard_cap_early: int = 20               # cumulative per-tile cap, no stairs, depth < 4
    hard_cap_stairs: int = 8               # cumulative per-tile cap, stairs known
    dead_end_none_deep: int = 20           # dead-end finder cap, no stairs, depth >= 4
    dead_end_none: int = 15                # dead-end finder cap, no stairs, depth < 4
    dead_end_stairs: int = 6               # dead-end finder cap, stairs known
    wall_none_deep: int = 20               # wall finder cap, no stairs, depth >= 4
    wall_none: int = 15                    # wall finder cap, no stairs, depth < 4
    wall_stairs: int = 5                   # wall finder cap, stairs known


@dataclass
class CombatPolicy:
    player_speed: int = 12
    fast_ratio: float = 1.3                # speed > ratio*player_speed → ranged-first engagement
    heavy_wounded_min_hp: int = 16         # heavy-hitter kiting gate: hp <= max(this, frac*max)
    heavy_wounded_frac: float = 0.65
    ranged_fail_streak: int = 3            # consecutive "Never mind." failures → cooldown
    ranged_fail_cooldown: int = 50         # turns ranged stays disabled after a storm
    fast_wait_streak: int = 6              # bounded WAIT vs hovering fast monsters
    elbereth_dust_turns: int = 8
    elbereth_permanent_turns: int = 12
    threat_damage_per_level: float = 2.5   # base_dmg = level * this


@dataclass
class NutritionPolicy:
    corpse_fresh_turns: int = 25           # witness-based freshness cutoff
    # Prayer cooldown model (guards.can_pray), turns
    pray_major_first: int = 101            # HP <= 5 or fainting, first prayer
    pray_major: int = 450                  # HP <= 5 or fainting, subsequent
    pray_routine_first: int = 301
    pray_routine: int = 850
    # Moderate emergency (inventory resource turn): quaff/consider at this HP
    moderate_emergency_hp_frac: float = 0.20
    moderate_emergency_min_hp: int = 5
    # Food pathing radii (nav): normal, hungry (level-wide), weak bridge-before-descend, conveyor
    food_radius: int = 8
    food_radius_hungry: int = 99
    food_bridge_radius: int = 10
    conveyor_radius: int = 15


@dataclass
class NavigationPolicy:
    # POI deferral on stairs (defer_for_poi)
    poi_defer_turns: int = 50
    # Optional-loot pathing radii
    container_radius: int = 10
    gold_radius: int = 12
    # Locked-door handling
    door_attempt_cap: int = 30
    # Water crossing: >= N adjacent '}' tiles → not a lone sink → levitation prep
    water_body_min_tiles: int = 2
    # Sokoban pushable-boulder pathing cost penalty
    boulder_push_cost: float = 30.0
    # Hazard costs
    trap_cost: float = 500.0
    monster_cost: float = 1000.0
    visit_cost_per: float = 2.0
    visit_cost_cap: float = 50.0


@dataclass
class StrategyPolicy:
    # MacroAscensionDirector phase transitions
    excalibur_xl: int = 5
    excalibur_timeout_steps: int = 2000
    early_exit_xl: int = 4
    early_exit_depth: int = 4
    poison_exit_depth: int = 6
    sokoban_exit_depth: int = 9
    minetown_ac_target: int = 0
    minetown_timeout_steps: int = 3000
    minetown_exit_depth: int = 10
    # Stair-farming deferral
    farming_unvisited: int = 15
    farming_turns_d1: int = 120
    farming_turns_d23: int = 150
    # Mines entry guard
    mines_enter_xl: int = 6
    mines_alt_xl: int = 5                  # with Excalibur: enter at this XL if healthy enough
    mines_alt_hp: int = 35
    # Emergency nutrition triggers (corp_agent Priority 0.0 gate)
    emergency_hunger: HungerState = HungerState.WEAK
    # Emergency triage thresholds (corp_agent Priority 0.5 gate)
    triage_hp_frac: float = 0.55
    triage_min_hp: int = 7
    triage_adj_hp_frac: float = 0.60
    triage_adj_min_hp: int = 8
    # Zero-turn anti-stall guard
    zero_turn_limit: int = 4
    # Excalibur fountain dipping
    dip_attempt_cap: int = 10


@dataclass
class PolicyConfig:
    survival: SurvivalPolicy = field(default_factory=SurvivalPolicy)
    descent: DescentPolicy = field(default_factory=DescentPolicy)
    search: SearchPolicy = field(default_factory=SearchPolicy)
    combat: CombatPolicy = field(default_factory=CombatPolicy)
    nutrition: NutritionPolicy = field(default_factory=NutritionPolicy)
    navigation: NavigationPolicy = field(default_factory=NavigationPolicy)
    strategy: StrategyPolicy = field(default_factory=StrategyPolicy)

    @classmethod
    def defaults(cls) -> "PolicyConfig":
        """Pre-R1 behavior, byte-for-byte."""
        return cls()

    def to_dict(self) -> dict:
        def conv(obj):
            if hasattr(obj, "__dataclass_fields__"):
                return {f.name: conv(getattr(obj, f.name)) for f in fields(obj)}
            if isinstance(obj, HungerState):
                return int(obj)
            return obj
        return conv(self)

    def copy(self) -> "PolicyConfig":
        import copy
        return copy.deepcopy(self)


_DEFAULTS: PolicyConfig | None = None


def default_config() -> PolicyConfig:
    """Module-level shared defaults (for static guards). Single instance, treated as immutable."""
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = PolicyConfig.defaults()
    return _DEFAULTS
