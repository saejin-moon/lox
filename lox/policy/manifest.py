"""
LOX-ψ Policy Layer (LOX-ψ): Vocabulary Manifest (R3, MACRO.md §4).

Machine-generated from code — never hand-written (drift-guard rule §9). The manifest is
both the validator's symbol table and the author prompt's vocabulary section: the LLM can
only reference what appears here, and the validator rejects anything outside it.

Param-leaf bounds are derived from PolicyConfig defaults (±50% for floats/ints, exact for
bools) with a small explicit-override table for fields whose useful range is asymmetric.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields

from lox.policy.config import PolicyConfig
from lox.env.blstats import HungerState

# ---------------------------------------------------------------------------
# Closed vocabulary tables (bound to code — MACRO.md §2.3 / §4.2)
# ---------------------------------------------------------------------------

# name: {"args": [(type, ...)], "desc": str}. Arity = len(args).
#   ("int",) / ("float",) / ("str",) / ("op",) bare; ("float", 0.0, 1.0) range-checked.
PREDICATE_SIGNATURES: dict[str, dict] = {
    # R2 strategy-plan predicates
    "true":            {"args": [], "desc": "always true"},
    "false":           {"args": [], "desc": "always false"},
    "xl_ge":           {"args": [("int",)], "desc": "experience level >= n"},
    "xl_le":           {"args": [("int",)], "desc": "experience level <= n"},
    "depth_ge":        {"args": [("int",)], "desc": "dungeon depth >= n"},
    "depth_le":        {"args": [("int",)], "desc": "dungeon depth <= n"},
    "depth_between":   {"args": [("int",), ("int",)], "desc": "depth in [lo, hi]"},
    "in_dungeons":     {"args": [], "desc": "currently in the Dungeons of Doom"},
    "in_mines":        {"args": [], "desc": "currently in the Gnomish Mines"},
    "in_sokoban":      {"args": [], "desc": "currently in Sokoban"},
    "lawful":          {"args": [], "desc": "lawful alignment"},
    "has_poison_res":  {"args": [], "desc": "poison resistance obtained"},
    "has_reflection":  {"args": [], "desc": "reflection obtained"},
    "has_excalibur":   {"args": [], "desc": "Excalibur forged"},
    "sokoban_completed": {"args": [], "desc": "Sokoban prize collected"},
    "minetown_visited":  {"args": [], "desc": "Minetown reached"},
    "donations_ge":    {"args": [("int",)], "desc": "temple donations >= n"},
    "ac_le":           {"args": [("int",)], "desc": "armor class <= n"},
    "steps_in_goal_ge": {"args": [("int",)], "desc": "steps in current goal >= n"},
    "turn_ge":         {"args": [("int",)], "desc": "absolute turn >= t"},
    # R3 manifest predicates (MACRO.md §2.3)
    "hp_frac":         {"args": [("op",), ("float", 0.0, 1.0)], "desc": "hp/max_hp compared to v, e.g. (hp_frac <= 0.40)"},
    "hunger_ge":       {"args": [("level",)], "desc": "hunger at least level: satiated|normal|hungry|weak|fainting"},
    "adjacent_hostiles": {"args": [], "desc": "a hostile monster is adjacent"},
    "has_healing":     {"args": [], "desc": "carries usable healing"},
    "is_fighting":     {"args": [], "desc": "in active combat"},
    "encumbrance_le":  {"args": [("int",)], "desc": "encumbrance level <= n (0..4)"},
    "donations_lt":    {"args": [("int",)], "desc": "temple donations < n"},
    "minetown_known":  {"args": [], "desc": "Minetown discovered"},
    "stairs_known":    {"args": [], "desc": "stairs located on this floor"},
    "failures_in_10_episodes_ge": {"args": [("int",), ("str",)], "desc": "goal failed >= n times in last 10 episodes"},
    # R4 tactic-rule match predicates
    "monster":        {"args": [("str",)], "desc": "primary target monster name contains substring (case-insensitive)"},
    "item":           {"args": [("str",)], "desc": "target/hero tile item name contains substring (water/trap/altar/fountain/throne/sink)"},
    # R6 Ascension Knowledge Stack milestone predicates
    "has_mr":         {"args": [], "desc": "magic resistance (gray dragon gear / ring of magic resistance)"},
    "has_light":      {"args": [], "desc": "carries a light source (lamp/lantern/candles)"},
    "has_wishing_wand": {"args": [], "desc": "carries an identified wand of wishing"},
    "has_amulet":     {"args": [], "desc": "carries the Amulet of Yendor"},
    "has_candelabrum": {"args": [], "desc": "carries the Candelabrum of Invocation (implies Vlad defeated)"},
    "castle_done":    {"args": [], "desc": "Castle prize acquired (bag of holding / dragon scale mail)"},
    "vlad_done":      {"args": [], "desc": "Vlad defeated (Candelabrum obtained)"},
    "invocation_done": {"args": [], "desc": "Invocation performed at the Vibrating Square"},
    # Gen-3 Sensory & Epistemic Predicates
    "stairs_dist":    {"args": [("op",), ("int",)], "desc": "A* distance to stairs down compared to n, e.g. (stairs_dist <= 5)"},
    "unvisited_count": {"args": [("op",), ("int",)], "desc": "unvisited floor tiles remaining compared to n, e.g. (unvisited_count <= 10)"},
    "is_corridor":    {"args": [], "desc": "hero is currently standing in a corridor tile (#)"},
    "is_doorway":     {"args": [], "desc": "hero is currently standing in a doorway tile (+ or ')"},
    "count_hostiles": {"args": [("op",), ("int",)], "desc": "count of monsters in radius <= 3 compared to n, e.g. (count_hostiles == 0)"},
    "closest_threat": {"args": [("op",), ("float",)], "desc": "threat score of closest monster compared to v, e.g. (closest_threat >= 5.0)"},
    "closest_speed":  {"args": [("op",), ("int",)], "desc": "speed of closest monster compared to n (player base is 12), e.g. (closest_speed > 12)"},
    "has_ranged_target": {"args": [], "desc": "a hostile monster is in raycasted line-of-sight"},
    "is_monster_fleeing": {"args": [], "desc": "primary target monster is fleeing"},
    "carried_food_count": {"args": [("op",), ("int",)], "desc": "count of safe carried food rations compared to n, e.g. (carried_food_count == 0)"},
    "turns_since_pray": {"args": [("op",), ("int",)], "desc": "turns elapsed since last prayer compared to n, e.g. (turns_since_pray >= 400)"},
    "can_safely_pray": {"args": [], "desc": "prayer cooldown has safely elapsed without divine wrath"},
    "has_intrinsic":  {"args": [("str",)], "desc": "hero has intrinsic (poison_res, reflection, cold_res, stealth)"},
    "current_ac":     {"args": [("op",), ("int",)], "desc": "hero Armour Class compared to n (lower is better), e.g. (current_ac <= 0)"},
    "weapon_enchantment": {"args": [("op",), ("int",)], "desc": "current main weapon enchantment compared to n, e.g. (weapon_enchantment >= 2)"},
    "has_item":       {"args": [("str",), ("str",)], "desc": "inventory contains item of category and name, e.g. (has_item 'scroll' 'teleport')"},
    "hunger_level":   {"args": [("op",), ("level",)], "desc": "hunger level compared to level (satiated|normal|hungry|weak|fainting)"},
}

VERBS: dict[str, str] = {
    "ranged_only":        "never melee vs matched target; missiles/wands only",
    "ranged_then_kill":   "soften at range, then finish in melee",
    "retreat":            "max-separation escape step away from the target",
    "retreat_when_wounded": "disengage when HP below the wounded gate",
    "avoid":              "do not engage; route around",
    "kite":               "hit-and-run: strike then step away",
    "elbereth_first":     "engrave Elbereth before engaging",
    "never_melee":        "melee attacks forbidden vs matched target",
}

# Certified goal handlers (goal handlers enter the vocabulary by certification only — §10)
CERTIFIED_GOALS: dict[str, dict] = {
    "explore_floor":    {"params": []},
    "forge_excalibur":  {"params": ["strategy.excalibur_timeout_steps"]},
    "hunt_poison_res":  {"params": []},
    "enter_sokoban":    {"params": ["descent.sokoban_hunt_step_cap",
                                     "descent.sokoban_entry_depth_lo",
                                     "descent.sokoban_entry_depth_hi"]},
    "goto_minetown":    {"params": ["strategy.minetown_timeout_steps",
                                     "descent.minetown_donations_done"]},
    "descend":          {"params": ["descent.min_hp_frac", "descent.deep_min_hp_frac",
                                     "descent.deep_min_depth"]},
    # R6 Ascension Knowledge Stack (mortality-driven order; each certified before
    # the loop may re-prioritize around it — AGENT_PLAN R6 execution plan)
    "early_survival_stack": {"params": ["strategy.survival_ac_target",
                                          "strategy.survival_timeout_steps",
                                          "strategy.survival_exit_depth"]},
    "equip_upgrade":    {"params": ["strategy.equip_ac_target",
                                      "strategy.equip_timeout_steps",
                                      "strategy.equip_exit_depth"]},
    "survival_intrinsics": {"params": ["strategy.intrinsics_min_depth",
                                         "strategy.intrinsics_exit_depth",
                                         "strategy.intrinsics_timeout_steps"]},
    "enter_gehennom":   {"params": ["strategy.gehennom_min_depth"]},
    "castle_wishing":    {"params": ["strategy.castle_min_depth"]},
    "vlad_invocation":   {"params": ["strategy.vlad_min_depth"]},
    "ascension_run":     {"params": []},
}

# Param-leaf bounds overrides (asymmetric useful ranges; otherwise ±50% of default)
PARAM_BOUNDS_OVERRIDES: dict[str, tuple] = {
    "strategy.minetown_ac_target": (-10, 3),   # AC target: negative AC is better
    "descent.mines_retreat_xl": (1, 12),
    "descent.mines_enter_xl": (1, 12),
    "navigation.monster_cost": (100.0, 4000.0),
    "navigation.trap_cost": (50.0, 2000.0),
}

# Non-tunable leaves (typed non-primitives the diff DSL must never set)
NON_TUNABLE_FIELDS = {"emergency_hunger"}


# ---------------------------------------------------------------------------
# Param leaf extraction + bounds derivation
# ---------------------------------------------------------------------------

@dataclass
class ParamLeaf:
    path: str
    type: str           # "int" | "float" | "bool"
    default: int | float | bool
    min: int | float | bool
    max: int | float | bool

    def coerce(self, value) -> "int | float | bool | None":
        """Type-coerces value for this leaf; returns None on type mismatch (ERR_BOUNDS)."""
        try:
            if self.type == "bool":
                if isinstance(value, bool):
                    return value
                return None
            if self.type == "int":
                f = float(value)
                return int(round(f))
            if self.type == "float":
                return float(value)
        except (TypeError, ValueError):
            return None
        return None

    def within(self, value) -> bool:
        return self.min <= value <= self.max


def _bounds_for(path: str, default, typ: str):
    if path in PARAM_BOUNDS_OVERRIDES:
        return PARAM_BOUNDS_OVERRIDES[path]
    if typ == "bool":
        return (False, True)
    if isinstance(default, bool):
        return (False, True)
    if typ == "float":
        # 0..1 fractions (hp/rest/triage gates): ±50%, capped at the unit interval
        # (matches the MACRO.md §4.1 example: 0.60 default → [0.30, 0.90]).
        if default > 0 and path.rsplit(".", 1)[-1].endswith("_frac") and default <= 1.0:
            return (default / 2.0, round(min(1.0, default * 1.5), 4))
        if default > 0:
            return (default / 2.0, default * 2.0)
        if default < 0:
            return (default * 2.0, default / 2.0)
        return (-1.0, 1.0)
    # int
    if default > 0:
        return (max(0, default // 2), default * 2)
    if default < 0:
        return (default * 2, 0)
    return (0, 8)


@dataclass
class VocabularyManifest:
    domain: str
    version: int
    predicates: dict[str, dict]                 # name -> signature dict
    verbs: dict[str, str]                       # name -> description
    goals: dict[str, dict]                      # name -> {"params": [...]}
    param_leaves: dict[str, dict]               # dotted path -> {type,min,max,default}
    macros: dict[str, str] = field(default_factory=dict)   # name -> rendered body

    # -- lookups -------------------------------------------------------------
    def is_predicate(self, name: str) -> bool:
        return name in self.predicates

    def is_verb(self, name: str) -> bool:
        return name in self.verbs

    def is_goal(self, name: str) -> bool:
        return name in self.goals

    def leaf(self, path: str):
        return self.param_leaves.get(path)

    def goal_owned_params(self, goal: str) -> list[str]:
        return self.goals.get(goal, {}).get("params", [])

    # -- prompt rendering (§4.2 rule 5: the manifest is the prompt) -----------
    def render(self, compact: bool = False) -> str:
        """Renders the vocabulary. `compact=True` emits one line per symbol group with
        signatures only (no descriptions) — the token-efficient default for author
        sessions; full descriptions remain available via the read_manifest tool."""
        if compact:
            return self._render_compact()
        lines = [f"domain: {self.domain}", f"version: {self.version}", "predicates:"]
        for name, sig in sorted(self.predicates.items()):
            parts = [str(a[0]) if len(a) == 1 else "/".join(str(x) for x in a) for a in sig["args"]]
            args = ", ".join(parts)
            lines.append(f"  {name}: ({args})  # {sig['desc']}")
        lines.append("verbs:")
        for name, desc in self.verbs.items():
            lines.append(f"  {name}: {desc}")
        lines.append("goals:")
        for name, spec in self.goals.items():
            params = ", ".join(spec["params"]) if spec["params"] else "(none)"
            lines.append(f"  {name}: {{params: [{params}]}}")
        lines.append("param_leaves:  # path: type [min..max] (default)")
        for path, spec in sorted(self.param_leaves.items()):
            lines.append(f"  policy_params.{path}: {spec['type']} [{spec['min']}..{spec['max']}] default {spec['default']}")
        if self.macros:
            lines.append("macros:")
            for name, body in self.macros.items():
                lines.append(f"  {name}: {body}")
        else:
            lines.append("macros: (none live)")
        return "\n".join(lines)

    def _render_compact(self) -> str:
        def sig(name, spec):
            args = ",".join(str(a[0]) for a in spec["args"])
            return f"{name}({args})" if args else name

        def leaf(p, s):
            if s["type"] == "bool":
                return f"{p}={s['default']}(bool)"
            return f"{p}={s['default']}({s['type']} {s['min']}..{s['max']})"

        lines = [f"domain: {self.domain}",
                 f"program_version: {self.version}",
                 "predicates: " + " ".join(sig(n, s) for n, s in sorted(self.predicates.items())),
                 "verbs: " + " ".join(sorted(self.verbs)),
                 "goals: " + " ".join(
                     f"{n}[{','.join(s['params']) or '-'}]" for n, s in self.goals.items())]
        lines.append("params: " + " ".join(leaf(p, s) for p, s in sorted(self.param_leaves.items())))
        lines.append("macros: " + (", ".join(f"{n}={b}" for n, b in self.macros.items())
                                   if self.macros else "(none live)"))
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manifest generation from code
# ---------------------------------------------------------------------------

def param_leaves_from_config(cfg: PolicyConfig | None = None) -> dict[str, dict]:
    """Walks the PolicyConfig dataclass tree → dotted-path leaves with derived bounds."""
    if cfg is None:
        cfg = PolicyConfig.defaults()
    leaves: dict[str, dict] = {}
    for sec in fields(cfg):
        section = getattr(cfg, sec.name)
        if not hasattr(section, "__dataclass_fields__"):
            continue
        for f in fields(section):
            value = getattr(section, f.name)
            if f.name in NON_TUNABLE_FIELDS:
                continue
            if isinstance(value, bool):
                typ, mn, mx = "bool", False, True
            elif isinstance(value, int):
                typ = "int"
                mn, mx = _bounds_for(f"{sec.name}.{f.name}", value, "int")
            elif isinstance(value, float):
                typ = "float"
                mn, mx = _bounds_for(f"{sec.name}.{f.name}", value, "float")
            else:
                continue  # HungerState and other non-primitives: not tunable via diff
            leaves[f"{sec.name}.{f.name}"] = {"type": typ, "min": mn, "max": mx, "default": value}
    return leaves


def build_manifest(program_version: int, domain: str = "nethack",
                   live_macros: dict[str, str] | None = None,
                   cfg: PolicyConfig | None = None) -> "VocabularyManifest":
    if domain != "nethack":
        try:
            from lox.executor.interface import get_adapter  # noqa: PLC0415
            adapter = get_adapter(domain)
            m = adapter.manifest(program_version)
            if live_macros:
                m.macros = dict(live_macros)
            return m
        except Exception:
            pass
    return VocabularyManifest(
        domain=domain,
        version=program_version,
        predicates=PREDICATE_SIGNATURES,
        verbs=dict(VERBS),
        goals=dict(CERTIFIED_GOALS),
        param_leaves=param_leaves_from_config(cfg),
        macros=dict(live_macros or {}),
    )