"""
LOX-ψ Policy Layer (LOX-ψ): the versioned PolicyProgram container (R2).

The policy program is the single source of truth the executor compiles:
  strategy_plan (ordered goals) + params overlay + macros (R3) + profiles (R4) + nogoods.
Stored at data/policy_program.json; every accepted revision bumps `version` and archives
the previous file (R3). The default program reproduces the pre-R2 MacroAscensionDirector
phase machine exactly.
"""
from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field

from lox.policy import tactics as tactic_mod

DEFAULT_PROGRAM_PATH = "data/policy_program.json"

# Default interlock tactic rules (R4): lethal poison → elbereth-first kiting, heavy
# hitters → retreat-when-wounded. Same behavior as the pre-R4 hardcoded branches, now
# visible and editable in the program (removal blocked by validator invariant #9b).
DEFAULT_TACTIC_RULES: list[dict] = tactic_mod.build_default_tactic_rules()


# ---------------------------------------------------------------------------
# Default strategy plan — byte-equivalent to the pre-R2 phase machine
# ---------------------------------------------------------------------------

DEFAULT_STRATEGY_PLAN: list[dict] = [
    {
        "goal": "explore_floor",
        "when": "(true)",
        "until": "(or (xl_ge 4) (depth_ge 4))",
    },
    {
        # R6 #1 (NEW, highest mortality impact): AC ladder + food security +
        # wand-wielder counterplay hold pattern at DL <= 6. Thresholds are
        # policy params (strategy.survival_*) — the literal string below mirrors
        # the defaults for inspection; the interpreter honors the params.
        "goal": "early_survival_stack",
        "when": "(true)",
        "until": "(or (ac_le 4) (depth_ge 7) (steps_in_goal_ge 1500))",
        "max_steps": 1500,
    },
    {
        "goal": "forge_excalibur",
        "when": "(and (lawful) (xl_ge 5) (not has_excalibur))",
        "until": "(has_excalibur)",
        "max_steps": 2000,
    },
    {
        "goal": "hunt_poison_res",
        "when": "(true)",
        "until": "(or (has_poison_res) (depth_ge 6))",
    },
    {
        "goal": "enter_sokoban",
        "when": "(true)",
        "until": "(or (has_reflection) (depth_ge 9) (sokoban_completed))",
        "directive": "ENTER_SOKOBAN",
    },
    {
        # R6 #2: weapon enchantment + armor ladder to AC <= -10 by DL 8.
        "goal": "equip_upgrade",
        "when": "(or (has_reflection) (depth_ge 6))",
        "until": "(or (ac_le -10) (depth_ge 8) (steps_in_goal_ge 2500))",
        "max_steps": 2500,
    },
    {
        "goal": "goto_minetown",
        "when": "(true)",
        "until": "(or (ac_le 0) (steps_in_goal_ge 3000) (depth_ge 10))",
        "directive": "GOTO_MINETOWN",
    },
    {
        # R6 #3: MR acquisition + drain counters — only relevant at DL 8-12,
        # which we must first reach reliably (activation gated by policy param).
        "goal": "survival_intrinsics",
        "when": "(depth_ge 8)",
        "until": "(or (has_mr) (depth_ge 13) (steps_in_goal_ge 3000))",
        "max_steps": 3000,
    },
    {
        "goal": "descend",
        "when": "(true)",
        "until": "(depth_ge 14)",
        "directive": "DESCEND",
    },
    {
        # R6 #4: gehennom survival (light logistics, maze mapping, undead tactics).
        "goal": "enter_gehennom",
        "when": "(and (in_dungeons) (depth_ge 14))",
        "until": "(or (depth_ge 22) (steps_in_goal_ge 5000))",
        "max_steps": 5000,
        "directive": "DESCEND",
    },
    {
        # R6 #5: Castle drawbridge + wand-of-wishing priority protocol.
        "goal": "castle_wishing",
        "when": "(depth_ge 22)",
        "until": "(or (castle_done) (depth_ge 30) (steps_in_goal_ge 6000))",
        "max_steps": 6000,
    },
    {
        # R6 #6: Vlad's Tower -> Candelabrum -> Invocation protocol.
        "goal": "vlad_invocation",
        "when": "(depth_ge 30)",
        "until": "(or (has_candelabrum) (depth_ge 38) (steps_in_goal_ge 6000))",
        "max_steps": 6000,
    },
    {
        # R6 #7: Wizard-hall discipline, Rider/Planes handling -> ascension.
        "goal": "ascension_run",
        "when": "(has_amulet)",
        "until": "(false)",
    },
]

DEFAULT_PROGRAM: dict = {
    "version": 1,
    "domain": "nethack",
    "params": {},
    "macros": [],
    "tactic_rules": DEFAULT_TACTIC_RULES,
    "nogoods": [],
    "strategy_plan": DEFAULT_STRATEGY_PLAN,
    "provenance": {"author": "r6-milestones", "note": "R2 default + R6 ascension stack (13 goals, mortality-driven order)"},
}


# ---------------------------------------------------------------------------
# Program container
# ---------------------------------------------------------------------------

@dataclass
class GoalSpec:
    goal: str
    when: str = "(true)"
    until: str = "(false)"
    directive: str | None = None
    max_steps: int | None = None
    raw: dict = field(default_factory=dict)


@dataclass
class PolicyProgram:
    version: int
    domain: str
    params: dict                                    # dotted-path overlay onto PolicyConfig
    strategy_plan: list[GoalSpec]
    macros: list[dict] = field(default_factory=list)
    tactic_rules: list[dict] = field(default_factory=list)   # R4: engine-evaluated (tactics.py)
    nogoods: list[dict] = field(default_factory=list)        # R3 fold-in (mask + declarative forms)
    role_profiles: dict = field(default_factory=dict)        # R4: role → {param-path: value}
    domain_profiles: dict = field(default_factory=dict)      # R4: domain → {param-path: value}
    provenance: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "PolicyProgram":
        plan = [GoalSpec(
            goal=g["goal"],
            when=g.get("when", "(true)"),
            until=g.get("until", "(false)"),
            directive=g.get("directive"),
            max_steps=g.get("max_steps"),
            raw=copy.deepcopy(g),
        ) for g in d.get("strategy_plan", [])]
        return cls(
            version=int(d.get("version", 1)),
            domain=d.get("domain", "nethack"),
            params=copy.deepcopy(d.get("params", {})),
            strategy_plan=plan,
            macros=copy.deepcopy(d.get("macros", [])),
            tactic_rules=copy.deepcopy(d.get("tactic_rules", [])),
            nogoods=copy.deepcopy(d.get("nogoods", [])),
            role_profiles=copy.deepcopy(d.get("role_profiles", {})),
            domain_profiles=copy.deepcopy(d.get("domain_profiles", {})),
            provenance=copy.deepcopy(d.get("provenance", {})),
        )

    @classmethod
    def load(cls, path: str = DEFAULT_PROGRAM_PATH) -> "PolicyProgram":
        """Loads the program from disk; falls back to the embedded default if absent/corrupt."""
        if not os.path.exists(path):
            return cls.from_dict(DEFAULT_PROGRAM)
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            prog = cls.from_dict(d)
            if not prog.strategy_plan:
                raise ValueError("empty strategy_plan")
            return prog
        except Exception as e:
            # A corrupt program must never crash the agent: fall back to defaults, loudly.
            print(f"[policy] WARNING: failed to load {path} ({e}); using default program")
            return cls.from_dict(DEFAULT_PROGRAM)

    def save(self, path: str = DEFAULT_PROGRAM_PATH) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "domain": self.domain,
            "params": copy.deepcopy(self.params),
            "macros": copy.deepcopy(self.macros),
            "tactic_rules": copy.deepcopy(self.tactic_rules),
            "nogoods": copy.deepcopy(self.nogoods),
            "role_profiles": copy.deepcopy(self.role_profiles),
            "domain_profiles": copy.deepcopy(self.domain_profiles),
            "strategy_plan": [copy.deepcopy(g.raw) for g in self.strategy_plan],
            "provenance": copy.deepcopy(self.provenance),
        }

    # ------------------------------------------------------------------
    # Param overlay: dotted paths onto PolicyConfig, type-coerced, unknown paths rejected
    # ------------------------------------------------------------------
    def apply_overlay(self, config):
        """Applies `params` overlay onto a PolicyConfig IN PLACE (R2: trust-the-author,
        bounds registry arrives with the R3 validator). Returns list of applied paths."""
        applied = []
        for dotted, value in self.params.items():
            parts = dotted.split(".")
            obj = config
            for p in parts[:-1]:
                obj = getattr(obj, p)          # unknown section → AttributeError → reject
            fld = getattr(type(obj), parts[-1], None)
            if fld is None:
                raise ValueError(f"ERR_UNKNOWN_SYMBOL: param path {dotted!r} not in PolicyConfig")
            cur = getattr(obj, parts[-1])
            try:
                if isinstance(cur, bool):
                    setattr(obj, parts[-1], bool(value))
                elif isinstance(cur, int):
                    setattr(obj, parts[-1], int(round(float(value))))
                elif isinstance(cur, float):
                    setattr(obj, parts[-1], float(value))
                else:
                    setattr(obj, parts[-1], value)
            except (TypeError, ValueError) as e:
                raise ValueError(f"ERR_SIGNATURE: param {dotted!r} type mismatch: {e}") from e
            applied.append(dotted)
        return applied
