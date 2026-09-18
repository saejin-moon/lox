"""
CORP-Ω Policy Layer: the versioned PolicyProgram container (R2).

The policy program is the single source of truth the executor compiles:
  strategy_plan (ordered goals) + params overlay + macros (R3) + profiles (R4) + nogoods.
Stored at data/policy_program.json; every accepted revision bumps `version` and archives
the previous file (R3). The default program reproduces the pre-R2 MacroAscensionDirector
phase machine exactly.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

DEFAULT_PROGRAM_PATH = "data/policy_program.json"


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
        "goal": "goto_minetown",
        "when": "(true)",
        "until": "(or (ac_le 0) (steps_in_goal_ge 3000) (depth_ge 10))",
        "directive": "GOTO_MINETOWN",
    },
    {
        "goal": "descend",
        "when": "(true)",
        "until": "(false)",
        "directive": "DESCEND",
    },
]

DEFAULT_PROGRAM: dict = {
    "version": 1,
    "domain": "nethack",
    "params": {},
    "macros": [],
    "strategy_plan": DEFAULT_STRATEGY_PLAN,
    "provenance": {"author": "r2-default", "note": "equivalent to pre-R2 MacroAscensionDirector phase machine"},
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
            raw=g,
        ) for g in d.get("strategy_plan", [])]
        return cls(
            version=int(d.get("version", 1)),
            domain=d.get("domain", "nethack"),
            params=dict(d.get("params", {})),
            strategy_plan=plan,
            macros=list(d.get("macros", [])),
            provenance=dict(d.get("provenance", {})),
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
            "params": self.params,
            "macros": self.macros,
            "strategy_plan": [g.raw for g in self.strategy_plan],
            "provenance": self.provenance,
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
