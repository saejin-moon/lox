"""
LOX-ψ Policy Layer (CORP): tactic-rule evaluator (R4, AGENT_PLAN §8).

Tactic rules replace the hardcoded monster-name-set decisions (LETHAL_POISON_NAMES /
HEAVY_HITTERS branches) in the combat manager. Rules evaluate in program order, first
match wins; the matched verb overrides the combat manager's default branch for that
target. Rules can only CHOOSE among the closed verb set — they can never disable a
safety interlock (floating-eye melee lockout, grid-bug diagonal tactics, cockatrice
no-touch, gas-spore no-melee, critical-HP universal retreat stay hardcoded, enforced
by NogoodStore + branch classification).

Match predicates live in the same closed vocabulary as strategy-plan conditions:
`(monster "ogre")`, `(item "throne")`, plus all manifest predicates (`hp_frac <= ...`,
`has_poison_res`, ...). Default interlock rules carry `"interlock": true` — the
validator rejects any diff that removes them (invariant #9b).
"""
from __future__ import annotations

from dataclasses import dataclass

from corp.policy.predicates import evaluate as eval_node, nethack_bindings
from corp.policy.predicates import parse as parse_condition

# The closed verb set (mirrors corp/policy/manifest.py VERBS; kept literal here so
# tactics.py stays import-light for the combat manager hot path).
TACTIC_VERBS = {
    "ranged_only", "ranged_then_kill", "retreat", "retreat_when_wounded",
    "avoid", "kite", "elbereth_first", "never_melee",
}

# ---------------------------------------------------------------------------
# Canonical name sets (R4: the tactic-decision data lives HERE, not in the combat
# manager, so the default program can ship them as rules without an import cycle).
# INSTAKILL_NAMES stays in combat_manager: it drives scan_monsters threat weighting,
# and its melee lockouts are NogoodStore interlocks, not tactic decisions.
# ---------------------------------------------------------------------------

# Lethal poison biters in NetHack 3.6.6 (instakill / fatal strength drain without
# poison resistance). Default response: elbereth-first kiting, never melee.
TACTIC_LETHAL_POISON_NAMES = (
    "giant spider",
    "killer bee",
    "queen bee",
    "soldier ant",
    "scorpion",
    "pit viper",
    "water moccasin",
    "snake",
)

# Heavy hitter monster classes/names that hit for 10-30+ HP per turn. Default
# response: retreat-when-wounded (wounded gate: hp <= max(16, 65% max_hp)).
TACTIC_HEAVY_HITTER_NAMES = (
    "ogre",
    "ogre lord",
    "ogre king",
    "gnome lord",
    "gnome king",
    "dwarf lord",
    "dwarf king",
    "soldier ant",
    "rothe",
    "mumak",
    "leocrotta",
    "stalker",
    "giant",
    "ettin",
    "minotaur",
)


@dataclass(slots=True)
class TacticMatch:
    verb: str
    note: str | None
    index: int
    interlock: bool = False


class TacticRuleEngine:
    """Compiles the program's tactic_rules once (episode-static) and evaluates them
    per turn against a combat context. Pure CPU, first match wins."""

    def __init__(self, rules: list[dict] | None = None):
        self._compiled: list[tuple[dict, object | None]] = []
        for i, r in enumerate(rules or []):
            verb = r.get("do", "")
            if verb not in TACTIC_VERBS:
                continue  # validator guarantees this; defensive skip
            node = None
            when = r.get("when")
            if when:
                try:
                    node = parse_condition(when)
                except ValueError:
                    node = None  # unparseable conditions can never match (validator also blocks)
            self._compiled.append((r, node))

    def __len__(self) -> int:
        return len(self._compiled)

    def evaluate(self, ctx: dict) -> TacticMatch | None:
        """Evaluates rules in order against the combat context. First match wins.
        Returns None when no rule matches (default combat branch applies)."""
        bindings = nethack_bindings(ctx)
        for i, (rule, node) in enumerate(self._compiled):
            if node is None:
                continue
            try:
                if not eval_node(node, bindings):
                    continue
            except (ValueError, KeyError, TypeError):
                continue  # evaluation error → rule never fires (validator shadow-tested)
            return TacticMatch(verb=rule["do"], note=rule.get("note"),
                               index=i, interlock=bool(rule.get("interlock")))
        return None


# ---------------------------------------------------------------------------
# Default interlock rules — byte-equivalent triggers to the pre-R4 hardcoded branches.
# Parity is enforced by tests/test_policy_tactics.py against the combat manager's
# legacy name sets (drift guard, MACRO.md §9).
# ---------------------------------------------------------------------------

def build_default_tactic_rules(
    lethal_poison_names=TACTIC_LETHAL_POISON_NAMES,
    heavy_hitter_names=TACTIC_HEAVY_HITTER_NAMES,
) -> list[dict]:
    """Builds the default program's tactic rules from the combat manager's canonical
    name sets. One rule per name (substring matching); responses identical to the
    pre-R4 branches: lethal poison → elbereth-first kiting; heavy hitters →
    retreat-when-wounded. INSTAKILL_NAMES intentionally stay as scan_monsters threat
    data: their melee lockouts are NogoodStore interlocks, not tactic decisions."""
    rules: list[dict] = []
    for name in lethal_poison_names:
        rules.append({
            "when": f'(and (monster "{name}") (not has_poison_res))',
            "do": "elbereth_first",
            "note": f"interlock: lethal poison biter {name!r} — engrave/ward, never melee",
            "interlock": True,
        })
    for name in heavy_hitter_names:
        rules.append({
            "when": f'(monster "{name}")',
            "do": "retreat_when_wounded",
            "note": f"interlock: heavy hitter {name!r} — never trade blows when wounded",
            "interlock": True,
        })
    return rules