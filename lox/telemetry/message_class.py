"""
LOX-ψ Telemetry: message-class classifier.

Per-turn NetHack messages are the *why* behind state changes, but they are not
persisted today. Persisting every raw message would be as voluminous as ticks, so
we persist CLASS TRANSITIONS: a compact, low-cardinality code (death/hunger/prayer/
damage/shop/trap/stair/status/levelup/item/monster) plus a short detail.

The classifier is pure and shared by the telemetry writer (episode_runner) and the
evidence reader (report digests), so what is written and what is displayed agree.
"""
from __future__ import annotations

# Ordered rules — first match wins. Death/hunger precede damage so a fatal blow is
# classified by its cause, not the attack verb.
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("death", ("you die", "you are dead", "killed by", "you are killed", "die...")),
    ("hunger", ("faint from lack", "you are starving", "you faint", "your stomach",
                "you feel weak", "you are hungry", "you feel hungry")),
    ("prayer", ("you pray", "deity", "thou art", "you feel a surge", "altar",
                "you feel the presence", "divine")),
    ("levelup", ("welcome to experience level", "you feel more experienced",
                 "you feel very experienced", "you advance to level")),
    ("shop", ("shop", "store", "unpaid", "shopkeeper", "you pay", "bill",
              "closed for inventory", "how dare you")),
    ("trap", ("trap", "you trigger", "you fall into", "a dart", "an arrow",
              "you are caught", "pit")),
    ("status", ("confus", "you are blind", "blind", "stun", "hallucin", "petrif",
                "you are turning to stone", "green slime", "you feel deathly sick",
                "terminally ill", "strangl")),
    ("stair", ("staircase", "stairs", "ladder", "you descend", "you climb")),
    ("damage", ("hits", "bites", "claws", "kicks", "stings", "strikes", "touches",
                "breathes", "you are hit", "you feel a bite", "zaps", "burns",
                "you get hurt", "damage")),
    ("item", ("you pick up", "you see here", "you drop", "you wield", "you wear",
              "you put on", "you are carrying", "you quaff", "you read")),
    ("monster", ("you see", "appears", "monster", "comes into view")),
)

# Classes worth persisting as evidence (everything else is dropped to bound volume).
SIGNIFICANT_CLASSES: frozenset[str] = frozenset(
    {"death", "hunger", "prayer", "levelup", "shop", "trap", "status", "stair",
     "damage", "item"})


def classify_message(message: str) -> str | None:
    """Maps a raw message to a significant class code, or None if not significant."""
    m = (message or "").strip().lower()
    if not m:
        return None
    for code, keywords in _RULES:
        if any(k in m for k in keywords):
            return code if code in SIGNIFICANT_CLASSES else None
    return None
