"""
LOX-ψ Policy Layer: behavior-primitive manifest (S1, AGENT_PLAN §1.2).

The plan vocabulary for the S3 handler ladder: every mechanism-level action the
dispatcher can execute, with MACHINE-READABLE preconditions and effects. A handler
plan is a sequence of these primitives — the LLM composes them, humans own the
mechanisms and interlocks (AGENTS.md §4).

Vocabulary split (the safety boundary):
  * PREDICATES / VERBS / GOALS  (manifest.py)  — the DIFF vocabulary (strategy).
  * PRIMITIVES                  (this module)  — the PLAN vocabulary (mechanism
    invocation). A primitive can never add a mechanism, only reference one that
    already exists in `lox/workers/dispatcher.py`; the interlock field records
    the human-owned guardrail the dispatcher enforces for that task.

`preconditions` entries are either:
  * an S-expression over the closed manifest predicate vocabulary, or
  * "mechanism:<note>" — a mechanical precondition the dispatcher checks (item in
    slot, adjacent door, …) that is deliberately NOT an LLM-visible strategy gate.

`DISPATCHER_TASKS` is the full set of task names `ActionDispatcher.create_fiber`
accepts; a test proves every task is covered by a primitive (as its task or alias).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BehaviorPrimitive:
    name: str                       # plan vocabulary name (handlers reference this)
    task: str                       # canonical dispatcher Task name
    args: tuple[str, ...]           # positional arg slots (documentation + handler schema)
    preconditions: tuple[str, ...]  # manifest predicate exprs and/or "mechanism:*"
    effects: str                    # declarative state change
    desc: str
    interlock: str | None = None    # AGENTS.md §4 guardrail the dispatcher enforces
    aliases: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# The primitive table (one canonical entry per dispatcher mechanism)
# ---------------------------------------------------------------------------

_PRIM_LIST: tuple[BehaviorPrimitive, ...] = (
    # -- movement / position ------------------------------------------------
    BehaviorPrimitive(
        "goto", "STEP", ("direction | tile_class",),
        ("(true)",),
        "hero position moves one tile toward the target",
        "single-step movement; the navigation manager/A* chooses the step direction",
        interlock="§4.2 non-pet monster tiles are unwalkable (A* +1000 cost), never bumped "
                  "intentionally; §4.15 diagonal door alignment excludes blocked/shop tiles",
        aliases=("MOVE",),
    ),
    BehaviorPrimitive(
        "jump", "JUMP", ("direction",),
        ("mechanism: an adjacent jumpable obstacle/dead-end",),
        "hero leaps over an adjacent tile",
        "jump (bounded leap; used to cross water/traps in place)",
        interlock="§4.2 never jump onto a non-pet monster tile",
    ),
    BehaviorPrimitive(
        "descend", "DESCEND", (),
        ("(stairs_known)",),
        "hero transfers to the next dungeon level (depth +1)",
        "take the downstairs",
        interlock="§4.4 multi-stairs steering: DL 2–4 under-leveled agents route to the "
                  "Main Dungeons staircase; §4.12 stairs_up anchored to the arrival tile",
    ),
    BehaviorPrimitive(
        "ascend", "ASCEND", (),
        ("(stairs_known)",),
        "hero transfers to the previous dungeon level (depth −1)",
        "take the upstairs",
        interlock="§4.4 trapped in the Mines under-leveled → ASCEND out; §4.9 Sokoban "
                  "ascend floors 1–3 → floor 4 prize",
    ),
    BehaviorPrimitive(
        "wait", "WAIT", (),
        ("(true)",),
        "one turn passes; timed effects (regen, cooldowns) advance",
        "pass a turn in place",
        interlock="§4.16 search-burst damage abort: HP drop aborts idle bursts",
    ),
    BehaviorPrimitive(
        "search", "SEARCH", (),
        ("(true)",),
        "adjacent walls/floor/traps are searched once",
        "one search pulse at the current position",
        interlock="§4.5 no secret-door search on corridor tiles; dead ends ≤ 6 searches, "
                  "perimeter walls ≤ 5; §4.16 search-burst aborts on HP drop/attack msgs",
    ),
    # -- items / inventory --------------------------------------------------
    BehaviorPrimitive(
        "pickup", "PICKUP", (),
        ("mechanism: at least one item lies on the floor tile underfoot",),
        "top floor item moves to the inventory",
        "pick up items underfoot",
    ),
    BehaviorPrimitive(
        "loot", "LOOT", ("direction",),
        ("mechanism: a container/chest is adjacent",),
        "container contents move to inventory (or a prompt opens)",
        "loot a container",
        interlock="§4.10 menu/pager dismissal guards on (X of Y) prompts",
        aliases=("LOOT_CONTAINER",),
    ),
    BehaviorPrimitive(
        "eat", "EAT", ("slot | floor",),
        ("mechanism: an edible item is selected (inventory slot or floor tile)",),
        "nutrition increases; a corpse may be rotten or grant an intrinsic",
        "eat food or a corpse",
        interlock="§4.1 corpse freshness: witnessed kills ≤ 25 turns; pre-existing level "
                  "corpses are rotten; never eat floor corpses with adjacent hostiles; "
                  "lichen never rots",
    ),
    BehaviorPrimitive(
        "quaff", "QUAFF", ("slot",),
        ("mechanism: a potion occupies the given inventory slot",),
        "potion effect applies; the potion is consumed",
        "quaff an identified or unidentified potion",
        interlock="§4.6 emergency health triage quaffs full/extra/healing at HP ≤ 55% "
                  "(60% with hostiles); §4.1 rotten-potion prompts guarded by AutoMore",
    ),
    BehaviorPrimitive(
        "read", "READ", ("slot",),
        ("mechanism: a scroll or spellbook occupies the given slot",),
        "scroll effect applies / spell learned; the item is consumed",
        "read a scroll or spellbook",
        interlock="§4.10 ESC guards on pager/menu prompts",
    ),
    BehaviorPrimitive(
        "wield", "WIELD", ("slot",),
        ("mechanism: a weapon/weapon-tool occupies the given slot",),
        "wielded weapon changes",
        "wield a weapon from inventory",
    ),
    BehaviorPrimitive(
        "wear", "WEAR", ("slot",),
        ("mechanism: wearable armor occupies the given slot",),
        "worn armor changes; AC may improve",
        "wear body armor",
    ),
    BehaviorPrimitive(
        "puton", "PUTON", ("slot",),
        ("mechanism: a ring/amulet occupies the given slot",),
        "accessory slot changes; intrinsics may apply",
        "put on a ring or amulet",
    ),
    BehaviorPrimitive(
        "takeoff", "TAKEOFF", ("slot",),
        ("mechanism: worn armor occupies the given slot",),
        "worn armor changes; AC may worsen",
        "take off worn armor",
        aliases=("REMOVE",),
    ),
    BehaviorPrimitive(
        "drop", "DROP", ("slot",),
        ("mechanism: an inventory item occupies the given slot",),
        "item moves to the floor tile underfoot",
        "drop an inventory item",
        interlock="§4.9 unpaid-item debt relief: never leave a shop carrying unpaid items",
    ),
    BehaviorPrimitive(
        "quiver", "QUIVER", ("slot",),
        ("mechanism: missiles occupy the given slot",),
        "quivered missile changes (used by fire)",
        "quiver missiles for ranged fire",
    ),
    # -- combat -------------------------------------------------------------
    BehaviorPrimitive(
        "attack", "MELEE_ATTACK", ("target | direction",),
        ("(adjacent_hostiles)",),
        "melee attack resolves against the adjacent monster",
        "deliberate melee attack (combat manager must have issued it)",
        interlock="§4.2 no intentional attack unless the combat manager issued it; "
                  "§4.7 floating-eye melee lockout; §4.8 grid-bug diagonal exploit; "
                  "§4.13 never trade blows when wounded vs heavy hitters",
        aliases=("FIGHT", "FORCE_FIGHT"),
    ),
    BehaviorPrimitive(
        "fire", "FIRE", ("target",),
        ("mechanism: missiles are quivered",),
        "ranged projectile attack resolves",
        "fire the quivered missile at a target",
    ),
    BehaviorPrimitive(
        "throw", "THROW", ("slot", "target | direction"),
        ("mechanism: a throwable item occupies the given slot",),
        "thrown item lands/attacks at the target",
        "throw an inventory item",
    ),
    BehaviorPrimitive(
        "zap", "ZAP", ("slot", "target | direction"),
        ("mechanism: a wand occupies the given slot",),
        "wand beam/effect resolves at the target",
        "zap a wand at a target or direction",
        interlock="§4.7 floating-eye threat: missiles/wands only; §4.9 wand-of-striking "
                  "drawbridge protocol; wand-of-wishing protocol",
        aliases=("ZAP_WAND",),
    ),
    BehaviorPrimitive(
        "cast", "CAST_SPELL", ("slot", "target | direction"),
        ("mechanism: a known spell and sufficient energy",),
        "spell effect applies; energy decreases",
        "cast a memorized spell",
        aliases=("CAST",),
    ),
    BehaviorPrimitive(
        "turn_undead", "TURN", (),
        ("mechanism: class can turn undead and energy is available",),
        "adjacent undead are turned/frightened",
        "use the turn-undead class ability",
        aliases=("TURN_UNDEAD",),
    ),
    BehaviorPrimitive(
        "engrave", "ENGRAVE", ("text",),
        ("(true)",),
        "an engraving is written on the floor tile (Elbereth, dust marks)",
        "engrave text/Elbereth (wand charges or fingers)",
        interlock="§4.7 Elbereth before floating-eye/poison engagement; §4.16 "
                  "hallucination freezes vectorized map updates",
        aliases=("ENGRAVE_DUST", "ENGRAVE_WAND"),
    ),
    # -- dungeon features / tools ------------------------------------------
    BehaviorPrimitive(
        "open", "OPEN", ("direction",),
        ("(true)",),
        "an adjacent door opens",
        "open an adjacent door",
        interlock="§4.3/§4.12 shop doors on Depth ≥ 2 are never kicked; unlock first",
    ),
    BehaviorPrimitive(
        "close", "CLOSE", ("direction",),
        ("(true)",),
        "an adjacent door closes",
        "close an adjacent door",
        aliases=("CLOSE_DOOR",),
    ),
    BehaviorPrimitive(
        "kick", "KICK", ("direction",),
        ("mechanism: an adjacent kickable tile (locked door, monster)",),
        "kick attempt resolves; a locked door may break",
        "kick an adjacent tile",
        interlock="§4.3 shop doors on Depth ≥ 2 are NEVER kicked (shopkeeper wand death)",
    ),
    BehaviorPrimitive(
        "force_lock", "FORCE_LOCK", ("direction",),
        ("mechanism: an adjacent locked door and a lock-picking tool",),
        "a lock is forced open (faster than applying the tool repeatedly)",
        "force a lock with a tool",
        interlock="§4.3/§4.12 never force a shop door on Depth ≥ 2",
        aliases=("FORCE",),
    ),
    BehaviorPrimitive(
        "apply", "APPLY", ("slot", "direction | target"),
        ("mechanism: a tool occupies the given slot",),
        "tool effect resolves (unlock, probe, …)",
        "apply a tool (lockpick, key, mirror, …)",
        interlock="§4.3/§4.12 unlock tools before routing around locked doors",
    ),
    BehaviorPrimitive(
        "dip", "DIP", ("slot",),
        ("mechanism: an inventory item occupies the given slot and a dungeon feature "
         "is underfoot/adjacent",),
        "item may be altered (Excalibur forge, potion dilution, …)",
        "dip an item into a dungeon feature",
        interlock="§4.9 Excalibur dipping requires lawful, XL ≥ 5, uncursed long sword, "
                  "and a fountain",
    ),
    BehaviorPrimitive(
        "untrap", "UNTRAP", ("direction",),
        ("mechanism: a known/adjacent trap and an untrapping tool",),
        "a trap is disarmed/removed",
        "untrap an adjacent trap",
    ),
    BehaviorPrimitive(
        "rub", "RUB", ("slot",),
        ("mechanism: a lamp/ring/artifact occupies the given slot",),
        "the rubbed item may activate (lamp, ring of levitation, …)",
        "rub an item",
    ),
    BehaviorPrimitive(
        "sit", "SIT", (),
        ("mechanism: a throne/altar/fountain is underfoot",),
        "feature-specific effect resolves (throne sit, …)",
        "sit on the feature underfoot",
    ),
    BehaviorPrimitive(
        "wipe", "WIPE", ("direction",),
        ("mechanism: an adjacent feature to wipe (e.g. a magic portal)",),
        "the adjacent feature is wiped/cleaned",
        "wipe an adjacent feature",
    ),
    BehaviorPrimitive(
        "altar_test", "ALTAR_TEST", (),
        ("mechanism: an altar is underfoot/adjacent",),
        "altar alignment/BUC information is revealed (epistemic probe)",
        "probe an altar (drop-test) for BUC/alignment",
        interlock="§4.9 altar BUC/EpistemicWorker identification protocol",
    ),
    BehaviorPrimitive(
        "offer", "OFFER", ("slot",),
        ("mechanism: an altar is underfoot/adjacent and a sacrifice is in the slot",),
        "sacrifice resolves (BUC/artifact/crowning effects)",
        "offer a sacrifice at an altar",
        interlock="§4.9 altar BUC/EpistemicWorker identification protocol",
    ),
    # -- social / menus -----------------------------------------------------
    BehaviorPrimitive(
        "pay", "PAY", ("amount",),
        ("mechanism: a shopkeeper is adjacent and gold is carried",),
        "shop debt is paid; unpaid items become owned",
        "pay a shopkeeper",
        interlock="§4.9 unpaid-item debt relief (PAY or drop before leaving the shop)",
    ),
    BehaviorPrimitive(
        "twoweapon", "TWOWEAPON", (),
        ("mechanism: two eligible weapons are wielded/available",),
        "two-weapon combat toggles on/off",
        "toggle two-weapon combat",
    ),
    BehaviorPrimitive(
        "chat", "CHAT", ("direction",),
        ("mechanism: a peaceful/tame creature is adjacent",),
        "the adjacent creature may reply (rumors, shop hints)",
        "chat with an adjacent creature",
    ),
    BehaviorPrimitive(
        "look", "LOOK_HERE", (),
        ("(true)",),
        "current-tile description is emitted to the message line",
        "inspect the tile underfoot",
    ),
    BehaviorPrimitive(
        "swap_weapon", "SWAP_WEAPON", ("slot",),
        ("mechanism: an alternate weapon occupies the given slot",),
        "the wielded weapon is swapped with the alternate",
        "swap to the alternate weapon (a/x)",
        aliases=("SWAP",),
    ),
    BehaviorPrimitive(
        "cancel", "ESC", (),
        ("(true)",),
        "the current prompt/menu is cancelled (pager dismissal)",
        "press ESC to dismiss a prompt/menu",
        interlock="§4.10 (X of Y)/(end) menus → SPACE/ESC via AutoMoreWrapper; "
                  "#enhance fibers terminate with ESC",
        aliases=("ESCAPE",),
    ),
    BehaviorPrimitive(
        "enhance", "ENHANCE", (),
        ("(true)",),
        "weapon/armor skill advances (menu terminated with ESC)",
        "train a weapon or armor skill",
        interlock="§4.10 #enhance fibers terminate with ESC (pager dismissal guards)",
    ),
    BehaviorPrimitive(
        "pray", "PRAY", (),
        ("(not (adjacent_hostiles))",),
        "deity intervention resolves (heal, nutrition reset, …)",
        "pray to the deity (safe-pray gate only)",
        interlock="§4.6 pray ≤ 25% HP when can_safely_pray; §4.14 pray at WEAK+ starvation "
                  "(no food) resets nutrition to 900",
    ),
)


PRIMITIVES: dict[str, BehaviorPrimitive] = {p.name: p for p in _PRIM_LIST}

# Every task name ActionDispatcher.create_fiber accepts (lox/workers/dispatcher.py).
DISPATCHER_TASKS: frozenset[str] = frozenset(
    p.task for p in _PRIM_LIST
) | frozenset(a for p in _PRIM_LIST for a in p.aliases)

# Non-primitive (composite/HTN/meta) task names — intentionally NOT plan vocabulary.
COMPOSITE_TASKS: frozenset[str] = frozenset({
    "SURVIVE_AND_ASCEND", "TACTICAL_COMBAT", "SPATIAL_NAVIGATION",
    "RESOURCE_MANAGEMENT", "EMERGENCY_NUTRITION", "EMERGENCY_TRIAGE",
    "SHOP_ACTION", "EPISTEMIC_ACTION", "SKILL_ENHANCEMENT", "MEDUSA_PREP",
})


# ---------------------------------------------------------------------------
# Rendering (author evidence — preconditions/effects are machine-readable)
# ---------------------------------------------------------------------------

def render_primitives() -> str:
    """Markdown table of the plan vocabulary for the author's env-schema evidence."""
    lines = ["## BEHAVIOR PRIMITIVES (plan vocabulary — S3 handler plans compose these)",
             "| primitive | task | args | preconditions | effects | interlock |",
             "|---|---|---|---|---|---|"]
    for p in PRIMITIVES.values():
        pre = "; ".join(p.preconditions)
        lines.append(
            f"| {p.name} | {p.task} | {', '.join(p.args) or '—'} | {pre} | "
            f"{p.effects} | {p.interlock or '—'} |")
    lines.append("")
    lines.append("aliases: " + "; ".join(
        f"{p.name}={','.join(p.aliases)}" for p in PRIMITIVES.values() if p.aliases))
    return "\n".join(lines)


def primitive_for_task(task: str) -> BehaviorPrimitive | None:
    """Reverse lookup: dispatcher task/alias → canonical primitive."""
    for p in PRIMITIVES.values():
        if p.task == task or task in p.aliases:
            return p
    return None