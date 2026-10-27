"""
Populates canonical NetHack early-game, exploration, and role strategy guides into data/wiki_index.db.
"""

from __future__ import annotations

import sqlite3

STRATEGY_ARTICLES = [
    (
        "Strategy: Early Game Survival & Progression Guide",
        """# NetHack Early Game Survival & Tactical Progression Guide

## 1. Golden Rules of Survival (Levels 1 to 5)
- **Do Not Rush**: NetHack is a turn-based tactical game. Never hold down movement keys. Check your HP on every turn.
- **HP Management**: When HP drops below 50%, immediately retreat, find a corridor or doorway, or engrave Elbereth. Never fight at low HP.
- **Armor Class (AC) is King**: In NetHack, lower/negative AC is better (AC 10 is naked; AC -10 is heavily protected). Wear your starting shield and armor on Turn 1 using `seq(Action.WEAR, slot)`.
- **Pet Management**: Your starting pet (dog or kitten) is your best early-game ally. Let your pet fight tough monsters (like grid bugs, jackals, sewer rats) while you stand behind it. Never hit your pet!
- **Elbereth (The Ultimate Escape Button)**: Engraving 'Elbereth' in the dust with your fingers (`seq(Action.ENGRAVE, '-', 'Elbereth\\r')`) prevents almost all non-human monsters from attacking you. Use it whenever cornered or below 50% HP.

## 2. Food, Nutrition, and Intrinsics
- **Hunger States**: 0=Satiated, 1=Not Hungry, 2=Hungry, 3=Weak, 4=Fainting. Eat food rations (%) or fresh corpses when state reaches 2 (Hungry) or 3 (Weak).
- **Poison Resistance**: Essential early intrinsic. Acquired by eating corpses of killer bees (q), poisonous snakes (s), or giant ants (a).
- **Never Eat**:
  - Chickatrice (c) corpses (causes instant petrification/stoning).
  - Kobold (k) corpses (poisonous unless you already have poison resistance).
  - Rotten corpses (tainted corpses cause fatal food poisoning).
- **Floating Eyes (e)**: NEVER attack blue floating eyes in melee! Hitting a floating eye paralyzes you for 50 turns while nearby monsters kill you. Kill floating eyes from range by throwing daggers (`seq(Action.THROW, slot, dir)`).

## 3. Early Branches and Goals
- **Minetown (Dlvl 2-4)**: Downstairs in Dungeons of Doom lead to Gnomish Mines. Minetown has a temple priest: donate gold (400 * XL) to buy permanent divine protection (lowers AC by 2-4 points!).
- **Sokoban (Dlvl 3-5)**: Complete the 4-level boulder puzzle for a guaranteed Bag of Holding (carrying capacity) or Amulet of Reflection (reflects breath attacks and death wands).""",
    ),
    (
        "Strategy: Dungeon Exploration, Doors, and Stairs",
        """# NetHack Dungeon Exploration, Door Breaching, and Stair Navigation Guide

## 1. Escaping Dungeon Level 1
- **The #1 Objective**: Find the down staircase ('>') and descend using `Action.DOWN` ('>'). Remaining on Dlvl 1 past 300 turns causes starvation and difficult monster spawns.
- **Enclosed Starting Rooms**: NetHack levels generate with rooms connected by corridors ('#') and closed doors ('+'). You cannot reach the rest of the floor without opening or breaching doors!

## 2. Navigating Doors & Corridors
- **Opening Doors**: When adjacent to a closed door ('+'), execute `seq(Action.OPEN, dir)`.
- **Locked Doors (The Kick Rule)**: If a door resists or is locked (\"The door resists!\"), immediately kick it down using `seq(Action.KICK, dir)`. A Valkyrie with 16+ Strength breaks open a locked door in 1-2 kicks.
- **Secret Doors & Corridors**: If a room appears to have no open exits, or a corridor terminates in a dead end, search adjacent wall segments 10-20 times (`Action.SEARCH`) to reveal secret doors.

## 3. Target Persistence & Avoiding Exploration Oscillations
- **The 2-Tile Ping-Pong Trap**: Never recompute `min(frontier_tiles, ...)` from scratch every single turn! Moving 1 step toward a northern frontier inverts Manhattan distance ties, causing the agent to turn around and step South on the next turn. This results in an infinite 2-tile oscillation (N, S, N, S) until max_steps!
- **Target Persistence Rule**: Maintain `self.target_tile`. Lock onto an unvisited frontier tile or door coordinate and follow the A* / BFS path step (`state.path_step`) until the target is reached, impassable, or after 25 turns.
- **Cycle Detection Ring Buffer**: Keep a history of the last 8 visited positions (`collections.deque(maxlen=8)`). If `state.pos in self.pos_history[:-1]`, you are trapped in a cyclic loop! Immediately discard `self.target_tile` and take a random cardinal step to break the attractor.""",
    ),
    (
        "Strategy: Valkyrie Ascension Playbook",
        """# NetHack Valkyrie Ascension & Early-Game Playbook

## 1. Character Strengths & Setup
- **Class**: Valkyrie (Lawful Human Female).
- **Starting Equipment**:
  - `a`: a +1 long sword
  - `b`: a +1 dagger
  - `c`: a +0 small shield (AC 9 -> AC 8)
  - `d`: a food ration
- **Turn 1 Optimization**:
  - Wear shield immediately: `seq(Action.WEAR, 'c')` to gain +1 AC.
  - Wield primary weapon: `seq(Action.WIELD, 'a')`.

## 2. Forging Excalibur (The Game-Winning Power Spike)
- **Requirement**: Lawful alignment, XL 5+, wielding a non-artifact long sword.
- **Procedure**: Stand on a fountain tile ('{') and execute `seq(Action.DIP, 'a', 'y')`.
- **Probability**: Each dip has a 1/6 chance to convert the long sword into the artifact blade **Excalibur**!
- **Benefits**:
  - +1d10 physical damage bonus against all monsters.
  - Grants intrinsic Searching (detects secret doors and traps automatically).
  - Grants Level Drain Resistance (essential against wights, wraiths, vampires).

## 3. Mid-Game Milestones
- **Acquire Reflection**: Wear shield of reflection, amulet of reflection (from Sokoban), or gray/silver dragon scale mail.
- **Acquire Magic Resistance**: Cloak of magic resistance or gray dragon scale mail (protects against touch of death and destroy armor).
- **Acquire Speed**: Speed boots or wand of speed.""",
    ),
    (
        "Strategy: Tactical Combat, Kiting, and Monster Threats",
        """# NetHack 3.6 Tactical Combat, Kiting, and High-Threat Monster Guide

## 1. Golden Rules of Tactical Combat
- **Chokepoint Advantage**: Never fight in open rooms when outnumbered. Step back into a doorway or corridor ('#') so monsters can only approach and attack one at a time.
- **HP Threshold Rule**: Retreat when HP drops below 50%. If HP < 30%, disengage immediately, seek upstairs or corridors, or engrave Elbereth.
- **Kiting Tough Enemies**: If an enemy is faster than you or hits hard, attack once and step backward toward visited territory to let your pet attack or force the enemy into a chokepoint.

## 2. High-Threat Monsters & Countermeasures
- **Floating Eyes ('e')**:
  - DANGER: Attacking a floating eye in melee paralyzes you for up to 50 turns while adjacent monsters finish you off.
  - RULE: NEVER attack 'e' in melee unless blindfolded. Kill it from range: throw daggers (`seq(Action.THROW, slot, dir)`), fire arrows, or zap wands.
- **Gas Spores ('e')**:
  - DANGER: Explodes on death in melee, dealing massive fire/explosion damage (often one-shotting early characters).
  - RULE: Disengage and kill with ranged weapons or let your pet handle it.
- **Soldier Ants ('a')**:
  - DANGER: Speed 18 (faster than player) with high-damage poisonous stings.
  - RULE: Flee to corridors, engrave Elbereth, or use poison resistance.
- **Cockatrice / Chickatrice ('c')**:
  - DANGER: Touching or hitting without gloves causes fatal stoning within 5 turns.
  - RULE: Never eat cockatrice corpses. Kill with ranged attacks or while wearing leather gloves.
- **Rust Monsters ('R')**:
  - DANGER: Hits corrode iron weapons and armor, degrading AC permanently.
  - RULE: Use non-metallic weapons (clubs, spears, artifact Excalibur which is rustproof) or throw daggers.

## 3. The Power of Elbereth in NetHack 3.6
- **Dust-Engraving**: Engraving 'Elbereth' in the dust with your fingers (`seq(Action.ENGRAVE, '-', 'Elbereth\\r')`) takes 1 turn.
- **Effect**: Most non-human monsters cannot attack the player while standing on the word Elbereth. Use it as an emergency lifesaver when cornered.""",
    ),
    (
        "Strategy: Nutrition, Safe Corpses, and Altars",
        """# NetHack 3.6 Nutrition, Corpse Consumption, and Divine Altar Guide

## 1. Hunger States & Nutrition Management
- **Hunger Levels**:
  - 0 = Satiated (do not eat)
  - 1 = Not Hungry (normal state)
  - 2 = Hungry (look for food or corpses)
  - 3 = Weak (eat immediately!)
  - 4 = Fainting / Fainted (starvation imminent)
- **Starting Food**: Valkyries start with food rations ('%'). Rations provide 800 nutrition points and never rot. Save them for emergencies when corpses are unavailable.

## 2. Corpse Consumption Rules
- **Eat Fresh Meat**: Eating fresh corpses dropped by slain monsters preserves rations. Always eat corpses immediately after killing (`seq(Action.EAT, 'y')`).
- **Safe Early Corpses**:
  - Newts, lichens, sewer rats, jackals, sewer rats, foxes, hobbits, dwarves.
- **Deadly Corpses (NEVER EAT)**:
  - **Kobolds ('k')**: Poisonous! Only safe if you have poison resistance.
  - **Chickatrice / Cockatrice ('c')**: Causes instant fatal stoning (petrification).
  - **Tainted / Rotten Corpses**: Corpses left on the ground for more than ~50 turns become tainted. Eating tainted food causes fatal food poisoning.

## 3. Altars & Divine Protection
- **Identifying Altars ('_')**:
  - Co-aligned altars (Lawful for Valkyrie) accept sacrifices.
  - `#offer` fresh corpses on co-aligned altars to gain divine favor, heal HP, and unlock divine gifts (artifacts like Mjollnir).
- **Minetown Temple Priest**:
  - Minetown features an aligned temple. Donate gold (400 * XP level) to the priest to receive permanent divine protection (AC reduction of 2 to 4 points!).""",
    ),
    (
        "Strategy: Item Identification, Beatitude, and Inventory Management",
        """# NetHack 3.6 Item Identification, Beatitude, and Kit Management

## 1. Beatitude: Cursed, Uncursed, and Blessed
- **Beatitude Rule**:
  - NEVER wear or wield unknown items without testing beatitude first!
  - Cursed armor or weapons weld to your body and cannot be removed without a scroll of remove curse.
  - Cursed rings can impart negative stats (e.g. ring of teleportation, ring of polymorph).
- **Pet Testing**:
  - Drop unknown items on the floor. Pets refuse to step on cursed items, or step on them with a reluctant message ("Your dog hesitates"). If your pet freely steps over an item, it is uncursed or blessed.
- **Altars**:
  - Dropping an item on an altar immediately identifies its beatitude: flashes purple/black (blessed), pale/no flash (uncursed), or red/brown (cursed).

## 2. Valkyrie Starting Gear & Inventory Discipline
- **Starting Setup**:
  - Slot `a`: +1 long sword
  - Slot `b`: +1 dagger
  - Slot `c`: +0 small shield
  - Slot `d`: food ration
- **Turn 1 Optimization**:
  - Equip small shield: `seq(Action.WEAR, 'c')` (AC drops from 10 to 9 or 8).
  - Wield long sword: `seq(Action.WIELD, 'a')`.
- **Identifying Potions and Scrolls**:
  - Clear potions are always water. Dipping water into an altar blesses it (holy water).
  - Potions of extra/full healing increase max HP when quaffed at full health.
  - Scrolls of enchant weapon / armor improve equipment pluses (+1 to +5).""",
    ),
]

REDIRECTS = [
    ("early game", "Strategy: Early Game Survival & Progression Guide"),
    ("survival", "Strategy: Early Game Survival & Progression Guide"),
    ("early game survival", "Strategy: Early Game Survival & Progression Guide"),
    ("exploration", "Strategy: Dungeon Exploration, Doors, and Stairs"),
    ("doors", "Strategy: Dungeon Exploration, Doors, and Stairs"),
    ("stairs", "Strategy: Dungeon Exploration, Doors, and Stairs"),
    ("valkyrie", "Strategy: Valkyrie Ascension Playbook"),
    ("excalibur", "Strategy: Valkyrie Ascension Playbook"),
    ("combat", "Strategy: Tactical Combat, Kiting, and Monster Threats"),
    ("kiting", "Strategy: Tactical Combat, Kiting, and Monster Threats"),
    ("monsters", "Strategy: Tactical Combat, Kiting, and Monster Threats"),
    ("floating eye", "Strategy: Tactical Combat, Kiting, and Monster Threats"),
    ("food", "Strategy: Nutrition, Safe Corpses, and Altars"),
    ("corpses", "Strategy: Nutrition, Safe Corpses, and Altars"),
    ("altars", "Strategy: Nutrition, Safe Corpses, and Altars"),
    ("hunger", "Strategy: Nutrition, Safe Corpses, and Altars"),
    ("inventory", "Strategy: Item Identification, Beatitude, and Inventory Management"),
    ("beatitude", "Strategy: Item Identification, Beatitude, and Inventory Management"),
    ("cursed", "Strategy: Item Identification, Beatitude, and Inventory Management"),
    ("armor", "Strategy: Item Identification, Beatitude, and Inventory Management"),
]


def populate_strategy_guides(db_path: str = "data/wiki_index.db") -> None:
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    for title, content in STRATEGY_ARTICLES:
        cur.execute("DELETE FROM articles WHERE title = ?;", (title,))
        cur.execute(
            "INSERT INTO articles (title, content) VALUES (?, ?);", (title, content)
        )
        art_id = cur.lastrowid
        cur.execute("DELETE FROM wiki_fts WHERE title = ?;", (title,))
        cur.execute(
            "INSERT INTO wiki_fts (rowid, title, content) VALUES (?, ?, ?);",
            (art_id, title, content),
        )

    cur.execute(
        "CREATE TABLE IF NOT EXISTS redirects (alias TEXT PRIMARY KEY, target_title TEXT);"
    )
    for alias, target in REDIRECTS:
        cur.execute(
            "INSERT OR REPLACE INTO redirects (alias, target_title) VALUES (?, ?);",
            (alias, target),
        )

    con.commit()
    con.close()
    print("Successfully populated strategy guides into wiki database!")


if __name__ == "__main__":
    populate_strategy_guides()
