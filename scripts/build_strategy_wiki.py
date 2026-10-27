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
    (
        "Strategy: The Castle, Drawbridge Breaching, and Wand of Wishing",
        """# NetHack 3.6 The Castle, Drawbridge Breaching, and Wishing Guide

## 1. Finding & Breaching The Castle (Dlvl 25-29)
- **Castle Layout**: Surrounded by a deep water moat with a central closed drawbridge. The courtyard is filled with dragons, giants, liches, and soldier swarms.
- **Breaching the Drawbridge**:
  - NEVER stand directly adjacent to the closed drawbridge when destroying it (the blast or collapse kills you!). Stand 2-3 tiles back.
  - Destroying options:
    - Zap wand of striking (`seq(Action.ZAP, slot, dir)`).
    - Cast force bolt.
    - Play the pass-tune on any musical instrument (uncursed bugle, horn, flute, harp) via `#apply`.
  - Alternatively: freeze the moat with a cold wand / cone of cold, or use levitation / flying boots to cross without opening the drawbridge.

## 2. Defensive Courtyard Clear
- **Chokepoint Advantage**: Retreat outside the breach into a corridor or doorway. Let dragons and giants come to you one at a time.
- **Scroll of Scare Monster / Elbereth**: Stand behind an engraved Elbereth or on an uncursed scroll of scare monster so monsters flee rather than melee you.

## 3. The Wand of Wishing (The Ultimate Power Spike)
- **Location**: Located in a closed chest in the back-right barracks tower of the Castle.
- **Charges**: Generated with (0:3) charges.
- **Recharging Rule**:
  - Always recharge the wand ONCE with a blessed scroll of charging (`seq(Action.READ, slot)` -> select wand).
  - This restores the wand to (1:3), granting 3 additional wishes! Never recharge a wand twice (it will explode).
- **Ascension Wishing Priority List**:
  1. `blessed scroll of charging` (if you have none in inventory, to recharge the wand).
  2. `blessed greased +2 gray dragon scale mail` (grants permanent Magic Resistance).
  3. `blessed fireproof +2 cloak of displacement` (causes attacks to miss by targeting ghost image).
  4. `blessed +2 pair of speed boots` (very fast speed intrinsic).
  5. `blessed +2 pair of gauntlets of power` (sets Strength to 25; allows Valkyrie to throw Mjollnir as a returning lightning boomerang!).
  6. `blessed magic marker` (write scrolls of enchant armor, enchant weapon, genocide, gold detection, and magic mapping).
  7. `blessed amulet of life saving` (worn around neck: revives you with full HP if you die).""",
    ),
    (
        "Strategy: Gehennom Navigation, Demon Lords, and The Vibrating Square",
        """# NetHack 3.6 Gehennom Mazes, Demon Lords, and The Vibrating Square Guide

## 1. Surviving Gehennom (Dlvl 30-50)
- **Mandatory Intrinsics**:
  - **Fire Resistance**: Gehennom is filled with lava and fire traps.
  - **Magic Resistance**: Essential against spellcasting demon lords and arch-liches.
  - **Reflection**: Bounces dragon breath and death rays.
  - **Poison Resistance**: Many demons inflict deadly poison.
- **Ring of Conflict**: Wearing a ring of conflict (`seq(Action.PUTON, slot)`) causes demon hordes and monsters to attack each other instead of swarming you!
- **Blessed Genocide (Scroll of Genocide)**:
  - Read blessed scroll of genocide: eradicate **liches** ('L') to eliminate arch-liches and master liches permanently from the entire game!
  - Optionally genocide **mind flayers** ('h') to prevent deadly brain-eating stat drains.

## 2. Demon Lords & Boss Encounters
- **Valley of the Dead (Dlvl 30)**: First level of Gehennom. Full of graveyards; use Excalibur (level drain resistance) to dispatch wights and wraiths safely.
- **Juiblex**: Demon lord of acid. Never melee without acid resistance. Zap with wand of digging to drop through his floor, or kill from range.
- **Orcus**: High demon who spawns with a **wand of death**! Reflection and Magic Resistance are 100% mandatory. Kill immediately or steal/zap wand first.
- **Vlad's Tower (Branch up from Dlvl 36-39)**: 3-level branch. Vlad the Impaler is easy to kill; he drops the essential **Candelabrum of Invocation**.
- **Wizard's Tower (Dlvl 42-45)**: Enter via magic portal. The Wizard of Yendor drops the essential **Book of the Dead**.

## 3. Locating The Vibrating Square
- **The Portal to Moloch's Sanctum**: On the deepest regular maze level of Gehennom (usually Dlvl 45-48), one floor tile is the **Vibrating Square**.
- **The Gold Detection Trick (Instant Location)**:
  1. Become confused (eat a tripe ration, quaff uncursed potion of booze, or read scroll while confused).
  2. While `state.is_confused` is True, read an uncursed **scroll of gold detection** (`seq(Action.READ, slot)`).
  3. Instead of gold, this instantly maps all traps and portals on the floor, marking the Vibrating Square with a '^' on your screen!
  4. Navigate to the Vibrating Square coordinate and drop an identifying object or engrave on it.""",
    ),
    (
        "Strategy: The Invocation Ritual and Moloch's Sanctum",
        """# NetHack 3.6 The Invocation Ritual and Moloch's Sanctum Guide

## 1. The Three Invocation Artifacts
To perform the Invocation Ritual and unlock the stairs to Moloch's Sanctum, you must have all 3 items in your inventory:
1. **The Bell of Opening**: Awarded upon completing your role's Quest (Dlvl 14-16).
2. **The Candelabrum of Invocation**: Dropped by Vlad the Impaler in Vlad's Tower.
3. **The Book of the Dead**: Dropped by the Wizard of Yendor in the Wizard's Tower.
- **Prep Work**: You must attach **7 lit candles** (beeswax or tallow candles found in Minetown or shops) to the Candelabrum of Invocation before the ritual.

## 2. Performing the Invocation Ritual
1. Stand directly on the **Vibrating Square** coordinate.
2. Apply the **Bell of Opening**: `seq(Action.APPLY, bell_letter)`.
3. Apply the **Candelabrum of Invocation** (with 7 candles lit): `seq(Action.APPLY, candelabrum_letter)`.
4. Read the **Book of the Dead**: `seq(Action.READ, book_letter)`.
- **Result**: The floor trembles and a staircase opens down to **Moloch's Sanctum**!

## 3. Moloch's Sanctum & Retrieving The Amulet of Yendor
- Moloch's Sanctum is an un-mappable graveyard maze surrounding the High Temple of Moloch.
- Enter the inner sanctum, defeat the High Priest of Moloch, and retrieve the genuine **Amulet of Yendor** ('\"').
- **The Ascent & The Mysterious Force**:
  - Once carrying the Amulet of Yendor, you must climb back UP through all 50 levels of the dungeon to Dlvl 1.
  - The **Mysterious Force**: Every time you climb up stairs, there is a chance you are pushed back down 1-3 levels. Stay calm, turn around, and climb back up.
  - The Wizard of Yendor will periodically resurrect and harass you until you reach the Elemental Planes.""",
    ),
    (
        "Strategy: The Elemental Planes and The Astral Plane (Ascension)",
        """# NetHack 3.6 The Elemental Planes and Astral Plane Ascension Guide

## 1. Entering the Elemental Planes
- Carry the real Amulet of Yendor to **Dungeon Level 1**.
- Climb the upstairs ('<') while carrying the Amulet to enter the **Elemental Planes**.

## 2. Navigating the Four Elemental Planes
On each elemental plane, your goal is finding and entering the magic portal ('^') to the next plane:
- **The Universal Portal Finding Trick**:
  - Read an uncursed **scroll of gold detection while confused** (`state.is_confused`). This instantly reveals the exact coordinates of the magic portal!
- **Plane of Earth**:
  - Surrounded by solid rock. Use a **wand of digging** (`seq(Action.ZAP, slot, dir)`) to dig corridors toward the portal.
- **Plane of Air**:
  - Very dangerous! Lightning strikes and high-damage Air Elementals.
  - Put on **Ring of Conflict** immediately so elementals fight each other.
  - Levitate or wear speed boots and dash directly into the portal.
- **Plane of Fire**:
  - Lava fields everywhere. Fire resistance and levitation/water-walking boots are mandatory.
- **Plane of Water**:
  - Air bubbles in an endless sea. Magical breathing (amulet of magical breathing or oilskin sack) prevents drowning. Travel inside air bubbles toward the portal.

## 3. The Astral Plane (The Final Test)
- **The Layout**: Three corridors leading to three High Altars: **Lawful**, **Neutral**, and **Chaotic**.
- **The Three Riders of the Apocalypse**:
  - **Death**: Touch rapidly drains HP. Do not fight in melee! Zap wand of death, throw Mjollnir from range, or teleport monsters away.
  - **Pestilence**: Touch inflicts deadly illness. Cure immediately with a unicorn horn (`seq(Action.APPLY, slot)`) or full healing potion.
  - **Famine**: Touch inflicts immediate Weak/Fainting hunger. Quaff potions of full healing or eat rations immediately.
- **Ascension Execution**:
  1. Put on **Ring of Conflict** so Angels, Priests, and Riders fight each other.
  2. Use **wands of teleportation** to beam away blocking hordes and clear a straight path.
  3. Identify the **co-aligned High Altar** (for Valkyrie: **Lawful High Altar**).
  4. Step onto the High Altar.
  5. Execute `#offer` with the Amulet of Yendor (`seq(Action.OFFER, amulet_letter)`).
  6. **VICTORY**: You ascend to demigod status and win the game!""",
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
    ("castle", "Strategy: The Castle, Drawbridge Breaching, and Wand of Wishing"),
    ("wishing", "Strategy: The Castle, Drawbridge Breaching, and Wand of Wishing"),
    ("wand of wishing", "Strategy: The Castle, Drawbridge Breaching, and Wand of Wishing"),
    ("drawbridge", "Strategy: The Castle, Drawbridge Breaching, and Wand of Wishing"),
    ("gehennom", "Strategy: Gehennom Navigation, Demon Lords, and The Vibrating Square"),
    ("demon", "Strategy: Gehennom Navigation, Demon Lords, and The Vibrating Square"),
    ("vibrating square", "Strategy: Gehennom Navigation, Demon Lords, and The Vibrating Square"),
    ("orcus", "Strategy: Gehennom Navigation, Demon Lords, and The Vibrating Square"),
    ("juiblex", "Strategy: Gehennom Navigation, Demon Lords, and The Vibrating Square"),
    ("vlad", "Strategy: Gehennom Navigation, Demon Lords, and The Vibrating Square"),
    ("invocation", "Strategy: The Invocation Ritual and Moloch's Sanctum"),
    ("sanctum", "Strategy: The Invocation Ritual and Moloch's Sanctum"),
    ("book of the dead", "Strategy: The Invocation Ritual and Moloch's Sanctum"),
    ("bell of opening", "Strategy: The Invocation Ritual and Moloch's Sanctum"),
    ("candelabrum", "Strategy: The Invocation Ritual and Moloch's Sanctum"),
    ("amulet of yendor", "Strategy: The Invocation Ritual and Moloch's Sanctum"),
    ("astral plane", "Strategy: The Elemental Planes and The Astral Plane (Ascension)"),
    ("elemental planes", "Strategy: The Elemental Planes and The Astral Plane (Ascension)"),
    ("planes", "Strategy: The Elemental Planes and The Astral Plane (Ascension)"),
    ("riders", "Strategy: The Elemental Planes and The Astral Plane (Ascension)"),
    ("death", "Strategy: The Elemental Planes and The Astral Plane (Ascension)"),
    ("pestilence", "Strategy: The Elemental Planes and The Astral Plane (Ascension)"),
    ("famine", "Strategy: The Elemental Planes and The Astral Plane (Ascension)"),
    ("ascension", "Strategy: The Elemental Planes and The Astral Plane (Ascension)"),
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
