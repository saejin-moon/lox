# Specification 01: Sensory Boundary & NLE Interface

## 1. Scope & Execution Objectives

The Sensory Boundary acts as the low-level physical barrier between the raw NetHack C-engine (`nle.env.NetHackEnv`) and the higher-order reasoning layers. It guarantees:
1. **Zero-Latency Prompt Sanitation**: Swallows `--More--` dialogs, multi-page inventory screens, and confirmation queries deterministically without invoking the HTN planner.
2. **Permanent Inventory UID Normalization**: Decouples NetHack's volatile inventory letters (`a-zA-Z`) from persistent item identities.
3. **Sub-0.05 ms Anomaly Detection**: Monitors life-critical invariant deltas on every game tick, immediately generating hard interrupts (`SIG_ANOMALY_INTERRUPT`) when fatal states threaten the agent.

---

## 2. Observation Tensor Contracts

The NetHack Learning Environment outputs a multi-modal observation dictionary $O_t$:

```python
from typing import TypedDict
import numpy as np
import numpy.typing as npt

class RawNLEObservation(TypedDict):
    glyphs: npt.NDArray[np.int16]       # Shape (21, 79), range [0, 5976]: Visual glyph IDs
    chars: npt.NDArray[np.uint8]        # Shape (21, 79), ASCII characters of map
    colors: npt.NDArray[np.uint8]       # Shape (21, 79), ANSI color attributes
    blstats: npt.NDArray[np.int64]      # Shape (25,): Bottom-line numeric status vector
    message: npt.NDArray[np.uint8]      # Shape (256,): Raw ASCII byte buffer of terminal message
    inv_glyphs: npt.NDArray[np.int16]   # Shape (55,): Glyphs of inventory items
    inv_strs: npt.NDArray[np.uint8]     # Shape (55, 80): Text strings of inventory descriptions
    inv_letters: npt.NDArray[np.uint8]  # Shape (55,): ASCII letters ('a'-'z', 'A'-'Z')
    tty_chars: npt.NDArray[np.uint8]    # Shape (24, 80): Full terminal grid representation
    tty_cursor: npt.NDArray[np.int32]   # Shape (2,): (row, col) cursor position
```

### Bottom-Line Status Vector (`blstats`) Mapping
The 25-element integer array is mapped into a strongly typed state representation:

```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class BottomLineStats:
    x: int               # blstats[0]: X coordinate (1-79)
    y: int               # blstats[1]: Y coordinate (0-20)
    strength_pct: int    # blstats[2]: Strength percentage (if 18/**)
    strength: int        # blstats[3]: Base strength
    dexterity: int       # blstats[4]: Dexterity
    constitution: int    # blstats[5]: Constitution
    intelligence: int    # blstats[6]: Intelligence
    wisdom: int          # blstats[7]: Wisdom
    charisma: int        # blstats[8]: Charisma
    score: int           # blstats[9]: Current score
    hp: int              # blstats[10]: Current Hit Points
    max_hp: int          # blstats[11]: Maximum Hit Points
    depth: int           # blstats[12]: Current Dungeon Depth
    gold: int            # blstats[13]: Carried gold
    energy: int          # blstats[14]: Current power/energy
    max_energy: int      # blstats[15]: Maximum power/energy
    ac: int              # blstats[16]: Armor Class (lower is better)
    monster_level: int   # blstats[17]: Experience level
    experience: int      # blstats[18]: Raw experience points
    turn: int            # blstats[19]: Elapsed game turns
    hunger_state: int    # blstats[20]: 0=Satiated, 1=Normal, 2=Hungry, 3=Weak, 4=Fainting
    encumbrance: int     # blstats[21]: 0=Unencumbered, 1=Burdened, 2=Stressed, ...
    dungeon_number: int  # blstats[22]: Dungeon branch ID (0=Doom, 1=Mines, 2=Sokoban, ...)
    level_number: int    # blstats[23]: Branch-relative level index
    condition_bits: int  # blstats[24]: Bitmask of status conditions
```

### Condition Bitmask Indices (`blstats[24]`)
```python
class ConditionFlag:
    STONE           = 1 << 0   # Petrifying (fatal in ~3 turns)
    SLIME           = 1 << 1   # Turning to green slime (fatal in ~5 turns)
    STRANGLE        = 1 << 2   # Being strangled (choking, fatal)
    FOOD_POISONING  = 1 << 3   # Fatal in ~3 turns
    TERM_ILLNESS    = 1 << 4   # Terminal illness
    BLIND           = 1 << 5   # Blindness
    DEAF            = 1 << 6   # Deafness
    STUN            = 1 << 7   # Stunned (random movement)
    CONFUSED        = 1 << 8   # Confused (erratic actions)
    HALLUCINATING   = 1 << 9   # Hallucinating (monsters/items appear random)
    LEVITATING      = 1 << 10  # Levitating
    FLYING          = 1 << 11  # Flying
    RIDE            = 1 << 12  # Riding a steed
```

---

## 3. The `AutoMoreWrapper` State Machine

NetHack terminal output is inherently conversational. An action may trigger multiple dialog steps before returning to an interactive state. The `AutoMoreWrapper` encapsulates `nle.step()` inside a deterministic loop that guarantees the environment only yields control when the prompt is fully resolved.

```mermaid
flowchart TD
    A[Raw Step Returned] --> B{Examine message & tty}
    B -->|Contains '--More--'| C[Send Spacebar ' ']
    C --> A
    B -->|Contains 'Do you want to die? [yn]'| D[Send 'n']
    D --> A
    B -->|Contains 'Are you sure? [yn]'| E[Resolve Safe Context]
    E --> A
    B -->|Inventory Menu '[1 of 2]'| F[Parse Page & Send Spacebar]
    F --> A
    B -->|Prompt Clean & Interactive| G[Yield Normalized Observation to HTN]
```

### Handled Query Patterns
1. **`--More--` Dialogue Flush**:
   - Condition: `b"--More--"` present in `raw_obs["message"]` or raw `tty_chars`.
   - Action: Emit `ord(' ')` (spacebar) synchronously.
2. **Death Queries**:
   - Pattern: `"Do you really want to quit? [yn]"` or `"Do you want your possessions identified? [yn]"`.
   - Action: Emit `'y'` or `'n'` according to session policy.
3. **Directional Interrogatives**:
   - Pattern: `"In what direction?"`.
   - Action: If in interactive dispatch, look up the pending target coordinate from the active Primitive Task and emit the corresponding direction key (`h, j, k, l, y, u, b, n`).
4. **Item Slot Queries**:
   - Pattern: `"What do you want to eat? [a-z or ?*]"`.
   - Action: Query the Normalized Inventory Tracker for the requested item UID, resolve its current dynamic letter, and emit that keystroke.

---

## 4. Persistent Inventory UID Normalization

In NetHack, inventory slots are designated by ASCII letters (`a`–`z`, `A`–`Z`). When an item in slot `c` is eaten or dropped, NetHack may reassign slot `c` to another item or shift later items upward.

To maintain continuous Bayesian belief states over items, the Sensory Boundary provides a **Persistent Item Tracker**:

```python
from dataclasses import dataclass, field
import uuid

@dataclass
class NormalizedItem:
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    raw_str: str = ""
    current_letter: str = ""
    glyph: int = 0
    item_class: str = ""       # "weapon", "armor", "potion", "scroll", "wand", "ring", etc.
    quantity: int = 1
    equipped: bool = False
    first_seen_turn: int = 0
    last_updated_turn: int = 0

class InventoryNormalizer:
    def __init__(self):
        self.active_items: dict[str, NormalizedItem] = {} # Keyed by UID
        self.letter_to_uid: dict[str, str] = {}           # Maps 'a' -> UID

    def synchronize(self, inv_strs: np.ndarray, inv_letters: np.ndarray, inv_glyphs: np.ndarray, turn: int):
        """
        Runs Hungarian bipartite matching between existing active items and new raw strings
        to ensure stable UID tracking even across letter shifts.
        """
        # 1. Parse raw inventory records
        # 2. Match exact name + glyph strings to existing UIDs
        # 3. For newly discovered items, allocate new unique UIDs
        # 4. Refresh letter_to_uid map
        pass

    def get_letter(self, uid: str) -> str:
        return self.active_items[uid].current_letter
```

---

## 5. Anomaly Sentry & Hard Interrupt Protocol

The Anomaly Sentry is a 60 Hz zero-allocation evaluation module that runs after every physical tick. It enforces deterministic safety invariants.

### Invariant Rules Table

| Anomaly Class | Detection Predicate | Severity | Action Triggered |
| :--- | :--- | :--- | :--- |
| **Burst HP Drop** | $HP_t < 0.80 \cdot HP_{t-1}$ | Critical | Hard interrupt; cancel active HTN branch; invoke emergency survival reflex. |
| **Critical HP Floor** | $HP_t \le \max(4, 0.20 \cdot MaxHP_t)$ | Fatal Risk | Force immediate retreat/Elbereth/prayer evaluation. |
| **Lethal Status Flag** | `condition & (STONE \| SLIME \| STRANGLE \| FOOD_POISON)` | Instant Fatal | Emergency curative dispatch (lizard corpse, wand of fire, cure sickness). |
| **Control Impairment**| `condition & (STUN \| CONFUSED \| BLIND \| HALLU)` | High Danger | Halt navigation; seek defensive corner or wait behind pet. |
| **Spatial Stalemate** | $(x_t, y_t) == (x_{t-15}, y_{t-15})$ and $D_t == D_{t-15}$ | Stalemate | Trigger Anomaly Deadlock; if HTN has 0 fixes, escalate to Deliberative LLM Core. |
| **Hunger Transition** | `hunger_state >= 3` (Weak or Fainting) | Critical | Force interrupt; inject high-priority `FORAGE_FOOD` or `EAT` compound goal. |

### Interrupt Dispatch Contract
```python
from enum import Enum, auto

class AnomalySeverity(Enum):
    WARNING = auto()
    CRITICAL = auto()
    FATAL = auto()

@dataclass(slots=True, frozen=True)
class AnomalyEvent:
    severity: AnomalySeverity
    condition_name: str
    message: str
    turn: int
    delta_hp: int = 0
    current_hp: int = 0
    max_hp: int = 0

class AnomalySentry:
    def __init__(self):
        self.prev_blstats: BottomLineStats | None = None
        self.position_history: list[tuple[int, int]] = []

    def evaluate(self, current: BottomLineStats, raw_message: str) -> AnomalyEvent | None:
        if self.prev_blstats is None:
            self.prev_blstats = current
            return None

        # 1. Sudden HP Drop Check
        if current.hp < int(self.prev_blstats.hp * 0.80):
            return AnomalyEvent(
                severity=AnomalySeverity.CRITICAL,
                condition_name="BURST_DAMAGE",
                message=f"HP dropped from {self.prev_blstats.hp} to {current.hp}",
                turn=current.turn,
                delta_hp=self.prev_blstats.hp - current.hp,
                current_hp=current.hp,
                max_hp=current.max_hp
            )

        # 2. Instant Fatal Condition Check
        lethal_bits = (ConditionFlag.STONE | ConditionFlag.SLIME | 
                      ConditionFlag.STRANGLE | ConditionFlag.FOOD_POISONING)
        if current.condition_bits & lethal_bits:
            return AnomalyEvent(
                severity=AnomalySeverity.FATAL,
                condition_name="LETHAL_STATUS",
                message=f"Lethal status detected: bits={bin(current.condition_bits)}",
                turn=current.turn,
                current_hp=current.hp,
                max_hp=current.max_hp
            )

        # 3. Stalemate Tracker (Last 15 turns)
        self.position_history.append((current.x, current.y))
        if len(self.position_history) > 15:
            self.position_history.pop(0)
            if len(set(self.position_history)) == 1:
                return AnomalyEvent(
                    severity=AnomalySeverity.WARNING,
                    condition_name="SPATIAL_STALEMATE",
                    message="Agent stationary for 15 consecutive turns",
                    turn=current.turn,
                    current_hp=current.hp,
                    max_hp=current.max_hp
                )

        self.prev_blstats = current
        return None
```
