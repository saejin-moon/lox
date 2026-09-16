# Specification 02: Epistemic POMDP Belief Engine

## 1. Mathematical Formulation of NetHack POMDP

NetHack is formally modeled as a Partially Observable Markov Decision Process:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \gamma \rangle$$

* $\mathcal{S}$: True hidden game state (exact monster inventory, true item identities, exact charges, dungeon layout).
* $\mathcal{A}$: Set of valid keyboard actions.
* $\mathcal{T}(s' \mid s, a)$: Stochastic state transition distribution (procedural generation, hit rolls, monster AI).
* $\Omega$: Observation space (ASCII map glyphs, status line, terminal text messages).
* $\mathcal{O}(o \mid s', a)$: Observation emission function (obscuring invisible monsters, un-IDed item appearances, trap flags).

The Epistemic Engine maintains a Bayesian belief state $b_t(s)$ that maps raw observations and historical experiments into actionable probability distributions over hidden variables.

---

## 2. Probabilistic Item State Vector

Every item $i \in \text{Inventory} \cup \text{FloorItems}$ is modeled as a probabilistic belief tuple:

$$B(i) = \langle \vec{P}_{\text{Identity}}, \vec{P}_{\text{BUC}}, \text{Charges}_{\text{est}}, \text{Enchantment}_{\text{est}}, \text{Erosion} \rangle$$

### Probability Distributions
1. **Identity Distribution**:
   $$\vec{P}_{\text{Identity}} = [p_1, p_2, \dots, p_K], \quad \sum_{k=1}^K p_k = 1.0, \quad p_k = P(\text{Item}_i = \text{Candidate}_k)$$
   Where candidates are drawn from the item appearance family (e.g., appearance `"bubbly potion"` $\implies \{\text{healing}, \text{extra healing}, \text{sleeping}, \dots\}$).
2. **BUC Distribution (Blessed / Uncursed / Cursed)**:
   $$\vec{P}_{\text{BUC}} = \langle P(\text{Blessed}), P(\text{Uncursed}), P(\text{Cursed}) \rangle, \quad \sum_{c \in \{\text{B}, \text{U}, \text{C}\}} P(c) = 1.0$$
3. **Charge & Enchantment Bounds**:
   $$\text{Charges}_{\text{est}} = [\text{min\_charges}, \text{max\_charges}], \quad \text{Enchantment}_{\text{est}} = [\text{min\_enchant}, \text{max\_enchant}]$$

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class ItemBeliefState:
    uid: str
    appearance: str
    item_class: str
    candidate_identities: list[str] = field(default_factory=list)
    identity_probs: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # BUC probabilities: [P(Blessed), P(Uncursed), P(Cursed)]
    p_buc: np.ndarray = field(default_factory=lambda: np.array([0.10, 0.80, 0.10]))
    
    min_charges: int = 0
    max_charges: int = 0
    charges_known: bool = False
    
    min_enchantment: int = -5
    max_enchantment: int = 7
    enchantment_known: bool = False
    
    erosion_level: int = 0      # 0=None, 1=Corroded, 2=Damaged, etc.
```

---

## 3. Bayesian Observation Listeners

The Epistemic Engine listens to specific environmental interactions and applies exact Bayesian likelihood updates:

$$P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}$$

### 1. `AltarListener` (Drop on Altar)
* **Mechanic**: Dropping any item onto an altar produces an immediate flash message, provided sensory faculties are uncompromised.
* **Sensory Gate Requirement**: 
  - Verification strictly requires $\neg \text{IS\_BLIND}$ and $\neg \text{IS\_HALLUCINATING}$.
  - If the player is **blind**, no visual flash text is emitted (would cause false uncursed inference).
  - If **hallucinating**, flash colors are randomized into illusory hues.
* **Likelihood Model** (when $\neg \text{IS\_BLIND} \wedge \neg \text{IS\_HALLUCINATING}$):
  - Observation: `"There is a flash of black light"` $\implies P(E \mid \text{Cursed}) = 1.0, P(E \mid \text{Other}) = 0.0$.
  - Observation: `"There is an amber/purple flash"` $\implies P(E \mid \text{Blessed}) = 1.0, P(E \mid \text{Other}) = 0.0$.
  - Observation: No flash text $\implies P(E \mid \text{Uncursed}) = 1.0$.
* **Posterior Collapse**: A single valid altar test collapses $\vec{P}_{\text{BUC}}$ to a deterministic one-hot vector in 1 game turn.

### 2. `PetMovementListener` (Pet Hesitation Test)
* **Mechanic**: Domestic pets (dog/cat/horse) instinctively refuse to step on cursed non-comestible items unless starved.
* **Comestible Exclusion Gate**:
  - If `item.class == FOOD`, domestic pets will greedily step on and eat the food regardless of cursed status.
  - **Rule**: Pet testing is strictly invalidated for comestibles ($P(E \mid \text{FOOD}) = \text{Uniform}$).
* **Likelihood Model** (for non-food items):
  - Item located at coordinate $(x, y)$.
  - Pet is adjacent and attempts to path toward the player or a monster across $(x, y)$:
    * Pet steps willingly onto $(x, y) \implies P(\text{Cursed} \mid \text{Step}) = 0.0$. Update $\vec{P}_{\text{BUC}}$:
      $$P(\text{Cursed}) \leftarrow 0.0, \quad P(\text{Blessed}) \leftarrow \frac{P(\text{Blessed})}{P(\text{Blessed}) + P(\text{Uncursed})}$$
    * Pet refuses to step on $(x, y)$, whining or stopping $\implies P(\text{Cursed} \mid \text{Refusal}) = 0.99$.

### 3. `PriceIDListener` (Merchant Cost Calculation)
* **Mechanic**: NetHack 3.6.6 shopkeepers determine buy and sell transactions through separate mathematical pathways:
* **Selling to Shopkeeper** (Dropping item in shop):
  - Charisma does **not** affect selling price!
  - Base offer: $P_{\text{sell}} = \lfloor \frac{1}{2} C_{\text{base}} \rfloor$ (or $\lfloor \frac{1}{3} C_{\text{base}} \rfloor$ if wearing a dunce cap, unclothed shirt, or Tourist $< \text{lvl 15}$).
  - In $\frac{1}{4}$ of shopkeepers, an unidentified surcharge pays $\frac{1}{4}$ less ($\frac{3}{8} C_{\text{base}}$).
* **Buying from Shopkeeper** (Picking up item in shop):
  - Base price modified by Charisma multiplier $M_{\text{cha}}$:
    $$\text{Cha} \le 5: 2.0\times \quad|\quad 6–7: 1.5\times \quad|\quad 8–10: 1.33\times \quad|\quad 11–15: 1.0\times \quad|\quad 16–17: 0.75\times \quad|\quad 18: 0.67\times \quad|\quad \ge 19: 0.5\times$$
  - Sucker surcharge: $\frac{4}{3}\times$ if sucker criteria met.
  - Unidentified item surcharge: in $\frac{1}{3}$ of items, shopkeeper adds $\frac{4}{3}\times$ surcharge.
* **Candidate Elimination Algorithm**:
  1. Record merchant offer $P_{\text{offer}}$ (buy or sell).
  2. Invert pricing equation to derive candidate base costs: $\hat{C}_{\text{base}} \in \text{InvertPrice}(P_{\text{offer}})$.
  3. Query the static spoiler table (`data/wiki_index.db`).
  4. Eliminate all candidate identities whose official base cost $\ne \hat{C}_{\text{base}}$.
  5. Re-normalize $\vec{P}_{\text{Identity}}$.

### 4. `EngraveTestListener` (Wand Scratching)
* **Mechanic**: Engraving on the floor with a wand (`E` then select wand) produces diagnostic messages without expending a directional combat charge.
* **Safety Interlock (3.6.0+ Explosion Guard)**:
  - In NetHack 3.6.0+, **cursed wands have a 1% chance of exploding when used to engrave**, destroying the wand and inflicting blast damage.
  - **Rule**: Wand engrave-testing is strictly forbidden unless $P(\text{Cursed}) = 0.0$ (previously verified via altar or pet).
* **Deterministic Effect Table (NetHack 3.6.6 Ground Truth)**:

| Terminal Message / Visual Effect | Eliminated Wand Candidates | Retained Wand Candidates |
| :--- | :--- | :--- |
| `"The bugs on the <floor> stop moving!"` | All except Sleep and Death | **Sleep, Death** |
| `"The engraving on the <floor> vanishes!"` | All except Vanishing set | **Cancellation, Teleportation, Make Invisible, Cold** *(if burnt)* |
| `"A few ice cubes drop from the wand."` | All except Cold | **Cold** |
| `"This <wand> is a wand of digging!"` / `"Gravel flies up..."` | All except Digging | **Digging** *(self-identifies)* |
| `"This <wand> is a wand of fire!"` | All except Fire | **Fire** *(self-identifies)* |
| `"Lightning arcs from the wand..."` | All except Lightning | **Lightning** *(self-identifies)* |
| `"The <floor> is riddled by bullet holes!"` | All except Magic Missile | **Magic Missile** |
| `"The engraving now reads: <random>"` | All except Polymorph | **Polymorph** |
| `"The bugs on the <floor> slow down!"` | All except Slow Monster | **Slow Monster** |
| `"The bugs on the <floor> speed up!"` | All except Speed Monster | **Speed Monster** |
| `"The wand unsuccessfully fights your attempt to write!"` | All except Striking | **Striking** |
| No message or effect | Non-inert wands | **Locking, Opening, Probing, Undead Turning, Secret Door Detection, Stasis, Exhausted** |

---

## 4. Shannon Entropy Safe-Gates

To prevent fatal blunders (e.g., equipping a cursed amulet of strangulation, reading an unidentified scroll of fire next to potions, or quaffing a cursed potion of death), the Epistemic Engine calculates the **Shannon Entropy** of each item state:

$$\mathcal{H}(B(i)) = -\sum_{k=1}^K p_k \log_2 p_k$$
$$\mathcal{H}_{\text{BUC}}(B(i)) = -\sum_{c \in \{\text{B}, \text{U}, \text{C}\}} P(c) \log_2 P(c)$$

```mermaid
flowchart TD
    A[HTN Proposes Primitive: EQUIP / QUAFF / READ] --> B{Calculate BUC Entropy}
    B -->|P(Cursed) > 0.05| C[Veto Action: Cursed Risk]
    B -->|H(Identity) > tau_safe| D[Veto Action: Identity Entropy Too High]
    C --> E[Inject HTN Sub-Goal: RESOLVE_ITEM_ENTROPY]
    D --> E
    E --> F[Schedule Altar / Pet / Price / Engrave Test]
    B -->|P(Cursed) <= 0.05 AND H(Identity) <= tau_safe| G[Authorize Execution]
```

### Risk Evaluation Function
For any proposed action $a \in \{\text{EQUIP}, \text{QUAFF}, \text{READ}\}$ on item $i$:

$$\text{Risk}(a, i) = P(\text{Cursed}) \cdot \mathcal{L}_{\text{CursedLock}} + \sum_{k \in \text{Candidates}} p_k \cdot \text{FatalityScore}(k)$$

* If $\text{Risk}(a, i) > \tau_{\text{risk\_threshold}}$, the HTN planner immediately rejects the decomposition.
* The system substitutes a safe information-gathering sub-goal:
  `TASK: RESOLVE_ITEM_ENTROPY(target_item=i)`
  which routes the agent toward an altar, drops the item in front of a pet, or seeks a shopkeeper before any attempt to use the item is permitted.
