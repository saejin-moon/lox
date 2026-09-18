# MACRO.md — The CORP Policy Diff & Macro Expansion Specification (v1.0)

Complete specification for the S-expression policy-diff language, the `defmacro` system, the exposed vocabulary,
and the validation machinery. This is the implementation contract for `corp/policy/dsl.py`, `corp/policy/macros.py`,
`corp/policy/validator.py`, `corp/policy/program.py`, and the `corp/policy/grammar/` GBNF files.

**Design axiom**: the LLM expresses *what to value and when*. It can never express *how to act*. Everything it may
reference is a named, typed, pre-bound primitive; everything it may *create* is definitional sugar over those
primitives. There is no code emission, no escape hatch, no recursion.

---

## 1. The Two-Layer Model

```
LAYER 1 — THE POLICY PROGRAM (data/policy_program.json)          [storage, typed, versioned]
  The single source of truth the executor compiles. Contains only:
  closed primitives + macro definitions + their compositions.

LAYER 2 — POLICY DIFFS (S-expressions, emitted by the LLM)       [transient, validated, ledgered]
  Minimal edits against Layer 1. Accepted diffs are applied, the program version bumps,
  the diff is archived with provenance, and the ledger records the measured delta.
```

The LLM **never** sees or edits the executor. The executor **never** sees a diff — only validated, expanded,
compiled programs.

---

## 2. Lexical Layer

### 2.1 Tokens
| Token | Form | Notes |
|---|---|---|
| Open/close paren | `(` `)` | the only grouping syntax |
| Symbol | `[a-z_][a-z0-9_.]*` | primitive names, macro names, param paths; case-sensitive, lowercase |
| Keyword | `:` prefix? No — keywords are positional | see §3 |
| String | `"..."` double-quoted, no escapes except `\"` | used for monster/item names, reasons, notes |
| Number | int or float (IEEE754 literal) | compared with declared param type; int↔float coercion per param spec |
| Boolean | `true` / `false` | only where a param/predicate declares bool |
| Comment | `;` to end of line | stripped pre-parse; **never persisted** into the program |

### 2.2 Depth budget
- Diff form nesting: **≤ 3** (`(rule add tactic_rules (when (and ...)))` = 3).
- Macro *body* nesting: **≤ 3**, counted *after* expansion (a macro used inside a macro consumes the caller's budget).
- Rationale: caps expressiveness at "value statements", prevents program-in-disguise. The GBNF grammar enforces
  this at decode time; the parser re-enforces it post-decode (API authors).

### 2.3 Size budgets (per accepted revision)
| Budget | Limit | Purpose |
|---|---|---|
| Diff length | ≤ 40 forms | keep revisions attributable |
| `set` ops | ≤ 10 | param tuning stays surgical |
| `rule` ops | ≤ 5 | tactic changes stay reviewable |
| `defmacro` ops | ≤ 3 per revision; ≤ 64 live per program | bound vocabulary growth |
| Goal ops | ≤ 2 | strategy re-planning is rare and loud |
| `nogood` ops | ≤ 10 | usually auto-generated from autopsies |

---

## 3. Grammar (all forms)

```
<diff>       := (revision <int> (parent <int>) (author <string>) (domain <symbol>) (reason <string>))
<top-form>   := <set> | <rule-op> | <goal-op> | <nogood-op> | <defmacro> | (note <string>)

<set>        := (set <param-path> <value>)
<param-path> := policy_params.<dotted-path>          ; must exist in schema, leaf, within bounds

<rule-op>    := (rule add tactic_rules <rule>)
              | (rule remove tactic_rules <rule-index:int> | <rule>)
<rule>       := (when <expr>) (do <verb>) [(unless <expr>)] [(note <string>)]

<goal-op>    := (goal prioritize <goal-name> [(when <expr>)] [(until <expr>)])
              | (goal deprioritize <goal-name> [(after <expr>)])
              | (goal set_threshold <goal-name> <param-path> <value>)   ; goals own tuning params too

<nogood-op>  := (nogood add (when <expr>) (cause <string>) [(forbid <task-name>)])

<defmacro>   := (defmacro <name> <expr>)             ; name: new, unique, not colliding with primitives
<expr>       := <predicate-call> | <macro-call> | (and <expr>*) | (or <expr>*) | (not <expr>)
              | (compare <predicate> <op> <value>)   ; sugar: (hp_frac <= 0.5)
<predicate-call> := (<name> <arg>*)                  ; name must be in the exposed vocabulary
<macro-call> := (<name> <arg>*)                      ; name must be a live macro
<verb>       := ranged_only | ranged_then_kill | retreat | retreat_when_wounded | avoid
              | kite | elbereth_first | never_melee                  ; closed set, per-domain
```

**Parse rules**
1. First form must be `revision`; `parent` must equal the current program version (forced rebase otherwise).
2. `domain` must match an active `DomainAdapter` name; all paths/names resolve against that domain's vocabulary.
3. Unknown symbol anywhere → reject (`ERR_UNKNOWN_SYMBOL`). The LLM cannot introduce atoms that aren't bound.
4. Comments and `note` strings are preserved in the **ledger** but never affect execution.

---

## 4. The Exposed Vocabulary (what the LLM actually gets)

Every domain adapter exposes a **Vocabulary Manifest** — a machine-generated document injected into the author
prompt (and mirrored in the validator). The LLM can only use what appears here.

### 4.1 Manifest structure (generated per domain, per program version)

```yaml
domain: nethack
version: 14
predicates:                        # name: signature -> returns bool
  hp_frac:        (cmp: <=|>=|<|>|==, value: float 0..1)   # hp / max_hp
  hunger_ge:      (level: satiated|normal|hungry|weak|fainting)
  depth_between:  (lo: int, hi: int)
  xl_ge:          (n: int 1..30)
  lawful:         ()
  has_poison_res: ()
  has_reflection: ()
  has_healing:    ()
  adjacent_hostiles: ()
  stairs_known:   ()
  minetown_known: ()
  donations_lt:   (n: int)
  turn_ge:        (t: int)
  encumbrance_le: (n: int 0..4)
  failures_in_10_episodes_ge: (n: int, goal: <goal-name>)
verbs:
  ranged_only: {desc: "never melee vs matched target"}
  retreat:     {desc: "max-separation escape step"}
  ...                                      # closed per-domain list
goals:                             # certified goal handlers, with their owned params
  descend:       {params: [min_hp_frac, deep_min_hp_frac, deep_min_depth]}
  enter_sokoban: {params: [hunt_steps_cap]}
param_leaves:                      # policy_params.* with bounds
  survival.rest_below_frac: {type: float, min: 0.30, max: 0.90, default: 0.60}
  ...
macros:                            # currently live macros (name + expanded form)
  when_endangered: "(and (adjacent_hostiles) (hp_frac <= 0.40) (not has_healing))"
```

### 4.2 Exposure rules
1. **Predicates are bound, not computed.** Each name maps to a Python callable + JSON signature in the adapter's
   `predicates()` registry. Signature mismatch (arity, arg type, arg range) → reject.
2. **Verbs are closed.** A verb not in the adapter registry → reject. Adding a verb is a *code change*, never a diff.
3. **Goals are certified.** A goal name not in the adapter's certified handler set → reject. (R8 will define the
   process by which LLM-*proposed* handlers enter this set — through shadow certification, never through a diff.)
4. **Param leaves are typed.** Each path has `{type, min, max, default}`. `set` outside bounds → reject. Writes to
   non-leaf or unknown paths → reject.
5. **The manifest is the prompt.** The reviser renders it (plus a tail of the ledger + run report + RAG slices)
   into the author prompt. The LLM cannot reference anything outside it — by construction and by rejection.

---

## 5. Macro Semantics (`defmacro`)

### 5.1 What a macro is
A macro is a **named, parameterless-or-parameterized expression abbreviation** whose body is a valid expression
over already-live primitives and already-live macros. It is **purely definitional**: `(defmacro m body)` means
"replace every `(m ...)` call with the appropriate substitution of `body`, then re-validate."

### 5.2 Expansion algorithm (`policy/macros.py`)
```
expand(expr, macro_env, budget):
  1. if expr is (and/or/not ...) or predicate call: recurse into children, decrement budget per level
  2. if expr is (m a1 ... ak) where m is a live macro:
       a. bind body parameters (see 5.3) — or verify parameterless
       b. substitute → body'
       c. recurse expand(body')                                 # the expansion is re-parsed as an expr
       d. budget -= depth(body')                                 # EXPANDED depth counts against caller budget
  3. if depth(expanded) > 3 → reject (ERR_DEPTH_BUDGET)
  4. unknown symbol at any leaf → reject (ERR_UNKNOWN_SYMBOL)
```

### 5.3 Semantic rules
| Rule | Definition | Enforcement |
|---|---|---|
| **Definitional only** | A macro may contain only predicates, comparisons, `and/or/not`, and calls to live macros | closure checker walks the body; any non-expression form → reject |
| **No recursion** | A macro body may not reference itself, directly or transitively | expansion cycle detection (name stack); max expansion depth 4 |
| **No shadowing** | Macro names may not collide with primitives, verbs, goals, param leaves, or other macros | manifest namespace check |
| **No parameter capture** | Macro parameters are positional and substituted textually *before* re-validation; bodies may not introduce free symbols beyond the manifest + own parameters | free-symbol check post-substitution |
| **Budgeted composition** | Every macro call consumes nesting budget from its call site — a macro that expands to depth-2 body used at depth-2 leaves total depth 4 → reject | post-expansion depth check |
| **Immutable once shipped** | Redefining a live macro requires `defmacro` with the same name — allowed, but the *old* expansion is preserved in program history and any rule referencing it is re-validated (may reject rules that break) | re-validation pass |
| **Deterministic** | Expansion is a pure function of (expr, manifest). No randomness, no time, no I/O | implementation constraint, unit-tested |

### 5.4 Why this is safe (the one-paragraph proof for the paper)
Expansion maps every macro to a finite expression over the closed primitive set; closure and depth checks bound
the result; every expanded expression is re-validated against the same signature tables as hand-written ones;
and evaluation is delegated to the *executor's* predicate bindings, which the diff never touches. Therefore a
macro adds **no new behavior class** — only new *names* for compositions of old behavior classes.

### 5.5 Macro testing (mandatory before acceptance)
1. **Closure test**: expansion contains only manifest primitives (post-substitution).
2. **Termination/depth test**: expansion terminates and respects the depth budget on a canonical input.
3. **Fixture evaluation**: expand and evaluate against 20 fixture states from the domain's certification set —
   must produce a deterministic bool and never raise.
4. **Non-regression**: applying the diff must keep the certification subset green (validator step 4).

A macro failing any of these → whole diff rejected (macros are never partially accepted).

---

## 6. Validation Pipeline (order is normative — `policy/validator.py`)

```
1. PARSE        s-expr reader; GBNF pre-guarantee locally, repair-retry once on API path
2. HEADER       revision/parent/domain match current program
3. VOCAB        every symbol resolves against the domain manifest (§4)
4. BOUNDS       every `set` within declared leaf bounds; type-coercion per leaf spec
5. BUDGETS      §2.2 depth, §2.3 form-count limits
6. MACRO CHECK  defmacro closure + cycle + no-shadow (§5.3)
7. EXPANSION    expand all forms (including new macros applied to new rules); depth re-check
8. PROGRAM MOUNT apply diff → candidate program; Pydantic schema validation of whole program
9. INVARIANTS   no goal removed; ≥1 descent-class goal present; nogood count monotone per domain
                unless the diff explicitly removes a nogood with a `cause`
10. SHADOW TEST new/changed macros + rules evaluated on 20-domain fixture states (deterministic, no raise)
11. CERTIFICATION  domain's certification subset (run_skill_certifications) green
12. QUICK BATCH    3 episodes × 5k steps: no score collapse (>50% drop vs current program mean) → reject
13. COMMIT         program version++, diff archived, ledger appended, git commit
```

**Error taxonomy** (stable codes, returned to the author LLM for one repair retry):
`ERR_PARSE`, `ERR_HEADER`, `ERR_UNKNOWN_SYMBOL`, `ERR_SIGNATURE`, `ERR_BOUNDS`, `ERR_DEPTH_BUDGET`,
`ERR_FORM_BUDGET`, `ERR_MACRO_CYCLE`, `ERR_MACRO_SHADOW`, `ERR_MACRO_CLOSURE`, `ERR_INVARIANT`,
`ERR_SHADOW_FAIL`, `ERR_CERT_FAIL`, `ERR_BATCH_REGRESS`.

**Rejection is first-class**: a rejected diff is *kept* (with its error code and reason) in the ledger. Rejected
diffs are training signal for the next revision — the loop is allowed to fail loudly and often.

---

## 7. Runtime Evaluation (what the executor does with expanded programs)

- Expanded predicates compile to the existing 64-bit `PredicateBit` mask compiler (`planner/predicates.py`
  heritage, migrated to `policy/predicates.py`), so a `when` clause costs the same sub-microsecond mask test as
  every hand-written guard in the codebase today.
- Tactic rules evaluate in listed order, first match wins; the matched verb overrides the combat manager's
  default branch for that monster/item class.
- Goal gates evaluate top-down each turn in the GoalInterpreter; `until`/`after` transitions emit `goal_events`
  rows (DuckDB) that feed `failures_in_10_episodes` back to the LLM.
- **Hot path guarantee**: a fully-macro'd program compiles once at episode start (expansion + mask compilation
  are episode-static; `turn_ge`/`failures_*` predicates are the only runtime-parameterized ones and are
  re-evaluated per turn). No expansion happens mid-episode.

---

## 8. Domain Scoping & Overlays

- Each `DomainAdapter` exposes its own manifest slice (§4.1). The diff's `(domain <name>)` header selects it.
- Cross-domain reuse: `domain_profiles` overlay shared tuning (e.g., `survival.*` defaults) onto all domains;
  domain-specific leaves override. Name collisions across domains are fine — resolution is per-domain, and the
  manifest states its provenance.
- NetHack exposes the richest vocabulary (~20 predicates, ~10 goals); transfer domains start minimal (5–8
  predicates, 3–5 goals) and grow only through certified code changes — the asymmetry is itself reported in the
  paper (vocabulary size vs. improvement curve).

---

## 9. Implementation Checklist (maps 1:1 to code)

| Component | File | Tests |
|---|---|---|
| Reader (§2–3) | `corp/policy/dsl.py` | grammar round-trip, depth rejection, unknown-symbol rejection, comment stripping |
| Manifest (§4) | `corp/policy/program.py` + adapter registries | signature tables generated from code; drift test (code↔manifest CI check) |
| Expansion (§5) | `corp/policy/macros.py` | closure, cycle, shadow, depth-budget, fixture-eval, determinism |
| Gates (§6) | `corp/policy/validator.py` | one test per error code; ordering test (bounds before shadow, etc.) |
| Ledger (§6.13) | `corp/policy/ledger.py` | append-only; JSONL schema; rejection retention |
| Runtime (§7) | `corp/policy/goal_interpreter.py` + compiled masks | mask-equivalence vs hand-written guards; hot-path timing < 1ms/turn |
| Grammar | `corp/policy/grammar/*.gbnf` | llama.cpp decode-fuzz: 1000 samples all parse |
| Budgets | validator constants | form-count limits honored; documented in the manifest |
| Drift guard | CI | adapter registries ↔ manifest diff check (a predicate added in code without manifest regeneration fails CI) |

**Drift guard is critical**: the manifest is *generated* from the adapter registries, never hand-written. A
predicate added in code without regenerating the manifest fails CI — otherwise the LLM's view of the world
silently diverges from the executor's.

---

## 10. Non-Goals (explicit)

- No parameters/loops/recursion in macros (no computation beyond boolean composition and comparison).
- No arithmetic on game state (comparisons only — the LLM cannot compute `(hp - dmg)`).
- No macro over `do` verbs, param paths, or goal names (macros compose *conditions* only).
- No runtime diff application (mid-episode revision) — v1 is episode-static; the in-game `DeadlockResolver`
  emits diffs that take effect **next episode** (unification with `plan_queue` transient patches is removed).
- No cross-domain macros (a NetHack macro cannot appear in a Craftax diff).
- No deletion of certified goal handlers via diffs (goals enter by certification, leave by human decision).
