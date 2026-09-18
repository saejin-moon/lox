# CORP-Ω: Grammar-Constrained Policy-Diff Synthesis for Symbolic HTN Executors
## Roadmap from current state (NetHack: Median Depth 2–4, Mean Score 281–848) to a General Multi-Domain Method

> **Mission**: Beat AutoAscend (NetHack: Median Depth 10.0 / Mean Score 10,713.6 / 4.8% Valkyrie ascension) and
> demonstrate that an **offline LLM authoring S-expression policy diffs** against a declarative **policy program**,
> executed by a **CPU-only symbolic HTN executor**, is a *general, statistically rigorous protocol for
> LLM-maintained embodied agents* — validated on NetHack (hard showcase) **and 2–3 seeded transfer domains**
> (ablation core), across roles, with bounded self-extension (macros) in the core experiments.
>
> **Status of the four impact trades (all accepted)**:
> 1. Multi-domain benchmark suite — MiniHack/Craftax-class seeded domains join the ablation core; NetHack demoted
>    to hard showcase. Track B (transfer) promoted into the **main results**.
> 2. Frontier API LLMs allowed as the policy **author** (llama.cpp stays the reproducible path). The **executor**
>    remains strictly CPU-only.
> 3. **Macro-extension** (`defmacro`) promoted into the core DSL: the LLM composes new predicates/goals from
>    existing primitives via definitional expansion — bounded self-extension lands in the main paper, not a
>    future-work section. Full Track A (goal-handler authorship) remains post-core.
> 4. **Ascension parity demoted to stretch goal.** The R6 endgame knowledge port is pursued only if the core
>    evidence is already secured; novelty lives in transfer + self-extension + learning, not in the NetHack
>    community headline.
>
> **Honest baseline note**: `NetHackChallenge-v0` forbids seeding — NetHack claims require 100-episode batches.
> Seeded domains get full seed-controlled statistics. AutoAscend's ~2 person-years of human expert labor is the
> comparison axis for the development-loop claim.

---

## 1. Target Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│ OFFLINE LLM REVISION LOOP (runs indefinitely between batches)      │
│  author   : frontier API model (primary) + llama.cpp (repro path)  │
│  input    : run-report bundle (DuckDB) + current policy program    │
│             + revision ledger + domain wiki/docs RAG               │
│  output   : S-EXPRESSION POLICY DIFF (GBNF-constrained, depth ≤ 3) │
└───────────────┬────────────────────────────────────────────────────┘
                ▼ validator gate: grammar → schema → bounds → macro expansion
                  → certification subset → quick-batch non-regression
┌────────────────────────────────────────────────────────────────────┐
│ POLICY PROGRAM  (data/policy_program.json — canonical, versioned)  │
│  policy_params · strategy_plan · tactic_rules · macros             │
│  role/domain profiles · nogoods · provenance/ledger                │
└───────────────┬────────────────────────────────────────────────────┘
                ▼ compile: typed PolicyConfig → HTN method registry
┌────────────────────────────────────────────────────────────────────┐
│ SYMBOLIC EXECUTOR CORE (CPU-only, stable)                          │
│  GoalInterpreter → HTN planner ← Domain Managers ← A*/epistemics   │
│         ▲                                                          │
│  ┌──────┴───────────────────────────────────────────────┐          │
│  │ DomainAdapter ABC — one adapter per environment       │          │
│  │  nethack (existing) · minihack · craftax · (crafter)  │          │
│  └───────────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────────┘
```

**Division of authority (the safety + novelty claim):**
- The LLM expresses **what to value and when** — goals, priorities, thresholds, tactics, and (via `defmacro`)
  *compositions of existing primitives*. It can never express *how to act*: no code emission, no escape hatch.
  Macro expansion is purely definitional — a macro is sugar over already-verified primitives, so verification
  stays trivial and the system remains closed.
- The executor owns *how*: pathfinding, combat mechanics, inventory actions, menu/prompt handling.

---

## 2. The Policy Diff DSL (emission format)

**Decision**: the LLM emits **S-expression policy diffs** — depth ≤ 3 nesting, one form per line — against the
policy program. Storage remains typed JSON. (S-expressions over bespoke line-DSL and raw JSON: one recursive
grammar + ~50-line parser, LLMs know Lisp deeply from training data, trivially GBNF-constrainable, connects to
the program-synthesis / DSL literature — DreamCoder et al. — for the paper.)

### 2.1 Why diffs, and why not raw JSON or code
- Full-document regeneration causes semantic drift in untouched sections (unattributable revisions).
- Long JSON is bracket-fragile and token-expensive at the iteration frequency we need.
- **Code emission is explicitly forbidden** — the safety claim. Macros are the *controlled* widening:
  definitional expansion over verified primitives, never new mechanisms.
- Diffs over a stable program give 1:1 attribution between LLM hypothesis and measured delta — the ablation
  (LLM-revision vs frozen vs random-perturbation-with-identical-gates) stays falsifiable.

### 2.2 Format spec (v1)

```lisp
(revision 15
  (parent 14)
  (author "claude-opus@api")            ; or "llama3.1-70b@local"
  (domain nethack)                       ; diffs are domain-scoped
  (reason "coyote-fight death t2891: hp 23->6 while resting; widen retreat"))

(set policy_params.survival.rest_below_frac 0.65)          ; bounded: 0.40..0.85

;; BOUNDED SELF-EXTENSION: a macro is definitional sugar over existing primitives
(defmacro when_endangered
  (and (adjacent_hostiles) (hp_frac <= 0.40) (not has_healing)))

(rule add tactic_rules
  (when (and (monster "coyote") (when_endangered)))
  (do retreat))

(rule add tactic_rules
  (when (item "throne"))
  (do avoid)
  (note "2 electric-shock deaths in runs 21-22"))

(goal prioritize hunt_poison_res
  (when (and (depth >= 5) (not has_poison_res)))
  (until has_poison_res))

(goal deprioritize enter_sokoban
  (after (failures_in_10_episodes >= 2)))

(nogood add
  (when (and (task "SIT") (depth <= 5)))
  (cause "throne electric shock fatality"))
```

### 2.3 Closed vocabulary (the LLM composes; it cannot invent mechanisms)

| Category | Primitives (bound to code) |
|---|---|
| Predicates | `hp_frac<=/>=`, `hunger>=`, `depth in a-b`, `xl>=`, `lawful`, `has_poison_res`, `has_reflection`, `has_healing`, `minetown_known`, `minetown_visited`, `donations<n`, `stairs_known`, `adjacent_hostiles`, `is_fighting`, `encumbrance<=`, `turn>=`, `failures_in_10_episodes>=`, `(*domain-specific bindings per adapter*)` |
| Param paths | `policy_params.*` (typed leaves; each has min/max bounds in the schema) |
| Tactic verbs | `ranged_only`, `ranged_then_kill`, `retreat`, `retreat_when_wounded`, `avoid`, `kite`, `elbereth_first`, `never_melee` |
| Goal names | `explore_floor`, `forge_excalibur`, `hunt_poison_res`, `enter_sokoban`, `goto_minetown`, `donate_temple`, `descend`, `equip_upgrade`, `shop_sell_loot`, `ascend` (each = a certified goal handler; per-domain sets) |
| Nogoods | `WHEN <predicate> DO forbid(task)` |
| Macros | `defmacro name expr` — pure definitional expansion over the above; expansion happens at validation; expanded forms are re-validated against the same closure |

**Normative spec**: the complete language, macro semantics, exposed-vocabulary manifest, validation pipeline,
error taxonomy, and implementation checklist live in **`MACRO.md`** — the implementation contract for
`corp/policy/dsl.py`, `macros.py`, `validator.py`, `program.py`, and the GBNF grammar files.

### 2.4 Constrained decoding
llama.cpp natively supports GBNF: ship the **s-expression grammar (depth ≤ 3, closed vocabulary)** so invalid
output is impossible at decode time locally. Frontier API paths (the primary author) get structured
output/parse-and-reject with one repair retry. Nothing unparseable reaches the validator.

---

## 3. The Policy Program (canonical storage, `data/policy_program.json`)

Typed JSON with:

1. **`policy_params`** — every tunable the symbolic core reads. Each leaf: `{type, min, max, default, description, tuning_history[]}`.
2. **`strategy_plan`** — ordered goals replacing `MacroAscensionDirector`'s hardcoded 6-phase machine: `{goal, when?, until?, on_fail?, priority}` evaluated top-down each turn by the GoalInterpreter.
3. **`tactic_rules`** — ordered match→policy rules replacing the hardcoded monster sets (`INSTAKILL_NAMES`, `LETHAL_POISON_NAMES`, `HEAVY_HITTERS`) and inline guard branches.
4. **`macros`** — definitional expansions validated at compile time (depth-bounded, closure-checked).
5. **`role_profiles` / `domain_profiles`** — per-role overlays (NetHack) and per-domain overlays (transfer suite): weapon tiers, keep-distance flags, shop behavior, skill priorities.
6. **`nogoods`** — current `data/nogoods.json` folded in.
7. **`provenance`** — version, parent, author model, prompt/revision ids, accept/reject history with measured deltas and cost.

---

## 4. Complete Codebase Refactor Blueprint

This is the engineering contract. Migration order is dependency-ordered; every behavior-preserving step must
reproduce decision traces on certification episodes before the next step lands.

### 4.1 Target module tree

```
corp/
├── policy/                          # NEW — the entire LLM-facing surface
│   ├── config.py                    # PolicyConfig typed tree + bounds registry + (de)serialization
│   ├── program.py                   # PolicyProgram Pydantic models + versioned load/save + overlay merge
│   ├── predicates.py                # Named predicate registry (replaces/adapts planner/predicates.py bitmask compiler)
│   ├── macros.py                    # defmacro definitional expansion + closure checker
│   ├── dsl.py                       # S-expression reader (depth ≤ 3) → typed diff objects
│   ├── grammar/                     # GBNF grammar files (nethack.sexpr.gbnf, minihack.sexpr.gbnf, ...)
│   ├── macros.py                    # defmacro definitional expansion + closure checker
│   ├── grammar/                     # GBNF grammar files (nethack.sexpr.gbnf, minihack.sexpr.gbnf, ...)
│   ├── validator.py                 # Full gate pipeline (parse → schema → bounds → certs → quick-batch)
│   ├── reviser.py                   # Run-report bundle builder + provider calls + diff proposal
│   ├── goal_interpreter.py          # strategy_plan evaluator → navigation directives
│   ├── profiles.py                  # role/domain overlay resolution
│   ├── ledger.py                    # Revision ledger (append-only JSONL): cost, latency, accept/reject, deltas
│   ├── corpus.py                    # Per-domain RAG corpus: ingest + FTS index + rag_context slices
│   └── report.py                    # Run-report bundle builder (DuckDB → death taxonomy, stall census, goal stats)
├── executor/                        # NEW — the domain-agnostic boundary
│   ├── interface.py                 # DomainAdapter ABC + DomainSpec (see 4.3)
│   ├── registry.py                  # adapter lookup by domain name
│   └── nethack/                     # adapter facade over the existing managers (thin!)
├── agent/                           # corp_agent.py slims to: loop + priority cascade + delegation
├── domain/                          # existing managers (nethack-specific, unchanged behavior)
├── deliberative/                    # AutopsyEngine + DeadlockResolver rewritten onto diff contract
├── env/ · navigation/ · planner/    # existing (planner/predicates.py migrates into policy/predicates.py)
├── telemetry/                       # existing + goal_events table + revision ledger views
└── workers/dispatcher.py            # existing
scripts/
├── run_benchmark.py                 # --domain flag; adapter-aware
├── run_baseline_suite.py            # per-domain baselines
├── run_revision_loop.py             # NEW — the unattended loop
└── run_ablation.py                  # NEW — three-arm harness (LLM-revision | frozen | random-gated)
data/
├── policy_program.json              # canonical program (versioned, git-tracked)
├── policy_program/                  # version history
└── revision_ledger.jsonl            # append-only
```

### 4.2 Refactor steps (dependency-ordered)

| # | Step | Files touched | Behavior change | Gate |
|---|---|---|---|---|
| 1 | Commit current state; start ledger | git, `data/revision_ledger.jsonl` | none | — |
| 2 | `policy/config.py` extraction: ~80–100 tunables out of managers | nav/combat/inventory/guards/corp_agent | **none** | decision-trace equivalence on certification episodes |
| 3 | `policy/program.py` + `policy/program.json` (defaults == current behavior) | new | none | equivalence |
| 4 | `policy/goal_interpreter.py` replaces `MacroAscensionDirector` phases; directives via existing `get_navigation_directive()` seam | macro_director → goal_interpreter, nav_manager seam | none | equivalence + `goal_events` DuckDB table |
| 5 | `policy/predicates.py` named registry (migrate `planner/predicates.py`) | planner/predicates → policy/predicates | none | equivalence |
| 6 | `policy/dsl.py` + `grammar/` + `policy/macros.py` | new | new capability (off by default) | grammar unit tests + closure checker tests |
| 7 | `policy/validator.py` + `policy/report.py` + `policy/ledger.py` + `policy/reviser.py` | new | new capability | gate pipeline tests; repair-retry tests |
| 8 | `scripts/run_revision_loop.py` + `scripts/run_ablation.py` | new | new capability | 10-revision unattended dry-run with MockProvider |
| 9 | `deliberative/` unification: AutopsyEngine + DeadlockResolver emit diffs through the validator | deliberative/* | low (same intent, new format) | existing deliberative tests adapted |
| 10 | `executor/interface.py` DomainAdapter ABC; wrap NetHack managers as `executor/nethack/` (thin facade) | new + thin re-exports | none | NetHack benchmark unchanged |
| 11 | `executor/minihack/` adapter (first transfer domain) | new | new capability | MiniHack certification set |
| 12 | `executor/craftax/` (or Crafter) adapter (second transfer domain) | new | new capability | domain-2 certification set |
| 13 | Tactic-rules evaluator in `combat_manager.py`; profiles in `policy/profiles.py` | combat_manager, benchmark | none-by-default | equivalence on default program |
| 14 | Per-domain RAG corpus build (`policy/corpus.py` + `scripts/build_corpus.py` → `data/corpus/<domain>/`): ingest docs/paper/source-docstrings, FTS index, RAG slices wired into `adapter.rag_context()` and the reviser prompt | new | new capability | corpus coverage check (task/achievement lists present, slices non-empty); author-prompt smoke test |

### 4.3 The `DomainAdapter` ABC (the generalization contract)

```python
class DomainAdapter(ABC):
    spec: DomainSpec                      # name, obs vector layout, grid shape, action space
    def predicates(self) -> dict[str, PredicateBinding]   # domain-specific named predicates
    def goal_handlers(self) -> dict[str, GoalHandler]     # certified goal handlers
    def make_env(self, seed: int | None, **cfg) -> EnvLike
    def report_bundle(self, duckdb) -> dict               # run-report for the reviser
    def certification_suite(self) -> list[CertCase]       # per-domain skill cases
    def rag_context(self) -> list[str]                    # wiki/docs slices for the author LLM
```

The policy diff DSL references **only** what an adapter's `predicates()` and `goal_handlers()` expose — the
same grammar, different bindings. Porting a domain = implementing the adapter + writing its certification set;
the loop, validator, DSL, and program machinery are untouched.

### 4.4 Interfaces the refactor must preserve (regression anchors)

- `CORPAgent.select_action()` priority ordering and all documented interlocks (AGENTS.md §3) — untouched
- `get_navigation_directive()` seam signature — GoalInterpreter plugs in here
- Dispatcher fiber semantics (`THROW`/`EAT`/etc. multi-key handling, prompt interception in `AutoMoreWrapper`)
- Telemetry schemas (ticks/episodes) + `goal_events` added, never removed

---

## 5. Phases R0–R9

### R0 — Checkpoint & Measurement (½ day) — **[2026-09-18 DONE]**
- [x] Commit ALL uncommitted work → commits `a39a560` (core), `891deb5` (bench+tests), `cd6dc98` (docs); + `.gitignore` exceptions for policy program/ledger
- [x] NetHack baselines **launched** (background, nice-10): CORP 100ep@20k (`data/r0_baseline_corp.log` → `data/baseline_corp_100ep.json`), then AutoAscend 100ep@50k val (`data/r0_baseline_autoascend.log` → `data/autoascend_val_100ep.json`). NOTE: earlier 100-ep attempts (Sep 18 09:47) died at startup with 0 episodes — verified dead and relaunched. Freeze as reference when both finish.
- [x] Trajectory reconstructed → `data/trajectory.csv` + `data/trajectory.png` (106 runs, Sep 15 → Sep 18: mean score 2→850, best-ever run `7dQBM4`: depth 11, score 2,933). Script: `scripts/trajectory.py`
- [x] Ledger started → `data/revision_ledger.jsonl` (schema v1 recorded in init entry)

### R1 — PolicyConfig Extraction (2–3 days) — **[2026-09-18 DONE — 1 day]**
> Wired: ~75 tunables across guards/navigation/combat/inventory/macro_director/corp_agent via typed
> `PolicyConfig` (`corp/policy/config.py`) with `defaults()` byte-identical to pre-R1 behavior; `CORPAgent`
> accepts `policy_config=` and propagates to all managers. Smoke test: 2-ep live run in expected distribution
> (DL 5/1, scores 413/288, SPS ~650 — no perf regression). Remaining literals are domain *data* (name sets)
> reserved for R4 tactic rules.
> Original acceptance criteria:
Refactor steps 2–3. Extract tunables from `navigation_manager.py` (HP descent gates 0.45/0.70, rest 0.60/0.85, search caps 15/20/6/5, food radii 8/99/10/15, door caps), `combat_manager.py` (retreat 0.25, speed ratio 1.3, wounded-vs-heavy 0.65, ranged cooldown 50, escape grace), `inventory_manager.py` + `guards.py` (nutrition priorities, prayer gaps 301/850/450, eat-before-descend radius 10), `corp_agent.py` (triage 0.55/0.60, zero-turn guard 4).
**Accept**: 183+ tests pass; default program reproduces decision traces byte-for-byte.

### R2 — Policy Program + Goal Interpreter (3–4 days)
Refactor steps 4–5. Strategy-plan interpreter replaces `MacroAscensionDirector`; named predicate registry; `goal_events` DuckDB table feeding `failures_in_10_episodes`.
**Accept**: default program equivalence; goal telemetry live.

### R3 — Revision Loop + Diff DSL + Macros (3–4 days) — **LLM goes live**
Refactor steps 6–8. Dual author (frontier API primary + llama.cpp GBNF repro path); **tiered authorship**:
frontier for cold-start programs and stalled loops, mid-tier (Gemma-class ~30B via OpenRouter, or local) for
routine tuning — acceptance rate per author model is tracked in the ledger and becomes its own ablation ("how
smart must the policy author be?"). `defmacro` machinery with closure checker; full validator gate pipeline;
unattended `run_revision_loop.py`; deliberative unification (step 9). RAG: NetHack slices from
`data/wiki_index.db`; transfer-domain corpora arrive in R5 (before their author prompts go live).
**Accept**: ≥10 gated revisions unattended (dry-run first with MockProvider); ≥1 accepted revision with measured improvement on the next batch; 0 rejected-revision leaks.

### R4 — Tactic Rules + Profiles + Ablation Harness (3–4 days)
Refactor steps 13 + profiles. Three-arm harness (`run_ablation.py`): LLM-revision | frozen | random-perturbation-with-identical-gates — **this control arm is what makes every later claim falsifiable**.
**Accept**: harness runs unattended; tactic-rule parity on default program; all-role reports.

### R5 — Domain Suite + Tuning Campaign (2–4 weeks) — **transfer becomes a main result**
Refactor steps 10–12 + 14. Port `DomainAdapter` to MiniHack, then Craftax/Crafter. Per-domain: **RAG corpus
build first** (`policy/corpus.py` + `scripts/build_corpus.py` → `data/corpus/<domain>/`; NetHack reuses
`data/wiki_index.db`; MiniHack = NetHack wiki + MiniHack task docs; Craftax = official docs + paper + source
docstrings + achievement/symbol tables — no community wiki exists, the corpus is self-built and load-bearing
there since author models have thin parametric knowledge of it) → certification set → cold-start program →
100-ep three-arm batches. NetHack tuning campaign continues nightly in parallel (100-ep batches driving
revisions until plateau: target median depth ≥ 5, mean score ≥ 1,500).
**Accept**: improvement curves on ≥2 transfer domains from cold-start programs; NetHack batch improvement beyond R0 baseline σ; **workshop paper checkpoint**.

### R6 — STRETCH: Ascension Knowledge Stack (4–8 weeks, only if R5 secured)
The NetHack-community headline, demoted per Trade 4. Ordered by mortality impact: survival intrinsics/MR (DL 8–12), armor/weapon upgrade loop (alone should push median depth 6–8), Gehennom survival, Castle/wishing, Vlad → Candelabrum → Invocation → Ascension run, role quest branches.
**Accept**: ascension rate ≥ AutoAscend's 4.8% on Valkyrie (100-ep batches); each milestone certified before the LLM may re-prioritize around it.

### R7 — Cross-Role Generalization (2–4 weeks, overlaps R6)
`role_profiles` refinement by the loop (parameters generalize; capabilities need building): armor/weapon tiers generalize across fighters first. Spellcasting infrastructure (wizard/priest/healer) is **scope-gated**: required only if the paper claims all-role parity — otherwise report role-coverage honestly.
**Accept**: fighter roles median depth ≥ 8 on 50-ep batches; ≥2 roles ascending (if R6 pursued); honest role-coverage table otherwise.

### R8 — Track A: LLM-Authored Goal Handlers (research stretch)
The full self-extension: the LLM proposes new *goal handlers* (declarative sub-programs + action schemas) against domain RAG; verified in shadow/certification harness before entering production. Only after the closed-vocabulary loop has a long, clean acceptance ledger.
**Accept**: ≥3 LLM-proposed goal handlers certified and shipped with measured contribution; 0 unverified handlers in production.

### R9 — Paper
- **R5 checkpoint (workshop)**: "Agent-as-Developer" — trajectory figure, ledger, DSL artifact
- **Main-track**: *"Grammar-constrained policy-diff synthesis: offline LLMs as optimizers of declarative agent policies across domains"* — requires (a) three-arm ablations with seeds, (b) ≥2-domain improvement curves, (c) NetHack ≥ AutoAscend at mid-game (ascension parity if R6 pursued), (d) cross-role evidence. If Track B domains stall, fall back to AAAI/IJCAI/CoG
- **Nogood learning** framed as online constraint accumulation (ablated)

---

## 6. Honest Diff from AutoAscend

| Dimension | AutoAscend | CORP-Ω |
|---|---|---|
| Control architecture | Priority-ordered expert strategies (human-authored, frozen) | Same genre, compiled from a declarative policy program |
| Strategy authorship | ~2 person-years expert labor | Offline LLM policy diffs + macros + human plan sketches, gated revision loop |
| Learning across episodes | None (tabula rasa) | CDCL nogoods + gated policy revisions (failures become permanent knowledge) |
| Knowledge | Hardcoded ID tables | Named predicate vocabulary + RAG-grounded authorship + Bayesian epistemic gates (BUC, Shannon) |
| Scope | NetHack-specific | DomainAdapter boundary — same loop demonstrated on multiple environments |
| Must cite / ablate | — | FunSearch/AlphaEvolve (propose-verify loops), Voyager (skill code — we forbid), DSPy/TextGrad (prompt-space); lineage of ported tables; random-gated control |

**Not claimed as novel**: the priority-cascade executor, HTN decomposition, A* exploration, NetHack tactics.

---

## 7. Verification Discipline

```bash
uv run pytest                                        # 183+ tests, never skip
./scripts/run_skill_certifications.sh                # NetHack per-skill cases
uv run python scripts/run_benchmark.py --domain nethack --episodes 10 --max-steps 20000 --role valkyrie --clean-parquet
uv run python scripts/run_baseline_suite.py --episodes 100 --step-limit 50000 --role val
uv run python scripts/run_revision_loop.py --max-revisions 10 --author api --repro local
uv run python scripts/run_ablation.py --arms llm,frozen,random --episodes 100 --domain all
```

- Refactor steps marked "none" require decision-trace equivalence on certification episodes
- Every accept/reject committed to the ledger; every program version in git
- Claims: seeded domains at 100-ep batches with seeds; NetHack at 100-ep batches (σ acknowledged); 10-ep = engineering signal only
- RAG corpus coverage check per domain before author prompts go live (task/achievement lists present, slices non-empty, index freshness vs source commit)

---

## 8. Known Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LLM overfits lucky episodes (NetHack unseedable) | Gated acceptance on 100-ep batches; seeded transfer domains carry the statistical claims; random-gated control arm |
| Refactor regressions (steps 2–5 touch everything) | Decision-trace equivalence + certification suite before any behavior change; one step per commit |
| Transfer domains stall (adapters costlier than projected) | MiniHack first (shares NetHack's glyph interface — cheapest port); NetHack-only evidence still carries the workshop paper |
| Frontier-API author drifts / unsafe proposals | Grammar + closure validation + certification gate; macros are definitional sugar only; no code path exists |
| Loop plateaus (accepts nothing) | Report bundle includes gate-rejection reasons; oscillation detector freezes param subtrees; author-model switch |
| Ascension stack slips (R6 demoted by design) | Paper's core claims (transfer + self-extension + learning) are R4/R5-complete; R6 only upgrades the showcase |
| Cost tracking | Ledger records provider-reported tokens + wall-clock from R0; labor claims scoped to wall-clock |
