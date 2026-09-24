# LOX-ψ Gen-2: Policy Synthesis Engine (LLM-Oriented Creation of Symbolic policies)
## Roadmap from a gated revision loop to a knowledge-growing agent that beats AutoAscend — and its equivalents everywhere else

> **Mission**: an LLM author — given *everything* (full environment schema, every action, all run telemetry,
> trajectories, the wiki) — **grows a declarative policy program** through a safety-gated synthesis loop until
> the resulting agent decisively beats AutoAscend on NetHack (median depth 10.0 / mean score 10,713.6 /
> 4.8% Valkyrie ascension) and the strongest published equivalents on MiniHack and Craftax.
>
> **Non-negotiables**: (1) the contribution is fully novel — no expert HTN is copied (no AutoAscend lineage);
> (2) humans provide only primitives, interlocks, and certification gates; (3) the LLM's only write path is a
> grammar-constrained diff validated by machine-checkable gates; (4) CPU-only execution.
>
> **Honest baseline note**: `NetHackChallenge-v0` forbids seeding → NetHack claims need 100-ep batches
> (σ acknowledged). Seeded domains (MiniHack, Craftax) carry full seed-controlled statistics.

---

## 0. Why AutoAscend is beatable (structural argument)

AutoAscend is a static expert system with four structural weaknesses, each mapped to a pillar below:

| AutoAscend weakness | Our attack |
|---|---|
| Fixed knowledge, frozen at ship time | **Evolution engine**: knowledge grows every campaign; nogoods + program revisions compound |
| Single strategy, no self-observation | **Evidence engine**: trajectories, failure counters, counterfactual probes → the author sees failure modes and rewrites strategy |
| One program | **Population search**: K candidates evaluated in parallel (~6k NetHack eps/hr); winners breed the next generation |
| ~2 person-years of hand knowledge | **Author agent + RAG**: the wiki (FTS5, full 3.6.6 dump) + telemetry give the LLM the raw material experts used — acquired in machine-time, not person-years |

Parity is the crossover; the design goal is that our loop's improvement rate **compounds** while theirs is zero.

---

## 0.5 WHERE WE ARE (state of record, 2026-09-24) — read before working

**Suite**: **578 passing tests (~18s)** (`uv run pytest -q`). Telemetry: **7.2M+ ticks in DuckDB**.
**Program**: live NetHack program = **v6** (`data/policy_program.json`; interlock-only tactic rules — the 3
llm-authored combat rules were reverted after an A/B showed they cost 45% mean score). MiniHack program v4.
**Current NetHack baseline band (100-ep batches, valkyrie, v6 rules)**: mean score **330–396**, median depth
**2–3** (unseeded batch σ is wide; the 396.4 control ran on pre-refactor code — treat 396 as the optimistic
edge). Death taxonomy: starvation faints 18–28% (chronic, NOT fixed — the food chase is locked OFF, see
AGENTS.md §4.14), DL-1–3 melee bites dominant, occasional prayer-prompt deaths. **128 of ~330 recent episodes
died at depth 1 after a median of ~4,400 turns** — stalling + early survival is the wall, not turn budget.
AutoAscend empirical foil: median depth 10.0 / mean score 10,713.6 / 4.8% ascension (5 eps @ 20k).

### Autonomous Agent Handoff: Immediate Next Run (MiniHack Campaign)
Any agent picking up this workspace can immediately execute the MiniHack synthesis run:
1. **Launch local vLLM on RTX 3080 Ti (12GB VRAM)**:
   ```bash
   uv run vllm serve groxaxo/Huihui-Qwen3.5-9B-abliterated-GPTQ-Pro-4bit-g64 \
     --quantization gptq_marlin \
     --dtype float16 \
     --trust-remote-code \
     --host 0.0.0.0 \
     --port 8000 \
     --max-model-len 32768 \
     --gpu-memory-utilization 0.90 \
     --enforce-eager \
     --disable-uvicorn-access-log
   ```
2. **Execute Agentic Authoring Session (Pure Pythonic Infix AST)**:
   ```bash
   uv run python scripts/run_author_session.py \
     --provider llama_cpp \
     --base-url http://localhost:8000/v1 \
     --model groxaxo/Huihui-Qwen3.5-9B-abliterated-GPTQ-Pro-4bit-g64 \
     --domain minihack \
     --no-commit
   ```
3. **Execute 100-Episode MiniHack Batch (30 CPU Workers)**:
   ```bash
   uv run python scripts/run_minihack_batch.py \
     --task MiniHack-ExploreMaze-Easy-Mapped-v0 \
     --episodes 100 \
     --jobs 30
   ```

**Decision ledger (do not re-litigate)**:
- **Pure Pythonic Infix AST (Zero S-Expressions) & AST-to-HTN Execution**: All outer Lisp parentheses eliminated from authoring surface and prompt (`data/prompts/v6.md`). Infix conditions (e.g. `stairs_known and not is_fighting`, `monster == "j" and hp_frac <= 0.35`) supported directly via sandboxed AST parsing (`lox/policy/infix.py` `parse_pythonic_diff` -> `dsl.py` -> `ast_to_canonical_expr`) with zero parenthesis hallucination, depth <= 3 bound, and direct AST evaluation in `GoalInterpreter` and `TacticRuleEngine` feeding HTN planner methods.
- **Context Capacity to 128k Tokens**: `MAX_TRANSCRIPT_CHARS=450000` (~128k tokens) and `MAX_TOOL_OUTPUT_CHARS=32000` (~9k tokens), allowing massive SQL queries and 2000-tick trajectory autopsies without truncation.
- **No AutoAscend port** — the contribution is grown knowledge beating engineered knowledge; porting an
  expert HTN falsifies the claim and breaks the baseline-free novel-env story (user decision, final).
- **1a**: combat-verb tactic rules MUST be target-conditional — `(monster "…")`/`(item "…")` required in
  `when` (validator invariant + author prompt). Macro bodies are parameterless (no string atoms), so the
  target match lives in the rule's own `when`; macros compose state conditions around it.
- **1b**: tiered quick-batch gate — rule-ADD revisions gate on 10ep×10k; param-only stay 3ep×5k
  (`make_quick_batch_gate(args, parent_program=…)` in run_revision_loop.py).
- **1c**: transfer report bundles carry per-seed spread; report-episodes defaults 10.
- **Storage**: per-file `.sexpr` authoring tree (`data/program/<domain>/`) + compiled JSON artifact
  (`lox/policy/compiler.py` → `data/compiled/<domain>.json`). Executor contract unchanged.
- **Naming**: **LOX-ψ** (LLM-Oriented Creation of Symbolic policies).
- **Food chase stays disabled** (dose-response A/B, 4×100 eps: none 396.4 > tight 275.4 > design 310.8 >
  wide 250.0; faints RISE with chase capability). Locked by `tests/test_food_security.py`.

**Session-critical facts**:
- Providers: `.env` holds `GEMINI_API_KEY`; default model `gemma-4-26b-a4b-it` (works on Google AI Studio;
  `gemini-2.5-flash`/`gemma-3-27b-it` 404 for this account). Revision-loop/ablation scripts auto-load `.env`.
  `generate_text` = native google-genai with thinking-high (env-tunable); full revision call ≈ 90–130 s.
- **MiniHack/NLE resets are NOT seed-deterministic** (same seed → different episode steps across runs,
  measured). Transfer gates on small samples wobble around thresholds; never gate a single comparison on 3 eps.
- `run_parallel_batch.py` `--jobs` default is conservative (8) — campaigns must pass **`--jobs 30`**
  (36 cores; ~6k NetHack eps/hr; leave 4–6 cores headroom).
- Flake protocol: if exactly one test fails, rerun once before diagnosing (two known timing-sensitive
  tests: combat-manager suites; a full-suite order dependence was observed once, not reproduced).
- `docs/architecture/*.md` are LEGACY (pre-Gen-2 topology, old naming) — superseded by AGENTS.md §2; do not
  trust or rename them; they get rewritten in S1.
- `MACRO.md` says macros are "parameterless" (line 276) but also describes parameterized substitution (§153):
  the IMPLEMENTATION is parameterless today; parameterization is the S3 ladder's first rung.
- MockProvider canned diffs are domain-aware and model the corrected author idioms (target-conditional
  rules); keep them aligned when the DSL changes.

---

## 1. Target architecture (already partially built)

```
AUTHOR AGENT (agentic LLM: gemini primary, llama.cpp repro)
  tools: query_duckdb(sql, read-only) · wiki_search(q) · read_trajectory(ep, turn_range)
         read_env_schema() · read_manifest() · read_program_tree()
  output: S-expression policy diff → VALIDATOR GATES → authoring tree commit
POLICY PROGRAM  data/program/<domain>/  (per-file .sexpr workspace; compiled to JSON)
EVOLUTION ENGINE  population × parallel gated evaluation → selection (ledger lineage)
EXECUTOR  unchanged contract: compiled JSON → GoalInterpreter → HTN → managers
CURRICULA  data/curricula/<skill>/ — short targeted episodes for fast per-skill iteration
```

### 1.1 Storage decision (made): per-file tree + compiled artifact

The LLM authors into **individual `.sexpr` files** under `data/program/<domain>/`
(`macros/*.sexpr`, `rules/*.sexpr`, `goals/*.sexpr`, `handlers/*.sexpr`, `params.json`, `nogoods.jsonl`).
Rationale: per-unit git diffs, per-file provenance, natural review units for humans, no monolithic merge
conflicts for the evolution engine, and file identity = rule identity. `lox/policy/compiler.py` assembles +
validates the tree into `data/compiled/<domain>.json` — the executor's contract is unchanged (zero executor
rewrites). The validator remains the ONLY writer of accepted trees; every compiled artifact is archived under
`data/programs/archive/`.

### 1.2 The Policy ISA (S1 — the big enabler)

The strategy content of the current hardcoded cascade (~4k lines across managers) is re-expressed as the
default program in the DSL. The ISA:

- **Predicates** (30+ today): hp_frac, hunger, depth, xl, inventory/weapon/armor state, monster/item
  presence, dungeon topology, epistemic flags (BUC, identity), failure counters.
- **Verbs** (8): ranged_only, ranged_then_kill, retreat, retreat_when_wounded, avoid, kite, elbereth_first,
  never_melee — target-conditional only (v6 invariant).
- **Goals** (13 today, extensible by certification): ordered strategy_plan entries with when/until/directive.
- **Behavior primitives** (NEW, S1/S3 — the plan vocabulary): goto(tile_class), pickup, eat(slot|floor),
  wield/wear/puton/quaff/read/zap/apply(slot), engrave(text), descend/ascend, pray, search, attack, wait —
  each mapped to existing dispatcher Tasks with machine-readable preconditions/effects in the manifest.
- **Macros**: parameterized in S3 (MACRO.md already specifies substitution semantics); boolean composition
  only — never behavior.
- **Handlers** (S3): declarative sub-programs `(goal-handler <name> (when …) (plan <primitive>+) (until …))`
  proposed by the LLM, shadow-validated, certified, then promotable into goals.
- **Params** (~100 typed leaves, bounds-enforced).

Hardcoded code retains ONLY the interlocks (AGENTS.md §4) and the mechanisms. Everything the LLM may want to
change is data.

### 1.3 Evidence engine (S0 — everything the author sees)

- `query_duckdb(sql)` — read-only author tool over 6.6M+ ticks, episodes, goal_events, death taxonomy.
- `wiki_search(q)` — the full NetHack 3.6.6 wiki FTS5 dump (the manual the experts used).
- `read_trajectory(ep_id, turn_range)` — flight-recorder markdown autopsies (per-episode, per-death last-N).
- `read_env_schema()` — every action, its dispatcher task, preconditions, effects; every observation channel.
- `read_manifest()` / `read_program_tree()` — the full vocabulary and the current program source.
- Goal-failure counters wired from goal_events into the condition vocabulary (deprioritize what keeps
  failing); per-death state dumps; stall census.
- **Counterfactual probes** (S2): candidate params re-evaluated on death-adjacent situations (paired batches
  on NetHack; seeded replay on transfer domains).

### 1.4 Evolution engine (S2)

- Population of K candidates as program trees; each generation: LLM proposes per-candidate mutations
  (conditioned on that candidate's evidence) → gates → parallel tiered evaluation → selection by gated
  metrics (NetHack: mean score + median depth, 100-ep CIs; curricula: skill pass-rate; transfer:
  steps-on-success / achievement curves) → winners become the base; losers archived with full lineage.
- This is FunSearch/AlphaEvolve's selection shape applied to **executable agent policies** — with the
  difference that our mutation operator is an evidence-conditioned agentic author, our search space is a
  *declarative, safety-bounded policy language*, and our evaluation is whole-episode agent performance with
  interlock invariants.

### 1.5 Curricula (S4)

`data/curricula/<skill>/`: miniature scenarios isolating one skill (retreat discipline, corpse timing,
altar BUC, wand usage, store etiquette, boulder routes). MiniHack custom levels for transfer-usable skills;
NetHack fixture episodes for NetHack-specific ones. The loop iterates against 30-second episodes (~100× the
full-game iteration rate); certified improvements deploy to the full game. Train on synthetic, evaluate on
vanilla — no test contamination.

### 1.6 Prompt program (S0, continuous)

`data/prompts/<version>.md` — the system prompt is a first-class, versioned artifact. `run_prompt_ab.py`
evaluates prompt candidates by (a) author-acceptance rate, (b) validator-reject taxonomy, (c) downstream
program performance on fixed eval batches. The prompt iterates under the same gated discipline as the
program. Expected: many iterations; the prompt is where the author's *judgment* lives (e.g., "descend before
detours", "never propose catch-all combat rules", "prefer wiki-cited rationales").

---

## 2. Phases (S0–S6), each with entry/exit gates and kill criteria

### S0 — Evidence + Prompt Harness & High-Performance Spine (≈1 wk) — NEXT UP
- [ ] Author agent with tool loop (`run_author_session.py`): query_duckdb, wiki_search, read_trajectory,
      read_env_schema, read_manifest, read_program_tree → one validated diff per session.
- [ ] Wire goal-failure counters (goal_events → condition vocabulary) + trajectory slices in bundles.
- [ ] `run_prompt_ab.py` + first prompt-iteration campaign (≥5 prompt versions evaluated).
- [ ] **Spine Acceleration**:
      - `@njit` accelerate `GridAStar.find_path` (`astar.py`) and `FrontierExplorer` BFS (`frontier.py`) — target <20 µs.
      - Static 512-entry NumPy LUT in `threat_scan.py` (eliminate `glyphs.tobytes()` and `permonst` C-API allocations).
      - Profile decision latency: drop from 1,444 µs to <250 µs/turn, pushing throughput to >8k eps/hr on 30 cores.
- [ ] Re-run the 3-arm minihack ablation with the agentic author (post-1a/1b/1c gates).
- **Exit**: agentic author measurably outperforms the bundle-only author; decision latency <300 µs; test suite 100% green.
- **Kill**: no improvement → the bottleneck is expression, jump S1 authoring-tree work early.

### S1 — Policy ISA, Monolith Modularization & Core Primitives (2–3 wk)
- [ ] `lox/policy/compiler.py` + authoring tree migration (v6 → tree; compiled artifact byte-equivalent).
- [ ] **Modularize `NavigationManager`**: Split 1,212-line monolith into `stair_routing.py`, `search_policy.py`,
      and `feature_navigation.py` to eliminate 4,400-turn perimeter wall search loops causing DL1 stalls.
- [ ] **Mechanical Primitive Scaffolding (The AutoAscend prerequisites)**:
      - `safe_fountain_dip`: uncursed long sword dipping for Excalibur at XL ≥ 5 with water demon abort.
      - `corridor_funnel`: strict retreat-to-doorway behavior against fast/poison biters.
      - `price_id`: shopkeeper buy/sell price testing for scroll/wand/potion identification.
- [ ] Behavior-primitive manifest (preconditions/effects for every dispatcher Task).
- [ ] Re-express the hardcoded cascade as the default program (decision-trace equivalence, per-manager).
- **Exit**: default program equivalent to today's agent; all 544 tests green; perimeter search stalls eliminated;
  the LLM can express every strategy decision the cascade used to hardcode.
- **Kill**: any cascade behavior that cannot be expressed without weakening an interlock → that behavior
  stays in code (guardrail), ISA narrows honestly.

### S2 — Evolution Engine (2 wk)
- [ ] `run_evolution.py`: population of K candidates, per-candidate evidence, gated tiered evaluation, selection, lineage.
- [ ] Counterfactual probe tooling (seeded replay on transfer domains; paired batches on NetHack).
- **Exit**: evolution beats the single-program loop on minihack (seeded, p<0.05 on steps-on-success) and
  does not regress NetHack vs the frozen v6 control (329–396 band, 100-ep batches).
- **Kill**: population search ≤ single-program after 3 generations → author/prompt iteration first.

### S3 — Handler Ladder & Deterministic Sokoban Solver (2–4 wk)
- [ ] Parameterized macros (validator substitution semantics already specified in MACRO.md).
- [ ] **Deterministic Sokoban Solver Primitive**: implement boulder-push A* solver as a certified handler
      (guaranteeing Sokoban completion and bag/reflection prize without human heuristic creep).
- [ ] Handler-plan schema + shadow harness (sandboxed episodes, side-effect telemetry, ≥50 shadow eps,
      0 invariant breaks, measured contribution vs frozen).
- [ ] ≥3 LLM-authored handlers promoted with measured contribution; 0 unverified handlers in production.
- **Exit**: at least one handler the handwritten layer lacked, with a measured win.
- **Kill**: promoted handlers never beat handwritten equivalents → widen the primitive set; re-examine
  whether plans need iteration constructs (bounded loops) — still no general code.

### S4 — NetHack AutoAscend Push (ongoing, mortality-driven)
- [ ] **Survival floor**: eliminate DL1 deaths (128/~330 episodes died at depth 1, median 4.4k turns wasted) →
      median depth ≥ 5, mean score ≥ 1,500 on 100-ep batches.
- [ ] **Mid-game**: AC/MR/weapon curves (Excalibur + Minetown divine protection + Sokoban prize) → median depth ≥ 8.
- [ ] **Endgame** (R6 milestones 4–7 as certified handlers): Medusa mirror → Castle drawbridge striking →
      Wand of Wishing protocol → Gehennom → Vlad → Invocation → Ascension.
- [ ] **Ascension rate ≥ 4.8% (parity)** then **≥ 15% (dominance)** and mean score ≥ 10,713 (parity) then
      ≥ 2× (dominance), on 100-ep batches with bootstrap CIs.
- **Kill**: 3 consecutive stalled generations at any stage → escalate ISA expressiveness or curriculum
  coverage before more compute.

### S5 — Transfer Dominance: Craftax & MiniHack (parallel with S4)
- [ ] **`CraftaxAdapter` Implementation**:
      - Observation decoder: decode 8268-float array into player status, inventory DAG, and 63x63 local grid.
      - **JAX Vectorized Batching**: Implement `CraftaxBatchWrapper` using `jax.vmap` to eliminate single-step
        CPU dispatch overhead (17.2 ms/step → <0.3 ms/step in batches).
      - Primitives: `mine(block)`, `craft(recipe)`, `drink()`, `eat()`, `attack(mob)`, `sleep()`, `place_torch()`.
      - Goal handlers: `gather_wood`, `craft_pickaxe`, `mine_stone`, `craft_sword`, `survive_night`, `dungeon_crawl`.
- [ ] **Craftax cold-start** (corpus self-built from source docstrings): beat published Craftax baselines:
      - Craftax-Classic: beat PPO (~15%) and CALM (~35%) on achievement curve.
      - Full Craftax: surpass published RL (<5%) across the 67-achievement progression tree.
- [ ] MiniHack: dominance on steps-on-success + extension beyond ExploreMaze (combat/procgen task families).
- **Exit**: ≥2 domains where the *same synthesis machinery* beats env-specific published agents.

### S6 — The Breakthrough Paper (Solo-Author Oral Target)
- Target Venues: **NeurIPS / ICML / ICLR Oral** or **Nature Machine Intelligence**.
- Main-track claim: *"Evidence-conditioned policy synthesis: a safety-gated LLM grows an agent program that
  surpasses years of human expert engineering"* — requires NetHack dominance over AutoAscend + Craftax transfer.
- Airtight Evidence Package:
  1. **Clean-room guarantee**: Zero AutoAscend code/tables imported; full audit trail.
  2. **Mandatory 3-arm ablation**: `LLM-Grown Policy > Frozen Initial Policy > Random Mutation Policy`.
  3. **Multi-domain transfer**: Same synthesis loop dominates on NetHack (ancient C), MiniHack, and Craftax (modern JAX).
  4. **Open-source audit artifact**: Append-only `data/revision_ledger.jsonl` with every prompt, diff, and evaluation trace.

---

## 3. Novelty ledger (what is and is not ours)

**Not novel (must cite, must ablate)**: FunSearch/AlphaEvolve (evolution + evaluators — ours differs:
declarative safety-bounded policy language, agentic evidence-conditioned author, whole-agent evaluation);
Voyager (LLM writes skill *code* — we forbid code entirely); DSPy/TextGrad (prompt-space optimization — we
optimize a policy artifact); grammar-constrained decoding (standard); NetHack LLM benchmarks (Balrog et al. —
they *play*, they don't grow a program); CALM (LLM agent on Craftax with reflection — no persistent
synthesized program, no cross-episode artifact); AutoAscend (static expert system — our foil).

**Novel (the contribution)**:
1. A **declarative policy program as a synthesis substrate for embodied agents** — typed bounded params +
   ordered rules + goals + handlers, compiled to an executor contract (vs functions, prompts, or code).
2. **Safety-preserving bounded self-extension**: the policy language itself grows via machine-verified
   definitional extension (closure, interlock invariants, certification) — provable boundedness enabling
   unattended operation; the LLM can never express mechanism-level action.
3. **Evidence-conditioned population synthesis** over executable environments at CPU scale, with per-candidate
   provenance (the ledger is the paper's audit artifact).
4. **Cross-episode nonmonotonic learning inside the artifact** (CDCL nogoods + gated revisions) — knowledge
   that survives death, which none of the cited systems have.
5. **Domain-portable synthesis**: identical machinery from cold-start on docs-only corpora (Craftax) — the
   "no expert baseline to copy" condition, which is precisely the point.

**Significance**: if S4/S5 land, this is the first demonstration that a safety-gated LLM synthesis loop
outgrows person-decades of expert engineering on a hard POMDP *and* transfers — with every accepted and
rejected step auditable. The negative results (v6 combat-rule revert, food-chase dose-response) are part of
the safety story, not noise.

---

## 4. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Author regresses the program (v6 lesson: −45% missed by a 3-ep gate) | Tiered gates (10ep×10k for rule/handler adds), target-conditionality invariant, evolution selection is *relative* — a regression loses, it doesn't ship |
| NetHack σ masks real effects | 100-ep batches minimum, CIs, dose-response arms (3-arm + population arms) on seeded domains carry the statistics |
| ISA too weak → author stalls | S1 kill-criterion forces honest narrowing; curricula localize skill deficits fast |
| ISA too strong → unsound strategies evolve | Interlocks are inviolable; certification gates handlers; shadow harness before promotion |
| Compute ceiling | 30-worker pools ≈ 6k NetHack eps/hr; curricula ~100× cheaper per skill iteration; scale-out is linear (process-safe) |
| Prompt drift/overfitting to eval batches | Prompt versions evaluated on held-out batches; the eval set rotates |
| "Copied AutoAscend" accusation | Hard: no AutoAscend code or tables are read or ported (verified by review); wiki RAG is public game documentation, the same source the experts used |

---

## 5. Session protocol (any fresh session)

1. Read `AGENTS.md` → this file → `MACRO.md` (DSL spec).
2. `uv run pytest -q` → expect **321 passed**; fix flaky isolation, never skip.
3. Check campaign state: `data/revision_ledger.jsonl` tail, `data/programs/archive/`, running campaigns
   (`ps aux | grep run_`), unconsolidated parquet (`scripts/clean_telemetry.py`).
4. Current phase: **S0** (evidence + prompt harness). Phase gates/kill criteria in §2 — do not skip ahead.
5. Compute discipline: 30-worker pools; consolidate DuckDB at campaign boundaries.
