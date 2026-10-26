# LOX v3.0 — Autonomous Empirical Policy Synthesis Engine

> **Architectural Motto:** *"Hard Code Nothing."*

**LOX v3.0** is an autonomous, high-throughput neuro-symbolic program synthesis engine that discovers, tests, attributes, and refines executable Python control policies and reusable macro libraries for complex, long-horizon environments—starting with the **NetHack Learning Environment (NLE)**—from first principles.

Unlike traditional Deep Reinforcement Learning (which trains opaque neural weights over billions of frames) or online LLM agents (which query a slow, expensive language model at every environment step), **LOX puts the LLM in the outer scientific loop and compiled Python in the inner execution loop**:
- **Inner Loop (CPU)**: Compiled Python 3.11+ generator policies execute directly against NLE's C engine at **50,000–96,000+ steps/sec** across worker cores (`0` LLM tokens per step).
- **Outer Loop (Stateless Scientist LLM)**: Between batches, a stateless Author Agent queries a multi-table **DuckDB** analytical warehouse (`runs`, `ticks`, `function_stats`, `expert_runs`, `expert_ticks`) and an offline **NetHack 3.6.6 Wiki**, formulates a falsifiable causal hypothesis, dry-runs candidate code in a 20ms compiler-in-the-loop sandbox, and triggers targeted evolutionary experiments.

---

## Table of Contents

1. [Core Architectural Philosophy ("Hard Code Nothing")](#1-core-architectural-philosophy-hard-code-nothing)
2. [System Architecture & Dataflow](#2-system-architecture--dataflow)
3. [Theoretical & Research Foundations](#3-theoretical--research-foundations)
4. [The Four Modification Modes](#4-the-four-modification-modes)
5. [Multi-Table DuckDB Schema & Outlier Attribution](#5-multi-table-duckdb-schema--outlier-attribution)
6. [Parameter-Free Sequential Hypothesis Testing](#6-parameter-free-sequential-hypothesis-testing)
7. [Repository Structure](#7-repository-structure)
8. [Quickstart & CLI Usage](#8-quickstart--cli-usage)
9. [References & Citations](#9-references--citations)

---

## 1. Core Architectural Philosophy ("Hard Code Nothing")

Earlier generations of LLM-guided game agents rely heavily on human-engineered wrappers: hardcoded monster threat tiers, pre-written corpse safety tables, hidden auto-explore loops, or fixed finite-state machines. While those crutches produce quick early-game demos, they impose a strict glass ceiling when the agent encounters mechanics the human architect forgot to hardcode.

**LOX v3.0 enforces a strict separation between the Architect (the harness) and the Author (the policy agent):**

1. **Zero Hardcoded Strategy in the Harness ([`lox/env/`](lox/env/))**:
   - The harness contains **zero** game heuristics, **zero** monster/corpse/item lists, and **zero** hidden macro adapters.
   - It provides a **pure 1-to-1 map** between policy yields and native NLE C actions ([`Action`](lox/env/actions.py) IntEnum, [`key(ch)`](lox/env/actions.py) raw ASCII bytes, and [`seq(...)`](lox/env/actions.py) multi-keystroke sequences).
   - It exposes the complete NetHack game state lazily via [`GameState`](lox/env/state.py): all 27 `blstats` fields, `BL_MASK_*` status bitmasks, 21×79 screen grids (`glyphs`, `chars`, `colors`, `specials`), inventory slots, and pre-indexed C symbol table lookups over `nle.nethack.permonst` and `nle.nethack.objclass`, alongside generic Numba-JIT grid math ([`bfs_distance_grid`](lox/env/state.py), [`astar_next_step`](lox/env/state.py)).
2. **Near-Full Python 3.11+ Policy Language & Macro Library ([`lox/lang/`](lox/lang/))**:
   - The Author Agent writes full Python 3.11+ in `data/workspace/policy.py` and `data/workspace/macros.py`: classes, stateful generator coroutines (`yield` / `yield from`), `match/case` structural pattern matching, comprehensions, `@dataclass`, closures, and safe standard library modules (`math`, `collections`, `heapq`, `itertools`, `functools`, `re`, `numpy`, `random`, `bisect`, `copy`).
   - Compile-time AST instrumentation ([`PolicyInstrumentTransformer`](lox/lang/instrument.py) & [`ExecutionTracer`](lox/lang/instrument.py)) injects `<1 µs` per-function call counting, per-function turn attribution, zero-yield infinite-loop protection ([`PolicyLoopError`](lox/lang/instrument.py)), and tunable numerical parameter hooks (`self.param(name, default, min_val, max_val)`).

---

## 2. System Architecture & Dataflow

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     STATELESS AUTHOR AGENT (Per-Generation)                      │
│  • 100% Static Cached System Prompt (Policy Language Specs + DuckDB Schema)      │
│  • Dynamic User Prompt: Mode Instructions + Active Code + Contrastive Digest     │
│  • Tools: query_db(sql) | query_wiki(q) | query_policy_docs(t) | dry_run_policy  │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Emits Causal YAML Hypothesis + Python Code
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        HYBRID CADENCE / BANDIT SCHEDULER                         │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ 1. FULL_WRITE  │  │ 2. MACRO_WRITE │  │ 3. FUNC_MOD_ADD  │  │ 4. PARAM_TUNE│  │
│  │   10,000 eps   │  │  0 eps (skip)  │  │ 1,000 trig eps/fn│  │  0 LLM tokens│  │
│  │ (Gen 0 init)   │  │ -> macros.py   │  │ (untrig dropped) │  │  (CRN-SPSA)  │  │
│  └───────┬────────┘  └───────┬────────┘  └────────┬─────────┘  └──────┬───────┘  │
└──────────┼───────────────────┼────────────────────┼───────────────────┼──────────┘
           │                   │ (Instant Next Gen) │                   │
           ▼                   ▼                    ▼                   ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                 PERSISTENT MULTI-CORE C++ NLE WORKER POOL                        │
│  • Direct nle.nethack.Nethack.step(byte) (50k–96k+ steps/sec across workers)     │
│  • Conditional Trigger Filter: discards episodes where target_fn call_count == 0 │
│  • Unreachable Function Circuit Breaker + Sequential Futility Early Stopping     │
└───────────────────────────────────────┬──────────────────────────────────────────┘
                                        │ Recorded Triggered Episodes + 4KB Replay Bytes
                                        ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│           MULTI-TABLE DUCKDB WAREHOUSE & OUTLIER SUMMARIZER (lox.duckdb)         │
│  • RLE Phase Timeline Compressor + Critical Inflection & Stall Detector          │
│  • Go-Explore Outlier Good-Run Marker (is_outlier_good = TRUE via Z-score / IQR) │
│  • Spectrum-Based Function Attribution (Ochiai Good-Run & Liblit Failure Scores) │
│  • Consolidated Human Ascensions & NLE Traces (expert_runs & expert_ticks)       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Theoretical & Research Foundations

LOX v3.0 synthesizes insights from eight distinct research traditions across reinforcement learning, program synthesis, software fault localization, distributed systems observability, and sequential statistics:

### 3.1 Evolutionary Program Synthesis & Library Learning
- **FunSearch** *(Romera-Paredes et al., Nature 2024)* & **AlphaEvolve** *(Novikov et al., 2025)* demonstrated that pairing an LLM code generator with a fast, automated programmatic evaluator can discover novel algorithms when evolving executable functions rather than raw solutions. LOX extends this paradigm from static combinatorial scoring functions to **stateful, multi-thousand-turn closed-loop control policies**.
- **DreamCoder** *(Ellis et al., PLDI 2021)* & **Voyager** *(Wang et al., 2023)* established the **Wake-Sleep Library Learning** paradigm: agents only scale to deep horizons when they factor recurring low-level action sequences into reusable abstractions. LOX implements this directly via its dedicated `MACRO_WRITE` mode (`data/workspace/macros.py`), allowing the Author Agent to build reusable generator subroutines (`yield from macro_name(state)`) without burning evaluation episodes.

### 3.2 Go-Explore & Statistical Outlier Archiving
- **Go-Explore** *(Ecoffet et al., Nature 2021)* & **MAP-Elites** *(Mouret & Clune, 2015)* showed that in sparse-reward, high-mortality environments like Montezuma's Revenge and NetHack, average-case batch statistics wash out rare breakthroughs. Progress depends on **identifying, archiving, and returning to exceptional outlier trajectories**.
- In LOX, every evaluation batch automatically scores all episodes on a composite progression index and flags statistical breakthroughs (`Z-score >= 1.65` or `> Q75 + 1.5 * IQR`) with `is_outlier_good = TRUE` in the `runs` table. Furthermore, every run stores its raw keystroke stream (`action_bytes`, ~4 KB), enabling **14ms deterministic C-level replay** ([`NetHackHarness.replay_bytes`](lox/env/harness.py)) to any turn of any outlier run.

### 3.3 Coverage-Guided Greybox Fuzzing & Conditional Episode Filtering
- **AFL** *(Zalewski, 2014)* & **AFLFast** *(Böhme et al., CCS 2017)* revolutionized software testing by instrumenting basic-block edges at compile time and discarding inputs that fail to exercise targeted code paths.
- LOX applies this exact principle to policy evaluation in `FUNC_MOD_ADD` mode: [`PolicyInstrumentTransformer`](lox/lang/instrument.py) instruments every function and macro in the AST. When the Author modifies or adds $K$ functions, the worker pool requires $1,000 \times K$ episodes that **actually execute** at least one modified function. Any episode where the hero dies before triggering the modified code is **immediately discarded in the worker** (`triggered == False`) and never pollutes database statistics or hypothesis tests.

### 3.4 Spectrum-Based & Statistical Fault Localization (SBFL)
- How does the Author Agent know *which* function caused outlier runs to succeed and *which* function caused normal runs to die? LOX draws directly from software debugging literature:
  - **Ochiai Similarity Coefficient** *(Abreu et al., 2007)*: Originally used to correlate test coverage spectra with software faults, LOX inverts Ochiai to measure **Good-Run Function Attribution**:
    $$\text{Ochiai}_{\text{good}}(f) = \frac{e_{\text{good}}(f)}{\sqrt{(e_{\text{good}}(f) + e_{\text{normal}}(f)) \cdot N_{\text{good}}}}$$
    Functions with high $\text{Ochiai}_{\text{good}}(f)$ are the statistical drivers of `is_outlier_good = TRUE` breakthroughs.
  - **Liblit Statistical Failure Importance** *(Liblit et al., PLDI 2005)*: Computes the harmonic contrast between a function's probability of being active on the fatal turn versus its baseline activation rate, pinpointing the exact subroutine responsible for mortality spikes.

### 3.5 Tail-Based Distributed Tracing & RLE Trajectory Compression
- **Dapper** *(Sigelman et al., 2010)* & **Sifter / Tail-Based Sampling** *(Las-Casas et al., SoCC 2019)* established that high-throughput distributed systems should make trace retention decisions *at the end of a request*—keeping 100% of anomalous/outlier traces while downsampling routine executions.
- [`BatchDatabaseLogger`](lox/db/logger.py) implements **Smart Tail-Based Tick Sampling**: every run in `runs` receives a full **Run-Length Encoded (RLE) Phase Timeline**, **Critical Inflection Log**, and **Stall Heatmap** ([`summarize_episode_ticks`](lox/db/summarizer.py)), while the `ticks` table stores full tick-by-tick histories for all `is_outlier_good = TRUE` runs and diagnostic exemplars without multi-gigabyte storage bloat.

### 3.6 Simultaneous Perturbation Stochastic Approximation with Common Random Numbers (CRN-SPSA)
- **SPSA** *(Spall, IEEE TAC 1992, 1998)* & **CRN-SPSA** *(Kleinman, Spall & Naiman, 1999)* proved that in high-dimensional stochastic optimization ($p$ continuous/integer parameters), perturbing all parameters simultaneously along a Rademacher vector $\Delta_k \in \{-1, +1\}^p$ requires only **2 evaluations per gradient step** regardless of dimension $p$, and evaluating $\theta + c_k \Delta_k$ and $\theta - c_k \Delta_k$ on **identical random seeds** (Common Random Numbers) collapses gradient variance.
- LOX uses **CRN-SPSA** in `PARAM_TUNE` mode ([`run_spsa_param_tuning`](lox/engine/modes.py)) to tune all `self.param(...)` thresholds in the policy simultaneously using **zero LLM tokens**, then bakes the winning constants back into the Python AST.

### 3.7 Anytime-Valid Confidence Sequences & Non-Parametric Rank Testing
- **Empirical Bernstein Confidence Sequences** *(Howard et al., Annals of Statistics 2021; Waudby-Smith & Ramdas, JRSS-B 2024)* provide time-uniform confidence intervals $[L_t, U_t]$ that remain statistically valid under continuous sequential monitoring and early stopping.
- **Wilcoxon Signed-Rank Test** *(Wilcoxon, 1945)* provides non-parametric paired comparison robust to heavy-tailed NetHack score/depth distributions.
- Together, these allow LOX to eliminate "LLM effect-size guessing" completely ([`lox/engine/hypothesis.py`](lox/engine/hypothesis.py)).

### 3.8 Extreme-Value Multi-Armed Bandits for Mode Scheduling
- **Max $K$-Armed Bandits** *(Cicirello & Smith, 2005; Streeter & Smith, 2006)* model mutation operator selection in evolutionary algorithms where the value of an operator is governed by the *maximum* breakthrough it produces per unit compute rather than its mean. [`ModeScheduler`](lox/engine/scheduler.py) combines a deterministic warmup cadence with Extreme-Value UCB credit assignment.

---

## 4. The Four Modification Modes

| Mode | Episode Budget | Trigger / Filter Rule | Purpose |
| :--- | :--- | :--- | :--- |
| **`FULL_WRITE`** | `10,000` episodes | All episodes recorded | Synthesizes `class Policy` from scratch on Generation 0 (no starter policy fed). |
| **`MACRO_WRITE`** | `0` episodes *(instant)* | Syntax + AST sandbox check | Synthesizes reusable generator/helper macros into `data/workspace/macros.py` and immediately starts the next generation. |
| **`FUNC_MOD_ADD`** | `1,000` triggered eps $\times K$ functions | **Must call $\ge 1$ modified function**; untriggered episodes are discarded | Surgically adds or modifies methods on `Policy` ([`splice_functions_into_policy`](lox/lang/validator.py)). |
| **`PARAM_TUNE`** | `0` LLM tokens (`CRN-SPSA`) | Paired seed mini-batches + validation batch | Optimizes all `self.param(name, default, min, max)` constants via CRN-SPSA and bakes winning values into the AST. |

---

## 5. Multi-Table DuckDB Schema & Outlier Attribution

All experimental telemetry and reference human ascensions live in a single consolidated DuckDB database (`data/lox.duckdb`) queryable by the Author Agent via `query_db(sql)`:

1. **`batches`**: Batch-level metadata (`batch_id`, `generation`, `mode`, `target_functions`, `episodes_recorded`, `episodes_discarded`, `avg_depth`, `p90_depth`, `max_depth`, `outlier_count`, `hypothesis_passed`, `wilcoxon_p`, `ci_lower`, `ci_upper`).
2. **`runs`**: Every recorded episode (`run_id`, `batch_id`, `seed`, `depth`, `max_depth`, `score`, `turns`, `xp_level`, `final_hp`, `final_ac`, `end_reason`, `summary_digest`, `game_log`, `milestones_json`, `triggered_functions`, `last_active_function`, `outlier_z_score`, `batch_percentile`, `is_outlier_good`, `action_bytes`).
3. **`ticks`**: Tail-sampled step-by-step trace records (`run_id`, `turn`, `step_idx`, `depth`, `hp`, `ac`, `xp_level`, `hunger_state`, `y`, `x`, `action_name`, `raw_key`, `active_function`, `message`).
4. **`function_stats`**: Per-function execution spectrum (`function_name`, `episodes_triggered`, `trigger_rate`, `total_calls`, `total_turns_active`, `avg_depth_when_triggered`, `death_rate_while_active`, `ochiai_good_run_score`, `liblit_failure_score`).
5. **`policies`**: Full source code and parameter snapshots for every generation (`policy_id`, `policy_code`, `macros_code`, `params_json`, `is_promoted`).
6. **`experiments`**: Causal hypotheses and statistical verdicts (`observation_from_db`, `mechanism`, `passed`, `mean_delta`, `ci_lower`, `ci_upper`, `wilcoxon_p`, `verdict_reason`).
7. **`expert_runs`** & **`expert_ticks`**: Verified human NetHack ascensions (`archive.alt.org` xlogfile + dumplogs, *Hambro et al., 2022*) and NLE reference trajectories compiled into the **exact same `summary_digest` / `game_log` / overview stats format** as `runs` and `ticks`.

---

## 6. Parameter-Free Sequential Hypothesis Testing

The Author Agent never guesses a numerical improvement target. Instead, it emits a structured qualitative causal hypothesis:

```yaml
mode: FUNC_MOD_ADD
target_functions:
  - handle_combat
  - retreat_to_stairs
observation_from_db: "In runs WHERE is_outlier_good = TRUE, Ochiai score for retreat_to_stairs is 0.74, whereas 62% of normal deaths occur in handle_combat when HP < 30%."
mechanism: "Check HP ratio before melee engagement and pathfind toward '<' or '#' corridor tiles when surrounded by >= 2 hostiles."
```

Candidate policies are evaluated on paired seeds against the active baseline using three objective statistical gates ([`lox/engine/hypothesis.py`](lox/engine/hypothesis.py)):
1. **Sequential Futility Early Stopping**: If after $n \ge 60$ paired triggered seeds the 95% Empirical Bernstein Confidence Sequence upper bound satisfies $U_n \le 0.0$, the evaluation terminates early to save compute.
2. **Positive Mean Progression Delta**: $\bar{d} > 0.0$ and strictly more improved seeds than regressed seeds ($\text{wins} > \text{losses}$).
3. **Statistical Significance**: Either the 95% Confidence Sequence lower bound $L_N > 0.0$ or the one-sided Paired Wilcoxon Signed-Rank $p < 0.05$.

---

## 7. Repository Structure

```text
lox/
├── env/
│   ├── actions.py          # 1-to-1 Action IntEnum, key(), seq(), DIR_DELTAS
│   ├── state.py            # Lazy zero-copy GameState, NLE C LUTs, Numba BFS & A*
│   └── harness.py          # Direct C++ nle.nethack.Nethack harness & 14ms byte replay
├── lang/
│   ├── validator.py        # Python 3.11+ AST sandbox validator & surgical function splicer
│   ├── instrument.py       # ExecutionTracer, TunableParam, PolicyInstrumentTransformer
│   ├── macros.py           # Persistent MacroLibrary manager (data/workspace/macros.py)
│   └── compiler.py         # Compiles macros.py + policy.py into PolicyRunner instances
├── db/
│   ├── schema.py           # 8-table DuckDB schema initialization & connections
│   ├── summarizer.py       # RLE phase compressor, inflection detector, outlier marker, Ochiai/Liblit
│   └── logger.py           # High-throughput DuckDB batch logger with tail-based tick sampling
├── datasets/
│   ├── wiki.py             # Offline NetHack 3.6.6 Wiki SQLite FTS5 BM25 search engine
│   └── download_experts.py # Compiles NAO human ascensions & NLE traces into expert_runs / expert_ticks
├── runner/
│   └── pool.py             # Persistent multi-core NLE worker pool & conditional trigger filter
├── author/
│   ├── prompts.py          # Static cacheable System Prompt & dynamic User Prompt builder
│   ├── tools.py            # AuthorToolRegistry: query_db, query_wiki, query_policy_docs, dry_run_policy
│   └── agent.py            # Stateless ReAct AuthorAgent with compiler-in-the-loop self-repair
├── engine/
│   ├── hypothesis.py       # Confidence Sequences, Wilcoxon Signed-Rank, Futility Early Stopping
│   ├── modes.py            # CRN-SPSA parameter optimizer & AST constant baker
│   ├── scheduler.py        # Hybrid cadence + Extreme-Value UCB mode scheduler
│   └── orchestrator.py     # End-to-end CampaignOrchestrator
└── ui/
    └── console.py          # Compact ANSI console UI with single-line progress & summary tables
scripts/
├── run_lox.py              # Main CLI entrypoint for running synthesis campaigns
├── download_experts.py     # Populates expert_runs and expert_ticks in data/lox.duckdb
└── build_wiki_db.py        # Builds data/wiki_index.db from data/nethackwikidata.json
tests/
├── test_harness_speed.py   # 1-to-1 actions, GameState queries, replay bit-identity, throughput
├── test_lang_instrument.py # Python 3.11+ features, sandbox security, tracer, loop guard, splicing
├── test_db_summarizer.py   # RLE summarizer, outlier marker, Ochiai/Liblit, DuckDB logger & tools
├── test_modes_and_eval.py  # Trigger filtering, unreachable circuit breaker, hypothesis CS, SPSA, mock campaign
└── test_author_and_ui.py   # Author YAML/code parser, macro execution tracing, ConsoleUI rendering
```

---

## 8. Quickstart & CLI Usage

### 1. Run the Automated Verification Suite
```bash
.venv/bin/pytest -v
```

### 2. Initialize DuckDB & Populate Expert Runs
```bash
.venv/bin/python scripts/download_experts.py --max-ascensions 50
```

### 3. Launch a Synthesis Campaign
```bash
export OPENROUTER_API_KEY="your-api-key"

# Run with Gemma 4 31B (or DeepSeek R1 / GLM-5 / Qwen / Claude)
.venv/bin/python scripts/run_lox.py \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --workers 16 \
  --generations 50 \
  --full-write-episodes 10000 \
  --func-mod-episodes 1000
```

---

## 9. References & Citations

1. **Küttler, H., Nardelli, N., Miller, A. H., Raileanu, R., Selvatici, M., Grefenstette, E., & Rocktäschel, T.** (2020). *The NetHack Learning Environment.* Advances in Neural Information Processing Systems (NeurIPS), 33, 7671–7684.
2. **Hambro, E., Raileanu, R., Rothermel, D., Mella, V., Rocktäschel, T., Küttler, H., & Murray, N.** (2022). *Insights from the NeurIPS 2021 NetHack Challenge.* Proceedings of Machine Learning Research (PMLR), 176, 41–52.
3. **Hambro, E., Mohanty, S., Babaev, D., Byeon, M., Chakraborty, D., Grefenstette, E., Jiang, M., Küttler, H., et al.** (2022). *Dungeons and Data: A Large-Scale NetHack Dataset (NLD).* Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track.
4. **Ecoffet, A., Huizinga, J., Lehman, J., Stanley, K. O., & Clune, J.** (2021). *First return, then explore (Go-Explore).* Nature, 590(7847), 580–586.
5. **Mouret, J.-B., & Clune, J.** (2015). *Illuminating search spaces by mapping elites (MAP-Elites).* arXiv:1504.04909.
6. **Romera-Paredes, B., Barekatain, M., Novikov, A., Balog, M., Kumar, M. P., Dupont, E., Ruiz, F. J. R., Ellenberger, J. S., Wang, P., Fawzi, O., Kohli, P., & Fawzi, A.** (2024). *Mathematical discoveries from program search with large language models (FunSearch).* Nature, 625(7995), 468–475.
7. **Novikov, A., Vu, N., Eisenberger, M., Dupont, E., Huang, P.-S., Wagner, A. Z., Shirobokov, S., Kozlovskii, B., Ruiz, F. J. R., Mehrabian, A., Kumar, M. P., See, A., Chaudhuri, S., Holland, G., Davies, A., Nowozin, S., Kohli, P., & Balog, M.** (2025). *AlphaEvolve: A coding agent for scientific and algorithmic discovery.* Google DeepMind Technical Report.
8. **Ellis, K., Wong, C., Nye, M., Sablé-Meyer, M., Morales, L., Hewitt, L., Cary, L., Solar-Lezama, A., & Tenenbaum, J. B.** (2021). *DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning.* Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation (PLDI), 835–850.
9. **Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., & Anandkumar, A.** (2023). *Voyager: An Open-Ended Embodied Agent with Large Language Models.* Transactions on Machine Learning Research (TMLR).
10. **Ma, Y. J., Liang, W., Wang, G., Huang, D.-A., Bastani, O., Jayaraman, D., Zhu, Y., Fan, L., & Anandkumar, A.** (2024). *Eureka: Human-Level Reward Design via Coding Large Language Models.* International Conference on Learning Representations (ICLR).
11. **Böhme, M., Pham, V.-T., & Roychoudhury, A.** (2017). *Coverage-Based Greybox Fuzzing as Markov Chain (AFLFast).* IEEE Transactions on Software Engineering / ACM CCS.
12. **Abreu, R., Zoeteweij, P., & van Gemund, A. J. C.** (2007). *On the Accuracy of Spectrum-based Fault Localization (Ochiai).* Testing: Academic and Industrial Conference Practice and Research Techniques (TAICPART-MUTATION), IEEE, 89–98.
13. **Liblit, B., Naik, M., Zheng, A. X., Aiken, A., & Jordan, M. I.** (2005). *Scalable statistical bug isolation.* Proceedings of the 2005 ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), 15–26.
14. **Las-Casas, P., Papakerashvili, G., Anand, V., & Mace, J.** (2019). *Sifter: Scalable Sampling for Distributed Traces, without Feature Engineering.* Proceedings of the ACM Symposium on Cloud Computing (SoCC), 312–324.
15. **Spall, J. C.** (1992). *Multivariate stochastic approximation using a simultaneous perturbation gradient approximation (SPSA).* IEEE Transactions on Automatic Control, 37(3), 332–341.
16. **Kleinman, N. L., Spall, J. C., & Naiman, D. Q.** (1999). *Simulation-based optimization with stochastic approximation using common random numbers.* Management Science, 45(11), 1570–1578.
17. **Waudby-Smith, I., & Ramdas, A.** (2024). *Estimating means of bounded random variables by betting (Empirical Bernstein Confidence Sequences).* Journal of the Royal Statistical Society Series B: Statistical Methodology, 86(1), 1–27.
18. **Howard, S. R., Ramdas, A., McAuliffe, J., & Sekhon, J.** (2021). *Time-uniform, nonparametric, nonasymptotic confidence sequences.* The Annals of Statistics, 49(2), 1055–1080.
19. **Wilcoxon, F.** (1945). *Individual Comparisons by Ranking Methods.* Biometrics Bulletin, 1(6), 80–83.
20. **Cicirello, V. A., & Smith, S. F.** (2005). *The Max K-Armed Bandit: A New Model of Exploration Applied to Search Heuristic Selection.* Proceedings of the Twentieth National Conference on Artificial Intelligence (AAAI), 1355–1361.
