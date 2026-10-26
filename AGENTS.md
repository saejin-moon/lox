# LOX v3.0 Operational Guide (`AGENTS.md`)

Welcome to **LOX v3.0**, an autonomous, high-throughput empirical policy synthesis engine for NetHack built under the architectural motto: **"Hard Code Nothing."**

---

## 1. Core Architectural Philosophy ("Hard Code Nothing")

1. **Zero Hardcoded Strategy in the Harness**:
   - The Architect's harness (`lox/env/`) contains **zero** game heuristics, **zero** hardcoded monster/corpse/item lists, and **zero** hidden macro adapters.
   - It provides a **pure 1-to-1 connection** between policy yields and native NLE C actions (`Action` enum, `key(ch)`, `seq(...)` multi-keystroke sequences) and exposes the complete NetHack game state lazily via `GameState` (`blstats`, 21×79 screen grids, inventory, and direct C symbol table queries over `nle.nethack.permonst` and `nle.nethack.objclass`).
2. **Near-Full Python Policy Language & Persistent Macro Library**:
   - Policies (`data/workspace/policy.py`) and reusable macros (`data/workspace/macros.py`) support full Python 3.11+ syntax (`class`, generators `yield` / `yield from`, `match/case`, comprehensions, `@dataclass`, closures, type hints, and safe stdlib modules `math`, `collections`, `heapq`, `itertools`, `functools`, `re`, `numpy`, `random`, `bisect`, `copy`).
   - Compile-time AST instrumentation (`PolicyInstrumentTransformer` & `ExecutionTracer`) injects `<1 µs` per-function call tracking, per-function turn attribution, loop-watchdog protection (`PolicyLoopError`), and tunable parameter hooks (`self.param(name, default, min_val, max_val)`).
3. **Go-Explore Outlier Good-Run Attribution & Multi-Table DuckDB**:
   - Every evaluation batch computes statistical progression scores across all recorded episodes and flags top-performing statistical exceptions with `is_outlier_good = TRUE` (`outlier_z_score`, `batch_percentile`).
   - Computes **Ochiai Good-Run Similarity** and **Liblit Failure Importance** per function (`function_stats` table) so the Author Agent can immediately see which functions drive outlier success vs. mortality.
   - Consolidates real human NetHack ascensions (`archive.alt.org` xlogfile + dumplogs) and NLE reference traces directly into DuckDB's `expert_runs` and `expert_ticks` tables using the exact same `summary_digest` / `game_log` / overview stats structure as `runs` and `ticks`.
4. **Stateless Per-Generation Author Agent & 4 Modification Modes**:
   - Each generation starts a **fresh, stateless LLM session** with a 100% static cached System Prompt (Policy Language Specs + DuckDB Schema) and 4 high-signal tools: `query_db(sql)`, `query_wiki(query)`, `query_policy_docs(topic)`, and `dry_run_policy(code)`.
   - **4 Modification Modes**:
     - `FULL_WRITE` (10,000 episodes): Synthesizes `class Policy` from scratch on Generation 0 without a pre-fed starter policy.
     - `MACRO_WRITE` (0 episodes — instant commit): Synthesizes reusable macros into `data/workspace/macros.py` and immediately advances to the next generation.
     - `FUNC_MOD_ADD` (1,000 triggered episodes per modified/added function): Splices targeted methods onto `Policy`; episodes that do not call any targeted function are immediately discarded (`triggered == False`) and excluded from stats and DB storage.
     - `PARAM_TUNE` (0 LLM tokens — CRN-SPSA): Simultaneously tunes all `self.param(...)` numerical thresholds using Common Random Numbers SPSA and bakes winning constants back into the AST.
5. **Parameter-Free Hypothesis Testing**:
   - The Author Agent states a qualitative causal YAML hypothesis (`observation_from_db`, `mechanism`, `target_functions`) without guessing numerical effect sizes.
   - Candidate policies are evaluated via **Anytime-Valid Empirical Bernstein Confidence Sequences** (`[ci_lower, ci_upper]`), **One-Sided Paired Wilcoxon Signed-Rank Tests** (`wilcoxon_p < 0.05`), and **Sequential Futility Early Stopping**.

---

## 2. Repository Structure

```
lox/
├── env/
│   ├── actions.py          # 1-to-1 Action IntEnum, key(), seq(), DIR_DELTAS
│   ├── state.py            # Lazy zero-copy GameState, NLE C LUTs, Numba BFS & A*
│   └── harness.py          # Direct C++ nle.nethack.Nethack harness & 14ms byte replay
├── lang/
│   ├── validator.py        # Python 3.11+ AST validator & surgical function splicer
│   ├── instrument.py       # ExecutionTracer, TunableParam, PolicyInstrumentTransformer
│   ├── macros.py           # Persistent MacroLibrary manager (data/workspace/macros.py)
│   └── compiler.py         # Compiles macros.py + policy.py into PolicyRunner instances
├── db/
│   ├── schema.py           # 8-table DuckDB schema (batches, runs, ticks, function_stats, policies, experiments, expert_runs, expert_ticks)
│   ├── summarizer.py       # RLE phase compressor, inflection detector, Z/IQR outlier marker, Ochiai/Liblit scorer
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

## 3. Running Commands & Tests

Always use the project's virtual environment (`/home/moose/git/lox/.venv/bin/python` and `/home/moose/git/lox/.venv/bin/pytest`):

```bash
# 1. Run the automated test suite
.venv/bin/pytest -v

# 2. Populate or refresh the expert runs table in DuckDB
.venv/bin/python scripts/download_experts.py

# 3. Launch a synthesis campaign (e.g. with Gemma 4 31B or GLM-5 via OpenRouter)
.venv/bin/python scripts/run_lox.py \
  --provider openrouter \
  --model google/gemma-4-31b-it \
  --workers 16 \
  --generations 50
```
