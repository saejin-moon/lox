# LOX Operational Guide (`AGENTS.md`)

Welcome to **LOX**, an autonomous, high-throughput empirical policy synthesis engine for **NetHack 3.6.6** built under the architectural motto: **"Hard Code Nothing."**

---

## 1. Core Architectural Philosophy ("Hard Code Nothing")

1. **Zero Hardcoded Strategy in the Harness**:
   - The Architect's harness (`lox/env/`), native Rust extension (`crates/lox-core/`), AST compiler (`lox/lang/`), deficit analyzer (`lox/author/swarm_deficits.py`), and referee (`lox/engine/swarm_referee.py`) contain **zero** game heuristics, **zero** hardcoded monster/corpse/item lists, and **zero** hidden macro adapters.
   - `lox/env/` provides a **pure 1-to-1 connection** between policy yields/returns and native NLE C actions (`Action`, `key(ch)`, `seq(...)`, `repeat(...)`, `extcmd(...)`, `DIR_DELTAS`) and exposes complete NetHack game state lazily via `GameState` (`blstats`, 21x79 screen grids, inventory, `WorldMemory`, and direct C symbol table queries over `nle.nethack.permonst` and `nle.nethack.objclass`).
2. **Option-A Modular Policy Language & Sandboxed Compiler**:
   - Policies (`data/workspace/policy.py`) and 1-level / 2-level modular skill packages (`data/workspace/skills/<domain>/<module>.py`) support Python 3.11+ syntax (`class`, `step(state)` or generator `run(state)`, `match/case`, comprehensions, `@dataclass`, type hints, and allowlisted stdlib + `numpy`).
   - Compile-time AST validation (`lox/lang/sandbox.py`, `contracts_check.py`), sanitized stdlib module proxies (`lox/lang/compiler.py`), and instrumentation (`lox/lang/instrument.py`) enforce strict sandbox isolation: shared `SAFE_STDLIB_SUBMODULES` & `ALLOWED_LOX_SUBMODULES`, stripped `._*` and `FORBIDDEN_IMPORT_NAMES` across direct and aliased module references, read-only `_TracerProxy` loop watchdog, monotonic generator `frame_id` leaf attribution, and pre-rollout `pyright` + `symtable` verification.
3. **Layout A (`2R + 2P`) Democratized Peer Swarm**:
   - **`reactive_1`**: Picks up any active qualifying reactive stepping stone (`stepping_stones`, max 2 revisions), otherwise targets empirical failure cluster #1 from `DeficitAnalyzer`.
   - **`reactive_2`**: Targets an orthogonal empirical failure cluster from `champion_runs` / `champion_ticks` (using recency-weighted Ochiai & Liblit attribution over unmasked sub-functions).
   - **`proactive_1`**: Starts from Champion and contrasts `champion_ticks` against `expert_ticks` / `expert_runs` + the offline NetHack 3.6.6 Wiki across `Dlvl 1..p90+2` to synthesize a self-contained missing capability module.
   - **`proactive_2`**: Advances a 4-generation sheltered proactive branch (`proactive_branches`) gated on branch-internal non-catastrophic `S_select` safety and graduating to Champion only on strict `S_confirm`.
   - A 1-shot `SwarmDirectivePlanner` (`lox/author/directives.py`) partitions orthogonal investigative scopes across all 4 peers before concurrent execution, and `run_side_cohesion_refactor` keeps workspace files modularized outside promotion gating.
4. **3-Stage Split-Sample Paired CRN Promotion Gate (`SwarmReferee`)**:
   - **Stage 1 (`incubate_candidate`)**: Lock-protected compile, smoke rollout, failure-shard & frontier checkpoint replays, and a paired micro-pilot (`D_succ` guards + `D_fail` targets + unseen probe seeds).
   - **Stage 2 (`S_select`)**: Evaluates Incumbent + finalists + safe AST-disjoint 2-way stacks composed strictly from individually safe positive winners; ranks by precision-aware Winsorized LCB and advances only the single `#1` challenger `C*`.
   - **Stage 3 (`S_confirm` + Stage-3b O'Brien-Fleming / Triggered Sparse-Surgical Confirmation, max `N=128`)**: Confirms `C*` vs. Incumbent on strictly unseen paired seeds using 4 calibrated safety invariants (exceptions, `auto_esc`, paired McNemar `horizon_waste`, paired CVaR-25%) + exact paired Sign-Permutation test (`FWER <= 0.05`) + optional MiniHack `--scenario-gate` (`off`/`advisory`/`strict`).
5. **Unified DuckDB Warehouse (16 Tables, 8 Views) & Native Rust Hot Path**:
   - `data/lox.duckdb` stores 16 tables (`batches`, `runs`, `ticks`, `function_stats`, `policies`, `experiments`, `discoveries`, `expert_runs`, `expert_ticks`, `synthesis_logs`, `llm_cost_rates`, `archive_policies`, `theories`, `tournament_attempts`, `stepping_stones`, `proactive_branches`) and 8 analysis views (`champion_runs`, `champion_function_stats`, `champion_ticks`, `latest_pilot_runs`, `latest_pilot_ticks`, `peer_runs`, `peer_ticks`, `all_ticks`).
   - `crates/lox-core/` (`lox_core` via PyO3/maturin) accelerates BFS distance grids, A* next-step routing, floor memory updates, exit extraction, vectorized trap masks, and unexplored frontier masks (`path.rs`, `replay.rs`), with strict `ValueError` propagation for shape mismatches in `lox/env/grid_utils.py`.

---

## 2. Repository Structure

```text
lox/
├── core/
│   └── contracts.py          # Typed EpisodeRun, BatchEvaluation, CandidatePackage, invariant guards
├── env/
│   ├── actions.py            # 1-to-1 Action IntEnum, key(), seq(), repeat(), extcmd(), DIR_DELTAS
│   ├── types.py              # MonsterInfo, ObjectInfo, InventoryItem, VisibleMonster, VisibleObject, C LUTs
│   ├── memory.py             # Zero-heuristic FloorMemory & WorldMemory (Rust + Python passability parity)
│   ├── state.py              # Lazy zero-copy GameState, slot-cached blstats, pathfinding, prompt detectors
│   ├── grid_utils.py         # Rust/Numba BFS distance grids & A* next-step routing (strict shape checks)
│   ├── harness.py            # Direct C++ nle.nethack.Nethack harness & deterministic byte replay
│   ├── base.py               # BenchmarkAdapter protocol & NetHack registry
│   ├── nethack/
│   │   ├── __init__.py       # NetHack package exports
│   │   └── adapter.py        # NetHackBenchmarkAdapter & progression metric
│   └── benchmarks/
│       └── nethack_scenarios.py # Prompt-torture & archetype MiniHack scenarios
├── lang/
│   ├── sandbox.py            # Python 3.11+ AST sandbox, SAFE_STDLIB_SUBMODULES & hygiene auditor
│   ├── splicer.py            # Surgical AST function splicer, 3-way AST merge, #EVOLVE-BLOCK bounds
│   ├── contracts_check.py    # Skill key normalization, undefined-symbol & cross-module checks, pyright
│   ├── contracts.py          # Optional typed policy contracts (GoalIntent, SafetyReport, WorldTracker)
│   ├── validator.py          # Thin re-export facade for sandbox, splicer, and contracts_check
│   ├── instrument.py         # ExecutionTracer, read-only _TracerProxy, loop watchdog, frame_id stack
│   ├── compiler.py           # Sanitized stdlib proxies, scoped __import__, & PolicyRunner compiler
│   ├── bundle.py             # XML/Markdown `<policy_code>` / `<skill_code>` unpacker & disk sync
│   ├── modularize.py         # Bit-identical AST side cohesion refactorer
│   └── starter.py            # Clean zero-heuristic Option-A Gen-0 starter package builder
├── db/
│   ├── schema.py             # 16-table / 8-view DuckDB schema, migrations, cost rates, init cache
│   ├── summarizer.py         # RLE phase compressor, inflection detector, Z/IQR outliers, Ochiai/Liblit
│   ├── shards.py             # Organic scenario & failure-shard extraction
│   ├── logger.py             # High-throughput transactional DuckDB logger
│   └── theories.py           # DuckDB-backed TheoryDocumentStore
├── datasets/
│   ├── bootstrap_frontier.py # Seeds Go-Explore FrontierArchive from historical DuckDB runs
│   ├── symbols.py            # C-level permonst/objclass lookup registry
│   ├── wiki.py               # Offline NetHack 3.6.6 Wiki SQLite FTS5 BM25 search engine
│   └── download_experts.py   # Compiles NAO human ascensions & ttyrecs into expert_runs / expert_ticks
├── runner/
│   └── pool.py               # Persistent multi-core NLE worker pool
├── author/
│   ├── prompts.py            # Static Option-A Peer Swarm system prompt & Turn-0 user prompt
│   ├── directives.py         # 1-shot SwarmDirectivePlanner, prune_unused_skills, branch extractor
│   ├── llm_client.py         # Provider-first API key resolver, HTTP retry client, safe usage extraction
│   ├── parsing.py            # Safe YAML/Python/skill/tool-arg extraction helpers
│   ├── agent.py              # Stateless AuthorAgent with compiler-in-the-loop self-repair
│   ├── swarm_deficits.py     # Deterministic SQL/AST DeficitAnalyzer & expert-parity gaps
│   ├── swarm_peer.py         # SwarmPeerAgent, parse_peer_output skill merge & isolated incubation
│   └── tools/
│       ├── __init__.py       # Package re-exports
│       ├── specs.py          # OpenAI tool schemas (16 peer tools)
│       ├── db_tools.py       # Single-pass SQL lexer + enable_external_access=false DB/Wiki/Symbol tools
│       ├── code_tools.py     # Safe read_skill, read_policy, diff_policies, read_symbol, grep, architecture_map
│       ├── micro_pilot.py    # Horizon-matched Paired CRN Micro-Pilot
│       ├── dry_run.py        # Live NetHack smoke + replay + scenario + micro-pilot verifier
│       └── registry.py       # AuthorToolRegistry composition
├── engine/
│   ├── hypothesis.py         # Confidence Sequences, Wilcoxon, exact Sign-Permutation, Winsorized/CVaR metrics
│   ├── referee_gates.py      # TournamentVerdict, calibrated safety invariants, stepping-stone selector
│   ├── swarm_referee.py      # SwarmReferee 3-stage split-sample CRN tournament & AST stacker
│   ├── lifecycle.py          # Stepping-stone, proactive-branch, near-miss, & Layout A assignment lifecycles
│   ├── reporting.py          # Adaptive max_steps & campaign status report writer
│   ├── swarm_orchestrator.py # End-to-end Layout A SwarmCampaignOrchestrator
│   ├── frontier.py           # Go-Explore FrontierArchive byte-replay checkpoints
│   └── map_elites.py         # 3D MAP-Elites v2 diversity archive
└── ui/
    └── console.py            # Thread-safe compact ANSI console UI
crates/
└── lox-core/                 # Native Rust PyO3/maturin hot-path acceleration (path.rs, replay.rs, lib.rs)
scripts/
├── run_swarm.py              # Main CLI entrypoint for Layout A Peer Swarm synthesis campaigns
├── status_lox.py             # Live campaign telemetry & read-only DuckDB status dashboard
├── reset_clean.py            # Clean workspace/frontier/DB reset utility (preserves wiki & experts)
├── modularize_policy.py      # Standalone AST modularizer CLI with bit-identical verification
├── download_experts.py       # Populates expert_runs and expert_ticks in DuckDB
├── build_wiki_db.py          # Builds data/wiki_index.db from NetHackWiki XML + Guidebook 3.6.6
└── build_core.sh             # Rebuilds & installs lox-core PyO3 wheel (--release)
tests/                        # Automated unit, schema/prompt contract, & adversarial regression suites
```

---

## 3. Engineering Guardrails (Preventative Rules)

Every contribution to LOX **must** adhere to these enforceable rules:

1. **Zero `x or default` on Numeric or Boolean Fields**:
   - Never write `float(d.get("sign_perm_p", 1.0) or 1.0)`, `float(d.get("avg_ac", 10.0) or 10.0)`, or `int(r.get("seed") or idx)`. Values like `0.0` (exact `p == 0.0`, `AC == 0`) and `0` (`seed == 0`) are falsy in Python!
   - Also beware `d.get(key, default)`: if the key exists with value `None`, `.get` returns `None`. Always check `is None` explicitly or use `opt_float(d, key, default)` / `opt_int(d, key, default)` from `lox/engine/referee_gates.py` (or `_tick_int` / `_opt_int` in `db/`).
2. **Mandatory Static Type Gate (`pyright`) Alongside `ruff` & `pytest`**:
   - `uv run pyright lox/` must pass with **0 errors** before finishing any change.
3. **No Silent `except Exception: pass` on Persistence, Telemetry, Lifecycle, or Guard Logic**:
   - Either propagate the error or log a warning via `ui.warn(...)` / `_LOG.warning(...)` / explicit error return so broken DB writes, lifecycle updates, or safety checks never fail invisibly.
4. **Test-Verified Prompt Docs & SQL Templates**:
   - Every `GameState` / `VisibleObject` / `ObjectInfo` attribute referenced in `lox/author/prompts.py` and every SQL query template in `prompts.py` / `swarm_peer.py` must be verified against live objects and `init_db()` in `tests/author/test_prompt_and_schema_contracts.py`.
5. **Cross-Cutting Touchpoint Checklist**:
   - **New DuckDB table/view**: update `lox/db/schema.py` (schema + `ALTER TABLE` migration if modifying an existing table) + `get_schema_markdown()` + `lox/db/logger.py` + `scripts/reset_clean.py`.
   - **Mid-generation policy/skill mutation** (e.g. side cohesion refactor): immediately recompute `cur_fp = _package_fingerprint(...)`.
   - **New tournament / candidate field**: wire consistently across Mode A, Mode B, early returns, and all `PeerCandidate` constructors.
   - **New sandbox builtin, module, or submodule allowlist entry**: update `lox/lang/sandbox.py`, `lox/lang/compiler.py`, `lox/lang/contracts_check.py`, and `lox/lang/validator.py` together.
6. **Module Cohesion & Delete-Don't-Accrete**:
   - Keep modules cohesive and focused (~700 LOC soft target when creating/splitting modules; never add unrelated responsibilities to existing files), and delete superseded architectures rather than carrying parallel legacy code paths.
7. **Boundary & Adversarial Test Rule**:
   - Every gate, parser, sandbox, SQL guard, or tool change must include tests covering `0.0`, `None`, empty input, malformed LLM JSON args, security/escape attempts, and non-mock code paths.
8. **Live Campaign DB Safety**:
   - Never mutate a real campaign database during tests or ad-hoc inspection; use `connect_db(..., read_only=True)` or copy the file to `/tmp/` with trap cleanup before running schema migrations or writes.

---

## 4. Running Commands, Linting, Type-Checking & Tests

Always use `uv`, `ruff`, `pyright`, `maturin`, and `pytest`:

```bash
# 1. Run Rust unit tests & rebuild native lox_core bindings
cargo test --manifest-path crates/lox-core/Cargo.toml
uv run maturin develop --release -m crates/lox-core/Cargo.toml

# 2. Lint, format, and type-check lox/ (target: 0 errors)
uv run ruff check --fix .
uv run ruff format .
uv run pyright lox/

# 3. Run the full automated test suite (196 tests)
uv run pytest -v

# 4. Build/refresh offline NetHackWiki index & expert ascensions
uv run python scripts/build_wiki_db.py --download-xml
uv run python scripts/download_experts.py --max-ascensions 1000 --fetch-dumplogs 25

# 5. Inspect live campaign status safely in read-only mode
uv run python scripts/status_lox.py --db-path data/lox.duckdb --since-hours 12

# 6. Launch a Layout A Peer Swarm synthesis campaign
uv run python scripts/run_swarm.py \
  --provider openrouter \
  --model z-ai/glm-5.3-flash:nitro \
  --workers 4 \
  --seeds 64 \
  --race-seeds 32 \
  --generations 50
```
