# LOX-ψ: LLM-Oriented Creation of Symbolic policies

**LOX-ψ** is a policy synthesis engine: a safety-gated loop in which an LLM author grows a declarative
policy program (macros, rules, goals, handlers) from full run telemetry, the game wiki, and self-built
documentation corpora — until the resulting agent surpasses expert-engineered baselines for NetHack (NLE 3.6.6), with transfer adapters for MiniHack and Craftax.

Humans provide primitives, interlocks, and certification gates. The LLM provides the strategy — through a
grammar-constrained diff language that provably cannot express mechanism-level action.

An event-driven, bicameral cognitive architecture engineered to solve the NetHack Learning Environment (NLE) and systematically outperform existing rule-based benchmarks (AutoAscend) across both forced random personas and self-selected optimal playstyles.

---

## 1. System Manifesto & Core Innovations

NetHack is an archetypal Partially Observable Markov Decision Process (POMDP) characterized by extreme procedural generation, lethal state traps, non-linear strategic progression, and thousands of obscure, interacting game mechanics.

Traditional reinforcement learning agents struggle due to sparse rewards and extreme catastrophic forgetting. Monolithic rule-based agents (such as AutoAscend) achieve high depth by hand-crafting extensive heuristics, but suffer from **fatal brittleness**:
1. **Zero Cross-Generational Learning**: They repeat identical fatal blunders across runs.
2. **Ad-Hoc Identification**: They lack Bayesian probabilistic modeling of curse/enchantment/charge states, frequently risking game-ending curses.
3. **Stalemate & Oscillation Loops**: When conflicting heuristics trigger, they enter deadlock loops and starve.

This architecture resolves these failure modes through a **Bicameral Neuro-Symbolic Operating System**:
* **Fast Core (<0.5 ms/tick)**: A deterministic Hierarchical Task Network (HTN) planner with active branch pruning running strictly on CPU.
* **Slow Deliberative Core (Asynchronous / Triggered)**: An out-of-band LLM reasoning engine (supporting OpenRouter, Google Gemma, and local `llama.cpp` hosting QwQ-32B) that runs multi-step causal `<think>` analysis only during cold-boots, anomaly deadlocks, and post-mortem autopsies.
* **Epistemic POMDP Engine**: Continuous Bayesian belief tracking over item identity, BUC status (Blessed/Uncursed/Cursed), charges, and erosion with active entropy reduction experiments.
* **CDCL-Style Episodic Memory**: Conflict-Driven Clause Learning that synthesizes autopsy findings into permanent negative constraints (Nogoods), permanently cutting fatal decision trees in future generations.
* **Strict-Null Semantic RAG**: In-process, zero-VRAM hybrid search (`bm25s` + `FastEmbed` ONNX + `FlashRank` Cross-Encoder) over the NetHack Wiki with hard thresholding ($\tau \ge 0.82$) to mathematically prevent hallucination.
* **Role-Agnostic Design & Dual Evaluation Modes**: A unified core engine operating on continuous persona trait vectors, supporting both zero-shot random persona survival (`MODE_RANDOM_GENERALIST`) and autonomous competence-driven ascension optimization (`MODE_COMPETENCE_SELECTION`).

---

## 2. Architectural Documentation Index

The complete specification is detailed across nine modular documents in `docs/architecture/`:

| Spec Document | Title | Core Focus |
| :--- | :--- | :--- |
| [00_master_topology.md](file:///home/bae/corp/docs/architecture/00_master_topology.md) | **High-Level Topology & Data Flow** | Component bus, timing budgets, data pipelines, and AutoAscend architectural gap analysis. |
| [01_sensory_and_nle_interface.md](file:///home/bae/corp/docs/architecture/01_sensory_and_nle_interface.md) | **Sensory Boundary & NLE Interface** | Observation tensor normalization, AutoMore wrapper, multi-char prompts, and Anomaly Sentry interrupt rules. |
| [02_epistemic_pomdp_engine.md](file:///home/bae/corp/docs/architecture/02_epistemic_pomdp_engine.md) | **Epistemic POMDP Belief Engine** | Bayesian belief state algebra, observation listeners (altar, pet, price ID, engrave, sink), and entropy gates. |
| [03_fast_htn_planner.md](file:///home/bae/corp/docs/architecture/03_fast_htn_planner.md) | **Fast Core: HTN Planner** | Formal HTN grammar, dynamic persona profiling, precondition guards, cycle detection, and Nogood branch pruning. |
| [04_deliberative_llm_core.md](file:///home/bae/corp/docs/architecture/04_deliberative_llm_core.md) | **Slow Core: Deliberative LLM Engine** | Multi-backend client (OpenRouter, Google Gemma, local llama.cpp QwQ-32B), `<think>` protocol, and JSON schemas. |
| [05_memory_systems_and_rag.md](file:///home/bae/corp/docs/architecture/05_memory_systems_and_rag.md) | **Dual-Store Memory Architecture** | Offline NetHack Wiki ETL, LanceDB + bm25s hybrid search, Strict-Null reranker, and CDCL Nogood store. |
| [06_domain_managers_and_workers.md](file:///home/bae/corp/docs/architecture/06_domain_managers_and_workers.md) | **Domain Managers & Worker Layer** | Spatial navigation (BFS/A*), tactical combat kiting, nutrition/inventory optimization, Sokoban, and atomic keystrokes. |
| [07_autopsy_and_benchmarking.md](file:///home/bae/corp/docs/architecture/07_autopsy_and_benchmarking.md) | **Autopsy Pipeline & Evaluation** | Flight recorder, LLM post-mortem analysis, Competence Discovery Engine, and head-to-head AutoAscend benchmarks. |
| [08_roadmap_and_tech_stack.md](file:///home/bae/corp/docs/architecture/08_roadmap_and_tech_stack.md) | **Implementation Roadmap & Tech Stack** | Evaluated dependency suite, phased milestones, interface verification, and validation plan. |
| [09_bottlenecks_edge_cases_and_flaws.md](file:///home/bae/corp/docs/architecture/09_bottlenecks_edge_cases_and_flaws.md) | **Bottlenecks, Designs & Fatal Flaws** | Process concurrency, inventory hashing, micro-action fibers, 64-bit predicate registry, and NetHack instakill interlocks. |
| [10_rust_acceleration_and_pyo3_bindings.md](file:///home/bae/corp/docs/architecture/10_rust_acceleration_and_pyo3_bindings.md) | **Rust Acceleration & PyO3 Spine** | "Python Brain, Rust Spine" architecture, 80x21 bitmask A*, SIMD Nogood cuts, and maturin build config. |

---

## 3. Technology Highlights

* **Language**: Hybrid Python 3.11/3.12 + **Rust 2021** native extension (`lox_core` compiled via `maturin` and managed by `uv`).
* **Environment**: `nle` (NetHack Learning Environment 0.9.x) + `minihack` for unit-level tactical regressions.
* **Lexical & Vector Retrieval**: `SQLite FTS5` (C-based column-weighted BM25) + `LanceDB` (Apache Arrow embedded) + `bm25s`.
* **Embeddings & Reranker**: `FastEmbed` (`BAAI/bge-small-en-v1.5`) + `FlashRank` (`ms-marco-TinyBERT`) running 100% on CPU ONNX runtime (0 MB VRAM).
* **Deliberative Reasoning**: Local `llama.cpp` (`llama-server`) running `QwQ-32B` (AWQ/GGUF 4-bit) or remote OpenRouter / Google Gemma endpoints via a unified typed client.
