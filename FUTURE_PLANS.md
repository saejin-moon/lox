# LOX: Roadmap to a NeurIPS / ICLR Oral & Best Paper Contender (`FUTURE_PLANS.md`)

This document details the architectural extensions, experimental matrix, ablation suite, external baselines, theoretical propositions, and figure layout needed to position **LOX** for a **Spotlight / Oral / Best Paper Finalist** slot at **ICLR, NeurIPS, or ICML**.

---

## 1. Core Paper Positioning: Solving the Long-Horizon Agent Trilemma

Existing approaches to open-ended, long-horizon sequential decision-making (`25,000`-step NetHack, multi-stage Craftax) are trapped in a strict three-way trade-off:

| Paradigm | Representative Works | Zero Human Heuristics? | Test-Time Speed & Cost | Interpretable & Modular? | Long-Horizon Credit Assignment |
|---|---|---|---|---|---|
| **Deep Neural RL** | `IMPALA`, `PPO`, `CDGPT5` | Yes | Fast (`~5k sps`), `$0` | No (black-box weights) | Poor on sparse combinatorial tech trees |
| **Handcrafted Symbolic Bots** | `AutoAscend` (NeurIPS '22 Award), `BotHack` | **No** (years of human coding) | Fast (`~8k sps`), `$0` | Yes (readable code) | Human-debugged |
| **Per-Step LLM Agents** | `BALROG`, `ReAct`, `Reflexion` | Yes | **Slow (`~0.5 sps`), `$10–$50`/ep** | Partial (text logs) | Context window overflows after ~500 steps |
| **Black-Box Evolutionary Code Search** | `FunSearch`, `Eureka` | Yes | Fast (`~8k sps`), `$0` | Monolithic scripts | Scalar fitness only; blind to *which* function failed at step 4,000 |
| **LOX (Ours)** | **Grey-Box Programmatic Synthesis** | **Yes ("Hard Code Nothing")** | **Fast (`8,000+ sps`), `$0` test-time** | **Yes (modular AST + macro library)** | **`<1 µs` AST tracing + `Ochiai`/`Liblit` SBFL + `14ms` prefix shards** |

**The Core Thesis of LOX**:
> Instead of placing an LLM inside the runtime environment loop (where it is `10,000x` too slow and goes blind over `25,000` steps) or mutating code blindly against a scalar reward (`FunSearch`/`Eureka`), **LOX places a stateless LLM inside a grey-box compiler and fault-localization loop**. By combining compile-time AST instrumentation (`<1 µs` per call), Spectrum-Based Fault Localization (`Ochiai` and `Liblit`), `14ms` C-level prefix-shard replay, Anytime-Valid Confidence Sequences, and zero-token CRN-SPSA parameter tuning, LOX synthesizes human-readable, `8,000+ steps/sec` symbolic control policies from scratch with **zero hardcoded domain heuristics**.

---

## 2. Next-Wave Architectural Upgrades (Pushing NetHack from Dlvl 3–4 $\to$ Mid-Game Dlvl 8–15+)

To reach the "Symbolic Bot Without Humans" milestone (Sokoban, Minetown, Oracle, `avg_depth >= 5.0–8.0`), implement these four targeted upgrades:

### 2.1 Persistent Frontier Shard Archive ("Go-Explore in Program Space")
- **Why**: By Generation 10+, the policy reliably clears `Dlvl 1–3`, so new bugs only appear on `Dlvl 5+` (e.g., Shopkeepers, Minetown altars, Sokoban boulders, Cockatrice stoning, floating eyes). Running from Turn 0 wastes 90% of steps re-clearing `Dlvl 1–3`.
- **Implementation (`lox/db/shards.py` + `lox/db/schema.py`)**:
  - Add a persistent `frontier_shards` table in DuckDB keyed by behavioral cell `(dungeon_num, depth, situation_tag)` (e.g., `(0, 4, "oracle_level")`, `(2, 3, "minetown")`, `(4, 1, "sokoban")`, `(0, 5, "locked_shop")`, `(0, 3, "floating_eye_melee")`).
  - Whenever *any* episode across any generation reaches a new cell or encounters a new mid-game hazard, save its exact `action_bytes[:byte_offset]` snapshot into `frontier_shards`.
  - During `FUNC_MOD_ADD` and `dry_run_policy`, stratify shards across both **recent failure windows** and **frontier cells**, allowing the Author Agent to immediately test mid-game skills in `14ms` without replaying 2,000 early-game turns.

### 2.2 Differential Trace Slicing (Failure vs. Outlier/Expert Contrast)
- **Why**: Showing the LLM *only* its failed run log forces it to guess what a good trajectory looks like in that same state.
- **Implementation (`lox/db/summarizer.py` + `lox/author/prompts.py`)**:
  - When a target function $f$ has high `Liblit` failure importance, automatically retrieve:
    1. Two failed trajectories where $f$ was the `last_active_function` before death/stall.
    2. Two `is_outlier_good = TRUE` (or `expert_runs`) trajectories at the same `depth` and hazard class.
  - Present a compact **Differential Trace Slice** in the generation digest (`Failing trace in handle_combat vs. Outlier trace at Dlvl 3`), directly grounding the LLM's causal YAML hypothesis.

### 2.3 Quality-Diversity Crossover Mode (`NICHE_MERGE`)
- **Why**: `CampaignOrchestrator` already maintains a 3-niche `MAP-Elites` archive (`descent`, `outlier`, `survivability`), but never recombines them.
- **Implementation (`lox/engine/orchestrator.py` + `lox/engine/scheduler.py`)**:
  - When `descent` and `survivability` hold distinct policy IDs, allow the scheduler to trigger a `NICHE_MERGE` generation that presents both policies' best-performing methods (ranked by `Ochiai` score in their respective batches) so the Author Agent can splice `survivability`'s combat/nutrition methods into `descent`'s navigation backbone.

### 2.4 One-Flag Ablation Switches in `scripts/run_lox.py`
Add explicit CLI flags to `scripts/run_lox.py` so every ablation experiment in the paper is a single command:
- `--no-sbfl`: Masks `ochiai_good`, `liblit_fail`, and per-function call/turn attribution in `function_stats` and batch digests (reduces LOX to scalar/log feedback).
- `--no-shards`: Disables prefix-shard extraction in both `dry_run_policy` and `FUNC_MOD_ADD` (forces Turn-0-only evaluation).
- `--no-spsa`: Disables `PARAM_TUNE` (CRN-SPSA) in `ModeScheduler`.
- `--no-macros`: Disables `MacroHarvester`, `MACRO_WRITE`, and external `skills/` modules.
- `--no-pilot`: Sets `pilot_revisions=0` (syntax/compile check only, no 30-episode empirical pilot critic).

---

## 3. The Paper-Winning Ablation Suite

Top-tier reviewers look for a clean, monotonic **Component Ablation Table** and **Generation Scaling Curves** (3–5 seeds per condition, 30–50 generations per seed):

### Ablation 1: Grey-Box Fault Localization (`Full LOX` vs. `--no-sbfl`)
- **Hypothesis Tested**: Does Spectrum-Based Fault Localization (`Ochiai` + `Liblit` via `<1 µs` AST instrumentation) actually improve LLM code synthesis over raw trajectory logs?
- **Metrics to Report**:
  1. **Fault-Localization Hit Rate (%)**: How often the LLM modifies the actual method responsible for the top stall/death cause.
  2. **Pilot Pass Rate (%)** and **Rollout Promotion Rate (%)**.
  3. **Final Avg Depth / P90 Depth / Score** at Generation 30.

### Ablation 2: `14ms` Prefix-Shard Replay (`Full LOX` vs. `--no-shards`)
- **Hypothesis Tested**: Do C-level byte-replay shards solve the mid-game credit assignment bottleneck where bugs on `Dlvl 3+` rarely trigger in short Turn-0 rollouts?
- **Metrics to Report**:
  1. **Wall-Clock Time per Generation (sec)** (shards run 200 steps in ~15ms vs. 2,000+ steps from Turn 0).
  2. **Mid-Game Hazard Survival Rate (%)** on `Dlvl >= 3` (locked doors, stairs descent, multi-monster rooms).
  3. **P90 and Max Depth** across generations.

### Ablation 3: In-Turn Pilot Critic & Scientific Investigation Loop (`Full LOX` vs. `--no-pilot`)
- **Hypothesis Tested**: Does gating full rollouts behind a 30-episode paired pilot + mandatory `query_wiki`/`query_db` revision loop prevent broken hypotheses from wasting full evaluation budgets?
- **Metrics to Report**:
  1. **Full-Rollout Promotion Rate (%)** (expected to jump from ~15–20% without pilot to ~50–70% with pilot).
  2. **Depth Gained per 100,000 Environment Steps**.

### Ablation 4: Hybrid LLM + Zero-Token CRN-SPSA (`Full LOX` vs. `--no-spsa`)
- **Hypothesis Tested**: LLMs are good at discrete control-flow synthesis (`if/else`, BFS, door kicking) but poor at guessing continuous thresholds (`flee_hp_ratio`, `hunger_threshold`, `search_chance`). Does interleaving CRN-SPSA (`PARAM_TUNE`) with LLM code edits improve survival at zero token cost?
- **Metrics to Report**:
  1. **Depth & Score progression per 1M LLM tokens**.
  2. **Before/After `PARAM_TUNE` survival deltas**.

### Ablation 5: Modular Skill Graph & Bottom-Up Macro Harvesting (`Full LOX` vs. `--no-macros`)
- **Hypothesis Tested**: As synthesized policies grow past 350+ lines of Python, monolithic prompting degrades. Does bottom-up `MacroHarvester` + context-scoped AST outlines maintain synthesis accuracy at scale?
- **Metrics to Report**:
  1. **Prompt Token Count vs. Policy Total LOC** across Generations 0 $\to$ 50.
  2. **Compile & Pilot Pass Rate at Generation 25–50** (monolithic vs. context-scoped + harvested macros).

---

## 4. External Baselines to Benchmark Against

To make the empirical section bulletproof across both **NetHack (`nle`)** and **Craftax**:

1. **`Eureka` / `FunSearch` Style Black-Box Evolutionary Code Synthesis (Implemented on the Exact Same Harness)**:
   - Run the same LLM (`google/gemma-4-31b-it` and `z-ai/glm-5`) generating full `class Policy` candidates scored purely by scalar batch mean depth/score (no `DuckDB` SQL tool, no `Ochiai`/`Liblit` table, no `FUNC_MOD_ADD` surgical method splicing, no prefix shards).
   - **Why this is critical**: It proves on identical hardware, identical LLM, and identical `GameState` API that **LOX's grey-box program-analysis loop** is what drives the performance leap.
2. **Live Per-Step LLM Agents (`BALROG`-Style `Zero-Shot` / `ReAct`)**:
   - Evaluate the same LLM prompted on every step with the screen/message observation over 100 episodes.
   - Report the **Test-Time Compute & Cost Table**:
     - Per-Step LLM: `~0.5 steps/sec`, `~$15.00` per episode, `100%` network dependency.
     - LOX Synthesized Policy: `~8,000 steps/sec` (**`16,000x` faster**), **`$0.00` test-time cost**, runs offline on CPU in `<200 lines` of auditable Python.
3. **Deep RL Baselines (`IMPALA` / `Sample Factory PPO`)**:
   - Compare against published / reference `NLE` (`NetHackScore-v0`) and `Craftax-Symbolic` neural RL checkpoints at matched environment-step budgets (`10M`, `50M`, `100M` steps).
4. **Cross-Backbone & Cross-Role Generalization Matrix**:
   - **3 LLM Backbones**:
     1. `google/gemma-4-31b-it` (Open-weight 31B)
     2. `z-ai/glm-5` (Frontier open-weight)
     3. `anthropic/claude-sonnet-4.5` or `openai/gpt-5` (Proprietary frontier)
   - **3 NetHack Starting Roles (Zero Engine Changes)**:
     1. `val-hum-law-fem` (Valkyrie — melee/armor specialization)
     2. `wiz-elf-cha-mal` (Wizard — force bolt/spell/kite specialization)
     3. `mon-hum-neu-mal` (Monk — unarmed/martial arts/vegetarian conduct)
   - **2 Distinct Benchmarks**:
     1. **NetHack (`nle`)**: Avg Depth, P90 Depth, Max Depth, Score, Ascension/Mid-Game Milestone Rates.
     2. **Craftax (`lox/env/craftax`)**: Progression Score, Achievement Unlock Rates (`COLLECT_WOOD` $\to$ `PLACE_TABLE` $\to$ `MAKE_WOOD_PICKAXE` $\to$ `COLLECT_STONE` $\to$ `MAKE_STONE_PICKAXE` $\to$ `COLLECT_IRON` $\to$ `PLACE_FURNACE` $\to$ `MAKE_IRON_PICKAXE` $\to$ `COLLECT_DIAMOND`).

---

## 5. Mathematical & Theoretical Contributions (Section 4 of the Paper)

Just like `THIEF` pairs systems engineering with clean propositions, include two formal propositions in the LOX paper:

1. **Proposition 1 (Anytime-Valid Type-I Error Control under Sequential Futility Stopping)**:
   - Formalize the Empirical Bernstein Confidence Sequence $[L_t, U_t]$ in `lox/engine/hypothesis.py` (`compute_confidence_sequence`) and prove that stopping early when $U_t \le 0$ (`should_stop_early_for_futility`) preserves the family-wise false-promotion rate $\mathbb{P}(\exists t \ge 1 : \mu \le 0 \text{ and } L_t > 0) \le \alpha$ while reducing expected evaluation cost on regressed candidates by $60\text{--}75\%$.
2. **Proposition 2 (Variance Reduction of Common-Random-Number Prefix-Shard Estimators)**:
   - Formalize a trajectory return $G = G_{0:\tau} + G_{\tau:T}$ split at prefix byte offset $\tau$ (where the modified function $f$ first activates). Prove that evaluating $\Delta_f = G^{\text{cand}}_{\tau:T}(\omega) - G^{\text{base}}_{\tau:T}(\omega)$ from the shared C-state snapshot $s_\tau$ eliminates the pre-trigger variance $\text{Var}(G_{0:\tau})$, strictly increasing the signal-to-noise ratio $\text{SNR}_{\text{shard}} / \text{SNR}_{\text{full}} = 1 + \frac{\text{Var}(G_{0:\tau})}{\text{Var}(G_{\tau:T})}$ and reducing simulation steps by $T / H_{\text{shard}}$ (`25,000 / 200 = 125x`).

---

## 6. Target Paper Figures & Tables Checklist

1. **Figure 1 (Hero Architecture Diagram)**:
   - Closed-loop grey-box synthesis cycle: `Stateless Author Agent` $\xrightarrow{\text{AST Pass}}$ `Instrumented Python Policy (<1 µs tracing)` $\xrightarrow{\text{16-Core C++ NLE/Craftax Pool + 14ms Byte Shards}}$ `RLE Summarizer & Ochiai/Liblit Scorer` $\xrightarrow{\text{DuckDB Telemetry}}$ `Bernstein CS + Wilcoxon + CRN-SPSA`.
2. **Figure 2 (Main Scaling Curves on NetHack & Craftax)**:
   - 2-panel plot across Generations 0 $\to$ 50 showing **Avg Depth / P90 Depth (NetHack)** and **Achievement Unlock Rate / Score (Craftax)** for `LOX (Full)` vs. `Eureka/FunSearch-style Black-Box Code Search`, `No-SBFL`, `No-Shards`, and `Neural PPO`.
3. **Figure 3 (Pareto Frontier: Performance vs. Test-Time Speed & API Cost)**:
   - Log-scale scatter plot of **Steps/Sec (`0.5` vs `8,000+`)** and **Test-Time Cost per 1,000 Episodes (`$15,000` vs `$0.00`)** comparing `LOX` against `BALROG` per-step LLM agents.
4. **Figure 4 (Phylogenetic Tree of Synthesized Skills & `Ochiai`/`Liblit` Heatmap)**:
   - Visualizes a real campaign from `data/lox.duckdb`: how `Gen 0` (`explore`, `handle_combat`) branches into door kicking (`Gen 1`), prompt handling (`Gen 2`), corpse/prayer survival, SPSA threshold tuning (`Gen 3`), and mid-game skills, annotated with the exact `Ochiai` and `Liblit` scores that triggered each edit.
5. **Table 1 (Main Benchmark Table — NetHack 3 Roles + Craftax across 3 LLM Backbones)**.
6. **Table 2 (Systematic Component Ablation Table — 6 Ablations with Paired Wilcoxon $p$-values)**.
