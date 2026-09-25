# LOX-ψ Development Roadmap & Backlog

## Future Milestone: The Diagnostic Critic & ECO Subsystem

> **Status**: Deferred per design decision (2026-09-24). The synthesis loop will currently run with single-model autonomous architect with interactive multi-turn self-repair. The Critic will be integrated in a subsequent phase when multi-agent orchestrations are enabled.

---

### Overview & Motivation
The Diagnostic Critic decouples **Data Analysis** (big data statistical aggregation over 24M+ ticks) from **Policy Synthesis** (translating insights into AST diffs and macros). It functions as an automated Site Reliability Engineer / Data Scientist for the policy author.

---

### Architecture & Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. DATA INGESTION                                                          │
│    Reads latest 100-200 episode batch from DuckDB                          │
│    Calls analyze_bottlenecks(), trace_causal_pivot(), compare_trajectories()│
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 2. THE DIAGNOSTIC CRITIC (Fast Model / Gemini Flash)                       │
│    Analyzes telemetry, failure distributions, and causal pivot divergences │
│    DOES NOT WRITE CODE OR DIFFS                                            │
│    Emits structured Engineering Change Order (ECO) in JSON                 │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      │
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 3. THE POLICY ARCHITECT (Gemma 4 31B)                                      │
│    Receives: ECO + Macro Manifest + Wiki FTS5                              │
│    Authors: Clean Pythonic Infix AST diffs + Parameterized defmacro         │
│    Pre-tests with simulate_diff() before finalizing                        │
└────────────────────────────────────────────────────────────────────────────┘
```

---

### Structured ECO Schema

```json
{
  "eco_id": "ECO-YYYYMMDD-XX",
  "primary_bottleneck": "Short description of the dominant bottleneck holding back median depth",
  "empirical_evidence": {
    "median_depth": 2.0,
    "max_depth": 6,
    "top_death_reasons": [
      {"cause": "You faint from lack of food", "count": 212, "pct": 4.5},
      {"cause": "The giant rat bites!", "count": 205, "pct": 4.3}
    ],
    "goal_time_allocation": {
      "explore": 0.78,
      "descend": 0.04,
      "forage": 0.08
    }
  },
  "root_cause_diagnosis": "Detailed breakdown of the mechanical or strategic flaw leading to the bottleneck",
  "strategic_directives": [
    "Concrete strategic actions for the Architect to implement",
    "e.g. Define macro emergency_dive when hunger >= HUNGRY and stairs known",
    "e.g. Reprioritize descend above explore during food crisis"
  ],
  "target_subsystems": ["descent", "nutrition", "macros"],
  "counter_indicators": "Policies or actions to explicitly avoid based on negative regression data"
}
```

---

### Planned Implementation Tasks

1. **`lox/policy/critic_agent.py`**:
   - `generate_eco(batch_id, telemetry_summary)`: Standalone agent function invoking fast model with structured JSON output schema.
2. **`scripts/run_critic_evaluation.py`**:
   - A/B benchmark harness comparing synthesis convergence rate with vs without Critic ECO pre-briefs.
3. **Integration into Campaign Runner**:
   - Add `--enable-critic` flag to `scripts/run_depth10_campaign.py`.
