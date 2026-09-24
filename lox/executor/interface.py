"""
LOX-ψ Executor Boundary: the DomainAdapter ABC (R5, AGENT_PLAN §4.3).

The generalization contract: the policy diff DSL references ONLY what an adapter's
`predicates()` and `goal_handlers()` expose — the same grammar, different bindings.
Porting a domain = implementing the adapter + writing its certification set; the loop,
validator, DSL, and program machinery are untouched.

Adapters registered via `lox.executor.register(adapter)`; looked up by
`lox.executor.get_adapter(domain)`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DomainSpec:
    name: str
    obs_layout: str                  # description of the observation vector/dict
    grid_shape: tuple[int, int]      # (rows, cols)
    n_actions: int
    seeded: bool                     # whether make_env honors seeds
    step_limit_default: int = 5000
    notes: str = ""


@dataclass(slots=True)
class PredicateBinding:
    """A named predicate bound to a domain callable. Signature mirrors the manifest:
    (args: list[tuple], desc: str, fn: callable)."""
    args: list[tuple]
    desc: str
    fn: Callable


@dataclass(slots=True)
class ParamLeaf:
    path: str                        # dotted path WITHOUT domain prefix (e.g. "explore.max_steps")
    type: str                        # int | float | bool
    default: int | float | bool
    min: int | float | bool
    max: int | float | bool
    desc: str = ""


@dataclass(slots=True)
class GoalHandler:
    """A certified goal: name + owned params + a step function."""
    name: str
    owned_params: list[str] = field(default_factory=list)
    desc: str = ""


@dataclass(slots=True)
class CertCase:
    """One certification case: run fn(seed) and require predicate(result) True."""
    name: str
    seed: int
    max_steps: int
    check: Callable[[dict], bool]    # result dict from the episode runner
    desc: str = ""


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class DomainAdapter(ABC):
    """One adapter per environment. The revision loop, validator, DSL, corpus, and
    ablation harness are domain-agnostic and consume only this interface."""

    spec: DomainSpec

    @abstractmethod
    def predicates(self) -> dict[str, PredicateBinding]:
        """Domain-specific named predicates (the manifest's predicate table)."""

    @abstractmethod
    def param_leaves(self) -> dict[str, ParamLeaf]:
        """Domain param leaves with bounds (diff-tunable via (set policy_params.<path> ...))."""

    @abstractmethod
    def goal_handlers(self) -> dict[str, GoalHandler]:
        """Certified goal handlers (goal names usable by strategy_plan / goal ops)."""

    @abstractmethod
    def verbs(self) -> dict[str, str]:
        """Closed tactic-verb set for this domain (may be a subset of the master list)."""

    @abstractmethod
    def make_env(self, seed: int | None = None, **cfg: Any) -> Any:
        """Constructs a fresh env instance."""

    @abstractmethod
    def run_episode(self, env: Any, program: Any, seed: int | None,
                    max_steps: int) -> dict:
        """Runs ONE episode with the domain agent, driving the program's compiled
        config. Returns {reward, steps, success, coverage, goal_events: [...], ...}."""

    def report_bundle(self, results: list[dict]) -> dict:
        """Run-report for the reviser. R5 metric-gap fix: ExploreMaze reward saturates
        at 100% success — the batch report therefore carries STEP-EFFICIENCY metrics
        (median steps on successes + coverage) so revisions are distinguishable."""
        n = len(results) or 1
        successes = [r for r in results if r.get("success")]
        steps_success = sorted(r.get("steps", 0) for r in successes)
        median_steps_success = (steps_success[len(steps_success) // 2]
                                if steps_success else None)
        return {
            "domain": self.spec.name,
            "batch": {
                "episodes": len(results),
                "mean_reward": sum(r.get("reward", 0.0) for r in results) / n,
                "success_rate": len(successes) / n,
                "mean_steps": sum(r.get("steps", 0) for r in results) / n,
                "median_steps_on_success": median_steps_success,
                "mean_coverage": sum(r.get("coverage", 0) for r in results) / n,
                # per-seed spread: the author must see variance so it stops
                # chasing noise (ledger shows oscillating monster_cost revisions)
                "steps_on_success_min_max": ([steps_success[0], steps_success[-1]]
                                              if steps_success else None),
                "per_episode": [
                    {"seed": r.get("seed"), "steps": r.get("steps"),
                     "success": r.get("success")} for r in results
                ],
            },
            "metric_note": ("reward saturates at 100% success on ExploreMaze; "
                            "optimize median_steps_on_success (lower is better) "
                            "without dropping success_rate"),
            "goal_stats": {},
            "variance_note": "seeded domain" if self.spec.seeded else "unseeded",
        }

    @abstractmethod
    def certification_suite(self) -> list[CertCase]:
        """Per-domain skill cases (validator gate 11)."""

    @abstractmethod
    def rag_context(self, query: str = "", k: int = 4) -> list[str]:
        """Wiki/docs slices for the author LLM (from the domain's corpus)."""

    def default_program(self) -> dict:
        """Cold-start policy program for this domain (goals use certified handlers)."""
        raise NotImplementedError

    def manifest(self, version: int):
        """Machine-generates the domain's VocabularyManifest from the registries
        (drift-guard: the manifest is derived from code, never hand-written)."""
        from lox.policy.manifest import VocabularyManifest  # noqa: PLC0415
        return VocabularyManifest(
            domain=self.spec.name,
            version=version,
            predicates={n: {"args": b.args, "desc": b.desc}
                        for n, b in self.predicates().items()},
            verbs=dict(self.verbs()),
            goals={n: {"params": g.owned_params} for n, g in self.goal_handlers().items()},
            param_leaves={p.path: {"type": p.type, "min": p.min, "max": p.max,
                                   "default": p.default}
                          for p in self.param_leaves().values()},
            macros={},
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, DomainAdapter] = {}


def register(adapter: DomainAdapter) -> DomainAdapter:
    _REGISTRY[adapter.spec.name] = adapter
    return adapter


def get_adapter(domain: str) -> DomainAdapter:
    if domain not in _REGISTRY:
        _lazy_load(domain)
    if domain not in _REGISTRY:
        raise KeyError(f"no adapter registered for domain {domain!r} "
                       f"(available: {sorted(_REGISTRY)})")
    return _REGISTRY[domain]


def available_domains() -> list[str]:
    return sorted(_REGISTRY)


def _lazy_load(domain: str) -> None:
    if domain == "nethack":
        from lox.executor.nethack_adapter import NethackAdapter  # noqa: PLC0415
        register(NethackAdapter())
    elif domain == "minihack":
        from lox.executor.minihack_adapter import MiniHackAdapter  # noqa: PLC0415
        register(MiniHackAdapter())
    elif domain == "craftax":
        from lox.executor.craftax_adapter import CraftaxAdapter  # noqa: PLC0415
        register(CraftaxAdapter())