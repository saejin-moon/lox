"""
LOX-ψ NetHack DomainAdapter facade (R5, refactor step 10).

Thin facade over the existing subsystems — the NetHack flow is the incumbent path
(lox_agent priority cascade + benchmark harness); this adapter exposes it through the
DomainAdapter boundary so the revision loop, corpus, and ablation harness are
domain-agnostic. Behavior is unchanged: run_episode delegates to LoxAgent.
"""
from __future__ import annotations

from typing import Any

from lox.executor.interface import (
    CertCase, DomainAdapter, DomainSpec, GoalHandler, ParamLeaf, PredicateBinding,
)


class NethackAdapter(DomainAdapter):
    spec = DomainSpec(
        name="nethack",
        obs_layout="NLE dict: glyphs/chars/blstats(26)/message + AutoMoreWrapper instrumentation",
        grid_shape=(21, 79),
        n_actions=121,
        seeded=False,   # NetHackChallenge-v0 forbids seeding
        step_limit_default=20000,
        notes="hard showcase domain; full priority cascade + HTN + nogoods",
    )

    # ------------------------------------------------------------------
    def predicates(self) -> dict[str, PredicateBinding]:
        from lox.policy.manifest import PREDICATE_SIGNATURES  # noqa: PLC0415
        from lox.policy.predicates import nethack_bindings  # noqa: PLC0415
        out: dict[str, PredicateBinding] = {}
        for name, sig in PREDICATE_SIGNATURES.items():
            out[name] = PredicateBinding(
                args=sig["args"], desc=sig["desc"],
                fn=lambda *a, _name=name, **kw: _nethack_call(_name, a, kw),
            )
        return out

    def param_leaves(self) -> dict[str, ParamLeaf]:
        from lox.policy.manifest import param_leaves_from_config  # noqa: PLC0415
        return {
            path: ParamLeaf(path=path, type=s["type"], default=s["default"],
                            min=s["min"], max=s["max"])
            for path, s in param_leaves_from_config().items()
        }

    def goal_handlers(self) -> dict[str, GoalHandler]:
        specs = {
            "explore_floor": {"params": []},
            "forge_excalibur": {"params": ["strategy.excalibur_timeout_steps"]},
            "hunt_poison_res": {"params": []},
            "enter_sokoban": {"params": ["descent.sokoban_hunt_step_cap",
                                         "descent.sokoban_entry_depth_lo",
                                         "descent.sokoban_entry_depth_hi"]},
            "goto_minetown": {"params": ["strategy.minetown_timeout_steps",
                                         "descent.minetown_donations_done"]},
            "descend": {"params": ["descent.min_hp_frac", "descent.deep_min_hp_frac",
                                   "descent.deep_min_depth"]},
            # R6 Ascension Knowledge Stack milestones (mortality-driven order)
            "early_survival_stack": {"params": ["strategy.survival_ac_target",
                                                  "strategy.survival_timeout_steps",
                                                  "strategy.survival_exit_depth"]},
            "equip_upgrade": {"params": ["strategy.equip_ac_target",
                                          "strategy.equip_timeout_steps",
                                          "strategy.equip_exit_depth"]},
            "survival_intrinsics": {"params": ["strategy.intrinsics_min_depth",
                                                "strategy.intrinsics_exit_depth",
                                                "strategy.intrinsics_timeout_steps"]},
            "enter_gehennom": {"params": ["strategy.gehennom_min_depth"]},
            "castle_wishing": {"params": ["strategy.castle_min_depth"]},
            "vlad_invocation": {"params": ["strategy.vlad_min_depth"]},
            "ascension_run": {"params": []},
        }
        return {name: GoalHandler(name=name, owned_params=spec["params"])
                for name, spec in specs.items()}

    def verbs(self) -> dict[str, str]:
        from lox.policy.manifest import VERBS  # noqa: PLC0415
        return dict(VERBS)

    # ------------------------------------------------------------------
    def make_env(self, seed: int | None = None, **cfg: Any) -> Any:
        from lox.env.nle_wrapper import make_env  # noqa: PLC0415
        return make_env(**cfg)

    def run_episode(self, env: Any, program: Any, seed: int | None,
                    max_steps: int) -> dict:
        from lox.agent.lox_agent import LoxAgent  # noqa: PLC0415
        agent = LoxAgent(env=env)
        result = agent.run_episode(max_steps=max_steps)
        return {
            "reward": float(result.final_score),
            "steps": int(result.total_turns),
            "success": getattr(result, "ascended", False),
            "max_depth": int(result.max_depth),
            "death_message": getattr(result, "death_message", ""),
            "goal_events": [],
        }

    def certification_suite(self) -> list[CertCase]:
        # NetHack certifications run via scripts/run_skill_certifications.py (the
        # validator's cert gate delegates to it in the revision loop); this suite
        # exposes the same contract for the ablation harness.
        return [
            CertCase(name="nethack_smoke", seed=0, max_steps=2000,
                     check=lambda r: r.get("steps", 0) > 0,
                     desc="agent runs a full episode without crashing"),
        ]

    def rag_context(self, query: str = "", k: int = 4) -> list[str]:
        from lox.policy.corpus import rag_slices  # noqa: PLC0415
        return rag_slices("nethack", query=query or "survival tactics", k=k)

    def default_program(self) -> dict:
        import copy  # noqa: PLC0415
        return copy.deepcopy(DEFAULT_PROGRAM_DICT)


def _nethack_call(name: str, args: tuple, kw: dict) -> bool:
    """Predicate evaluation against the current GoalInterpreter-style context dict
    passed via kw['ctx'] (the adapter binds ctx at evaluation time)."""
    ctx = kw.get("ctx")
    if ctx is None:
        raise ValueError("nethack predicates require a ctx kwarg")
    bindings = nethack_bindings(ctx)
    if name not in bindings:
        raise ValueError(f"ERR_UNKNOWN_SYMBOL: {name!r}")
    return bool(bindings[name](*args))


from lox.policy.program import DEFAULT_PROGRAM as DEFAULT_PROGRAM_DICT  # noqa: E402