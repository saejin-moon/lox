"""
The 4 Modification Modes of LOX & CRN-SPSA Parameter Optimizer:
1. `FULL_WRITE`: Initial full policy synthesis (default: 10,000 episodes).
2. `FUNC_MOD_ADD`: Targeted function modification/addition (default: 1,000 triggered episodes per function).
3. `MACRO_WRITE`: Reusable macro synthesis in `macros.py` (default: 0 episodes -> immediate next gen).
4. `PARAM_TUNE`: Zero-order CRN-SPSA parameter finetuning over `self.param(...)` declarations (0 LLM tokens).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import numpy as np

from lox.engine.hypothesis import evaluate_paired_hypothesis, progression_metric
from lox.runner.pool import PersistentEvalPool
from lox.ui.console import ui


@dataclass(frozen=True, slots=True)
class ModeConfig:
    """Configuration for a policy modification mode."""

    name: str
    episodes_per_unit: int
    requires_llm: bool
    requires_trigger_filter: bool


MODES: dict[str, ModeConfig] = {
    "FULL_WRITE": ModeConfig(
        name="FULL_WRITE",
        episodes_per_unit=10000,
        requires_llm=True,
        requires_trigger_filter=False,
    ),
    "FUNC_MOD_ADD": ModeConfig(
        name="FUNC_MOD_ADD",
        episodes_per_unit=1000,
        requires_llm=True,
        requires_trigger_filter=True,
    ),
    "MACRO_WRITE": ModeConfig(
        name="MACRO_WRITE",
        episodes_per_unit=0,
        requires_llm=True,
        requires_trigger_filter=False,
    ),
    "PARAM_TUNE": ModeConfig(
        name="PARAM_TUNE",
        episodes_per_unit=500,
        requires_llm=False,
        requires_trigger_filter=False,
    ),
}


class _ParamConstantUpdater(ast.NodeTransformer):
    """Updates default values inside `self.param("name", default, ...)` or `param("name", default, ...)` calls."""

    def __init__(self, new_defaults: dict[str, float | int]):
        super().__init__()
        self.new_defaults = new_defaults

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        is_param_call = False
        if isinstance(node.func, ast.Name) and node.func.id == "param":
            is_param_call = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "param":
            is_param_call = True

        if (
            is_param_call
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            p_name = node.args[0].value
            if p_name in self.new_defaults:
                val = self.new_defaults[p_name]
                node.args[1] = ast.Constant(value=val)
        return node


def bake_params_into_policy_ast(
    policy_code: str,
    optimized_params: dict[str, float | int],
) -> str:
    """Rewrites `self.param(name, default, ...)` calls in `policy_code` with `optimized_params`."""
    if not optimized_params or not policy_code.strip():
        return policy_code
    try:
        tree = ast.parse(policy_code)
        tree = _ParamConstantUpdater(optimized_params).visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return policy_code


def run_spsa_param_tuning(
    pool: PersistentEvalPool,
    batch_id: str,
    policy_code: str,
    macros_code: str,
    registered_params: dict[str, dict[str, Any]],
    baseline_runs: list[dict[str, Any]],
    iterations: int = 5,
    minibatch_size: int = 24,
    validation_episodes: int = 100,
    seed: int = 777,
    episodes_per_eval: int | None = None,
) -> dict[str, Any]:
    """
    Simultaneous Perturbation Stochastic Approximation with Common Random Numbers (CRN-SPSA).
    Perturbs all `self.param(...)` thresholds simultaneously in normalized [0, 1]^p space,
    evaluates theta+ and theta- on the EXACT same seed mini-batch (Common Random Numbers),
    follows the empirical gradient, and validates the final parameter vector against baseline.
    """
    if episodes_per_eval is not None:
        minibatch_size = int(episodes_per_eval)
    if not registered_params:
        return {
            "tuned": False,
            "optimized_params": {},
            "new_policy_code": policy_code,
            "val_result": None,
            "verdict": {
                "passed": False,
                "mean_delta": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "wilcoxon_p": 1.0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
                "reason": "No self.param(...) declarations found in policy.",
            },
        }

    param_names = sorted(registered_params.keys())
    p_dim = len(param_names)
    rng = np.random.default_rng(seed)

    # Normalize initial defaults to [0, 1]^p
    mins = np.array(
        [float(registered_params[k]["min_val"]) for k in param_names], dtype=np.float64
    )
    maxs = np.array(
        [float(registered_params[k]["max_val"]) for k in param_names], dtype=np.float64
    )
    spans = np.maximum(1e-6, maxs - mins)
    defaults = np.array(
        [float(registered_params[k]["default"]) for k in param_names], dtype=np.float64
    )
    is_ints = [bool(registered_params[k].get("is_int", False)) for k in param_names]

    theta = np.clip((defaults - mins) / spans, 0.0, 1.0)
    best_theta = theta.copy()
    best_gain = 0.0

    def _denorm(vec: np.ndarray) -> dict[str, float | int]:
        out: dict[str, float | int] = {}
        raw = mins + np.clip(vec, 0.0, 1.0) * spans
        for i, k in enumerate(param_names):
            if is_ints[i]:
                out[k] = int(round(float(raw[i])))
            else:
                out[k] = round(float(raw[i]), 4)
        return out

    base_by_seed = {int(r["seed"]): progression_metric(r) for r in baseline_runs}
    base_seeds = (
        [int(r["seed"]) for r in baseline_runs]
        if baseline_runs
        else list(range(10000, 10200))
    )

    # Spall SPSA standard gain schedule
    c0 = 0.12
    a0 = 0.18
    A = max(1.0, 0.1 * iterations)
    alpha_exp = 0.602
    gamma_exp = 0.101

    for k in range(iterations):
        ui.status(
            f"CRN-SPSA tuning parameters iterate {k + 1}/{iterations} (evaluating mini-batches on Common Random Numbers)..."
        )
        ck = c0 / ((k + 1.0) ** gamma_exp)
        ak = a0 / ((k + 1.0 + A) ** alpha_exp)
        delta_vec = rng.choice([-1.0, 1.0], size=p_dim)

        theta_plus = np.clip(theta + ck * delta_vec, 0.0, 1.0)
        theta_minus = np.clip(theta - ck * delta_vec, 0.0, 1.0)

        # Common Random Numbers: evaluate theta_plus and theta_minus on the exact same mini-batch of seeds
        mb_seeds = [
            int(s)
            for s in rng.choice(
                base_seeds, size=min(minibatch_size, len(base_seeds)), replace=False
            )
        ]
        j_base_mb = (
            float(np.mean([base_by_seed[s] for s in mb_seeds if s in base_by_seed]))
            if base_by_seed
            else 0.0
        )

        res_plus = pool.evaluate_batch(
            batch_id=f"{batch_id}_spsa_{k}p",
            policy_code=policy_code,
            macros_code=macros_code,
            target_episodes=len(mb_seeds),
            priority_seeds=mb_seeds,
            param_overrides=_denorm(theta_plus),
            record_ticks=False,
            show_progress=False,
        )
        res_minus = pool.evaluate_batch(
            batch_id=f"{batch_id}_spsa_{k}m",
            policy_code=policy_code,
            macros_code=macros_code,
            target_episodes=len(mb_seeds),
            priority_seeds=mb_seeds,
            param_overrides=_denorm(theta_minus),
            record_ticks=False,
            show_progress=False,
        )

        j_plus = (
            float(np.mean([progression_metric(r) for r in res_plus["runs"]]))
            if res_plus["runs"]
            else 0.0
        )
        j_minus = (
            float(np.mean([progression_metric(r) for r in res_minus["runs"]]))
            if res_minus["runs"]
            else 0.0
        )

        gain_plus = j_plus - j_base_mb
        gain_minus = j_minus - j_base_mb

        if gain_plus > best_gain:
            best_gain = gain_plus
            best_theta = theta_plus.copy()
        if gain_minus > best_gain:
            best_gain = gain_minus
            best_theta = theta_minus.copy()

        grad = ((j_plus - j_minus) / (2.0 * ck)) * (1.0 / delta_vec)
        theta = np.clip(theta + ak * grad, 0.0, 1.0)

    cand_params = _denorm(best_theta if best_gain > 0.0 else theta)
    new_policy_code = bake_params_into_policy_ast(policy_code, cand_params)
    new_macros_code = (
        bake_params_into_policy_ast(macros_code, cand_params)
        if macros_code.strip()
        else macros_code
    )

    val_seeds = base_seeds[: min(validation_episodes, len(base_seeds))]
    val_result = pool.evaluate_batch(
        batch_id=batch_id,
        policy_code=new_policy_code,
        macros_code=new_macros_code,
        target_episodes=len(val_seeds) if val_seeds else validation_episodes,
        priority_seeds=val_seeds,
        param_overrides=cand_params,
        record_ticks=True,
        show_progress=True,
    )
    verdict = evaluate_paired_hypothesis(val_result["runs"], baseline_runs)

    return {
        "tuned": True,
        "optimized_params": cand_params,
        "new_policy_code": new_policy_code,
        "new_macros_code": new_macros_code,
        "val_result": val_result,
        "verdict": verdict,
    }
