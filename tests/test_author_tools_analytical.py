"""
Unit tests for Gen-3 analytical author tools:
- analyze_bottlenecks()
- trace_causal_pivot()
- compare_trajectories()
- simulate_diff()
"""
import pytest
from lox.policy.author_tools import (
    analyze_bottlenecks,
    trace_causal_pivot,
    compare_trajectories,
    simulate_diff,
    execute_tool_call,
    parse_tool_calls,
)
from lox.policy.manifest import build_manifest
from lox.policy.program import PolicyProgram


def test_parse_tool_calls_analytical():
    text = """
    I will analyze bottlenecks first:
    analyze_bottlenecks(50)
    trace_causal_pivot("ep_valk_123")
    compare_trajectories("ep_deep", "ep_stalled")
    simulate_diff("set nutrition.food_radius = 5")
    """
    calls = parse_tool_calls(text)
    names = [c[0] for c in calls]
    assert "analyze_bottlenecks" in names
    assert "trace_causal_pivot" in names
    assert "compare_trajectories" in names
    assert "simulate_diff" in names


def test_analyze_bottlenecks_runs_cleanly():
    res = analyze_bottlenecks(10)
    assert isinstance(res, str)
    assert "BATCH BOTTLENECK ANALYSIS" in res
    assert "Median Depth" in res


def test_simulate_diff_valid():
    manifest = build_manifest("nethack")
    program = PolicyProgram.load("data/policy_program.json")
    valid_diff = f"""
    revision: {program.version + 1}, parent: {program.version}, author: "test", domain: "nethack", reason: "testing"
    set nutrition.food_radius = 5
    """
    res = simulate_diff(valid_diff, manifest=manifest, program=program)
    assert res.startswith("PASS")


def test_simulate_diff_invalid():
    manifest = build_manifest("nethack")
    program = PolicyProgram.load("data/policy_program.json")
    invalid_diff = """
    revision: 1, parent: 0, author: "test", domain: "nethack", reason: "testing"
    set totally_invalid_param_path = 9999
    """
    res = simulate_diff(invalid_diff, manifest=manifest, program=program)
    assert res.startswith("FAIL")


def test_execute_tool_call_dispatch():
    manifest = build_manifest("nethack")
    program = PolicyProgram.load("data/policy_program.json")
    res = execute_tool_call("analyze_bottlenecks", [20], manifest=manifest, program=program)
    assert "BATCH BOTTLENECK ANALYSIS" in res

    # 0 args dispatch test
    res_zero = execute_tool_call("analyze_bottlenecks", [], manifest=manifest, program=program)
    assert "BATCH BOTTLENECK ANALYSIS" in res_zero
