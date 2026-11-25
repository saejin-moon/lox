"""OpenAI-compatible tool specifications for Option-A Peer Swarm Author sessions."""

from __future__ import annotations

import copy as _copy
from typing import Any

from lox.lang.validator import normalize_mode

OPENAI_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "query_db",
            "description": (
                "Execute ONE read-only SELECT SQL query on the DuckDB database "
                "(tables/views: champion_runs, champion_ticks, champion_function_stats, "
                "peer_runs, peer_ticks, all_ticks, runs, ticks, batches, function_stats, "
                "policies, experiments, theories, expert_runs, expert_ticks). "
                "Never pass multiple ;-separated statements."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "Single read-only SQL SELECT query (one statement only).",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_wiki",
            "description": "Search the offline NetHack 3.6.6 Wiki for monster stats, items, corpses, intrinsics, or branch mechanics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search topic or entity name (e.g., 'floating eye', 'Excalibur', 'hungry', 'altar').",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of wiki articles to return (1-5).",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_monster",
            "description": (
                "Query exact C-level engine permonst struct for a monster name "
                "(level, speed, AC, magic resistance, nutrition, corpse weight, conveyed intrinsics, resistances)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Monster name to inspect (e.g., 'jackal', 'floating eye', 'orc', 'lichen').",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_item",
            "description": (
                "Query exact C-level engine objclass struct for an item name "
                "(class, cost, weight, and nutrition for food items)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Item name to inspect (e.g., 'food ration', 'potion of extra healing', 'long sword').",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_policy_docs",
            "description": "Retrieve Policy Language documentation, GameState fields, Action names, and active skills listing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "'all', 'policy', 'skills', 'state', or 'actions'.",
                        "default": "all",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_theories",
            "description": (
                "Search the DuckDB `theories` knowledge store for prior peer THEORY experiments "
                "and campaign findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keywords (e.g. 'doorway stall', 'starvation', 'oscillation').",
                    },
                    "doc_type": {
                        "type": "string",
                        "description": "Optional filter (e.g. 'THEORY').",
                        "default": "",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max documents to return (default 6).",
                        "default": 6,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_skill",
            "description": (
                "Read Python source (full or line slice `start_line..end_line`) of a skill module "
                "(e.g. 'world', 'survival/core', 'goals/explore') from workspace or a historical `policy_id`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Skill module path/key (e.g. 'world', 'survival/core', 'navigation/core').",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed start line for partial reads (default: 1).",
                        "default": 1,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed inclusive end line (omit for full file).",
                    },
                    "policy_id": {
                        "type": "string",
                        "description": "Optional historical policy_id (e.g. 'policy_swarm_g12') to read that policy's snapshot.",
                        "default": "",
                    },
                },
                "required": ["skill_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_symbol",
            "description": "Read an exact function, class, or `Class.method` definition with line numbers (e.g. 'survival/core.check_interrupt' or 'Policy.step').",
            "parameters": {
                "type": "object",
                "properties": {
                    "qualified_name": {
                        "type": "string",
                        "description": "Qualified or bare symbol name (e.g. 'goals/core.GoalRunner.tick', 'world.WorldState', 'step_toward').",
                    }
                },
                "required": ["qualified_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_policy",
            "description": "Read the current workspace `policy.py` (when `policy_id` is omitted or `'current'`) or a historical `policy_id` snapshot from DuckDB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id": {
                        "type": "string",
                        "description": "Optional policy ID from the `policies` table (e.g. 'policy_swarm_g12'), or omit/'current' to read active policy.py.",
                        "default": "",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diff_policies",
            "description": "Compute unified diffs of `policy.py` and `skills/**/*.py` between two stored `policy_id` snapshots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_id_a": {
                        "type": "string",
                        "description": "Baseline/earlier policy_id (e.g. 'policy_swarm_g12').",
                    },
                    "policy_id_b": {
                        "type": "string",
                        "description": "Target/newer policy_id (e.g. 'policy_swarm_g16').",
                    },
                },
                "required": ["policy_id_a", "policy_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dry_run_policy",
            "description": (
                "Compile and test candidate Python policy or skill edits in NetHackHarness. "
                "Runs Pyright + cross-module checks, smoke check + behavioral critic, failure & frontier replays, "
                "and Paired CRN Micro-Pilot when configured. Pass ONLY new/modified functions inside "
                "<skill_code module='domain/submodule'>...</skill_code> / <policy_code>...</policy_code> — do NOT "
                "re-paste unchanged skill functions or wrap in <hypothesis> tags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Candidate code bundle to validate and dry-run. Wrap any new/updated skills inside "
                            "<skill_code module='domain/submodule'>...</skill_code> tags and policy edits/root inside "
                            "<policy_code>...</policy_code> (raw Python also accepted)."
                        ),
                    },
                    "seed": {
                        "type": "integer",
                        "description": "NetHack seed to test on.",
                        "default": 42,
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "Number of steps to simulate (10-2000).",
                        "default": 150,
                    },
                    "replay_run_id": {
                        "type": "string",
                        "description": "Optional run_id from DuckDB to fast-forward into that exact mid-game state before executing candidate policy.",
                        "default": "",
                    },
                    "replay_steps": {
                        "type": "integer",
                        "description": "Number of initial actions from replay_run_id to fast-forward (0 defaults to 25 steps before run end).",
                        "default": 0,
                    },
                    "smoke_only": {
                        "type": "boolean",
                        "description": "Set to true to run only the fast single-seed smoke check without consuming a paired micro-pilot.",
                        "default": False,
                    },
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_wiki_article",
            "description": (
                "Reads full untruncated NetHackWiki or 3.6.6 Guidebook article/section (use after `query_wiki` "
                "or when an article was partially shown). Pass optional `section` heading or `offset`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Exact NetHackWiki article title.",
                    },
                    "section": {
                        "type": "string",
                        "description": "Optional section heading substring.",
                        "default": "",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Character offset for pagination.",
                        "default": 0,
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Optional character cap (0 = full article).",
                        "default": 0,
                    },
                },
                "required": ["title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_references",
            "description": "Lists definitions and all call/import sites for a symbol across `policy.py` and `skills/**/*.py`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Identifier/function/attribute name to find.",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_skills",
            "description": "Searches a regex or substring across `policy.py` and all `skills/**/*.py` modules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Case-insensitive regex or text pattern.",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "architecture_map",
            "description": "AST-indexed overview of all modules (with LOC), `register_goal(...)` Goals, and typed `ws.domain(...)` sub-states.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "benchmark_expert_parity",
            "description": "Compares champion `runs` vs human `expert_runs` across progression, AC, XL, intrinsics, and branches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of expert reference digests to return (1-30).",
                        "default": 10,
                    }
                },
                "required": [],
            },
        },
    },
]


def get_tool_specs(mode: str = "SWARM_PEER") -> list[dict[str, Any]]:
    """
    Returns mode-filtered OpenAI tool definitions.
    In `SWARM_PEER` mode, omits redundant `query_policy_docs` (already in the static system prompt).
    """
    canon = normalize_mode(mode)
    if canon != "SWARM_PEER":
        return _copy.deepcopy(OPENAI_TOOL_SPECS)

    return [
        _copy.deepcopy(spec)
        for spec in OPENAI_TOOL_SPECS
        if str((spec.get("function") or {}).get("name", "")) != "query_policy_docs"
    ]
