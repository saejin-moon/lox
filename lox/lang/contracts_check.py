"""
Undefined-Symbol Gate, Cross-Module Contract Validator, and Pyright `.pyi` Stub Runner.
"""

from __future__ import annotations

import ast
import os
import re
import symtable
from typing import Any

from lox.lang.sandbox import (
    SAFE_BUILTINS_NAMES,
    PolicyValidationError,
    expand_leading_tabs,
    normalize_skill_key,
    resolve_skill_import,
    skill_key_to_module,
    skill_key_to_relpath,
    smart_dedent,
)


def _add_assign_target_names(tgt: ast.AST, out: set[str]) -> None:
    if isinstance(tgt, ast.Name):
        out.add(tgt.id)
    elif isinstance(tgt, (ast.Tuple, ast.List)):
        for elt in tgt.elts:
            _add_assign_target_names(elt, out)
    elif isinstance(tgt, ast.Starred):
        _add_assign_target_names(tgt.value, out)


def _extract_module_exports(tree: ast.Module) -> tuple[set[str], bool]:
    """Returns `(exported_names, has_star_import)` for a skill module AST."""
    exports: set[str] = set()
    has_star = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exports.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                _add_assign_target_names(t, exports)
        elif isinstance(node, ast.AnnAssign):
            _add_assign_target_names(node.target, exports)
        elif isinstance(node, ast.Import):
            for al in node.names:
                exports.add(al.asname or al.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for al in node.names:
                if al.name == "*":
                    has_star = True
                else:
                    exports.add(al.asname or al.name)
    return exports, has_star


def _collect_scope_locals(node: ast.AST) -> set[str]:
    """Collects parameter/local assignment/import/match names inside a function or comprehension scope."""
    locals_set: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        args = node.args
        for a in (
            list(getattr(args, "posonlyargs", []))
            + list(args.args)
            + list(args.kwonlyargs)
        ):
            locals_set.add(a.arg)
        if args.vararg is not None:
            locals_set.add(args.vararg.arg)
        if args.kwarg is not None:
            locals_set.add(args.kwarg.arg)
    if isinstance(node, ast.Lambda):
        return locals_set

    stack: list[ast.AST] = list(ast.iter_child_nodes(node))
    while stack:
        cur = stack.pop()
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            locals_set.add(cur.name)
            continue
        if isinstance(cur, ast.Assign):
            for t in cur.targets:
                _add_assign_target_names(t, locals_set)
        elif isinstance(cur, ast.AnnAssign):
            _add_assign_target_names(cur.target, locals_set)
        elif isinstance(cur, (ast.For, ast.AsyncFor)):
            _add_assign_target_names(cur.target, locals_set)
        elif isinstance(cur, (ast.With, ast.AsyncWith)):
            for it in cur.items:
                if it.optional_vars is not None:
                    _add_assign_target_names(it.optional_vars, locals_set)
        elif isinstance(cur, ast.ExceptHandler) and cur.name:
            locals_set.add(cur.name)
        elif isinstance(cur, ast.NamedExpr):
            _add_assign_target_names(cur.target, locals_set)
        elif isinstance(cur, ast.comprehension):
            _add_assign_target_names(cur.target, locals_set)
        elif isinstance(cur, (ast.Import, ast.ImportFrom)):
            for al in cur.names:
                if al.name != "*":
                    locals_set.add(al.asname or al.name.split(".")[0])
        elif isinstance(cur, ast.MatchAs) and cur.name:
            locals_set.add(cur.name)
        elif isinstance(cur, ast.MatchStar) and cur.name:
            locals_set.add(cur.name)
        elif isinstance(cur, ast.MatchMapping) and cur.rest:
            locals_set.add(cur.rest)
        for ch in ast.iter_child_nodes(cur):
            stack.append(ch)
    return locals_set


def _walk_with_shadowed(tree: ast.AST):
    """Yields `(ast_node, shadowed_names)` respecting local bindings in inner scopes."""

    def _visit(node: ast.AST, shadowed: frozenset[str]):
        yield node, shadowed
        next_shadowed = shadowed
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            next_shadowed = shadowed | frozenset(_collect_scope_locals(node))
        elif isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
            gen_names: set[str] = set()
            for comp in node.generators:
                _add_assign_target_names(comp.target, gen_names)
            next_shadowed = shadowed | frozenset(gen_names)
        for ch in ast.iter_child_nodes(node):
            yield from _visit(ch, next_shadowed)

    return _visit(tree, frozenset())


def check_undefined_symbols(
    policy_code: str,
    skills: dict[str, str] | None = None,
    allowed_globals: frozenset[str] = frozenset(),
) -> None:
    """
    Fast (<2ms) CPython `symtable` undefined-symbol and cross-skill export gate.
    Uses `SAFE_BUILTINS_NAMES` (matching runtime `SAFE_BUILTINS` 1-to-1) so builtins
    absent from the sandbox (e.g. `print`, `dir`, `id`) are caught at compile time.
    """
    known_builtins = SAFE_BUILTINS_NAMES | frozenset(
        {"__name__", "__doc__", "__builtins__", "__package__", "__class__", "skills"}
    )
    global_env = known_builtins | frozenset(allowed_globals)

    skill_map: dict[str, str] = {}
    for raw_k, raw_v in (skills or {}).items():
        nk = normalize_skill_key(raw_k)
        if nk:
            skill_map[nk] = raw_v
    known_skill_keys = set(skill_map.keys())
    skill_pkg_names = {k.split("/", 1)[0] for k in known_skill_keys if "/" in k}
    skill_exports: dict[str, set[str]] = {}
    skill_has_star: dict[str, bool] = {}

    sources: list[tuple[str, str]] = []
    parsed_skill_trees: dict[str, ast.Module] = {}
    for s_name, s_src in sorted(skill_map.items()):
        if not s_src or not s_src.strip():
            continue
        cleaned = smart_dedent(expand_leading_tabs(s_src)).strip()
        rel_fn = skill_key_to_relpath(s_name)
        try:
            tree = ast.parse(cleaned, filename=rel_fn)
        except SyntaxError as exc:
            raise PolicyValidationError(
                f"SyntaxError in {rel_fn} line {exc.lineno}: {exc.msg}"
            ) from exc
        exp, has_star = _extract_module_exports(tree)
        skill_exports[s_name] = exp
        skill_has_star[s_name] = has_star
        parsed_skill_trees[s_name] = tree
        sources.append((rel_fn, cleaned))

    # Propagate transitive `from skills.<mod> import *` exports across skill modules
    for _ in range(len(parsed_skill_trees) + 1):
        changed_exp = False
        for s_name, s_tree in parsed_skill_trees.items():
            for st in s_tree.body:
                if isinstance(st, ast.ImportFrom) and str(st.module or "").startswith(
                    "skills."
                ):
                    dep = resolve_skill_import(str(st.module), known_skill_keys)
                    if dep in skill_exports and any(al.name == "*" for al in st.names):
                        before_n = len(skill_exports[s_name])
                        skill_exports[s_name] |= skill_exports[dep]
                        if (
                            not skill_has_star.get(dep, False)
                            and len(skill_exports[s_name]) > before_n
                        ):
                            changed_exp = True
        if not changed_exp:
            break

    cleaned_pol = smart_dedent(expand_leading_tabs(policy_code or "")).strip()
    if cleaned_pol:
        sources.append(("policy.py", cleaned_pol))

    for filename, src in sources:
        try:
            tree = ast.parse(src, filename=filename)
        except SyntaxError as exc:
            raise PolicyValidationError(
                f"SyntaxError in {filename} line {exc.lineno}: {exc.msg}"
            ) from exc

        from lox.lang.bundle import extract_skill_imports

        imp_info = extract_skill_imports(
            tree, known_skill_keys, pkg_names=skill_pkg_names
        )
        missing_mods = list(imp_info.missing_refs)
        if skills is not None:
            for rk in sorted(imp_info.referenced_keys - known_skill_keys):
                missing_mods.append(skill_key_to_module(rk))
            # Also catch `from skills import foo` when skills == {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "skills":
                    for al in node.names:
                        if al.name == "*":
                            continue
                        cand = normalize_skill_key(al.name)
                        if cand not in known_skill_keys and cand not in skill_pkg_names:
                            missing_mods.append(f"skills.{al.name}")
        if missing_mods and (skill_map or skills is not None):
            raise PolicyValidationError(
                f"Undefined skill module '{missing_mods[0]}' imported in {filename}"
            )
        for s_mod, sym_name in imp_info.direct_symbols.values():
            if s_mod in skill_exports and not skill_has_star.get(s_mod, False):
                if sym_name not in skill_exports[s_mod]:
                    raise PolicyValidationError(
                        f"Symbol '{sym_name}' not exported by {skill_key_to_relpath(s_mod)} (referenced in {filename})"
                    )
        skill_aliases: dict[str, str] = dict(imp_info.module_aliases)

        for node, shadowed in _walk_with_shadowed(tree):
            if isinstance(node, ast.Attribute) and not node.attr.startswith("_"):
                if (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Attribute)
                    and isinstance(node.value.value.value, ast.Name)
                    and node.value.value.value.id == "skills"
                    and "skills" not in shadowed
                ):
                    s_mod = f"{node.value.value.attr}/{node.value.attr}"
                    if s_mod in skill_exports and not skill_has_star.get(s_mod, False):
                        if node.attr not in skill_exports[s_mod]:
                            raise PolicyValidationError(
                                f"Symbol 'skills.{node.value.value.attr}.{node.value.attr}.{node.attr}' "
                                f"not found in {skill_key_to_relpath(s_mod)} (referenced in {filename})"
                            )
                elif (
                    isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name)
                    and node.value.value.id == "skills"
                    and "skills" not in shadowed
                ):
                    s_mod = node.value.attr
                    if s_mod in skill_exports and not skill_has_star.get(s_mod, False):
                        if node.attr not in skill_exports[s_mod]:
                            raise PolicyValidationError(
                                f"Symbol 'skills.{s_mod}.{node.attr}' not found in {skill_key_to_relpath(s_mod)} (referenced in {filename})"
                            )
                elif (
                    isinstance(node.value, ast.Name)
                    and node.value.id in skill_aliases
                    and node.value.id not in shadowed
                ):
                    s_mod = skill_aliases[node.value.id]
                    if s_mod in skill_exports and not skill_has_star.get(s_mod, False):
                        if node.attr not in skill_exports[s_mod]:
                            raise PolicyValidationError(
                                f"Symbol '{node.value.id}.{node.attr}' not found in {skill_key_to_relpath(s_mod)} (referenced in {filename})"
                            )

        try:
            top_st = symtable.symtable(src, filename, "exec")
        except SyntaxError as exc:
            raise PolicyValidationError(
                f"SyntaxError in {filename} line {exc.lineno}: {exc.msg}"
            ) from exc

        mod_defined: set[str] = set()
        for sym in top_st.get_symbols():
            if sym.is_local() or sym.is_imported():
                mod_defined.add(sym.get_name())
        # Also include explicit star-imported symbols from known skills or 2-level skill packages
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and any(
                al.name == "*" for al in node.names
            ):
                mod = str(node.module or "")
                if mod == "skills":
                    for k in known_skill_keys:
                        mod_defined.add(k.split("/", 1)[0])
                elif mod.startswith("skills."):
                    tail = mod[len("skills.") :]
                    if tail in skill_pkg_names and tail not in known_skill_keys:
                        for k in known_skill_keys:
                            if k.startswith(f"{tail}/"):
                                mod_defined.add(k.split("/", 1)[1])
                    else:
                        s_mod = resolve_skill_import(mod, known_skill_keys)
                        if s_mod in skill_exports:
                            mod_defined.update(skill_exports[s_mod])

        valid_globals = global_env | mod_defined

        def _check_table(tbl: symtable.SymbolTable, is_root: bool) -> None:
            for sym in tbl.get_symbols():
                nm = sym.get_name()
                if is_root:
                    if (
                        sym.is_referenced()
                        and not sym.is_local()
                        and not sym.is_imported()
                        and nm not in valid_globals
                    ):
                        raise PolicyValidationError(
                            f"Undefined global symbol '{nm}' at module level in {filename}"
                        )
                else:
                    if (
                        sym.is_global()
                        and sym.is_referenced()
                        and nm not in valid_globals
                    ):
                        scope_nm = tbl.get_name()
                        raise PolicyValidationError(
                            f"Undefined symbol '{nm}' in {filename}::{scope_nm}"
                        )
            for child in tbl.get_children():
                _check_table(child, is_root=False)

        _check_table(top_st, is_root=True)


# ---------------------------------------------------------------------------
# Cross-Module Contract Checker & Pyright `.pyi` Stubs
# ---------------------------------------------------------------------------

_CACHED_STUB_FILES: dict[str, str] | None = None


def _build_class_stub(cls: Any) -> list[str]:
    """Builds compact `.pyi` stub lines for a Python class via runtime inspection."""
    import inspect

    bases = [
        b.__name__
        for b in cls.__bases__
        if b is not object and b.__module__ == cls.__module__
    ]
    base_clause = f"({', '.join(bases)})" if bases else ""
    lines = [f"class {cls.__name__}{base_clause}:"]
    seen_members: set[str] = set()

    def _clean_sig_params(sig_obj: Any, fallback: str) -> str:
        raw = str(sig_obj)
        paren_depth = 0
        bracket_depth = 0
        cut_idx = -1
        for i, ch in enumerate(raw):
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(0, paren_depth - 1)
            elif ch in "[{":
                bracket_depth += 1
            elif ch in "]}":
                bracket_depth = max(0, bracket_depth - 1)
            elif paren_depth == 0 and bracket_depth == 0 and raw.startswith("->", i):
                cut_idx = i
                break
        if cut_idx >= 0:
            raw = raw[:cut_idx].rstrip()
        raw = re.sub(r"=<[^>]+>", "=...", raw)
        return raw if raw.startswith("(") else fallback

    if getattr(cls, "__init__", object.__init__) is not object.__init__:
        seen_members.add("__init__")
        try:
            init_sig = _clean_sig_params(
                inspect.signature(cls.__init__),
                "(self, *args: Any, **kwargs: Any)",
            )
        except Exception:
            init_sig = "(self, *args: Any, **kwargs: Any)"
        lines.append(f"    def __init__{init_sig} -> None: ...")
    annotations = getattr(cls, "__annotations__", {}) or {}
    for attr_name in annotations:
        if attr_name.startswith("__"):
            continue
        seen_members.add(attr_name)
        lines.append(f"    {attr_name}: Any")
    for attr_name, val in cls.__dict__.items():
        if attr_name.startswith("__"):
            continue
        if isinstance(val, property):
            seen_members.add(attr_name)
            lines.append(f"    @property\n    def {attr_name}(self) -> Any: ...")
        elif callable(val) and not isinstance(val, (staticmethod, classmethod)):
            seen_members.add(attr_name)
            try:
                sig = _clean_sig_params(
                    inspect.signature(val), "(self, *args: Any, **kwargs: Any)"
                )
            except Exception:
                sig = "(self, *args: Any, **kwargs: Any)"
            lines.append(f"    def {attr_name}{sig} -> Any: ...")
        elif isinstance(val, (staticmethod, classmethod)):
            seen_members.add(attr_name)
            fn = val.__func__
            dec = "@staticmethod" if isinstance(val, staticmethod) else "@classmethod"
            try:
                sig = _clean_sig_params(
                    inspect.signature(fn), "(*args: Any, **kwargs: Any)"
                )
            except Exception:
                sig = "(*args: Any, **kwargs: Any)"
            lines.append(f"    {dec}\n    def {attr_name}{sig} -> Any: ...")
        elif attr_name not in seen_members:
            seen_members.add(attr_name)
            lines.append(f"    {attr_name}: Any")
    if len(lines) == 1:
        lines.append("    pass")
    return lines


def _get_lox_stub_files() -> dict[str, str]:
    """Builds and caches minimal `.pyi` stubs matching live `lox.*` runtime APIs 1-to-1."""
    global _CACHED_STUB_FILES
    if _CACHED_STUB_FILES is not None:
        return _CACHED_STUB_FILES

    import inspect

    stub_map: dict[str, str] = {
        "lox/__init__.pyi": "",
        "lox/env/__init__.pyi": "",
        "lox/lang/__init__.pyi": "",
    }
    try:
        from lox.env import actions as act_mod
        from lox.env import state as st_mod

        # lox/env/state.pyi
        st_lines = [
            "from __future__ import annotations",
            "from typing import Any, Iterator",
            "import numpy as np",
            "",
        ]
        for cls_name in (
            "InventoryItem",
            "MonsterInfo",
            "ObjectInfo",
            "VisibleMonster",
            "VisibleObject",
            "FloorMemory",
            "WorldMemory",
            "GameState",
        ):
            cls_obj = getattr(st_mod, cls_name, None)
            if cls_obj is not None:
                st_lines.extend(_build_class_stub(cls_obj))
                st_lines.append("")
        st_lines.append("def bfs_distance_grid(*args: Any, **kwargs: Any) -> Any: ...")
        st_lines.append("def astar_next_step(*args: Any, **kwargs: Any) -> Any: ...")
        stub_map["lox/env/state.pyi"] = "\n".join(st_lines)

        # lox/env/actions.pyi
        act_lines = [
            "from __future__ import annotations",
            "from enum import IntEnum",
            "from typing import Any",
            "",
        ]
        action_cls = getattr(act_mod, "Action", None)
        if action_cls is not None:
            act_lines.append("class Action(IntEnum):")
            for m_name, m_val in action_cls.__members__.items():
                act_lines.append(f"    {m_name} = {int(m_val.value)}")
            act_lines.append("")
        for cls_name in ("RawAction", "ActionSequence"):
            cls_obj = getattr(act_mod, cls_name, None)
            if cls_obj is not None:
                act_lines.extend(_build_class_stub(cls_obj))
                act_lines.append("")
        for fn_name in (
            "key",
            "seq",
            "repeat",
            "extcmd",
            "menu_pick",
            "delta_to_dir",
            "dir_to_action",
        ):
            fn_obj = getattr(act_mod, fn_name, None)
            if callable(fn_obj):
                try:
                    sig = str(inspect.signature(fn_obj))
                except Exception:
                    sig = "(*args: Any, **kwargs: Any)"
                act_lines.append(f"def {fn_name}{sig} -> Any: ...")
        for const_name in (
            "DIR_DELTAS",
            "OPPOSITE_DIR",
            "CARDINAL_DELTAS",
            "ACTION_BY_NAME",
            "ACTION_BY_CHAR",
        ):
            if hasattr(act_mod, const_name):
                act_lines.append(f"{const_name}: Any")
        stub_map["lox/env/actions.pyi"] = "\n".join(act_lines)
    except Exception:
        pass

    stub_map["_lox_prelude.pyi"] = "\n".join(
        [
            "from __future__ import annotations",
            "from typing import Any as Any",
            "import math as math, collections as collections, heapq as heapq, itertools as itertools, functools as functools, re as re, enum as enum, dataclasses as dataclasses, typing as typing, random as random, operator as operator, bisect as bisect, copy as copy",
            "from collections import deque as deque, defaultdict as defaultdict, Counter as Counter",
            "from dataclasses import dataclass as dataclass",
            "import numpy as np",
            "import numpy as numpy",
            "from lox.env.actions import Action as Action, RawAction as RawAction, ActionSequence as ActionSequence, DIR_DELTAS as DIR_DELTAS, ACTION_BY_NAME as ACTION_BY_NAME, ACTION_BY_CHAR as ACTION_BY_CHAR, key as key, seq as seq, repeat as repeat, extcmd as extcmd",
            "from lox.env.state import GameState as GameState, WorldMemory as WorldMemory, FloorMemory as FloorMemory, InventoryItem as InventoryItem, MonsterInfo as MonsterInfo, ObjectInfo as ObjectInfo, VisibleMonster as VisibleMonster, VisibleObject as VisibleObject, bfs_distance_grid as bfs_distance_grid, astar_next_step as astar_next_step",
            "from lox.lang.contracts import GoalIntent as GoalIntent, SafetyReport as SafetyReport, WorldTracker as WorldTracker, CapabilityScorer as CapabilityScorer, ThreatEvaluator as ThreatEvaluator",
            "def param(name: str, default: Any, min_val: Any = ..., max_val: Any = ...) -> Any: ...",
        ]
    )

    try:
        import importlib

        mod_path = "lox.lang.contracts"
        mod_obj = importlib.import_module(mod_path)
        m_lines = [
            "from __future__ import annotations",
            "from typing import Any",
            "",
        ]
        for exp_name in dir(mod_obj):
            if exp_name.startswith("_"):
                continue
            exp_val = getattr(mod_obj, exp_name)
            if isinstance(exp_val, type) and exp_val.__module__ == mod_path:
                m_lines.extend(_build_class_stub(exp_val))
                m_lines.append("")
            elif callable(exp_val) and getattr(exp_val, "__module__", "") == mod_path:
                try:
                    sig = str(inspect.signature(exp_val))
                except Exception:
                    sig = "(*args: Any, **kwargs: Any)"
                m_lines.append(f"def {exp_name}{sig} -> Any: ...")
        stub_map["lox/lang/contracts.pyi"] = "\n".join(m_lines)
    except Exception:
        pass

    _CACHED_STUB_FILES = stub_map
    return stub_map


def validate_cross_module_contracts(
    policy_code: str, skills: dict[str, str] | None = None
) -> None:
    """Checks static cross-module `skills.*` imports, signatures, and `self.*` calls."""
    skills_map: dict[str, str] = {}
    for raw_k, raw_v in (skills or {}).items():
        nk = normalize_skill_key(raw_k)
        if nk:
            skills_map[nk] = raw_v
    known_keys = set(skills_map.keys())
    pkg_names = {k.split("/", 1)[0] for k in known_keys if "/" in k}
    parsed_skills: dict[str, ast.Module] = {}
    skill_fn_sigs: dict[
        str, dict[str, tuple[list[str], int, int, bool, bool, set[str], set[str]]]
    ] = {}

    for s_name, s_src in skills_map.items():
        if not str(s_src or "").strip():
            continue
        rel_fn = skill_key_to_relpath(s_name)
        try:
            s_tree = ast.parse(s_src, filename=rel_fn)
        except SyntaxError as exc:
            raise PolicyValidationError(
                f"SyntaxError in {rel_fn} line {exc.lineno}: {exc.msg}"
            ) from exc
        parsed_skills[s_name] = s_tree
        fn_map: dict[
            str, tuple[list[str], int, int, bool, bool, set[str], set[str]]
        ] = {}
        for node in s_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                pos_param_names = [a.arg for a in node.args.posonlyargs] + [
                    a.arg for a in node.args.args
                ]
                pos_args = len(pos_param_names)
                req_pos = max(0, pos_args - len(node.args.defaults))
                has_varargs = node.args.vararg is not None
                has_kwargs = node.args.kwarg is not None
                req_kwonly = {
                    a.arg
                    for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                    if d is None
                }
                kw_names = {a.arg for a in (node.args.args + node.args.kwonlyargs)}
                fn_map[node.name] = (
                    pos_param_names,
                    req_pos,
                    pos_args,
                    has_varargs,
                    has_kwargs,
                    kw_names,
                    req_kwonly,
                )
        skill_fn_sigs[s_name] = fn_map

    skill_exports_cache: dict[str, set[str]] = {}

    def _collect_skill_exports(
        mod_name: str, _visited: set[str] | None = None
    ) -> set[str]:
        if mod_name in skill_exports_cache:
            return skill_exports_cache[mod_name]
        visited = set(_visited or set())
        if mod_name in visited:
            return set()
        visited.add(mod_name)
        s_tree = parsed_skills.get(mod_name)
        if s_tree is None:
            return set()
        exports: set[str] = set()
        sig_map = skill_fn_sigs.setdefault(mod_name, {})
        for node in s_tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exports.add(node.name)
            elif isinstance(node, ast.Assign):
                for tgt in node.targets:
                    _add_assign_target_names(tgt, exports)
            elif isinstance(node, ast.AnnAssign):
                _add_assign_target_names(node.target, exports)
            elif isinstance(node, ast.ImportFrom) and node.module:
                src_mod = str(node.module)
                if src_mod.startswith("skills."):
                    dep_mod = resolve_skill_import(src_mod, known_keys)
                    dep_exports = _collect_skill_exports(dep_mod, visited)
                    dep_sigs = skill_fn_sigs.get(dep_mod, {})
                    for al in node.names:
                        if al.name == "*":
                            exports |= dep_exports
                            for k_fn, v_sig in dep_sigs.items():
                                sig_map.setdefault(k_fn, v_sig)
                        else:
                            bound = al.asname or al.name
                            if al.name in dep_exports:
                                exports.add(bound)
                            if al.name in dep_sigs:
                                sig_map.setdefault(bound, dep_sigs[al.name])
                else:
                    for al in node.names:
                        if al.name != "*":
                            exports.add(al.asname or al.name)
            elif isinstance(node, ast.Import):
                for al in node.names:
                    exports.add(al.asname or al.name.split(".")[0])
        skill_exports_cache[mod_name] = exports
        return exports

    for mod_k in list(parsed_skills.keys()):
        _collect_skill_exports(mod_k)

    try:
        p_tree = ast.parse(policy_code or "", filename="policy.py")
    except SyntaxError as exc:
        raise PolicyValidationError(
            f"SyntaxError in policy.py line {exc.lineno}: {exc.msg}"
        ) from exc

    modules_to_check: list[tuple[str, ast.Module]] = [("policy.py", p_tree)] + [
        (skill_key_to_relpath(k), v) for k, v in parsed_skills.items()
    ]

    from lox.lang.bundle import extract_skill_imports

    for fname, tree in modules_to_check:
        imp_info = extract_skill_imports(tree, known_keys, pkg_names=pkg_names)
        missing_mods = list(imp_info.missing_refs)
        for rk in sorted(imp_info.referenced_keys - known_keys):
            missing_mods.append(skill_key_to_module(rk))
        if missing_mods:
            raise PolicyValidationError(
                f"Cross-module error in {fname}: imports missing skill module '{missing_mods[0]}'."
            )
        for s_mod, sym_name in imp_info.direct_symbols.values():
            if s_mod in parsed_skills:
                mod_exports = _collect_skill_exports(s_mod)
                if sym_name not in mod_exports:
                    raise PolicyValidationError(
                        f"Cross-module error in {fname}: '{skill_key_to_module(s_mod)}' has no symbol '{sym_name}'."
                    )
        module_aliases: dict[str, str] = dict(imp_info.module_aliases)
        direct_imports: dict[str, tuple[str, str]] = dict(imp_info.direct_symbols)

        for node, shadowed in _walk_with_shadowed(tree):
            if not isinstance(node, ast.Call):
                continue
            target: tuple[str, str] | None = None
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in direct_imports
                and node.func.id not in shadowed
            ):
                target = direct_imports[node.func.id]
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in module_aliases
                and node.func.value.id not in shadowed
            ):
                target = (module_aliases[node.func.value.id], node.func.attr)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Attribute)
                and isinstance(node.func.value.value.value, ast.Name)
                and node.func.value.value.value.id == "skills"
                and "skills" not in shadowed
            ):
                s_mod = f"{node.func.value.value.attr}/{node.func.value.attr}"
                target = (s_mod, node.func.attr)
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Attribute)
                and isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "skills"
                and "skills" not in shadowed
            ):
                target = (node.func.value.attr, node.func.attr)
            if target is None:
                continue
            s_mod, fn_name = target
            sig = skill_fn_sigs.get(s_mod, {}).get(fn_name)
            if sig is None:
                continue
            (
                pos_param_names,
                req_pos,
                max_pos,
                has_varargs,
                has_kwargs,
                kw_names,
                req_kwonly,
            ) = sig
            if any(isinstance(a, ast.Starred) for a in node.args) or any(
                k.arg is None for k in node.keywords
            ):
                continue
            n_pos = len(node.args)
            kw_passed = {k.arg for k in node.keywords if k.arg is not None}
            if not has_varargs and n_pos > max_pos:
                raise PolicyValidationError(
                    f"Call contract error in {fname} (line {getattr(node, 'lineno', '?')}): "
                    f"skills.{s_mod}.{fn_name}() accepts at most {max_pos} positional args, got {n_pos}."
                )
            if not has_kwargs:
                unknown_kws = kw_passed - kw_names
                if unknown_kws:
                    raise PolicyValidationError(
                        f"Call contract error in {fname} (line {getattr(node, 'lineno', '?')}): "
                        f"skills.{s_mod}.{fn_name}() got unexpected keyword argument(s): {sorted(unknown_kws)}."
                    )
            req_pos_names = pos_param_names[:req_pos]
            missing_pos = [
                p
                for idx_p, p in enumerate(req_pos_names)
                if idx_p >= n_pos and p not in kw_passed
            ]
            missing_kwonly = sorted(req_kwonly - kw_passed)
            if missing_pos or missing_kwonly:
                raise PolicyValidationError(
                    f"Call contract error in {fname} (line {getattr(node, 'lineno', '?')}): "
                    f"skills.{s_mod}.{fn_name}() missing required argument(s): {missing_pos + missing_kwonly}."
                )


def _insert_prelude_after_future(src: str, prelude_header: str) -> tuple[str, int]:
    """
    Inserts `_lox_prelude` import after leading docstring and `from __future__` imports.
    Returns `(rewritten_src, inserted_at_1idx_line)` so diagnostic line numbers adjust accurately.
    """
    text = str(src or "")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return prelude_header + text, 1
    insert_after_line = 0
    for st in tree.body:
        if (
            isinstance(st, ast.Expr)
            and isinstance(st.value, ast.Constant)
            and isinstance(st.value.value, str)
            and insert_after_line == 0
        ):
            insert_after_line = int(getattr(st, "end_lineno", st.lineno) or st.lineno)
        elif isinstance(st, ast.ImportFrom) and st.module == "__future__":
            insert_after_line = int(getattr(st, "end_lineno", st.lineno) or st.lineno)
        else:
            break
    if insert_after_line <= 0:
        return prelude_header + text, 1
    lines = text.splitlines(keepends=True)
    head = "".join(lines[:insert_after_line])
    if head and not head.endswith("\n"):
        head += "\n"
    tail = "".join(lines[insert_after_line:])
    return head + prelude_header + tail, insert_after_line + 1


def run_pyright_on_package(
    policy_code: str,
    skills: dict[str, str] | None = None,
    timeout_sec: float = 8.0,
) -> list[str]:
    """
    Runs `pyright --outputjson` against `(policy.py, skills/*.py)` plus generated
    `lox/*.pyi` stubs in an isolated temp dir. Returns list of concise error lines.
    """
    import json
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    pyright_bin = shutil.which("pyright")
    if not pyright_bin:
        return []

    stubs = _get_lox_stub_files()
    skills_map = dict(skills or {})
    prelude_header = "from _lox_prelude import *  # pyright: ignore\n"
    prelude_lines = prelude_header.count("\n")
    prelude_offsets: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="lox_pyright_") as tmp_root:
        root = Path(tmp_root)
        pol_wrapped, pol_ins_line = _insert_prelude_after_future(
            policy_code or "", prelude_header
        )
        prelude_offsets["policy.py"] = pol_ins_line
        (root / "policy.py").write_text(pol_wrapped, encoding="utf-8")
        skills_dir = root / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        (skills_dir / "__init__.py").write_text("", encoding="utf-8")
        for s_name, s_src in skills_map.items():
            clean_k = normalize_skill_key(s_name)
            if clean_k and str(s_src or "").strip():
                dst = skills_dir / f"{clean_k}.py"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if "/" in clean_k:
                    pkg_init = dst.parent / "__init__.py"
                    if not pkg_init.exists():
                        pkg_init.write_text("", encoding="utf-8")
                s_wrapped, s_ins_line = _insert_prelude_after_future(
                    str(s_src), prelude_header
                )
                prelude_offsets[f"skills/{clean_k}.py"] = s_ins_line
                dst.write_text(s_wrapped, encoding="utf-8")

        typings_dir = root / "typings"
        for rel_path, stub_src in stubs.items():
            dst = typings_dir / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(stub_src, encoding="utf-8")

        cfg = {
            "include": ["policy.py", "skills"],
            "stubPath": "typings",
            "pythonVersion": "3.11",
            "typeCheckingMode": "basic",
            "reportMissingImports": "error",
            "reportMissingModuleSource": "none",
            "reportUndefinedVariable": "error",
            "reportAttributeAccessIssue": "error",
            "reportCallIssue": "error",
            "reportArgumentType": "warning",
            "reportGeneralTypeIssues": "warning",
            "reportOptionalMemberAccess": "none",
            "reportOptionalSubscript": "none",
            "reportOptionalCall": "none",
            "reportOptionalOperand": "none",
            "reportOptionalIterable": "none",
            "reportReturnType": "warning",
        }
        (root / "pyrightconfig.json").write_text(json.dumps(cfg), encoding="utf-8")
        try:
            proc = subprocess.run(
                [pyright_bin, "--outputjson"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            data = json.loads(proc.stdout or "{}")
            diags = data.get("generalDiagnostics") or []
            errors: list[str] = []
            for d in diags:
                if str(d.get("severity", "")).lower() != "error":
                    continue
                rule = str(d.get("rule", "") or "")
                msg = str(d.get("message", "") or "").strip()
                if 'Cannot access attribute "param" for class "Policy"' in msg:
                    continue
                fpath = str(d.get("file", "") or "").replace("\\", "/")
                if "/skills/" in fpath:
                    rel_f = f"skills/{fpath.split('/skills/', 1)[1]}"
                else:
                    rel_f = os.path.basename(fpath) if fpath else "policy.py"
                raw_line = int(
                    ((d.get("range") or {}).get("start") or {}).get("line", 0)
                )
                line_1idx = raw_line + 1
                ins_line = prelude_offsets.get(rel_f, 1)
                if line_1idx > ins_line:
                    line_1idx = max(1, line_1idx - prelude_lines)
                if msg:
                    errors.append(f"{rel_f}:{line_1idx} [{rule or 'pyright'}]: {msg}")
            return errors[:12]
        except Exception as exc:
            return [
                f"policy.py:0 [pyright]: typecheck failed ({type(exc).__name__}: {exc})"
            ]
