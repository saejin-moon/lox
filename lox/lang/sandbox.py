"""
Python AST Sandbox Validator & Skill Key Utilities.
Supports full Python 3.11+ syntax while preventing OS, filesystem, subprocess,
module-reexport, and dynamic-execution sandbox escapes.
"""

from __future__ import annotations

import ast
import textwrap


class PolicyValidationError(Exception):
    """Raised when policy or skill code contains syntax errors or forbidden sandbox escapes."""


MODE_ALIASES: dict[str, str] = {
    "CREATE": "CREATE",
    "FULL_WRITE": "CREATE",
    "WRITE": "WRITE",
    "FUNC_MOD_ADD": "WRITE",
    "REFACTOR": "REFACTOR",
    "STRUCTURAL_REFACTOR": "REFACTOR",
    "STRUCTURAL_REWRITE": "REFACTOR",
    "SWARM_PEER": "SWARM_PEER",
}


def normalize_mode(mode: str | None) -> str:
    """Normalizes legacy or lowercase mode names to canonical (`CREATE`, `WRITE`, `REFACTOR`, `SWARM_PEER`)."""
    if not mode:
        return "WRITE"
    u = str(mode).strip().upper()
    return MODE_ALIASES.get(u, u)


ALLOWED_IMPORT_MODULES: frozenset[str] = frozenset(
    {
        "__future__",
        "math",
        "collections",
        "heapq",
        "itertools",
        "functools",
        "re",
        "enum",
        "dataclasses",
        "typing",
        "numpy",
        "np",
        "random",
        "operator",
        "bisect",
        "copy",
        "skills",
        "lox",
    }
)

# Explicit allowlist of safe `lox.*` submodules permitted inside policy/skill code.
ALLOWED_LOX_SUBMODULES: frozenset[str] = frozenset(
    {
        "lox.env.actions",
        "lox.env.state",
        "lox.lang.contracts",
    }
)

# Explicit allowlist of safe submodules under permitted stdlib/numpy root modules.
SAFE_STDLIB_SUBMODULES: dict[str, tuple[str, ...]] = {
    "collections": ("abc",),
    "numpy": ("linalg", "random", "fft", "ma", "exceptions"),
}

# Per-module attribute denylist blocking host filesystem I/O, pickle deserialization,
# dynamic eval(), and C-level getattr bypasses inside allowlisted modules.
FORBIDDEN_MODULE_ATTRS: dict[str, frozenset[str]] = {
    "operator": frozenset({"attrgetter", "methodcaller"}),
    "typing": frozenset({"get_type_hints", "ForwardRef", "eval_type_backport"}),
    "numpy": frozenset(
        {
            "load",
            "save",
            "savez",
            "savez_compressed",
            "fromfile",
            "fromregex",
            "memmap",
            "loadtxt",
            "savetxt",
            "genfromtxt",
            "DataSource",
            "lib",
            "ctypeslib",
        }
    ),
}
FORBIDDEN_MODULE_ATTRS["np"] = FORBIDDEN_MODULE_ATTRS["numpy"]


def _validate_stdlib_submodule(mod_name: str, root_mod: str) -> None:
    """Rejects nested or unauthorized submodules under allowlisted stdlib roots (e.g. `collections.sys`)."""
    if "." not in mod_name:
        return
    parts = mod_name.split(".")
    if len(parts) != 2 or parts[1] not in SAFE_STDLIB_SUBMODULES.get(root_mod, ()):
        raise PolicyValidationError(
            f"Import of submodule '{mod_name}' is forbidden in policy sandbox."
        )


# Dangerous stdlib/internal names that must never be imported even as re-exports from allowed modules
# (e.g., `from dataclasses import sys`, `from random import _os`).
FORBIDDEN_IMPORT_NAMES: frozenset[str] = frozenset(
    {
        "sys",
        "os",
        "builtins",
        "importlib",
        "subprocess",
        "pathlib",
        "shutil",
        "socket",
        "io",
        "ctypes",
        "pickle",
        "marshal",
        "inspect",
        "gc",
        "signal",
        "threading",
        "multiprocessing",
        "posix",
        "nt",
        "nh",
        "nle",
        "lox_core",
        "attrgetter",
        "methodcaller",
        "get_type_hints",
        "ForwardRef",
        "eval_type_backport",
    }
)

FORBIDDEN_CALLS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "attrgetter",
        "methodcaller",
        "get_type_hints",
        "eval_type_backport",
    }
)

FORBIDDEN_DUNDERS: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__closure__",
        "__loader__",
        "__spec__",
        "__import__",
        "__mro__",
        "__bases__",
        "__dict__",
    }
)

ALLOWED_DUNDERS: frozenset[str] = frozenset(
    {
        "__init__",
        "__post_init__",
        "__name__",
        "__bool__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__eq__",
        "__ne__",
        "__hash__",
        "__repr__",
        "__str__",
        "__len__",
        "__iter__",
        "__next__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "__missing__",
        "__contains__",
        "__call__",
        "__enter__",
        "__exit__",
    }
)

# Authoritative list of builtin names exposed to policy/skill code (shared with compiler.py).
SAFE_BUILTINS_NAMES: frozenset[str] = frozenset(
    {
        "abs",
        "all",
        "any",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "complex",
        "dict",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
        "Exception",
        "RuntimeError",
        "ValueError",
        "KeyError",
        "IndexError",
        "AttributeError",
        "TypeError",
        "ZeroDivisionError",
        "StopIteration",
        "AssertionError",
        "NotImplementedError",
        "LookupError",
        "True",
        "False",
        "None",
        "__import__",
        "__build_class__",
    }
)


_PRELOADED_MODULE_NAMES: frozenset[str] = frozenset(
    ALLOWED_IMPORT_MODULES - {"__future__", "skills"}
)

_STRICT_FORBIDDEN_ATTRS: frozenset[str] = frozenset(
    {
        "sys",
        "os",
        "builtins",
        "importlib",
        "subprocess",
        "pathlib",
        "shutil",
        "socket",
        "ctypes",
        "pickle",
        "marshal",
        "inspect",
        "gc",
        "signal",
        "threading",
        "multiprocessing",
        "posix",
        "nt",
        "nh",
        "nle",
        "lox_core",
        # Frame / generator / coroutine / traceback / code introspection (sandbox escape vectors)
        "gi_frame",
        "gi_code",
        "gi_yieldfrom",
        "cr_frame",
        "cr_code",
        "cr_await",
        "ag_frame",
        "ag_code",
        "ag_await",
        "tb_frame",
        "tb_next",
        "f_back",
        "f_globals",
        "f_builtins",
        "f_locals",
        "f_code",
        "co_consts",
        "co_code",
        "co_names",
        "func_globals",
        "func_code",
        "func_closure",
        # Arbitrary host file I/O & pickle serialization methods (e.g. ndarray.tofile/dump)
        "tofile",
        "dump",
        "dumps",
        # C-level getattr / eval helpers
        "attrgetter",
        "methodcaller",
        "get_type_hints",
        "ForwardRef",
        "eval_type_backport",
    }
)


class SandboxASTVisitor(ast.NodeVisitor):
    """Validates an AST for sandbox safety while allowing rich Python constructs."""

    def __init__(self) -> None:
        super().__init__()
        self._imported_modules: set[str] = set(_PRELOADED_MODULE_NAMES)
        self._module_roots: dict[str, str] = {m: m for m in _PRELOADED_MODULE_NAMES}

    def _track_module_alias(self, target: ast.expr, value: ast.expr | None) -> None:
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Name)
            and value.id in self._imported_modules
        ):
            self._imported_modules.add(target.id)
            self._module_roots[target.id] = self._module_roots.get(value.id, value.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        for tgt in node.targets:
            self._track_module_alias(tgt, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._track_module_alias(node.target, node.value)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            mod_name = str(alias.name or "")
            root_mod = mod_name.split(".")[0]
            if root_mod == "__future__" or root_mod not in ALLOWED_IMPORT_MODULES:
                raise PolicyValidationError(
                    f"Import of module '{mod_name}' is forbidden in policy sandbox. "
                    f"Allowed modules: {sorted(ALLOWED_IMPORT_MODULES)}"
                )
            if root_mod == "lox" and mod_name not in ALLOWED_LOX_SUBMODULES:
                raise PolicyValidationError(
                    f"Import of internal module '{mod_name}' is forbidden in policy sandbox. "
                    f"Allowed lox modules: {sorted(ALLOWED_LOX_SUBMODULES)}"
                )
            elif root_mod != "skills":
                _validate_stdlib_submodule(mod_name, root_mod)
            bound_as = alias.asname or root_mod
            if root_mod != "skills":
                self._imported_modules.add(bound_as)
                self._module_roots[bound_as] = root_mod
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        full_mod = str(node.module or "")
        root_mod = full_mod.split(".")[0]
        if root_mod not in ALLOWED_IMPORT_MODULES:
            raise PolicyValidationError(
                f"Import from '{node.module}' is forbidden in policy sandbox. "
                f"Allowed modules: {sorted(ALLOWED_IMPORT_MODULES)}"
            )
        if full_mod == "__future__":
            for alias in node.names:
                if alias.name != "annotations":
                    raise PolicyValidationError(
                        f"Only `from __future__ import annotations` is allowed (got '{alias.name}')."
                    )
            return
        if root_mod == "lox":
            if full_mod not in ALLOWED_LOX_SUBMODULES:
                raise PolicyValidationError(
                    f"Import from internal module '{full_mod}' is forbidden in policy sandbox. "
                    f"Allowed lox modules: {sorted(ALLOWED_LOX_SUBMODULES)}"
                )
            for alias in node.names:
                nm = str(alias.name or "")
                if nm.startswith("_") or nm in FORBIDDEN_IMPORT_NAMES:
                    raise PolicyValidationError(
                        f"Import of private/internal symbol '{nm}' from '{full_mod}' is forbidden in policy sandbox."
                    )
        elif root_mod != "skills":
            _validate_stdlib_submodule(full_mod, root_mod)
            mod_denied = FORBIDDEN_MODULE_ATTRS.get(root_mod, frozenset())
            for alias in node.names:
                nm = str(alias.name or "")
                if (
                    nm.startswith("_")
                    or nm in FORBIDDEN_IMPORT_NAMES
                    or nm in mod_denied
                ):
                    raise PolicyValidationError(
                        f"Import of private/re-exported symbol '{nm}' from '{full_mod}' is forbidden in policy sandbox."
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise PolicyValidationError(
                    f"Call to builtin '{node.func.id}()' is forbidden in policy sandbox."
                )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_DUNDERS or (
            node.attr.startswith("__") and node.attr not in ALLOWED_DUNDERS
        ):
            raise PolicyValidationError(
                f"Access to dunder attribute '{node.attr}' is forbidden in policy sandbox."
            )
        if node.attr in _STRICT_FORBIDDEN_ATTRS:
            raise PolicyValidationError(
                f"Access to internal/system attribute '{node.attr}' is forbidden in policy sandbox."
            )
        if isinstance(node.value, ast.Name) and node.value.id in self._imported_modules:
            root_mod = self._module_roots.get(node.value.id, node.value.id)
            mod_denied = FORBIDDEN_MODULE_ATTRS.get(root_mod, frozenset())
            if (
                node.attr.startswith("_")
                or node.attr in FORBIDDEN_IMPORT_NAMES
                or node.attr in mod_denied
            ):
                raise PolicyValidationError(
                    f"Access to private/forbidden module attribute '{node.value.id}.{node.attr}' is forbidden in policy sandbox."
                )
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "_lox_tracer"
            and isinstance(node.ctx, (ast.Store, ast.Del))
        ):
            raise PolicyValidationError(
                "Mutating internal '_lox_tracer' is forbidden in policy sandbox."
            )
        self.generic_visit(node)


def expand_leading_tabs(code: str) -> str:
    """Expands tabs only in leading line indentation, preserving tab characters inside string literals."""
    if "\t" not in (code or ""):
        return code or ""
    out_lines: list[str] = []
    for line in (code or "").splitlines(keepends=True):
        i = 0
        n = len(line)
        while i < n and line[i] in (" ", "\t"):
            i += 1
        if i > 0 and "\t" in line[:i]:
            out_lines.append(line[:i].replace("\t", "    ") + line[i:])
        else:
            out_lines.append(line)
    return "".join(out_lines)


def format_code_with_line_numbers(
    code: str,
    highlight_line: int | None = None,
    col_offset: int | None = None,
    window: int = 4,
) -> str:
    """
    Formats code lines with 1-based line numbers, highlighting the specified line and column with a caret.
    """
    lines = code.splitlines()
    if not lines:
        return "(empty)"
    total_lines = len(lines)

    if highlight_line is not None and 1 <= highlight_line <= total_lines:
        start = max(1, highlight_line - window)
        end = min(total_lines, highlight_line + window)
    else:
        start = 1
        end = min(total_lines, window * 2 + 1)

    num_digits = max(2, len(str(end)))
    out: list[str] = []
    for ln in range(start, end + 1):
        line_content = lines[ln - 1]
        is_hl = ln == highlight_line
        marker = "> " if is_hl else "  "
        out.append(f"{marker}{str(ln).rjust(num_digits)} | {line_content}")
        if is_hl and col_offset is not None and col_offset > 0:
            out.append(" " * (2 + num_digits + 3) + " " * (col_offset - 1) + "^")
    return "\n".join(out)


def validate_python_code(code: str, filename: str = "<policy>") -> ast.Module:
    """Parses and validates Python source code, returning the parsed AST Module."""
    if not code or not code.strip():
        raise PolicyValidationError(f"Empty code submitted for {filename}.")
    cleaned = textwrap.dedent(expand_leading_tabs(code)).strip()
    try:
        tree = ast.parse(cleaned, filename=filename)
    except SyntaxError as exc:
        snippet = format_code_with_line_numbers(
            cleaned,
            highlight_line=exc.lineno,
            col_offset=exc.offset,
        )
        raise PolicyValidationError(
            f"SyntaxError in {filename} at line {exc.lineno}: {exc.msg}\n\n{snippet}"
        ) from exc

    visitor = SandboxASTVisitor()
    visitor.visit(tree)
    return tree


def is_ast_equivalent(code1: str, code2: str) -> bool:
    """Returns True if code1 and code2 parse to structurally equivalent ASTs (ignoring comments/whitespace)."""
    if not code1 or not code2:
        return False
    if code1.strip() == code2.strip():
        return True
    try:
        t1 = ast.dump(ast.parse(textwrap.dedent(expand_leading_tabs(code1)).strip()))
        t2 = ast.dump(ast.parse(textwrap.dedent(expand_leading_tabs(code2)).strip()))
        return t1 == t2
    except Exception:
        return False


def smart_dedent(code: str) -> str:
    """Smart dedents code even when comments or docstrings appear at column 0."""
    norm = expand_leading_tabs(code or "")
    lines = norm.splitlines()
    code_lines = [
        line_str
        for line_str in lines
        if line_str.strip() and not line_str.strip().startswith("#")
    ]
    if not code_lines:
        return norm
    common_indent = min(
        len(line_str) - len(line_str.lstrip()) for line_str in code_lines
    )
    if common_indent > 0:
        new_lines = []
        for line_str in lines:
            if not line_str.strip():
                new_lines.append("")
            elif (
                line_str.strip().startswith("#")
                and (len(line_str) - len(line_str.lstrip())) < common_indent
            ):
                new_lines.append(line_str.strip())
            else:
                new_lines.append(
                    line_str[common_indent:]
                    if len(line_str) >= common_indent
                    else line_str.lstrip()
                )
        return "\n".join(new_lines)
    return norm


def normalize_skill_key(raw: str) -> str:
    """Normalizes `skills/a/b.py`, `skills.a.b`, `a.b`, or `a/b` into `'a/b'` (max depth 2)."""
    s = str(raw or "").strip().replace("\\", "/")
    if s.startswith("skills/"):
        s = s[len("skills/") :]
    elif s.startswith("skills."):
        s = s[len("skills.") :]
    if s.endswith(".py"):
        s = s[:-3]
    s = s.replace(".", "/").strip("/")
    parts = [p for p in s.split("/") if p]
    if len(parts) > 2:
        raise PolicyValidationError(
            f"Skill package depth > 2 is not allowed (`{raw}`): use `skills/<domain>/<module>.py`."
        )
    return "/".join(parts)


def skill_key_to_module(key: str) -> str:
    """Converts normalized key `'a/b'` or `'a'` to `'skills.a.b'` or `'skills.a'`."""
    clean = normalize_skill_key(key)
    return f"skills.{clean.replace('/', '.')}" if clean else "skills"


def skill_key_to_relpath(key: str) -> str:
    """Converts normalized key `'a/b'` or `'a'` to `'skills/a/b.py'` or `'skills/a.py'`."""
    clean = normalize_skill_key(key)
    return f"skills/{clean}.py" if clean else "skills/__init__.py"


def resolve_skill_import(mod_str: str, known_keys: set[str]) -> str:
    """Resolves `'skills.domain.sub'` or `'skills.domain'` against normalized `known_keys`."""
    tail = mod_str[len("skills.") :] if mod_str.startswith("skills.") else mod_str
    parts = [p for p in tail.replace("/", ".").split(".") if p]
    if len(parts) >= 2:
        two = f"{parts[0]}/{parts[1]}"
        if two in known_keys or not known_keys:
            return two
    if parts:
        one = parts[0]
        if one in known_keys:
            return one
        return "/".join(parts[:2])
    return ""
