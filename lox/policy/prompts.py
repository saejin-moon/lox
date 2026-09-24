"""
LOX-ψ Policy Layer (LOX-ψ): the prompt program (S0, AGENT_PLAN §1.6).

The author system prompt is a first-class, versioned artifact stored under
data/prompts/<version>.md and evaluated under the same gated discipline as the
policy program itself (run_prompt_ab.py: acceptance rate, reject taxonomy,
downstream batch). The loader never hard-fails: a missing version falls back to
the bundled default so the loop keeps operating on fresh installs.

Prompt files may carry a `{tool_docs}` placeholder (replaced with the generated
tool reference); otherwise the tool reference is appended as a trailing section.
"""
from __future__ import annotations

import glob
import os
import re

PROMPTS_DIR = "data/prompts"
VERSION_RE = re.compile(r"^v\d+$")


def list_prompt_versions(prompts_dir: str = PROMPTS_DIR) -> list[str]:
    """All vN prompt versions on disk, in numeric order."""
    if not os.path.isdir(prompts_dir):
        return []
    out = []
    for p in glob.glob(os.path.join(prompts_dir, "v*.md")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if VERSION_RE.match(stem):
            out.append(stem)
    def _key(v: str):
        return int(v[1:])
    return sorted(out, key=_key)


def load_prompt_text(name_or_path: str | None = None,
                     prompts_dir: str = PROMPTS_DIR) -> str:
    """Loads a prompt version by name ('v2'), file stem ('v2.md'), or path.
    Unknown/None falls back to the newest on-disk version, then the bundled
    default — the session must never hard-fail on a prompt lookup."""
    if name_or_path:
        if os.path.isfile(name_or_path):
            with open(name_or_path, "r", encoding="utf-8") as f:
                return f.read()
        for candidate in (name_or_path, f"{name_or_path}.md"):
            p = os.path.join(prompts_dir, candidate)
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    return f.read()
    versions = list_prompt_versions(prompts_dir)
    if versions:
        return load_prompt_text(versions[-1], prompts_dir)
    return BUNDLED_DEFAULT_PROMPT


def render_system_prompt(prompt_text: str, tool_docs: str) -> str:
    """Injects the generated tool reference into a prompt version."""
    if "{tool_docs}" in prompt_text:
        return prompt_text.replace("{tool_docs}", tool_docs)
    return prompt_text.rstrip() + "\n\n# AVAILABLE TOOLS (call with (tool <name> <args>...))\n" + tool_docs


# The v1-equivalent baseline (matches lox.policy.reviser.SYSTEM_PROMPT content;
# used when data/prompts/ has no versions yet — fresh installs, test sandboxes).
BUNDLED_DEFAULT_PROMPT = """You are the strategy-tuning author of a symbolic NetHack agent (LOX-ψ).
Your ONLY output is ONE policy diff — an S-expression document whose first form is
(revision ...). You may ONLY use the vocabulary manifest provided in the user message:
predicates, verbs, certified goals, and policy_params.* leaves with their bounds. Every
change needs a reason grounded in the run report (deaths, stalls, goal stats). You cannot
write code, invent predicates, or exceed declared bounds. Emit the diff and nothing else.

Diff forms you may use (one per line, omit any part you do not need — never write
square brackets, they are only used below to mark optional parts):
  (set policy_params.<dotted.leaf> <value>)
  (rule add tactic_rules (when <expr>) (do <verb>) (unless <expr>)? (note "<str>")?)
  (rule remove tactic_rules <index>)
  (goal prioritize <goal> (when <expr>)? (until <expr>)?)
  (goal deprioritize <goal> (after <expr>)?)
  (goal set_threshold <goal> <owned-param-path> <value>)
  (nogood add (when <expr>) (cause "<str>") (forbid <task>)?)
  (defmacro <name> <expr>)   ; parameterless sugar over manifest predicates — no recursion
  (note "<str>")

Rules:
- header: (revision <parent+1> (parent <parent>) (author "<model>") (domain <domain>) (reason "<grounded rationale>"))
- conditions can be written in natural Pythonic infix or parenthesized calls:
  (when hp_frac <= 0.40 and not is_fighting)
  (when monster == "j" and hp_frac <= 0.35)
  (when stairs_known and not adjacent_hostiles)
  Combinators: and, or, not, comparisons (<=, >=, <, >, ==, !=).
- condition depth <= 3.
- keep diffs surgical: tune params, add/retire tactic rules, reprioritize goals.
- combat tactic rules MUST be target-conditional: the (when ...) of a combat-verb rule
  must contain a (monster "...") or (item "...") match. Global catch-alls such as
  (when (adjacent_hostiles)) are REJECTED by the validator — they mis-fire across the
  whole early game (measured: a catch-all cost 45% mean score).
- do NOT restate unchanged parts of the program; do NOT remove certified goals.
- emit the diff forms flat, one per line, no leading indentation needed.
"""
