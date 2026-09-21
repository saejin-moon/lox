"""
LOX-ψ Policy Layer (CORP): role/domain profile overlays (R4, AGENT_PLAN §8).

Overlay precedence (normative):
    PolicyConfig defaults ← domain_profile ← role_profile ← program.params (global)

Profiles live in the policy program (`role_profiles` / `domain_profiles` dicts mapping
role/domain name → {dotted-param-path: value}). They are human-curated overlays — NOT
LLM-authored diffs — so the LLM's `set` ops always win (params apply last).

Roles may tune: weapon-tier preferences, combat aggression fracs, food radii, search
caps, shop behavior. Roles may NOT touch (protected, enforced here): the prayer cooldown
model, corpse freshness, and the interlock branches (which live in code, not params).
"""
from __future__ import annotations

from corp.policy.config import PolicyConfig

# Dotted paths a profile overlay may never set (safety invariants — AGENTS.md §3)
PROTECTED_PROFILE_PATHS = {
    "nutrition.pray_major_first", "nutrition.pray_major",
    "nutrition.pray_routine_first", "nutrition.pray_routine",
    "nutrition.corpse_fresh_turns",
    "nutrition.emergency_hunger",
    "strategy.emergency_hunger",
}


class ProfileError(ValueError):
    pass


def apply_profile_overlay(config: PolicyConfig, overlay: dict, source: str) -> list[str]:
    """Applies one {dotted-path: value} overlay onto a PolicyConfig IN PLACE.
    Returns the applied paths. Raises ProfileError on protected/unknown paths."""
    applied = []
    for dotted, value in (overlay or {}).items():
        if dotted in PROTECTED_PROFILE_PATHS:
            raise ProfileError(f"{source}: path {dotted!r} is protected (safety invariant)")
        parts = dotted.split(".")
        obj = config
        for p in parts[:-1]:
            if not hasattr(obj, p):
                raise ProfileError(f"{source}: unknown param section {dotted!r}")
            obj = getattr(obj, p)
        if not hasattr(type(obj), parts[-1]) and not hasattr(obj, parts[-1]):
            raise ProfileError(f"{source}: unknown param {dotted!r}")
        cur = getattr(obj, parts[-1])
        try:
            if isinstance(cur, bool):
                setattr(obj, parts[-1], bool(value))
            elif isinstance(cur, int) and not isinstance(cur, bool):
                setattr(obj, parts[-1], int(round(float(value))))
            elif isinstance(cur, float):
                setattr(obj, parts[-1], float(value))
            else:
                setattr(obj, parts[-1], value)
        except (TypeError, ValueError) as e:
            raise ProfileError(f"{source}: type mismatch for {dotted!r}: {e}") from e
        applied.append(dotted)
    return applied


def resolve_config(program, role: str = "", domain: str = "") -> PolicyConfig:
    """Builds a PolicyConfig for (program, role, domain) with the normative precedence:
    defaults ← domain_profile ← role_profile ← program.params."""
    cfg = PolicyConfig.defaults()
    domain_profiles = getattr(program, "domain_profiles", {}) or {}
    role_profiles = getattr(program, "role_profiles", {}) or {}
    if domain and domain in domain_profiles:
        apply_profile_overlay(cfg, domain_profiles[domain], f"domain_profile[{domain}]")
    role_key = (role or "").lower()
    if role_key and role_key in role_profiles:
        apply_profile_overlay(cfg, role_profiles[role_key], f"role_profile[{role_key}]")
    if program.params:
        program.apply_overlay(cfg)
    return cfg