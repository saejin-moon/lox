"""
CORP-Ω Executor Boundary: domain registry + lazy adapter loading (R5).
"""
from corp.executor.interface import (  # noqa: F401
    DomainAdapter, DomainSpec, PredicateBinding, ParamLeaf, GoalHandler, CertCase,
    register, get_adapter, available_domains,
)