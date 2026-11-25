"""LOX Author Agent & Option-A Peer Swarm package."""

from lox.author.agent import AuthorAgent, SynthesisProposal
from lox.author.directives import (
    SwarmDirectivePlanner,
    extract_policy_branches,
    prune_unused_skills,
)
from lox.author.swarm_deficits import DeficitAnalyzer, DeficitCluster, DeficitReport
from lox.author.swarm_peer import PeerCandidate, SwarmPeerAgent
from lox.author.tools import OPENAI_TOOL_SPECS, AuthorToolRegistry, get_tool_specs

__all__ = [
    "AuthorAgent",
    "AuthorToolRegistry",
    "DeficitAnalyzer",
    "DeficitCluster",
    "DeficitReport",
    "OPENAI_TOOL_SPECS",
    "PeerCandidate",
    "SwarmDirectivePlanner",
    "SwarmPeerAgent",
    "SynthesisProposal",
    "extract_policy_branches",
    "get_tool_specs",
    "prune_unused_skills",
]
