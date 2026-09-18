"""
CORP-Ω Policy Layer: revision ledger (R3, MACRO.md §6 step 13).

Append-only JSONL at data/revision_ledger.jsonl. Every author-LLM call appends exactly
one entry — accepted AND rejected (rejection is first-class: rejected diffs are training
signal for the next revision). Schema v1 was initialized in R0; this module is the writer
that the revision loop, the deliberative layer, and the ablation harness all share.
"""
from __future__ import annotations

import json
import os
import time

DEFAULT_LEDGER_PATH = "data/revision_ledger.jsonl"


class RevisionLedger:
    def __init__(self, path: str = DEFAULT_LEDGER_PATH):
        self.path = path

    def append(self, entry: dict) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        entry = {"ts": time.time(), **entry}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def all(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def tail(self, n: int = 5) -> list[dict]:
        return self.all()[-n:]

    # ------------------------------------------------------------------
    def revision_entry(
        self,
        *,
        revision: int,
        parent_version: int,
        author_model: str,
        provider: str,
        accepted: bool,
        reject_code: str | None = None,
        reject_detail: str = "",
        gate: str | None = None,
        reason: str = "",
        diff_text: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        tokens_thought: int = 0,
        latency_ms: float = 0.0,
        thinking: str = "",
        response_text: str = "",
        cost_usd_est: float = 0.0,
        wall_sec: float = 0.0,
        delta: dict | None = None,
    ) -> dict:
        return {
            "type": "policy_revision",
            "revision": revision,
            "parent_version": parent_version,
            "author_model": author_model,
            "provider": provider,
            "prompt_id": f"rev-{parent_version + 1}",
            "accepted": accepted,
            "reject_code": reject_code,
            "reject_detail": reject_detail,
            "gate": gate,
            "reason": reason,
            "diff": diff_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_thought": tokens_thought,
            "latency_ms": latency_ms,
            "thinking": thinking,
            "response_text": response_text,
            "cost_usd_est": cost_usd_est,
            "wall_sec": wall_sec,
            "delta": delta or {},
        }

    def acceptance_rate(self, author_model: str | None = None, last_n: int = 100) -> float:
        """Acceptance rate per author model — the R3 tiered-authorship ablation metric."""
        entries = [e for e in self.all()[-last_n:]
                   if e.get("type") == "policy_revision"
                   and (author_model is None or e.get("author_model") == author_model)]
        if not entries:
            return 0.0
        return sum(1 for e in entries if e.get("accepted")) / len(entries)