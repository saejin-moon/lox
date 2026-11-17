"""
Go-Explore Frontier Checkpoint Archive.
Stores verified milestone trajectories (Dlvl >= 3, altars, shops, XP >= 5) as raw action byte sequences.
Allows fast-forwarding (~15ms) a live NetHack harness directly into deep mid-game states
so candidate policies can be tested on mid-game tactical situations without grinding from Dlvl 1.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lox.env.harness import NetHackHarness

DEFAULT_FRONTIER_DIR = "data/frontier"


@dataclass
class FrontierCheckpoint:
    checkpoint_id: str
    seed: int
    depth: int
    milestone: str
    turns: int
    action_bytes: bytes
    game_log: str = ""
    created_at: float = 0.0
    visit_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["action_bytes_len"] = len(self.action_bytes)
        del d["action_bytes"]
        return d


class FrontierArchive:
    """Manages Go-Explore milestone frontier checkpoints for fast replay and mid-game evaluation."""

    def __init__(self, frontier_dir: str = DEFAULT_FRONTIER_DIR) -> None:
        self.frontier_dir = Path(frontier_dir)
        self.frontier_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: dict[str, FrontierCheckpoint] = {}
        self.load_checkpoints()

    def add_checkpoint(
        self,
        seed: int,
        depth: int,
        milestone: str,
        turns: int,
        action_bytes: bytes,
        game_log: str = "",
        checkpoint_id: str = "",
    ) -> FrontierCheckpoint:
        if not checkpoint_id:
            checkpoint_id = (
                f"fc_{milestone.lower()}_d{depth}_s{seed}_{int(time.time())}"
            )

        cp = FrontierCheckpoint(
            checkpoint_id=checkpoint_id,
            seed=seed,
            depth=depth,
            milestone=milestone,
            turns=turns,
            action_bytes=bytes(action_bytes),
            game_log=game_log,
            created_at=time.time(),
        )
        self.checkpoints[checkpoint_id] = cp
        self._persist_checkpoint(cp)
        return cp

    def _persist_checkpoint(self, cp: FrontierCheckpoint) -> None:
        try:
            meta_path = self.frontier_dir / f"{cp.checkpoint_id}.json"
            bytes_path = self.frontier_dir / f"{cp.checkpoint_id}.bytes"

            meta = cp.to_dict()
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            bytes_path.write_bytes(cp.action_bytes)
        except Exception:
            pass

    def load_checkpoints(self) -> int:
        loaded = 0
        try:
            for meta_path in self.frontier_dir.glob("*.json"):
                cid = meta_path.stem
                bytes_path = self.frontier_dir / f"{cid}.bytes"
                if not bytes_path.exists():
                    continue
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    act_bytes = bytes_path.read_bytes()
                    cp = FrontierCheckpoint(
                        checkpoint_id=cid,
                        seed=int(meta["seed"]),
                        depth=int(meta["depth"]),
                        milestone=str(meta["milestone"]),
                        turns=int(meta["turns"]),
                        action_bytes=act_bytes,
                        game_log=str(meta.get("game_log", "")),
                        created_at=float(meta.get("created_at", 0.0)),
                        visit_count=int(meta.get("visit_count", 0) or 0),
                    )
                    self.checkpoints[cid] = cp
                    loaded += 1
                except Exception:
                    continue
        except Exception:
            pass
        return loaded

    def sample_checkpoint(
        self,
        min_depth: int | None = None,
        milestone: str | None = None,
        prefer_deep: bool = True,
    ) -> FrontierCheckpoint | None:
        pool = list(self.checkpoints.values())
        if min_depth is not None:
            pool = [c for c in pool if c.depth >= min_depth]
        if milestone is not None:
            pool = [c for c in pool if c.milestone.lower() == milestone.lower()]
        if not pool:
            return None
        if prefer_deep and len(pool) > 1:
            max_d = max(c.depth for c in pool)
            deep_tier = [c for c in pool if c.depth >= max(min_depth or 1, max_d - 2)]
            if deep_tier:
                pool = deep_tier
        # Go-Explore inverse-visit exploration weighting: depth / sqrt(visit_count + 1)
        weights = [
            max(0.1, float(c.depth)) / math.sqrt(max(0, int(c.visit_count)) + 1.0)
            for c in pool
        ]
        chosen = random.choices(pool, weights=weights, k=1)[0]
        chosen.visit_count += 1
        self._persist_checkpoint(chosen)
        return chosen

    def replay_into(
        self,
        harness: NetHackHarness,
        checkpoint: FrontierCheckpoint,
    ) -> Any:
        """
        Fast-forwards harness into checkpoint state in ~10-20ms by replaying raw action bytes.
        If replaying all bytes reaches termination (e.g. a fatal run tail), automatically
        rewinds before termination so candidate evaluation starts in a live active game state.
        """
        buf = bytes(checkpoint.action_bytes or b"")
        ms = str(checkpoint.milestone or "").lower()
        if len(buf) > 30 and any(k in ms for k in ("died", "killed", "starv", "faint")):
            rewind = max(1, len(buf) - 30)
            state = harness.replay_bytes(
                seed=checkpoint.seed,
                recorded_bytes=buf,
                stop_at_byte_idx=rewind,
            )
            harness.step_count = 0
            return state
        state = harness.replay_bytes(seed=checkpoint.seed, recorded_bytes=buf)
        if harness.done and len(buf) > 1:
            rewind = max(1, len(buf) - 30)
            state = harness.replay_bytes(
                seed=checkpoint.seed,
                recorded_bytes=buf,
                stop_at_byte_idx=rewind,
            )
        harness.step_count = 0
        return state
