"""
Flight Recorder: 100-turn in-game ring buffer for post-mortem autopsy and regression analysis.
"""

from collections import deque
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from corp.env.blstats import BottomLineStats
from corp.env.anomaly_sentry import AnomalyEvent


@dataclass(slots=True, frozen=True)
class FlightRecord:
    turn: int
    action: int | str
    blstats: BottomLineStats
    message: str
    tty_screen: str
    anomalies: tuple[AnomalyEvent, ...]


class FlightRecorderRingBuffer:
    """
    Fixed-capacity circular buffer that captures the last N turns of gameplay
    for post-mortem causal analysis when player dies.
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: deque[FlightRecord] = deque(maxlen=capacity)

    def clear(self):
        self.buffer.clear()

    def record(
        self,
        turn: int,
        action: int | str,
        blstats: BottomLineStats,
        message: str,
        tty_chars: npt.NDArray[np.uint8] | None = None,
        anomalies: list[AnomalyEvent] | None = None,
    ):
        """Records a single physical step into the ring buffer."""
        screen_text = ""
        if tty_chars is not None:
            # Decode 24x80 ASCII screen
            lines = [
                bytes(tty_chars[r]).decode("latin-1", errors="replace").rstrip()
                for r in range(min(24, tty_chars.shape[0]))
            ]
            screen_text = "\n".join(lines)

        rec = FlightRecord(
            turn=turn,
            action=action,
            blstats=blstats,
            message=message,
            tty_screen=screen_text,
            anomalies=tuple(anomalies or []),
        )
        self.buffer.append(rec)

    def get_recent_history(self, count: int = 20) -> list[FlightRecord]:
        """Returns the most recent N records in chronological order."""
        history = list(self.buffer)
        return history[-count:]

    def export_autopsy_summary(self, recent_turns: int = 15) -> str:
        """
        Formats a structured markdown autopsy report summarizing the trajectory
        leading up to termination.
        """
        if not self.buffer:
            return "No flight records recorded."

        records = self.get_recent_history(recent_turns)
        last_rec = records[-1]
        bs = last_rec.blstats

        lines = [
            f"# Flight Recorder Autopsy: Game Ended at Turn {bs.turn}",
            f"- **Final HP**: {bs.hp}/{bs.max_hp} | **AC**: {bs.ac} | **Dungeon Depth**: {bs.depth}",
            f"- **Position**: ({bs.x}, {bs.y}) | **Score**: {bs.score} | **Gold**: {bs.gold}",
            f"- **Final Message**: `{last_rec.message}`",
            "",
            "## Trajectory Preceding Death (Last Turns):",
            "| Turn | HP | Action | Message | Anomalies Detected |",
            "| :--- | :--- | :--- | :--- | :--- |",
        ]

        for r in records:
            anom_str = ", ".join(a.condition_name for a in r.anomalies) if r.anomalies else "-"
            sanitized_msg = r.message.replace("|", "/")
            lines.append(
                f"| {r.turn} | {r.blstats.hp}/{r.blstats.max_hp} | `{r.action}` | {sanitized_msg} | {anom_str} |"
            )

        lines.extend([
            "",
            "## Final Terminal Screen:",
            "```text",
            last_rec.tty_screen,
            "```",
        ])

        return "\n".join(lines)
