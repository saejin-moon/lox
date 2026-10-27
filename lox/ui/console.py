"""
Clean, non-bloated terminal UI for LOX.
Provides single-line in-place progress bars, compact 1-line tool logs,
and scannable generation/hypothesis summary tables without spamming stdout.
"""

from __future__ import annotations

import sys
from typing import Any


class ConsoleUI:
    """Compact ANSI terminal logger designed for high signal-to-noise ratio."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"

    def __init__(self, enabled: bool = True, use_color: bool | None = None):
        self.enabled = enabled
        self.use_color = sys.stdout.isatty() if use_color is None else use_color
        self._in_progress_line = False

    def _c(self, color: str, text: str) -> str:
        if not self.use_color:
            return text
        return f"{color}{text}{self.RESET}"

    def _clear_progress(self) -> None:
        if self._in_progress_line and self.enabled:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            self._in_progress_line = False

    def banner(
        self,
        model: str,
        provider: str,
        db_path: str,
        workers: int,
        role: str,
    ) -> None:
        if not self.enabled:
            return
        self._clear_progress()
        line = "─" * 74
        print(self._c(self.CYAN, f"┌{line}┐"))
        title = "LOX  │  Autonomous Policy Synthesis Engine ('Hard Code Nothing')"
        print(
            self._c(self.CYAN, "│ ")
            + self._c(self.BOLD, f"{title:<72}")
            + self._c(self.CYAN, " │")
        )
        meta = f"Model: {model} ({provider})  │  Role: {role}  │  Workers: {workers}  │  DB: {db_path}"
        print(self._c(self.CYAN, "│ ") + f"{meta[:72]:<72}" + self._c(self.CYAN, " │"))
        print(self._c(self.CYAN, f"└{line}┘"))

    def gen_header(
        self,
        gen: int,
        mode: str,
        target_episodes: int,
        target_fns: list[str] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._clear_progress()
        fn_str = f" │ Targets: {', '.join(target_fns)}" if target_fns else ""
        eps_str = (
            f"{target_episodes:,} eps"
            if target_episodes > 0
            else "0 eps (instant commit)"
        )
        hdr = f"═══ Gen {gen:03d} │ Mode: {mode} │ Budget: {eps_str}{fn_str} "
        pad = max(0, 76 - len(hdr))
        print("\n" + self._c(self.BOLD + self.BLUE, hdr + "═" * pad))

    def tool_call(
        self,
        turn_idx: int,
        tool_name: str,
        arg_summary: str,
        result_summary: str,
        elapsed_ms: float,
    ) -> None:
        if not self.enabled:
            return
        self._clear_progress()
        arg_clean = " ".join(arg_summary.strip().split())
        if len(arg_clean) > 42:
            arg_clean = arg_clean[:39] + "..."
        res_clean = " ".join(result_summary.strip().split())
        if len(res_clean) > 46:
            res_clean = res_clean[:43] + "..."
        print(
            f"  {self._c(self.DIM, f'[T{turn_idx}]')} "
            f"{self._c(self.CYAN, f'{tool_name:<18}')} "
            f"{self._c(self.DIM, f'({arg_clean})')} -> {res_clean} "
            f"{self._c(self.DIM, f'[{elapsed_ms:.0f}ms]')}"
        )

    def hypothesis_box(self, hypothesis: dict[str, Any]) -> None:
        if not self.enabled or not hypothesis:
            return
        self._clear_progress()
        targets = (
            hypothesis.get("target_functions")
            or hypothesis.get("targeted_skill")
            or "policy"
        )
        if isinstance(targets, list):
            targets = ", ".join(str(t) for t in targets)
        skill_name = hypothesis.get("skill_name")
        persistence = hypothesis.get("state_persistence")
        obs = str(
            hypothesis.get("observation_from_db")
            or hypothesis.get("causal_finding")
            or "N/A"
        )
        mech = str(hypothesis.get("mechanism") or "N/A")
        if len(obs) > 90:
            obs = obs[:87] + "..."
        if len(mech) > 90:
            mech = mech[:87] + "..."
        extra_lines = ""
        if skill_name:
            extra_lines += (
                f"\n             │ {self._c(self.BOLD, 'Skill:  ')} {skill_name}"
            )
        if persistence:
            pers_str = str(persistence)[:80]
            extra_lines += (
                f"\n             │ {self._c(self.DIM, 'State:  ')} {pers_str}"
            )

        print(
            f"  {self._c(self.MAGENTA + self.BOLD, 'Hypothesis')} │ "
            f"{self._c(self.BOLD, 'Target:')} {targets}{extra_lines}\n"
            f"             │ {self._c(self.DIM, 'Finding:')} {obs}\n"
            f"             │ {self._c(self.DIM, 'Action: ')} {mech}"
        )

    def progress(
        self,
        triggered_done: int,
        triggered_target: int,
        raw_done: int,
        steps_per_sec: float,
        avg_depth: float,
        max_depth: int,
        outliers: int = 0,
    ) -> None:
        if not self.enabled:
            return
        pct = (
            min(100.0, 100.0 * triggered_done / max(1, triggered_target))
            if triggered_target > 0
            else 100.0
        )
        bar_len = 16
        filled = int(round(bar_len * pct / 100.0))
        bar = "█" * filled + "░" * (bar_len - filled)
        msg = (
            f"\r  {self._c(self.CYAN, f'[{bar}]')} {pct:5.1f}% │ "
            f"Trig: {triggered_done:,}/{triggered_target:,} "
            f"{self._c(self.DIM, f'(raw {raw_done:,})')} │ "
            f"{steps_per_sec:,.0f} sps │ "
            f"Depth: {avg_depth:.2f} (max {max_depth}) │ "
            f"Outliers: {outliers}"
        )
        sys.stdout.write(msg + "\033[K")
        sys.stdout.flush()
        self._in_progress_line = True

    def batch_summary(
        self,
        batch_stats: dict[str, Any],
        verdict: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._clear_progress()
        eps = batch_stats.get("episodes_recorded", 0)
        disc = batch_stats.get("episodes_discarded", 0)
        avg_d = batch_stats.get("avg_depth", 0.0)
        p90_d = batch_stats.get("p90_depth", 0.0)
        max_d = batch_stats.get("max_depth", 0)
        avg_s = batch_stats.get("avg_score", 0.0)
        avg_t = batch_stats.get("avg_turns", 0.0)
        outliers = batch_stats.get("outlier_count", 0)
        sps = batch_stats.get("steps_per_sec", 0.0)
        wall_s = batch_stats.get("wall_sec", 0.0)

        print(
            f"  {self._c(self.BOLD, 'Rollout')}    │ "
            f"Recorded: {eps:,} {self._c(self.DIM, f'(discarded {disc:,})')} │ "
            f"Depth: {self._c(self.BOLD, f'{avg_d:.2f}')} "
            f"(P90 {p90_d:.1f}, Max {max_d}) │ "
            f"Score: {avg_s:.1f} │ Turns: {avg_t:.0f} │ "
            f"Outliers: {self._c(self.YELLOW, str(outliers))} │ "
            f"{sps:,.0f} sps ({wall_s:.1f}s)"
        )

        if verdict is not None:
            passed = verdict.get("passed", False)
            status_str = (
                self._c(self.GREEN + self.BOLD, "PROMOTED ✓")
                if passed
                else self._c(self.RED + self.BOLD, "REJECTED ✗")
            )
            mean_d = verdict.get("mean_delta", 0.0)
            ci_l = verdict.get("ci_lower", 0.0)
            ci_u = verdict.get("ci_upper", 0.0)
            p_val = verdict.get("wilcoxon_p", 1.0)
            wins = verdict.get("wins", 0)
            losses = verdict.get("losses", 0)
            ties = verdict.get("ties", 0)
            reason = verdict.get("reason", "")
            print(
                f"  {self._c(self.BOLD, 'Verdict')}    │ {status_str} │ "
                f"Δ: {mean_d:+.3f} (95% CS [{ci_l:+.3f}, {ci_u:+.3f}]) │ "
                f"W/L/T: {wins}/{losses}/{ties} (p={p_val:.4f}) │ {self._c(self.DIM, reason)}"
            )

    def status(self, msg: str) -> None:
        """Prints a temporary status line in place (cleared on next log)."""
        if not self.enabled:
            return
        if self._in_progress_line:
            sys.stdout.write("\r\033[K")
        sys.stdout.write(f"\r  {self._c(self.DIM, '…')} {self._c(self.DIM, msg)}")
        sys.stdout.flush()
        self._in_progress_line = True

    def info(self, msg: str) -> None:
        if not self.enabled:
            return
        self._clear_progress()
        print(f"  {self._c(self.CYAN, '•')} {msg}")

    def warn(self, msg: str) -> None:
        if not self.enabled:
            return
        self._clear_progress()
        print(f"  {self._c(self.YELLOW, '!')} {msg}")

    def error(self, msg: str) -> None:
        if not self.enabled:
            return
        self._clear_progress()
        print(f"  {self._c(self.RED, '✗')} {msg}")


ui = ConsoleUI()
