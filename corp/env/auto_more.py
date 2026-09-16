"""
AutoMoreWrapper: Deterministic dialog, menu, and prompt flushing for NLE.
Ensures the higher-order reasoning layer only observes clean, interactive game states.
"""

from typing import Any
import gymnasium as gym
import numpy as np
from nle import nethack

from corp.env.blstats import BottomLineStats
from corp.env.inventory_tracker import InventoryNormalizer
from corp.env.anomaly_sentry import AnomalySentry, AnomalyEvent
from corp.env.flight_recorder import FlightRecorderRingBuffer


class AutoMoreWrapper(gym.Wrapper):
    """
    Wraps Gymnasium NetHack environment to swallow '--More--', multi-page menus,
    and confirmation queries deterministically without stalling the HTN planner.
    """

    ACTION_MORE = 19       # nethack.MiscAction.MORE (\r / enter)
    ACTION_SPACE = 107     # Spacebar
    ACTION_ESC = 108       # Escape if mapped, or 107
    ACTION_N = 5           # CompassDirection.SE / 'n'
    ACTION_Y = 7           # CompassDirection.NW / 'y'

    def __init__(
        self,
        env: gym.Env,
        max_dialog_steps: int = 50,
        flight_recorder_capacity: int = 100,
    ):
        super().__init__(env)
        self.max_dialog_steps = max_dialog_steps

        # Subsystems
        self.inventory_tracker = InventoryNormalizer()
        self.anomaly_sentry = AnomalySentry()
        self.flight_recorder = FlightRecorderRingBuffer(capacity=flight_recorder_capacity)

        # Internal state
        self.current_blstats: BottomLineStats | None = None
        self.accumulated_message: str = ""
        self.step_count: int = 0

    def reset(self, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self.anomaly_sentry.reset()
        self.flight_recorder.clear()
        self.step_count = 0

        # Flush any startup banners / dialogs
        obs, flushed_msg, terminated, truncated = self._flush_dialogs(obs)

        # Synchronize initial state
        self.current_blstats = BottomLineStats.from_blstats(obs["blstats"])
        if "inv_strs" in obs and "inv_letters" in obs and "inv_glyphs" in obs:
            self.inventory_tracker.synchronize(
                obs["inv_strs"],
                obs["inv_letters"],
                obs["inv_glyphs"],
                obs.get("inv_oclasses"),
                turn=self.current_blstats.turn,
            )

        anomalies = self.anomaly_sentry.evaluate(self.current_blstats, flushed_msg)
        self.flight_recorder.record(
            turn=self.current_blstats.turn,
            action="RESET",
            blstats=self.current_blstats,
            message=flushed_msg,
            tty_chars=obs.get("tty_chars"),
            anomalies=anomalies,
        )

        info["blstats"] = self.current_blstats
        info["anomalies"] = anomalies
        info["full_message"] = flushed_msg
        info["inventory_tracker"] = self.inventory_tracker

        return obs, info

    def step(self, action: int | Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1

        total_reward = float(reward)

        # Flush any dialogs triggered by this action
        obs, full_msg, sub_term, sub_trunc = self._flush_dialogs(obs)
        terminated = terminated or sub_term
        truncated = truncated or sub_trunc

        # Synchronize bottom-line stats
        self.current_blstats = BottomLineStats.from_blstats(obs["blstats"])

        # Synchronize inventory if tensors present
        if "inv_strs" in obs and "inv_letters" in obs and "inv_glyphs" in obs:
            self.inventory_tracker.synchronize(
                obs["inv_strs"],
                obs["inv_letters"],
                obs["inv_glyphs"],
                obs.get("inv_oclasses"),
                turn=self.current_blstats.turn,
            )

        # Evaluate physical invariants via Anomaly Sentry
        anomalies = self.anomaly_sentry.evaluate(self.current_blstats, full_msg)

        # Record to flight recorder
        self.flight_recorder.record(
            turn=self.current_blstats.turn,
            action=action,
            blstats=self.current_blstats,
            message=full_msg,
            tty_chars=obs.get("tty_chars"),
            anomalies=anomalies,
        )

        info["blstats"] = self.current_blstats
        info["anomalies"] = anomalies
        info["full_message"] = full_msg
        info["inventory_tracker"] = self.inventory_tracker

        return obs, total_reward, terminated, truncated, info

    def _flush_dialogs(self, obs: dict[str, Any]) -> tuple[dict[str, Any], str, bool, bool]:
        """
        Synchronously flushes '--More--', confirmation prompts, and multi-page text
        until reaching a fully interactive prompt.
        """
        dialog_steps = 0
        messages: list[str] = []
        terminated = False
        truncated = False

        initial_msg = self._extract_message(obs)
        if initial_msg:
            messages.append(initial_msg)

        while dialog_steps < self.max_dialog_steps:
            more_needed = self._is_more(obs)
            yn_needed = self._is_yn_prompt(obs)

            if not more_needed and not yn_needed:
                break

            action_to_send = self.ACTION_MORE
            if yn_needed:
                action_to_send = self._resolve_yn_action(obs)

            obs, r, term, trunc, _ = self.env.step(action_to_send)
            dialog_steps += 1
            if term:
                terminated = True
            if trunc:
                truncated = True

            new_msg = self._extract_message(obs)
            if new_msg and (not messages or new_msg != messages[-1]):
                messages.append(new_msg)

            if terminated or truncated:
                break

        full_message = " ".join(messages)
        return obs, full_message, terminated, truncated

    def _extract_message(self, obs: dict[str, Any]) -> str:
        msg_bytes = bytes(obs.get("message", b""))
        clean = msg_bytes.split(b"\x00")[0].decode("latin-1", errors="replace").strip()
        # Strip out trailing --More--
        if clean.endswith("--More--"):
            clean = clean[:-8].strip()
        return clean

    def _is_more(self, obs: dict[str, Any]) -> bool:
        """Detects if terminal is paused at a --More-- prompt."""
        # 1. Message check
        msg_raw = bytes(obs.get("message", b""))
        if b"--More--" in msg_raw:
            return True

        # 2. TTY screen check
        tty_bytes = bytes(obs.get("tty_chars", b""))
        if b"--More--" in tty_bytes or b"(1 of " in tty_bytes or b"(end)" in tty_bytes:
            return True

        return False

    def _is_yn_prompt(self, obs: dict[str, Any]) -> bool:
        """Detects if terminal is asking a [yn] query."""
        msg = self._extract_message(obs)
        return (
            "[yn]" in msg.lower()
            or "[y/n]" in msg.lower()
            or "[ynq]" in msg.lower()
            or "[y,n]" in msg.lower()
        )

    def _resolve_yn_action(self, obs: dict[str, Any]) -> int:
        """Determines whether to answer 'y' or 'n' to a prompt."""
        msg = self._extract_message(obs).lower()
        # Confirm eating when prompted on the floor
        if "eat it" in msg or "eat one" in msg:
            return self.ACTION_Y
        # Confirm divine prayer when deliberately initiated
        if "really pray" in msg or "sure you want to pray" in msg:
            return self.ACTION_Y
        # Confirm attack on peaceful monsters when combat engine committed to strike
        if "really attack" in msg:
            return self.ACTION_Y
        # Identify items post-mortem
        if "possessions identified" in msg:
            return self.ACTION_Y
        # Safe default for dangers or quitting
        if "really quit" in msg or "die" in msg:
            return self.ACTION_N
        # Safe default is 'n'
        return self.ACTION_N
