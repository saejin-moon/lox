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
    ACTION_ESC = 38        # Command.ESC
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

        # Character to action index mapping for text dialog typing
        self.char_to_action: dict[str, int] = {}
        if hasattr(env.unwrapped, "actions"):
            for i, a in enumerate(env.unwrapped.actions):
                if hasattr(a, "value") and 32 <= a.value <= 126:
                    self.char_to_action[chr(a.value)] = i

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

        if terminated or truncated:
            full_msg = self._extract_message(obs)
            self.current_blstats = BottomLineStats.from_blstats(obs["blstats"])
            anomalies = self.anomaly_sentry.evaluate(self.current_blstats, full_msg)
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
            if self._is_wish_prompt(obs):
                has_ref = False
                if self.inventory_tracker is not None:
                    active = self.inventory_tracker.get_active_items() if hasattr(self.inventory_tracker, "get_active_items") else [it for it in getattr(self.inventory_tracker, "active_items", {}).values() if getattr(it, "is_active", True)]
                    has_ref = any("reflection" in it.raw_str.lower() for it in active)
                wish = "blessed +2 gray dragon scale mail" if has_ref else "blessed +2 silver dragon scale mail"
                for ch in wish:
                    ch_idx = self.char_to_action.get(ch)
                    if ch_idx is not None:
                        obs, r, term, trunc, _ = self.env.step(ch_idx)
                        dialog_steps += 1
                obs, r, term, trunc, _ = self.env.step(self.ACTION_MORE)
                dialog_steps += 1
                new_msg = self._extract_message(obs)
                if new_msg and (not messages or new_msg != messages[-1]):
                    messages.append(new_msg)
                if term or trunc:
                    terminated = terminated or term
                    truncated = truncated or trunc
                    break
                continue

            more_needed = self._is_more(obs)
            yn_needed = self._is_yn_prompt(obs)
            cancel_needed = self._is_unhandled_prompt(obs)

            if not more_needed and not yn_needed and not cancel_needed:
                break

            action_to_send = self.ACTION_SPACE
            if yn_needed:
                action_to_send = self._resolve_yn_action(obs)
            elif cancel_needed:
                action_to_send = self.ACTION_ESC if dialog_steps >= 5 else self.ACTION_SPACE
            elif more_needed:
                tty_bytes = bytes(obs.get("tty_chars", b""))
                if (b" of " in tty_bytes or b"(end)" in tty_bytes) and dialog_steps >= 5:
                    action_to_send = self.ACTION_ESC
                else:
                    action_to_send = self.ACTION_SPACE

            try:
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
            except RuntimeError as e:
                if "finished NetHack" in str(e):
                    terminated = True
                    break
                raise

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
        """Detects if terminal is paused at a --More-- prompt or paginated menu."""
        # 1. Message check
        msg_raw = bytes(obs.get("message", b""))
        if b"--More--" in msg_raw:
            return True

        # 2. TTY screen check
        tty_bytes = bytes(obs.get("tty_chars", b""))
        if (
            b"--More--" in tty_bytes
            or b" of " in tty_bytes
            or b"(end)" in tty_bytes
        ):
            return True

        return False

    def _is_yn_prompt(self, obs: dict[str, Any]) -> bool:
        """Detects if terminal is asking a [yn] query."""
        msg = self._extract_message(obs).lower()
        return (
            "[yn]" in msg
            or "[y/n]" in msg
            or "[ynq]" in msg
            or "[y,n]" in msg
            or "really attack" in msg
            or "really flee" in msg
            or "really quit" in msg
        )

    def _resolve_yn_action(self, obs: dict[str, Any]) -> int:
        """Determines whether to answer 'y' or 'n' to a prompt."""
        msg = self._extract_message(obs).lower()
        # Confirm eating when prompted on the floor (unless tainted, rotten, or petrifying!)
        if "eat it" in msg or "eat one" in msg:
            if any(bad in msg for bad in ("rotten", "tainted", "smells terrible", "chickatrice", "cockatrice", "medusa")):
                return self.ACTION_N
            return self.ACTION_Y
        # Confirm divine prayer when deliberately initiated by HTN planner
        if "really pray" in msg or "sure you want to pray" in msg:
            return self.ACTION_Y
        # Confirm altar sacrifice when deliberately initiated
        if "sacrifice" in msg:
            return self.ACTION_Y
        # Confirm adding to existing engraving in dust/floor
        if "add to it" in msg:
            return self.ACTION_Y
        # Confirm dipping into fountain
        if "dip it into the fountain" in msg or "dip into the fountain" in msg or "fountain" in msg:
            return self.ACTION_Y
        # Confirm paying shopkeeper
        if "pay " in msg or "pay?" in msg or "will you pay" in msg or "pay for" in msg:
            return self.ACTION_Y
        # Never attack peaceful monsters or pets when prompted
        if "really attack" in msg:
            return self.ACTION_N
        # Identify items post-mortem
        if "possessions identified" in msg:
            return self.ACTION_Y
        # Safe default for dangers or quitting
        if "really quit" in msg or "die" in msg:
            return self.ACTION_N
        # Safe default is 'n'
        return self.ACTION_N

    def _is_unhandled_prompt(self, obs: dict[str, Any]) -> bool:
        """Detects if terminal is stuck in an unhandled item prompt."""
        msg = self._extract_message(obs).lower()
        if (
            "you don't have that object" in msg
            or "what do you want to write with" in msg
            # Free-text input prompts (e.g. from #loot / #force side effects) — ESC cancels
            or "what are you looking for" in msg
            or "what do you want to drop" in msg
            or "what do you want to put in" in msg
            or "what do you want to take out" in msg
            or "what do you want to throw" in msg
            or "what do you want to zap" in msg
            or "what do you want to use or apply" in msg
            or "what do you want to eat" in msg
            or "what do you want to quaff" in msg
            or "what do you want to read" in msg
            or "talk to whom" in msg
        ):
            return True
        return False

    def _is_wish_prompt(self, obs: dict[str, Any]) -> bool:
        """Detects if terminal is prompting for a wish."""
        msg = self._extract_message(obs).lower()
        return "for what do you wish" in msg


