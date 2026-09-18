"""Probe: catch turns where dialog flushing consumes anomalous env steps."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from corp.env.nle_wrapper import make_env
from corp.env.auto_more import AutoMoreWrapper
from corp.agent.corp_agent import CORPAgent
from corp.planner.nogood import NogoodStore
from corp.deliberative.providers.mock_provider import MockProvider

stats = {"dialog_turns": 0, "max_dialog": 0, "worst_msgs": []}


def main():
    # Instrument AutoMoreWrapper._flush_dialogs
    orig_flush = AutoMoreWrapper._flush_dialogs

    def counting_flush(self, obs):
        before = self._dialog_counter if hasattr(self, "_dialog_counter") else 0
        # count actual env steps by wrapping self.env.step
        n = {"steps": 0}
        orig_env_step = self.env.step

        def counting_step(action):
            n["steps"] += 1
            return orig_env_step(action)

        self.env.step = counting_step
        try:
            out = orig_flush(self, obs)
        finally:
            self.env.step = orig_env_step
        if n["steps"] > stats["max_dialog"]:
            stats["max_dialog"] = n["steps"]
            stats["worst_msgs"] = [str(out[0].get("message", ""))[:80] if isinstance(out[0], dict) else ""]
        if n["steps"] > 8:
            stats["dialog_turns"] += 1
            if len(stats["worst_msgs"]) < 10:
                msg = out[0].get("message", "") if isinstance(out[0], dict) else ""
                stats["worst_msgs"].append(f"{n['steps']} steps: {str(msg)[:70]}")
        if n["steps"] > 30:
            print(f"    SPIKE {n['steps']} steps, msg={str(out[0].get('message',''))[:70] if isinstance(out[0], dict) else ''}", flush=True)
        return out

    AutoMoreWrapper._flush_dialogs = counting_flush

    for ep in range(1):
        env = make_env(character="val-hum-law-fem")
        agent = CORPAgent(
            env=env,
            nogood_store=NogoodStore(),
            llm_provider=MockProvider(),
            eval_type="debug",
            mode="random",
            seed=42 + ep,
            role="valkyrie",
        )
        result = agent.run_episode(max_steps=20000)
        print(
            f"Ep{ep}: turns={result.turns} depth={result.max_depth} score={result.final_score} "
            f"dialog_flush_turns>8={stats['dialog_turns']} max_dialog={stats['max_dialog']}",
            flush=True,
        )
        stats["dialog_turns"] = 0
        stats["max_dialog"] = 0
        for m in stats["worst_msgs"][:5]:
            print(f"   {m}")
        stats["worst_msgs"] = []


if __name__ == "__main__":
    main()
