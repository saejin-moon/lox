#!/usr/bin/env python3
"""
CORP-Ω R5: NetHack tuning campaign runner (AGENT_PLAN §8).

Nightly cadence: one gated revision batch per night (run_revision_loop), then a
10-ep engineering batch to measure the accepted program. Plateau detection: three
consecutive nights with no accepted revision beyond 1 batch-σ → the loop pauses and
writes a plateau report (the human decides: new capabilities vs accept plateau).

  uv run python scripts/run_campaign.py --nights 7 --provider gemini --model gemma-4-26b-a4b-it
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEDGER_PATH = "data/revision_ledger.jsonl"
PLATEAU_REPORT = "data/plateau_report.json"


def run_night(args, night: int) -> dict:
    """One campaign night: revision loop + engineering batch."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    loop_log = f"logs/campaign_night{night}_loop_{ts}.log"
    print(f"\n=== night {night + 1}/{args.nights} ===")

    cmd = [
        "uv", "run", "python", "scripts/run_revision_loop.py",
        "--provider", args.provider, "--max-revisions", str(args.revisions_per_night),
        "--no-git" if args.no_git else "--live-gates",
    ]
    if args.model:
        cmd += ["--model", args.model]
    with open(loop_log, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=args.night_timeout)
    if proc.returncode != 0:
        print(f"  [warn] revision loop exited {proc.returncode} (see {loop_log})")

    accepted = _accepted_revisions_since(args.campaign_start_ts)
    print(f"  accepted revisions tonight: {len(accepted)}")

    # Engineering batch: 10-ep measure of the current program
    batch_out = f"data/campaign_night{night}_batch.json"
    subprocess.run(
        ["uv", "run", "python", "scripts/run_benchmark.py", "--role", "valkyrie",
         "--episodes", str(args.batch_episodes), "--max-steps", "20000",
         "--output", batch_out],
        check=False, capture_output=True, timeout=args.night_timeout)
    mean_score = None
    try:
        with open(batch_out) as f:
            mean_score = json.load(f).get("mean_score")
    except Exception:
        pass
    print(f"  engineering batch: mean_score={mean_score}")
    return {"night": night + 1, "accepted": len(accepted), "mean_score": mean_score,
            "revisions": [r.get("revision") for r in accepted]}


def _accepted_revisions_since(ts: float) -> list[dict]:
    out = []
    if not os.path.exists(LEDGER_PATH):
        return out
    with open(LEDGER_PATH) as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type") == "policy_revision" and e.get("accepted") and e.get("ts", 0) > ts:
                out.append(e)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="CORP R5 tuning campaign")
    p.add_argument("--nights", type=int, default=7)
    p.add_argument("--provider", type=str, default="gemini")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--revisions-per-night", type=int, default=3)
    p.add_argument("--batch-episodes", type=int, default=10)
    p.add_argument("--plateau-nights", type=int, default=3,
                   help="consecutive nights without an accepted revision → plateau")
    p.add_argument("--night-timeout", type=int, default=21600)
    p.add_argument("--no-git", action="store_true")
    args = p.parse_args()

    start_ts = time.time()
    history = []
    for night in range(args.nights):
        night_result = run_night(args, night)
        history.append(night_result)
        if night_result["accepted"] == 0:
            dry = sum(1 for h in history[-args.plateau_nights:] if h["accepted"] == 0)
            if dry >= args.plateau_nights:
                report = {
                    "plateau_detected": True,
                    "nights_without_acceptance": dry,
                    "history": history,
                    "recommendation": "loop paused: 3+ consecutive nights without an "
                                      "accepted revision. Human decision required: new "
                                      "capabilities (R6/R8) vs accept the plateau.",
                }
                with open(PLATEAU_REPORT, "w") as f:
                    json.dump(report, f, indent=2)
                print(f"\n[campaign] PLATEAU detected after {dry} dry nights "
                      f"-> {PLATEAU_REPORT}")
                return 1
        else:
            # acceptance resets the dry-night counter
            continue

    print(f"\n[campaign] {args.nights} nights complete, no plateau detected")
    with open("data/campaign_history.json", "w") as f:
        json.dump({"history": history}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())