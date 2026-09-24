"""
LOX-ψ Policy Layer (LOX-ψ): run-report bundle builder.

Builds the author's evidence from DuckDB. Two design rules (evidence audit, S1.5):

1. SCOPE — every aggregate is scoped to ONE run (`run_id`), defaulting to the most
   recent run. The previous bundle mixed the last 100 episodes across dozens of
   runs/programs, so deltas were not attributable to the revision under test.
   The bundle carries an explicit `scope` header so the author knows what it is
   looking at, plus a `prev_run` aggregate for change attribution.

2. CONDENSATION — raw ticks are never shipped. Instead the bundle carries compact
   per-episode DIGESTS (depth/goal timelines, action mix, stall + revisit stats,
   HP curve, normalized death cause) and event-anchored run-length trajectory
   slices. Raw ticks stay one tool call away (`read_trajectory` / `query_duckdb`).

Missing tables/DBs degrade to an empty-but-shaped bundle so the loop never crashes.
"""
from __future__ import annotations

import json
import os
import re
import statistics

from lox.policy.ledger import RevisionLedger, DEFAULT_LEDGER_PATH

DEFAULT_DB_PATH = "data/lox_telemetry.duckdb"
VARIANCE_NOTE = "NetHackChallenge unseedable; deltas < 1 batch-sigma are noise"

SURVIVED_MARKERS = ("", "survived", "alive", "ascended")
_DEATH_CATEGORY_JUNK = ("", "none", "unknown", "null")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else 0


def _is_survived(death_message: str) -> bool:
    return (death_message or "").strip().lower() in SURVIVED_MARKERS


def _table_cols(con, name: str) -> set[str] | None:
    try:
        return {r[0] for r in con.execute(f"DESCRIBE {name}").fetchall()}
    except Exception:  # noqa: BLE001
        return None


def _normalize_cause(death_message: str, death_category: str | None) -> str:
    """A stable, low-cardinality death cause. Prefers episodes.death_category when
    it carries signal; otherwise normalizes the raw message to its first sentence
    (starvation deaths are recorded as 'You faint from lack of food.  The kobold
    zombie hits! ...', so the raw string fragments the taxonomy). Melee kills
    ('The giant rat bites!') are bucketed to 'melee (<verb>)' to keep the taxonomy
    compact while preserving the death mode."""
    cat = (death_category or "").strip().lower()
    if cat not in _DEATH_CATEGORY_JUNK:
        return cat
    msg = (death_message or "").strip()
    if _is_survived(msg):
        return "survived"
    first = msg.split(". ", 1)[0].split("! ", 1)[0].strip().rstrip(".!")
    low = first.lower()
    for verb in ("bites", "hits", "claws", "kicks", "stings", "strikes",
                 "touches", "breathes", "gazes", "spits", "butts"):
        if verb in low.split():
            return f"melee ({verb})"
    return low if low else "unknown"


def _scope_base(run_id: str) -> str:
    """Parallel batches write one run_id per worker ('<base>_w<n>'); the batch group is
    the prefix. Strip the worker suffix so the scope covers the whole batch of 100
    episodes, not a single worker's handful."""
    return re.sub(r"_w\d+$", "", run_id or "")


def _resolve_run(con, run_id: str | None, domain: str = "nethack") -> tuple[str | None, str]:
    if run_id:
        return _scope_base(run_id), "explicit"
    try:
        ecols = _table_cols(con, "episodes")
        if ecols and "domain" in ecols:
            row = con.execute(
                "SELECT run_id FROM episodes WHERE domain = ? ORDER BY rowid DESC LIMIT 1",
                (domain,),
            ).fetchone()
            if row and row[0]:
                base = _scope_base(row[0])
                return base, ("latest_batch" if base != row[0] else "latest_run")

        if domain != "nethack":
            tcols_lineage = _table_cols(con, "run_lineage")
            if tcols_lineage and "domain" in tcols_lineage:
                row = con.execute(
                    "SELECT base_run_id FROM run_lineage WHERE domain = ? ORDER BY created_ts DESC LIMIT 1",
                    (domain,),
                ).fetchone()
                if row and row[0]:
                    return row[0], "latest_domain_batch"
            return None, "empty"

        if ecols:
            row = con.execute("SELECT run_id FROM episodes ORDER BY rowid DESC LIMIT 1").fetchone()
            if row and row[0]:
                base = _scope_base(row[0])
                return base, ("latest_batch" if base != row[0] else "latest_run")

        return None, "empty"
    except Exception:  # noqa: BLE001
        return None, "empty"


def _run_episode_ids(con, run_id: str | None) -> list[str]:
    if run_id is None:
        return []
    try:
        return [r[0] for r in con.execute(
            "SELECT episode_id FROM episodes WHERE run_id LIKE ?",
            (run_id + "%",)).fetchall()]
    except Exception:  # noqa: BLE001
        return []


# ---------------------------------------------------------------------------
# Per-episode digest (condensed evidence; ~100-250 tokens each)
# ---------------------------------------------------------------------------

def _digest_episode(con, tcols: set[str] | None, ep: dict) -> str:
    """Compact multi-line digest of one episode. Never raises."""
    ep_id = ep["episode_id"]
    cols = ["turn", "depth", "hp", "max_hp", "action_name"]
    for extra in ("x", "y"):
        if tcols and extra in tcols:
            cols.append(extra)
    try:
        rows = con.execute(
            f"SELECT {', '.join(cols)} FROM ticks WHERE episode_id = ? ORDER BY step",
            (ep_id,)).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    idx = {c: i for i, c in enumerate(cols)}

    head = (f"ep {ep_id} | cause={_normalize_cause(ep.get('death_message'), ep.get('death_category'))}"
            f" | max_depth={ep.get('max_depth')} | turns={ep.get('total_turns')}"
            f" | score={ep.get('final_score')}")
    lines = [head]
    if not rows:
        lines.append("  (no tick telemetry)")
        return "\n".join(lines)

    # depth timeline (segments of constant depth)
    segs: list[tuple[int, int]] = []   # (depth, start_turn)
    start_turn = rows[0][idx["turn"]]
    last_depth = rows[0][idx["depth"]]
    for r in rows[1:]:
        if r[idx["depth"]] != last_depth:
            segs.append((last_depth, start_turn))
            last_depth = r[idx["depth"]]
            start_turn = r[idx["turn"]]
    segs.append((last_depth, start_turn))
    timeline = " → ".join(f"d{d}@{t}" for d, t in segs[:12])
    if len(segs) > 12:
        timeline += f" …(+{len(segs) - 12} more)"
    lines.append(f"  depth: {timeline}")

    # action mix (top 5)
    counts: dict[str, int] = {}
    for r in rows:
        a = r[idx["action_name"]] or "?"
        counts[a] = counts.get(a, 0) + 1
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    total = len(rows)
    lines.append("  actions: " + ", ".join(
        f"{a} {100 * c // total}%({c})" for a, c in top))

    # stall: longest turn span without a depth increase
    max_stall = 0
    stall_start = 0
    for i in range(1, len(rows)):
        if rows[i][idx["depth"]] > rows[i - 1][idx["depth"]]:
            max_stall = max(max_stall, i - stall_start)
            stall_start = i
    max_stall = max(max_stall, len(rows) - stall_start)
    # revisits: fraction of steps landing on an already-visited tile
    revisit_txt = ""
    if "x" in idx and "y" in idx:
        seen = set()
        rev = 0
        for r in rows:
            p = (r[idx["x"]], r[idx["y"]])
            if p in seen:
                rev += 1
            else:
                seen.add(p)
        revisit_txt = f"; revisits {100 * rev // max(1, total)}% of steps"
    lines.append(f"  stall: longest depth-gain gap {max_stall} turns{revisit_txt}")

    # hp curve
    min_frac = 1.0
    below = 0
    for r in rows:
        mhp = max(1, r[idx["max_hp"]] or 1)
        frac = (r[idx["hp"]] or 0) / mhp
        min_frac = min(min_frac, frac)
        if frac <= 0.30:
            below += 1
    lines.append(f"  hp: min {100 * min_frac:.0f}% ; {100 * below // total}% of turns <=30% hp"
                 f" ; final {rows[-1][idx['hp']]}/{rows[-1][idx['max_hp']]}")

    # goal timeline (scoped to this episode)
    try:
        gts = con.execute(
            "SELECT turn, goal, event, steps_in_goal FROM goal_events "
            "WHERE episode_id = ? ORDER BY turn", (ep_id,)).fetchall()
        if gts:
            parts = [f"{g}(t{t} {ev}" + (f" {s}st" if s else "") + ")"
                     for t, g, ev, s in gts[:10]]
            lines.append("  goals: " + " → ".join(parts))
    except Exception:  # noqa: BLE001
        pass
    # event census (anomalies + message-class transitions) — the *why*
    try:
        ev = con.execute(
            "SELECT kind, code, COUNT(*) FROM events WHERE episode_id = ? "
            "GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 8", (ep_id,)).fetchall()
        if ev:
            anom = [f"{c}×{n}" for k, c, n in ev if k == "anomaly"]
            msgs = [f"{c}×{n}" for k, c, n in ev if k == "message"]
            if anom:
                lines.append("  anomalies: " + ", ".join(anom))
            if msgs:
                lines.append("  message classes: " + ", ".join(msgs))
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def build_episode_digests(db_path: str, run_id: str | None = None,
                          n: int = 4, domain: str = "nethack") -> list[str]:
    """Condensed per-episode digests for representative episodes of one run:
    the longest deaths by normalized cause + the deepest + the longest-stalling.
    Returns a list of compact markdown strings."""
    if not os.path.exists(db_path):
        return []
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
    except Exception:  # noqa: BLE001
        return []
    try:
        ecols = _table_cols(con, "episodes")
        if not ecols:
            return []
        rid, _src = _resolve_run(con, run_id, domain=domain)
        if rid is None:
            return []
        tcols = _table_cols(con, "ticks")
        wanted = ["episode_id", "death_message", "death_category", "max_depth",
                  "final_score", "total_turns", "run_id"]
        sel = [c for c in wanted if c in ecols]
        where = "WHERE run_id LIKE ?"
        params = [rid + "%"]
        rows = con.execute(
            f"SELECT {', '.join(sel)} FROM episodes {where} "
            f"ORDER BY total_turns DESC LIMIT 50", params).fetchall()
        eps = [dict(zip(sel, r)) for r in rows]
        if not eps:
            return []
        # representative selection: modal-cause example + deepest + longest
        by_cause: dict[str, dict] = {}
        for e in eps:
            cause = _normalize_cause(e.get("death_message"), e.get("death_category"))
            if cause not in by_cause:
                by_cause[cause] = e
        chosen: list[dict] = []
        for e in sorted(by_cause.values(), key=lambda x: -(x.get("total_turns") or 0)):
            chosen.append(e)
        for e in sorted(eps, key=lambda x: -(x.get("max_depth") or 0)):
            if all(e["episode_id"] != c["episode_id"] for c in chosen):
                chosen.append(e)
                break
        for e in eps:  # longest-stalling (by turns)
            if all(e["episode_id"] != c["episode_id"] for c in chosen):
                chosen.append(e)
                break
        return [_digest_episode(con, tcols, e) for e in chosen[:max(1, n)]]
    except Exception:  # noqa: BLE001 — digests are advisory
        return []
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Event-anchored, run-length-encoded trajectory slices
# ---------------------------------------------------------------------------

def _rle_window(rows: list[tuple], cols: list[str]) -> str:
    """Collapses consecutive ticks that share (depth, hp, max_hp, action) into one
    `t<from>-<to> d<depth> hp<h>/<mh> <action> ×count` line. No stride sampling:
    every material change boundary is preserved (stride aliases short kill windows)."""
    idx = {c: i for i, c in enumerate(cols)}
    if not rows:
        return "(no ticks)"
    out = []
    r0 = rows[0]
    cur = (r0[idx["depth"]], r0[idx["hp"]], r0[idx["max_hp"]], r0[idx["action_name"]])
    start_t = r0[idx["turn"]]
    last_t = r0[idx["turn"]]
    count = 1
    for r in rows[1:]:
        k = (r[idx["depth"]], r[idx["hp"]], r[idx["max_hp"]], r[idx["action_name"]])
        if k == cur:
            count += 1
            last_t = r[idx["turn"]]
        else:
            out.append(f"t{start_t}-{last_t} d{cur[0]} hp{cur[1]}/{cur[2]} {cur[3]} ×{count}")
            cur = k
            start_t = last_t = r[idx["turn"]]
            count = 1
    out.append(f"t{start_t}-{last_t} d{cur[0]} hp{cur[1]}/{cur[2]} {cur[3]} ×{count}")
    return "\n".join(out)


def build_run_distribution(db_path: str, run_id: str | None = None,
                           d1_stall_turns: int = 5000, domain: str = "nethack") -> dict:
    """DISTRIBUTIONAL shape of one run — the batch-level view a per-episode digest
    cannot give: how many episodes never left depth 1, how many burned the budget
    stalled there, the batch-wide action mix, and the event/anomaly census.

    'N of M episodes spent >5k turns at depth 1' is the kind of statement the author
    can act on; raw counts are not."""
    dist: dict = {"episodes": 0, "depth_histogram": {}, "never_left_d1": 0,
                  "long_d1_stalls": 0, "median_turns": 0, "action_mix": [],
                  "anomalies": [], "message_classes": []}
    if not os.path.exists(db_path):
        return dist
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
    except Exception:  # noqa: BLE001
        return dist
    try:
        ecols = _table_cols(con, "episodes")
        if not ecols:
            return dist
        rid, _src = _resolve_run(con, run_id, domain=domain)
        if rid is None:
            return dist
        where = "WHERE run_id LIKE ?"
        params = [rid + "%"]
        rows = con.execute(
            f"SELECT max_depth, total_turns FROM episodes {where}", params).fetchall()
        dist["episodes"] = len(rows)
        if rows:
            buckets = {"1": 0, "2-3": 0, "4-6": 0, "7-9": 0, "10+": 0}
            for depth, turns in rows:
                d = depth or 0
                key = ("1" if d <= 1 else "2-3" if d <= 3 else "4-6" if d <= 6
                       else "7-9" if d <= 9 else "10+")
                buckets[key] += 1
                if d <= 1:
                    dist["never_left_d1"] += 1
                    if (turns or 0) >= d1_stall_turns:
                        dist["long_d1_stalls"] += 1
            dist["depth_histogram"] = buckets
            dist["median_turns"] = int(_median([t for _d, t in rows]))

        if rid is not None:
            try:
                act = con.execute(
                    """SELECT action_name, COUNT(*) FROM ticks
                       WHERE episode_id IN (SELECT episode_id FROM episodes WHERE run_id LIKE ?)
                       GROUP BY 1 ORDER BY 2 DESC LIMIT 8""", (rid + "%",)).fetchall()
                total = sum(c for _a, c in act) or 1
                dist["action_mix"] = [{"action": a or "?", "pct": round(100 * c / total),
                                       "count": int(c)} for a, c in act]
            except Exception:  # noqa: BLE001
                pass
            try:
                ev = con.execute(
                    """SELECT kind, code, COUNT(*) FROM events WHERE run_id LIKE ?
                       GROUP BY 1, 2 ORDER BY 3 DESC LIMIT 12""", (rid + "%",)).fetchall()
                dist["anomalies"] = [{"code": c, "count": int(n)}
                                     for k, c, n in ev if k == "anomaly"]
                dist["message_classes"] = [{"class": c, "count": int(n)}
                                           for k, c, n in ev if k == "message"]
            except Exception:  # noqa: BLE001 — events table may not exist yet
                pass
        return dist
    except Exception:  # noqa: BLE001
        return dist
    finally:
        con.close()


def build_trajectory_slices(db_path: str, n_episodes: int = 3, last_ticks: int = 40,
                            run_id: str | None = None, domain: str = "nethack") -> list[dict]:
    """Event-anchored (death) trajectory slices for the run: RLE-encoded ticks +
    goal events. Keyed {episode_id, death_message, cause, slice}."""
    if not os.path.exists(db_path):
        return []
    try:
        import duckdb
        con = duckdb.connect(db_path, read_only=True)
    except Exception:  # noqa: BLE001
        return []
    try:
        ecols = _table_cols(con, "episodes")
        if not ecols or "death_message" not in ecols:
            return []
        rid, _src = _resolve_run(con, run_id, domain=domain)
        if rid is None:
            return []
        tcols = _table_cols(con, "ticks")
        tsel = [c for c in ("step", "turn", "depth", "hp", "max_hp", "action_name")
                if tcols and c in tcols]
        where = "WHERE death_message IS NOT NULL AND death_message != '' " \
                "AND LOWER(death_message) NOT IN ('survived','alive','ascended') AND run_id LIKE ?"
        params: list = [rid + "%"]
        eps = con.execute(
            f"SELECT episode_id, death_message, death_category, max_depth, total_turns "
            f"FROM episodes {where} ORDER BY rowid DESC LIMIT ?",
            params + [int(n_episodes)]).fetchall()
        out = []
        for ep_id, death, category, max_depth, total_turns in eps:
            rows = con.execute(
                f"SELECT {', '.join(tsel)} FROM ticks WHERE episode_id = ? "
                f"ORDER BY turn DESC LIMIT ?", (ep_id, int(last_ticks))).fetchall()
            rows.reverse()
            goals = con.execute(
                "SELECT turn, goal, event FROM goal_events WHERE episode_id = ? "
                "ORDER BY turn DESC LIMIT 8", (ep_id,)).fetchall()
            lines = [f"### episode {ep_id} — cause={_normalize_cause(death, category)} "
                     f"death={death!r} max_depth={max_depth} turns={total_turns}",
                     _rle_window(rows, tsel)]
            if goals:
                lines.append("recent goal events: " + "; ".join(
                    f"t{t} {g}:{ev}" for t, g, ev in goals))
            try:
                evs = con.execute(
                    "SELECT turn, kind, code FROM events WHERE episode_id = ? "
                    "ORDER BY turn DESC LIMIT 12", (ep_id,)).fetchall()
                if evs:
                    lines.append("events: " + "; ".join(
                        f"t{t} {c}" for t, k, c in reversed(evs)))
            except Exception:  # noqa: BLE001
                pass
            out.append({"episode_id": ep_id, "death_message": death,
                        "cause": _normalize_cause(death, category),
                        "slice": "\n".join(lines)})
        return out
    except Exception:  # noqa: BLE001
        return []
    finally:
        con.close()


def build_failure_counters_section(db_path: str, last_n_episodes: int = 10,
                                   run_id: str | None = None) -> list[dict]:
    """Per-goal failure counts (goal_events → bundle). Same counts the runtime
    failure_counters predicate consumes."""
    from lox.policy.failure_counters import load_failure_counters
    counters = load_failure_counters(db_path, last_n_episodes=last_n_episodes)
    return [{"goal": g, "failures": c}
            for g, c in sorted(counters.items(), key=lambda kv: -kv[1])]


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------

def _batch_summary(con, ecols: set[str], run_id: str | None) -> dict:
    where = "WHERE run_id LIKE ?" if run_id is not None else ""
    params = [run_id + "%"] if run_id is not None else []
    cols = ["max_depth", "final_score", "total_turns", "death_message"]
    available = [c for c in cols if c in ecols]
    if "max_depth" not in available:
        return {"episodes": 0, "median_depth": 0.0, "mean_score": 0.0,
                "best_score": 0, "survival_rate": 0.0, "mean_turns": 0.0}
    row = con.execute(
        f"""SELECT COUNT(*), MEDIAN(max_depth), AVG(final_score), MAX(final_score),
                   SUM(CASE WHEN LOWER(COALESCE(death_message,'')) IN ('', 'survived','alive','ascended')
                            THEN 1 ELSE 0 END), AVG(total_turns)
            FROM episodes {where}""", params).fetchone()
    if not row or not row[0]:
        return {"episodes": 0, "median_depth": 0.0, "mean_score": 0.0,
                "best_score": 0, "survival_rate": 0.0, "mean_turns": 0.0}
    return {"episodes": int(row[0]), "median_depth": float(row[1] or 0),
            "mean_score": float(row[2] or 0), "best_score": int(row[3] or 0),
            "survival_rate": float(row[4] or 0) / max(1, int(row[0])),
            "mean_turns": float(row[5] or 0)}


def build_report_bundle(
    db_path: str = DEFAULT_DB_PATH,
    ledger_path: str = DEFAULT_LEDGER_PATH,
    domain: str = "nethack",
    run_id: str | None = None,
    last_n_episodes: int = 100,
    digest_episodes: int = 4,
    candidate_id: str | None = None,
    population_id: str | None = None,
) -> dict:
    bundle: dict = {
        "domain": domain,
        "scope": {"run_id": run_id, "source": "none", "eval_type": None,
                  "episode_count": 0},
        "batch": {"episodes": 0, "median_depth": 0.0, "mean_score": 0.0,
                  "best_score": 0, "survival_rate": 0.0, "mean_turns": 0.0},
        "prev_run": {},
        "death_taxonomy": [],
        "stall_census": [],
        "goal_stats": [],
        "failure_counters": [],
        "distribution": {},
        "episode_digests": [],
        "trajectory_slices": [],
        "prev_revisions": [],
        "variance_note": VARIANCE_NOTE,
    }

    # ---- DuckDB: scoped batch + taxonomy + goals + digests + slices -----------
    if os.path.exists(db_path):
        try:
            import duckdb
            con = duckdb.connect(db_path, read_only=True)
        except Exception:  # noqa: BLE001
            con = None
        if con is not None:
            try:
                ecols = _table_cols(con, "episodes")
                if ecols:
                    lineage = None
                    if candidate_id:
                        # Per-candidate evidence: resolve the candidate's evaluation run
                        # group via the lineage registry, not the global latest run.
                        try:
                            from lox.evolution import RunLineage  # noqa: PLC0415
                            rg = RunLineage(db_path)
                            bases = rg.runs_for_candidate(candidate_id, population_id)
                            if bases:
                                rid, src, lineage = bases[0], "candidate", rg.lookup(bases[0])
                            else:
                                rid, src = None, "candidate_missing"
                        except Exception:  # noqa: BLE001
                            rid, src = _resolve_run(con, run_id, domain=domain)
                    else:
                        rid, src = _resolve_run(con, run_id, domain=domain)
                    eval_type = None
                    if rid is not None and "eval_type" in ecols:
                        try:
                            eval_type = con.execute(
                                "SELECT eval_type FROM episodes WHERE run_id LIKE ? LIMIT 1",
                                (rid + "%",)).fetchone()[0]
                        except Exception:  # noqa: BLE001
                            eval_type = None
                    ep_ids = _run_episode_ids(con, rid)
                    bundle["scope"] = {"run_id": rid, "source": src,
                                       "eval_type": eval_type,
                                       "episode_count": len(ep_ids)}
                    if lineage:
                        bundle["scope"]["lineage"] = {
                            k: lineage.get(k) for k in
                            ("candidate_id", "population_id", "generation",
                             "parent_id", "program_version")}
                    bundle["batch"] = (_batch_summary(con, ecols, rid)
                                       if rid is not None else bundle["batch"])
                    # previous distinct run (change attribution)
                    if rid is not None:
                        try:
                            prev_where = "run_id NOT LIKE ?"
                            prev_params = [rid + "%"]
                            if "domain" in ecols:
                                prev_where += " AND domain = ?"
                                prev_params.append(domain)
                            prev = con.execute(
                                f"""SELECT run_id FROM episodes WHERE {prev_where}
                                   ORDER BY rowid DESC LIMIT 1""", prev_params).fetchone()
                            if prev and prev[0] is not None:
                                bundle["prev_run"] = {
                                    "run_id": _scope_base(prev[0]),
                                    **_batch_summary(con, ecols, _scope_base(prev[0]))}
                        except Exception:  # noqa: BLE001
                            pass

                    # death taxonomy: NORMALIZED cause (not the raw 486-string space)
                    try:
                        if rid is None:
                            raise ValueError("no scoped run")
                        tsel = ["death_message", "max_depth", "episode_id"]
                        for c in ("death_category", "final_hp"):
                            if c in ecols:
                                tsel.append(c)
                        rows = con.execute(
                            f"""SELECT {', '.join(tsel)} FROM episodes
                                WHERE {'run_id LIKE ? AND ' if rid is not None else ''}
                                      death_message IS NOT NULL AND death_message != ''
                                      AND LOWER(death_message) NOT IN ('survived','alive','ascended')""",
                            ([rid + "%"] if rid is not None else [])).fetchall()
                        hp_by_ep = {}
                        if ep_ids:
                            try:
                                hp_rows = con.execute(
                                    """SELECT episode_id,
                                              arg_max(hp, step) AS hp,
                                              arg_max(max_hp, step) AS max_hp
                                       FROM ticks
                                       WHERE episode_id IN (SELECT episode_id FROM episodes
                                                            WHERE run_id LIKE ?)
                                       GROUP BY episode_id""", (rid + "%",)).fetchall()
                                hp_by_ep = {r[0]: (r[1], r[2]) for r in hp_rows}
                            except Exception:  # noqa: BLE001
                                pass
                        i_msg, i_depth, i_ep = tsel.index("death_message"), tsel.index("max_depth"), tsel.index("episode_id")
                        i_cat = tsel.index("death_category") if "death_category" in tsel else None
                        agg: dict[str, dict] = {}
                        for row in rows:
                            msg, depth, ep_id = row[i_msg], row[i_depth], row[i_ep]
                            cause = _normalize_cause(msg, row[i_cat] if i_cat is not None else None)
                            a = agg.setdefault(cause, {"cause": cause, "count": 0,
                                                       "depths": [], "hp_fracs": [],
                                                       "example": msg[:120]})
                            a["count"] += 1
                            a["depths"].append(depth)
                            if ep_id in hp_by_ep and hp_by_ep[ep_id][1]:
                                hp, mhp = hp_by_ep[ep_id]
                                a["hp_fracs"].append((hp or 0) / max(1, mhp or 1))
                        taxonomy = sorted(agg.values(), key=lambda a: -a["count"])
                        bundle["death_taxonomy"] = [
                            {"cause": a["cause"], "count": a["count"],
                             "median_depth_at_death": int(_median(a["depths"])),
                             "median_hp_frac_at_death": round(_median(a["hp_fracs"]), 3),
                             "example": a["example"]}
                            for a in taxonomy[:12]]
                    except Exception:  # noqa: BLE001
                        pass

                    # goal stats + stall census, SCOPED to this run's episodes
                    try:
                        if rid is None:
                            raise ValueError("no scoped run")
                        rows = con.execute(
                            """SELECT goal,
                                      SUM(CASE WHEN event='completed' THEN 1 ELSE 0 END),
                                      SUM(CASE WHEN event='skipped_budget' THEN 1 ELSE 0 END),
                                      SUM(CASE WHEN event='skipped_when' THEN 1 ELSE 0 END),
                                      MEDIAN(CASE WHEN event='completed' THEN steps_in_goal END),
                                      MEDIAN(CASE WHEN event IN ('skipped_budget','skipped_when')
                                                  THEN steps_in_goal END)
                               FROM goal_events
                               WHERE episode_id IN (SELECT episode_id FROM episodes
                                                    WHERE run_id LIKE ?)
                               GROUP BY goal ORDER BY goal""", (rid + "%",)).fetchall()
                        bundle["goal_stats"] = [
                            {"goal": r[0], "completions": int(r[1] or 0),
                             "skips_budget": int(r[2] or 0), "skips_when": int(r[3] or 0),
                             "median_steps": int(r[4] or 0)}
                            for r in rows]
                        bundle["stall_census"] = sorted(
                            [{"goal": r[0],
                              "stalls": int(r[2] or 0) + int(r[3] or 0),
                              "median_stall_turns": int(r[5] or 0)}
                             for r in rows if (r[2] or 0) + (r[3] or 0) > 0],
                            key=lambda s: -s["stalls"])[:8]
                    except Exception:  # noqa: BLE001
                        pass
            finally:
                con.close()

    # ---- S0: failure counters + condensed digests + RLE slices ---------------
    # Skip when a candidate id was requested but has no registered lineage: a
    # missing candidate must yield EMPTY evidence, never another candidate's batch.
    if bundle["scope"].get("source") != "candidate_missing":
        try:
            bundle["failure_counters"] = build_failure_counters_section(
                db_path, run_id=bundle["scope"].get("run_id"))
        except Exception:  # noqa: BLE001
            pass
        try:
            bundle["distribution"] = build_run_distribution(
                db_path, run_id=bundle["scope"].get("run_id"), domain=domain)
        except Exception:  # noqa: BLE001
            pass
        try:
            bundle["episode_digests"] = build_episode_digests(
                db_path, run_id=bundle["scope"].get("run_id"), n=digest_episodes, domain=domain)
        except Exception:  # noqa: BLE001
            pass
        try:
            bundle["trajectory_slices"] = build_trajectory_slices(
                db_path, run_id=bundle["scope"].get("run_id"), domain=domain)
        except Exception:  # noqa: BLE001
            pass

    # ---- Ledger: prev_revisions ----------------------------------------------
    ledger = RevisionLedger(ledger_path)
    prev = []
    for e in ledger.all()[-5:]:
        if e.get("type") == "policy_revision":
            prev.append({
                "version": e.get("revision"),
                "accepted": bool(e.get("accepted")),
                "delta_score": (e.get("delta") or {}).get("after"),
                "reason": e.get("reason", ""),
            })
    bundle["prev_revisions"] = prev
    return bundle


if __name__ == "__main__":  # pragma: no cover — debugging helper
    print(json.dumps(build_report_bundle(), indent=2)[:4000])