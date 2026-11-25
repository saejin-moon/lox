"""
Contract tests verifying that:
1. Every `VisibleObject` / `GameState` attribute documented in `prompts.py` exists on real instances (`H-1`).
2. Every SQL template in `prompts.py` and `swarm_peer.py` executes cleanly against a live `init_db()` DuckDB instance (`M-1`).
"""

from __future__ import annotations

import re

from lox.author.prompts import CORE_HARNESS_DOCS, build_system_prompt
from lox.db.schema import init_db
from lox.env.harness import NetHackHarness
from lox.env.types import ObjectInfo, VisibleObject


def test_prompts_visible_object_docs_match_real_api():
    """Verifies H-1: prompts document real VisibleObject/ObjectInfo attrs and allowed modules."""
    assert "NOT `obj.is_food`" not in CORE_HARNESS_DOCS
    assert "obj.info.is_food" not in CORE_HARNESS_DOCS
    assert "obj.info.nutrition" not in CORE_HARNESS_DOCS
    assert "lox.env.state" in CORE_HARNESS_DOCS

    oinfo = ObjectInfo(
        obj_id=1,
        name="food ration",
        descr="food ration",
        oclass=5,
        oclass_name="food",
        char="%",
        weight=20,
        cost=45,
        delay=1,
        color=6,
        prob=100,
    )
    for info_attr in ("oclass", "oclass_name", "cost", "weight", "delay"):
        assert hasattr(oinfo, info_attr), (
            f"ObjectInfo missing documented attr `{info_attr}`"
        )
    assert not hasattr(oinfo, "nutrition")

    vobj = VisibleObject(
        y=5,
        x=10,
        glyph=0,
        char="%",
        color=7,
        name="food ration",
        is_corpse=False,
        corpse_monster=None,
        info=oinfo,
    )
    for attr in (
        "y",
        "x",
        "name",
        "char",
        "color",
        "glyph",
        "is_boulder",
        "is_corpse",
        "is_food",
        "is_weapon",
        "is_armor",
        "is_potion",
        "is_scroll",
        "is_wand",
        "is_ring",
        "is_amulet",
        "is_tool",
        "corpse_monster",
        "info",
    ):
        assert hasattr(vobj, attr), f"VisibleObject missing documented attr `{attr}`"

    harness = NetHackHarness(max_steps=10)
    try:
        st = harness.reset(seed=42)
        for attr in (
            "hp",
            "max_hp",
            "ac",
            "strength",
            "strength_pct",
            "in_yn_prompt",
            "in_getobj_prompt",
            "in_getdir_prompt",
            "in_text_prompt",
            "in_menu",
            "trap_mask",
            "unexplored_frontier_mask",
            "visible_objects",
            "visible_monsters",
            "adjacent_monsters",
            "bfs_distances",
            "path_step",
            "mem",
            "kv",
        ):
            assert hasattr(st, attr), f"GameState missing documented attr `{attr}`"
    finally:
        harness.close()


def test_prompts_and_peer_sql_templates_execute_against_live_schema(tmp_path):
    """Verifies M-1: every SQL query template in prompts.py & swarm_peer.py executes on live DuckDB."""
    db_path = str(tmp_path / "contract_test.duckdb")
    con = init_db(db_path)
    try:
        sys_prompt = build_system_prompt(db_path=db_path, is_swarm=True, is_peer=True)
        # Ensure nonexistent `visible_monsters` is NOT suggested on `expert_ticks`
        assert "expert_ticks" in sys_prompt
        assert "visible_monsters LIKE" not in sys_prompt

        sql_blocks = re.findall(r"`(SELECT\s+[^`]+;)`", sys_prompt, flags=re.IGNORECASE)
        assert len(sql_blocks) >= 5, (
            f"Expected >=5 SQL templates, found {len(sql_blocks)}"
        )
        for raw_sql in sql_blocks:
            concrete_sql = (
                raw_sql.replace("<D>", "1")
                .replace("<G>", "1")
                .replace("<N>", "1")
                .replace("<S>", "42")
                .replace("<your_peer_id>", "reactive_1")
                .replace("<name>", "jackal")
                .replace("<char>", "d")
            )
            cur = con.execute(concrete_sql)
            cur.fetchall()

        # Also verify the situational expert_ticks filter advertised in prompts & swarm_peer.py
        con.execute(
            "SELECT action_name, COUNT(*) AS n FROM expert_ticks "
            "WHERE depth = 1 AND (message LIKE '%jackal%' OR screen_ascii LIKE '%d%') "
            "GROUP BY action_name LIMIT 5"
        ).fetchall()
    finally:
        con.close()
