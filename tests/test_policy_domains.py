"""R5 tests: DomainAdapter boundary, MiniHack adapter, corpus, transfer revision loop."""
import json
import os
from argparse import Namespace

import pytest

from lox.executor import get_adapter, available_domains
from lox.executor.interface import DomainSpec
from lox.policy.corpus import build_corpus, rag_slices, corpus_coverage
from lox.policy.manifest import build_manifest


class TestRegistry:
    def test_nethack_adapter_registered(self):
        ad = get_adapter("nethack")
        assert isinstance(ad.spec, DomainSpec) and ad.spec.name == "nethack"
        assert not ad.spec.seeded
        assert len(ad.predicates()) > 20
        assert "descend" in ad.goal_handlers()
        assert ad.param_leaves()["survival.rest_below_frac"].max == 0.9

    def test_unknown_domain_raises(self):
        with pytest.raises(KeyError):
            get_adapter("does_not_exist")

    def test_manifest_generated_from_adapter(self):
        ad = get_adapter("nethack")
        m = ad.manifest(5)
        assert m.domain == "nethack" and m.version == 5
        assert "survival.rest_below_frac" in m.param_leaves


class TestMiniHackAdapter:
    def test_spec_and_vocab(self):
        ad = get_adapter("minihack")
        assert ad.spec.seeded
        m = ad.manifest(1)
        assert m.domain == "minihack"
        assert set(m.goals) == {"explore_floor", "reach_stairs"}
        assert "explore.stuck_patience" in m.param_leaves
        # minihack verbs are a legal subset (retreat/avoid are real combat-manager verbs)
        from lox.policy.tactics import TACTIC_VERBS
        assert set(m.verbs) <= TACTIC_VERBS

    def test_cold_start_program_valid(self):
        ad = get_adapter("minihack")
        prog = ad.default_program()
        assert prog["domain"] == "minihack"
        assert prog["strategy_plan"][0]["goal"] == "reach_stairs"
        from lox.policy.program import PolicyProgram
        p = PolicyProgram.from_dict(prog)
        assert [g.goal for g in p.strategy_plan] == ["reach_stairs", "explore_floor"]

    def test_certification_mapped_easy(self):
        """Cert case 1: mapped Easy maze — pure A* must reach the stairs."""
        ad = get_adapter("minihack")
        case = ad.certification_suite()[0]
        env = ad.make_env(seed=case.seed, task="MiniHack-ExploreMaze-Easy-Mapped-v0")
        r = ad.run_episode(env, ad.default_program(), seed=case.seed,
                           max_steps=case.max_steps)
        env.close()
        assert case.check(r), r

    def test_certification_unmapped_progress(self):
        """Cert case 2: unmapped Hard maze — frontier exploration makes progress."""
        ad = get_adapter("minihack")
        case = ad.certification_suite()[1]
        env = ad.make_env(seed=case.seed, task="MiniHack-ExploreMaze-Hard-v0")
        r = ad.run_episode(env, ad.default_program(), seed=case.seed,
                           max_steps=case.max_steps)
        env.close()
        assert case.check(r), {k: r[k] for k in ("steps", "success", "coverage")}

    def test_params_overlay_changes_behavior(self):
        ad = get_adapter("minihack")
        prog = ad.default_program()
        prog["params"] = {"explore.stuck_patience": 55}
        from lox.executor.minihack_adapter import MiniHackAgent, MiniHackConfig
        agent = MiniHackAgent(None, prog, max_steps=10)
        assert agent.cfg.explore.stuck_patience == 55


class TestCorpus:
    def test_build_and_query(self, tmp_path):
        import lox.policy.corpus as corpus_mod
        old_root = corpus_mod.CORPUS_ROOT
        corpus_mod.CORPUS_ROOT = str(tmp_path)
        try:
            n = build_corpus("testdom", [
                ("doc1", "Elbereth protects the player from melee attacks by most monsters."),
                ("doc2", "Poison resistance prevents instakill from giant spider bites."),
            ])
            assert n >= 2
            hits = rag_slices("testdom", "giant spider", k=2)
            assert any("spider" in h for h in hits)
            cov = corpus_coverage("testdom")
            assert cov["documents"] == 2 and cov["probe_query_nonempty"]
            assert rag_slices("testdom", "", k=2) == []  # empty query → RAG off
        finally:
            corpus_mod.CORPUS_ROOT = old_root

    def test_real_corpora_exist(self):
        for domain in ("nethack", "minihack"):
            cov = corpus_coverage(domain)
            assert cov["exists"], f"{domain} corpus missing — run build_corpus"
            assert cov["chunks"] > 20
            assert cov["probe_query_nonempty"]


class TestTransferLoop:
    @pytest.mark.asyncio
    async def test_minihack_mock_revision(self, tmp_path):
        """The revision loop runs on a transfer domain: manifest from the adapter,
        domain-legal diffs gated, seeded episode batches as the report."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_revision_loop", os.path.join(os.path.dirname(__file__), "..", "scripts", "run_revision_loop.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        program_path = str(tmp_path / "policy_program_minihack.json")
        ledger_path = str(tmp_path / "ledger.jsonl")
        ns = Namespace(max_revisions=1, provider="mock", model=None,
                       program_path=program_path, ledger_path=ledger_path,
                       db_path="/nonexistent", domain="minihack", report_episodes=2,
                       live_gates=False, fixture_shadow=True,
                       quick_batch_episodes=3, quick_batch_steps=5000, no_git=True)
        rc = await mod.run_loop(ns)
        assert rc == 0
        final = json.load(open(program_path))
        assert final["domain"] == "minihack"
        # NLE MiniHack resets are NOT seed-deterministic (same seed yields
        # different episode step counts across runs — measured), and the mock's
        # domain template is mildly step-regressive, so the live transfer gate
        # (median_steps_on_success > 1.2× baseline) correctly rejects it on
        # some draws. Both gate-consistent outcomes prove the loop machinery:
        if final["version"] == 2:
            # the mock's domain template tuned a minihack param leaf
            assert final["params"].get("explore.stuck_patience") == 40
        else:
            assert final["version"] == 1  # gate rejection: program untouched


class TestCraftaxAdapter:
    def test_registered_and_manifest(self):
        from lox.executor import get_adapter
        ad = get_adapter("craftax")
        assert ad.spec.name == "craftax"
        assert ad.spec.grid_shape == (63, 63)
        assert ad.spec.n_actions == 43
        assert len(ad.predicates()) >= 10
        assert len(ad.goal_handlers()) >= 5
        manifest = ad.manifest(version=1)
        assert manifest.domain == "craftax"
        assert "collect_wood" in manifest.goals
        assert "has_wood" in manifest.predicates

    def test_default_program(self):
        from lox.executor import get_adapter
        ad = get_adapter("craftax")
        prog = ad.default_program()
        assert prog["domain"] == "craftax"
        assert any(g["goal"] == "collect_wood" for g in prog["strategy_plan"])