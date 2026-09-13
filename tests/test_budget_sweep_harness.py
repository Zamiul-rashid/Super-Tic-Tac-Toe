"""M7 regression tests for scripts/run_thorough_tournament.py.

The defect that motivates the whole module: the search-budget sweep seeded each
budget with ``seed=100 + sims``, so 512 played openings from seed 612 and 2000
played openings from seed 2100. The two arms never faced the same positions, yet
the difference between them was read as the effect of search budget. A
512-vs-2000 score gap measured that way is confounded with the opening corpus.

Also covered: entrant names derived from the checkpoint that loaded (the script
labelled everything "ckpt-iter6155" regardless, and `best.pt` was observed at
5750), rank ordering in the ratings CSV, provenance, and bot cleanup.

Cheap heuristic bots stand in for the neural entrants throughout; nothing here
runs a real 2000-simulation search.
"""
import csv
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import torch

import scripts.run_thorough_tournament as thorough
from sttt.bots import create_bot


def write_checkpoint(path, iteration):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {"w": torch.zeros(2)}, "iteration": iteration, "arch": "mlp"}, path)
    return path


def cheap_bot_factory(spec, **kwargs):
    return create_bot("tactical", name=kwargs.get("name") or str(spec))


class TestSharedOpeningCorpus(unittest.TestCase):
    """Every search budget must face the identical held-out openings."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.ckpt = write_checkpoint(self.root / "big_run" / "latest.pt", iteration=6155)

    def _sweep(self, budgets=(512, 1024, 2000)):
        seeds_seen = []
        real_matchup = thorough.run_matchup

        def record(*args, **kwargs):
            seeds_seen.append(kwargs.get("seed"))
            return real_matchup(*args, **kwargs)

        with mock.patch.object(thorough, "create_bot", side_effect=cheap_bot_factory), \
             mock.patch.object(thorough, "run_matchup", side_effect=record):
            result = thorough.run_budget_sweep(
                str(self.ckpt), self.root / "out", budgets=list(budgets),
                games=4, seed=4242, opponent_depth=2, opponent_nodes=10000)
        return result, seeds_seen

    def test_every_budget_uses_the_same_opening_seed(self):
        """Directly reproduces `seed=100 + sims`, which gave 512 seed 612 and
        2000 seed 2100."""
        _, seeds = self._sweep()
        self.assertEqual(len(seeds), 3)
        self.assertEqual(set(seeds), {4242}, f"budgets used different opening seeds: {seeds}")

    def test_arms_are_played_from_identical_opening_positions(self):
        result, _ = self._sweep()
        arms = result["budgets"]
        openings = {}
        for budget, arm in arms.items():
            openings[budget] = {row["pair_id"]: row["opening_moves"] for row in arm["game_rows"]}
        reference = openings[next(iter(openings))]
        for budget, corpus in openings.items():
            self.assertEqual(corpus, reference, f"budget {budget} faced different openings")

    def test_budget_differences_are_reported_with_pair_bootstrap_intervals(self):
        result, _ = self._sweep()
        comparisons = result["comparisons"]
        self.assertTrue(comparisons)
        for entry in comparisons.values():
            self.assertIn("difference", entry)
            self.assertLessEqual(entry["ci_low"], entry["difference"])
            self.assertLessEqual(entry["difference"], entry["ci_high"])
            self.assertGreater(entry["pairs"], 0)

    def test_measured_latency_is_recorded_per_budget(self):
        """A budget sweep may report latency; it may not claim an equal-time test."""
        result, _ = self._sweep()
        for arm in result["budgets"].values():
            self.assertIn("elapsed_seconds", arm)
            self.assertIn("seconds_per_game", arm)
            self.assertGreaterEqual(arm["elapsed_seconds"], 0.0)


class TestProvenance(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def test_names_derive_from_the_loaded_checkpoint_not_a_label(self):
        ckpt = write_checkpoint(self.root / "big_run" / "best.pt", iteration=5750)
        with mock.patch.object(thorough, "create_bot", side_effect=cheap_bot_factory):
            result = thorough.run_budget_sweep(str(ckpt), self.root / "out", budgets=[512],
                                               games=4, seed=7, opponent_depth=2,
                                               opponent_nodes=10000)
        name = result["budgets"][512]["entrant"]
        self.assertIn("iter5750", name)
        self.assertNotIn("6155", name)

    def test_manifest_records_the_settings_a_replay_needs(self):
        ckpt = write_checkpoint(self.root / "big_run" / "latest.pt", iteration=6155)
        out = self.root / "out"
        with mock.patch.object(thorough, "create_bot", side_effect=cheap_bot_factory):
            thorough.run_budget_sweep(str(ckpt), out, budgets=[512], games=4, seed=7,
                                      opponent_depth=4, opponent_nodes=25000,
                                      backend="python", device="cpu", leaf_batch=8)
        manifest = json.loads((out / "manifest.json").read_text())
        config = manifest["config"]
        for key in ("budgets", "backend", "device", "leaf_batch",
                    "opponent_depth", "opponent_nodes", "opening_seed"):
            self.assertIn(key, config)
        self.assertEqual(config["opponent_depth"], 4)
        self.assertEqual(config["opponent_nodes"], 25000)
        self.assertEqual(manifest["checkpoints"]["candidate"]["iteration"], 6155)
        self.assertEqual(len(manifest["checkpoints"]["candidate"]["sha256"]), 64)
        self.assertIn("native", manifest["environment"])

    def test_checkpoint_is_frozen_before_play(self):
        ckpt = write_checkpoint(self.root / "big_run" / "latest.pt", iteration=6155)
        out = self.root / "out"
        with mock.patch.object(thorough, "create_bot", side_effect=cheap_bot_factory):
            thorough.run_budget_sweep(str(ckpt), out, budgets=[512], games=4, seed=7,
                                      opponent_depth=2, opponent_nodes=10000)
        frozen = out / "evaluation-inputs" / "candidate" / "latest.pt"
        self.assertTrue(frozen.is_file())
        self.assertEqual(frozen.read_bytes(), ckpt.read_bytes())

    def test_per_game_rows_are_exported_for_every_budget(self):
        ckpt = write_checkpoint(self.root / "big_run" / "latest.pt", iteration=6155)
        out = self.root / "out"
        with mock.patch.object(thorough, "create_bot", side_effect=cheap_bot_factory):
            thorough.run_budget_sweep(str(ckpt), out, budgets=[512, 1024], games=4, seed=7,
                                      opponent_depth=2, opponent_nodes=10000)
        for budget in (512, 1024):
            path = out / f"budget_{budget}_games.csv"
            self.assertTrue(path.is_file(), path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)


class TestChampionshipCleanup(unittest.TestCase):
    def test_bots_close_when_the_tournament_raises(self):
        closed, opened = [], []

        class FakeBot:
            def __init__(self, name):
                self.name = name
                opened.append(name)

            def close(self):
                closed.append(self.name)

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ckpt = write_checkpoint(root / "big_run" / "latest.pt", iteration=6155)
            with mock.patch.object(thorough, "create_bot",
                                   side_effect=lambda spec, **kw: FakeBot(kw.get("name") or str(spec))), \
                 mock.patch.object(thorough, "run_tournament",
                                   side_effect=RuntimeError("engine died")):
                with self.assertRaises(RuntimeError):
                    thorough.run_grand_championship(str(ckpt), root / "out", games_per_matchup=4)

        self.assertTrue(opened, "no bots created; the test proves nothing")
        self.assertCountEqual(closed, opened)


class TestCliSmoke(unittest.TestCase):
    def test_missing_checkpoint_exits_without_playing(self):
        with tempfile.TemporaryDirectory() as tmp:
            absent = pathlib.Path(tmp) / "absent.pt"
            with mock.patch.object(thorough, "run_budget_sweep") as never_sweep, \
                 mock.patch.object(thorough, "run_grand_championship") as never_champ:
                with self.assertRaises(SystemExit):
                    thorough.main(["--checkpoint", str(absent), "--output", tmp])
            never_sweep.assert_not_called()
            never_champ.assert_not_called()

    def test_budgets_are_configurable_from_the_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ckpt = write_checkpoint(root / "latest.pt", iteration=1)
            with mock.patch.object(thorough, "run_budget_sweep") as sweep, \
                 mock.patch.object(thorough, "run_grand_championship"):
                thorough.main(["--checkpoint", str(ckpt), "--output", str(root / "out"),
                               "--budgets", "512", "1024", "--games", "4",
                               "--skip-championship"])
            sweep.assert_called_once()
            self.assertEqual(sweep.call_args.kwargs["budgets"], [512, 1024])


if __name__ == "__main__":
    unittest.main()
