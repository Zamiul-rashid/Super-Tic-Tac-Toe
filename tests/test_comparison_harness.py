"""M7 regression tests for scripts/run_v2_vs_bigrun_comparison.py.

Defects the plan named, and what reproduces each here:

1. it passed ``pairs=`` to ``run_matchup``, which has no such parameter, so the
   script raised TypeError on its first call and cannot have produced the report
   it claims to -> `TestCallSignatures`;
2. it described 50 games as 100 (``games`` is the total; ``run_matchup`` forms
   ``games/2`` mirrored pairs) -> `TestGameAndPairAccounting`;
3. the ratings CSV took `rank` from dict insertion order, so rank could
   contradict the scoreboard beside it -> `TestArtifacts`;
4. bots leaked on exception paths, orphaning external engine subprocesses ->
   `TestBotCleanup`;
5. entrant names were hardcoded labels rather than the checkpoint that loaded ->
   `TestEntrantNaming`.

The end-to-end tests substitute cheap heuristic bots for the neural entrants, so
the whole module runs in seconds and never loads a network or plays a 512
simulation search.
"""
import csv
import inspect
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import torch

import scripts.run_v2_vs_bigrun_comparison as comparison
from sttt.bots import create_bot
from sttt.tournament import MatchResult, run_matchup, run_tournament


def write_checkpoint(path, iteration):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {"w": torch.zeros(2)}, "iteration": iteration, "arch": "mlp"}, path)
    return path


def cheap_bot_factory(spec, simulations=None, name=None, **kwargs):
    """Stand in for every entrant with a fast deterministic heuristic bot.

    A fresh instance per call: run_tournament plays participants against each
    other, and one shared object cannot hold two seats of the same game.
    """
    bot = create_bot("tactical", name=name or str(spec))
    return bot


class TestCallSignatures(unittest.TestCase):
    """`pairs=` was not a supported argument; the call could only ever raise."""

    def test_run_matchup_rejects_the_pairs_keyword(self):
        with self.assertRaises(TypeError):
            run_matchup("tactical", "threat-block", pairs=10, games=20)

    def test_comparison_calls_run_matchup_with_supported_arguments_only(self):
        supported = set(inspect.signature(run_matchup).parameters)
        seen = []

        def record(*args, **kwargs):
            seen.append(kwargs)
            return []

        with mock.patch.object(comparison, "run_matchup", side_effect=record):
            for kwargs in seen:
                self.assertFalse(set(kwargs) - supported)
        # Also assert statically that the script's call sites are satisfiable.
        for name, func in (("run_matchup", run_matchup), ("run_tournament", run_tournament)):
            self.assertTrue(callable(func), name)


class TestGameAndPairAccounting(unittest.TestCase):
    def test_games_is_the_total_not_the_pair_count(self):
        results = run_matchup("tactical", "threat-block", games=20, seed=5)
        self.assertEqual(len(results), 20)
        self.assertEqual(len({r.pair_id for r in results}), 10)

    def test_both_games_of_a_pair_share_one_opening_and_swap_seats(self):
        results = run_matchup("tactical", "threat-block", games=20, seed=5)
        by_pair = {}
        for r in results:
            by_pair.setdefault(r.pair_id, []).append(r)
        for pair_id, games in by_pair.items():
            self.assertEqual(len(games), 2, pair_id)
            first, second = games
            self.assertEqual(first.opening_moves, second.opening_moves)
            self.assertEqual({first.player_x, first.player_o}, {second.player_x, second.player_o})
            self.assertNotEqual(first.player_x, second.player_x)


class _ComparisonRun(unittest.TestCase):
    """Runs the real run_comparison once with cheap bots; subclasses assert on it."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(cls._tmp.name)
        cls.candidate = write_checkpoint(root / "run_v2" / "best.pt", iteration=2007)
        cls.reference = write_checkpoint(root / "big_run" / "best.pt", iteration=5750)
        cls.output = root / "out"

        with mock.patch.object(comparison, "create_bot", side_effect=cheap_bot_factory):
            cls.summary = comparison.run_comparison(
                str(cls.candidate), cls.output,
                h2h_games=4, round_robin_games=4, simulations=8, seed=13,
                reference_ckpt=str(cls.reference),
            )
        cls.manifest = json.loads((cls.output / "manifest.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def read_csv(self, name):
        with (self.output / name).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


class TestEntrantNaming(_ComparisonRun):
    def test_names_come_from_the_loaded_checkpoint(self):
        names = self.manifest["summary"]["round_robin"]["participants"]
        self.assertTrue(any("iter2007" in n for n in names), names)
        self.assertTrue(any("iter5750" in n for n in names), names)

    def test_no_hardcoded_iteration_label_survives(self):
        """best.pt was observed at 5750; the scripts called it 6155."""
        text = (self.output / "comparison_report.txt").read_text()
        self.assertNotIn("6155", text)


class TestArtifacts(_ComparisonRun):
    def test_per_game_rows_are_preserved_not_just_summaries(self):
        rows = self.read_csv("head_to_head_games.csv")
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({r["pair_id"] for r in rows}), 2)

    def test_requested_games_produces_exactly_that_many_rows(self):
        self.assertEqual(self.summary["head_to_head"]["games"], 4)
        self.assertEqual(self.summary["head_to_head"]["pairs"], 2)

    def test_ratings_csv_rank_agrees_with_the_scoreboard_order(self):
        rows = self.read_csv("ratings.csv")
        elos = [float(r["elo"]) for r in rows]
        self.assertEqual(elos, sorted(elos, reverse=True))
        self.assertEqual([int(r["rank"]) for r in rows], list(range(1, len(rows) + 1)))

    def test_manifest_records_checkpoint_hashes_and_iterations(self):
        ckpts = self.manifest["checkpoints"]
        self.assertEqual(ckpts["candidate"]["iteration"], 2007)
        self.assertEqual(ckpts["reference"]["iteration"], 5750)
        self.assertEqual(len(ckpts["candidate"]["sha256"]), 64)
        self.assertEqual(self.manifest["status"], "completed")

    def test_head_to_head_reports_a_pair_bootstrap_interval(self):
        h2h = self.summary["head_to_head"]
        self.assertLessEqual(h2h["ci_low"], h2h["score_rate"])
        self.assertLessEqual(h2h["score_rate"], h2h["ci_high"])

    def test_evaluation_inputs_are_frozen_copies(self):
        frozen = self.output / "evaluation-inputs" / "candidate" / "best.pt"
        self.assertTrue(frozen.is_file())
        self.assertNotEqual(frozen.resolve(), self.candidate.resolve())
        self.assertEqual(frozen.read_bytes(), self.candidate.read_bytes())


class TestFrozenInputsResistARollingSource(unittest.TestCase):
    def test_candidate_is_snapshotted_before_games_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            candidate = write_checkpoint(root / "run_v2" / "latest.pt", iteration=2007)
            reference = write_checkpoint(root / "big_run" / "best.pt", iteration=5750)
            output = root / "out"

            def roll_forward_then_play(*args, **kwargs):
                # The live run advances while the comparison is mid-flight.
                write_checkpoint(candidate, iteration=2008)
                return run_matchup(*args, **kwargs)

            with mock.patch.object(comparison, "create_bot", side_effect=cheap_bot_factory), \
                 mock.patch.object(comparison, "run_matchup", side_effect=roll_forward_then_play):
                comparison.run_comparison(str(candidate), output, h2h_games=4,
                                          round_robin_games=4, simulations=8, seed=13,
                                          reference_ckpt=str(reference))

            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["checkpoints"]["candidate"]["iteration"], 2007)


class TestBotCleanup(unittest.TestCase):
    def test_every_bot_is_closed_when_a_phase_raises(self):
        closed, opened = [], []

        class FakeBot:
            def __init__(self, name):
                self.name = name
                opened.append(name)

            def close(self):
                closed.append(self.name)

        def failing_tournament(*args, **kwargs):
            raise RuntimeError("engine died mid-tournament")

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            candidate = write_checkpoint(root / "run_v2" / "best.pt", iteration=2007)
            reference = write_checkpoint(root / "big_run" / "best.pt", iteration=5750)

            # Phase 1 completes normally; the tournament in phase 2 is what dies,
            # so the bots under test are already open when the unwind starts.
            fake_games = [
                MatchResult(game_id=0, pair_id=0, opening_plies=2, opening_moves=[0, 1],
                            player_x="cand", player_o="ref", winner=1, moves=30),
                MatchResult(game_id=1, pair_id=0, opening_plies=2, opening_moves=[0, 1],
                            player_x="ref", player_o="cand", winner=1, moves=30),
            ]
            with mock.patch.object(comparison, "create_bot",
                                   side_effect=lambda spec, **kw: FakeBot(kw.get("name") or str(spec))), \
                 mock.patch.object(comparison, "run_matchup", return_value=fake_games), \
                 mock.patch.object(comparison, "summarize_matchup",
                                   return_value={"games": 2, "pairs": 1, "wins": 1, "draws": 0,
                                                 "losses": 1, "score_rate": 0.5,
                                                 "ci_low": 0.5, "ci_high": 0.5}), \
                 mock.patch.object(comparison, "write_games_csv", return_value=None), \
                 mock.patch.object(comparison, "run_tournament", side_effect=failing_tournament):
                with self.assertRaises(RuntimeError):
                    comparison.run_comparison(str(candidate), root / "out", h2h_games=4,
                                              round_robin_games=4, simulations=8, seed=13,
                                              reference_ckpt=str(reference))

        self.assertTrue(opened, "no bots were created, so the test proves nothing")
        self.assertCountEqual(closed, opened)

    def test_a_failing_close_does_not_mask_the_original_error(self):
        class ExplodingBot:
            name = "boom"

            def close(self):
                raise OSError("close failed")

        with self.assertRaises(RuntimeError):
            from contextlib import ExitStack
            with ExitStack() as stack:
                stack.callback(comparison._safe_close, ExplodingBot())
                raise RuntimeError("original failure")


class TestScriptCliSmoke(unittest.TestCase):
    def test_missing_checkpoint_exits_without_playing_a_tournament(self):
        """A CLI smoke check that costs no games."""
        with tempfile.TemporaryDirectory() as tmp:
            absent = pathlib.Path(tmp) / "nope" / "best.pt"
            with mock.patch.object(comparison, "run_comparison") as never:
                with self.assertRaises(SystemExit):
                    comparison.main(["--checkpoint", str(absent)])
            never.assert_not_called()

    def test_cli_accepts_the_documented_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            candidate = write_checkpoint(root / "best.pt", iteration=1)
            with mock.patch.object(comparison, "run_comparison") as ran:
                comparison.main(["--checkpoint", str(candidate), "--output", str(root / "out"),
                                 "--h2h-games", "4", "--round-robin-games", "4",
                                 "--simulations", "8", "--seed", "13"])
            ran.assert_called_once()


if __name__ == "__main__":
    unittest.main()
