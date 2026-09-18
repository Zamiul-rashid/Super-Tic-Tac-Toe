"""Tests for the matched-opening search-budget comparison.

Covers the primitives in `sttt.tournament` / `sttt.evaluation` (opening-corpus
save/load round trip, shared openings across arms, rotated budget order,
per-move latency attribution, move disagreement, reanalyse-record conversion)
and the `scripts/evaluation_suite.py` harness that wires them together
(`run_budget_comparison`), using cheap heuristic bots throughout so the whole
module runs in seconds -- nothing here loads a network or plays a real search.
"""
import csv
import json
import pathlib
import tempfile
import unittest
from unittest import mock

import torch

from sttt.bots import StyleBot, TacticalBot, create_bot
from sttt.evaluation import (
    candidate_move_seconds,
    game_record_for_reanalysis,
    move_disagreement,
)
from sttt.reanalysis import load_records, replay_states, select_tasks, write_jsonl
from sttt.tournament import (
    MatchResult,
    generate_opening_corpus,
    load_opening_corpus,
    run_matched_budget_comparison,
    save_opening_corpus,
)

import scripts.evaluation_suite as suite


def write_checkpoint(path, iteration):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {"w": torch.zeros(2)}, "iteration": iteration, "arch": "mlp"}, path)
    return path


def cheap_bot_factory(spec, **kwargs):
    return create_bot("tactical", name=kwargs.get("name") or str(spec))


class _LoggingBot(StyleBot):
    """StyleBot that logs every `reset()` call under `key`, to observe play order."""

    def __init__(self, key, log, **kwargs):
        super().__init__(**kwargs)
        self._key = key
        self._log = log

    def reset(self):
        self._log.append(self._key)
        super().reset()


class TestOpeningCorpusRoundTrip(unittest.TestCase):
    def test_saved_corpus_reloads_identically(self):
        corpus = generate_opening_corpus(seed=42, num_pairs=10, plies=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "openings.json"
            save_opening_corpus(path, corpus, seed=42, plies=2)
            loaded, meta = load_opening_corpus(path)
        self.assertEqual(loaded, corpus)
        self.assertEqual(meta, {"seed": 42, "opening_plies": 2})

    def test_generation_is_deterministic_from_the_seed(self):
        a = generate_opening_corpus(seed=7, num_pairs=5, plies=2)
        b = generate_opening_corpus(seed=7, num_pairs=5, plies=2)
        self.assertEqual(a, b)
        # And matches the pairing scheme run_matchup itself uses.
        from sttt.tournament import paired_opening
        for entry in a:
            _, moves = paired_opening(seed=7, pair=entry["pair_id"], plies=2)
            self.assertEqual(entry["opening_moves"], moves)


class TestMatchedBudgetComparisonPlay(unittest.TestCase):
    def test_every_arm_faces_the_identical_saved_openings(self):
        corpus = generate_opening_corpus(seed=5, num_pairs=6, plies=2)
        candidate_bots = {8: StyleBot(style="random", name="cand-8"),
                          32: StyleBot(style="corners", name="cand-32")}
        opponent = TacticalBot(name="opp")
        results = run_matched_budget_comparison(candidate_bots, opponent, corpus, seed=11)

        corpus_by_pair = {c["pair_id"]: tuple(c["opening_moves"]) for c in corpus}
        for budget, games in results.items():
            openings = {r.pair_id: tuple(r.opening_moves) for r in games}
            self.assertEqual(openings, corpus_by_pair, f"budget {budget} saw different openings")

    def test_every_pair_is_played_both_colours_per_arm(self):
        corpus = generate_opening_corpus(seed=1, num_pairs=4, plies=2)
        candidate_bots = {16: StyleBot(style="random", name="cand-16")}
        opponent = TacticalBot(name="opp")
        results = run_matched_budget_comparison(candidate_bots, opponent, corpus, seed=2)
        games = results[16]
        self.assertEqual(len(games), 8)  # 4 pairs x 2 colours
        by_pair = {}
        for r in games:
            by_pair.setdefault(r.pair_id, []).append(r)
        for pair_id, pair_games in by_pair.items():
            self.assertEqual(len(pair_games), 2, pair_id)
            colours = {pair_games[0].player_x, pair_games[0].player_o}
            self.assertEqual(colours, {"cand-16", "opp"})
            self.assertNotEqual(pair_games[0].player_x, pair_games[1].player_x)

    def test_budget_order_rotates_across_opening_pairs(self):
        corpus = generate_opening_corpus(seed=3, num_pairs=4, plies=2)
        log = []
        candidate_bots = {
            1: _LoggingBot(1, log, style="random", name="cand-1"),
            2: _LoggingBot(2, log, style="center", name="cand-2"),
            3: _LoggingBot(3, log, style="corners", name="cand-3"),
        }
        opponent = TacticalBot(name="opp")
        run_matched_budget_comparison(candidate_bots, opponent, corpus, seed=1)
        # Each key's reset() fires twice per pair (X-game, then O-game),
        # consecutively, so every 6 log entries is one pair's block of 3 keys.
        self.assertEqual(len(log), 4 * 3 * 2)
        first_key_per_pair = [log[i * 6] for i in range(4)]
        self.assertEqual(first_key_per_pair, [1, 2, 3, 1])

    def test_save_moves_records_full_sequence_and_latency(self):
        corpus = generate_opening_corpus(seed=9, num_pairs=2, plies=2)
        candidate_bots = {4: StyleBot(style="random", name="cand-4")}
        opponent = StyleBot(style="corners", name="opp")
        results = run_matched_budget_comparison(candidate_bots, opponent, corpus, seed=21, save_moves=True)
        for r in results[4]:
            self.assertIsNotNone(r.moves_played)
            self.assertIsNotNone(r.move_latencies)
            self.assertEqual(r.moves_played[:2], r.opening_moves)
            self.assertEqual(len(r.moves_played), r.moves)
            self.assertEqual(len(r.move_latencies), r.moves - 2)
            self.assertTrue(all(lat >= 0.0 for lat in r.move_latencies))

    def test_save_moves_false_leaves_fields_none(self):
        corpus = generate_opening_corpus(seed=9, num_pairs=1, plies=2)
        candidate_bots = {4: StyleBot(style="random", name="cand-4")}
        opponent = StyleBot(style="corners", name="opp")
        results = run_matched_budget_comparison(candidate_bots, opponent, corpus, seed=21, save_moves=False)
        for r in results[4]:
            self.assertIsNone(r.moves_played)
            self.assertIsNone(r.move_latencies)

    def test_reproducible_given_the_same_seed(self):
        corpus = generate_opening_corpus(seed=9, num_pairs=3, plies=2)
        candidate_bots = lambda: {4: StyleBot(style="random", name="cand-4")}
        opponent = lambda: StyleBot(style="corners", name="opp")
        first = run_matched_budget_comparison(candidate_bots(), opponent(), corpus, seed=21)
        second = run_matched_budget_comparison(candidate_bots(), opponent(), corpus, seed=21)
        self.assertEqual([r.winner for r in first[4]], [r.winner for r in second[4]])


class TestCandidateMoveSeconds(unittest.TestCase):
    def _result(self, player_x, player_o, latencies):
        return MatchResult(game_id=0, pair_id=0, opening_plies=2, opening_moves=[10, 20],
                           player_x=player_x, player_o=player_o, winner=1, moves=2 + len(latencies),
                           moves_played=[10, 20] + list(range(len(latencies))), move_latencies=latencies)

    def test_filters_to_the_candidates_own_plies_when_x(self):
        result = self._result("cand", "opp", [0.1, 0.2, 0.3, 0.4])
        # opening_plies=2: ply 2 (even) is X=cand, ply 3 (odd) is O=opp, ...
        self.assertEqual(candidate_move_seconds(result, "cand", 2), [0.1, 0.3])

    def test_filters_to_the_candidates_own_plies_when_o(self):
        result = self._result("opp", "cand", [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(candidate_move_seconds(result, "cand", 2), [0.2, 0.4])

    def test_raises_without_save_moves(self):
        result = MatchResult(game_id=0, pair_id=0, opening_plies=2, opening_moves=[0, 1],
                             player_x="cand", player_o="opp", winner=1, moves=10)
        with self.assertRaises(ValueError):
            candidate_move_seconds(result, "cand", 2)


class TestMoveDisagreement(unittest.TestCase):
    def _result(self, game_id, pair_id, moves):
        return MatchResult(game_id=game_id, pair_id=pair_id, opening_plies=2, opening_moves=moves[:2],
                           player_x="a", player_o="b", winner=0, moves=len(moves), moves_played=moves,
                           move_latencies=[0.0] * (len(moves) - 2))

    def test_identical_sequences_have_zero_disagreement(self):
        a = [self._result(0, 0, [1, 2, 3, 4, 5])]
        b = [self._result(0, 0, [1, 2, 3, 4, 5])]
        stats = move_disagreement(a, b, opening_plies=2)
        self.assertEqual(stats["disagreed_positions"], 0)
        self.assertEqual(stats["compared_positions"], 3)
        self.assertEqual(stats["disagreement_rate"], 0.0)

    def test_stops_counting_at_first_divergence(self):
        a = [self._result(0, 0, [1, 2, 3, 4, 5, 6])]
        b = [self._result(0, 0, [1, 2, 3, 9, 9, 9])]  # diverges at ply 3 (index into the sequence)
        stats = move_disagreement(a, b, opening_plies=2)
        # ply 2 (index 2) matches (3==3), ply 3 (index 3) diverges (4 != 9) and stops there.
        self.assertEqual(stats["compared_positions"], 2)
        self.assertEqual(stats["disagreed_positions"], 1)
        self.assertAlmostEqual(stats["disagreement_rate"], 0.5)

    def test_only_games_present_in_both_arms_are_compared(self):
        a = [self._result(0, 0, [1, 2, 3, 4]), self._result(2, 1, [1, 2, 3, 4])]
        b = [self._result(0, 0, [1, 2, 3, 4])]
        stats = move_disagreement(a, b, opening_plies=2)
        self.assertEqual(stats["matched_games"], 1)


class TestReanalysisRecordFormat(unittest.TestCase):
    def test_recorded_moves_replay_to_the_recorded_result(self):
        corpus = generate_opening_corpus(seed=9, num_pairs=3, plies=2)
        candidate_bots = {4: StyleBot(style="random", name="cand-4")}
        opponent = StyleBot(style="corners", name="opp")
        results = run_matched_budget_comparison(candidate_bots, opponent, corpus, seed=21, save_moves=True)

        records = [game_record_for_reanalysis(r, "cand-4", opening_plies=2, record_id=f"g{i}")
                  for i, r in enumerate(results[4])]
        self.assertTrue(records)
        for result, record in zip(results[4], records):
            final = replay_states(record["actions"])[-1]
            self.assertEqual(final.result, record["result"])
            self.assertEqual(len(record["actions"]), result.moves)
            self.assertEqual(len(record["movers"]), len(record["actions"]))
            self.assertEqual(record["movers"][:2], ["opening", "opening"])
            self.assertIn("learner", record["movers"])  # candidate played at least one ply
            self.assertIn("opponent", record["movers"])

    def test_records_round_trip_through_load_records_and_select_tasks(self):
        corpus = generate_opening_corpus(seed=4, num_pairs=2, plies=2)
        candidate_bots = {4: StyleBot(style="random", name="cand-4")}
        opponent = StyleBot(style="corners", name="opp")
        results = run_matched_budget_comparison(candidate_bots, opponent, corpus, seed=6, save_moves=True)
        records = [game_record_for_reanalysis(r, "cand-4", opening_plies=2, record_id=f"g{i}")
                  for i, r in enumerate(results[4])]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "games.jsonl"
            write_jsonl(path, records)
            loaded = load_records(path)
        self.assertEqual(loaded, records)
        tasks = select_tasks(loaded, sides=("learner",))
        self.assertTrue(tasks)
        for task in tasks:
            self.assertTrue(task["plies"])


class TestBudgetComparisonHarness(unittest.TestCase):
    """scripts/evaluation_suite.py::run_budget_comparison, with cheap heuristic bots."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.ckpt = write_checkpoint(self.root / "run" / "latest.pt", iteration=4530)

    def _run(self, **kwargs):
        out = self.root / "out"
        with mock.patch.object(suite, "create_bot", side_effect=cheap_bot_factory):
            summary = suite.run_budget_comparison(str(self.ckpt), out, budgets=[4, 8, 16],
                                                   opponent="tactical", pairs=3, seed=13, **kwargs)
        return out, summary

    def test_parallel_workers_reproduce_the_sequential_games(self):
        # Real (tiny) checkpoint bots: spawned workers can't see mock patches.
        from sttt.learning import Network
        torch.manual_seed(0)
        self.ckpt = self.root / "real" / "latest.pt"
        self.ckpt.parent.mkdir(parents=True)
        torch.save({"model": Network().state_dict(), "iteration": 1, "arch": "mlp"}, self.ckpt)
        runs = {}
        for workers in (1, 2):
            out = self.root / f"out-w{workers}"
            suite.run_budget_comparison(str(self.ckpt), out, budgets=[2, 4], opponent="tactical",
                                        pairs=2, seed=13, backend="python", workers=workers)
            runs[workers] = [{k: v for k, v in r.items() if k != "move_latencies"}
                             for r in load_records(out / "games-budget-compare.jsonl")]
        self.assertEqual(len(runs[1]), 2 * 2 * 2)
        self.assertEqual(runs[1], runs[2])

    def test_opponent_is_required(self):
        with self.assertRaises(ValueError):
            suite.run_budget_comparison(str(self.ckpt), self.root / "out", budgets=[4, 8], opponent=None)

    def test_openings_file_is_saved_and_matches_pairs_requested(self):
        out, _ = self._run()
        openings_path = out / "openings.json"
        self.assertTrue(openings_path.is_file())
        corpus, meta = load_opening_corpus(openings_path)
        self.assertEqual(len(corpus), 3)
        self.assertEqual(meta["opening_plies"], 2)

    def test_summary_json_and_markdown_are_written(self):
        out, summary = self._run()
        self.assertTrue((out / "summary.json").is_file())
        self.assertTrue((out / "summary.md").is_file())
        on_disk = json.loads((out / "summary.json").read_text())
        self.assertEqual(on_disk["opponent"], "tactical")
        self.assertEqual(on_disk["total_games"], 3 * 3 * 2)  # 3 budgets x 3 pairs x 2 colours
        md = (out / "summary.md").read_text()
        self.assertIn("Matched-opening search budget comparison", md)

    def test_primary_comparison_is_the_two_largest_budgets(self):
        _, summary = self._run()
        comparisons = summary["comparisons"]
        self.assertIn("16-8", comparisons)
        self.assertTrue(comparisons["16-8"]["primary"])
        self.assertFalse(comparisons["8-4"]["primary"])
        self.assertFalse(comparisons["16-4"]["primary"])
        self.assertEqual(sum(c["primary"] for c in comparisons.values()), 1)

    def test_move_sequences_are_saved_and_reanalyse_readable(self):
        out, summary = self._run()
        records_path = out / "games-budget-compare.jsonl"
        self.assertTrue(records_path.is_file())
        records = load_records(records_path)
        self.assertEqual(len(records), 3 * 3 * 2)
        for record in records:
            final = replay_states(record["actions"])[-1]
            self.assertEqual(final.result, record["result"])

    def test_per_budget_games_csv_is_written(self):
        out, _ = self._run()
        for budget in (4, 8, 16):
            path = out / f"budget_{budget}_games.csv"
            self.assertTrue(path.is_file())
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)  # 3 pairs x 2 colours

    def test_existing_openings_file_is_reused_not_regenerated(self):
        out = self.root / "out"
        with mock.patch.object(suite, "create_bot", side_effect=cheap_bot_factory):
            suite.run_budget_comparison(str(self.ckpt), out, budgets=[4, 8], opponent="tactical",
                                        pairs=2, seed=13)
        openings_before = (out / "openings.json").read_text()
        with mock.patch.object(suite, "create_bot", side_effect=cheap_bot_factory), \
             mock.patch.object(suite, "generate_opening_corpus") as gen:
            suite.run_budget_comparison(str(self.ckpt), out, budgets=[4, 8], opponent="tactical",
                                        pairs=2, seed=13, save_moves=False)
            gen.assert_not_called()
        self.assertEqual((out / "openings.json").read_text(), openings_before)

    def test_mismatched_reuse_request_raises(self):
        out = self.root / "out"
        with mock.patch.object(suite, "create_bot", side_effect=cheap_bot_factory):
            suite.run_budget_comparison(str(self.ckpt), out, budgets=[4, 8], opponent="tactical",
                                        pairs=2, seed=13)
        with mock.patch.object(suite, "create_bot", side_effect=cheap_bot_factory):
            with self.assertRaises(ValueError):
                suite.run_budget_comparison(str(self.ckpt), out, budgets=[4, 8], opponent="tactical",
                                            pairs=5, seed=13)

    def test_external_opponent_spec_is_created_with_strict_fallback(self):
        calls = []

        def recording_factory(spec, **kwargs):
            calls.append((spec, kwargs))
            return create_bot("tactical", name=kwargs.get("name") or str(spec))

        with mock.patch.object(suite, "create_bot", side_effect=recording_factory):
            suite.run_budget_comparison(str(self.ckpt), self.root / "out2", budgets=[4],
                                        opponent="utttai:1", pairs=2, seed=13)
        opponent_calls = [kw for spec, kw in calls if spec == "utttai:1"]
        self.assertEqual(len(opponent_calls), 1)
        self.assertEqual(opponent_calls[0].get("fallback"), "raise")
        # A non-external spec (the candidate checkpoint) must not receive a
        # `fallback` kwarg it doesn't accept.
        candidate_calls = [kw for spec, kw in calls if spec != "utttai:1"]
        self.assertTrue(candidate_calls)
        self.assertNotIn("fallback", candidate_calls[0])

    def test_progress_json_is_marked_failed_not_left_running(self):
        out = self.root / "out"
        with mock.patch.object(suite, "create_bot", side_effect=cheap_bot_factory), \
             mock.patch.object(suite, "run_matched_budget_comparison",
                               side_effect=RuntimeError("engine died")):
            with self.assertRaisesRegex(RuntimeError, "engine died"):
                suite.run_budget_comparison(str(self.ckpt), out, budgets=[4, 8], opponent="tactical",
                                            pairs=2, seed=13)
        manifest = json.loads((out / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        progress = json.loads((out / "progress.json").read_text())
        self.assertEqual(progress["status"], "failed")
        self.assertIn("error", progress)


class TestBudgetCompareCli(unittest.TestCase):
    def test_cli_dispatches_to_run_budget_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ckpt = write_checkpoint(root / "latest.pt", iteration=1)
            with mock.patch.object(suite, "run_budget_comparison") as ran:
                suite.main(["--checkpoint", str(ckpt), "--output", str(root / "out"),
                           "--budget-compare", "--budgets", "4", "8", "--opponent", "tactical",
                           "--pairs", "3"])
            ran.assert_called_once()
            self.assertEqual(ran.call_args.kwargs["opponent"], "tactical")
            self.assertEqual(ran.call_args.kwargs["pairs"], 3)
            self.assertEqual(ran.call_args.kwargs["budgets"], [4, 8])


if __name__ == "__main__":
    unittest.main()
