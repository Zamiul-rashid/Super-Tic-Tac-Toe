"""M7: the evaluation harness must produce evidence a decision can rest on.

These cover `sttt.evaluation`, the shared layer both comparison scripts use:

* entrant identity derived from the checkpoint that actually loaded, not a
  hardcoded label (the old scripts called `runs/big_run/latest.pt`
  "ckpt-iter6155" and `best.pt` the same, though best.pt was observed at 5750);
* one rating->rank ordering, so a CSV `rank` column cannot contradict the
  scoreboard printed beside it;
* opening-pair-level bootstrap, because mirrored games share an opening and are
  not independent samples;
* a provenance manifest carrying checkpoint SHA-256, iteration, architecture and
  native build identity.

No test here plays a real neural game; the fixtures are constructed MatchResults
and tiny state dicts.
"""
import csv
import json
import pathlib
import tempfile
import unittest

import torch

from sttt.evaluation import (
    checkpoint_identity,
    entrant_name,
    evaluation_manifest,
    freeze_checkpoint,
    pair_bootstrap_difference,
    pair_bootstrap_score,
    rank_ratings,
    write_ratings_csv,
)
from sttt.tournament import MatchResult


def make_pair(pair_id, subject, opponent, winner_even, winner_odd, opening=(0, 1)):
    """One mirrored opening pair: subject is X in the even game, O in the odd one."""
    return [
        MatchResult(game_id=2 * pair_id, pair_id=pair_id, opening_plies=len(opening),
                    opening_moves=list(opening), player_x=subject, player_o=opponent,
                    winner=winner_even, moves=30),
        MatchResult(game_id=2 * pair_id + 1, pair_id=pair_id, opening_plies=len(opening),
                    opening_moves=list(opening), player_x=opponent, player_o=subject,
                    winner=winner_odd, moves=30),
    ]


def write_checkpoint(path, iteration=1600, arch="mlp"):
    torch.save({"model": {"w": torch.zeros(2)}, "iteration": iteration, "arch": arch}, path)
    return path


class _Rating:
    def __init__(self, name, elo):
        self.name, self.elo = name, elo
        self.error_margin = 10.0
        self.wins, self.draws, self.losses, self.games = 1, 0, 1, 2


class _Glicko:
    rating, rd = 1500.0, 50.0


class TestCheckpointIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_identity_reports_iteration_arch_and_digest(self):
        ckpt = write_checkpoint(self.dir / "latest.pt", iteration=5750, arch="resnet")
        ident = checkpoint_identity(ckpt)
        self.assertEqual(ident["iteration"], 5750)
        self.assertEqual(ident["arch"], "resnet")
        self.assertEqual(len(ident["sha256"]), 64)
        self.assertTrue(ident["sha256"].startswith(ident["sha8"]))

    def test_two_checkpoints_with_equal_iteration_get_distinct_names(self):
        """The defect: names were labels, so two different files shared one name."""
        a = write_checkpoint(self.dir / "a.pt", iteration=6155)
        b = self.dir / "b.pt"
        torch.save({"model": {"w": torch.ones(2)}, "iteration": 6155, "arch": "mlp"}, b)
        self.assertNotEqual(entrant_name(a, 512), entrant_name(b, 512))

    def test_name_records_the_loaded_iteration_not_the_filename(self):
        ckpt = write_checkpoint(self.dir / "best.pt", iteration=5750)
        self.assertIn("iter5750", entrant_name(ckpt, 512))
        self.assertNotIn("6155", entrant_name(ckpt, 512))

    def test_name_includes_the_search_budget(self):
        ckpt = write_checkpoint(self.dir / "latest.pt")
        self.assertNotEqual(entrant_name(ckpt, 512), entrant_name(ckpt, 2000))
        self.assertTrue(entrant_name(ckpt, 2000).endswith("s2000"))

    def test_checkpoint_without_iteration_falls_back_to_stem(self):
        ckpt = self.dir / "mystery.pt"
        torch.save({"model": {"w": torch.zeros(2)}}, ckpt)
        name = entrant_name(ckpt, 512)
        self.assertIn("mystery", name)
        self.assertNotIn("iterNone", name)

    def test_missing_checkpoint_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            checkpoint_identity(self.dir / "absent.pt")


class TestFreezeCheckpoint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_frozen_copy_is_independent_of_a_rolling_source(self):
        src = write_checkpoint(self.dir / "latest.pt", iteration=2007)
        frozen_dir = self.dir / "frozen"
        dest, digest = freeze_checkpoint(src, frozen_dir)

        write_checkpoint(src, iteration=2008)          # the run rolls forward
        self.assertEqual(checkpoint_identity(dest)["iteration"], 2007)
        self.assertEqual(checkpoint_identity(dest)["sha256"], digest)

    def test_source_is_preserved(self):
        src = write_checkpoint(self.dir / "latest.pt")
        before = src.read_bytes()
        freeze_checkpoint(src, self.dir / "frozen")
        self.assertEqual(src.read_bytes(), before)


class TestRankOrdering(unittest.TestCase):
    def setUp(self):
        # Deliberately NOT in rating order: this is the insertion order the old
        # `enumerate(tourn["elo_ratings"])` would have numbered 1, 2, 3.
        self.elo = {"weak": _Rating("weak", 1200.0),
                    "strong": _Rating("strong", 1800.0),
                    "middle": _Rating("middle", 1500.0)}
        self.glicko = {name: _Glicko() for name in self.elo}

    def test_rank_ratings_sorts_by_rating_descending(self):
        self.assertEqual(rank_ratings(self.elo), ["strong", "middle", "weak"])

    def test_csv_rank_column_matches_rating_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "ratings.csv"
            write_ratings_csv({"elo_ratings": self.elo, "glicko_ratings": self.glicko}, path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([r["participant"] for r in rows], ["strong", "middle", "weak"])
        self.assertEqual([int(r["rank"]) for r in rows], [1, 2, 3])
        # rank 1 must be the highest rating, which is what the old code lost
        self.assertEqual(float(rows[0]["elo"]), 1800.0)

    def test_ties_do_not_drop_or_duplicate_participants(self):
        elo = {"a": _Rating("a", 1500.0), "b": _Rating("b", 1500.0)}
        self.assertCountEqual(rank_ratings(elo), ["a", "b"])


class TestPairBootstrap(unittest.TestCase):
    def test_score_point_estimate_matches_direct_computation(self):
        # 4 pairs: subject wins both games of pair 0, splits 1 and 2, loses pair 3.
        results = (make_pair(0, "S", "O", 1, -1)      # S wins as X, S wins as O
                   + make_pair(1, "S", "O", 1, 1)     # S wins as X, S loses as O
                   + make_pair(2, "S", "O", 0, 0)     # both drawn
                   + make_pair(3, "S", "O", -1, 1))   # S loses both
        stats = pair_bootstrap_score(results, "S", iterations=500, seed=7)
        self.assertEqual(stats["games"], 8)
        self.assertEqual(stats["pairs"], 4)
        self.assertAlmostEqual(stats["score_rate"], 0.5)

    def test_interval_brackets_the_point_estimate(self):
        results = []
        for k in range(20):
            results += make_pair(k, "S", "O", 1 if k % 2 else -1, -1 if k % 2 else 1)
        stats = pair_bootstrap_score(results, "S", iterations=800, seed=11)
        self.assertLessEqual(stats["ci_low"], stats["score_rate"])
        self.assertLessEqual(stats["score_rate"], stats["ci_high"])

    def test_resampling_is_over_pairs_not_games(self):
        """A perfectly mirrored pair contributes one sample, not two.

        With every pair identical there is no between-pair variation, so a
        pair-level bootstrap must return a zero-width interval. A game-level
        bootstrap would resample the two halves independently and manufacture
        spread that the paired design deliberately cancels.
        """
        results = []
        for k in range(10):
            results += make_pair(k, "S", "O", 1, -1)   # S wins both games of every pair
        stats = pair_bootstrap_score(results, "S", iterations=300, seed=3)
        self.assertAlmostEqual(stats["score_rate"], 1.0)
        self.assertAlmostEqual(stats["ci_low"], 1.0)
        self.assertAlmostEqual(stats["ci_high"], 1.0)

    def test_deterministic_for_a_fixed_seed(self):
        results = []
        for k in range(12):
            results += make_pair(k, "S", "O", 1, 1)
        a = pair_bootstrap_score(results, "S", iterations=400, seed=5)
        b = pair_bootstrap_score(results, "S", iterations=400, seed=5)
        self.assertEqual(a, b)

    def test_unknown_subject_is_rejected(self):
        with self.assertRaises(ValueError):
            pair_bootstrap_score(make_pair(0, "S", "O", 1, 1), "nobody", iterations=10, seed=1)

    def test_budget_difference_requires_a_shared_opening_corpus(self):
        """The whole point of M7: budgets must be compared on the same openings."""
        low = make_pair(0, "S", "O", 1, 1, opening=(0, 1))
        high = make_pair(0, "S", "O", 1, 1, opening=(4, 5))     # different opening
        with self.assertRaises(ValueError) as ctx:
            pair_bootstrap_difference(low, high, "S", iterations=10, seed=1)
        self.assertIn("opening", str(ctx.exception).lower())

    def test_budget_difference_point_estimate(self):
        low, high = [], []
        for k in range(8):
            low += make_pair(k, "S", "O", -1, 1)    # S loses both -> 0.0
            high += make_pair(k, "S", "O", 1, -1)   # S wins both  -> 1.0
        stats = pair_bootstrap_difference(low, high, "S", iterations=400, seed=9)
        self.assertAlmostEqual(stats["difference"], 1.0)
        self.assertEqual(stats["pairs"], 8)


class TestEvaluationManifest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_manifest_carries_provenance_and_is_json_serializable(self):
        ckpt = write_checkpoint(self.dir / "latest.pt", iteration=2007)
        manifest = evaluation_manifest(
            kind="budget-sweep",
            checkpoints={"candidate": ckpt},
            config={"simulations": [512, 1024], "backend": "cpp", "device": "cpu"},
            seeds={"openings": 4242},
        )
        text = json.dumps(manifest)          # must not raise
        self.assertIn("candidate", manifest["checkpoints"])
        self.assertEqual(manifest["checkpoints"]["candidate"]["iteration"], 2007)
        self.assertEqual(len(manifest["checkpoints"]["candidate"]["sha256"]), 64)
        for key in ("schema_version", "kind", "created_at", "config", "seeds", "environment"):
            self.assertIn(key, manifest)
        self.assertIn("torch", manifest["environment"])
        self.assertIn("native", manifest["environment"])
        self.assertIn("2007", text)

    def test_status_defaults_to_running_so_an_abort_is_not_scored_as_evidence(self):
        manifest = evaluation_manifest(kind="h2h", checkpoints={}, config={}, seeds={})
        self.assertEqual(manifest["status"], "running")


if __name__ == "__main__":
    unittest.main()
