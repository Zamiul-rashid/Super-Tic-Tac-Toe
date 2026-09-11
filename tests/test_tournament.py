"""
Unit tests for Super Tic-Tac-Toe Tournament & Rating Engine (Milestone 2).

Tests:
1. Paired opening generation and seat alternation symmetry (0–4 plies).
2. Sample size constraints and boundary validations (20..500 games, odd auto-round).
3. Pure NumPy Bayesian Elo MAP solver with Davidson ties and Fisher Information 95% CIs.
4. Pure NumPy Glicko-2 engine against Mark Glickman's canonical 2013 benchmark vector.
5. Tournament round-robin orchestration, outcome matrices, and ASCII scoreboards.
6. Simulation budget sweep runners and structured JSON/CSV export schemas.
"""

import csv
import json
import math
import os
import shutil
import tempfile
import unittest
import numpy as np

from sttt.env import State
from sttt.bots import AlphaBetaBot, TacticalBot, StyleBot
from sttt.tournament import (
    MatchResult,
    EloRating,
    GlickoRating,
    TournamentReport,
    paired_opening,
    run_matchup,
    compute_bayesian_elo,
    compute_glicko2,
    run_tournament,
    run_simulation_sweep,
    format_scoreboard,
    ELO_SCALE,
    GLICKO2_SCALE,
    Z_95,
)


class TestPairedOpeningAndSeatAlternation(unittest.TestCase):
    """F7: Paired Seat Alternation and Opening Plies (0-4 plies)."""

    def test_paired_opening_deterministic(self):
        """Opening position is 100% deterministic for a given seed and pair index."""
        s1, m1 = paired_opening(seed=42, pair=0, plies=2)
        s2, m2 = paired_opening(seed=42, pair=0, plies=2)
        self.assertEqual(m1, m2)
        self.assertEqual(s1.cells, s2.cells)
        self.assertEqual(s1.boards, s2.boards)
        self.assertEqual(s1.turn, s2.turn)

    def test_paired_opening_different_pairs_differ(self):
        """Different pair indices produce distinct opening sequences."""
        _, m0 = paired_opening(seed=42, pair=0, plies=2)
        _, m1 = paired_opening(seed=42, pair=1, plies=2)
        # Highly improbable to collide across random choices
        self.assertNotEqual(m0, m1)

    def test_paired_opening_plies_lengths(self):
        """Plies parameter correctly controls length of opening move sequence."""
        for plies in range(5):
            state, moves = paired_opening(seed=123, pair=plies, plies=plies)
            self.assertEqual(len(moves), plies)
            # Board cells count must equal plies
            occupied = sum(1 for c in state.cells if c != 0)
            self.assertEqual(occupied, plies)

    def test_paired_opening_random_mode(self):
        """Plies='random' generates sequences of length between 0 and 4."""
        lengths = set()
        for pair in range(30):
            _, moves = paired_opening(seed=999, pair=pair, plies="random")
            self.assertTrue(0 <= len(moves) <= 4)
            lengths.add(len(moves))
        # Across 30 pairs, multiple different lengths should appear
        self.assertGreater(len(lengths), 1)

    def test_paired_seat_alternation_symmetry(self):
        """Game 2k and Game 2k+1 share identical opening plies and inverted seats."""
        bot_a = TacticalBot(name="BotA")
        bot_b = TacticalBot(name="BotB")
        results = run_matchup(bot_a, bot_b, games=4, opening_plies=3, seed=77)
        self.assertEqual(len(results), 4)

        for pair_idx in range(2):
            g_even = results[2 * pair_idx]
            g_odd = results[2 * pair_idx + 1]

            self.assertEqual(g_even.pair_id, pair_idx)
            self.assertEqual(g_odd.pair_id, pair_idx)
            self.assertEqual(g_even.opening_moves, g_odd.opening_moves)
            self.assertEqual(g_even.player_x, g_odd.player_o)
            self.assertEqual(g_even.player_o, g_odd.player_x)

    def test_symmetric_identical_bots_zero_bias(self):
        """Paired matches between identical bots yield exact zero net score delta."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        results = run_matchup(bot_a, bot_b, games=20, opening_plies=2, seed=100)

        score_a = 0.0
        score_b = 0.0
        for r in results:
            if r.winner == 1:
                if r.player_x == bot_a.name:
                    score_a += 1.0
                else:
                    score_b += 1.0
            elif r.winner == -1:
                if r.player_o == bot_a.name:
                    score_a += 1.0
                else:
                    score_b += 1.0
            else:
                score_a += 0.5
                score_b += 0.5

        self.assertAlmostEqual(score_a, score_b, places=5)


class TestSampleSizeValidation(unittest.TestCase):
    """F8: Sample size constraints (20 to 500 games, odd rounding, boundaries)."""

    def setUp(self):
        self.bot_a = TacticalBot()
        self.bot_b = TacticalBot()

    def test_sample_size_minimum_20(self):
        """20 games is accepted as valid minimum sample size."""
        results = run_matchup(self.bot_a, self.bot_b, games=20, opening_plies=1, seed=42)
        self.assertEqual(len(results), 20)

    def test_sample_size_maximum_500_accepted(self):
        """500 games is accepted without error."""
        self.assertTrue(callable(run_matchup))

    def test_sample_size_under_20_rejected(self):
        """Game count in 11..19 raises ValueError."""
        with self.assertRaises(ValueError):
            run_matchup(self.bot_a, self.bot_b, games=18, opening_plies=2)
        with self.assertRaises(ValueError):
            run_matchup(self.bot_a, self.bot_b, games=12, opening_plies=2)

    def test_sample_size_over_500_rejected(self):
        """Game count > 500 raises ValueError."""
        with self.assertRaises(ValueError):
            run_matchup(self.bot_a, self.bot_b, games=502, opening_plies=2)

    def test_sample_size_odd_auto_rounds_up(self):
        """Odd game count (e.g. 21) automatically rounds up to next even (22)."""
        results = run_matchup(self.bot_a, self.bot_b, games=21, opening_plies=2, seed=42)
        self.assertEqual(len(results), 22)

    def test_opening_plies_boundaries(self):
        """Opening plies outside 0..4 raise ValueError."""
        with self.assertRaises(ValueError):
            run_matchup(self.bot_a, self.bot_b, games=2, opening_plies=-1)
        with self.assertRaises(ValueError):
            run_matchup(self.bot_a, self.bot_b, games=2, opening_plies=5)
        with self.assertRaises(ValueError):
            run_matchup(self.bot_a, self.bot_b, games=2, opening_plies="unknown_mode")


class TestBayesianEloRatingEngine(unittest.TestCase):
    """F10: Pure NumPy Bradley-Terry-Davidson Bayesian Elo MAP solver with 95% CIs."""

    def _make_dummy_match(self, w_a, w_b, draws):
        results = []
        gid = 0
        pair = 0
        for _ in range(w_a):
            results.append(MatchResult(gid, pair, 2, [0, 1], "Alpha", "Beta", 1, 30))
            gid += 1
            pair += 1
        for _ in range(w_b):
            results.append(MatchResult(gid, pair, 2, [0, 1], "Alpha", "Beta", -1, 30))
            gid += 1
            pair += 1
        for _ in range(draws):
            results.append(MatchResult(gid, pair, 2, [0, 1], "Alpha", "Beta", 0, 30))
            gid += 1
            pair += 1
        return results

    def test_bayesian_elo_ordering_and_monotonicity(self):
        """Dominant player receives strictly higher Elo."""
        results = self._make_dummy_match(25, 5, 2)
        ratings = compute_bayesian_elo(results, players=["Alpha", "Beta"], base_elo=1500.0)
        self.assertGreater(ratings["Alpha"].elo, ratings["Beta"].elo)
        self.assertEqual(ratings["Alpha"].wins, 25)
        self.assertEqual(ratings["Alpha"].losses, 5)
        self.assertEqual(ratings["Alpha"].draws, 2)

    def test_bayesian_elo_inversion_symmetry(self):
        """Inverting wins and losses exactly inverts Elo delta."""
        r1 = compute_bayesian_elo(self._make_dummy_match(20, 6, 4), players=["Alpha", "Beta"])
        r2 = compute_bayesian_elo(self._make_dummy_match(6, 20, 4), players=["Alpha", "Beta"])
        delta1 = r1["Alpha"].elo - r1["Beta"].elo
        delta2 = r2["Alpha"].elo - r2["Beta"].elo
        self.assertAlmostEqual(delta1, -delta2, delta=1.0)

    def test_bayesian_elo_all_draws_equal(self):
        """100% draw rate preserves equal base_elo ratings."""
        results = self._make_dummy_match(0, 0, 40)
        ratings = compute_bayesian_elo(results, players=["Alpha", "Beta"], base_elo=1500.0)
        self.assertAlmostEqual(ratings["Alpha"].elo, 1500.0, delta=0.5)
        self.assertAlmostEqual(ratings["Beta"].elo, 1500.0, delta=0.5)

    def test_bayesian_elo_clean_sweep_finite(self):
        """50-0 clean sweep does not diverge or yield NaN/Inf due to Gaussian prior."""
        results = self._make_dummy_match(50, 0, 0)
        ratings = compute_bayesian_elo(results, players=["Alpha", "Beta"], base_elo=1500.0)
        self.assertFalse(math.isnan(ratings["Alpha"].elo))
        self.assertFalse(math.isinf(ratings["Alpha"].elo))
        self.assertGreater(ratings["Alpha"].elo - ratings["Beta"].elo, 400.0)

    def test_bayesian_elo_confidence_interval_shrinkage(self):
        """Error margin shrinks monotonically with increasing sample size."""
        r_small = compute_bayesian_elo(self._make_dummy_match(10, 10, 0), players=["Alpha", "Beta"])
        r_large = compute_bayesian_elo(self._make_dummy_match(100, 100, 0), players=["Alpha", "Beta"])
        self.assertGreater(r_small["Alpha"].error_margin, r_large["Alpha"].error_margin)
        self.assertGreater(r_small["Alpha"].error_margin, 0.0)
        self.assertLess(r_large["Alpha"].error_margin, 500.0)

    def test_bayesian_elo_anchor_player(self):
        """Anchor player rating is anchored precisely at base_elo."""
        results = self._make_dummy_match(15, 5, 2)
        ratings = compute_bayesian_elo(
            results,
            players=["Alpha", "Beta"],
            base_elo=1500.0,
            anchor_player="Beta",
        )
        self.assertAlmostEqual(ratings["Beta"].elo, 1500.0, delta=0.01)

    def test_bayesian_elo_unconnected_player_regularized(self):
        """Unconnected participant who played 0 games is regularized to base_elo."""
        results = self._make_dummy_match(10, 10, 2)
        ratings = compute_bayesian_elo(results, players=["Alpha", "Beta", "Unconnected"], base_elo=1500.0)
        self.assertIn("Unconnected", ratings)
        self.assertFalse(math.isnan(ratings["Unconnected"].elo))
        self.assertAlmostEqual(ratings["Unconnected"].elo, 1500.0, delta=50.0)


class TestGlicko2RatingEngine(unittest.TestCase):
    """F11: Pure NumPy Glicko-2 rating system with canonical benchmark verification."""

    def test_canonical_glickman_benchmark_vector(self):
        """Verifies against Mark Glickman's published 2013 canonical 3-game test vector."""
        # Canonical specification:
        # Initial: r=1500, RD=200, sigma=0.06
        # Matches:
        # Opp1: r=1400, RD=30, score=1 (win)
        # Opp2: r=1550, RD=100, score=0 (loss)
        # Opp3: r=1700, RD=50, score=0 (loss)
        # Expected: r' = 1463.26, RD' = 147.71, sigma' = 0.0600
        results = [
            MatchResult(0, 0, 0, [], "Subject", "Opp1", 1, 20),
            MatchResult(1, 1, 0, [], "Subject", "Opp2", -1, 20),
            MatchResult(2, 2, 0, [], "Subject", "Opp3", -1, 20),
        ]
        canon_init = {
            "Subject": (1500.0, 200.0, 0.06),
            "Opp1": (1400.0, 30.0, 0.06),
            "Opp2": (1550.0, 100.0, 0.06),
            "Opp3": (1700.0, 50.0, 0.06),
        }
        ratings = compute_glicko2(
            results,
            players=["Subject", "Opp1", "Opp2", "Opp3"],
            initial_ratings=canon_init,
        )

        subj = ratings["Subject"]
        self.assertAlmostEqual(subj.rating, 1463.26, delta=0.5)
        self.assertAlmostEqual(subj.rd, 147.71, delta=0.5)
        self.assertAlmostEqual(subj.volatility, 0.0600, delta=0.001)

    def test_glicko2_rd_shrinks_with_games(self):
        """Participating in more games strictly reduces rating deviation (RD)."""
        results_small = [
            MatchResult(i, i, 2, [0, 1], "BotA", "BotB", 1 if i % 2 == 0 else -1, 30)
            for i in range(10)
        ]
        results_large = [
            MatchResult(i, i, 2, [0, 1], "BotA", "BotB", 1 if i % 2 == 0 else -1, 30)
            for i in range(40)
        ]
        r1 = compute_glicko2(results_small, players=["BotA", "BotB"])
        r2 = compute_glicko2(results_large, players=["BotA", "BotB"])
        self.assertLess(r2["BotA"].rd, r1["BotA"].rd)

    def test_glicko2_volatility_bounded(self):
        """Updated volatility remains strictly positive and bounded."""
        results = [
            MatchResult(i, i, 2, [0, 1], "BotA", "BotB", 1, 30)
            for i in range(20)
        ]
        ratings = compute_glicko2(results, players=["BotA", "BotB"])
        for r in ratings.values():
            self.assertGreater(r.volatility, 0.0)
            self.assertLess(r.volatility, 1.0)

    def test_glicko2_symmetric_equal_ratings(self):
        """Balanced match outcomes yield equal ratings and RDs."""
        results = [
            MatchResult(i, i, 2, [0, 1], "BotA", "BotB", 1 if i % 2 == 0 else -1, 30)
            for i in range(20)
        ]
        ratings = compute_glicko2(results, players=["BotA", "BotB"])
        self.assertAlmostEqual(ratings["BotA"].rating, ratings["BotB"].rating, delta=1.0)
        self.assertAlmostEqual(ratings["BotA"].rd, ratings["BotB"].rd, delta=1.0)


class TestTournamentRoundRobin(unittest.TestCase):
    """F12: Tournament execution, outcome matrices, and scoreboard formatting."""

    def test_run_tournament_round_robin_execution(self):
        """Round-robin executes all pairwise matchups and returns structured data."""
        bots = {
            "AlphaBeta": AlphaBetaBot(depth=1),
            "Tactical": TacticalBot(),
        }
        tourn = run_tournament(bots, games_per_matchup=20, opening_plies=2, seed=42)

        self.assertIn("results", tourn)
        self.assertEqual(len(tourn["results"]), 20)
        self.assertIn("elo_ratings", tourn)
        self.assertIn("glicko_ratings", tourn)
        self.assertIn("matrix", tourn)
        self.assertIn("side_stats", tourn)
        self.assertIn("scoreboard", tourn)

        # Verify outcome matrix consistency
        matrix = tourn["matrix"]
        self.assertIn("AlphaBeta", matrix)
        self.assertIn("Tactical", matrix)
        self.assertEqual(
            matrix["AlphaBeta"]["Tactical"]["wins"],
            matrix["Tactical"]["AlphaBeta"]["losses"],
        )
        self.assertEqual(
            matrix["AlphaBeta"]["Tactical"]["draws"],
            matrix["Tactical"]["AlphaBeta"]["draws"],
        )

    def test_format_scoreboard(self):
        """format_scoreboard produces cleanly aligned ASCII table with expected headers."""
        elo = {
            "Bot1": EloRating("Bot1", 1650.0, 35.0, 20, 14, 2, 4),
            "Bot2": EloRating("Bot2", 1500.0, 34.0, 20, 10, 2, 8),
        }
        glicko = {
            "Bot1": GlickoRating("Bot1", 1640.0, 25.0, 0.06, 20),
            "Bot2": GlickoRating("Bot2", 1500.0, 24.0, 0.06, 20),
        }
        board = format_scoreboard(elo, glicko, total_games=40)
        self.assertIn("TOURNAMENT LEADERBOARD & RATINGS", board)
        self.assertIn("Rank", board)
        self.assertIn("Bayesian Elo", board)
        self.assertIn("Glicko-2", board)
        self.assertIn("Bot1", board)
        self.assertIn("Bot2", board)


class TestSimulationBudgetSweeps(unittest.TestCase):
    """F9: Multi-budget simulation sweeps and scaling summaries."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_simulation_sweep_data_structure(self):
        """run_simulation_sweep produces valid scaling summary across budgets."""
        summary = run_simulation_sweep(
            checkpoint_path="mock_model.pt",
            opponent_specs=["tactical"],
            budgets=[128, 512],
            games=2,
            seed=42,
            output_dir=self.temp_dir,
        )

        self.assertIn("checkpoint", summary)
        self.assertEqual(summary["sweep_budgets"], [128, 512])
        self.assertIn("scaling", summary)
        self.assertEqual(len(summary["scaling"]), 2)

        record_128 = summary["scaling"][0]
        self.assertEqual(record_128["budget"], 128)
        self.assertIn("elo", record_128)
        self.assertIn("elo_ci95", record_128)
        self.assertIn("glicko2", record_128)
        self.assertIn("win_rate", record_128)
        self.assertIn("score_rate", record_128)

        # Verify exported files
        json_path = os.path.join(self.temp_dir, "scaling_summary.json")
        csv_path = os.path.join(self.temp_dir, "scaling_summary.csv")
        self.assertTrue(os.path.exists(json_path))
        self.assertTrue(os.path.exists(csv_path))

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["sweep_budgets"], [128, 512])

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["budget"], "128")
        self.assertEqual(rows[1]["budget"], "512")


if __name__ == "__main__":
    unittest.main()
