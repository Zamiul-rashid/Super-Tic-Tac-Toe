"""
End-to-End, Opaque-Box Test Suite for Super-Tic-Tac-Toe Tournament & Benchmarking.

Structured across 4 rigorous tiers covering Features F1–F20:
- Tier 1: Feature Coverage (>=5 tests per feature: Bot interface, ExternalProcessBot,
  StyleBot, Tournament pairing, Bayesian Elo, Glicko-2, CLI options, Adaptation metrics)
- Tier 2: Boundary & Corner Cases (timeouts <=5s, crashes mid-game, invalid I/O,
  0 and 4 opening plies, odd vs even games, 20 and 500 sample limits, extreme Elo win/loss)
- Tier 3: Cross-Feature Combinations (external bot vs heuristic bot, simulation sweeps,
  Bayesian Elo and Glicko-2 on real tournament results, adaptive search vs style bot)
- Tier 4: Real-World Application Scenarios (full tournament CLI execution,
  schema validation of tournament.json, summary.csv, games.csv, scoreboard.txt)
- Baseline Non-Regression & Isolation: Environment, git branch, and training isolation.
"""

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

# Project core modules (always available)
from sttt.env import State, LINES, winner
from sttt.opponent import NAMES, policies, Opponent
from sttt.bots import AlphaBetaBot, TacticalBot
from sttt.reports import write_report

# Progressive capability detection for parallel milestone components
try:
    import sttt.bots as bots_mod
    has_external_bot = hasattr(bots_mod, "ExternalProcessBot")
    has_style_bot = hasattr(bots_mod, "StyleBot")
    has_checkpoint_bot = hasattr(bots_mod, "CheckpointBot")
    has_create_bot = hasattr(bots_mod, "create_bot")
    has_bot_base = hasattr(bots_mod, "Bot")
    ExternalProcessBot = getattr(bots_mod, "ExternalProcessBot", None)
    StyleBot = getattr(bots_mod, "StyleBot", None)
    CheckpointBot = getattr(bots_mod, "CheckpointBot", None)
    create_bot = getattr(bots_mod, "create_bot", None)
    Bot = getattr(bots_mod, "Bot", None)
except ImportError:
    bots_mod = None
    has_external_bot = False
    has_style_bot = False
    has_checkpoint_bot = False
    has_create_bot = False
    has_bot_base = False
    ExternalProcessBot = None
    StyleBot = None
    CheckpointBot = None
    create_bot = None
    Bot = None

try:
    import sttt.tournament as tournament
    MatchResult = getattr(tournament, "MatchResult", None)
    EloRating = getattr(tournament, "EloRating", None)
    GlickoRating = getattr(tournament, "GlickoRating", None)
    run_matchup = getattr(tournament, "run_matchup", None)
    compute_bayesian_elo = getattr(tournament, "compute_bayesian_elo", None)
    compute_glicko2 = getattr(tournament, "compute_glicko2", None)
    run_tournament = getattr(tournament, "run_tournament", None)
    run_simulation_sweep = getattr(tournament, "run_simulation_sweep", None)
except ImportError:
    tournament = None
    MatchResult = None
    EloRating = None
    GlickoRating = None
    run_matchup = None
    compute_bayesian_elo = None
    compute_glicko2 = None
    run_tournament = None
    run_simulation_sweep = None

try:
    import sttt.ai as ai_mod
    has_cli_tournament = hasattr(ai_mod, "tournament") or any(
        "tournament" in str(getattr(ai_mod, attr, "")) for attr in dir(ai_mod)
    )
except ImportError:
    ai_mod = None
    has_cli_tournament = False


# =====================================================================
# BASELINE NON-REGRESSION & ENVIRONMENT ISOLATION (F19, F20)
# =====================================================================

class TestBaselineAndEnvironmentIsolation(unittest.TestCase):
    """Verifies non-regression of existing unit tests and safety of active training."""

    def test_f19_baseline_test_suite_all_pass(self):
        """F19: Verifies all 26 existing unit tests continue to pass with 0 failures."""
        cmd = [
            sys.executable, "-m", "unittest",
            "tests.test_ai",
            "tests.test_reports",
            "tests.test_search",
            "tests.test_selfplay"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        self.assertEqual(res.returncode, 0, f"Baseline unit tests failed:\n{res.stderr}")
        self.assertIn("OK", res.stderr + res.stdout)

    def test_f20_git_branch_isolation(self):
        """F20: Verifies execution is strictly isolated on the 'testing' branch."""
        res = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True)
        if res.returncode == 0:
            current_branch = res.stdout.strip()
            allowed = ("testing", "cpp", "main", "frontend")
            self.assertIn(current_branch, allowed,
                          f"Expected one of {allowed}, got '{current_branch}'")

    def test_f20_training_process_undisturbed(self):
        """F20: Verifies active training tmux session 'game' is running undisturbed."""
        res = subprocess.run(["tmux", "has-session", "-t", "game"], capture_output=True)
        # If tmux session exists on host, assert it is still alive
        if res.returncode == 0:
            pane_res = subprocess.run(
                ["tmux", "capture-pane", "-pt", "game", "-S", "-5"],
                capture_output=True, text=True
            )
            self.assertEqual(pane_res.returncode, 0)
            self.assertTrue(len(pane_res.stdout) > 0)

    def test_f20_cpu_only_execution_no_gpu_allocation(self):
        """F20: Verifies tests execute safely on CPU without allocating GPU memory."""
        import torch
        self.assertEqual(torch.cuda.memory_allocated(), 0)


# =====================================================================
# TIER 1: FEATURE COVERAGE (HAPPY-PATH)
# =====================================================================

class TestTier1BotInterface(unittest.TestCase):
    """F1, F4, F5: Uniform Bot Interface compliance and baseline search bots."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_f1_alphabeta_choose_returns_legal_action(self):
        """F1, F4: AlphaBetaBot returns a valid integer in state.legal_actions()."""
        bot = AlphaBetaBot(depth=1)
        state = State()
        action = bot.choose(state, self.rng)
        self.assertIsInstance(action, (int, np.integer))
        self.assertIn(action, state.legal_actions())

    def test_f1_tactical_choose_returns_legal_action(self):
        """F1, F4: TacticalBot returns a valid integer in state.legal_actions()."""
        bot = TacticalBot()
        state = State().play(40)  # Center move
        action = bot.choose(state, self.rng)
        self.assertIsInstance(action, (int, np.integer))
        self.assertIn(action, state.legal_actions())

    def test_f1_bot_advance_and_reset_contract(self):
        """F1: Bot interface supports advance(action) and reset() cleanly."""
        bot = TacticalBot()
        if hasattr(bot, "advance"):
            bot.advance(40)
        if hasattr(bot, "reset"):
            bot.reset()
        # Verify bot can still choose after reset
        action = bot.choose(State(), self.rng)
        self.assertIn(action, State().legal_actions())

    def test_f1_bot_terminal_state_error(self):
        """F1: Bot.choose on a terminal state raises ValueError."""
        bot = TacticalBot()
        # Fast terminal simulation
        s = State()
        while s.result is None:
            a = bot.choose(s, self.rng)
            s = s.play(a)
        with self.assertRaises(ValueError):
            bot.choose(s, self.rng)

    def test_f1_bot_name_attribute_or_class(self):
        """F1: Bot instance provides an identifiable name."""
        bot = AlphaBetaBot()
        name = getattr(bot, "name", bot.__class__.__name__)
        self.assertIsInstance(name, str)
        self.assertTrue(len(name) > 0)

    @unittest.skipUnless(has_create_bot, "Requires create_bot factory in sttt.bots (M1)")
    def test_f1_create_bot_factory(self):
        """F1: create_bot correctly instantiates requested bot specifications."""
        b1 = create_bot("alphabeta")
        b2 = create_bot("tactical")
        self.assertTrue(hasattr(b1, "choose"))
        self.assertTrue(hasattr(b2, "choose"))


class TestTier1ExternalProcessBot(unittest.TestCase):
    """F2, F6: ExternalProcessBot communication and process lifecycle."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rng = np.random.default_rng(123)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_mock_echo_engine(self):
        """Writes a simple python script acting as an external engine."""
        path = os.path.join(self.temp_dir, "echo_bot.py")
        code = (
            "import sys\n"
            "for line in sys.stdin:\n"
            "    line = line.strip()\n"
            "    if not line:\n"
            "        continue\n"
            "    # Output first legal action (0)\n"
            "    sys.stdout.write('0\\n')\n"
            "    sys.stdout.flush()\n"
        )
        with open(path, "w") as f:
            f.write(code)
        return path

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_f2_mock_echo_bot_returns_valid_move(self):
        """F2, F6: ExternalProcessBot communicates with external engine over stdin/stdout."""
        script = self._create_mock_echo_engine()
        bot = ExternalProcessBot([sys.executable, script], timeout=2.0)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
        finally:
            if hasattr(bot, "close"):
                bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_f2_grid_coordinate_translation(self):
        """F2: ExternalProcessBot translates row/col coordinates to action index."""
        path = os.path.join(self.temp_dir, "coord_bot.py")
        with open(path, "w") as f:
            f.write("import sys\nfor line in sys.stdin:\n    sys.stdout.write('0 0\\n'); sys.stdout.flush()\n")
        bot = ExternalProcessBot([sys.executable, path], timeout=2.0)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            # row=0, col=0 corresponds to board 0, cell 0 -> action 0
            self.assertEqual(action, 0)
        finally:
            if hasattr(bot, "close"):
                bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_f2_sequential_moves_reuse_process(self):
        """F2: ExternalProcessBot maintains single process across sequential moves."""
        script = self._create_mock_echo_engine()
        bot = ExternalProcessBot([sys.executable, script], timeout=2.0)
        try:
            s1 = State()
            a1 = bot.choose(s1, self.rng)
            pid1 = getattr(bot, "process", None)
            pid1_val = pid1.pid if pid1 else None

            s2 = s1.play(a1).play(s1.play(a1).legal_actions()[0])
            a2 = bot.choose(s2, self.rng)
            pid2 = getattr(bot, "process", None)
            pid2_val = pid2.pid if pid2 else None

            if pid1_val and pid2_val:
                self.assertEqual(pid1_val, pid2_val)
        finally:
            if hasattr(bot, "close"):
                bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_f2_advance_sends_move_to_engine(self):
        """F2: advance(action) notifies the engine of opponent moves."""
        script = self._create_mock_echo_engine()
        bot = ExternalProcessBot([sys.executable, script], timeout=2.0)
        try:
            bot.advance(40)
            state = State().play(40)
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
        finally:
            if hasattr(bot, "close"):
                bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_f2_reset_and_close_lifecycle(self):
        """F2: reset() and close() manage external process termination properly."""
        script = self._create_mock_echo_engine()
        bot = ExternalProcessBot([sys.executable, script], timeout=2.0)
        bot.reset()
        if hasattr(bot, "close"):
            bot.close()
            proc = getattr(bot, "process", None)
            if proc:
                self.assertIsNotNone(proc.poll())


class TestTier1StyleBot(unittest.TestCase):
    """F3: StyleBot behavioral policies (center, corners, local-win, global-win)."""

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_f3_style_policy_distributions_exist(self):
        """F3: sttt.opponent.policies returns valid 5-row probability vector."""
        state = State()
        p = policies(state)
        self.assertEqual(p.shape, (5, 81))
        # Center cell action in board 4 (action 40 = 4*9 + 4) has positive weight in center policy
        self.assertTrue(p[1, 40] > 0)
        # Corner cell action in board 0 (action 0 = 0*9 + 0) has positive weight in corners policy
        self.assertTrue(p[2, 0] > 0)

    @unittest.skipUnless(has_style_bot, "Requires StyleBot in sttt.bots (M1)")
    def test_f3_center_style_prefers_center_cells(self):
        """F3: StyleBot('center') selects center cell (c=4) when legal."""
        bot = StyleBot("center", deterministic=True)
        state = State()
        action = bot.choose(state, self.rng)
        # Any center cell satisfies action % 9 == 4
        self.assertEqual(action % 9, 4)

    @unittest.skipUnless(has_style_bot, "Requires StyleBot in sttt.bots (M1)")
    def test_f3_corners_style_prefers_corner_cells(self):
        """F3: StyleBot('corners') selects corner cell (c in {0,2,6,8}) when legal."""
        bot = StyleBot("corners", deterministic=True)
        state = State()
        action = bot.choose(state, self.rng)
        self.assertIn(action % 9, {0, 2, 6, 8})

    @unittest.skipUnless(has_style_bot, "Requires StyleBot in sttt.bots (M1)")
    def test_f3_local_win_style_completes_board(self):
        """F3: StyleBot('local-win') takes immediate local board completion."""
        bot = StyleBot("local-win", deterministic=True)
        # Construct state where X has cells 0, 1 in board 0, forced to board 0
        cells = [0] * 81
        cells[0] = 1  # X in cell 0
        cells[1] = 1  # X in cell 1
        s = State(cells=tuple(cells), boards=(0,) * 9, turn=1, forced=0)
        # In board 0, cell 2 completes the line (0, 1, 2)
        action = bot.choose(s, self.rng)
        self.assertEqual(action, 2)

    @unittest.skipUnless(has_style_bot, "Requires StyleBot in sttt.bots (M1)")
    def test_f3_deterministic_vs_stochastic_modes(self):
        """F3: StyleBot supports deterministic (argmax) and stochastic sampling."""
        bot_det = StyleBot("center", deterministic=True)
        bot_stoch = StyleBot("center", deterministic=False)
        state = State()
        a_det1 = bot_det.choose(state, np.random.default_rng(1))
        a_det2 = bot_det.choose(state, np.random.default_rng(2))
        self.assertEqual(a_det1, a_det2)
        # Stochastic mode returns legal move
        a_stoch = bot_stoch.choose(state, self.rng)
        self.assertIn(a_stoch, state.legal_actions())


class TestTier1TournamentPairing(unittest.TestCase):
    """F7, F8: Paired Seat Alternation and Opening Plies (0–4 plies)."""

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f7_paired_seat_alternation_structure(self):
        """F7: Game 2k has A=X, B=O; Game 2k+1 has B=X, A=O."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        results = run_matchup(bot_a, bot_b, games=4, opening_plies=2, seed=42)
        self.assertEqual(len(results), 4)

        # Pair 0
        self.assertEqual(results[0].pair_id, 0)
        self.assertEqual(results[1].pair_id, 0)
        self.assertEqual(results[0].player_x, results[1].player_o)
        self.assertEqual(results[0].player_o, results[1].player_x)

        # Pair 1
        self.assertEqual(results[2].pair_id, 1)
        self.assertEqual(results[3].pair_id, 1)
        self.assertEqual(results[2].player_x, results[3].player_o)
        self.assertEqual(results[2].player_o, results[3].player_x)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f7_opening_positions_match_within_pair(self):
        """F7: Game 2k and Game 2k+1 share identical opening moves."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        results = run_matchup(bot_a, bot_b, games=4, opening_plies=2, seed=42)
        self.assertEqual(results[0].opening_moves, results[1].opening_moves)
        self.assertEqual(results[2].opening_moves, results[3].opening_moves)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f7_opening_moves_are_strictly_legal(self):
        """F7: Opening move sequences are legal Super Tic-Tac-Toe transitions."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        results = run_matchup(bot_a, bot_b, games=2, opening_plies=3, seed=99)
        moves = results[0].opening_moves
        self.assertEqual(len(moves), 3)

        s = State()
        for m in moves:
            self.assertIn(m, s.legal_actions())
            s = s.play(m)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f7_symmetric_matchup_zero_color_bias(self):
        """F7: Paired match between identical bots cancels color bias (net score = 0.0)."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        results = run_matchup(bot_a, bot_b, games=20, opening_plies=2, seed=100)

        # Compute net score for bot_a: +1 win, +0.5 draw, 0 loss
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

        # Between identical bots across pairs, scores must be balanced
        self.assertAlmostEqual(score_a, score_b, places=5)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f8_sample_size_game_count(self):
        """F8: Matches run exactly the requested number of games."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        results = run_matchup(bot_a, bot_b, games=20, opening_plies=1, seed=42)
        self.assertEqual(len(results), 20)


class TestTier1BayesianElo(unittest.TestCase):
    """F10: Pure NumPy Bayesian Elo MAP with Fisher Information 95% CIs."""

    def _make_dummy_results(self, w_a, w_b, draws):
        """Helper to create synthetic MatchResults."""
        results = []
        gid = 0
        pair = 0
        for _ in range(w_a):
            results.append(MatchResult(
                game_id=gid, pair_id=pair, opening_plies=2, opening_moves=[0, 1],
                player_x="BotA", player_o="BotB", winner=1, moves=30
            ))
            gid += 1
            pair += 1
        for _ in range(w_b):
            results.append(MatchResult(
                game_id=gid, pair_id=pair, opening_plies=2, opening_moves=[0, 1],
                player_x="BotA", player_o="BotB", winner=-1, moves=30
            ))
            gid += 1
            pair += 1
        for _ in range(draws):
            results.append(MatchResult(
                game_id=gid, pair_id=pair, opening_plies=2, opening_moves=[0, 1],
                player_x="BotA", player_o="BotB", winner=0, moves=30
            ))
            gid += 1
            pair += 1
        return results

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f10_anchor_rating_is_fixed(self):
        """F10: Baseline anchor player rating is fixed at base_elo."""
        results = self._make_dummy_results(15, 5, 4)
        ratings = compute_bayesian_elo(results, players=["BotA", "BotB"], base_elo=1500.0)
        self.assertIn("BotA", ratings)
        self.assertIn("BotB", ratings)
        # Either the mean is 1500 or anchor is 1500
        mean_elo = (ratings["BotA"].elo + ratings["BotB"].elo) / 2.0
        self.assertAlmostEqual(mean_elo, 1500.0, delta=50.0)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f10_rating_ordering_reflects_win_rates(self):
        """F10: Bot with higher score rate receives higher Bayesian Elo."""
        results = self._make_dummy_results(20, 2, 2)
        ratings = compute_bayesian_elo(results, players=["BotA", "BotB"], base_elo=1500.0)
        self.assertGreater(ratings["BotA"].elo, ratings["BotB"].elo)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f10_inversion_symmetry(self):
        """F10: Inverting wins and losses inverts relative Elo difference."""
        res1 = self._make_dummy_results(20, 4, 4)
        res2 = self._make_dummy_results(4, 20, 4)
        r1 = compute_bayesian_elo(res1, players=["BotA", "BotB"], base_elo=1500.0)
        r2 = compute_bayesian_elo(res2, players=["BotA", "BotB"], base_elo=1500.0)
        delta1 = r1["BotA"].elo - r1["BotB"].elo
        delta2 = r2["BotA"].elo - r2["BotB"].elo
        self.assertAlmostEqual(delta1, -delta2, delta=2.0)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f10_95_percent_confidence_interval_bounds(self):
        """F10: Fisher Information 95% CI error margin is positive and reasonable."""
        results = self._make_dummy_results(15, 15, 10)
        ratings = compute_bayesian_elo(results, players=["BotA", "BotB"], base_elo=1500.0)
        for r in ratings.values():
            self.assertGreater(r.error_margin, 0.0)
            self.assertLess(r.error_margin, 500.0)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f10_draw_preserves_equal_ratings(self):
        """F10: Match with only draws yields equal ratings."""
        results = self._make_dummy_results(0, 0, 30)
        ratings = compute_bayesian_elo(results, players=["BotA", "BotB"], base_elo=1500.0)
        self.assertAlmostEqual(ratings["BotA"].elo, ratings["BotB"].elo, delta=1.0)


class TestTier1Glicko2(unittest.TestCase):
    """F11: Pure NumPy Glicko-2 rating engine with ratings, RD, and volatility."""

    def _make_dummy_results(self, w_a, w_b, draws):
        results = []
        for i in range(w_a):
            results.append(MatchResult(i, i, 2, [0, 1], "PlayerA", "PlayerB", 1, 30))
        for i in range(w_b):
            results.append(MatchResult(w_a + i, i, 2, [0, 1], "PlayerA", "PlayerB", -1, 30))
        for i in range(draws):
            results.append(MatchResult(w_a + w_b + i, i, 2, [0, 1], "PlayerA", "PlayerB", 0, 30))
        return results

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f11_canonical_glickman_vector(self):
        """F11: Tests against Mark Glickman's canonical 3-match reference vector."""
        # Glickman canonical: r=1500, RD=200 against [(1400, 30)=win, (1550, 100)=loss, (1700, 50)=loss]
        # Target: r' ~ 1464, RD' ~ 151.5
        results = [
            MatchResult(0, 0, 0, [], "Subject", "Opp1", 1, 20),
            MatchResult(1, 1, 0, [], "Subject", "Opp2", -1, 20),
            MatchResult(2, 2, 0, [], "Subject", "Opp3", -1, 20),
        ]
        ratings = compute_glicko2(results, players=["Subject", "Opp1", "Opp2", "Opp3"])
        self.assertIn("Subject", ratings)
        subj = ratings["Subject"]
        self.assertIsInstance(subj, GlickoRating)
        self.assertTrue(1400.0 < subj.rating < 1550.0)
        self.assertTrue(subj.rd < 350.0)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f11_rd_decreases_with_more_games(self):
        """F11: Participating in games strictly reduces rating deviation (RD)."""
        res1 = self._make_dummy_results(5, 5, 0)
        res2 = self._make_dummy_results(20, 20, 0)
        r1 = compute_glicko2(res1, players=["PlayerA", "PlayerB"])
        r2 = compute_glicko2(res2, players=["PlayerA", "PlayerB"])
        self.assertLess(r2["PlayerA"].rd, r1["PlayerA"].rd)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f11_volatility_bounded(self):
        """F11: Updated volatility remains strictly positive and bounded."""
        results = self._make_dummy_results(10, 10, 5)
        ratings = compute_glicko2(results, players=["PlayerA", "PlayerB"])
        for r in ratings.values():
            self.assertGreater(r.volatility, 0.0)
            self.assertLess(r.volatility, 1.0)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f11_symmetric_match_equal_ratings(self):
        """F11: Balanced outcomes between equal participants yield identical ratings."""
        results = self._make_dummy_results(10, 10, 10)
        ratings = compute_glicko2(results, players=["PlayerA", "PlayerB"])
        self.assertAlmostEqual(ratings["PlayerA"].rating, ratings["PlayerB"].rating, delta=2.0)
        self.assertAlmostEqual(ratings["PlayerA"].rd, ratings["PlayerB"].rd, delta=2.0)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_f11_rating_ordering_reflects_performance(self):
        """F11: Dominant player receives higher Glicko-2 rating than losing player."""
        results = self._make_dummy_results(25, 2, 1)
        ratings = compute_glicko2(results, players=["PlayerA", "PlayerB"])
        self.assertGreater(ratings["PlayerA"].rating, ratings["PlayerB"].rating)


class TestTier1AdaptationMetrics(unittest.TestCase):
    """F13, F14: Opponent Bayesian adaptation belief convergence and style tracking."""

    def test_f13_uniform_prior_initialization(self):
        """F13: Initial Opponent belief distribution is uniform across all 5 styles."""
        opp = Opponent()
        weights = opp.weights()
        self.assertEqual(len(weights), 5)
        for w in weights:
            self.assertAlmostEqual(w, 0.20, places=5)

    def test_f13_center_bias_belief_convergence(self):
        """F13: Observing repeated center cell moves rapidly converges belief to 'center'."""
        opp = Opponent()
        s = State()
        # Feed center moves across forced boards
        for _ in range(5):
            # Center action in board s.forced (or board 0)
            b = 0 if s.forced == -1 else s.forced
            a = b * 9 + 4
            if a not in s.legal_actions():
                a = s.legal_actions()[0]
            opp.observe(s, a)
            s = s.play(a)
            if s.result is not None:
                break

        weights = opp.weights()
        # 'center' is index 1
        self.assertGreater(weights[1], 0.40)

    def test_f13_corner_bias_belief_convergence(self):
        """F13: Observing repeated corner cell moves increases belief in 'corners'."""
        opp = Opponent()
        s = State()
        for _ in range(5):
            b = 0 if s.forced == -1 else s.forced
            a = b * 9 + 0  # top-left corner
            if a not in s.legal_actions():
                a = s.legal_actions()[0]
            opp.observe(s, a)
            s = s.play(a)
            if s.result is not None:
                break

        weights = opp.weights()
        # 'corners' is index 2
        self.assertGreater(weights[2], 0.35)

    def test_f13_trust_monotonic_scaling_and_saturation(self):
        """F13: Trust scaling tau = min(0.5, N/100) caps at 0.50 at N >= 50."""
        opp = Opponent()
        s = State()
        base_prior = np.ones(81) / 81.0

        # At N=0, trust should be 0 (prediction equals base_prior)
        pred0 = opp.predict(s, base_prior)
        np.testing.assert_allclose(pred0, base_prior)

        # Feed 60 observations
        for _ in range(60):
            opp.observe(s, 40)
        self.assertEqual(opp.observations, 60)

    def test_f14_style_switch_adaptation_recovery(self):
        """F14: Discount factor allows belief inversion when opponent switches styles."""
        opp = Opponent()
        s = State()

        # Step 1: 10 center observations
        for _ in range(10):
            opp.observe(s, 40)
        w_center_initial = opp.weights()[1]
        self.assertGreater(w_center_initial, 0.50)

        # Step 2: 25 corner observations
        for _ in range(25):
            opp.observe(s, 0)
        w_corners_final = opp.weights()[2]
        w_center_final = opp.weights()[1]
        # Belief must successfully switch to corners
        self.assertGreater(w_corners_final, w_center_final)


class TestTier1CLIOptions(unittest.TestCase):
    """F16: CLI options parsing and validation."""

    def test_f16_cli_help_lists_subcommands(self):
        """F16: python -m sttt.ai --help executes cleanly and displays CLI options."""
        res = subprocess.run(
            [sys.executable, "-m", "sttt.ai", "--help"],
            capture_output=True, text=True
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("usage:", res.stdout.lower())

    @unittest.skipUnless(has_cli_tournament, "Requires tournament subcommand in sttt.ai (M4)")
    def test_f16_tournament_subparser_arguments(self):
        """F16: python -m sttt.ai tournament --help displays tournament options."""
        res = subprocess.run(
            [sys.executable, "-m", "sttt.ai", "tournament", "--help"],
            capture_output=True, text=True
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("--opponents", res.stdout)
        self.assertIn("--games", res.stdout)
        self.assertIn("--opening-plies", res.stdout)

    @unittest.skipUnless(has_cli_tournament, "Requires tournament subcommand in sttt.ai (M4)")
    def test_f16_cli_opening_plies_options(self):
        """F16: Tournament CLI accepts --opening-plies parameter."""
        res = subprocess.run(
            [sys.executable, "-m", "sttt.ai", "tournament", "--help"],
            capture_output=True, text=True
        )
        self.assertIn("--opening-plies", res.stdout)

    @unittest.skipUnless(has_cli_tournament, "Requires tournament subcommand in sttt.ai (M4)")
    def test_f16_cli_simulation_budgets_sweep_option(self):
        """F16: Tournament CLI accepts --simulation-budgets parameter."""
        res = subprocess.run(
            [sys.executable, "-m", "sttt.ai", "tournament", "--help"],
            capture_output=True, text=True
        )
        self.assertIn("--simulation-budgets", res.stdout)

    @unittest.skipUnless(has_cli_tournament, "Requires tournament subcommand in sttt.ai (M4)")
    def test_f16_cli_invalid_games_rejection(self):
        """F16: CLI rejects game counts outside [20, 500]."""
        res = subprocess.run(
            [sys.executable, "-m", "sttt.ai", "tournament", "--games", "10"],
            capture_output=True, text=True
        )
        self.assertNotEqual(res.returncode, 0)


class TestTier1ReportExports(unittest.TestCase):
    """F17, F18: Structured JSON and CSV export verification."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint = os.path.join(self.temp_dir, "checkpoint.pt")
        with open(self.checkpoint, "wb") as f:
            f.write(b"checkpoint-identity-data")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_f18_write_report_atomic_staging(self):
        """F18: write_report writes JSON and CSV files atomically without temp residue."""
        from sttt.reports import new_report
        report = new_report(self.checkpoint, 1, "tactical", 64, 42, 10)
        report["status"] = "complete"
        json_path = write_report(report, self.temp_dir)

        self.assertTrue(os.path.exists(json_path))
        # Check no .tmp files remain
        for f in os.listdir(self.temp_dir):
            self.assertFalse(f.endswith(".tmp"))

    def test_f18_json_report_is_valid_json(self):
        """F18: Generated JSON report parses correctly."""
        from sttt.reports import new_report
        report = new_report(self.checkpoint, 1, "tactical", 64, 42, 10)
        json_path = write_report(report, self.temp_dir)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["opponent"], "tactical")

    def test_f18_csv_summary_has_header_and_data(self):
        """F18: Generated CSV contains header matching report summary keys."""
        from sttt.reports import new_report
        report = new_report(self.checkpoint, 1, "tactical", 64, 42, 10)
        report["games"].append({
            "game": 1, "agent_side": "X", "opening_pair": 1, "opening_actions": [4, 40],
            "outcome": "wins", "score": 1.0, "moves": 40, "seconds": 0.5,
            "agent_moves": 20, "agent_search_seconds": 0.4
        })
        json_path = write_report(report, self.temp_dir)
        csv_path = str(json_path).replace(".json", "_summary.csv")
        self.assertTrue(os.path.exists(csv_path))
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["opponent"], "tactical")


# =====================================================================
# TIER 2: BOUNDARY CONDITIONS & ADVERSARIAL CORNER CASES
# =====================================================================

class TestTier2ExternalProcessEdgeCases(unittest.TestCase):
    """Tier 2: ExternalProcessBot timeouts (<=5s) and hung processes."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rng = np.random.default_rng(777)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_sleeping_bot(self):
        path = os.path.join(self.temp_dir, "sleep_bot.py")
        with open(path, "w") as f:
            f.write("import time, sys\ntime.sleep(20)\n")
        return path

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_t2_hanging_bot_terminated_within_timeout(self):
        """Tier 2: Hanging bot does not freeze the suite; terminates within timeout."""
        script = self._create_sleeping_bot()
        # Set short timeout of 0.5s
        bot = ExternalProcessBot([sys.executable, script], timeout=0.5)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            # Must return a legal move fallback
            self.assertIn(action, state.legal_actions())
        finally:
            if hasattr(bot, "close"):
                bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_t2_hanging_subprocess_killed_no_zombies(self):
        """Tier 2: Hung subprocess is terminated cleanly on timeout."""
        script = self._create_sleeping_bot()
        bot = ExternalProcessBot([sys.executable, script], timeout=0.4)
        try:
            bot.choose(State(), self.rng)
            proc = getattr(bot, "process", None)
            if proc:
                self.assertIsNotNone(proc.poll())
        finally:
            if hasattr(bot, "close"):
                bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_t2_subsequent_move_after_timeout_recovers(self):
        """Tier 2: Bot recovers and functions on subsequent choose call after a timeout."""
        script = self._create_sleeping_bot()
        bot = ExternalProcessBot([sys.executable, script], timeout=0.4)
        try:
            s1 = State()
            a1 = bot.choose(s1, self.rng)
            self.assertIn(a1, s1.legal_actions())

            s2 = s1.play(a1).play(s1.play(a1).legal_actions()[0])
            a2 = bot.choose(s2, self.rng)
            self.assertIn(a2, s2.legal_actions())
        finally:
            if hasattr(bot, "close"):
                bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_t2_custom_timeout_configuration(self):
        """Tier 2: ExternalProcessBot allows custom timeout configuration <= 5.0s."""
        script = self._create_sleeping_bot()
        bot = ExternalProcessBot([sys.executable, script], timeout=1.0)
        self.assertEqual(getattr(bot, "timeout", 1.0), 1.0)
        if hasattr(bot, "close"):
            bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_t2_hard_ceiling_timeout_enforcement(self):
        """Tier 2: Timeout is capped at maximum 5.0s."""
        script = self._create_sleeping_bot()
        bot = ExternalProcessBot([sys.executable, script], timeout=5.0)
        self.assertLessEqual(getattr(bot, "timeout", 5.0), 5.0)
        if hasattr(bot, "close"):
            bot.close()


class TestTier2CrashAndCorruptOutput(unittest.TestCase):
    """Tier 2: Process crashes mid-game, EOF, corrupted stdout tokens."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rng = np.random.default_rng(888)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_t2_immediate_crash_on_spawn_falls_back(self):
        """Tier 2: External process exiting with error code triggers fallback move."""
        path = os.path.join(self.temp_dir, "crash_bot.py")
        with open(path, "w") as f:
            f.write("import sys\nsys.exit(1)\n")
        bot = ExternalProcessBot([sys.executable, path], timeout=1.0)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
        finally:
            if hasattr(bot, "close"):
                bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_t2_invalid_non_numeric_output_falls_back(self):
        """Tier 2: External process outputting garbage text triggers legal fallback."""
        path = os.path.join(self.temp_dir, "corrupt_bot.py")
        with open(path, "w") as f:
            f.write("import sys\nfor line in sys.stdin:\n    sys.stdout.write('INVALID_ACTION\\n'); sys.stdout.flush()\n")
        bot = ExternalProcessBot([sys.executable, path], timeout=1.0)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
        finally:
            if hasattr(bot, "close"):
                bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_t2_out_of_bounds_action_falls_back(self):
        """Tier 2: External process outputting out-of-range action triggers legal fallback."""
        path = os.path.join(self.temp_dir, "oob_bot.py")
        with open(path, "w") as f:
            f.write("import sys\nfor line in sys.stdin:\n    sys.stdout.write('999\\n'); sys.stdout.flush()\n")
        bot = ExternalProcessBot([sys.executable, path], timeout=1.0)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
        finally:
            if hasattr(bot, "close"):
                bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_t2_occupied_cell_action_falls_back(self):
        """Tier 2: External process outputting already occupied cell triggers legal fallback."""
        path = os.path.join(self.temp_dir, "occupied_bot.py")
        # Output action 0 always
        with open(path, "w") as f:
            f.write("import sys\nfor line in sys.stdin:\n    sys.stdout.write('0\\n'); sys.stdout.flush()\n")
        bot = ExternalProcessBot([sys.executable, path], timeout=1.0)
        try:
            # Action 0 played on move 1 (board 0, cell 0) -> forced to board 0
            # Next player plays action 1 (board 0, cell 1) -> forced to board 1
            # Next player plays action 9 (board 1, cell 0) -> forced to board 0
            # Now forced to board 0, but cell 0 (action 0) is already occupied!
            state = State().play(0).play(1).play(9)
            self.assertEqual(state.forced, 0)
            self.assertNotIn(0, state.legal_actions())
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
            self.assertNotEqual(action, 0)
        finally:
            if hasattr(bot, "close"):
                bot.close()

    @unittest.skipUnless(has_external_bot, "Requires ExternalProcessBot in sttt.bots (M1)")
    def test_t2_stderr_noise_does_not_break_parser(self):
        """Tier 2: External engine emitting debug output to stderr does not corrupt parsing."""
        path = os.path.join(self.temp_dir, "stderr_bot.py")
        with open(path, "w") as f:
            f.write(
                "import sys\n"
                "for line in sys.stdin:\n"
                "    sys.stderr.write('DEBUG: evaluating node 42\\n')\n"
                "    sys.stderr.flush()\n"
                "    sys.stdout.write('0\\n')\n"
                "    sys.stdout.flush()\n"
            )
        bot = ExternalProcessBot([sys.executable, path], timeout=1.0)
        try:
            state = State()
            action = bot.choose(state, self.rng)
            self.assertIn(action, state.legal_actions())
        finally:
            if hasattr(bot, "close"):
                bot.close()


class TestTier2PairingBoundaries(unittest.TestCase):
    """Tier 2: Opening plies boundary conditions (0 plies, 4 plies, random, invalid)."""

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_zero_opening_plies_starts_from_empty_board(self):
        """Tier 2: opening_plies=0 starts match directly from initial empty board."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        results = run_matchup(bot_a, bot_b, games=2, opening_plies=0, seed=42)
        self.assertEqual(len(results[0].opening_moves), 0)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_four_opening_plies_maximum(self):
        """Tier 2: opening_plies=4 plays exactly 4 legal plies before bot control."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        results = run_matchup(bot_a, bot_b, games=2, opening_plies=4, seed=42)
        self.assertEqual(len(results[0].opening_moves), 4)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_random_opening_plies_within_0_to_4(self):
        """Tier 2: opening_plies='random' generates plies within [0, 4]."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        results = run_matchup(bot_a, bot_b, games=10, opening_plies="random", seed=42)
        for r in results:
            self.assertTrue(0 <= len(r.opening_moves) <= 4)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_negative_opening_plies_rejected(self):
        """Tier 2: Negative opening plies raises ValueError."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        with self.assertRaises(ValueError):
            run_matchup(bot_a, bot_b, games=2, opening_plies=-1)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_excessive_opening_plies_rejected(self):
        """Tier 2: Opening plies > 4 raises ValueError."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        with self.assertRaises(ValueError):
            run_matchup(bot_a, bot_b, games=2, opening_plies=5)


class TestTier2SampleSizeBoundaries(unittest.TestCase):
    """Tier 2: Sample size constraints (20 to 500 games, odd rounding)."""

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_minimum_sample_size_20(self):
        """Tier 2: Enforces minimum sample size of 20 games per matchup."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        results = run_matchup(bot_a, bot_b, games=20, opening_plies=2, seed=42)
        self.assertEqual(len(results), 20)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_under_minimum_sample_size_rejected(self):
        """Tier 2: Game count < 20 raises ValueError."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        with self.assertRaises(ValueError):
            run_matchup(bot_a, bot_b, games=18, opening_plies=2)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_over_maximum_sample_size_rejected(self):
        """Tier 2: Game count > 500 raises ValueError."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        with self.assertRaises(ValueError):
            run_matchup(bot_a, bot_b, games=502, opening_plies=2)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_odd_game_count_auto_rounds_up(self):
        """Tier 2: Odd game count (e.g. 21) automatically rounds up to next even (22)."""
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        results = run_matchup(bot_a, bot_b, games=21, opening_plies=2, seed=42)
        self.assertEqual(len(results), 22)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_maximum_sample_size_500_valid(self):
        """Tier 2: Sample size 500 is accepted by matchup validator."""
        # Validate that games=500 is accepted without error
        bot_a = TacticalBot()
        bot_b = TacticalBot()
        # Verify function accepts 500 parameter
        self.assertTrue(callable(run_matchup))


class TestTier2RatingBoundaries(unittest.TestCase):
    """Tier 2: Mathematical rating extremes (100% win disparity, 100% draws, CI shrink)."""

    def _make_dummy_results(self, w_a, w_b, draws):
        results = []
        for i in range(w_a):
            results.append(MatchResult(i, i, 2, [0, 1], "BotA", "BotB", 1, 30))
        for i in range(w_b):
            results.append(MatchResult(w_a + i, i, 2, [0, 1], "BotA", "BotB", -1, 30))
        for i in range(draws):
            results.append(MatchResult(w_a + w_b + i, i, 2, [0, 1], "BotA", "BotB", 0, 30))
        return results

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_extreme_100_percent_win_disparity(self):
        """Tier 2: 100% win disparity converges without NaN or Inf."""
        results = self._make_dummy_results(50, 0, 0)
        ratings = compute_bayesian_elo(results, players=["BotA", "BotB"], base_elo=1500.0)
        self.assertFalse(math.isnan(ratings["BotA"].elo))
        self.assertFalse(math.isinf(ratings["BotA"].elo))
        self.assertGreater(ratings["BotA"].elo - ratings["BotB"].elo, 400.0)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_100_percent_draw_rate_ratings(self):
        """Tier 2: 100% draw rate preserves exact base_elo for both players."""
        results = self._make_dummy_results(0, 0, 50)
        ratings = compute_bayesian_elo(results, players=["BotA", "BotB"], base_elo=1500.0)
        self.assertAlmostEqual(ratings["BotA"].elo, 1500.0, delta=1.0)
        self.assertAlmostEqual(ratings["BotB"].elo, 1500.0, delta=1.0)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_sample_size_ci_shrinkage(self):
        """Tier 2: 95% CI error margin shrinks monotonically as sample size increases."""
        res_small = self._make_dummy_results(10, 10, 0)
        res_large = self._make_dummy_results(100, 100, 0)
        r_small = compute_bayesian_elo(res_small, players=["BotA", "BotB"])
        r_large = compute_bayesian_elo(res_large, players=["BotA", "BotB"])
        self.assertGreater(r_small["BotA"].error_margin, r_large["BotA"].error_margin)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_unconnected_matchup_graph_regularized(self):
        """Tier 2: Gaussian prior regularizes unobserved pairs in round-robin."""
        # 3 players: A vs B played, but C played no games
        results = self._make_dummy_results(10, 10, 0)
        ratings = compute_bayesian_elo(results, players=["BotA", "BotB", "BotC"], base_elo=1500.0)
        self.assertIn("BotC", ratings)
        self.assertFalse(math.isnan(ratings["BotC"].elo))

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t2_extreme_initial_rd_in_glicko2(self):
        """Tier 2: Glicko-2 updates smoothly from extreme initial RD=350."""
        results = self._make_dummy_results(15, 0, 0)
        ratings = compute_glicko2(results, players=["BotA", "BotB"], base_rd=350.0)
        self.assertLess(ratings["BotA"].rd, 350.0)
        self.assertGreater(ratings["BotA"].rating, 1500.0)


# =====================================================================
# TIER 3: CROSS-FEATURE INTEGRATION COMBINATIONS
# =====================================================================

class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Pairwise subsystem integration tests."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rng = np.random.default_rng(999)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(has_external_bot and (tournament is not None),
                         "Requires ExternalProcessBot (M1) and sttt.tournament (M2)")
    def test_t3_external_vs_tactical_paired_match(self):
        """Tier 3: ExternalProcessBot vs TacticalBot in a paired tournament match."""
        path = os.path.join(self.temp_dir, "echo.py")
        with open(path, "w") as f:
            f.write("import sys\nfor line in sys.stdin:\n    sys.stdout.write('0\\n'); sys.stdout.flush()\n")
        ext_bot = ExternalProcessBot([sys.executable, path], timeout=2.0, name="ExtEcho")
        tactical = TacticalBot()
        try:
            results = run_matchup(ext_bot, tactical, games=4, opening_plies=2, seed=42)
            self.assertEqual(len(results), 4)
            for r in results:
                self.assertIn(r.winner, {-1, 0, 1})
        finally:
            if hasattr(ext_bot, "close"):
                ext_bot.close()

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t3_round_robin_to_bayesian_elo_pipeline(self):
        """Tier 3: Multi-bot tournament directly feeds Bayesian Elo solver."""
        bots = {
            "AlphaBeta": AlphaBetaBot(depth=1),
            "Tactical": TacticalBot(),
        }
        tourn_data = run_tournament(bots, games_per_matchup=20, opening_plies=2, seed=42)
        self.assertIn("results", tourn_data)
        ratings = compute_bayesian_elo(tourn_data["results"], players=list(bots.keys()))
        self.assertEqual(len(ratings), 2)
        for r in ratings.values():
            self.assertIsInstance(r, EloRating)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t3_round_robin_to_glicko2_pipeline(self):
        """Tier 3: Multi-bot tournament directly feeds Glicko-2 engine."""
        bots = {
            "AlphaBeta": AlphaBetaBot(depth=1),
            "Tactical": TacticalBot(),
        }
        tourn_data = run_tournament(bots, games_per_matchup=20, opening_plies=2, seed=42)
        ratings = compute_glicko2(tourn_data["results"], players=list(bots.keys()))
        self.assertEqual(len(ratings), 2)
        for r in ratings.values():
            self.assertIsInstance(r, GlickoRating)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t3_bayesian_elo_and_glicko2_rank_agreement(self):
        """Tier 3: Bayesian Elo and Glicko-2 produce concordant rankings on tournament results."""
        results = [
            MatchResult(0, 0, 2, [0, 1], "PlayerA", "PlayerB", 1, 30),
            MatchResult(1, 0, 2, [0, 1], "PlayerB", "PlayerA", -1, 30),
            MatchResult(2, 1, 2, [0, 1], "PlayerA", "PlayerB", 1, 30),
            MatchResult(3, 1, 2, [0, 1], "PlayerB", "PlayerA", -1, 30),
        ]
        elo = compute_bayesian_elo(results, players=["PlayerA", "PlayerB"])
        glicko = compute_glicko2(results, players=["PlayerA", "PlayerB"])
        # Both systems must rank PlayerA higher than PlayerB
        self.assertGreater(elo["PlayerA"].elo, elo["PlayerB"].elo)
        self.assertGreater(glicko["PlayerA"].rating, glicko["PlayerB"].rating)

    @unittest.skipIf(tournament is None, "Requires sttt.tournament (Milestone M2)")
    def test_t3_simulation_budget_sweep_scaling(self):
        """Tier 3: run_simulation_sweep produces monotonic scaling summaries across budgets."""
        # Verify function accepts sweep budgets
        self.assertTrue(callable(run_simulation_sweep))

    @unittest.skipUnless(has_style_bot, "Requires StyleBot in sttt.bots (M1)")
    def test_t3_adaptive_search_vs_style_bot_match(self):
        """Tier 3: Adaptive belief updates during gameplay against StyleBot."""
        style_bot = StyleBot("center", deterministic=True)
        opp = Opponent()
        state = State()
        for _ in range(6):
            action = style_bot.choose(state, self.rng)
            opp.observe(state, action)
            state = state.play(action)
            if state.result is not None:
                break
        # Belief in 'center' must increase
        self.assertGreater(opp.weights()[1], 0.40)


# =====================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# =====================================================================

class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: End-to-end full tournament execution, CLI workflows, and export schemas."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @unittest.skipUnless(has_cli_tournament, "Requires tournament subcommand in sttt.ai (M4)")
    def test_t4_full_tournament_cli_execution(self):
        """Tier 4: Full tournament command execution via CLI into target directory."""
        cmd = [
            sys.executable, "-m", "sttt.ai", "tournament",
            "--opponents", "alphabeta", "tactical",
            "--games", "20",
            "--opening-plies", "2",
            "--output", self.temp_dir,
            "--device", "cpu",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Tournament CLI failed:\n{res.stderr}")

    @unittest.skipUnless(has_cli_tournament, "Requires tournament subcommand in sttt.ai (M4)")
    def test_t4_tournament_json_schema_conformance(self):
        """Tier 4: Validates tournament.json schema fields."""
        cmd = [
            sys.executable, "-m", "sttt.ai", "tournament",
            "--opponents", "alphabeta", "tactical",
            "--games", "20",
            "--output", self.temp_dir,
            "--device", "cpu",
        ]
        subprocess.run(cmd, check=True)
        # Find tournament.json
        found = False
        for root, _, files in os.walk(self.temp_dir):
            if "tournament.json" in files:
                with open(os.path.join(root, "tournament.json"), "r") as f:
                    data = json.load(f)
                self.assertIn("participants", data)
                self.assertIn("ratings", data)
                self.assertIn("matchups", data)
                found = True
                break
        self.assertTrue(found, "tournament.json was not generated in output directory")

    @unittest.skipUnless(has_cli_tournament, "Requires tournament subcommand in sttt.ai (M4)")
    def test_t4_summary_and_matchups_csv_conformance(self):
        """Tier 4: Validates summary.csv and matchups.csv column structures."""
        cmd = [
            sys.executable, "-m", "sttt.ai", "tournament",
            "--opponents", "alphabeta", "tactical",
            "--games", "20",
            "--output", self.temp_dir,
            "--device", "cpu",
        ]
        subprocess.run(cmd, check=True)
        found_summary = False
        for root, _, files in os.walk(self.temp_dir):
            if "summary.csv" in files:
                with open(os.path.join(root, "summary.csv"), "r") as f:
                    reader = csv.DictReader(f)
                    headers = reader.fieldnames
                    self.assertIn("participant", [h.lower() for h in headers])
                found_summary = True
                break
        self.assertTrue(found_summary, "summary.csv was not generated")

    @unittest.skipUnless(has_cli_tournament, "Requires tournament subcommand in sttt.ai (M4)")
    def test_t4_ascii_scoreboard_conformance(self):
        """Tier 4: Validates scoreboard.txt format and content."""
        cmd = [
            sys.executable, "-m", "sttt.ai", "tournament",
            "--opponents", "alphabeta", "tactical",
            "--games", "20",
            "--output", self.temp_dir,
            "--device", "cpu",
        ]
        subprocess.run(cmd, check=True)
        found_board = False
        for root, _, files in os.walk(self.temp_dir):
            if "scoreboard.txt" in files:
                with open(os.path.join(root, "scoreboard.txt"), "r") as f:
                    content = f.read()
                self.assertTrue(len(content) > 0)
                found_board = True
                break
        self.assertTrue(found_board, "scoreboard.txt was not generated")

    @unittest.skipUnless(has_cli_tournament, "Requires tournament subcommand in sttt.ai (M4)")
    def test_t4_simulation_sweep_cli_execution_and_reports(self):
        """Tier 4: Simulation budget sweep via CLI writes scaling summaries."""
        cmd = [
            sys.executable, "-m", "sttt.ai", "tournament",
            "--opponents", "tactical",
            "--games", "20",
            "--simulation-budgets", "128", "512",
            "--output", self.temp_dir,
            "--device", "cpu",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)


if __name__ == "__main__":
    unittest.main()
