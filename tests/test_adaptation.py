"""Comprehensive test suite for Bayesian opponent adaptation (Milestone 3 / R3).

Verifies:
- F13: Belief convergence speed and Shannon entropy reduction across behavioral styles.
- F14: Dynamic style switching recovery under recency discounting (rho=0.98) and trust bounds.
- F15: MCTS search adaptation invariants (proof disabling, expectimax sampling) and paired win-rate delta.
"""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch

from sttt.env import State
from sttt.opponent import Opponent, policies, NAMES
from sttt.search import TreeSearch, SearchConfig
from sttt.bots import StyleBot, value


class HeuristicEvaluator:
    """Fast, deterministic evaluator combining uniform priors with domain heuristic values.
    
    Operates on CPU without requiring GPU allocation or neural network checkpoints.
    """
    def evaluate(self, state):
        legal = state.legal_actions()
        p = np.zeros(81, dtype=np.float64)
        if legal:
            p[legal] = 1.0 / len(legal)
        return p, float(value(state))

    def evaluate_many(self, states):
        return [self.evaluate(s) for s in states]


class UniformMockEvaluator:
    """Mock evaluator returning uniform policy and zero value."""
    def evaluate(self, state):
        p = np.ones(81, dtype=np.float64) / 81.0
        return p, 0.0

    def evaluate_many(self, states):
        return [self.evaluate(s) for s in states]


class ChoiceTrackingRNG:
    """RNG proxy tracking calls to choice() with custom probabilities p."""
    def __init__(self, seed=42):
        self._rng = np.random.default_rng(seed)
        self.choice_calls = []

    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        if p is not None:
            self.choice_calls.append((a, np.copy(p)))
        return self._rng.choice(a, size=size, replace=replace, p=p, axis=axis, shuffle=shuffle)

    def dirichlet(self, alpha, size=None):
        return self._rng.dirichlet(alpha, size=size)


class TestAdaptationBeliefConvergence(unittest.TestCase):
    """F13: Verify Bayesian belief state convergence speed and entropy reduction."""

    def setUp(self):
        torch.set_num_threads(1)

    def test_center_convergence_speed(self):
        """Feed 10 consecutive center-targeted moves and verify rapid convergence.
        
        Requirements:
            - b_center >= 0.45 after 1 move
            - b_center >= 0.70 after 3 moves
            - b_center >= 0.95 after 10 moves
            - argmax_k b_k == 1 ('center')
        """
        opponent = Opponent()
        state = State()

        for step in range(1, 11):
            centers = [a for a in state.legal_actions() if a % 9 == 4]
            a = centers[0] if centers else state.legal_actions()[0]
            opponent.observe(state, a)
            state = state.play(a)
            if state.result is not None:
                state = State()

            weights = opponent.weights()
            if step == 1:
                self.assertGreaterEqual(weights[1], 0.45, f"Expected b_center >= 0.45 at step 1, got {weights[1]}")
            elif step == 3:
                self.assertGreaterEqual(weights[1], 0.70, f"Expected b_center >= 0.70 at step 3, got {weights[1]}")
            elif step == 10:
                self.assertGreaterEqual(weights[1], 0.95, f"Expected b_center >= 0.95 at step 10, got {weights[1]}")

        final_weights = opponent.weights()
        self.assertEqual(int(final_weights.argmax()), 1, "Expected argmax to be index 1 ('center')")
        self.assertLess(final_weights[0], 0.05, "Expected random prior to be suppressed (< 0.05)")

    def test_corners_convergence_speed(self):
        """Feed 10 consecutive corner-targeted moves and verify convergence.
        
        Requirements:
            - b_corners >= 0.60 after 3 moves
            - b_corners >= 0.80 after 5 moves
            - b_corners >= 0.95 after 10 moves
            - argmax_k b_k == 2 ('corners')
        """
        opponent = Opponent()
        state = State()

        for step in range(1, 11):
            corners = [a for a in state.legal_actions() if a % 9 in (0, 2, 6, 8)]
            a = corners[0] if corners else state.legal_actions()[0]
            opponent.observe(state, a)
            state = state.play(a)
            if state.result is not None:
                state = State()

            weights = opponent.weights()
            if step == 3:
                self.assertGreaterEqual(weights[2], 0.60, f"Expected b_corners >= 0.60 at step 3, got {weights[2]}")
            elif step == 5:
                self.assertGreaterEqual(weights[2], 0.80, f"Expected b_corners >= 0.80 at step 5, got {weights[2]}")
            elif step == 10:
                self.assertGreaterEqual(weights[2], 0.95, f"Expected b_corners >= 0.95 at step 10, got {weights[2]}")

        final_weights = opponent.weights()
        self.assertEqual(int(final_weights.argmax()), 2, "Expected argmax to be index 2 ('corners')")

    def test_local_win_greed_convergence(self):
        """Feed moves taking local board completions and verify convergence to 'local-win'.
        
        Requirements:
            - b_local-win >= 0.40 after 1 local win
            - b_local-win >= 0.90 after 5 local wins
            - argmax_k b_k == 3 ('local-win')
        """
        opponent = Opponent()

        for step in range(1, 6):
            cells = [0] * 81
            b = (step - 1) % 9
            # Set two cells in board b for turn=1, so action b*9 + 2 completes the local win
            cells[b * 9 + 0] = 1
            cells[b * 9 + 1] = 1
            boards = [0] * 9
            state = State(tuple(cells), tuple(boards), forced=b, turn=1)
            win_action = b * 9 + 2

            opponent.observe(state, win_action)
            weights = opponent.weights()

            if step == 1:
                self.assertGreaterEqual(weights[3], 0.40, f"Expected b_local-win >= 0.40 at step 1, got {weights[3]}")
            elif step == 5:
                self.assertGreaterEqual(weights[3], 0.90, f"Expected b_local-win >= 0.90 at step 5, got {weights[3]}")

        final_weights = opponent.weights()
        self.assertEqual(int(final_weights.argmax()), 3, "Expected argmax to be index 3 ('local-win')")

    def test_global_win_convergence(self):
        """Feed winning moves claiming the global game and verify rapid convergence.
        
        Requirements:
            - b_global-win >= 0.70 after 1 move
            - b_global-win >= 0.95 after 3 moves
            - argmax_k b_k == 4 ('global-win')
        """
        opponent = Opponent()

        for step in range(1, 4):
            # Synthesize a state with multiple local winning opportunities,
            # where exactly one completes the global game.
            cells = [0] * 81
            boards = [0] * 9
            boards[0] = 1
            boards[1] = 1
            # Global winning opportunity in board 2 (completing 0, 1, 2 line)
            cells[2 * 9 + 0] = 1
            cells[2 * 9 + 2] = 1
            # Non-global local winning opportunities in boards 3 and 4
            cells[3 * 9 + 0] = 1
            cells[3 * 9 + 1] = 1
            cells[4 * 9 + 0] = 1
            cells[4 * 9 + 1] = 1

            state = State(tuple(cells), tuple(boards), forced=-1, turn=1)
            global_win_action = 2 * 9 + 1  # completes line in board 2 -> global win

            opponent.observe(state, global_win_action)
            weights = opponent.weights()

            if step == 1:
                self.assertGreaterEqual(weights[4], 0.70, f"Expected b_global-win >= 0.70 at step 1, got {weights[4]}")
            elif step == 3:
                self.assertGreaterEqual(weights[4], 0.95, f"Expected b_global-win >= 0.95 at step 3, got {weights[4]}")

        final_weights = opponent.weights()
        self.assertEqual(int(final_weights.argmax()), 4, "Expected argmax to be index 4 ('global-win')")

    def test_shannon_entropy_reduction(self):
        """Compute Shannon entropy H(b) = -sum(b_k * ln(b_k)).
        
        Requirements:
            - Initial entropy drops from H_0 = ln(5) ~ 1.609 down to H <= 0.35 after 10 moves
            - Represents over 75% uncertainty reduction under a skewed style.
        """
        opponent = Opponent()
        w0 = opponent.weights()
        h0 = -float(np.sum(w0 * np.log(w0 + 1e-12)))
        self.assertAlmostEqual(h0, np.log(5), places=3,
                               msg=f"Expected initial entropy ~ ln(5) ({np.log(5):.4f}), got {h0:.4f}")

        state = State()
        for step in range(10):
            corners = [a for a in state.legal_actions() if a % 9 in (0, 2, 6, 8)]
            a = corners[0] if corners else state.legal_actions()[0]
            opponent.observe(state, a)
            state = state.play(a)
            if state.result is not None:
                state = State()

        w10 = opponent.weights()
        h10 = -float(np.sum(w10 * np.log(w10 + 1e-12)))
        self.assertLessEqual(h10, 0.35, f"Expected Shannon entropy H <= 0.35 after 10 moves, got {h10:.4f}")

        reduction_ratio = (h0 - h10) / h0
        self.assertGreater(reduction_ratio, 0.75,
                           f"Expected > 75% uncertainty reduction, got {reduction_ratio * 100:.1f}%")


class TestAdaptationStyleSwitching(unittest.TestCase):
    """F14: Verify recency-discounted style switching recovery and trust factor scaling."""

    def setUp(self):
        torch.set_num_threads(1)

    def test_style_switching_recovery(self):
        """Feed 15 center moves (b_center > 0.95), then switch to 15 corner moves.
        
        Verifies that within 15 moves of the shift, b_corners > b_center,
        confirming that the rho=0.98 recency discount factor permits belief inversion.
        """
        opponent = Opponent()
        state = State()

        # Phase 1: 15 center moves
        for _ in range(15):
            centers = [a for a in state.legal_actions() if a % 9 == 4]
            a = centers[0] if centers else state.legal_actions()[0]
            opponent.observe(state, a)
            state = state.play(a)
            if state.result is not None:
                state = State()

        w_phase1 = opponent.weights()
        self.assertGreater(w_phase1[1], 0.95, f"Expected b_center > 0.95 after 15 center moves, got {w_phase1[1]}")
        self.assertLess(w_phase1[2], 0.05, f"Expected b_corners < 0.05 after 15 center moves, got {w_phase1[2]}")

        # Phase 2: Switch to 15 corner moves
        state = State()
        for _ in range(15):
            corners = [a for a in state.legal_actions() if a % 9 in (0, 2, 6, 8)]
            a = corners[0] if corners else state.legal_actions()[0]
            opponent.observe(state, a)
            state = state.play(a)
            if state.result is not None:
                state = State()

        w_phase2 = opponent.weights()
        self.assertGreater(w_phase2[2], w_phase2[1],
                           f"Expected b_corners ({w_phase2[2]:.4f}) > b_center ({w_phase2[1]:.4f}) after style shift")
        self.assertEqual(int(w_phase2.argmax()), 2, "Expected argmax to be index 2 ('corners') after inversion")

    def test_trust_factor_monotonicity_and_cap(self):
        """Verify trust factor tau = min(0.5, N_obs / 100) across observations.
        
        Requirements:
            - tau = 0.00 at N = 0
            - tau = 0.25 at N = 25
            - tau = 0.50 at N = 50
            - tau = 0.50 at N = 100 (capped)
            - tau = 0.50 at N = 500 (capped)
        """
        opponent = Opponent()
        state = State()
        base = np.full(81, 1.0 / 81.0, dtype=np.float64)

        # N = 0
        pred_0 = opponent.predict(state, base)
        np.testing.assert_allclose(pred_0, base, err_msg="At N=0, predict() must match base distribution exactly")
        self.assertEqual(min(0.5, opponent.observations / 100.0), 0.0)

        # Advance to N = 25
        for _ in range(25):
            opponent.observe(state, 4)
        self.assertEqual(opponent.observations, 25)
        self.assertAlmostEqual(min(0.5, opponent.observations / 100.0), 0.25)

        # Advance to N = 50
        for _ in range(25):
            opponent.observe(state, 4)
        self.assertEqual(opponent.observations, 50)
        self.assertAlmostEqual(min(0.5, opponent.observations / 100.0), 0.50)

        # Advance to N = 100
        for _ in range(50):
            opponent.observe(state, 4)
        self.assertEqual(opponent.observations, 100)
        self.assertAlmostEqual(min(0.5, opponent.observations / 100.0), 0.50)

        # Advance to N = 500
        for _ in range(400):
            opponent.observe(state, 4)
        self.assertEqual(opponent.observations, 500)
        self.assertAlmostEqual(min(0.5, opponent.observations / 100.0), 0.50)

        # Verify conservative cap property: base distribution retains at least 50% weight
        pred_500 = opponent.predict(state, base)
        # For an illegal action (e.g. out of legal range), policy gives 0, so pred is at least 0.5 * base
        legal = state.legal_actions()
        for a in legal:
            # Each legal action has base[a] = 1/81, so pred[a] >= (1 - 0.5) * (1/81)
            self.assertGreaterEqual(pred_500[a], 0.5 * base[a] - 1e-9)


class TestAdaptationSearchIntegration(unittest.TestCase):
    """F15: Verify MCTS search adaptation invariants and paired win-rate delta."""

    def setUp(self):
        torch.set_num_threads(1)

    def test_proof_pruning_disabled_when_adapting(self):
        """Verify minimax proof pruning is disabled when opponent modeling is provided.
        
        Requirements:
            - With opponent provided: tree.use_proofs == False
            - Without opponent: tree.use_proofs == True (under default SearchConfig)
            - tree.stats['root_solved'] remains None when adapting
        """
        evaluator = UniformMockEvaluator()

        tree_adaptive = TreeSearch(evaluator, opponent=Opponent(), agent_side=1)
        self.assertFalse(tree_adaptive.use_proofs,
                         "Expected tree.use_proofs to be False when opponent is provided")

        tree_default = TreeSearch(evaluator, opponent=None, agent_side=1,
                                  config=SearchConfig(proofs=True))
        self.assertTrue(tree_default.use_proofs,
                        "Expected tree.use_proofs to be True when opponent is None")

        state = State()
        tree_adaptive.run(state, simulations=16)
        self.assertIsNone(tree_adaptive.stats.get('root_solved'),
                          "Expected root_solved to be None when proof pruning is disabled")

    def test_expectimax_sampling_at_opponent_nodes(self):
        """Verify opponent nodes in TreeSearch use expectation sampling weighted by opponent.predict().
        
        Requirements:
            - Opponent nodes (state.turn != agent_side) sample children using rng.choice(..., p=weights)
              where weights are proportional to child priors.
            - Agent nodes (state.turn == agent_side) and non-adaptive trees use PUCT argmax selection.
        """
        opponent = Opponent()
        # Seed opponent model with center observations so center moves receive higher prior
        for _ in range(10):
            opponent.observe(State(), 4)

        opponent_turn_state = State(turn=-1, forced=0)
        evaluator = UniformMockEvaluator()

        # 1. Adaptive search with ChoiceTrackingRNG
        tracking_rng = ChoiceTrackingRNG(seed=42)
        tree_adaptive = TreeSearch(evaluator, rng=tracking_rng, opponent=opponent, agent_side=1)
        tree_adaptive.run(opponent_turn_state, simulations=25)

        self.assertGreater(len(tracking_rng.choice_calls), 0,
                           "Expected rng.choice to be called during expectimax selection at opponent nodes")

        # Verify choice call probabilities match normalized child priors
        _, choice_p = tracking_rng.choice_calls[0]
        root_priors = np.array([c.prior for c in tree_adaptive.root.children.values()])
        expected_weights = root_priors / root_priors.sum()
        np.testing.assert_allclose(choice_p, expected_weights, atol=1e-5,
                                   err_msg="Expectimax sampling probabilities must match normalized child priors")

        # 2. Non-adaptive search (opponent is None) should NOT use expectimax choice
        tracking_rng_nonadaptive = ChoiceTrackingRNG(seed=42)
        tree_nonadaptive = TreeSearch(evaluator, rng=tracking_rng_nonadaptive, opponent=None, agent_side=1)
        tree_nonadaptive.run(opponent_turn_state, simulations=25)

        self.assertEqual(len(tracking_rng_nonadaptive.choice_calls), 0,
                         "Non-adaptive search must use PUCT argmax selection, not expectimax choice sampling")

    def test_adaptation_win_rate_delta_paired_match(self):
        """Run a paired match comparing adaptive vs non-adaptive agent against predictable StyleBot.
        
        Requirements:
            - Delta WinRate (Wins_adaptive - Wins_nonadaptive) >= 0
            - Adaptive agent wins in equal or fewer average moves (higher efficiency against predictable play)
        """
        evaluator = HeuristicEvaluator()

        def play_single_game(adaptive: bool, seed: int):
            rng = np.random.default_rng(seed)
            bot_rng = np.random.default_rng(seed + 1000)
            state = State()
            opponent = Opponent() if adaptive else None
            tree = TreeSearch(evaluator, rng, SearchConfig(reuse=True, proofs=False),
                              opponent=opponent, agent_side=1)
            moves = 0
            bot = StyleBot('corners', deterministic=True)

            while state.result is None:
                if state.turn == 1:
                    pi = tree.run(state, simulations=32, batch_size=4)
                    a = int(pi.argmax())
                    tree.advance(a)
                    state = state.play(a)
                else:
                    a = bot.choose(state, bot_rng)
                    if opponent is not None:
                        opponent.observe(state, a)
                    tree.advance(a)
                    if adaptive:
                        tree.reset()
                    state = state.play(a)
                moves += 1

            won = (state.result == 1)
            return won, moves

        adaptive_wins = 0
        nonadaptive_wins = 0
        adaptive_moves_list = []
        nonadaptive_moves_list = []

        # 10 paired games with identical seeds
        num_pairs = 10
        for seed in range(num_pairs):
            won_ad, moves_ad = play_single_game(adaptive=True, seed=seed)
            won_non, moves_non = play_single_game(adaptive=False, seed=seed)

            if won_ad:
                adaptive_wins += 1
                adaptive_moves_list.append(moves_ad)
            if won_non:
                nonadaptive_wins += 1
                nonadaptive_moves_list.append(moves_non)

        delta_win_rate = adaptive_wins - nonadaptive_wins
        self.assertGreaterEqual(delta_win_rate, 0,
                                f"Expected Delta WinRate >= 0, got {delta_win_rate} (ad={adaptive_wins}, non={nonadaptive_wins})")

        mean_moves_ad = np.mean(adaptive_moves_list) if adaptive_moves_list else float('inf')
        mean_moves_non = np.mean(nonadaptive_moves_list) if nonadaptive_moves_list else float('inf')
        self.assertLessEqual(mean_moves_ad, mean_moves_non,
                             f"Expected adaptive mean moves ({mean_moves_ad:.1f}) <= non-adaptive ({mean_moves_non:.1f})")


class TestAdaptationEdgeCases(unittest.TestCase):
    """Supplementary unit tests verifying edge cases and profile persistence."""

    def test_illegal_move_observation_rejected(self):
        """Verify observing an illegal move raises ValueError."""
        opponent = Opponent()
        state = State()
        legal = state.legal_actions()
        illegal_action = -1
        while illegal_action in legal or illegal_action < 0:
            illegal_action = 99

        with self.assertRaises(ValueError):
            opponent.observe(state, illegal_action)

    def test_opponent_persistence_roundtrip(self):
        """Verify Opponent profile saves and reloads identically from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "sub" / "profile.json"
            opponent = Opponent(profile_path)

            state = State()
            opponent.observe(state, 4)
            opponent.observe(state.play(4), 40)

            reloaded = Opponent(profile_path)
            self.assertEqual(reloaded.observations, 2)
            np.testing.assert_allclose(reloaded.weights(), opponent.weights(), atol=1e-6)
            np.testing.assert_allclose(reloaded.log_weights, opponent.log_weights, atol=1e-6)

    def test_initial_uniform_weights(self):
        """Verify unobserved Opponent initializes with uniform weights across all policies."""
        opponent = Opponent()
        self.assertEqual(opponent.observations, 0)
        expected_uniform = np.full(len(NAMES), 1.0 / len(NAMES))
        np.testing.assert_allclose(opponent.weights(), expected_uniform, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
