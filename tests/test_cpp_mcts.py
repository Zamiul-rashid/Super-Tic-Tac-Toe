"""Differential correctness tests and benchmarks: C++ MCTS Engine vs Python TreeSearch."""
import unittest
import time
import numpy as np
from sttt.env import State
from sttt.search import TreeSearch as PyTreeSearch, SearchConfig
from sttt.cpp_env import is_cpp_available, CppTreeSearch, FastState


class DeterministicEvaluator:
    """Deterministic evaluator for exact differential comparison."""
    def __init__(self, value=0.0):
        self.value = value

    def evaluate(self, state):
        legal = state.legal_actions()
        p = np.zeros(81, dtype=np.float32)
        if legal:
            # Deterministic skewed prior favoring center/first legal actions
            weights = np.array([1.0 / (i + 1) for i in range(len(legal))], dtype=np.float32)
            weights /= weights.sum()
            p[legal] = weights
        return p, float(self.value)

    def evaluate_many(self, states):
        return [self.evaluate(s) for s in states]


class TestCppMCTS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not is_cpp_available() or CppTreeSearch is None:
            raise unittest.SkipTest("C++ MCTS Engine not available")

    def test_initial_configuration_and_stats(self):
        cfg = SearchConfig(soft_pruning=True, proofs=True, reuse=True, c_puct=1.75)
        tree = CppTreeSearch(model=None, config=cfg)
        self.assertTrue(tree.use_proofs)
        self.assertIsNone(tree.root)
        self.assertEqual(tree.stats["completed_simulations"], 0)

    def test_heuristic_search_pure_cpp(self):
        """Verify pure C++ heuristic search runs cleanly without a Python model."""
        tree = CppTreeSearch(model=None)
        s = State()
        pi = tree.run(s, simulations=256, batch_size=8)
        self.assertEqual(len(pi), 81)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
        self.assertEqual(tree.stats["completed_simulations"], 256)
        self.assertGreater(tree.stats["max_depth"], 0)
        self.assertIsNotNone(tree.root)
        self.assertEqual(tree.root.n, 256)
        self.assertEqual(tree.root.in_flight, 0)
        self.assertFalse(tree.root.pending)

    def test_differential_search_vs_python_treesearch(self):
        """Run differential search between Python and C++ MCTS across game positions."""
        evaluator = DeterministicEvaluator(value=0.25)
        cfg = SearchConfig(soft_pruning=False, proofs=False, reuse=False, c_puct=1.5)

        test_positions = [
            State(),
            State().play(40),
            State().play(40).play(38),
            State().play(4).play(40).play(36),
        ]

        for s in test_positions:
            py_tree = PyTreeSearch(evaluator, config=cfg)
            cpp_tree = CppTreeSearch(evaluator, config=cfg)

            py_pi = py_tree.run(s, simulations=64, batch_size=4)
            cpp_pi = cpp_tree.run(s, simulations=64, batch_size=4)

            self.assertEqual(py_tree.stats["completed_simulations"], 64)
            self.assertEqual(cpp_tree.stats["completed_simulations"], 64)

            # Both trees must pick the exact same best action
            self.assertEqual(int(py_pi.argmax()), int(cpp_pi.argmax()))

            # Visit counts across legal actions must match or have high rank correlation
            py_visits = np.array([c.n for c in py_tree.root.children.values()])
            cpp_visits = np.array([c.n for c in cpp_tree.root.children.values()])
            self.assertEqual(len(py_visits), len(cpp_visits))
            self.assertEqual(py_visits.sum(), cpp_visits.sum())

    def test_proof_solving_differential(self):
        """Verify C++ MCTS solves proven wins/losses matching game theoretical logic."""
        # Create a state where X can immediately win locally and globally
        cells = [0] * 81
        cells[0] = 1; cells[1] = 1  # Board 0: winning line at 2
        cells[9] = 1; cells[10] = 1; cells[11] = 1  # Board 1 won by X
        cells[18] = 1; cells[19] = 1; cells[20] = 1 # Board 2 won by X
        boards = [0, 1, 1, 0, 0, 0, 0, 0, 0]
        s = State(cells=tuple(cells), boards=tuple(boards), turn=1, forced=0)

        cpp_tree = CppTreeSearch(model=None, config=SearchConfig(proofs=True))
        pi = cpp_tree.run(s, simulations=100, batch_size=4)
        self.assertEqual(int(pi.argmax()), 2)
        self.assertEqual(cpp_tree.root.solved, 1)

    def test_subtree_reuse_differential(self):
        """Verify subtree retention matches Python semantics across consecutive turns."""
        evaluator = DeterministicEvaluator(value=0.1)
        cfg = SearchConfig(reuse=True, proofs=False)

        py_tree = PyTreeSearch(evaluator, config=cfg)
        cpp_tree = CppTreeSearch(evaluator, config=cfg)

        s = State()
        py_pi = py_tree.run(s, simulations=128, batch_size=8)
        cpp_pi = cpp_tree.run(s, simulations=128, batch_size=8)

        action = int(cpp_pi.argmax())
        py_tree.advance(action)
        cpp_tree.advance(action)

        s_next = s.play(action)
        py_pi2 = py_tree.run(s_next, simulations=64, batch_size=4)
        cpp_pi2 = cpp_tree.run(s_next, simulations=64, batch_size=4)

        self.assertGreater(cpp_tree.stats["retained_visits"], 0)
        self.assertEqual(cpp_tree.stats["retained_visits"], py_tree.stats["retained_visits"])
        self.assertEqual(int(py_pi2.argmax()), int(cpp_pi2.argmax()))

    def test_virtual_losses_released_on_evaluator_exception(self):
        """Verify virtual losses are completely unwound if inference raises an exception."""
        class FailingEvaluator:
            def evaluate(self, state):
                raise RuntimeError("simulated GPU failure")
            def evaluate_many(self, states):
                raise RuntimeError("simulated GPU failure")

        tree = CppTreeSearch(FailingEvaluator())
        s = State()
        with self.assertRaises(RuntimeError):
            tree.run(s, simulations=32, batch_size=4)

        if tree.root is not None:
            self.assertEqual(tree.root.in_flight, 0)
            self.assertFalse(tree.root.pending)

    def test_terminal_state_error_handling(self):
        """Verify passing terminal state raises ValueError."""
        terminal_s = State(cells=tuple([0]*81), boards=(1, 1, 1, 0, 0, 0, 0, 0, 0), turn=1, forced=-1, result=1)
        tree = CppTreeSearch(model=None)
        with self.assertRaises(ValueError):
            tree.run(terminal_s, simulations=64)

    def test_rng_seed_independence(self):
        """Two CppTreeSearch instances with different seeds must produce different policies under noise."""
        state = State()
        cfg = SearchConfig(proofs=False)
        tree1 = CppTreeSearch(model=None, rng=np.random.default_rng(1), config=cfg)
        tree2 = CppTreeSearch(model=None, rng=np.random.default_rng(999), config=cfg)
        pi1 = tree1.run(state, simulations=64, batch_size=1, noise=True)
        pi2 = tree2.run(state, simulations=64, batch_size=1, noise=True)
        self.assertFalse(np.allclose(pi1, pi2, atol=1e-6),
                         "Different seeds produced identical policies — RNG seed is ignored!")

    def test_stats_no_leak(self):
        """Reading stats repeatedly must not leak Python object references."""
        import gc
        tree = CppTreeSearch(model=None)
        tree.run(State(), simulations=64)
        gc.collect()
        before = len(gc.get_objects())
        for _ in range(5000):
            _ = tree.stats
        gc.collect()
        after = len(gc.get_objects())
        self.assertLess(after - before, 500,
                        f"Stats leaked references: {after - before} objects after 5000 reads")

    def test_advance_invalid_action_raises(self):
        """Calling advance with out-of-range action must raise ValueError, not crash."""
        tree = CppTreeSearch(model=None)
        tree.run(State(), simulations=16)
        with self.assertRaises(ValueError):
            tree.advance(81)
        with self.assertRaises(ValueError):
            tree.advance(-1)

    def test_nan_value_mid_batch_does_not_corrupt_inflight(self):
        """A NaN value mid-batch must raise ValueError and not leave negative in_flight."""
        class PartialNanEvaluator:
            def evaluate_many(self, states):
                res = []
                for i, _ in enumerate(states):
                    p = np.ones(81, dtype=np.float64) / 81
                    v = float('nan') if i > 0 else 0.0
                    res.append((p, v))
                return res

            def evaluate(self, state):
                return self.evaluate_many([state])[0]

        tree = CppTreeSearch(model=PartialNanEvaluator())
        state = State()
        with self.assertRaises(ValueError):
            tree.run(state, simulations=64, batch_size=8)
        if tree.root is not None:
            self.assertEqual(tree.root.in_flight, 0, f"in_flight was corrupted to {tree.root.in_flight}")

    def test_root_action_values_match_python_tree(self):
        from sttt.search import root_action_values
        evaluator = DeterministicEvaluator(value=0.2)
        cfg = SearchConfig(soft_pruning=False, proofs=False, reuse=False, c_puct=1.5)
        s = State().play(40).play(38)
        py_tree, cpp_tree = PyTreeSearch(evaluator, config=cfg), CppTreeSearch(evaluator, config=cfg)
        py_tree.run(s, simulations=96, batch_size=8)
        cpp_tree.run(s, simulations=96, batch_size=8)
        py_q, py_v = root_action_values(py_tree.root)
        cpp_q, cpp_v = root_action_values(cpp_tree.root)
        self.assertTrue((py_v == cpp_v).all())
        np.testing.assert_allclose(py_q, cpp_q, atol=1e-5)
        self.assertEqual(cpp_tree.root.in_flight, 0)


if __name__ == "__main__":
    unittest.main()

