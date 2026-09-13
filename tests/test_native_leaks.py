"""M1 regression tests: reference accounting and exception translation.

The stats dictionary is rebuilt on every ``.stats`` read, so a single leaked
reference per read compounds over a long training run. The plan's fixture is
20,000 discarded reads with garbage collection, asserting no growth
proportional to the read count.
"""
import gc
import unittest

from sttt.env import State
from sttt.search import SearchConfig
from sttt.cpp_env import is_cpp_available, CppTreeSearch


@unittest.skipUnless(is_cpp_available() and CppTreeSearch is not None,
                     "C++ MCTS Engine not available")
class TestStatsReferenceAccounting(unittest.TestCase):
    READS = 20_000

    def test_discarded_stats_reads_do_not_retain(self):
        tree = CppTreeSearch(model=None, config=SearchConfig(proofs=False))
        tree.run(State(), simulations=32, batch_size=4)

        gc.collect()
        baseline = len(gc.get_objects())

        for _ in range(self.READS):
            tree.stats            # built and immediately discarded

        gc.collect()
        grown = len(gc.get_objects()) - baseline

        # One leaked object per read would show up as ~20,000 here. Allow a
        # small constant for interpreter bookkeeping, but nothing proportional.
        self.assertLess(grown, self.READS // 100,
                        f"retained {grown} objects across {self.READS} discarded reads")

    def test_stats_dict_is_complete_and_fresh(self):
        tree = CppTreeSearch(model=None, config=SearchConfig(proofs=False))
        tree.run(State(), simulations=32, batch_size=4)
        expected = {"completed_simulations", "neural_positions", "inference_batches",
                    "max_depth", "hard_pruned_choices", "soft_rechecks",
                    "retained_visits", "root_solved",
                    "arena_nodes", "arena_capacity"}
        first = tree.stats
        self.assertEqual(set(first), expected)
        # A fresh dict each read: mutating one snapshot cannot corrupt the next.
        first["completed_simulations"] = -999
        self.assertNotEqual(tree.stats["completed_simulations"], -999)

    def test_each_read_returns_a_distinct_snapshot(self):
        """Consecutive reads must not alias one cached dict."""
        tree = CppTreeSearch(model=None, config=SearchConfig(proofs=False))
        tree.run(State(), simulations=16)
        a, b = tree.stats, tree.stats
        self.assertIsNot(a, b)
        self.assertEqual(a, b)


@unittest.skipUnless(is_cpp_available() and CppTreeSearch is not None,
                     "C++ MCTS Engine not available")
class TestNativeExceptionTranslation(unittest.TestCase):
    """No C++ exception may cross the boundary; each maps to a Python error."""

    def test_evaluator_exception_propagates_unchanged(self):
        class Boom:
            def evaluate(self, state):
                raise KeyError("evaluator blew up")

            def evaluate_many(self, states):
                raise KeyError("evaluator blew up")

        tree = CppTreeSearch(model=Boom(), config=SearchConfig(proofs=False))
        with self.assertRaises(KeyError):
            tree.run(State(), simulations=16, batch_size=4)

    def test_tree_usable_after_evaluator_exception(self):
        class Flaky:
            def __init__(self):
                self.fail = True

            def _p(self, state):
                import numpy as np
                p = np.zeros(81, dtype=np.float32)
                legal = state.legal_actions()
                p[legal] = 1.0 / len(legal)
                return p, 0.0

            def evaluate(self, state):
                if self.fail:
                    raise RuntimeError("nope")
                return self._p(state)

            def evaluate_many(self, states):
                if self.fail:
                    raise RuntimeError("nope")
                return [self._p(s) for s in states]

        model = Flaky()
        tree = CppTreeSearch(model=model, config=SearchConfig(proofs=False))
        with self.assertRaises(RuntimeError):
            tree.run(State(), simulations=16, batch_size=4)
        model.fail = False
        pi = tree.run(State(), simulations=32, batch_size=4)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)

    def test_invalid_state_type_raises_typeerror(self):
        tree = CppTreeSearch(model=None)
        with self.assertRaises(TypeError):
            tree.run("not a state", simulations=8)

    def test_terminal_state_rejected(self):
        tree = CppTreeSearch(model=None)
        s = State()
        while s.result is None:
            s = s.play(s.legal_actions()[0])
        with self.assertRaises(ValueError):
            tree.run(s, simulations=8)

    def test_nonpositive_budgets_rejected(self):
        tree = CppTreeSearch(model=None)
        for sims, batch in ((0, 1), (-5, 1), (8, 0), (8, -2)):
            with self.assertRaises(ValueError):
                tree.run(State(), simulations=sims, batch_size=batch)


if __name__ == "__main__":
    unittest.main()
