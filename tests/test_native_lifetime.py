"""M1 regression tests: node-view lifetime, advance() legality, benchmark threads.

A ``FastNode`` holds an arena *index*, not a pointer. ``reset()``, ``init_root()``
and re-rooting all rebuild the arena, after which the same index names a
different node -- the pre-patch getters returned that unrelated node's
statistics as if nothing had happened. Views now carry the engine generation
they were captured under and fail loudly once it moves on.
"""
import unittest

from sttt.env import State
from sttt.search import SearchConfig
# cpp_env puts the built extension on sys.path; import it only afterwards.
from sttt.cpp_env import is_cpp_available, CppTreeSearch

try:
    import sttt_cpp
except ImportError:              # pragma: no cover - mirrors cpp_env's own guard
    sttt_cpp = None


@unittest.skipUnless(is_cpp_available() and CppTreeSearch is not None,
                     "C++ MCTS Engine not available")
class TestNodeViewLifetime(unittest.TestCase):
    def _searched_tree(self, simulations=64):
        tree = CppTreeSearch(model=None, config=SearchConfig(proofs=False))
        tree.run(State(), simulations=simulations, batch_size=4)
        return tree

    def test_view_valid_while_generation_holds(self):
        tree = self._searched_tree()
        root = tree.root
        self.assertGreater(root.n, 0)
        self.assertTrue(root.children)          # dict of {action: node}
        self.assertEqual(root.in_flight, 0)

    def test_view_stale_after_reset(self):
        tree = self._searched_tree()
        root = tree.root
        tree.reset()
        for attr in ("n", "total", "prior", "solved", "in_flight", "pending",
                     "state", "children"):
            with self.assertRaises(RuntimeError, msg=f"{attr} must reject a stale view"):
                getattr(root, attr)

    def test_view_stale_after_advance(self):
        """Re-rooting discards the sibling subtrees the view may point into."""
        tree = self._searched_tree()
        root = tree.root
        action = next(iter(root.children))
        tree.advance(action)
        with self.assertRaises(RuntimeError):
            root.n

    def test_child_view_stale_after_reset(self):
        tree = self._searched_tree()
        child = next(iter(tree.root.children.values()))
        self.assertGreaterEqual(child.n, 0)
        tree.reset()
        with self.assertRaises(RuntimeError):
            child.n

    def test_fresh_view_after_reset_is_usable(self):
        """The failure must be recoverable: re-reading tree.root works."""
        tree = self._searched_tree()
        tree.reset()
        self.assertIsNone(tree.root)
        tree.run(State(), simulations=32, batch_size=4)
        self.assertGreater(tree.root.n, 0)


@unittest.skipUnless(is_cpp_available() and CppTreeSearch is not None,
                     "C++ MCTS Engine not available")
class TestAdvanceLegality(unittest.TestCase):
    def test_out_of_range_action_rejected(self):
        tree = CppTreeSearch(model=None)
        tree.run(State(), simulations=16)
        for bad in (-1, 81, 1000):
            with self.assertRaises(ValueError):
                tree.advance(bad)

    def test_in_range_illegal_action_rejected(self):
        """The pre-patch code re-rooted onto a board the rules cannot produce."""
        tree = CppTreeSearch(model=None)
        s = State()
        tree.run(s, simulations=16)
        tree.advance(40)                      # centre of the centre board
        legal = set(s.play(40).legal_actions())
        illegal = next(a for a in range(81) if a not in legal)
        with self.assertRaises(ValueError):
            tree.advance(illegal)

    def test_legal_action_still_accepted(self):
        tree = CppTreeSearch(model=None)
        s = State()
        tree.run(s, simulations=16)
        for action in list(s.legal_actions())[:1]:
            tree.advance(action)
        self.assertIsNotNone(tree.root)


class TestBenchmarkThreadSafety(unittest.TestCase):
    """benchmark_rollouts must never abort the interpreter."""

    @unittest.skipUnless(is_cpp_available() and sttt_cpp is not None,
                         "C++ engine not available")
    def test_more_threads_than_games_does_not_abort(self):
        elapsed, moves, games_ps, moves_ps = sttt_cpp.benchmark_rollouts(1, 2, 12345)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertGreater(moves, 0)

    @unittest.skipUnless(is_cpp_available() and sttt_cpp is not None,
                         "C++ engine not available")
    def test_zero_games_is_safe(self):
        elapsed, moves, games_ps, moves_ps = sttt_cpp.benchmark_rollouts(0, 4, 7)
        self.assertGreaterEqual(elapsed, 0.0)
        self.assertEqual(moves, 0)

    @unittest.skipUnless(is_cpp_available() and sttt_cpp is not None,
                         "C++ engine not available")
    def test_multithreaded_run_completes(self):
        elapsed, moves, games_ps, moves_ps = sttt_cpp.benchmark_rollouts(64, 4, 99)
        self.assertGreater(moves, 0)
        self.assertGreaterEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
