"""M1 regression tests: every float crossing the native boundary is validated.

Each test here reproduces a defect where a malformed evaluator output or config
value was converted with an unchecked ``PyFloat_AsDouble`` and silently became a
real prior (or a NaN c_puct that made every PUCT score NaN). The native layer
must raise ``ValueError`` at the boundary instead, and must leave the tree
reusable rather than holding in-flight counters for paths it never backed up.
"""
import math
import unittest

import numpy as np

from sttt.env import State
from sttt.search import SearchConfig
from sttt.cpp_env import is_cpp_available, CppTreeSearch


def _good_policy(state):
    p = np.zeros(81, dtype=np.float32)
    legal = state.legal_actions()
    if legal:
        p[legal] = 1.0 / len(legal)
    return p


class _Evaluator:
    """Evaluator whose policy can be corrupted at a chosen flat index."""

    def __init__(self, corrupt_index=None, corrupt_value=None, value=0.0):
        self.corrupt_index = corrupt_index
        self.corrupt_value = corrupt_value
        self.value = value
        self.calls = 0

    def _policy(self, state):
        p = _good_policy(state)
        if self.corrupt_index is None:
            return p
        out = p.astype(object)          # object dtype carries non-float corruption
        out[self.corrupt_index] = self.corrupt_value
        return list(out)

    def evaluate(self, state):
        self.calls += 1
        return self._policy(state), float(self.value)

    def evaluate_many(self, states):
        return [self.evaluate(s) for s in states]


@unittest.skipUnless(is_cpp_available() and CppTreeSearch is not None,
                     "C++ MCTS Engine not available")
class TestNativePolicyValidation(unittest.TestCase):
    """A malformed policy must raise, not become a prior."""

    def _assert_rejected(self, corrupt_value, batch_size):
        model = _Evaluator(corrupt_index=0, corrupt_value=corrupt_value)
        tree = CppTreeSearch(model=model, config=SearchConfig(proofs=False))
        with self.assertRaises(ValueError) as ctx:
            tree.run(State(), simulations=16, batch_size=batch_size)
        self.assertIn("policy", str(ctx.exception).lower())

    def test_nan_policy_entry_rejected(self):
        self._assert_rejected(float("nan"), batch_size=1)

    def test_inf_policy_entry_rejected(self):
        self._assert_rejected(float("inf"), batch_size=1)

    def test_negative_policy_entry_rejected(self):
        self._assert_rejected(-0.5, batch_size=1)

    def test_non_numeric_policy_entry_rejected(self):
        self._assert_rejected("not-a-number", batch_size=1)

    def test_none_policy_entry_rejected(self):
        self._assert_rejected(None, batch_size=1)

    def test_batched_nan_policy_entry_rejected(self):
        # batch_size > 1 takes the evaluate_many path, validated separately.
        self._assert_rejected(float("nan"), batch_size=8)

    def test_batched_negative_policy_entry_rejected(self):
        self._assert_rejected(-1.0, batch_size=8)

    def test_tree_reusable_after_rejected_batch(self):
        """A rejected batch must release every pending path, not strand counters."""
        model = _Evaluator(corrupt_index=3, corrupt_value=float("nan"))
        tree = CppTreeSearch(model=model, config=SearchConfig(proofs=False))
        with self.assertRaises(ValueError):
            tree.run(State(), simulations=16, batch_size=8)

        # Same tree, now with a well-formed evaluator: it must search cleanly.
        model.corrupt_index = None
        pi = tree.run(State(), simulations=32, batch_size=8)
        self.assertEqual(len(pi), 81)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
        self.assertEqual(tree.root.in_flight, 0)
        self.assertFalse(tree.root.pending)


@unittest.skipUnless(is_cpp_available() and CppTreeSearch is not None,
                     "C++ MCTS Engine not available")
class TestNativeConfigValidation(unittest.TestCase):
    """Nonfinite or out-of-range config values must be rejected at construction."""

    def _assert_config_rejected(self, **kwargs):
        cfg = SearchConfig(**kwargs)
        with self.assertRaises(ValueError):
            CppTreeSearch(model=None, config=cfg)

    def test_nan_c_puct_rejected(self):
        self._assert_config_rejected(c_puct=float("nan"))

    def test_inf_c_puct_rejected(self):
        self._assert_config_rejected(c_puct=float("inf"))

    def test_nan_soft_margin_rejected(self):
        self._assert_config_rejected(soft_margin=float("nan"))

    def test_nan_soft_strength_rejected(self):
        self._assert_config_rejected(soft_strength=float("nan"))

    def test_zero_revisit_interval_rejected(self):
        self._assert_config_rejected(revisit_interval=0)

    def test_negative_min_visits_rejected(self):
        self._assert_config_rejected(min_visits=-1)

    def test_valid_config_still_accepted(self):
        cfg = SearchConfig(c_puct=1.75, soft_margin=0.1, soft_strength=0.5,
                           min_visits=0, revisit_interval=1)
        tree = CppTreeSearch(model=None, config=cfg)
        pi = tree.run(State(), simulations=32, batch_size=4)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)

    def test_finite_c_puct_survives_round_trip(self):
        """Guard against the validator silently zeroing an accepted value."""
        tree = CppTreeSearch(model=None, config=SearchConfig(c_puct=2.5))
        self.assertAlmostEqual(tree.run(State(), simulations=16).sum(), 1.0, places=5)
        self.assertFalse(math.isnan(float(tree.stats["completed_simulations"])))


if __name__ == "__main__":
    unittest.main()
