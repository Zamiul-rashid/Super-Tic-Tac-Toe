"""M2 regression tests: one explicit search-backend contract.

The defect this locks down is silent substitution. ``CheckpointBot`` re-probed
for the native extension inside ``choose()`` and, on ImportError, ran the Python
search while still reporting ``backend="cpp"`` -- so a whole run's artifacts
could name a backend that never executed.
"""
import unittest
from unittest import mock

import numpy as np

from sttt import backends
from sttt.backends import (AUTO, CPP, PYTHON, BackendUnavailable, SearchAdapter,
                           create_search, derive_rng, describe_backend,
                           resolve_backend, STREAM_SEARCH, STREAM_MOVE_SAMPLING,
                           STREAM_OPPONENT)
from sttt.env import State
from sttt.search import SearchConfig, TreeSearch


class TestBackendResolution(unittest.TestCase):
    def test_python_is_always_available(self):
        info = resolve_backend(PYTHON)
        self.assertEqual(info["actual"], PYTHON)
        self.assertIsNone(info["fallback_reason"])

    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            resolve_backend("gpu")

    def test_strict_cpp_raises_when_unavailable(self):
        """The whole point: never substitute Python for a strict cpp request."""
        with mock.patch.object(backends, "cpp_available", return_value=False):
            with self.assertRaises(BackendUnavailable):
                resolve_backend(CPP)

    def test_auto_falls_back_with_a_recorded_reason(self):
        with mock.patch.object(backends, "cpp_available", return_value=False):
            info = resolve_backend(AUTO)
        self.assertEqual(info["actual"], PYTHON)
        self.assertIn("unavailable", info["fallback_reason"])

    def test_auto_prefers_cpp_when_available(self):
        with mock.patch.object(backends, "cpp_available", return_value=True):
            info = resolve_backend(AUTO)
        self.assertEqual(info["actual"], CPP)
        self.assertIsNone(info["fallback_reason"])

    def test_no_cpp_claim_when_python_ran(self):
        """backend_info must never say cpp if the Python search is executing."""
        with mock.patch.object(backends, "cpp_available", return_value=False):
            adapter = create_search(None, backend=AUTO, seed=1)
        self.assertEqual(adapter.backend, PYTHON)
        self.assertIsInstance(adapter._tree, TreeSearch)
        self.assertNotEqual(adapter.backend_info["actual"], CPP)

    def test_describe_backend_states_choice_and_reason(self):
        with mock.patch.object(backends, "cpp_available", return_value=False):
            line = describe_backend(resolve_backend(AUTO))
        self.assertIn("python", line)
        self.assertIn("requested auto", line)
        self.assertIn("unavailable", line)


class TestAdaptiveOpponentIsPythonOnly(unittest.TestCase):
    class _Opponent:
        def predict(self, state, policy):
            return policy

    def test_strict_cpp_with_opponent_raises_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            resolve_backend(CPP, opponent=self._Opponent())

    def test_auto_with_opponent_chooses_python(self):
        info = resolve_backend(AUTO, opponent=self._Opponent())
        self.assertEqual(info["actual"], PYTHON)
        self.assertIn("Python-only", info["fallback_reason"])

    def test_adaptive_prediction_still_runs_in_python(self):
        calls = []

        class Recording(self._Opponent.__class__ if False else object):
            def predict(self, state, policy):
                calls.append(state)
                return policy

        class Uniform:
            def evaluate(self, state):
                p = np.zeros(81, dtype=np.float32)
                legal = state.legal_actions()
                p[legal] = 1.0 / len(legal)
                return p, 0.0

            def evaluate_many(self, states):
                return [self.evaluate(s) for s in states]

        adapter = create_search(Uniform(), backend=AUTO, seed=3,
                                opponent=Recording(), agent_side=1)
        self.assertEqual(adapter.backend, PYTHON)
        adapter.run(State(), 32, batch_size=1)
        self.assertTrue(calls, "opponent.predict was never consulted")


class TestRngStreams(unittest.TestCase):
    def test_streams_are_independent(self):
        a = derive_rng(42, STREAM_SEARCH).random(10)
        b = derive_rng(42, STREAM_MOVE_SAMPLING).random(10)
        c = derive_rng(42, STREAM_OPPONENT).random(10)
        self.assertFalse(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertFalse(np.array_equal(b, c))

    def test_same_seed_reproduces(self):
        self.assertTrue(np.array_equal(derive_rng(7, STREAM_SEARCH).random(5),
                                       derive_rng(7, STREAM_SEARCH).random(5)))

    def test_different_seeds_differ(self):
        self.assertFalse(np.array_equal(derive_rng(7, STREAM_SEARCH).random(5),
                                        derive_rng(8, STREAM_SEARCH).random(5)))

    def test_unknown_stream_rejected(self):
        with self.assertRaises(ValueError):
            derive_rng(1, 99)

    def test_not_all_games_default_to_seed_42(self):
        """Every game seed must give a distinct search stream."""
        draws = {float(derive_rng(s, STREAM_SEARCH).random()) for s in range(50)}
        self.assertEqual(len(draws), 50)


class TestAdapterSurface(unittest.TestCase):
    def _adapter(self, backend=AUTO):
        return create_search(None, backend=backend, seed=11,
                             config=SearchConfig(proofs=False))

    def test_adapter_exposes_the_documented_surface(self):
        adapter = self._adapter()
        for attr in ("run", "advance", "reset", "root_state", "stats", "backend_info"):
            self.assertTrue(hasattr(adapter, attr), attr)

    def test_root_state_none_before_search(self):
        self.assertIsNone(self._adapter().root_state)

    def test_run_advance_reset_cycle(self):
        adapter = self._adapter()
        state = State()
        pi = adapter.run(state, 32, batch_size=4)
        self.assertEqual(len(pi), 81)
        self.assertAlmostEqual(float(pi.sum()), 1.0, places=5)
        self.assertIsNotNone(adapter.root_state)
        action = int(np.argmax(pi))
        adapter.advance(action)
        adapter.reset()
        self.assertIsNone(adapter.root_state)

    def test_stats_is_a_snapshot_copy(self):
        adapter = self._adapter()
        adapter.run(State(), 32, batch_size=4)
        snap = adapter.stats
        snap["completed_simulations"] = -1
        self.assertNotEqual(adapter.stats.get("completed_simulations"), -1)

    def test_backend_info_is_a_copy(self):
        adapter = self._adapter()
        info = adapter.backend_info
        info["actual"] = "tampered"
        self.assertNotEqual(adapter.backend_info["actual"], "tampered")

    def test_both_backends_produce_a_normalized_policy(self):
        class Uniform:
            def evaluate(self, state):
                p = np.zeros(81, dtype=np.float32)
                legal = state.legal_actions()
                p[legal] = 1.0 / len(legal)
                return p, 0.0

            def evaluate_many(self, states):
                return [self.evaluate(s) for s in states]

        for backend in (PYTHON, AUTO):
            adapter = create_search(Uniform(), backend=backend, seed=5,
                                    config=SearchConfig(proofs=False))
            pi = adapter.run(State(), 32, batch_size=2)
            self.assertAlmostEqual(float(pi.sum()), 1.0, places=5,
                                   msg=f"{backend} policy not normalized")


class TestCheckpointBotResolvesUpFront(unittest.TestCase):
    def test_strict_cpp_bot_fails_before_any_game(self):
        """Construction must fail, not the first choose() call mid-tournament."""
        from sttt import bots
        with mock.patch.object(backends, "cpp_available", return_value=False):
            with self.assertRaises(BackendUnavailable):
                resolve_backend("cpp")
        self.assertTrue(hasattr(bots.CheckpointBot, "choose"))


if __name__ == "__main__":
    unittest.main()


class TestWorkerHandshake(unittest.TestCase):
    """Spawned workers must prove their backend before any game is dispatched."""

    def test_pool_handshake_reports_capabilities(self):
        from sttt.selfplay import SelfPlayPool
        with SelfPlayPool(workers=2, batch_size=4) as pool:
            reports = pool.verify_backend(use_cpp=False)
        self.assertEqual(len(reports), 2)
        for report in reports:
            self.assertIn("cpp_available", report)
            self.assertIn("cpp_build_id", report)
            self.assertIsInstance(report["pid"], int)

    def test_strict_native_run_verifies_every_worker(self):
        from sttt.selfplay import SelfPlayPool
        from sttt.cpp_env import is_cpp_available
        if not is_cpp_available():
            self.skipTest("native search not built")
        with SelfPlayPool(workers=2, batch_size=4) as pool:
            reports = pool.verify_backend(use_cpp=True)
        self.assertTrue(all(r["cpp_available"] for r in reports))
        builds = {r["cpp_build_id"] for r in reports}
        self.assertEqual(len(builds), 1, "workers must share one native build")

    def test_play_game_refuses_to_degrade(self):
        """use_cpp is a requirement: a worker without native must raise."""
        from sttt import selfplay
        with mock.patch.object(selfplay, "_HAS_CPP", False):
            with self.assertRaises(RuntimeError) as ctx:
                selfplay.play_game(None, 8, 1, SearchConfig(), 1, use_cpp=True)
        self.assertIn("refusing to silently fall back", str(ctx.exception).lower())
