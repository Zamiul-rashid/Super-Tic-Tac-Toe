"""M5 regression tests: learning-rate schedules in units of completed iterations.

The unit is the whole point. A cosine horizon expressed in optimizer *updates*
and stepped once per update is a different schedule from the same horizon in
training *iterations*: at 100 updates per iteration, ``T_max=5000`` stepped per
update bottoms out after 50 iterations rather than 5,000.
"""
import math
import unittest
from types import SimpleNamespace

from sttt.training_schedule import (CONSTANT, COSINE, DEFAULT_LR, LRSchedule,
                                    apply_lr, build_schedule)


class _Optimizer:
    def __init__(self, lr=DEFAULT_LR):
        self.param_groups = [{"lr": lr, "params": []}]


class TestConstantSchedule(unittest.TestCase):
    def test_constant_never_moves(self):
        schedule = LRSchedule(kind=CONSTANT, lr_start=3e-4)
        for _ in range(50):
            self.assertEqual(schedule.current_lr, 3e-4)
            schedule.advance()

    def test_constant_ignores_horizon(self):
        schedule = LRSchedule(kind=CONSTANT, lr_start=1e-3, horizon=0)
        self.assertEqual(schedule.lr_at(10_000), 1e-3)


class TestCosineCurve(unittest.TestCase):
    def setUp(self):
        self.schedule = LRSchedule(kind=COSINE, lr_start=1e-3, lr_min=1e-5, horizon=5000)

    def test_starts_at_lr_start(self):
        self.assertAlmostEqual(self.schedule.lr_at(0), 1e-3)

    def test_midpoint_is_the_arithmetic_mean(self):
        self.assertAlmostEqual(self.schedule.lr_at(2500), (1e-3 + 1e-5) / 2, places=12)

    def test_reaches_the_floor_at_the_horizon(self):
        self.assertAlmostEqual(self.schedule.lr_at(5000), 1e-5, places=12)

    def test_clamps_past_the_horizon_and_never_restarts(self):
        floor = self.schedule.lr_at(5000)
        for k in (5001, 7500, 100_000):
            self.assertAlmostEqual(self.schedule.lr_at(k), floor, places=12)

    def test_monotonically_decreasing(self):
        previous = math.inf
        for k in range(0, 5001, 25):
            current = self.schedule.lr_at(k)
            self.assertLessEqual(current, previous + 1e-15)
            previous = current

    def test_horizon_is_iterations_not_updates(self):
        """A 10-iteration horizon must not be exhausted by 10 optimizer steps."""
        schedule = LRSchedule(kind=COSINE, lr_start=1e-3, lr_min=1e-5, horizon=10)
        schedule.advance()                      # one COMPLETED iteration
        self.assertGreater(schedule.current_lr, 1e-5 * 2)


class TestResumeContinuity(unittest.TestCase):
    def _sequence(self, schedule, steps):
        out = []
        for _ in range(steps):
            out.append(schedule.current_lr)
            schedule.advance()
        return out

    def test_save_load_midway_reproduces_the_same_sequence(self):
        """Two uninterrupted iterations == one, save/load, one."""
        straight = self._sequence(
            LRSchedule(kind=COSINE, lr_start=1e-3, lr_min=1e-5, horizon=100), 2)

        interrupted = LRSchedule(kind=COSINE, lr_start=1e-3, lr_min=1e-5, horizon=100)
        first = [interrupted.current_lr]
        interrupted.advance()
        restored = LRSchedule.from_state(interrupted.state_dict())
        first.append(restored.current_lr)
        self.assertEqual(straight, first)

    def test_state_round_trip_preserves_every_field(self):
        schedule = LRSchedule(kind=COSINE, lr_start=2e-3, lr_min=1e-6, horizon=42,
                              completed=7, phase=3)
        restored = LRSchedule.from_state(schedule.state_dict())
        self.assertEqual(restored.state_dict(), schedule.state_dict())

    def test_state_is_plain_data(self):
        """Must load under weights_only=True: no scheduler object, no callable."""
        state = LRSchedule(kind=COSINE, lr_start=1e-3, horizon=5).state_dict()
        for value in state.values():
            self.assertIsInstance(value, (int, float, str))

    def test_interrupted_iteration_does_not_advance(self):
        schedule = LRSchedule(kind=COSINE, lr_start=1e-3, lr_min=1e-5, horizon=100)
        before = schedule.state_dict()
        # No advance() call: the iteration never completed.
        self.assertEqual(schedule.state_dict(), before)


class TestValidation(unittest.TestCase):
    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            LRSchedule(kind="linear")

    def test_nonpositive_lr_rejected(self):
        for bad in (0.0, -1e-3):
            with self.assertRaises(ValueError):
                LRSchedule(lr_start=bad)

    def test_nonfinite_lr_rejected(self):
        for bad in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                LRSchedule(lr_start=bad)

    def test_floor_above_start_rejected(self):
        with self.assertRaises(ValueError):
            LRSchedule(kind=COSINE, lr_start=1e-5, lr_min=1e-3, horizon=10)

    def test_cosine_needs_a_positive_horizon(self):
        for bad in (0, -5):
            with self.assertRaises(ValueError):
                LRSchedule(kind=COSINE, lr_start=1e-3, horizon=bad)


class TestBuildSchedule(unittest.TestCase):
    def _args(self, **kw):
        base = dict(lr=None, lr_schedule=None, lr_min=0.0, lr_iterations=0,
                    reset_lr_schedule=False)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_fresh_run_defaults_to_legacy_constant(self):
        schedule = build_schedule(self._args(), None)
        self.assertEqual(schedule.kind, CONSTANT)
        self.assertEqual(schedule.lr_start, DEFAULT_LR)

    def test_legacy_resume_keeps_its_restored_lr(self):
        """A decayed run must not silently jump back to 1e-3."""
        schedule = build_schedule(self._args(), None, legacy_lr=2.5e-4)
        self.assertAlmostEqual(schedule.lr_start, 2.5e-4)

    def test_resume_continues_the_saved_phase(self):
        saved = LRSchedule(kind=COSINE, lr_start=1e-3, lr_min=1e-5,
                           horizon=5000, completed=1200).state_dict()
        schedule = build_schedule(self._args(), saved)
        self.assertEqual(schedule.completed, 1200)
        self.assertEqual(schedule.horizon, 5000)

    def test_resume_does_not_recompute_horizon_from_iterations(self):
        saved = LRSchedule(kind=COSINE, lr_start=1e-3, lr_min=1e-5,
                           horizon=5000, completed=10).state_dict()
        schedule = build_schedule(self._args(lr_iterations=99), saved)
        self.assertEqual(schedule.horizon, 5000)

    def test_silent_mid_phase_change_is_refused(self):
        saved = LRSchedule(kind=COSINE, lr_start=1e-3, lr_min=1e-5,
                           horizon=5000, completed=10).state_dict()
        with self.assertRaises(ValueError):
            build_schedule(self._args(lr=5e-4), saved)
        with self.assertRaises(ValueError):
            build_schedule(self._args(lr_schedule="constant"), saved)

    def test_explicit_reset_starts_a_new_recorded_phase(self):
        saved = LRSchedule(kind=COSINE, lr_start=1e-3, lr_min=1e-5,
                           horizon=5000, completed=10, phase=0).state_dict()
        schedule = build_schedule(
            self._args(lr=3e-4, lr_schedule="constant", reset_lr_schedule=True), saved)
        self.assertEqual(schedule.phase, 1)
        self.assertEqual(schedule.completed, 0)
        self.assertAlmostEqual(schedule.lr_start, 3e-4)


class TestApplyLR(unittest.TestCase):
    def test_apply_lr_sets_every_group(self):
        optimizer = _Optimizer(lr=1e-3)
        optimizer.param_groups.append({"lr": 1e-3, "params": []})
        apply_lr(optimizer, 4.2e-4)
        for group in optimizer.param_groups:
            self.assertAlmostEqual(group["lr"], 4.2e-4)


if __name__ == "__main__":
    unittest.main()
