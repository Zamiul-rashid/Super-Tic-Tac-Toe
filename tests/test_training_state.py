"""M4 regression tests: AMP scaler lifetime, checkpoint schema, run ownership.

The scaler defect is the subtle one. ``torch.amp.GradScaler`` was constructed
*inside* the iteration loop, so each iteration discarded the loss scale the
previous one had converged on and restarted scale discovery -- which skips
optimizer updates while it halves the scale back down. Over a long run that is
a permanent, silent tax on every iteration's first steps.
"""
import ast
import pathlib
import tempfile
import unittest

import numpy as np
import torch

from sttt.ai import RunOwnership

AI_SOURCE = pathlib.Path(__file__).resolve().parents[1] / "sttt" / "ai.py"


def _function(name):
    tree = ast.parse(AI_SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in ai.py")


class TestScalerLifetime(unittest.TestCase):
    def test_scaler_is_not_constructed_inside_the_iteration_loop(self):
        loop = _function("_train_loop")
        for node in ast.walk(loop):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                self.assertNotEqual(
                    name, "GradScaler",
                    "GradScaler must be created once per process, not per iteration")

    def test_scaler_is_constructed_in_train(self):
        train = _function("train")
        names = [getattr(n.func, "attr", None) or getattr(n.func, "id", None)
                 for n in ast.walk(train) if isinstance(n, ast.Call)]
        self.assertIn("GradScaler", names)

    def test_scaler_state_round_trips(self):
        """A restored scaler must keep the scale, not restart discovery."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for GradScaler state")
        scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.0)
        # Touch the scale so the state dict is populated the way a real
        # iteration would leave it.
        scaler.scale(torch.zeros(1, device="cuda", requires_grad=True).sum()).backward()
        state = scaler.state_dict()
        restored = torch.amp.GradScaler("cuda", enabled=True)
        restored.load_state_dict(state)
        self.assertEqual(restored.get_scale(), 1024.0)


class TestCheckpointSchema(unittest.TestCase):
    REQUIRED = {"model", "optimizer", "iteration", "replay", "arch",
                "training_config", "search_config", "numpy_rng_state",
                "torch_rng_state", "scaler", "precision", "backend_info",
                "population_games", "metrics"}

    def test_checkpoint_dict_literal_carries_every_required_field(self):
        loop = _function("_train_loop")
        found = set()
        for node in ast.walk(loop):
            if isinstance(node, ast.Dict):
                keys = {k.value for k in node.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                if "model" in keys and "optimizer" in keys:
                    found = keys
        missing = self.REQUIRED - found
        self.assertFalse(missing, f"checkpoint schema missing: {sorted(missing)}")

    def test_metrics_report_policy_and_value_separately(self):
        loop = _function("_train_loop")
        source = ast.dump(loop)
        for field in ("policy_loss", "value_loss", "grad_norm",
                      "optimizer_updates", "skipped_updates"):
            self.assertIn(field, source, f"{field} is not recorded")


class TestNonfiniteGuards(unittest.TestCase):
    def test_loop_rejects_nonfinite_loss(self):
        source = AI_SOURCE.read_text()
        self.assertIn("torch.isfinite(loss)", source,
                      "a nonfinite loss must not be committed to the model")

    def test_loop_fails_when_every_update_is_skipped(self):
        source = AI_SOURCE.read_text()
        self.assertIn("optimizer_updates == 0", source,
                      "an iteration with no applied update must not count as training")


class TestRunOwnership(unittest.TestCase):
    def test_second_writer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first = RunOwnership(directory).acquire()
            try:
                with self.assertRaises(RuntimeError):
                    RunOwnership(directory).acquire()
            finally:
                first.release()

    def test_lock_is_released_for_the_next_run(self):
        with tempfile.TemporaryDirectory() as directory:
            RunOwnership(directory).acquire().release()
            second = RunOwnership(directory).acquire()
            second.release()

    def test_context_manager_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            with RunOwnership(directory):
                pass
            RunOwnership(directory).acquire().release()

    def test_lock_file_records_the_owning_pid(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            owner = RunOwnership(directory).acquire()
            try:
                self.assertEqual(
                    int(pathlib.Path(owner.path).read_text().strip()), os.getpid())
            finally:
                owner.release()

    def test_distinct_directories_do_not_conflict(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, second = RunOwnership(a).acquire(), RunOwnership(b).acquire()
            first.release()
            second.release()


if __name__ == "__main__":
    unittest.main()
