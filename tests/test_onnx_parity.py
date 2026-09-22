"""The ONNX network must be the network that was measured.

If the export is subtly wrong the container plays a different agent than
docs/report/ describes, and nothing else in the system would notice. These
tests are the only thing standing between those two outcomes.
"""
import json
import unittest
from pathlib import Path

import numpy as np

from sttt.env import State
from sttt.web.evaluator import OnnxEvaluator

REPO = Path(__file__).resolve().parent.parent
SHIPPED = REPO / "models" / "model-6000.onnx"

try:
    import torch  # noqa: F401
    import onnxruntime  # noqa: F401
    HAVE_EXPORT_DEPS = True
except ImportError:
    HAVE_EXPORT_DEPS = False


def sample_positions(count, seed=0):
    rng = np.random.default_rng(seed)
    states = []
    while len(states) < count:
        state = State()
        while state.result is None and len(states) < count:
            states.append(state)
            state = state.play(int(rng.choice(state.legal_actions())))
    return states


@unittest.skipUnless(HAVE_EXPORT_DEPS, "needs torch and onnxruntime")
class ExportParityTests(unittest.TestCase):
    """Export a randomly initialised net of the shipping shape and round-trip it."""

    @classmethod
    def setUpClass(cls):
        import sys
        import tempfile

        import torch
        sys.path.insert(0, str(REPO / "scripts"))
        import export_onnx

        from sttt.unet import UNet
        torch.manual_seed(0)
        # Default dimensions, not a tiny stand-in: load_model rebuilds the
        # architecture from the `arch` tag and ignores per-layer widths, so a
        # shrunken net cannot round-trip. Testing the shape that actually ships
        # is the more faithful check anyway.
        model = UNet().eval()
        cls.tmp = tempfile.TemporaryDirectory()
        checkpoint = Path(cls.tmp.name) / "tiny.pt"
        torch.save({"model": model.state_dict(), "arch": "unet", "iteration": 1}, checkpoint)
        cls.onnx = Path(cls.tmp.name) / "tiny.onnx"
        cls.export_onnx = export_onnx
        cls.record = export_onnx.export(checkpoint, cls.onnx, positions=64)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_export_records_its_provenance(self):
        sidecar = json.loads(self.onnx.with_suffix(".onnx.json").read_text())
        self.assertEqual(sidecar["arch"], "unet")
        self.assertEqual(sidecar["opset"], 18)
        self.assertEqual(len(sidecar["source_sha256"]), 64)
        self.assertLess(sidecar["parity"]["max_logit_abs_diff"], 1e-4)

    def test_every_batch_size_agrees_including_one(self):
        """A graph that folded its batch dimension passes at 64 and fails at 1."""
        import torch
        model, _ = __import__("sttt.learning", fromlist=["load_model"]).load_model(
            str(Path(self.tmp.name) / "tiny.pt"))
        states = sample_positions(64, seed=3)
        evaluator = OnnxEvaluator(self.onnx)
        from sttt.encoding import encode_states
        for size in (1, 2, 5, 16, 64):
            with self.subTest(batch=size):
                batch = states[:size]
                features, _ = encode_states(batch)
                with torch.inference_mode():
                    t_logits, t_value = model(torch.from_numpy(features))
                onnx_out = evaluator.evaluate_many(batch)
                self.assertEqual(len(onnx_out), size)
                values = np.array([v for _, v in onnx_out])
                np.testing.assert_allclose(values, t_value.numpy(), atol=1e-4)

    def test_probabilities_are_normalised_and_illegal_actions_are_zero(self):
        evaluator = OnnxEvaluator(self.onnx)
        for state in sample_positions(24, seed=7):
            policy, _ = evaluator.evaluate(state)
            legal = set(state.legal_actions())
            self.assertAlmostEqual(float(policy.sum()), 1.0, places=9)
            illegal = [i for i in range(81) if i not in legal]
            self.assertTrue(np.all(policy[illegal] == 0.0))
            self.assertTrue(np.all(policy[sorted(legal)] > 0.0))


@unittest.skipUnless(SHIPPED.exists(), "models/model-6000.onnx not present")
class ShippedModelTests(unittest.TestCase):
    """The committed artifact must match the checkpoint it claims to come from."""

    def setUp(self):
        self.sidecar = json.loads(SHIPPED.with_suffix(".onnx.json").read_text())

    def test_sidecar_names_a_checkpoint_that_still_hashes_the_same(self):
        import hashlib
        source = REPO / self.sidecar["source_checkpoint"]
        if not source.exists():
            self.skipTest(f"source checkpoint not in this checkout: {source}")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(digest, self.sidecar["source_sha256"],
                         "models/model-6000.onnx was exported from a different file "
                         "than its sidecar names; re-run scripts/export_onnx.py")

    @unittest.skipUnless(HAVE_EXPORT_DEPS, "needs torch")
    def test_shipped_onnx_matches_the_checkpoint(self):
        import torch
        from sttt.encoding import encode_states
        from sttt.learning import load_model
        source = REPO / self.sidecar["source_checkpoint"]
        if not source.exists():
            self.skipTest("source checkpoint not in this checkout")
        model, _ = load_model(str(source))
        states = sample_positions(128, seed=11)
        features, _ = encode_states(states)
        with torch.inference_mode():
            _, t_value = model(torch.from_numpy(features))
        onnx_values = np.array([v for _, v in OnnxEvaluator(SHIPPED).evaluate_many(states)])
        np.testing.assert_allclose(onnx_values, t_value.numpy(), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
