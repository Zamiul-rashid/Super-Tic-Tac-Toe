"""M3 regression tests: encoding ownership, backend explicitness, batch bounds.

Two defects are locked down here.

``encode_batch`` returned ``np.frombuffer`` over immutable Python bytes: a
read-only view that owns no storage, which is not a valid owner for
``torch.from_numpy``.

``evaluate_many`` contained a *second* copy of its body after an unconditional
``return``, and that dead copy was the only one honouring ``use_fp16`` -- so
``--fp16`` silently did nothing for self-play inference.
"""
import ast
import gc
import pathlib
import unittest

import numpy as np
import torch

from sttt.env import State
from sttt.learning import INPUTS, encode, encode_states, legal_masks
from sttt.cpp_env import encode_batch, is_cpp_available


def random_states(n, seed=0):
    rng = np.random.default_rng(seed)
    states, s = [State()], State()
    for _ in range(n - 1):
        if s.result is not None:
            break
        s = s.play(int(rng.choice(s.legal_actions())))
        states.append(s)
    return states


class TestEncodingOwnership(unittest.TestCase):
    def test_encode_states_returns_owned_writable_contiguous(self):
        for backend in ("python", "cpp") if is_cpp_available() else ("python",):
            features, mask = encode_states(random_states(6), backend=backend)
            self.assertTrue(features.flags.owndata, f"{backend}: not owned")
            self.assertTrue(features.flags.writeable, f"{backend}: not writable")
            self.assertTrue(features.flags.c_contiguous, f"{backend}: not contiguous")
            self.assertEqual(features.dtype, np.float32)
            self.assertEqual(mask.dtype, np.bool_)

    def test_features_are_safe_for_torch_from_numpy(self):
        features, _ = encode_states(random_states(4))
        tensor = torch.from_numpy(features)
        tensor[0, 0] = 5.0            # read-only storage would raise here
        self.assertEqual(float(features[0, 0]), 5.0)

    @unittest.skipUnless(is_cpp_available(), "native encoder not built")
    def test_encode_batch_is_owned(self):
        arr = encode_batch(random_states(3))
        self.assertTrue(arr.flags.owndata)
        self.assertTrue(arr.flags.writeable)

    def test_storage_outlives_the_source_states(self):
        states = random_states(5)
        features, mask = encode_states(states)
        checksum = float(features.sum())
        del states
        gc.collect()
        self.assertEqual(float(features.sum()), checksum)
        self.assertEqual(int(mask.sum()), int(mask.sum()))


class TestEncodingParity(unittest.TestCase):
    @unittest.skipUnless(is_cpp_available(), "native encoder not built")
    def test_native_matches_python_reference_exactly(self):
        for seed in range(5):
            states = random_states(12, seed=seed)
            f_py, m_py = encode_states(states, backend="python")
            f_cpp, m_cpp = encode_states(states, backend="cpp")
            np.testing.assert_array_equal(f_py, f_cpp)
            np.testing.assert_array_equal(m_py, m_cpp)

    def test_reference_encoder_matches_single_state_encode(self):
        states = random_states(6)
        features, _ = encode_states(states, backend="python")
        for i, state in enumerate(states):
            np.testing.assert_array_equal(features[i], encode(state))

    def test_masks_match_legal_actions(self):
        states = random_states(8)
        mask = legal_masks(states)
        for i, state in enumerate(states):
            self.assertEqual(sorted(np.flatnonzero(mask[i]).tolist()),
                             sorted(state.legal_actions()))

    def test_feature_width_is_the_declared_input_size(self):
        features, _ = encode_states(random_states(3))
        self.assertEqual(features.shape[1], INPUTS)


class TestBackendExplicitness(unittest.TestCase):
    def test_unknown_backend_rejected(self):
        with self.assertRaises(ValueError):
            encode_states([State()], backend="gpu")

    def test_strict_cpp_refuses_to_substitute_python(self):
        import sttt.learning as learning
        real = learning.encode_states
        # Simulate a machine with no native encoder.
        import sttt.cpp_env as cpp_env
        original = cpp_env.is_cpp_available
        cpp_env.is_cpp_available = lambda: False
        try:
            with self.assertRaises(RuntimeError):
                real([State()], backend="cpp")
        finally:
            cpp_env.is_cpp_available = original

    def test_empty_batch_has_correct_shape(self):
        features, mask = encode_states([], backend="python")
        self.assertEqual(features.shape, (0, INPUTS))
        self.assertEqual(mask.shape, (0, 81))

    def test_mixed_state_types_encode_identically(self):
        if not is_cpp_available():
            self.skipTest("native encoder not built")
        from sttt.cpp_env import to_fast_state
        states = random_states(6)
        mixed = [s if i % 2 else to_fast_state(s) for i, s in enumerate(states)]
        f_plain, _ = encode_states(states, backend="cpp")
        f_mixed, _ = encode_states(mixed, backend="cpp")
        np.testing.assert_array_equal(f_plain, f_mixed)


class TestNoDeadCodeInEvaluateMany(unittest.TestCase):
    def test_evaluate_many_has_no_unreachable_statements(self):
        """The fp16 path was unreachable; assert no return-then-code remains."""
        source = pathlib.Path(__file__).resolve().parents[1] / "sttt" / "learning.py"
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_many":
                for i, stmt in enumerate(node.body):
                    if isinstance(stmt, ast.Return):
                        self.assertEqual(
                            node.body[i + 1:], [],
                            "unreachable statements after return in evaluate_many")

    def test_fp16_flag_is_actually_consulted(self):
        source = pathlib.Path(__file__).resolve().parents[1] / "sttt" / "learning.py"
        tree = ast.parse(source.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_many":
                body_src = ast.dump(node)
                self.assertIn("use_fp16", body_src,
                              "evaluate_many no longer consults use_fp16")
                return
        self.fail("evaluate_many not found")


if __name__ == "__main__":
    unittest.main()
