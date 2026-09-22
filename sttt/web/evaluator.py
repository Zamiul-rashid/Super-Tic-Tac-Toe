"""ONNX Runtime evaluator: the serving replacement for a torch checkpoint.

The searches never depend on torch. Both call the evaluator by name --
``sttt/search.py`` at the Python tree and ``cpp/src/mcts.cpp`` via
``PyObject_CallMethod(model, "evaluate_many", ...)`` at the native one -- so
anything exposing ``evaluate`` and ``evaluate_many`` drops straight in. That is
what lets the container ship without PyTorch, which is ~500 MB of the image.

This mirrors ``BasePolicyValue.evaluate_many`` in ``sttt/learning.py`` exactly.
Where it diverges, the served agent stops being the agent that was measured.
"""
import os

import numpy as np

from ..encoding import encode_states


def _masked_softmax(logits, mask):
    """Row-wise softmax over legal actions only; illegal actions get exactly 0.

    The torch path fills masked logits with -inf (its -1e4 variant is guarded by
    `device.type == 'cuda'` for fp16 autocast and never applies on CPU), then
    softmaxes in float32 and renormalises in float64. Both steps are reproduced
    here, including the renormalisation -- dropping it leaves rows summing to
    1 +/- 1e-7, which biases the search priors in the last digits.
    """
    logits = np.where(mask, logits.astype(np.float32), -np.inf)
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits, dtype=np.float32)
    probabilities = probabilities.astype(np.float64)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


class OnnxEvaluator:
    """Policy/value evaluator backed by one shared ONNX Runtime session.

    An ``InferenceSession`` is thread-safe, so every game session shares one --
    a game costs a search tree, not a copy of the network.
    """

    def __init__(self, path, threads=None, encode_backend="auto"):
        import onnxruntime as ort

        options = ort.SessionOptions()
        # Left alone, ONNX Runtime takes every core. One busy game must not be
        # able to starve the host it is self-hosted on.
        if threads:
            options.intra_op_num_threads = int(threads)
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.encode_backend = encode_backend
        self.path = str(path)

    @classmethod
    def from_env(cls):
        return cls(os.environ.get("STTT_WEB_MODEL", "models/model-6000.onnx"),
                   threads=os.environ.get("STTT_WEB_THREADS", 4))

    def evaluate_many(self, states):
        if not states:
            return []
        if any(s.result is not None for s in states):
            raise ValueError("Neural evaluation expects nonterminal positions")
        features, mask = encode_states(states, backend=self.encode_backend)
        logits, values = self.session.run(None, {self.input_name: features})
        probabilities = _masked_softmax(np.asarray(logits), mask)
        return list(zip(probabilities, np.asarray(values, dtype=np.float64).ravel().tolist()))

    def evaluate(self, state):
        return self.evaluate_many([state])[0]
