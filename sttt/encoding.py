"""Torch-free state encoding shared by training, search and the ONNX runtime.

These functions were part of ``learning.py``, which imports torch at module
level. Nothing in them needs torch -- they are numpy plus the optional native
encoder -- and the ONNX serving path must import them without dragging a
500 MB dependency into the container. ``learning`` re-exports them, so every
existing caller keeps working unchanged.
"""
import numpy as np

INPUTS = 81*3 + 9*4 + 10

def encode(state):
    cells = np.array(state.cells) * state.turn
    boards = np.array(state.boards)
    return np.concatenate([(cells == k).astype(np.float32) for k in (0,1,-1)] +
                          [(boards == k).astype(np.float32) for k in (0,state.turn,-state.turn,2)] +
                          [np.eye(10, dtype=np.float32)[state.forced+1]])

def legal_masks(states):
    """bool[N, 81] legal-action masks, owned and contiguous."""
    mask = np.zeros((len(states), 81), dtype=bool)
    for i, state in enumerate(states):
        mask[i, state.legal_actions()] = True
    return mask


def encode_states(states, backend='auto'):
    """Owned float32[N, 289] features plus bool[N, 81] masks.

    `backend` is explicit on purpose. The previous code probed
    `is_cpp_available()` and used native encoding whenever the extension
    happened to be importable -- so a run that deliberately requested the
    Python reference path was still encoded natively, which is exactly the
    comparison the reference path exists to make.

    The returned features are always an owned, C-contiguous array. The native
    encoder hands back a read-only view over immutable Python bytes, which is
    not a valid owner for `torch.from_numpy`; copying is the correct behaviour
    until the extension exposes an owner-backed buffer.
    """
    if backend not in ('auto', 'cpp', 'python'):
        raise ValueError(f"unknown encode backend {backend!r}")
    if not states:
        return np.empty((0, INPUTS), dtype=np.float32), np.empty((0, 81), dtype=bool)

    use_cpp = False
    if backend in ('auto', 'cpp'):
        try:
            from .cpp_env import encode_batch as _cpp_encode_batch, is_cpp_available
            use_cpp = bool(is_cpp_available())
        except ImportError:
            use_cpp = False
        if backend == 'cpp' and not use_cpp:
            raise RuntimeError(
                'encode backend "cpp" requested but the native encoder is '
                'unavailable; refusing to substitute the Python encoder.')

    if use_cpp:
        features = np.ascontiguousarray(_cpp_encode_batch(states), dtype=np.float32)
    else:
        features = np.ascontiguousarray(
            np.stack([encode(s) for s in states]), dtype=np.float32)
    if not features.flags.owndata or not features.flags.writeable:
        features = features.copy()
    return features, legal_masks(states)
