"""C++ Accelerated Bitboard Engine integration for Super Tic-Tac-Toe.

Provides high-performance bitboard state representation, move generation,
random rollouts, and Alpha-Beta search.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

# Attempt to load native sttt_cpp module
_cpp_dir = str(Path(__file__).resolve().parent.parent / "cpp")
if _cpp_dir not in sys.path:
    sys.path.insert(0, _cpp_dir)

try:
    import sttt_cpp
    _CPP_AVAILABLE = True
except ImportError:
    sttt_cpp = None
    _CPP_AVAILABLE = False


def is_cpp_available() -> bool:
    """Return True if C++ bitboard extension is compiled and available."""
    return _CPP_AVAILABLE


if _CPP_AVAILABLE:
    FastState = sttt_cpp.FastState
    FastNode = getattr(sttt_cpp, "FastNode", None)
    FastTreeSearch = getattr(sttt_cpp, "FastTreeSearch", None)
    CppTreeSearch = FastTreeSearch
    # NOTE: CppTreeSearch does NOT support the `opponent` parameter for adaptive
    # opponent modeling. When opponent modeling is needed (e.g., human play mode),
    # use Python TreeSearch from sttt.search instead.
    benchmark_rollouts = sttt_cpp.benchmark_rollouts
    benchmark_mcts = getattr(sttt_cpp, "benchmark_mcts", None)
    alphabeta_search = sttt_cpp.alphabeta
    cpp_evaluate = sttt_cpp.evaluate
else:
    FastState = None
    FastNode = None
    FastTreeSearch = None
    CppTreeSearch = None
    benchmark_rollouts = None
    benchmark_mcts = None
    alphabeta_search = None
    cpp_evaluate = None


from .env import State
import numpy as np


def encode_batch(states) -> np.ndarray:
    """Encode a sequence of FastState, CppState, or State into a (N, 289) float32 numpy array."""
    if not _CPP_AVAILABLE:
        raise RuntimeError("C++ bitboard engine (sttt_cpp) is not compiled or available")
    if not states:
        return np.empty((0, 289), dtype=np.float32)
    converted = []
    need_convert = False
    for s in states:
        if isinstance(s, (FastState, CppState)):
            converted.append(s)
        else:
            converted.append(to_fast_state(s))
            need_convert = True
    raw = sttt_cpp.encode_batch(converted if need_convert else states)
    return np.frombuffer(raw, dtype=np.float32).reshape(len(states), 289)


def to_python_state(fast_state: FastState) -> State:
    """Convert a C++ FastState instance to a standard Python State."""
    return State(
        cells=fast_state.cells,
        boards=fast_state.boards,
        turn=fast_state.turn,
        forced=fast_state.forced,
        result=fast_state.result,
    )


def to_fast_state(py_state: State) -> FastState:
    """Convert a standard Python State instance to a C++ FastState."""
    if not _CPP_AVAILABLE:
        raise RuntimeError("C++ bitboard engine (sttt_cpp) is not compiled or available")
    return FastState(
        cells=py_state.cells,
        boards=py_state.boards,
        turn=py_state.turn,
        forced=py_state.forced,
        result=py_state.result,
    )


class CppState:
    """Drop-in replacement for sttt.env.State backed by C++ bitboards."""

    def __init__(self, cells=None, boards=None, turn=1, forced=-1, result=None, _fast=None):
        if not _CPP_AVAILABLE:
            raise RuntimeError("C++ engine not available. Run 'make -C cpp' first.")
        if _fast is not None:
            self._fast = _fast
        elif cells is None and boards is None and turn == 1 and forced == -1 and result is None:
            self._fast = FastState()
        else:
            self._fast = FastState(cells, boards, turn, forced, result)

    @property
    def cells(self) -> tuple[int, ...]:
        return self._fast.cells

    @property
    def boards(self) -> tuple[int, ...]:
        return self._fast.boards

    @property
    def turn(self) -> int:
        return self._fast.turn

    @property
    def forced(self) -> int:
        return self._fast.forced

    @property
    def result(self) -> int | None:
        return self._fast.result

    def legal_actions(self) -> list[int]:
        return self._fast.legal_actions()

    def play(self, action: int) -> CppState:
        nxt = self._fast.play(action)
        return CppState(_fast=nxt)

    def play_inplace(self, action: int) -> CppState:
        self._fast.play_inplace(action)
        return self

    def encode(self) -> list[float]:
        return self._fast.encode()

    def encode_bytes(self) -> bytes:
        return self._fast.encode_bytes()

    def evaluate(self) -> float:
        return self._fast.evaluate()

    def value(self) -> float:
        return self._fast.evaluate()

    def render(self) -> str:
        return self._fast.render()

    def __eq__(self, other) -> bool:
        if isinstance(other, CppState):
            return self._fast == other._fast
        if isinstance(other, (State, FastState)):
            return self._fast == other
        return False

    def __hash__(self) -> int:
        return hash(self._fast)

    def __repr__(self) -> str:
        return repr(self._fast)

    def __reduce__(self):
        return (CppState, (self.cells, self.boards, self.turn, self.forced, self.result))


class CppAlphaBetaBot:
    """High-speed C++ Alpha-Beta bot executing millions of nodes/sec."""

    def __init__(self, depth=3, node_budget=100000, name=None):
        if not _CPP_AVAILABLE:
            raise RuntimeError("C++ engine not available. Run 'make -C cpp' first.")
        self.depth = depth
        self.node_budget = node_budget
        self._name = name or f"cpp-alphabeta-d{depth}"

    @property
    def name(self) -> str:
        return self._name

    def choose(self, state, rng=None) -> int:
        if state.result is not None or not state.legal_actions():
            raise ValueError("Cannot choose a move in a terminal state")
        fast_s = state._fast if isinstance(state, CppState) else to_fast_state(state)
        action, _ = alphabeta_search(fast_s, self.depth, self.node_budget)
        if action < 0:
            legal = state.legal_actions()
            return int(rng.choice(legal) if rng is not None else legal[0])
        return action

