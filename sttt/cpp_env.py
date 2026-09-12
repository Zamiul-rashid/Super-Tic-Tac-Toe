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
    benchmark_rollouts = sttt_cpp.benchmark_rollouts
    alphabeta_search = sttt_cpp.alphabeta
else:
    FastState = None
    benchmark_rollouts = None
    alphabeta_search = None


from .env import State


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

    def encode(self) -> list[float]:
        return self._fast.encode()

    def render(self) -> str:
        return self._fast.render()

    def __eq__(self, other) -> bool:
        if isinstance(other, CppState):
            return self._fast == other._fast
        if isinstance(other, State):
            return (self.cells == other.cells and
                    self.boards == other.boards and
                    self.turn == other.turn and
                    self.forced == other.forced and
                    self.result == other.result)
        return False

    def __hash__(self) -> int:
        return hash(self._fast)

    def __repr__(self) -> str:
        return repr(self._fast)
