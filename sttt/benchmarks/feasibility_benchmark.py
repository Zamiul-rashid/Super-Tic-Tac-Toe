"""Feasibility Benchmark: Python vs C++ Bitboard Engine for Super Tic-Tac-Toe."""
import time
import random
import os
import sys
import subprocess
from pathlib import Path

# Enforce strict 16-18GB memory cap
try:
    import resource
    _MAX_MEM_BYTES = 17 * 1024 * 1024 * 1024  # 17 GiB
    resource.setrlimit(resource.RLIMIT_AS, (_MAX_MEM_BYTES, _MAX_MEM_BYTES))
except Exception:
    pass

# Ensure paths
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "cpp"))

from sttt.env import State
from sttt.cpp_env import CppState, FastState, is_cpp_available, benchmark_rollouts, alphabeta_search
from sttt.bots import AlphaBetaBot
from sttt.learning import encode as py_encode
import numpy as np


def benchmark_legal_actions(iterations=100_000):
    print("\n--- 1. Legal Move Generation ---")
    s_py = State()
    s_cpp = FastState()

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = s_py.legal_actions()
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = s_cpp.legal_actions()
    t_cpp = time.perf_counter() - t0

    py_rate = iterations / t_py
    cpp_rate = iterations / t_cpp
    speedup = cpp_rate / py_rate

    print(f"Python legal_actions(): {py_rate:,.0f} calls/sec ({t_py*1e6/iterations:.2f} us/call)")
    print(f"C++ FastState.legal_actions() [Py binding]: {cpp_rate:,.0f} calls/sec ({t_cpp*1e6/iterations:.2f} us/call)")
    print(f"-> Speedup via Python binding: {speedup:.1f}x")
    return speedup


def benchmark_move_play(iterations=100_000):
    print("\n--- 2. Move Execution (play) ---")
    s_py = State()
    s_cpp = FastState()
    action = 40  # center cell

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = s_py.play(action)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = s_cpp.play(action)
    t_cpp = time.perf_counter() - t0

    t0 = time.perf_counter()
    # In-place in C++
    for _ in range(iterations):
        temp = FastState()
        temp.play_inplace(action)
    t_inplace = time.perf_counter() - t0

    py_rate = iterations / t_py
    cpp_rate = iterations / t_cpp
    inplace_rate = iterations / t_inplace

    print(f"Python State.play(): {py_rate:,.0f} moves/sec ({t_py*1e6/iterations:.2f} us/move)")
    print(f"C++ FastState.play() [functional]: {cpp_rate:,.0f} moves/sec ({t_cpp*1e6/iterations:.2f} us/move)")
    print(f"C++ FastState.play_inplace(): {inplace_rate:,.0f} moves/sec ({t_inplace*1e6/iterations:.2f} us/move)")
    print(f"-> Functional Speedup: {cpp_rate / py_rate:.1f}x")
    print(f"-> In-place Speedup: {inplace_rate / py_rate:.1f}x")
    return cpp_rate / py_rate


def benchmark_neural_encoding(iterations=50_000):
    print("\n--- 3. State Feature Encoding (289 floats) ---")
    s_py = State().play(40).play(38)
    s_cpp = FastState().play(40).play(38)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = py_encode(s_py)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = s_cpp.encode()
    t_cpp = time.perf_counter() - t0

    py_rate = iterations / t_py
    cpp_rate = iterations / t_cpp
    speedup = cpp_rate / py_rate

    print(f"Python encode(): {py_rate:,.0f} states/sec ({t_py*1e6/iterations:.2f} us/state)")
    print(f"C++ FastState.encode(): {cpp_rate:,.0f} states/sec ({t_cpp*1e6/iterations:.2f} us/state)")
    print(f"-> Speedup: {speedup:.1f}x")
    return speedup


def run_rollout_benchmark(num_games=1000):
    print(f"\n--- 4. Random Rollouts ({num_games} games) ---")

    # Python rollouts
    t0 = time.perf_counter()
    py_moves = 0
    for _ in range(num_games):
        s = State()
        while s.result is None:
            leg = s.legal_actions()
            a = random.choice(leg)
            s = s.play(a)
            py_moves += 1
    t_py = time.perf_counter() - t0
    py_gps = num_games / t_py
    py_mps = py_moves / t_py

    print(f"Python: {py_gps:,.1f} games/sec | {py_mps:,.1f} moves/sec (time: {t_py:.3f}s)")

    # C++ single-threaded (native C++ loop releasing GIL)
    cpp_games = 50_000
    res_1 = benchmark_rollouts(cpp_games, 1, 42)
    t_cpp1, moves_cpp1, gps_cpp1, mps_cpp1 = res_1
    print(f"C++ (1 thread): {gps_cpp1:,.1f} games/sec | {mps_cpp1:,.1f} moves/sec (time: {t_cpp1:.3f}s)")
    print(f"-> Single-core Speedup: {gps_cpp1 / py_gps:.1f}x (Games) | {mps_cpp1 / py_mps:.1f}x (Moves)")

    # C++ multi-threaded (10 threads)
    res_10 = benchmark_rollouts(200_000, 10, 42)
    t_cpp10, moves_cpp10, gps_cpp10, mps_cpp10 = res_10
    print(f"C++ (10 threads): {gps_cpp10:,.1f} games/sec | {mps_cpp10:,.1f} moves/sec (time: {t_cpp10:.3f}s)")
    print(f"-> Multi-core Speedup: {gps_cpp10 / py_gps:.1f}x (Games) | {mps_cpp10 / py_mps:.1f}x (Moves)")

    return gps_cpp1 / py_gps, gps_cpp10 / py_gps


def benchmark_alphabeta():
    print("\n--- 5. Alpha-Beta Search Comparison ---")
    py_bot = AlphaBetaBot(depth=3, node_budget=50_000)
    rng = np.random.default_rng(42)
    s_py = State()
    s_cpp = FastState()

    # Python AlphaBeta
    t0 = time.perf_counter()
    a_py = py_bot.choose(s_py, rng)
    t_py = time.perf_counter() - t0

    # C++ AlphaBeta
    t0 = time.perf_counter()
    a_cpp, nodes_cpp = alphabeta_search(s_cpp, 3, 50_000)
    t_cpp = time.perf_counter() - t0

    print(f"Python AlphaBeta(depth=3): action={a_py}, time={t_py*1000:.2f}ms")
    print(f"C++ AlphaBeta(depth=3): action={a_cpp}, nodes={nodes_cpp}, time={t_cpp*1000:.2f}ms")
    if t_cpp > 0:
        print(f"-> AlphaBeta Speedup: {t_py / t_cpp:.1f}x")


def benchmark_heuristic_eval(iterations=100_000):
    print("\n--- 6. Zero-Sum Heuristic Evaluation ---")
    from sttt.bots import value as py_value
    s_py = State().play(40).play(38)
    s_cpp = FastState().play(40).play(38)

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = py_value(s_py)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = s_cpp.evaluate()
    t_cpp = time.perf_counter() - t0

    py_rate = iterations / t_py
    cpp_rate = iterations / t_cpp
    speedup = cpp_rate / py_rate

    print(f"Python value(state): {py_rate:,.0f} evals/sec ({t_py*1e6/iterations:.2f} us/eval)")
    print(f"C++ FastState.evaluate(): {cpp_rate:,.0f} evals/sec ({t_cpp*1e6/iterations:.2f} us/eval)")
    print(f"-> Speedup: {speedup:.1f}x")
    return speedup


def benchmark_batched_encoding(batch_size=64, iterations=5_000):
    print(f"\n--- 7. Batched Feature Encoding (Batch size {batch_size}) ---")
    from sttt.cpp_env import encode_batch
    states_py = [State().play(40).play(38) for _ in range(batch_size)]
    states_cpp = [FastState().play(40).play(38) for _ in range(batch_size)]

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = np.stack([py_encode(s) for s in states_py])
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = encode_batch(states_cpp)
    t_cpp = time.perf_counter() - t0

    py_states_sec = (batch_size * iterations) / t_py
    cpp_states_sec = (batch_size * iterations) / t_cpp
    speedup = cpp_states_sec / py_states_sec

    print(f"Python np.stack(encode): {py_states_sec:,.0f} states/sec ({t_py*1e3/iterations:.2f} ms/batch)")
    print(f"C++ encode_batch: {cpp_states_sec:,.0f} states/sec ({t_cpp*1e3/iterations:.2f} ms/batch)")
    print(f"-> Batched Speedup: {speedup:.1f}x")
    return speedup


def main():
    print("=" * 65)
    print("  Super Tic-Tac-Toe Python vs C++ Bitboard Feasibility Benchmark")
    print("=" * 65)

    if not is_cpp_available():
        print("ERROR: C++ module not found. Please build it first.")
        return

    benchmark_legal_actions()
    benchmark_move_play()
    benchmark_neural_encoding()
    run_rollout_benchmark()
    benchmark_alphabeta()
    benchmark_heuristic_eval()
    benchmark_batched_encoding()

    print("\n" + "=" * 65)
    print("Benchmark completed successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
