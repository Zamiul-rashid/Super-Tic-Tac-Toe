"""MCTS Search Benchmark: Python TreeSearch vs High-Performance C++ MCTS Engine."""
import time
import os
import sys
from pathlib import Path
import numpy as np

# Enforce strict 16-18GB memory cap
try:
    import resource
    _MAX_MEM_BYTES = 17 * 1024 * 1024 * 1024  # 17 GiB
    resource.setrlimit(resource.RLIMIT_AS, (_MAX_MEM_BYTES, _MAX_MEM_BYTES))
except Exception:
    pass

repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "cpp"))

from sttt.env import State
from sttt.search import TreeSearch as PyTreeSearch, SearchConfig
from sttt.cpp_env import is_cpp_available, CppTreeSearch, benchmark_mcts, FastState
from sttt.bots import value as py_evaluate


class HeuristicEvaluator:
    """Heuristic evaluator wrapping sttt.bots.value."""
    def evaluate(self, state):
        legal = state.legal_actions()
        p = np.zeros(81, dtype=np.float32)
        if legal:
            p[legal] = 1.0 / len(legal)
        v = py_evaluate(state)
        return p, float(v)

    def evaluate_many(self, states):
        return [self.evaluate(s) for s in states]


class FastEvaluator:
    """Zero-overhead evaluator for raw MCTS tree traversal throughput."""
    def evaluate(self, state):
        legal = state.legal_actions()
        p = np.zeros(81, dtype=np.float32)
        if legal:
            p[legal] = 1.0 / len(legal)
        return p, 0.0

    def evaluate_many(self, states):
        return [self.evaluate(s) for s in states]


def run_mcts_benchmarks():
    print("================================================================================")
    print("        Super Tic-Tac-Toe MCTS Search Engine Benchmark (Python vs C++)          ")
    print("================================================================================")

    if not is_cpp_available() or CppTreeSearch is None:
        print("[Error] C++ engine or CppTreeSearch not available!")
        return

    s = State()
    cfg = SearchConfig(proofs=True, reuse=False)

    # 1. Raw Tree Traversal Throughput (TreeSearch with FastEvaluator)
    print("\n--- 1. Neural Tree Search Traversal (Python Evaluator Callback) ---")
    sims = 1000
    batch_size = 8
    fast_eval = FastEvaluator()

    # Python TreeSearch
    t0 = time.perf_counter()
    py_tree = PyTreeSearch(fast_eval, config=cfg)
    _ = py_tree.run(s, simulations=sims, batch_size=batch_size)
    t_py = time.perf_counter() - t0
    py_sims_sec = sims / t_py

    # C++ CppTreeSearch
    t0 = time.perf_counter()
    cpp_tree = CppTreeSearch(fast_eval, config=cfg)
    _ = cpp_tree.run(s, simulations=sims, batch_size=batch_size)
    t_cpp = time.perf_counter() - t0
    cpp_sims_sec = sims / t_cpp

    speedup_eval = cpp_sims_sec / py_sims_sec
    print(f"Python TreeSearch (1000 sims, batch={batch_size}): {py_sims_sec:,.1f} sims/sec ({t_py*1000:.2f} ms)")
    print(f"C++ CppTreeSearch (1000 sims, batch={batch_size}): {cpp_sims_sec:,.1f} sims/sec ({t_cpp*1000:.2f} ms)")
    print(f"-> Speedup Factor: {speedup_eval:.1f}x")

    # 2. Pure Native C++ Heuristic MCTS (Zero Python Overhead, GIL Released)
    print("\n--- 2. Native C++ MCTS Engine Throughput (GIL Released) ---")
    native_sims = 20000
    t0 = time.perf_counter()
    native_tree = CppTreeSearch(model=None, config=cfg)
    _ = native_tree.run(s, simulations=native_sims, batch_size=batch_size)
    t_native = time.perf_counter() - t0
    native_sims_sec = native_sims / t_native

    print(f"Python TreeSearch baseline: {py_sims_sec:,.1f} sims/sec")
    print(f"C++ Native MCTS (single thread, {native_sims} sims): {native_sims_sec:,.1f} sims/sec ({t_native*1000:.2f} ms)")
    print(f"-> Native MCTS Speedup: {native_sims_sec / py_sims_sec:.1f}x vs Python TreeSearch")

    # 3. Multi-Threaded C++ MCTS Throughput
    print("\n--- 3. Multi-Threaded C++ MCTS Scaling (sttt_cpp.benchmark_mcts) ---")
    thread_counts = [1, 2, 4, 8, 10]
    scaling_results = {}
    for threads in thread_counts:
        budget = 50000 * threads
        res = benchmark_mcts(budget, 8, threads)
        rate = res["simulations_per_sec"]
        scaling_results[threads] = res
        print(f"  {threads:2d} Thread{'s' if threads > 1 else ' '}: {rate:,.0f} sims/sec "
              f"({res['total_simulations']} sims in {res['elapsed_seconds']*1000:.1f} ms) "
              f"[{rate / py_sims_sec:,.1f}x vs Python]")

    # 4. Latency Analysis
    print("\n--- 4. MCTS Simulation Latency Profile ---")
    budgets = [64, 128, 512, 1024, 2048]
    print(f"{'Budget':<8} | {'Python Latency':<16} | {'C++ (Model) Latency':<20} | {'C++ Native Latency':<18} | {'Speedup':<10}")
    print("-" * 84)
    for b in budgets:
        # Python
        t0 = time.perf_counter()
        pt = PyTreeSearch(fast_eval, config=cfg)
        pt.run(s, simulations=b, batch_size=8)
        py_ms = (time.perf_counter() - t0) * 1000

        # C++ with Python model
        t0 = time.perf_counter()
        ct = CppTreeSearch(fast_eval, config=cfg)
        ct.run(s, simulations=b, batch_size=8)
        cpp_ms = (time.perf_counter() - t0) * 1000

        # C++ native
        t0 = time.perf_counter()
        nt = CppTreeSearch(model=None, config=cfg)
        nt.run(s, simulations=b, batch_size=8)
        nat_ms = (time.perf_counter() - t0) * 1000

        sp = py_ms / nat_ms if nat_ms > 0 else 0.0
        print(f"{b:<8} | {py_ms:>11.2f} ms    | {cpp_ms:>15.2f} ms     | {nat_ms:>13.3f} ms     | {sp:>7.1f}x")

    print("\n================================================================================")
    print("                           MCTS Benchmark Complete                              ")
    print("================================================================================")


if __name__ == "__main__":
    run_mcts_benchmarks()
