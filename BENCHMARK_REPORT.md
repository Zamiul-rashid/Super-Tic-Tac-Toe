# Super Tic-Tac-Toe C++ Bitboard Engine Benchmark Report

**Date**: 2026-09-12  
**Branch**: `cpp`  
**Hardware / Toolchain**: GCC 15.2.0, x86_64, Linux, Python 3.14.4 (`.venv`)  
**Status**: Target >= 50x speedup achieved and exceeded across all core primitives and rollouts.

---

## 1. Executive Summary

The high-performance C++ bitboard engine (`sttt_cpp`) replaces Python object allocations and dictionary/tuple traversals with 64-bit/16-bit packed bitboards, branchless compile-time lookup tables (`WIN_TABLE`), and hardware bit-scan intrinsics (`__builtin_ctz`, `blsr`).

Key highlights:
- **Memory Footprint**: `sizeof(BoardState)` is **46 bytes** (verified at compile time via `static_assert(sizeof(BoardState) == 46)`), fitting within a single 64-byte L1 cache line.
- **Rollout Throughput**: Single-core C++ rollouts achieve **675,528 games/sec** (**117.5x speedup** vs Python's 5,751 games/sec), exceeding the 50x target by 2.35x.
- **Multi-Core Scaling**: 10-thread C++ rollouts achieve **4,813,524 games/sec** (**837.0x speedup** vs Python).
- **Native Move Generation**: Native C++ move generator produces **156.3 million calls/sec** (6.4 ns/call).
- **Alpha-Beta Search**: Depth-3 search executes in **0.13 ms** (**520.2x speedup** vs Python's 67.88 ms).

---

## 2. Benchmark Comparison Tables

### 2.1 Game Rollouts Throughput (Target: >= 50x Speedup)

| Engine Configuration | Games / sec | Moves / sec | Latency / Game | Speedup Factor |
|----------------------|------------:|------------:|---------------:|---------------:|
| Python Baseline (`sttt.env.State`) | 5,751 | 338,174 | 173.9 µs | 1.0x (Baseline) |
| C++ Binding (1 thread) | 675,528 | 39,814,052 | 1.48 µs | **117.5x** |
| C++ Native (`bench_main`, 1 thread) | 708,462 | 41,734,173 | 1.41 µs | **123.2x** |
| C++ Binding (10 threads) | 4,813,524 | 283,570,256 | 0.21 µs | **837.0x** |
| C++ Native (`bench_main`, 10 threads) | 5,072,322 | 298,849,619 | 0.20 µs | **881.9x** |

*Acceptance Criteria: Target >= 50x rollout speedup is met and substantially exceeded (117.5x single-core, 837.0x multi-core).*

---

### 2.2 Core State Transitions & Move Execution

| Operation | Python Baseline | C++ Binding | C++ Native | Speedup (Binding) | Speedup (Native) |
|-----------|----------------:|------------:|-----------:|------------------:|-----------------:|
| Legal Move Generation | 338,301 calls/s (2.96 µs) | 4,971,558 calls/s (0.20 µs) | 156.3M calls/s (6.4 ns) | **14.7x** | **462.0x** |
| Move Play (Functional) | 229,602 moves/s (4.36 µs) | 27,122,204 moves/s (0.04 µs) | 400.0M+ moves/s (<2.5 ns) | **118.1x** | **>1,740x** |
| Move Play (In-place) | N/A (immutable) | 15,005,192 moves/s (0.07 µs) | 400.0M+ moves/s (<2.5 ns) | **65.4x** | **>1,740x** |
| Zero-Sum Heuristic Evaluation | 70,847 evals/s (14.1 µs) | 10,619,660 evals/s (0.09 µs) | 35.0M+ evals/s (28.5 ns) | **149.9x** | **>490x** |

---

### 2.3 Neural Network Feature Encoding (289 floats)

| Encoding Mode | Python (`sttt.learning.encode`) | C++ (`sttt_cpp`) | Speedup |
|---------------|--------------------------------:|-----------------:|--------:|
| Single State (`encode()`) | 103,249 states/s (9.69 µs) | 559,956 states/s (1.79 µs) | **5.4x** |
| Batched Encoding (Batch 64, `encode_batch`) | 90,527 states/s (0.71 ms/batch) | 8,156,303 states/s (0.01 ms/batch) | **90.1x** |
| Native C++ Direct Memory (`encode(float*)`) | N/A | 45,454,545 states/s (22.0 ns) | **440.2x** |

---

### 2.4 Alpha-Beta Search Performance

| Search Depth | Explored Nodes | Python Time | C++ Time | Speedup Factor | C++ Search Rate |
|--------------|---------------:|------------:|---------:|---------------:|----------------:|
| Depth 1 | 81 | ~0.8 ms | <0.01 ms | >80x | 5.4 M nodes/s |
| Depth 2 | 200 | ~2.5 ms | 0.01 ms | >180x | 13.6 M nodes/s |
| Depth 3 | 1,453 | 67.88 ms | 0.13 ms | **520.2x** | 12.2 M nodes/s |
| Depth 4 | 3,479 | ~210 ms | 0.27 ms | >750x | 12.7 M nodes/s |
| Depth 5 | 20,514 | ~1,850 ms | 2.11 ms | >870x | 9.7 M nodes/s |
| Depth 6 | 72,318 | ~8,200 ms | 6.57 ms | >1,200x | 11.0 M nodes/s |

---

### 2.5 Perft (Move Path Enumeration Validation)

Recursive move tree enumeration from root state confirms 100% legal tree generation with zero branch divergence:

| Depth | Total Nodes | Native C++ Time | Throughput |
|-------|------------:|----------------:|-----------:|
| 1 | 81 | 0.0000 s | 122.0 M nodes/s |
| 2 | 720 | 0.0000 s | 660.6 M nodes/s |
| 3 | 6,336 | 0.0000 s | 575.1 M nodes/s |
| 4 | 55,080 | 0.0001 s | 570.2 M nodes/s |
| 5 | 473,256 | 0.0009 s | 525.6 M nodes/s |

---

### 2.6 Phase 2: Monte Carlo Tree Search (MCTS) Benchmark (Target: Accelerated Neural Self-Play)

Benchmark executed comparing Python reference `sttt.search.TreeSearch` against C++ high-performance engine `sttt_cpp.FastTreeSearch` / `CppTreeSearch` across identical search configurations (PUCT $c_{puct}=1.5$, proofs enabled, batch size 8).

#### 2.6.1 Neural Model Tree Traversal (Python Evaluator Callback)

| Engine Implementation | Simulations / sec | 1000-Sim Search Latency | Speedup Factor |
|-----------------------|------------------:|------------------------:|---------------:|
| Python `TreeSearch` | 11,819 sims/s | 84.61 ms | 1.0x (Baseline) |
| C++ `CppTreeSearch` (Python Callback) | 210,785 sims/s | 4.74 ms | **17.8x** |

#### 2.6.2 Native C++ MCTS Engine Throughput (Zero Python Overhead, GIL Released)

| Execution Configuration | Simulations / sec | 20,000-Sim Latency | Speedup vs Python |
|-------------------------|------------------:|-------------------:|------------------:|
| Python `TreeSearch` Baseline | 11,819 sims/s | 1,692.2 ms (est.) | 1.0x (Baseline) |
| C++ Native MCTS (1 thread) | 629,286 sims/s | 31.78 ms | **53.2x** |
| C++ Native MCTS (2 threads) | 1,133,629 sims/s | 17.65 ms | **95.9x** |
| C++ Native MCTS (4 threads) | 1,917,583 sims/s | 10.43 ms | **162.2x** |
| C++ Native MCTS (8 threads) | 2,718,519 sims/s | 7.36 ms | **230.0x** |
| C++ Native MCTS (10 threads) | 2,860,016 sims/s | 6.99 ms | **242.0x** |

#### 2.6.3 Latency Profile Across Simulation Budgets (Batch Size = 8)

| Simulation Budget | Python Latency | C++ (Model) Latency | C++ Native Latency | Native Speedup |
|-------------------|---------------:|--------------------:|-------------------:|---------------:|
| 64 simulations | 5.10 ms | 0.34 ms | 0.075 ms | **68.0x** |
| 128 simulations | 9.54 ms | 0.62 ms | 0.111 ms | **85.9x** |
| 512 simulations | 37.36 ms | 2.32 ms | 0.457 ms | **81.8x** |
| 1,024 simulations | 81.36 ms | 4.38 ms | 0.970 ms | **83.9x** |
| 2,048 simulations | 181.57 ms | 9.34 ms | 1.947 ms | **93.2x** |

---

## 3. Memory Footprint Attestation

```cpp
#pragma pack(push, 1)
struct BoardState {
    uint16_t x_cells[9];   // 18 bytes: 9 bits per subgrid for X
    uint16_t o_cells[9];   // 18 bytes: 9 bits per subgrid for O
    uint16_t macro_x;      //  2 bytes: 9 bits for won boards by X
    uint16_t macro_o;      //  2 bytes: 9 bits for won boards by O
    uint16_t macro_draw;   //  2 bytes: 9 bits for drawn boards
    int8_t   turn;         //  1 byte:  1 (X) or -1 (O)
    int8_t   forced;       //  1 byte:  -1 (free) or 0..8
    int8_t   result;       //  1 byte:  1, -1, 0, or RESULT_ONGOING (2)
    uint8_t  padding;      //  1 byte:  alignment
};
#pragma pack(pop)

static_assert(sizeof(BoardState) == 46, "BoardState size must be 46 bytes");
```

- Total struct size: **46 bytes**.
- Cache line alignment: 46 bytes < 64-byte L1 cache line.
- Requirement (< 64 bytes per state): **SATISFIED**.
- MCTS Node size (`sizeof(MCTSNode)`): **96 bytes** (flat contiguous arena allocation, zero heap fragmentation).

---

## 4. Test Suite and Parity Non-Regression

- Total discovered unit tests: **258 unit tests** in `.venv/bin/python -m unittest discover tests`.
- Full test pass rate: **258/258 (100% PASS, 0 failures, 0 errors)**.
- Lockstep differential parity verified across **> 120,000 paired moves** with 100% exact match against Python `sttt.env.State`.
- Bit-for-bit identical MCTS visit distributions across all 81 actions verified against Python `TreeSearch`.
- Zero compiler warnings or errors under `-Wall -Wextra -Werror` in GCC 15.
