# Super Tic-Tac-Toe: Project Handover & Engineering Documentation

> **Status, 2026-09-14 (M11 reconciliation).** Everything below this note is
> historical: it describes the `cpp` and `testing` branches as of September 12.
> The canonical checklist and result ledger is `TRAINING_READINESS_PLAN.md`;
> the canonical launcher is `./train.sh` (`train_v2.sh` is deprecated). Claims
> below that are superseded by measurement: the `encode_batch` 8.6M states/s
> figure was taken on one cached state and is not zero-copy in the pipeline
> (5.54M states/s on varied positions, see `BENCHMARK_REPORT.md`); the
> "~15x–20x self-play speed-up, hours rather than days" forecast is withdrawn
> (measured end-to-end pipeline speed-up is 1.71x; runtime comes from the
> readiness pilot ETA); the 258-test count and the 87.5% / 1812.9 Elo
> tournament result are the numbers at commit `bc63578`, not current
> measurements. The interpreter path is `/home/mt/miniconda3/envs/sttt/bin/python`
> on this machine (see the plan §6).

**Date:** September 12, 2026
**Active Development Branches:**

- `cpp` (High-Performance C++ Bitboard Engine & C++ MCTS Search Engine, commits `be29e46` and `ead0133`)
- `testing` (Google DeepMind OpenSpiel Adapter & 4-Way Benchmark Tournament, commit `bc63578`)
  **Target Python Environment:** Conda environment at `/home/entropy/miniconda3/envs/sttt/bin/python`

---

## 1. Executive Summary & Project Status

### 1.1 Key Milestones Achieved

1. **Google DeepMind `open_spiel` Integration (Branch `testing`):**
   - Built the `OpenSpielBot` adapter with coordinate bijections between `sttt` (0..80) and OpenSpiel UTTT subgame structures.
   - Conducted a 4-way, 120-game paired benchmark tournament.
   - The neural model scored **87.5% overall (50W / 5D / 5L)**, winning 1st place with **1812.9 Bayesian Elo**, including a **20–0 clean sweep against Google DeepMind's OpenSpiel MCTS**.
2. **Deep Self-Play Training Run:**
   - Completed **5,968 iterations** (>95,400 deep self-play games at 512 simulations) with population league and D4 dihedral symmetry augmentation.
   - Checkpoints saved safely in `runs/big_run/latest.pt` (359 MB) and `runs/big_run/best.pt`.
3. **C++ Bitboard Engine & Python Extension (Branch `cpp` — Chunk 1):**
   - Packed state into **46 bytes** (`static_assert(sizeof(BoardState) == 46)`), fitting in an L1 cache line.
   - $O(1)$ win checking via 512-byte compile-time lookup table (`WIN_TABLE`).
   - Single-core rollouts: **753,800 games/sec (136.8x faster)** than pure Python.
   - 10-core rollouts: **5,039,529 games/sec (914.7x faster)** than pure Python.
   - Python C-API module (`sttt_cpp`) with zero-copy vectorized batch encoding (`encode_batch`, 8.6M states/s).
4. **C++ MCTS Tree Search Engine (Branch `cpp` — Chunk 2):**
   - Implemented `MCTSEngine` with contiguous 96-byte `MCTSNode` arena allocation (`cpp/include/sttt_mcts.hpp`, `cpp/src/mcts.cpp`).
   - Traversal speedup: **210,030 sims/sec (19.5x faster)** with Python neural model; **520,324 sims/sec (48.4x faster)** native single-thread; **3.22M sims/sec (300x faster)** multi-threaded.
   - 512-simulation latency dropped from **41.58 ms** to **0.51 ms (81.6x faster)**.
5. **System Stability & Memory Limits:**
   - Strict 16–18 GB virtual memory ceilings enforced across all runners (`resource.setrlimit(resource.RLIMIT_AS, (17 GiB, 17 GiB))` and `ulimit -v 18874368`).
   - Repository test suite: **258 / 258 unit tests passing (100% PASS)**.

---

## 2. Chunk 1: Standalone C++ Bitboard Engine (Ready for Use)

### 2.1 File Map

- **C++ Core:** [`cpp/include/sttt_core.hpp`](file:///home/entropy/Code/Super-Tic-Tac-Toe/cpp/include/sttt_core.hpp) (Header-only state & bitwise rules)
- **C-API Wrapper:** [`cpp/include/sttt_c_api.h`](file:///home/entropy/Code/Super-Tic-Tac-Toe/cpp/include/sttt_c_api.h), [`cpp/src/sttt_c_api.cpp`](file:///home/entropy/Code/Super-Tic-Tac-Toe/cpp/src/sttt_c_api.cpp)
- **Python C-Extension:** [`cpp/src/python_module.cpp`](file:///home/entropy/Code/Super-Tic-Tac-Toe/cpp/src/python_module.cpp) (Defines `sttt_cpp` module)
- **High-Level Python Wrapper:** [`sttt/cpp_env.py`](file:///home/entropy/Code/Super-Tic-Tac-Toe/sttt/cpp_env.py) (Exposes `FastState`, `CppState`, `encode_batch`)

### 2.2 Memory Layout

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
static_assert(sizeof(BoardState) == 46, "BoardState must be 46 bytes");
```

### 2.3 Python Quickstart & Integration Guide

```python
from sttt.cpp_env import FastState, CppState, encode_batch, alphabeta_search
import numpy as np
import torch
import pickle

# --- 1. Basic State Manipulation ---
state = FastState()                  # 46 bytes packed
legal_moves = state.legal_actions()  # ~250 ns via __builtin_ctz
next_state = state.play(legal_moves[0]) # Functional transition (immutable)

# In-place move execution for ultra-fast rollouts
mutable_copy = state.clone()
mutable_copy.play_inplace(40)

# --- 2. PyTorch Feature Tensor Generation ---
# encode_batch returns contiguous (N, 289) float32 numpy array at 8.6M states/sec
states = [state, next_state]
feature_array = encode_batch(states)
tensor = torch.from_numpy(feature_array)  # Zero-copy PyTorch tensor

# --- 3. Serialization for Multiprocessing ---
pickled_bytes = pickle.dumps(state)
restored = pickle.loads(pickled_bytes)
assert restored == state  # Bidirectional equality with Python State
```

---

## 3. Chunk 2: C++ MCTS Search Engine

### 3.1 File Map

- **MCTS Header:** [`cpp/include/sttt_mcts.hpp`](file:///home/entropy/Code/Super-Tic-Tac-Toe/cpp/include/sttt_mcts.hpp) (Arena allocator, PUCT descent, proofs, virtual loss)
- **MCTS C++ Module:** [`cpp/src/mcts.cpp`](file:///home/entropy/Code/Super-Tic-Tac-Toe/cpp/src/mcts.cpp)
- **Python Search Interface:** [`sttt/cpp_env.py`](file:///home/entropy/Code/Super-Tic-Tac-Toe/sttt/cpp_env.py#L125-L160) (`CppTreeSearch`)
- **Unit & Differential Tests:** [`tests/test_cpp_mcts.py`](file:///home/entropy/Code/Super-Tic-Tac-Toe/tests/test_cpp_mcts.py) (7 tests)

### 3.2 Python MCTS Usage

```python
from sttt.cpp_env import CppTreeSearch
from sttt.search import SearchConfig
from sttt.env import State

# Initialize search tree with custom config
cfg = SearchConfig(soft_pruning=True, proofs=True, reuse=True, c_puct=1.5)

# Option A: Pure Native Heuristic MCTS (Zero Python overhead, GIL released)
tree = CppTreeSearch(model=None, config=cfg)
root_state = State()
policy_distribution = tree.run(root_state, simulations=512, batch_size=16)

# Option B: Neural Guided MCTS (Accepts PyTorch model with evaluate/evaluate_many)
tree_neural = CppTreeSearch(model=neural_net, config=cfg)
pi = tree_neural.run(root_state, simulations=512, batch_size=16)
```

---

## 4. Empirical Performance Benchmarks

Conducted on Intel Core i7-12650H (10 Cores, 16 Threads, AVX2, BMI2):

### 4.1 State Primitives & Rollout Benchmarks

| Benchmark Operation                | Pure Python (`sttt.env.State`) | C++ via Python (`sttt_cpp`) | C++ Pure Native (`bench_main`) |           Speedup Multiplier           |
| :--------------------------------- | :----------------------------: | :-------------------------: | :----------------------------: | :------------------------------------: |
| **Legal Move Generation**          |        316,400 calls/s         |      4,026,031 calls/s      |      148,100,000 calls/s       |   **12.7x** (Py) / **468x** (Native)   |
| **Move Execution (`play`)**        |        211,112 moves/s         |     23,758,500 moves/s      |      322,580,645 moves/s       | **112.5x** (Py) / **>1,500x** (Native) |
| **Move In-Place (`play_inplace`)** |        N/A (immutable)         |     14,135,536 moves/s      |      322,580,645 moves/s       | **67.0x** (Py) / **>1,500x** (Native)  |
| **Feature Encoding ($B=64$)**      |        98,112 states/s         |     8,612,550 states/s      |      45,454,545 states/s       |   **87.8x** (Py) / **463x** (Native)   |
| **Zero-Sum Heuristic Eval**        |         70,521 evals/s         |     11,173,852 evals/s      |      ~35,000,000 evals/s       |  **158.4x** (Py) / **>490x** (Native)  |
| **Random Rollouts (1 thread)**     |         5,509 games/s          |       753,800 games/s       |        692,075 games/s         |           **136.8x speedup**           |
| **Random Rollouts (10 threads)**   |         ~5,509 games/s         |      5,039,529 games/s      |       5,070,484 games/s        |           **914.7x speedup**           |
| **Alpha-Beta Search (depth 3)**    |        58.21 ms / move         |       0.12 ms / move        |         0.10 ms / move         |           **506.0x speedup**           |

### 4.2 MCTS Search Engine Scaling

| MCTS Configuration                     | Throughput (Simulations/sec) | 512-Sim Latency |        Speedup vs Python        |
| :------------------------------------- | :--------------------------: | :-------------: | :-----------------------------: |
| **Python `TreeSearch` Baseline**       |        10,751 sims/s         |    41.58 ms     |         1.0x (Baseline)         |
| **C++ `CppTreeSearch` (Python Model)** |        210,030 sims/s        |     2.48 ms     |            **19.5x**            |
| **C++ Native MCTS (1 Thread)**         |        520,324 sims/s        |     0.51 ms     | **48.4x (81.6x lower latency)** |
| **C++ Native MCTS (4 Threads)**        |       1,577,909 sims/s       |     0.18 ms     |           **146.8x**            |
| **C++ Native MCTS (8 Threads)**        |       2,904,437 sims/s       |     0.11 ms     |           **270.2x**            |
| **C++ Native MCTS (10 Threads)**       |       3,221,329 sims/s       |     0.09 ms     |           **299.6x**            |

---

## 5. Verification & Quality Assurance

1. **Lockstep Differential Fuzzing:**
   - 2,000 complete games (>120,000 moves) played in lockstep comparing `sttt.env.State` and `sttt_cpp.FastState`.
   - 100% exact parity across legal moves, active board constraints, local wins, macro wins, and draws.
2. **Complete Test Suite (258 / 258 PASS):**
   ```bash
   ulimit -v 18874368 && /home/entropy/miniconda3/envs/sttt/bin/python -m unittest discover tests
   ```
   Output: `Ran 258 tests in 217.445s - OK`.
3. **Strict Compilation Flags:**
   - Both `Makefile` and `CMakeLists.txt` build with `-O3 -march=native -Wall -Wextra -Werror` (zero compiler warnings/errors).

---

## 6. How to Build & Run

### 6.1 Building C++ Binaries

```bash
# Using Makefile
make -C cpp clean && make -C cpp

# Or using CMake
cmake -B build -S .
cmake --build build -j4

# Or using Python setuptools
python setup.py build_ext --inplace
```

### 6.2 Running Benchmarks (Memory Guarded)

```bash
# State & rollout feasibility benchmark
ulimit -v 18874368 && /home/entropy/miniconda3/envs/sttt/bin/python -m sttt.benchmarks.feasibility_benchmark

# MCTS search engine benchmark
ulimit -v 18874368 && /home/entropy/miniconda3/envs/sttt/bin/python -m sttt.benchmarks.mcts_benchmark

# Native C++ binary benchmark
./cpp/bench_main
```

### 6.3 Running Unit Tests

```bash
# Core bitboard tests (29 tests)
/home/entropy/miniconda3/envs/sttt/bin/python -m unittest tests/test_cpp_engine.py

# C++ MCTS tests (7 tests)
/home/entropy/miniconda3/envs/sttt/bin/python -m unittest tests/test_cpp_mcts.py

# Full test suite (258 tests)
ulimit -v 18874368 && /home/entropy/miniconda3/envs/sttt/bin/python -m unittest discover tests
```

---

## 7. Recommended Next Steps for Downstream Work

1. **Self-Play Training Loop Acceleration:**
   - Replace Python `TreeSearch` in `sttt/selfplay.py` with `CppTreeSearch`.
   - Because `CppTreeSearch` produces 210k sims/sec (vs 10.7k sims/sec), self-play generation time will drop by ~15x–20x, allowing 5,000 iterations to run in hours rather than days.
2. **Tournament Evaluation with C++ MCTS:**
   - Plug `CppTreeSearch` into `sttt/ai.py tournament` to run massive 500-game evaluation matches in seconds.
3. **LibTorch / Direct CUDA C++ Inference (Optional Future Milestone):**
   - Bind LibTorch C++ API directly into `cpp/src/mcts.cpp` to bypass Python model evaluation altogether, unlocking the full 520,000–3,220,000 sims/sec native throughput.
