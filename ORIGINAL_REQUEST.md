# Original User Request

## Initial Request — 2026-09-10T19:19:14Z

Build a production-grade automated testing and benchmarking tournament suite for the Super Tic-Tac-Toe AI project on the `testing` branch. The suite integrates external state-of-the-art open-source engines (e.g. CodinGame Legend MCTS bitboard bot, utttai AlphaZero), supports high-volume 100–500 game paired tournament matches, computes Elo/Glicko-2 ratings, conducts simulation sweeps (128 to 5000), and exports structured visual benchmark charts.

Working directory: /home/entropy/Code/Super-Tic-Tac-Toe
Integrity mode: development

## Requirements

### R1. External Engine Adapter Subsystem
Build pluggable engine adapters for open-source Super Tic-Tac-Toe opponents in `sttt/bots.py`:
- `ExternalProcessBot`: Spawns external engine binaries (e.g. CodinGame C++/Rust engines) via `subprocess.Popen` communicating over `stdin`/`stdout` with configurable timeouts and robust error handling.
- `StyleBot`: Wraps behavioral policies (`center`, `corners`, `local-win`, `global-win`) into the standard bot interface.
- Native heuristic search bots: `AlphaBetaBot` and `TacticalBot`.
- Generational neural checkpoint matching: Load and pit different model checkpoints against each other.

### R2. Tournament & Rating Engine (`sttt/tournament.py`)
Implement an automated multi-round tournament orchestrator:
- Paired seat alternation: Each matchup plays in pairs from identical seeded random opening plies (0–4 plies), alternating who plays X and O to eliminate first-player color bias.
- Configurable sample sizes (20 to 500 games per matchup).
- Simulation budget sweeps (128, 512, 1024, 5000 simulations).
- Mathematical Bayesian Elo and Glicko-2 skill rating calculation from tournament outcome matrices with 95% confidence intervals.

### R3. Opponent Adaptation Verifier (`tests/test_adaptation.py`)
A dedicated automated test suite that verifies the Bayesian opponent modeling module (`sttt/opponent.py`:
- Measures how quickly the agent's belief state detects skewed styles (e.g. corner-heavy, center-heavy, local greed).
- Validates win-rate delta with adaptation enabled vs. disabled against predictable opponents.

### R4. CLI Integration & Visualization
- Add `tournament` command to `sttt/ai.py` allowing one-line tournament execution:
  `python -m sttt.ai tournament --checkpoint runs/big_run/latest.pt --opponents alphabeta tactical corners --games 50 --simulations 512`
- Export clean tournament scoreboards, win-rate breakdown charts, and CSV summaries to `runs/<run>/tournaments/`.

## Acceptance Criteria

### Adapter Integrity
- [ ] All external and internal bots conform to a uniform `choose(state, rng)` interface.
- [ ] `ExternalProcessBot` cleanly handles process crashes, timeouts (max 5s per move), and invalid output without freezing.
- [ ] Mock external echo bot unit tests pass.

### Tournament Verification
- [ ] Automated tournament executes cleanly with opening pairs ensuring zero first-player color bias.
- [ ] Elo ratings and error margins are mathematically calculated and exported to CSV/JSON reports.
- [ ] Simulation sweeps (128 to 5000) produce structured scaling reports.

### Codebase Non-Regression
- [ ] All 26 existing unit tests continue to pass with 0 failures on Conda `sttt`.
- [ ] All code strictly lives on the `testing` branch without disturbing active training in tmux.

## Follow-up — 2026-09-11T18:30:28Z

This is a single self-contained feature; keep it small and focused. Use at most 5 agents total.

Integrate Google DeepMind's `open_spiel` Ultimate Tic-Tac-Toe engine as a benchmark opponent in the `sttt` framework on branch `testing`. Build the `OpenSpielBot` adapter, add unit tests, and verify by running an automated test tournament between the neural agent and OpenSpiel.

Working directory: /home/entropy/Code/Super-Tic-Tac-Toe
Integrity mode: development

## Requirements

### R1. OpenSpiel Engine Adapter (`sttt/bots.py`)
- Install / verify `open_spiel` in the Conda environment (`/home/entropy/miniconda3/envs/sttt`).
- Implement `OpenSpielBot` in `sttt/bots.py` conforming to the uniform `Bot` interface (`name`, `choose`, `advance`, `reset`, `close`).
- Implement bidirectional state/action translation between `sttt.State` and OpenSpiel's `ultimate_tic_tac_toe` environment.
- Support configurable OpenSpiel bot algorithms (e.g. `mcts`, `random`) with simulation budgets.
- Register `openspiel-mcts` in the `create_bot` polymorphic factory.

### R2. Verification & Test Tournament Execution
- Add unit tests in `tests/test_openspiel.py` verifying state mapping, legal action consistency, move generation, and reset/advance semantics.
- Execute an automated test tournament (minimum 20 paired games) between `OpenSpielBot` and the neural agent / tactical bot using `python -m sttt.ai tournament`.
- Verify that tournament ratings (Bayesian Elo and Glicko-2) and summary reports export cleanly.

## Acceptance Criteria

### Functional & Behavioral Verification
- [ ] `OpenSpielBot` loads `pyspiel.load_game("ultimate_tic_tac_toe")` and correctly chooses legal moves in any legal board state.
- [ ] Action index translation between `sttt` (0..80) and `open_spiel` is verified exact with zero illegal moves across full games.
- [ ] `create_bot("openspiel-mcts")` successfully instantiates the bot.

### Tournament Verification
- [ ] An automated 20-game paired tournament runs to completion without errors or hangs.
- [ ] Matchup produces valid Elo/Glicko-2 ratings and saves output to `runs/tournaments/`.

### Non-Regression
- [ ] All existing 206 unit tests on branch `testing` continue to pass without regression.
- [ ] Active background training session in tmux remains undisturbed.

## Follow-up — 2026-09-12T10:55:02Z

Build an ultra-fast C++ bitboard game engine and rules subsystem with Python bindings on branch `cpp`, delivering orders-of-magnitude faster game rollouts, legal move generation, and MCTS simulation throughput than the Python baseline.

Working directory: /home/entropy/Code/Super-Tic-Tac-Toe
Integrity mode: development

## Requirements

### R1. Branch Isolation & Build Infrastructure (`cpp` branch)
- Check out a dedicated `cpp` git branch branched cleanly from current codebase.
- Ensure the active background training session in tmux `game:0` is completely untouched and uninterrupted.
- Set up a robust CMake / `pybind11` (or `setuptools`) build pipeline configured for the Conda environment (`/home/entropy/miniconda3/envs/sttt`).

### R2. High-Performance C++ Bitboard Game Engine
- Represent Super Tic-Tac-Toe board state using 64-bit/16-bit bitboards for maximum cache locality and SIMD/bitwise efficiency.
- Implement O(1) win checking via bitwise masks across subgrids and global grid.
- Implement ultra-fast legal move mask computation and state transitions (`apply_move`, `undo_move` or lightweight copy-on-write).
- Support random rollout playouts entirely in C++ with minimal branching and zero heap allocations.

### R3. Python Bindings & Drop-In Compatibility
- Expose the C++ engine to Python (via `pybind11`) as a high-performance drop-in replacement or accelerator for `sttt.env.State`.
- Provide tensor/NumPy export functions for board representations compatible with the neural network's spatial input format.

### R4. Differential Parity Verification
- Implement a comprehensive differential test suite pitting C++ bitboard state transitions against Python `sttt.State` across 10,000+ random and edge-case game positions.
- Verify 100% exact parity for: legal moves, active board constraints, local wins, global wins, and draw detection.

### R5. Throughput Benchmarks & Profiling
- Measure raw moves/second, random rollouts/second, and MCTS simulation rate comparing Python vs C++ bitboard engine.
- Generate structured benchmark reports and comparison tables demonstrating the achieved speedup factor (targeting >= 50x speedup).

## Acceptance Criteria

### Correctness & Integrity
- [ ] Dedicated git branch `cpp` created without disturbing active tmux training in `game:0`.
- [ ] Differential test suite verifies 100% identical state transitions and terminal outcomes across >= 10,000 paired moves vs Python `sttt.State`.
- [ ] All 251 existing Python unit tests continue to pass.

### Performance & Compilation
- [ ] C++ extension builds cleanly in `/home/entropy/miniconda3/envs/sttt` with zero compiler warnings/errors.
- [ ] Benchmark script measures >= 50x throughput speedup on game rollouts vs pure Python.
- [ ] Memory footprint per state is minimal (< 64 bytes per state in C++).

## Follow-up — 2026-09-12T11:34:14Z

Resume active execution following server restart.

Current state:
1. Branch `cpp` contains the clean compilation of `sttt_cpp` with `-Werror`, 46-byte packed `BoardState`, 100% 251/251 passing tests, and verified benchmarks (136x single-thread, 914x 10-thread rollout speedups) in commit `be29e46`.
2. Resume the verification audit and proceed with Phase 2: Implement the C++ MCTS search engine and node arena to accelerate neural self-play simulations.
