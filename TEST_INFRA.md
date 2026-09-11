# Super-Tic-Tac-Toe: Test Infrastructure Specification

**Document Version**: 1.0.0  
**Author**: E2E Test Writer  
**Target Branch**: `testing`  
**Test Harness**: Python `unittest` (`/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest`)  
**Scope**: Full Test Suite Architecture, Feature Inventory Mapping, and Quality Assurance Gateways

---

## 1. Test Philosophy & Principles

The testing framework for the Super-Tic-Tac-Toe project adheres to a strict, production-grade test philosophy engineered for deterministic verification, zero regression, and absolute isolation:

1. **Opaque-Box Verification (Black-Box / Contract-Driven)**:
   - Tests evaluate purely observable public behavior: interface compliance, method return values, standard input/output streams, exit codes, and structured file artifacts (JSON, CSV, TXT).
   - Tests have zero dependence on private or volatile internal implementation details (`_private_methods`, internal node layouts, or ephemeral variables).
   - Expected outputs are derived directly from authoritative mathematical definitions (Davidson Elo, Glickman Glicko-2), game rules (`LINES`, `winner`), or documented interface contracts in `PROJECT.md` and `ORIGINAL_REQUEST.md`.

2. **Zero External Test Dependencies**:
   - The test infrastructure operates exclusively using Python 3.14 standard library (`unittest`, `subprocess`, `tempfile`, `json`, `csv`, `math`, `shutil`) and core project dependencies (`numpy==2.5.3`, `torch==2.14.0`).
   - Does not require `pytest`, `scipy`, or `matplotlib`. When optional libraries like `matplotlib` are absent, headless fallback mechanisms (ASCII scoreboards, pure text reports) are asserted to function gracefully without crashing.

3. **Progressive Testability & Graceful Feature Detection**:
   - Milestone development follows a decoupled parallel architecture (M1: Adapters, M2: Tournament, M3: Adaptation, M4: CLI, M5: E2E Hardening).
   - Tests are written against the interface contracts defined in `PROJECT.md`. For components under active construction, tests employ dynamic capability detection (`unittest.skipUnless`) so that existing baseline tests continue passing with zero failures at every milestone commit, while automatically engaging full assertions once the respective milestone components are wired.

4. **Self-Contained & Isolated Execution**:
   - Every test sets up its own isolated environment (e.g. `tempfile.TemporaryDirectory()`), provides explicit random seeds (`np.random.default_rng(seed)`), and cleans up all subprocesses and disk artifacts upon completion.
   - Tests never depend on execution order or shared mutable global state.

5. **Strict Environment & Training Isolation**:
   - A live training process runs concurrently in tmux session `game` (`runs/big_run/latest.pt`).
   - Tests execute exclusively on CPU (`device='cpu'`), enforce thread caps (`torch.set_num_threads(1)`), and operate strictly in read-only mode regarding active run directories.
   - Subprocess bots spawned by tests are strictly bounded by non-blocking timeouts ($\le 5.0\text{s}$) with automatic SIGTERM/SIGKILL process teardown.

---

## 2. Feature Inventory Mapping (F1–F20)

Every feature defined in `PROJECT.md` is mapped to its primary specification source, implementation target, milestone, and assigned test tiers:

| Feature ID | Feature Name | Specification Source | Target Module | Milestone | Test Tiers | Primary Test File & Test Class |
|---|---|---|---|---|---|---|
| **F1** | Uniform Bot Interface | R1, AC1 | `sttt/bots.py` | M1 | Tier 1, Tier 2 | `test_e2e_tournament.py::TestTier1BotInterface` |
| **F2** | ExternalProcessBot | R1, AC2 | `sttt/bots.py` | M1 | Tier 1, Tier 2, Tier 3 | `test_e2e_tournament.py::TestTier1ExternalProcessBot`, `TestTier2ExternalProcessEdgeCases` |
| **F3** | StyleBot | R1 | `sttt/bots.py` | M1 | Tier 1, Tier 2 | `test_e2e_tournament.py::TestTier1StyleBot`, `TestTier2StyleBotCornerCases` |
| **F4** | Heuristic Search Bots | R1 | `sttt/bots.py` | M1 | Tier 1, Tier 3 | `test_e2e_tournament.py::TestTier1HeuristicBots`, `TestTier3CrossFeatureCombinations` |
| **F5** | CheckpointBot | R1 | `sttt/bots.py` | M1 | Tier 1, Tier 3 | `test_e2e_tournament.py::TestTier1CheckpointBot`, `TestTier3CrossFeatureCombinations` |
| **F6** | Mock Bot Unit Tests | AC3 | `tests/test_bots.py` | M1 | Tier 1, Tier 2 | `tests/test_bots.py`, `test_e2e_tournament.py::TestTier1ExternalProcessBot` |
| **F7** | Paired Seat Alternation | R2, AC4 | `sttt/tournament.py` | M2 | Tier 1, Tier 2, Tier 3 | `test_e2e_tournament.py::TestTier1TournamentPairing`, `TestTier2PairingBoundaries` |
| **F8** | Configurable Sample Sizes | R2 | `sttt/tournament.py` | M2 | Tier 1, Tier 2 | `test_e2e_tournament.py::TestTier1TournamentPairing`, `TestTier2PairingBoundaries` |
| **F9** | Simulation Budget Sweeps | R2, AC6 | `sttt/tournament.py` | M2 | Tier 1, Tier 3, Tier 4 | `test_e2e_tournament.py::TestTier1BudgetSweeps`, `TestTier3CrossFeatureCombinations`, `TestTier4RealWorldScenarios` |
| **F10** | Bayesian Elo Calculation | R2, AC5 | `sttt/tournament.py` | M2 | Tier 1, Tier 2, Tier 3 | `test_e2e_tournament.py::TestTier1BayesianElo`, `TestTier2RatingBoundaries` |
| **F11** | Glicko-2 Rating Engine | R2, AC5 | `sttt/tournament.py` | M2 | Tier 1, Tier 2, Tier 3 | `test_e2e_tournament.py::TestTier1Glicko2`, `TestTier2RatingBoundaries` |
| **F12** | Tournament Engine Tests | AC4, AC5 | `tests/test_tournament.py` | M2 | Tier 1, Tier 2, Tier 3 | `tests/test_tournament.py`, `test_e2e_tournament.py` |
| **F13** | Adaptation Belief Convergence | R3 | `sttt/opponent.py` | M3 | Tier 1, Tier 2 | `test_e2e_tournament.py::TestTier1AdaptationMetrics` |
| **F14** | Adaptation Style Switching | R3 | `sttt/opponent.py` | M3 | Tier 1, Tier 2 | `test_e2e_tournament.py::TestTier1AdaptationMetrics`, `TestTier2AdaptationCornerCases` |
| **F15** | Adaptation Win-Rate Delta | R3 | `tests/test_adaptation.py` | M3 | Tier 1, Tier 3 | `tests/test_adaptation.py`, `test_e2e_tournament.py::TestTier3CrossFeatureCombinations` |
| **F16** | Tournament CLI Subcommand | R4 | `sttt/ai.py` | M4 | Tier 1, Tier 2, Tier 4 | `test_e2e_tournament.py::TestTier1CLIOptions`, `TestTier2CLIBoundaries`, `TestTier4RealWorldScenarios` |
| **F17** | Scoreboards & Visualizations | R4 | `sttt/reports.py` | M4 | Tier 1, Tier 4 | `test_e2e_tournament.py::TestTier1Visualizations`, `TestTier4RealWorldScenarios` |
| **F18** | Tournament Report Exports | R4 | `sttt/reports.py` | M4 | Tier 1, Tier 4 | `test_e2e_tournament.py::TestTier1ReportExports`, `TestTier4RealWorldScenarios` |
| **F19** | Codebase Non-Regression | AC7 | `tests/` | M5 | Baseline | `test_ai.py`, `test_reports.py`, `test_search.py`, `test_selfplay.py` (26 tests) |
| **F20** | Git & Training Isolation | AC8 | Git / tmux | M5 | Baseline | Environment audits, branch status checks, CPU-only verification |

---

## 3. Test Architecture & 4-Tier Hierarchy

The E2E test suite in `tests/test_e2e_tournament.py` is organized into a 4-tier testing hierarchy designed to guarantee functional completeness and adversarial robustness:

```
+-------------------------------------------------------------------------+
|                  TIER 4: REAL-WORLD APPLICATION SCENARIOS               |
|  - Full CLI end-to-end execution with artifact export validation        |
|  - Multi-budget simulation sweep workflows                              |
|  - Schema conformance: tournament.json, summary.csv, scoreboard.txt     |
+-------------------------------------------------------------------------+
|                TIER 3: CROSS-FEATURE INTEGRATION COMBINATIONS           |
|  - ExternalProcessBot vs AlphaBetaBot paired match                      |
|  - Real paired tournament -> Bayesian Elo + Glicko-2 ranking pipeline    |
|  - Multi-budget simulation sweep scaling verification                   |
|  - Adaptive search vs StyleBot paired competition                       |
+-------------------------------------------------------------------------+
|               TIER 2: BOUNDARY CONDITIONS & ADVERSARIAL CORNERS         |
|  - Process timeouts (<=5.0s hard cutoff), hung subprocess kills         |
|  - Subprocess mid-game crashes, EOF, corrupted stdout tokens            |
|  - Opening plies boundaries (0 plies, 4 plies, random, invalid plies)   |
|  - Sample sizes (min 20, max 500, odd count auto-rounding)              |
|  - Mathematical rating extremes (100% win disparity, 100% draws, CI)    |
+-------------------------------------------------------------------------+
|                 TIER 1: FEATURE COVERAGE (HAPPY-PATH)                   |
|  - >=5 tests per core feature: F1, F2, F3, F7, F10, F11, F13, F16, F18   |
|  - Uniform bot protocol (choose, advance, reset, close, name)           |
|  - Paired seat alternation color bias cancellation symmetry             |
|  - Bradley-Terry-Davidson Bayesian Elo MAP convergence                  |
|  - Glicko-2 canonical benchmark vector (Glickman 2013)                  |
+-------------------------------------------------------------------------+
|             BASELINE NON-REGRESSION: 26 EXISTING UNIT TESTS             |
|  - test_ai.py (8), test_reports.py (4), test_search.py (12), selfplay (2)|
+-------------------------------------------------------------------------+
```

### 3.1 Tier Breakdown & Test Counts

1. **Tier 1: Feature Coverage (>=5 tests per feature)**:
   - Covers primary happy-path behavior across all core subsystems.
   - *Target Count*: $\ge 40$ test cases.
   - *Key Areas*: Bot interface adherence, external engine I/O, style distributions, pairing symmetry, Bayesian Elo MAP, Glicko-2 canonical vectors, CLI argument validation, belief convergence.

2. **Tier 2: Boundary & Corner Cases (>=5 tests per category)**:
   - Tests boundary constraints, failure handling, and adversarial inputs.
   - *Target Count*: $\ge 25$ test cases.
   - *Key Areas*: Subprocess 5s timeout expiration, crash/SIGKILL recovery, corrupt non-integer engine output, 0 opening plies, 4 opening plies, sample size limits ($N=20$, $N=500$, odd numbers), extreme rating disparities ($100\%$ wins, $0\%$ wins, $100\%$ draws).

3. **Tier 3: Cross-Feature Combinations**:
   - Validates pairwise interactions across multiple subsystems in realistic combinations.
   - *Target Count*: $\ge 5$ test cases.
   - *Key Areas*: External bot vs internal bot paired tournament; tournament results feeding Bayesian Elo and Glicko-2; multi-budget simulation sweep; adaptive bot vs behavioral style bot.

4. **Tier 4: Real-World Application Scenarios**:
   - Executes full end-to-end user workflows from command-line invocation to filesystem inspection.
   - *Target Count*: $\ge 5$ test cases.
   - *Key Areas*: Complete tournament execution via CLI into `runs/<run>/tournaments/`; verification of JSON report schema; verification of CSV summary and game logs; ASCII scoreboard generation.

---

## 4. Quality Assurance & Coverage Thresholds

| Metric | Target Threshold | Verification Method |
|---|---|---|
| **Feature Coverage** | 100% of F1–F20 mapped | Section 2 Feature Inventory |
| **Tier 1 Test Density** | $\ge 5$ tests per feature | `tests/test_e2e_tournament.py` |
| **Tier 2 Test Density** | $\ge 5$ tests per boundary class | `tests/test_e2e_tournament.py` |
| **Baseline Non-Regression** | 26 passed, 0 failed, 0 errors | `.venv/bin/python -m unittest discover tests` |
| **Total Test Count** | $\ge 95$ total tests | Baseline (26) + E2E suite ($\ge 70$) |
| **Pass Rate** | 100% (0 failures, 0 unexpected errors) | `unittest` exit code 0 |
| **Execution Performance** | Full test suite execution $\le 30.0\text{s}$ | `time python -m unittest` |
| **Subprocess Timeout Cap** | Hard ceiling at $\le 5.0\text{s}$ per action | Mock sleep tests verify timeout enforcement |
| **Device Sandboxing** | CPU only (`device='cpu'`) | No CUDA allocations during tests |

---

## 5. Test Execution Instructions

### 5.1 Run Full Test Suite
```bash
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest discover -v tests
```

### 5.2 Run E2E Tournament Suite Only
```bash
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest -v tests/test_e2e_tournament.py
```

### 5.3 Run by Specific Tier
```bash
# Run Tier 1 Feature Coverage
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest -v tests.test_e2e_tournament.TestTier1BotInterface
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest -v tests.test_e2e_tournament.TestTier1ExternalProcessBot
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest -v tests.test_e2e_tournament.TestTier1TournamentPairing
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest -v tests.test_e2e_tournament.TestTier1BayesianElo
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest -v tests.test_e2e_tournament.TestTier1Glicko2

# Run Tier 2 Boundary & Corner Cases
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest -v tests.test_e2e_tournament.TestTier2ExternalProcessEdgeCases
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest -v tests.test_e2e_tournament.TestTier2PairingBoundaries
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest -v tests.test_e2e_tournament.TestTier2RatingBoundaries

# Run Tier 3 Cross-Feature Combinations
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest -v tests.test_e2e_tournament.TestTier3CrossFeatureCombinations

# Run Tier 4 Real-World Application Scenarios
/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -m unittest -v tests.test_e2e_tournament.TestTier4RealWorldScenarios
```
