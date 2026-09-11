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
