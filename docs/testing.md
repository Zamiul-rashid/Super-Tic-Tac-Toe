# Testing

The suite uses Python `unittest` and covers the rules engine, search, native
bindings, training state, external engines, population sampling, evaluation
provenance, reports, and complete tournament workflows.

## Commands

```bash
# Full suite
python -m unittest discover -s tests -v

# Focused evaluation harnesses
python -m unittest \
  tests.test_budget_sweep_harness \
  tests.test_comparison_harness \
  tests.test_benchmark_integrity -v

# Native correctness and lifetime boundaries
python -m unittest \
  tests.test_native_validation \
  tests.test_native_lifetime \
  tests.test_native_leaks -v

# Training resume, schedules, replay, and population
python -m unittest \
  tests.test_training_state \
  tests.test_training_schedule \
  tests.test_replay_sampling \
  tests.test_population -v
```

Use the same interpreter that built the native extension. In this repository it
can also be selected explicitly:

```bash
STTT_PY=/path/to/python
"$STTT_PY" -m unittest discover -s tests -v
```

## Test layers

| Layer | Scope | Representative files |
| --- | --- | --- |
| Unit | Rules, encoding, loss, schedules, reports | `test_ai.py`, `test_encoding.py`, `test_learning_loss.py` |
| Boundary | Invalid states, process failures, timeouts, lifetime | `test_bots_adversarial.py`, `test_native_lifetime.py` |
| Integration | Search backends, population, resumable state | `test_cpp_mcts.py`, `test_population.py`, `test_training_state.py` |
| Evaluation | Paired openings, provenance, ratings, artifacts | `test_budget_sweep_harness.py`, `test_comparison_harness.py` |
| End-to-end | CLI and tournament workflows | `test_e2e_tournament.py`, `test_cli_tournament.py` |

Tests create isolated temporary directories and should not write into production
run directories. CPU-only tests must not disturb a live GPU training process.
External-process tests bound subprocess lifetimes and verify cleanup on failures.

## Readiness gates

`scripts/check_training_ready.py` provides staged environment checks beyond the
unit suite:

```bash
python scripts/check_training_ready.py --backend cpp --device cpu --stage build
python scripts/check_training_ready.py --backend cpp --device cpu --stage native
python scripts/check_training_ready.py --backend cpp --device cpu --stage cpu
python scripts/check_training_ready.py --backend cpp --device cuda --stage gpu
```

Each stage writes a manifest under a fresh `runs/readiness/` directory. GPU pilot
runs require an explicit full checkpoint and invoke `scripts/train.sh`, so they
should be scheduled like any other training workload.

The dated original certification and milestone-specific test counts are kept in
[history/test-readiness-2026-09-10.md](history/test-readiness-2026-09-10.md) and
[history/training-readiness.md](history/training-readiness.md). Current test
counts should always come from test discovery, not those historical documents.
