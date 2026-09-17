# Repository Guide for Agents

This file applies to the entire repository. Follow direct user instructions first, then this guide. Keep changes focused, preserve unrelated work, and leave the repository easier to operate than you found it.

## Repository Map

| Path | Purpose |
| --- | --- |
| `sttt/` | Main Python package: game rules, agents, search, training, evaluation, reports, and native-backend integration. |
| `sttt/env.py` | Authoritative Python game state and rules implementation. |
| `sttt/ai.py` | Main command-line entry points for training and evaluation. |
| `sttt/search.py` | Python search and MCTS implementation. |
| `sttt/backends.py` | Backend selection and shared backend interface. |
| `sttt/cpp_env.py` | Python bindings and adapters for the native engine. |
| `sttt/learning.py`, `sttt/unet.py` | Model, training, and loss-related code. |
| `sttt/selfplay.py` | Self-play generation and worker-pool orchestration. |
| `sttt/population.py`, `sttt/opponent.py` | Population definitions and opponent selection. |
| `sttt/evaluation.py`, `sttt/tournament.py`, `sttt/reports.py` | Match evaluation, championships, and result reporting. |
| `sttt/bootstrap.py` | Runtime/bootstrap helpers. |
| `cpp/` | Native engine source, build files, and native benchmarks/tests. |
| `configs/` | Versioned runtime configuration. `configs/population/*.json` is the source of truth for named populations. |
| `scripts/` | Reusable operational entry points for training, evaluation, readiness checks, and benchmarks. |
| `tests/` | Automated Python tests. Add regression coverage here for behavioral changes. |
| `docs/` | Current documentation, grouped by topic. Start with `docs/README.md`. |
| `docs/report/` | Research manuscript, source evidence snapshot, figures, and image-generation provenance. Rebuild measured figures with `scripts/report_figures.py`; distinguish archived results from new experiments. |
| `docs/history/` | Historical plans and records. These are context, not current operating instructions. |
| `engines/` | External engine integrations or engine assets. |
| `runs/`, `data/` | Generated runtime output and datasets. Treat these as artifacts, not source code. |

## Canonical Workflows

Use the generalized launchers instead of creating a script for each experiment. Change arguments, configuration, checkpoint paths, and output paths at invocation time.

```bash
# Install the Python package in editable mode.
python -m pip install -e .

# Build the native extension in place.
python setup.py build_ext --inplace

# Run the Python test suite.
python -m pytest tests

# Check that the repository is ready for training.
python scripts/check_training_ready.py

# Resume or start training from a selected checkpoint.
scripts/train.sh --checkpoint PATH_TO_CHECKPOINT --output PATH_TO_NEW_RUN

# Run evaluations or a championship from selected inputs.
scripts/evaluate.sh --help
```

Read `scripts/README.md`, `docs/training.md`, and `docs/testing.md` before changing operational commands. Use `python -m sttt.ai --help` when a lower-level command is needed.

## Editing Rules

- Do not add per-run, per-checkpoint, or date-stamped shell/Python scripts. Extend `scripts/train.sh`, `scripts/evaluate.sh`, or an existing generalized Python tool with an option instead.
- Keep policy and defaults in versioned configuration where practical. Do not bury reusable settings in shell history or a copied script.
- Update tests and current documentation whenever command names, paths, defaults, output formats, or public behavior change.
- Keep historical documents in `docs/history/`; do not use them as the canonical description of current behavior.
- Preserve the existing architecture and backend contract. Python and native implementations should agree on rules, legal moves, state transitions, and terminal outcomes.
- Prefer deterministic seeds for tests and reproducible experiments. Evaluation changes should preserve pairing/mirroring and provenance where those controls apply.
- External engine failures must be visible. Do not silently replace a requested engine with an easier opponent or fallback implementation.
- Avoid committing generated outputs or native build products, including run directories, datasets, compiled extensions, benchmark binaries, caches, and `build*` directories.
- Preserve unrelated modifications in a dirty worktree. Do not reformat, rename, delete, reset, or overwrite files outside the requested scope.

## Training and Evaluation Safety

- Before modifying, stopping, or restarting training, inspect active processes and the relevant tmux session. The conventional training session is named `train`.
- Never stop an active run unless the user explicitly asks. Repository cleanup and documentation work must not interrupt training.
- Never train into an existing run directory. Use a new output directory so checkpoints, logs, metrics, and configuration snapshots remain attributable.
- Treat the input checkpoint as immutable. Resuming should read from it and write all new state to the new run directory.
- Do not delete, rename, or overwrite checkpoints, championship results, or run metadata without explicit authorization and an exact target check.
- Record enough configuration and version information to reproduce important training and evaluation results.
- For championships, report the checkpoint, population/configuration, engine/backend, search settings (including AlphaBeta depth), game count, seed/opening policy, and output location.

## Native Engine Changes

- Rebuild the extension after changing `cpp/` or binding code. If the active environment matters, explicitly pass its interpreter, for example:

  ```bash
  make -C cpp PYTHON=/path/to/python
  ```

- Run native/backend parity tests as well as the directly affected tests.
- Do not commit compiled `.so` files or generated native executables.

## Verification Checklist

Before handing off a change:

1. Inspect references to moved or renamed files and check for active jobs when runtime behavior is involved.
2. Run the narrowest relevant tests, then broader tests when the risk warrants it.
3. Validate shell scripts with `bash -n` and Python scripts with compilation or import checks.
4. Run `git diff --check` and review the final diff for accidental generated files or unrelated edits.
5. State what changed, what was tested, and any remaining limitation.

When adding a new subsystem, extend this map and link its canonical documentation so future agents can locate it without reconstructing the repository history.
