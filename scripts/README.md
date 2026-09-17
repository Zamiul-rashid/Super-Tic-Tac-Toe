# Scripts

The repository has two stable operational entry points:

- `train.sh` resumes any full checkpoint into a new run directory.
- `evaluate.sh` runs a championship, AlphaBeta budget sweep, or checkpoint comparison.

Both accept paths and settings as command-line arguments. Do not create a new
launcher for each checkpoint or output directory.

## Train

```bash
scripts/train.sh \
  --checkpoint runs/source/latest.pt \
  --output runs/continuation \
  --iterations 1000 \
  --lr-horizon 5000
```

The concise positional form remains available:

```bash
scripts/train.sh runs/source/latest.pt runs/continuation
```

Use `--dry-run` to validate and print the resolved command. Arguments after `--`
are forwarded to `sttt.ai train` for uncommon settings.

## Evaluate

```bash
# Full round-robin plus the default depth-10 AlphaBeta sweep
scripts/evaluate.sh \
  --checkpoint runs/continuation/latest.pt \
  --output runs/continuation/evaluation

# Only the d10 sweep at one neural search budget
scripts/evaluate.sh \
  --mode sweep \
  --checkpoint runs/continuation/latest.pt \
  --output runs/continuation/d10 \
  --budgets "512" \
  --opponent-depth 10 \
  --opponent-nodes 50000000

# Controlled comparison of two checkpoints
scripts/evaluate.sh \
  --mode compare \
  --checkpoint runs/candidate/latest.pt \
  --reference runs/reference/best.pt \
  --output runs/comparisons/candidate-vs-reference

# Eight entrants, 28 pairings × 72 games = 2,016 games (including AlphaBeta d10)
scripts/evaluate.sh --mode championship \
  --checkpoint runs/current/latest.pt --output runs/championship-large \
  --championship-games 72 --seed 17092026 --backend cpp --device cpu \
  --championship-opponents "cpp-alphabeta:10:50000000 cpp-alphabeta:6:50000000 cpp-alphabeta:4:50000000 tactical threat-block openspiel-mcts:100 utttai:128"
```

The Python files here are reusable harnesses behind these entry points or
readiness/benchmark tools. Their filenames describe a capability, not a run.
Training always requires a new run directory. Evaluation also refuses a
non-empty output directory unless `--force` is supplied explicitly.

Championships journal completed games to `championship/games.partial.jsonl` and
publish `championship/progress.json` after each game. Final results still use
`championship_games.csv`, `tournament.json`, and the manifest. A partial journal
is evidence of completed games, not automatic resume support or a completed
championship. Manifests include explicit opponent specs, resolved settings,
package versions, and hashes of the source files used by the harness.

## Research report figures

`report_figures.py` renders measured charts and explanatory flowcharts from a
frozen evidence snapshot, without running training or evaluation:

```bash
python scripts/report_figures.py \
  --snapshot docs/report/evidence/snapshot.json \
  --output docs/report
```

To collect evidence from other run paths, edit a copy of
`docs/report/report-config.json` and pass it with `--config` instead of
`--snapshot`. Use the same renderer for different runs rather than creating a
new report script. See [the evidence guide](../docs/report/EVIDENCE.md).
