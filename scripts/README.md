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
```

The Python files here are reusable harnesses behind these entry points or
readiness/benchmark tools. Their filenames describe a capability, not a run.
Training always requires a new run directory. Evaluation also refuses a
non-empty output directory unless `--force` is supplied explicitly.
