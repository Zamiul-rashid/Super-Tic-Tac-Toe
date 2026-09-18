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

Opt-in game records and loss-review reanalysis are forwarded the same way, for
example `-- --save-game-records runs/continuation/records --reanalyse`. Review
saved records offline with `python -m sttt.ai reanalyse`. See the
[training guide](../docs/training.md#game-records-and-loss-review-opt-in).

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

### Matched-opening search-budget comparison

`--mode budget-compare` plays ONE checkpoint at several search budgets against
ONE fixed opponent, on a pre-generated opening corpus SAVED to disk (not just
seeded) and shared identically by every budget. Each opening pair is played
with both colour assignments, and the budget play order is rotated per pair
(pair `k` starts with `budgets[k % len(budgets)]`) so no arm's games are
confined to one block of wall-clock time -- a real concern when a separate
training run is competing for the same CPU cores. Full move sequences and
per-move latency are recorded by default (`--save-moves`); disable with
`--no-save-moves`. Requires `--opponent`.

```bash
# 50 opening pairs, both colours: 300 games total, 100 per budget.
scripts/evaluate.sh --mode budget-compare \
  --checkpoint runs/current/latest.pt --output runs/budget-compare-current \
  --budgets "128 512 2000" --opponent utttai:128 --pairs 50 \
  --backend cpp --device cpu --leaf-batch 16 --seed 4242
```

Artifacts in the output directory:

* `openings.json` -- the saved opening corpus (`--openings-file` to reuse a
  specific path; reusing an existing file with a matching pair count and
  opening-ply count skips regeneration and plays the identical openings again).
* `budget_<N>_games.csv` -- per-game rows for each budget, same schema as the
  sweep.
* `games-budget-compare.jsonl` -- every recorded game's full action sequence,
  in the format `python -m sttt.ai reanalyse --records` reads (`actions`,
  `id`, `result`, `opening_moves`, `movers`; the candidate's own plies are
  labelled `learner`).
* `games.partial.jsonl` / `progress.json` -- live per-game journal, same
  durability contract as a championship. `progress.json["status"]` is
  `"running"`, `"completed"`, or `"failed"` -- it is updated on a failure, not
  left claiming `"running"` after the process has already stopped.
* `manifest.json` -- provenance: checkpoint hash/iteration, resolved config,
  native build identity, environment.
* `summary.json` / `summary.md` -- per-arm score and mean per-move latency,
  plus paired score differences bootstrapped over opening pairs (the two
  largest budgets are marked `"primary": true`) and, when moves were saved,
  the fraction of commonly reached positions where two budgets chose a
  different move.

Timings recorded while another process (e.g. training) shares the machine are
measured, not controlled; treat them as approximate.

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

## Search diagnostics

`search_diagnostics.py` records root policy, visits, Q, model value, simulation
and evaluation counts, and timing for one checkpoint on fixed tactical and
opening positions, for both search backends:

```bash
python scripts/search_diagnostics.py --checkpoint runs/current/latest.pt \
  --output runs/current/search-diagnostics.json --budgets 128 512 2000 --leaf-batch 1 16
```
