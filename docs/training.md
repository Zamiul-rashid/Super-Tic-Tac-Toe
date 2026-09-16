# Training and evaluation

## Production launcher

Use one launcher for every online continuation. Only the checkpoint, output path,
and any settings that differ from the defaults need to change:

```bash
scripts/train.sh \
  --checkpoint runs/source/latest.pt \
  --output runs/continuation \
  --iterations 1000 \
  --lr-horizon 5000
```

The launcher:

- freezes an immutable copy under `OUTPUT/start-checkpoint/`;
- refuses an output directory that already contains `latest.pt`;
- defaults to CUDA, FP16, the C++ backend, eight workers, and 512 simulations;
- continues optimizer, replay, RNG, scaler, and learning-rate schedule state;
- writes the effective command and appends output to `OUTPUT/train.log`.

Use `scripts/train.sh --help` for every configurable setting. The old positional
form remains valid:

```bash
scripts/train.sh runs/source/latest.pt runs/continuation
```

For a long-running terminal session:

```bash
tmux new-session -s train
scripts/train.sh --checkpoint runs/source/latest.pt --output runs/continuation
```

Only one process may write to a run directory. To resume again, use its
`latest.pt` as the source and select a new output directory.

## Bootstrap flow

Generate native search data, pretrain a network, then use the same online
launcher:

```bash
python -m sttt.ai generate-dataset \
  --output data/bootstrap --games 50000 --workers 8

python -m sttt.ai pretrain \
  --dataset data/bootstrap --output runs/bootstrap \
  --arch unet --epochs 10 --batch 1024 --device cuda --fp16

scripts/train.sh \
  --checkpoint runs/bootstrap/latest.pt \
  --output runs/main
```

Pretraining defaults to a guarded warm-up schedule. The historical failure
analysis is preserved in [history/value-head-check.txt](history/value-head-check.txt).

## Population curriculum

`configs/population/baseline.json` is the source of truth. Checkpoints and metrics
record its name and SHA-256. If this table differs from the JSON file, the JSON
file wins.

| Family | Games per 100 | Budget |
| --- | ---: | --- |
| Self-play | 30 | Learner simulation budget |
| uttt.ai | 25 | 64/128/256 simulations; bounded move noise |
| AlphaBeta | 25 | Depth 4–8; 100k/250k/500k nodes |
| Historical checkpoints | 10 | 128/256/512 simulations |
| Best checkpoint | 5 | 128/256/512 simulations |
| Tactical variants | 2 | Depth 2; 6k/10k/16k nodes |
| OpenSpiel MCTS | 1 | 128/256/512/1024 simulations |
| Threat/blocking | 1 | Depth 2; 6k/12k/24k nodes |
| Behavioral styles | 1 | Randomized policy and move noise |

Quotas span iteration boundaries. The cursor is saved as `population_games` and
restored on resume. Changing the population configuration starts a recorded new
phase. Metrics distinguish requested opponents from the opponents actually used,
including history slots that fall back to self-play when no snapshot is present.

Training records learner targets in opponent games and both sides in self-play,
so game quotas are not replay-position quotas. `best.pt` is manually selected;
periodic AlphaBeta evaluation does not promote it automatically.

## Evaluation launcher

Use one entry point for championships, d10 sweeps, and checkpoint comparisons:

```bash
# Championship plus search-budget sweep
scripts/evaluate.sh \
  --checkpoint runs/main/latest.pt \
  --output runs/main/evaluation

# AlphaBeta d10 only
scripts/evaluate.sh \
  --mode sweep \
  --checkpoint runs/main/latest.pt \
  --output runs/main/d10 \
  --budgets "512" \
  --opponent-depth 10 \
  --opponent-nodes 50000000

# Candidate versus reference
scripts/evaluate.sh \
  --mode compare \
  --checkpoint runs/candidate/latest.pt \
  --reference runs/reference/best.pt \
  --output runs/comparisons/candidate-vs-reference
```

The harness freezes all checkpoint inputs, uses mirrored openings, stores
per-game rows, and records manifests with hashes and exact settings. Compare
runs only when their opponents, search budgets, seeds, and opening corpus match.
Reported Elo and Glicko ratings are local to that tournament pool.

## Readiness and diagnostics

Before a production run, use the staged readiness harness:

```bash
python scripts/check_training_ready.py --backend cpp --device cpu --stage cpu
python scripts/check_training_ready.py --backend cpp --device cuda --stage gpu
```

Pipeline performance measurements live behind
`scripts/benchmark_pipeline.py`. Historical milestone evidence is in
[history/training-readiness.md](history/training-readiness.md).
