# Population training plan and implementation

1. Keep current training running while preparing and testing the changes.
2. Add opt-in per-game opponent sampling: 60% current self-play, 15% frozen
   history, 10% AlphaBeta, 10% Tactical, 5% behavioral styles. These are sampling
   probabilities, not exact quotas in every small batch. If no history exists,
   its allocation becomes self-play.
3. Randomize opponent seat, 0–4 opening plies, AlphaBeta depth 1–4, search node
   budgets 500/1500/3000/6000, historical search budgets 64/128/256, and opponent
   random-move probability 0–12%. The opponent family stays fixed within a game.
4. Store only learner search targets in bot/history games; retain both sides in
   self-play. Record every sampled matchup in metrics. Save the NumPy generator
   state so resumes do not restart the same sampling sequence.
5. Optionally augment minibatches with all eight board rotations/reflections,
   transforming both board levels, forced board, move policy and legal mask.
6. Verify mixed worker inference, legality, history loading, perspective and
   symmetry consistency, then smoke-test training and resume on separate files.

## Prepared settings

For the next training launch, use the command below. Run it only after the
existing trainer has finished or after explicitly arranging a checkpoint-boundary
handoff. It does not hot-reconfigure an existing process. Do not launch another
writer to `runs/big_run` while its current trainer is running.

```bash
/home/entropy/miniconda3/envs/sttt/bin/python -u -m sttt.ai train \
  --resume runs/big_run/latest.pt --output runs/big_run \
  --device cuda --workers 8 --iterations 5000 --games 16 \
  --simulations 512 --batch 256 --buffer 150000 --steps 100 \
  --leaf-batch 16 --inference-batch 128 --inference-wait-ms 2 \
  --save-every 100 --eval-every 250 --eval-games 20 --eval-simulations 512 \
  --seed 42 --population --augment-symmetry
```

The 150k buffer retains more recent experience; it does not guarantee protection
against forgetting or allocate separate quotas to opponent families. Older
checkpoints remain CPU opponents in workers; the current learner continues to
use the shared inference owner. This may reduce throughput versus pure self-play.
Original training defaults remain unchanged unless the new flags are supplied.

## Generalization and simulation budget

Randomization and augmentation reduce repetition; neither guarantees that the
agent cannot overfit the training families. Use evaluation-only seeds and
openings, frozen checkpoints excluded from the training history directory, and
opponents outside this family to test generalization. Use
`--population-checkpoints DIRECTORY` to select a curated training-only history.
Existing periodic AlphaBeta evaluation remains a small diagnostic; it does not
select or promote a best checkpoint. The latest checkpoint is not necessarily best.

Keep 512 simulations for the initial comparison. A move to 256 could increase
game throughput but also changes search targets. Compare separate 256 and 512
continuations from the same frozen checkpoint under equal wall time, then evaluate
both at the same 512-simulation budget across multiple seeds. Existing results
do not establish that either training budget is superior. Increasing the buffer
to 150k is a proposed retention change, not a proven strength improvement.

The previous explanations attributing weak performance to pure self-play,
replay turnover or the old simulation-budget change were hypotheses. These
changes address plausible limitations but their strength benefit needs measurement.
