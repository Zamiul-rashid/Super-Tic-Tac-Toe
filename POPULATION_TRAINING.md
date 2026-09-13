# Population league

The curriculum is versioned configuration: `configs/population/baseline.json`
is the source of truth for quotas and budgets, and every checkpoint and metrics
row records the config's name and sha256. The table below is `baseline.json`;
if it disagrees with the file, the file wins. `candidate-utttai35.json` raises
uttt.ai to 35 and lowers AlphaBeta to 15 with everything else identical.

| Family | Games per 100 | Budget (baseline.json) |
| --- | ---: | --- |
| Self-play | 30 | Learner --simulations |
| uttt.ai stage-2 ONNX + official NMCTS | 25 | 64/128/256 per move; move noise epsilon 0–0.02 |
| AlphaBeta | 25 | Depth 3–8; 100k/250k/500k nodes |
| Older checkpoints | 10 | 128/256/512 simulations |
| best.pt | 5 | 128/256/512 simulations |
| Tactical variants | 2 | Depth 2; 6k/10k/16k nodes; epsilon 0–0.12 |
| Official OpenSpiel MCTS | 1 | 128/256/512/1024 per decision |
| Threat/blocking | 1 | Depth 2; 6k/12k/24k nodes |
| Behavioral styles | 1 | Randomized policy and move noise |

Quotas are exact per 100 scheduled games and span iteration boundaries: a
16-game iteration approximates the mix. The schedule cursor is saved as
`population_games` in latest.pt and restored on resume; changing the config
(a new sha256) starts a new phase and resets the cursor, and the checkpoint
keeps the phase history. Game order, seat, openings and variants are seeded.
A family's quota is not an independent random draw. Metrics record both the
requested family (`population_requested_counts`) and the family actually
played (`population_match_counts`), so a substitution (for example a history
slot played as self-play because no snapshot exists yet) is visible.

Historical opponents come from frozen model-*.pt files. Following checkpoint
cleanup, the history quota uses best.pt until new snapshots are available.
Neither history nor best reads the rolling latest.pt. Without either a best
or historical snapshot, missing history slots become self-play and actual
counts expose this. best.pt remains a manually selected checkpoint; periodic
AlphaBeta evaluation does not automatically promote it.

Training records only learner policy targets in opponent games and both sides
in self-play. Thus game quotas are not replay-position quotas. Metrics include
each matchup and actual population_match_counts. Replay retains the existing
FIFO behavior; no strength improvement or forgetting protection is guaranteed.

## Launch

Dependencies, source and model setup are documented in [engines/README.md](engines/README.md).
The local registry is already configured. Start the prepared script with:

```bash
bash runs/big_run/train.sh
```

It resumes latest.pt using CUDA, 8 workers, 16 games per iteration, 512 learner
simulations, a 150k replay buffer, 100 training steps, and symmetry augmentation.
It requests 5,000 additional iterations. Only start one writer for a run.

Alternatively:

```bash
python -m sttt.ai train --resume runs/big_run/latest.pt --output runs/big_run \
  --device cuda --workers 8 --iterations 5000 --games 16 --simulations 512 \
  --batch 256 --buffer 150000 --steps 100 --leaf-batch 16 \
  --inference-batch 128 --inference-wait-ms 2 --save-every 100 \
  --eval-every 250 --eval-games 20 --eval-simulations 512 --seed 42 \
  --population --population-checkpoints runs/big_run \
  --engine-config engines/registry.json --augment-symmetry
```

The trainer probes the real uttt.ai wrapper and requires compiled OpenSpiel
before starting workers. OpenSpiel and uttt.ai run on CPU. Their cost can reduce
throughput substantially; the old iteration-time estimate no longer applies.

The state_json wrapper accepts complete positions, including randomized openings
and injected moves. Opponent processes close even if collection raises an error.
OpenSpiel receives every played move and handles its separate board/cell actions.

## Evaluation

Use held-out seeds and openings, and opponents/checkpoints outside the training
pool. Existing AlphaBeta evaluation is diagnostic, not a comprehensive strength
measure. Compare continuations at equal wall time and evaluate at common search
budgets before attributing improvements to the league.
