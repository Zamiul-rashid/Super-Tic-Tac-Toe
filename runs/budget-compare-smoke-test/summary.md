# Matched-opening search budget comparison

Checkpoint: `/home/mt/Zami/Super-Tic-Tac-Toe/runs/budget-compare-smoke-test/evaluation-inputs/candidate/iter4530.pt`
Opponent: `utttai:128`
Opening pairs: 2 (each played both colours; 12 games total)
Openings file: `runs/budget-compare-smoke-test/openings.json`
Native build: `20260918084249_63a72ea` (source revision 63a72ea)

Timings are measured as observed, not controlled for other processes sharing the machine, and this is not an equal-time comparison across budgets.

## Per-arm results

| Budget | Entrant | Score | 95% CI | W/D/L | Mean s/move (candidate) |
| --- | --- | --- | --- | --- | --- |
| 8 | ckpt-iter4530-cb04e705-s8 | 0.0% | 0.0-0.0% | 0/0/4 | 0.007 |
| 16 | ckpt-iter4530-cb04e705-s16 | 0.0% | 0.0-0.0% | 0/0/4 | 0.010 |
| 32 | ckpt-iter4530-cb04e705-s32 | 12.5% | 0.0-25.0% | 0/1/3 | 0.015 |

## Paired score differences (bootstrapped over opening pairs)

| Comparison | Difference | 95% CI | Pairs | Move disagreement |
| --- | --- | --- | --- | --- |
| 16-8 | +0.0 pts | +0.0 to +0.0 | 2 | 66.7% |
| 32-8 | +12.5 pts | +0.0 to +25.0 | 2 | 66.7% |
| 32-16 (primary) | +12.5 pts | +0.0 to +25.0 | 2 | 15.4% |
