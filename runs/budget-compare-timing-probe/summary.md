# Matched-opening search budget comparison

Checkpoint: `/home/mt/Zami/Super-Tic-Tac-Toe/runs/budget-compare-timing-probe/evaluation-inputs/candidate/iter4530.pt`
Opponent: `utttai:128`
Opening pairs: 1 (each played both colours; 6 games total)
Openings file: `runs/budget-compare-timing-probe/openings.json`
Native build: `20260918084249_63a72ea` (source revision 63a72ea)

Timings are measured as observed, not controlled for other processes sharing the machine, and this is not an equal-time comparison across budgets.

## Per-arm results

| Budget | Entrant | Score | 95% CI | W/D/L | Mean s/move (candidate) |
| --- | --- | --- | --- | --- | --- |
| 128 | ckpt-iter4530-cb04e705-s128 | 50.0% | 50.0-50.0% | 0/2/0 | 0.063 |
| 512 | ckpt-iter4530-cb04e705-s512 | 25.0% | 25.0-25.0% | 0/1/1 | 0.210 |
| 2000 | ckpt-iter4530-cb04e705-s2000 | 75.0% | 75.0-75.0% | 1/1/0 | 0.749 |

## Paired score differences (bootstrapped over opening pairs)

| Comparison | Difference | 95% CI | Pairs | Move disagreement |
| --- | --- | --- | --- | --- |
| 512-128 | -25.0 pts | -25.0 to -25.0 | 1 | 40.0% |
| 2000-128 | +25.0 pts | +25.0 to +25.0 | 1 | 40.0% |
| 2000-512 (primary) | +50.0 pts | +50.0 to +50.0 | 1 | 15.4% |
