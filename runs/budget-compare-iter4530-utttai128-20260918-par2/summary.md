# Matched-opening search budget comparison

Checkpoint: `/home/mt/Zami/Super-Tic-Tac-Toe/runs/budget-compare-iter4530-utttai128-20260918-par2/evaluation-inputs/candidate/iter4530.pt`
Opponent: `utttai:128`
Opening pairs: 50 (each played both colours; 300 games total)
Openings file: `runs/budget-compare-iter4530-utttai128-20260918-par2/openings.json`
Native build: `20260918084249_63a72ea` (source revision 63a72ea)

Timings are measured as observed, not controlled for other processes sharing the machine, and this is not an equal-time comparison across budgets.

## Per-arm results

| Budget | Entrant | Score | 95% CI | W/D/L | Mean s/move (candidate) |
| --- | --- | --- | --- | --- | --- |
| 128 | ckpt-iter4530-cb04e705-s128 | 22.0% | 16.0-28.0% | 4/36/60 | 0.110 |
| 512 | ckpt-iter4530-cb04e705-s512 | 41.0% | 34.0-48.5% | 17/48/35 | 0.353 |
| 2000 | ckpt-iter4530-cb04e705-s2000 | 56.5% | 50.0-63.0% | 31/51/18 | 1.209 |

## Paired score differences (bootstrapped over opening pairs)

| Comparison | Difference | 95% CI | Pairs | Move disagreement |
| --- | --- | --- | --- | --- |
| 512-128 | +19.0 pts | +9.0 to +29.5 | 50 | 8.6% |
| 2000-128 | +34.5 pts | +25.0 to +44.0 | 50 | 16.0% |
| 2000-512 (primary) | +15.5 pts | +4.5 to +26.5 | 50 | 8.7% |
