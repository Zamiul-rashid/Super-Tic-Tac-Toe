# Full suite: iteration 2010

Frozen from `runs/big_run/latest.pt`. All benchmarks used the same `checkpoint.pt`,
512 simulations, CPU, seed 42, and 20 games per opponent/pairing.
The new population-training options were not active in the ongoing training run.

Automated tests: 210 run, 209 passed, 1 skipped (CUDA allocation check unavailable
in the test shell), zero failures. All ten jobs exited successfully.

| Opponent | Direct evaluation W/D/L | Score | Tournament W/D/L | Score |
| --- | --- | --- | --- | --- |
| Random | 19/1/0 | 97.5% | 18/2/0 | 95% |
| Center | 16/4/0 | 90% | 20/0/0 | 100% |
| Corners | 18/0/2 | 90% | 20/0/0 | 100% |
| Local-win | 17/3/0 | 92.5% | 15/4/1 | 85% |
| Global-win | 15/5/0 | 87.5% | 17/2/1 | 90% |
| Legacy-tactical | 15/5/0 | 87.5% | — | — |
| Tactical | 8/2/10 | 45% | 6/6/8 | 45% |
| AlphaBeta | 4/4/12 | 30% | 4/2/14 | 25% |

Score counts a draw as half a win. Direct evaluation and tournament use different
opening/RNG scheduling, so their games are not identical despite the same seed.
Legacy-tactical currently aliases the global-win policy in the evaluator.

Tournament: 560 total games, 8 players. Agent ranked third behind AlphaBeta and
Tactical, with 100 wins, 16 draws, 24 losses in its 140 games (77.1% score).
The aggregate score includes five simpler bots and should not obscure the deficit
against AlphaBeta/Tactical. These 20-game matchups do not establish a training trend.

See `manifest.json` for checkpoint SHA-256, exact commands and exit codes;
`tests.log` for automated tests; opponent directories for game records; and
`tournament/scoreboard.txt` and `tournament/tournament.json` for tournament details.
