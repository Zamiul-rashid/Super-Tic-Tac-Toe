"""Time per move for the native AlphaBeta at several depths, on one fixed
position set drawn from real games. Node cap 50,000,000 as in the championship."""
import time, json
import numpy as np
from sttt.env import State
from sttt.cpp_env import CppAlphaBetaBot, CppState

rng = np.random.default_rng(7)
# Build a position set: play games between two d4 bots, sample every 4th ply.
a, b = CppAlphaBetaBot(depth=4, node_budget=50_000_000), CppAlphaBetaBot(depth=4, node_budget=50_000_000)
positions = []
for g in range(6):
    s = State()
    for _ in range(int(rng.integers(0, 3))):          # random opening plies
        s = s.play(int(rng.choice(s.legal_actions())))
    ply = 0
    while s.result is None:
        if ply % 4 == 0:
            positions.append(s)
        s = s.play((a if s.turn == 1 else b).choose(s, rng))
        ply += 1
print(f"{len(positions)} positions from 6 games")

rows = {}
for depth in (2, 4, 6, 8, 10):
    bot = CppAlphaBetaBot(depth=depth, node_budget=50_000_000)
    times = []
    for s in positions:
        fs = CppState(cells=s.cells, boards=s.boards, turn=s.turn, forced=s.forced)
        t0 = time.perf_counter()
        bot.choose(fs, rng)
        times.append(time.perf_counter() - t0)
    times = np.array(times)
    rows[depth] = {"mean_ms": float(times.mean()*1000), "median_ms": float(np.median(times)*1000),
                   "p95_ms": float(np.quantile(times, .95)*1000), "max_ms": float(times.max()*1000),
                   "n": len(times)}
    print(f"depth {depth:2d}: mean {rows[depth]['mean_ms']:8.2f} ms  median {rows[depth]['median_ms']:8.2f} ms  "
          f"p95 {rows[depth]['p95_ms']:9.2f} ms  max {rows[depth]['max_ms']:9.2f} ms")
json.dump(rows, open('docs/report/evidence/alphabeta-depth-cost.json','w'), indent=2)
