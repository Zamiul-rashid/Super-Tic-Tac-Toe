# Search correctness audit (iteration 4530)

**Date**: 2026-09-18 · **Scope**: `sttt/search.py` (`TreeSearch`, `root_action_values`)
and the native `CppTreeSearch` (`cpp/src/mcts.cpp`, `cpp/include/sttt_mcts.hpp`).
**Regression tests**: `tests/test_search_audit.py` (both backends, stub evaluators, no checkpoint).
**Diagnostic data**: [`search-audit-iter4530.json`](search-audit-iter4530.json), produced by
`scripts/search_diagnostics.py`.

> **One correctness bug was found and fixed (batched search, below).** The fix changes
> search behaviour for every leaf batch > 1 whenever a terminal or proven leaf is selected
> while evaluations are pending. Strength, championship and self-play numbers produced
> before this fix (including all archived 1806/4193/4530 results) must **not** be pooled
> or directly compared with results produced after it. The native build reports the same
> `SOURCE_REVISION` (`b221f8b`) before and after the fix because the fix is uncommitted;
> only `BUILD_ID` (`20260918075553_b221f8b` pre-fix, `20260918080519_b221f8b` post-fix)
> distinguishes them. Commit the fix before running any post-fix strength experiment.

## Checks

| # | Check | Result | Evidence (tests in `tests/test_search_audit.py`) |
| --- | --- | --- | --- |
| 1 | Value perspective | Pass | Stub "X ahead by 0.5" (mover value `0.5·turn`): every visited node in both backends has `total/n = 0.5·turn` exactly, root Q = `0.5·root.turn` for X and O to move. Stub "mover is winning (+0.6)": every root Q = −0.6. |
| 2 | Terminal handling | Pass after fix | Immediate global win proven at root expansion (0 simulations, one-hot policy, Q = +1); forced loss proven −1 and stops early; draw proven 0 with Q = 0; forced block (exact value: draw) gives all mass to the block; a position with two immediately losing moves and one drawing move never gives the losing moves mass — at 128/512 sims and leaf batch 1/16/64. The draw case **failed** at leaf batch 16/64 before the fix. |
| 3 | Legal actions | Pass | Forced board, sent-to-closed-board (free choice minus closed boards) and full free choice, with a stub that puts policy mass on illegal actions: root and every expanded node have exactly the legal children; zero probability, visits and Q-mask on illegal actions. |
| 4 | Simulation accounting | Pass (convention below) | 128/512/2000 sims × batch 1/16 × both backends: `completed_simulations` = budget, `root.n` = budget, `neural_positions` = evaluator calls = budget − terminal backups + 1, `inference_batches` = evaluator batch calls + 1; after `advance`, `retained_visits` = the kept child's visits and a new 128-sim budget adds exactly 128. A proven retained root does no new work. |
| 5 | Python/C++ agreement | Pass | 8 random positions × 3 configs × batch 1/16 × two consecutive moves (reuse) with a tie-free deterministic stub: identical per-child visit counts, identical stats, policy and Q equal. With iteration 4530 all 78 backend pairs in the diagnostic also agree exactly. |
| 6 | Sequential vs batched | Pass after fix | After every run, all nodes have `in_flight = 0` and `pending = False`; `root.n = Σ child.n`; each expanded non-root node has `n = 1 + Σ child.n` (≥ with proofs); `|total| ≤ n`; returned policy = normalized root visits. Tactical choices on win/block/avoid positions are identical at batch 1/16/64. |

### Simulation-accounting convention

* `simulations` is the number of **new** backups requested per `run` call. Retained subtree
  visits are reported in `stats['retained_visits']` and are not charged to the budget.
* `completed_simulations` counts backups: evaluated leaves plus terminal/proven-leaf backups
  (which cost no network call). It is below the budget only when the root becomes proven
  (early proof termination); an immediately winning root finishes with 0 simulations.
* `neural_positions` counts evaluated positions **including** the root expansion, whose value
  is not backed up; `inference_batches` likewise includes the root call.
* Values are mover-relative everywhere (evaluator output, `node.total`, `node.solved`);
  `root_action_values` returns Q from the root player's view and exact ±1/0 for proven children.

### Documented tie-breaking (not bugs)

Ties go to the first candidate in legal-action order in both backends. Exact PUCT ties
(common with coarse stub priors such as multiples of 1/16) can resolve differently because
Python normalizes priors with numpy pairwise summation in float64 while C++ sums sequentially,
C++ stores leaf values as float32 and uses `soft_margin = 0.2f`, and after a pending child is
skipped inside a batch Python keeps candidate order while C++ swaps the last candidate into its
slot. With tie-free inputs the trees are identical.

## Bug: batched search spends its budget on a known leaf while evaluations are pending

**Where**: the leaf-collection loop in `TreeSearch.run` (`sttt/search.py`), both loops of
`PyFastTreeSearch_run_impl` (`cpp/src/mcts.cpp`) and `MCTSEngine::run_heuristic`
(`cpp/include/sttt_mcts.hpp`).

**Mechanism**: a pending leaf blocks every path through it, so once the unexplored branches
are pending, selection keeps returning the same terminal or proven leaf. Each return was
backed up immediately and counted as a completed simulation, so the loop only stopped when the
budget was exhausted — before the pending evaluations were even applied.

**Reproduction** (position `TERMINAL_OPTION`: 78 ends the game as a draw, 79/80 continue;
iteration 4530, leaf batch 16): 2000 simulations gave 78 → 1998 visits, 79 → 1, 80 → 1, with
3 network evaluations and the root never proven. Leaf batch 1 proves the draw in 129
simulations. `proven-draw` at batch 16 spent the full 128/512/2000 budget where batch 1 needs 6.
With a proof-free config and a stub that favours 79/80, batch 16 chose the terminal draw
(198/200 visits) while batch 1 spread visits 70/60/70. Self-play used leaf batch 64 (the 4530
training config: 512 sims, `leaf_batch` 64), so endgame policy targets could concentrate on
terminal or proven moves for reasons unrelated to their value.

**Fix** (the same guard at all four collection loops): after a terminal/proven-leaf backup, stop collecting if
any evaluation is pending, so the batch is flushed and selection resumes with real statistics.
Leaf batch 1 is unaffected (nothing is ever pending there): every batch-1 diagnostic row is
identical before and after the fix, as are all opening rows at batch 16. Regression tests:
`test_pending_evaluations_do_not_starve_behind_a_terminal_child` and
`test_draw_is_proven_exactly` (both failed for both backends before the fix). Cost: smaller
average batches in endgame positions (e.g. `avoid-sending-to-threat`, 2000 sims: 17 → 160
inference batches), no change in opening positions.

## Diagnostic: iteration 4530

Checkpoint `latest.pt` of `runs/report-championship-2016-20260917/championship/evaluation-inputs/candidate`
on the training host, sha256 `cb04e705…3f671c6d`, run post-fix on CPU with 2 torch threads,
default `SearchConfig` (identical to the checkpoint's), both backends, 128/512/2000 sims,
leaf batch 1 and 16, no noise. 13 positions: 7 tactical with known safe moves (global win for X,
for O and with free choice; forced block for X and O; avoid sending the opponent to a winning
board, twice), 3 exact endgames (forced loss, proven draw, terminal-draw option) and 3 openings.

* **Tactics**: all 84 tactical rows chose a winning/safe move (0 failures), pre- and post-fix.
  Wins are proven at root expansion with 0 simulations; forced losses are proven after 2.
* **Model value**: consistent with perspective — empty board +0.214 (X to move), after the
  center −0.331 (O to move), global-win positions +1.000 for either colour. The model is nearly
  neutral (+0.002) on the forced-block position (exact value: draw).
* **Backends**: Python and C++ gave identical visits, Q and stats on all 78 pairs; C++ is
  1.1–1.3× faster here because the neural network dominates (e.g. opening, 2000 sims, batch 16:
  0.53 s vs 0.71 s).
* **Budget**: from the empty board, the 128/512/2000-sim searches at batch 1 give the center
  (40) 116/428/1560 visits with Q 0.26/0.27/0.24.

### Hypotheses (suspicious, not changed)

1. **Virtual loss flattens root visits at large leaf batches.** Post-fix, C++, 512 sims, empty
   board: the most-visited share is 0.836 at batch 1, 0.416 at 16 and 0.061 at 64 (entropy
   1.09 → 2.53 → 4.18 nats; uniform over 81 moves is 4.39). After 1.e4 / 4 plies: 0.674 → 0.301.
   The chosen move is unchanged, but self-play at 512 sims × batch 64 therefore trains the policy
   head on much flatter targets than sequential search would produce. Each pending visit counts
   as a full loss (`total + in_flight`). Worth a controlled training comparison; not a correctness defect.
2. **Proven-lost roots stop searching immediately** and play by pre-proof visits (the
   `forced-loss` position picks 38 after 2 simulations). Against a fallible opponent, preferring
   the longest resistance may score better.
3. **Build provenance does not capture uncommitted source changes** (`SOURCE_REVISION` is the
   HEAD commit only), so pre- and post-fix artifacts are distinguishable only by `BUILD_ID`.

## Reproduce

```bash
make -C cpp clean && make -C cpp PYTHON=$(which python)
python -m unittest tests.test_search_audit -v
nice -n 19 env OMP_NUM_THREADS=2 python scripts/search_diagnostics.py \
  --checkpoint PATH/latest.pt --output docs/engineering/search-audit-iter4530.json \
  --budgets 128 512 2000 --leaf-batch 1 16
```
