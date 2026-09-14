# Super Tic Tac Toe Console Game

The implementation and validation checklist for long native/GPU training runs is
in [Training readiness and improvement plan](TRAINING_READINESS_PLAN.md).

## Learning AI (4 GB GPU preset)

The learning agent uses a 1.77M-parameter residual policy/value network by default
(`--arch mlp` selects the original 161k network). Existing checkpoints load with
their original architecture. CPU workers build search trees; one inference owner
evaluates batches on the GPU, then performs training updates between iterations.
The original two-human console game remains available through `sttt`.

```bash
conda create -n sttt python=3.14 pip -y
conda activate sttt
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
```

### Training (canonical launcher)

`./train.sh <checkpoint> <output-dir> [population-config.json]` is the one
production command: it freezes an immutable copy of the start checkpoint,
resumes it on CUDA with FP16 AMP, the native search backend and the cosine
learning-rate schedule, uses the population curriculum from
`configs/population/*.json` (default `baseline.json`, 8 workers, leaf batch 64), prints its
effective configuration and refuses an output directory that already holds a
run. `train_v2.sh` and the `run_v2`/`run_v3` naming are deprecated; existing
`runs/run_v2` and `runs/big_run` are historical inputs, never outputs.

Before a long run, certify the environment with the staged readiness runner
(`scripts/check_training_ready.py`, stages build/native/cpu/mixed/failure on
CPU, gpu/memory/pilot on CUDA). The pilot stage runs `train.sh` itself for
20 measured iterations and writes the ETA; that ETA, not a fixed speed-up
claim, is the runtime estimate. `TRAINING_READINESS_PLAN.md` is the canonical
checklist and result ledger.

### Bootstrapped training (two-stage)

    # Stage 1: network-free native search games (C++ heuristic MCTS vs itself and alpha-beta d4-d6)
    nice -n 19 python -m sttt.ai generate-dataset --output data/bootstrap --games 50000 --workers 8
    # Stage 2: supervised fit; writes runs/bootstrap/latest.pt with a warm replay buffer
    nice -n 19 python -m sttt.ai pretrain --dataset data/bootstrap --output runs/bootstrap --arch unet --fp16
    # Online: the canonical launcher resumes the bootstrapped checkpoint
    ./train.sh runs/bootstrap/latest.pt runs/run_v4 configs/population/baseline.json

Architectures: `--arch resnet` (1.8M MLP-ResNet, default), `--arch unet` (hierarchical conv
U-Net with a dense action-value head). Replay sampling: `--replay-sampling stratified`
balances each batch across nine-ply game phases (`train.sh` enables it).

**`--arch unet` caveat:** its value head is measured dead — on 4,096 dataset
positions the output is constant −1.0 (min = mean = max = −1.0, std = 0.0),
saturated pre-tanh at initialisation because the head reads the unnormalised
macro residual stream (`ResNet` avoids this; its trunk is LayerNormed
throughout). Policy and Q heads are healthy on the same positions. Not fixed
in this task — see the ledger in `TRAINING_READINESS_PLAN.md` §7 before
starting a long U-Net run.

### High-Performance C++ Bitboard Engine

Build the native C++ extension for hardware-accelerated bitboard rules, MCTS search, and batch encoding:

```bash
make -C cpp clean && make -C cpp
```

Self-play training automatically uses the C++ backend when compiled (`--backend auto`):

```bash
python -m sttt.ai train --output runs/default
```

You can explicitly select the backend:

```bash
python -m sttt.ai train --backend cpp --output runs/cpp_run     # C++ bitboard engine & MCTS arena
python -m sttt.ai train --backend python --output runs/py_run   # Pure Python reference fallback
```

First training run (160 games; a starting experiment, not a strength guarantee):

```bash
python -m sttt.ai train --output runs/default
```

Resume for another 20 iterations, including optimizer and replay:

```bash
python -m sttt.ai train --resume runs/default/latest.pt --output runs/default
```

Play and evaluate:

```bash
python -m sttt.ai play --checkpoint runs/default/latest.pt
python -m sttt.ai evaluate --checkpoint runs/default/latest.pt --games 20
```

### External-engine tournaments

The testing branch includes a strict external-process adapter and named engine
registry. Third-party binaries are not vendored. Configure verified local
commands in `engines/registry.json`, based on `engines/registry.example.json`:

```bash
python -m sttt.ai tournament --checkpoint runs/big_run/latest.pt \
  --engine-config engines/registry.json \
  --opponents codingame-legend utttai --games 20 --simulations 512 \
  --output runs/big_run/tournaments/top-engines-20
```

Missing executables, timeouts, crashes and illegal moves fail a configured
engine matchup instead of silently substituting a built-in bot. See
`engines/README.md` for the supported wire protocols.

### Batched search, pruning, and the 4 GB GPU

Continue the existing large model in a new output directory:

```bash
python -m sttt.ai train --resume runs/large_256/latest.pt \
  --output runs/batched_256 --device cuda --workers 8 \
  --iterations 100 --games 8 --simulations 256 \
  --leaf-batch 16 --inference-batch 128 --inference-wait-ms 2 \
  --batch 128 --eval-every 25 --eval-games 20 --eval-simulations 512
```

This adds 800 games. Checkpoints are saved every iteration and alpha-beta
evaluations every 25 iterations. To continue this new run later, resume from
`runs/batched_256/latest.pt`, not the original checkpoint again. Set
`--eval-every 0` to omit periodic evaluations. An evaluation interrupt stops that
test; a training interrupt stops training and leaves the last completed checkpoint.

- `--workers`: persistent spawned CPU processes playing separate games; no CUDA
  models or contexts are created in them.
- `--leaf-batch`: pending leaf positions collected within each search tree (16 by
  default). Virtual losses spread these requests across branches and are cleared
  after inference, including on errors.
- `--inference-batch`: maximum positions in one forward pass (128 by default).
  The owner combines worker requests, waiting at most `--inference-wait-ms` for
  additional requests. Underfilled batches run immediately when necessary.
- `--batch`: training minibatch size; separate from search inference batching.
  Replay remains in system RAM. Use `--device cpu` for a CUDA-free fallback.

Workers stay alive across iterations, but search trees are recreated for each
game and never reused after a network update. JSONL logs include actual mean/max
inference batch size, inference positions/time, self-play time, search depth,
completed simulations, retained visits and peak PyTorch-allocated GPU memory.
The allocator metric excludes some driver/runtime memory. High GPU utilization
is not guaranteed: tree traversal, IPC, encoding and checkpoint writes still take
CPU time. Measure games/second as well as batch occupancy.

Search includes two distinct mechanisms:

1. **Soft suppression:** PUCT still values promising branches. After eight visits,
   a branch whose estimated value trails a better sampled branch gets a smaller
   exploration bonus (never below 20% of its original bonus). No action is deleted.
   Every 64 selections at a node, the least recently checked eligible branch is
   revisited. Legal moves retain a small positive prior. The suppression rule is
   an experimental heuristic; use ablations to check whether it helps strength.
2. **Exact proof propagation:** only terminal game outcomes create proofs. A node
   is a proven win if any child is a proven loss for the next player. It becomes
   a proven draw/loss only once all legal children are solved. Proven losing moves
   are skipped while non-losing or unresolved alternatives exist. Search stops
   early when the root is solved, and the returned policy includes only moves
   that achieve its proven result. Neural values are never treated as proofs.

This is an MCTS solver extension, not an implementation of the full PN-MCTS
paper's proof-number selection formula. Research references are
[MCTS-Solver](https://staff.ru.is/yngvi/pdf/WinandsBS08.pdf),
[Batch Monte Carlo Tree Search](https://arxiv.org/abs/2104.04278), and
[Proof Number Based Monte-Carlo Tree Search](https://arxiv.org/abs/2303.09449).
The batched implementation uses virtual losses and a shared inference owner; it
does not reproduce every heuristic or the transposition table in Batch MCTS.
Turn off features independently with
`--no-soft-pruning`, `--no-proofs`, and `--no-reuse` for ablations. Reuse retains
the played subtree and frees siblings after both players' moves; visits from that
subtree contribute to the next policy. `--simulations` limits **new** simulations
per move; exact solving may finish earlier. Reported `hard_pruned_choices` counts
excluded choices across selections, not unique pruned nodes.

For human play, batched GPU search can also evaluate multiple pending branches:

```bash
python -m sttt.ai play --checkpoint runs/large_256/latest.pt \
  --device cuda --simulations 5000 --leaf-batch 64 --no-adapt
```

With adaptation enabled, minimax proof pruning is disabled: a forced loss against
perfect opposition may still win against a fallible human. Updating the human
profile invalidates expected-response tree statistics, so that tree resets.
Soft suppression and inference batching still operate. For fully reusable,
proof-aware competitive search, use `--no-adapt` as above.

Batching changes selection order and can change playing strength at a fixed
simulation budget. Keep the leaf batch fixed in comparisons and test both equal
simulation budgets and equal wall-clock budgets before claiming a speed/strength gain.

### Stronger test opponents and checkpoint comparisons

The default evaluation opponent is now `alphabeta`. Available opponents:

| Opponent          | Behavior                                                                                                   |
| ----------------- | ---------------------------------------------------------------------------------------------------------- |
| `random`          | Uniform legal moves                                                                                        |
| `legacy-tactical` | Original stochastic local/global-win-biased baseline                                                       |
| `tactical`        | Takes global wins, screens immediate global losses, searches two plies to block threats and assess routing |
| `alphabeta`       | Iterative-deepening negamax with alpha-beta pruning, global/local line evaluation and move ordering        |
| `checkpoint`      | Another neural checkpoint using reusable MCTS                                                              |

Alpha-beta defaults to `--opponent-depth 3 --opponent-nodes 3000`. The node cap
covers recursive search; immediate-win/reply-safety checks and move ordering have
additional cost. It returns the last completed depth when the budget is exhausted.
Its horizon evaluations are heuristics, not guarantees of optimal play.

Run one checkpoint across multiple budgets and seeds (a separate report per pair):

```bash
python -m sttt.ai evaluate --checkpoint runs/large_256/latest.pt \
  --opponent alphabeta --games 100 --seeds 0 1 2 \
  --simulation-budgets 128 512 1024 5000 --leaf-batch 16 --device cuda

python -m sttt.ai evaluate --checkpoint runs/batched_256/latest.pt \
  --opponent checkpoint --opponent-checkpoint runs/large_256/latest.pt \
  --simulations 512 --opponent-simulations 512 --games 100 --seeds 0 1 2
```

The first command runs 1,200 evaluation games and can take substantial time.
Agent and opponent random streams are separate for each game. Starting sides
alternate; use even game counts. By default, each pair shares two seeded random
opening moves, with the tested agent playing X once and O once. Openings are fixed
across budgets and recorded as zero-based action indices in JSON/CSV. This avoids
counting identical deterministic checkpoint games as independent evidence. Use
`--opening-moves 0` for empty-board tests, or up to 8 for more opening diversity;
compare runs with the same opening settings. Paired games are not statistically
independent, so uncertainty analysis should account for pairs.
Tests measure general competitive strength with
adaptation disabled. Proof/soft-pruning/reuse flags apply to the tested agent;
the checkpoint opponent keeps default search settings. The reports record search
version/config, device, batching, opponent budgets and opponent checkpoint hash.

### Saved evaluation results and visualizer

Each evaluation automatically writes three matching files to the checkpoint's
`evaluations/` directory. Pass `--output runs/my-tests` to choose another directory.
Unique run IDs keep repeated tests from overwriting each other.

- `eval-<id>.json`: settings, checkpoint SHA-256 and iteration, aggregate results,
  results by starting side, and individual games.
- `eval-<id>_games.csv`: one row per completed game with outcome, score, starting
  side, move count, elapsed seconds and agent search seconds. Settings repeat on
  each row so exports can be combined later.
- `eval-<id>_summary.csv`: one row per evaluation, including wins/draws/losses,
  win rate, score rate and average game length/time.

Score rate means `(wins + 0.5 * draws) / completed games`; rates in exports are
fractions from 0 to 1. Exports update after each game. Ctrl+C saves completed games
with `status=interrupted`; unfinished games are excluded. The current evaluator
tests general playing strength against the selected opponent, with adaptation
disabled. New tactical reports are labeled `tactical-v2`; legacy reports are not
directly comparable to this stronger baseline. Compare the same opponent and search budget, use several
seeds, and balance starting sides (an even game count).

Install plotting dependencies into the Conda environment and generate charts:

```bash
conda activate sttt
python -m pip install -r requirements.txt
python -m sttt.visualize runs/starter --output runs/charts
```

The visualizer recursively reads evaluation JSON and `metrics.jsonl`, producing
`evaluation.png` (outcomes, score by side, cumulative test score, game lengths)
and `training.png` (loss, replay size, iteration time). It also accepts several
run directories or individual JSON files for comparisons. CSV exports are for
spreadsheet/external analysis; the visualizer reads the richer JSON exports.
Reports remain separate and are numbered chronologically; the terminal prints
their checkpoint paths and run IDs. These are descriptive charts, not proof of
statistically significant improvement. Cumulative score is evaluation progress,
not learning during the test; falling training loss alone does not prove strength.

Use `--show` to open Matplotlib windows with zoom/pan and image saving (requires
a desktop display). PNG generation works offline and without a display or GPU.
Rerun the command after new evaluations; existing chart PNGs are replaced.

For a quick pipeline test use `train --iterations 1 --games 2 --simulations 8
--steps 4 --output runs/smoke`. Use `--device cpu` if CUDA is unavailable.
Training prints time and loss per iteration and saves `metrics.jsonl`, `latest.pt`
and historical model weights. Interrupting a partial iteration preserves the last
completed checkpoint. Resuming continues training but does not reproduce the exact
random stream of an uninterrupted run.

During play, enter `board cell` (both 1–9, numbered left-to-right, top-to-bottom).
The first move can use any board; won or drawn boards close permanently. A move
that sends the next player to a closed board grants free choice among open boards.
Use `--side O` to play second and `q` to exit.

The human profile in `profiles/player.json` updates after every valid human move.
It learns a discounted Bayesian mixture of five simple behavioral models and
changes the sampled opponent responses in search. Use `--profile profiles/name.json`
for another player and `--no-adapt` to compare behavior without that profile.
The neural network learns from self-play; human observations update the separate
profile, not the network weights. The current experts are deliberately limited;
this version does not yet include a recurrent opponent encoder,
symmetry augmentation or population training. Historical weights support
later comparisons. Winning strength and adaptation gains require evaluation;
a smoke-trained checkpoint is only evidence that the pipeline runs.

Code: `sttt/env.py` (rules), `learning.py` (network), `search.py` (MCTS),
`opponent.py` (online adaptation), `ai.py` (training/play/evaluation).

This Python console-based game offers a twist on the classic Tic Tac Toe by introducing Super Tic Tac Toe. The game is played on a larger grid consisting of nine smaller Tic Tac Toe boards, creating an engaging and strategic gameplay experience.

## How to Play

- **Installation**: Ensure Python 3.x is installed on your system.
- **Run the Game**: Execute the Python script `super_tic_tac_toe.py`.
- **Gameplay**: Players take turns placing their 'X' or 'O' markers in the local boards, aiming to win three boards in a row horizontally, vertically, or diagonally to claim victory.

## Features

- Implemented in Python for console-based interaction.
- Provides a challenging gameplay experience requiring strategic moves to win.
- Follows the rules of Super Tic Tac Toe for an engaging and enjoyable gaming session.

Enjoy playing Super Tic Tac Toe! For any inquiries or issues, contact the developer at [zamiulrashid1@gmail.com , abyashrirproyas@gmail.com](mailto:zamiulrashid1@gmail.com,mailto:abyashrirproyas@gmail.com).
