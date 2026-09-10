# Super Tic Tac Toe Console Game

## Learning AI (4 GB GPU preset)

The learning agent uses a 161k-parameter policy/value network, self-play and PUCT
search. GPU training defaults to batches of 128; replay and sequential search stay
on CPU. The original two-human console game remains available through `sttt`.

```bash
conda create -n sttt python=3.14 pip -y
conda activate sttt
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
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
tests general playing strength against random or simple tactical policies, with
adaptation disabled. Compare the same opponent and search budget, use several
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
this version does not yet include a recurrent opponent encoder, batched parallel
search, symmetry augmentation or population training. Historical weights support
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
