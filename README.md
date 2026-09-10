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
