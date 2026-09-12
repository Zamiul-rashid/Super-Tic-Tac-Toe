# Engine wrappers

The project includes a working uttt.ai wrapper and uses the official OpenSpiel
C++ game with its Python MCTS implementation. Training selects both through
the population league.

## Setup

From the project root, using the same Python interpreter as training:

```bash
python -m pip install --target engines/runtime -r engines/requirements.txt
python -m unittest tests.test_utttai_wrapper tests.test_openspiel
```

The runtime directory is ignored by Git. The upstream uttt.ai Python source,
Apache-2.0 license and provenance are in `engines/vendor/utttai`. The deployed
ONNX model is in `engines/models/utttai_stage2.onnx`; its checksum is checked
when loaded. No source, model or dependency is loaded from /tmp.

## uttt.ai

`engines/registry.json` is ready to use. Its `{python}` placeholder selects
the trainer's interpreter. `{simulations}` and `{seed}` are resolved per game;
tournaments default to 128 simulations and seed 0.

`python -m sttt.utttai_wrapper` runs a persistent process with one JSON
position per input line and one integer action per output line. Positions carry
`cells`, `boards`, `turn`, `forced` and `result`, using the local State
representation. This supports arbitrary openings, random opponent moves and
process resets. The wrapper validates legality against both engines.

For other clients, `--protocol action_index` accepts the previous move (-1
for the first request), legal count, and one legal action per line. This mode
requires empty-board games and no injected moves. The state_json protocol is
used for training.

The official NMCTS implementation runs on CPU with a fresh root each move,
exploration strength 2.0, and the deployed stage-2 ONNX policy/value network.
A 64/128/256 budget is used in training. Each process limits inference to one
thread. Missing dependencies, checksum errors, timeouts and illegal moves fail
the game; league play never substitutes a tactical opponent.

## OpenSpiel

The adapter requires the official compiled `pyspiel` extension. The prior local
Python substitute is rejected. OpenSpiel uses separate board-selection and
cell-selection actions on free-choice turns. The adapter translates these
and replays every opening/learner/noise move to keep the C++ state synchronized.
A budget is per OpenSpiel decision, so a free-choice move can use two searches.

Old benchmark results obtained with the Python substitute should not be
presented as measurements against official OpenSpiel.

## Tournament

```bash
python -m sttt.ai tournament --checkpoint runs/big_run/latest.pt \
  --engine-config engines/registry.json --opponents utttai openspiel-mcts:128 \
  --games 20 --simulations 512 --device cpu
```

The example registry also documents the optional CodinGame executable protocol;
that engine is not part of this training league.
