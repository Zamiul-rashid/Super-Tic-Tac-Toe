# Super Tic-Tac-Toe

Ultimate Tic-Tac-Toe with a Python rules engine, native C++ search backend,
neural self-play training, external-engine adapters, and reproducible tournament
evaluation.

## Setup

```bash
conda create -n sttt python=3.14 pip -y
conda activate sttt
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
make -C cpp
python -m unittest discover -s tests
```

The console game is available as `sttt` after installation.

## Play it in a browser

```bash
docker run -p 8000:8000 --cpus 4 ghcr.io/zamiul-rashid/super-tic-tac-toe:latest
```

A 385 MB self-hosted container: the iteration-6000 network, the native search,
and a board UI. No GPU and no PyTorch — inference runs on ONNX Runtime. See
[docs/web.md](docs/web.md).

## Stable commands

Resume a full checkpoint into a new run directory:

```bash
scripts/train.sh \
  --checkpoint runs/source/latest.pt \
  --output runs/continuation \
  --iterations 1000 \
  --lr-horizon 5000
```

Run the grand championship and AlphaBeta depth-10 sweep:

```bash
scripts/evaluate.sh \
  --checkpoint runs/continuation/latest.pt \
  --output runs/continuation/evaluation
```

These launchers accept paths and settings as arguments. A checkpoint or output
change should never require another script. See [scripts/README.md](scripts/README.md)
for all modes and examples.

## Training pipeline

The production launcher freezes the source checkpoint before loading it, refuses
to write into an existing run, and defaults to CUDA, FP16, the native backend,
cosine learning-rate scheduling, population training, and symmetry augmentation.

To bootstrap a new U-Net before online self-play:

```bash
python -m sttt.ai generate-dataset \
  --output data/bootstrap --games 50000 --workers 8

python -m sttt.ai pretrain \
  --dataset data/bootstrap --output runs/bootstrap --arch unet --fp16

scripts/train.sh \
  --checkpoint runs/bootstrap/latest.pt \
  --output runs/main
```

See [docs/training.md](docs/training.md) for the curriculum, checkpoint rules,
configuration, and evaluation guidance.

## Repository layout

| Path | Purpose |
| --- | --- |
| `sttt/` | Python game, search, training, evaluation, and adapters |
| `cpp/` | Native bitboard engine, MCTS, bindings, and benchmarks |
| `scripts/` | Stable launchers and reusable readiness/evaluation harnesses |
| `configs/` | Versioned training configuration |
| `engines/` | External-engine registry, wrappers, and vendored provenance |
| `tests/` | Unit, integration, adversarial, native, and E2E tests |
| `docs/` | Active documentation and historical engineering records |
| `runs/` | Generated checkpoints, metrics, and evaluation artifacts; ignored by Git |
| `data/` | Generated bootstrap datasets; ignored by Git |

## Documentation

- [Research report: methodology, literature, and measured results](docs/report/README.md)
- [Documentation index](docs/README.md)
- [Training and evaluation](docs/training.md)
- [Testing](docs/testing.md)
- [C++ engine design](docs/engineering/cpp-engine.md)
- [Native benchmark evidence](docs/engineering/cpp-benchmarks.md)
- [External engines](engines/README.md)

Historical requests, readiness ledgers, handovers, and implementation plans are
kept under [`docs/history/`](docs/history/) so they remain auditable without
cluttering the operational documentation.

## Direct CLI

The launchers cover repeatable production workflows. Lower-level commands remain
available for focused work:

```bash
python -m sttt.ai --help
python -m sttt.ai evaluate --checkpoint runs/main/latest.pt --games 20
python -m sttt.ai tournament --checkpoint runs/main/latest.pt \
  --opponents alphabeta tactical utttai --games 20
python -m sttt.visualize runs/main --output runs/main/charts
```

Evaluation ratings are local to the selected pool, budgets, and opening corpus.
Use mirrored openings and identical settings when comparing checkpoints.
