#!/usr/bin/env bash
# Stable production launcher for online training.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/train.sh --checkpoint PATH --output DIR [options] [-- TRAIN_ARGS...]
  scripts/train.sh CHECKPOINT OUTPUT [POPULATION_CONFIG]

Required:
  --checkpoint PATH             Full checkpoint to resume
  --output DIR                  New run directory (must not contain latest.pt)

Common options:
  --population-config PATH      Default: configs/population/baseline.json
  --population-checkpoints DIR  Historical model directory; default: output DIR
  --python PATH                 Python interpreter; default: $STTT_PY or python
  --iterations N                Additional iterations; default: $ITERATIONS or 5000
  --workers N                   Self-play workers; default: $WORKERS or 8
  --device auto|cpu|cuda        Default: cuda
  --backend auto|python|cpp     Default: cpp
  --lr FLOAT                    Default: 0.001
  --lr-min FLOAT                Default: 0.00001
  --lr-horizon N                Cosine horizon; default: $LR_HORIZON or iterations
  --max-iterations N            Optional cumulative stopping iteration
  --games N                     Games per iteration; default: 16
  --simulations N               Learner simulations per move; default: 512
  --fp16 | --no-fp16            Default: enabled
  --population | --no-population
  --augment-symmetry | --no-augment-symmetry
  --dry-run                     Validate and print the command without running it
  -h, --help                    Show this help

Arguments after -- are forwarded to `python -m sttt.ai train`, e.g. the
opt-in `--save-game-records DIR --reanalyse` loss review. The source
checkpoint is frozen into OUTPUT before use.
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PY=${STTT_PY:-python}
CHECKPOINT=""
OUTPUT=""
POPULATION_CONFIG=configs/population/baseline.json
POPULATION_CHECKPOINTS=""
ITERATION_COUNT=${ITERATIONS:-5000}
WORKER_COUNT=${WORKERS:-8}
LR_HORIZON_VALUE=${LR_HORIZON:-}
DEVICE=cuda
BACKEND=cpp
LR=0.001
LR_MIN=0.00001
MAX_ITERATIONS=""
GAMES=16
SIMULATIONS=512
LEAF_BATCH=64
INFERENCE_BATCH=1024
INFERENCE_WAIT_MS=2
STEPS=100
BATCH=512
BUFFER=200000
SAVE_EVERY=50
KEEP_CHECKPOINT_WINDOW=500
KEEP_CHECKPOINTS=10
EVAL_EVERY=300
EVAL_GAMES=20
EVAL_SIMULATIONS=512
SEED=42
FP16=1
POPULATION=1
AUGMENT_SYMMETRY=1
DRY_RUN=0
EXTRA_ARGS=()

# Keep the old concise positional form while exposing every changing value as
# a named option.
if [[ $# -ge 2 && ${1:-} != -* ]]; then
  CHECKPOINT=$1
  OUTPUT=$2
  if [[ $# -ge 3 && ${3:-} != -* ]]; then
    POPULATION_CONFIG=$3
    shift 3
  else
    shift 2
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT=${2:?missing value for --checkpoint}; shift 2 ;;
    --output) OUTPUT=${2:?missing value for --output}; shift 2 ;;
    --population-config) POPULATION_CONFIG=${2:?missing value for --population-config}; shift 2 ;;
    --population-checkpoints) POPULATION_CHECKPOINTS=${2:?missing value for --population-checkpoints}; shift 2 ;;
    --python) PY=${2:?missing value for --python}; shift 2 ;;
    --iterations) ITERATION_COUNT=${2:?missing value for --iterations}; shift 2 ;;
    --workers) WORKER_COUNT=${2:?missing value for --workers}; shift 2 ;;
    --device) DEVICE=${2:?missing value for --device}; shift 2 ;;
    --backend) BACKEND=${2:?missing value for --backend}; shift 2 ;;
    --lr) LR=${2:?missing value for --lr}; shift 2 ;;
    --lr-min) LR_MIN=${2:?missing value for --lr-min}; shift 2 ;;
    --lr-horizon) LR_HORIZON_VALUE=${2:?missing value for --lr-horizon}; shift 2 ;;
    --max-iterations) MAX_ITERATIONS=${2:?missing value for --max-iterations}; shift 2 ;;
    --games) GAMES=${2:?missing value for --games}; shift 2 ;;
    --simulations) SIMULATIONS=${2:?missing value for --simulations}; shift 2 ;;
    --leaf-batch) LEAF_BATCH=${2:?missing value for --leaf-batch}; shift 2 ;;
    --inference-batch) INFERENCE_BATCH=${2:?missing value for --inference-batch}; shift 2 ;;
    --inference-wait-ms) INFERENCE_WAIT_MS=${2:?missing value for --inference-wait-ms}; shift 2 ;;
    --steps) STEPS=${2:?missing value for --steps}; shift 2 ;;
    --batch) BATCH=${2:?missing value for --batch}; shift 2 ;;
    --buffer) BUFFER=${2:?missing value for --buffer}; shift 2 ;;
    --save-every) SAVE_EVERY=${2:?missing value for --save-every}; shift 2 ;;
    --keep-checkpoint-window) KEEP_CHECKPOINT_WINDOW=${2:?missing value for --keep-checkpoint-window}; shift 2 ;;
    --keep-checkpoints) KEEP_CHECKPOINTS=${2:?missing value for --keep-checkpoints}; shift 2 ;;
    --eval-every) EVAL_EVERY=${2:?missing value for --eval-every}; shift 2 ;;
    --eval-games) EVAL_GAMES=${2:?missing value for --eval-games}; shift 2 ;;
    --eval-simulations) EVAL_SIMULATIONS=${2:?missing value for --eval-simulations}; shift 2 ;;
    --seed) SEED=${2:?missing value for --seed}; shift 2 ;;
    --fp16) FP16=1; shift ;;
    --no-fp16) FP16=0; shift ;;
    --population) POPULATION=1; shift ;;
    --no-population) POPULATION=0; shift ;;
    --augment-symmetry) AUGMENT_SYMMETRY=1; shift ;;
    --no-augment-symmetry) AUGMENT_SYMMETRY=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$CHECKPOINT" ]] || { echo "--checkpoint is required" >&2; usage >&2; exit 2; }
[[ -n "$OUTPUT" ]] || { echo "--output is required" >&2; usage >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -f engines/registry.json ]] || { echo "missing engines/registry.json" >&2; exit 2; }
if [[ $POPULATION -eq 1 ]]; then
  [[ -f "$POPULATION_CONFIG" ]] || { echo "population config not found: $POPULATION_CONFIG" >&2; exit 2; }
fi
if [[ -e "$OUTPUT/latest.pt" ]]; then
  echo "$OUTPUT already contains latest.pt; choose a new output directory" >&2
  exit 2
fi

LR_HORIZON_VALUE=${LR_HORIZON_VALUE:-$ITERATION_COUNT}
POPULATION_CHECKPOINTS=${POPULATION_CHECKPOINTS:-$OUTPUT}

if [[ $DRY_RUN -eq 0 ]]; then
  mkdir -p "$OUTPUT"
  FROZEN=$("$PY" - "$CHECKPOINT" "$OUTPUT/start-checkpoint" <<'PYEOF'
import sys
from sttt.evaluation import checkpoint_identity, freeze_checkpoint

destination, digest = freeze_checkpoint(sys.argv[1], sys.argv[2])
identity = checkpoint_identity(destination)
print(
    f"frozen {destination} iteration={identity['iteration']} "
    f"arch={identity['arch']} sha256={digest}",
    file=sys.stderr,
)
print(destination)
PYEOF
  )
else
  FROZEN="$OUTPUT/start-checkpoint/$(basename "$CHECKPOINT")"
fi

COMMAND=(
  "$PY" -u -m sttt.ai train
  --resume "$FROZEN"
  --output "$OUTPUT"
  --device "$DEVICE"
  --backend "$BACKEND"
  --lr-schedule cosine --lr "$LR" --lr-min "$LR_MIN" --lr-iterations "$LR_HORIZON_VALUE"
  --engine-config engines/registry.json
  --workers "$WORKER_COUNT"
  --iterations "$ITERATION_COUNT"
  --games "$GAMES"
  --simulations "$SIMULATIONS"
  --leaf-batch "$LEAF_BATCH"
  --inference-batch "$INFERENCE_BATCH"
  --inference-wait-ms "$INFERENCE_WAIT_MS"
  --steps "$STEPS"
  --batch "$BATCH"
  --buffer "$BUFFER"
  --save-every "$SAVE_EVERY"
  --keep-checkpoint-window "$KEEP_CHECKPOINT_WINDOW"
  --keep-checkpoints "$KEEP_CHECKPOINTS"
  --eval-every "$EVAL_EVERY"
  --eval-games "$EVAL_GAMES"
  --eval-simulations "$EVAL_SIMULATIONS"
  --seed "$SEED"
)

if [[ $FP16 -eq 1 ]]; then COMMAND+=(--fp16); fi
if [[ $POPULATION -eq 1 ]]; then
  COMMAND+=(--population --population-config "$POPULATION_CONFIG" --population-checkpoints "$POPULATION_CHECKPOINTS")
fi
if [[ $AUGMENT_SYMMETRY -eq 1 ]]; then COMMAND+=(--augment-symmetry); fi
if [[ -n "$MAX_ITERATIONS" ]]; then COMMAND+=(--max-iterations "$MAX_ITERATIONS"); fi
COMMAND+=("${EXTRA_ARGS[@]}")

printf 'Training configuration:\n'
printf '  checkpoint: %s\n' "$CHECKPOINT"
printf '  frozen:     %s\n' "$FROZEN"
printf '  output:     %s\n' "$OUTPUT"
printf '  iterations: %s (cosine horizon %s)\n' "$ITERATION_COUNT" "$LR_HORIZON_VALUE"
printf '  workers:    %s\n' "$WORKER_COUNT"
printf '  command:   '
printf ' %q' "${COMMAND[@]}"
printf '\n'

if [[ $DRY_RUN -eq 1 ]]; then exit 0; fi

set +e
"${COMMAND[@]}" 2>&1 | tee -a "$OUTPUT/train.log"
result=${PIPESTATUS[0]}
set -e
printf 'Training exited with status %s.\n' "$result" | tee -a "$OUTPUT/train.log"
exit "$result"
