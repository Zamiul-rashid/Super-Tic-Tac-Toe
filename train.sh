#!/usr/bin/env bash
# Canonical training launcher. Resumes a FULL checkpoint with the cosine
# learning-rate schedule, FP16 AMP on CUDA and the native search backend.
# train_v2.sh and the run_v2/run_v3 naming are deprecated; this is the one
# production command. The readiness pilot gate runs this same script.
#
#   train.sh <checkpoint> <output-dir> [population-config.json]
#
# Every setting below is explicit. The launcher freezes an immutable copy of the
# source checkpoint into the output directory and resumes from THAT copy, so a
# rolling latest.pt cannot change under the run and the source run is never
# written to. The training process itself refuses a second writer on the same
# output directory before spawning any worker (RunOwnership).
#
# Verified 2026-09-14 on CUDA with the exact flags below
# (runs/readiness/cosine-burst-*): FP16 AMP active, GradScaler scale held at
# 65536 across iterations and across a resume, cosine schedule attached as
# phase 0 to a legacy checkpoint and continued on resume, peak allocated GPU
# memory 60 MiB at batch 512 / inference-batch 512 (540 MiB process footprint).
#
# Environment overrides:
#   STTT_PY     interpreter (default: python on PATH)
#   WORKERS     self-play worker processes (default 10)
#   ITERATIONS  additional iterations to run (default 5000)
#   LR_HORIZON  cosine horizon in completed iterations (default = ITERATIONS)
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 <checkpoint.pt> <new-output-dir> [population-config.json]" >&2
  exit 2
fi
SRC_CKPT=$1
OUTPUT=$2
POP_CONFIG=${3:-configs/population/baseline.json}

REPO=$(cd "$(dirname "$0")" && pwd)
cd "$REPO"
PY=${STTT_PY:-python}
WORKERS=${WORKERS:-10}
ITERATIONS=${ITERATIONS:-5000}
LR_HORIZON=${LR_HORIZON:-$ITERATIONS}

[[ -f "$SRC_CKPT" ]] || { echo "checkpoint not found: $SRC_CKPT" >&2; exit 2; }
[[ -f engines/registry.json ]] || { echo "missing engines/registry.json" >&2; exit 2; }
[[ -f "$POP_CONFIG" ]] || { echo "population config not found: $POP_CONFIG" >&2; exit 2; }
if [[ -e "$OUTPUT/latest.pt" ]]; then
  echo "$OUTPUT already holds a run; use a new output directory (never overwrite the source run)." >&2
  exit 2
fi
mkdir -p "$OUTPUT"

# Immutable start checkpoint: copied and hashed, verified unchanged during copy.
FROZEN=$("$PY" - "$SRC_CKPT" "$OUTPUT/start-checkpoint" <<'PYEOF'
import sys
from sttt.evaluation import freeze_checkpoint, checkpoint_identity
dest, digest = freeze_checkpoint(sys.argv[1], sys.argv[2])
ident = checkpoint_identity(dest)
print(f"frozen {dest} iteration={ident['iteration']} arch={ident['arch']} sha256={digest}", file=sys.stderr)
print(dest)
PYEOF
)

echo "=========================================================================="
echo " TRAINING: cosine LR, FP16 AMP, CUDA, native search"
echo "   start checkpoint : $FROZEN"
echo "   output           : $OUTPUT"
echo "   population config: $POP_CONFIG"
echo "   iterations       : $ITERATIONS   cosine horizon: $LR_HORIZON   workers: $WORKERS"
echo "=========================================================================="

"$PY" -u -m sttt.ai train \
  --resume "$FROZEN" \
  --output "$OUTPUT" \
  --device cuda \
  --backend cpp \
  --fp16 \
  --lr-schedule cosine --lr 0.001 --lr-min 0.00001 --lr-iterations "$LR_HORIZON" \
  --population --population-config "$POP_CONFIG" \
  --population-checkpoints "$OUTPUT" \
  --engine-config engines/registry.json \
  --workers "$WORKERS" \
  --iterations "$ITERATIONS" \
  --games 16 \
  --simulations 512 \
  --leaf-batch 16 \
  --inference-batch 512 \
  --inference-wait-ms 2 \
  --steps 100 \
  --batch 512 \
  --buffer 200000 \
  --save-every 50 \
  --keep-checkpoint-window 500 \
  --eval-every 100 \
  --eval-games 20 \
  --eval-simulations 512 \
  --augment-symmetry \
  --seed 42 \
  2>&1 | tee -a "$OUTPUT/train.log"

result=${PIPESTATUS[0]}
echo "Training exited with status ${result}." | tee -a "$OUTPUT/train.log"
exit "$result"
