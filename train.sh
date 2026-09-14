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
# output directory before spawning any worker (RunOwnership). The launcher is
# ARCH-independent: it resumes whatever architecture the frozen checkpoint
# carries (`resnet` or `unet`), so a bootstrapped U-Net checkpoint resumes here
# unchanged.
#
# Verified 2026-09-14 on CUDA with the exact flags below: FP16 AMP active,
# GradScaler state held across iterations and restored across a resume, cosine
# schedule attached as phase 0 to a legacy checkpoint and continued on resume,
# peak reserved GPU memory 88 MiB. Mean 16.2 s/iteration on a contended 4090
# (runs/readiness/m9-pilot-fast-*); measure your own with the pilot gate.
#
# Environment overrides:
#   STTT_PY     interpreter (default: python on PATH)
#   WORKERS     self-play worker processes (default 8; the CPU-load rule caps this box at 10)
#   ITERATIONS  additional iterations to run (default 5000)
#   LR_HORIZON  cosine horizon in completed iterations (default = ITERATIONS)
#
# Bootstrapped start: generate a dataset and pretrain first, then resume here:
#   nice -n 19 $PY -m sttt.ai generate-dataset --output data/bootstrap --games <N> --workers 8
#   nice -n 19 $PY -m sttt.ai pretrain --dataset data/bootstrap --output runs/bootstrap --arch unet --fp16
#   ./train.sh runs/bootstrap/latest.pt runs/run_v4 configs/population/baseline.json
#
# Storage: latest.pt (full resume state) is rewritten every iteration. pack_replay
# (sttt/ai.py) omits the replay's Q/Q-mask block entirely when no row carries a
# real target, so the two cases differ: `--arch resnet` never populates Q
# (ResNet.forward_all returns q=None) and settles at ~320 MB; `--arch unet`
# always does, adding a 200,000x81 fp32 Q tensor (~65 MB) plus a 200,000x81 bool
# Q-mask (~16 MB), ~401 MB total. A model-NNNN.pt weights snapshot (~7 MB) is
# written every 50 iterations and pruned to the last 500 iterations AND the
# last 10 snapshots, so the run directory settles at roughly 320 MB + 70 MB
# (resnet) or 400 MB + 70 MB (unet). best.pt is never pruned.
#
# Stratified replay sampling (sttt/replay_sampling.py, --replay-sampling
# stratified) is implemented and tested but deliberately NOT enabled below.
# Measured on the production replay (runs/run_v2/latest.pt, 200,000 rows): the
# thinnest ply bin (63-71) held only 176 rows, yet StratifiedSampler's flat
# 512/8=64-per-bin first pass gives it the same 64-row share as every other
# bin before the remaining-share redistribution ever runs -- a ~141x oversample
# of that bin sustained for hundreds of iterations (the FIFO replaces only
# ~0.3 rows/iteration in that bin). Do not re-enable this flag here without
# first adding a quota rule that bounds oversampling for thin bins.
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
WORKERS=${WORKERS:-8}
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
  --leaf-batch 64 \
  --inference-batch 1024 \
  --inference-wait-ms 2 \
  --steps 100 \
  --batch 512 \
  --buffer 200000 \
  --save-every 50 \
  --keep-checkpoint-window 500 \
  --keep-checkpoints 10 \
  --eval-every 300 \
  --eval-games 20 \
  --eval-simulations 512 \
  --augment-symmetry \
  --seed 42 \
  2>&1 | tee -a "$OUTPUT/train.log"

result=${PIPESTATUS[0]}
echo "Training exited with status ${result}." | tee -a "$OUTPUT/train.log"
exit "$result"
