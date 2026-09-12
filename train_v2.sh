#!/usr/bin/env bash
set -uo pipefail
cd /home/entropy/Code/Super-Tic-Tac-Toe

mkdir -p runs/run_v2

if [[ ! -f engines/registry.json ]]; then
  echo "Missing engines/registry.json; copy engines/registry.example.json and configure the uttt.ai wrapper." >&2
  exit 2
fi

echo "=========================================================================="
echo "          LAUNCHING RUN_V2: CLEAN SLATE ELITE TRAINING RUN               "
echo "=========================================================================="
echo "Config: Fresh ResNet (~1.8M params), FP16 AMP, C++ Bitboard Engine & MCTS"
echo "Population: 25% utttai (2320 Elo), 25% Alpha-Beta (d3-d8), 30% Self-Play"
echo "Buffer: 200,000 | Batch: 512 | Simulations: 512 | Workers: 8"
echo "=========================================================================="

/home/entropy/Code/Super-Tic-Tac-Toe/.venv/bin/python -u -m sttt.ai train \
  --output runs/run_v2 \
  --device cuda \
  --backend cpp \
  --fp16 \
  --workers 8 \
  --iterations 3000 \
  --games 16 \
  --simulations 512 \
  --batch 512 \
  --buffer 200000 \
  --steps 100 \
  --leaf-batch 16 \
  --inference-batch 512 \
  --inference-wait-ms 2 \
  --save-every 50 \
  --eval-every 100 \
  --eval-games 20 \
  --eval-simulations 512 \
  --seed 42 \
  --population \
  --population-checkpoints runs/run_v2 \
  --engine-config engines/registry.json \
  --augment-symmetry \
  2>&1 | tee -a runs/run_v2/train.log

result=${PIPESTATUS[0]}
echo "Training exited with status ${result}." | tee -a runs/run_v2/train.log
exit "$result"
