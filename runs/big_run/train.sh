#!/usr/bin/env bash
set -uo pipefail
cd /home/entropy/Code/Super-Tic-Tac-Toe

if [[ ! -f engines/registry.json ]]; then
  echo "Missing engines/registry.json; copy engines/registry.example.json and configure the uttt.ai wrapper." >&2
  exit 2
fi

/home/entropy/miniconda3/envs/sttt/bin/python -u -m sttt.ai train \
  --resume runs/big_run/latest.pt \
  --output runs/big_run \
  --device cuda \
  --workers 8 \
  --iterations 5000 \
  --games 16 \
  --simulations 512 \
  --batch 256 \
  --buffer 150000 \
  --steps 100 \
  --leaf-batch 16 \
  --inference-batch 128 \
  --inference-wait-ms 2 \
  --save-every 100 \
  --eval-every 250 \
  --eval-games 20 \
  --eval-simulations 512 \
  --seed 42 \
  --population \
  --population-checkpoints runs/big_run \
  --engine-config engines/registry.json \
  --augment-symmetry \
  2>&1 | tee -a runs/big_run/train.log

result=${PIPESTATUS[0]}
echo "Training exited with status ${result}." | tee -a runs/big_run/train.log
exit "$result"
