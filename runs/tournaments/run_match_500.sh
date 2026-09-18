#!/usr/bin/env bash
set -uo pipefail
cd /home/entropy/Code/Super-Tic-Tac-Toe

/home/entropy/miniconda3/envs/sttt/bin/python -u -m sttt.ai tournament \
  --checkpoint runs/big_run/latest.pt \
  --opponents alphabeta \
  --games 500 \
  --simulations 512 \
  --device cuda \
  --seed 42 \
  2>&1 | tee runs/tournaments/tournament_500_alphabeta.log

