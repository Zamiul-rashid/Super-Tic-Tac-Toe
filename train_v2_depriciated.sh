#!/usr/bin/env bash
# DEPRECATED. train_v2.sh hard-coded another machine's paths, a rolling
# latest.pt resume and a changing Elo label. The canonical launcher is
# ./train.sh <checkpoint> <output-dir> [population-config.json]; see
# TRAINING_READINESS_PLAN.md §M11.
echo "train_v2.sh is deprecated; use ./train.sh <checkpoint> <output-dir> [population-config.json]" >&2
exit 2
