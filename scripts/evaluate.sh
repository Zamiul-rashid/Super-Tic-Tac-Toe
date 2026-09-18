#!/usr/bin/env bash
# Stable entry point for checkpoint evaluation.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/evaluate.sh --checkpoint PATH --output DIR [options]

Modes:
  --mode full            Grand championship followed by a budget sweep (default)
  --mode championship    Grand championship only
  --mode sweep           Search-budget sweep only; d10 by default
  --mode compare         Candidate/reference comparison; requires --reference
  --mode budget-compare  Matched-opening search-budget comparison; requires --opponent

Common options:
  --checkpoint PATH
  --output DIR
  --python PATH                 Default: $STTT_PY or python
  --simulations N               Default: 512
  --games N                     Games per sweep budget; default: 20
  --championship-games N        Games per matchup; default: 20
  --championship-opponents "SPECS"  Space-separated bot specs; replaces default pool
  --budgets "512 1024 2000"     Default: 512 1024 2000
  --opponent-depth N            Default: 10
  --opponent-nodes N            Default: 50000000
  --backend auto|python|cpp     Default: auto
  --device cpu|cuda             Default: cpu
  --leaf-batch N                Default: 16
  --seed N                      Default: 4242
  --force                       Allow writing into a non-empty output directory

Compare-mode options:
  --reference PATH
  --h2h-games N                 Default: 50
  --round-robin-games N         Default: 20

Budget-compare-mode options:
  --opponent SPEC                bot spec, e.g. utttai:128 (required)
  --workers N                    Parallel game processes; default: 1
  --pairs N                      opening pairs, both colours each; default: 50
  --opening-plies N              random opening plies per pair; default: 2
  --openings-file PATH           save/reuse the opening corpus; default: OUTPUT/openings.json
  --save-moves / --no-save-moves Record full move sequences + per-move latency; default: on
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

PY=${STTT_PY:-python}
MODE=full
CHECKPOINT=""
REFERENCE=""
OUTPUT=""
SIMULATIONS=512
GAMES=20
CHAMPIONSHIP_GAMES=20
CHAMPIONSHIP_OPPONENTS=()
BUDGETS=(512 1024 2000)
OPPONENT_DEPTH=10
OPPONENT_NODES=50000000
BACKEND=auto
DEVICE=cpu
LEAF_BATCH=16
SEED=4242
H2H_GAMES=50
ROUND_ROBIN_GAMES=20
FORCE=0
OPPONENT=""
PAIRS=50
OPENING_PLIES=2
OPENINGS_FILE=""
SAVE_MOVES=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE=${2:?missing value for --mode}; shift 2 ;;
    --checkpoint) CHECKPOINT=${2:?missing value for --checkpoint}; shift 2 ;;
    --reference) REFERENCE=${2:?missing value for --reference}; shift 2 ;;
    --output) OUTPUT=${2:?missing value for --output}; shift 2 ;;
    --python) PY=${2:?missing value for --python}; shift 2 ;;
    --simulations) SIMULATIONS=${2:?missing value for --simulations}; shift 2 ;;
    --games) GAMES=${2:?missing value for --games}; shift 2 ;;
    --championship-games) CHAMPIONSHIP_GAMES=${2:?missing value for --championship-games}; shift 2 ;;
    --championship-opponents) read -r -a CHAMPIONSHIP_OPPONENTS <<< "${2:?missing value for --championship-opponents}"; shift 2 ;;
    --budgets) read -r -a BUDGETS <<< "${2:?missing value for --budgets}"; shift 2 ;;
    --opponent-depth) OPPONENT_DEPTH=${2:?missing value for --opponent-depth}; shift 2 ;;
    --opponent-nodes) OPPONENT_NODES=${2:?missing value for --opponent-nodes}; shift 2 ;;
    --backend) BACKEND=${2:?missing value for --backend}; shift 2 ;;
    --device) DEVICE=${2:?missing value for --device}; shift 2 ;;
    --leaf-batch) LEAF_BATCH=${2:?missing value for --leaf-batch}; shift 2 ;;
    --seed) SEED=${2:?missing value for --seed}; shift 2 ;;
    --h2h-games) H2H_GAMES=${2:?missing value for --h2h-games}; shift 2 ;;
    --round-robin-games) ROUND_ROBIN_GAMES=${2:?missing value for --round-robin-games}; shift 2 ;;
    --opponent) OPPONENT=${2:?missing value for --opponent}; shift 2 ;;
    --pairs) PAIRS=${2:?missing value for --pairs}; shift 2 ;;
    --opening-plies) OPENING_PLIES=${2:?missing value for --opening-plies}; shift 2 ;;
    --openings-file) OPENINGS_FILE=${2:?missing value for --openings-file}; shift 2 ;;
    --workers) WORKERS=${2:?missing value for --workers}; shift 2 ;;
    --save-moves) SAVE_MOVES=1; shift ;;
    --no-save-moves) SAVE_MOVES=0; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$CHECKPOINT" ]] || { echo "--checkpoint is required" >&2; exit 2; }
[[ -f "$CHECKPOINT" ]] || { echo "checkpoint not found: $CHECKPOINT" >&2; exit 2; }
[[ -n "$OUTPUT" ]] || { echo "--output is required" >&2; exit 2; }
if [[ -d "$OUTPUT" && -n $(find "$OUTPUT" -mindepth 1 -maxdepth 1 -print -quit) && $FORCE -eq 0 ]]; then
  echo "$OUTPUT is not empty; choose a new output directory or pass --force" >&2
  exit 2
fi

case "$MODE" in
  full|championship|sweep)
    COMMAND=(
      "$PY" -u scripts/evaluation_suite.py
      --checkpoint "$CHECKPOINT"
      --output "$OUTPUT"
      --championship-games "$CHAMPIONSHIP_GAMES"
      --simulations "$SIMULATIONS"
      --budgets "${BUDGETS[@]}"
      --games "$GAMES"
      --opponent-depth "$OPPONENT_DEPTH"
      --opponent-nodes "$OPPONENT_NODES"
      --backend "$BACKEND"
      --device "$DEVICE"
      --leaf-batch "$LEAF_BATCH"
      --seed "$SEED"
    )
    if [[ $MODE == championship ]]; then COMMAND+=(--skip-sweep); fi
    if [[ $MODE == sweep ]]; then COMMAND+=(--skip-championship); fi
    if [[ ${#CHAMPIONSHIP_OPPONENTS[@]} -gt 0 ]]; then
      COMMAND+=(--championship-opponents "${CHAMPIONSHIP_OPPONENTS[@]}")
    fi
    ;;
  compare)
    [[ -n "$REFERENCE" ]] || { echo "--reference is required in compare mode" >&2; exit 2; }
    [[ -f "$REFERENCE" ]] || { echo "reference checkpoint not found: $REFERENCE" >&2; exit 2; }
    COMMAND=(
      "$PY" -u scripts/compare_checkpoints.py
      --checkpoint "$CHECKPOINT"
      --reference "$REFERENCE"
      --output "$OUTPUT"
      --h2h-games "$H2H_GAMES"
      --round-robin-games "$ROUND_ROBIN_GAMES"
      --simulations "$SIMULATIONS"
      --seed "$SEED"
    )
    ;;
  budget-compare)
    [[ -n "$OPPONENT" ]] || { echo "--opponent is required in budget-compare mode" >&2; exit 2; }
    COMMAND=(
      "$PY" -u scripts/evaluation_suite.py
      --checkpoint "$CHECKPOINT"
      --output "$OUTPUT"
      --budget-compare
      --budgets "${BUDGETS[@]}"
      --opponent "$OPPONENT"
      --pairs "$PAIRS"
      --opening-plies "$OPENING_PLIES"
      --seed "$SEED"
      --backend "$BACKEND"
      --device "$DEVICE"
      --leaf-batch "$LEAF_BATCH"
    )
    if [[ -n "$OPENINGS_FILE" ]]; then COMMAND+=(--openings-file "$OPENINGS_FILE"); fi
    if [[ -n "${WORKERS:-}" ]]; then COMMAND+=(--workers "$WORKERS"); fi
    if [[ $SAVE_MOVES -eq 0 ]]; then COMMAND+=(--no-save-moves); fi
    ;;
  *) echo "invalid --mode: $MODE" >&2; usage >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT"
printf 'Evaluation command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
"${COMMAND[@]}" 2>&1 | tee -a "$OUTPUT/evaluation.log"
exit "${PIPESTATUS[0]}"
