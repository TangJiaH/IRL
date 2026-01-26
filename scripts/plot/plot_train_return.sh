#!/usr/bin/env bash
set -euo pipefail

CSV_PATH=${1:-""}
if [ -z "$CSV_PATH" ]; then
  echo "用法: bash scripts/plot/plot_train_return.sh <train_metrics.csv路径> [window] [ema] [save_path]"
  exit 1
fi

WINDOW=${2:-20}
EMA=${3:-0.0}
SAVE_PATH=${4:-""}

python scripts/plot/plot_train_return.py \
  --csv "$CSV_PATH" \
  --window "$WINDOW" \
  --ema "$EMA" \
  --save "$SAVE_PATH"
