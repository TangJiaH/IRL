#!/usr/bin/env bash
set -euo pipefail

expert_path="tacviewDataSet"
output_dir="runs/bc_group2"

WANDB_MODE=online python scripts/train/train_bc_singlecontrol.py \
  --expert-path ${expert_path} \
  --output-dir ${output_dir} \
  --epochs 30 --batch-size 256 --bc-lr 5e-4 \
  --bc-weight-decay 1e-4 --bc-grad-clip 0.5 \
  --bc-lr-factor 0.5 --bc-lr-patience 3 --bc-lr-min 1e-5 \
  --use-wandb --val-split 0.1 --test-split 0.1
