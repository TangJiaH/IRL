#!/usr/bin/env bash
set -euo pipefail

expert_path="tacviewDataSet"
output_dir="runs/bc_group2"

python scripts/train/train_bc_singlecontrol.py \
  --expert-path ${expert_path} \
  --output-dir ${output_dir} \
  --epochs 10 --batch-size 256 --bc-lr 1e-3
