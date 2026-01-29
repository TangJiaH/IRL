#!/usr/bin/env bash
set -euo pipefail

expert_path="tacviewDataSet"
output_dir="runs/bc_group5_dagger"

WANDB_MODE=online python scripts/train/train_bc_singlecontrol.py \
  --expert-path ${expert_path} \
  --output-dir ${output_dir} \
  --epochs 8 --batch-size 256 --bc-lr 1e-3 \
  --dagger-iterations 2 --dagger-episodes 4 --dagger-max-steps 300 --dagger-env-config 1/heading \
  --use-wandb --use-eval --eval-interval 1
