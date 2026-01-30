#!/usr/bin/env bash
set -euo pipefail

expert_path="generated_acmi"
output_dir="runs/bc_group5_dagger"

WANDB_MODE=online python scripts/train/train_bc_singlecontrol.py \
  --expert-path ${expert_path} \
  --output-dir ${output_dir} \
  --epochs 50 --batch-size 256 --bc-lr 1e-3 \
  --dagger-iterations 2 --dagger-episodes 4 --dagger-max-steps 300 --dagger-env-config 1/heading \
  --use-wandb --val-split 0.1 --test-split 0.1
