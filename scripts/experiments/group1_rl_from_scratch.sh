#!/usr/bin/env bash
set -euo pipefail

env="SingleControl"
scenario="1/heading"
algo="ppo"
exp="group1_rl_from_scratch"
seed=1

WANDB_MODE=online python scripts/train/train_jsbsim_rl.py \
  --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
  --seed ${seed} --n-training-threads 1 --n-rollout-threads 16 \
  --log-interval 1 --save-interval 1 \
  --num-mini-batch 4 --buffer-size 2000 --num-env-steps 2e7 \
  --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-param 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
  --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
  --use-wandb --use-eval --eval-interval 1
