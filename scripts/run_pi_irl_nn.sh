#!/bin/sh
script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(cd "${script_dir}/.." && pwd)

expert_path="${root_dir}/tacviewDataSet"
env_config="1/heading"
sample_episodes=50
max_steps=100
epochs=100
temperature=1.0
l2_reg=0.01
reward_lr=0.001
reward_hidden_sizes="32 32"
grad_clip=1.0

echo "expert_path=${expert_path}, env_config=${env_config}, sample_episodes=${sample_episodes}, max_steps=${max_steps}, epochs=${epochs}, temperature=${temperature}, l2_reg=${l2_reg}, reward_lr=${reward_lr}, reward_hidden_sizes=${reward_hidden_sizes}, grad_clip=${grad_clip}"

cd "${root_dir}"
python -m scripts.train.train_pi_irl \
    --expert-path "${expert_path}" \
    --env-config "${env_config}" \
    --sample-episodes "${sample_episodes}" \
    --max-steps "${max_steps}" \
    --epochs "${epochs}" \
    --temperature "${temperature}" \
    --l2-reg "${l2_reg}" \
    --use-reward-network \
    --reward-lr "${reward_lr}" \
    --reward-hidden-sizes ${reward_hidden_sizes} \
    --grad-clip "${grad_clip}"
