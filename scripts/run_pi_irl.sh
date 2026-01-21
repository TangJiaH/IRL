#!/bin/sh
script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(cd "${script_dir}/.." && pwd)

expert_path="${root_dir}/tacviewDataSet"
env_config="1/heading"
sample_episodes=10
max_steps=100
learning_rate=0.1
epochs=50
temperature=1.0
l2_reg=0.0

echo "expert_path=${expert_path}, env_config=${env_config}, sample_episodes=${sample_episodes}, max_steps=${max_steps}, learning_rate=${learning_rate}, epochs=${epochs}, temperature=${temperature}, l2_reg=${l2_reg}"

cd "${root_dir}"
python -m scripts.train.train_pi_irl \
    --expert-path "${expert_path}" \
    --env-config "${env_config}" \
    --sample-episodes "${sample_episodes}" \
    --max-steps "${max_steps}" \
    --learning-rate "${learning_rate}" \
    --epochs "${epochs}" \
    --temperature "${temperature}" \
    --l2-reg "${l2_reg}"
