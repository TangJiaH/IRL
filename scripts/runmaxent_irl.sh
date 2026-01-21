#!/bin/sh
script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(cd "${script_dir}/.." && pwd)

expert_path="${root_dir}/tacviewDataSet"
env_config="1/heading"
sample_episodes=50
max_steps=150
learning_rate=0.05
epochs=100
seed=42

echo "expert_path=${expert_path}, env_config=${env_config}, sample_episodes=${sample_episodes}, max_steps=${max_steps}, learning_rate=${learning_rate}, epochs=${epochs}, seed=${seed}"

cd "${root_dir}"
python -m scripts.train.train_maxent_irl \
    --expert-path "${expert_path}" \
    --env-config "${env_config}" \
    --sample-episodes "${sample_episodes}" \
    --max-steps "${max_steps}" \
    --learning-rate "${learning_rate}" \
    --epochs "${epochs}" \
    --seed "${seed}"
