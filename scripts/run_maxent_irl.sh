#!/bin/sh
script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(cd "${script_dir}/.." && pwd)

expert_path=${1:-dataset/A05}
env_config=${2:-1/heading}
sample_episodes=${3:-10}
max_steps=${4:-}
learning_rate=${5:-0.1}
epochs=${6:-50}

if [ -n "${max_steps}" ]; then
    max_steps_arg="--max-steps ${max_steps}"
else
    max_steps_arg=""
fi

echo "expert_path=${expert_path}, env_config=${env_config}, sample_episodes=${sample_episodes}, max_steps=${max_steps}, learning_rate=${learning_rate}, epochs=${epochs}"

python "${root_dir}/scripts/train/train_maxent_irl.py" \
    --expert-path "${expert_path}" \
    --env-config "${env_config}" \
    --sample-episodes "${sample_episodes}" \
    ${max_steps_arg} \
    --learning-rate "${learning_rate}" \
    --epochs "${epochs}"
