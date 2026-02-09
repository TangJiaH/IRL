#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(cd "${script_dir}/../.." && pwd)

output_dir="${root_dir}/generated_acmi"
episodes=1
steps=1000
dt=0.2
seed=1
target_interval=200
env_config="1/heading"
altitude_buffer=200.0

echo "output_dir=${output_dir}, episodes=${episodes}, steps=${steps}, dt=${dt}, seed=${seed}, target_interval=${target_interval}, env_config=${env_config}, altitude_buffer=${altitude_buffer}"

python "${root_dir}/scripts/data/generate_classic_compare.py" \
  --output-dir "${output_dir}" \
  --episodes "${episodes}" \
  --steps "${steps}" \
  --dt "${dt}" \
  --seed "${seed}" \
  --target-interval "${target_interval}" \
  --env-config "${env_config}" \
  --altitude-buffer "${altitude_buffer}"
