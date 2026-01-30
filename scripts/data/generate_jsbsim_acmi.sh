#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(cd "${script_dir}/../.." && pwd)

output_dir="${root_dir}/generated_acmi"
episodes=5
max_steps=1000
env_config="1/heading"
seed=1

echo "output_dir=${output_dir}, episodes=${episodes}, max_steps=${max_steps}, env_config=${env_config}, seed=${seed}"

python "${root_dir}/scripts/data/generate_jsbsim_acmi.py" \
  --output-dir "${output_dir}" \
  --episodes "${episodes}" \
  --max-steps "${max_steps}" \
  --env-config "${env_config}" \
  --seed "${seed}"
