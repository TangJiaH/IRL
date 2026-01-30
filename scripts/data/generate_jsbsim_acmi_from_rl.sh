#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(cd "${script_dir}/../.." && pwd)

model_dir="${1:-}"
output_dir="${2:-${root_dir}/generated_acmi}"
episodes="${3:-5}"
max_steps="${4:-1000}"
env_config="${5:-1/heading}"
seed="${6:-1}"

if [[ -z "${model_dir}" ]]; then
  echo "Usage: $0 <model_dir> [output_dir] [episodes] [max_steps] [env_config] [seed]" >&2
  exit 1
fi

echo "model_dir=${model_dir}, output_dir=${output_dir}, episodes=${episodes}, max_steps=${max_steps}, env_config=${env_config}, seed=${seed}"

python "${root_dir}/scripts/data/generate_jsbsim_acmi_from_rl.py" \
  --model-dir "${model_dir}" \
  --output-dir "${output_dir}" \
  --episodes "${episodes}" \
  --max-steps "${max_steps}" \
  --env-config "${env_config}" \
  --seed "${seed}"
