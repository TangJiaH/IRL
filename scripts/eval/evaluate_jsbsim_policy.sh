#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# 直接在此处修改配置，无需命令行参数。
model_dir="runs/example_model"
env_name="SingleControl"
scenario_name="1/heading"
episodes=20
max_steps=1000
output_csv=""

args=(
  --model-dir "${model_dir}"
  --env-name "${env_name}"
  --scenario-name "${scenario_name}"
  --episodes "${episodes}"
  --max-steps "${max_steps}"
)

if [[ -n "${output_csv}" ]]; then
  args+=(--output-csv "${output_csv}")
fi

PYTHONPATH="${ROOT_DIR}" python "${ROOT_DIR}/scripts/eval/evaluate_jsbsim_policy.py" "${args[@]}"
