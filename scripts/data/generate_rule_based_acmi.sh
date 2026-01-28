#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(cd "${script_dir}/../.." && pwd)

output_dir="${root_dir}/generated_acmi"
episodes=5
steps=1000
dt=0.2
seed=1
target_interval=200

echo "output_dir=${output_dir}, episodes=${episodes}, steps=${steps}, dt=${dt}, seed=${seed}, target_interval=${target_interval}"

python "${root_dir}/scripts/data/generate_rule_based_acmi.py" \
  --output-dir "${output_dir}" \
  --episodes "${episodes}" \
  --steps "${steps}" \
  --dt "${dt}" \
  --seed "${seed}" \
  --target-interval "${target_interval}"
