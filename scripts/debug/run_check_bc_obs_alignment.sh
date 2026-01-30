#!/usr/bin/env bash
set -euo pipefail

expert_path="${1:-}"
env_config="${2:-1/heading}"
num_samples="${3:-2000}"
env_steps="${4:-500}"
seed="${5:-1}"

if [[ -z "${expert_path}" ]]; then
  echo "Usage: $0 <expert_path> [env_config] [num_samples] [env_steps] [seed]" >&2
  echo "Example: $0 generated_acmi 1/heading 2000 500 1" >&2
  exit 1
fi

python scripts/debug/check_bc_obs_alignment.py \
  --expert-path "${expert_path}" \
  --env-config "${env_config}" \
  --num-samples "${num_samples}" \
  --env-steps "${env_steps}" \
  --seed "${seed}"
