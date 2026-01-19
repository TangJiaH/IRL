#!/usr/bin/env bash
set -euo pipefail

# Example script to run MaxEnt IRL training for HeadingReward weights.
# Usage:
#   ./scripts/run_maxent_irl.sh [EXPERT_PATH] [ENV_CONFIG] [SAMPLE_EPISODES] [MAX_STEPS] [LEARNING_RATE] [EPOCHS]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXPERT_PATH="${1:-dataset/A05}"
ENV_CONFIG="${2:-1/heading}"
SAMPLE_EPISODES="${3:-10}"
MAX_STEPS="${4:-}"
LEARNING_RATE="${5:-0.1}"
EPOCHS="${6:-50}"

MAX_STEPS_ARG=()
if [[ -n "${MAX_STEPS}" ]]; then
  MAX_STEPS_ARG=("--max-steps" "${MAX_STEPS}")
fi

python "${ROOT_DIR}/scripts/train/train_maxent_irl.py" \
  --expert-path "${EXPERT_PATH}" \
  --env-config "${ENV_CONFIG}" \
  --sample-episodes "${SAMPLE_EPISODES}" \
  "${MAX_STEPS_ARG[@]}" \
  --learning-rate "${LEARNING_RATE}" \
  --epochs "${EPOCHS}"
