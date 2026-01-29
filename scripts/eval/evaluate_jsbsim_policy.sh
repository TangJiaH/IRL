#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL_DIR="${MODEL_DIR:-}"
ENV_NAME="${ENV_NAME:-SingleControl}"
SCENARIO_NAME="${SCENARIO_NAME:-1/heading}"
EPISODES="${EPISODES:-20}"
MAX_STEPS="${MAX_STEPS:-1000}"
OUTPUT_CSV="${OUTPUT_CSV:-}"

usage() {
  cat <<'EOF'
Usage: scripts/eval/evaluate_jsbsim_policy.sh --model-dir <path> [options]

Options:
  --model-dir <path>     Directory containing actor_latest.pt and critic_latest.pt.
  --env-name <name>      SingleControl/SingleCombat/MultipleCombat (default: SingleControl).
  --scenario-name <name> Scenario name under envs/JSBSim/configs (default: 1/heading).
  --episodes <num>       Number of evaluation episodes (default: 20).
  --max-steps <num>      Max steps per episode (default: 1000).
  --output-csv <path>    Optional CSV output file.
  -h, --help             Show this help.

Environment variable overrides:
  MODEL_DIR, ENV_NAME, SCENARIO_NAME, EPISODES, MAX_STEPS, OUTPUT_CSV
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --scenario-name)
      SCENARIO_NAME="$2"
      shift 2
      ;;
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --output-csv)
      OUTPUT_CSV="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${MODEL_DIR}" ]]; then
  echo "Error: --model-dir is required." >&2
  usage >&2
  exit 1
fi

ARGS=(
  --model-dir "${MODEL_DIR}"
  --env-name "${ENV_NAME}"
  --scenario-name "${SCENARIO_NAME}"
  --episodes "${EPISODES}"
  --max-steps "${MAX_STEPS}"
)

if [[ -n "${OUTPUT_CSV}" ]]; then
  ARGS+=(--output-csv "${OUTPUT_CSV}")
fi

python "${ROOT_DIR}/scripts/eval/evaluate_jsbsim_policy.py" "${ARGS[@]}" "${EXTRA_ARGS[@]}"
