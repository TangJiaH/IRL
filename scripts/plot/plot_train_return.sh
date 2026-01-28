#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "$0")" && pwd)
root_dir=$(cd "${script_dir}/../.." && pwd)

csv_path="${root_dir}/results/SingleControl/1/heading/ppo/exp/run1/logs/train_metrics.csv"
window=20
ema=0.0
save_path=""

echo "csv_path=${csv_path}, window=${window}, ema=${ema}, save_path=${save_path}"

python "${root_dir}/scripts/plot/plot_train_return.py" \
  --csv "${csv_path}" \
  --window "${window}" \
  --ema "${ema}" \
  --save "${save_path}"
