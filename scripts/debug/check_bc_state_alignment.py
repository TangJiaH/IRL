#!/usr/bin/env python
import argparse
from typing import List, Tuple

import numpy as np

from algorithms.bc.tacview_bc_dataset import TacviewBCConfig, TacviewBCDataset
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 ACMI 注释状态与 BC 数据集状态是否一致。")
    parser.add_argument("--expert-path", type=str, required=True,
                        help="ACMI 专家数据路径（目录或单个文件）。")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="最多检查的样本数量。")
    return parser.parse_args()


def _parse_state_comments(path: str) -> List[np.ndarray]:
    states: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("//STATE"):
                continue
            payload = line[len("//STATE"):].strip()
            if payload.startswith(":") or payload.startswith("="):
                payload = payload[1:].strip()
            if payload.startswith("obs="):
                payload = payload[len("obs="):]
            values = [float(v) for v in payload.replace(",", " ").split()]
            if len(values) != 12:
                continue
            states.append(np.array(values, dtype=np.float32))
    return states


def main() -> None:
    args = parse_args()
    dataset = TacviewBCDataset(args.expert_path, config=TacviewBCConfig())
    if len(dataset) == 0:
        raise ValueError(f"No training samples found in {args.expert_path}")

    base = Path(args.expert_path)
    if base.is_file():
        acmi_files = [base]
    else:
        acmi_files = sorted(base.glob("*.acmi"))
        if not acmi_files:
            acmi_files = sorted(base.rglob("*.acmi"))
    if not acmi_files:
        raise ValueError(f"No ACMI files found in {args.expert_path}")

    total_samples = 0
    diffs: List[float] = []
    for file_path in acmi_files:
        if file_path.suffix.lower() != ".acmi":
            continue
        state_obs = _parse_state_comments(str(file_path))
        if not state_obs:
            continue
        for idx, state in enumerate(state_obs):
            if total_samples >= args.max_samples:
                break
            if idx >= len(dataset):
                break
            ds_obs, _ = dataset[idx]
            diffs.append(float(np.max(np.abs(ds_obs.numpy() - state))))
            total_samples += 1
        if total_samples >= args.max_samples:
            break

    if not diffs:
        raise ValueError("未找到可比较的 //STATE 注释或样本不足。")

    print(f"checked_samples={len(diffs)}")
    print(f"max_abs_diff_max={np.max(diffs):.6f}")
    print(f"max_abs_diff_mean={np.mean(diffs):.6f}")


if __name__ == "__main__":
    main()
