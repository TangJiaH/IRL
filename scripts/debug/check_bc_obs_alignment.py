#!/usr/bin/env python
import argparse
from typing import Tuple

import numpy as np

from algorithms.bc.tacview_bc_dataset import TacviewBCConfig, TacviewBCDataset
from envs.JSBSim.envs import SingleControlEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BC 数据集与 JSBSim 观测对齐检查（最小排查脚本）。")
    parser.add_argument("--expert-path", type=str, required=True,
                        help="ACMI/CSV 专家数据路径（目录或单个文件）。")
    parser.add_argument("--env-config", type=str, default="1/heading",
                        help="JSBSim 配置名（用于 SingleControlEnv）。")
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="采样的最大样本数（用于统计数据集分布）。")
    parser.add_argument("--env-steps", type=int, default=500,
                        help="环境采样步数。")
    parser.add_argument("--seed", type=int, default=1,
                        help="随机种子。")
    return parser.parse_args()


def _stats(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.min(arr, axis=0), np.mean(arr, axis=0), np.max(arr, axis=0)


def _format_stats(title: str, stats: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    min_v, mean_v, max_v = stats
    print(f"\n[{title}]")
    for idx in range(len(min_v)):
        print(f"  dim {idx:02d}: min={min_v[idx]: .4f} mean={mean_v[idx]: .4f} max={max_v[idx]: .4f}")


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    config = TacviewBCConfig()
    dataset = TacviewBCDataset(args.expert_path, config=config)
    if len(dataset) == 0:
        raise ValueError(f"No training samples found in {args.expert_path}")

    sample_count = min(args.num_samples, len(dataset))
    indices = rng.choice(len(dataset), size=sample_count, replace=False)
    obs_samples = np.stack([dataset[i][0].numpy() for i in indices], axis=0)
    act_samples = np.stack([dataset[i][1].numpy() for i in indices], axis=0)

    _format_stats("BC 数据集 obs (已归一化)", _stats(obs_samples))
    _format_stats("BC 数据集 action (离散索引)", _stats(act_samples))

    env = SingleControlEnv(args.env_config)
    env.seed(args.seed)
    obs = env.reset()
    env_obs = []
    for _ in range(args.env_steps):
        action = np.array([[rng.randint(0, n) for n in env.action_space.nvec]])
        obs, _, done, _ = env.step(action)
        env_obs.append(obs[0])
        if done.any():
            obs = env.reset()
    env.close()

    env_obs = np.stack(env_obs, axis=0)
    _format_stats("JSBSim 环境 obs (已归一化)", _stats(env_obs))

    print("\n提示：若同一维度在数据集与环境中分布差异很大（例如速度分量符号/尺度），")
    print("可能存在坐标系或归一化不一致，这会导致评估时策略输出异常（如持续下坠）。")


if __name__ == "__main__":
    main()
