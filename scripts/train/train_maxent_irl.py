#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import yaml

from algorithms.irl.maxent_irl import MaxEntIRL, RewardScales, load_acmi_trajectories, sample_env_trajectories
from envs.JSBSim.envs import SingleControlEnv
from envs.JSBSim.utils.utils import get_root_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MaxEnt IRL for HeadingReward weights.")
    parser.add_argument("--expert-path", type=str, default="dataset/A05",
                        help="专家轨迹数据集路径（目录或单个 .acmi 文件）。")
    parser.add_argument("--env-config", type=str, default="1/heading",
                        help="JSBSim 环境配置名（相对于 envs/JSBSim/configs）。")
    parser.add_argument("--sample-episodes", type=int, default=10,
                        help="用于估计模型期望的随机轨迹数量。")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="采样轨迹最大步数，默认使用环境 max_steps。")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="MaxEnt IRL 学习率。")
    parser.add_argument("--epochs", type=int, default=50,
                        help="MaxEnt IRL 迭代轮数。")
    parser.add_argument("--write-config", action="store_true",
                        help="将学习到的权重写回 JSBSim 配置文件。")
    parser.add_argument("--output-json", type=str, default=None,
                        help="输出权重到指定 JSON 文件。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scales = RewardScales()

    expert_trajectories = load_acmi_trajectories(args.expert_path, scales=scales)
    if not expert_trajectories:
        raise ValueError(f"未找到专家轨迹数据，请检查路径: {args.expert_path}")

    env = SingleControlEnv(args.env_config)
    sampled_trajectories = sample_env_trajectories(
        env, num_episodes=args.sample_episodes, max_steps=args.max_steps, scales=scales
    )
    if not sampled_trajectories:
        raise ValueError("未采样到随机轨迹，无法进行 MaxEnt IRL。")

    irl = MaxEntIRL(learning_rate=args.learning_rate, epochs=args.epochs)
    result = irl.fit(expert_trajectories, sampled_trajectories)

    weights = result["weights"].tolist()
    print("MaxEnt IRL 权重:", weights)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps({"HeadingReward_weights": weights}, indent=2), encoding="utf-8")

    if args.write_config:
        config_path = Path(get_root_dir()) / "configs" / f"{args.env_config}.yaml"
        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config_data["HeadingReward_weights"] = weights
        config_path.write_text(yaml.safe_dump(config_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"已更新配置文件: {config_path}")

    env.close()


if __name__ == "__main__":
    main()
