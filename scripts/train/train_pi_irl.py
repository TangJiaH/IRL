#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import yaml

from algorithms.irl.maxent_irl import RewardScales, load_acmi_trajectories, sample_env_trajectories
from algorithms.irl.path_integral_irl import PathIntegralIRL
from algorithms.irl.path_integral_irl_nn import PathIntegralIRLWithDeepLearning
from envs.JSBSim.envs import SingleControlEnv
from envs.JSBSim.utils.utils import get_root_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PI-IRL for HeadingReward weights.")
    parser.add_argument("--expert-path", type=str, default="dataset/A05",
                        help="专家轨迹数据集路径（目录或单个 .acmi 文件）。")
    parser.add_argument("--env-config", type=str, default="1/heading",
                        help="JSBSim 环境配置名（相对于 envs/JSBSim/configs）。")
    parser.add_argument("--sample-episodes", type=int, default=50,
                        help="用于估计模型期望的随机轨迹数量。")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="采样轨迹最大步数，默认使用环境 max_steps。")
    parser.add_argument("--learning-rate", type=float, default=0.1,
                        help="PI-IRL 学习率。")
    parser.add_argument("--epochs", type=int, default=100,
                        help="PI-IRL 迭代轮数。")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Path Integral 温度系数。")
    parser.add_argument("--l2-reg", type=float, default=0.01,
                        help="L2 正则系数。")
    normalize_group = parser.add_mutually_exclusive_group()
    normalize_group.add_argument("--normalize-returns", dest="normalize_returns", action="store_true",
                                 help="对路径积分回报进行标准化以提升数值稳定性。")
    normalize_group.add_argument("--no-normalize-returns", dest="normalize_returns", action="store_false",
                                 help="禁用路径积分回报标准化。")
    parser.set_defaults(normalize_returns=True)
    parser.add_argument("--uniform-mix", type=float, default=0.05,
                        help="与均匀分布混合的概率，用于稳定重要性采样。")
    parser.add_argument("--resample-count", type=int, default=None,
                        help="重要性采样重采样数量（用于降低蒙特卡洛估计方差）。")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，用于采样与重要性重采样。")
    parser.add_argument("--replay-buffer-size", type=int, default=None,
                        help="轨迹回放缓冲区大小（用于平滑训练）。")
    parser.add_argument("--replay-batch-size", type=int, default=None,
                        help="每轮从回放缓冲区采样的轨迹数量。")
    parser.add_argument("--temperature-decay", type=float, default=0.98,
                        help="温度系数衰减系数（每轮乘以该值）。")
    parser.add_argument("--min-temperature", type=float, default=0.1,
                        help="温度系数下限。")
    parser.add_argument("--optimizer", type=str, default="adam", choices=("adam", "sgd"),
                        help="优化器类型。")
    parser.add_argument("--lr-decay", type=float, default=0.99,
                        help="学习率衰减系数（每轮乘以该值）。")
    parser.add_argument("--adam-beta1", type=float, default=0.9,
                        help="Adam 一阶动量系数。")
    parser.add_argument("--adam-beta2", type=float, default=0.999,
                        help="Adam 二阶动量系数。")
    parser.add_argument("--adam-eps", type=float, default=1e-8,
                        help="Adam 数值稳定项。")
    parser.add_argument("--ensemble-runs", type=int, default=5,
                        help="多次训练并集成的次数。")
    parser.add_argument("--ensemble-seed-offset", type=int, default=1000,
                        help="集成训练的随机种子偏移量。")
    parser.add_argument("--use-reward-network", action="store_true",
                        help="使用神经网络奖励函数（需要安装 PyTorch）。")
    parser.add_argument("--reward-hidden-sizes", type=int, nargs="*", default=[32, 32],
                        help="奖励网络隐藏层大小列表。")
    parser.add_argument("--reward-lr", type=float, default=0.001,
                        help="奖励网络优化器学习率。")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="奖励网络梯度裁剪阈值。")
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
        raise ValueError("未采样到随机轨迹，无法进行 PI-IRL。")

    resample_count = args.resample_count
    if resample_count is None:
        resample_count = max(1, args.sample_episodes)

    replay_buffer_size = args.replay_buffer_size
    if replay_buffer_size is None:
        replay_buffer_size = max(1, len(sampled_trajectories))

    replay_batch_size = args.replay_batch_size
    if replay_batch_size is None:
        replay_batch_size = max(1, min(len(sampled_trajectories), args.sample_episodes))

    if args.use_reward_network:
        import torch

        def _build_reward_network(input_dim: int):
            layers = []
            in_dim = input_dim
            for hidden_dim in args.reward_hidden_sizes:
                layers.append(torch.nn.Linear(in_dim, hidden_dim))
                layers.append(torch.nn.ReLU())
                in_dim = hidden_dim
            layers.append(torch.nn.Linear(in_dim, 1))
            return torch.nn.Sequential(*layers)

        reward_network = _build_reward_network(expert_trajectories[0].shape[-1])
        irl = PathIntegralIRLWithDeepLearning(
            reward_network=reward_network,
            learning_rate=args.reward_lr,
            epochs=args.epochs,
            temperature=args.temperature,
            l2_reg=args.l2_reg,
            normalize_returns=args.normalize_returns,
            uniform_mix=args.uniform_mix,
            resample_count=resample_count,
            seed=args.seed,
            replay_buffer_size=replay_buffer_size,
            replay_batch_size=replay_batch_size,
            temperature_decay=args.temperature_decay,
            min_temperature=args.min_temperature,
            grad_clip=args.grad_clip,
        )
        result = irl.fit(expert_trajectories, sampled_trajectories)
        weights = None
    else:
        irl = PathIntegralIRL(
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            temperature=args.temperature,
            l2_reg=args.l2_reg,
            normalize_returns=args.normalize_returns,
            uniform_mix=args.uniform_mix,
            resample_count=resample_count,
            seed=args.seed,
            replay_buffer_size=replay_buffer_size,
            replay_batch_size=replay_batch_size,
            temperature_decay=args.temperature_decay,
            min_temperature=args.min_temperature,
            optimizer=args.optimizer,
            lr_decay=args.lr_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_eps=args.adam_eps,
            ensemble_runs=args.ensemble_runs,
            ensemble_seed_offset=args.ensemble_seed_offset,
        )
        result = irl.fit(expert_trajectories, sampled_trajectories)
        weights = result["weights"].tolist()

    if weights is not None:
        print("PI-IRL 权重:", weights)
    else:
        print("PI-IRL 已完成奖励网络训练。")

    if args.output_json and weights is not None:
        output_path = Path(args.output_json)
        output_path.write_text(json.dumps({"HeadingReward_weights": weights}, indent=2), encoding="utf-8")

    if args.write_config and weights is not None:
        config_path = Path(get_root_dir()) / "configs" / f"{args.env_config}.yaml"
        config_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config_data["HeadingReward_weights"] = weights
        config_path.write_text(yaml.safe_dump(config_data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"已更新配置文件: {config_path}")

    env.close()


if __name__ == "__main__":
    main()
