#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from algorithms.bc.tacview_bc_dataset import TacviewBCConfig, TacviewBCDataset
from algorithms.ppo.ppo_policy import PPOPolicy
from config import get_config
from envs.JSBSim.envs import SingleControlEnv


def parse_args() -> argparse.Namespace:
    parser = get_config()
    parser.add_argument("--expert-path", type=str, default="tacviewDataSet",
                        help="Tacview 专家数据路径（目录或单个文件）。")
    parser.add_argument("--output-dir", type=str, default="runs/bc_pretrain",
                        help="保存 actor_latest.pt 与 critic_latest.pt 的目录。")
    parser.add_argument("--epochs", type=int, default=10,
                        help="行为克隆训练轮数。")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="行为克隆批大小。")
    parser.add_argument("--bc-lr", type=float, default=1e-3,
                        help="行为克隆学习率。")
    parser.add_argument("--bc-weight-decay", type=float, default=1e-4,
                        help="行为克隆权重衰减（L2 正则）。")
    parser.add_argument("--bc-grad-clip", type=float, default=0.5,
                        help="行为克隆梯度裁剪阈值（0 表示不裁剪）。")
    parser.add_argument("--bc-lr-factor", type=float, default=0.5,
                        help="验证损失停滞时学习率衰减系数。")
    parser.add_argument("--bc-lr-patience", type=int, default=5,
                        help="验证损失停滞的轮次数后触发学习率衰减。")
    parser.add_argument("--bc-lr-min", type=float, default=1e-6,
                        help="学习率衰减的最小值。")
    parser.add_argument("--roll-rate-limit", type=float, default=1.2,
                        help="滚转角速度归一化上限（rad/s）。")
    parser.add_argument("--pitch-rate-limit", type=float, default=0.8,
                        help="俯仰角速度归一化上限（rad/s）。")
    parser.add_argument("--yaw-rate-limit", type=float, default=0.8,
                        help="偏航角速度归一化上限（rad/s）。")
    parser.add_argument("--speed-rate-limit", type=float, default=15.0,
                        help="速度变化率归一化上限（m/s^2）。")
    parser.add_argument("--stride", type=int, default=1,
                        help="轨迹采样步长。")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="可选的样本数量上限。")
    parser.add_argument("--dagger-iterations", type=int, default=0,
                        help="DAgger 聚合迭代次数。")
    parser.add_argument("--dagger-episodes", type=int, default=4,
                        help="每次 DAgger 迭代的采样回合数。")
    parser.add_argument("--dagger-max-steps", type=int, default=300,
                        help="每回合 DAgger 采样的最大步数。")
    parser.add_argument("--dagger-env-config", type=str, default="1/heading",
                        help="DAgger 采样使用的 SingleControl 配置名。")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="验证集比例（0-1）。")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="评估集比例（0-1）。")
    return parser.parse_args()


def _discretize(value: float, nvec: int, low: float, high: float) -> int:
    value = max(min(value, high), low)
    scaled = (value - low) / (high - low)
    return int(np.clip(round(scaled * (nvec - 1)), 0, nvec - 1))


def _wrap_angle_rad(angle: float) -> float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _label_from_transition(obs: np.ndarray, next_obs: np.ndarray, dt: float, config: TacviewBCConfig) -> np.ndarray:
    roll = np.arctan2(obs[4], obs[5])
    pitch = np.arctan2(obs[6], obs[7])
    roll_next = np.arctan2(next_obs[4], next_obs[5])
    pitch_next = np.arctan2(next_obs[6], next_obs[7])

    roll_rate = _wrap_angle_rad(roll_next - roll) / dt
    pitch_rate = _wrap_angle_rad(pitch_next - pitch) / dt
    yaw_rate = _wrap_angle_rad(next_obs[1] - obs[1]) / dt

    speed = obs[11] * 340.0
    speed_next = next_obs[11] * 340.0
    speed_rate = (speed_next - speed) / dt

    aileron_cmd = np.clip(roll_rate / config.roll_rate_limit, -1.0, 1.0)
    elevator_cmd = np.clip(pitch_rate / config.pitch_rate_limit, -1.0, 1.0)
    rudder_cmd = np.clip(yaw_rate / config.yaw_rate_limit, -1.0, 1.0)
    throttle_cmd = np.clip(0.65 + (speed_rate / config.speed_rate_limit) * 0.25, 0.4, 0.9)

    return np.array([
        _discretize(aileron_cmd, 41, -1.0, 1.0),
        _discretize(elevator_cmd, 41, -1.0, 1.0),
        _discretize(rudder_cmd, 41, -1.0, 1.0),
        _discretize(throttle_cmd, 30, 0.4, 0.9),
    ], dtype=np.int64)


def _collect_dagger_samples(policy: PPOPolicy, env: SingleControlEnv, episodes: int, max_steps: int,
                            config: TacviewBCConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    obs_list = []
    act_list = []
    for _ in range(episodes):
        obs = env.reset()
        rnn_states = np.zeros((obs.shape[0], policy.actor.recurrent_hidden_layers, policy.actor.recurrent_hidden_size))
        masks = np.ones((env.num_agents, 1), dtype=np.float32)
        for _ in range(max_steps):
            obs_tensor = torch.from_numpy(obs).to(device)
            rnn_tensor = torch.from_numpy(rnn_states).to(device)
            masks_tensor = torch.from_numpy(masks).to(device)
            with torch.no_grad():
                actions, _, rnn_tensor = policy.actor(obs_tensor, rnn_tensor, masks_tensor, deterministic=True)
            action_np = actions.cpu().numpy()
            next_obs, _, dones, _ = env.step(action_np)
            dt = config.min_dt
            for agent_idx in range(obs.shape[0]):
                label = _label_from_transition(obs[agent_idx], next_obs[agent_idx], dt, config)
                obs_list.append(obs[agent_idx])
                act_list.append(label)
            obs = next_obs
            rnn_states = rnn_tensor.cpu().numpy()
            if dones.any():
                break
    if not obs_list:
        return np.empty((0, 12), dtype=np.float32), np.empty((0, 4), dtype=np.int64)
    return np.stack(obs_list, axis=0), np.stack(act_list, axis=0)


def _train_bc_epoch(policy: PPOPolicy, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, args: argparse.Namespace) -> float:
    total_loss = 0.0
    total_samples = 0
    for obs_batch, action_batch in dataloader:
        obs_batch = obs_batch.to(device)
        action_batch = action_batch.to(device)
        batch_size = obs_batch.shape[0]

        rnn_states = torch.zeros((batch_size, args.recurrent_hidden_layers, args.recurrent_hidden_size), device=device)
        masks = torch.ones((batch_size, 1), device=device)

        action_log_probs, _ = policy.actor.evaluate_actions(obs_batch, rnn_states, action_batch, masks)
        loss = -action_log_probs.mean()

        optimizer.zero_grad()
        loss.backward()
        if args.bc_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), args.bc_grad_clip)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def _eval_bc_epoch(policy: PPOPolicy, dataloader: DataLoader,
                   device: torch.device, args: argparse.Namespace) -> float:
    total_loss = 0.0
    total_samples = 0
    policy.actor.eval()
    with torch.no_grad():
        for obs_batch, action_batch in dataloader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            batch_size = obs_batch.shape[0]

            rnn_states = torch.zeros((batch_size, args.recurrent_hidden_layers, args.recurrent_hidden_size), device=device)
            masks = torch.ones((batch_size, 1), device=device)

            action_log_probs, _ = policy.actor.evaluate_actions(obs_batch, rnn_states, action_batch, masks)
            loss = -action_log_probs.mean()

            total_loss += loss.item() * batch_size
            total_samples += batch_size
    policy.actor.train()
    return total_loss / max(total_samples, 1)


def _split_dataset(dataset: TacviewBCDataset, val_split: float, test_split: float,
                   seed: int) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if val_split < 0 or test_split < 0 or val_split + test_split >= 1:
        raise ValueError("val_split 和 test_split 必须在 [0,1) 且总和小于 1。")
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    test_size = int(total_size * test_split)
    train_size = total_size - val_size - test_size
    if train_size <= 0:
        raise ValueError("训练集样本数不足，请调整 val_split/test_split。")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size, test_size], generator=generator)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    config = TacviewBCConfig(
        roll_rate_limit=args.roll_rate_limit,
        pitch_rate_limit=args.pitch_rate_limit,
        yaw_rate_limit=args.yaw_rate_limit,
        speed_rate_limit=args.speed_rate_limit,
        stride=args.stride,
        max_samples=args.max_samples,
    )
    dataset = TacviewBCDataset(args.expert_path, config=config)
    if len(dataset) == 0:
        raise ValueError(f"No training samples found in {args.expert_path}")
    train_set, val_set, test_set = _split_dataset(dataset, args.val_split, args.test_split, args.seed)

    obs_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(12,))
    act_space = gym.spaces.MultiDiscrete([41, 41, 41, 30])

    policy = PPOPolicy(args, obs_space, act_space, device=device)
    policy.actor.train()
    optimizer = torch.optim.AdamW(
        policy.actor.parameters(),
        lr=args.bc_lr,
        weight_decay=args.bc_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.bc_lr_factor,
        patience=args.bc_lr_patience,
        min_lr=args.bc_lr_min,
    )

    env = None
    if args.dagger_iterations > 0:
        env = SingleControlEnv(args.dagger_env_config)

    for iteration in range(args.dagger_iterations + 1):
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False) if len(val_set) > 0 else None
        for epoch in range(args.epochs):
            avg_loss = _train_bc_epoch(policy, dataloader, optimizer, device, args)
            if val_loader is not None:
                val_loss = _eval_bc_epoch(policy, val_loader, device, args)
                scheduler.step(val_loss)
                print(f"迭代 {iteration + 1}/{args.dagger_iterations + 1} 轮次 {epoch + 1}/{args.epochs} "
                      f"- BC 损失: {avg_loss:.6f} - 验证损失: {val_loss:.6f}")
            else:
                scheduler.step(avg_loss)
                print(f"迭代 {iteration + 1}/{args.dagger_iterations + 1} 轮次 {epoch + 1}/{args.epochs} - BC 损失: {avg_loss:.6f}")

        if iteration < args.dagger_iterations:
            obs_new, actions_new = _collect_dagger_samples(
                policy, env, args.dagger_episodes, args.dagger_max_steps, config, device
            )
            dataset.add_samples(obs_new, actions_new)
            train_set, val_set, test_set = _split_dataset(dataset, args.val_split, args.test_split, args.seed)
            print(f"DAgger 聚合: 新增 {len(obs_new)} 条样本，总计 {len(dataset)}。")

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False) if len(test_set) > 0 else None
    if test_loader is not None:
        test_loss = _eval_bc_epoch(policy, test_loader, device, args)
        print(f"评估集 BC 损失: {test_loss:.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.actor.state_dict(), output_dir / "actor_latest.pt")
    torch.save(policy.critic.state_dict(), output_dir / "critic_latest.pt")

    print(f"已保存 BC actor 至 {output_dir / 'actor_latest.pt'}")
    print(f"已保存 BC critic 至 {output_dir / 'critic_latest.pt'}")


if __name__ == "__main__":
    main()
