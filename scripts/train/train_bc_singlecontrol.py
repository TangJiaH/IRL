#!/usr/bin/env python
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import DataLoader

from algorithms.bc.tacview_bc_dataset import TacviewBCConfig, TacviewBCDataset
from algorithms.ppo.ppo_policy import PPOPolicy
from config import get_config


def parse_args() -> argparse.Namespace:
    parser = get_config()
    parser.add_argument("--expert-path", type=str, default="tacviewDataSet",
                        help="Tacview expert data path (directory or file).")
    parser.add_argument("--output-dir", type=str, default="runs/bc_pretrain",
                        help="Directory to save actor_latest.pt and critic_latest.pt.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Behavior cloning epochs.")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Behavior cloning batch size.")
    parser.add_argument("--bc-lr", type=float, default=1e-3,
                        help="Behavior cloning learning rate.")
    parser.add_argument("--roll-rate-limit", type=float, default=1.2,
                        help="Roll rate normalization (rad/s).")
    parser.add_argument("--pitch-rate-limit", type=float, default=0.8,
                        help="Pitch rate normalization (rad/s).")
    parser.add_argument("--yaw-rate-limit", type=float, default=0.8,
                        help="Yaw rate normalization (rad/s).")
    parser.add_argument("--speed-rate-limit", type=float, default=15.0,
                        help="Speed change normalization (m/s^2).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Sample stride for trajectory steps.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Optional cap for total samples.")
    return parser.parse_args()


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

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    obs_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(12,))
    act_space = gym.spaces.MultiDiscrete([41, 41, 41, 30])

    policy = PPOPolicy(args, obs_space, act_space, device=device)
    policy.actor.train()
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.bc_lr)

    for epoch in range(args.epochs):
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
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} - BC loss: {avg_loss:.6f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(policy.actor.state_dict(), output_dir / "actor_latest.pt")
    torch.save(policy.critic.state_dict(), output_dir / "critic_latest.pt")

    print(f"Saved BC actor to {output_dir / 'actor_latest.pt'}")
    print(f"Saved BC critic to {output_dir / 'critic_latest.pt'}")


if __name__ == "__main__":
    main()
