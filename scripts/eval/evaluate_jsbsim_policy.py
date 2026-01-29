#!/usr/bin/env python
import argparse
import os
import random
import sys
from typing import List, Tuple

import numpy as np
import torch

from config import get_config
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import DummyVecEnv, ShareDummyVecEnv
from algorithms.ppo.ppo_policy import PPOPolicy


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SingleCombat":
                env = SingleCombatEnv(all_args.scenario_name)
            elif all_args.env_name == "SingleControl":
                env = SingleControlEnv(all_args.scenario_name)
            elif all_args.env_name == "MultipleCombat":
                env = MultipleCombatEnv(all_args.scenario_name)
            else:
                raise NotImplementedError(f"Unsupported env: {all_args.env_name}")
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.env_name == "MultipleCombat":
        return ShareDummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = get_config()
    group = parser.add_argument_group("Evaluation parameters")
    group.add_argument("--model-dir", type=str, required=True,
                       help="包含 actor_latest.pt 与 critic_latest.pt 的目录。")
    group.add_argument("--env-name", type=str, default="SingleControl",
                       help="环境名称（SingleControl/SingleCombat/MultipleCombat）。")
    group.add_argument("--scenario-name", type=str, default="1/heading",
                       help="JSBSim 配置名（位于 envs/JSBSim/configs）。")
    group.add_argument("--episodes", type=int, default=20,
                       help="评估回合数。")
    group.add_argument("--max-steps", type=int, default=1000,
                       help="每回合最大步数。")
    group.add_argument("--output-csv", type=str, default="",
                       help="可选，评估结果输出 CSV。")
    return parser.parse_args(args)


def _calc_rmse(accum_sq: float, steps: int) -> float:
    return float(np.sqrt(accum_sq / max(steps, 1)))


def main(args: List[str]) -> None:
    all_args = parse_args(args)

    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)

    device = torch.device("cuda:0" if all_args.cuda and torch.cuda.is_available() else "cpu")

    eval_envs = make_eval_env(all_args)
    obs_space = eval_envs.observation_space
    act_space = eval_envs.action_space

    policy = PPOPolicy(all_args, obs_space, act_space, device=device)
    actor_path = os.path.join(all_args.model_dir, "actor_latest.pt")
    critic_path = os.path.join(all_args.model_dir, "critic_latest.pt")
    policy.actor.load_state_dict(torch.load(actor_path, map_location=device))
    policy.critic.load_state_dict(torch.load(critic_path, map_location=device))
    policy.prep_rollout()

    episode_returns: List[float] = []
    episode_lengths: List[int] = []
    success_flags: List[int] = []
    rmse_heading: List[float] = []
    rmse_altitude: List[float] = []
    rmse_speed: List[float] = []

    total_episodes = 0
    eval_obs = eval_envs.reset()
    num_agents = eval_envs.envs[0].num_agents
    eval_masks = np.ones((all_args.n_eval_rollout_threads, num_agents, 1), dtype=np.float32)
    eval_rnn_states = np.zeros(
        (all_args.n_eval_rollout_threads, num_agents, all_args.recurrent_hidden_layers, all_args.recurrent_hidden_size),
        dtype=np.float32
    )

    episode_reward = np.zeros(all_args.n_eval_rollout_threads, dtype=np.float32)
    episode_steps = np.zeros(all_args.n_eval_rollout_threads, dtype=np.int32)
    delta_heading_sq = np.zeros(all_args.n_eval_rollout_threads, dtype=np.float32)
    delta_altitude_sq = np.zeros(all_args.n_eval_rollout_threads, dtype=np.float32)
    delta_speed_sq = np.zeros(all_args.n_eval_rollout_threads, dtype=np.float32)

    while total_episodes < all_args.episodes:
        with torch.no_grad():
            eval_actions, eval_rnn_states = policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
        eval_actions = np.array(np.split(eval_actions.cpu().numpy(), all_args.n_eval_rollout_threads))
        eval_rnn_states = np.array(np.split(eval_rnn_states.cpu().numpy(), all_args.n_eval_rollout_threads))

        eval_obs, eval_rewards, eval_dones, eval_infos = eval_envs.step(eval_actions)
        eval_dones_env = np.all(eval_dones.squeeze(axis=-1), axis=-1)

        for env_idx in range(all_args.n_eval_rollout_threads):
            obs = eval_obs[env_idx, 0]
            delta_heading_rad = obs[1]
            delta_altitude_m = obs[0] * 1000.0
            delta_speed_mps = obs[2] * 340.0
            delta_heading_sq[env_idx] += delta_heading_rad ** 2
            delta_altitude_sq[env_idx] += delta_altitude_m ** 2
            delta_speed_sq[env_idx] += delta_speed_mps ** 2

        episode_reward += eval_rewards.squeeze(-1).sum(axis=1)
        episode_steps += 1

        force_done = episode_steps >= all_args.max_steps
        eval_dones_env = np.logical_or(eval_dones_env, force_done)

        for env_idx, done_flag in enumerate(eval_dones_env):
            if not done_flag:
                continue
            info = eval_infos[env_idx] if env_idx < len(eval_infos) else {}
            success_flag = 1 if info.get("heading_turn_counts", 0) > 0 else 0
            length = int(min(episode_steps[env_idx], all_args.max_steps))
            episode_returns.append(float(episode_reward[env_idx]))
            episode_lengths.append(length)
            success_flags.append(success_flag)
            rmse_heading.append(_calc_rmse(delta_heading_sq[env_idx], length))
            rmse_altitude.append(_calc_rmse(delta_altitude_sq[env_idx], length))
            rmse_speed.append(_calc_rmse(delta_speed_sq[env_idx], length))

            total_episodes += 1
            if total_episodes >= all_args.episodes:
                break
            episode_reward[env_idx] = 0.0
            episode_steps[env_idx] = 0
            delta_heading_sq[env_idx] = 0.0
            delta_altitude_sq[env_idx] = 0.0
            delta_speed_sq[env_idx] = 0.0

        eval_masks = np.ones_like(eval_masks, dtype=np.float32)
        eval_masks[eval_dones_env == True] = 0.0
        eval_rnn_states[eval_dones_env == True] = 0.0

    eval_envs.close()

    success_rate = float(np.mean(success_flags)) if success_flags else 0.0
    mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
    mean_rmse_heading = float(np.mean(rmse_heading)) if rmse_heading else 0.0
    mean_rmse_altitude = float(np.mean(rmse_altitude)) if rmse_altitude else 0.0
    mean_rmse_speed = float(np.mean(rmse_speed)) if rmse_speed else 0.0

    print(f"success_rate: {success_rate:.3f}")
    print(f"avg_return: {mean_return:.3f}")
    print(f"rmse_heading: {mean_rmse_heading:.3f}")
    print(f"rmse_altitude: {mean_rmse_altitude:.3f}")
    print(f"rmse_speed: {mean_rmse_speed:.3f}")

    if all_args.output_csv:
        import csv
        with open(all_args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "success_rate",
                "avg_return",
                "rmse_heading",
                "rmse_altitude",
                "rmse_speed",
            ])
            writer.writerow([
                success_rate,
                mean_return,
                mean_rmse_heading,
                mean_rmse_altitude,
                mean_rmse_speed,
            ])


if __name__ == "__main__":
    main(sys.argv[1:])
