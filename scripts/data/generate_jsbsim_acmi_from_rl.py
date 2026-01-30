#!/usr/bin/env python
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from algorithms.ppo.ppo_policy import PPOPolicy
from config import get_config
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.envs import SingleControlEnv


def parse_args() -> argparse.Namespace:
    parser = get_config()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="包含 actor_latest.pt 与 critic_latest.pt 的目录。")
    parser.add_argument("--output-dir", type=str, default="generated_acmi",
                        help="输出目录。")
    parser.add_argument("--episodes", type=int, default=5,
                        help="生成的轨迹数量。")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="每条轨迹最大步数。")
    parser.add_argument("--env-config", type=str, default="1/heading",
                        help="JSBSim 配置名。")
    parser.add_argument("--seed", type=int, default=1,
                        help="随机种子。")
    return parser.parse_args()


def _action_to_controls(action: np.ndarray, nvec: np.ndarray) -> np.ndarray:
    aileron = action[0] * 2.0 / (nvec[0] - 1.0) - 1.0
    elevator = action[1] * 2.0 / (nvec[1] - 1.0) - 1.0
    rudder = action[2] * 2.0 / (nvec[2] - 1.0) - 1.0
    throttle = action[3] * 0.5 / (nvec[3] - 1.0) + 0.4
    return np.array([aileron, elevator, rudder, throttle], dtype=float)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    env = SingleControlEnv(args.env_config)
    env.seed(args.seed)

    obs_space = env.observation_space
    act_space = env.action_space
    policy = PPOPolicy(args, obs_space, act_space, device=device)

    actor_path = os.path.join(args.model_dir, "actor_latest.pt")
    critic_path = os.path.join(args.model_dir, "critic_latest.pt")
    policy.actor.load_state_dict(torch.load(actor_path, map_location=device))
    policy.critic.load_state_dict(torch.load(critic_path, map_location=device))
    policy.prep_rollout()

    for episode in range(args.episodes):
        obs = env.reset()
        rnn_states = np.zeros((env.num_agents, args.recurrent_hidden_layers, args.recurrent_hidden_size), dtype=np.float32)
        masks = np.ones((env.num_agents, 1), dtype=np.float32)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"rl_policy_{episode + 1}_{timestamp}.txt.acmi")

        env.render_with_tacview(
            render_mode="histroy_acmi",
            acmi_filename=out_path,
            eval_env=env,
            timestamp=0.0,
            _should_save_acmi=True,
        )

        for step in range(args.max_steps):
            with torch.no_grad():
                action_tensor, rnn_tensor = policy.act(
                    obs,
                    rnn_states,
                    masks,
                    deterministic=True,
                )
            action = action_tensor.cpu().numpy().astype(int)
            rnn_states = rnn_tensor.cpu().numpy()

            controls = _action_to_controls(action[0], env.action_space.nvec)
            target_heading = env.agents[env.ego_ids[0]].get_property_value(c.target_heading_deg)
            target_altitude = env.agents[env.ego_ids[0]].get_property_value(c.target_altitude_ft) * 0.3048
            target_speed = env.agents[env.ego_ids[0]].get_property_value(c.target_velocities_u_mps)

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(f"//TARGET heading_deg={target_heading:.3f} alt_m={target_altitude:.2f} speed_mps={target_speed:.2f}\n")
                f.write(
                    f"//ACTION aileron={controls[0]:.3f} elevator={controls[1]:.3f} "
                    f"rudder={controls[2]:.3f} throttle={controls[3]:.3f}\n"
                )

            obs, _, done, _ = env.step(action)
            env.render_with_tacview(
                render_mode="histroy_acmi",
                acmi_filename=out_path,
                eval_env=env,
                timestamp=float(step + 1) * env.time_interval,
                _should_save_acmi=True,
            )
            if done.any():
                break

        print(f"已生成: {out_path}")

    env.close()


if __name__ == "__main__":
    main()
