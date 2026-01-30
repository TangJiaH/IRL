#!/usr/bin/env python
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.envs import SingleControlEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 JSBSim 环境生成 ACMI 轨迹（带目标与动作注释）。")
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


def _action_to_controls(action: np.ndarray) -> np.ndarray:
    aileron = action[0] * 2.0 / 40.0 - 1.0
    elevator = action[1] * 2.0 / 40.0 - 1.0
    rudder = action[2] * 2.0 / 40.0 - 1.0
    throttle = action[3] * 0.5 / 29.0 + 0.4
    return np.array([aileron, elevator, rudder, throttle], dtype=float)


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    env = SingleControlEnv(args.env_config)
    env.seed(args.seed)

    for episode in range(args.episodes):
        obs = env.reset()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"jsbsim_{episode + 1}_{timestamp}.txt.acmi")

        env.render_with_tacview(
            render_mode="histroy_acmi",
            acmi_filename=out_path,
            eval_env=env,
            timestamp=0.0,
            _should_save_acmi=True,
        )

        for step in range(args.max_steps):
            action = np.array([[rng.randint(0, n) for n in env.action_space.nvec]])
            controls = _action_to_controls(action[0])

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
