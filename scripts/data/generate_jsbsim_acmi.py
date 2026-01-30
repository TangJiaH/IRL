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


def _clamp(value: float, low: float, high: float) -> float:
    return max(min(value, high), low)


def _discretize(value: float, low: float, high: float, n: int) -> int:
    value = _clamp(value, low, high)
    ratio = (value - low) / (high - low)
    return int(round(ratio * (n - 1)))


def _rule_based_controls(delta_heading: float, delta_altitude: float, delta_speed: float) -> np.ndarray:
    heading_cmd = _clamp(delta_heading / 30.0, -1.0, 1.0)
    altitude_cmd = _clamp(delta_altitude / 300.0, -1.0, 1.0)
    throttle_cmd = _clamp(0.65 + delta_speed * 0.002, 0.4, 0.9)
    rudder_cmd = 0.0
    return np.array([heading_cmd, altitude_cmd, rudder_cmd, throttle_cmd], dtype=float)


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
            ego_id = env.ego_ids[0]
            delta_heading = env.agents[ego_id].get_property_value(c.delta_heading)
            delta_altitude = env.agents[ego_id].get_property_value(c.delta_altitude)
            delta_speed = env.agents[ego_id].get_property_value(c.delta_velocities_u)
            controls = _rule_based_controls(delta_heading, delta_altitude, delta_speed)
            action = np.array([[
                _discretize(controls[0], -1.0, 1.0, env.action_space.nvec[0]),
                _discretize(controls[1], -1.0, 1.0, env.action_space.nvec[1]),
                _discretize(controls[2], -1.0, 1.0, env.action_space.nvec[2]),
                _discretize(controls[3], 0.4, 0.9, env.action_space.nvec[3]),
            ]])

            target_heading = env.agents[ego_id].get_property_value(c.target_heading_deg)
            target_altitude = env.agents[ego_id].get_property_value(c.target_altitude_ft) * 0.3048
            target_speed = env.agents[ego_id].get_property_value(c.target_velocities_u_mps)

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
