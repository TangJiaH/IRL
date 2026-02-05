#!/usr/bin/env python
import argparse
import csv
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


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float, integral_limit: float, output_limit: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        self.integral += error * dt
        self.integral = _clamp(self.integral, -self.integral_limit, self.integral_limit)
        derivative = 0.0
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return _clamp(output, -self.output_limit, self.output_limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 PID 控制器生成 JSBSim ACMI 轨迹（专家数据）。")
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
    parser.add_argument("--heading-kp", type=float, default=0.03,
                        help="外环航向 PID (deg->deg) 的 P 系数。")
    parser.add_argument("--heading-ki", type=float, default=0.001,
                        help="外环航向 PID 的 I 系数。")
    parser.add_argument("--heading-kd", type=float, default=0.012,
                        help="外环航向 PID 的 D 系数。")
    parser.add_argument("--roll-kp", type=float, default=0.045,
                        help="内环滚转 PID (deg->aileron) 的 P 系数。")
    parser.add_argument("--roll-ki", type=float, default=0.001,
                        help="内环滚转 PID 的 I 系数。")
    parser.add_argument("--roll-kd", type=float, default=0.015,
                        help="内环滚转 PID 的 D 系数。")
    parser.add_argument("--max-bank-deg", type=float, default=45.0,
                        help="外环输出的目标滚转角限幅（度）。")
    parser.add_argument("--altitude-kp", type=float, default=0.004)
    parser.add_argument("--altitude-ki", type=float, default=0.0005)
    parser.add_argument("--altitude-kd", type=float, default=0.002)
    parser.add_argument("--speed-kp", type=float, default=0.02)
    parser.add_argument("--speed-ki", type=float, default=0.0005)
    parser.add_argument("--speed-kd", type=float, default=0.01)
    parser.add_argument("--integral-limit", type=float, default=20.0,
                        help="PID 积分限幅。")
    parser.add_argument("--rudder-gain", type=float, default=0.15,
                        help="方向舵协调比例（与滚转误差相关）。")
    parser.add_argument("--sideslip-damping", type=float, default=0.03,
                        help="侧滑阻尼系数（基于 body-y 速度）。")
    parser.add_argument("--throttle-bias", type=float, default=0.65,
                        help="油门基准值。")
    parser.add_argument("--reward-csv", type=str, default="",
                        help="可选：保存每条轨迹奖励统计结果。")
    return parser.parse_args()


def _clamp(value: float, low: float, high: float) -> float:
    return max(min(value, high), low)


def _discretize(value: float, low: float, high: float, n: int) -> int:
    value = _clamp(value, low, high)
    ratio = (value - low) / (high - low)
    return int(round(ratio * (n - 1)))


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    env = SingleControlEnv(args.env_config)
    env.seed(args.seed)
    dt = env.time_interval
    altitude_limit = float(getattr(env.config, "altitude_limit", 2500.0))
    safe_altitude_m = altitude_limit + 300.0

    heading_pid = PIDController(
        args.heading_kp, args.heading_ki, args.heading_kd, args.integral_limit, output_limit=args.max_bank_deg
    )
    roll_pid = PIDController(
        args.roll_kp, args.roll_ki, args.roll_kd, args.integral_limit, output_limit=1.0
    )
    altitude_pid = PIDController(
        args.altitude_kp, args.altitude_ki, args.altitude_kd, args.integral_limit, output_limit=1.0
    )
    speed_pid = PIDController(
        args.speed_kp, args.speed_ki, args.speed_kd, args.integral_limit, output_limit=0.25
    )
    reward_summaries = []

    for episode in range(args.episodes):
        env.reset()
        heading_pid.reset()
        roll_pid.reset()
        altitude_pid.reset()
        speed_pid.reset()

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"pid_{episode + 1}_{timestamp}.txt.acmi")

        env.render_with_tacview(
            render_mode="histroy_acmi",
            acmi_filename=out_path,
            eval_env=env,
            timestamp=0.0,
            _should_save_acmi=True,
        )
        episode_rewards = []

        for step in range(args.max_steps):
            ego_id = env.ego_ids[0]
            current_heading = env.agents[ego_id].get_property_value(c.attitude_psi_deg)
            current_altitude = env.agents[ego_id].get_property_value(c.position_h_sl_m)
            current_speed = env.agents[ego_id].get_property_value(c.velocities_u_mps)
            current_roll_deg = np.degrees(env.agents[ego_id].get_property_value(c.attitude_roll_rad))
            current_side_vel = env.agents[ego_id].get_property_value(c.velocities_v_mps)

            target_heading = env.agents[ego_id].get_property_value(c.target_heading_deg)
            target_altitude = env.agents[ego_id].get_property_value(c.target_altitude_ft) * 0.3048
            target_speed = env.agents[ego_id].get_property_value(c.target_velocities_u_mps)

            heading_error = _wrap_deg(target_heading - current_heading)
            target_bank_deg = heading_pid.update(heading_error, dt)
            target_bank_deg = _clamp(target_bank_deg, -args.max_bank_deg, args.max_bank_deg)
            roll_error_deg = target_bank_deg - current_roll_deg

            altitude_error = target_altitude - current_altitude
            speed_error = target_speed - current_speed

            aileron_cmd = roll_pid.update(roll_error_deg, dt)
            elevator_cmd = altitude_pid.update(altitude_error, dt)
            throttle_cmd = _clamp(args.throttle_bias + speed_pid.update(speed_error, dt), 0.4, 0.9)
            rudder_cmd = _clamp(
                args.rudder_gain * (roll_error_deg / max(args.max_bank_deg, 1e-6)) - args.sideslip_damping * current_side_vel,
                -1.0,
                1.0,
            )

            if current_altitude < safe_altitude_m:
                elevator_cmd = max(elevator_cmd, 0.2)
                throttle_cmd = max(throttle_cmd, 0.7)

            action = np.array([[
                _discretize(aileron_cmd, -1.0, 1.0, env.action_space.nvec[0]),
                _discretize(elevator_cmd, -1.0, 1.0, env.action_space.nvec[1]),
                _discretize(rudder_cmd, -1.0, 1.0, env.action_space.nvec[2]),
                _discretize(throttle_cmd, 0.4, 0.9, env.action_space.nvec[3]),
            ]])

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(
                    f"//TARGET heading_deg={target_heading:.3f} alt_m={target_altitude:.2f} speed_mps={target_speed:.2f}\n"
                )
                f.write(
                    f"//PID errors heading_deg={heading_error:.3f} bank_target_deg={target_bank_deg:.3f} "
                    f"roll_error_deg={roll_error_deg:.3f} alt_m={altitude_error:.2f} speed_mps={speed_error:.2f}\n"
                )
                f.write(
                    f"//ACTION aileron={aileron_cmd:.3f} elevator={elevator_cmd:.3f} "
                    f"rudder={rudder_cmd:.3f} throttle={throttle_cmd:.3f}\n"
                )

            _, reward, done, _ = env.step(action)
            step_reward = float(np.array(reward).reshape(-1)[0])
            episode_rewards.append(step_reward)

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(f"//REWARD total={step_reward:.6f}\n")

            env.render_with_tacview(
                render_mode="histroy_acmi",
                acmi_filename=out_path,
                eval_env=env,
                timestamp=float(step + 1) * env.time_interval,
                _should_save_acmi=True,
            )
            if done.any():
                break

        total_reward = float(np.sum(episode_rewards))
        mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        reward_summaries.append({
            "episode": episode + 1,
            "file": out_path,
            "steps": len(episode_rewards),
            "reward_sum": total_reward,
            "reward_mean": mean_reward,
        })
        print(
            f"已生成: {out_path} | steps={len(episode_rewards)} "
            f"reward_sum={total_reward:.4f} reward_mean={mean_reward:.6f}"
        )

    if args.reward_csv and reward_summaries:
        os.makedirs(os.path.dirname(args.reward_csv) or ".", exist_ok=True)
        with open(args.reward_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "file", "steps", "reward_sum", "reward_mean"])
            writer.writeheader()
            writer.writerows(reward_summaries)
        print(f"奖励统计已保存: {args.reward_csv}")

    env.close()


if __name__ == "__main__":
    main()
