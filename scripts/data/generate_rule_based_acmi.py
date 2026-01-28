#!/usr/bin/env python
import argparse
import math
import os
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于规则生成单机飞行轨迹（ACMI）。")
    parser.add_argument("--output-dir", type=str, default="generated_acmi",
                        help="输出目录。")
    parser.add_argument("--episodes", type=int, default=5,
                        help="生成的轨迹数量。")
    parser.add_argument("--steps", type=int, default=1000,
                        help="每条轨迹步数。")
    parser.add_argument("--dt", type=float, default=0.2,
                        help="时间步长（秒）。")
    parser.add_argument("--seed", type=int, default=1,
                        help="随机种子。")
    parser.add_argument("--lat-range", type=float, nargs=2, default=[59.8, 60.2],
                        help="纬度范围（度）。")
    parser.add_argument("--lon-range", type=float, nargs=2, default=[119.8, 120.2],
                        help="经度范围（度）。")
    parser.add_argument("--alt-range", type=float, nargs=2, default=[14000.0, 30000.0],
                        help="初始高度范围（英尺）。")
    parser.add_argument("--speed-range", type=float, nargs=2, default=[400.0, 1200.0],
                        help="初始速度范围（英尺/秒）。")
    parser.add_argument("--target-alt-range", type=float, nargs=2, default=[14000.0, 30000.0],
                        help="目标高度范围（英尺）。")
    parser.add_argument("--target-speed-range", type=float, nargs=2, default=[400.0, 1200.0],
                        help="目标速度范围（英尺/秒）。")
    parser.add_argument("--target-interval", type=int, default=200,
                        help="目标刷新间隔（步）。")
    parser.add_argument("--heading-eps", type=float, default=5.0,
                        help="航向误差阈值（度）。")
    parser.add_argument("--alt-eps", type=float, default=10.0,
                        help="高度误差阈值（米）。")
    parser.add_argument("--speed-eps", type=float, default=2.0,
                        help="速度误差阈值（米/秒）。")
    parser.add_argument("--env-config", type=str, default="1/heading",
                        help="JSBSim 配置名（用于读取终止条件阈值）。")
    parser.add_argument("--altitude-buffer", type=float, default=200.0,
                        help="低空终止高度的安全缓冲（米）。")
    return parser.parse_args()


def _clamp(value: float, low: float, high: float) -> float:
    return max(min(value, high), low)


def _wrap_deg(angle: float) -> float:
    angle = (angle + 180.0) % 360.0 - 180.0
    return angle


def _step_position(lat_deg: float, lon_deg: float, heading_deg: float, speed_mps: float, dt: float) -> Tuple[float, float]:
    earth_radius = 6371000.0
    distance = speed_mps * dt
    heading_rad = math.radians(heading_deg)
    dlat = (distance * math.cos(heading_rad)) / earth_radius
    dlon = (distance * math.sin(heading_rad)) / (earth_radius * math.cos(math.radians(lat_deg)))
    return lat_deg + math.degrees(dlat), lon_deg + math.degrees(dlon)


def _sample_target(rng: np.random.RandomState, heading_range: Tuple[float, float],
                   alt_range_ft: Tuple[float, float], speed_range_ft_s: Tuple[float, float]) -> Tuple[float, float, float]:
    heading = rng.uniform(heading_range[0], heading_range[1])
    altitude = rng.uniform(alt_range_ft[0], alt_range_ft[1])
    speed = rng.uniform(speed_range_ft_s[0], speed_range_ft_s[1])
    return heading, altitude, speed


def _load_env_limits(env_config: str) -> Dict[str, float]:
    config_path = os.path.join("envs", "JSBSim", "configs", f"{env_config}.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return {
        "altitude_limit_m": float(data.get("altitude_limit", 2500)),
        "acceleration_limit_x": float(data.get("acceleration_limit_x", 10.0)),
        "acceleration_limit_y": float(data.get("acceleration_limit_y", 10.0)),
        "acceleration_limit_z": float(data.get("acceleration_limit_z", 10.0)),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    heading_range = (0.0, 360.0)
    g = 9.81
    env_limits = _load_env_limits(args.env_config)
    min_altitude_m = env_limits["altitude_limit_m"] + args.altitude_buffer
    max_g_x = env_limits["acceleration_limit_x"]
    max_g_y = env_limits["acceleration_limit_y"]
    max_g_z = env_limits["acceleration_limit_z"]

    for episode in range(args.episodes):
        lat = rng.uniform(args.lat_range[0], args.lat_range[1])
        lon = rng.uniform(args.lon_range[0], args.lon_range[1])
        altitude_ft = rng.uniform(args.alt_range[0], args.alt_range[1])
        speed_ft_s = rng.uniform(args.speed_range[0], args.speed_range[1])
        heading = rng.uniform(0.0, 360.0)
        roll = 0.0
        pitch = 0.0

        target_heading, target_alt_ft, target_speed_ft_s = _sample_target(
            rng, heading_range, tuple(args.target_alt_range), tuple(args.target_speed_range)
        )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"rule_based_{episode + 1}_{timestamp}.txt.acmi")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")
            f.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")

            for step in range(args.steps):
                if step % args.target_interval == 0:
                    target_heading, target_alt_ft, target_speed_ft_s = _sample_target(
                        rng, heading_range, tuple(args.target_alt_range), tuple(args.target_speed_range)
                    )

                heading_error = _wrap_deg(target_heading - heading)
                alt_error_m = (target_alt_ft - altitude_ft) * 0.3048
                speed_mps = speed_ft_s * 0.3048
                speed_error_mps = (target_speed_ft_s - speed_ft_s) * 0.3048

                max_turn_rate = 8.0
                turn_rate = _clamp(heading_error * 0.08, -max_turn_rate, max_turn_rate)
                if abs(heading_error) < args.heading_eps:
                    turn_rate *= 0.3

                max_climb_rate = 15.0
                climb_rate = _clamp(alt_error_m * 0.08, -max_climb_rate, max_climb_rate)
                if abs(alt_error_m) < args.alt_eps:
                    climb_rate *= 0.2

                max_accel = 5.0
                accel = _clamp(speed_error_mps * 0.2, -max_accel, max_accel)
                if abs(speed_error_mps) < args.speed_eps:
                    accel *= 0.3

                altitude_m = altitude_ft * 0.3048
                if altitude_m <= min_altitude_m and climb_rate < 0:
                    climb_rate = abs(climb_rate)

                lateral_acc = abs(math.radians(turn_rate) * speed_mps)
                g_y = lateral_acc / g
                if g_y > max_g_y:
                    scale = max_g_y / max(g_y, 1e-6)
                    turn_rate *= scale

                g_z = abs(climb_rate) / g
                if g_z > max_g_z:
                    scale = max_g_z / max(g_z, 1e-6)
                    climb_rate *= scale

                g_x = abs(accel) / g
                if g_x > max_g_x:
                    scale = max_g_x / max(g_x, 1e-6)
                    accel *= scale

                heading = (heading + turn_rate * args.dt) % 360.0
                roll = math.degrees(math.atan2(turn_rate * math.radians(1) * speed_mps, g))
                pitch = math.degrees(math.atan2(climb_rate, max(speed_mps, 1.0)))
                speed_mps = max(speed_mps + accel * args.dt, 50.0)
                speed_ft_s = speed_mps / 0.3048
                altitude_ft = max(altitude_ft + (climb_rate * args.dt) / 0.3048, min_altitude_m / 0.3048)
                lat, lon = _step_position(lat, lon, heading, speed_mps, args.dt)

                t = step * args.dt
                f.write(f"#{t:.2f}\n")
                f.write(
                    "A0100,"
                    f"T={lon:.6f}|{lat:.6f}|{altitude_ft * 0.3048:.2f}|{roll:.3f}|{pitch:.3f}|{heading:.3f},"
                    "Name=F16,Color=Red\n"
                )

        print(f"已生成: {out_path}")


if __name__ == "__main__":
    main()
