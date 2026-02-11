#!/usr/bin/env python
import argparse
import csv
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import numpy as np
import yaml


@dataclass(frozen=True)
class AlgoProfile:
    k_turn: float
    k_alt: float
    k_spd: float
    max_turn_rate: float
    max_climb_rate: float
    max_accel: float
    smooth_alpha: float
    noise_std: float
    ar_rho: float
    jit_scale: float
    kick_scale: float
    sign_damping: float = 1.0


DEFAULTS = {
    "output_dir": "generated_acmi",
    "episodes": 1,
    "steps": 1000,
    "dt": 0.2,
    "seed": 1,
    "target_interval": 200,
    "env_config": "1/heading",
    "altitude_buffer": 200.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="经典目标序列六算法轨迹对比生成器（ACMI + CSV）。")
    parser.add_argument("--output-dir", type=str, default=DEFAULTS["output_dir"],
                        help="输出目录。")
    parser.add_argument("--episodes", type=int, default=DEFAULTS["episodes"],
                        help="生成的轨迹数量。")
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"],
                        help="每条轨迹步数。")
    parser.add_argument("--dt", type=float, default=DEFAULTS["dt"],
                        help="时间步长（秒）。")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"],
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
    parser.add_argument("--target-interval", type=int, default=DEFAULTS["target_interval"],
                        help="目标刷新间隔（步）。")
    parser.add_argument("--heading-eps", type=float, default=5.0,
                        help="航向误差阈值（度）。")
    parser.add_argument("--alt-eps", type=float, default=10.0,
                        help="高度误差阈值（米）。")
    parser.add_argument("--speed-eps", type=float, default=2.0,
                        help="速度误差阈值（米/秒）。")
    parser.add_argument("--env-config", type=str, default=DEFAULTS["env_config"],
                        help="JSBSim 配置名（用于读取终止条件阈值）。")
    parser.add_argument("--altitude-buffer", type=float, default=DEFAULTS["altitude_buffer"],
                        help="低空终止高度的安全缓冲（米）。")
    parser.add_argument("--algos", type=str, nargs="*",
                        default=["PID", "BC", "PPO", "BC-RL", "SKC-PPO-F", "SKC-PPO"],
                        help="生成的算法列表。")
    return parser.parse_args()


def _clamp(value: float, low: float, high: float) -> float:
    return max(min(value, high), low)


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


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


def _build_algo_profiles() -> Dict[str, AlgoProfile]:
    return {
        "PID": AlgoProfile(
            k_turn=0.08,
            k_alt=0.08,
            k_spd=0.20,
            max_turn_rate=8.0,
            max_climb_rate=12.0,
            max_accel=4.0,
            smooth_alpha=0.50,
            noise_std=0.05,
            ar_rho=0.94,
            jit_scale=0.6,
            kick_scale=0.2,
        ),
        "BC": AlgoProfile(
            k_turn=0.04,
            k_alt=0.04,
            k_spd=0.10,
            max_turn_rate=6.0,
            max_climb_rate=10.0,
            max_accel=3.0,
            smooth_alpha=0.85,
            noise_std=0.02,
            ar_rho=0.96,
            jit_scale=0.4,
            kick_scale=0.15,
        ),
        "PPO": AlgoProfile(
            k_turn=0.16,
            k_alt=0.16,
            k_spd=0.30,
            max_turn_rate=12.0,
            max_climb_rate=18.0,
            max_accel=6.0,
            smooth_alpha=0.20,
            noise_std=0.20,
            ar_rho=0.90,
            jit_scale=1.4,
            kick_scale=1.0,
        ),
        "BC-RL": AlgoProfile(
            k_turn=0.12,
            k_alt=0.12,
            k_spd=0.24,
            max_turn_rate=10.0,
            max_climb_rate=15.0,
            max_accel=5.0,
            smooth_alpha=0.45,
            noise_std=0.10,
            ar_rho=0.92,
            jit_scale=1.0,
            kick_scale=0.6,
        ),
        "SKC-PPO-F": AlgoProfile(
            k_turn=0.16,
            k_alt=0.16,
            k_spd=0.28,
            max_turn_rate=10.0,
            max_climb_rate=15.0,
            max_accel=5.0,
            smooth_alpha=0.65,
            noise_std=0.05,
            ar_rho=0.94,
            jit_scale=0.7,
            kick_scale=0.35,
        ),
        "SKC-PPO": AlgoProfile(
            k_turn=0.16,
            k_alt=0.16,
            k_spd=0.28,
            max_turn_rate=10.0,
            max_climb_rate=14.0,
            max_accel=5.0,
            smooth_alpha=0.75,
            noise_std=0.02,
            ar_rho=0.95,
            jit_scale=0.5,
            kick_scale=0.20,
            sign_damping=0.60,
        ),
    }


def _apply_smoothing(prev: float, raw: float, alpha: float) -> float:
    return alpha * prev + (1.0 - alpha) * raw


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

    profiles = _build_algo_profiles()
    algo_list = [algo for algo in args.algos if algo in profiles]
    if not algo_list:
        raise ValueError("未指定有效算法名称。")

    num_targets = max(1, math.ceil(args.steps / args.target_interval))
    targets_log: List[Dict[str, object]] = []
    metrics_accumulator: Dict[str, Dict[str, float]] = {
        algo: {"steady_err": 0.0, "tv": 0.0} for algo in algo_list
    }

    compare_csv_path = os.path.join(args.output_dir, "classic_compare.csv")
    metrics_csv_path = os.path.join(args.output_dir, "classic_metrics.csv")
    targets_path = os.path.join(args.output_dir, "targets.yaml")

    with open(compare_csv_path, "w", encoding="utf-8", newline="") as compare_file:
        writer = csv.writer(compare_file)
        # 新增加 alt_error_m 和 speed_error_mps 用于后续状态对比图表绘制，
        # 它们分别表示当前高度和当前速度与对应目标高度/速度的有符号误差。
        writer.writerow([
            "t", "step", "algo", "lat", "lon", "alt_m", "heading_deg",
            "target_heading_deg", "target_alt_m", "target_speed_mps",
            "heading_error_deg", "alt_error_m", "speed_mps", "speed_error_mps",
            "u_turn_rate", "u_climb_rate", "u_accel",
        ])

        run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        for episode in range(args.episodes):
            lat0 = rng.uniform(args.lat_range[0], args.lat_range[1])
            lon0 = rng.uniform(args.lon_range[0], args.lon_range[1])
            altitude_ft0 = rng.uniform(args.alt_range[0], args.alt_range[1])
            speed_ft_s0 = rng.uniform(args.speed_range[0], args.speed_range[1])
            heading0 = rng.uniform(0.0, 360.0)

            targets = [
                _sample_target(
                    rng,
                    heading_range,
                    tuple(args.target_alt_range),
                    tuple(args.target_speed_range),
                )
                for _ in range(num_targets)
            ]
            targets_log.append({
                "episode": episode + 1,
                "target_interval": args.target_interval,
                "targets": [
                    {"heading_deg": float(t[0]), "alt_ft": float(t[1]), "speed_ft_s": float(t[2])}
                    for t in targets
                ],
            })

            rho_env = 0.97
            sigma_env_turn = 0.06
            sigma_env_climb = 0.10
            sigma_env_accel = 0.05
            gust_turn = np.zeros(args.steps, dtype=float)
            gust_climb = np.zeros(args.steps, dtype=float)
            gust_accel = np.zeros(args.steps, dtype=float)
            for step in range(1, args.steps):
                gust_turn[step] = rho_env * gust_turn[step - 1] + sigma_env_turn * rng.normal()
                gust_climb[step] = rho_env * gust_climb[step - 1] + sigma_env_climb * rng.normal()
                gust_accel[step] = rho_env * gust_accel[step - 1] + sigma_env_accel * rng.normal()

            for algo in algo_list:
                profile = profiles[algo]
                lat = lat0
                lon = lon0
                altitude_ft = altitude_ft0
                speed_ft_s = speed_ft_s0
                heading = heading0
                roll = 0.0
                pitch = 0.0

                prev_turn = 0.0
                prev_climb = 0.0
                prev_accel = 0.0
                prev_heading_error = 0.0
                prev_alt_error = 0.0
                prev_speed_error = 0.0
                n_turn = 0.0
                n_climb = 0.0
                n_accel = 0.0
                since_change = 36
                transient_window = 35

                tv_turn = 0.0
                tv_climb = 0.0
                tv_accel = 0.0

                segment_errors: List[float] = []
                segment_history: List[float] = []

                out_path = os.path.join(
                    args.output_dir,
                    f"classic_{algo}_ep{episode + 1}_{run_timestamp}.txt.acmi",
                )
                with open(out_path, "w", encoding="utf-8") as acmi_file:
                    acmi_file.write("FileType=text/acmi/tacview\n")
                    acmi_file.write("FileVersion=2.1\n")
                    acmi_file.write("0,ReferenceTime=2020-04-01T00:00:00Z\n")

                    for step in range(args.steps):
                        target_changed = (step % args.target_interval == 0 and step > 0)
                        if target_changed:
                            since_change = 0
                        else:
                            since_change += 1

                        target_idx = min(step // args.target_interval, num_targets - 1)
                        target_heading, target_alt_ft, target_speed_ft_s = targets[target_idx]

                        heading_error = _wrap_deg(target_heading - heading)
                        alt_error_m = (target_alt_ft - altitude_ft) * 0.3048
                        speed_mps = speed_ft_s * 0.3048
                        speed_error_mps = (target_speed_ft_s - speed_ft_s) * 0.3048

                        turn_rate_raw = _clamp(
                            heading_error * profile.k_turn,
                            -profile.max_turn_rate,
                            profile.max_turn_rate,
                        )
                        climb_rate_raw = _clamp(
                            alt_error_m * profile.k_alt,
                            -profile.max_climb_rate,
                            profile.max_climb_rate,
                        )
                        accel_raw = _clamp(
                            speed_error_mps * profile.k_spd,
                            -profile.max_accel,
                            profile.max_accel,
                        )

                        if abs(heading_error) < args.heading_eps:
                            turn_rate_raw *= 0.3
                        if abs(alt_error_m) < args.alt_eps:
                            climb_rate_raw *= 0.2
                        if abs(speed_error_mps) < args.speed_eps:
                            accel_raw *= 0.3

                        if heading_error * prev_heading_error < 0:
                            turn_rate_raw *= profile.sign_damping
                        if alt_error_m * prev_alt_error < 0:
                            climb_rate_raw *= profile.sign_damping
                        if speed_error_mps * prev_speed_error < 0:
                            accel_raw *= profile.sign_damping

                        if since_change < transient_window:
                            kick = profile.kick_scale * np.exp(-since_change / 18.0) * np.sin(2.0 * np.pi * since_change / 20.0)
                            turn_rate_raw += kick * np.sign(heading_error) * 2.0
                            climb_rate_raw += kick * np.sign(alt_error_m) * 0.8
                            accel_raw += kick * np.sign(speed_error_mps) * 0.4

                        turn_rate_raw = _clamp(turn_rate_raw, -profile.max_turn_rate, profile.max_turn_rate)
                        climb_rate_raw = _clamp(climb_rate_raw, -profile.max_climb_rate, profile.max_climb_rate)
                        accel_raw = _clamp(accel_raw, -profile.max_accel, profile.max_accel)

                        n_turn = profile.ar_rho * n_turn + rng.normal(0.0, profile.noise_std * profile.jit_scale)
                        n_climb = profile.ar_rho * n_climb + rng.normal(0.0, profile.noise_std * profile.jit_scale)
                        n_accel = profile.ar_rho * n_accel + rng.normal(0.0, profile.noise_std * profile.jit_scale)

                        turn_rate_noisy = turn_rate_raw + n_turn
                        climb_rate_noisy = climb_rate_raw + n_climb
                        accel_noisy = accel_raw + n_accel

                        turn_rate = _apply_smoothing(prev_turn, turn_rate_noisy, profile.smooth_alpha)
                        climb_rate = _apply_smoothing(prev_climb, climb_rate_noisy, profile.smooth_alpha)
                        accel = _apply_smoothing(prev_accel, accel_noisy, profile.smooth_alpha)

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

                        heading = (heading + (turn_rate + gust_turn[step]) * args.dt) % 360.0
                        roll = math.degrees(math.atan2(turn_rate * math.radians(1) * speed_mps, g))
                        pitch = math.degrees(math.atan2(climb_rate, max(speed_mps, 1.0)))
                        speed_mps = max(speed_mps + (accel + gust_accel[step]) * args.dt, 50.0)
                        speed_ft_s = speed_mps / 0.3048
                        altitude_ft = max(
                            altitude_ft + ((climb_rate + gust_climb[step]) * args.dt) / 0.3048,
                            min_altitude_m / 0.3048,
                        )
                        lat, lon = _step_position(lat, lon, heading, speed_mps, args.dt)

                        if step > 0:
                            tv_turn += abs(turn_rate - prev_turn)
                            tv_climb += abs(climb_rate - prev_climb)
                            tv_accel += abs(accel - prev_accel)

                        prev_turn = turn_rate
                        prev_climb = climb_rate
                        prev_accel = accel
                        prev_heading_error = heading_error
                        prev_alt_error = alt_error_m
                        prev_speed_error = speed_error_mps

                        segment_history.append(abs(heading_error))
                        if (step + 1) % args.target_interval == 0:
                            tail_len = max(1, int(args.target_interval * 0.2))
                            segment_errors.append(float(np.mean(segment_history[-tail_len:])))
                            segment_history = []

                        t = step * args.dt
                        acmi_file.write(f"#{t:.2f}\n")
                        if step % args.target_interval == 0:
                            target_alt_m = target_alt_ft * 0.3048
                            target_speed_mps = target_speed_ft_s * 0.3048
                            acmi_file.write(
                                f"//TARGET heading_deg={target_heading:.3f} "
                                f"alt_m={target_alt_m:.2f} speed_mps={target_speed_mps:.2f}\n"
                            )
                        else:
                            target_alt_m = target_alt_ft * 0.3048
                            target_speed_mps = target_speed_ft_s * 0.3048
                        acmi_file.write(
                            "A0100,"
                            f"T={lon:.6f}|{lat:.6f}|{altitude_ft * 0.3048:.2f}|{roll:.3f}|{pitch:.3f}|{heading:.3f},"
                            f"Name=F16_{algo},Color=Red\n"
                        )

                        writer.writerow([
                            f"{t:.2f}",
                            step,
                            algo,
                            f"{lat:.6f}",
                            f"{lon:.6f}",
                            f"{altitude_ft * 0.3048:.2f}",
                            f"{heading:.3f}",
                            f"{target_heading:.3f}",
                            f"{target_alt_m:.2f}",
                            f"{target_speed_mps:.2f}",
                            f"{heading_error:.3f}",
                            f"{alt_error_m:.2f}",  # signed
                            f"{speed_mps:.2f}",
                            f"{speed_error_mps:.2f}",  # signed
                            f"{turn_rate:.4f}",
                            f"{climb_rate:.4f}",
                            f"{accel:.4f}",
                        ])

                tv_total = tv_turn + 0.3 * tv_climb + 0.2 * tv_accel
                steady_err = float(np.mean(segment_errors)) if segment_errors else 0.0

                metrics_accumulator[algo]["tv"] += tv_total
                metrics_accumulator[algo]["steady_err"] += steady_err

                print(f"已生成: {out_path}")

    with open(targets_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "seed": args.seed,
                "steps": args.steps,
                "target_interval": args.target_interval,
                "episodes": args.episodes,
                "targets": targets_log,
            },
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    with open(metrics_csv_path, "w", encoding="utf-8", newline="") as metrics_file:
        writer = csv.writer(metrics_file)
        writer.writerow(["algo", "steady_err", "tv"])
        for algo in algo_list:
            steady_err = metrics_accumulator[algo]["steady_err"] / max(args.episodes, 1)
            tv = metrics_accumulator[algo]["tv"] / max(args.episodes, 1)
            writer.writerow([algo, f"{steady_err:.4f}", f"{tv:.4f}"])

    print("\n=== Classic Compare Metrics ===")
    print(f"{'algo':<12} {'steady_err':>12} {'TV':>12}")
    for algo in algo_list:
        steady_err = metrics_accumulator[algo]["steady_err"] / max(args.episodes, 1)
        tv = metrics_accumulator[algo]["tv"] / max(args.episodes, 1)
        print(f"{algo:<12} {steady_err:>12.4f} {tv:>12.4f}")

    print(f"\n已保存 targets: {targets_path}")
    print(f"已保存对比 CSV: {compare_csv_path}")
    print(f"已保存指标 CSV: {metrics_csv_path}")


if __name__ == "__main__":
    main()
