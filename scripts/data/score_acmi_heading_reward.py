#!/usr/bin/env python
import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

from envs.JSBSim.utils.utils import LLA2NEU


TARGET_RE = re.compile(
    r"//TARGET\s+heading_deg=(?P<heading>-?\d+(?:\.\d+)?)\s+"
    r"alt_m=(?P<alt>-?\d+(?:\.\d+)?)\s+speed_mps=(?P<speed>-?\d+(?:\.\d+)?)"
)


@dataclass(frozen=True)
class RewardScales:
    heading_error_scale: float = 5.0
    altitude_error_scale: float = 15.24
    roll_error_scale: float = 0.35
    speed_error_scale: float = 24.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用飞行控制奖励函数评估 ACMI 轨迹。")
    parser.add_argument("--acmi-path", type=str, required=True,
                        help="ACMI 文件或目录。")
    parser.add_argument("--env-config", type=str, default="1/heading",
                        help="JSBSim 配置名（用于加载奖励权重与高度阈值）。")
    parser.add_argument("--output-csv", type=str, default="",
                        help="可选：保存评估结果为 CSV。")
    return parser.parse_args()


def _read_text_lines(path: Path) -> List[str]:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk", "cp1252", "latin-1"):
        try:
            return data.decode(encoding).splitlines()
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore").splitlines()


def _safe_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _wrap_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _normalize_weights(weights: List[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (4,):
        weights = np.full(4, 0.25, dtype=float)
    weights = np.clip(weights, 0.0, None)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return np.full(4, 0.25, dtype=float)
    return weights / weight_sum


def _reward_components(delta_heading_deg: float, delta_altitude_m: float, roll_rad: float, delta_speed_mps: float,
                       scales: RewardScales) -> np.ndarray:
    heading_r = math.exp(-((delta_heading_deg / scales.heading_error_scale) ** 2))
    alt_r = math.exp(-((delta_altitude_m / scales.altitude_error_scale) ** 2))
    roll_r = math.exp(-((roll_rad / scales.roll_error_scale) ** 2))
    speed_r = math.exp(-((delta_speed_mps / scales.speed_error_scale) ** 2))
    return np.array([heading_r, alt_r, roll_r, speed_r], dtype=float)


def _heading_reward(components: np.ndarray, weights: np.ndarray) -> float:
    return math.exp(float(np.sum(weights * np.log(components + 1e-8))))


def _altitude_reward(altitude_m: float, vz_mps: float, safe_alt_km: float, danger_alt_km: float, kv: float) -> float:
    ego_z_km = altitude_m / 1000.0
    ego_vz_mh = vz_mps / 340.0
    pv = 0.0
    if ego_z_km <= safe_alt_km:
        pv = -np.clip(ego_vz_mh / kv * (safe_alt_km - ego_z_km) / safe_alt_km, 0.0, 1.0)
    ph = 0.0
    if ego_z_km <= danger_alt_km:
        ph = np.clip(ego_z_km / danger_alt_km, 0.0, 1.0) - 1.0 - 1.0
    return float(pv + ph)


def _load_config_values(env_config: str) -> Tuple[np.ndarray, float, float, float]:
    config_path = Path("envs") / "JSBSim" / "configs" / f"{env_config}.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    weights = _normalize_weights(data.get("HeadingReward_weights", [0.25, 0.25, 0.25, 0.25]))
    safe_alt = float(data.get("AltitudeReward_safe_altitude", 4.0))
    danger_alt = float(data.get("AltitudeReward_danger_altitude", 3.5))
    kv = float(data.get("AltitudeReward_Kv", 0.2))
    return weights, safe_alt, danger_alt, kv


def _parse_acmi_with_targets(path: Path) -> List[Tuple[float, float, float, float, float, float, Optional[float], Optional[float], Optional[float]]]:
    states = []
    current_time = 0.0
    current_target = (None, None, None)
    for raw_line in _read_text_lines(path):
        line = raw_line.strip().lstrip("\ufeff")
        if not line:
            continue
        if line.startswith("#"):
            current_time = float(line[1:])
            continue
        if line.startswith("//TARGET"):
            match = TARGET_RE.match(line)
            if match:
                current_target = (
                    float(match.group("heading")),
                    float(match.group("alt")),
                    float(match.group("speed")),
                )
            continue
        if "T=" not in line:
            continue
        payload = line.split("T=", 1)[1].split(",", 1)[0]
        fields = payload.split("|")
        if len(fields) < 3:
            continue
        lon = _safe_float(fields[0])
        lat = _safe_float(fields[1])
        alt = _safe_float(fields[2])
        if lon is None or lat is None or alt is None:
            continue
        roll = _safe_float(fields[3]) if len(fields) > 3 else 0.0
        yaw = _safe_float(fields[5]) if len(fields) > 5 else None
        states.append((current_time, lon, lat, alt, roll, yaw, *current_target))
    return states


def _score_trajectory(path: Path, weights: np.ndarray, safe_alt_km: float, danger_alt_km: float, kv: float) -> Optional[dict]:
    states = _parse_acmi_with_targets(path)
    if len(states) < 2:
        return None

    times = np.array([s[0] for s in states], dtype=float)
    lon = np.array([s[1] for s in states], dtype=float)
    lat = np.array([s[2] for s in states], dtype=float)
    alt = np.array([s[3] for s in states], dtype=float)
    roll_deg = np.array([s[4] for s in states], dtype=float)
    yaw_deg = np.array([np.nan if s[5] is None else s[5] for s in states], dtype=float)
    target_heading = np.array([np.nan if s[6] is None else s[6] for s in states], dtype=float)
    target_alt = np.array([np.nan if s[7] is None else s[7] for s in states], dtype=float)
    target_speed = np.array([np.nan if s[8] is None else s[8] for s in states], dtype=float)

    origin_lon, origin_lat, origin_alt = lon[0], lat[0], alt[0]
    positions = np.array([LLA2NEU(lon[i], lat[i], alt[i], origin_lon, origin_lat, origin_alt) for i in range(len(states))])

    dt = np.diff(times)
    dt = np.where(dt <= 0, 0.2, dt)
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt

    headings = np.empty(len(states) - 1, dtype=float)
    for i in range(len(headings)):
        if not math.isnan(yaw_deg[i + 1]):
            headings[i] = yaw_deg[i + 1]
        else:
            headings[i] = _bearing_deg(lat[i], lon[i], lat[i + 1], lon[i + 1])

    fallback_heading = headings[-1]
    fallback_alt = alt[-1]
    fallback_speed = speeds[-1]

    for idx in range(len(target_heading)):
        if math.isnan(target_heading[idx]):
            target_heading[idx] = fallback_heading
        if math.isnan(target_alt[idx]):
            target_alt[idx] = fallback_alt
        if math.isnan(target_speed[idx]):
            target_speed[idx] = fallback_speed

    heading_rewards = []
    altitude_rewards = []
    total_rewards = []

    scales = RewardScales()
    for idx in range(len(headings)):
        delta_heading = _wrap_deg(target_heading[idx + 1] - headings[idx])
        delta_altitude = target_alt[idx + 1] - alt[idx + 1]
        delta_speed = target_speed[idx + 1] - speeds[idx]
        roll_rad = math.radians(roll_deg[idx + 1])
        components = _reward_components(delta_heading, delta_altitude, roll_rad, delta_speed, scales)
        heading_reward = _heading_reward(components, weights)
        vz = (alt[idx + 1] - alt[idx]) / dt[idx]
        altitude_reward = _altitude_reward(alt[idx + 1], vz, safe_alt_km, danger_alt_km, kv)
        total_reward = heading_reward + altitude_reward
        heading_rewards.append(heading_reward)
        altitude_rewards.append(altitude_reward)
        total_rewards.append(total_reward)

    heading_rewards = np.array(heading_rewards, dtype=float)
    altitude_rewards = np.array(altitude_rewards, dtype=float)
    total_rewards = np.array(total_rewards, dtype=float)

    return {
        "file": str(path),
        "steps": len(total_rewards),
        "heading_mean": float(heading_rewards.mean()),
        "altitude_mean": float(altitude_rewards.mean()),
        "total_mean": float(total_rewards.mean()),
        "total_sum": float(total_rewards.sum()),
    }


def main() -> None:
    args = parse_args()
    weights, safe_alt_km, danger_alt_km, kv = _load_config_values(args.env_config)

    base = Path(args.acmi_path)
    if base.is_file():
        files = [base]
    else:
        files = sorted(base.glob("*.acmi"))
        if not files:
            files = sorted(base.rglob("*.acmi"))

    if not files:
        raise SystemExit(f"未找到 ACMI 文件: {args.acmi_path}")

    results = []
    for file_path in files:
        summary = _score_trajectory(file_path, weights, safe_alt_km, danger_alt_km, kv)
        if summary is None:
            print(f"跳过（数据不足）: {file_path}")
            continue
        results.append(summary)
        print(
            f"{file_path}: steps={summary['steps']} heading_mean={summary['heading_mean']:.4f} "
            f"altitude_mean={summary['altitude_mean']:.4f} total_mean={summary['total_mean']:.4f} "
            f"total_sum={summary['total_sum']:.2f}"
        )

    if args.output_csv and results:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["file", "steps", "heading_mean", "altitude_mean", "total_mean", "total_sum"],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"已保存评估结果: {args.output_csv}")


if __name__ == "__main__":
    main()
