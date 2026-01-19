import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU


@dataclass(frozen=True)
class RewardScales:
    heading_error_scale: float = 5.0
    altitude_error_scale: float = 15.24
    roll_error_scale: float = 0.35
    speed_error_scale: float = 24.0


def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (4,):
        weights = np.full(4, 0.25, dtype=float)
    weights = np.clip(weights, 0.0, None)
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return np.full(4, 0.25, dtype=float)
    return weights / weight_sum


def _safe_float(value: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _wrap_heading_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _reward_components(delta_heading_deg: float, delta_altitude_m: float, roll_rad: float, delta_speed_mps: float,
                       scales: RewardScales) -> np.ndarray:
    heading_r = math.exp(-((delta_heading_deg / scales.heading_error_scale) ** 2))
    alt_r = math.exp(-((delta_altitude_m / scales.altitude_error_scale) ** 2))
    roll_r = math.exp(-((roll_rad / scales.roll_error_scale) ** 2))
    speed_r = math.exp(-((delta_speed_mps / scales.speed_error_scale) ** 2))
    return np.array([heading_r, alt_r, roll_r, speed_r], dtype=float)


def _log_features_from_components(components: np.ndarray) -> np.ndarray:
    return np.log(np.clip(components, 1e-8, 1.0))


def _trajectory_feature_sums(trajectories: Sequence[np.ndarray]) -> np.ndarray:
    return np.array([traj.sum(axis=0) for traj in trajectories], dtype=float)


def parse_acmi_file(path: Path) -> List[Tuple[float, float, float, float, Optional[float], Optional[float]]]:
    states = []
    current_time = 0.0
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip().lstrip("\ufeff")
            if not line:
                continue
            if line.startswith("#"):
                current_time = float(line[1:])
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
            roll = _safe_float(fields[3]) if len(fields) > 3 else None
            yaw = _safe_float(fields[5]) if len(fields) > 5 else None
            states.append((current_time, lon, lat, alt, roll, yaw))
    return states


def trajectory_log_features_from_acmi(path: Path, scales: Optional[RewardScales] = None) -> Optional[np.ndarray]:
    scales = scales or RewardScales()
    states = parse_acmi_file(path)
    if len(states) < 2:
        return None

    times = np.array([s[0] for s in states], dtype=float)
    lon = np.array([s[1] for s in states], dtype=float)
    lat = np.array([s[2] for s in states], dtype=float)
    alt = np.array([s[3] for s in states], dtype=float)
    roll_deg = np.array([0.0 if s[4] is None else s[4] for s in states], dtype=float)
    yaw_deg = np.array([np.nan if s[5] is None else s[5] for s in states], dtype=float)

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

    # Use the final expert state as the target, so deltas represent convergence to the segment goal.
    target_heading = headings[-1]
    target_altitude = alt[-1]
    target_speed = speeds[-1]

    features = []
    for idx in range(len(headings)):
        delta_heading = _wrap_heading_deg(target_heading - headings[idx])
        delta_altitude = target_altitude - alt[idx + 1]
        delta_speed = target_speed - speeds[idx]
        roll_rad = math.radians(roll_deg[idx + 1])
        components = _reward_components(delta_heading, delta_altitude, roll_rad, delta_speed, scales)
        features.append(_log_features_from_components(components))

    return np.array(features, dtype=float)


def load_acmi_trajectories(path: str, scales: Optional[RewardScales] = None) -> List[np.ndarray]:
    base = Path(path)
    files = [base] if base.is_file() else sorted(base.glob("*.acmi"))
    trajectories = []
    for file_path in files:
        features = trajectory_log_features_from_acmi(file_path, scales=scales)
        if features is not None and len(features) > 0:
            trajectories.append(features)
    return trajectories


def sample_env_trajectories(env, num_episodes: int, max_steps: Optional[int] = None,
                            scales: Optional[RewardScales] = None, seed: Optional[int] = None) -> List[np.ndarray]:
    scales = scales or RewardScales()
    if seed is not None:
        env.seed(seed)
    agent_id = next(iter(env.agents.keys()))
    max_steps = max_steps or env.max_steps
    trajectories = []
    for _ in range(num_episodes):
        env.reset()
        episode_features = []
        for _ in range(max_steps):
            action = env.action_space.sample()
            env.step(np.array([action]))
            delta_heading = env.agents[agent_id].get_property_value(c.delta_heading)
            delta_altitude = env.agents[agent_id].get_property_value(c.delta_altitude)
            roll_rad = env.agents[agent_id].get_property_value(c.attitude_roll_rad)
            delta_speed = env.agents[agent_id].get_property_value(c.delta_velocities_u)
            components = _reward_components(delta_heading, delta_altitude, roll_rad, delta_speed, scales)
            episode_features.append(_log_features_from_components(components))
        if episode_features:
            trajectories.append(np.array(episode_features, dtype=float))
    return trajectories


class MaxEntIRL:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 50):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, expert_trajectories: Sequence[np.ndarray], sampled_trajectories: Sequence[np.ndarray],
            init_weights: Optional[Sequence[float]] = None) -> dict:
        if not expert_trajectories:
            raise ValueError("expert_trajectories must not be empty.")
        if not sampled_trajectories:
            raise ValueError("sampled_trajectories must not be empty.")

        expert_sums = _trajectory_feature_sums(expert_trajectories)
        sampled_sums = _trajectory_feature_sums(sampled_trajectories)

        expert_expectation = expert_sums.mean(axis=0)
        weights = _normalize_weights(init_weights or [0.25, 0.25, 0.25, 0.25])
        history = []
        for epoch in range(self.epochs):
            logits = sampled_sums @ weights
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / probs.sum()
            expected = probs @ sampled_sums
            grad = expert_expectation - expected
            weights = _normalize_weights(weights + self.learning_rate * grad)
            history.append({"epoch": epoch + 1, "weights": weights.copy(), "grad": grad.copy()})
        return {"weights": weights, "history": history}
