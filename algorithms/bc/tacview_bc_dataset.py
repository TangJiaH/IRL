import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import io
import numpy as np
import torch

from envs.JSBSim.utils.utils import LLA2NEU


@dataclass(frozen=True)
class TacviewBCConfig:
    roll_rate_limit: float = 1.2
    pitch_rate_limit: float = 0.8
    yaw_rate_limit: float = 0.8
    speed_rate_limit: float = 15.0
    min_dt: float = 0.2
    stride: int = 1
    max_samples: Optional[int] = None


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


def _wrap_heading_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _ned_to_body(v_north: float, v_east: float, v_down: float, roll_rad: float, pitch_rad: float, yaw_rad: float) -> Tuple[float, float, float]:
    cr = math.cos(roll_rad)
    sr = math.sin(roll_rad)
    cp = math.cos(pitch_rad)
    sp = math.sin(pitch_rad)
    cy = math.cos(yaw_rad)
    sy = math.sin(yaw_rad)

    r11 = cp * cy
    r12 = cp * sy
    r13 = -sp
    r21 = sr * sp * cy - cr * sy
    r22 = sr * sp * sy + cr * cy
    r23 = sr * cp
    r31 = cr * sp * cy + sr * sy
    r32 = cr * sp * sy - sr * cy
    r33 = cr * cp

    u = r11 * v_north + r12 * v_east + r13 * v_down
    v = r21 * v_north + r22 * v_east + r23 * v_down
    w = r31 * v_north + r32 * v_east + r33 * v_down
    return u, v, w


def _parse_target_comment(line: str) -> Optional[Tuple[float, float, float]]:
    if not line.startswith("//TARGET"):
        return None
    payload = line[len("//TARGET"):].strip()
    if payload.startswith(":") or payload.startswith("="):
        payload = payload[1:].strip()
    tokens = payload.replace(",", " ").split()
    values = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed = _safe_float(value)
        if parsed is not None:
            values[key.strip().lower()] = parsed
    heading = values.get("heading_deg")
    altitude = values.get("alt_m")
    speed = values.get("speed_mps")
    if heading is None or altitude is None or speed is None:
        return None
    return heading, altitude, speed


def parse_acmi_file(
    path: Path,
) -> List[Tuple[float, float, float, float, float, float, float, Optional[float], Optional[float], Optional[float]]]:
    states = []
    current_time = 0.0
    target_heading = None
    target_alt_m = None
    target_speed_mps = None
    for raw_line in _read_text_lines(path):
        line = raw_line.strip().lstrip("\ufeff")
        if not line:
            continue
        target_values = _parse_target_comment(line)
        if target_values is not None:
            target_heading, target_alt_m, target_speed_mps = target_values
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
        roll = _safe_float(fields[3]) if len(fields) > 3 else 0.0
        pitch = _safe_float(fields[4]) if len(fields) > 4 else 0.0
        yaw = _safe_float(fields[5]) if len(fields) > 5 else None
        states.append(
            (
                current_time,
                lon,
                lat,
                alt,
                roll or 0.0,
                pitch or 0.0,
                yaw or 0.0,
                target_heading,
                target_alt_m,
                target_speed_mps,
            )
        )
    return states


def parse_csv_file(
    path: Path,
) -> List[Tuple[float, float, float, float, float, float, float, Optional[float], Optional[float], Optional[float]]]:
    states = []
    csv_text = "\n".join(_read_text_lines(path))
    reader = csv.DictReader(io.StringIO(csv_text, newline=""))
    for row in reader:
        time_value = _safe_float(row.get("Unix time") or row.get("Time") or row.get("Timestamp"))
        if time_value is None:
            continue
        lon = _safe_float(row.get("Longitude"))
        lat = _safe_float(row.get("Latitude"))
        alt = _safe_float(row.get("Altitude"))
        if lon is None or lat is None or alt is None:
            continue
        roll = _safe_float(row.get("Roll")) or 0.0
        pitch = _safe_float(row.get("Pitch")) or 0.0
        yaw = _safe_float(row.get("Yaw") or row.get("Heading")) or 0.0
        states.append((time_value, lon, lat, alt, roll, pitch, yaw, None, None, None))
    return states


def _discretize(value: float, nvec: int, low: float, high: float) -> int:
    value = max(min(value, high), low)
    scaled = (value - low) / (high - low)
    return int(np.clip(round(scaled * (nvec - 1)), 0, nvec - 1))


def _build_samples(states: Sequence[Tuple[float, float, float, float, float, float, float, Optional[float], Optional[float], Optional[float]]],
                   config: TacviewBCConfig,
                   action_bins: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    if len(states) < 2:
        return np.empty((0, 12), dtype=np.float32), np.empty((0, 4), dtype=np.int64)

    times = np.array([s[0] for s in states], dtype=float)
    lon = np.array([s[1] for s in states], dtype=float)
    lat = np.array([s[2] for s in states], dtype=float)
    alt = np.array([s[3] for s in states], dtype=float)
    roll_deg = np.array([s[4] for s in states], dtype=float)
    pitch_deg = np.array([s[5] for s in states], dtype=float)
    yaw_deg = np.array([s[6] for s in states], dtype=float)
    target_heading_deg = np.array([s[7] if s[7] is not None else np.nan for s in states], dtype=float)
    target_alt_m = np.array([s[8] if s[8] is not None else np.nan for s in states], dtype=float)
    target_speed_mps = np.array([s[9] if s[9] is not None else np.nan for s in states], dtype=float)

    origin_lon, origin_lat, origin_alt = lon[0], lat[0], alt[0]
    positions = np.array(
        [LLA2NEU(lon[i], lat[i], alt[i], origin_lon, origin_lat, origin_alt) for i in range(len(states))],
        dtype=float,
    )

    dt = np.diff(times)
    dt = np.where(dt <= 0, config.min_dt, dt)
    deltas = np.diff(positions, axis=0)
    velocities = deltas / dt[:, None]
    speeds = np.linalg.norm(velocities, axis=1)

    headings = np.empty(len(states) - 1, dtype=float)
    for i in range(len(headings)):
        if not math.isnan(yaw_deg[i + 1]):
            headings[i] = yaw_deg[i + 1]
        else:
            headings[i] = _bearing_deg(lat[i], lon[i], lat[i + 1], lon[i + 1])

    default_target_heading = headings[-1]
    default_target_altitude = alt[-1]
    default_target_speed = speeds[-1]

    if np.isnan(target_heading_deg).all():
        target_heading_deg = np.full(len(states), default_target_heading, dtype=float)
    else:
        last_heading = target_heading_deg[~np.isnan(target_heading_deg)][0]
        for idx in range(len(target_heading_deg)):
            if np.isnan(target_heading_deg[idx]):
                target_heading_deg[idx] = last_heading
            else:
                last_heading = target_heading_deg[idx]

    if np.isnan(target_alt_m).all():
        target_alt_m = np.full(len(states), default_target_altitude, dtype=float)
    else:
        last_alt = target_alt_m[~np.isnan(target_alt_m)][0]
        for idx in range(len(target_alt_m)):
            if np.isnan(target_alt_m[idx]):
                target_alt_m[idx] = last_alt
            else:
                last_alt = target_alt_m[idx]

    if np.isnan(target_speed_mps).all():
        target_speed_mps = np.full(len(states), default_target_speed, dtype=float)
    else:
        last_speed = target_speed_mps[~np.isnan(target_speed_mps)][0]
        for idx in range(len(target_speed_mps)):
            if np.isnan(target_speed_mps[idx]):
                target_speed_mps[idx] = last_speed
            else:
                last_speed = target_speed_mps[idx]

    obs_list = []
    act_list = []
    for idx in range(len(headings) - 1):
        if config.stride > 1 and idx % config.stride != 0:
            continue
        target_heading = target_heading_deg[idx + 1]
        target_altitude = target_alt_m[idx + 1]
        target_speed = target_speed_mps[idx + 1]
        delta_heading = _wrap_heading_deg(target_heading - headings[idx])
        delta_altitude = target_altitude - alt[idx + 1]
        delta_speed = target_speed - speeds[idx]

        roll_rad = math.radians(roll_deg[idx + 1])
        pitch_rad = math.radians(pitch_deg[idx + 1])
        v_north, v_east, v_up = velocities[idx]
        v_down = -v_up
        yaw_rad = math.radians(headings[idx + 1])
        v_body_u, v_body_v, v_body_w = _ned_to_body(v_north, v_east, v_down, roll_rad, pitch_rad, yaw_rad)

        norm_obs = np.zeros(12, dtype=np.float32)
        norm_obs[0] = delta_altitude / 1000.0
        norm_obs[1] = math.radians(delta_heading)
        norm_obs[2] = delta_speed / 340.0
        norm_obs[3] = alt[idx + 1] / 5000.0
        norm_obs[4] = math.sin(roll_rad)
        norm_obs[5] = math.cos(roll_rad)
        norm_obs[6] = math.sin(pitch_rad)
        norm_obs[7] = math.cos(pitch_rad)
        norm_obs[8] = v_body_u / 340.0
        norm_obs[9] = v_body_v / 340.0
        norm_obs[10] = v_body_w / 340.0
        norm_obs[11] = speeds[idx] / 340.0
        norm_obs = np.clip(norm_obs, -10.0, 10.0)

        roll_rate = math.radians(_wrap_heading_deg(roll_deg[idx + 1] - roll_deg[idx])) / dt[idx]
        pitch_rate = math.radians(_wrap_heading_deg(pitch_deg[idx + 1] - pitch_deg[idx])) / dt[idx]
        yaw_rate = math.radians(_wrap_heading_deg(headings[idx + 1] - headings[idx])) / dt[idx]
        speed_rate = (speeds[idx + 1] - speeds[idx]) / dt[idx]

        aileron_cmd = np.clip(roll_rate / config.roll_rate_limit, -1.0, 1.0)
        elevator_cmd = np.clip(pitch_rate / config.pitch_rate_limit, -1.0, 1.0)
        rudder_cmd = np.clip(yaw_rate / config.yaw_rate_limit, -1.0, 1.0)
        throttle_cmd = np.clip(0.65 + (speed_rate / config.speed_rate_limit) * 0.25, 0.4, 0.9)

        action = np.array([
            _discretize(aileron_cmd, action_bins[0], -1.0, 1.0),
            _discretize(elevator_cmd, action_bins[1], -1.0, 1.0),
            _discretize(rudder_cmd, action_bins[2], -1.0, 1.0),
            _discretize(throttle_cmd, action_bins[3], 0.4, 0.9),
        ], dtype=np.int64)

        obs_list.append(norm_obs)
        act_list.append(action)
        if config.max_samples is not None and len(obs_list) >= config.max_samples:
            break

    if not obs_list:
        return np.empty((0, 12), dtype=np.float32), np.empty((0, 4), dtype=np.int64)

    return np.stack(obs_list, axis=0), np.stack(act_list, axis=0)


def _iter_tacview_files(path: str) -> Iterable[Path]:
    base = Path(path)
    if base.is_file():
        yield base
        return
    files = sorted(base.glob("*.acmi")) + sorted(base.glob("*.csv"))
    if not files:
        files = sorted(base.rglob("*.acmi")) + sorted(base.rglob("*.csv"))
    for file_path in files:
        if file_path.name.endswith(".zip.acmi"):
            continue
        yield file_path


class TacviewBCDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, config: Optional[TacviewBCConfig] = None,
                 action_bins: Sequence[int] = (41, 41, 41, 30)) -> None:
        self.config = config or TacviewBCConfig()
        self.action_bins = action_bins
        self.obs = []
        self.actions = []
        for file_path in _iter_tacview_files(path):
            if file_path.suffix.lower() == ".csv":
                states = parse_csv_file(file_path)
            else:
                states = parse_acmi_file(file_path)
            obs, actions = _build_samples(states, self.config, self.action_bins)
            if obs.size == 0:
                continue
            self.obs.append(obs)
            self.actions.append(actions)
        if self.obs:
            self.obs = np.concatenate(self.obs, axis=0)
            self.actions = np.concatenate(self.actions, axis=0)
        else:
            self.obs = np.empty((0, 12), dtype=np.float32)
            self.actions = np.empty((0, 4), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.obs[idx]), torch.from_numpy(self.actions[idx])

    def add_samples(self, obs: np.ndarray, actions: np.ndarray) -> None:
        if obs.size == 0 or actions.size == 0:
            return
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        if self.obs.size == 0:
            self.obs = obs
            self.actions = actions
        else:
            self.obs = np.concatenate([self.obs, obs], axis=0)
            self.actions = np.concatenate([self.actions, actions], axis=0)
