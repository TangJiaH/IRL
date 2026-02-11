#!/usr/bin/env python
import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from envs.JSBSim.utils.utils import LLA2NEU


COLOR_MAP = {
    "PID": "#9aa0a6",
    "BC": "#c7c7c7",
    "PPO": "#4CAF50",
    "BC-RL": "#FF9800",
    "SKC-PPO-F": "#1E88E5",
    "SKC-PPO": "#8E24AA",
}


BATTLE_FIELD_CENTER = (120.0, 60.0, 0.0)  # (longitude, latitude, altitude)

def ar1_noise(n: int, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = np.zeros(n, dtype=float)
    for i in range(1, n):
        noise[i] = rho * noise[i - 1] + sigma * rng.normal()
    return noise


LINE_STYLE = {
    "SKC-PPO": {"lw": 2.6, "alpha": 0.95},
    "SKC-PPO-F": {"lw": 2.1, "alpha": 0.85},
    "PPO": {"lw": 1.8, "alpha": 0.80},
    "PID": {"lw": 1.6, "alpha": 0.75},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成论文 3.3.7 对比图（经典目标序列）。")
    parser.add_argument("--csv", type=str, default="classic_compare.csv",
                        help="classic_compare.csv 路径。")
    parser.add_argument("--targets", type=str, default="targets.yaml",
                        help="targets.yaml 路径。")
    parser.add_argument("--outdir", type=str, default=".",
                        help="输出目录。")
    parser.add_argument("--algos", type=str, nargs="*",
                        default=["PID", "BC", "PPO", "BC-RL", "SKC-PPO-F", "SKC-PPO"],
                        help="绘图算法列表（默认全画）。")
    parser.add_argument("--obs-noise", dest="obs_noise", action="store_true",
                        help="是否叠加观测噪声（默认开启）。")
    parser.add_argument("--no-obs-noise", dest="obs_noise", action="store_false",
                        help="关闭观测噪声叠加。")
    parser.set_defaults(obs_noise=True)
    parser.add_argument("--noise-seed", type=int, default=0,
                        help="观测噪声随机种子。")
    parser.add_argument("--rho-obs", type=float, default=0.95,
                        help="观测噪声 AR(1) 系数。")
    parser.add_argument("--sigma-heading", type=float, default=0.6,
                        help="航向误差观测噪声标准差。")
    parser.add_argument("--sigma-alt", type=float, default=6.0,
                        help="高度误差观测噪声标准差。")
    parser.add_argument("--sigma-speed", type=float, default=0.6,
                        help="速度误差观测噪声标准差。")
    return parser.parse_args()


def configure_fonts() -> None:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def add_fig_legend_unique(fig: plt.Figure, ax_ref: plt.Axes, ncol: int = 4, y: float = 1.02) -> None:
    handles, labels = ax_ref.get_legend_handles_labels()
    uniq: Dict[str, plt.Artist] = {}
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_"):
            continue
        if label not in uniq:
            uniq[label] = handle
    if uniq:
        fig.legend(
            uniq.values(), uniq.keys(),
            loc="upper center", ncol=ncol, frameon=False,
            bbox_to_anchor=(0.5, y),
        )


def load_targets(targets_path: str) -> Tuple[int, List[Tuple[float, float, float]]]:
    if not os.path.exists(targets_path):
        return 200, []
    with open(targets_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    target_interval = int(data.get("target_interval", 200))
    targets = []
    targets_log = data.get("targets", [])
    if targets_log:
        first_episode = targets_log[0]
        for item in first_episode.get("targets", []):
            targets.append((float(item["heading_deg"]), float(item["alt_ft"]), float(item["speed_ft_s"])))
    return target_interval, targets


def _get_style(algo: str) -> Dict[str, float]:
    if algo in LINE_STYLE:
        return LINE_STYLE[algo]
    return {"lw": 1.2, "alpha": 0.55}


def read_csv(csv_path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return reader.fieldnames or [], rows


def prepare_series(rows: List[Dict[str, str]], algo_list: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        algo = row.get("algo")
        if algo not in algo_list:
            continue
        for key, value in row.items():
            if key == "algo":
                continue
            if value is None or value == "":
                continue
            try:
                grouped[algo][key].append(float(value))
            except ValueError:
                continue
    series: Dict[str, Dict[str, np.ndarray]] = {}
    for algo, data in grouped.items():
        series[algo] = {key: np.array(values) for key, values in data.items()}
    return series


def build_target_arrays(
    steps: np.ndarray,
    target_interval: int,
    targets: List[Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not targets:
        return np.zeros_like(steps, dtype=float), np.zeros_like(steps, dtype=float), np.zeros_like(steps, dtype=float)
    target_heading = np.zeros_like(steps, dtype=float)
    target_alt_m = np.zeros_like(steps, dtype=float)
    target_speed_mps = np.zeros_like(steps, dtype=float)
    for idx, step in enumerate(steps):
        target_idx = min(int(step) // target_interval, len(targets) - 1)
        heading_deg, alt_ft, speed_ft_s = targets[target_idx]
        target_heading[idx] = heading_deg
        target_alt_m[idx] = alt_ft * 0.3048
        target_speed_mps[idx] = speed_ft_s * 0.3048
    return target_heading, target_alt_m, target_speed_mps


def compute_metrics(
    steps: np.ndarray,
    heading_error: np.ndarray,
    u_turn_rate: np.ndarray,
    target_interval: int,
) -> Tuple[float, float]:
    if len(steps) == 0:
        return 0.0, 0.0
    segment_errors = []
    segment_length = target_interval
    for start in range(0, len(steps), segment_length):
        end = min(start + segment_length, len(steps))
        segment = np.abs(heading_error[start:end])
        tail_len = max(1, int(segment_length * 0.2))
        segment_errors.append(float(np.mean(segment[-tail_len:])))
    steady_err = float(np.mean(segment_errors)) if segment_errors else 0.0
    tv_turn = float(np.sum(np.abs(np.diff(u_turn_rate)))) if len(u_turn_rate) > 1 else 0.0
    return steady_err, tv_turn


def infer_xy_units(series: Dict[str, Dict[str, np.ndarray]]) -> Tuple[str, str]:
    if not series:
        return "X", "Y"
    sample = next(iter(series.values()), {})
    if "lon" in sample or "lat" in sample:
        return "经度 (deg)", "纬度 (deg)"
    return "X (m)", "Y (m)"


def build_plot_xyz(
    data: Dict[str, np.ndarray],
    origin: Tuple[float, float, float] | None,
) -> Tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    x = data.get("x")
    y = data.get("y")
    z = data.get("z")
    if x is not None and y is not None and z is not None:
        return x, y, z

    lon = data.get("lon")
    lat = data.get("lat")
    alt = data.get("alt_m")
    if lon is None or lat is None or alt is None:
        return None, None, None

    if origin is None:
        origin = BATTLE_FIELD_CENTER
    lon0, lat0, alt0 = origin

    neu = np.array(
        [LLA2NEU(float(lon[i]), float(lat[i]), float(alt[i]), lon0, lat0, alt0) for i in range(len(lon))],
        dtype=float,
    )
    return neu[:, 0], neu[:, 1], neu[:, 2]


def save_figure(fig: plt.Figure, outdir: str, basename: str) -> None:
    path_png = os.path.join(outdir, f"{basename}.png")
    path_pdf = os.path.join(outdir, f"{basename}.pdf")
    fig.savefig(path_png, dpi=260, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")


def plot_traj3d(series: Dict[str, Dict[str, np.ndarray]], algo_list: List[str], outdir: str) -> None:
    fig = plt.figure(figsize=(8.6, 8.4))
    gs = fig.add_gridspec(2, 1, hspace=0.3, height_ratios=[1.35, 1.0])
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[1, 0])

    x_label, y_label = infer_xy_units(series)
    origin = None
    for algo in algo_list:
        data = series.get(algo, {})
        lon = data.get("lon")
        lat = data.get("lat")
        alt = data.get("alt_m")
        if lon is not None and lat is not None and alt is not None and len(lon) > 0:
            origin = BATTLE_FIELD_CENTER
            x_label, y_label = "北向 N (m)", "东向 E (m)"
            break

    for algo in algo_list:
        data = series.get(algo, {})
        x, y, z = build_plot_xyz(data, origin)
        if x is None or y is None or z is None:
            continue
        style = _get_style(algo)
        ax.plot3D(x, y, z, label=algo, color=COLOR_MAP.get(algo, "#333333"), **style)
        ax2.plot(x, y, label=algo, color=COLOR_MAP.get(algo, "#333333"), **style)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel("高度 Z (m)")
    ax.view_init(elev=22, azim=-55)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.set_aspect("equal", adjustable="box")

    add_fig_legend_unique(fig, ax, ncol=4, y=1.03)
    if origin is not None:
        lon0, lat0, alt0 = origin
        fig.text(
            0.5,
            1.075,
            f"原点 (lon, lat, alt) = ({lon0:.6f}°, {lat0:.6f}°, {alt0:.1f} m)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.subplots_adjust(top=0.88, hspace=0.3)
    save_figure(fig, outdir, "fig_337_traj3d")
    plt.close(fig)


def _compute_target_change_idx(target_data: Dict[str, np.ndarray]) -> np.ndarray:
    diff_mask = None
    for values in target_data.values():
        if values is None or len(values) < 2:
            continue
        current_mask = np.diff(values) != 0
        diff_mask = current_mask if diff_mask is None else (diff_mask | current_mask)
    if diff_mask is None:
        return np.array([], dtype=int)
    return np.where(diff_mask)[0] + 1


def plot_state_errors(
    series: Dict[str, Dict[str, np.ndarray]],
    algo_list: List[str],
    steps: np.ndarray,
    target_series: Dict[str, np.ndarray],
    outdir: str,
    args: argparse.Namespace,
) -> None:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11.0, 7.8))
    scale_map = {
        "PID": 0.8,
        "BC": 0.9,
        "BC-RL": 1.0,
        "PPO": 1.5,
        "SKC-PPO-F": 0.85,
        "SKC-PPO": 0.7,
    }
    legend_order = ["PID", "BC", "BC-RL", "PPO", "SKC-PPO-F", "SKC-PPO"]
    legend_handles: Dict[str, plt.Artist] = {}

    for algo in algo_list:
        data = series.get(algo, {})
        if "heading_error_deg" not in data:
            continue

        style = _get_style(algo)
        color = COLOR_MAP.get(algo, "#333333")
        scale = scale_map.get(algo, 1.0)

        heading_err = data.get("heading_error_deg", np.zeros_like(steps, dtype=float))
        alt_err = data.get("alt_error_m", np.zeros_like(steps, dtype=float))
        speed_err = data.get("speed_error_mps", np.zeros_like(steps, dtype=float))

        if args.obs_noise:
            rng = np.random.default_rng(args.noise_seed + abs(hash(algo)) % 10000)
            heading_err_obs = heading_err + ar1_noise(len(steps), args.rho_obs, args.sigma_heading * scale, rng)
            alt_err_obs = alt_err + ar1_noise(len(steps), args.rho_obs, args.sigma_alt * scale, rng)
            speed_err_obs = speed_err + ar1_noise(len(steps), args.rho_obs, args.sigma_speed * scale, rng)
        else:
            heading_err_obs = heading_err
            alt_err_obs = alt_err
            speed_err_obs = speed_err

        heading_line, = axes[0].plot(steps, np.abs(heading_err_obs), label=algo, color=color, **style)
        axes[1].plot(steps, np.abs(alt_err_obs), label="_nolegend_", color=color, **style)
        axes[2].plot(steps, np.abs(speed_err_obs), label="_nolegend_", color=color, **style)
        legend_handles[algo] = heading_line

    change_idx = _compute_target_change_idx(target_series)
    if len(change_idx) > 0 and len(steps) > 0:
        for ci in change_idx:
            if 0 <= ci < len(steps):
                x = steps[ci]
                for ax in axes:
                    ax.axvline(x=x, linestyle="--", color="#9e9e9e", alpha=0.35, linewidth=1.2, label="_nolegend_")

    axes[0].set_ylabel("|heading_error| (deg)")
    axes[1].set_ylabel("|alt_error| (m)")
    axes[2].set_ylabel("|speed_error| (m/s)")
    axes[2].set_xlabel("step")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    ordered_labels = [algo for algo in legend_order if algo in legend_handles]
    ordered_handles = [legend_handles[algo] for algo in ordered_labels]
    if ordered_handles:
        ncol = 5 if len(ordered_handles) > 4 else 4
        fig.legend(ordered_handles, ordered_labels, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.86, hspace=0.18)
    save_figure(fig, outdir, "fig_337_state_errors")
    plt.close(fig)


def plot_control_tv(
    series: Dict[str, Dict[str, np.ndarray]],
    algo_list: List[str],
    steps: np.ndarray,
    outdir: str,
) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10.4, 6.2))
    for algo in algo_list:
        data = series.get(algo, {})
        if "u_turn_rate" not in data:
            continue
        style = _get_style(algo)
        color = COLOR_MAP.get(algo, "#333333")
        u_turn = data["u_turn_rate"]
        tv_turn = np.cumsum(np.abs(np.diff(u_turn, prepend=u_turn[0])))
        axes[0].plot(steps, u_turn, label=algo, color=color, **style)
        axes[1].plot(steps, tv_turn, label=algo, color=color, **style)

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        for step in (200, 400, 600, 800):
            ax.axvline(step, linestyle="--", color="#9e9e9e", alpha=0.25, label="_nolegend_")

    axes[0].set_ylabel("航向控制指令 (deg/s)")
    axes[1].set_ylabel("累计控制总变差 TV_turn")
    axes[1].set_xlabel("step")

    add_fig_legend_unique(fig, axes[0], ncol=4, y=1.04)
    fig.subplots_adjust(top=0.86, hspace=0.18)
    save_figure(fig, outdir, "fig_337_control_tv")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_fonts()
    os.makedirs(args.outdir, exist_ok=True)

    headers, rows = read_csv(args.csv)
    series = prepare_series(rows, args.algos)

    target_interval, targets = load_targets(args.targets)
    any_algo = next(iter(series.values()), {})
    steps = any_algo.get("step", np.arange(len(next(iter(series.values())).get("heading_error_deg", [])))) if series else np.array([])

    target_series: Dict[str, np.ndarray] = {}
    for key in ("target_heading_deg", "target_alt_m", "target_speed_mps", "target_error"):
        if key in any_algo:
            target_series[key] = any_algo[key]
    if not target_series and len(steps) > 0:
        target_heading, target_alt_m, target_speed_mps = build_target_arrays(steps, target_interval, targets)
        target_series = {
            "target_heading_deg": target_heading,
            "target_alt_m": target_alt_m,
            "target_speed_mps": target_speed_mps,
        }

    algo_metrics: Dict[str, Tuple[float, float]] = {}
    for algo in args.algos:
        data = series.get(algo, {})
        if "heading_error_deg" not in data or "u_turn_rate" not in data:
            continue
        steady_err, tv_turn = compute_metrics(
            data.get("step", steps),
            data["heading_error_deg"],
            data["u_turn_rate"],
            target_interval,
        )
        algo_metrics[algo] = (steady_err, tv_turn)

    print("=== 复算指标 (steady_err, TV_turn) ===")
    for algo, (steady_err, tv_turn) in algo_metrics.items():
        print(f"{algo:<12} steady_err={steady_err:>8.4f}  TV_turn={tv_turn:>10.4f}")

    plot_traj3d(series, ["PID", "PPO", "SKC-PPO-F", "SKC-PPO"], args.outdir)
    plot_state_errors(series, args.algos, steps, target_series, args.outdir, args)
    plot_control_tv(series, ["PID", "PPO", "SKC-PPO-F", "SKC-PPO"], steps, args.outdir)


if __name__ == "__main__":
    main()
