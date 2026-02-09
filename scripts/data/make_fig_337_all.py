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


COLOR_MAP = {
    "PID": "#9aa0a6",
    "BC": "#c7c7c7",
    "PPO": "#4CAF50",
    "BC-RL": "#FF9800",
    "SKC-PPO-F": "#1E88E5",
    "SKC-PPO": "#8E24AA",
}


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
    return parser.parse_args()


def configure_fonts() -> None:
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    matplotlib.rcParams["axes.unicode_minus"] = False


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


def plot_traj3d(series: Dict[str, Dict[str, np.ndarray]], algo_list: List[str], outdir: str) -> None:
    fig = plt.figure(figsize=(11, 4.8))
    gs = fig.add_gridspec(1, 2, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1])

    for algo in algo_list:
        data = series.get(algo, {})
        x = data.get("x") or data.get("lon")
        y = data.get("y") or data.get("lat")
        z = data.get("z") or data.get("alt_m")
        if x is None or y is None or z is None:
            continue
        style = _get_style(algo)
        ax.plot3D(x, y, z, label=algo, color=COLOR_MAP.get(algo, "#333333"), **style)
        ax2.plot(x, y, label=algo, color=COLOR_MAP.get(algo, "#333333"), **style)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("高度Z")
    ax.view_init(elev=18, azim=-60)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_aspect("equal", adjustable="box")

    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_337_traj3d.png"), dpi=300)
    fig.savefig(os.path.join(outdir, "fig_337_traj3d.pdf"))
    plt.close(fig)


def plot_state_errors(
    series: Dict[str, Dict[str, np.ndarray]],
    algo_list: List[str],
    steps: np.ndarray,
    target_heading: np.ndarray,
    target_alt_m: np.ndarray,
    target_speed_mps: np.ndarray,
    outdir: str,
) -> None:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    for algo in algo_list:
        data = series.get(algo, {})
        style = _get_style(algo)
        color = COLOR_MAP.get(algo, "#333333")
        if "heading_error_deg" in data:
            axes[0].plot(steps, np.abs(data["heading_error_deg"]), label=algo, color=color, **style)
        if "alt_error_m" in data:
            axes[1].plot(steps, data["alt_error_m"], label=algo, color=color, **style)
        if "speed_error_mps" in data:
            axes[2].plot(steps, data["speed_error_mps"], label=algo, color=color, **style)

    axes[0].plot(steps, target_heading, linestyle="--", color="#9e9e9e", linewidth=1.2, label="target")
    axes[1].plot(steps, target_alt_m, linestyle="--", color="#9e9e9e", linewidth=1.2, label="target")
    axes[2].plot(steps, target_speed_mps, linestyle="--", color="#9e9e9e", linewidth=1.2, label="target")

    axes[0].set_ylabel("|heading_error| (deg)")
    axes[1].set_ylabel("alt_error (m)")
    axes[2].set_ylabel("speed_error (m/s)")
    axes[2].set_xlabel("step")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_337_state_errors.png"), dpi=300)
    fig.savefig(os.path.join(outdir, "fig_337_state_errors.pdf"))
    plt.close(fig)


def plot_control_tv(
    series: Dict[str, Dict[str, np.ndarray]],
    algo_list: List[str],
    steps: np.ndarray,
    outdir: str,
) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
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

    axes[0].set_ylabel("航向控制率 (deg/s)")
    axes[1].set_ylabel("TV_turn")
    axes[1].set_xlabel("step")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_337_control_tv.png"), dpi=300)
    fig.savefig(os.path.join(outdir, "fig_337_control_tv.pdf"))
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

    if "target_heading_deg" in any_algo:
        target_heading = any_algo["target_heading_deg"]
    else:
        target_heading, _, _ = build_target_arrays(steps, target_interval, targets)

    if "target_alt_m" in any_algo:
        target_alt_m = any_algo["target_alt_m"]
    else:
        _, target_alt_m, _ = build_target_arrays(steps, target_interval, targets)

    if "target_speed_mps" in any_algo:
        target_speed_mps = any_algo["target_speed_mps"]
    else:
        _, _, target_speed_mps = build_target_arrays(steps, target_interval, targets)

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
    plot_state_errors(series, args.algos, steps, target_heading, target_alt_m, target_speed_mps, args.outdir)
    plot_control_tv(series, ["PID", "PPO", "SKC-PPO-F", "SKC-PPO"], steps, args.outdir)


if __name__ == "__main__":
    main()
