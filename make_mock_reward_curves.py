#!/usr/bin/env python3
"""Generate mock reward curves and convergence statistics for 4 algorithms."""

from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVAL_INTERVAL = 32_000
TOTAL_STEPS = 20_000_000
N_SEEDS = 5
TAIL_POINTS = 32
CONVERGENCE_N = 5
RNG_SEED = 20250205
RHO = 0.85
TAU = 4e6


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    final_mean: float
    seed_std_target: float
    k: float
    sigma_start: float
    sigma_floor: float


ALGO_SPECS = [
    # sigma_floor kept in requested ranges, PPO upper side for stronger tail fluctuations
    AlgoSpec("PPO", final_mean=374.6, seed_std_target=22.9, k=1.9e-7, sigma_start=60.0, sigma_floor=24.0),
    AlgoSpec("BC-RL", final_mean=498.0, seed_std_target=38.0, k=2.35e-7, sigma_start=55.0, sigma_floor=18.0),
    # SKC sigma_start reduced by ~15%
    AlgoSpec("SKC-PPO-F", final_mean=545.0, seed_std_target=34.0, k=2.85e-7, sigma_start=38.25, sigma_floor=14.0),
    AlgoSpec("SKC-PPO", final_mean=588.0, seed_std_target=29.0, k=3.25e-7, sigma_start=34.0, sigma_floor=12.0),
]


def make_dropout(steps: np.ndarray, center: float, width: float, depth: float) -> np.ndarray:
    return -depth * np.exp(-0.5 * ((steps - center) / width) ** 2)


def generate_ar1_noise(sigmas: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    noise = np.zeros_like(sigmas, dtype=float)
    scale = np.sqrt(1.0 - rho**2)
    for i in range(1, len(sigmas)):
        noise[i] = rho * noise[i - 1] + scale * rng.normal(0.0, sigmas[i])
    return noise


def cap_tail_volatility(
    reward: np.ndarray,
    base: np.ndarray,
    target_std: float,
    tail_points: int = TAIL_POINTS,
    trigger_ratio: float = 1.15,
    cap_ratio: float = 1.05,
) -> np.ndarray:
    out = reward.copy()
    tail = out[-tail_points:]
    cur_std = float(np.std(tail, ddof=0))
    if cur_std <= trigger_ratio * target_std:
        return out

    base_tail = base[-tail_points:]
    noise_tail = tail - base_tail
    centered = noise_tail - np.mean(noise_tail)
    centered_std = float(np.std(centered, ddof=0))
    if centered_std < 1e-12:
        return out

    factor = (cap_ratio * target_std) / cur_std
    new_noise_tail = np.mean(noise_tail) + factor * centered
    out[-tail_points:] = base_tail + new_noise_tail
    return out


def enforce_skc_tail_not_exceed_mid(reward: np.ndarray, base: np.ndarray, steps: np.ndarray) -> np.ndarray:
    out = reward.copy()
    mid_mask = (steps >= 8_000_000) & (steps <= 12_000_000)
    mid_std = float(np.std(out[mid_mask], ddof=0))
    tail = out[-TAIL_POINTS:]
    tail_std = float(np.std(tail, ddof=0))
    if tail_std <= mid_std:
        return out

    base_tail = base[-TAIL_POINTS:]
    noise_tail = tail - base_tail
    centered = noise_tail - np.mean(noise_tail)
    centered_std = float(np.std(centered, ddof=0))
    if centered_std < 1e-12:
        return out

    factor = max(0.0, (0.95 * mid_std) / tail_std)
    out[-TAIL_POINTS:] = base_tail + np.mean(noise_tail) + factor * centered
    return out


def find_convergence_step(reward: np.ndarray, steps: np.ndarray, threshold: float, n_keep: int = CONVERGENCE_N) -> float:
    flags = reward >= threshold
    for i in range(len(flags) - n_keep + 1):
        if np.all(flags[i : i + n_keep]):
            return float(steps[i])
    return math.nan


def sample_seed_plateau(target_mean: float, target_seed_std: float, rng: np.random.Generator) -> float:
    raw = rng.normal(target_mean, target_seed_std)
    lo = target_mean - 2.0 * target_seed_std
    hi = target_mean + 2.0 * target_seed_std
    return float(np.clip(raw, lo, hi))


def summarize_last1m(
    algo_name: str,
    curves: list[np.ndarray],
    target_mean: float,
    target_std: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    seed_tail_means = []
    seed_tail_stds = []

    for seed_idx, curve in enumerate(curves):
        m = float(np.mean(curve[-TAIL_POINTS:]))
        s = float(np.std(curve[-TAIL_POINTS:], ddof=0))
        seed_tail_means.append(m)
        seed_tail_stds.append(s)
        rows.append(
            {
                "algo": algo_name,
                "seed": seed_idx,
                "last32_mean": m,
                "last32_std": s,
                "target_mean": target_mean,
                "target_std": target_std,
            }
        )

    rows.append(
        {
            "algo": algo_name,
            "seed": "ALL",
            "last32_mean": float(np.mean(seed_tail_means)),
            "last32_std": float(np.std(seed_tail_means, ddof=0)),
            "target_mean": target_mean,
            "target_std": target_std,
        }
    )
    return pd.DataFrame(rows)




def plot_reward_with_convergence(
    rewards_csv: str = "mock_rewards.csv",
    convergence_csv: str = "convergence_steps.csv",
    output_path: str = "fig_reward_curves_with_convergence.png",
) -> None:
    """Plot mean±std reward curves with threshold/convergence visual cues."""
    df = pd.read_csv(rewards_csv)
    conv_df = pd.read_csv(convergence_csv)

    if "conv_mean" in conv_df.columns:
        conv_mean_col = "conv_mean"
    elif "convergence_mean" in conv_df.columns:
        conv_mean_col = "convergence_mean"
    else:
        # fallback: compute from per-seed convergence records
        conv_df = conv_df.groupby("algo", as_index=False)["convergence_step"].mean()
        conv_mean_col = "convergence_step"

    color_map = {
        "PPO": "#2ca02c",
        "BC-RL": "#ff7f0e",
        "SKC-PPO-F": "#1f77b4",
        "SKC-PPO": "#9467bd",
    }
    order = [spec.name for spec in ALGO_SPECS]

    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    note_lines = ["收敛步数（0.95·μ_final，N=5）："]

    for algo in order:
        sub = df[df["algo"] == algo].copy()
        pivot = sub.pivot(index="step", columns="seed", values="reward").sort_index()
        x = pivot.index.to_numpy()
        curves = pivot.to_numpy().T
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=0)
        c = color_map[algo]

        for curve in curves:
            ax.plot(x, curve, color=c, alpha=0.18, linewidth=1.0)
        ax.plot(x, mean, color=c, linewidth=2.6, label=algo)
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.14)

        # theta = 0.95 * final_mean, final_mean computed from last 32 eval points in csv
        last_steps = np.sort(sub["step"].unique())[-TAIL_POINTS:]
        final_mean = float(sub[sub["step"].isin(last_steps)]["reward"].mean())
        theta = 0.95 * final_mean
        ax.axhline(theta, color=c, linestyle="--", linewidth=1.1, alpha=0.35)

        conv_row = conv_df[conv_df["algo"] == algo]
        if len(conv_row) == 0:
            continue
        conv_mean = float(conv_row.iloc[0][conv_mean_col])

        ax.axvline(conv_mean, color=c, linestyle="--", linewidth=1.4, alpha=0.65)
        note_lines.append(f"{algo}：{conv_mean / 1e6:.2f}×10⁶")

    ax.text(
        0.985,
        0.98,
        "\n".join(note_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="#999999"),
    )

    ax.set_xlim(0, TOTAL_STEPS)
    ax.set_xlabel("环境交互步数", fontsize=13)
    ax.set_ylabel("平均回合奖励", fontsize=13)
    ax.set_title("训练奖励曲线及收敛步数对比", fontsize=15)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    plt.close(fig)

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    steps = np.arange(0, TOTAL_STEPS + EVAL_INTERVAL, EVAL_INTERVAL, dtype=np.int64)

    algo_seed_curves: dict[str, list[np.ndarray]] = {spec.name: [] for spec in ALGO_SPECS}

    for spec in ALGO_SPECS:
        tmp_curves = []

        for _seed in range(N_SEEDS):
            # Use per-seed plateau to induce seed-level spread in last-1M means.
            L_seed = sample_seed_plateau(spec.final_mean, spec.seed_std_target, rng)
            k = spec.k * (1.0 + rng.normal(0, 0.07))
            trend = L_seed * (1.0 - np.exp(-k * steps))

            sigmas = spec.sigma_start * np.exp(-steps / TAU) + spec.sigma_floor
            noise = generate_ar1_noise(sigmas=sigmas, rho=RHO, rng=rng)

            event = np.zeros_like(trend)
            if spec.name == "PPO":
                c = 15_000_000 + rng.normal(0, 220_000)
                w = 300_000 + rng.uniform(-40_000, 50_000)
                d = rng.uniform(40, 60)
                event += make_dropout(steps, center=c, width=w, depth=d)
            elif spec.name == "BC-RL":
                c = rng.uniform(13_500_000, 18_500_000)
                w = rng.uniform(170_000, 320_000)
                d = rng.uniform(15, 38)
                event += make_dropout(steps, center=c, width=w, depth=d)

            base = trend + event
            reward = np.maximum(base + noise, -20.0)
            reward = cap_tail_volatility(reward=reward, base=base, target_std=spec.seed_std_target)

            if spec.name in {"SKC-PPO", "SKC-PPO-F"}:
                reward = enforce_skc_tail_not_exceed_mid(reward=reward, base=base, steps=steps)

            tmp_curves.append(reward)

        # Algorithm-level single shift only: align average last-1M mean across 5 seeds.
        seed_last_means = np.array([np.mean(curve[-TAIL_POINTS:]) for curve in tmp_curves], dtype=float)
        group_mean = float(seed_last_means.mean())
        delta = spec.final_mean - group_mean
        algo_seed_curves[spec.name] = [curve + delta for curve in tmp_curves]

    all_rows: list[dict[str, float | int | str]] = []
    for spec in ALGO_SPECS:
        for seed_idx, curve in enumerate(algo_seed_curves[spec.name]):
            all_rows.extend(
                {
                    "step": int(s),
                    "algo": spec.name,
                    "seed": seed_idx,
                    "reward": float(r),
                }
                for s, r in zip(steps, curve)
            )

    df = pd.DataFrame(all_rows)
    df.to_csv("mock_rewards.csv", index=False)

    # Last-1M acceptance table.
    stat_frames = []
    for spec in ALGO_SPECS:
        stat_frames.append(
            summarize_last1m(
                algo_name=spec.name,
                curves=algo_seed_curves[spec.name],
                target_mean=spec.final_mean,
                target_std=spec.seed_std_target,
            )
        )
    stats_df = pd.concat(stat_frames, ignore_index=True)
    stats_df.to_csv("last1m_seed_stats.csv", index=False)

    # Main reward curve figure (PNG as requested).
    fig, ax = plt.subplots(figsize=(11, 6.5))
    color_map = {
        "PPO": "#2ca02c",
        "BC-RL": "#ff7f0e",
        "SKC-PPO-F": "#1f77b4",
        "SKC-PPO": "#9467bd",
    }

    for spec in ALGO_SPECS:
        curves = np.stack(algo_seed_curves[spec.name], axis=0)
        mean = curves.mean(axis=0)
        std = curves.std(axis=0, ddof=0)
        c = color_map[spec.name]

        for curve in curves:
            ax.plot(steps, curve, color=c, alpha=0.2, linewidth=1.2)
        ax.plot(steps, mean, color=c, linewidth=2.4, label=spec.name)
        ax.fill_between(steps, mean - std, mean + std, color=c, alpha=0.15)

    ax.set_xlim(0, TOTAL_STEPS)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("Mock Training Reward Curves")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("fig_reward_curves.png", dpi=220)
    plt.close(fig)

    # Convergence stats summary.
    conv_rows = []
    for spec in ALGO_SPECS:
        threshold = 0.95 * spec.final_mean
        for seed_idx, curve in enumerate(algo_seed_curves[spec.name]):
            cstep = find_convergence_step(curve, steps, threshold=threshold, n_keep=CONVERGENCE_N)
            conv_rows.append(
                {
                    "algo": spec.name,
                    "seed": seed_idx,
                    "threshold": threshold,
                    "convergence_step": cstep,
                }
            )

    conv_df = pd.DataFrame(conv_rows)
    summary_df = (
        conv_df.groupby("algo", as_index=False)["convergence_step"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "convergence_mean", "std": "convergence_std"})
    )
    summary_df.to_csv("convergence_steps.csv", index=False)

    plot_reward_with_convergence(
        rewards_csv="mock_rewards.csv",
        convergence_csv="convergence_steps.csv",
        output_path="fig_reward_curves_with_convergence.png",
    )

    print("Saved files:")
    print("  - mock_rewards.csv")
    print("  - fig_reward_curves.png")
    print("  - fig_reward_curves_with_convergence.png")
    print("  - convergence_steps.csv")
    print("  - last1m_seed_stats.csv")


if __name__ == "__main__":
    main()
