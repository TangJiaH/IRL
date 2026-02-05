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
TAIL_POINTS = 32  # ~last 1M steps when eval_interval=32k
CONVERGENCE_N = 5
RNG_SEED = 20250205
RHO = 0.85
TAU = 4e6


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    final_mean: float
    final_std: float
    k: float
    sigma_start: float
    sigma_floor: float


ALGO_SPECS = [
    AlgoSpec("PPO", final_mean=374.6, final_std=22.9, k=1.9e-7, sigma_start=60.0, sigma_floor=20.0),
    AlgoSpec("BC-RL", final_mean=498.0, final_std=38.0, k=2.35e-7, sigma_start=55.0, sigma_floor=18.0),
    AlgoSpec("SKC-PPO-F", final_mean=545.0, final_std=34.0, k=2.85e-7, sigma_start=45.0, sigma_floor=14.0),
    AlgoSpec("SKC-PPO", final_mean=588.0, final_std=29.0, k=3.25e-7, sigma_start=40.0, sigma_floor=12.0),
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
    if np.std(centered, ddof=0) < 1e-12:
        return out

    factor = (cap_ratio * target_std) / cur_std
    new_noise_tail = np.mean(noise_tail) + factor * centered
    out[-tail_points:] = base_tail + new_noise_tail
    return out


def align_tail_mean_only(reward: np.ndarray, target_mean: float, tail_points: int = TAIL_POINTS) -> np.ndarray:
    m_last = float(np.mean(reward[-tail_points:]))
    return reward + (target_mean - m_last)


def enforce_skc_tail_not_exceed_mid(reward: np.ndarray, base: np.ndarray, tail_points: int = TAIL_POINTS) -> np.ndarray:
    out = reward.copy()
    # Mid region: 8M~12M, tail region: last 1M
    mid_mask = (np.arange(len(out)) * EVAL_INTERVAL >= 8_000_000) & (np.arange(len(out)) * EVAL_INTERVAL <= 12_000_000)
    mid_std = float(np.std(out[mid_mask], ddof=0))
    tail = out[-tail_points:]
    tail_std = float(np.std(tail, ddof=0))

    if tail_std <= mid_std:
        return out

    base_tail = base[-tail_points:]
    noise_tail = tail - base_tail
    centered = noise_tail - np.mean(noise_tail)
    centered_std = float(np.std(centered, ddof=0))
    if centered_std < 1e-12:
        return out

    factor = max(0.0, (0.95 * mid_std) / tail_std)
    new_noise_tail = np.mean(noise_tail) + factor * centered
    out[-tail_points:] = base_tail + new_noise_tail
    return out


def find_convergence_step(reward: np.ndarray, steps: np.ndarray, threshold: float, n_keep: int = CONVERGENCE_N) -> float:
    flags = reward >= threshold
    for i in range(len(flags) - n_keep + 1):
        if np.all(flags[i : i + n_keep]):
            return float(steps[i])
    return math.nan


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    steps = np.arange(0, TOTAL_STEPS + EVAL_INTERVAL, EVAL_INTERVAL, dtype=np.int64)

    all_rows = []
    algo_seed_curves: dict[str, list[np.ndarray]] = {spec.name: [] for spec in ALGO_SPECS}

    for spec in ALGO_SPECS:
        for seed in range(N_SEEDS):
            # Exponential rise-to-plateau trend with per-seed small variation.
            L = spec.final_mean * (1.0 + rng.normal(0, 0.03))
            k = spec.k * (1.0 + rng.normal(0, 0.07))
            trend = L * (1.0 - np.exp(-k * steps))

            # AR(1) correlated noise with monotonically decaying sigma_t.
            sigmas = spec.sigma_start * np.exp(-steps / TAU) + spec.sigma_floor
            noise = generate_ar1_noise(sigmas=sigmas, rho=RHO, rng=rng)

            # Method-specific dropout events.
            event = np.zeros_like(trend)
            if spec.name == "PPO":
                # one clear rollback around 15M
                c = 15_000_000 + rng.normal(0, 220_000)
                w = 300_000 + rng.uniform(-40_000, 50_000)
                d = rng.uniform(45, 62)
                event += make_dropout(steps, center=c, width=w, depth=d)
            elif spec.name == "BC-RL":
                # one small late rollback (<40)
                c = rng.uniform(13_500_000, 18_500_000)
                w = rng.uniform(170_000, 320_000)
                d = rng.uniform(20, 38)
                event += make_dropout(steps, center=c, width=w, depth=d)
            # SKC-PPO / SKC-PPO-F: no large rollback events.

            base = trend + event
            reward = base + noise
            reward = np.maximum(reward, -20.0)

            # Tail volatility cap: shrink tail noise only if too large.
            reward = cap_tail_volatility(reward=reward, base=base, target_std=spec.final_std)

            # Extra SKC realism constraint: tail fluctuation <= mid fluctuation.
            if spec.name in {"SKC-PPO", "SKC-PPO-F"}:
                reward = enforce_skc_tail_not_exceed_mid(reward=reward, base=base)

            # Calibration rule: mean-align only (no scaling to match std per seed).
            reward = align_tail_mean_only(reward, target_mean=spec.final_mean)

            algo_seed_curves[spec.name].append(reward)
            all_rows.extend(
                {
                    "step": int(s),
                    "algo": spec.name,
                    "seed": seed,
                    "reward": float(r),
                }
                for s, r in zip(steps, reward)
            )

    df = pd.DataFrame(all_rows)
    df.to_csv("mock_rewards.csv", index=False)

    # Main reward curve figure
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
    fig.savefig("fig_reward_curves.pdf")
    plt.close(fig)

    # Convergence stats
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

    # Convergence bar chart
    order = [spec.name for spec in ALGO_SPECS]
    bar_means = [float(summary_df.loc[summary_df["algo"] == a, "convergence_mean"].iloc[0]) for a in order]
    bar_stds = [float(summary_df.loc[summary_df["algo"] == a, "convergence_std"].iloc[0]) for a in order]

    fig2, ax2 = plt.subplots(figsize=(8.8, 5.4))
    xs = np.arange(len(order))
    colors = [color_map[a] for a in order]
    ax2.bar(xs, bar_means, yerr=bar_stds, capsize=5, color=colors, alpha=0.85)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(order, rotation=0)
    ax2.set_ylabel("Convergence Step")
    ax2.set_title("Convergence Step (mean ± std across 5 seeds)")
    ax2.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig("fig_convergence_steps.pdf")
    plt.close(fig2)

    # Display terminal summary.
    print("Tail summary (last 32 eval points):")
    tail_start_step = steps[-TAIL_POINTS]
    for spec in ALGO_SPECS:
        sub = df[(df["algo"] == spec.name) & (df["step"] >= tail_start_step)]
        per_seed = sub.groupby("seed")["reward"].agg(["mean", "std"])
        print(f"\\n{spec.name} target mean={spec.final_mean:.3f}, target std={spec.final_std:.3f}")
        print(f"  average over seeds: mean={per_seed['mean'].mean():.3f}, std={per_seed['std'].mean():.3f}")
        for seed, row in per_seed.iterrows():
            print(f"  seed={seed}: mean={row['mean']:.3f}, std={row['std']:.3f}")

    print("\nSaved files:")
    print("  - mock_rewards.csv")
    print("  - convergence_steps.csv")
    print("  - fig_reward_curves.pdf")
    print("  - fig_convergence_steps.pdf")


if __name__ == "__main__":
    main()
