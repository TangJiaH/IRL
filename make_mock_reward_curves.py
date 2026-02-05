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


@dataclass(frozen=True)
class AlgoSpec:
    name: str
    final_mean: float
    final_std: float
    k: float
    noise_hi: float
    noise_lo: float
    heavy_tail: bool = False
    smooth: bool = False


ALGO_SPECS = [
    AlgoSpec("PPO", final_mean=374.6, final_std=22.9, k=1.9e-7, noise_hi=36.0, noise_lo=11.0, heavy_tail=True, smooth=False),
    AlgoSpec("BC-RL", final_mean=498.0, final_std=38.0, k=2.35e-7, noise_hi=28.0, noise_lo=16.0, heavy_tail=False, smooth=False),
    AlgoSpec("SKC-PPO-F", final_mean=545.0, final_std=34.0, k=2.85e-7, noise_hi=18.0, noise_lo=9.0, heavy_tail=False, smooth=True),
    AlgoSpec("SKC-PPO", final_mean=588.0, final_std=29.0, k=3.25e-7, noise_hi=14.0, noise_lo=7.0, heavy_tail=False, smooth=True),
]


def moving_average(x: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return x.copy()
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def make_dropout(steps: np.ndarray, center: float, width: float, depth: float) -> np.ndarray:
    return -depth * np.exp(-0.5 * ((steps - center) / width) ** 2)


def calibrate_tail(reward: np.ndarray, target_mean: float, target_std: float, n_tail: int = TAIL_POINTS) -> np.ndarray:
    out = reward.copy()
    tail = out[-n_tail:]
    mu = float(np.mean(tail))
    sigma = float(np.std(tail, ddof=0))

    if sigma < 1e-12:
        # avoid divide-by-zero; inject tiny spread before calibrating
        eps = np.linspace(-1e-3, 1e-3, len(tail))
        tail = tail + eps
        sigma = float(np.std(tail, ddof=0))

    a = target_std / sigma
    b = target_mean - a * mu
    out[-n_tail:] = a * tail + b
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
            # Exponential rise-to-plateau + per-seed variation in asymptote/speed
            L = spec.final_mean * (1.0 + rng.normal(0, 0.035))
            k = spec.k * (1.0 + rng.normal(0, 0.08))
            trend = L * (1.0 - np.exp(-k * steps))

            # Decaying noise envelope
            prog = steps / TOTAL_STEPS
            envelope = spec.noise_lo + (spec.noise_hi - spec.noise_lo) * np.exp(-3.0 * prog)
            noise = rng.normal(0.0, envelope)

            # Heavy-tailed PPO spikes
            if spec.heavy_tail:
                spike_mask = rng.random(len(steps)) < 0.06
                spike_noise = rng.standard_t(df=3, size=len(steps)) * (0.45 * envelope)
                noise = noise + spike_mask * spike_noise

            # BC-RL: larger mid/late instability
            if spec.name == "BC-RL":
                late_boost = 1.0 + 0.45 / (1.0 + np.exp(-(prog - 0.65) * 16.0))
                noise = noise * late_boost

            reward = trend + noise

            # Dropout events
            if spec.name == "PPO":
                c = 15_000_000 + rng.normal(0, 260_000)
                w = 320_000 + rng.uniform(-50_000, 70_000)
                d = rng.uniform(48, 68)
                reward += make_dropout(steps, center=c, width=w, depth=d)
            elif spec.name == "BC-RL":
                n_drop = int(rng.integers(1, 3))
                for _ in range(n_drop):
                    c = rng.uniform(13_000_000, 19_000_000)
                    w = rng.uniform(190_000, 420_000)
                    d = rng.uniform(28, 62)
                    reward += make_dropout(steps, center=c, width=w, depth=d)
            elif spec.name == "SKC-PPO-F":
                if rng.random() < 0.45:
                    reward += make_dropout(
                        steps,
                        center=rng.uniform(12_500_000, 18_500_000),
                        width=rng.uniform(160_000, 300_000),
                        depth=rng.uniform(8, 20),
                    )
            elif spec.name == "SKC-PPO":
                if rng.random() < 0.25:
                    reward += make_dropout(
                        steps,
                        center=rng.uniform(14_000_000, 18_500_000),
                        width=rng.uniform(140_000, 240_000),
                        depth=rng.uniform(4, 12),
                    )

            if spec.smooth:
                reward = moving_average(reward, window=3)

            reward = np.maximum(reward, -20.0)

            # Force tail mean/std calibration to exactly match targets
            reward = calibrate_tail(reward, spec.final_mean, spec.final_std, n_tail=TAIL_POINTS)

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
        conv_steps = []
        for seed_idx, curve in enumerate(algo_seed_curves[spec.name]):
            cstep = find_convergence_step(curve, steps, threshold=threshold, n_keep=CONVERGENCE_N)
            conv_steps.append(cstep)
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

    # Display terminal summary
    print("Tail calibration check (last 32 eval points per seed):")
    tail_start_step = steps[-TAIL_POINTS]
    for spec in ALGO_SPECS:
        sub = df[(df["algo"] == spec.name) & (df["step"] >= tail_start_step)]
        per_seed = sub.groupby("seed")["reward"].agg(["mean", "std"])
        print(f"\\n{spec.name} target mean={spec.final_mean:.3f}, std={spec.final_std:.3f}")
        for seed, row in per_seed.iterrows():
            print(f"  seed={seed}: mean={row['mean']:.3f}, std={row['std']:.3f}")

    print("\nSaved files:")
    print("  - mock_rewards.csv")
    print("  - convergence_steps.csv")
    print("  - fig_reward_curves.pdf")
    print("  - fig_convergence_steps.pdf")


if __name__ == "__main__":
    main()
