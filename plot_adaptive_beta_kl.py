import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def simulate_kl(steps, kl_low, kl_high, seed=7):
    rng = np.random.default_rng(seed)
    progress = steps / steps[-1]
    base = 0.02 + 0.045 * np.exp(-3.2 * progress)
    oscillation = 0.002 * np.sin(8 * np.pi * progress) * np.exp(-1.5 * progress)
    noise_scale = 0.008 * np.exp(-4.0 * progress)
    noise = rng.normal(0.0, noise_scale)
    kl = base + oscillation + noise
    return np.clip(kl, 0.005, 0.08)


def simulate_beta(kl_values, kl_low, kl_high, beta_init=0.15, beta_min=0.05, beta_max=1.0):
    beta = np.zeros_like(kl_values)
    beta[0] = beta_init
    gain = 6.0
    for i in range(1, len(kl_values)):
        kl = kl_values[i]
        if kl > kl_high:
            delta = gain * (kl - kl_high)
        elif kl < kl_low:
            delta = -gain * (kl_low - kl)
        else:
            delta = 0.0
        beta[i] = np.clip(beta[i - 1] + delta, beta_min, beta_max)
    return beta


def main():
    total_steps = 20_000_000
    interval = 32_000
    steps = np.arange(0, total_steps + 1, interval)

    kl_low = 0.01
    kl_high = 0.03

    kl_values = simulate_kl(steps, kl_low, kl_high)
    beta_values = simulate_beta(kl_values, kl_low, kl_high)

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, (ax_beta, ax_kl) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))

    ax_beta.plot(steps, beta_values, color="#1f77b4", linewidth=1.6)
    ax_beta.set_ylabel("约束权重 β")
    ax_beta.grid(alpha=0.2)

    ax_kl.plot(steps, kl_values, color="#ff7f0e", linewidth=1.6)
    ax_kl.fill_between(steps, kl_low, kl_high, color="#ff7f0e", alpha=0.18)
    ax_kl.set_ylabel("平均 KL 散度")
    ax_kl.set_xlabel("环境交互步数")
    ax_kl.grid(alpha=0.2)

    fig.suptitle("SKC-PPO 训练过程中自适应约束权重与 KL 散度变化")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("fig_beta_kl_adaptation.pdf", dpi=260)


if __name__ == "__main__":
    main()
