import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def simulate_kl(steps, kl_star, seed=7):
    rng = np.random.default_rng(seed)
    kl_trend = kl_star + 0.05 * np.exp(-steps / 2.5e6)
    early_std = 0.012
    late_std = 0.003
    rho = 0.85
    std_t = late_std + (early_std - late_std) * np.exp(-steps / 3e6)
    noise = np.zeros_like(steps, dtype=float)
    noise[0] = rng.normal(0.0, std_t[0])
    for i in range(1, len(steps)):
        noise[i] = rho * noise[i - 1] + rng.normal(0.0, std_t[i])
    kl = np.clip(kl_trend + noise, 0.002, 0.08)
    return kl


def simulate_beta(
    kl_values,
    kl_low,
    kl_high,
    kl_star,
    beta_init=0.15,
    beta_min=0.05,
    beta_max=0.9,
    eta=0.05,
    gamma=0.90,
    delta=0.10,
    b_clip=0.08,
):
    beta = np.zeros_like(kl_values)
    beta[0] = beta_init
    log_beta = np.log(beta_init)
    eta_used = eta
    for i in range(1, len(kl_values)):
        kl = kl_values[i]
        if kl > kl_high:
            e = (kl - kl_high) / kl_star
        elif kl < kl_low:
            e = (kl - kl_low) / kl_star
        else:
            e = 0.0
        if abs(e) < delta:
            e = 0.0
        b_new = log_beta + eta_used * e
        b_new = np.clip(b_new, log_beta - b_clip, log_beta + b_clip)
        b_smooth = gamma * log_beta + (1 - gamma) * b_new
        beta[i] = np.clip(np.exp(b_smooth), beta_min, beta_max)
        log_beta = np.log(beta[i])
        if i <= 5 and beta[i] >= beta_max:
            eta_used *= 0.8
            log_beta = np.log(beta[i - 1])
            b_new = log_beta + eta_used * e
            b_new = np.clip(b_new, log_beta - b_clip, log_beta + b_clip)
            b_smooth = gamma * log_beta + (1 - gamma) * b_new
            beta[i] = np.clip(np.exp(b_smooth), beta_min, beta_max)
            log_beta = np.log(beta[i])
    return beta


def main():
    total_steps = 20_000_000
    interval = 32_000
    steps = np.arange(0, total_steps + interval, interval)

    kl_low = 0.01
    kl_high = 0.03
    kl_star = 0.02

    kl_values = simulate_kl(steps, kl_star)
    beta_values = simulate_beta(kl_values, kl_low, kl_high, kl_star)

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, (ax_beta, ax_kl) = plt.subplots(2, 1, sharex=True, figsize=(11, 7))

    ax_beta.plot(steps, beta_values, color="#1f77b4", linewidth=1.6)
    ax_beta.set_ylabel("约束权重 β")
    ax_beta.grid(alpha=0.2)

    ax_kl.plot(steps, kl_values, color="#ff7f0e", linewidth=1.6)
    ax_kl.fill_between(steps, kl_low, kl_high, color="#ff7f0e", alpha=0.18)
    ax_kl.axhline(kl_low, color="#ff7f0e", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_kl.axhline(kl_high, color="#ff7f0e", linestyle="--", linewidth=1.0, alpha=0.6)
    ax_kl.set_ylabel("平均 KL 散度")
    ax_kl.set_xlabel("环境交互步数")
    ax_kl.grid(alpha=0.2)

    fig.suptitle("SKC-PPO 训练过程中自适应约束权重与 KL 散度变化")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("fig_beta_kl_adaptation.pdf", dpi=260)
    fig.savefig("fig_beta_kl_adaptation.png", dpi=260)

    beta_min = float(beta_values.min())
    beta_max = float(beta_values.max())
    post_mask = steps >= 10_000_000
    kl_in_range = (kl_values >= kl_low) & (kl_values <= kl_high)
    kl_ratio = float(kl_in_range[post_mask].mean()) if post_mask.any() else 0.0
    std_post = float(beta_values[post_mask].std()) if post_mask.any() else 0.0
    std_pre = float(beta_values[steps <= 5_000_000].std())

    print(f\"beta min={beta_min:.4f}, max={beta_max:.4f}\")
    print(f\"10M之后KL区间内比例={kl_ratio:.2%}\")
    print(f\"beta std(<=5M)={std_pre:.4f}, std(>=10M)={std_post:.4f}\")


if __name__ == "__main__":
    main()
