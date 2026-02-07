import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def main():
    algorithms = ["PID", "BC", "PPO", "BC-RL", "SKC-PPO-F", "SKC-PPO"]

    metrics = {
        "调节步长": {
            "mean": [102, 136, 178, 114, 87, 79],
            "std": [11, 17, 34, 19, 9, 7],
        },
        "稳态绝对误差": {
            "mean": [0.046, 0.081, 0.069, 0.056, 0.038, 0.031],
            "std": [0.010, 0.019, 0.028, 0.015, 0.010, 0.008],
        },
        "控制量总变差": {
            "mean": [18.4, 11.9, 31.6, 26.9, 15.8, 14.1],
            "std": [2.1, 1.5, 6.8, 5.7, 2.3, 2.0],
        },
        "平均回合奖励": {
            "mean": [432, 393, 375, 498, 545, 588],
            "std": [26, 31, 23, 38, 34, 29],
        },
    }

    failure_mean = np.array([6.8, 9.5, 14.2, 8.1, 4.3, 2.1], dtype=float)
    n_trials = 1000
    p = failure_mean / 100.0
    failure_std = 100.0 * np.sqrt(p * (1 - p) / n_trials)

    color_map = {
        "PID": "#9aa0a6",
        "BC": "#c7c7c7",
        "PPO": "#4CAF50",
        "BC-RL": "#FF9800",
        "SKC-PPO-F": "#1E88E5",
        "SKC-PPO": "#8E24AA",
    }

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, wspace=0.25, hspace=0.35)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    titles = [
        "(a) 调节步长",
        "(b) 稳态绝对误差",
        "(c) 控制量总变差",
        "(d) 平均回合奖励",
        "(e) 失败率",
    ]
    ylabels = [
        "调节步长",
        "稳态绝对误差",
        "控制量总变差",
        "平均回合奖励",
        "失败率（%）",
    ]

    error_kw = {"elinewidth": 1.0, "capsize": 3, "capthick": 1.0, "ecolor": "black"}
    x = np.arange(len(algorithms))

    for ax, title, ylabel, metric_name in zip(axes[:4], titles[:4], ylabels[:4], metrics.keys()):
        means = np.array(metrics[metric_name]["mean"], dtype=float)
        stds = np.array(metrics[metric_name]["std"], dtype=float)
        bars = ax.bar(
            x,
            means,
            yerr=stds,
            error_kw=error_kw,
            color=[color_map[algo] for algo in algorithms],
            edgecolor="none",
            linewidth=0.6,
        )
        for idx, bar in enumerate(bars):
            if algorithms[idx] == "SKC-PPO":
                bar.set_edgecolor("black")
                bar.set_linewidth(1.2)
            if metric_name == "调节步长":
                bar.set_label(algorithms[idx])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=20)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax_failure = axes[4]
    bars = ax_failure.bar(
        x,
        failure_mean,
        yerr=failure_std,
        error_kw=error_kw,
        color=[color_map[algo] for algo in algorithms],
        edgecolor="none",
        linewidth=0.6,
    )
    for idx, bar in enumerate(bars):
        if algorithms[idx] == "SKC-PPO":
            bar.set_edgecolor("black")
            bar.set_linewidth(1.2)
    ax_failure.set_title(titles[4])
    ax_failure.set_ylabel(ylabels[4])
    ax_failure.set_xticks(x)
    ax_failure.set_xticklabels(algorithms, rotation=20)
    ax_failure.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("各算法在独立测试集上的性能指标对比（含失败率）", y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig("fig_metrics_with_failure_rate.png", dpi=260)
    fig.savefig("fig_metrics_with_failure_rate.pdf", dpi=260)

    formatted = ", ".join(f"{val:.3f}" for val in failure_std)
    print(f"失败率标准差: {formatted}")


if __name__ == "__main__":
    main()
