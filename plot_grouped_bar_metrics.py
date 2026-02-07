import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    titles = ["(a) 调节步长", "(b) 稳态绝对误差", "(c) 控制量总变差", "(d) 平均回合奖励"]
    ylabels = ["调节步长", "稳态绝对误差", "控制量总变差", "平均回合奖励"]

    x = np.arange(len(algorithms))
    color_map = {
        "PID": "#9aa0a6",
        "BC": "#c7c7c7",
        "PPO": "#4CAF50",
        "BC-RL": "#FF9800",
        "SKC-PPO-F": "#1E88E5",
        "SKC-PPO": "#8E24AA",
    }
    error_kw = {"elinewidth": 1.0, "capsize": 3, "capthick": 1.0, "ecolor": "black"}

    for plot_idx, (ax, title, ylabel, metric_name) in enumerate(
        zip(axes, titles, ylabels, metrics.keys())
    ):
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
            if plot_idx == 0:
                bar.set_label(algorithms[idx])

        ax.set_title(title)
        ax.set_xlabel("算法")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=20)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("各算法在独立测试集上的性能指标对比", y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig("fig_grouped_bar_metrics.png", dpi=260)
    fig.savefig("fig_grouped_bar_metrics.pdf", dpi=260)


if __name__ == "__main__":
    main()
