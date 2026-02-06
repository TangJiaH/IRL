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
    base_color = "#a6bddb"
    highlight_color = "#1b3a6b"

    for ax, title, ylabel, metric_name in zip(axes, titles, ylabels, metrics.keys()):
        means = np.array(metrics[metric_name]["mean"], dtype=float)
        stds = np.array(metrics[metric_name]["std"], dtype=float)
        colors = [base_color] * len(algorithms)
        colors[-1] = highlight_color

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            error_kw={"ecolor": "black", "elinewidth": 1.2},
            color=colors,
            edgecolor="black",
            linewidth=1.1,
        )

        for idx, bar in enumerate(bars):
            if algorithms[idx] == "SKC-PPO":
                bar.set_linewidth(1.4)

        ax.set_title(title)
        ax.set_xlabel("算法")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=20)
        ax.grid(axis="y", alpha=0.3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=base_color, ec="black", lw=1.1),
        plt.Rectangle((0, 0), 1, 1, color=highlight_color, ec="black", lw=1.3),
    ]
    fig.legend(handles, ["其他算法", "SKC-PPO"], loc="upper center", ncol=2, frameon=False)

    fig.suptitle("各算法在独立测试集上的性能指标对比", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig("fig_grouped_bar_metrics.png", dpi=260)
    fig.savefig("fig_grouped_bar_metrics.pdf", dpi=260)


if __name__ == "__main__":
    main()
