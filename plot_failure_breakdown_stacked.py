import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def main():
    algorithms = ["PID", "BC", "PPO", "BC-RL", "SKC-PPO-F", "SKC-PPO"]
    unreach_heading = np.array([45, 62, 58, 39, 21, 12], dtype=float)
    extreme_state = np.array([12, 18, 41, 23, 13, 6], dtype=float)
    overload = np.array([11, 15, 43, 19, 9, 3], dtype=float)

    total_failures = unreach_heading + extreme_state + overload
    failure_rate = total_failures / 1000.0 * 100.0

    colors = {
        "UnreachHeading": "#90caf9",
        "ExtremeState": "#ffcc80",
        "Overload": "#ef9a9a",
    }

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    x = np.arange(len(algorithms))
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x,
        unreach_heading,
        color=colors["UnreachHeading"],
        edgecolor="black",
        linewidth=0.6,
        label="UnreachHeading",
    )
    ax.bar(
        x,
        extreme_state,
        bottom=unreach_heading,
        color=colors["ExtremeState"],
        edgecolor="black",
        linewidth=0.6,
        label="ExtremeState",
    )
    ax.bar(
        x,
        overload,
        bottom=unreach_heading + extreme_state,
        color=colors["Overload"],
        edgecolor="black",
        linewidth=0.6,
        label="Overload",
    )

    for idx, rate in enumerate(failure_rate):
        ax.text(
            x[idx],
            total_failures[idx] + 2.0,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("各算法失败类型构成统计")
    ax.set_ylabel("失败回合数（/1000）")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()

    fig.savefig("fig_failure_breakdown_stacked.png", dpi=260)
    fig.savefig("fig_failure_breakdown_stacked.pdf", dpi=260)

    for name, total, rate in zip(algorithms, total_failures, failure_rate):
        print(f"{name}: 失败总数={int(total)}, 失败率={rate:.1f}%")


if __name__ == "__main__":
    main()
