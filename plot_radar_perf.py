import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def normalize_metrics(data):
    metrics = np.array(list(data.values()), dtype=float)
    max_vals = metrics.max(axis=0)
    min_vals = metrics.min(axis=0)
    normalized = np.empty_like(metrics)
    for idx in range(metrics.shape[1]):
        col = metrics[:, idx]
        if idx == 3:
            normalized[:, idx] = (col - min_vals[idx]) / (max_vals[idx] - min_vals[idx])
        else:
            normalized[:, idx] = (max_vals[idx] - col) / (max_vals[idx] - min_vals[idx])
    return normalized


def main():
    data = {
        "PID": [102, 0.046, 18.4, 432],
        "BC": [136, 0.081, 11.9, 393],
        "PPO": [178, 0.069, 31.6, 375],
        "BC-RL": [114, 0.056, 26.9, 498],
        "SKC-PPO-F": [87, 0.038, 15.8, 545],
        "SKC-PPO": [79, 0.031, 14.1, 588],
    }

    labels = ["收敛速度", "稳态精度", "控制平滑性", "综合性能"]
    algorithms = list(data.keys())
    normalized = normalize_metrics(data)

    matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))

    for idx, name in enumerate(algorithms):
        values = normalized[idx]
        values = np.concatenate([values, values[:1]])
        if name == "SKC-PPO":
            ax.plot(angles, values, linewidth=2.6, alpha=0.9, label=name)
        else:
            ax.plot(angles, values, linewidth=1.4, alpha=0.55, label=name)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("各算法综合性能雷达图（归一化）", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    fig.tight_layout()
    fig.savefig("fig_radar_performance.pdf", dpi=260)
    fig.savefig("fig_radar_performance.png", dpi=260)

    print("归一化指标表（收敛速度, 稳态精度, 控制平滑性, 综合性能）：")
    for name, row in zip(algorithms, normalized):
        formatted = ", ".join(f"{val:.3f}" for val in row)
        print(f"{name}: {formatted}")


if __name__ == "__main__":
    main()
