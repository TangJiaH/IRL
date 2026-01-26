#!/usr/bin/env python
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def smooth_moving_avg(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def smooth_ema(series: pd.Series, alpha: float) -> pd.Series:
    if alpha <= 0 or alpha >= 1:
        return series
    return series.ewm(alpha=alpha, adjust=False).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制训练回报曲线，并支持平滑处理。")
    parser.add_argument("--csv", type=str, required=True, help="train_metrics.csv 路径")
    parser.add_argument("--window", type=int, default=20, help="滑动平均窗口大小")
    parser.add_argument("--ema", type=float, default=0.0, help="EMA 平滑系数(0-1)，0 表示不开启")
    parser.add_argument("--save", type=str, default="", help="保存图像文件路径（可选）")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["episode_return"].notna() & (df["episode_return"] != "")]
    df["episode_return"] = df["episode_return"].astype(float)

    x = df["episode"] if "episode" in df.columns else range(len(df))
    y = df["episode_return"]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, label="原始回报", alpha=0.4)

    if args.window > 1:
        plt.plot(x, smooth_moving_avg(y, args.window), label=f"滑动平均({args.window})")
    if 0 < args.ema < 1:
        plt.plot(x, smooth_ema(y, args.ema), label=f"EMA({args.ema})")

    plt.xlabel("episode")
    plt.ylabel("episode_return")
    plt.title("训练回报曲线")
    plt.legend()
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save)
        print(f"已保存图像到 {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
