#!/usr/bin/env python3
"""Create the optimizer figures used in Chapter 04 from raw CSV files."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOTAL_TRAINING_SAMPLES = 1713


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fold_weights(rows):
    folds = sorted({int(row["fold"]) for row in rows})
    quotient, remainder = divmod(TOTAL_TRAINING_SAMPLES, len(folds))
    return {
        fold: quotient + int(position < remainder)
        for position, fold in enumerate(folds)
    }


def weighted_mean_and_sd(values, weights):
    total_weight = sum(weights)
    mean = sum(value * weight for value, weight in zip(values, weights)) / total_weight
    variance = sum(
        weight * (value - mean) ** 2
        for value, weight in zip(values, weights)
    ) / total_weight
    return mean, math.sqrt(variance)


def label_for_optimizer(description):
    if description.startswith("sgd_momentum"):
        return "SGD + momentum"
    if description.startswith("adagrad"):
        return "AdaGrad"
    if description.startswith("rmsprop"):
        return "RMSprop"
    if description.startswith("adam"):
        return "Adam"
    if description.startswith("sgd"):
        return "SGD"
    raise ValueError(f"Unknown optimizer description: {description}")


def summarize_fold_results(rows):
    weights = fold_weights(rows)
    grouped = defaultdict(list)
    for row in rows:
        value = float(row["validation_mse"])
        if not math.isfinite(value):
            raise ValueError("Validation MSE contains a non-finite value")
        grouped[row["candidate"]].append((int(row["fold"]), value))

    summaries = []
    for candidate, values in grouped.items():
        values.sort()
        fold_values = [value for _, value in values]
        sample_weights = [weights[fold] for fold, _ in values]
        mean, sd = weighted_mean_and_sd(fold_values, sample_weights)
        summaries.append(
            {
                "candidate": candidate,
                "label": label_for_optimizer(rows[next(
                    index for index, row in enumerate(rows)
                    if row["candidate"] == candidate
                )]["optimizer"]),
                "values": fold_values,
                "mean": mean,
                "sd": sd,
            }
        )
    return sorted(summaries, key=lambda summary: summary["mean"])


def plot_ranking(summaries, output_path):
    labels = [summary["label"] for summary in summaries]
    positions = list(range(len(summaries)))
    figure, axis = plt.subplots(figsize=(7.4, 4.7))
    axis.boxplot(
        [summary["values"] for summary in summaries],
        positions=positions,
        widths=0.48,
        patch_artist=True,
        boxprops={"facecolor": "#d9e8f5", "edgecolor": "#315f8c"},
        medianprops={"color": "#17324d", "linewidth": 1.4},
        whiskerprops={"color": "#315f8c"},
        capprops={"color": "#315f8c"},
        flierprops={"marker": "", "markersize": 0},
    )
    for position, summary in zip(positions, summaries):
        offsets = [-0.13, -0.065, 0.0, 0.065, 0.13]
        axis.scatter(
            [position + offset for offset in offsets],
            summary["values"],
            color="#17202a",
            s=20,
            zorder=3,
        )
        axis.errorbar(
            position,
            summary["mean"],
            yerr=summary["sd"],
            fmt="o",
            color="#b23a48",
            markerfacecolor="#b23a48",
            markeredgecolor="white",
            markeredgewidth=0.7,
            capsize=3,
            zorder=4,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Physical validation MSE")
    axis.set_xlabel("Optimizer candidate, ordered by weighted mean")
    axis.set_yscale("log")
    axis.grid(True, axis="y", which="both", linestyle=":", alpha=0.55)
    axis.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="#17202a", linestyle="None", label="Fold"),
            plt.Line2D([], [], marker="o", color="#b23a48", linestyle="None", label="Weighted mean"),
        ],
        loc="upper right",
        frameon=True,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def plot_history(rows, summaries, output_path):
    weights = fold_weights(rows)
    grouped = defaultdict(list)
    for row in rows:
        value = float(row["validation_mse"])
        if not math.isfinite(value):
            raise ValueError("History validation MSE contains a non-finite value")
        grouped[(row["candidate"], int(row["fold"]))].append(
            (int(row["epoch"]), value)
        )

    figure, axis = plt.subplots(figsize=(8.0, 4.9))
    colors = plt.get_cmap("tab10").colors
    for color_index, summary in enumerate(summaries):
        candidate = summary["candidate"]
        by_epoch = defaultdict(list)
        for (candidate_key, fold), points in grouped.items():
            if candidate_key == candidate:
                for epoch, value in points:
                    by_epoch[epoch].append((fold, value))
        epochs = sorted(by_epoch)
        means = []
        deviations = []
        for epoch in epochs:
            points = sorted(by_epoch[epoch])
            values = [value for _, value in points]
            point_weights = [weights[fold] for fold, _ in points]
            mean, sd = weighted_mean_and_sd(values, point_weights)
            means.append(mean)
            deviations.append(sd)
        if len(epochs) != 10:
            raise ValueError(f"Expected ten history checkpoints for {summary['label']}")
        color = colors[color_index % len(colors)]
        axis.plot(epochs, means, marker="o", markersize=3.2, label=summary["label"], color=color)
        lower = [max(mean - sd, 1e-12) for mean, sd in zip(means, deviations)]
        upper = [mean + sd for mean, sd in zip(means, deviations)]
        axis.fill_between(epochs, lower, upper, color=color, alpha=0.10)

    axis.set_xlabel("Completed epoch")
    axis.set_ylabel("Physical validation MSE")
    axis.set_yscale("log")
    axis.set_xticks(list(range(10, 101, 10)))
    axis.grid(True, which="both", linestyle=":", alpha=0.55)
    axis.legend(loc="upper right", frameon=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/cross_validation/optimizer_comparison"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Report/Images/Chapter04/optimizer"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fold_rows = read_rows(args.results_dir / "fold_results.csv")
    history_rows = read_rows(args.results_dir / "training_history.csv")
    if len(fold_rows) != 25 or len({row["candidate"] for row in fold_rows}) != 5:
        raise ValueError("Expected 25 fold rows for five optimizer candidates")
    summaries = summarize_fold_results(fold_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_ranking(summaries, args.output_dir / "optimizer_ranking.png")
    plot_history(history_rows, summaries, args.output_dir / "optimizer_history.png")
    for summary in summaries:
        print(f"{summary['label']}: {summary['mean']:.12f} +/- {summary['sd']:.12f}")


if __name__ == "__main__":
    main()
