#!/usr/bin/env python3
"""Create the learning-rate figures used in Chapter 04.

The script reads the committed cross-validation summary and diagnostics. The
candidate scores use the recorded sample-weighted values. Validation histories
are weighted with the fold validation sizes, peak diagnostics are maxima over
the five folds, and activation variances are pooled from population moments.

Usage:
    python3 plot_report_learning_rate.py [--results-dir PATH] [--output-dir PATH]
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CANDIDATES = [
    "const_1e-5",
    "const_1e-3",
    "const_1e-1",
    "step_s30_1e-1",
    "step_s30_1e-3",
    "step_s30_1e-5",
    "step_s15_1e-1",
    "step_s15_1e-3",
    "step_s15_1e-5",
    "cosine_1e-1",
    "warmup_cosine_1e-1",
]
WINNER = "const_1e-3"
STABLE = {
    "const_1e-5",
    "const_1e-3",
    "step_s30_1e-3",
    "step_s30_1e-5",
    "step_s15_1e-3",
    "step_s15_1e-5",
}


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_schedule(candidate_name):
    return candidate_name.split("schedule=", 1)[1].rstrip("]")


def display_name(candidate):
    return candidate.replace("_", "\n")


def weighted_mean_std(values):
    total_weight = sum(weight for _, weight in values)
    mean = sum(value * weight for value, weight in values) / total_weight
    variance = sum(weight * (value - mean) ** 2 for value, weight in values)
    return mean, math.sqrt(variance / total_weight)


def search_directory(results_dir):
    return results_dir / "learning_rate_tuning" / "run"


def load_inputs(results_dir):
    search_dir = search_directory(results_dir)
    summary = {}
    fold_scores = {}
    fold_weights = {}
    for row in read_rows(search_dir / "cv_summary.csv"):
        candidate_index = int(row["candidate_index"])
        candidate = parse_schedule(row["candidate_name"])
        summary[candidate] = {
            "mean": float(row["mean_validation_physical_mse"]),
            "std": float(row["validation_stddev"]),
        }
        fold = int(row["fold"])
        fold_scores[(candidate, fold)] = float(row["validation_physical_mse"])
        fold_weights[(candidate, fold)] = int(row["validation_samples"])
        if candidate_index != CANDIDATES.index(candidate):
            raise ValueError(f"Candidate index mismatch for {candidate}")

    history = defaultdict(lambda: defaultdict(list))
    for candidate_index, candidate in enumerate(CANDIDATES):
        for fold in range(5):
            path = search_dir / f"candidate_{candidate_index:03d}" / f"fold_{fold:03d}" / "epoch_metrics.csv"
            for row in read_rows(path):
                if not row["validation_physical_mse"]:
                    continue
                history[candidate][int(row["epoch"])].append(
                    (
                        float(row["validation_physical_mse"]),
                        fold_weights[(candidate, fold)],
                    )
                )

    return summary, history, fold_scores, fold_weights, search_dir


def diagnostic_rows(search_dir, candidate_index, filename):
    rows = []
    candidate_dir = search_dir / f"candidate_{candidate_index:03d}"
    for fold in range(5):
        path = candidate_dir / f"fold_{fold:03d}" / filename
        for row in read_rows(path):
            row["fold"] = fold
            rows.append(row)
    return rows


def peak_diagnostics(search_dir):
    peak_gradients = {}
    peak_updates = {}
    for candidate_index, candidate in enumerate(CANDIDATES):
        gradient_rows = diagnostic_rows(search_dir, candidate_index, "gradient_norms.csv")
        update_rows = diagnostic_rows(
            search_dir, candidate_index, "parameter_update_ratios.csv"
        )
        peak_gradients[candidate] = max(
            float(row["maximum_norm"])
            for row in gradient_rows
            if row["parameter_scope"] == "all"
        )
        peak_updates[candidate] = max(
            float(row["mean_ratio"])
            for row in update_rows
            if row["parameter_scope"] == "weights"
        )
    return peak_gradients, peak_updates


def winner_gradient_profile(search_dir):
    profile = defaultdict(lambda: defaultdict(list))
    for row in diagnostic_rows(search_dir, CANDIDATES.index(WINNER), "gradient_norms.csv"):
        if row["parameter_scope"] != "all":
            continue
        if int(row["epoch"]) in (1, 100):
            profile[int(row["epoch"])][int(row["layer_index"])].append(
                float(row["rms_norm"])
            )
    return {
        epoch: {
            layer: sum(values) / len(values)
            for layer, values in layers.items()
        }
        for epoch, layers in profile.items()
    }


def pooled_variance(rows):
    total = sum(int(row["count"]) for row in rows)
    first_moment = sum(int(row["count"]) * float(row["mean"]) for row in rows)
    second_moment = sum(
        int(row["count"])
        * (float(row["variance"]) + float(row["mean"]) ** 2)
        for row in rows
    )
    mean = first_moment / total
    return second_moment / total - mean**2


def winner_activation_variance(search_dir):
    grouped = defaultdict(list)
    rows = diagnostic_rows(
        search_dir, CANDIDATES.index(WINNER), "activation_statistics.csv"
    )
    for row in rows:
        if row["phase"] == "post_activation" and int(row["epoch"]) in (1, 50, 100):
            grouped[(int(row["epoch"]), int(row["layer_index"]))].append(row)
    return {
        epoch: {
            layer: pooled_variance(grouped[(epoch, layer)])
            for layer in (1, 5, 7, 9, 11)
        }
        for epoch in (1, 50, 100)
    }


def rate_history(search_dir, candidate_index):
    path = search_dir / f"candidate_{candidate_index:03d}" / "fold_000" / "epoch_metrics.csv"
    return [
        (int(row["epoch"]), float(row["effective_learning_rate"]))
        for row in read_rows(path)
    ]


def colors(candidates):
    result = []
    for candidate in candidates:
        if candidate == WINNER:
            result.append("tab:orange")
        elif candidate in STABLE:
            result.append("tab:blue")
        else:
            result.append("tab:red")
    return result


def plot_ranking(summary, output_dir):
    order = sorted(CANDIDATES, key=lambda candidate: summary[candidate]["mean"])
    means = [summary[candidate]["mean"] for candidate in order]
    deviations = [summary[candidate]["std"] for candidate in order]
    positions = list(range(len(order)))

    fig, axis = plt.subplots(figsize=(8.8, 5.4))
    axis.errorbar(
        means,
        positions,
        xerr=deviations,
        fmt="none",
        ecolor="0.25",
        elinewidth=1.0,
        capsize=3,
    )
    axis.scatter(means, positions, c=colors(order), s=38, zorder=3)
    axis.set_yticks(positions, order)
    axis.invert_yaxis()
    axis.set_xscale("log")
    axis.set_xlabel("Sample-weighted mean validation MSE (physical units)")
    axis.set_title("Learning-rate schedule cross-validation ranking")
    axis.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "learning_rate_ranking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_history(history, output_dir):
    figure, axes = plt.subplots(2, 1, figsize=(8.8, 7.0), sharex=True)
    groups = [
        (list(STABLE), "Stable-scale candidates"),
        ([candidate for candidate in CANDIDATES if candidate not in STABLE],
         "Candidates reaching $10^{-1}$"),
    ]
    for axis, (group, title) in zip(axes, groups):
        for candidate in CANDIDATES:
            if candidate not in group:
                continue
            epochs = sorted(history[candidate])
            means = [weighted_mean_std(history[candidate][epoch])[0] for epoch in epochs]
            axis.plot(epochs, means, marker="o", markersize=2.5, linewidth=1.3, label=candidate)
        axis.set_yscale("log")
        axis.set_ylabel("Validation MSE")
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Completed epoch")
    axes[-1].set_xticks(range(10, 101, 10))
    figure.suptitle("Fold-weighted validation histories", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_dir / "learning_rate_history.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_schedules(search_dir, output_dir):
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    for candidate_index, candidate in enumerate(CANDIDATES):
        values = rate_history(search_dir, candidate_index)
        epochs = [epoch for epoch, _ in values]
        rates = [rate for _, rate in values]
        axis.plot(
            epochs,
            rates,
            linewidth=1.4 if candidate == WINNER else 1.0,
            color=colors([candidate])[0],
            label=candidate,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Completed epoch")
    axis.set_ylabel("Effective learning rate")
    axis.set_title("Effective learning rates recorded by the trainer")
    axis.set_xticks(range(0, 101, 10))
    axis.grid(alpha=0.3)
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output_dir / "learning_rate_schedules.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_diagnostics(search_dir, output_dir):
    peak_gradients, peak_updates = peak_diagnostics(search_dir)
    gradient_profile = winner_gradient_profile(search_dir)
    activation_variance = winner_activation_variance(search_dir)
    positions = list(range(len(CANDIDATES)))
    labels = [display_name(candidate) for candidate in CANDIDATES]

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    figure.suptitle("Learning-rate diagnostics from the five-fold runs", fontsize=12)

    axes[0, 0].bar(positions, [peak_gradients[candidate] for candidate in CANDIDATES], color=colors(CANDIDATES))
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Peak gradient maximum norm")
    axes[0, 0].set_ylabel("Maximum norm, scope=all")

    axes[0, 1].bar(positions, [peak_updates[candidate] for candidate in CANDIDATES], color=colors(CANDIDATES))
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Peak actual weight movement")
    axes[0, 1].set_ylabel("Mean update ratio, scope=weights")

    for epoch, style in ((1, "--"), (100, "-")):
        layers = sorted(gradient_profile[epoch])
        axes[1, 0].plot(
            layers,
            [gradient_profile[epoch][layer] for layer in layers],
            marker="o",
            linestyle=style,
            label=f"Epoch {epoch}",
        )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Selected candidate gradient depth")
    axes[1, 0].set_xlabel("Stable model layer index")
    axes[1, 0].set_ylabel("RMS gradient norm")
    axes[1, 0].set_xticks([0, 4, 6, 8, 10, 12])
    axes[1, 0].legend(fontsize=8)

    for epoch, style in ((1, "--"), (50, ":"), (100, "-")):
        layers = sorted(activation_variance[epoch])
        axes[1, 1].plot(
            layers,
            [activation_variance[epoch][layer] for layer in layers],
            marker="o",
            linestyle=style,
            label=f"Epoch {epoch}",
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Selected candidate post-activation variance")
    axes[1, 1].set_xlabel("Stable activation layer index")
    axes[1, 1].set_ylabel("Population variance")
    axes[1, 1].set_xticks([1, 5, 7, 9, 11])
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    axes[0, 0].set_xticks(positions, labels, rotation=40, ha="right", fontsize=7)
    axes[0, 1].set_xticks(positions, labels, rotation=40, ha="right", fontsize=7)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output_dir / "learning_rate_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/cross_validation/learning_rate_tuning"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Report/Images/Chapter04/learning_rate"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary, history, _, _, search_dir = load_inputs(args.results_dir)
    plot_ranking(summary, args.output_dir)
    plot_history(history, args.output_dir)
    plot_schedules(search_dir, args.output_dir)
    plot_diagnostics(search_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
