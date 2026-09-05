#!/usr/bin/env python3
"""Create report figures from the recorded physics-weight sweep.

The diagnostic panels deliberately aggregate over folds and over comparable
diagnostic rows. They do not plot a layer index, because the retained run is
an archival artifact and its layer metadata is not the current experiment
recipe.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
SELECTED_INDICES = [0, 3, 6]
COLORS = {0: "#1f4e79", 3: "#b45f06", 6: "#6a1b9a"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def weighted_mean_std(values: list[tuple[float, int]]) -> tuple[float, float]:
    total = sum(weight for _, weight in values)
    mean = sum(value * weight for value, weight in values) / total
    variance = sum(weight * (value - mean) ** 2 for value, weight in values) / total
    return mean, math.sqrt(variance)


def load_cv_summary(results_dir: Path) -> tuple[list[float], list[float]]:
    rows = read_csv(results_dir / "sweep" / "cv_summary.csv")
    means: dict[int, float] = {}
    deviations: dict[int, float] = {}
    for row in rows:
        index = int(row["candidate_index"])
        if row["success"] != "1":
            raise ValueError(f"failed candidate in cv_summary.csv: {row}")
        means[index] = float(row["mean_validation_physical_mse"])
        deviations[index] = float(row["validation_stddev"])
    return (
        [means[index] for index in range(len(LAMBDA_VALUES))],
        [deviations[index] for index in range(len(LAMBDA_VALUES))],
    )


def load_masked_mse(results_dir: Path) -> tuple[list[float], list[float]]:
    rows = read_csv(results_dir / "sweep_physics.csv")
    small: dict[int, list[tuple[float, int]]] = defaultdict(list)
    large: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for row in rows:
        value = float(row["physics_weight"])
        index = min(range(len(LAMBDA_VALUES)), key=lambda candidate: abs(LAMBDA_VALUES[candidate] - value))
        small[index].append(
            (float(row["val_mse_small_angle"]), int(row["small_angle_count"]))
        )
        large[index].append(
            (float(row["val_mse_large_angle"]), int(row["large_angle_count"]))
        )
    small_means = [weighted_mean_std(small[index])[0] for index in range(len(LAMBDA_VALUES))]
    large_means = [weighted_mean_std(large[index])[0] for index in range(len(LAMBDA_VALUES))]
    return small_means, large_means


def validation_counts(results_dir: Path) -> dict[tuple[int, int], int]:
    rows = read_csv(results_dir / "sweep" / "cv_summary.csv")
    return {
        (int(row["candidate_index"]), int(row["fold"])): int(
            row["validation_samples"]
        )
        for row in rows
    }


def load_histories(
    results_dir: Path, counts: dict[tuple[int, int], int]
) -> dict[int, tuple[list[int], list[float], list[float]]]:
    histories: dict[int, dict[int, list[tuple[float, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for candidate_index in SELECTED_INDICES:
        for fold in range(5):
            path = (
                results_dir
                / "sweep"
                / f"candidate_{candidate_index:03d}"
                / f"fold_{fold:03d}"
                / "epoch_metrics.csv"
            )
            for row in read_csv(path):
                if not row["validation_physical_mse"]:
                    continue
                epoch = int(row["epoch"])
                histories[candidate_index][epoch].append(
                    (
                        float(row["validation_physical_mse"]),
                        counts[(candidate_index, fold)],
                    )
                )

    result: dict[int, tuple[list[int], list[float], list[float]]] = {}
    for candidate_index, epochs in histories.items():
        ordered = sorted(epochs)
        means: list[float] = []
        deviations: list[float] = []
        for epoch in ordered:
            mean, deviation = weighted_mean_std(epochs[epoch])
            means.append(mean)
            deviations.append(deviation)
        result[candidate_index] = (ordered, means, deviations)
    return result


def load_global_diagnostics(
    results_dir: Path, candidate_index: int
) -> dict[str, tuple[list[int], list[float], list[tuple[float, float]]]]:
    gradient: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    update: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    variance: dict[int, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for fold in range(5):
        base = (
            results_dir
            / "sweep"
            / f"candidate_{candidate_index:03d}"
            / f"fold_{fold:03d}"
        )
        for row in read_csv(base / "gradient_norms.csv"):
            if row["parameter_scope"] != "all":
                continue
            gradient[int(row["epoch"])][fold].append(float(row["rms_norm"]) ** 2)
        for row in read_csv(base / "parameter_update_ratios.csv"):
            if row["parameter_scope"] != "weights":
                continue
            update[int(row["epoch"])][fold].append(float(row["mean_ratio"]))
        for row in read_csv(base / "activation_statistics.csv"):
            if row["phase"] != "post_activation":
                continue
            variance[int(row["epoch"])][fold].append(float(row["variance"]))

    def finish(data, transform):
        epochs = sorted(data)
        means = []
        bands = []
        for epoch in epochs:
            fold_values = [transform(values) for values in data[epoch].values()]
            means.append(sum(fold_values) / len(fold_values))
            bands.append((min(fold_values), max(fold_values)))
        return epochs, means, bands

    return {
        "gradient": finish(gradient, lambda values: math.sqrt(sum(values))),
        "update": finish(update, lambda values: sum(values) / len(values)),
        "variance": finish(variance, lambda values: sum(values) / len(values)),
    }


def candidate_labels() -> list[str]:
    return ["0", "0.05", "0.1", "0.25", "0.5", "1", "2"]


def plot_performance(results_dir: Path, output_dir: Path) -> None:
    means, deviations = load_cv_summary(results_dir)
    small, large = load_masked_mse(results_dir)
    histories = load_histories(results_dir, validation_counts(results_dir))
    labels = candidate_labels()
    x = list(range(len(LAMBDA_VALUES)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    axes[0].errorbar(x, means, yerr=deviations, fmt="o-", capsize=3, color="#1f4e79")
    axes[0].set_title("Cross-validated validation error")
    axes[0].set_ylabel("Physical validation MSE")

    axes[1].plot(x, small, "o-", label=r"$|\alpha|\leq 10^\circ$", color="#2e7d32")
    axes[1].plot(x, large, "s-", label=r"$|\alpha|>10^\circ$", color="#b45f06")
    axes[1].set_title("Validation error by physics mask")
    axes[1].set_ylabel("Physical MSE")
    axes[1].legend(fontsize=8)

    for candidate_index, (epochs, means_history, deviations_history) in histories.items():
        color = COLORS[candidate_index]
        axes[2].plot(
            epochs,
            means_history,
            label=fr"$\lambda={LAMBDA_VALUES[candidate_index]:g}$",
            color=color,
        )
        lower = [mean - deviation for mean, deviation in zip(means_history, deviations_history)]
        upper = [mean + deviation for mean, deviation in zip(means_history, deviations_history)]
        axes[2].fill_between(epochs, lower, upper, color=color, alpha=0.12)
    axes[2].set_title("Validation history")
    axes[2].set_ylabel("Physical validation MSE")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=8)

    for axis in axes[:2]:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=45)
        axis.set_xlabel(r"Physics weight $\lambda$")
        axis.grid(True, alpha=0.25)
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.25)
    fig.suptitle("Physics-weight sweep, global performance summaries", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "physics_performance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(results_dir: Path, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    titles = [
        "Aggregated gradient norm",
        "Actual weight update ratio",
        "Post-activation variance",
    ]
    ylabels = [
        r"$\sqrt{\sum_\ell \mathrm{rms\_norm}_\ell^2}$",
        r"Mean $r_\theta$ (weights scope)",
        "Population variance",
    ]
    keys = ["gradient", "update", "variance"]
    for candidate_index in SELECTED_INDICES:
        diagnostics = load_global_diagnostics(results_dir, candidate_index)
        for axis, key, title, ylabel in zip(axes, keys, titles, ylabels):
            epochs, means, bands = diagnostics[key]
            color = COLORS[candidate_index]
            axis.plot(
                epochs,
                means,
                label=fr"$\lambda={LAMBDA_VALUES[candidate_index]:g}$",
                color=color,
            )
            lower = [band[0] for band in bands]
            upper = [band[1] for band in bands]
            axis.fill_between(epochs, lower, upper, color=color, alpha=0.10)
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=8)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    fig.suptitle("Physics-weight diagnostics, fold and diagnostic-row aggregates", y=1.02)
    fig.tight_layout()
    fig.savefig(
        output_dir / "physics_aggregate_diagnostics.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_performance(args.results_dir, args.output_dir)
    plot_diagnostics(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
