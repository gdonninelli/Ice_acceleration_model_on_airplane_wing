#!/usr/bin/env python3
"""Create the compact dropout figures used by Chapter 04.

The script reads the committed raw sweep and diagnostic CSV files.  It does
not rerun training or derive values from a plot.  All diagnostic aggregates
are calculated across the five recorded folds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def rate_from_name(name: str) -> float:
    match = re.search(r"dropout-rate=([^\]]+)", name)
    if match is None:
        raise ValueError(f"Cannot parse dropout rate from {name!r}")
    return float(match.group(1))


def candidate_directories(root: Path) -> dict[float, Path]:
    output: dict[float, Path] = {}
    for candidate in sorted(root.glob("candidate_*")):
        metadata = json.loads(
            (candidate / "fold_000" / "metadata.json").read_text(encoding="utf-8")
        )
        rate = float(metadata["selected_hyperparameters"]["dropout-rate"])
        output[rate] = candidate
    return output


def save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_ranking(root: Path, output: Path) -> None:
    rows = read_csv(root / "cv_summary.csv")
    aggregate: dict[int, dict[str, float | str]] = {}
    for row in rows:
        index = int(row["candidate_index"])
        aggregate[index] = {
            "rate": rate_from_name(row["candidate_name"]),
            "mean": float(row["mean_validation_physical_mse"]),
            "std": float(row["validation_stddev"]),
        }
    ordered = sorted(aggregate.values(), key=lambda item: float(item["rate"]))
    rates = [float(item["rate"]) for item in ordered]
    means = [float(item["mean"]) for item in ordered]
    deviations = [float(item["std"]) for item in ordered]

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    colors = ["#9b3d2e" if rate == 0.0 else "#477998" for rate in rates]
    axis.errorbar(
        range(len(rates)),
        means,
        yerr=deviations,
        fmt="none",
        ecolor="#343434",
        capsize=3,
        linewidth=1,
    )
    axis.scatter(range(len(rates)), means, c=colors, s=48, zorder=3)
    axis.set_xticks(range(len(rates)), [f"{rate:g}" for rate in rates])
    axis.set_xlabel("Dropout rate p")
    axis.set_ylabel("Mean physical validation MSE")
    axis.set_title("Dropout-rate cross-validation ranking")
    axis.grid(axis="y", alpha=0.25)
    axis.text(
        0.02,
        0.97,
        "Error bars: weighted fold standard deviation",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    save(figure, output / "dropout_ranking.png")


def plot_paired_differences(root: Path, output: Path) -> None:
    rows = read_csv(root.parent / "sweep_dropout.csv")
    by_rate: dict[float, dict[int, float]] = defaultdict(dict)
    for row in rows:
        by_rate[float(row["dropout_rate"])] [int(row["fold"])] = float(row["val_mse"])
    reference = by_rate[0.0]
    rates = sorted(by_rate)

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    folds = sorted(reference)
    for fold in folds:
        differences = [by_rate[rate][fold] - reference[fold] for rate in rates]
        axis.plot(rates, differences, "o-", linewidth=1, alpha=0.65, label=f"fold {fold}")
    means = np.array([
        np.mean([by_rate[rate][fold] - reference[fold] for fold in folds])
        for rate in rates
    ])
    deviations = np.array([
        np.std([by_rate[rate][fold] - reference[fold] for fold in folds], ddof=1)
        for rate in rates
    ])
    axis.errorbar(
        rates,
        means,
        yerr=deviations,
        fmt="ks-",
        linewidth=2,
        markersize=4,
        capsize=3,
        label="mean +/- fold std",
    )
    axis.axhline(0.0, color="#343434", linewidth=1)
    axis.set_xlabel("Dropout rate p")
    axis.set_ylabel("Validation MSE difference from p = 0")
    axis.set_title("Paired validation differences")
    axis.set_xticks(rates, [f"{rate:g}" for rate in rates])
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
    save(figure, output / "dropout_paired_differences.png")


def plot_history(root: Path, output: Path) -> None:
    candidates = candidate_directories(root)
    colors = {0.0: "#9b3d2e", 0.1: "#477998", 0.2: "#5b8e7d", 0.3: "#c28f2c", 0.5: "#6f5a8d"}

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for rate, candidate in sorted(candidates.items()):
        objective_by_epoch: dict[int, list[float]] = defaultdict(list)
        validation_by_epoch: dict[int, list[float]] = defaultdict(list)
        for fold in sorted(candidate.glob("fold_*")):
            for row in read_csv(fold / "epoch_metrics.csv"):
                epoch = int(row["epoch"])
                objective_by_epoch[epoch].append(float(row["train_objective"]))
                if row["validation_physical_mse"]:
                    validation_by_epoch[epoch].append(
                        float(row["validation_physical_mse"])
                    )
        epochs = sorted(objective_by_epoch)
        axes[0].plot(
            epochs,
            [np.mean(objective_by_epoch[epoch]) for epoch in epochs],
            color=colors[rate],
            label=f"p = {rate:g}",
        )
        validation_epochs = sorted(validation_by_epoch)
        axes[1].plot(
            validation_epochs,
            [np.mean(validation_by_epoch[epoch]) for epoch in validation_epochs],
            "o-",
            color=colors[rate],
            label=f"p = {rate:g}",
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean SIMM training objective")
    axes[0].set_title("Training objective")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean physical validation MSE")
    axes[1].set_title("Validation error")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.suptitle("Dropout convergence, means over five folds")
    save(figure, output / "dropout_history.png")


def aggregate_fold_values(
    candidate: Path,
    filename: str,
    scope: str | None,
    value_column: str,
    phase: str | None = None,
) -> dict[int, list[float]]:
    values: dict[int, list[float]] = defaultdict(list)
    for fold in sorted(candidate.glob("fold_*")):
        by_epoch: dict[int, list[float]] = defaultdict(list)
        for row in read_csv(fold / filename):
            if scope is not None and row.get("parameter_scope") != scope:
                continue
            if phase is not None and row.get("phase") != phase:
                continue
            by_epoch[int(row["epoch"])].append(float(row[value_column]))
        for epoch, epoch_values in by_epoch.items():
            values[epoch].append(float(np.mean(epoch_values)))
    return values


def plot_aggregate_metric(
    axis: plt.Axes,
    candidates: dict[float, Path],
    filename: str,
    scope: str | None,
    value_column: str,
    title: str,
    ylabel: str,
    phase: str | None = None,
    logarithmic: bool = False,
) -> None:
    aggregates = {
        rate: aggregate_fold_values(candidate, filename, scope, value_column, phase)
        for rate, candidate in candidates.items()
    }
    rates = sorted(candidates)
    for epoch, style in ((1, "o-"), (100, "s--")):
        means = [np.mean(aggregates[rate][epoch]) for rate in rates]
        minimums = [np.min(aggregates[rate][epoch]) for rate in rates]
        maximums = [np.max(aggregates[rate][epoch]) for rate in rates]
        axis.plot(rates, means, style, label=f"epoch {epoch}")
        axis.fill_between(rates, minimums, maximums, alpha=0.12)
    if logarithmic:
        axis.set_yscale("log")
    axis.set_xticks(rates, [f"{rate:g}" for rate in rates])
    axis.set_xlabel("Dropout rate p")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25, which="both")
    axis.legend(fontsize=8)


def plot_stability(root: Path, output: Path) -> None:
    candidates = candidate_directories(root)

    figure, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    plot_aggregate_metric(
        axes[0, 0],
        candidates,
        "gradient_norms.csv",
        "all",
        "rms_norm",
        "Aggregate gradient norm",
        "Mean RMS gradient norm",
        logarithmic=True,
    )
    plot_aggregate_metric(
        axes[0, 1],
        candidates,
        "parameter_update_ratios.csv",
        "all",
        "mean_ratio",
        "Aggregate update ratio",
        "Mean update ratio",
        logarithmic=True,
    )
    plot_aggregate_metric(
        axes[1, 0],
        candidates,
        "activation_statistics.csv",
        None,
        "variance",
        "Aggregate post-activation variance",
        "Mean population variance",
        phase="post_activation",
    )

    # The fourth panel retains the maximum gradient diagnostic for every rate.
    peak_gradients = []
    rates = sorted(candidates)
    for rate in rates:
        values = []
        for fold in sorted(candidates[rate].glob("fold_*")):
            values.extend(
                float(row["maximum_norm"])
                for row in read_csv(fold / "gradient_norms.csv")
                if row["parameter_scope"] == "all"
            )
        peak_gradients.append(max(values))
    axis = axes[1, 1]
    bars = axis.bar([f"{rate:g}" for rate in rates], peak_gradients, color="#477998")
    bars[0].set_color("#9b3d2e")
    axis.set_xlabel("Dropout rate p")
    axis.set_ylabel("Largest recorded maximum gradient norm")
    axis.set_title("Peak gradient norm")
    axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Dropout diagnostics aggregated by rate and fold"
    )
    save(figure, output / "dropout_stability.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/cross_validation/dropout_tuning/sweep"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Report/Images/Chapter04/dropout"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_ranking(args.input, args.output_dir)
    plot_paired_differences(args.input, args.output_dir)
    plot_history(args.input, args.output_dir)
    plot_stability(args.input, args.output_dir)
    print(f"Wrote dropout report figures to {args.output_dir}")


if __name__ == "__main__":
    main()
