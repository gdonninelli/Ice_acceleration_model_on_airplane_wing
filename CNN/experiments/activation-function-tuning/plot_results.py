#!/usr/bin/env python3
"""Generate plots for the activation-function tuning experiment.

Usage:
    python3 plot_results.py [--results-dir PATH] [--out-dir PATH]
"""

import argparse
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTIVATIONS = [
    "tanh",
    "sigmoid",
    "relu",
    "leakyrelu-alpha-0.01",
    "leakyrelu-alpha-0.05",
    "leakyrelu-alpha-0.1",
]
DISPLAY_NAMES = {
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "relu": "ReLU",
    "leakyrelu-alpha-0.01": "LeakyReLU\nα=0.01",
    "leakyrelu-alpha-0.05": "LeakyReLU\nα=0.05",
    "leakyrelu-alpha-0.1": "LeakyReLU\nα=0.1",
}
CONFIG_STR = "conv5x5-dense-{1024,512,256,128}  Adam lr=1e-5"


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def std(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / len(values))


def read_fold_rows(results_dir):
    path = os.path.join(results_dir, "fold_results.csv")
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        activation: [row for row in rows if row["activation"] == activation]
        for activation in ACTIVATIONS
    }


def diagnostics_base(results_dir, activation):
    return os.path.join(
        results_dir,
        "activation-function-tuning",
        "search",
        f"candidate_{ACTIVATIONS.index(activation):03d}",
    )


def diagnostic_folders(base):
    if not os.path.isdir(base):
        return []
    return sorted(
        name
        for name in os.listdir(base)
        if name.startswith("fold_") and os.path.isdir(os.path.join(base, name))
    )


def read_epoch_metrics(results_dir, activation):
    values = {}
    base = diagnostics_base(results_dir, activation)
    for fold in diagnostic_folders(base):
        path = os.path.join(base, fold, "epoch_metrics.csv")
        if not os.path.isfile(path):
            continue
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                validation = row.get("validation_physical_mse", "").strip()
                if validation:
                    values.setdefault(int(row["epoch"]), []).append(float(validation))
    return values


def read_diagnostic_metric(results_dir, activation, filename, value_key, scope=None):
    values = {}
    base = diagnostics_base(results_dir, activation)
    for fold in diagnostic_folders(base):
        path = os.path.join(base, fold, filename)
        if not os.path.isfile(path):
            continue
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if scope is not None and row.get("parameter_scope") != scope:
                    continue
                value = row.get(value_key, "").strip()
                if value:
                    values.setdefault(int(row["epoch"]), []).append(float(value))
    return {epoch: mean(points) for epoch, points in values.items()}


def plot_val_mse(rows_by_activation, out_dir):
    means = []
    errors = []
    for activation in ACTIVATIONS:
        values = [float(row["validation_mse"]) for row in rows_by_activation[activation]]
        means.append(mean(values))
        errors.append(std(values))

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(ACTIVATIONS)), means, yerr=errors, capsize=5, color="tab:blue")
    best = means.index(min(means))
    bars[best].set_color("tab:orange")
    ax.set_title(f"Cross-validated validation MSE by activation\n{CONFIG_STR}")
    ax.set_ylabel("Mean validation MSE (physical units)")
    ax.set_xticks(range(len(ACTIVATIONS)), [DISPLAY_NAMES[a] for a in ACTIVATIONS])
    ax.grid(axis="y", alpha=0.3)
    for index, value in enumerate(means):
        ax.text(index, value + errors[index], f"{value:.6f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot1_val_mse_by_activation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_fold_mse(rows_by_activation, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.13
    offsets = [(index - (len(ACTIVATIONS) - 1) / 2) * width for index in range(len(ACTIVATIONS))]
    for offset, activation in zip(offsets, ACTIVATIONS):
        rows = rows_by_activation[activation]
        ax.bar(
            [int(row["fold"]) + offset for row in rows],
            [float(row["validation_mse"]) for row in rows],
            width=width,
            label=DISPLAY_NAMES[activation].replace("\n", " "),
        )
    ax.set_title(f"Validation MSE by fold and activation\n{CONFIG_STR}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Validation MSE (physical units)")
    ax.set_xticks(range(5))
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot2_val_mse_by_fold.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_val_curves(results_dir, out_dir):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for activation in ACTIVATIONS:
        epoch_values = read_epoch_metrics(results_dir, activation)
        if not epoch_values:
            continue
        epochs = sorted(epoch_values)
        averages = [mean(epoch_values[epoch]) for epoch in epochs]
        deviations = [std(epoch_values[epoch]) for epoch in epochs]
        label = DISPLAY_NAMES[activation].replace("\n", " ")
        ax.plot(epochs, averages, marker="o", linewidth=1.5, label=label)
        ax.fill_between(
            epochs,
            [value - deviation for value, deviation in zip(averages, deviations)],
            [value + deviation for value, deviation in zip(averages, deviations)],
            alpha=0.1,
        )
    ax.set_title(f"Validation MSE over training checkpoints\n{CONFIG_STR}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE (physical units)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot3_val_curves_epoch.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_diagnostics_over_epochs(results_dir, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"Training diagnostics over epochs\n{CONFIG_STR}")
    for activation in ACTIVATIONS:
        label = DISPLAY_NAMES[activation].replace("\n", " ")
        gradients = read_diagnostic_metric(
            results_dir, activation, "gradient_norms.csv", "mean_norm"
        )
        updates = read_diagnostic_metric(
            results_dir, activation, "parameter_update_ratios.csv", "mean_ratio", "weights"
        )
        if gradients:
            epochs = sorted(gradients)
            axes[0].plot(epochs, [gradients[epoch] for epoch in epochs], label=label)
        if updates:
            epochs = sorted(updates)
            axes[1].plot(epochs, [updates[epoch] for epoch in epochs], label=label)
    axes[0].set_title("Mean gradient norm")
    axes[0].set_ylabel("Mean norm (all layers)")
    axes[1].set_title("Mean parameter update ratio")
    axes[1].set_ylabel("|Δw| / |w| (weights scope)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "plot4_diagnostics_over_epochs.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=os.environ.get(
            "ACTIVATION_RESULTS_DIR",
            "results/cross_validation/activation-function-tuning",
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"),
    )
    args = parser.parse_args()
    if not os.path.isdir(args.results_dir):
        print(f"ERROR: results directory not found: {args.results_dir}", file=sys.stderr)
        return 1
    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_fold_rows(args.results_dir)
    print(f"Results: {args.results_dir}")
    print(f"Output:  {args.out_dir}")
    plot_val_mse(rows, args.out_dir)
    plot_fold_mse(rows, args.out_dir)
    plot_val_curves(args.results_dir, args.out_dir)
    plot_diagnostics_over_epochs(args.results_dir, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
