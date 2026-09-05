#!/usr/bin/env python3
"""Create the activation-function figures used in Chapter 04.

The script reads the committed CSV diagnostics. Candidate ranking uses the
official sample-weighted values in ``cv_summary.csv``. Diagnostic aggregation
is kept explicit: gradient and update peaks are maxima over all folds, while
the selected-candidate depth profiles are averaged over folds and activation
statistics are pooled from their population moments.

Usage:
    python3 plot_report_activation.py [--results-dir PATH] [--output-dir PATH]
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


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
    "leakyrelu-alpha-0.01": "LeakyReLU\nalpha=0.01",
    "leakyrelu-alpha-0.05": "LeakyReLU\nalpha=0.05",
    "leakyrelu-alpha-0.1": "LeakyReLU\nalpha=0.1",
}
WINNER = "leakyrelu-alpha-0.05"


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_activation(candidate_name):
    return candidate_name.split("activation=", 1)[1].rstrip("]")


def weighted_mean_std(values):
    total_weight = sum(weight for _, weight in values)
    mean = sum(value * weight for value, weight in values) / total_weight
    variance = sum(weight * (value - mean) ** 2 for value, weight in values)
    return mean, math.sqrt(variance / total_weight)


def load_inputs(results_dir):
    fold_rows = read_rows(results_dir / "fold_results.csv")
    fold_weights = {
        (row["activation"], int(row["fold"])): int(row["validation_samples"])
        for row in fold_rows
    }

    summary = {}
    for row in read_rows(
        results_dir
        / "activation-function-tuning"
        / "search"
        / "cv_summary.csv"
    ):
        activation = parse_activation(row["candidate_name"])
        summary.setdefault(
            activation,
            {
                "mean": float(row["mean_validation_physical_mse"]),
                "std": float(row["validation_stddev"]),
            },
        )

    history = defaultdict(lambda: defaultdict(list))
    for row in read_rows(results_dir / "training_history.csv"):
        activation = row["activation"]
        fold = int(row["fold"])
        epoch = int(row["epoch"])
        history[activation][epoch].append(
            (
                float(row["validation_mse"]),
                fold_weights[(activation, fold)],
            )
        )

    search_dir = results_dir / "activation-function-tuning" / "search"
    return summary, history, fold_weights, search_dir


def diagnostic_rows(search_dir, candidate_index, filename):
    rows = []
    candidate_dir = search_dir / f"candidate_{candidate_index:03d}"
    for fold in range(5):
        path = candidate_dir / f"fold_{fold:03d}" / filename
        for row in read_rows(path):
            row["fold"] = fold
            rows.append(row)
    return rows


def aggregate_histogram(search_dir, candidate_index, epoch, layer_index, phase):
    rows = [
        row
        for row in diagnostic_rows(search_dir, candidate_index, "activation_histograms.csv")
        if int(row["epoch"]) == epoch
        and int(row["layer_index"]) == layer_index
        and row["phase"] == phase
    ]
    if not rows:
        raise ValueError(
            f"No histogram rows for candidate {candidate_index}, epoch {epoch}, "
            f"layer {layer_index}, phase {phase}"
        )
    edges = [float(value) for value in rows[0]["bin_edges"].split(";")]
    counts = [0] * (len(edges) - 1)
    for row in rows:
        row_edges = [float(value) for value in row["bin_edges"].split(";")]
        if row_edges != edges:
            raise ValueError("Histogram bin edges differ between folds")
        for index, value in enumerate(row["counts"].split(";")):
            counts[index] += int(value)
    return edges, counts


def activation_derivative(activation, value):
    if activation == "tanh":
        output = math.tanh(value)
        return 1.0 - output * output
    if activation == "sigmoid":
        output = 1.0 / (1.0 + math.exp(-value))
        return output * (1.0 - output)
    if activation == "relu":
        return 1.0 if value > 0.0 else 0.0
    alpha = float(activation.rsplit("-", 1)[1])
    return 1.0 if value > 0.0 else alpha


def plateau_threshold(activation, derivative_limit=0.01):
    if activation == "tanh":
        return math.atanh(math.sqrt(1.0 - derivative_limit))
    if activation == "sigmoid":
        root = math.sqrt(1.0 - 4.0 * derivative_limit)
        return math.log((1.0 + root) / (1.0 - root))
    return None


def histogram_density(edges, counts):
    total = sum(counts)
    width = edges[1] - edges[0]
    return [count / (total * width) for count in counts]


def plateau_mass(activation, edges, counts):
    centers = [(left + right) / 2.0 for left, right in zip(edges, edges[1:])]
    threshold = plateau_threshold(activation)
    if threshold is None:
        selected = [count for center, count in zip(centers, counts) if center < 0.0]
    else:
        selected = [
            count
            for center, count in zip(centers, counts)
            if abs(center) >= threshold
        ]
    return sum(selected) / sum(counts)


def plot_ranking(summary, output_dir):
    order = sorted(ACTIVATIONS, key=lambda activation: summary[activation]["mean"])
    means = [summary[activation]["mean"] for activation in order]
    deviations = [summary[activation]["std"] for activation in order]
    colors = ["tab:orange" if activation == WINNER else "tab:blue" for activation in order]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    positions = list(range(len(order)))
    ax.errorbar(
        means,
        positions,
        xerr=deviations,
        fmt="none",
        ecolor="0.25",
        elinewidth=1.0,
        capsize=4,
    )
    ax.scatter(means, positions, c=colors, s=42, zorder=3)
    ax.set_yticks(positions, [DISPLAY_NAMES[activation] for activation in order])
    ax.invert_yaxis()
    ax.set_xlabel("Sample-weighted mean validation MSE (physical units)")
    ax.set_xlim(left=0.003)
    ax.grid(axis="x", alpha=0.3)
    for position, value in zip(positions, means):
        ax.text(value + 0.00004, position, f"{value:.6f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "activation_ranking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_history(history, output_dir):
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    for activation in ACTIVATIONS:
        epochs = sorted(history[activation])
        means = []
        deviations = []
        for epoch in epochs:
            mean, deviation = weighted_mean_std(history[activation][epoch])
            means.append(mean)
            deviations.append(deviation)
        label = DISPLAY_NAMES[activation].replace("\n", " ")
        line = ax.plot(epochs, means, marker="o", markersize=2.8, linewidth=1.4, label=label)[0]
        lower = [
            mean - deviation if mean > deviation else float("nan")
            for mean, deviation in zip(means, deviations)
        ]
        upper = [mean + deviation for mean, deviation in zip(means, deviations)]
        ax.fill_between(epochs, lower, upper, color=line.get_color(), alpha=0.10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE (physical units)")
    ax.set_yscale("log")
    ax.set_xticks(range(10, 101, 10))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "activation_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_input_distributions(search_dir, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(11.2, 7.2), sharex=True)
    x_values = [-10.0 + 20.0 * index / 400.0 for index in range(401)]
    axes = list(axes.flat)

    for index, activation in enumerate(ACTIVATIONS):
        axis = axes[index]
        distributions = {}
        for epoch in (1, 100):
            edges, counts = aggregate_histogram(
                search_dir, index, epoch, layer_index=5, phase="pre_activation"
            )
            distributions[epoch] = (edges, counts, histogram_density(edges, counts))

        threshold = plateau_threshold(activation)
        if threshold is None:
            axis.axvspan(-10.0, 0.0, color="#d95f02", alpha=0.12, zorder=0)
            plateau_label = "negative branch"
        else:
            axis.axvspan(-10.0, -threshold, color="#d95f02", alpha=0.12, zorder=0)
            axis.axvspan(threshold, 10.0, color="#d95f02", alpha=0.12, zorder=0)
            plateau_label = r"low slope: $|f'(z)|\leq0.01$"

        for epoch, color in ((1, "0.55"), (100, "tab:blue")):
            edges, counts, density = distributions[epoch]
            centers = [(left + right) / 2.0 for left, right in zip(edges, edges[1:])]
            width = edges[1] - edges[0]
            if epoch == 1:
                axis.bar(
                    centers,
                    density,
                    width=width,
                    color=color,
                    edgecolor="0.25",
                    linewidth=0.25,
                    alpha=0.38,
                    align="center",
                    zorder=1,
                )
            else:
                step_density = density + [density[-1]]
                axis.step(
                    edges,
                    step_density,
                    where="post",
                    color=color,
                    linewidth=1.4,
                    zorder=3,
                )
                axis.fill_between(
                    edges,
                    step_density,
                    step="post",
                    color=color,
                    alpha=0.10,
                    zorder=2,
                )

        derivative_axis = axis.twinx()
        derivative_axis.plot(
            x_values,
            [activation_derivative(activation, value) for value in x_values],
            color="black",
            linewidth=1.2,
            zorder=4,
        )
        if threshold is not None:
            derivative_axis.axhline(
                0.01, color="#d95f02", linestyle=":", linewidth=0.8, zorder=4
            )
        derivative_axis.set_ylim(0.0, 1.05)
        derivative_axis.tick_params(axis="y", labelsize=7)
        axis.set_xlim(-10.0, 10.0)
        maximum_density = max(
            max(distributions[epoch][2]) for epoch in (1, 100)
        )
        axis.set_ylim(0.0, maximum_density * 1.25)
        axis.axvline(0.0, color="0.25", linestyle="--", linewidth=0.6, zorder=4)
        axis.grid(axis="y", alpha=0.25)
        axis.set_title(
            f"{DISPLAY_NAMES[activation].replace(chr(10), ' ')}\n{plateau_label}",
            fontsize=9,
        )
        mass = plateau_mass(activation, distributions[100][0], distributions[100][1])
        axis.text(
            0.98,
            0.95,
            f"epoch 100 mass: {100.0 * mass:.1f}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )
        if index % 3 == 0:
            axis.set_ylabel("Empirical density")
        if index >= 3:
            axis.set_xlabel("Pre-activation value $z$")
        if index == 2:
            derivative_axis.set_ylabel("Absolute derivative $|f'(z)|$")

    figure.suptitle(
        "Recorded pre-activation distributions at the first dense nonlinearity",
        fontsize=12,
    )
    figure.legend(
        handles=[
            Patch(facecolor="0.55", edgecolor="0.25", alpha=0.38, label="Epoch 1"),
            Line2D([0], [0], color="tab:blue", linewidth=1.4, label="Epoch 100"),
            Line2D([0], [0], color="black", linewidth=1.2, label="$|f'(z)|$"),
            Patch(
                facecolor="#d95f02",
                edgecolor="none",
                alpha=0.12,
                label="Plateau or low-slope region",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        fontsize=8,
        frameon=False,
    )
    figure.text(
        0.5,
        0.01,
        "Histograms pool the five folds at layer 5; the fixed edge bins include values outside [-10, 10].",
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.88))
    figure.savefig(output_dir / "activation_input_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def pooled_variance(rows):
    count = sum(int(row["count"]) for row in rows)
    first_moment = sum(int(row["count"]) * float(row["mean"]) for row in rows)
    second_moment = sum(
        int(row["count"])
        * (float(row["variance"]) + float(row["mean"]) ** 2)
        for row in rows
    )
    mean = first_moment / count
    return second_moment / count - mean**2


def plot_diagnostics(search_dir, output_dir):
    peak_gradients = {}
    peak_updates = {}
    for index, activation in enumerate(ACTIVATIONS):
        gradient_rows = diagnostic_rows(search_dir, index, "gradient_norms.csv")
        update_rows = diagnostic_rows(search_dir, index, "parameter_update_ratios.csv")
        peak_gradients[activation] = max(
            float(row["maximum_norm"])
            for row in gradient_rows
            if row["parameter_scope"] == "all"
        )
        peak_updates[activation] = max(
            float(row["mean_ratio"])
            for row in update_rows
            if row["parameter_scope"] == "weights"
        )

    selected_gradients = diagnostic_rows(search_dir, ACTIVATIONS.index(WINNER), "gradient_norms.csv")
    selected_gradient_profile = defaultdict(lambda: defaultdict(list))
    for row in selected_gradients:
        if row["parameter_scope"] == "all" and int(row["epoch"]) in (1, 100):
            selected_gradient_profile[int(row["epoch"])][int(row["layer_index"])].append(
                float(row["rms_norm"])
            )

    selected_activation_rows = diagnostic_rows(
        search_dir, ACTIVATIONS.index(WINNER), "activation_statistics.csv"
    )
    selected_variances = defaultdict(dict)
    for epoch in (1, 50, 100):
        for layer_index in (1, 5, 7, 9, 11):
            rows = [
                row
                for row in selected_activation_rows
                if int(row["epoch"]) == epoch
                and int(row["layer_index"]) == layer_index
                and row["phase"] == "post_activation"
            ]
            selected_variances[epoch][layer_index] = pooled_variance(rows)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.1))
    fig.suptitle("Activation diagnostics for the recorded five-fold search", fontsize=12)
    positions = list(range(len(ACTIVATIONS)))
    labels = [DISPLAY_NAMES[activation].replace("\n", " ") for activation in ACTIVATIONS]
    colors = ["tab:orange" if activation == WINNER else "tab:blue" for activation in ACTIVATIONS]

    axes[0, 0].bar(positions, [peak_gradients[a] for a in ACTIVATIONS], color=colors)
    axes[0, 0].set_title("Peak gradient maximum norm")
    axes[0, 0].set_ylabel("Maximum norm, scope=all")
    axes[0, 0].set_yscale("log")

    axes[0, 1].bar(positions, [peak_updates[a] for a in ACTIVATIONS], color=colors)
    axes[0, 1].set_title("Peak actual parameter movement")
    axes[0, 1].set_ylabel("Mean update ratio, scope=weights")
    axes[0, 1].set_yscale("log")

    for epoch, style in ((1, "--"), (100, "-")):
        layers = sorted(selected_gradient_profile[epoch])
        means = [
            sum(selected_gradient_profile[epoch][layer])
            / len(selected_gradient_profile[epoch][layer])
            for layer in layers
        ]
        axes[1, 0].plot(layers, means, marker="o", linestyle=style, label=f"Epoch {epoch}")
    axes[1, 0].set_title("Selected candidate gradient depth")
    axes[1, 0].set_xlabel("Stable model layer index")
    axes[1, 0].set_ylabel("RMS gradient norm")
    axes[1, 0].set_xticks([0, 4, 6, 8, 10, 12])
    axes[1, 0].legend(fontsize=8)

    for epoch, style in ((1, "--"), (50, ":"), (100, "-")):
        layers = sorted(selected_variances[epoch])
        axes[1, 1].plot(
            layers,
            [selected_variances[epoch][layer] for layer in layers],
            marker="o",
            linestyle=style,
            label=f"Epoch {epoch}",
        )
    axes[1, 1].set_title("Selected candidate post-activation variance")
    axes[1, 1].set_xlabel("Stable activation layer index")
    axes[1, 1].set_ylabel("Population variance")
    axes[1, 1].set_xticks([1, 5, 7, 9, 11])
    axes[1, 1].legend(fontsize=8)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
    axes[0, 0].set_xticks(positions, labels, rotation=35, ha="right", fontsize=8)
    axes[0, 1].set_xticks(positions, labels, rotation=35, ha="right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_dir / "activation_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/cross_validation/activation-function-tuning"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Report/Images/Chapter04/activation"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, history, _, search_dir = load_inputs(args.results_dir)
    plot_ranking(summary, args.output_dir)
    plot_history(history, args.output_dir)
    plot_input_distributions(search_dir, args.output_dir)
    plot_diagnostics(search_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
