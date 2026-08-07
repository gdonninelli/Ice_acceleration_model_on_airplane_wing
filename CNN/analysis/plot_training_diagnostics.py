#!/usr/bin/env python3
"""Render cluster-safe plots from CNN training diagnostic artifacts."""

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


EPSILON = 1e-12


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def number(value: str) -> float:
    if value == "" or value is None:
        return math.nan
    return float(value)


def directory_index(path: Path, prefix: str) -> int | None:
    for part in path.parts:
        match = re.fullmatch(rf"{prefix}_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def discover_runs(root: Path,
                  candidates: set[int] | None,
                  folds: set[int] | None) -> list[Path]:
    paths = []
    if (root / "epoch_metrics.csv").exists():
        paths.append(root)
    paths.extend(path.parent for path in root.rglob("epoch_metrics.csv"))
    selected = []
    for path in sorted(set(paths)):
        candidate = directory_index(path, "candidate")
        fold = directory_index(path, "fold")
        if candidates is not None and candidate not in candidates:
            continue
        if folds is not None and fold not in folds:
            continue
        selected.append(path)
    return selected


def selected_epochs(rows: list[dict[str, str]], requested: set[int] | None) -> list[int]:
    available = sorted({int(row["epoch"]) for row in rows})
    return available if requested is None else [epoch for epoch in available if epoch in requested]


def run_label(root: Path, run: Path) -> str:
    relative = run.relative_to(root) if run != root else Path("run")
    return "_".join(relative.parts).replace(" ", "_")


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_objectives(run: Path, output: Path, prefix: str) -> None:
    rows = read_csv(run / "epoch_metrics.csv")
    if not rows:
        return
    epochs = np.array([int(row["epoch"]) for row in rows])
    objective = np.array([number(row["train_objective"]) for row in rows])
    validation = np.array([number(row["validation_physical_mse"]) for row in rows])
    figure, left = plt.subplots(figsize=(8, 4.5))
    left.plot(epochs, objective, color="#1f5a7a", label="training objective")
    left.set_xlabel("Epoch")
    left.set_ylabel("SIMM training objective", color="#1f5a7a")
    left.grid(alpha=0.25)
    right = left.twinx()
    valid = np.isfinite(validation)
    if np.any(valid):
        right.plot(epochs[valid], validation[valid], "o-", color="#b54935",
                   label="validation physical MSE")
    right.set_ylabel("Validation physical MSE", color="#b54935")
    plt.title("Training and validation metrics")
    save_figure(output / f"{prefix}_training_validation.png")


def plot_gradients(run: Path, output: Path, prefix: str,
                   requested: set[int] | None) -> None:
    rows = [row for row in read_csv(run / "gradient_norms.csv")
            if row["parameter_scope"] == "all"]
    epochs = selected_epochs(rows, requested)
    if not rows or not epochs:
        return
    plt.figure(figsize=(8, 4.8))
    for epoch in epochs:
        current = sorted((row for row in rows if int(row["epoch"]) == epoch),
                         key=lambda row: int(row["layer_index"]))
        depth = [int(row["layer_index"]) for row in current]
        values = np.maximum([number(row["rms_norm"]) for row in current], EPSILON)
        plt.plot(depth, values, "o-", label=f"epoch {epoch}")
    plt.yscale("log")
    plt.xlabel("Stable model layer index")
    plt.ylabel("Epoch RMS gradient L2 norm")
    plt.title("Gradient magnitude across depth")
    plt.grid(alpha=0.25, which="both")
    plt.legend(fontsize=8, ncol=2)
    save_figure(output / f"{prefix}_gradient_depth.png")


def plot_updates(run: Path, output: Path, prefix: str) -> None:
    rows = [row for row in read_csv(run / "parameter_update_ratios.csv")
            if row["parameter_scope"] == "all"]
    if not rows:
        return
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer_index"]), row["layer_name"])].append(row)
    plt.figure(figsize=(8, 4.8))
    for (index, name), values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["epoch"]))
        plt.plot([int(row["epoch"]) for row in values],
                 np.maximum([number(row["mean_ratio"]) for row in values], EPSILON),
                 "o-", label=f"{index}: {name}")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Mean actual parameter update ratio")
    plt.title("Optimizer update size by trainable layer")
    plt.grid(alpha=0.25, which="both")
    plt.legend(fontsize=7)
    save_figure(output / f"{prefix}_parameter_update_ratio.png")


def plot_histograms(run: Path, output: Path, prefix: str,
                    requested: set[int] | None) -> None:
    rows = read_csv(run / "activation_histograms.csv")
    epochs = selected_epochs(rows, requested)
    for phase in ("pre_activation", "post_activation"):
        for epoch in epochs:
            current = sorted(
                (row for row in rows
                 if row["phase"] == phase and int(row["epoch"]) == epoch),
                key=lambda row: int(row["layer_index"]))
            if not current:
                continue
            columns = min(3, len(current))
            row_count = math.ceil(len(current) / columns)
            figure, axes = plt.subplots(row_count, columns,
                                        figsize=(4.2 * columns, 3.2 * row_count),
                                        squeeze=False)
            for axis, row in zip(axes.flat, current):
                edges = np.array([float(value) for value in row["bin_edges"].split(";")])
                counts = np.array([float(value) for value in row["counts"].split(";")])
                axis.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
                         color="#477998", alpha=0.8)
                axis.set_title(f'{row["layer_index"]}: {row["layer_name"]}', fontsize=9)
                axis.set_xlabel("Activation value")
                axis.set_ylabel("Global count")
            for axis in axes.flat[len(current):]:
                axis.set_visible(False)
            figure.suptitle(f"{phase.replace('_', ' ').title()}, epoch {epoch}")
            save_figure(output / f"{prefix}_{phase}_histograms_epoch_{epoch:04d}.png")


def post_variance_rows(run: Path) -> list[dict[str, str]]:
    return [row for row in read_csv(run / "activation_statistics.csv")
            if row["phase"] == "post_activation"]


def plot_variance_depth(run: Path, output: Path, prefix: str,
                        requested: set[int] | None) -> None:
    rows = post_variance_rows(run)
    epochs = selected_epochs(rows, requested)
    if not rows or not epochs:
        return
    plt.figure(figsize=(8, 4.8))
    all_values = []
    for epoch in epochs:
        current = sorted((row for row in rows if int(row["epoch"]) == epoch),
                         key=lambda row: int(row["layer_index"]))
        values = np.array([number(row["variance"]) for row in current])
        all_values.extend(values)
        plt.plot([int(row["layer_index"]) for row in current], values,
                 "o-", label=f"epoch {epoch}")
    positive = np.array([value for value in all_values if value > 0])
    if positive.size and positive.max() / positive.min() > 1000:
        plt.yscale("log")
    plt.xlabel("Stable model layer index (activation layers only)")
    plt.ylabel("Post-activation population variance")
    plt.title("Activated-value variance across depth")
    plt.grid(alpha=0.25, which="both")
    plt.legend(fontsize=8, ncol=2)
    save_figure(output / f"{prefix}_activated_variance_depth.png")


def plot_variance_heatmap(run: Path, output: Path, prefix: str) -> None:
    rows = post_variance_rows(run)
    if not rows:
        return
    epochs = sorted({int(row["epoch"]) for row in rows})
    layers = sorted({int(row["layer_index"]) for row in rows})
    epoch_index = {value: index for index, value in enumerate(epochs)}
    layer_index = {value: index for index, value in enumerate(layers)}
    matrix = np.full((len(epochs), len(layers)), np.nan)
    for row in rows:
        matrix[epoch_index[int(row["epoch"])],
               layer_index[int(row["layer_index"])]] = number(row["variance"])
    positive = matrix[np.isfinite(matrix) & (matrix > 0)]
    display = matrix
    label = "Variance"
    if positive.size and positive.max() / positive.min() > 1000:
        display = np.log10(np.maximum(matrix, EPSILON))
        label = "log10 variance"
    plt.figure(figsize=(max(6, len(layers) * 0.7), 5))
    image = plt.imshow(display, aspect="auto", origin="lower", cmap="viridis")
    plt.xticks(range(len(layers)), layers)
    tick_step = max(1, len(epochs) // 12)
    ticks = list(range(0, len(epochs), tick_step))
    plt.yticks(ticks, [epochs[index] for index in ticks])
    plt.xlabel("Stable model layer index (activation layers only)")
    plt.ylabel("Epoch")
    plt.title("Activated-value variance heatmap")
    plt.colorbar(image, label=label)
    save_figure(output / f"{prefix}_activated_variance_heatmap.png")


def plot_variance_epochs(run: Path, output: Path, prefix: str) -> None:
    rows = post_variance_rows(run)
    if not rows:
        return
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["layer_index"]), row["layer_name"])].append(row)
    plt.figure(figsize=(8, 4.8))
    all_values = []
    for (index, name), values in sorted(grouped.items()):
        values.sort(key=lambda row: int(row["epoch"]))
        variances = np.array([number(row["variance"]) for row in values])
        all_values.extend(variances)
        plt.plot([int(row["epoch"]) for row in values], variances,
                 "o-", label=f"{index}: {name}")
    positive = np.array([value for value in all_values if value > 0])
    if positive.size and positive.max() / positive.min() > 1000:
        plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Post-activation population variance")
    plt.title("Activated-value variance by layer")
    plt.grid(alpha=0.25, which="both")
    plt.legend(fontsize=7)
    save_figure(output / f"{prefix}_activated_variance_epochs.png")


def plot_learning_rate(run: Path, output: Path, prefix: str) -> None:
    rows = read_csv(run / "epoch_metrics.csv")
    if not rows:
        return
    plt.figure(figsize=(8, 4.2))
    plt.plot([int(row["epoch"]) for row in rows],
             [number(row["effective_learning_rate"]) for row in rows], "o-")
    plt.xlabel("Epoch")
    plt.ylabel("Effective learning rate")
    plt.title("Learning-rate schedule")
    plt.grid(alpha=0.25)
    save_figure(output / f"{prefix}_learning_rate.png")


def topology_signature(run: Path) -> tuple[tuple[int, str, bool], ...]:
    path = run / "metadata.json"
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as source:
        metadata = json.load(source)
    return tuple((int(layer["index"]), str(layer["name"]), bool(layer["activation"]))
                 for layer in metadata.get("model_layers", []))


def plot_cv_comparisons(root: Path, runs: list[Path], output: Path) -> None:
    cv_runs = [run for run in runs if directory_index(run, "fold") is not None]
    groups: dict[tuple[tuple[int, str, bool], ...], list[Path]] = defaultdict(list)
    for run in cv_runs:
        groups[topology_signature(run)].append(run)
    for group_index, compatible in enumerate(groups.values()):
        if not compatible:
            continue
        plt.figure(figsize=(8, 4.8))
        plotted = False
        for run in compatible:
            rows = read_csv(run / "epoch_metrics.csv")
            values = [(int(row["epoch"]), number(row["validation_physical_mse"]))
                      for row in rows]
            values = [(epoch, value) for epoch, value in values if math.isfinite(value)]
            if not values:
                continue
            plotted = True
            plt.plot([item[0] for item in values], [item[1] for item in values],
                     "o-", label=str(run.relative_to(root)))
        if plotted:
            plt.xlabel("Epoch")
            plt.ylabel("Validation physical MSE")
            plt.title("Cross-validation fold comparison (compatible topology)")
            plt.grid(alpha=0.25)
            plt.legend(fontsize=7)
            save_figure(output / f"cv_fold_comparison_topology_{group_index:02d}.png")
        else:
            plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="Experiment/run result directory")
    parser.add_argument("--output-dir", type=Path,
                        help="Plot directory (default: <input>/plots)")
    parser.add_argument("--candidate", type=int, action="append",
                        help="Select a numeric candidate index; repeat as needed")
    parser.add_argument("--fold", type=int, action="append",
                        help="Select a numeric fold index; repeat as needed")
    parser.add_argument("--epoch", type=int, action="append",
                        help="Select an epoch for depth/histogram plots; repeat as needed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input.resolve()
    output = (args.output_dir or (root / "plots")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(root,
                         set(args.candidate) if args.candidate else None,
                         set(args.fold) if args.fold else None)
    if not runs:
        raise SystemExit(f"No epoch_metrics.csv files found under {root}")
    requested = set(args.epoch) if args.epoch else None
    for run in runs:
        prefix = run_label(root, run)
        plot_objectives(run, output, prefix)
        plot_gradients(run, output, prefix, requested)
        plot_updates(run, output, prefix)
        plot_histograms(run, output, prefix, requested)
        plot_variance_depth(run, output, prefix, requested)
        plot_variance_epochs(run, output, prefix)
        plot_variance_heatmap(run, output, prefix)
        plot_learning_rate(run, output, prefix)
    plot_cv_comparisons(root, runs, output)
    print(f"Wrote diagnostics plots to {output}")


if __name__ == "__main__":
    main()
