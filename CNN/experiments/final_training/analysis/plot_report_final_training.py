#!/usr/bin/env python3
"""Create the final-training and angle-split figures used in Chapter 04."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_final_metrics(path: Path) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    metrics = {}
    for row in rows:
        try:
            metrics[row["metric"]] = float(row["value"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid final metric row in {path}: {row}") from error

    required = {"selected_epoch", "test_physical_mse"}
    missing = required - metrics.keys()
    if missing:
        raise ValueError(f"Missing final metrics in {path}: {sorted(missing)}")
    return metrics


def read_epoch_metrics(path: Path) -> tuple[list[int], list[float], list[float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    epochs = []
    training_mse = []
    validation_mse = []
    for row in rows:
        try:
            epoch = int(row["epoch"])
            train = float(row["training_physical_mse"])
            validation = float(row["validation_physical_mse"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid epoch metric row in {path}: {row}") from error
        if not all(math.isfinite(value) for value in (train, validation)):
            raise ValueError(f"Non-finite epoch metric row in {path}: {row}")
        epochs.append(epoch)
        training_mse.append(train)
        validation_mse.append(validation)

    if not epochs:
        raise ValueError(f"No epoch metrics found in {path}")
    return epochs, training_mse, validation_mse


def read_test_metrics(path: Path) -> dict[str, tuple[int, float]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    metrics = {}
    for row in rows:
        try:
            subset = row["subset"]
            samples = int(row["samples"])
            physical_mse = float(row["physical_mse"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid test metric row in {path}: {row}") from error
        if samples <= 0 or not math.isfinite(physical_mse):
            raise ValueError(f"Invalid test metric values in {path}: {row}")
        metrics[subset] = (samples, physical_mse)

    required = {"overall", "abs_aoa_le_10_deg", "abs_aoa_gt_10_deg"}
    missing = required - metrics.keys()
    if missing:
        raise ValueError(f"Missing angle subsets in {path}: {sorted(missing)}")
    return metrics


def plot_history(
    epochs: list[int],
    training_mse: list[float],
    validation_mse: list[float],
    selected_epoch: int,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharex=True)
    figure.suptitle("Final production training")
    stopping_epoch = epochs[-1]
    for axis, scale, title in zip(
        axes,
        ("linear", "log"),
        ("Linear y-axis", "Logarithmic y-axis"),
    ):
        axis.plot(epochs, training_mse, color="tab:blue", linewidth=1.4,
                  label="Training physical MSE")
        axis.plot(epochs, validation_mse, color="tab:orange", linewidth=1.4,
                  label="Validation physical MSE")
        axis.axvline(selected_epoch, color="tab:green", linestyle="--",
                     linewidth=1.2, label=f"Selected epoch ({selected_epoch})")
        if stopping_epoch != selected_epoch:
            axis.axvline(stopping_epoch, color="0.35", linestyle=":",
                         linewidth=1.2, label=f"Stopping epoch ({stopping_epoch})")
        if scale == "log":
            axis.set_yscale("log")
        else:
            axis.set_ylim(bottom=0.0)
        axis.set_xlabel("Completed epoch")
        axis.set_ylabel("Physical MSE")
        axis.set_title(title)
        axis.grid(True, which="both" if scale == "log" else "major", alpha=0.25)
        axis.legend(frameon=False, loc="best", fontsize=8)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_angle_mse(
    test_metrics: dict[str, tuple[int, float]],
    output_path: Path,
) -> None:
    subset_names = ["abs_aoa_le_10_deg", "abs_aoa_gt_10_deg"]
    labels = [r"$|\alpha| \leq 10^\circ$", r"$|\alpha| > 10^\circ$"]
    values = [test_metrics[name][1] for name in subset_names]
    counts = [test_metrics[name][0] for name in subset_names]
    overall = test_metrics["overall"][1]

    figure, axis = plt.subplots(figsize=(7.4, 5.0))
    bars = axis.bar(labels, values, color=["tab:blue", "tab:orange"], width=0.58)
    axis.axhline(overall, color="0.25", linestyle="--", linewidth=1.1,
                 label=f"Overall MSE ({overall:.6g})")
    axis.set_ylabel("Test physical MSE")
    axis.set_title("Test error by angle-of-attack regime")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    axis.set_ylim(0.0, max(values) * 1.32)
    for bar, count, value in zip(bars, counts, values):
        axis.text(bar.get_x() + bar.get_width() / 2.0, value,
                  f"{value:.6f}\nn={count}", ha="center", va="bottom",
                  fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot final-training history and test MSE by angle regime."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/ordinary-training/production_slurm-56688223"),
        help="Directory containing final_metrics.csv, test_metrics.csv, and epoch_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Report/Images/Chapter04/final_training"),
        help="Directory for the generated report figures.",
    )
    args = parser.parse_args()

    final_metrics = read_final_metrics(args.run_dir / "final_metrics.csv")
    epochs, training_mse, validation_mse = read_epoch_metrics(
        args.run_dir / "epoch_metrics.csv"
    )
    test_metrics = read_test_metrics(args.run_dir / "test_metrics.csv")
    overall_test_mse = test_metrics["overall"][1]
    if not math.isclose(
        overall_test_mse,
        final_metrics["test_physical_mse"],
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError("Overall test MSE does not match final_metrics.csv")
    subset_count = sum(
        test_metrics[name][0] for name in ("abs_aoa_le_10_deg", "abs_aoa_gt_10_deg")
    )
    if subset_count != test_metrics["overall"][0]:
        raise ValueError("Angle subset counts do not match the overall test count")
    selected_epoch = int(final_metrics["selected_epoch"])
    if selected_epoch not in epochs:
        raise ValueError(f"Selected epoch {selected_epoch} is not in {args.run_dir / 'epoch_metrics.csv'}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "final_training_history.png"
    angle_path = args.output_dir / "test_mse_by_angle.png"
    plot_history(epochs, training_mse, validation_mse, selected_epoch, history_path)
    plot_angle_mse(test_metrics, angle_path)
    print(f"Saved plot to: {history_path}")
    print(f"Saved plot to: {angle_path}")


if __name__ == "__main__":
    main()
